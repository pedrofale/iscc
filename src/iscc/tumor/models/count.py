"""
Genotype-level (count-based) tumor engine — phase 3b of DESIGN_scalability.md.

Instead of one Python object per cell, the population is represented as **per-deme genotype
counts** over a shared **genotype registry**. A "cell" is a unit of count in a (genotype,
deme) bucket; cells of the same genotype in the same deme are statistically exchangeable, so
counting them loses no information.

This realises the same birth/death/mutation/dispersal process as the cell-level
`GlandularTumor` (validated by statistical equivalence — it is NOT byte-identical, since it
draws "how many of the count" rather than "which cell"), but each event is O(1) integer
arithmetic on counts. A genotype is carried by a representative cell (reusing its genome /
mutate / get_snvs / get_cnvs / get_exp and CINner fitness).

Normal cells (epithelial/stromal/immune) are seeded as **static** genotype counts (they don't
divide or mutate); they provide tissue structure, cell-type labels, and the local immune density
that the cancer death rate reads. Treatment runs on this engine (see grow/_apply_treatment) and
immune killing is additive contact pressure (see _death_rate); the immune compartment itself is
static (recruitment/migration are future work).
"""
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..components.selection import Selection
from ..components.cell import CancerCell, EpithelialCell, StromalCell, ImmuneCell
from .glandular import bresenham_circumference, get_inside
from ...constants import normal_names

CELLTYPES = ["cancer", "epithelial", "stromal", "immune"]


class GenotypeTumor:
    def __init__(self, config=None, seed=42, genome_params=None, selection_params=None,
                 cancer_cell_params=None, deme_params=None, spatial_params=None,
                 epithelial_cell_params=None, stromal_cell_params=None, immune_cell_params=None,
                 genome_mode="abstract", genome_spec=None):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.type = "genotype"

        # Real-genome mode (DESIGN_inference A.5): the genome is wired from a GenomeSpec built
        # from human chromosome-arm data (arm lengths -> segment_sizes, per-arm oncogene/TSG
        # content) and selection uses the per-arm copy-number model (selection_mode="arm",
        # s_arm vector) instead of the abstract random gene-driver layout. The engine is
        # otherwise unchanged. ``selection_params`` (notably ``s_arm``) still wins, so the ABC
        # layer can inject the per-arm coefficients it is inferring.
        self.genome_mode = genome_mode
        self.genome_spec = genome_spec
        if genome_mode == "real":
            if genome_spec is None:
                raise ValueError("genome_mode='real' requires genome_spec")
            genome_params, selection_params = genome_spec.engine_params(selection_params)

        if config is not None:
            with open(config) as f:
                cfg = yaml.safe_load(f)
            genome_params = cfg["genome_params"]
            selection_params = cfg["selection_params"]
            deme_params = cfg["deme_params"]
            spatial_params = cfg["spatial_params"]
            cp = cfg["cell_params"]
            cancer_cell_params = cp["cancer"]
            epithelial_cell_params = cp.get("epithelial", {})
            stromal_cell_params = cp.get("stromal", {})
            immune_cell_params = cp.get("immune", {})

        self.genome_params = genome_params
        self.n_segments = genome_params["n_segments"]
        self.segment_size = genome_params.get("segment_size", 1000)
        # Per-segment sizes (real-genome mode: proportional to chromosome-arm length); falls
        # back to a uniform scalar so the abstract mode is unchanged.
        self.segment_sizes = genome_params.get("segment_sizes")
        self._normal_params = {
            "epithelial": (EpithelialCell, epithelial_cell_params or {}),
            "stromal": (StromalCell, stromal_cell_params or {}),
            "immune": (ImmuneCell, immune_cell_params or {}),
        }

        self.selection = Selection(n_segments=self.n_segments, segment_size=self.segment_size,
                                   segment_sizes=self.segment_sizes, rng=self.rng, **selection_params)
        self.n_genes = self.selection.n_genes
        self._cancer_params = cancer_cell_params

        # per-cell-type baseline expression (seeded, shared by all cells of a type)
        self.celltype_exps = {}
        for ct in CELLTYPES:
            exp = self.rng.beta(0.1, 1.0, size=self.n_genes)
            exp[self.selection.get_tsgs()] = 0.8
            exp[self.selection.get_oncogenes()] = 0.01
            self.celltype_exps[ct] = exp

        # genotype registry: id -> representative cell. Normals keyed by their type name.
        self.genotypes = {}
        self.genotypes_parents = {}
        self.genotypes_counts = Counter()
        self._next_ord = 0

        # per-immune-cell contact kill hazard, and per-step treatment overrides
        # (gid -> extra death hazard / overridden immune resistance), refreshed each
        # treated step by _apply_treatment so they never corrupt the shared genotype.
        self._immune_prob_kill = (immune_cell_params or {}).get("prob_kill", 0.01)
        self._tx_death_add = {}
        self._tx_immune_resist = {}

        # founder cancer genotype
        founder = CancerCell(
            n_segments=self.n_segments, segment_size=self.segment_size,
            segment_sizes=self.segment_sizes, seed=seed,
            n_onc=len(self.selection.get_oncogenes()), n_tsg=len(self.selection.get_tsgs()),
            n_disp=len(self.selection.get_dispersal_genes()),
            n_ir=len(self.selection.get_immune_resistant()),
            n_tr=len(self.selection.get_treatment_resistant()),
            **cancer_cell_params,
        )
        founder.set_genotype_id()
        self._register(founder)
        self.founder_id = founder.genotype_id

        # grid of demes; each deme is a dict {genotype_id: count}
        self.grid_size = spatial_params["grid_size"]
        self.carrying_capacity = deme_params.get("carrying_capacity", 1)
        self.maximum_death_rate = deme_params.get("maximum_death_rate", 0.5)
        self.structure_radius = spatial_params.get("structure_radius", 0)
        n_demes = self.grid_size * self.grid_size
        self.demes = [dict() for _ in range(n_demes)]
        self.deme_coords = [(i // self.grid_size, i % self.grid_size) for i in range(n_demes)]

        if self.structure_radius > 0:
            self._seed_structure()
        else:
            center = (self.grid_size // 2) * self.grid_size + (self.grid_size // 2)
            self._add(center, self.founder_id, 1)

        # Optional immune microenvironment: seed immune cells in every deme so that
        # cancer growing into them experiences local immune pressure (and so that
        # immunotherapy has a substrate to act on). The count scales with carrying
        # capacity (immune_density = immune cells per capacity unit) so the immune
        # fraction is not washed out once a deme fills with cancer. Static for now
        # (no division/migration yet).
        self.immune_density = spatial_params.get("immune_density", 0.0)
        n_immune = int(round(self.immune_density * max(self.carrying_capacity, 1)))
        if n_immune > 0:
            imm = self._normal_genotype("immune")
            for i in range(n_demes):
                self._add(i, imm, n_immune)

        self.deme_rates = np.array([self._deme_rate(i) for i in range(n_demes)], dtype=float)
        self.traces = []
        self.step = 0
        self.cell_data = None

    # --- genotype registry ---------------------------------------------------
    def _register(self, rep):
        rep.ord = self._next_ord
        self._next_ord += 1
        self.genotypes[rep.genotype_id] = rep
        return rep.genotype_id

    def _normal_genotype(self, type_name):
        """Return the (cached) representative genotype id for a normal cell type."""
        if type_name in self.genotypes:
            return type_name
        cls, params = self._normal_params[type_name]
        rep = cls(n_segments=self.n_segments, segment_size=self.segment_size,
                  segment_sizes=self.segment_sizes, **params)
        rep.genotype_id = type_name
        self._register(rep)
        return type_name

    def _is_cancer(self, gid):
        return self.genotypes[gid].type == "cancer"

    # --- structure seeding (mirrors GlandularTumor.make_structure) ------------
    def _seed_structure(self):
        center = self.grid_size // 2
        border = bresenham_circumference(center, center, self.structure_radius)
        epi = self._normal_genotype("epithelial")
        for (r, c) in border:
            if 0 <= r < self.grid_size and 0 <= c < self.grid_size:
                self._add(r * self.grid_size + c, epi, self.carrying_capacity)
        circle = get_inside(border)
        occupied = set(border) | set(circle)

        in_border = bresenham_circumference(center, center, self.structure_radius - 1)
        in_border = [(r, c) for (r, c) in in_border if 0 <= r < self.grid_size and 0 <= c < self.grid_size]
        if in_border:
            pos = in_border[int(self.rng.choice(len(in_border)))]
            self._add(pos[0] * self.grid_size + pos[1], self.founder_id, 1)

        stroma = self._normal_genotype("stromal")
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if (r, c) not in occupied and not self.demes[r * self.grid_size + c]:
                    self._add(r * self.grid_size + c, stroma, self.carrying_capacity)

    # --- count bookkeeping ---------------------------------------------------
    def _add(self, deme_idx, gid, n):
        deme = self.demes[deme_idx]
        deme[gid] = deme.get(gid, 0) + n
        self.genotypes_counts[gid] += n

    def _remove(self, deme_idx, gid, n):
        deme = self.demes[deme_idx]
        deme[gid] -= n
        if deme[gid] <= 0:
            del deme[gid]
        self.genotypes_counts[gid] -= n
        if self.genotypes_counts[gid] <= 0:
            del self.genotypes_counts[gid]

    def _neighbors(self, deme_idx):
        r, c = self.deme_coords[deme_idx]
        out = []
        for rr, cc in [(r - 1, c), (r, c + 1), (r + 1, c), (r, c - 1)]:
            if 0 <= rr < self.grid_size and 0 <= cc < self.grid_size:
                out.append(rr * self.grid_size + cc)
        return out

    # --- rates (mirror Deme.get_cancer_death_rate) ---------------------------
    def _immune_fraction(self, deme):
        total = sum(deme.values())
        if total == 0:
            return 0.0
        immune = sum(cnt for gid, cnt in deme.items() if self.genotypes[gid].type == "immune")
        return immune / total

    def _death_rate(self, gid, deme_idx):
        """Cancer death rate = crowding-modulated baseline + local immune killing + treatment.

        Mirrors the corrected Deme.get_cancer_death_rate: immune killing is additive contact
        pressure (more local immune cells -> higher death, attenuated by immune resistance),
        and active therapy adds a death hazard (chemo/targeted) or strips immune resistance
        (immunotherapy, via the per-step _tx_* overrides).
        """
        rep = self.genotypes[gid]
        deme = self.demes[deme_idx]
        total = sum(deme.values())
        crowd = self.carrying_capacity if total > self.carrying_capacity else 1.0
        death = min(rep.evolutionary_parameters["death_rate"] * crowd, self.maximum_death_rate)

        ir = self._tx_immune_resist.get(gid, rep.evolutionary_parameters["immune_resistance"])
        ir = min(max(ir, 0.0), 1.0)
        death += self._immune_prob_kill * self._immune_fraction(deme) * (1.0 - ir)

        death += self._tx_death_add.get(gid, 0.0)
        return death

    def _cancer_gids(self, deme):
        return [gid for gid in deme if self._is_cancer(gid)]

    def _deme_rate(self, deme_idx):
        """Total event rate of a deme. Only cancer genotypes are dynamic; normals are static."""
        deme = self.demes[deme_idx]
        rate = 0.0
        for gid in self._cancer_gids(deme):
            div = self.genotypes[gid].evolutionary_parameters["division_rate"]
            rate += deme[gid] * (div + self._death_rate(gid, deme_idx))
        return rate

    def _refresh_rate(self, deme_idx):
        self.deme_rates[deme_idx] = self._deme_rate(deme_idx)

    # --- simulation ----------------------------------------------------------
    def is_extinct(self):
        return self.deme_rates.sum() == 0

    def get_tumor_size(self):
        return int(sum(self.genotypes_counts.values()))

    def get_cancer_size(self):
        return int(sum(c for g, c in self.genotypes_counts.items() if self._is_cancer(g)))

    def update(self, rng):
        if self.is_extinct():
            return
        di = int(rng.choice(len(self.demes), p=self.deme_rates / self.deme_rates.sum()))
        deme = self.demes[di]
        # pick a cancer genotype in the deme proportionally to count * (div + death),
        # ordered by creation ordinal (NOT the id()-based genotype_id) for reproducibility
        gids = sorted(self._cancer_gids(deme), key=lambda g: self.genotypes[g].ord)
        weights = np.array([
            deme[gid] * (self.genotypes[gid].evolutionary_parameters["division_rate"]
                         + self._death_rate(gid, di))
            for gid in gids
        ])
        gid = gids[int(rng.choice(len(gids), p=weights / weights.sum()))]
        rep = self.genotypes[gid]
        div = rep.evolutionary_parameters["division_rate"]
        death = self._death_rate(gid, di)

        affected = [di]
        if rng.random() < death / (div + death):
            self._remove(di, gid, 1)                       # death
        else:
            disp = rep.evolutionary_parameters["dispersal_rate"]
            mut_prob = rep.mutation_rate / (rep.mutation_rate + disp)
            if rng.random() < mut_prob:
                child = rep.divide()
                if child.mutate(rng, self.selection):
                    self._register(child)
                    self.genotypes_parents[child.genotype_id] = rep.genotype_id
                    self._add(di, child.genotype_id, 1)
                else:
                    # no-op mutation (saturated allele): same genotype, grows in place
                    self._add(di, rep.genotype_id, 1)
            else:
                nbrs = self._neighbors(di)
                tgt = nbrs[int(rng.choice(len(nbrs)))] if nbrs else di
                self._add(tgt, gid, 1)
                affected.append(tgt)

        for idx in affected:
            self._refresh_rate(idx)

    # --- treatment (genotype-level) -----------------------------------------
    def _is_treatment_target(self, treatment, rep):
        """Is this cancer genotype sensitive to the therapy?

        Broad therapies (chemo, or gene-targeted therapy with no checkpoints) hit every
        cancer cell; gene-specific therapies delegate to the treatment's expression-based
        `is_target`. The representative's baseline expression is set so `expresses` is
        deterministic.
        """
        if rep.type != "cancer":
            return False
        targets = getattr(treatment, "targets", None)
        if not targets:
            return True
        rep.baseline_exp = self.celltype_exps["cancer"]
        return bool(treatment.is_target(rep, mut_effects=self.selection.mut_effects))

    def _apply_treatment(self, treatment, dosage):
        """Recompute per-genotype treatment effects for the current dosage.

        Treatment is a transient, dose-dependent modifier, not a genotype property, so the
        effect lives in self._tx_* (refreshed every treated step) rather than mutating the
        shared genotype. The per-cell stochastic effectiveness/toxicity of the cell engine
        becomes an expected, dose-scaled intensity over the exchangeable count.
        """
        self._tx_death_add = {}
        self._tx_immune_resist = {}
        if treatment is None or dosage <= 0:
            return
        affects = getattr(treatment, "affects", "death_rate")
        kill_rate = getattr(treatment, "kill_rate", 0.8)
        for gid in list(self.genotypes_counts):
            if not self._is_cancer(gid):
                continue
            rep = self.genotypes[gid]
            target = self._is_treatment_target(treatment, rep)
            tr = rep.evolutionary_parameters["treatment_resistance"]
            p = treatment.effectiveness if target else treatment.toxicity
            intensity = dosage * p * (1.0 - min(max(tr, 0.0), 1.0))  # in [0, 1]
            if intensity <= 0:
                continue
            if affects == "immune_resistance":
                ir = rep.evolutionary_parameters["immune_resistance"]
                self._tx_immune_resist[gid] = ir * (1.0 - intensity)
            else:
                base = rep.evolutionary_parameters["death_rate"]
                self._tx_death_add[gid] = intensity * max(kill_rate - base, 0.0)

    def grow(self, n_steps=1000, seed=None, treatment=None, **kwargs):
        if seed is None:
            seed = self.seed
        self.traces.append(dict(genotypes_counts=dict(self.genotypes_counts)))
        prev_dose = 0.0
        for local_step in range(n_steps - 1):
            rng = np.random.default_rng(seed + self.step + local_step)
            if treatment is not None:
                dose = treatment.get_dosage(self.step + local_step, self.get_tumor_size())
                self._apply_treatment(treatment, dose)
                # death rates changed for every genotype while a dose is on (or just turned
                # off); refresh the whole rate vector so event sampling stays correct.
                if dose > 0 or prev_dose > 0:
                    self.deme_rates = np.array(
                        [self._deme_rate(i) for i in range(len(self.demes))], dtype=float)
                prev_dose = dose
            self.update(rng)
            self.traces.append(dict(genotypes_counts=dict(self.genotypes_counts)))
        self.step += n_steps
        self.make_cell_data()
        return self.traces

    # --- materialisation: counts -> per-cell matrices ------------------------
    def make_cell_data(self, cell_prefix="C", **kwargs):
        gene_names = self.selection.get_gene_names()
        onc_idx, tsg_idx = self.selection.get_oncogenes(), self.selection.get_tsgs()
        disp_idx, ir_idx, tr_idx = (self.selection.get_dispersal_genes(),
                                    self.selection.get_immune_resistant(),
                                    self.selection.get_treatment_resistant())
        snv_cache, cnv_cache, exp_cache, evo_cache = {}, {}, {}, {}
        for gid in self.genotypes_counts:
            rep = self.genotypes[gid]
            snv = rep.get_snvs()
            snv_cache[gid] = snv
            cnv_cache[gid] = rep.get_cnvs()
            if rep.type == "cancer":
                rep.baseline_exp = self.celltype_exps["cancer"]
                exp_cache[gid] = rep.get_exp(self.selection.mut_effects)
            else:
                exp_cache[gid] = self.celltype_exps[rep.type]
            # per-genotype evolutionary parameters + unique-mutated-driver tallies
            evo = dict(rep.evolutionary_parameters)
            evo["n_mut_onc"] = int((snv[onc_idx] > 0).sum())
            evo["n_mut_tsg"] = int((snv[tsg_idx] > 0).sum())
            evo["n_mut_disp"] = int((snv[disp_idx] > 0).sum())
            evo["n_mut_ir"] = int((snv[ir_idx] > 0).sum())
            evo["n_mut_tr"] = int((snv[tr_idx] > 0).sum())
            evo_cache[gid] = evo

        rows_snv, rows_cnv, rows_exp, rows_evo, crd, types, demes_col, names = [], [], [], [], [], [], [], []
        i = 0
        for deme_idx, deme in enumerate(self.demes):
            r, c = self.deme_coords[deme_idx]
            for gid in sorted(deme.keys(), key=lambda g: self.genotypes[g].ord):
                for _ in range(deme[gid]):
                    rows_snv.append(snv_cache[gid]); rows_cnv.append(cnv_cache[gid])
                    rows_exp.append(exp_cache[gid]); rows_evo.append(evo_cache[gid])
                    crd.append((r, c)); types.append(gid); demes_col.append(deme_idx)
                    names.append(f"{cell_prefix}{i}"); i += 1

        idx = pd.Index(names)
        empty = np.empty((0, self.n_genes))
        self.cell_data = dict(
            cell_evo=pd.DataFrame(rows_evo, index=idx),
            cell_snv=pd.DataFrame(np.array(rows_snv) if rows_snv else empty, index=idx, columns=gene_names),
            cell_cnv=pd.DataFrame(np.array(rows_cnv) if rows_cnv else empty, index=idx, columns=gene_names),
            cell_exp=pd.DataFrame(np.array(rows_exp) if rows_exp else empty, index=idx, columns=gene_names),
            cell_crd=pd.DataFrame(crd if crd else np.empty((0, 2)), index=idx, columns=["row", "col"]).astype(int),
            cell_type=pd.DataFrame(types, index=idx, columns=["cell_id"]),
            cell_deme=pd.DataFrame(demes_col, index=idx, columns=["deme_id"]),
        )
        return self.cell_data

    # --- plotting (shared, engine-agnostic) ----------------------------------
    def plot_muller(self, ax=None, **kwargs):
        from .. import viz
        return viz.plot_muller(self.traces, self.genotypes_parents, ax=ax, **kwargs)

    def plot_grid(self, color=None, ax=None, **kwargs):
        from .. import viz
        if self.cell_data is None or self.cell_data["cell_type"].shape[0] != self.get_tumor_size():
            self.make_cell_data()
        return viz.plot_grid(self.cell_data, self.grid_size, self.traces,
                             self.genotypes_parents, color=color, ax=ax, **kwargs)

    def get_genotype_frequencies(self, normalize=True):
        gids = [g for g in self.genotypes_counts if g not in normal_names]
        counts = np.array([self.genotypes_counts[g] for g in gids], dtype=float)
        if normalize and counts.sum() > 0:
            counts = counts / counts.sum()
        return gids, counts

    def get_gene_data(self, **kwargs):
        return self.selection.get_gene_data(**kwargs)

    # --- output --------------------------------------------------------------
    def write(self, output_path):
        """Write the canonical iscc layout (so isccsample/isccdata can consume it)."""
        Path(output_path).mkdir(parents=True, exist_ok=True)
        if self.traces:
            pd.DataFrame([t["genotypes_counts"] for t in self.traces]).fillna(0).to_csv(
                os.path.join(output_path, "trace_counts.csv"))
            pd.DataFrame([self.genotypes_parents]).to_csv(os.path.join(output_path, "parents.csv"))
            gids, _ = self.get_genotype_frequencies()
            pd.DataFrame(gids).to_csv(os.path.join(output_path, "genotypes.csv"))

        gene_data = self.selection.get_gene_data()
        Path(os.path.join(output_path, "gene_data")).mkdir(parents=True, exist_ok=True)
        for mat in gene_data:
            pd.DataFrame(gene_data[mat]).to_csv(os.path.join(output_path, "gene_data", f"{mat}.csv"))

        if self.cell_data is None:
            self.make_cell_data()
        Path(os.path.join(output_path, "cell_data")).mkdir(parents=True, exist_ok=True)
        for mat in self.cell_data:
            self.cell_data[mat].to_csv(os.path.join(output_path, "cell_data", f"{mat}.csv"))
