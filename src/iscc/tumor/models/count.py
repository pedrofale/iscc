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
                 genome_mode="abstract", genome_spec=None,
                 update_mode="exact", tau=1.0, snapshot_every=1):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.type = "genotype"

        # Update mode (DESIGN_scalability §7). "exact" = the reference one-birth/death-per-update
        # Gillespie engine (default, unchanged). "tau" = tau-leaping: advance ALL clones once per
        # discrete generation of length `tau` by drawing Poisson(rate*count*tau) births/deaths per
        # (deme, genotype), so wall-time scales with #clones x #generations rather than #cells.
        # `snapshot_every` records a full per-clone count snapshot every k generations (so
        # plot_muller/plot_grid keep working, now on a real-time x-axis -- see self.trace_times).
        self.update_mode = update_mode
        self.tau = tau
        self.snapshot_every = snapshot_every
        self.time = 0.0
        self.trace_times = []

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
            self.update_mode = cfg.get("update_mode", update_mode)
            self.tau = cfg.get("tau", tau)
            self.snapshot_every = cfg.get("snapshot_every", snapshot_every)

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
        # Number of founder cancer cells to seed (an established micro-lesion). A single founder
        # has P(extinction) ≈ death/division (~7% for the defaults) regardless of carrying
        # capacity, so a one-cell start makes runs/demos randomly cancer-free; seeding a small
        # cluster removes that founder bottleneck. Default 1 preserves prior behaviour.
        self.initial_cancer_cells = deme_params.get("initial_cancer_cells", 1)
        n_demes = self.grid_size * self.grid_size
        self.demes = [dict() for _ in range(n_demes)]
        self.deme_coords = [(i // self.grid_size, i % self.grid_size) for i in range(n_demes)]

        if self.structure_radius > 0:
            self._seed_structure()
        else:
            center = (self.grid_size // 2) * self.grid_size + (self.grid_size // 2)
            self._add(center, self.founder_id, max(1, min(self.initial_cancer_cells, self.carrying_capacity)))

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
            self._add(pos[0] * self.grid_size + pos[1], self.founder_id,
                      max(1, min(self.initial_cancer_cells, self.carrying_capacity)))

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
        if self.update_mode == "tau":
            return self._grow_tau(n_steps=n_steps, seed=seed, treatment=treatment, **kwargs)
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

    # --- tau-leaping (generation-batched clonal update, DESIGN §7) -----------
    def _tau_substep(self, rng, dt):
        """One synchronous tau-leap of length `dt`.

        Per (deme, genotype) with count c, draw the two reaction channels independently from
        the CURRENT (pre-step) state -- births ~ Poisson(division_rate*c*dt) and
        deaths ~ Poisson(death_rate*c*dt) -- then split births into a mutation branch
        (in-place division, attempts a mutation exactly as the exact engine) and a dispersal
        branch (daughter placed in a neighbour deme) by Binomial(n_births, mut_prob), and apply
        everything in batch. The per-channel rates match the exact engine's
        (event rate = c*(div+death), death share = death/(div+death)), so the two engines agree
        in distribution as dt -> 0. Genotype ids are iterated in creation-ordinal order and all
        randomness comes from the seeded `rng`, so runs are reproducible. Death rates are read
        from current deme totals, so carrying-capacity crowding self-limits across substeps.
        """
        deaths, dispersals, mutants = [], [], []
        for di in range(len(self.demes)):
            deme = self.demes[di]
            if not deme:
                continue
            for gid in sorted(self._cancer_gids(deme), key=lambda g: self.genotypes[g].ord):
                c = deme[gid]
                rep = self.genotypes[gid]
                div = rep.evolutionary_parameters["division_rate"]
                death = self._death_rate(gid, di)
                disp = rep.evolutionary_parameters["dispersal_rate"]
                n_div = int(rng.poisson(div * c * dt))
                n_death = int(rng.poisson(death * c * dt))
                if n_death:
                    deaths.append((di, gid, n_death))
                if n_div:
                    denom = rep.mutation_rate + disp
                    mut_prob = rep.mutation_rate / denom if denom > 0 else 0.0
                    n_mut = int(rng.binomial(n_div, mut_prob))
                    n_disp = n_div - n_mut
                    if n_mut:
                        mutants.append((di, gid, n_mut))
                    if n_disp:
                        dispersals.append((di, gid, n_disp))

        # deaths first, capped at the pre-step count so a clone never goes negative
        for di, gid, n in deaths:
            n = min(n, self.demes[di].get(gid, 0))
            if n:
                self._remove(di, gid, n)
        # mutation-branch births: each is one division that attempts a mutation. A successful
        # mutate() spawns a new genotype (count 1); a saturated allele grows the parent in place.
        # This per-mutation genotype creation is intrinsic to the infinite-sites model and costs
        # exactly the same as in the exact engine (the separate #genotypes concern of §3).
        for di, gid, n in mutants:
            rep = self.genotypes[gid]
            for _ in range(n):
                child = rep.divide()
                if child.mutate(rng, self.selection):
                    self._register(child)
                    self.genotypes_parents[child.genotype_id] = rep.genotype_id
                    self._add(di, child.genotype_id, 1)
                else:
                    self._add(di, gid, 1)
        # dispersal-branch births: same genotype, one daughter into a uniformly random neighbour
        for di, gid, n in dispersals:
            nbrs = self._neighbors(di)
            if not nbrs:
                self._add(di, gid, n)
                continue
            idx = rng.integers(0, len(nbrs), size=n)
            u, cnts = np.unique(idx, return_counts=True)
            for k, cnt in zip(u, cnts):
                self._add(nbrs[int(k)], gid, int(cnt))

    def _tau_generation(self, rng, tau):
        """Advance the whole tumour by one generation of length `tau`, adaptively sub-stepping so
        the largest single-cell event probability per substep stays in the accurate Poisson
        regime (this also prevents carrying-capacity overshoot, since death rates are re-read each
        substep). Records a full per-clone snapshot every `snapshot_every` generations."""
        ACCURACY = 0.34  # keep rate*dt <= this so Poisson tau-leaping stays accurate
        max_cell_rate = 0.0
        for gid in self.genotypes_counts:
            if self._is_cancer(gid):
                ep = self.genotypes[gid].evolutionary_parameters
                max_cell_rate = max(max_cell_rate,
                                    ep["division_rate"] + ep["dispersal_rate"])
        max_cell_rate += self.maximum_death_rate
        n_sub = max(1, int(np.ceil(max_cell_rate * tau / ACCURACY)))
        dt = tau / n_sub
        for _ in range(n_sub):
            self._tau_substep(rng, dt)
        self.step += 1
        self.time += tau
        if self.step % self.snapshot_every == 0:
            self.traces.append(dict(genotypes_counts=dict(self.genotypes_counts)))
            self.trace_times.append(self.time)

    def _grow_tau(self, n_steps=1000, seed=None, treatment=None, tau=None, **kwargs):
        """Tau-leaping growth: `n_steps` is the number of generations to advance. A full per-clone
        snapshot is recorded every `snapshot_every` generations, so plot_muller/plot_grid keep
        working unchanged -- now on a REAL-TIME x-axis (self.trace_times, in generation units)."""
        if seed is None:
            seed = self.seed
        if tau is None:
            tau = self.tau
        rng = np.random.default_rng(seed + self.step)
        # snapshot at t=0 (matches the exact engine's initial trace)
        self.traces.append(dict(genotypes_counts=dict(self.genotypes_counts)))
        self.trace_times.append(self.time)
        prev_dose = 0.0
        for _ in range(n_steps):
            if self.get_cancer_size() == 0:
                break
            if treatment is not None:
                dose = treatment.get_dosage(self.step, self.get_tumor_size())
                self._apply_treatment(treatment, dose)
                prev_dose = dose
            self._tau_generation(rng, tau)
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
        # cell_rna_vaf (F7b): allele-expression-weighted true RNA-VAF, the alt fraction the scRNA
        # reads should carry. With m mutant + w wt copies at a locus and per-locus expression
        # effect e (selection.mut_effects: oncogene=2, TSG=0.5, else 1), the fraction of expression
        # from mutant alleles is (m·e)/(m·e + w) = (v·e)/(v·e + (1-v)) with v = DNA-VAF (cell_snv) —
        # copy number cancels. e=1 -> RNA-VAF == DNA-VAF; e>1 inflates, e<1 deflates (that
        # divergence + expression gating is why scRNA SNV calling is hard). obs_fidelity (the read
        # layer, reads/rna.py) distorts this further; it deliberately does NOT live in the engine.
        flat_eff = (np.concatenate(self.selection.mut_effects)
                    if self.selection.mut_effects else np.ones(self.n_genes))
        v = self.cell_data["cell_snv"].values
        num = v * flat_eff
        denom = num + (1.0 - v)
        rna_vaf = np.divide(num, denom, out=np.zeros_like(v, dtype=float), where=denom > 0)
        self.cell_data["cell_rna_vaf"] = pd.DataFrame(rna_vaf, index=idx, columns=gene_names)
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
