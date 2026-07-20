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
from ..components.epistasis import bits_to_events
from ..components.cell import CancerCell, EpithelialCell, StromalCell, ImmuneCell
from .glandular import bresenham_circumference, get_inside
from ..programs import ProgramModel
from ...constants import normal_names, DEFAULT_LAYOUT_SEED, LAYOUT_OFFSET_F8_PROGRAMS

CELLTYPES = ["cancer", "epithelial", "stromal", "immune"]


class GenotypeTumor:
    def __init__(self, config=None, seed=42, genome_params=None, selection_params=None,
                 cancer_cell_params=None, deme_params=None, spatial_params=None,
                 epithelial_cell_params=None, stromal_cell_params=None, immune_cell_params=None,
                 genome_mode="abstract", genome_spec=None,
                 update_mode="exact", tau=1.0, snapshot_every=1, microenv_params=None,
                 layout_seed=None, expression_params=None):
        self.seed = seed
        # EVOLUTION rng (per-run: spatial seeding; grow() draws its own fresh default_rng(seed+step)).
        self.rng = np.random.default_rng(seed)
        # GENOME LAYOUT rng (config-determined, SHARED across same-config runs): the gene-role
        # layout (Selection) and shared per-cell-type baseline expression. Defaulting layout_seed to
        # DEFAULT_LAYOUT_SEED makes any two same-config runs share their driver identities by
        # construction (comparability by default) — recurrence / cohort analysis is meaningful — while
        # they still differ in evolution. See DESIGN_cohort.md §1. Byte-identical to the previous
        # single-rng plumbing at the default seed 42 (structure_radius=0 never consumes self.rng at
        # construction, and evolution never reads it).
        self.layout_seed = DEFAULT_LAYOUT_SEED if layout_seed is None else layout_seed
        self.layout_rng = np.random.default_rng(self.layout_seed)
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
            microenv_params = cfg.get("microenv_params", microenv_params)
            expression_params = cfg.get("expression_params", expression_params)
            if cfg.get("layout_seed") is not None:
                self.layout_seed = cfg["layout_seed"]
                self.layout_rng = np.random.default_rng(self.layout_seed)
        # F8 microenvironment-driven expression (DESIGN_features §H): OPTIONAL and OFF by default
        # (absent -> None -> `make_cell_data` output is bit-identical to the base engine). When
        # enabled, a per-deme expression modifier (hypoxia field + cell-cell communication) is
        # applied at materialisation only; GROWTH is untouched, so a tumour is byte-identical with
        # F8 on or off at the same seed (the modifier draws from a dedicated rng). Fitness-coupling
        # of the microenvironment (hypoxia slowing division, etc.) is a FUTURE EXTENSION.
        self.microenv_params = microenv_params
        # R13 gene-program expression (DESIGN_expression.md): OPTIONAL and OFF by default, on exactly
        # the F8 discipline — absent -> None -> `make_cell_data` is bit-identical to the base engine,
        # and GROWTH never reads it at all, so a tumour is byte-identical with the program layer on or
        # off at a given seed. Programs are a READOUT: the genotype and the niche drive them, they
        # never feed back into fitness (that loop is R8b/R12-v3). Built after `self.selection` below,
        # since the model needs the gene count. See `iscc.tumor.programs`.
        self.expression_params = expression_params
        self.programs = None

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

        # ``layout_seed`` is handed over (as well as ``layout_rng``) so the optional epistasis network
        # can draw from its own layout SUB-STREAM — part of the shared landscape, but independent of
        # the gene-role layout drawn from layout_rng. See DESIGN_epistasis.md / constants.py.
        self.selection = Selection(n_segments=self.n_segments, segment_size=self.segment_size,
                                   segment_sizes=self.segment_sizes, rng=self.layout_rng,
                                   layout_seed=self.layout_seed, **selection_params)
        self.n_genes = self.selection.n_genes
        self._cancer_params = cancer_cell_params

        # R13 program layer. The dictionary (gene->program map, `loading`, regulators, `s_g`) is part
        # of the SHARED landscape and so draws from `layout_seed`'s program sub-stream — two patients
        # with the same config get the SAME programs, exactly as they already get the same oncogenes.
        # The per-cell `z` and each SNV's functional class are per-RUN events and draw from `seed`.
        if self.expression_params is not None:
            self.programs = ProgramModel(
                n_genes=self.n_genes, segment_sizes=self.selection.segment_sizes,
                layout_seed=self.layout_seed, run_seed=self.seed,
                expression_params=self.expression_params)

        # per-cell-type baseline expression (part of the SHARED landscape -> layout_rng, so a shared
        # cell state has the same baseline profile across patients; see make_celltype_exps note in tumor.py)
        self.celltype_exps = {}
        for ct in CELLTYPES:
            exp = self.layout_rng.beta(0.1, 1.0, size=self.n_genes)
            exp[self.selection.get_tsgs()] = 0.8
            exp[self.selection.get_oncogenes()] = 0.01
            self.celltype_exps[ct] = exp

        # F8 program designation: pick the hypoxia-responsive and CCI-receptor-target gene sets
        # (the ground-truth cell-extrinsic modules). A DEDICATED rng (not self.rng) keeps the
        # growth trajectory byte-identical whether F8 is on or off at a given seed.
        # WHICH genes respond to hypoxia is a property of the GENOME, not of a run, so the stream is
        # the LAYOUT one: two patients in a cohort must share their niche programs exactly as they
        # share their oncogenes. (This previously read `self.seed + 9973` — the per-run EVOLUTION
        # seed — which silently gave every patient a different hypoxia programme; F8 predates
        # `layout_seed` and was never migrated. The dedicated-stream intent was right, the seed
        # source was wrong.)
        self._hypoxia_genes = np.array([], dtype=int)
        self._cci_target_genes = np.array([], dtype=int)
        if self.microenv_params:
            prog_rng = np.random.default_rng(self.layout_seed + LAYOUT_OFFSET_F8_PROGRAMS)
            hyp = ((self.microenv_params.get("hypoxia") or {}) if isinstance(self.microenv_params, dict) else {})
            cci = ((self.microenv_params.get("cci") or {}) if isinstance(self.microenv_params, dict) else {})
            n_hyp = int(hyp.get("n_genes", 0))
            n_cci = int(cci.get("n_target_genes", 0))
            if n_hyp > 0:
                self._hypoxia_genes = prog_rng.choice(self.n_genes, size=min(n_hyp, self.n_genes),
                                                      replace=False)
            if n_cci > 0:
                self._cci_target_genes = prog_rng.choice(self.n_genes, size=min(n_cci, self.n_genes),
                                                         replace=False)

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
        # carrying_capacity is a real per-deme CAP now (DESIGN_crowding.md, Option A): crowding
        # death rises RELATIVE to each clone's own evolved division rate, so demes cap near K even
        # for fast-evolved clones. Setting it to None or 0 disables crowding entirely -> the
        # "well-mixed" regime (unbounded growth in a deme), used by the single-deme SISTEM benchmark.
        self.carrying_capacity = deme_params.get("carrying_capacity", 10)
        self._crowding = self.carrying_capacity is not None and self.carrying_capacity > 0
        # Positive fill/normalisation capacity: used for structure filling, immune seeding and the
        # microenvironment density fields. Falls back to 1 when crowding is off (no capacity).
        self._cap = int(self.carrying_capacity) if self._crowding else 1
        # Firmness of the cap: the crowding slope is steepened by (1 + crowding_margin) so the
        # per-deme fixed point sits at K/(1+margin) (slightly below K) and the restoring force above
        # K is firm rather than marginal. 0 recovers the plain fixed-point-at-K form. See _death_rate.
        self.crowding_margin = deme_params.get("crowding_margin", 0.1)
        # maximum_death_rate MUST be >= max_birth_rate for the cap to bind: a clone whose division
        # evolved up to max_birth_rate needs its crowding death to be able to reach (and exceed) that
        # rate. Default 1.0 (>= the 0.8 max_birth_rate default). A lower value re-opens the overfill bug.
        self.maximum_death_rate = deme_params.get("maximum_death_rate", 1.0)
        self.structure_radius = spatial_params.get("structure_radius", 0)
        # Number of founder cancer cells to seed (an established micro-lesion). A single founder
        # has P(extinction) ≈ death/division (~7% for the defaults) regardless of carrying
        # capacity, so a one-cell start makes runs/demos randomly cancer-free; seeding a small
        # cluster removes that founder bottleneck. Default 1 preserves prior behaviour.
        self.initial_cancer_cells = deme_params.get("initial_cancer_cells", 1)
        # Founder cells actually seeded: the requested cluster, capped by K when crowding is on
        # (can't over-seed a deme past its capacity) and left uncapped in the well-mixed regime.
        self._n_founder = max(1, min(self.initial_cancer_cells, self._cap)
                              if self._crowding else self.initial_cancer_cells)
        n_demes = self.grid_size * self.grid_size
        self.demes = [dict() for _ in range(n_demes)]
        self.deme_coords = [(i // self.grid_size, i % self.grid_size) for i in range(n_demes)]

        if self.structure_radius > 0:
            self._seed_structure()
        else:
            center = (self.grid_size // 2) * self.grid_size + (self.grid_size // 2)
            self._add(center, self.founder_id, self._n_founder)

        # Optional immune microenvironment: seed immune cells in every deme so that
        # cancer growing into them experiences local immune pressure (and so that
        # immunotherapy has a substrate to act on). The count scales with carrying
        # capacity (immune_density = immune cells per capacity unit) so the immune
        # fraction is not washed out once a deme fills with cancer. Static for now
        # (no division/migration yet).
        self.immune_density = spatial_params.get("immune_density", 0.0)
        n_immune = int(round(self.immune_density * self._cap))
        if n_immune > 0:
            imm = self._normal_genotype("immune")
            for i in range(n_demes):
                self._add(i, imm, n_immune)
        # Whether any immune cells exist (static: only seeded here). When false, the immune-killing
        # term is always zero, so the per-genotype death-rate can skip the O(#genotypes-in-deme)
        # immune sum entirely — a big win for the clone-heavy tau-leaping path (see _immune_fraction).
        self._has_immune = "immune" in self.genotypes

        self.deme_rates = np.array([self._deme_rate(i) for i in range(n_demes)], dtype=float)
        self.traces = []
        self.step = 0
        self.cell_data = None

    # --- genotype registry ---------------------------------------------------
    def _is_viable(self, rep):
        """Whether a freshly mutated genotype satisfies the CINner viability limits
        (``max_ploidy`` / ``max_cn`` / ``max_nullisomy`` / ``max_mut_drivers``; see
        Selection.update_viability). A non-viable daughter is REJECTED AT BIRTH: the division is
        still consumed, but it yields no cell and no genotype is registered.

        WHY REJECT-AT-BIRTH. The alternative is to check lazily, when a cell is next *sampled* for
        an event -- which is what the cell-level engine used to do. Both rules agree on the long-run
        dynamics: the division yields no surviving descendant either way, and the event is consumed
        either way, so the parent pays for it identically. They differ only in a transient: under
        the lazy rule the doomed cell occupies a carrying-capacity slot until it happens to be
        picked. That dwell time is not a modelled quantity -- it is set by the deme's total event
        rate, so it has no parameter behind it and no biological meaning -- and while it lasts the
        cell is counted by ``get_tumor_size`` and can be sampled into ``cell_data``, i.e. genomes
        that breach the configured limits leak into the emitted assay data. Rejecting at birth
        instead gives the invariant the limits are documented to provide: no cell breaching them
        ever exists. It is also the complete seam -- ``update_evolutionary_parameters`` (the only
        thing that computes viability) runs only inside ``Cell.mutate``, so a successful mutation is
        the ONLY way a genotype can become non-viable. Founders and normal cells are viable by
        construction, in this engine and the cell engine alike.

        BOTH ENGINES NOW REJECT AT BIRTH. ``Deme.apply_event`` applies the same rule for the
        cell-level engine (its ``sample_event`` viability check survives only as a backstop), so the
        limits are a true invariant everywhere rather than a property of this engine alone. That
        change was a no-op at the shipped defaults, where the limits are never reached -- see
        tests/test_count_engine.py::test_cell_engine_default_limits_are_a_noop.
        """
        return self.selection.update_viability(rep.genome_summary) != 0

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
                self._add(r * self.grid_size + c, epi, self._cap)
        circle = get_inside(border)
        occupied = set(border) | set(circle)

        in_border = bresenham_circumference(center, center, self.structure_radius - 1)
        in_border = [(r, c) for (r, c) in in_border if 0 <= r < self.grid_size and 0 <= c < self.grid_size]
        if in_border:
            pos = in_border[int(self.rng.choice(len(in_border)))]
            self._add(pos[0] * self.grid_size + pos[1], self.founder_id, self._n_founder)

        stroma = self._normal_genotype("stromal")
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if (r, c) not in occupied and not self.demes[r * self.grid_size + c]:
                    self._add(r * self.grid_size + c, stroma, self._cap)

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
    def _immune_fraction(self, deme, total=None):
        # No immune cells anywhere -> the fraction is identically zero; skip the per-genotype scan
        # (this is O(#genotypes-in-deme) and dominates the clone-heavy tau-leaping loop otherwise).
        if not getattr(self, "_has_immune", True):
            return 0.0
        if total is None:
            total = sum(deme.values())
        if total == 0:
            return 0.0
        immune = sum(cnt for gid, cnt in deme.items() if self.genotypes[gid].type == "immune")
        return immune / total

    def _death_rate(self, gid, deme_idx, total=None):
        """Cancer death rate = crowding-modulated baseline + local immune killing + treatment.

        Mirrors the corrected Deme.get_cancer_death_rate: immune killing is additive contact
        pressure (more local immune cells -> higher death, attenuated by immune resistance),
        and active therapy adds a death hazard (chemo/targeted) or strips immune resistance
        (immunotherapy, via the per-step _tx_* overrides). ``total`` (the deme's cell count) may be
        passed in when the caller already computed it, so a per-deme loop doesn't recompute it once
        per genotype.
        """
        rep = self.genotypes[gid]
        deme = self.demes[deme_idx]
        if total is None:
            total = sum(deme.values())
        base = rep.evolutionary_parameters["death_rate"]
        if self._crowding:
            # Density-dependent death RELATIVE to this clone's OWN (evolved) division rate
            # (DESIGN_crowding.md, Option A). The crowding slope is the clone's net growth rate
            # (div - base), steepened by (1 + crowding_margin), so at occupancy == K/(1+margin)
            # death == div and above it death > div — a true restoring force to the carrying
            # capacity even for a clone whose division has evolved up to max_birth_rate (the old
            # step-function `death_rate * K` capped at maximum_death_rate could not, because an
            # evolved division rate outran the absolute cap). maximum_death_rate must be >=
            # max_birth_rate (default 1.0) or the clamp below re-opens the overfill bug.
            div = rep.evolutionary_parameters["division_rate"]
            slope = max(0.0, div - base) * (1.0 + self.crowding_margin)
            death = base + slope * (total / self.carrying_capacity)
        else:
            # well-mixed regime (carrying_capacity None/0): no crowding ceiling -> unbounded growth.
            death = base
        death = min(death, self.maximum_death_rate)

        ir = self._tx_immune_resist.get(gid, rep.evolutionary_parameters["immune_resistance"])
        ir = min(max(ir, 0.0), 1.0)
        death += self._immune_prob_kill * self._immune_fraction(deme, total) * (1.0 - ir)

        death += self._tx_death_add.get(gid, 0.0)
        return death

    def _cancer_gids(self, deme):
        return [gid for gid in deme if self._is_cancer(gid)]

    def _deme_rate(self, deme_idx):
        """Total event rate of a deme. Only cancer genotypes are dynamic; normals are static."""
        deme = self.demes[deme_idx]
        total = sum(deme.values())
        rate = 0.0
        for gid in self._cancer_gids(deme):
            div = self.genotypes[gid].evolutionary_parameters["division_rate"]
            rate += deme[gid] * (div + self._death_rate(gid, deme_idx, total))
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
        total = sum(deme.values())
        weights = np.array([
            deme[gid] * (self.genotypes[gid].evolutionary_parameters["division_rate"]
                         + self._death_rate(gid, di, total))
            for gid in gids
        ])
        gid = gids[int(rng.choice(len(gids), p=weights / weights.sum()))]
        rep = self.genotypes[gid]
        div = rep.evolutionary_parameters["division_rate"]
        death = self._death_rate(gid, di, total)

        affected = [di]
        if rng.random() < death / (div + death):
            self._remove(di, gid, 1)                       # death
        else:
            disp = rep.evolutionary_parameters["dispersal_rate"]
            mut_prob = rep.mutation_rate / (rep.mutation_rate + disp)
            if rng.random() < mut_prob:
                child = rep.divide()
                if child.mutate(rng, self.selection):
                    # A daughter breaching the viability limits is never born (see _is_viable):
                    # the division is consumed but adds no cell.
                    if self._is_viable(child):
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
            total = sum(deme.values())
            for gid in sorted(self._cancer_gids(deme), key=lambda g: self.genotypes[g].ord):
                c = deme[gid]
                rep = self.genotypes[gid]
                div = rep.evolutionary_parameters["division_rate"]
                death = self._death_rate(gid, di, total)
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
        # mutate() spawns a new genotype (count 1) unless it breaches the viability limits, in
        # which case nothing is born; a saturated allele grows the parent in place.
        # This per-mutation genotype creation is intrinsic to the infinite-sites model and costs
        # exactly the same as in the exact engine (the separate #genotypes concern of §3).
        for di, gid, n in mutants:
            rep = self.genotypes[gid]
            for _ in range(n):
                child = rep.divide()
                if child.mutate(rng, self.selection):
                    # non-viable daughter: division consumed, no cell added (see _is_viable)
                    if self._is_viable(child):
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

    # --- F8 microenvironment fields (DESIGN_features §H) ---------------------
    # All operate on the regular grid_size x grid_size deme lattice (deme i at
    # (i//G, i%G)); computed once per make_cell_data (a snapshot), so tau-leaping-compatible.
    # NOTE (future extension): these modulate the expression READOUT only. Coupling the
    # microenvironment to FITNESS (hypoxia slowing division, etc.) is deliberately left for later.
    def _deme_density(self):
        """Per-deme occupancy = total cells / carrying capacity (drives O2 consumption)."""
        cc = self._cap
        return np.array([sum(d.values()) / cc for d in self.demes], dtype=float)

    def _emitter_density(self, emitter_type):
        """Per-deme density of a ligand-emitting cell type (e.g. immune)."""
        cc = self._cap
        out = np.zeros(len(self.demes))
        for i, deme in enumerate(self.demes):
            out[i] = sum(c for gid, c in deme.items()
                         if self.genotypes[gid].type == emitter_type) / cc
        return out

    def _o2_field(self, D=1.0, k=1.0, s=0.2, source="uniform", n_iter=500, tol=1e-5):
        """Steady-state O2 on the deme grid (BioFVM-style), returned as hypoxia = 1 - O2.

        Solves D∇²O2 + supply·(1-O2) - k·density·O2 = 0 by Jacobi relaxation with zero-flux edges.
        O2 ∈ [0,1] (a weighted average of neighbours and the supply target 1), so hypoxia ∈ [0,1].

        `source` sets where O2 comes from:
          * ``"uniform"`` — supplied everywhere (a well-vascularised-tissue assumption). Gives a
            hypoxic core only when the tumour has an oxygenated (empty/normal) margin.
          * ``"perfused"`` — supplied ONLY by non-cancer (perfused stroma / empty) tissue:
            supply ∝ (1 - cancer density). A solid cancer mass or a cancer-filled duct then develops
            a hypoxic core from the O2 diffusion limit (comedonecrosis in DCIS) even with no empty
            margin, because the vasculature does not live inside the tumour.
        """
        G = self.grid_size
        dens = self._deme_density().reshape(G, G)
        if source == "perfused":
            perfusion = np.clip(1.0 - self._emitter_density("cancer"), 0.0, 1.0).reshape(G, G)
            supply = s * perfusion
        else:
            supply = s
        nn = np.full((G, G), 4.0)
        nn[0, :] -= 1; nn[-1, :] -= 1; nn[:, 0] -= 1; nn[:, -1] -= 1
        O2 = np.ones((G, G))
        for _ in range(n_iter):
            nb = np.zeros((G, G))
            nb[1:, :] += O2[:-1, :]; nb[:-1, :] += O2[1:, :]
            nb[:, 1:] += O2[:, :-1]; nb[:, :-1] += O2[:, 1:]
            new = (D * nb + supply) / (D * nn + supply + k * dens)
            if np.max(np.abs(new - O2)) < tol:
                O2 = new
                break
            O2 = new
        return np.clip(1.0 - O2.ravel(), 0.0, 1.0)

    def _cci_field(self, emitter_type="immune", lengthscale=2.0):
        """Per-deme ligand signal = neighbourhood-averaged emitter density (Gaussian `lengthscale`)."""
        emit = self._emitter_density(emitter_type)
        coords = np.array(self.deme_coords, dtype=float)
        diff = coords[:, None, :] - coords[None, :, :]
        d2 = np.sum(diff * diff, axis=-1)
        L = max(float(lengthscale), 1e-9)
        W = np.exp(-d2 / (2.0 * L * L))
        wsum = W.sum(axis=1)
        return np.divide(W @ emit, wsum, out=np.zeros_like(emit), where=wsum > 0)

    def _microenv_deme_mod(self):
        """The F8 per-deme x gene expression modifier (n_demes x n_genes), or None if disabled.

        `mod[deme, hypoxia_genes] *= 1 + strength·hypoxia[deme]` and likewise for CCI target genes.
        Stores the ground-truth programs + fields on `self.microenv_truth` for validation/benchmarks.
        """
        mp = self.microenv_params
        if not mp:
            return None
        hyp = (mp.get("hypoxia") or {}) if isinstance(mp, dict) else {}
        cci = (mp.get("cci") or {}) if isinstance(mp, dict) else {}
        n_demes = len(self.demes)
        mod = np.ones((n_demes, self.n_genes))
        hypoxia = np.zeros(n_demes)
        cci_signal = np.zeros(n_demes)
        if len(self._hypoxia_genes) and float(hyp.get("strength", 0.0)) != 0.0:
            hypoxia = self._o2_field(D=float(hyp.get("o2_diffusion", 1.0)),
                                     k=float(hyp.get("o2_consumption", 1.0)),
                                     s=float(hyp.get("o2_supply", 0.2)),
                                     source=hyp.get("o2_source", "uniform"))
            mod[:, self._hypoxia_genes] *= (1.0 + float(hyp["strength"]) * hypoxia[:, None])
        if len(self._cci_target_genes) and float(cci.get("strength", 0.0)) != 0.0:
            cci_signal = self._cci_field(cci.get("emitter_type", "immune"),
                                         float(cci.get("lengthscale", 2.0)))
            mod[:, self._cci_target_genes] *= (1.0 + float(cci["strength"]) * cci_signal[:, None])
        self.microenv_truth = dict(
            hypoxia_genes=np.asarray(self._hypoxia_genes), cci_target_genes=np.asarray(self._cci_target_genes),
            hypoxia=hypoxia, cci=cci_signal)
        return mod

    # --- materialisation: counts -> per-cell matrices ------------------------
    def make_cell_data(self, cell_prefix="C", **kwargs):
        gene_names = self.selection.get_gene_names()
        onc_idx, tsg_idx = self.selection.get_oncogenes(), self.selection.get_tsgs()
        disp_idx, ir_idx, tr_idx = (self.selection.get_dispersal_genes(),
                                    self.selection.get_immune_resistant(),
                                    self.selection.get_treatment_resistant())
        snv_cache, cnv_cache, exp_cache, evo_cache = {}, {}, {}, {}
        # R13: per-genotype allele-resolved expression + the per-clone program drive (routes 1+2).
        # Both are per-CLONE, so they belong in this cache; the per-CELL `z` is drawn later.
        exp_p_cache, exp_m_cache, drive_cache = {}, {}, {}
        P = self.programs
        for gid in self.genotypes_counts:
            rep = self.genotypes[gid]
            snv = rep.get_snvs()
            snv_cache[gid] = snv
            cnv_cache[gid] = rep.get_cnvs()
            if P is None:
                # legacy path — untouched, so output is bit-identical when the layer is off
                if rep.type == "cancer":
                    rep.baseline_exp = self.celltype_exps["cancer"]
                    exp_cache[gid] = rep.get_exp(self.selection.mut_effects)
                else:
                    exp_cache[gid] = self.celltype_exps[rep.type]
            else:
                rep.baseline_exp = self.celltype_exps[rep.type]
                if rep.type == "cancer":
                    ep, em = rep.get_exp_alleles(dosage_sensitivity=P.dosage_sensitivity,
                                                 snv_exp_effect=P.snv_exp_effect,
                                                 dosage_saturation=P.dosage_saturation)
                    # Route 1 reads the EVOLVED per-clone phenotype, so whatever moved fitness —
                    # CINner drivers, and now R14's epistasis multiplier — propagates into the
                    # programs by construction. Route 2 reads the clone's SNV VAFs.
                    drive_cache[gid] = P.clone_drive(rep.evolutionary_parameters,
                                                     rep.baseline_rates, snv)
                else:
                    # Normal cells are diploid and unmutated: a flat allele split, no genotype drive.
                    half = self.celltype_exps[rep.type] / 2.0
                    ep, em = half.copy(), half.copy()
                    drive_cache[gid] = np.zeros(P.n_programs)
                drive_cache[gid] = drive_cache[gid] + P.celltype_bias(rep.type)
                exp_p_cache[gid], exp_m_cache[gid] = ep, em
                exp_cache[gid] = ep + em
            # per-genotype evolutionary parameters + unique-mutated-driver tallies
            evo = dict(rep.evolutionary_parameters)
            evo["n_mut_onc"] = int((snv[onc_idx] > 0).sum())
            evo["n_mut_tsg"] = int((snv[tsg_idx] > 0).sum())
            evo["n_mut_disp"] = int((snv[disp_idx] > 0).sum())
            evo["n_mut_ir"] = int((snv[ir_idx] > 0).sum())
            evo["n_mut_tr"] = int((snv[tr_idx] > 0).sum())
            evo_cache[gid] = evo

        # F8: per-deme expression modifier (None -> disabled -> exp is bit-identical to the base
        # engine). Cache the modified expression per (deme, genotype) since many cells share both.
        deme_mod = self._microenv_deme_mod()
        mod_exp_cache = {}

        # R13 route 3 — niche -> program: the F8 fields drive per-deme program activity, generalising
        # F8's hard-wired hypoxia/CCI gene sets (which still apply via `deme_mod`; the two routes
        # compose, see DESIGN_expression.md §3.1).
        niche_drive = None
        if P is not None and getattr(self, "microenv_truth", None) is not None:
            niche_drive = P.niche_drive({"hypoxia": self.microenv_truth["hypoxia"],
                                         "cci": self.microenv_truth["cci"]})

        rows_snv, rows_cnv, rows_exp, rows_evo, crd, types, demes_col, names = [], [], [], [], [], [], [], []
        rows_exp_p, rows_exp_m, rows_drive = [], [], []
        i = 0
        for deme_idx, deme in enumerate(self.demes):
            r, c = self.deme_coords[deme_idx]
            for gid in sorted(deme.keys(), key=lambda g: self.genotypes[g].ord):
                if deme_mod is None:
                    exp_row = exp_cache[gid]
                else:
                    exp_row = mod_exp_cache.get((deme_idx, gid))
                    if exp_row is None:
                        exp_row = exp_cache[gid] * deme_mod[deme_idx]
                        mod_exp_cache[(deme_idx, gid)] = exp_row
                if P is not None:
                    mod_row = 1.0 if deme_mod is None else deme_mod[deme_idx]
                    ep_row, em_row = exp_p_cache[gid] * mod_row, exp_m_cache[gid] * mod_row
                    drive_row = drive_cache[gid]
                    if niche_drive is not None:
                        drive_row = drive_row + niche_drive[deme_idx]
                for _ in range(deme[gid]):
                    rows_snv.append(snv_cache[gid]); rows_cnv.append(cnv_cache[gid])
                    rows_exp.append(exp_row); rows_evo.append(evo_cache[gid])
                    crd.append((r, c)); types.append(gid); demes_col.append(deme_idx)
                    names.append(f"{cell_prefix}{i}"); i += 1
                    if P is not None:
                        rows_exp_p.append(ep_row); rows_exp_m.append(em_row)
                        rows_drive.append(drive_row)

        # R13: apply the per-CELL program activity. `z` is per-cell (cycling is a cell state, not a
        # clone constant) while the genotype drive is per-clone — so the drive sets each clone's MEAN
        # and `activity_noise` supplies the within-clone spread. One matmul over all cells keeps this
        # affordable despite breaking the per-(deme, genotype) expression sharing.
        Z = None
        if P is not None and rows_exp_p:
            drive = np.asarray(rows_drive, dtype=float)
            Z = P.sample_z(drive, P.activity_rng())
            prog_mult = P.gene_multiplier(Z)                      # (n_cells, n_genes)
            ep = np.asarray(rows_exp_p) * prog_mult
            em = np.asarray(rows_exp_m) * prog_mult
            rows_exp_p, rows_exp_m = ep, em
            rows_exp = ep + em

        idx = pd.Index(names)
        empty = np.empty((0, self.n_genes))
        self.cell_data = dict(
            cell_evo=pd.DataFrame(rows_evo, index=idx),
            cell_snv=pd.DataFrame(np.array(rows_snv) if rows_snv else empty, index=idx, columns=gene_names),
            cell_cnv=pd.DataFrame(np.array(rows_cnv) if rows_cnv else empty, index=idx, columns=gene_names),
            # NB `len(...)`, not truthiness: with the program layer on, `rows_exp` is an ndarray.
            cell_exp=pd.DataFrame(np.array(rows_exp) if len(rows_exp) else empty, index=idx, columns=gene_names),
            cell_crd=pd.DataFrame(crd if crd else np.empty((0, 2)), index=idx, columns=["row", "col"]).astype(int),
            cell_type=pd.DataFrame(types, index=idx, columns=["cell_id"]),
            cell_deme=pd.DataFrame(demes_col, index=idx, columns=["deme_id"]),
        )
        # cell_rna_vaf (F7b): the EXPECTED allele FRACTION in RNA (not an observed VAF). With m
        # mutant + w wt copies at a locus and per-locus expression effect e (selection.mut_effects:
        # oncogene=2, TSG=0.5, else 1), the fraction of expression from mutant alleles is
        # (m·e·base)/(m·e·base + w·base) = (m·e)/(m·e + w) = (v·e)/(v·e + (1-v)) with v = DNA-VAF
        # (cell_snv). The per-gene baseline expression CANCELS in this fraction, which is why at a
        # neutral locus (e=1) the *expected fraction* equals DNA-VAF; e>1 inflates, e<1 deflates.
        # This is NOT what a caller observes: the OBSERVED scRNA-VAF is this fraction SAMPLED at a
        # depth equal to the gene's expression (the F3 UMI count, which carries the per-gene
        # baseline + cell-type + CNV + SNV-effect scaling). Per-gene expression varies enormously,
        # so most loci drop out or are noisy and the observed VAF does NOT match DNA-VAF at neutral
        # loci — the read layer (reads/rna.py) does the depth-aware sampling and the obs_fidelity
        # distortion; both deliberately live outside the engine.
        flat_eff = (np.concatenate(self.selection.mut_effects)
                    if self.selection.mut_effects else np.ones(self.n_genes))
        v = self.cell_data["cell_snv"].values
        num = v * flat_eff
        denom = num + (1.0 - v)
        rna_vaf = np.divide(num, denom, out=np.zeros_like(v, dtype=float), where=denom > 0)
        self.cell_data["cell_rna_vaf"] = pd.DataFrame(rna_vaf, index=idx, columns=gene_names)
        # WGD ground truth (DESIGN_focal_cna.md v1): the per-genotype `is_wgd` flag surfaced per cell,
        # so downstream CNA / BAF benchmarks (e.g. Numbat) can score WGD detection against the truth.
        # Gated on WGD being enabled — off -> the frame is absent and the base schema is unchanged
        # (the F8 discipline, mirroring cell_microenv below). `types` is the per-cell genotype id.
        if self._cancer_params is not None and self._cancer_params.get("wgd_rate", 0):
            self.cell_data["cell_wgd"] = pd.DataFrame(
                {"is_wgd": [bool(self.genotypes[g].is_wgd) for g in types]}, index=idx)
        # F8: surface the per-cell cell-extrinsic levels (ground truth for the intrinsic-vs-extrinsic
        # decomposition benchmark). Only added when F8 is enabled, so the base schema is unchanged.
        if deme_mod is not None:
            dcol = np.asarray(demes_col, dtype=int)
            hyp, cci = self.microenv_truth["hypoxia"], self.microenv_truth["cci"]
            self.cell_data["cell_microenv"] = pd.DataFrame(
                {"hypoxia_level": hyp[dcol] if dcol.size else np.array([]),
                 "cci_level": cci[dcol] if dcol.size else np.array([])}, index=idx)

        # R13: the allele layers + the RNA BAF, and the per-cell program activity. Only added when
        # the program layer is on, so the base schema is unchanged.
        if P is not None:
            if P.allele_specific:
                ep = np.asarray(rows_exp_p) if len(rows_exp_p) else empty
                em = np.asarray(rows_exp_m) if len(rows_exp_m) else empty
                tot = ep + em
                # BAF in RNA: the fraction of a gene's expression coming from the `p` homolog. 0.5 at
                # a balanced diploid locus; an allelic imbalance (an amplified or lost homolog, or an
                # NMD SNV in cis) pushes it away from 0.5. This is the layer Numbat / CalicoST read,
                # and it is exactly what summing p+m in `get_exp` used to destroy.
                baf = np.divide(ep, tot, out=np.full_like(tot, 0.5, dtype=float), where=tot > 0)
                self.cell_data["cell_exp_p"] = pd.DataFrame(ep, index=idx, columns=gene_names)
                self.cell_data["cell_exp_m"] = pd.DataFrame(em, index=idx, columns=gene_names)
                self.cell_data["cell_rna_baf"] = pd.DataFrame(baf, index=idx, columns=gene_names)
            if Z is not None and P.n_programs:
                self.cell_data["cell_program"] = pd.DataFrame(
                    Z, index=idx, columns=list(P.dictionary.program_names))
            self.program_truth = P.truth()
        return self.cell_data

    # --- plotting (shared, engine-agnostic) ----------------------------------
    def plot_muller(self, ax=None, by_drivers=False, **kwargs):
        """Muller plot. ``by_drivers=True`` colours by distinct DRIVER-mutation combinations (Noble's
        demon convention) instead of by genotype, collapsing passenger-only diversity; combine with
        ``min_freq`` for a size threshold. See ``viz.plot_muller``."""
        from .. import viz
        driver_map = self._driver_signatures() if by_drivers else None
        return viz.plot_muller(self.traces, self.genotypes_parents, ax=ax,
                               driver_map=driver_map, **kwargs)

    def _driver_signatures(self):
        """Map every cancer genotype id -> a hashable signature of its mutated DRIVER genes (the
        oncogene/TSG loci, ``driver_types != 0``, carrying an SNV). Covers ancestors too (needed to
        build the driver-clone ancestry). Genotypes sharing a signature are one 'driver clone'."""
        dts = getattr(self.selection, "driver_types", None)
        if not dts:
            return {}
        driver_idx = np.flatnonzero(np.concatenate(dts) != 0)
        if driver_idx.size == 0:
            return {}
        out = {}
        for gid, rep in self.genotypes.items():
            if gid in normal_names or not hasattr(rep, "get_snvs"):
                continue
            vafs = rep.get_snvs()
            out[str(gid)] = tuple(int(i) for i in driver_idx[vafs[driver_idx] > 0])
        return out

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

    # --- epistasis ground truth (R14, DESIGN_epistasis.md §4) ----------------
    def epistasis_ground_truth(self):
        """The planted network — the answer key for MHN/TreeMHN/CBN/REVOLVER edge + order recovery.

        Returns the true ``E`` matrix, the marginal effects ``beta``, the interaction edges, the
        conjunctive dependency DAG, the mutually-exclusive pairs and the event->gene modules; or
        ``None`` when epistasis is off. Shared by every patient of a cohort (drawn from the layout
        stream), which is what makes pooling patients to recover ONE network well-posed.
        """
        if self.selection.epistasis is None:
            return None
        return self.selection.epistasis.ground_truth()

    def event_table(self):
        """Per-clone event sets and the ORDER this lineage acquired them (empty when epistasis is off).

        One row per cancer genotype: its cell count, the events it carries, and the realised order
        along its lineage — the per-patient ordering ground truth. ``event_groups`` is the truthful
        form (one tuple per mutating division; events inside a group are TIED, having been acquired
        together, so no order between them exists); ``event_order`` is that flattened for exports
        needing a linear path. **Score ordering against ``event_groups``** — ``event_order`` breaks
        ties arbitrarily.
        """
        net = self.selection.epistasis
        rows = []
        cols = ["genotype_id", "n_cells", "events", "event_order", "event_groups"]
        if net is None:
            return pd.DataFrame(columns=cols)
        for gid, count in self.genotypes_counts.items():
            if gid in normal_names:
                continue
            rep = self.genotypes[gid]
            rows.append(dict(genotype_id=gid, n_cells=int(count),
                             events=tuple(bits_to_events(rep.event_bits, net.n_events)),
                             event_order=tuple(rep.event_order),
                             event_groups=tuple(rep.event_groups)))
        return pd.DataFrame(rows, columns=cols)

    # --- operating-envelope QC (read-only; DESIGN_operating_envelope.md) ------
    def diagnose(self, thresholds=None, verbose=False):
        """Flag degenerate ("crappy") tumour regimes after growth, with actionable hints.

        Computes phenotype metrics (size, clonal diversity, TMB, clone spatial confinement,
        fraction-genome-altered, hypoxia core–rim contrast, 1/f fit) and checks each against an
        overridable threshold: extinct / monoclonal / hypermutated / well-mixed /
        no-microenvironment-gradient / CNA-runaway / trivial-genome. Returns a ``TumorDiagnosis``.
        Read-only — it never draws from ``self.rng`` or changes the counts, so it cannot alter
        simulation output (like the F8 ground truth)."""
        from ..diagnostics import diagnose
        return diagnose(self, thresholds=thresholds, verbose=verbose)

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
