"""
Gene programs: the expression backbone (R13) and the cell-state model (R12).

`DESIGN_expression.md` §3-§4 and `DESIGN_celltrajectory.md` describe ONE object from two sides, and
this module is that object — build it once, use it twice. A cell's expression state is a vector of
activities ``z = (z_1 … z_K)`` over K gene PROGRAMS (the recurrent meta-programs: cell cycle,
hypoxia, EMT, stress, stemness…; Gavish & Tirosh, Nature 2023), and expression is

    exp_{g,a} = base_{type,g}/2 · exp(Σ_k z_k·loading_{k,g}) · dosage(CN_{g,a}; s_g) · snv(class_{g,a}) · niche_g

R12 owns how ``z`` MOVES (differentiation hierarchy, landscape deformation, niche); R13 owns
``z`` → counts plus the gene-level dosage/SNV overlays. This module provides the shared pieces: the
program dictionary (which genes each program loads on, and how strongly), the per-cell ``z`` sampler,
and the genotype/niche coupling that biases ``z``.

WHY THIS EXISTS — non-circularity. The DNA↔RNA integration tools (clonealign, inferCNV, Numbat,
cardelino, PhylEx) all INVERT a genotype→expression law. Before this module iscc's forward model was
approximately the law they assume: linear dosage, genes independent, alleles summed. A benchmark
against that is partly circular. The load-bearing property here is that **programs ⟂ CNAs**:
programs are FUNCTIONAL gene sets scattered across the genome, CNAs are CONTIGUOUS segments. That
orthogonality is what makes the benchmarks non-trivial, and ``program_genomic_scatter`` is the knob
that operationalises it (turn it down to build a deliberately CNA-mimicking program as a control).

READOUT-ONLY. Programs are driven BY the genotype and the niche; they never feed back into fitness.
Program→fitness is an eco-evolutionary loop (R8b / R12-v3) and is deliberately out of scope — this
keeps the F8 discipline, and it is what lets the whole layer stay a pure materialisation-time
concern: growth is byte-identical whether the program layer is on or off at a given seed.

STREAMS. Everything that is a property of the GENOME/LANDSCAPE — the gene→program map, `loading`,
the program-regulator assignment, and the per-gene dosage sensitivity `s_g` — is drawn from the
LAYOUT stream (`layout_seed + LAYOUT_OFFSET_PROGRAMS`, spawned into independent sub-streams per
component) so that two patients sharing a config share their programs, exactly as they already share
their oncogenes. Everything that is an EVENT — the per-cell `z` noise, a given SNV's functional class
— is drawn from the per-run evolution seed. See `constants.py`'s layout sub-stream registry.
"""
import numpy as np

from ..constants import LAYOUT_OFFSET_PROGRAMS

# Offsets into the per-run EVOLUTION stream (`seed + OFFSET`), for the draws that are properties of a
# RUN rather than of the landscape. Distinct from the layout registry in `constants.py`.
RUN_OFFSET_ACTIVITY = 5501     # the per-cell activity vector `z` (drawn at materialisation)
RUN_OFFSET_SNV_CLASS = 7717    # each SNV's functional class (LoF / missense / splice / silent)

# The interpretable program names. `seeded_programs` anchors program k to a known signature so that
# the phenotype/niche maps below can refer to programs by NAME rather than by index — otherwise
# "which program is proliferation" would depend on the draw. Programs beyond the seeded ones are
# unnamed ("background_1", …) and are the generic co-expression background.
DEFAULT_SEEDED_PROGRAMS = ("proliferation", "emt", "hypoxia", "drug_resistance", "immune_evasion")

# Route 1 (the DEFAULT coupling): each evolved per-clone phenotype drives one program. The chain is
# mutation → (the existing CINner fitness model) → phenotype → program → expression, so a mutation
# that raises the division rate RAISES the proliferation program by construction, with no new
# gene-tagging machinery. See DESIGN_expression.md §3.1.
# Note that MULTIPLE phenotypes may map to the SAME program — `clone_drive` sums their contributions
# (`drive[k] += ...`). The invasive `emt` program is driven by BOTH `breach` (the compartment-selection
# invasion trait, DESIGN_phenotype_plasticity.md §2 / DESIGN_ductal_field.md §5) and `dispersal_rate`.
# `breach` is the biologically apt, always-live genetic arm of the invasive phenotype; `dispersal_rate`
# is retained but is INERT whenever `prop_dispersal == 0` (dispersal never varies ⇒ zero fold-change ⇒
# no drive), which is the default in the structured configs — so relying on it alone gives the emt
# program no genetic signal. Keeping both means emt has a robust genetic arm regardless of prop_dispersal.
DEFAULT_PHENOTYPE_PROGRAM_MAP = {
    "division_rate": "proliferation",
    "dispersal_rate": "emt",
    "breach": "emt",
    "treatment_resistance": "drug_resistance",
    "immune_resistance": "immune_evasion",
}

# Route 3: which niche field drives which program (generalises F8's hypoxia/CCI gene sets).
# Available niche fields: the F8 fields "hypoxia" / "cci", and — in a structured (gland) tumour — the
# compartment fields "epithelial" / "stromal" (per-deme live fraction; DESIGN_phenotype_plasticity.md
# §2 Part D). Mapping "epithelial" -> "emt" makes the invasive program env-responsive to the
# epithelial interface, which is the genetic-vs-niche expression confound v1 ships.
DEFAULT_NICHE_PROGRAM_MAP = {"hypoxia": "hypoxia"}

# The SNV functional classes (axis B). `silent` and `missense` leave mRNA level alone (a missense
# changes protein ACTIVITY, not abundance — which is exactly why it is invisible to expression-based
# clone assignment); `lof` triggers nonsense-mediated decay on its allele; `splice` shifts expression.
SNV_CLASSES = ("silent", "missense", "splice", "lof")


def _mean_sd(value, default_mean, default_sd):
    """Accept a knob written as a scalar, a (mean, sd) pair, or a {'mean':, 'sd':} mapping."""
    if value is None:
        return float(default_mean), float(default_sd)
    if isinstance(value, dict):
        return float(value.get("mean", default_mean)), float(value.get("sd", default_sd))
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 1:
            return float(value[0]), float(default_sd)
        return float(value[0]), float(value[1])
    return float(value), 0.0


class ProgramDictionary:
    """The gene × program landscape: which genes each program loads on, and how strongly.

    A property of the GENOME, so it is drawn from the layout stream and is IDENTICAL across two runs
    (patients) that share a config — the same requirement the shared driver landscape already meets.

    Parameters mirror `DESIGN_expression.md` §4.1 (`program_params`):

    n_programs (K)
        How many programs exist.
    n_genes_per_program
        Genes each program loads on. An int, or a ``[lo, hi]`` pair drawn uniformly per program.
    program_overlap
        Expected fraction of a program's genes that are SHARED with an already-drawn program.
        0 = disjoint modules; high = entangled programs (harder to identify — real programs overlap).
    loading_strength
        Effect size: the log-fold a program imposes on its genes. Scalar, ``(mean, sd)`` or
        ``{'mean':, 'sd':}``.
    loading_sparsity
        Shape of the loading distribution WITHIN a program, as the sigma of a mean-1 lognormal
        multiplier. 0 = every gene loads equally; larger = heavy-tailed (a few strong marker genes
        plus a long tail of weak ones), which is what real programs look like.
    program_genomic_scatter
        Fraction of a program's genes drawn SCATTERED genome-wide (the default, and the realistic
        choice) rather than as a CONTIGUOUS positional block. **This is the knob that operationalises
        programs ⟂ CNAs.** At 1.0 a program is functional and positionally unstructured, so it is
        distinguishable from a copy-number segment; at 0.0 the program is a contiguous block that
        MIMICS a CNA — the deliberate control for "can the tool tell a program from a CNA?".
    program_signs
        ``"up"`` = programs only up-regulate their genes; ``"bidirectional"`` = each gene is up- or
        down-regulated (a real program has both).
    seeded_programs
        Names anchored to programs 0…n-1 so the phenotype/niche maps can refer to them by name.
    """

    def __init__(self, n_genes, segment_sizes, rng, n_programs=8, n_genes_per_program=30,
                 program_overlap=0.1, loading_strength=None, loading_sparsity=1.0,
                 program_genomic_scatter=1.0, program_signs="up", seeded_programs=None):
        self.n_genes = int(n_genes)
        self.n_programs = int(n_programs)
        self.segment_sizes = [int(s) for s in segment_sizes]
        self._seg_offsets = np.concatenate([[0], np.cumsum(self.segment_sizes)]).astype(int)
        self.program_overlap = float(program_overlap)
        self.program_genomic_scatter = float(program_genomic_scatter)
        self.program_signs = program_signs
        self.loading_sparsity = float(loading_sparsity)
        self.loading_strength_mean, self.loading_strength_sd = _mean_sd(loading_strength, 1.0, 0.3)

        if seeded_programs is None:
            seeded_programs = list(DEFAULT_SEEDED_PROGRAMS)
        seeded = [str(s) for s in seeded_programs][:self.n_programs]
        # Programs past the seeded ones are deliberately anonymous — generic co-expression background,
        # the structure NO phenotype and NO niche field drives. Named "background_N" rather than
        # "program_N" so it reads as intentional, and not after a real meta-program, which would
        # claim a biological identity they have not been calibrated to.
        self.program_names = seeded + [f"background_{k}"
                                       for k in range(1, self.n_programs - len(seeded) + 1)]
        self.program_index = {name: k for k, name in enumerate(self.program_names)}

        self.loading = np.zeros((self.n_programs, self.n_genes))
        self.program_genes = []
        used = np.array([], dtype=int)
        for k in range(self.n_programs):
            n_k = self._n_genes_for(n_genes_per_program, rng)
            genes = self._draw_genes(n_k, used, rng)
            self.program_genes.append(genes)
            self.loading[k, genes] = self._draw_loadings(len(genes), rng)
            used = np.union1d(used, genes)

    def _n_genes_for(self, spec, rng):
        if isinstance(spec, (list, tuple, np.ndarray)):
            lo, hi = int(spec[0]), int(spec[-1])
            n = int(rng.integers(lo, hi + 1)) if hi > lo else lo
        else:
            n = int(spec)
        return max(0, min(n, self.n_genes))

    def _draw_genes(self, n_k, used, rng):
        """Pick this program's genes: some SHARED with earlier programs (`program_overlap`), the
        rest fresh — and each fresh gene either scattered genome-wide or drawn into a contiguous
        positional block (`program_genomic_scatter`)."""
        n_shared = 0
        if len(used) and self.program_overlap > 0:
            n_shared = min(int(rng.binomial(n_k, min(self.program_overlap, 1.0))), len(used))
        shared = rng.choice(used, size=n_shared, replace=False) if n_shared else np.array([], dtype=int)

        n_fresh = n_k - len(shared)
        taken = set(shared.tolist())
        fresh = []
        # Split the fresh genes between the scattered (functional) and clustered (CNA-mimicking) modes.
        n_scattered = int(rng.binomial(n_fresh, np.clip(self.program_genomic_scatter, 0.0, 1.0)))
        n_clustered = n_fresh - n_scattered

        if n_clustered > 0:
            # A contiguous run inside one segment — the positional structure a CNA would impose.
            seg = int(rng.integers(0, len(self.segment_sizes)))
            size = self.segment_sizes[seg]
            run = min(n_clustered, size)
            start = int(rng.integers(0, max(1, size - run + 1)))
            block = self._seg_offsets[seg] + np.arange(start, start + run)
            for g in block:
                if g not in taken:
                    taken.add(int(g)); fresh.append(int(g))

        # Scattered genes (plus any the clustered block could not supply): uniform genome-wide.
        need = n_k - len(shared) - len(fresh)
        if need > 0:
            pool = np.setdiff1d(np.arange(self.n_genes), np.fromiter(taken, dtype=int, count=len(taken)))
            if len(pool):
                pick = rng.choice(pool, size=min(need, len(pool)), replace=False)
                fresh.extend(int(g) for g in pick)

        genes = np.concatenate([shared.astype(int), np.array(fresh, dtype=int)]) if (len(shared) or fresh) \
            else np.array([], dtype=int)
        return np.sort(genes.astype(int))

    def _draw_loadings(self, n, rng):
        """Loading magnitudes: |Normal(mean, sd)| shaped by a mean-1 lognormal so that a program has a
        few strong markers and many weak genes (`loading_sparsity`), optionally signed."""
        if n == 0:
            return np.array([])
        mag = np.abs(rng.normal(self.loading_strength_mean, self.loading_strength_sd, size=n))
        s = self.loading_sparsity
        if s > 0:
            # mean-1 lognormal: E[w] = exp(mu + s^2/2) = 1 when mu = -s^2/2.
            mag = mag * rng.lognormal(mean=-0.5 * s * s, sigma=s, size=n)
        if self.program_signs == "bidirectional":
            mag = mag * rng.choice([-1.0, 1.0], size=n)
        return mag

    def gene_program_map(self):
        """gene index -> list of program indices loading on it (the ground-truth membership map)."""
        out = {}
        for k, genes in enumerate(self.program_genes):
            for g in genes:
                out.setdefault(int(g), []).append(k)
        return out


class ProgramModel:
    """Program dictionary + genotype/niche coupling + the per-cell `z` sampler + the gene overlays.

    This is the object `count.py` holds when `expression_params` is supplied. It is constructed once
    per tumour (the landscape) and queried at materialisation (the per-cell draw).

    All of it is OFF unless `expression_params` is given, and each sub-block (`program_params`,
    `dosage_params`, `snv_effect_params`) is independently optional — a missing block is the identity.
    """

    def __init__(self, n_genes, segment_sizes, layout_seed, run_seed=0, expression_params=None):
        ep = dict(expression_params or {})
        self.n_genes = int(n_genes)
        self.segment_sizes = [int(s) for s in segment_sizes]
        self.run_seed = int(run_seed)

        self.program_params = ep.get("program_params")
        self.activity_params = dict(ep.get("activity_params") or {})
        self.coupling_params = dict(ep.get("coupling_params") or {})
        self.dosage_params = ep.get("dosage_params")
        self.snv_effect_params = ep.get("snv_effect_params")

        # Independent LAYOUT sub-streams per landscape component. Spawning (rather than drawing all
        # three from one generator) is what makes the components independent: changing `n_programs`
        # must not reshuffle `s_g`, just as it must not reshuffle the oncogenes (which live on the
        # untouched base layout stream). See constants.py.
        ss = np.random.SeedSequence(int(layout_seed) + LAYOUT_OFFSET_PROGRAMS)
        dict_ss, reg_ss, sg_ss = ss.spawn(3)

        # --- the program dictionary (optional) ---
        self.dictionary = None
        if self.program_params is not None:
            self.dictionary = ProgramDictionary(n_genes, segment_sizes,
                                                np.random.default_rng(dict_ss),
                                                **dict(self.program_params))
        # --- route 2: direct program regulators (drawn even so they are stable) ---
        self._regulator_matrix = None
        self.regulator_genes = {}
        if self.dictionary is not None:
            self._draw_regulators(np.random.default_rng(reg_ss))

        # --- axis A: per-gene dosage sensitivity s_g (optional) ---
        self.dosage_sensitivity = None
        self.dosage_saturation = None
        self.allele_specific = False
        if self.dosage_params is not None:
            dp = dict(self.dosage_params)
            mean = float(dp.get("dosage_sensitivity_mean", 0.7))
            sd = float(dp.get("dosage_sensitivity_sd", 0.25))
            self.dosage_saturation = dp.get("dosage_saturation")
            self.allele_specific = bool(dp.get("allele_specific", False))
            # s_g ∈ [0, 1]: 0 = fully buffered/compensated (copy number invisible in RNA), 1 = full
            # linear dosage (the law the tools assume). Most genes sit in between — that partial,
            # PER-GENE buffering is precisely the confounder that makes clonealign/inferCNV a fair
            # test rather than a tautology.
            rng_sg = np.random.default_rng(sg_ss)
            self.dosage_sensitivity = np.clip(rng_sg.normal(mean, sd, size=self.n_genes), 0.0, 1.0)

        # --- axis B: per-SNV functional class (optional; EVENT-level -> run stream) ---
        self.snv_class = None
        self.snv_exp_effect = None
        if self.snv_effect_params is not None:
            self._draw_snv_classes()

    # ---------------------------------------------------------------- landscape draws
    def _draw_regulators(self, rng):
        """Route 2: tag a sparse subset of genes as DIRECT program regulators — mutating one shifts
        `z` for its target programs with NO fitness change (R12's plasticity cases: differentiation
        block, de-differentiation, an aberrant program switched on).

        Sparse on purpose: if most drivers moved a program, every clone would be its own expression
        state and clone-vs-state clustering would be trivially easy — the opposite of the benchmark.
        """
        cp = self.coupling_params
        prop = float(cp.get("prop_program_regulator", 0.0))
        n_per = int(cp.get("n_programs_per_regulator", 1))
        K = self.dictionary.n_programs
        if prop <= 0 or K == 0:
            return
        is_reg = rng.random(self.n_genes) < prop
        genes = np.where(is_reg)[0]
        if not len(genes):
            return
        mat = np.zeros((K, self.n_genes))
        for g in genes:
            targets = rng.choice(K, size=min(n_per, K), replace=False)
            # A regulator can activate or repress its target program.
            signs = rng.choice([-1.0, 1.0], size=len(targets))
            mat[targets, g] = signs
            self.regulator_genes[int(g)] = [(int(t), float(s)) for t, s in zip(targets, signs)]
        self._regulator_matrix = mat

    def _draw_snv_classes(self):
        """Draw each SNV's functional class, and turn it into an expression multiplier.

        The class is a property of the SITE (infinite-sites: a given position is mutated once in the
        population, so every clone carrying it inherits the same class) but of THIS RUN, not of the
        landscape — so it draws from the run stream.

        The multiplier is deliberately kept SEPARATE from the fitness effect (`selection.mut_effects`),
        which is the knob these two shared before: entangling "this mutation is advantageous" with
        "this mutation raises expression" is what made the SNV-based benchmarks circular.
        """
        sp = dict(self.snv_effect_params)
        p = np.array([float(sp.get("p_silent", 0.55)), float(sp.get("p_missense", 0.30)),
                      float(sp.get("p_splice", 0.05)), float(sp.get("p_lof", 0.10))])
        # SNV_CLASSES order is (silent, missense, splice, lof)
        if p.sum() <= 0:
            raise ValueError("snv_effect_params: class probabilities must sum to > 0")
        p = p / p.sum()
        rng = np.random.default_rng(self.run_seed + RUN_OFFSET_SNV_CLASS)
        self.snv_class = rng.choice(len(SNV_CLASSES), size=self.n_genes, p=p)

        nmd_strength = float(sp.get("nmd_strength", 0.2))
        snv_expression_effect = float(sp.get("snv_expression_effect", 0.5))
        # silent / missense leave mRNA abundance alone; LoF is degraded by NMD on its own allele;
        # splice shifts expression by the (fitness-independent) expression effect.
        effect_by_class = np.array([1.0, 1.0, snv_expression_effect, nmd_strength])
        self.snv_exp_effect = effect_by_class[self.snv_class]

    # ---------------------------------------------------------------- coupling (routes 1-3)
    @property
    def n_programs(self):
        return 0 if self.dictionary is None else self.dictionary.n_programs

    def _phenotype_level(self, phenotype, evo, baseline_rates):
        """Program activity ∝ the phenotype RELATIVE to its wild-type baseline (monotone, normalised,
        and 0 for a wild-type clone so the founder sits at the origin)."""
        if phenotype in ("treatment_resistance", "immune_resistance", "breach", "stromal_survival"):
            # already normalised to [0, 1) with wild-type = 0 (compartment-selection traits included)
            return float(evo.get(phenotype, 0.0))
        base = float((baseline_rates or {}).get(phenotype, 0.0))
        if base <= 0:
            return 0.0
        return float(evo.get(phenotype, base)) / base - 1.0

    def clone_drive(self, evo, baseline_rates, snv_vaf):
        """The per-CLONE program drive: routes 1 (phenotype-mediated) + 2 (direct regulators).

        Per-clone by construction — a phenotype is a property of a genotype — so this sets the MEAN
        of the program across that clone's cells; `activity_noise` supplies the within-clone spread
        (cycling is a per-cell state, not a clone constant).
        """
        K = self.n_programs
        drive = np.zeros(K)
        if K == 0:
            return drive
        cp = self.coupling_params
        idx = self.dictionary.program_index

        # Route 1 — phenotype-mediated (the default). mutation → CINner fitness → phenotype → program.
        # `phenotype_program_strength` is either a scalar gain shared by every phenotype, OR a per-
        # phenotype dict {phenotype: gain, "__default__": gain}. The per-phenotype form exists because
        # `_phenotype_level` is a RAW fold-change, and different phenotypes live on wildly different
        # fold-change scales: division_rate is bounded above by `max_birth_rate` (fold-change caps at
        # ~1.5 and saturates), whereas dispersal_rate is unbounded (fold-change can reach tens). A
        # single shared gain therefore makes the proliferation drive tiny relative to the EMT drive and
        # to `activity_noise`, so proliferation stops tracking division. A larger division gain
        # restores parity — biologically, the proliferation- and EMT-signature amplitudes are
        # independently calibrated, not tied to a common gain.
        strength_cfg = cp.get("phenotype_program_strength", 0.5)
        pmap = cp.get("phenotype_program_map", DEFAULT_PHENOTYPE_PROGRAM_MAP)
        for phenotype, prog in dict(pmap).items():
            k = idx.get(prog)
            if k is None:
                continue
            if isinstance(strength_cfg, dict):
                strength = float(strength_cfg.get(phenotype, strength_cfg.get("__default__", 0.5)))
            else:
                strength = float(strength_cfg)
            if strength == 0:
                continue
            drive[k] += strength * self._phenotype_level(phenotype, evo, baseline_rates)

        # Route 2 — direct regulators, graded by the mutation's VAF (so a clone that has amplified the
        # mutant allele drives the program harder). No fitness change.
        if self._regulator_matrix is not None:
            bias = float(cp.get("program_bias_strength", 0.5))
            if bias != 0:
                drive = drive + bias * (self._regulator_matrix @ np.asarray(snv_vaf, dtype=float))
        return drive

    def niche_drive(self, fields):
        """Route 3 — niche → program: the per-DEME drive from the F8 fields (n_demes × K).

        Generalises F8: any program can be tagged niche-responsive via `niche_program_map`, instead of
        hypoxia/CCI being hard-wired to their own private gene sets.
        """
        K = self.n_programs
        n_demes = len(next(iter(fields.values()))) if fields else 0
        D = np.zeros((n_demes, K))
        if K == 0 or not fields:
            return D
        cp = self.coupling_params
        nmap = cp.get("niche_program_map")
        if not nmap:
            return D
        strength = float(cp.get("niche_program_strength", 1.0))
        idx = self.dictionary.program_index
        for field_name, prog in dict(nmap).items():
            k = idx.get(prog)
            field = fields.get(field_name)
            if k is None or field is None:
                continue
            D[:, k] += strength * np.asarray(field, dtype=float)
        return D

    def celltype_bias(self, celltype):
        """Baseline program activity for a cell type (normal vs cancer), as a (K,) vector."""
        K = self.n_programs
        bias = np.zeros(K)
        raw = (self.activity_params.get("celltype_program_bias") or {}).get(celltype)
        if raw is None:
            return bias
        arr = np.atleast_1d(np.asarray(raw, dtype=float))
        if arr.size == 1:
            return bias + float(arr[0])
        bias[:min(K, arr.size)] += arr[:min(K, arr.size)]
        return bias

    # ---------------------------------------------------------------- the per-cell z sampler
    def _draw_activity(self, rng, shape):
        ap = self.activity_params
        dist = ap.get("activity_dist", "lognormal")
        mean = float(ap.get("activity_mean", 1.0))
        sd = float(ap.get("activity_sd", 0.5))
        if dist == "normal":
            return rng.normal(mean, sd, size=shape)
        if dist == "gamma":
            if mean <= 0 or sd <= 0:
                return np.full(shape, max(mean, 0.0))
            return rng.gamma(shape=(mean / sd) ** 2, scale=sd * sd / mean, size=shape)
        if dist == "lognormal":
            if mean <= 0:
                return np.zeros(shape)
            if sd <= 0:
                return np.full(shape, mean)
            # moment-match a lognormal to the requested (mean, sd)
            sigma2 = np.log1p((sd / mean) ** 2)
            return rng.lognormal(mean=np.log(mean) - 0.5 * sigma2, sigma=np.sqrt(sigma2), size=shape)
        raise ValueError(f"unknown activity_dist {dist!r} (normal | lognormal | gamma)")

    def _activation_mask(self, rng, n_cells):
        """Which programs are ON in each cell (`n_active_programs_per_cell`) — activity is sparse."""
        K = self.n_programs
        n_active = self.activity_params.get("n_active_programs_per_cell")
        if n_active is None or int(n_active) >= K:
            return np.ones((n_cells, K))
        n_active = int(n_active)
        if n_active <= 0:
            return np.zeros((n_cells, K))
        r = rng.random((n_cells, K))
        cutoff = np.partition(r, n_active - 1, axis=1)[:, n_active - 1][:, None]
        return (r <= cutoff).astype(float)

    def sample_z(self, drive, rng):
        """Draw the per-cell activity vector `z` given each cell's per-clone + niche drive.

        `drive` is (n_cells, K). The genotype/niche drive is ALWAYS on (it is a graded bias); the
        sparse random activation applies to the stochastic baseline activity, so a clone that drives
        the proliferation program raises it in all its cells while `activity_noise` still gives each
        cell its own level.
        """
        drive = np.atleast_2d(np.asarray(drive, dtype=float))
        n_cells, K = drive.shape
        if K == 0:
            return np.zeros((n_cells, 0))
        base = self._draw_activity(rng, (n_cells, K))
        z = self._activation_mask(rng, n_cells) * base + drive
        noise = float(self.activity_params.get("activity_noise", 0.0))
        if noise > 0:
            z = z + rng.normal(0.0, noise, size=(n_cells, K))
        return z

    def gene_multiplier(self, Z):
        """exp(Z · loading) — the program contribution to expression, (n_cells × n_genes)."""
        if self.dictionary is None or Z.shape[1] == 0:
            return np.ones((Z.shape[0], self.n_genes))
        return np.exp(np.asarray(Z) @ self.dictionary.loading)

    def activity_rng(self):
        """The `z` stream: run-seeded and re-created per materialisation, so repeated
        `make_cell_data()` calls on the same tumour reproduce the same cells."""
        return np.random.default_rng(self.run_seed + RUN_OFFSET_ACTIVITY)

    # ---------------------------------------------------------------- ground truth
    def truth(self):
        """`program_truth`: everything a benchmark needs to score an inferred program against."""
        out = {
            "dosage_sensitivity": self.dosage_sensitivity,
            "snv_class": self.snv_class,
            "snv_class_names": np.array(SNV_CLASSES),
            "snv_exp_effect": self.snv_exp_effect,
        }
        if self.dictionary is not None:
            d = self.dictionary
            out.update({
                "loading": d.loading,
                "program_names": np.array(d.program_names),
                "program_genes": [np.asarray(g) for g in d.program_genes],
                "gene_program_map": d.gene_program_map(),
                "regulator_genes": dict(self.regulator_genes),
            })
        return out
