"""Assay batch-effect model (DESIGN_features.md §B).

A **batch** is one realization of an assay instance — a single reaction / lane / plate
keyed by a seed. The *hyper-parameters* (`BatchHyperParams`) are protocol-typical
magnitudes; one seed draws a concrete *realization* (`Batch`) whose technical signature
is applied consistently across every cell of that instance and is reproducible from the
seed. Two `Batch`es with identical hyper-parameters but different seeds = same biology,
two technical signatures (§B.3).

The hyper-parameter names here are exactly what `estimate()` (DESIGN_inference.md §B / M2)
will fit from real data, so they are named for that purpose:

    sigma_batch       per-gene batch-factor LogNormal sd  (Splatter `batch.facScale`)
    mu_lib            mean library size (counts / cell)    -> sequencing depth
    sigma_lib         per-cell library-size LogNormal sd
    dispersion        NB overdispersion phi  (var = mu + dispersion * mu^2)
    ambient_frac      fraction of the library that is ambient ("soup") contamination
    doublet_rate      fraction of barcodes that are doublets (two cells merged)
    dropout_mid       logistic dropout midpoint in expected-count space (0 disables)
    dropout_shape     logistic dropout steepness
    well_sigma        per-cell "well" (plate-position) LogNormal sd  (Smart-seq3 nesting)
    depth_batch_sigma per-batch depth-shift LogNormal sd (so batches differ in depth)
    kappa             Dirichlet concentration (only for the `dm` count model)

The **count emission is a pluggable step** (§B.2): everything above (biology -> library ->
batch) is shared and only the final draw is swapped via ``COUNT_MODELS``.
"""
from dataclasses import dataclass, asdict

import numpy as np


# --------------------------------------------------------------------------------------
# Pluggable count-emission step (DESIGN_features §B.2).
#
# Each emitter has the signature  emit(rng, comp, lib, batch) -> counts (n_cells, n_genes)
# where `comp` is the per-cell normalized composition (rows sum to 1, already carrying the
# per-gene batch factor beta and the biological signal lambda) and `lib` is the per-cell
# library size. Keeping this signature shared means a new count model is a drop-in, not a
# rewrite: the biology->library->batch pipeline never changes.
# --------------------------------------------------------------------------------------
def _emit_nb(rng, comp, lib, batch):
    """Default `count_model="nb"`: y_cg ~ NB(mu = lib_c * comp_cg, phi_b).

    Gamma-Poisson mixture so var = mu + phi*mu^2 (the field-standard Splatter/DESeq2/edgeR
    parameterization, the one `estimate()` fits).
    """
    mu = np.maximum(lib[:, None] * comp, 0.0)
    phi = batch.dispersion
    if phi <= 0:
        return rng.poisson(mu)
    shape = 1.0 / phi
    gamma = rng.gamma(shape=shape, scale=mu * phi)
    return rng.poisson(gamma)


def _emit_dm(rng, comp, lib, batch):
    """Optional `count_model="dm"` (Dirichlet-Multinomial) — planned drop-in alternative.

    The compositional model: a fixed library total with gene-gene competition for reads.
    The shared pipeline already produces exactly the two pieces this needs — the per-cell
    composition `comp` (= normalized beta_gb * lambda_cg) and the per-cell library `lib` —
    so wiring it up is only this function:

        for each cell c:
            N_c = round(lib_c)                              # fixed total
            p_c ~ Dirichlet(kappa_b * comp_c)              # proportions
            y_c ~ Multinomial(N_c, p_c)

    Left unimplemented for F3 (NB is the default and what M2 fits); the seam is here so it
    is a one-function add, not a rewrite.
    """
    raise NotImplementedError(
        "count_model='dm' (Dirichlet-Multinomial) is not implemented yet. The biology->"
        "library->batch pipeline already yields `comp` and `lib`; only this emission step "
        "needs: N_c=round(lib_c); p_c~Dirichlet(kappa_b*comp_c); y_c~Multinomial(N_c,p_c)."
    )


COUNT_MODELS = {"nb": _emit_nb, "dm": _emit_dm}


@dataclass
class BatchHyperParams:
    """Protocol-typical magnitudes for the scRNA batch model (the targets of `estimate()`)."""
    protocol: str = "10x"
    sigma_batch: float = 0.10
    mu_lib: float = 4000.0
    sigma_lib: float = 0.35
    dispersion: float = 0.30
    ambient_frac: float = 0.05
    doublet_rate: float = 0.05
    dropout_mid: float = 0.0
    dropout_shape: float = 1.0
    well_sigma: float = 0.0
    depth_batch_sigma: float = 0.05
    kappa: float = 50.0

    def to_dict(self):
        return asdict(self)


class Batch:
    """One seeded realization of an assay instance (a reaction / lane / plate).

    Draws and stores the technical state shared across all cells of the batch (per-gene
    factor, realized depth, dispersion, ambient soup profile) and exposes the per-cell
    operations used by the modality assay (library sizes, composition, count emission,
    ambient, dropout, doublets). All draws come from a single seeded RNG, so the whole
    technical signature is reproducible from `seed` and differs between seeds.
    """

    def __init__(self, hypers: BatchHyperParams, seed=0, label=None):
        self.h = hypers
        self.seed = int(seed)
        self.label = str(label) if label is not None else f"batch{self.seed}"
        self.rng = np.random.default_rng(self.seed)
        self._realized = False

    @property
    def dispersion(self):
        return self._phi

    def realize(self, genes, base_profile):
        """Draw the batch-level technical state. `base_profile` = pooled expression (soup)."""
        n = len(genes)
        self.genes = list(genes)
        # per-gene batch factor beta_gb ~ LogNormal(0, sigma_batch^2), shared across cells
        self.beta = self.rng.lognormal(mean=0.0, sigma=self.h.sigma_batch, size=n)
        # per-batch realized depth mu_lib,b (so two batches differ in depth)
        self.mu_lib_b = float(self.h.mu_lib * self.rng.lognormal(0.0, self.h.depth_batch_sigma))
        # per-batch dispersion phi_b
        self._phi = float(self.h.dispersion)
        # ambient ("soup") profile for this batch: normalized pooled expression
        amb = np.asarray(base_profile, dtype=float)
        tot = amb.sum()
        self.ambient_profile = amb / tot if tot > 0 else np.full(n, 1.0 / n)
        self._realized = True
        return self

    def library_factors(self, n_cells):
        """Per-cell library size ell_c ~ LogNormal(log mu_lib,b, sigma_lib^2).

        For Smart-seq3 (`well_sigma>0`) an extra per-cell "well" multiplier nests a
        plate-position effect inside the batch.
        """
        mu_ln = np.log(self.mu_lib_b) - 0.5 * self.h.sigma_lib ** 2
        lib = self.rng.lognormal(mean=mu_ln, sigma=self.h.sigma_lib, size=n_cells)
        if self.h.well_sigma > 0:
            lib = lib * self.rng.lognormal(0.0, self.h.well_sigma, size=n_cells)
        return lib

    def composition(self, probs):
        """Apply the per-gene batch factor to normalized expression, renormalize per cell.

        `probs` (n_cells, n_genes) is normalized biological expression (lambda). The batch
        factor reshapes the per-gene composition (Splatter-style); library size sets the
        total separately, so the realized library stays centered on the requested depth.
        """
        base = probs * self.beta[None, :]
        tot = base.sum(axis=1, keepdims=True)
        return np.divide(base, tot, out=np.zeros_like(base), where=tot > 0)

    def emit(self, comp, lib, count_model="nb"):
        if count_model not in COUNT_MODELS:
            raise ValueError(
                f"unknown count_model {count_model!r}; choose from {sorted(COUNT_MODELS)}"
            )
        return COUNT_MODELS[count_model](self.rng, comp, lib, self)

    def add_ambient(self, counts, lib):
        """Add ambient contamination: Poisson counts from the batch soup profile."""
        if self.h.ambient_frac <= 0:
            return counts
        amb_mu = (self.h.ambient_frac * lib)[:, None] * self.ambient_profile[None, :]
        return counts + self.rng.poisson(amb_mu)

    def apply_dropout(self, counts, comp, lib):
        """Optional Splatter-style logistic dropout (zero-inflation), keyed to mean count.

        Returns (counts, p_drop). `p_drop` is None when dropout is disabled.
        """
        if self.h.dropout_mid <= 0:
            return counts, None
        mu = np.maximum(lib[:, None] * comp, 1e-12)
        p_drop = 1.0 / (1.0 + np.exp(self.h.dropout_shape * (np.log(mu) - np.log(self.h.dropout_mid))))
        keep = self.rng.random(counts.shape) >= p_drop
        return counts * keep, p_drop

    def apply_doublets(self, counts):
        """Merge a `doublet_rate` fraction of barcodes with a random partner cell.

        Returns (counts, is_doublet) where `is_doublet` marks the merged barcodes.
        """
        n = counts.shape[0]
        is_doublet = np.zeros(n, dtype=bool)
        if self.h.doublet_rate <= 0 or n < 2:
            return counts, is_doublet
        n_dbl = int(round(self.h.doublet_rate * n))
        if n_dbl == 0:
            return counts, is_doublet
        idx = self.rng.choice(n, size=n_dbl, replace=False)
        partners = self.rng.choice(n, size=n_dbl, replace=True)
        counts = counts.copy()
        for i, p in zip(idx, partners):
            if p == i:
                p = (p + 1) % n
            counts[i] = counts[i] + counts[p]
            is_doublet[i] = True
        return counts, is_doublet


# ======================================================================================
# DNA assay batch model (DESIGN_features.md §C / §D) — shared by bulk and single-cell.
#
# The physics of DNA coverage differ from scRNA expression, so the pluggable choices flip
# (§C C1): the *depth* default is **Dirichlet-Multinomial** (compositional — a fixed read
# budget partitioned across loci, proportions driven by COPY NUMBER) rather than NB, and
# the *allele* layer is a separate binomial/beta-binomial draw with explicit ADO for
# single-cell. The unifying knob across bulk vs single-cell is **kappa = the amplification
# regime**: large kappa  ≈ multinomial/Poisson (un-amplified BULK), small kappa = lumpy
# (whole-cell-amplified SINGLE-CELL).
#
# All hyper-parameter names here are M4 `estimate()` targets (DESIGN_inference): `kappa`
# (amplification regime), `capture_sigma` (per-target/amplicon efficiency), `ado_rate`,
# `beta_binom_conc`, the GC-curve params, and `error_rate`.
# ======================================================================================

# ---- Pluggable DNA depth-emission step (DESIGN_features §C C1) ------------------------
# Each emitter has the signature  emit(rng, weights, N, h) -> counts (n_loci,)  where
# `weights` are non-negative per-locus relative coverage drivers (∝ CN · length · GC /
# mappability efficiency), `N` is the total read budget, and `h` is the hyper-parameters.
# Keeping the (weights, N) interface shared means DM and NB are true drop-ins.
def _dna_depth_dm(rng, weights, N, h):
    """Default DNA depth model: Dirichlet-Multinomial (compositional, fixed budget N).

        p_seg ~ Dirichlet(kappa * p_bar),  p_bar ∝ weights;   y ~ Multinomial(N, p)

    Large `kappa` → p≈p_bar → multinomial/Poisson (BULK); small `kappa` → lumpy
    over-Poisson coverage (SINGLE-CELL amplification). Because the budget is fixed, an
    amplified (high-CN) segment compositionally *steals* reads from the rest — exactly the
    coupling independent-per-bin models miss, and the reason depth is RELATIVE not absolute.
    """
    w = np.asarray(weights, dtype=float)
    s = w.sum()
    N = int(round(N))
    if s <= 0 or N <= 0:
        return np.zeros(len(w), dtype=int)
    p_bar = w / s
    alpha = np.maximum(h.kappa * p_bar, 1e-12)
    p = rng.dirichlet(alpha)
    return rng.multinomial(N, p)


def _dna_depth_nb(rng, weights, N, h):
    """Alternative DNA depth model: independent Negative-Binomial per locus/bin.

    The field convention for BULK CNA tools (HMMcopy / CNVkit / Control-FREEC): each bin
    drawn independently with mean ∝ its weight (no fixed-total coupling, so the total
    fluctuates and high-CN bins do not steal from others). `nb_dispersion` sets the
    over-Poisson lumpiness (var = mu + phi*mu^2); phi<=0 collapses to Poisson.
    """
    w = np.asarray(weights, dtype=float)
    s = w.sum()
    if s <= 0:
        return np.zeros(len(w), dtype=int)
    mu = np.maximum(N * (w / s), 0.0)  # expected per-locus coverage
    phi = h.nb_dispersion
    if phi <= 0:
        return rng.poisson(mu)
    shape = 1.0 / phi
    gamma = rng.gamma(shape=shape, scale=mu * phi)
    return rng.poisson(gamma)


DNA_DEPTH_MODELS = {"dm": _dna_depth_dm, "nb": _dna_depth_nb}


@dataclass
class DNABatchHyperParams:
    """Protocol-typical magnitudes for the DNA batch model (the targets of `estimate()`).

        breadth          capture breadth {wgs, wes, panel} (sets locus set + depth regime)
        depth_model      depth emission {"dm" (default), "nb"}
        mu_depth         mean per-locus coverage (the depth regime; breadth sets it)
        kappa            DM concentration = AMPLIFICATION REGIME (large=bulk, small=single-cell)
        nb_dispersion    NB per-bin overdispersion phi (only for depth_model="nb")
        gc_curve_sigma   per-batch GC->coverage curve strength (the GC bias a panel-of-normals fits)
        mappability_sigma  unused placeholder kept for symmetry with the genome mappability field
        capture_sigma    per-target (WES) / per-amplicon (panel) capture-efficiency LogNormal sd
        error_rate       per-base sequencing error (false alt/ref on the allele layer)
        depth_batch_sigma per-batch depth-shift LogNormal sd (batches differ in depth)
        ado_rate         single-cell allelic dropout Bernoulli prob (one allele lost at a locus)
        beta_binom_conc  single-cell allele-fraction overdispersion (Beta-Binomial concentration)
        doublet_rate     single-cell doublet fraction
        ffpe_ct_rate     optional FFPE C>T deamination extra-error at C-sites
    """
    breadth: str = "wgs"
    depth_model: str = "dm"
    mu_depth: float = 30.0
    kappa: float = 2000.0
    nb_dispersion: float = 0.1
    gc_curve_sigma: float = 0.20
    mappability_sigma: float = 0.10
    capture_sigma: float = 0.0
    error_rate: float = 0.001
    depth_batch_sigma: float = 0.05
    ado_rate: float = 0.0
    beta_binom_conc: float = 30.0
    doublet_rate: float = 0.0
    ffpe_ct_rate: float = 0.0

    def to_dict(self):
        return asdict(self)


class DNABatch:
    """One seeded realization of a DNA assay instance (a library / run / chip).

    Mirrors `Batch` (scRNA) for DNA (DESIGN_features §D): draws the run-level technical
    state shared across the assay — a **per-batch GC->coverage curve**, a per-locus
    mappability multiplier, a systematic **per-target/amplicon capture efficiency**, a
    per-batch **depth shift**, and a per-locus **error rate** (+ optional FFPE C>T) — then
    exposes the per-locus operations the bulk/single-cell DNA assays compose: the pluggable
    depth draw and the allele layer (binomial for bulk; beta-binomial + ADO for single-cell).
    """

    def __init__(self, hypers: DNABatchHyperParams, seed=0, label=None):
        self.h = hypers
        self.seed = int(seed)
        self.label = str(label) if label is not None else f"dnabatch{self.seed}"
        self.rng = np.random.default_rng(self.seed)
        self._realized = False

    def realize(self, gc, mappability, ct_sites=None):
        """Draw the batch-level technical state for the observed loci.

        `gc` and `mappability` are *genome* properties (stable across batches); the GC
        *curve* and capture draws are the per-batch (technical) signature. The combined
        per-locus efficiency is normalized to mean 1 so it reshapes coverage (bias) without
        moving the absolute depth scale, which `mu_depth` / the budget set separately.
        """
        gc = np.asarray(gc, dtype=float)
        mappability = np.asarray(mappability, dtype=float)
        n = len(gc)
        self.n = n
        # per-batch realized depth (so two batches differ in depth)
        self.depth = float(self.h.mu_depth * self.rng.lognormal(0.0, self.h.depth_batch_sigma))
        # per-batch GC->coverage curve: a smooth unimodal bias peaking at gc_opt.
        # gc_curve_sigma scales both the peak jitter and the curvature (bias strength).
        self.gc_opt = float(0.42 + self.rng.normal(0.0, 0.05) * self.h.gc_curve_sigma / 0.20)
        self.gc_strength = float(self.rng.uniform(4.0, 10.0) * self.h.gc_curve_sigma / 0.20)
        gc_mult = np.exp(-self.gc_strength * (gc - self.gc_opt) ** 2)
        # systematic per-target/amplicon capture efficiency MEAN (WES/panel); 1 for WGS.
        if self.h.capture_sigma > 0:
            self.capture_eff = self.rng.lognormal(0.0, self.h.capture_sigma, size=n)
        else:
            self.capture_eff = np.ones(n)
        eff = gc_mult * mappability * self.capture_eff
        m = eff.mean()
        self.efficiency = eff / m if m > 0 else np.ones(n)
        # per-locus effective error (+ FFPE C>T deamination at C-sites)
        err = np.full(n, self.h.error_rate, dtype=float)
        if self.h.ffpe_ct_rate > 0 and ct_sites is not None:
            err = err + self.h.ffpe_ct_rate * np.asarray(ct_sites, dtype=float)
        self.error = np.clip(err, 0.0, 1.0)
        self._realized = True
        return self

    # -- depth -------------------------------------------------------------------------
    def emit_depth(self, weights, N):
        """Draw per-locus coverage via the pluggable depth model (DM default / NB)."""
        return DNA_DEPTH_MODELS[self.h.depth_model](self.rng, weights, N, self.h)

    # -- allele layer ------------------------------------------------------------------
    def alleles_binomial(self, coverage, true_af):
        """BULK allele draw: alt ~ Binomial(coverage, p_eff) with sequencing error.

        p_eff = true_af*(1-e) + (1-true_af)*e folds the per-base error in both directions
        (false alt on ref bases, false ref on alt bases).
        """
        e = self.error
        p = np.clip(np.asarray(true_af) * (1 - e) + (1 - np.asarray(true_af)) * e, 0.0, 1.0)
        return self.rng.binomial(coverage, p)

    def apply_ado(self, true_af):
        """SINGLE-CELL allelic dropout: with prob `ado_rate`, one allele is lost at a het
        locus, flipping the observed fraction to 0 or 1 (the dominant single-cell allele
        artifact — modelled as a separate Bernoulli layer, NOT via the depth distribution).

        Returns (af_observed, ado_mask). Only heterozygous loci (0<af<1) can drop.
        """
        af = np.asarray(true_af, dtype=float)
        mask = np.zeros(af.shape, dtype=bool)
        if self.h.ado_rate <= 0:
            return af.copy(), mask
        het = (af > 0) & (af < 1)
        ado = (self.rng.random(af.shape) < self.h.ado_rate) & het
        drop_alt = self.rng.random(af.shape) < 0.5
        af_obs = af.copy()
        af_obs[ado & drop_alt] = 0.0       # alt allele lost -> looks homozygous reference
        af_obs[ado & ~drop_alt] = 1.0      # ref allele lost -> looks homozygous alt
        return af_obs, ado

    def allele_balance(self, af, apply_error=True):
        """SINGLE-CELL realized per-locus allele fraction theta (Beta-Binomial mean), WITHOUT
        sampling reads.

        theta ~ Beta(c*p, c*(1-p)) with c = `beta_binom_conc` (small c = lumpier allele balance).
        ``apply_error`` controls whether the per-base sequencing-error floor is folded in:
          * ``True`` (count layer): p = af*(1-e)+(1-af)*e — the OBSERVED allele fraction a count
            caller sees (`alleles_betabinom` samples reads from it).
          * ``False`` (read layer): p = af — the TRUE molecular allele balance (amplification
            overdispersion only), so the read emitter sets per-copy alt multiplicity WITHOUT the
            error floor and the simulator's per-base error (DWGSIM `-e`) is the single read-error
            source (no double-counting; mirrors the scRNA path).
        Degenerate p in {0,1} (e.g. after ADO) collapses to a fixed fraction.
        """
        af = np.asarray(af, dtype=float)
        if apply_error:
            p = np.clip(af * (1 - self.error) + (1 - af) * self.error, 0.0, 1.0)
        else:
            p = np.clip(af, 0.0, 1.0)
        c = self.h.beta_binom_conc
        a = np.maximum(c * p, 1e-9)
        b = np.maximum(c * (1 - p), 1e-9)
        theta = self.rng.beta(a, b)
        return np.where(p <= 0, 0.0, np.where(p >= 1, 1.0, theta))

    def alleles_betabinom(self, coverage, af):
        """SINGLE-CELL allele draw: alt ~ Binomial(coverage, allele_balance(af))."""
        theta = self.allele_balance(af)
        return self.rng.binomial(coverage, np.clip(theta, 0.0, 1.0))
