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
