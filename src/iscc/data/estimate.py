"""Splatter-style `estimate_rna()` for the scRNA assay.

Fits the hyper-parameters of the batch model (``RNABatchHyperParams`` in
``batch.py``) from a real count matrix, so realistic technical magnitudes are *learned, not
guessed*. The fitted ``RNABatchHyperParams`` (plus per-gene tables) drive ``scRNA`` /
``run_scrna_batches`` directly — same field names, no redefinition.

What is fit (and from what):

    mu_lib            mean of per-cell totals                       (always)
    sigma_lib         sd of log per-cell totals                     (always)
    dispersion phi    mean-variance trend  var = mu + phi*mu^2,     (always)
                      de-biased for the library-size CV (see below)
    gene_means/gamma  Splatter per-gene mean-expression table       (always, informational)
    dropout_mid/shape logistic zero-fraction-vs-mean curve          (10x; off for Smart-seq3)
    sigma_batch       per-gene cross-batch log-fold-change scale    (needs >=2 batches)
    depth_batch_sigma sd of log per-batch depth                     (needs >=2 batches)

Library-size de-biasing of the dispersion.  For a gene with cell-independent composition
``p_g`` and library ``lib`` (mean ``mu_lib``, squared CV ``c2 = Var(lib)/mu_lib^2``):

    E[y_g]   = mu_lib * p_g                                   =: m_g
    Var[y_g] = m_g + m_g^2 * ( phi*(1+c2) + c2 )

so the raw mean-variance slope ``S = (Var-mean)/mean^2`` measures ``phi*(1+c2)+c2``, *not*
``phi``: library-size variation inflates it. We back out the negative-binomial dispersion as
``phi = (S - c2) / (1 + c2)``. On synthetic data with cell-constant biology this recovers the
true ``phi``; on real data (which also has biological cell-cell variation) it is an effective
dispersion, reported honestly.

Protocol-aware: ``estimate_rna(adata, protocol="10x")`` fits dropout and keeps droplet ambient/
doublet priors; ``protocol="smartseq3"`` disables dropout, enables the per-cell well term, and
keeps the plate layout (wells per plate, plates, the shared per-plate depth offset) and the
low-ambient priors. Counts-only data cannot identify the ambient soup fraction, the doublet rate or
the plate layout (those need empty droplets / genotype demultiplexing / the wet-lab plate map), so
those fields are carried from the protocol preset. ``RNAEstimate.fitted`` lists what the data
actually determined and ``RNAEstimate.carried`` lists what came from the preset, so a report can
say which is which instead of implying the whole parameter set was measured.

numpy / scipy / sklearn only (no JAX); stateless and config-driven.
"""
from dataclasses import dataclass, field

import numpy as np

from .batch import RNABatchHyperParams
from .rna import PROTOCOL_PRESETS, resolve_protocol

# Fields the batch model owns but a counts-only estimate cannot identify (need empty droplets /
# demultiplexing / the wet-lab plate map); carried from the protocol preset, never claimed as
# "fit". The plate layout belongs here too: a count matrix does not say how many wells a plate had,
# how many plates the cells were spread over, or how much depth the cells on one plate shared —
# dropping them silently turned a fitted Smart-seq3 estimate into a plate-less one.
_PRIOR_ONLY = ("ambient_frac", "doublet_rate", "kappa",
               "well_sigma", "n_wells", "n_plates", "plate_sigma")


@dataclass
class RNAEstimate:
    """Fitted scRNA technical parameters, ready to drive ``scRNA`` / ``run_scrna_batches``.

    ``hypers`` is a ``RNABatchHyperParams`` with the fitted magnitudes; ``scrna_kwargs()`` returns
    the same as a kwargs dict. The per-gene tables (``gene_means`` + the gamma fit) are the
    Splatter mean-expression model, kept for reporting / a future expression generator.

    ``fitted`` and ``carried`` split the hyper-parameters into the ones this count matrix actually
    determined and the ones taken from the protocol preset because counts alone cannot identify
    them (ambient soup, doublets, the plate layout, and the cross-batch terms when there is only
    one batch), so a report can be explicit about which is which.
    """
    hypers: RNABatchHyperParams
    protocol: str
    n_cells: int
    n_genes: int
    n_batches: int
    cv_lib: float
    gene_means: np.ndarray                 # per-gene mean library-normalized expression
    gamma_shape: float                     # gamma fit to gene_means (Splatter)
    gamma_scale: float
    fitted: list                           # hyper-param names actually fit from this data
    per_batch: dict = field(default_factory=dict)   # label -> {mu_lib, dispersion, n_cells}
    carried: list = field(default_factory=list)     # names taken from the protocol preset

    def scrna_kwargs(self):
        """Kwargs to splat into ``scRNA(...)`` or ``run_scrna_batches(...)``."""
        h = self.hypers.to_dict()
        h.pop("kappa", None)               # set via count_model, not a constructor knob here
        return h

    def __repr__(self):
        h = self.hypers
        return (f"RNAEstimate(protocol={self.protocol!r}, n_cells={self.n_cells}, "
                f"n_genes={self.n_genes}, n_batches={self.n_batches}, "
                f"mu_lib={h.mu_lib:.0f}, sigma_lib={h.sigma_lib:.3f}, "
                f"dispersion={h.dispersion:.3f}, sigma_batch={h.sigma_batch:.3f}, "
                f"depth_batch_sigma={h.depth_batch_sigma:.3f}, "
                f"dropout_mid={h.dropout_mid:.2f}, n_wells={h.n_wells}, "
                f"fitted={self.fitted}, carried={self.carried})")


# --------------------------------------------------------------------------------------
# Dense-matrix helpers
# --------------------------------------------------------------------------------------
def _dense_counts(x):
    """Accept AnnData / DataFrame / sparse / ndarray -> dense (cells, genes) float array."""
    if hasattr(x, "X"):
        x = x.X
    if hasattr(x, "toarray"):
        x = x.toarray()
    if hasattr(x, "values"):
        x = x.values
    return np.asarray(x, dtype=float)


def _batch_labels(adata, batch_key, n_cells):
    """Extract per-cell batch labels (None if single batch / no key)."""
    if batch_key is None:
        return None
    if hasattr(adata, "obs") and batch_key in getattr(adata, "obs", {}):
        return np.asarray(adata.obs[batch_key].values).astype(str)
    return None


# --------------------------------------------------------------------------------------
# Individual estimators
# --------------------------------------------------------------------------------------
def _fit_library_size(counts):
    """LogNormal fit to per-cell totals -> (mu_lib, sigma_lib, cv_lib).

    The model draws ``lib`` so that ``E[lib] = mu_lib`` and ``sd(log lib) = sigma_lib``; invert
    both directly from the empirical per-cell totals.
    """
    lib = counts.sum(axis=1)
    lib = lib[lib > 0]
    if lib.size == 0:
        return 0.0, 0.0, 0.0
    mu_lib = float(lib.mean())
    sigma_lib = float(np.std(np.log(lib)))
    cv_lib = float(np.std(lib) / (np.mean(lib) + 1e-12))
    return mu_lib, sigma_lib, cv_lib


def _fit_dispersion(counts, cv_lib):
    """Negative-binomial dispersion ``phi`` from the mean-variance trend, library-de-biased.

    Per gene ``S_g = (var - mean)/mean^2`` estimates ``phi*(1+c2) + c2`` (see module docstring),
    where ``c2 = cv_lib^2``. We take a robust (median) ``S`` over adequately expressed genes and
    invert: ``phi = (S - c2)/(1 + c2)``, clamped at 0.
    """
    m = counts.mean(axis=0)
    v = counts.var(axis=0)
    c2 = float(cv_lib) ** 2
    # restrict to genes with enough signal so the per-gene variance is well estimated
    thr = max(1.0, float(np.median(m[m > 0])) if np.any(m > 0) else 1.0)
    keep = m >= thr
    if keep.sum() < 5:                                  # too few -> relax
        keep = m > 0
    mm, vv = m[keep], v[keep]
    s = (vv - mm) / (mm ** 2)
    S = float(np.median(s))
    phi = (S - c2) / (1.0 + c2)
    return max(phi, 0.0)


def _fit_mean_expression(counts):
    """Splatter per-gene mean-expression model: library-normalize, mean per gene, fit gamma.

    Returns ``(gene_means, shape, scale)``. ``gene_means`` is the per-gene mean of
    library-size-normalized counts (scaled to the median library size).
    """
    lib = counts.sum(axis=1, keepdims=True)
    scale_to = np.median(lib[lib > 0]) if np.any(lib > 0) else 1.0
    norm = np.divide(counts, lib, out=np.zeros_like(counts), where=lib > 0) * scale_to
    gene_means = norm.mean(axis=0)
    pos = gene_means[gene_means > 0]
    if pos.size < 2:
        return gene_means, float("nan"), float("nan")
    # method-of-moments gamma (robust, no optimizer); shape=mean^2/var, scale=var/mean
    mu, var = float(pos.mean()), float(pos.var())
    if var <= 0:
        return gene_means, float("nan"), float("nan")
    shape = mu ** 2 / var
    scale = var / mu
    return gene_means, shape, scale


def _fit_dropout(counts):
    """Logistic dropout-vs-mean (Splatter): zero_frac_g = 1/(1+exp(shape*(log m_g - log mid))).

    Fit by least squares on the per-gene (log mean, zero fraction) curve. Returns
    ``(dropout_mid, dropout_shape)`` or ``(0.0, 1.0)`` if no usable curve / fit fails.
    """
    from scipy.optimize import curve_fit

    m = counts.mean(axis=0)
    zero = (counts == 0).mean(axis=0)
    use = m > 0
    if use.sum() < 8 or zero[use].max() < 0.02:         # no meaningful dropout to fit
        return 0.0, 1.0
    x = np.log(m[use])
    y = zero[use]

    def logistic(lm, log_mid, shape):
        return 1.0 / (1.0 + np.exp(shape * (lm - log_mid)))

    try:
        p0 = [float(np.median(x)), 1.0]
        popt, _ = curve_fit(logistic, x, y, p0=p0,
                            bounds=([x.min() - 5, 1e-2], [x.max() + 5, 20.0]), maxfev=5000)
        log_mid, shape = popt
        mid = float(np.exp(log_mid))
        if not np.isfinite(mid) or mid <= 0:
            return 0.0, 1.0
        return mid, float(shape)
    except (RuntimeError, ValueError):
        return 0.0, 1.0


def _fit_batch_params(counts, labels):
    """Cross-batch hyper-parameters (needs >=2 batches).

    Returns ``(sigma_batch, depth_batch_sigma, per_batch)``.

    * ``depth_batch_sigma`` = sd over batches of ``log(mean library size)``: how much per-batch
      sequencing depth wobbles (the model draws ``mu_lib,b = mu_lib * LogNormal(0, this^2)``).
    * ``sigma_batch`` = scale of the per-gene cross-batch log-fold-change ``log beta_gb``.
      For each batch take the per-gene composition (mean count normalized to sum 1), log it, then
      strip the additive gene effect and per-batch depth effect (a two-way centering); the
      residual is the gene x batch interaction = ``log beta_gb`` (mean-0, sd ``sigma_batch``).
      Estimated on well-expressed genes to limit sampling noise.
    """
    labs = np.asarray(labels).astype(str)
    uniq = list(dict.fromkeys(labs))                    # preserve order
    per_batch = {}
    log_depths = []
    comp_logs = []                                      # per-batch log composition over shared genes
    n_genes = counts.shape[1]

    # choose genes well-expressed in *every* batch so the per-batch mean is precise
    expressed = np.ones(n_genes, dtype=bool)
    batch_means = {}
    for b in uniq:
        cb = counts[labs == b]
        m = cb.mean(axis=0)
        batch_means[b] = m
        expressed &= m >= 1.0
    if expressed.sum() < 10:                            # relax if too strict
        pooled = counts.mean(axis=0)
        expressed = pooled >= max(0.5, np.median(pooled[pooled > 0]) if np.any(pooled > 0) else 0.5)

    for b in uniq:
        cb = counts[labs == b]
        lib = cb.sum(axis=1)
        lib = lib[lib > 0]
        mu_b = float(lib.mean()) if lib.size else 0.0
        per_batch[b] = dict(mu_lib=mu_b,
                            dispersion=_fit_dispersion(cb, np.std(cb.sum(1)) / (cb.sum(1).mean() + 1e-12)),
                            n_cells=int(cb.shape[0]))
        if mu_b > 0:
            log_depths.append(np.log(mu_b))
        m = batch_means[b][expressed]
        comp = m / m.sum() if m.sum() > 0 else np.full(m.size, 1.0 / max(m.size, 1))
        comp_logs.append(np.log(np.maximum(comp, 1e-12)))

    depth_batch_sigma = float(np.std(log_depths)) if len(log_depths) > 1 else 0.0

    # two-way centering of the log-composition matrix (batches x genes): residual = log beta
    L = np.vstack(comp_logs)                            # (n_batch, n_gene_expressed)
    L = L - L.mean(axis=1, keepdims=True)               # remove per-batch depth/normalization
    L = L - L.mean(axis=0, keepdims=True)               # remove per-gene effect
    B = L.shape[0]
    # residual is the centered interaction; correct for the df lost to batch-centering
    df_corr = np.sqrt(B / (B - 1)) if B > 1 else 1.0
    sigma_batch = float(np.std(L) * df_corr)
    return sigma_batch, depth_batch_sigma, per_batch


# --------------------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------------------
def estimate_rna(adata, protocol="10x", batch_key=None, count_model="nb", fit_dropout=None):
    """Fit ``RNABatchHyperParams`` from a real scRNA count matrix (Splatter-style).

    Parameters
    ----------
    adata : AnnData | DataFrame | ndarray
        Raw integer counts, cells x genes. (Library-size / dispersion fits assume *counts*,
        not log/normalized values.)
    protocol : str
        ``"10x"`` or ``"smartseq3"`` (aliases accepted). Sets which components are active and
        supplies priors for the fields counts-only data cannot identify (ambient, doublets).
    batch_key : str | None
        ``adata.obs`` column with batch labels. With >=2 batches, ``sigma_batch`` and
        ``depth_batch_sigma`` are fit; otherwise they default to 0 (single-batch data).
    fit_dropout : bool | None
        Force dropout fitting on/off. Default: on for 10x, off for Smart-seq3.

    Returns
    -------
    RNAEstimate
    """
    proto = resolve_protocol(protocol)
    preset = dict(PROTOCOL_PRESETS[proto])
    counts = _dense_counts(adata)
    if counts.ndim != 2:
        raise ValueError(f"expected a 2D (cells x genes) count matrix, got shape {counts.shape}")
    n_cells, n_genes = counts.shape
    fitted = []

    # always-fit core params
    mu_lib, sigma_lib, cv_lib = _fit_library_size(counts)
    dispersion = _fit_dispersion(counts, cv_lib)
    gene_means, gshape, gscale = _fit_mean_expression(counts)
    fitted += ["mu_lib", "sigma_lib", "dispersion"]

    # dropout (protocol-aware)
    if fit_dropout is None:
        fit_dropout = (proto == "10x")
    if fit_dropout:
        dropout_mid, dropout_shape = _fit_dropout(counts)
        if dropout_mid > 0:
            fitted.append("dropout_mid")
    else:
        dropout_mid, dropout_shape = 0.0, float(preset["dropout_shape"])

    # batch hyper-parameters (need >=2 batches)
    labels = _batch_labels(adata, batch_key, n_cells)
    per_batch = {}
    sigma_batch = float(preset["sigma_batch"])
    depth_batch_sigma = float(preset["depth_batch_sigma"])
    n_batches = 1
    if labels is not None and len(set(labels.tolist())) >= 2:
        n_batches = len(set(labels.tolist()))
        sigma_batch, depth_batch_sigma, per_batch = _fit_batch_params(counts, labels)
        fitted += ["sigma_batch", "depth_batch_sigma"]
        # Dispersion is a within-batch quantity: estimating it on the pooled matrix lets
        # cross-batch depth and beta differences leak in and inflate phi. Average the per-batch
        # NB dispersions instead (each already library-de-biased on its own cells).
        phis = [v["dispersion"] for v in per_batch.values() if np.isfinite(v["dispersion"])]
        if phis:
            dispersion = float(np.mean(phis))

    # The whole plate layer (well spread, wells per plate, plates, shared per-plate depth) comes
    # from the protocol preset: a bare count matrix carries no plate map. Smart-seq3 therefore keeps
    # its 384-well layout instead of silently degrading to a plate-less protocol.
    hypers = RNABatchHyperParams(
        protocol=proto,
        sigma_batch=sigma_batch,
        mu_lib=mu_lib,
        sigma_lib=sigma_lib,
        dispersion=dispersion,
        ambient_frac=float(preset["ambient_frac"]),     # prior (see _PRIOR_ONLY)
        doublet_rate=float(preset["doublet_rate"]),     # prior
        dropout_mid=dropout_mid,
        dropout_shape=dropout_shape,
        well_sigma=float(preset["well_sigma"]),         # prior
        n_wells=int(preset["n_wells"]),                 # prior
        n_plates=int(preset["n_plates"]),               # prior
        plate_sigma=float(preset["plate_sigma"]),       # prior
        depth_batch_sigma=depth_batch_sigma,
        kappa=float(preset["kappa"]),
    )
    # Honest bookkeeping: everything the preset supplied, plus dropout when it was not fitted.
    carried = list(_PRIOR_ONLY)
    if "dropout_mid" not in fitted:
        carried.append("dropout_shape")
    if "sigma_batch" not in fitted:
        carried += ["sigma_batch", "depth_batch_sigma"]
    return RNAEstimate(
        hypers=hypers, protocol=proto, n_cells=n_cells, n_genes=n_genes, n_batches=n_batches,
        cv_lib=cv_lib, gene_means=gene_means, gamma_shape=gshape, gamma_scale=gscale,
        fitted=fitted, per_batch=per_batch, carried=carried,
    )
