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
    dropout_mid/shape logistic zero-fraction-vs-mean curve          (only with fit_dropout=True)
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
dispersion, reported as such.

The dispersion is fitted within cell types.  Pooled over a sample of several cell types, the slope
``S`` also holds the variance between types, and a simulated tumour draws the variance between its
own cell types on top of the fitted dispersion, so a pooled fit counts that variance twice and gives
too many zeros at every mean above about 0.1 UMIs. On PBMC3k the pooled fit is 0.61 and the fit
within its 8 annotated types is 0.36. ``cell_types`` takes the types, or finds them by k-means
(``"auto"``, the default, 0.345 on PBMC3k); ``RNAEstimate.dispersion_pooled`` keeps the pooled value.
Within the annotated types, a negative binomial at 0.35 gives the zero fraction of PBMC3k at every
mean (median excess of the real zeros 0.0007 above 0.1 UMIs), so no zeros are left for a dropout.

Dropout is not fit by default. The logistic curve is fit to the total zero fraction, which already
includes the zeros the negative binomial draws, and the generator applies the dropout on top of
those draws and after the library size is set; fitting it therefore counts zeros twice and shrinks
the simulated libraries (on PBMC3k: median 1,227 against 2,198 UMIs). UMI counts are not
zero-inflated beyond the negative binomial (Svensson 2020, Nat Biotechnol 38:147), and Splatter
also leaves dropout off by default; ``fit_dropout=True`` fits it for data that need it.

Protocol-aware: ``estimate_rna(adata, protocol="10x")`` keeps droplet ambient/doublet priors;
``protocol="smartseq3"`` enables the per-cell well term, and
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
# how many plates the cells were spread over, or how much depth the cells on one plate shared.
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
    dispersion_pooled: float = float("nan")         # dispersion over all cells, types pooled
    n_types: int = 1                                # cell groups the dispersion was fitted within

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
                f"dispersion={h.dispersion:.3f} (pooled {self.dispersion_pooled:.3f}, "
                f"{self.n_types} types), sigma_batch={h.sigma_batch:.3f}, "
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


def _cell_type_labels(adata, cell_types, n_cells):
    """``cell_types`` as ``_cell_typings`` takes it: "auto", None, or one label per cell (an
    ``adata.obs`` column name is looked up)."""
    if cell_types is None or (isinstance(cell_types, str) and cell_types == "auto"):
        return cell_types
    if isinstance(cell_types, str):
        if not (hasattr(adata, "obs") and cell_types in adata.obs):
            raise ValueError(f"cell_types={cell_types!r} is neither 'auto' nor a column of adata.obs")
        return np.asarray(adata.obs[cell_types].values).astype(str)
    return np.asarray(cell_types).astype(str)


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


# Smallest group the dispersion is fitted in; smaller groups are left out of the average.
_MIN_TYPE_CELLS = 30
# Cells per k-means group the automatic typing aims for at least, so small data get fewer groups.
_CELLS_PER_TYPE = 50


def _type_labels(counts, genes, n_types, seed=0):
    """Cell groups by k-means on the leading principal components of log-normalised counts.

    Uses only the columns ``genes``: the 2,000 most variable of them after ``log1p(1e4 * count /
    library)``, with the library over every gene, centred, then up to 20 principal components.
    """
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    lib = counts.sum(axis=1)
    scale = np.divide(1e4, lib, out=np.zeros_like(lib, dtype=float), where=lib > 0)
    y = counts[:, genes] * scale[:, None]
    np.log1p(y, out=y)
    top = np.argsort(y.var(axis=0))[::-1][:2000]
    y = y[:, top]
    y -= y.mean(axis=0)
    n_pc = int(min(20, y.shape[0] - 1, y.shape[1]))
    pcs = PCA(n_components=n_pc, random_state=seed).fit_transform(y)
    return KMeans(n_clusters=n_types, n_init=4, random_state=seed).fit_predict(pcs)


def _cell_typings(counts, cell_types, n_types, seed=0):
    """How to group the cells for the dispersion fit: a list of ``(labels, genes)`` pairs, or None.

    The dispersion is fitted within each group of ``labels`` on the columns ``genes`` (None: every
    gene), and the pairs are averaged. None (returned for ``cell_types`` None, and for ``"auto"`` on
    fewer than ``2 * _CELLS_PER_TYPE`` cells) pools every cell; per-cell labels are used as given.
    ``"auto"`` finds ``n_types`` groups by k-means, cross-fitted: the genes are split at random into
    two halves, the cells are grouped on one half and the dispersion fitted on the other, and both
    ways round. A clustering partly follows the noise of the genes it is built on, so the
    within-group variance of those same genes would understate the dispersion; the other half does
    not share that noise.
    """
    n = counts.shape[0]
    if cell_types is None:
        return None
    if not isinstance(cell_types, str):
        labels = np.asarray(cell_types).astype(str)
        if labels.shape != (n,):
            raise ValueError(f"cell_types has {labels.size} labels for {n} cells")
        return [(labels, None)]
    k = int(min(int(n_types), n // _CELLS_PER_TYPE))
    expressed = np.flatnonzero(counts.sum(axis=0) > 0)
    if k < 2 or expressed.size < 20:
        return None
    perm = np.random.default_rng(seed).permutation(expressed)
    halves = (np.sort(perm[: perm.size // 2]), np.sort(perm[perm.size // 2:]))
    return [(_type_labels(counts, halves[a], k, seed=seed), halves[1 - a]) for a in (0, 1)]


def _fit_dispersion_within(counts, typings):
    """NB dispersion within cell groups, from the within-group moments pooled over the groups.

    Within group ``k`` (``n_k`` cells, library squared CV ``c2_k`` over every gene), gene ``g`` has
    ``Var - mean = mean^2 * (phi*(1+c2_k) + c2_k)`` (see the module notes). Summing ``n_k*(Var - mean)``
    and ``n_k*mean^2`` over the groups gives one slope per gene, ``S_g = phi*(1+c2_g) + c2_g`` with
    ``c2_g`` the ``n_k*mean^2``-weighted ``c2_k``, so ``phi_g = (S_g - c2_g)/(1 + c2_g)``; the estimate
    is the median ``phi_g`` over the genes whose overall mean reaches ``_fit_dispersion``'s threshold.
    The variance is unbiased (ddof 1) and ``mean^2`` is corrected by ``Var/n_k``, and the genes are
    chosen on the overall mean, so groups of about a hundred cells add no small-sample bias that a
    median of per-group fits would. With several typings (the two halves of a cross-fit), their
    estimates are averaged. ``typings`` None, or no group of at least ``_MIN_TYPE_CELLS`` cells,
    pools every cell with ``_fit_dispersion``.
    """
    lib_all = counts.sum(axis=1)
    m_all = counts.mean(axis=0)
    vals = []
    for labels, genes in (typings or []):
        cols = np.arange(counts.shape[1]) if genes is None else np.asarray(genes)
        m = m_all[cols]
        if not np.any(m > 0):
            continue
        thr = max(1.0, float(np.median(m[m > 0])))
        keep = m >= thr
        if keep.sum() < 5:
            keep = m > 0
        cols = cols[keep]
        num, den, c2w = np.zeros(cols.size), np.zeros(cols.size), np.zeros(cols.size)
        for g in np.unique(labels):
            rows = np.flatnonzero(labels == g)
            n_k = rows.size
            if n_k < _MIN_TYPE_CELLS:
                continue
            sub = counts[np.ix_(rows, cols)]
            mk = sub.mean(axis=0)
            vk = sub.var(axis=0, ddof=1)
            lib = lib_all[rows]
            c2 = float(np.var(lib, ddof=1) / (np.mean(lib) ** 2 + 1e-12))
            m2 = np.maximum(mk ** 2 - vk / n_k, 0.0)
            num += n_k * (vk - mk)
            den += n_k * m2
            c2w += n_k * m2 * c2
        ok = den > 0
        if ok.any():
            c2g = c2w[ok] / den[ok]
            vals.append(float(np.median((num[ok] / den[ok] - c2g) / (1.0 + c2g))))
    if not vals:
        return _fit_dispersion(counts, float(np.std(lib_all) / (np.mean(lib_all) + 1e-12)))
    return max(float(np.mean(vals)), 0.0)


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


def _fit_dropout(counts, dispersion):
    """Logistic dropout-vs-mean (Splatter), fitted to the zeros the negative binomial does not make.

    The simulation draws counts from the negative binomial and then drops a count with probability
    ``1/(1+exp(shape*(log m - log mid)))``, so the zero fraction of a gene is
    ``p0 + (1 - p0) * p_drop`` with ``p0`` the negative binomial's own zeros. The curve is therefore
    fitted to ``(zero - p0) / (1 - p0)``, with ``p0`` averaged over the cells of their libraries and
    the fitted ``dispersion``; fitting it to every zero counted the negative binomial's zeros twice.
    The moment fit of the dispersion absorbs part of any real dropout, so the midpoint is a lower
    bound: on counts drawn with a midpoint of 1.5 it comes out near 0.03, and the refit has 0.25 of
    its counts at zero against 0.32 in the data (0.12 without dropout).
    Fit by least squares on the per-gene (log mean, excess zero fraction) curve. Returns
    ``(dropout_mid, dropout_shape)``, or ``(0.0, 1.0)`` when the data hold no excess zeros or the fit
    fails.
    """
    from scipy.optimize import curve_fit

    m = counts.mean(axis=0)
    zero = (counts == 0).mean(axis=0)
    lib = counts.sum(axis=1)
    scale = lib / max(float(lib.mean()), 1e-12)
    d = max(float(dispersion), 1e-8)
    p0 = np.empty_like(m, dtype=float)
    for lo in range(0, len(m), 2048):                   # cells x genes in blocks, to bound memory
        mu = scale[:, None] * m[None, lo:lo + 2048]
        p0[lo:lo + 2048] = np.mean((1.0 + mu * d) ** (-1.0 / d), axis=0)
    use = (m > 0) & (p0 < 1.0 - 1e-6)
    excess = np.clip((zero - p0) / np.maximum(1.0 - p0, 1e-12), 0.0, 1.0)
    if use.sum() < 8 or excess[use].max() < 0.02:       # no zeros beyond the negative binomial's
        return 0.0, 1.0
    x = np.log(m[use])
    y = excess[use]

    def logistic(lm, log_mid, shape):
        return 1.0 / (1.0 + np.exp(shape * (lm - log_mid)))

    try:
        p_init = [float(np.median(x)), 1.0]
        popt, _ = curve_fit(logistic, x, y, p0=p_init,
                            bounds=([x.min() - 5, 1e-2], [x.max() + 5, 20.0]), maxfev=5000)
        log_mid, shape = popt
        mid = float(np.exp(log_mid))
        if not np.isfinite(mid) or mid <= 0:
            return 0.0, 1.0
        return mid, float(shape)
    except (RuntimeError, ValueError):
        return 0.0, 1.0


def _fit_batch_params(counts, labels, typings=None):
    """Cross-batch hyper-parameters (needs >=2 batches).

    Returns ``(sigma_batch, depth_batch_sigma, per_batch)``. The dispersion of each batch is fitted
    within the cell groups of ``typings`` (see ``_cell_typings``; None pools the cells of the batch).

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
        in_b = labs == b
        cb = counts[in_b]
        lib = cb.sum(axis=1)
        lib = lib[lib > 0]
        mu_b = float(lib.mean()) if lib.size else 0.0
        per_batch[b] = dict(mu_lib=mu_b,
                            dispersion=_fit_dispersion_within(
                                cb, None if typings is None else [(t[in_b], g) for t, g in typings]),
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
def estimate_rna(adata, protocol="10x", batch_key=None, count_model="nb", fit_dropout=False,
                 cell_types="auto", n_types=8):
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
        ``depth_batch_sigma`` are fit; otherwise they are carried from the protocol preset.
    fit_dropout : bool
        Fit the logistic dropout curve (default False). The negative binomial already draws the
        zeros of UMI data, and the dropout is applied on top of it, so fitting it double-counts
        zeros and shrinks the simulated library sizes (see the module notes).
    count_model : str
        Not used by the fit; the fitted dispersion is the negative-binomial one.
    cell_types : "auto" | str | array-like | None
        The cell types the dispersion is fitted within. A simulated tumour draws the variation
        between its own cell types, so a dispersion that also holds the variation between the cell
        types of the reference counts it twice and gives too many zeros: on PBMC3k, pooling every
        cell gives 0.61 and the 8 annotated types give 0.36. ``"auto"`` (default) groups the cells
        by k-means into ``n_types`` groups (fewer for small data, about 50 cells per group at least),
        cross-fitted over two halves of the genes (see ``_cell_typings``); this gives 0.345 on
        PBMC3k. An ``adata.obs`` column name or one label per cell uses those types. None pools every
        cell. Each batch is fitted within its own cells.
    n_types : int
        Number of k-means groups for ``cell_types="auto"`` (default 8, the number of annotated cell
        types in PBMC3k).

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
    dispersion_pooled = _fit_dispersion(counts, cv_lib)
    typings = _cell_typings(counts, _cell_type_labels(adata, cell_types, n_cells), n_types)
    dispersion = _fit_dispersion_within(counts, typings)
    gene_means, gshape, gscale = _fit_mean_expression(counts)
    fitted += ["mu_lib", "sigma_lib", "dispersion"]

    # dropout (protocol-aware)
    if fit_dropout:
        dropout_mid, dropout_shape = _fit_dropout(counts, dispersion)
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
        sigma_batch, depth_batch_sigma, per_batch = _fit_batch_params(counts, labels, typings)
        fitted += ["sigma_batch", "depth_batch_sigma"]
        # Dispersion is a within-batch quantity: estimating it on the pooled matrix lets
        # cross-batch depth and beta differences leak in and inflate phi. Average the per-batch
        # NB dispersions instead (each already library-de-biased on its own cells).
        phis = [v["dispersion"] for v in per_batch.values() if np.isfinite(v["dispersion"])]
        if phis:
            dispersion = float(np.mean(phis))

    # The whole plate layer (well spread, wells per plate, plates, shared per-plate depth) comes
    # from the protocol preset: a bare count matrix carries no plate map. Smart-seq3 therefore keeps
    # its 384-well layout.
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
        dispersion_pooled=dispersion_pooled,
        n_types=1 if typings is None else int(max(len(np.unique(t)) for t, _ in typings)),
    )
