"""Method-of-moments / 1-D MLE ``estimate_dna()`` for the DNA assay (DESIGN_inference §C.1, M4).

The DNA analogue of the scRNA ``estimate()`` (``estimate.py``, M2): fits the technical
hyper-parameters of the DESIGN_features §D DNA batch model (``DNABatchHyperParams`` in
``batch.py``) from a real DNA-seq dataset reduced to **counts-level summary statistics** —
per-locus coverage, alt/ref counts, and a **per-locus called copy number** — so realistic
technical magnitudes are *learned, not guessed*. The fitted ``DNABatchHyperParams`` drive
``bulkDNA`` / ``scDNA`` / ``run_dna_batches`` directly (same field names, no redefinition).

This is **not ABC**: every DNA assay parameter maps directly to a marginal summary statistic, so
estimation is closed-form / 1-D MLE — fast and simulation-free. ABC stays reserved for the biology.

Conditional on called copy number (the one wrinkle).  Our depth model is copy-number-scaled
(coverage ∝ CN × efficiency). Real data does not hand you true CN, so estimation is a two-pass:
(1) run a CN caller (ASCAT/Sequenza for bulk; Ginkgo/HMMcopy for single-cell) to get per-locus CN,
then (2) estimate the *residual technical noise given that CN*. We are explicit that the technical
params are fit **conditional on the caller's CN** — exactly the CN regime re-injected when
simulating. This is the DNA analogue of M2's library-size de-biasing of the dispersion: separate
the technical layer from the part that is really biology.

What is fit (and from what), per modality:

    mu_depth          mean per-locus coverage                              (bulk + sc)
    kappa (DM)        DM concentration: coverage overdispersion AFTER       (bulk + sc)
                      dividing out CN x efficiency (index of dispersion)
    nb_dispersion(NB) NB per-locus phi from the same CN-conditioned residual(bulk + sc; depth_model=nb)
    gc_curve_sigma    curvature (log-space sd) of the coverage-vs-GC fit    (bulk + sc; needs GC)
    capture_sigma     LogNormal sd of per-target/amplicon coverage after    (WES/panel only)
                      removing GC+CN (depth-noise de-biased)
    error_rate        alt fraction at non-variant / hom-ref loci            (bulk + sc; needs mask)
    depth_batch_sigma sd of log mean-depth across batches                   (>=2 batches)
    ado_rate          fraction of known-het loci collapsing to ~0/~1 BAF    (single-cell only)
    beta_binom_conc   BAF overdispersion at het loci beyond Binomial        (single-cell only)
    ffpe_ct_rate      excess alt at C>T-eligible loci vs the error floor     (bulk + sc; needs ct mask)
    doublet_rate      NOT identifiable from counts -> prior-only (flagged)  (carried from preset)
    breadth, depth_model  chosen, not estimated (you know the assay)

The ``_PRIOR_ONLY`` escape that §B's ``estimate()`` uses (ambient/doublet) carries the genuinely
unidentifiable fields honestly instead of pretending to fit them. ``ado_rate`` /
``beta_binom_conc`` are the per-cell amplification/dropout layer — fittable only from single-cell
data; they are flagged not-fit (and carried from the preset) on bulk data, and ``kappa`` means
different things per modality (large from pooled bulk, small from lumpy MDA/MALBAC single-cell).

numpy / scipy only (no JAX / R); stateless and config-driven.
"""
from dataclasses import dataclass, field

import numpy as np

from .batch import DNABatchHyperParams
from .dna import DNA_BREADTH_PRESETS

# Single-cell depth_scale (scDNA.depth_scale): single-cell coverage is shallower than bulk.
_SC_DEPTH_SCALE = 0.3

# Amplification-regime / per-cell-layer defaults, mirroring bulkDNA/scDNA.mode_defaults().
_MODE_DEFAULTS = {
    "bulk": dict(kappa=2000.0, nb_dispersion=0.1, ado_rate=0.0,
                 beta_binom_conc=200.0, doublet_rate=0.0),
    "sc":   dict(kappa=5.0, nb_dispersion=0.5, ado_rate=0.20,
                 beta_binom_conc=30.0, doublet_rate=0.02),
}

# Fields a counts-only estimate cannot identify even with CN (need demultiplexing); carried from
# the preset, never claimed as "fit" — the DNA analogue of estimate()'s ambient/doublet escape.
_PRIOR_ONLY = ("doublet_rate",)

_HYPER_KEYS = set(DNABatchHyperParams().to_dict().keys())


def _preset_hypers(modality, breadth, depth_model):
    """Protocol-typical ``DNABatchHyperParams`` for (modality, breadth) — the prior baseline.

    Reconstructs exactly what ``bulkDNA`` / ``scDNA`` would build from the breadth preset +
    ``mode_defaults`` (incl. the single-cell ``depth_scale``), so prior-only / not-fit fields are
    carried honestly from the same place the generator uses.
    """
    if modality not in _MODE_DEFAULTS:
        raise ValueError(f"modality must be 'bulk' or 'sc', got {modality!r}")
    if breadth not in DNA_BREADTH_PRESETS:
        raise ValueError(f"unknown breadth {breadth!r}; choose from {sorted(DNA_BREADTH_PRESETS)}")
    params = {k: v for k, v in DNA_BREADTH_PRESETS[breadth].items() if k in _HYPER_KEYS}
    params.update(_MODE_DEFAULTS[modality])
    params["breadth"] = breadth
    params["depth_model"] = depth_model
    if modality == "sc":
        params["mu_depth"] = float(params.get("mu_depth", 30.0)) * _SC_DEPTH_SCALE
    return DNABatchHyperParams(**params)


@dataclass
class DNAEstimate:
    """Fitted DNA technical parameters, ready to drive ``bulkDNA`` / ``scDNA`` / ``run_dna_batches``.

    ``hypers`` is a ``DNABatchHyperParams`` with the fitted magnitudes; ``dna_kwargs()`` returns
    the hyper fields (minus ``breadth`` / ``depth_model``, which are explicit assay constructor
    args) as a kwargs dict, so the fit splats straight back into the assay without redefinition.
    ``fitted`` lists the fields actually learned from this data; everything else is carried from
    the (modality, breadth) preset and flagged honestly. ``diagnostics`` keeps realized intermediate
    quantities (per-sample kappa, realized GC/capture spread, het-BAF variance) for reporting.
    """
    hypers: DNABatchHyperParams
    modality: str
    breadth: str
    depth_model: str
    n_loci: int
    n_cells: int
    n_batches: int
    fitted: list
    diagnostics: dict = field(default_factory=dict)

    def dna_kwargs(self):
        """Hyper-parameter overrides to splat into ``bulkDNA`` / ``scDNA`` (minus breadth/model).

        ``breadth`` and ``depth_model`` are explicit assay constructor args, so drive the assay as
        ``scDNA(breadth=est.breadth, depth_model=est.depth_model, **est.dna_kwargs())``.
        """
        h = self.hypers.to_dict()
        h.pop("breadth", None)
        h.pop("depth_model", None)
        return h

    def __repr__(self):
        h = self.hypers
        disp = (f"kappa={h.kappa:.1f}" if self.depth_model == "dm"
                else f"nb_dispersion={h.nb_dispersion:.3f}")
        return (f"DNAEstimate(modality={self.modality!r}, breadth={self.breadth!r}, "
                f"n_loci={self.n_loci}, n_cells={self.n_cells}, n_batches={self.n_batches}, "
                f"mu_depth={h.mu_depth:.1f}, {disp}, gc_curve_sigma={h.gc_curve_sigma:.3f}, "
                f"capture_sigma={h.capture_sigma:.3f}, error_rate={h.error_rate:.2e}, "
                f"ado_rate={h.ado_rate:.3f}, beta_binom_conc={h.beta_binom_conc:.1f}, "
                f"fitted={self.fitted})")


# --------------------------------------------------------------------------------------
# Input normalization
# --------------------------------------------------------------------------------------
def _as2d(a, name):
    """Coerce a per-locus (1D) or cells x loci (2D) array to (n_samples, n_loci) float."""
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        a = a[None, :]
    elif a.ndim != 2:
        raise ValueError(f"{name} must be 1D (n_loci) or 2D (n_cells, n_loci), got {a.shape}")
    return a


# --------------------------------------------------------------------------------------
# Individual estimators
# --------------------------------------------------------------------------------------
def _fit_gc_curve(percopy, gc, mappability):
    """Quadratic (LOESS-equivalent) fit of log per-copy coverage on GC.

    The generative GC->coverage bias is ``exp(-strength*(gc-opt)^2)`` — exactly quadratic in
    log-space — so a degree-2 fit recovers the curve shape. ``percopy`` is per-locus coverage with
    the called CN divided out (and mappability removed if supplied); its smooth dependence on GC is
    the GC curve, and the *curvature* (log-space sd of the fitted curve) is ``gc_curve_sigma``.

    Returns ``(gc_fit, gc_sigma)`` where ``gc_fit`` is the centred fitted log-curve per locus
    (mean 0, a multiplicative shape) and ``gc_sigma`` its sd.
    """
    g = np.asarray(gc, dtype=float)
    pc = np.asarray(percopy, dtype=float)
    if mappability is not None:
        m = np.asarray(mappability, dtype=float)
        pc = np.divide(pc, m, out=np.zeros_like(pc), where=m > 0)
    ok = np.isfinite(pc) & (pc > 0) & np.isfinite(g)
    fit = np.zeros_like(pc)
    if ok.sum() < 5 or np.ptp(g[ok]) <= 0:
        return fit, 0.0
    coef = np.polyfit(g[ok], np.log(pc[ok]), 2)
    fit = np.polyval(coef, g)
    fit = fit - np.mean(fit[ok])                       # centre: a mean-1 multiplicative shape
    return fit, float(np.std(fit[ok]))


def _fit_depth_dispersion(coverage, cn, gc_fit, mappability, depth_model):
    """CN-conditioned coverage overdispersion -> ``kappa`` (DM) or ``nb_dispersion`` (NB).

    The crux of §C.1. Build the systematic expected per-locus coverage ``mu`` from the called CN x
    the fitted efficiency shape (GC curve x mappability), normalized to the *observed* total per
    sample (bulk: 1 sample of loci; single-cell: each cell is a sample). The residual of the
    observed coverage about ``mu`` is then pure technical overdispersion:

      * DM: the index of dispersion ``D = <(X-mu)^2/mu>`` estimates ``(1-1/n)*(N+kappa)/(kappa+1)``
        (Dirichlet-Multinomial), inverted for ``kappa = (N - R)/(R - 1)`` with ``R = D/(1-1/n)``.
        Large kappa (pooled bulk) ~ multinomial/Poisson; small kappa (lumpy single-cell MDA/MALBAC).
      * NB: ``phi = median_l ((X-mu)^2 - mu)/mu^2`` (the standard NB method-of-moments residual).

    This is the DNA analogue of M2's library-size de-biasing of the NB dispersion: divide out the
    part that is really CN (biology) before reading the technical dispersion.
    """
    eff_shape = np.exp(gc_fit)
    if mappability is not None:
        eff_shape = eff_shape * np.asarray(mappability, dtype=float)
    kappas, phis, d_indices = [], [], []
    n_loci = coverage.shape[1]
    cbar = 1.0 - 1.0 / n_loci if n_loci > 1 else 1.0
    for X, C in zip(coverage, cn):
        pred = np.asarray(C, dtype=float) * eff_shape
        sp = pred.sum()
        N = float(X.sum())
        if sp <= 0 or N <= 0:
            continue
        mu = N * pred / sp
        keep = mu > 0
        if keep.sum() < 5:
            continue
        Xk, muk = X[keep], mu[keep]
        D = float(np.mean((Xk - muk) ** 2 / muk))    # index of dispersion (depth-noise level)
        d_indices.append(D)
        if depth_model == "nb":
            # phi = E[((X-mu)^2 - mu)/mu^2]; MEAN over loci is unbiased for the per-locus variance,
            # whereas a single-sample median is biased low (median of a chi^2_1 residual ~ 0.45*var).
            phi_l = ((Xk - muk) ** 2 - muk) / muk ** 2
            phis.append(float(np.mean(phi_l)))
        else:
            R = D / cbar
            if R > 1.0:
                kappas.append((N - R) / (R - 1.0))
    d_index = float(np.median(d_indices)) if d_indices else 1.0
    if depth_model == "nb":
        nb = float(np.median(phis)) if phis else float("nan")
        return max(nb, 0.0), {"per_sample_nb": phis, "d_index": d_index}
    kappa = float(np.median(kappas)) if kappas else float("nan")
    return max(kappa, 1e-6), {"per_sample_kappa": kappas, "d_index": d_index}


def _fit_capture_sigma(coverage, cn, gc_fit, mappability):
    """Per-target (WES) / per-amplicon (panel) capture LogNormal sd, after removing GC + CN.

    Per locus, the coverage-per-copy with the GC curve (and mappability) divided out is
    ``capture_eff x depth-noise``; its log-space variance is ``capture_sigma^2 + depth_noise_var``.
    We subtract a depth-noise floor so the estimate isolates the systematic capture spread. The
    floor is the delta-method log-variance of a count with the *measured* index of dispersion
    ``d_index`` (Var(log X) ~ d_index/mu): at the DM overdispersion these data carry, the floor is
    far above the naive Poisson ``1/mu``. Deep panels make it negligible; for WES it is a genuine
    (M2-style) de-biasing. A panel-of-normals would pin it exactly; without one this is the honest
    counts-only estimate.
    """
    eff_shape = np.exp(gc_fit)
    if mappability is not None:
        eff_shape = eff_shape * np.asarray(mappability, dtype=float)
    eff_shape = np.where(eff_shape > 0, eff_shape, np.nan)
    # per-locus coverage with GC+mappability+CN removed (mean over samples)
    percopy = np.divide(coverage, np.maximum(cn, 1e-9), out=np.zeros_like(coverage), where=cn > 0)
    resid = percopy / eff_shape[None, :]
    locus_mean = np.nanmean(np.where(resid > 0, resid, np.nan), axis=0)
    ok = np.isfinite(locus_mean) & (locus_mean > 0)
    if ok.sum() < 5:
        return 0.0
    var_log = float(np.var(np.log(locus_mean[ok])))
    # Depth-noise floor on log(locus_mean). With S samples/cells the per-locus mean averages out
    # the depth noise as ~ S/reads (Poisson) — for a single bulk sample this is small relative to a
    # real per-target capture spread; for single-cell it is averaged over many cells, so the
    # systematic capture cleanly separates. (A single bulk WES sample cannot fully separate capture
    # from dispersion without a panel-of-normals, so its capture_sigma is honestly biased high.)
    n_samp = coverage.shape[0]
    total_reads = coverage.sum(axis=0)
    floor = float(np.mean(n_samp / np.maximum(total_reads[ok], 1.0)))
    return float(np.sqrt(max(var_log - floor, 0.0)))


def _fit_error_rate(coverage, alt, homref_mask):
    """Sequencing error floor = pooled alt fraction at non-variant / hom-ref loci.

    At a hom-ref locus the only alt signal is per-base error (true VAF 0 -> p_eff = error_rate), so
    the pooled alt/coverage over hom-ref loci is a direct estimate of ``error_rate``.
    """
    if homref_mask is None or homref_mask.sum() == 0:
        return None
    cov = coverage[:, homref_mask]
    a = alt[:, homref_mask]
    denom = float(cov.sum())
    if denom <= 0:
        return None
    return float(a.sum() / denom)


def _fit_ffpe_ct_rate(coverage, alt, homref_mask, ct_sites, error_rate):
    """Excess alt at C>T-eligible hom-ref loci over the (non-CT) error floor.

    FFPE deamination adds ``ffpe_ct_rate`` on top of ``error_rate`` at C-sites, so the alt fraction
    at hom-ref CT loci minus the baseline ``error_rate`` estimates it. Returns ``None`` if there is
    no CT signal to fit.
    """
    if ct_sites is None or homref_mask is None or error_rate is None:
        return None
    ct = np.asarray(ct_sites, dtype=bool)
    sel = homref_mask & ct
    if sel.sum() == 0:
        return None
    denom = float(coverage[:, sel].sum())
    if denom <= 0:
        return None
    ct_alt_frac = float(alt[:, sel].sum() / denom)
    return max(ct_alt_frac - error_rate, 0.0)


def _fit_ado_and_conc(coverage, alt, het_mask, min_cov=8, extreme=0.05):
    """Single-cell ADO rate + Beta-Binomial concentration from known-het loci.

    At a het locus (true BAF ~ 0.5) allelic dropout collapses the cell to ~0 or ~1 BAF. The
    non-collapsed het observations are Beta-Binomial around 0.5, whose BAF variance beyond the
    Binomial floor gives the concentration by method-of-moments:

        Var(BAF) = p(1-p)/(c+1) * (1 + c/n)   with p=0.5
      => c = (p(1-p) - V) / (V - p(1-p)*<1/n>)

    The raw fraction of extreme het BAFs over-counts ADO: at low coverage the Beta-Binomial alone
    lands in the extremes. We **de-bias** with the coverage-aware Beta-Binomial extreme probability
    ``q`` at the fitted concentration (the analogue of M2 subtracting the Binomial/library floor):

        frac_extreme = ado + (1-ado)*q   =>   ado = (frac_extreme - q) / (1 - q)

    Returns ``(ado_rate, beta_binom_conc, diagnostics)``; either may be ``None`` if not estimable.
    """
    if het_mask is None:
        return None, None, {}
    het_mask = np.asarray(het_mask, dtype=bool)
    if het_mask.ndim == 2:                             # per-observation (cell, locus) het mask
        sel = het_mask
    else:                                              # per-locus mask -> broadcast over cells
        sel = np.broadcast_to(het_mask, coverage.shape)
    if sel.sum() == 0:
        return None, None, {}
    cov = coverage[sel]
    a = alt[sel]
    ok = cov >= min_cov
    if ok.sum() < 10:
        return None, None, {}
    cov_ok = cov[ok]
    baf = (a[ok] / cov_ok)
    is_extreme = (baf <= extreme) | (baf >= 1.0 - extreme)
    frac_extreme = float(np.mean(is_extreme))

    inner = baf[~is_extreme]
    cov_inner = cov_ok[~is_extreme]
    diag = {"frac_extreme_het": frac_extreme, "n_het_obs": int(ok.sum())}
    if inner.size < 10:
        return frac_extreme, None, diag
    V = float(np.mean((inner - 0.5) ** 2))
    m1 = float(np.mean(1.0 / np.maximum(cov_inner, 1.0)))
    p1p = 0.25
    denom = V - p1p * m1
    conc = float(max((p1p - V) / denom, 1e-3)) if denom > 0 else None
    diag.update(het_baf_var=V, het_inv_cov=m1)

    # de-bias the extreme fraction by the Beta-Binomial extreme probability at the fitted conc
    ado_rate = frac_extreme
    if conc is not None:
        from scipy.stats import betabinom

        ab = conc * 0.5
        ns = cov_ok.astype(int)
        klow = np.floor(extreme * ns).astype(int)
        khigh = np.ceil((1.0 - extreme) * ns).astype(int)
        q = float(np.mean(betabinom.cdf(klow, ns, ab, ab)
                          + (1.0 - betabinom.cdf(khigh - 1, ns, ab, ab))))
        ado_rate = float(np.clip((frac_extreme - q) / max(1.0 - q, 1e-6), 0.0, 1.0))
        diag.update(bb_extreme_q=q)
    return ado_rate, conc, diag


def _fit_depth_batch_sigma(batch_depths):
    """sd of log mean-depth across batches/samples (needs >=2)."""
    d = np.asarray([x for x in batch_depths if x and x > 0], dtype=float)
    if d.size < 2:
        return None
    return float(np.std(np.log(d)))


# --------------------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------------------
def estimate_dna(coverage, alt_counts, called_cn, *, modality="bulk", breadth="wgs",
                 depth_model="dm", gc=None, mappability=None, ct_sites=None,
                 variant_mask=None, het_mask=None, batch_depths=None):
    """Fit ``DNABatchHyperParams`` from counts-level DNA summary statistics (MoM/MLE; M4 §C.1).

    Parameters
    ----------
    coverage : array
        Per-locus coverage. 1D ``(n_loci,)`` for bulk; 2D ``(n_cells, n_loci)`` for single-cell.
    alt_counts : array
        Alt read counts, same shape as ``coverage``.
    called_cn : array
        **Caller** per-locus total copy number (the two-pass conditioning), broadcastable to
        ``coverage``'s shape. Estimation is explicitly conditional on this CN.
    modality : {"bulk", "sc"}
        Bulk vs single-cell. ``ado_rate`` / ``beta_binom_conc`` are fit only for ``"sc"``; on bulk
        they are flagged not-fit and carried from the preset (and vice-versa is moot — they don't
        exist for bulk). ``kappa`` means different regimes per modality.
    breadth : {"wgs", "wes", "panel"}
        Capture breadth (chosen, not estimated). ``capture_sigma`` is fit only for WES/panel.
    depth_model : {"dm", "nb"}
        Depth emission (chosen). Selects ``kappa`` (DM) vs ``nb_dispersion`` (NB) as the fit target.
    gc, mappability : array | None
        Reference per-locus GC content / mappability ``(n_loci,)``. GC enables ``gc_curve_sigma``
        and (with CN) sharpens the dispersion / capture fits; mappability is divided out when given.
    ct_sites : array | None
        Boolean per-locus C>T-eligible mask -> ``ffpe_ct_rate``.
    variant_mask : array | None
        Boolean per-locus mask, True at known variant sites. Hom-ref loci (``~variant_mask``) give
        the ``error_rate`` floor.
    het_mask : array | None
        Boolean per-locus mask of known-het loci (single-cell ``ado_rate`` / ``beta_binom_conc``).
    batch_depths : sequence | None
        Per-batch mean depths (>=2) -> ``depth_batch_sigma``.

    Returns
    -------
    DNAEstimate
    """
    coverage = _as2d(coverage, "coverage")
    alt = _as2d(alt_counts, "alt_counts")
    cn = _as2d(called_cn, "called_cn")
    n_cells, n_loci = coverage.shape
    if alt.shape != coverage.shape:
        raise ValueError(f"alt_counts {alt.shape} must match coverage {coverage.shape}")
    if cn.shape[1] != n_loci:
        raise ValueError(f"called_cn has {cn.shape[1]} loci, coverage has {n_loci}")
    if cn.shape[0] == 1 and n_cells > 1:
        cn = np.repeat(cn, n_cells, axis=0)            # broadcast a per-locus CN over cells

    preset = _preset_hypers(modality, breadth, depth_model)
    h = preset.to_dict()
    fitted, diag = [], {}

    # mu_depth: mean per-locus coverage (compositional total / n_loci, so CN-invariant in the mean)
    h["mu_depth"] = float(coverage.mean())
    fitted.append("mu_depth")

    # GC curve (needs reference GC); also feeds the dispersion / capture conditioning
    percopy_locus = np.divide(coverage, np.maximum(cn, 1e-9), out=np.zeros_like(coverage),
                              where=cn > 0).mean(axis=0)
    gc_fit = np.zeros(n_loci)
    if gc is not None:
        gc_fit, gc_sigma = _fit_gc_curve(percopy_locus, gc, mappability)
        h["gc_curve_sigma"] = gc_sigma
        fitted.append("gc_curve_sigma")
        diag["gc_curve_sigma"] = gc_sigma

    # CN-conditioned coverage overdispersion -> kappa (DM) or nb_dispersion (NB)
    disp, disp_diag = _fit_depth_dispersion(coverage, cn, gc_fit, mappability, depth_model)
    diag.update(disp_diag)
    if np.isfinite(disp):
        if depth_model == "nb":
            h["nb_dispersion"] = disp
            fitted.append("nb_dispersion")
        else:
            h["kappa"] = disp
            fitted.append("kappa")

    # capture efficiency spread (WES / panel only)
    if breadth in ("wes", "panel"):
        h["capture_sigma"] = _fit_capture_sigma(coverage, cn, gc_fit, mappability)
        fitted.append("capture_sigma")
        diag["capture_sigma"] = h["capture_sigma"]

    # error floor + FFPE C>T excess (need a variant mask)
    homref = None
    if variant_mask is not None:
        homref = ~np.asarray(variant_mask, dtype=bool)
        if ct_sites is not None:                       # measure the floor off non-CT loci
            err = _fit_error_rate(coverage, alt, homref & ~np.asarray(ct_sites, dtype=bool))
        else:
            err = _fit_error_rate(coverage, alt, homref)
        if err is not None:
            h["error_rate"] = err
            fitted.append("error_rate")
        ffpe = _fit_ffpe_ct_rate(coverage, alt, homref, ct_sites, h["error_rate"])
        if ffpe is not None:
            h["ffpe_ct_rate"] = ffpe
            fitted.append("ffpe_ct_rate")

    # single-cell per-cell amplification/dropout layer (ado + beta-binomial concentration)
    if modality == "sc":
        ado, conc, ado_diag = _fit_ado_and_conc(coverage, alt, het_mask)
        diag.update(ado_diag)
        if ado is not None:
            h["ado_rate"] = ado
            fitted.append("ado_rate")
        if conc is not None:
            h["beta_binom_conc"] = conc
            fitted.append("beta_binom_conc")

    # cross-batch depth spread (needs >=2 batches)
    n_batches = 1
    if batch_depths is not None:
        dbs = _fit_depth_batch_sigma(batch_depths)
        if dbs is not None:
            h["depth_batch_sigma"] = dbs
            fitted.append("depth_batch_sigma")
            n_batches = len([x for x in batch_depths if x and x > 0])
            diag["depth_batch_sigma"] = dbs

    hypers = DNABatchHyperParams(**h)
    return DNAEstimate(
        hypers=hypers, modality=modality, breadth=breadth, depth_model=depth_model,
        n_loci=n_loci, n_cells=(1 if modality == "bulk" else n_cells),
        n_batches=n_batches, fitted=fitted, diagnostics=diag,
    )


# --------------------------------------------------------------------------------------
# Convenience: build the reduced summary statistics straight from a run DNA assay instance
# --------------------------------------------------------------------------------------
def estimate_dna_from_assay(assay, *, batch_depths=None, het_atol=0.15):
    """Fit from a *run* ``bulkDNA`` / ``scDNA`` instance (extracts the reduced statistics).

    Pulls coverage / alt / **true CN** (the perfect-caller stand-in: in synthetic recovery the CN
    is exact, exactly the two-pass conditioning) / GC / mappability / C-site / variant + het masks
    from the assay's surfaced ground truth, then defers to :func:`estimate_dna`. Modality and
    breadth are read off the assay. ``het_mask`` is loci whose true alt fraction is ~0.5.
    """
    from .dna import bulkDNA, scDNA, genome_features

    if isinstance(assay, bulkDNA):
        modality = "bulk"
        df = assay.observed_data
        genes = list(df.index)
        coverage = df["coverage"].values
        alt = df["alt_counts"].values
        cn = df["true_cn"].values
        af = df["true_alt_fraction"].values
    elif isinstance(assay, scDNA):
        modality = "sc"
        genes = list(assay.genes)
        coverage = assay.coverage.values
        alt = assay.alt_counts.values
        cn = assay.true_cn.values
        af_cells = assay.true_alt_fraction.values             # per-cell (cells x loci) VAF
        af = af_cells.mean(axis=0)                            # per-locus mean (variant floor only)
    else:
        raise TypeError(f"expected a run bulkDNA/scDNA instance, got {type(assay).__name__}")

    gc, mappability, ct_sites = genome_features(genes)
    variant_mask = af > 1e-9
    if modality == "sc":
        # per-OBSERVATION het mask: a locus with mean VAF ~0.5 may be a clonal mixture (cells
        # hom-ref + cells hom-alt), not a per-cell het — only genuinely-het cells inform ADO/conc.
        het_mask = (af_cells > het_atol) & (af_cells < 1.0 - het_atol)
    else:
        het_mask = np.abs(af - 0.5) < het_atol
    return estimate_dna(
        coverage, alt, cn, modality=modality, breadth=assay.breadth,
        depth_model=assay.hypers.depth_model, gc=gc, mappability=mappability,
        ct_sites=ct_sites, variant_mask=variant_mask, het_mask=het_mask,
        batch_depths=batch_depths,
    )
