"""Method-of-moments ``estimate_visium()`` for the Visium assay.

The spatial analogue of the scRNA ``estimate()`` and DNA ``estimate_dna()``: fits the technical
hyper-parameters of the spatial batch model (``VisiumBatchHyperParams`` in ``batch.py``) from a
real Visium AnnData reduced
to **per-spot counts + spot coordinates** — so realistic spatial-technical magnitudes are *learned,
not guessed*. The fitted ``VisiumBatchHyperParams`` drive ``Visium`` directly (same field names).

This is **not ABC**: every fitted Visium parameter maps to a marginal / spatial summary statistic, so
estimation is closed-form / 1-D curve-fit — fast and simulation-free (ABC stays for the biology).

What is fit (and from what):

    mu_counts          mean per-spot library size (total UMIs / spot)            (always)
    sigma_counts       NON-spatial (nugget) part of the per-spot log-library sd  (always)
    field_sigma        SPATIAL (autocorrelated) part of the per-spot log-library (always)
                       sd — the capture-field strength
    field_lengthscale  capture-field autocorrelation scale: squared-exponential   (always)
                       length-scale fit of the per-spot library-size residual
                       autocorrelation (Moran's-I correlogram / variogram range)
    kappa (dm)         per-spot count overdispersion (DM concentration)           (always)
    nb_dispersion (nb) per-spot count overdispersion phi (NB)                     (count_model=nb)
    sigma_batch        per-gene cross-section log-fold-change scale               (needs >=2 sections)

The headline decomposition.  The per-spot library total is ``mu_counts * field_s * noise_s`` with
``field_s`` the spatially-autocorrelated capture field (mean 1) and ``noise_s`` the i.i.d. LogNormal
depth noise. So the variance of ``log(total)`` splits into a SPATIAL part (the field, recovered as
the autocorrelation amplitude ``a`` at zero lag) and a NON-SPATIAL nugget (the i.i.d. noise,
``1-a``):  ``field_sigma = sqrt(a * Var[log total])``, ``sigma_counts = sqrt((1-a) * Var[log total])``.
This is exactly what separates the spatial technical layer from the per-spot depth noise — the
spatial analogue of the scRNA estimator's library-size de-biasing of the dispersion.

The ``_PRIOR_ONLY`` escape (as in ``estimate``/``estimate_dna``) carries the genuinely
unidentifiable-from-one-section fields honestly: ``ambient_frac`` (needs off-tissue/background
spots), ``edge_sigma`` and ``diffusion_sigma`` (confounded with biology in a single section), and
``sigma_batch`` on single-section data (the per-gene factor is constant within a section, so it is
confounded with the true expression — identifiable only across >=2 sections, exactly as the scRNA
estimator needs >=2 batches).

Spots outside the tissue lift two of those. Given a raw matrix of all spots with Space Ranger's
``in_tissue`` flag, a spot outside the tissue holds no cells, so its counts are bleed from the
tissue, and ``_fit_bleed`` fits SpotClean's spot-swapping model to them (Ni et al. 2022, the model
:func:`iscc.data.visium_background.bleed` simulates): the bleeding rate, its distal share (the
slide-wide floor) and the kernel sd become ``bleed_rate``, ``bleed_distal`` and ``bleed_sigma``,
which replace ``ambient_frac`` and ``diffusion_sigma`` (both set to 0, since the bleed carries
the floor and moves counts between spots). The spots within ``bleed_min_dist`` pitches of the tissue
are left out of that fit, because a spot just outside the tissue flag can still hold tissue.
``_fit_off_tissue``, the earlier ambient-plus-Gaussian description, is still reported in the
diagnostics. The same raw matrix then fits ``edge_sigma`` (with ``edge_reference="tissue"``) from the
tissue's outermost ring only, as the drop there beyond what the bleed predicts (``_fit_edge``):
farther in, real depth keeps changing with the tissue itself, which one section cannot separate
from biology.

numpy / scipy only (no JAX / R); stateless and config-driven.
"""
from dataclasses import dataclass, field

import numpy as np

from .batch import VisiumBatchHyperParams

# Fields a single-section counts+coords estimate cannot identify; carried from the preset, never
# claimed as "fit" — the spatial analogue of estimate()'s ambient/doublet escape.
_PRIOR_ONLY = ("ambient_frac", "edge_sigma", "diffusion_sigma")


@dataclass
class VisiumEstimate:
    """Fitted Visium technical parameters, ready to drive ``Visium(...)``.

    ``hypers`` is a ``VisiumBatchHyperParams`` with the fitted magnitudes; ``visium_kwargs()``
    returns the hyper fields (minus ``count_model``, an explicit assay constructor arg) as a kwargs
    dict, so the fit splats straight back into the assay without redefinition. ``fitted`` lists the
    fields actually learned; everything else is carried from the preset and flagged honestly.
    ``spots_per_tissue`` / ``occupied_fraction`` are spatial-extent diagnostics (not hyper fields).
    """
    hypers: VisiumBatchHyperParams
    count_model: str
    n_spots: int
    n_genes: int
    n_sections: int
    spots_per_tissue: int
    occupied_fraction: float
    fitted: list
    diagnostics: dict = field(default_factory=dict)

    def visium_kwargs(self):
        """Hyper-parameter overrides to splat into ``Visium`` (minus ``count_model``).

        ``count_model`` is an explicit assay constructor arg, so drive the assay as
        ``Visium(count_model=est.count_model, **est.visium_kwargs())``.
        """
        h = self.hypers.to_dict()
        h.pop("count_model", None)
        h.pop("protocol", None)
        return h

    def __repr__(self):
        h = self.hypers
        disp = (f"kappa={h.kappa:.1f}" if self.count_model == "dm"
                else f"nb_dispersion={h.nb_dispersion:.3f}")
        return (f"VisiumEstimate(count_model={self.count_model!r}, n_spots={self.n_spots}, "
                f"spots_per_tissue={self.spots_per_tissue}, "
                f"occupied_fraction={self.occupied_fraction:.2f}, "
                f"mu_counts={h.mu_counts:.0f}, sigma_counts={h.sigma_counts:.3f}, "
                f"field_sigma={h.field_sigma:.3f}, field_lengthscale={h.field_lengthscale:.2f}, "
                f"{disp}, sigma_batch={h.sigma_batch:.3f}, fitted={self.fitted})")


# --------------------------------------------------------------------------------------
# Input normalization
# --------------------------------------------------------------------------------------
def _dense(x):
    """AnnData / DataFrame / sparse / ndarray -> dense (spots, genes) float array."""
    if hasattr(x, "X"):
        x = x.X
    if hasattr(x, "toarray"):
        x = x.toarray()
    if hasattr(x, "values"):
        x = x.values
    return np.asarray(x, dtype=float)


def _spot_coords(adata, coords):
    if coords is not None:
        return np.asarray(coords, dtype=float)
    if hasattr(adata, "obsm") and "spatial" in getattr(adata, "obsm", {}):
        return np.asarray(adata.obsm["spatial"], dtype=float)
    raise ValueError("no spot coordinates: pass `coords` or an AnnData with obsm['spatial'].")


# --------------------------------------------------------------------------------------
# Individual estimators
# --------------------------------------------------------------------------------------
def _autocorrelation_fit(coords, resid, nbins=12, max_frac=0.6):
    """Squared-exponential length-scale + zero-lag amplitude of a spatial residual.

    Builds the empirical autocorrelation correlogram ``r(d) = <z_i z_j>_{|x_i-x_j|≈d} / Var[z]``
    over distance bins (z = mean-centred residual), then least-squares fits the generative SE form
    ``r(d) = a * exp(-d^2 / (2*ell^2))``. ``a`` is the spatially-correlated fraction of the residual
    variance (the field), ``ell`` the autocorrelation length (= ``field_lengthscale``). Returns
    ``(ell, a, diagnostics)`` (``ell``/``a`` may be ``None`` if too few pairs).
    """
    coords = np.asarray(coords, dtype=float)
    z = np.asarray(resid, dtype=float)
    z = z - z.mean()
    var = float(np.mean(z * z))
    n = z.shape[0]
    if n < 8 or var <= 0:
        return None, None, {}
    iu = np.triu_indices(n, k=1)
    d = coords[iu[0]] - coords[iu[1]]
    dist = np.sqrt(np.sum(d * d, axis=1))
    zz = z[iu[0]] * z[iu[1]] / var
    dmax = np.percentile(dist, 100 * max_frac)             # structure lives at shorter lags
    edges = np.linspace(dist.min(), dmax, nbins + 1)
    centers, corr = [], []
    for i in range(nbins):
        m = (dist >= edges[i]) & (dist < edges[i + 1])
        if m.sum() >= 20:
            centers.append(0.5 * (edges[i] + edges[i + 1]))
            corr.append(float(zz[m].mean()))
    centers, corr = np.asarray(centers), np.asarray(corr)
    if centers.size < 3:
        return None, None, {"corr_centers": centers.tolist(), "corr": corr.tolist()}
    from scipy.optimize import curve_fit

    def se(dd, a, ell):
        return a * np.exp(-dd * dd / (2.0 * ell * ell))

    try:
        a0 = float(np.clip(corr[0], 0.05, 1.0))
        ell0 = float(max(centers.mean(), 1.0))
        popt, _ = curve_fit(se, centers, corr, p0=[a0, ell0],
                            bounds=([0.0, 1e-3], [1.5, centers.max() * 6]), maxfev=20000)
        a = float(np.clip(popt[0], 0.0, 1.0))
        ell = float(popt[1])
    except (RuntimeError, ValueError):
        return None, None, {"corr_centers": centers.tolist(), "corr": corr.tolist()}
    return ell, a, {"corr_centers": centers.tolist(), "corr": corr.tolist()}


def _fit_library_and_field(counts, coords):
    """Per-spot library (``mu_counts``) + the spatial/nugget split of its log-residual.

    ``mu_counts`` is the mean per-spot total (the field is mean-1, so it does not move the mean).
    ``Var[log total]`` is split by the autocorrelation amplitude ``a`` into the spatial field
    (``field_sigma``) and the i.i.d. nugget (``sigma_counts``); the length-scale is the SE range.
    Only on-tissue (non-empty) spots inform the fit.
    """
    total = counts.sum(axis=1)
    occ = total > 0
    tot = total[occ]
    if tot.size < 8:
        return None
    mu_counts = float(tot.mean())
    logt = np.log(tot)
    var_log = float(np.var(logt))
    resid = logt - logt.mean()
    ell, a, diag = _autocorrelation_fit(coords[occ], resid)
    if a is None:                                          # no spatial structure resolvable
        return dict(mu_counts=mu_counts, sigma_counts=float(np.sqrt(max(var_log, 0.0))),
                    field_sigma=0.0, field_lengthscale=None, var_log=var_log, amp=0.0, diag=diag)
    field_sigma = float(np.sqrt(max(a * var_log, 0.0)))
    sigma_counts = float(np.sqrt(max((1.0 - a) * var_log, 0.0)))
    return dict(mu_counts=mu_counts, sigma_counts=sigma_counts, field_sigma=field_sigma,
                field_lengthscale=ell, var_log=var_log, amp=a, diag=diag)


def _fit_count_overdispersion(counts, count_model):
    """Per-spot count overdispersion -> ``kappa`` (DM) or ``nb_dispersion`` (NB).

    Builds the expected per-spot per-gene mean ``mu_sg = N_s * p_g`` from the section-mean
    composition ``p`` and the per-spot total ``N_s``, then reads the residual overdispersion:

      * DM: index of dispersion ``D = <(y-mu)^2/mu>`` inverts to ``kappa = (N - R)/(R - 1)`` with
        ``R = D/(1 - 1/G)`` (Dirichlet-Multinomial), median over spots.
      * NB: ``phi = <((y-mu)^2 - mu)/mu^2>`` (NB method-of-moments), median over spots.

    The section-mean composition stands in for the per-spot expectation; biological heterogeneity
    inflates it, so this is an *effective* overdispersion, reported honestly.
    """
    total = counts.sum(axis=1)
    occ = total > 0
    C = counts[occ]
    if C.shape[0] < 5:
        return None
    pooled = C.sum(axis=0)
    s = pooled.sum()
    if s <= 0:
        return None
    p = pooled / s
    n_genes = C.shape[1]
    cbar = 1.0 - 1.0 / n_genes if n_genes > 1 else 1.0
    kappas, phis = [], []
    for y in C:
        N = float(y.sum())
        mu = N * p
        keep = mu > 0
        if keep.sum() < 5:
            continue
        yk, muk = y[keep], mu[keep]
        if count_model == "nb":
            phis.append(float(np.mean(((yk - muk) ** 2 - muk) / muk ** 2)))
        else:
            D = float(np.mean((yk - muk) ** 2 / muk))
            R = D / cbar
            if R > 1.0:
                kappas.append((N - R) / (R - 1.0))
    if count_model == "nb":
        return max(float(np.median(phis)), 0.0) if phis else None
    return max(float(np.median(kappas)), 1e-6) if kappas else None


def _fit_sigma_batch(counts, labels):
    """Per-gene cross-section log-fold-change scale ``sigma_batch`` (needs >=2 sections).

    For each section take the per-gene composition (mean count normalized to sum 1), log it, then
    two-way-centre the section x gene log-composition matrix (remove per-section depth + per-gene
    effect); the residual sd is ``sigma_batch`` (the gene x section interaction = ``log beta_g``).
    Mirrors ``estimate._fit_batch_params``. ``None`` if fewer than two sections.
    """
    labs = np.asarray(labels).astype(str)
    uniq = list(dict.fromkeys(labs))
    if len(uniq) < 2:
        return None
    n_genes = counts.shape[1]
    expressed = np.ones(n_genes, dtype=bool)
    means = {}
    for b in uniq:
        m = counts[labs == b].mean(axis=0)
        means[b] = m
        expressed &= m >= 1.0
    if expressed.sum() < 10:
        pooled = counts.mean(axis=0)
        expressed = pooled >= max(0.5, np.median(pooled[pooled > 0]) if np.any(pooled > 0) else 0.5)
    rows = []
    for b in uniq:
        m = means[b][expressed]
        comp = m / m.sum() if m.sum() > 0 else np.full(m.size, 1.0 / max(m.size, 1))
        rows.append(np.log(np.maximum(comp, 1e-12)))
    L = np.vstack(rows)
    L = L - L.mean(axis=1, keepdims=True)
    L = L - L.mean(axis=0, keepdims=True)
    B = L.shape[0]
    df_corr = np.sqrt(B / (B - 1)) if B > 1 else 1.0
    return float(np.std(L) * df_corr)


# --------------------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------------------
def _pitch(coords):
    """Spot centre-to-centre spacing: the median nearest-neighbour distance over the grid."""
    from scipy.spatial import cKDTree
    d, _ = cKDTree(coords).query(coords, k=2)
    return float(np.median(d[:, 1]))


def _bleed_kernel(coords, sigma, cutoff=5.0):
    """Column-normalised Gaussian bleed kernel ``K[s, t]``: the share of spot ``t``'s bled mass that
    lands on spot ``s`` (``s != t``), for a kernel sd ``sigma`` in the units of ``coords``. Sparse,
    truncated at ``cutoff`` sd."""
    from scipy import sparse
    from scipy.spatial import cKDTree
    pairs = cKDTree(coords).query_pairs(cutoff * sigma, output_type="ndarray")
    if len(pairs) == 0:
        return sparse.csc_matrix((len(coords), len(coords)))
    d2 = np.sum((coords[pairs[:, 0]] - coords[pairs[:, 1]]) ** 2, axis=1)
    w = np.exp(-d2 / (2.0 * sigma * sigma))
    rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
    cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
    K = sparse.csc_matrix((np.concatenate([w, w]), (rows, cols)), shape=(len(coords),) * 2)
    colsum = np.asarray(K.sum(axis=0)).ravel()
    return K @ sparse.diags(np.divide(1.0, colsum, out=np.zeros_like(colsum), where=colsum > 0))


def _fit_off_tissue(counts, coords, in_tissue, sigmas=None):
    """Bleed and ambient RNA from the spots outside the tissue.

    A spot outside the tissue holds no cells, so whatever it counts came from elsewhere. Its total is
    modelled as a slide-wide ambient floor plus mRNA bled from every tissue spot through a Gaussian
    kernel:

        total_s / mu_in = a + beta * sum_t K_sigma[s, t] * total_t / mu_in      (s outside the tissue)

    with ``mu_in`` the mean total of a tissue spot, ``a`` the ambient floor as a fraction of it,
    ``beta`` the fraction of a spot's counts that bleeds to other spots, and ``K_sigma`` the
    column-normalised kernel of :func:`_bleed_kernel`. For each ``sigma`` on a grid (in units of the
    spot pitch) ``(a, beta)`` is fit within [0, 1] under a robust ``soft_l1`` loss, since spots
    outside the tissue have heavy-tailed totals (debris, tissue the flag missed); the ``sigma`` with
    the least robust cost wins. The tissue geometry enters through ``K``, so a ragged or curved edge
    is handled.

    Also returned: the off-tissue profile (median total against distance to the tissue), the inward
    tissue profile (median total against distance into the tissue, relative to the interior) with the
    part of its edge loss that bleed alone predicts, and the correlation of the ambient gene profile
    (spots far from the tissue) with the pooled tissue profile, which ``iscc`` takes as the soup.
    """
    from scipy.optimize import least_squares
    from scipy.spatial import cKDTree
    in_tissue = np.asarray(in_tissue, dtype=bool)
    if in_tissue.sum() < 10 or (~in_tissue).sum() < 10:
        return None
    pitch = _pitch(coords)
    xy = coords / pitch                                       # distances in spot pitches from here on
    total = np.asarray(counts.sum(axis=1), dtype=float).ravel()   # dense or sparse counts
    mu_in = float(total[in_tissue].mean())
    y = total / mu_in
    out = ~in_tissue
    d_out = cKDTree(xy[in_tissue]).query(xy[out])[0]          # off-tissue spot -> nearest tissue spot
    d_in = cKDTree(xy[out]).query(xy[in_tissue])[0]           # tissue spot -> nearest off-tissue spot
    src = np.where(in_tissue, y, 0.0)

    sigmas = np.round(np.arange(0.2, 3.01, 0.05), 3) if sigmas is None else np.asarray(sigmas, float)
    y_out = y[out]
    scale = max(1.4826 * float(np.median(np.abs(y_out - np.median(y_out)))), 1e-3)
    best = None
    for sig in sigmas:
        bled = (_bleed_kernel(xy, sig) @ src)[out]
        fit = least_squares(lambda p: p[0] + p[1] * bled - y_out, [0.05, 0.2],
                            bounds=([0.0, 0.0], [1.0, 1.0]), loss="soft_l1", f_scale=scale)
        if best is None or fit.cost < best[0]:
            best = (fit.cost, float(sig), float(fit.x[0]), float(fit.x[1]))
    _, sigma, a, beta = best

    # iscc's `_diffuse` smooths composition with a row-normalised kernel that includes the spot
    # itself; at this sd on a hexagonal grid it keeps this share of a spot's own expression
    ring = np.array([0.0] + [1.0] * 6 + [np.sqrt(3.0)] * 6 + [2.0] * 6 + [np.sqrt(7.0)] * 12 + [3.0] * 6)
    smooth_keep = float(1.0 / np.exp(-ring ** 2 / (2.0 * sigma * sigma)).sum())

    def profile(dist, val, step=0.5, dmax=8.0):
        bins = np.arange(0.0, dmax + step, step)
        idx = np.digitize(dist, bins) - 1
        rows = []
        for i in range(len(bins) - 1):
            m = idx == i
            if m.sum() >= 5:
                rows.append((float(0.5 * (bins[i] + bins[i + 1])), float(np.median(val[m])), int(m.sum())))
        return rows

    # inward profile, and the edge loss bleed alone predicts: a tissue spot keeps (1 - beta) of its own
    # mass and receives beta-weighted mass only from other tissue spots, so its edge neighbours lose
    # what bleeds out of the tissue
    K = _bleed_kernel(xy, sigma)
    inflow = K @ src
    kept = (1.0 - beta) * src + beta * inflow
    interior = d_in >= 5.0
    rel_obs = y[in_tissue] / np.median(y[in_tissue][interior]) if interior.any() else None
    rel_bleed = (kept[in_tissue] / src[in_tissue].clip(1e-12))
    rel_bleed = rel_bleed / np.median(rel_bleed[interior]) if interior.any() else None

    far = out.copy()
    far[out] = d_out >= max(4.0 * sigma, 3.0)
    amb_r = None
    amb = np.asarray(counts[far].sum(axis=0), dtype=float).ravel()
    tis = np.asarray(counts[in_tissue].sum(axis=0), dtype=float).ravel()
    if far.sum() >= 10 and amb.sum() > 0 and tis.sum() > 0:
        keep = (amb + tis) > 0
        amb_r = float(np.corrcoef(np.log1p(amb[keep] / amb.sum() * 1e6),
                                  np.log1p(tis[keep] / tis.sum() * 1e6))[0, 1])

    return dict(
        pitch=pitch, mu_in=mu_in, sigma_pitch=sigma, ambient_frac=a, bleed_frac=beta,
        smooth_keep=smooth_keep,
        n_in=int(in_tissue.sum()), n_out=int(out.sum()), n_far=int(far.sum()),
        off_profile=profile(d_out, y[out]),
        off_profile_model=profile(d_out, a + beta * inflow[out]),
        in_profile=profile(d_in, rel_obs) if rel_obs is not None else None,
        in_profile_bleed=profile(d_in, rel_bleed) if rel_bleed is not None else None,
        ambient_profile_r=amb_r,
    )


def _fit_bleed(counts, coords, in_tissue, sigmas=None, min_dist=0.0):
    """SpotClean's spot-swapping parameters from the spots outside the tissue.

    The model of :func:`iscc.data.visium_background.bleed`: a spot outside the tissue counts, as a
    fraction of the mean tissue spot,

        y_s = r (1 - g) sum_t W_sigma[s, t] src_t  +  r g mean_t(src_t)

    with ``src`` the tissue spots' totals (0 elsewhere), ``W_sigma`` SpotClean's kernel (Gaussian,
    normalised over every spot of the slide including the source) and the mean over all slide spots.
    For each ``sigma`` on a grid (in spot pitches) the two non-negative coefficients ``r (1 - g)`` and
    ``r g`` are fit under a robust ``soft_l1`` loss with ``r <= 1``; the ``sigma`` with the least cost
    wins. ``min_dist`` (pitches) drops the off-tissue spots nearer the tissue than that, since spots
    just outside the tissue flag can still hold some tissue, which inflates the bleed they imply.

    Returns a dict with ``bleed_rate`` (r), ``bleed_distal`` (g), ``sigma_pitch``, ``n_fit``, the
    share of the slide's UMIs counted outside the tissue (``background_share``), and the cost.
    """
    from scipy.optimize import least_squares
    from scipy.spatial import cKDTree
    from .visium_background import bleed_kernel
    in_tissue = np.asarray(in_tissue, dtype=bool)
    if in_tissue.sum() < 10 or (~in_tissue).sum() < 10:
        return None
    pitch = _pitch(coords)
    xy = coords / pitch
    total = np.asarray(counts.sum(axis=1), dtype=float).ravel()
    mu_in = float(total[in_tissue].mean())
    y = total / mu_in
    out = ~in_tissue
    d_out = cKDTree(xy[in_tissue]).query(xy[out])[0]
    use = d_out >= float(min_dist)
    if use.sum() < 10:
        return None
    src = np.where(in_tissue, y, 0.0)
    m = float(src.mean())                                     # the distal share reaches every spot alike
    y_fit = y[out][use]
    scale = max(1.4826 * float(np.median(np.abs(y_fit - np.median(y_fit)))), 1e-3)
    sigmas = np.round(np.arange(0.25, 8.01, 0.25), 3) if sigmas is None else np.asarray(sigmas, float)
    best = None
    for sig in sigmas:
        prox = (bleed_kernel(xy, sig) @ src)[out][use]
        fit = least_squares(lambda p: p[0] * prox + p[1] * m - y_fit, [0.2, 0.05],
                            bounds=([0.0, 0.0], [1.0, 1.0]), loss="soft_l1", f_scale=scale)
        b1, b2 = (float(v) for v in fit.x)
        if b1 + b2 > 1.0:                                     # r = b1 + b2 must stay a share
            continue
        if best is None or fit.cost < best[0]:
            best = (float(fit.cost), float(sig), b1, b2)
    if best is None:
        return None
    cost, sigma, b1, b2 = best
    r = b1 + b2
    return dict(bleed_rate=r, bleed_distal=(b2 / r if r > 0 else 0.0), sigma_pitch=sigma,
                n_fit=int(use.sum()), min_dist=float(min_dist), cost=cost,
                background_share=float(total[out].sum() / total.sum()))


def _fit_edge(counts, coords, in_tissue, bleed_rate, bleed_distal, sigma_pitch):
    """The tissue-edge capture loss beyond what spot swapping explains, from the outermost ring.

    Bleed alone dims a tissue's outer spots, since some of their counts leave the tissue; the rest of
    the drop on the outermost ring (the tissue spots nearest a spot without tissue) is the edge
    falloff of ``edge_reference="tissue"``. ``edge_sigma`` is chosen so that the assay's model, the
    falloff on a flat tissue followed by the bleed, puts the outermost ring at the observed depth
    relative to the interior (5+ pitches in). Only the outermost ring is used: farther in, real depth
    keeps changing with the tissue itself.
    """
    from scipy.optimize import brentq
    from scipy.spatial import cKDTree
    from .visium_background import bleed_kernel
    in_tissue = np.asarray(in_tissue, dtype=bool)
    pitch = _pitch(coords)
    xy = coords / pitch
    total = np.asarray(counts.sum(axis=1), dtype=float).ravel()
    d_in = cKDTree(xy[~in_tissue]).query(xy[in_tissue])[0]
    ring = np.abs(d_in - d_in.min()) < 0.25
    interior = d_in >= 5.0
    if ring.sum() < 5 or interior.sum() < 5:
        return None
    obs = float(np.median(total[in_tissue][ring]) / np.median(total[in_tissue][interior]))
    W = bleed_kernel(xy, sigma_pitch)
    depth = np.zeros(len(xy))
    depth[in_tissue] = d_in - d_in.min()

    def ring_depth(e):
        src = np.where(in_tissue, 1.0 - e * np.exp(-3.0 * depth), 0.0)
        out = ((1.0 - bleed_rate) * src + bleed_rate * (1.0 - bleed_distal) * (W @ src)
               + bleed_rate * bleed_distal * src.mean())[in_tissue]
        return float(np.median(out[ring]) / np.median(out[interior]))

    pred0 = ring_depth(0.0)
    if obs >= pred0:
        e = 0.0
    elif obs <= ring_depth(0.9):
        e = 0.9
    else:
        e = float(brentq(lambda x: ring_depth(x) - obs, 0.0, 0.9, xtol=1e-4))
    return dict(edge_sigma=e, observed_ring=obs, bleed_predicted_ring=pred0, n_ring=int(ring.sum()))


def estimate_visium(adata, *, count_model="dm", coords=None, section_labels=None,
                    occupied_fraction=None, in_tissue=None, bleed_min_dist=1.5):
    """Fit ``VisiumBatchHyperParams`` from a real Visium AnnData (MoM / curve-fit).

    Parameters
    ----------
    adata : AnnData | DataFrame | ndarray
        Per-spot integer counts, spots x genes (raw, not normalized).
    count_model : {"dm", "nb"}
        Count emission (chosen). Selects ``kappa`` (DM) vs ``nb_dispersion`` (NB) as the fit target.
    coords : array | None
        Per-spot ``(row, col)`` coordinates ``(n_spots, 2)``. Defaults to ``adata.obsm['spatial']``.
    section_labels : sequence | None
        Per-spot section label. With >=2 sections, ``sigma_batch`` (the per-gene factor) is fit;
        otherwise it is carried from the preset and flagged not-fit (single-section confounding).
    occupied_fraction : float | None
        Fraction of laid spots that are on-tissue. Defaults to the fraction of non-empty spots
        (spots with >0 counts). Sets ``spots_per_tissue`` = on-tissue spot count.
    in_tissue : sequence of bool | None
        Per-spot tissue flag (Space Ranger's ``in_tissue``). Defaults to ``adata.obs['in_tissue']``
        when present. Given a raw matrix of ALL spots with this flag, the library, field and
        overdispersion fits use the tissue spots only, and the spots outside the tissue fit
        SpotClean's spot-swapping model (:func:`_fit_bleed`): ``bleed_rate``, ``bleed_distal`` and
        ``bleed_sigma`` become fitted (``ambient_frac`` and ``diffusion_sigma`` are set to 0), and
        the fit is reported in ``diagnostics["bleed"]`` beside ``diagnostics["off_tissue"]``. Without
        it, the fit is as before.
    bleed_min_dist : float, default 1.5
        Spots outside the tissue nearer to it than this (in spot pitches) are left out of the bleed
        fit. On 10x's breast section the first ring alone pushes the bleeding rate to its bound
        (0.99) while every cut from 1.5 pitches out gives 0.2-0.3.

    Returns
    -------
    VisiumEstimate
    """
    if in_tissue is None and hasattr(adata, "obs") and "in_tissue" in getattr(adata, "obs", {}):
        in_tissue = adata.obs["in_tissue"].to_numpy()
    if in_tissue is not None:
        from scipy import sparse
        in_tissue = np.asarray(in_tissue).astype(bool)
        X = adata.X if hasattr(adata, "X") else adata
        counts_all = sparse.csr_matrix(X) if sparse.issparse(X) else _dense(adata)
        # Only a real raw matrix counts anything outside the tissue: iscc's own AnnData carries the
        # flag too, but its empty spots emit nothing, so it takes the original path unchanged.
        if not in_tissue.all() and counts_all[~in_tissue].sum() > 0:
            coords_all = _spot_coords(adata, coords)
            est = estimate_visium(_dense(counts_all[in_tissue]), count_model=count_model,
                                  coords=coords_all[in_tissue],
                                  section_labels=(None if section_labels is None
                                                  else np.asarray(section_labels)[in_tissue]),
                                  occupied_fraction=float(in_tissue.mean()), in_tissue=None)
            est.n_spots = int(len(in_tissue))
            off = _fit_off_tissue(counts_all, coords_all, in_tissue)
            if off is not None:
                est.diagnostics["off_tissue"] = off
            bl = _fit_bleed(counts_all, coords_all, in_tissue, min_dist=bleed_min_dist)
            if bl is not None:
                h = est.hypers.to_dict()
                h["bleed_rate"] = bl["bleed_rate"]
                h["bleed_distal"] = bl["bleed_distal"]
                h["bleed_sigma"] = bl["sigma_pitch"] * float(h["spot_pitch"])
                h["ambient_frac"] = 0.0
                h["diffusion_sigma"] = 0.0
                est.fitted += ["bleed_rate", "bleed_distal", "bleed_sigma"]
                est.diagnostics["bleed"] = bl
                edge = _fit_edge(counts_all, coords_all, in_tissue, bl["bleed_rate"], bl["bleed_distal"],
                                 bl["sigma_pitch"])
                if edge is not None:
                    h["edge_sigma"] = edge["edge_sigma"]
                    h["edge_reference"] = "tissue"
                    est.fitted += ["edge_sigma"]
                    est.diagnostics["edge"] = edge
                est.hypers = VisiumBatchHyperParams(**h)
            return est
    if count_model not in ("dm", "nb"):
        raise ValueError(f"count_model must be 'dm' or 'nb', got {count_model!r}")
    counts = _dense(adata)
    if counts.ndim != 2:
        raise ValueError(f"expected a 2D (spots x genes) count matrix, got shape {counts.shape}")
    coords = _spot_coords(adata, coords)
    n_spots, n_genes = counts.shape
    if coords.shape[0] != n_spots:
        raise ValueError(f"coords has {coords.shape[0]} spots, counts has {n_spots}")

    h = VisiumBatchHyperParams(count_model=count_model).to_dict()
    fitted, diag = [], {}

    total = counts.sum(axis=1)
    occ = total > 0
    n_occ = int(occ.sum())
    occupied = float(occupied_fraction) if occupied_fraction is not None else (n_occ / max(n_spots, 1))
    spots_per_tissue = n_occ

    # mu_counts / sigma_counts / field_sigma / field_lengthscale (the per-spot library + field split)
    lf = _fit_library_and_field(counts, coords)
    if lf is not None:
        h["mu_counts"] = lf["mu_counts"]
        h["sigma_counts"] = lf["sigma_counts"]
        fitted += ["mu_counts", "sigma_counts"]
        h["field_sigma"] = lf["field_sigma"]
        fitted.append("field_sigma")
        if lf["field_lengthscale"] is not None:
            h["field_lengthscale"] = lf["field_lengthscale"]
            fitted.append("field_lengthscale")
        diag.update({k: lf[k] for k in ("var_log", "amp")})
        diag["correlogram"] = lf["diag"]

    # per-spot count overdispersion -> kappa (DM) or nb_dispersion (NB)
    disp = _fit_count_overdispersion(counts, count_model)
    if disp is not None:
        if count_model == "nb":
            h["nb_dispersion"] = disp
            fitted.append("nb_dispersion")
        else:
            h["kappa"] = disp
            fitted.append("kappa")
        diag["overdispersion"] = disp

    # per-gene batch factor (needs >=2 sections; otherwise prior-only)
    n_sections = 1
    if section_labels is not None:
        sb = _fit_sigma_batch(counts, section_labels)
        n_sections = len(set(np.asarray(section_labels).astype(str).tolist()))
        if sb is not None:
            h["sigma_batch"] = sb
            fitted.append("sigma_batch")
            diag["sigma_batch"] = sb

    hypers = VisiumBatchHyperParams(**h)
    return VisiumEstimate(
        hypers=hypers, count_model=count_model, n_spots=n_spots, n_genes=n_genes,
        n_sections=n_sections, spots_per_tissue=spots_per_tissue, occupied_fraction=occupied,
        fitted=fitted, diagnostics=diag,
    )


def estimate_visium_from_assay(assay, *, section_labels=None):
    """Fit from a *run* ``Visium`` instance (extracts per-spot counts + coords + occupancy).

    The synthetic recovery convenience: pulls the spot count matrix, the spot coordinates, and the
    on-tissue fraction straight from the run assay, then defers to :func:`estimate_visium`.
    """
    counts = assay.spot_counts.values
    coords = assay.spot_coords
    occupied_fraction = float((assay.obs["n_cells"].values > 0).mean())
    return estimate_visium(counts, count_model=assay.count_model, coords=coords,
                           section_labels=section_labels, occupied_fraction=occupied_fraction)
