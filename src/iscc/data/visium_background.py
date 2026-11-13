"""Spot swapping on a Visium slide: mRNA that bleeds from where it was released to other spots.

A Visium spot outside the tissue holds no cells, yet a real slide counts reads there: mRNA released
from the tissue drifts laterally before it is captured. SpotClean (Ni et al. 2022) models this as
a bleed of each spot's counts into every other spot of the slide, tissue or not:

    M'_s = (1 - r) M_s  +  r (1 - g) sum_t W[s, t] M_t  +  r g (1 / N) sum_t M_t

with ``M`` the expected UMI counts per spot and gene before bleeding, ``r`` the bleeding rate (the
share of a spot's counts that leaves it), ``g`` the distal share of the bled counts (spread
uniformly over all ``N`` spots of the slide; the slide-wide ambient floor), and ``W`` a Gaussian
kernel of sd ``sigma`` over the slide's spot grid, normalised over destination spots for each
source. Total mass is conserved: the slide counts what the tissue released, redistributed.

:func:`bleed` applies this to the expected counts of :class:`iscc.data.Visium` after pooling and
before the count draw, so the draw's noise acts on the bled expectation, as in SpotClean's model.
It is off unless ``bleed_rate > 0``, and then draws no random numbers, so the assay's random stream
is unchanged.
"""
import numpy as np

__all__ = ["bleed_kernel", "bleed"]


def bleed_kernel(coords, sigma, cutoff=5.0, include_self=True):
    """Column-normalised Gaussian kernel over a spot grid.

    ``W[s, t]`` is the share of spot ``t``'s proximally bled counts that lands on spot ``s``. The
    kernel is truncated at ``cutoff`` sd and stored sparse, so a full slide (4,992 spots) costs a few
    neighbours per spot rather than a dense square matrix.

    Parameters
    ----------
    coords : array of shape (n_spots, 2)
        Spot centres, in the units ``sigma`` is given in.
    sigma : float
        Kernel sd.
    cutoff : float, default 5.0
        Truncation radius, in units of ``sigma``.
    include_self : bool, default True
        Whether a spot's own position takes a share of its bled counts, as in SpotClean's kernel,
        which weighs every spot by ``exp(-d^2 / 2 sigma^2)`` including the source (``d = 0``).

    Returns
    -------
    scipy.sparse.csc_matrix of shape (n_spots, n_spots)
        Columns sum to 1.
    """
    from scipy import sparse
    from scipy.spatial import cKDTree

    coords = np.asarray(coords, dtype=float)
    n = len(coords)
    pairs = cKDTree(coords).query_pairs(cutoff * float(sigma), output_type="ndarray")
    if len(pairs):
        d2 = np.sum((coords[pairs[:, 0]] - coords[pairs[:, 1]]) ** 2, axis=1)
        w = np.exp(-d2 / (2.0 * float(sigma) ** 2))
        rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
        cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
        vals = np.concatenate([w, w])
    else:
        rows = cols = np.zeros(0, dtype=int)
        vals = np.zeros(0)
    if include_self:
        rows = np.concatenate([rows, np.arange(n)])
        cols = np.concatenate([cols, np.arange(n)])
        vals = np.concatenate([vals, np.ones(n)])
    W = sparse.csc_matrix((vals, (rows, cols)), shape=(n, n))
    colsum = np.asarray(W.sum(axis=0)).ravel()
    return W @ sparse.diags(np.divide(1.0, colsum, out=np.zeros_like(colsum), where=colsum > 0))


def bleed(lib, comp, coords, rate, distal=0.0, sigma=0.0, include_self=True):
    """Redistribute a slide's expected counts by SpotClean's spot-swapping model.

    Parameters
    ----------
    lib : array of shape (n_spots,)
        Expected UMIs per spot before bleeding (zero for spots without tissue).
    comp : array of shape (n_spots, n_genes)
        Per-spot composition (rows sum to 1 where ``lib > 0``).
    coords : array of shape (n_spots, 2)
        Spot centres of the whole slide, in the units of ``sigma``.
    rate : float
        Bleeding rate ``r`` in [0, 1]: the share of each spot's counts that leaves it.
    distal : float, default 0.0
        Distal share ``g`` in [0, 1] of the bled counts, spread uniformly over every spot of the
        slide; the rest bleeds through the kernel.
    sigma : float, default 0.0
        Kernel sd. With ``sigma <= 0`` the proximal part stays at its source spot.
    include_self : bool, default True
        Passed to :func:`bleed_kernel`.

    Returns
    -------
    lib : ndarray of shape (n_spots,)
        Expected UMIs per spot after bleeding; it sums to the input total.
    comp : ndarray of shape (n_spots, n_genes)
        Per-spot composition after bleeding.
    received : ndarray of shape (n_spots,)
        The share of each spot's counts after bleeding that came from other spots: the ground truth
        of the per-spot contamination SpotClean estimates (1 for a spot outside the tissue).
    """
    rate, distal = float(rate), float(distal)
    if not (0.0 <= rate <= 1.0 and 0.0 <= distal <= 1.0):
        raise ValueError(f"bleed rate and distal share must lie in [0, 1], got {rate:g} and {distal:g}")
    lib = np.asarray(lib, dtype=float)
    comp = np.asarray(comp, dtype=float)
    M = lib[:, None] * comp
    n = len(lib)
    prox = rate * (1.0 - distal)
    if sigma > 0 and prox > 0:
        W = bleed_kernel(coords, sigma, include_self=include_self)
        moved = prox * (W @ M)
        self_keep = prox * W.diagonal()                    # the proximal share that lands back home
    else:
        moved = prox * M
        self_keep = np.full(n, prox)
    out = (1.0 - rate) * M + moved + (rate * distal / n) * M.sum(axis=0, keepdims=True)
    new_lib = out.sum(axis=1)
    new_comp = np.divide(out, new_lib[:, None], out=np.zeros_like(out), where=new_lib[:, None] > 0)
    own = (1.0 - rate + self_keep + rate * distal / n) * lib
    received = np.divide(new_lib - own, new_lib, out=np.zeros_like(new_lib), where=new_lib > 0)
    return new_lib, new_comp, np.clip(received, 0.0, 1.0)
