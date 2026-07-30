"""Materialize a physical tissue section from a count-based (deme) tumor.

The count engine (``GenotypeTumor``) tracks per-deme cell **counts**, not sub-deme positions —
8-59 cells share each integer deme coordinate. Spatial assays (Visium, MERFISH) and
cell-resolution views therefore have to place each deme's cells within its unit cell before they
can lay spots or draw a slide. A flat physical **section** additionally keeps only a fraction of
each deme's 3-D column (``section_frac`` = section thickness / column depth).

This module is the single canonical implementation of that step, so every spatial consumer places
cells the same way.
"""
import numpy as np
import pandas as pd


def spatialize(cell_data, section_frac=1.0, jitter=0.5, seed=0):
    """Give each cell a sub-deme ``(row, col)`` position, optionally thinning to a section.

    Each deme's cells are scattered uniformly within the deme's unit cell (its integer
    coordinate ``± jitter``); the count engine tracks only per-deme counts, so the intra-deme
    layout is cosmetic while the composition (and the sampled section depth) is exact. When
    ``section_frac < 1`` a random ``section_frac`` of each deme's cells is kept, modelling a thin
    flat section through the 3-D cell column.

    Parameters
    ----------
    cell_data : dict
        Per-cell ground-truth tables from the sampling stage; uses ``cell_deme`` (deme id per
        cell) and ``cell_crd`` (integer deme coordinates). All per-cell tables are carried over.
    section_frac : float, default 1.0
        Fraction of each deme's cells kept in the flat section. ``1.0`` keeps the whole column;
        smaller values model a thinner slice (fewer cells, so each spot pools fewer of them).
    jitter : float, default 0.5
        Half-width of the uniform scatter within a deme's unit cell. ``0`` leaves a deme's cells
        stacked at its centre; ``0.5`` fills the whole unit cell.
    seed : int, default 0
        RNG seed for the per-deme section subsample and the jitter.

    Returns
    -------
    dict
        A copy of ``cell_data`` restricted to the section's cells, with ``cell_crd`` replaced by
        continuous sub-deme ``(row, col)`` positions.
    """
    rng = np.random.default_rng(seed)
    crd = cell_data["cell_crd"]
    labels = np.asarray(crd.index)
    deme = cell_data["cell_deme"]["deme_id"].reindex(crd.index).to_numpy()
    rc = crd[["row", "col"]].to_numpy(float)

    keep = []
    for d in np.unique(deme):
        idx = np.where(deme == d)[0]
        k = idx.size if section_frac >= 1.0 else max(1, int(round(section_frac * idx.size)))
        keep.append(idx if k >= idx.size else rng.choice(idx, size=k, replace=False))
    keep = np.sort(np.concatenate(keep))
    keep_labels = labels[keep]
    xy = rc[keep] + rng.uniform(-jitter, jitter, size=(keep.size, 2))

    out = {k: (v.reindex(keep_labels).copy() if hasattr(v, "reindex") else v)
           for k, v in cell_data.items()}
    out["cell_crd"] = pd.DataFrame(xy, index=keep_labels, columns=["row", "col"])
    return out


def tissue_image(coords, grid_side, px=14, darkness=0.72, sigma_frac=0.32):
    """Rasterize cell positions into an H&E-like greyscale **morphology** background image.

    Bins the ``(row, col)`` positions into a ``grid_side*px`` square raster, smooths the density,
    and maps it to greyscale (denser tissue → darker). This is the tissue *image* a spatial slide
    sits on — morphology only, carrying no per-cell data — suitable for
    ``adata.uns['spatial'][lib]['images']['hires']``.

    Parameters
    ----------
    coords : array-like, shape (n_cells, 2)
        Sub-deme cell positions as ``(row, col)`` (e.g. from :func:`spatialize`).
    grid_side : int or tuple of int
        Capture-area extent in coordinate units: a scalar for a square area, or ``(rows, cols)``
        to match a non-square spot array (e.g. the 78 x 64 Visium v1 slide).
    px : int, default 14
        Pixels per coordinate unit; also the ``tissue_hires_scalef`` to record so that
        ``obsm['spatial']`` maps onto the image.
    darkness : float, default 0.72
        Maximum darkening of the densest tissue (0 = white everywhere, 1 = black at peak density).
    sigma_frac : float, default 0.32
        Gaussian smoothing sigma as a fraction of ``px``.

    Returns
    -------
    image : numpy.ndarray, shape (rows*px, cols*px, 3), float32
        RGB morphology image in ``[0, 1]``.
    scalef : float
        The ``tissue_hires_scalef`` (= ``px``) to store alongside the image.
    """
    from scipy.ndimage import gaussian_filter

    coords = np.asarray(coords, dtype=float)
    h_units, w_units = (int(grid_side), int(grid_side)) if np.ndim(grid_side) == 0 else \
        (int(grid_side[0]), int(grid_side[1]))
    H, W = h_units * int(px), w_units * int(px)
    dens = np.zeros((H, W))
    ri = np.clip((coords[:, 0] * px).astype(int), 0, H - 1)
    ci = np.clip((coords[:, 1] * px).astype(int), 0, W - 1)
    np.add.at(dens, (ri, ci), 1.0)
    dens = gaussian_filter(dens, sigma=px * sigma_frac)
    peak = dens.max()
    if peak > 0:
        dens /= peak
    image = np.stack([1.0 - darkness * dens] * 3, axis=-1).astype(np.float32)
    return image, float(px)
