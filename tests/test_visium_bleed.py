"""Spot swapping in the Visium assay (`bleed_rate`, `bleed_distal`, `bleed_sigma`) and the tissue-
referenced edge falloff (`edge_reference="tissue"`). Both are off by default, and then the assay is
unchanged; with them on, a slide counts reads outside the tissue, as a real slide does."""
import hashlib

import numpy as np
import pandas as pd
import pytest

from iscc.data import Visium
from iscc.data.batch import VisiumBatch, VisiumBatchHyperParams
from iscc.data.estimate_visium import _fit_bleed
from iscc.data.visium_background import bleed, bleed_kernel

PITCH = 2.0


def _section(radius=9.0, side=40, n_genes=30, seed=0, density=1.2):
    """A disk of tissue in the middle of a square slide: two profiles, 3-6 cells per spot area."""
    rng = np.random.default_rng(seed)
    genes = [f"G{g}" for g in range(n_genes)]
    base = {k: rng.gamma(2.0, 1.0, n_genes) for k in "AB"}
    ids, crd, exp, typ = [], [], [], []
    centre = np.array([side / 2.0, side / 2.0])
    for _ in range(int(np.pi * radius ** 2 * density)):
        p = centre + rng.uniform(-radius, radius, 2)
        if np.linalg.norm(p - centre) > radius:
            continue
        k = "A" if p[0] < centre[0] else "B"
        ids.append(f"C{len(ids)}")
        crd.append(p)
        exp.append(base[k] * rng.lognormal(0.0, 0.2, n_genes))
        typ.append(k)
    return {"cell_exp": pd.DataFrame(exp, index=ids, columns=genes),
            "cell_crd": pd.DataFrame(crd, index=ids, columns=["row", "col"]),
            "cell_type": pd.DataFrame(typ, index=ids, columns=["cell_id"])}, side


def _run(**kw):
    cd, side = _section()
    return Visium(seed=3, spot_pitch=PITCH, spot_radius=1.0, **kw).run(cd, grid_side=side)


def _digest(vz):
    return hashlib.sha256(np.ascontiguousarray(vz.spot_counts.to_numpy(np.int64)).tobytes()).hexdigest()[:16]


# digest of the default assay on this section, recorded with the code before this change (27c1b78)
DEFAULT_DIGEST = "cdf7ab731f44d5f7"


def test_kernel_columns_sum_to_one():
    xy = np.array([[i * PITCH, j * PITCH] for i in range(6) for j in range(6)], float)
    for inc in (True, False):
        W = bleed_kernel(xy, 1.7, include_self=inc)
        assert np.allclose(np.asarray(W.sum(axis=0)).ravel(), 1.0)
    assert bleed_kernel(xy, 1.7, include_self=False).diagonal().max() == 0.0


def test_bleed_conserves_mass_and_reaches_empty_spots():
    rng = np.random.default_rng(0)
    xy = np.array([[i * PITCH, j * PITCH] for i in range(12) for j in range(12)], float)
    tissue = (np.abs(xy - 11).max(axis=1) < 6)
    lib = np.where(tissue, rng.uniform(5e3, 2e4, len(xy)), 0.0)
    comp = rng.dirichlet(np.ones(8), len(xy))
    new_lib, new_comp, rec = bleed(lib, comp, xy, rate=0.4, distal=0.25, sigma=1.7)
    assert np.isclose(new_lib.sum(), lib.sum())
    assert np.allclose((new_lib[:, None] * new_comp).sum(axis=0), (lib[:, None] * comp).sum(axis=0))
    assert (new_lib[~tissue] > 0).all()                            # the distal floor reaches every spot
    assert np.allclose(rec[~tissue], 1.0) and (rec[tissue] < 1.0).all()
    edge = tissue & (np.abs(xy - 11).max(axis=1) >= 4)             # the tissue's outer rings lose depth
    inner = tissue & (np.abs(xy - 11).max(axis=1) < 2)
    assert (new_lib[edge] / lib[edge]).mean() < (new_lib[inner] / lib[inner]).mean()


def test_rate_zero_is_the_identity():
    rng = np.random.default_rng(1)
    xy = rng.uniform(0, 20, (30, 2))
    lib = rng.uniform(0, 10, 30)
    comp = rng.dirichlet(np.ones(5), 30)
    new_lib, new_comp, rec = bleed(lib, comp, xy, rate=0.0, distal=0.5, sigma=2.0)
    assert np.allclose(new_lib, lib) and np.allclose(new_comp, comp) and np.allclose(rec, 0.0)
    with pytest.raises(ValueError):
        bleed(lib, comp, xy, rate=1.5)


def test_default_assay_unchanged():
    a = _run()
    b = _run(bleed_rate=0.0, edge_reference="grid")
    assert _digest(a) == _digest(b)
    assert "bled_in" not in a.obs
    if DEFAULT_DIGEST is not None:
        assert _digest(a) == DEFAULT_DIGEST
    off = a.obs["n_cells"].to_numpy() == 0
    assert a.spot_counts.to_numpy()[off].sum() == 0                # without bleed nothing leaves the tissue


def test_bleed_puts_counts_outside_the_tissue():
    vz = _run(bleed_rate=0.3, bleed_distal=0.2, bleed_sigma=PITCH, ambient_frac=0.0)
    off = vz.obs["n_cells"].to_numpy() == 0
    tot = vz.spot_counts.to_numpy().sum(axis=1)
    assert tot[off].mean() > 0
    assert 0.0 < tot[off].sum() / tot.sum() < 0.5
    assert np.allclose(vz.obs["bled_in"].to_numpy()[off], 1.0)
    # same seed, same draws upstream of the count draw: the expected depth is only moved, not made
    ref = _run(ambient_frac=0.0)
    assert np.isclose(tot.sum(), ref.spot_counts.to_numpy().sum(), rtol=0.05)


def test_tissue_edge_reference_dims_the_tissue_edge_not_the_slide_edge():
    xy = np.array([[i * PITCH, j * PITCH] for i in range(20) for j in range(20)], float)
    tissue = np.linalg.norm(xy - 19.0, axis=1) < 10.0
    h = VisiumBatchHyperParams(edge_sigma=0.3, edge_reference="tissue")
    f = VisiumBatch(h, seed=0)._edge_falloff(xy, in_tissue=tissue)
    from scipy.spatial import cKDTree
    d = cKDTree(xy[~tissue]).query(xy[tissue])[0]
    ft = f[tissue]
    assert ft[d == d.min()].mean() < ft[d == d.max()].mean()        # dimmest at the tissue's edge
    assert np.isclose(ft[d == d.min()].min(), 0.7)
    g = VisiumBatch(VisiumBatchHyperParams(edge_sigma=0.3), seed=0)._edge_falloff(xy, in_tissue=tissue)
    assert not np.allclose(f, g)                                    # the grid reference differs
    with pytest.raises(ValueError):
        VisiumBatch(VisiumBatchHyperParams(edge_sigma=0.3, edge_reference="slide"), seed=0)._edge_falloff(xy)


def test_fit_recovers_the_bleed_from_iscc_output():
    true = dict(bleed_rate=0.3, bleed_distal=0.3, bleed_sigma=PITCH)
    cd, side = _section(radius=12.0, side=60)
    vz = Visium(seed=5, spot_pitch=PITCH, spot_radius=1.0, ambient_frac=0.0, **true).run(cd, grid_side=side)
    in_t = vz.obs["n_cells"].to_numpy() > 0
    fit = _fit_bleed(vz.spot_counts.to_numpy(float), vz.spot_coords, in_t)
    assert abs(fit["sigma_pitch"] - 1.0) <= 0.3
    assert abs(fit["bleed_rate"] - 0.3) <= 0.1
    assert abs(fit["bleed_distal"] - 0.3) <= 0.15


def test_estimate_visium_recovers_bleed_and_edge_from_a_raw_slide():
    """iscc's AnnData of a slide with bleed carries counts outside the tissue, so estimate_visium reads
    it like a raw Space Ranger matrix: the bleed from the spots outside, the edge from the outer ring."""
    from iscc.data import estimate_visium
    true = dict(bleed_rate=0.3, bleed_distal=0.3, bleed_sigma=PITCH, edge_sigma=0.25,
                edge_reference="tissue", ambient_frac=0.0, field_sigma=0.0)
    # an interior 5+ pitches from the edge, dense enough that no spot inside the tissue is left empty
    cd, side = _section(radius=30.0, side=100, density=5.0)
    vz = Visium(seed=7, spot_pitch=PITCH, spot_radius=1.0, **true).run(cd, grid_side=side)
    est = estimate_visium(vz.to_anndata(), coords=vz.spot_coords, bleed_min_dist=0.0)
    assert {"bleed_rate", "bleed_distal", "bleed_sigma", "edge_sigma"} <= set(est.fitted)
    assert est.hypers.bleed_rate == pytest.approx(0.3, abs=0.1)
    assert est.hypers.edge_reference == "tissue"
    assert est.hypers.edge_sigma == pytest.approx(0.25, abs=0.1)
