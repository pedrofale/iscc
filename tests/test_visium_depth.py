"""Visium depth, count noise and spot geometry against what the estimator fits and a real slide shows.

* The mean depth of a tissue spot is ``mu_counts``, whatever share of the slide the tissue covers:
  the capture field is normalised over the tissue spots, and ambient RNA is part of the library.
* With spot swapping, ``mu_counts`` is the depth before swapping, and ``estimate_visium`` on a raw
  slide returns it, not the shallower depth the tissue keeps.
* The Dirichlet-multinomial's technical dispersion does not depend on the gene set, so a fitted
  ``kappa`` transfers, and the default is near-multinomial.
* The default spot is Visium v1's 55 um spot at a 100 um pitch, with a deme of ~50 um.
"""
import numpy as np
import pandas as pd
import pytest

from iscc.data import Visium, VisiumBatchHyperParams, estimate_visium

PITCH = 2.0


def _disk(radius=12.0, side=60, n_genes=30, density=4.0, seed=0):
    """A disk of tissue in the middle of a square slide (an eighth of its spots at the defaults),
    cells scattered uniformly at ``density`` cells per unit area, two profiles."""
    rng = np.random.default_rng(seed)
    genes = [f"G{g}" for g in range(n_genes)]
    base = {k: rng.gamma(2.0, 1.0, n_genes) for k in "AB"}
    n = int(np.pi * radius ** 2 * density)
    r = radius * np.sqrt(rng.uniform(size=n))
    th = rng.uniform(0.0, 2.0 * np.pi, n)
    crd = side / 2.0 + np.c_[r * np.cos(th), r * np.sin(th)]
    kind = np.where(crd[:, 0] < side / 2.0, "A", "B")
    ids = [f"C{i}" for i in range(n)]
    exp = np.vstack([base[k] for k in kind]) * rng.lognormal(0.0, 0.2, (n, n_genes))
    return {"cell_exp": pd.DataFrame(exp, index=ids, columns=genes),
            "cell_crd": pd.DataFrame(crd, index=ids, columns=["row", "col"]),
            "cell_type": pd.DataFrame(kind, index=ids, columns=["cell_id"])}, side


def _tissue_depth(vz):
    tis = vz.obs["n_cells"].to_numpy() > 0
    return vz.spot_counts.to_numpy().sum(axis=1)[tis].mean(), tis


# ----------------------------------------------------------------- depth over the tissue -------
@pytest.mark.parametrize("edge_reference", ["grid", "tissue"])
def test_capture_field_is_mean_one_over_the_tissue(edge_reference):
    """Normalised over the whole slide, a section on part of it took the field's mean over the
    tissue as a depth factor, anywhere from about 0.5 to 1.5 depending on the seed."""
    cd, side = _disk()
    for seed in range(1, 6):
        vz = Visium(seed=seed, edge_reference=edge_reference).run(cd, grid_side=side)
        tis = vz.obs["n_cells"].to_numpy() > 0
        assert 0.1 < tis.mean() < 0.5                               # the tissue covers part of the slide
        assert vz.capture_field[tis].mean() == pytest.approx(1.0, abs=1e-9)


def test_tissue_depth_is_mu_counts_on_a_partly_covered_slide():
    cd, side = _disk()
    ratios = []
    for seed in range(1, 6):
        vz = Visium(seed=seed, mu_counts=8000.0).run(cd, grid_side=side)
        depth, _ = _tissue_depth(vz)
        ratios.append(depth / 8000.0)
    ratios = np.asarray(ratios)
    assert np.all(np.abs(ratios - 1.0) < 0.15), ratios               # per seed: the per-spot noise only
    assert abs(ratios.mean() - 1.0) < 0.05, ratios


def test_ambient_rna_is_part_of_the_library():
    cd, side = _disk()
    flat = dict(field_sigma=0.0, sigma_counts=1e-6, edge_sigma=0.0, mu_counts=8000.0)
    without, _ = _tissue_depth(Visium(seed=2, ambient_frac=0.0, **flat).run(cd, grid_side=side))
    with_soup, _ = _tissue_depth(Visium(seed=2, ambient_frac=0.2, **flat).run(cd, grid_side=side))
    assert without == pytest.approx(8000.0, rel=0.01)
    assert with_soup == pytest.approx(without, rel=0.01)


# ----------------------------------------------------------------- spot swapping --------------
SWAP = dict(bleed_rate=0.3, bleed_distal=0.3, bleed_sigma=PITCH, edge_sigma=0.25,
            edge_reference="tissue", ambient_frac=0.0, field_sigma=0.0, sigma_counts=0.2,
            mu_counts=8000.0)


def test_swapping_moves_depth_off_the_tissue_and_conserves_it():
    """Swapping draws no random numbers, so the same seed releases the same library: the slide counts
    what the tissue released, and the tissue keeps less of it."""
    cd, side = _disk()
    vz = Visium(seed=4, **SWAP).run(cd, grid_side=side)
    ref = Visium(seed=4, **dict(SWAP, bleed_rate=0.0)).run(cd, grid_side=side)
    depth, tis = _tissue_depth(vz)
    released, _ = _tissue_depth(ref)
    assert vz.spot_counts.to_numpy().sum() == pytest.approx(ref.spot_counts.to_numpy().sum(), rel=0.01)
    assert released == pytest.approx(SWAP["mu_counts"], rel=0.06)
    assert depth < 0.95 * released


def test_estimate_visium_returns_the_depth_before_swapping():
    """On a raw slide the tissue spots were counted after swapping; the fit must return the depth the
    assay draws before it, or a fitted run on the same outline comes out shallower than the data."""
    cd, side = _disk(radius=20.0, side=80, density=5.0)
    vz = Visium(seed=7, **SWAP).run(cd, grid_side=side)
    observed, tis = _tissue_depth(vz)
    est = estimate_visium(vz.to_anndata(), coords=vz.spot_coords, bleed_min_dist=0.0)
    bl = est.diagnostics["bleed"]
    assert bl["mu_counts_observed"] == pytest.approx(observed)
    assert 0.5 < bl["tissue_retained"] < 1.0
    assert est.hypers.mu_counts == pytest.approx(observed / bl["tissue_retained"])
    assert est.hypers.mu_counts == pytest.approx(SWAP["mu_counts"], rel=0.05)
    # a fitted run on the same outline reproduces the observed tissue depth
    refit = Visium(seed=8, **est.visium_kwargs()).run(cd, grid_side=side)
    assert _tissue_depth(refit)[0] == pytest.approx(observed, rel=0.05)


# ----------------------------------------------------------------- count noise ----------------
def _technical_dispersion(vz):
    """Index of dispersion of the counts around the true per-spot composition (ambient off)."""
    tis = vz.obs["n_cells"].to_numpy() > 0
    Y = vz.spot_counts.to_numpy(float)[tis]
    P = vz.raw_expr.to_numpy()[tis] * vz.batch.beta[None, :]
    mu = Y.sum(axis=1, keepdims=True) * P / P.sum(axis=1, keepdims=True)
    return np.mean([np.mean((y - m) ** 2 / m) for y, m in zip(Y, mu)]) / (1.0 - 1.0 / Y.shape[1])


@pytest.mark.parametrize("kappa", [50.0, 5_000.0])
def test_dm_dispersion_does_not_depend_on_the_gene_set(kappa):
    """(N + kappa) / (1 + kappa) for every gene set: a kappa fitted on a real transcriptome applies to
    the growth genes unchanged."""
    out = []
    for n_genes in (20, 300):
        cd, side = _disk(n_genes=n_genes)
        vz = Visium(seed=3, kappa=kappa, ambient_frac=0.0, mu_counts=8000.0).run(cd, grid_side=side)
        n = vz.spot_counts.to_numpy().sum(axis=1)[vz.obs["n_cells"].to_numpy() > 0].mean()
        out.append(_technical_dispersion(vz))
        assert out[-1] == pytest.approx((n + kappa) / (1.0 + kappa), rel=0.15)
    assert out[0] == pytest.approx(out[1], rel=0.15)


def test_default_count_noise_is_near_multinomial():
    """The old default of 50 inflated each gene's variance ~390-fold at 20,000 UMIs per spot; the real
    section's whole dispersion, tissue heterogeneity included, is 1.2-fold."""
    cd, side = _disk(n_genes=300)
    vz = Visium(seed=3, ambient_frac=0.0).run(cd, grid_side=side)
    assert vz.hypers.kappa >= 1e5
    assert _technical_dispersion(vz) < 1.2


def test_default_kappa_keeps_the_genes_of_a_6000_gene_spot():
    """At the default depth over the 6,000 growth genes, kappa 50 detected ~280 genes per spot and put
    77% of its counts in 50 of them; the default must stay close to the composition itself."""
    from types import SimpleNamespace
    from iscc.data.batch import COUNT_MODELS
    h = VisiumBatchHyperParams()
    rng = np.random.default_rng(0)
    comp = rng.lognormal(0.0, 1.0, (40, 6000))
    comp /= comp.sum(axis=1, keepdims=True)
    lib = np.full(40, h.mu_counts)
    Y = COUNT_MODELS["dm"](np.random.default_rng(1), comp, lib, SimpleNamespace(h=h))
    expected = (1.0 - (1.0 - comp) ** h.mu_counts).sum(axis=1).mean()
    assert (Y > 0).sum(axis=1).mean() > 0.95 * expected
    top = np.sort(Y, axis=1)[:, -50:].sum(axis=1) / Y.sum(axis=1)
    assert top.mean() < 1.2 * np.sort(comp, axis=1)[:, -50:].sum(axis=1).mean()


def test_fitted_kappa_is_passed_through():
    cd, side = _disk()
    vz = Visium(seed=3).run(cd, grid_side=side)
    tis = vz.obs["n_cells"].to_numpy() > 0
    est = estimate_visium(vz.spot_counts.to_numpy()[tis], coords=vz.spot_coords[tis])
    assert "kappa" in est.fitted
    assert est.visium_kwargs()["kappa"] == est.hypers.kappa


# ----------------------------------------------------------------- spot geometry --------------
def test_default_spot_is_visium_v1():
    """55 um spots at 100 um centre to centre, at ~50 um per deme."""
    h = VisiumBatchHyperParams()
    assert h.spot_pitch == 2.0
    assert 2.0 * h.spot_radius / h.spot_pitch == pytest.approx(55.0 / 100.0)
    cd, side = _disk()
    ad = Visium(seed=1).run(cd, grid_side=side).to_anndata()
    sf = next(iter(ad.uns["spatial"].values()))["scalefactors"]
    assert sf["spot_diameter_fullres"] == pytest.approx(1.1)


def test_default_spot_holds_about_one_deme_of_cells():
    """A thin section keeps ~12 cells per deme (a fifth of a 60-cell column). A v1 spot then pools
    about ten, as real Visium spots over dense tumour do; a spot as wide as the pitch pooled ~40."""
    cd, side = _disk(radius=16.0, density=12.0)
    vz = Visium(seed=1).place_grid(cd, grid_side=side)
    n = vz._grid["n_cells"]
    xy = vz._grid["spot_coords"]
    inner = np.linalg.norm(xy - side / 2.0, axis=1) < 12.0
    assert 7 <= np.median(n[inner]) <= 16
