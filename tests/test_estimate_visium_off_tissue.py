"""``estimate_visium`` with an ``in_tissue`` flag: bleed and ambient RNA from the spots outside the
tissue, recovered from a section simulated with known values."""
import numpy as np
import pytest

from iscc.data import estimate_visium
from iscc.data.estimate_visium import _bleed_kernel, _fit_off_tissue

PITCH = 2.0


def _hex(n_rows=44, n_cols=40):
    rows, cols = np.meshgrid(np.arange(n_rows), np.arange(n_cols), indexing="ij")
    r = rows.ravel() * PITCH * np.sqrt(3.0) / 2.0
    c = cols.ravel() * PITCH + (rows.ravel() % 2) * PITCH / 2.0
    return np.column_stack([r, c])


def _section(a, beta, sigma_pitch, mu=5000.0, n_genes=30, seed=0):
    """Tissue spots draw their totals; a spot outside the tissue receives ``a`` mean tissue totals of
    ambient RNA plus ``beta`` of every tissue spot's total through the bleed kernel."""
    rng = np.random.default_rng(seed)
    coords = _hex()
    in_t = np.linalg.norm(coords - coords.mean(axis=0), axis=1) < 24.0
    T = np.where(in_t, mu * rng.lognormal(0.0, 0.3, len(coords)), 0.0)
    expected = np.where(in_t, T, a * T[in_t].mean() + beta * (_bleed_kernel(coords / PITCH, sigma_pitch) @ T))
    totals = rng.poisson(expected)
    profile = rng.dirichlet(np.ones(n_genes))
    counts = np.vstack([rng.multinomial(n, profile) for n in totals]).astype(float)
    return counts, coords, in_t


@pytest.mark.parametrize("a,beta,sigma", [(0.01, 0.2, 1.0), (0.03, 0.1, 0.6)])
def test_off_tissue_fit_recovers_known_values(a, beta, sigma):
    counts, coords, in_t = _section(a, beta, sigma)
    off = _fit_off_tissue(counts, coords, in_t)
    assert off["pitch"] == pytest.approx(PITCH, rel=1e-6)
    assert off["ambient_frac"] == pytest.approx(a, rel=0.2)
    assert off["bleed_frac"] == pytest.approx(beta, abs=0.03)
    assert off["sigma_pitch"] == pytest.approx(sigma, abs=0.15)
    # the soup and the tissue share one gene profile here
    assert off["ambient_profile_r"] > 0.9


def test_estimate_visium_reads_the_tissue_flag():
    counts, coords, in_t = _section(0.01, 0.2, 1.0)
    est = estimate_visium(counts, coords=coords, in_tissue=in_t)
    # the off-tissue spots fit SpotClean's bleed, which carries the floor and moves counts, so the
    # in-spot soup and the composition smoothing are switched off
    assert {"bleed_rate", "bleed_distal", "bleed_sigma"} <= set(est.fitted)
    bl = est.diagnostics["bleed"]
    assert "off_tissue" in est.diagnostics
    assert est.hypers.bleed_rate == pytest.approx(bl["bleed_rate"])
    assert est.hypers.bleed_sigma == pytest.approx(bl["sigma_pitch"] * est.hypers.spot_pitch)
    assert 0.0 < est.hypers.bleed_rate <= 1.0 and 0.0 <= est.hypers.bleed_distal <= 1.0
    assert est.hypers.ambient_frac == 0.0 and est.hypers.diffusion_sigma == 0.0
    assert est.n_spots == len(in_t)
    assert est.spots_per_tissue == int(in_t.sum())
    # the library fit sees the tissue spots only
    assert est.hypers.mu_counts == pytest.approx(counts[in_t].sum(axis=1).mean(), rel=0.1)


def test_without_spots_outside_the_tissue_nothing_changes():
    counts, coords, in_t = _section(0.01, 0.2, 1.0)
    tissue = counts[in_t]
    old = estimate_visium(tissue, coords=coords[in_t])
    new = estimate_visium(tissue, coords=coords[in_t], in_tissue=np.ones(in_t.sum(), dtype=bool))
    assert old.hypers.to_dict() == new.hypers.to_dict()
    assert old.fitted == new.fitted
    assert "ambient_frac" not in old.fitted


def test_empty_spots_outside_the_tissue_take_the_original_path():
    """iscc's own AnnData carries an in_tissue flag, but its empty spots count nothing: the fit must
    not read them as zero bleed and zero ambient RNA."""
    counts, coords, in_t = _section(0.01, 0.2, 1.0)
    counts[~in_t] = 0.0
    old = estimate_visium(counts, coords=coords)
    new = estimate_visium(counts, coords=coords, in_tissue=in_t)
    assert old.hypers.to_dict() == new.hypers.to_dict()
    assert old.fitted == new.fitted
    assert "off_tissue" not in new.diagnostics
