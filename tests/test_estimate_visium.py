"""F6 / M4 Visium-half estimate tests (DESIGN_inference §C.2).

Recovery + honesty, mirroring test_estimate_dna:
  * simulate a Visium AnnData with KNOWN hypers -> estimate_visium -> recover mu_counts /
    sigma_counts and the capture-field autocorrelation length (field_lengthscale) within tolerance;
  * the `.fitted` map is honest: identifiable fields are listed; the prior-only fields
    (ambient_frac / edge_sigma / diffusion_sigma) are NOT claimed as fit;
  * both count models route to the right overdispersion target (kappa vs nb_dispersion).
"""
import numpy as np
import pandas as pd
import pytest

from iscc.data import Visium, estimate_visium, estimate_visium_from_assay
from iscc.data.estimate_visium import _PRIOR_ONLY

GENES = [f"G_{g}" for g in range(40)]


def make_cell_data(grid=30, seed=0):
    """Dense single-section tissue: 1-3 cells per integer coordinate over the whole grid, so the
    estimate sees a few hundred occupied spots (enough pairs for the autocorrelation fit)."""
    rng = np.random.default_rng(seed)
    base = rng.gamma(2.0, 1.0, len(GENES))
    ids, coords, exp, ctype = [], [], [], []
    i = 0
    for r in range(grid):
        for c in range(grid):
            for _ in range(int(rng.integers(1, 4))):
                exp.append(base * rng.lognormal(0.0, 0.2, len(GENES)))
                coords.append((r, c)); ctype.append("A" if r < grid / 2 else "B")
                ids.append(f"C{i}"); i += 1
    exp = pd.DataFrame(exp, index=ids, columns=GENES)
    return {
        "cell_exp": exp,
        "cell_crd": pd.DataFrame(coords, index=ids, columns=["row", "col"]),
        "cell_type": pd.DataFrame(ctype, index=ids, columns=["cell_id"]),
    }


TRUE = dict(mu_counts=6000.0, sigma_counts=0.30, field_sigma=0.40,
            field_lengthscale=6.0, edge_sigma=0.0, kappa=80.0)


def _simulate(seed=7, **over):
    cd = make_cell_data()
    h = dict(TRUE, **over)
    return Visium(seed=seed, spot_pitch=2.0, spot_radius=1.0, count_model="dm", **h).run(
        cd, grid_side=30)


class TestRecovery:
    def test_recovers_library_and_field_lengthscale(self):
        est = estimate_visium_from_assay(_simulate())
        h = est.hypers
        assert abs(h.mu_counts - TRUE["mu_counts"]) / TRUE["mu_counts"] < 0.15
        assert abs(h.sigma_counts - TRUE["sigma_counts"]) < 0.12
        # autocorrelation length recovered within a factor ~2 (the headline spatial param)
        assert 0.5 * TRUE["field_lengthscale"] < h.field_lengthscale < 2.0 * TRUE["field_lengthscale"]

    def test_field_strength_recovered(self):
        est = estimate_visium_from_assay(_simulate())
        assert abs(est.hypers.field_sigma - TRUE["field_sigma"]) < 0.15

    def test_spots_per_tissue_and_occupancy(self):
        a = _simulate()
        est = estimate_visium_from_assay(a)
        n_occ = int((a.obs.n_cells.values > 0).sum())
        assert est.spots_per_tissue == n_occ
        assert 0.0 < est.occupied_fraction <= 1.0


class TestFittedHonesty:
    def test_fitted_lists_identifiable_only(self):
        est = estimate_visium_from_assay(_simulate())
        for f in ("mu_counts", "sigma_counts", "field_sigma", "field_lengthscale", "kappa"):
            assert f in est.fitted
        # prior-only (unidentifiable from one section) fields are never claimed as fit
        for f in _PRIOR_ONLY:
            assert f not in est.fitted

    def test_single_section_sigma_batch_prior_only(self):
        est = estimate_visium_from_assay(_simulate())          # one section -> no sigma_batch fit
        assert "sigma_batch" not in est.fitted
        assert est.n_sections == 1


class TestCountModelRouting:
    def test_dm_fits_kappa_not_nb(self):
        est = estimate_visium_from_assay(_simulate())
        assert "kappa" in est.fitted and "nb_dispersion" not in est.fitted

    def test_nb_fits_dispersion(self):
        a = _simulate()
        # re-estimate the same counts under the NB model -> nb_dispersion is the overdispersion target
        est = estimate_visium(a.spot_counts.values, count_model="nb", coords=a.spot_coords)
        assert "nb_dispersion" in est.fitted and "kappa" not in est.fitted


def test_requires_coords():
    a = _simulate()
    with pytest.raises(ValueError):
        estimate_visium(a.spot_counts.values, count_model="dm")    # no coords / obsm['spatial']
