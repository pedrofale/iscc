"""Tests for the DNA `estimate_dna()` (DESIGN_inference §C.1 / milestone M4 DNA half).

The DNA analogue of `tests/test_estimate_scrna.py`, with the same two pillars:
  * **Estimation recovery** — simulate DNA with known ``DNABatchHyperParams`` via the F4/F5
    generative path -> ``estimate_dna()`` -> recover them within tolerance. Covered for bulk-WGS
    and single-cell-WGS (the required cases) plus bulk-panel (capture) and an NB-depth case.
  * **Modality / breadth awareness + honest flags** — single-cell-only fields (``ado_rate`` /
    ``beta_binom_conc``) are fit only from single-cell data and flagged not-fit on bulk;
    ``doublet_rate`` is prior-only; ``capture_sigma`` is fit only for WES/panel.

Recovery is method-of-moments on marginal summary statistics (no simulation loop). The CN passed to
``estimate_dna`` is the *true* CN — the perfect-caller stand-in for the two-pass conditioning.

Note on ``depth_batch_sigma``: the F4/F5 generator computes a per-batch depth shift but bulk
coverage uses ``N = mu_depth * n_loci`` (the knob is currently dormant), so multi-batch *simulated*
coverage carries no depth spread. We therefore test the estimator directly by injecting explicit
per-batch mean depths and recovering their log-sd.
"""
import numpy as np
import pandas as pd
import pytest

from iscc.data.dna import bulkDNA, scDNA, genome_features
from iscc.data import estimate_dna, estimate_dna_from_assay, DNABatchHyperParams


# --------------------------------------------------------------------------------------
# Controlled ground-truth dataset: a few CN levels + known het loci, cell-constant biology
# --------------------------------------------------------------------------------------
N_SEG, SZ = 20, 50
N_GENES = N_SEG * SZ
GENES = [f"G_{s}_{p}" for s in range(N_SEG) for p in range(SZ)]
HET_SEGS = (0, 3, 7)                       # seg 3 is the amplified segment (mixed-CN hets)


def make_cell_data(n_cells=60, amp_seg=3, amp_cn=6.0, seed=0):
    """cell_data with one amplified segment and several het (VAF=0.5) loci per het segment.

    Cell-constant CN/VAF isolates the *technical* layer (the analogue of M2's cell-constant
    biology), so the residual coverage dispersion measures the amplification regime, not biology.
    """
    cells = [f"C{i}" for i in range(n_cells)]
    cnv = np.full((n_cells, N_GENES), 2.0)
    cnv[:, amp_seg * SZ:(amp_seg + 1) * SZ] = amp_cn
    af = np.zeros((n_cells, N_GENES))
    for s in HET_SEGS:
        for p in range(0, SZ, 5):          # 10 het loci per het segment
            af[:, s * SZ + p] = 0.5
    return {
        "cell_cnv": pd.DataFrame(cnv, index=cells, columns=GENES),
        "cell_snv": pd.DataFrame(af, index=cells, columns=GENES),
        "cell_type": pd.DataFrame(["cloneA"] * n_cells, index=cells, columns=["cell_id"]),
    }


def _realized_gc_sigma(assay):
    """sd of the batch's realized log GC->coverage curve (the honest recovery target)."""
    b = assay.batch
    return float(np.std(-b.gc_strength * (assay.gc - b.gc_opt) ** 2))


# --------------------------------------------------------------------------------------
# Bulk WGS recovery (the headline case): mu_depth, kappa, GC, error
# --------------------------------------------------------------------------------------
class TestBulkWGSRecovery:
    KAPPA, ERR = 2000.0, 0.01

    @pytest.fixture(scope="class")
    def fit(self):
        cd = make_cell_data()
        assay = bulkDNA(breadth="wgs", seed=1, kappa=self.KAPPA, error_rate=self.ERR).run(cd)
        est = estimate_dna_from_assay(assay)
        return est, assay

    def test_mu_depth_recovered(self, fit):
        est, _ = fit
        assert est.hypers.mu_depth == pytest.approx(30.0, rel=0.05)   # wgs preset mean depth

    def test_kappa_recovered(self, fit):
        """CN-conditioned coverage overdispersion -> DM concentration (the §C.1 crux)."""
        est, _ = fit
        assert "kappa" in est.fitted
        assert est.hypers.kappa == pytest.approx(self.KAPPA, rel=0.25)

    def test_gc_curve_tracks_realized_bias(self, fit):
        """gc_curve_sigma recovers the batch's realized GC-curve strength (cf. M2 depth spread)."""
        est, assay = fit
        assert "gc_curve_sigma" in est.fitted
        assert est.hypers.gc_curve_sigma == pytest.approx(_realized_gc_sigma(assay), abs=0.04)

    def test_error_rate_recovered(self, fit):
        est, _ = fit
        assert "error_rate" in est.fitted
        assert est.hypers.error_rate == pytest.approx(self.ERR, rel=0.2)

    def test_single_cell_fields_not_fit_on_bulk(self, fit):
        """ado_rate / beta_binom_conc are the per-cell layer: not fit from bulk, carried from preset."""
        est, _ = fit
        assert "ado_rate" not in est.fitted
        assert "beta_binom_conc" not in est.fitted
        assert est.hypers.ado_rate == 0.0                 # bulk preset
        assert "doublet_rate" not in est.fitted           # prior-only always

    def test_capture_not_fit_for_wgs(self, fit):
        est, _ = fit
        assert "capture_sigma" not in est.fitted
        assert est.hypers.capture_sigma == 0.0


# --------------------------------------------------------------------------------------
# Single-cell WGS recovery: kappa (small/lumpy), ado_rate, beta_binom_conc, error
# --------------------------------------------------------------------------------------
class TestScWGSRecovery:
    KAPPA, ADO, CONC, ERR = 5.0, 0.20, 30.0, 0.005

    @pytest.fixture(scope="class")
    def fit(self):
        cd = make_cell_data(n_cells=200)
        assay = scDNA(n_cells=200, breadth="wgs", seed=2, kappa=self.KAPPA, ado_rate=self.ADO,
                      beta_binom_conc=self.CONC, error_rate=self.ERR).run(cd)
        est = estimate_dna_from_assay(assay)
        return est, assay

    def test_mu_depth_recovered(self, fit):
        est, _ = fit
        assert est.hypers.mu_depth == pytest.approx(9.0, rel=0.05)    # wgs * sc depth_scale 0.3

    def test_kappa_small_regime_recovered(self, fit):
        """Lumpy MDA/MALBAC amplification = small kappa, well identified per-cell."""
        est, _ = fit
        assert est.hypers.kappa == pytest.approx(self.KAPPA, rel=0.35)

    def test_ado_rate_recovered(self, fit):
        est, _ = fit
        assert "ado_rate" in est.fitted
        assert est.hypers.ado_rate == pytest.approx(self.ADO, abs=0.06)

    def test_beta_binom_conc_recovered(self, fit):
        """BAF overdispersion at hets; concentration is hard, so accept within a factor of ~2."""
        est, _ = fit
        assert "beta_binom_conc" in est.fitted
        assert 0.5 * self.CONC < est.hypers.beta_binom_conc < 2.0 * self.CONC

    def test_error_rate_recovered(self, fit):
        est, _ = fit
        assert est.hypers.error_rate == pytest.approx(self.ERR, abs=0.003)


# --------------------------------------------------------------------------------------
# Breadth-aware extras (cheap): bulk panel capture_sigma + FFPE; NB depth model
# --------------------------------------------------------------------------------------
class TestBreadthAndModelVariants:
    def test_bulk_panel_capture_sigma_recovered(self):
        """Deep panel: per-amplicon capture dominates -> recovers the realized LogNormal spread."""
        cd = make_cell_data()
        assay = bulkDNA(breadth="panel", seed=3, capture_sigma=0.45, error_rate=0.005).run(cd)
        est = estimate_dna_from_assay(assay)
        realized = float(np.std(np.log(assay.batch.capture_eff)))
        assert "capture_sigma" in est.fitted
        assert est.hypers.capture_sigma == pytest.approx(realized, abs=0.1)

    def test_ffpe_ct_rate_recovered_above_error_floor(self):
        cd = make_cell_data()
        assay = bulkDNA(breadth="panel", seed=4, mu_depth=2000.0, error_rate=0.005,
                        ffpe_ct_rate=0.03).run(cd)
        est = estimate_dna_from_assay(assay)
        assert "ffpe_ct_rate" in est.fitted
        assert est.hypers.ffpe_ct_rate == pytest.approx(0.03, abs=0.015)
        # error floor measured off non-CT loci, so it is not contaminated by the FFPE excess
        assert est.hypers.error_rate == pytest.approx(0.005, abs=0.003)

    def test_nb_depth_dispersion_recovered(self):
        cd = make_cell_data()
        assay = bulkDNA(breadth="wgs", seed=1, depth_model="nb", nb_dispersion=0.3,
                        error_rate=0.01).run(cd)
        est = estimate_dna_from_assay(assay)
        assert est.depth_model == "nb"
        assert "nb_dispersion" in est.fitted
        assert "kappa" not in est.fitted
        assert est.hypers.nb_dispersion == pytest.approx(0.3, rel=0.25)


# --------------------------------------------------------------------------------------
# depth_batch_sigma (estimator-level; generator knob is dormant — see module docstring)
# --------------------------------------------------------------------------------------
class TestDepthBatchSigma:
    def test_recovers_log_sd_of_injected_batch_depths(self):
        cd = make_cell_data()
        shifts = [0.0, 0.25, -0.2, 0.1, -0.15, 0.3]
        assays = [bulkDNA(breadth="wgs", seed=10 + i, mu_depth=30.0 * np.exp(s),
                          error_rate=0.01).run(cd) for i, s in enumerate(shifts)]
        depths = [a.observed_data.coverage.mean() for a in assays]
        est = estimate_dna_from_assay(assays[0], batch_depths=depths)
        assert "depth_batch_sigma" in est.fitted
        assert est.n_batches == len(depths)
        assert est.hypers.depth_batch_sigma == pytest.approx(float(np.std(np.log(depths))), abs=0.02)

    def test_single_batch_does_not_fit_depth_batch_sigma(self):
        cd = make_cell_data()
        est = estimate_dna_from_assay(bulkDNA(breadth="wgs", seed=1).run(cd))
        assert "depth_batch_sigma" not in est.fitted
        assert est.n_batches == 1


# --------------------------------------------------------------------------------------
# API: drives the assay, validates inputs, prior-only honesty
# --------------------------------------------------------------------------------------
class TestApi:
    def test_dna_kwargs_drive_the_assay(self):
        cd = make_cell_data()
        est = estimate_dna_from_assay(scDNA(n_cells=40, breadth="wgs", seed=2).run(cd))
        kw = est.dna_kwargs()
        assert "breadth" not in kw and "depth_model" not in kw   # explicit constructor args
        a = scDNA(n_cells=40, breadth=est.breadth, depth_model=est.depth_model, **kw).run(cd)
        assert a.coverage.shape == (40, N_GENES)

    def test_explicit_array_api_matches_assay_helper(self):
        cd = make_cell_data()
        assay = bulkDNA(breadth="wgs", seed=1).run(cd)
        df = assay.observed_data
        gc, mapp, ct = genome_features(list(df.index))
        est = estimate_dna(df["coverage"].values, df["alt_counts"].values, df["true_cn"].values,
                           modality="bulk", breadth="wgs", gc=gc, mappability=mapp, ct_sites=ct,
                           variant_mask=df["true_alt_fraction"].values > 0)
        assert est.hypers.mu_depth == pytest.approx(float(df["coverage"].mean()), rel=1e-9)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            estimate_dna(np.ones((3, 10)), np.ones((3, 8)), np.full((3, 10), 2.0))

    def test_unknown_modality_and_breadth_raise(self):
        cov = np.full(20, 30.0)
        with pytest.raises(ValueError):
            estimate_dna(cov, np.zeros(20), np.full(20, 2.0), modality="triple")
        with pytest.raises(ValueError):
            estimate_dna(cov, np.zeros(20), np.full(20, 2.0), breadth="exome")
