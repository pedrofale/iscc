"""Validation tests for the F4 (bulk) + F5 (single-cell) DNA coverage model.

Internal-validation targets from DESIGN_features §C C1 / §G:
  * coverage tracks copy number (bulk + single-cell);
  * a large-CN amplicon compositionally reduces other loci's coverage (bulk DM);
  * VAF recovers the true alt fraction at high depth (bulk);
  * ADO shows as allele loss at heterozygous loci (single-cell);
  * breadth behaves (panel = deep / few loci, WGS = shallow / many);
  * the NB depth alternative and multi-batch (same biology, different technical) work.
"""
import numpy as np
import pandas as pd
import pytest

from iscc.data.dna import bulkDNA, scDNA, run_dna_batches


N_SEG, SZ = 4, 25
N_GENES = N_SEG * SZ
GENES = [f"G_{s}_{p}" for s in range(N_SEG) for p in range(SZ)]
SEG_OF = np.array([s for s in range(N_SEG) for _ in range(SZ)])


def make_cnv_cell_data(n_cells=40, amp_seg=1, amp_cn=8.0, het_loci=(5, 30), seed=0):
    """cell_data with one amplified segment and a few heterozygous (VAF=0.5) loci.

    `het_loci` are absolute locus indices; index 5 is in segment 0 (diploid), 30 in the
    amplified segment 1 — so we can check VAF recovery at both CN levels.
    """
    cells = [f"C{i}" for i in range(n_cells)]
    cnv = np.full((n_cells, N_GENES), 2.0)
    cnv[:, amp_seg * SZ:(amp_seg + 1) * SZ] = amp_cn
    af = np.zeros((n_cells, N_GENES))
    for l in het_loci:
        af[:, l] = 0.5
    return {
        "cell_cnv": pd.DataFrame(cnv, index=cells, columns=GENES),
        "cell_snv": pd.DataFrame(af, index=cells, columns=GENES),
        "cell_type": pd.DataFrame(["cloneA"] * n_cells, index=cells, columns=["cell_id"]),
    }


# ----------------------------------------------------------------------- BULK (F4) ----
class TestBulkCoverage:
    def test_coverage_tracks_copy_number(self):
        cd = make_cnv_cell_data(amp_cn=8.0)
        od = bulkDNA(breadth="wgs", seed=1).run(cd).observed_data
        cov_amp = od[od.segment == 1].coverage.mean()
        cov_dip = od[od.segment == 0].coverage.mean()
        # CN8 vs CN2 -> ~4x coverage (DM is compositional so allow a generous band).
        assert 3.0 < cov_amp / cov_dip < 5.5

    def test_log2_ratio_recovers_cn(self):
        cd = make_cnv_cell_data(amp_cn=8.0)
        od = bulkDNA(breadth="wgs", seed=3).run(cd).observed_data
        # log2(CN8/CN2) = 2; amplicon log2-ratio clearly positive, diploid near 0/negative.
        assert od[od.segment == 1].log2_ratio.mean() > 1.3
        assert od[od.segment == 0].log2_ratio.mean() < 0.5

    def test_compositional_read_stealing(self):
        """At a fixed read budget, adding a large amplicon must REDUCE other loci's coverage
        (the Dirichlet-Multinomial coupling the per-bin NB model misses)."""
        diploid = make_cnv_cell_data(amp_cn=2.0)   # no amplification
        amplified = make_cnv_cell_data(amp_cn=20.0)  # strong amplification on seg 1
        N = 200000
        od0 = bulkDNA(breadth="wgs", seed=5, n_reads=N).run(diploid).observed_data
        od1 = bulkDNA(breadth="wgs", seed=5, n_reads=N, depth_model="dm").run(amplified).observed_data
        # both spend the same total budget ...
        assert od0.coverage.sum() == N and od1.coverage.sum() == N
        # ... so the diploid (non-amplified) segments lose coverage when seg1 is amplified.
        other = SEG_OF[od0.index.map(lambda g: GENES.index(g)).values] != 1
        assert od1.coverage.values[other].mean() < od0.coverage.values[other].mean()

    def test_vaf_recovers_true_alt_fraction_at_high_depth(self):
        cd = make_cnv_cell_data(amp_cn=2.0, het_loci=(5,))
        od = bulkDNA(breadth="panel", seed=7, target_genes=[GENES[5]],
                     mu_depth=5000.0, error_rate=0.0).run(cd).observed_data
        row = od.loc[GENES[5]]
        assert row.true_alt_fraction == pytest.approx(0.5)
        assert row.vaf == pytest.approx(0.5, abs=0.05)  # deep -> tight around truth

    def test_error_rate_creates_false_alt(self):
        cd = make_cnv_cell_data(amp_cn=2.0, het_loci=())  # no real variants
        od = bulkDNA(breadth="panel", seed=9, mu_depth=2000.0, error_rate=0.02).run(cd).observed_data
        # all loci are reference; error rate seeds a small but non-zero alt signal.
        assert (od.alt_counts > 0).mean() > 0.5
        assert od.vaf.mean() < 0.1

    def test_nb_depth_model_runs_and_tracks_cn(self):
        cd = make_cnv_cell_data(amp_cn=8.0)
        od = bulkDNA(breadth="wgs", seed=11, depth_model="nb").run(cd).observed_data
        assert od[od.segment == 1].coverage.mean() > od[od.segment == 0].coverage.mean()
        assert (od.alt_counts <= od.coverage).all()


# ------------------------------------------------------------------ SINGLE-CELL (F5) --
class TestScDNA:
    def test_coverage_tracks_cn_per_cell(self):
        cd = make_cnv_cell_data(amp_cn=8.0)
        sc = scDNA(n_cells=30, breadth="wgs", seed=2).run(cd)
        cov = sc.coverage.values
        amp = cov[:, SEG_OF == 1].mean()
        dip = cov[:, SEG_OF == 0].mean()
        assert amp / dip > 2.0  # lumpy (low-kappa) but still tracks CN

    def test_ado_causes_allele_loss(self):
        """ADO flips heterozygous loci to homozygous (VAF -> 0 or 1) — absent without ADO."""
        cd = make_cnv_cell_data(amp_cn=2.0, het_loci=(5,))
        with_ado = scDNA(n_cells=80, breadth="panel", seed=4, ado_rate=0.5,
                         target_genes=[GENES[5]], mu_depth=300.0).run(cd)
        no_ado = scDNA(n_cells=80, breadth="panel", seed=4, ado_rate=0.0,
                       target_genes=[GENES[5]], mu_depth=300.0).run(cd)
        # at the het locus, ADO produces extreme observed VAFs (allele loss).
        vaf_ado = with_ado.vaf[GENES[5]].values
        extreme_ado = ((vaf_ado < 0.1) | (vaf_ado > 0.9)).mean()
        vaf_no = no_ado.vaf[GENES[5]].values
        extreme_no = ((vaf_no < 0.1) | (vaf_no > 0.9)).mean()
        assert with_ado.ado_mask[GENES[5]].mean() > 0.2
        assert extreme_ado > extreme_no

    def test_binary_mode_values_zero_or_one(self):
        cd = make_cnv_cell_data(amp_cn=2.0)
        sc = scDNA(n_cells=20, breadth="wgs", data_mode="binary", seed=6,
                   error_rate=0.0).run(cd)
        assert set(np.unique(sc.observed_snvs.values)) <= {0, 1}

    def test_alt_le_coverage_and_ground_truth_surfaced(self):
        cd = make_cnv_cell_data(amp_cn=8.0)
        sc = scDNA(n_cells=25, breadth="wgs", seed=8).run(cd)
        assert (sc.alt_counts.values <= sc.coverage.values).all()
        # ground truth preserved alongside observed data
        assert sc.true_cn.loc[:, GENES[SZ]].mean() == 8.0
        assert "clone" in sc.obs.columns and "ado_frac" in sc.obs.columns

    def test_doublets_flagged(self):
        cd = make_cnv_cell_data(n_cells=100, amp_cn=2.0)
        sc = scDNA(n_cells=100, breadth="wgs", seed=10, doublet_rate=0.1).run(cd)
        assert sc.obs["is_doublet"].sum() > 0


# ---------------------------------------------------------------------- BREADTH -------
class TestBreadth:
    def test_panel_deep_few_loci_wgs_shallow_many(self):
        cd = make_cnv_cell_data(amp_cn=2.0)
        wgs = bulkDNA(breadth="wgs", seed=1).run(cd).observed_data
        panel = bulkDNA(breadth="panel", seed=1).run(cd).observed_data
        wes = bulkDNA(breadth="wes", seed=1).run(cd).observed_data
        # panel observes far fewer loci than WGS, at far greater depth.
        assert len(panel) < len(wgs)
        assert panel.coverage.mean() > 10 * wgs.coverage.mean()
        # WES is intermediate: a subset of the genome, deeper than WGS, shallower than panel.
        assert len(panel) <= len(wes) < len(wgs)
        assert wgs.coverage.mean() < wes.coverage.mean() < panel.coverage.mean()

    def test_wes_capture_creates_systematic_per_target_variation(self):
        cd = make_cnv_cell_data(amp_cn=2.0)
        # at fixed CN, WES per-target capture bias spreads coverage more than plain WGS GC bias.
        wgs = bulkDNA(breadth="wgs", seed=1, mu_depth=200.0).run(cd).observed_data
        wes = bulkDNA(breadth="wes", seed=1, mu_depth=200.0).run(cd).observed_data
        cv = lambda x: x.coverage.std() / x.coverage.mean()
        assert cv(wes) > cv(wgs)


# ---------------------------------------------------------------------- BATCHES -------
class TestMultiBatch:
    def test_bulk_batches_same_biology_different_technical(self):
        cd = make_cnv_cell_data(amp_cn=8.0)
        assays = run_dna_batches(cd, mode="bulk", n_batches=2, base_seed=42, breadth="wgs")
        a, b = assays
        # same biological ground truth ...
        assert np.allclose(a.observed_data.true_alt_fraction, b.observed_data.true_alt_fraction)
        assert np.allclose(a.observed_data.true_cn, b.observed_data.true_cn)
        # ... different technical realization (coverage differs).
        assert not np.array_equal(a.observed_data.coverage, b.observed_data.coverage)
        assert a.batch.label != b.batch.label

    def test_sc_shared_design_same_cells(self):
        cd = make_cnv_cell_data(n_cells=60, amp_cn=4.0)
        assays = run_dna_batches(cd, mode="sc", n_batches=2, base_seed=42,
                                 design="shared", breadth="wgs", n_cells=20)
        a, b = assays
        assert list(a.cells) == list(b.cells)               # shared biology
        assert not np.array_equal(a.coverage.values, b.coverage.values)  # different run
