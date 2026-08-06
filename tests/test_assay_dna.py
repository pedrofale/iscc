"""Validation tests for the F4 (bulk) + F5 (single-cell) DNA coverage model.

Internal-validation targets from DESIGN_features §C C1 / §G:
  * coverage tracks copy number (bulk + single-cell);
  * a large-CN amplicon compositionally reduces other loci's coverage (bulk DM);
  * VAF recovers the true alt fraction at high depth (bulk);
  * ADO shows as allele loss at heterozygous loci (single-cell);
  * breadth behaves (panel = deep / few loci, WGS = shallow / many);
  * the NB depth alternative and multi-batch (same biology, different technical) work;
  * the bulk pool reports its true tumour purity, and germline sites are flagged.
"""
import numpy as np
import pandas as pd
import pytest

from iscc.data.dna import bulkDNA, scDNA, run_dna_batches
from iscc.sample.dissociation.dissociation import biological_type


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


# Loci used by the purity / germline tests: germline hets sit at VAF 0.5 in EVERY cell
# (tumour and normal alike), somatic clonal hets at VAF 0.5 in the cancer cells only.
GERMLINE_LOCI = (2, 40, 60)
SOMATIC_LOCI = (7, 55)


def make_mixed_pool(n_cancer=400, n_normal=600,
                    normal_types=("immune", "stromal", "epithelial", "host"),
                    germline_loci=GERMLINE_LOCI, somatic_loci=SOMATIC_LOCI):
    """A pool of known composition: `n_cancer` cancer cells + `n_normal` normal cells.

    Everything is diploid, so the arithmetic is exact: purity = n_cancer / (n_cancer +
    n_normal); a germline het reads VAF 0.5 whatever the purity; a clonal somatic het reads
    VAF 0.5 x purity.
    """
    cells = ([f"T{i}" for i in range(n_cancer)] + [f"N{i}" for i in range(n_normal)])
    types = (["1"] * n_cancer
             + [normal_types[i % len(normal_types)] for i in range(n_normal)])
    n = len(cells)
    af = np.zeros((n, N_GENES))
    for l in germline_loci:
        af[:, l] = 0.5
    for l in somatic_loci:
        af[:n_cancer, l] = 0.5
    return {
        "cell_cnv": pd.DataFrame(np.full((n, N_GENES), 2.0), index=cells, columns=GENES),
        "cell_snv": pd.DataFrame(af, index=cells, columns=GENES),
        "cell_type": pd.DataFrame(types, index=cells, columns=["cell_id"]),
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


# -------------------------------------------------------- BULK PURITY + GERMLINE ------
class TestBulkPurity:
    def test_host_is_not_cancer(self):
        """Regression: the resident parenchyma of a metastatic deposit is NORMAL tissue.

        It was previously missing from the non-cancer type list, so every host cell was
        counted as a tumour cell and the purity of any met-bearing sample came out too high.
        """
        assert biological_type("host") == "host"
        for t in ("immune", "stromal", "epithelial"):
            assert biological_type(t) == t
        assert biological_type("7") == "cancer"

    def test_purity_of_known_mixture(self):
        cd = make_mixed_pool(n_cancer=400, n_normal=600)
        assay = bulkDNA(breadth="wgs", seed=1).run(cd)
        assert assay.n_cells_pooled == 1000
        assert assay.purity == pytest.approx(0.4)

    def test_purity_correct_for_a_met_bearing_sample(self):
        """A pool whose whole normal compartment is host parenchyma (a metastatic deposit).

        Before the fix this reported purity 1.0 — the exact silent failure the flag guards.
        """
        cd = make_mixed_pool(n_cancer=400, n_normal=600, normal_types=("host",))
        assay = bulkDNA(breadth="wgs", seed=1).run(cd)
        assert assay.purity == pytest.approx(0.4)

    def test_purity_follows_cell_subset(self):
        cd = make_mixed_pool(n_cancer=400, n_normal=600)
        subset = [f"T{i}" for i in range(100)] + [f"N{i}" for i in range(300)]
        assay = bulkDNA(breadth="wgs", seed=1).run(cd, cell_subset=subset)
        assert assay.n_cells_pooled == 400
        assert assay.purity == pytest.approx(0.25)

    def test_purity_is_none_when_cell_types_absent(self):
        """No cell-type table -> purity is unknowable; report nothing rather than 1.0."""
        cd = make_cnv_cell_data(n_cells=10)
        cd.pop("cell_type")
        assay = bulkDNA(breadth="wgs", seed=1).run(cd)
        assert assay.purity is None
        assert "is_germline" not in assay.observed_data.columns

    def test_purity_surfaced_in_anndata(self):
        cd = make_mixed_pool(n_cancer=400, n_normal=600)
        adata = bulkDNA(breadth="wgs", seed=1).run(cd).to_anndata()
        assert adata.obs["purity"].iloc[0] == pytest.approx(0.4)
        assert adata.uns["purity"] == pytest.approx(0.4)
        assert adata.uns["n_cells_pooled"] == 1000

    def test_anndata_purity_is_nan_when_unknown(self):
        cd = make_cnv_cell_data(n_cells=10)
        cd.pop("cell_type")
        adata = bulkDNA(breadth="wgs", seed=1).run(cd).to_anndata()
        assert np.isnan(adata.obs["purity"].iloc[0])


class TestBulkGermline:
    LOCI = [GENES[l] for l in GERMLINE_LOCI + SOMATIC_LOCI]

    def _deep_assay(self, cd, germline_sites):
        return bulkDNA(breadth="panel", seed=13, target_genes=self.LOCI,
                       mu_depth=4000.0, error_rate=0.0).run(cd, germline_sites=germline_sites)

    def test_germline_het_vaf_anchors_at_half_regardless_of_purity(self):
        """A germline het is in EVERY cell, so its VAF is 0.5 whatever the purity — the
        anchor a caller uses to read purity off the data. A clonal somatic het instead
        reads 0.5 x purity, so it moves with the mixture."""
        cd = make_mixed_pool(n_cancer=400, n_normal=600)   # purity 0.4
        assay = self._deep_assay(cd, germline_sites=[GENES[l] for l in GERMLINE_LOCI])
        od = assay.observed_data
        assert assay.purity == pytest.approx(0.4)

        germ = od[od.is_germline]
        som = od[~od.is_germline]
        assert len(germ) == len(GERMLINE_LOCI) and len(som) == len(SOMATIC_LOCI)
        # germline hets: truth exactly 0.5, observed tight around it at this depth.
        assert germ.true_alt_fraction.values == pytest.approx(0.5)
        assert germ.vaf.mean() == pytest.approx(0.5, abs=0.03)
        # clonal somatic hets: 0.5 x purity = 0.2, clearly separated from the anchor.
        assert som.true_alt_fraction.values == pytest.approx(0.2)
        assert som.vaf.mean() == pytest.approx(0.2, abs=0.03)

    def test_ccf_from_vaf_cn_and_purity_recovers_clonality(self):
        """CCF = VAF x CN / purity must return ~1 for a clonal somatic mutation — and only
        the somatic ones, which is why the germline flag has to be there to drop them."""
        cd = make_mixed_pool(n_cancer=400, n_normal=600)
        assay = self._deep_assay(cd, germline_sites=[GENES[l] for l in GERMLINE_LOCI])
        od = assay.observed_data
        ccf = od.vaf * od.true_cn / assay.purity
        assert ccf[~od.is_germline].mean() == pytest.approx(1.0, abs=0.1)
        # germline sites put through the same formula give an impossible CCF > 1.
        assert (ccf[od.is_germline] > 1.5).all()

    def test_germline_anchor_tracks_a_different_purity(self):
        cd = make_mixed_pool(n_cancer=750, n_normal=250)   # purity 0.75
        assay = self._deep_assay(cd, germline_sites=[GENES[l] for l in GERMLINE_LOCI])
        od = assay.observed_data
        assert assay.purity == pytest.approx(0.75)
        assert od[od.is_germline].vaf.mean() == pytest.approx(0.5, abs=0.03)
        assert od[~od.is_germline].vaf.mean() == pytest.approx(0.375, abs=0.03)

    def test_germline_sites_accepts_every_supported_form(self):
        cd = make_mixed_pool(n_cancer=400, n_normal=600)
        names = [GENES[l] for l in GERMLINE_LOCI]
        by_name = self._deep_assay(cd, names).observed_data.is_germline
        forms = {
            "positions": list(GERMLINE_LOCI),
            "mask_all_loci": np.isin(np.arange(N_GENES), GERMLINE_LOCI),
            "mask_observed": np.array([g in set(names) for g in self.LOCI]),
            "series": pd.Series(True, index=names),
        }
        for label, sites in forms.items():
            got = self._deep_assay(cd, sites).observed_data.is_germline
            assert got.equals(by_name), label

    def test_no_germline_sites_means_no_column(self):
        cd = make_mixed_pool(n_cancer=400, n_normal=600)
        od = self._deep_assay(cd, germline_sites=None).observed_data
        assert "is_germline" not in od.columns

    def test_germline_sites_read_from_cell_data_when_present(self):
        """Falls back to the sample's own germline ground truth when the caller passes none."""
        cd = make_mixed_pool(n_cancer=400, n_normal=600)
        cd["germline_sites"] = list(GERMLINE_LOCI)
        od = self._deep_assay(cd, germline_sites=None).observed_data
        assert od.is_germline.tolist() == [True] * len(GERMLINE_LOCI) + [False] * len(SOMATIC_LOCI)

    def test_germline_flag_reaches_anndata_var(self):
        cd = make_mixed_pool(n_cancer=400, n_normal=600)
        adata = self._deep_assay(cd, [GENES[l] for l in GERMLINE_LOCI]).to_anndata()
        assert "is_germline" in adata.var.columns
        assert adata.var["is_germline"].sum() == len(GERMLINE_LOCI)

    def test_bad_boolean_mask_length_is_rejected(self):
        cd = make_mixed_pool(n_cancer=10, n_normal=10)
        with pytest.raises(ValueError):
            self._deep_assay(cd, np.zeros(3, dtype=bool))


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
