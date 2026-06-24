"""Tests for iscc.data: bulkDNA, scDNA, scRNA assays."""
import numpy as np
import pandas as pd
import pytest

from conftest import N_GENES, N_SEGMENTS, SEGMENT_SIZE


class TestBulkDNA:
    def test_run_produces_observed_data(self, synthetic_cell_data):
        from iscc.data.dna import bulkDNA
        assay = bulkDNA(fpr=0.01, fnr=0.01, n_reads=500)
        assay.run(synthetic_cell_data)
        assert hasattr(assay, "observed_data")
        assert "coverage" in assay.observed_data.columns
        assert "alt_counts" in assay.observed_data.columns

    def test_run_output_indexed_by_genes(self, synthetic_cell_data):
        from iscc.data.dna import bulkDNA
        assay = bulkDNA(fpr=0.01, fnr=0.01, n_reads=500)
        assay.run(synthetic_cell_data)
        assert len(assay.observed_data) == N_GENES

    def test_total_coverage_equals_n_reads(self, synthetic_cell_data):
        from iscc.data.dna import bulkDNA
        n_reads = 400
        assay = bulkDNA(fpr=0.0, fnr=0.0, n_reads=n_reads)
        assay.run(synthetic_cell_data)
        assert assay.observed_data["coverage"].sum() == n_reads

    def test_alt_counts_le_coverage(self, synthetic_cell_data):
        from iscc.data.dna import bulkDNA
        assay = bulkDNA(fpr=0.1, fnr=0.1, n_reads=1000)
        assay.run(synthetic_cell_data)
        assert (assay.observed_data["alt_counts"] <= assay.observed_data["coverage"]).all()

    def test_zero_snv_zero_fpr_gives_zero_alt(self):
        """All-zero SNV matrix with FPR=0 should produce zero alt counts."""
        from iscc.data.dna import bulkDNA
        n_cells, n_genes = 10, N_GENES
        names = [f"C{i}" for i in range(n_cells)]
        cols = [f"G{j}" for j in range(n_genes)]
        cell_data = {
            "cell_snv": pd.DataFrame(np.zeros((n_cells, n_genes)), index=names, columns=cols)
        }
        assay = bulkDNA(fpr=0.0, fnr=0.0, n_reads=1000)
        assay.run(cell_data)
        assert (assay.observed_data["alt_counts"] == 0).all()

    def test_all_snv_zero_fnr_gives_full_coverage_alt(self):
        """All-mutated SNV with FNR=0 should give alt_counts == coverage."""
        from iscc.data.dna import bulkDNA
        n_cells, n_genes = 10, N_GENES
        names = [f"C{i}" for i in range(n_cells)]
        cols = [f"G{j}" for j in range(n_genes)]
        cell_data = {
            "cell_snv": pd.DataFrame(np.ones((n_cells, n_genes)), index=names, columns=cols)
        }
        assay = bulkDNA(fpr=0.0, fnr=0.0, n_reads=1000)
        assay.run(cell_data)
        assert (assay.observed_data["alt_counts"] == assay.observed_data["coverage"]).all()

    def test_target_genes_fraction(self, synthetic_cell_data):
        from iscc.data.dna import bulkDNA
        assay = bulkDNA(fpr=0.01, fnr=0.01, n_reads=500, target_genes=0.5)
        assay.run(synthetic_cell_data)
        assert len(assay.observed_data) == int(0.5 * N_GENES)

    def test_write_creates_file(self, synthetic_cell_data, tmp_path):
        from iscc.data.dna import bulkDNA
        assay = bulkDNA(fpr=0.01, fnr=0.01, n_reads=200)
        assay.run(synthetic_cell_data)
        assay.write(str(tmp_path))
        assert (tmp_path / "counts.csv").exists()


class TestScDNA:
    def test_run_binary_mode_shape(self, synthetic_cell_data):
        from iscc.data.dna import scDNA
        assay = scDNA(n_cells=10, data_mode="binary", fpr=0.01, fnr=0.01, n_reads=200)
        assay.run(synthetic_cell_data)
        assert assay.observed_snvs.shape == (10, N_GENES)

    def test_run_binary_values_are_zero_or_one(self, synthetic_cell_data):
        from iscc.data.dna import scDNA
        assay = scDNA(n_cells=10, data_mode="binary", fpr=0.0, fnr=0.0, n_reads=200)
        assay.run(synthetic_cell_data)
        assert set(assay.observed_snvs.values.flatten()).issubset({0, 1})

    def test_run_counts_mode_produces_coverage_and_alt(self, synthetic_cell_data):
        from iscc.data.dna import scDNA
        assay = scDNA(n_cells=10, data_mode="counts", fpr=0.01, fnr=0.01, n_reads=200)
        assay.run(synthetic_cell_data)
        assert hasattr(assay, "coverage")
        assert hasattr(assay, "alt_counts")
        assert assay.coverage.shape == (10, N_GENES)
        assert assay.alt_counts.shape == (10, N_GENES)

    def test_n_cells_capped_at_available(self, synthetic_cell_data):
        """Requesting more cells than available should use all available cells."""
        from iscc.data.dna import scDNA
        n_available = len(synthetic_cell_data["cell_snv"])
        assay = scDNA(n_cells=n_available + 100, data_mode="binary", fpr=0.0, fnr=0.0)
        assay.run(synthetic_cell_data)
        assert assay.observed_snvs.shape[0] == n_available

    def test_alt_counts_le_coverage_counts_mode(self, synthetic_cell_data):
        from iscc.data.dna import scDNA
        assay = scDNA(n_cells=10, data_mode="counts", fpr=0.1, fnr=0.1, n_reads=300)
        assay.run(synthetic_cell_data)
        assert (assay.alt_counts.values <= assay.coverage.values).all()


class TestScRNA:
    def test_run_produces_umi_counts(self, synthetic_cell_data):
        from iscc.data.rna import scRNA
        assay = scRNA(n_reads=100, n_cells=10)
        assay.run(synthetic_cell_data)
        assert hasattr(assay, "observed_counts")
        assert assay.observed_counts.shape == (10, N_GENES)

    def test_umi_counts_are_non_negative_integers(self, synthetic_cell_data):
        from iscc.data.rna import scRNA
        assay = scRNA(n_reads=100, n_cells=10)
        assay.run(synthetic_cell_data)
        vals = assay.observed_counts.values
        assert (vals >= 0).all()
        assert np.issubdtype(vals.dtype, np.integer)

    def test_library_size_varies_and_centered_on_n_reads(self, synthetic_cell_data):
        """Realistic scRNA: per-cell library size varies (not fixed) and averages ~ n_reads."""
        from iscc.data.rna import scRNA
        n_reads = 2000
        assay = scRNA(n_reads=n_reads, n_cells=20, lib_size_sigma=0.35)
        assay.run(synthetic_cell_data)
        row_sums = assay.observed_counts.sum(axis=1).values
        assert row_sums.std() > 0                      # not a fixed depth
        assert 0.5 * n_reads < row_sums.mean() < 2.0 * n_reads

    def test_n_cells_capped_at_available(self, synthetic_cell_data):
        from iscc.data.rna import scRNA
        n_available = len(synthetic_cell_data["cell_exp"])
        assay = scRNA(n_reads=100, n_cells=n_available + 50)
        assay.run(synthetic_cell_data)
        assert assay.observed_counts.shape[0] == n_available

    def test_write_creates_file(self, synthetic_cell_data, tmp_path):
        from iscc.data.rna import scRNA
        assay = scRNA(n_reads=100, n_cells=5)
        assay.run(synthetic_cell_data)
        assay.write(str(tmp_path))
        assert (tmp_path / "umis.csv").exists()
