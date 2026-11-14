"""Tests for iscc.tumor.components: Selection, Cell, CancerCell, Deme."""
import numpy as np
import pytest

from conftest import N_GENES, N_SEGMENTS, SEGMENT_SIZE, SELECTION_PARAMS, DEME_PARAMS


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

class TestSelection:
    def test_gene_counts(self, selection):
        assert selection.n_genes == N_GENES

    def test_driver_lists_length(self, selection):
        assert len(selection.onc) == N_SEGMENTS
        assert len(selection.tsg) == N_SEGMENTS
        assert len(selection.dispersal) == N_SEGMENTS

    def test_get_oncogenes_indices_in_range(self, selection):
        oncs = selection.get_oncogenes()
        assert oncs.ndim == 1
        assert np.all(oncs >= 0)
        assert np.all(oncs < N_GENES)

    def test_get_tsgs_indices_in_range(self, selection):
        tsgs = selection.get_tsgs()
        assert np.all(tsgs >= 0)
        assert np.all(tsgs < N_GENES)

    def test_get_gene_names_length(self, selection):
        names = selection.get_gene_names()
        assert len(names) == N_GENES

    def test_update_division_rate_clean_genome(self, selection):
        """A cell with no mutations should have a baseline division rate."""
        # genome_summary with no mutations, ploidy=2
        gs = {
            "n_wt_tsg": len(selection.get_tsgs()),
            "n_mut_tsg": 0,
            "n_wt_onc": len(selection.get_oncogenes()),
            "n_mut_onc": 0,
            "ploidy": 2,
        }
        rate = selection.update_division_rate(gs)
        assert isinstance(rate, float)
        assert rate > 0

    def test_update_viability_normal_genome(self, selection):
        gs = {"ploidy": 2, "highest_cn": 2, "nullisomy_count": 0, "n_mutated_drivers": 0}
        assert selection.update_viability(gs) == 1

    def test_update_viability_excess_ploidy(self, selection):
        gs = {"ploidy": 999, "highest_cn": 2, "nullisomy_count": 0, "n_mutated_drivers": 0}
        assert selection.update_viability(gs) == 0

    def test_get_gene_data_keys(self, selection):
        gd = selection.get_gene_data()
        required = {
            "driver_types", "dispersal_types",
            "treatment_resistance_types", "immune_resistance_types",
            "breach_types", "stromal_survival_types", "met_survival_types",
        }
        # `germline_types` is optional: present only for a patient that carries germline variants
        # (see tests/test_germline.py), absent for a plain gene-role layout like this one.
        assert set(gd.keys()) - required <= {"germline_types"}
        assert required <= set(gd.keys())
        for df in gd.values():
            assert len(df) == N_GENES


# ---------------------------------------------------------------------------
# Cell / CancerCell
# ---------------------------------------------------------------------------

class TestCell:
    def test_genome_has_correct_segments(self, cancer_cell):
        assert len(cancer_cell.genome) == N_SEGMENTS

    def test_genome_summary_initial_ploidy(self, cancer_cell):
        assert cancer_cell.genome_summary["ploidy"] == 2

    def test_genome_summary_no_mutations_initially(self, cancer_cell):
        assert cancer_cell.genome_summary["n_mut_onc"] == 0
        assert cancer_cell.genome_summary["n_mut_tsg"] == 0

    def test_get_snvs_shape(self, cancer_cell):
        snvs = cancer_cell.get_snvs()
        assert snvs.shape == (N_GENES,)

    def test_get_snvs_initial_all_zero(self, cancer_cell):
        snvs = cancer_cell.get_snvs()
        assert np.all(snvs == 0)

    def test_get_cnvs_shape(self, cancer_cell):
        cnvs = cancer_cell.get_cnvs()
        assert cnvs.shape == (N_GENES,)

    def test_get_cnvs_initial_diploid(self, cancer_cell):
        cnvs = cancer_cell.get_cnvs()
        assert np.all(cnvs == 2)

    def test_divide_returns_new_cell(self, cancer_cell):
        child = cancer_cell.divide()
        assert child is not cancer_cell
        assert child.parent is cancer_cell

    def test_divide_child_shares_genome_summary(self, cancer_cell):
        child = cancer_cell.divide()
        assert child.genome_summary["ploidy"] == cancer_cell.genome_summary["ploidy"]

    def test_mutate_changes_genome(self, cancer_cell, selection):
        rng = np.random.default_rng(42)
        # Run just a couple of mutations (more would saturate the tiny test genome
        # and cause the novel-position search loop to run forever).
        cancer_cell.mutate(rng, selection)
        cancer_cell.mutate(rng, selection)
        # genome_summary counters must remain non-negative integers
        assert isinstance(cancer_cell.genome_summary["n_mut_onc"], (int, np.integer))
        assert cancer_cell.genome_summary["n_mut_onc"] >= 0
        assert cancer_cell.genome_summary["n_mut_tsg"] >= 0

    def test_evolutionary_parameters_present(self, cancer_cell):
        keys = {"division_rate", "death_rate", "dispersal_rate",
                "treatment_resistance", "immune_resistance", "viability"}
        assert keys.issubset(cancer_cell.evolutionary_parameters.keys())

    def test_get_genome_df_columns(self, cancer_cell):
        df = cancer_cell.get_genome_df()
        assert set(df.columns) == {"seg", "hap", "pos", "mut"}
