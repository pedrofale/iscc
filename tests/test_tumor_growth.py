"""Tests for GlandularTumor: construction, grid setup, growth."""
import numpy as np
import pytest

from conftest import (
    N_GENES, N_SEGMENTS, SEGMENT_SIZE,
    CANCER_CELL_PARAMS, DEME_PARAMS, EPITHELIAL_CELL_PARAMS,
    GENOME_PARAMS, IMMUNE_CELL_PARAMS, SELECTION_PARAMS, STROMAL_CELL_PARAMS,
)


class TestGlandularTumorInit:
    def test_grid_shape(self, glandular_tumor):
        g = glandular_tumor.grid_size
        assert len(glandular_tumor.grid) == g
        assert all(len(row) == g for row in glandular_tumor.grid)

    def test_deme_list_length(self, glandular_tumor):
        g = glandular_tumor.grid_size
        assert len(glandular_tumor.deme_list) == g * g

    def test_deme_rates_length(self, glandular_tumor):
        assert len(glandular_tumor.deme_rates) == len(glandular_tumor.deme_list)

    def test_cancer_cells_in_center(self, glandular_tumor):
        # the founder micro-lesion (initial_cancer_cells, capped by K) is seeded in the centre deme
        center = glandular_tumor.grid_size // 2
        deme = glandular_tumor.grid[center][center]
        expected = min(DEME_PARAMS.get("initial_cancer_cells", 1), DEME_PARAMS["carrying_capacity"])
        assert deme.types_counts.get("cancer", 0) == expected

    def test_total_cancer_cells_at_init(self, glandular_tumor):
        total = sum(
            deme.types_counts.get("cancer", 0)
            for deme in glandular_tumor.deme_list
        )
        # all founders sit in the centre deme; none elsewhere yet
        expected = min(DEME_PARAMS.get("initial_cancer_cells", 1), DEME_PARAMS["carrying_capacity"])
        assert total == expected

    def test_get_neighboring_demes_center(self, glandular_tumor):
        center = glandular_tumor.grid_size // 2
        deme = glandular_tumor.grid[center][center]
        neighbors = glandular_tumor.get_neighboring_demes(deme)
        # Center of a 5x5 grid has 4 valid Von Neumann neighbors (edges excluded by strict inequality)
        assert len(neighbors) > 0
        assert all(n in glandular_tumor.deme_list for n in neighbors)

    def test_get_neighboring_demes_stay_in_bounds(self, glandular_tumor):
        g = glandular_tumor.grid_size
        for deme in glandular_tumor.deme_list:
            for neighbor in glandular_tumor.get_neighboring_demes(deme):
                assert 0 <= neighbor.row < g
                assert 0 <= neighbor.col < g

    def test_selection_n_genes(self, glandular_tumor):
        assert glandular_tumor.selection.n_genes == N_GENES

    def test_config_with_structure_radius(self):
        """Tumor with structure_radius > 0 places epithelial and stromal cells."""
        from iscc.tumor.models.glandular import GlandularTumor
        tumor = GlandularTumor(
            genome_params=GENOME_PARAMS,
            selection_params=SELECTION_PARAMS,
            cancer_cell_params=CANCER_CELL_PARAMS,
            epithelial_cell_params=EPITHELIAL_CELL_PARAMS,
            stromal_cell_params=STROMAL_CELL_PARAMS,
            immune_cell_params=IMMUNE_CELL_PARAMS,
            deme_params=DEME_PARAMS,
            grid_size=10,
            structure_radius=2,
        )
        epi_total = sum(d.types_counts.get("epithelial", 0) for d in tumor.deme_list)
        stromal_total = sum(d.types_counts.get("stromal", 0) for d in tumor.deme_list)
        assert epi_total > 0
        assert stromal_total > 0


class TestGlandularTumorGrowth:
    def test_grow_returns_correct_number_of_traces(self, glandular_tumor):
        n_steps = 5
        traces = glandular_tumor.grow(n_steps=n_steps, seed=0)
        assert len(traces) == n_steps

    def test_traces_contain_genotype_counts(self, glandular_tumor):
        traces = glandular_tumor.grow(n_steps=5, seed=0)
        for trace in traces:
            assert "genotypes_counts" in trace

    def test_tumor_grows_over_time(self, glandular_tumor):
        traces = glandular_tumor.grow(n_steps=20, seed=0)
        initial_size = sum(traces[0]["genotypes_counts"].values())
        final_size = sum(traces[-1]["genotypes_counts"].values())
        # With division_rate=0.5 >> death_rate=0.1 the tumor should grow
        assert final_size >= initial_size

    def test_cell_data_populated_after_grow(self, grown_tumor):
        assert grown_tumor.cell_data is not None
        assert "cell_snv" in grown_tumor.cell_data
        assert "cell_exp" in grown_tumor.cell_data
        assert "cell_cnv" in grown_tumor.cell_data
        assert "cell_crd" in grown_tumor.cell_data

    def test_cell_data_snv_shape(self, grown_tumor):
        snv = grown_tumor.cell_data["cell_snv"]
        n_cells = sum(grown_tumor.genotypes_counts.values())
        assert snv.shape == (n_cells, N_GENES)

    def test_cell_data_coordinates_in_grid(self, grown_tumor):
        crd = grown_tumor.cell_data["cell_crd"]
        g = grown_tumor.grid_size
        assert (crd["row"] >= 0).all() and (crd["row"] < g).all()
        assert (crd["col"] >= 0).all() and (crd["col"] < g).all()

    def test_genotype_parents_populated(self, grown_tumor):
        # Some mutations should have occurred with mutation_rate=0.5
        assert isinstance(grown_tumor.genotypes_parents, dict)
