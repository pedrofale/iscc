"""The whole simulated tumour as ONE AnnData, and the tissue-coordinate convention.

Two things are checked here.

1. **Nothing is left behind.** ``to_anndata(tumor)`` has to carry every per-cell matrix the run
   produced (as ``X`` + layers), every per-cell annotation (as ``obs``/``obsm``), the per-gene roles
   (``var``/``varm``) and the tumour-level truth (clone tree, programs, parameters) in ``uns`` — and
   all of it has to survive a write/read round-trip through a real ``.h5ad`` file. The mapping is
   driven by SHAPE, so the optional layers (microenvironment, glands, metastasis, programs,
   allele-resolved expression) are picked up in the runs that produce them without anyone listing
   them anywhere.

2. **Coordinates point the same way everywhere.** Every export writes ``obsm["spatial"]`` as
   ``(x, y) = (col, row)`` — the 10x/squidpy order the Visium export already used. Before this, the
   cell-level objects wrote ``(row, col)``, so a squidpy plot of cells came out transposed relative
   to a Visium slide cut from the same tissue.
"""
import os

import numpy as np
import pandas as pd
import pytest

from conftest import GENOME_PARAMS, SELECTION_PARAMS, CANCER_CELL_PARAMS, DEME_PARAMS
from iscc.tumor.models import GenotypeTumor
from iscc.integrations.anndata import (to_anndata, from_anndata, biological_types,
                                       _AUTO_MAX_CLONES, _MIN_CELLS_LADDER)

ad = pytest.importorskip("anndata")


# --------------------------------------------------------------------------- fixtures
SPATIAL = {"grid_size": 14, "structure_radius": 3, "n_glands": 2, "gland_radius": 3,
           "min_gland_sep": 7, "K_duct": 24, "K_stroma": 14, "stroma_fill_frac": 0.3,
           "immune_density": 0.1}
SEL = {**SELECTION_PARAMS, "prop_breach": 0.2, "prop_stromal_survival": 0.2}
DEME = {**DEME_PARAMS, "carrying_capacity": 14, "initial_cancer_cells": 6}
CANCER = {**CANCER_CELL_PARAMS, "wgd_rate": 0.05}

MICROENV = {
    "hypoxia": {"strength": 0.8, "n_genes": 6, "o2_consumption": 1.5, "o2_supply": 0.3},
    "cci": {"strength": 0.6, "n_target_genes": 6, "emitter_type": "cancer", "lengthscale": 2.5},
}
EXPRESSION = {
    "program_params": {"n_programs": 4, "n_genes_per_program": 6, "program_overlap": 0.1,
                       "loading_strength": {"mean": 1.0, "sd": 0.3}, "loading_sparsity": 1.0,
                       "program_genomic_scatter": 1.0},
    "activity_params": {"n_active_programs_per_cell": 2, "activity_dist": "lognormal",
                        "activity_mean": 1.0, "activity_sd": 0.5, "activity_noise": 0.2},
    "coupling_params": {"phenotype_program_strength": 0.5, "prop_program_regulator": 0.05,
                        "program_bias_strength": 0.5},
    "dosage_params": {"dosage_sensitivity_mean": 0.7, "dosage_sensitivity_sd": 0.25,
                      "dosage_saturation": 8, "allele_specific": True},
    "snv_effect_params": {"p_lof": 0.1, "p_missense": 0.3, "p_splice": 0.05, "p_silent": 0.55,
                          "nmd_strength": 0.2, "snv_expression_effect": 0.5},
}


def _grow(steps=60, **kw):
    t = GenotypeTumor(seed=3, genome_params=GENOME_PARAMS, selection_params=SEL,
                      cancer_cell_params=CANCER, deme_params=DEME, spatial_params=SPATIAL, **kw)
    t.grow(n_steps=steps, seed=3)
    return t


@pytest.fixture(scope="module")
def plain():
    """A structured tumour with none of the optional layers on."""
    return _grow()


@pytest.fixture(scope="module")
def loaded():
    """Every optional layer on: microenvironment, gene programs, allele resolution, germline."""
    return _grow(microenv_params=MICROENV, expression_params=EXPRESSION,
                 germline_params={"het_frac": 0.05, "hom_frac": 0.33})


@pytest.fixture(scope="module")
def clonal():
    """A tumour with real clonal STRUCTURE: a few hundred cancer cells over a branching tree.

    The structured fixtures above are microenvironment-heavy (a dozen cancer cells), which cannot
    tell one clone labelling from another. This one can.
    """
    t = GenotypeTumor(seed=3, genome_params=GENOME_PARAMS, selection_params=SEL,
                      cancer_cell_params=CANCER,
                      deme_params={"carrying_capacity": 16, "initial_cancer_cells": 5},
                      spatial_params={"grid_size": 22, "structure_radius": 3,
                                      "immune_density": 0.1},
                      update_mode="tau", tau=1.0)
    t.grow(n_steps=80, seed=3)
    return t


@pytest.fixture(scope="module")
def metastatic():
    """A tumour with a colonised metastatic deposit (adds the compartment + host-cell layers)."""
    met = {**SPATIAL, "met_grid_size": 10, "K_met": 16, "host_fill_frac": 0.4,
           "met_seed_kappa": 0.08, "met_hazard": 0.5, "met_transit_floor": 0.03,
           "epithelial_barrier": 1.2, "stromal_hazard": 0.7}
    sel = {**SEL, "prop_met_survival": 0.2, "breach_effects": 2.0,
           "stromal_survival_effects": 2.0, "met_survival_effects": 2.2}
    t = GenotypeTumor(seed=3, genome_params=GENOME_PARAMS, selection_params=sel,
                      cancer_cell_params=CANCER, deme_params=DEME, spatial_params=met,
                      update_mode="tau", tau=1.0)
    for _ in range(60):
        t.grow(n_steps=2, seed=t.seed)
        comp = (t.cell_data or {}).get("cell_compartment")
        if comp is not None and int((comp["compartment"].values == 1).sum()) >= 20:
            break
    return t


# --------------------------------------------------------------------------- shape-driven mapping
class TestEverythingIsCarried:
    def test_every_gene_wide_matrix_becomes_a_layer(self, loaded):
        a = to_anndata(loaded)
        gene_frames = {k for k, v in loaded.cell_data.items()
                       if list(v.columns) == list(loaded.cell_data["cell_exp"].columns)}
        assert gene_frames - {"cell_exp"} == set(a.layers)
        assert len(a.layers) >= 5                       # snv, snv_p, snv_m, cnv, rna_vaf at least
        np.testing.assert_allclose(a.X, loaded.cell_data["cell_exp"].values, rtol=1e-5, atol=1e-4)
        for name in a.layers:
            np.testing.assert_allclose(a.layers[name], loaded.cell_data[name].values,
                                       rtol=1e-4, atol=1e-4)

    def test_allele_layers_appear_only_when_the_program_layer_is_on(self, plain, loaded):
        assert "cell_exp_p" not in to_anndata(plain).layers
        assert {"cell_exp_p", "cell_exp_m", "cell_rna_baf"} <= set(to_anndata(loaded).layers)

    def test_every_annotation_column_becomes_an_obs_column(self, loaded):
        a = to_anndata(loaded)
        genes = list(loaded.cell_data["cell_exp"].columns)
        for name, frame in loaded.cell_data.items():
            if list(frame.columns) == genes or name in ("cell_crd", "cell_program", "cell_type"):
                continue
            for col in frame.columns:
                assert col in a.obs.columns, f"{name}.{col} was dropped"
                np.testing.assert_array_equal(np.asarray(a.obs[col].values),
                                              np.asarray(frame[col].values))

    def test_optional_annotations_are_absent_when_their_layer_is_off(self, plain, loaded):
        off = to_anndata(plain).obs.columns
        on = to_anndata(loaded).obs.columns
        assert "hypoxia_level" not in off and "cci_level" not in off
        assert {"hypoxia_level", "cci_level"} <= set(on)
        assert "gland_id" in off                        # the ductal field IS on in both

    def test_evolutionary_rates_and_driver_tallies_reach_obs(self, plain):
        a = to_anndata(plain)
        assert {"division_rate", "death_rate", "dispersal_rate"} <= set(a.obs.columns)
        assert {"n_mut_onc", "n_mut_tsg", "n_mut_breach"} <= set(a.obs.columns)

    def test_obs_extra_off_keeps_only_identity_and_coordinates(self, plain):
        a = to_anndata(plain, obs_extra=False)
        assert set(a.obs.columns) == {"genotype", "cell_type", "clone", "row", "col"}
        assert set(a.layers)                                # the matrices are untouched

    def test_programs_land_in_obsm_not_in_obs(self, loaded):
        a = to_anndata(loaded)
        prog = loaded.cell_data["cell_program"]
        assert "program" in a.obsm
        np.testing.assert_allclose(a.obsm["program"], prog.values)
        assert list(a.uns["program_names"]) == [str(c) for c in prog.columns]
        for col in prog.columns:
            assert col not in a.obs.columns

    def test_cells_and_genes_keep_their_names(self, plain):
        a = to_anndata(plain)
        assert list(a.obs_names) == [str(i) for i in plain.cell_data["cell_exp"].index]
        assert list(a.var_names) == list(plain.cell_data["cell_exp"].columns)

    def test_an_unknown_frame_is_routed_by_its_shape(self, plain):
        """The mapping must not be a hard-coded key list: a frame the export has never heard of
        goes to layers or obs purely on whether its columns are the gene index."""
        cd = dict(plain.cell_data)
        idx = cd["cell_exp"].index
        genes = list(cd["cell_exp"].columns)
        cd["cell_brand_new_matrix"] = pd.DataFrame(
            np.arange(len(idx) * len(genes)).reshape(len(idx), len(genes)) % 7.0,
            index=idx, columns=genes)
        cd["cell_brand_new_labels"] = pd.DataFrame(
            {"niche_score": np.linspace(0.0, 1.0, len(idx)),
             "sector": np.arange(len(idx)) % 3}, index=idx)

        a = to_anndata(cd, layers="all")
        assert "cell_brand_new_matrix" in a.layers
        np.testing.assert_allclose(a.layers["cell_brand_new_matrix"],
                                   cd["cell_brand_new_matrix"].values, rtol=1e-6)
        assert {"niche_score", "sector"} <= set(a.obs.columns)
        np.testing.assert_allclose(a.obs["niche_score"].values,
                                   cd["cell_brand_new_labels"]["niche_score"].values)
        assert "cell_brand_new_labels" not in a.layers

    def test_metastasis_layer_is_picked_up(self, plain, metastatic):
        assert "compartment" not in to_anndata(plain).obs.columns
        a = to_anndata(metastatic)
        comp = a.obs["compartment"].values
        np.testing.assert_array_equal(
            comp, metastatic.cell_data["cell_compartment"]["compartment"].values)
        assert set(np.unique(comp)) == {0, 1}, "the deposit should be colonised"
        # host parenchyma is a normal type, not a cancer clone
        types = a.obs["cell_type"].astype(str).values
        assert (types == "host").sum() > 0
        assert a.obs["clone"].isna().values[types == "host"].all()

    def test_a_tumour_with_nothing_materialised_is_materialised_first(self):
        t = _grow()                                     # its own tumour: this one gets emptied
        n = t.cell_data["cell_exp"].shape[0]
        t.cell_data = None
        a = to_anndata(t)
        assert a.n_obs == n and a.n_vars == t.n_genes

    def test_clashing_annotation_names_are_kept_apart(self, plain):
        cd = dict(plain.cell_data)
        idx = cd["cell_exp"].index
        cd["cell_extra"] = pd.DataFrame({"deme_id": np.zeros(len(idx), dtype=int)}, index=idx)
        a = to_anndata(cd, layers="all")
        assert "deme_id" in a.obs.columns and "deme_id_1" in a.obs.columns
        np.testing.assert_array_equal(a.obs["deme_id"].values,
                                      cd["cell_deme"]["deme_id"].values)
        assert (a.obs["deme_id_1"].values == 0).all()


class TestIdentityColumns:
    def test_genotype_clone_and_biological_type(self, plain):
        a = to_anndata(plain)
        gid = plain.cell_data["cell_type"]["cell_id"].astype(str).values
        np.testing.assert_array_equal(a.obs["genotype"].astype(str).values, gid)
        # the coarse type is derivable from the id alone: a normal cell's id IS its type
        truth = np.array([plain.genotypes[g].type for g in gid])
        np.testing.assert_array_equal(a.obs["cell_type"].astype(str).values, truth)
        assert set(np.unique(truth)) - {"cancer"}, "fixture should contain normal cells too"

    def test_clone_is_the_shared_clade_label(self, plain):
        a = to_anndata(plain, clone_min_cells=10)
        assert a.uns["clone_definition"] == "clade"
        labels = set(a.obs["clone"].dropna().astype(str))
        assert any(l.startswith("clone_") for l in labels)
        # normal cells belong to no clone
        normal = a.obs["cell_type"].astype(str).values != "cancer"
        assert a.obs["clone"].isna().values[normal].all()

    def test_clone_falls_back_to_the_genotype_id(self, plain):
        a = to_anndata(plain, clone_min_cells=None)
        assert a.uns["clone_definition"] == "genotype"
        np.testing.assert_array_equal(a.obs["clone"].astype(str).values,
                                      a.obs["genotype"].astype(str).values)

    def test_biological_types_helper(self):
        np.testing.assert_array_equal(
            biological_types(["12", "immune", "stromal", "epithelial", "host", "3"]),
            np.array(["cancer", "immune", "stromal", "epithelial", "host", "cancer"], dtype=object))

    def test_the_export_seam_is_importable_from_the_package(self):
        """``from_anndata`` / ``biological_types`` are part of the seam, not private helpers."""
        import iscc.integrations as I
        assert I.from_anndata is from_anndata and I.biological_types is biological_types
        assert {"to_anndata", "from_anndata", "biological_types"} <= set(I.__all__)


class TestCloneAnnotationIsUsable:
    """The ``obs["clone"]`` labels have to name most of the cancer cells, at every scale.

    A fixed ``clone_min_cells`` cannot: the share of cells left in the ancestral ``"other"`` bucket
    rises and falls with the threshold differently in every tumour (at realistic scale the old
    default of 50 stranded 61% of cancer cells in ``"other"`` while handing back only 6 clones). The
    default now searches for the threshold that strands the fewest cells while keeping the number of
    clones readable, and records what it chose.
    """

    def test_the_labels_describe_themselves(self, clonal):
        a = to_anndata(clonal)
        assert a.uns["clone_definition"] == "clade"
        assert a.uns["clone_min_cells_mode"] == "auto"
        assert int(a.uns["clone_min_cells"]) in _MIN_CELLS_LADDER
        assert 1 <= int(a.uns["clone_n_clones"]) <= _AUTO_MAX_CLONES
        assert 0.0 <= float(a.uns["clone_other_frac"]) <= 1.0
        doc = a.uns["clone_definition_doc"]
        assert str(a.uns["clone_min_cells"]) in doc and "other" in doc

    def test_auto_strands_fewer_cells_than_the_old_fixed_default(self, clonal):
        """50 is one of the candidates, so the search can only match or beat it where it applies."""
        auto = to_anndata(clonal)
        fixed = to_anndata(clonal, clone_min_cells=50)
        assert fixed.uns["clone_min_cells_mode"] == "fixed"
        assert fixed.uns["clone_min_cells"] == 50
        assert 2 <= int(fixed.uns["clone_n_clones"]) <= _AUTO_MAX_CLONES, "fixture lost its clones"
        assert auto.uns["clone_other_frac"] < fixed.uns["clone_other_frac"]
        assert auto.uns["clone_n_clones"] > fixed.uns["clone_n_clones"]

    def test_the_number_of_clones_stays_plottable(self, clonal, plain, loaded, metastatic):
        for t in (clonal, plain, loaded, metastatic):
            assert int(to_anndata(t).uns["clone_n_clones"]) <= _AUTO_MAX_CLONES

    def test_the_labels_match_the_recorded_definition(self, clonal):
        a = to_anndata(clonal)
        labelled = a.obs["clone"].dropna().astype(str)
        assert int(a.uns["clone_n_clones"]) == len(set(labelled) - {"other"})
        assert float(a.uns["clone_other_frac"]) == pytest.approx(
            (labelled == "other").mean(), abs=1e-9)
        # every labelled cell is a cancer cell, and the normal ones stay out of it
        assert (a.obs["cell_type"].astype(str)[a.obs["clone"].notna()] == "cancer").all()

    def test_no_clone_bookkeeping_when_the_labelling_is_off(self, clonal):
        a = to_anndata(clonal, clone_min_cells=None)
        assert a.uns["clone_definition"] == "genotype"
        assert "clone_min_cells" not in a.uns


class TestGeneAnnotation:
    def test_gene_roles_reach_var(self, plain):
        a = to_anndata(plain)
        gene_data = plain.get_gene_data()
        for key, frame in gene_data.items():
            assert key in a.var.columns
            np.testing.assert_array_equal(a.var[key].values, frame.iloc[:, 0].values)

    def test_germline_annotation_only_when_the_patient_carries_it(self, plain, loaded):
        assert "germline_types" not in to_anndata(plain).var.columns
        assert "germline_types" in to_anndata(loaded).var.columns

    def test_segment_index(self, plain):
        a = to_anndata(plain)
        sizes = plain.selection.segment_sizes
        np.testing.assert_array_equal(a.var["segment"].values,
                                      np.repeat(np.arange(len(sizes)), sizes))
        assert a.var["position_in_segment"].max() == max(sizes) - 1

    def test_program_loading_and_membership_in_varm(self, loaded):
        a = to_anndata(loaded)
        loading = loaded.program_truth["loading"]
        assert a.varm["program_loading"].shape == (a.n_vars, loading.shape[0])
        np.testing.assert_allclose(a.varm["program_loading"], loading.T)
        # the int-keyed gene->program map becomes a boolean genes x programs matrix
        member = a.varm["program_member"]
        assert member.dtype == bool and member.shape == a.varm["program_loading"].shape
        for gene, programs in loaded.program_truth["gene_program_map"].items():
            for p in programs:
                assert member[int(gene), int(p)]
        assert member.sum() == sum(len(v) for v in
                                   loaded.program_truth["gene_program_map"].values())

    def test_dosage_and_snv_class_reach_var(self, loaded):
        a = to_anndata(loaded)
        np.testing.assert_allclose(a.var["dosage_sensitivity"].values,
                                   loaded.program_truth["dosage_sensitivity"])
        np.testing.assert_array_equal(a.var["snv_class"].values, loaded.program_truth["snv_class"])
        names = np.asarray(loaded.program_truth["snv_class_names"], dtype=str)
        np.testing.assert_array_equal(a.var["snv_class_name"].astype(str).values,
                                      names[np.asarray(loaded.program_truth["snv_class"], int)])


class TestTumorLevelTruth:
    def test_clone_tree_edges_and_newick(self, plain):
        a = to_anndata(plain)
        edges = a.uns["clone_tree"]
        assert set(edges.columns) == {"child", "parent"}
        assert len(edges) == len(plain.genotypes_parents)
        assert dict(zip(edges["child"], edges["parent"])) == {
            str(k): str(v) for k, v in plain.genotypes_parents.items()}
        assert a.uns["clone_tree_newick"].endswith(";")

    def test_trace_counts_is_one_frame(self, plain):
        # tumor.traces is a list of dicts of dicts, which h5ad cannot store as-is
        counts = a_counts = to_anndata(plain).uns["trace_counts"]
        assert isinstance(counts, pd.DataFrame)
        assert len(a_counts) == len(plain.traces)
        last = plain.traces[-1]["genotypes_counts"]
        for gid, n in last.items():
            assert counts[str(gid)].iloc[-1] == float(n)

    def test_run_parameters(self, plain):
        a = to_anndata(plain)
        assert a.uns["grid_size"] == plain.grid_size
        assert a.uns["seed"] == plain.seed
        cfg = a.uns["config"]
        assert cfg["n_genes"] == plain.n_genes
        assert cfg["carrying_capacity"] == plain.carrying_capacity
        assert cfg["genome_params"]["n_segments"] == plain.n_segments


# --------------------------------------------------------------------------- h5ad round-trip
class TestRoundTrip:
    def test_h5ad_round_trip_preserves_everything(self, loaded, tmp_path):
        a = to_anndata(loaded)
        path = tmp_path / "tumour.h5ad"
        a.write_h5ad(str(path), compression="gzip")
        b = ad.read_h5ad(str(path))

        assert b.shape == a.shape
        assert list(b.obs_names) == list(a.obs_names) and list(b.var_names) == list(a.var_names)
        np.testing.assert_allclose(b.X, a.X)
        assert set(b.layers) == set(a.layers)
        for name in a.layers:
            np.testing.assert_allclose(b.layers[name], a.layers[name])
        assert set(b.obsm) == set(a.obsm)
        for key in a.obsm:
            np.testing.assert_allclose(b.obsm[key], a.obsm[key])
        assert set(b.varm) == set(a.varm)
        for key in a.varm:
            np.testing.assert_array_equal(b.varm[key], a.varm[key])
        assert set(b.obs.columns) == set(a.obs.columns)
        for col in a.obs.columns:
            np.testing.assert_array_equal(b.obs[col].astype(str).values,
                                          a.obs[col].astype(str).values)
        assert set(b.var.columns) == set(a.var.columns)
        for col in a.var.columns:
            np.testing.assert_array_equal(b.var[col].astype(str).values,
                                          a.var[col].astype(str).values)
        assert set(b.uns) == set(a.uns)
        assert b.uns["clone_tree_newick"] == a.uns["clone_tree_newick"]
        pd.testing.assert_frame_equal(b.uns["trace_counts"], a.uns["trace_counts"])
        pd.testing.assert_frame_equal(b.uns["clone_tree"], a.uns["clone_tree"])
        assert b.uns["grid_size"] == loaded.grid_size and b.uns["seed"] == loaded.seed
        assert b.uns["config"]["carrying_capacity"] == loaded.carrying_capacity
        assert b.uns["config"]["genome_params"]["n_segments"] == loaded.n_segments
        np.testing.assert_array_equal(b.uns["program_names"], a.uns["program_names"])

    def test_write_h5ad_method(self, loaded, tmp_path):
        path = tmp_path / "via_method.h5ad"
        out = loaded.write_h5ad(path)
        assert os.path.exists(str(out)) and os.path.getsize(str(path)) > 0
        b = ad.read_h5ad(str(path))
        assert b.n_obs == loaded.cell_data["cell_exp"].shape[0]
        assert b.n_vars == loaded.n_genes
        assert "spatial" in b.obsm and "clone_tree_newick" in b.uns

    def test_from_anndata_closes_the_loop(self, loaded, tmp_path):
        a = to_anndata(loaded)
        path = tmp_path / "loop.h5ad"
        a.write_h5ad(str(path), compression="gzip")
        cd = from_anndata(ad.read_h5ad(str(path)))

        assert set(cd) >= {"cell_exp", "cell_snv", "cell_cnv", "cell_crd", "cell_type"}
        np.testing.assert_allclose(cd["cell_exp"].values, loaded.cell_data["cell_exp"].values,
                                   rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(cd["cell_snv"].values, loaded.cell_data["cell_snv"].values,
                                   rtol=1e-4, atol=1e-4)
        pd.testing.assert_frame_equal(cd["cell_crd"], loaded.cell_data["cell_crd"],
                                      check_dtype=False)
        np.testing.assert_array_equal(cd["cell_type"]["cell_id"].values,
                                      loaded.cell_data["cell_type"]["cell_id"].astype(str).values)
        assert list(cd["cell_exp"].columns) == list(loaded.cell_data["cell_exp"].columns)

    def test_from_anndata_recovers_coordinates_without_obs_columns(self, plain):
        a = to_anndata(plain)
        del a.obs["row"], a.obs["col"]
        cd = from_anndata(a)
        pd.testing.assert_frame_equal(cd["cell_crd"], plain.cell_data["cell_crd"],
                                      check_dtype=False)


# --------------------------------------------------------------------------- E2: the axis order
class TestSpatialAxes:
    def test_spatial_is_x_then_y(self, plain):
        a = to_anndata(plain)
        crd = plain.cell_data["cell_crd"]
        np.testing.assert_array_equal(a.obsm["spatial"][:, 0], crd["col"].values)
        np.testing.assert_array_equal(a.obsm["spatial"][:, 1], crd["row"].values)
        assert a.uns["spatial_axes"] == "xy"

    def test_raw_row_and_col_are_never_lost(self, plain):
        a = to_anndata(plain)
        crd = plain.cell_data["cell_crd"]
        np.testing.assert_array_equal(a.obs["row"].values, crd["row"].values)
        np.testing.assert_array_equal(a.obs["col"].values, crd["col"].values)

    def test_rowcol_escape_hatch(self, plain):
        a = to_anndata(plain, spatial="rowcol")
        crd = plain.cell_data["cell_crd"]
        np.testing.assert_array_equal(a.obsm["spatial"][:, 0], crd["row"].values)
        assert a.uns["spatial_axes"] == "row_col"

    def test_bad_axis_order_is_rejected(self, plain):
        with pytest.raises(ValueError, match="spatial"):
            to_anndata(plain, spatial="yx")

    @staticmethod
    def _axis_correlations(cells, spot_x, members):
        """Correlate each spot's x with the mean of axis 0 and axis 1 of the cells it captured.

        Membership comes from the assay itself (cell identity), not from the coordinates, so the
        measurement cannot absorb a flipped axis order. Returns ``(r_axis0, r_axis1)``: with both
        objects on the same convention the first is ~1 and the second ~0; a transposed cell export
        swaps them.
        """
        c = np.asarray(cells.obsm["spatial"], dtype=float)
        keep = [i for i, m in enumerate(members) if len(m) >= 3]
        x = np.array([spot_x[i] for i in keep])
        a0 = np.array([c[members[i], 0].mean() for i in keep])
        a1 = np.array([c[members[i], 1].mean() for i in keep])
        return float(np.corrcoef(x, a0)[0, 1]), float(np.corrcoef(x, a1)[0, 1])

    def test_cell_level_and_visium_agree_on_orientation(self, plain):
        """The bug this fixes: a squidpy plot of the cells came out transposed relative to a Visium
        slide built from the SAME tissue, because the cell export wrote (row, col) while Visium
        wrote (col, row). Both now put the column index on axis 0."""
        from iscc.data.visium import Visium
        # section_frac=None + an explicit capture side keeps the spot grid in the tumour's own
        # coordinate frame, so the two objects are directly comparable
        vz = Visium(seed=1, section_frac=None).run(plain.cell_data, grid_side=plain.grid_size)
        spots = vz.to_anndata()
        members = vz._grid["members"]                    # positional cell indices per spot
        spot_x = np.asarray(spots.obsm["spatial"][:, 0], dtype=float)

        same, crossed = self._axis_correlations(to_anndata(plain), spot_x, members)
        assert same > 0.99, f"spot x vs cell axis-0 correlation only {same:.3f}"
        assert abs(crossed) < 0.2, f"spot x leaks into cell axis 1 at r={crossed:.3f}"

        # the pre-fix order is exactly what "rowcol" reproduces, and it is transposed
        old0, old1 = self._axis_correlations(to_anndata(plain, spatial="rowcol"), spot_x, members)
        assert abs(old0) < 0.2 and old1 > 0.99, (old0, old1)

    def test_assay_exports_share_the_convention(self, plain):
        from iscc.data.rna import scRNA
        from iscc.data.imaging import scSpatial
        from iscc.data.dna import scDNA
        crd = plain.cell_data["cell_crd"]
        for assay in (scRNA(seed=1, n_cells=40), scSpatial(seed=1, n_panel_genes=20),
                      scDNA(seed=1, n_cells=40)):
            a = assay.run(plain.cell_data).to_anndata()
            want = crd.reindex([str(c) for c in a.obs_names])
            np.testing.assert_array_equal(a.obsm["spatial"][:, 0], want["col"].values)
            np.testing.assert_array_equal(a.obsm["spatial"][:, 1], want["row"].values)
            assert a.uns["spatial_axes"] == "xy"

    def test_visium_convention_is_unchanged(self, plain):
        from iscc.data.visium import Visium
        vz = Visium(seed=1, section_frac=1.0).run(plain.cell_data)
        a = vz.to_anndata()
        np.testing.assert_array_equal(a.obsm["spatial"][:, 0], a.obs["col"].values)
        np.testing.assert_array_equal(a.obsm["spatial"][:, 1], a.obs["row"].values)
        # Visium always wrote (x, y); it just never said so, unlike the other four exporters.
        assert a.uns["spatial_axes"] == "xy"


# --------------------------------------------------------------------------- the dict path
class TestCellDataDictPath:
    def test_dict_keeps_the_historical_schema(self, plain):
        a = to_anndata(plain.cell_data)
        assert set(a.layers) == {"cell_snv", "cell_cnv"}
        assert set(a.obs.columns) == {"clone", "deme", "row", "col"}
        assert a.var.shape[1] == 0 and not a.varm
        assert set(a.uns) == {"source", "spatial_axes"}
        np.testing.assert_array_equal(a.obs["clone"].astype(str).values,
                                      plain.cell_data["cell_type"]["cell_id"].astype(str).values)

    def test_dict_can_opt_into_the_full_mapping(self, loaded):
        a = to_anndata(loaded.cell_data, layers="all")
        assert "cell_rna_vaf" in a.layers and "program" in a.obsm
        assert {"genotype", "cell_type", "clone", "gland_id"} <= set(a.obs.columns)
        assert a.var.shape[1] == 0, "a bare dict carries no gene-level truth"

    def test_dict_spatial_is_also_xy(self, plain):
        a = to_anndata(plain.cell_data)
        np.testing.assert_array_equal(a.obsm["spatial"][:, 0],
                                      plain.cell_data["cell_crd"]["col"].values)

    def test_unknown_source_is_rejected(self):
        with pytest.raises(TypeError, match="cell_data"):
            to_anndata("not a tumour")

    def test_missing_x_frame_is_rejected(self, plain):
        with pytest.raises(KeyError):
            to_anndata(plain, X="cell_nope")


# --------------------------------------------------------------------------- size / cost
def test_export_is_smaller_than_the_dense_matrices(loaded, tmp_path):
    """gzip has to actually pay off — the frames are dominated by repeated per-clone rows."""
    a = to_anndata(loaded)
    path = tmp_path / "size.h5ad"
    a.write_h5ad(str(path), compression="gzip")
    dense = a.X.nbytes + sum(a.layers[k].nbytes for k in a.layers)
    assert os.path.getsize(str(path)) < dense
