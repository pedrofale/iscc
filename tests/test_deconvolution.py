"""Spatial-deconvolution benchmark tests (cell2location + RCTD) — the FLAGSHIP integration demo.

Two layers, mirroring test_integration.py:
  * the iscc-side data generation and scoring (``validation/deconv_common.py``) are ALWAYS exercised —
    a small deterministic tumour must yield a four-type Visium section with a well-formed true per-spot
    composition, matched vs mismatched references whose composition differs measurably, and scoring
    helpers that behave;
  * the REAL tools run only when their dedicated env is present (``iscc-cell2location`` / ``iscc-rctd``);
    otherwise the test skips, so the core ``iscc`` suite stays green without the heavy stacks.

Non-circularity: the cell types differ transcriptionally because the R13 program layer gives each a
distinct COMBINATION of scattered functional programmes (an emergent property), not a hand-drawn marker
table matching the deconvolution model.
"""
import os
import sys
import tempfile

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "validation"))
import deconv_common as D  # noqa: E402

# A small, fast, deterministic four-type tumour (smaller grid than the paper figure).
GENOME = {"n_segments": 8, "segment_size": 25}
SPATIAL = {"grid_size": 18, "structure_radius": 7, "immune_density": 0.12}
DEME = {"carrying_capacity": 10, "initial_cancer_cells": 8}


@pytest.fixture(scope="module")
def tumor():
    return D.grow_tumor(seed=3, steps=1200, genome=GENOME, spatial=SPATIAL, deme=DEME)


@pytest.fixture(scope="module")
def section(tumor):
    return D.build_section(tumor, spot_radius=0.9, spot_pitch=1.5, section_radius=7.0)


# --------------------------------------------------------------------- data generation --------
class TestDataGeneration:
    def test_four_types_present(self, tumor):
        types = D.coarse_types(tumor)
        vc = types.value_counts()
        assert {"cancer", "epithelial", "stromal", "immune"} <= set(vc.index)
        assert vc["cancer"] >= 20                                  # a real lesion to deconvolve

    def test_section_true_composition_wellformed(self, section):
        T = section["true_type"]
        assert T.shape[1] == 4
        assert len(section["type_categories"]) == 4
        # every occupied spot's proportions sum to 1 (composition, not counts)
        assert np.allclose(T.sum(1), 1.0, atol=1e-6)
        assert section["true_type"].max() <= 1.0 and section["true_type"].min() >= 0.0
        # spots are genuine mixtures, not all pure
        assert (T.max(1) < 0.999).mean() > 0.2

    def test_expression_is_realistic_not_a_pileup(self, section):
        # the R13 program layer must not blow one marker gene up to the whole library
        tot = section["spot_counts"].sum(0)
        assert float(tot.max() / tot.sum()) < 0.25

    def test_regional_reference_composition_differs(self, tumor, section):
        # the whole point: a different-region reference has a measurably different composition
        oracle = D.build_reference(tumor, section, mode="oracle", n_cells=300, seed=1)
        regional = D.build_reference(tumor, section, mode="regional", offset=9.0, n_cells=300, seed=1)
        oc = np.array([oracle["composition"][c] for c in section["type_categories"]])
        rc = np.array([regional["composition"][c] for c in section["type_categories"]])
        assert np.abs(oc - rc).sum() > 0.15                       # a real, measurable mismatch

    def test_dissociation_depletes_fragile_types(self, tumor, section):
        # F2 under-recovers immune (fragile) relative to no dissociation
        no_diss = D.build_reference(tumor, section, mode="regional", offset=4.0,
                                    dissociation_strength=0.0, n_cells=None, seed=1)
        diss = D.build_reference(tumor, section, mode="regional", offset=4.0,
                                 dissociation_strength=1.0, n_cells=None, seed=1)
        assert diss["composition"]["immune"] <= no_diss["composition"]["immune"] + 1e-9

    def test_population_labels_split_cancer_into_clones(self, tumor, section):
        ref = D.build_reference(tumor, section, mode="oracle", label_by="population", n_cells=300, seed=1)
        cats = set(ref["categories"])
        assert any(c.startswith("clone") for c in cats)           # cancer -> CNA clones
        assert "cancer" not in cats and "immune" in cats          # normals keep their type


# --------------------------------------------------------------------- scoring helpers --------
class TestScoring:
    def test_score_proportions_perfect(self, section):
        T = section["true_type"]
        sc = D.score_proportions(T, T, section["type_categories"])
        assert sc["jsd"] == pytest.approx(0.0, abs=1e-6)
        assert sc["rmse"] == pytest.approx(0.0, abs=1e-6)
        assert sc["flat_r"] == pytest.approx(1.0)

    def test_score_proportions_penalises_missing_type(self, section):
        # a reference lacking a type (inferred column absent) is penalised, not silently ignored
        T = section["true_type"]
        cats = section["type_categories"]
        sc = D.score_proportions(T, T[:, :3], cats, inferred_categories=cats[:3])
        assert sc["jsd"] > 0.0                                     # missing 'immune' mass hurts

    def test_score_proportions_chance(self, section):
        rng = np.random.default_rng(0)
        T = section["true_type"]
        rand = rng.dirichlet(np.ones(4), size=T.shape[0])
        sc = D.score_proportions(T, rand, section["type_categories"])
        assert sc["flat_r"] < 0.5                                  # random is not inflated


# --------------------------------------------------------------------- real tools --------------
def _oracle_beats_chance(tool_runner, tumor, section):
    ref = D.build_reference(tumor, section, mode="oracle", n_cells=350, seed=1)
    with tempfile.TemporaryDirectory() as work:
        props, cats = tool_runner(ref, section, work)
    sc = D.score_proportions(section["true_type"], props.values, section["type_categories"], cats)
    assert props.shape[0] == section["true_type"].shape[0]
    assert sc["flat_r"] > 0.5                                      # a matched reference works well
    return sc


@pytest.mark.skipif(not D.rctd_available(),
                    reason="iscc-rctd env (R + spacexr) not installed")
class TestRCTDReal:
    def test_oracle_reference_recovers_composition(self, tumor, section):
        _oracle_beats_chance(lambda ref, sec, wd: D.run_rctd(ref, sec, wd, mode="full"),
                             tumor, section)


@pytest.mark.skipif(not D.cell2location_available(),
                    reason="iscc-cell2location env not installed")
class TestCell2locationReal:
    def test_oracle_reference_recovers_composition(self, tumor, section):
        _oracle_beats_chance(
            lambda ref, sec, wd: D.run_cell2location(ref, sec, wd, epochs=80, epochs_sp=400),
            tumor, section)
