"""Smoke tests for the shared realistic regime (``validation/realistic_regime.py``).

Keeps the ONE canonical grid-170 ductal-field regime — that every benchmark and science notebook now
grows from — importable and functionally intact: it grows a structured, mixed-compartment tumour and
samples tractable, memory-safe pieces without materialising the whole cm-scale tissue. Runs at the
``small`` preset (identical breach-gated biology, tiny field) so it stays fast.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "validation"))
import realistic_regime as R  # noqa: E402


def test_canonical_blocks_trace_to_example_config():
    # The regime IS the tutorial config — grid-170 breach-gated ductal field, 600-gene genome.
    assert R.SPATIAL["grid_size"] == 170
    assert R.SPATIAL["breach_gated_invasion"] is True
    assert R.GENOME["n_segments"] * R.GENOME["segment_size"] == 600
    assert R.MAX_CELLS and R.COARSEN is True


def test_config_for_merges_overrides_without_mutating_base():
    cfg = R.config_for(scale="small", cancer={"wgd_rate": 0.05}, spatial={"stromal_hazard": 0.9})
    assert cfg["spatial_params"]["grid_size"] == R.SCALES["small"]["grid_size"]
    assert cfg["cancer_cell_params"]["wgd_rate"] == 0.05
    assert cfg["spatial_params"]["stromal_hazard"] == 0.9
    # canonical blocks are untouched (overrides are merged onto copies)
    assert "wgd_rate" not in R.CANCER or R.CANCER.get("wgd_rate") != 0.05
    assert R.SPATIAL["grid_size"] == 170


def test_bad_scale_rejected():
    with pytest.raises(ValueError):
        R.config_for(scale="galactic")


def test_grow_is_memory_safe_by_default_and_samples_tractably():
    # Grow a tiny structured lesion; NOT materialised by default (cm-scale discipline).
    t = R.grow_realistic(seed=3, target_cancer=400, scale="small", max_cells=800)
    assert R.n_cancer(t) >= 400
    assert t.get_cancer_size() == R.n_cancer(t)
    assert t.cell_data is None  # grow_realistic left it as counts, never expanded

    # Sampling materialises only the sampled piece, bounded to max_cells.
    diss = R.dissociated(t, max_cells=300)
    sec = R.section(t, depth_frac=0.4, max_cells=300)
    assert 0 < len(diss["cell_type"]) <= 300
    assert 0 < len(sec["cell_type"]) <= 300

    # The sample is a MIXED microenvironment (not a pre-filtered cancer-only matrix).
    types = {t.genotypes[x].type for x in diss["cell_type"]["cell_id"].astype(str).values
             if x in t.genotypes}
    assert "cancer" in types
    assert types & set(R.NORMALS)  # at least one normal compartment present
