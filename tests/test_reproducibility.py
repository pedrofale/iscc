"""Reproducibility: the same config + seed must produce an identical tumor.

Guards the seeded-RNG plumbing (Selection driver layout, baseline expression, spatial seeding and
the per-run growth streams) that together make simulations deterministic, under both engines.
"""
import pytest

from conftest import (
    GENOME_PARAMS, SELECTION_PARAMS, CANCER_CELL_PARAMS,
    EPITHELIAL_CELL_PARAMS, STROMAL_CELL_PARAMS, IMMUNE_CELL_PARAMS, DEME_PARAMS,
)
from iscc.tumor.models import GenotypeTumor

STEPS = {"exact": 400, "tau": 20}


def _build(seed, mode):
    t = GenotypeTumor(
        genome_params=GENOME_PARAMS, selection_params=SELECTION_PARAMS,
        cancer_cell_params=CANCER_CELL_PARAMS, epithelial_cell_params=EPITHELIAL_CELL_PARAMS,
        stromal_cell_params=STROMAL_CELL_PARAMS, immune_cell_params=IMMUNE_CELL_PARAMS,
        deme_params=DEME_PARAMS, spatial_params={"grid_size": 7, "structure_radius": 2},
        update_mode=mode, seed=seed,
    )
    t.grow(n_steps=STEPS[mode], seed=seed)
    return t


@pytest.mark.parametrize("mode", list(STEPS))
def test_same_seed_gives_identical_tumor(mode):
    a, b = _build(0, mode), _build(0, mode)
    assert a.get_tumor_size() == b.get_tumor_size()
    # identical driver/selection layout
    assert list(a.selection.get_oncogenes()) == list(b.selection.get_oncogenes())
    assert list(a.selection.get_tsgs()) == list(b.selection.get_tsgs())
    # identical per-cell ground truth (values, order, index, columns)
    for key in ["cell_snv", "cell_cnv", "cell_exp", "cell_crd"]:
        assert a.cell_data[key].equals(b.cell_data[key]), f"{key} differs for same seed"


@pytest.mark.parametrize("mode", list(STEPS))
def test_different_seed_changes_something(mode):
    a, b = _build(0, mode), _build(1, mode)
    assert (a.get_tumor_size() != b.get_tumor_size()
            or not a.cell_data["cell_snv"].equals(b.cell_data["cell_snv"]))
