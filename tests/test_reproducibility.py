"""Reproducibility: the same config + seed must produce an identical tumor.

Guards the seeded-RNG plumbing (Selection driver layout, baseline expression, spatial
seeding) and the insertion-ordered `Deme.cells` that together make simulations
deterministic.
"""
from conftest import (
    GENOME_PARAMS, SELECTION_PARAMS, CANCER_CELL_PARAMS,
    EPITHELIAL_CELL_PARAMS, STROMAL_CELL_PARAMS, IMMUNE_CELL_PARAMS, DEME_PARAMS,
)
from iscc.tumor.models.glandular import GlandularTumor


def _build(seed):
    t = GlandularTumor(
        genome_params=GENOME_PARAMS, selection_params=SELECTION_PARAMS,
        cancer_cell_params=CANCER_CELL_PARAMS, epithelial_cell_params=EPITHELIAL_CELL_PARAMS,
        stromal_cell_params=STROMAL_CELL_PARAMS, immune_cell_params=IMMUNE_CELL_PARAMS,
        deme_params=DEME_PARAMS, grid_size=7, structure_radius=2, seed=seed,
    )
    t.grow(n_steps=40, seed=seed)
    return t


def test_same_seed_gives_identical_tumor():
    a, b = _build(0), _build(0)
    assert a.get_tumor_size() == b.get_tumor_size()
    # identical driver/selection layout
    assert list(a.selection.get_oncogenes()) == list(b.selection.get_oncogenes())
    assert list(a.selection.get_tsgs()) == list(b.selection.get_tsgs())
    # identical per-cell ground truth (values, order, index, columns)
    for key in ["cell_snv", "cell_cnv", "cell_exp", "cell_crd"]:
        assert a.cell_data[key].equals(b.cell_data[key]), f"{key} differs for same seed"


def test_different_seed_changes_something():
    a, b = _build(0), _build(1)
    differs = (
        list(a.selection.get_oncogenes()) != list(b.selection.get_oncogenes())
        or not a.cell_data["cell_snv"].equals(b.cell_data["cell_snv"])
    )
    assert differs
