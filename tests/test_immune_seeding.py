"""Where immune cells are seeded: in the stroma at ``immune_density`` and in the duct wall at
``immune_duct_density``, each a fraction of the deme's own capacity, never in an empty lumen; in every
deme of a field without glands."""
from conftest import GENOME_PARAMS, SELECTION_PARAMS, CANCER_CELL_PARAMS
from iscc.tumor.models import GenotypeTumor

FIELD = {"grid_size": 20, "structure_radius": 3, "n_glands": 1, "K_duct": 20, "K_stroma": 10,
         "stroma_fill_frac": 0.3}


def _tumour(spatial):
    return GenotypeTumor(seed=1, genome_params=GENOME_PARAMS, selection_params=SELECTION_PARAMS,
                         cancer_cell_params=CANCER_CELL_PARAMS,
                         deme_params={"carrying_capacity": 10, "initial_cancer_cells": 5},
                         spatial_params=spatial)


def _immune_by_compartment(t):
    lumen = {i for demes in t.gland_lumen_demes for i in demes}
    out = {"stroma": set(), "wall": set(), "lumen": set()}
    for i, deme in enumerate(t.demes):
        part = "lumen" if i in lumen else ("wall" if t.gland_id[i] != -1 else "stroma")
        out[part].add(deme.get("immune", 0))
    return out


def test_stroma_wall_and_lumen():
    t = _tumour({**FIELD, "immune_density": 0.3, "immune_duct_density": 0.1})
    assert _immune_by_compartment(t) == {"stroma": {3}, "wall": {2}, "lumen": {0}}


def test_duct_density_defaults_to_the_stroma_density():
    t = _tumour({**FIELD, "immune_density": 0.3})
    assert _immune_by_compartment(t) == {"stroma": {3}, "wall": {6}, "lumen": {0}}


def test_field_without_glands_has_immune_cells_everywhere():
    t = _tumour({"grid_size": 8, "structure_radius": 0, "immune_density": 0.4})
    assert all(deme.get("immune", 0) == 4 for deme in t.demes)


def test_no_immune_cells_by_default():
    t = _tumour(dict(FIELD))
    assert "immune" not in t.genotypes and not t._has_immune
