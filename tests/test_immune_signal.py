"""The immune niche field: the immune density of each deme, smoothed over the lattice, drives the
program it is mapped to (the interferon response of cancer cells near T and NK cells)."""
import numpy as np

from conftest import GENOME_PARAMS, SELECTION_PARAMS, CANCER_CELL_PARAMS
from iscc.tumor.models import GenotypeTumor
from iscc.tumor.programs import ProgramModel


def _expression(strength):
    return {"program_params": {"n_programs": 6, "n_genes_per_program": 8,
                               "seeded_programs": ["proliferation", "emt", "hypoxia", "drug_resistance",
                                                   "immune_evasion", "interferon_response"]},
            "coupling_params": {"niche_program_map": {"immune": "interferon_response"},
                                "niche_program_strength": {"__default__": 1.0, "immune": strength}},
            "activity_params": {"activity_mean": 1.0, "activity_sd": 0.3, "activity_noise": 0.05}}


def _tumour(strength, immune_density=0.3, **spatial):
    t = GenotypeTumor(seed=3, genome_params=GENOME_PARAMS, selection_params=SELECTION_PARAMS,
                      cancer_cell_params=CANCER_CELL_PARAMS,
                      deme_params={"carrying_capacity": 10, "initial_cancer_cells": 5},
                      spatial_params={"grid_size": 8, "structure_radius": 0,
                                      "immune_density": immune_density, **spatial},
                      expression_params=_expression(strength))
    t.grow(n_steps=200, seed=3)
    t.make_cell_data()
    return t


def test_niche_strength_per_field():
    P = ProgramModel(n_genes=100, segment_sizes=[50, 50], layout_seed=42, run_seed=0,
                     expression_params={"program_params": {"n_programs": 6, "n_genes_per_program": 10},
                                        "coupling_params": {
                                            "niche_program_map": {"hypoxia": "hypoxia", "immune": "emt"},
                                            "niche_program_strength": {"__default__": 2.0, "immune": 10.0}}})
    k = P.dictionary.program_index
    D = P.niche_drive({"hypoxia": np.array([0.0, 0.5]), "immune": np.array([0.1, 0.2])})
    np.testing.assert_allclose(D[:, k["hypoxia"]], [0.0, 1.0])
    np.testing.assert_allclose(D[:, k["emt"]], [1.0, 2.0])


def test_immune_field_is_the_smoothed_immune_density():
    t = _tumour(10.0)
    field = t.microenv_truth["immune"]
    assert field.shape == (len(t.demes),)
    # uniform seeding at 0.3 of K: the field is 0.3 everywhere, smoothing or not
    np.testing.assert_allclose(field, 0.3)
    np.testing.assert_allclose(t._immune_field(lengthscale=0), 0.3)


def test_immune_field_drives_only_its_program():
    on, off = _tumour(10.0), _tumour(0.0)
    zon, zoff = on.cell_data["cell_program"], off.cell_data["cell_program"]
    assert zon.index.equals(zoff.index)
    assert zon["interferon_response"].mean() > zoff["interferon_response"].mean()
    others = [c for c in zon.columns if c != "interferon_response"]
    np.testing.assert_allclose(zon[others].to_numpy(), zoff[others].to_numpy())


def test_no_field_without_the_map_or_without_immune_cells():
    t = _tumour(10.0, immune_density=0.0)
    np.testing.assert_allclose(t.microenv_truth["immune"], 0.0)
    plain = GenotypeTumor(seed=3, genome_params=GENOME_PARAMS, selection_params=SELECTION_PARAMS,
                          cancer_cell_params=CANCER_CELL_PARAMS,
                          deme_params={"carrying_capacity": 10, "initial_cancer_cells": 5},
                          spatial_params={"grid_size": 8, "structure_radius": 0, "immune_density": 0.3},
                          expression_params={"program_params": {"n_programs": 6, "n_genes_per_program": 8}})
    plain.grow(n_steps=50, seed=3)
    plain.make_cell_data()
    assert "immune" not in (getattr(plain, "microenv_truth", None) or {})
