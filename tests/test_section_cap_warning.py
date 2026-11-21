"""The warning that ``max_cells`` thins a physical section fires for sections only, at the user's call."""
import warnings

import pytest

from conftest import GENOME_PARAMS, SELECTION_PARAMS, CANCER_CELL_PARAMS, DEME_PARAMS
from iscc.sample import Resection
from iscc.tumor import GenotypeTumor
from iscc.tumor.models.materialise import SectionCapWarning


@pytest.fixture(scope="module")
def tumour():
    t = GenotypeTumor(seed=3, genome_params=GENOME_PARAMS, selection_params=SELECTION_PARAMS,
                      cancer_cell_params=CANCER_CELL_PARAMS,
                      deme_params={**DEME_PARAMS, "carrying_capacity": 14, "initial_cancer_cells": 6},
                      spatial_params={"grid_size": 14})
    t.grow(n_steps=60, seed=3)
    return t


def test_a_capped_dissociation_does_not_warn(tumour):
    with warnings.catch_warnings():
        warnings.simplefilter("error", SectionCapWarning)
        Resection(tumour).dissociate(max_cells=5, seed=0)


def test_a_capped_section_warns_at_the_callers_line(tumour):
    with pytest.warns(SectionCapWarning) as record:
        Resection(tumour).slice(depth_frac=0.9, max_cells=5)
    assert record[0].filename == __file__
