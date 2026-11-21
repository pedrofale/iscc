"""A saved and loaded tumour or cohort is the same object as the one saved.

The file leaves out the per-cell tables, so these tests check that the loaded tumour rebuilds them
exactly, and that it samples and grows on exactly as the original does.
"""
import functools
import json

import pandas as pd
import pytest

from conftest import GENOME_PARAMS, SELECTION_PARAMS, CANCER_CELL_PARAMS, DEME_PARAMS
from iscc.cohort import Cohort, grow_tumor
from iscc.sample import Resection
from iscc.tumor import GenotypeTumor
from iscc.tumor.components.cell import Cell
from iscc.tumor.io import MAGIC, read_header

SPATIAL = {"grid_size": 14, "structure_radius": 3, "n_glands": 2, "gland_radius": 3,
           "min_gland_sep": 7, "K_duct": 24, "K_stroma": 14, "stroma_fill_frac": 0.3}
PARAMS = dict(genome_params=GENOME_PARAMS,
              selection_params={**SELECTION_PARAMS, "prop_breach": 0.2},
              cancer_cell_params={**CANCER_CELL_PARAMS, "wgd_rate": 0.05},
              deme_params={**DEME_PARAMS, "carrying_capacity": 14, "initial_cancer_cells": 6},
              spatial_params=SPATIAL)


def _grow():
    t = GenotypeTumor(seed=3, **PARAMS)
    t.grow(n_steps=60, seed=3)
    return t


def _assert_same_frames(a, b):
    assert set(a) == set(b)
    for k in a:
        pd.testing.assert_frame_equal(a[k], b[k])


@pytest.fixture(scope="module")
def tumour():
    return _grow()


def test_loaded_tumour_rebuilds_the_same_cell_data(tumour, tmp_path):
    loaded = GenotypeTumor.load(tumour.save(tmp_path / "t.iscc"))
    _assert_same_frames(tumour.cell_data, loaded.cell_data)


def test_save_leaves_the_original_untouched(tumour, tmp_path):
    before = tumour.cell_data
    tumour.save(tmp_path / "t.iscc")
    assert tumour.cell_data is before


def test_loaded_tumour_samples_and_grows_on_as_the_original(tmp_path):
    a = _grow()
    b = GenotypeTumor.load(a.save(tmp_path / "t.iscc"))
    _assert_same_frames(Resection(a).dissociate(max_cells=200, seed=0),
                        Resection(b).dissociate(max_cells=200, seed=0))
    a.grow(n_steps=20)
    b.grow(n_steps=20)
    # New clones take the next ids of the process, so the two tumours name them differently.
    assert sorted(a.genotypes_counts.values()) == sorted(b.genotypes_counts.values())
    assert a.get_cancer_size() == b.get_cancer_size()


def test_clones_born_after_loading_in_a_new_process_take_new_ids(tumour, tmp_path):
    path = tumour.save(tmp_path / "t.iscc")
    Cell._genotype_counter = 0                      # a new process
    loaded = GenotypeTumor.load(path)
    old = dict(loaded.genotypes_parents)
    loaded.grow(n_steps=20)
    assert all(loaded.genotypes_parents[g] == p for g, p in old.items())
    assert Cell._genotype_counter >= max(int(g) for g in old)


def test_header_names_the_class_and_the_cell_data(tumour, tmp_path):
    header = read_header(tumour.save(tmp_path / "t.iscc"))
    assert header["format"] == 1
    assert header["kind"] == "GenotypeTumor"
    assert header["cell_data"] is True


def test_tumour_grown_without_materialising_loads_without_cell_data(tmp_path):
    t = GenotypeTumor(seed=3, **PARAMS)
    t.grow(n_steps=30, seed=3, materialize=False)
    assert GenotypeTumor.load(t.save(tmp_path / "t.iscc")).cell_data is None


def test_file_of_another_kind_is_refused(tumour, tmp_path):
    with pytest.raises(TypeError, match="GenotypeTumor.load"):
        Cohort.load(tumour.save(tmp_path / "t.iscc"))
    other = tmp_path / "other.pkl"
    other.write_bytes(b"not a tumour")
    with pytest.raises(ValueError, match="not a file written"):
        GenotypeTumor.load(other)


def test_file_of_another_format_is_refused(tumour, tmp_path):
    path = tumour.save(tmp_path / "t.iscc")
    raw = path.read_bytes()
    header_end = raw.index(b"\n", len(MAGIC)) + 1
    header = json.loads(raw[len(MAGIC):header_end])
    header["format"] = 99
    path.write_bytes(MAGIC + json.dumps(header).encode() + b"\n" + raw[header_end:])
    with pytest.raises(ValueError, match="format 99"):
        GenotypeTumor.load(path)


def test_cohort_round_trip_rebuilds_patients_when_read(tmp_path):
    grow = functools.partial(grow_tumor, grow_steps=40, **PARAMS)
    cohort = Cohort(grow, patient_seeds=[1, 2]).run()
    loaded = Cohort.load(cohort.save(tmp_path / "c.iscc"))
    assert loaded.patient_seeds == [1, 2]
    assert all(p.tumor.cell_data is None for p in loaded.patients)
    for a, b in zip(cohort.patients, loaded.patients):
        _assert_same_frames(a.cell_data, b.cell_data)
