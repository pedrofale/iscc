"""Tests for the genotype-level (count-based) engine, GenotypeTumor (DESIGN phase 3b).

Validates: builds/grows/materialises the standard schema, is reproducible (seed -> identical
matrices), and is *statistically equivalent* to the cell-level GlandularTumor (it is NOT
byte-identical — it draws different random variables — so we check survival and survivor-size
distributions match, not exact values).
"""
import hashlib

import numpy as np
import pytest

from conftest import (
    GENOME_PARAMS, SELECTION_PARAMS, CANCER_CELL_PARAMS,
    EPITHELIAL_CELL_PARAMS, STROMAL_CELL_PARAMS, IMMUNE_CELL_PARAMS, DEME_PARAMS,
)
from iscc.tumor.models import GenotypeTumor, GlandularTumor

SPATIAL = {"grid_size": 15, "n_structures": 1, "structure_radius": 0}
# death_rate 0 -> the founder can never stochastically die out, so schema/reproducibility
# checks are deterministic. The equivalence test uses a small death rate on purpose.
NO_DEATH = {**CANCER_CELL_PARAMS, "death_rate": 0.0}
LOW_DEATH = {**CANCER_CELL_PARAMS, "death_rate": 0.05}


def _count(seed, steps, cancer=NO_DEATH):
    t = GenotypeTumor(
        seed=seed, genome_params=GENOME_PARAMS, selection_params=SELECTION_PARAMS,
        cancer_cell_params=cancer, deme_params=DEME_PARAMS, spatial_params=SPATIAL,
    )
    t.grow(n_steps=steps, seed=seed)
    return t


def _cell(seed, steps, cancer=NO_DEATH):
    t = GlandularTumor(
        seed=seed, genome_params=GENOME_PARAMS, selection_params=SELECTION_PARAMS,
        cancer_cell_params=cancer, epithelial_cell_params=EPITHELIAL_CELL_PARAMS,
        stromal_cell_params=STROMAL_CELL_PARAMS, immune_cell_params=IMMUNE_CELL_PARAMS,
        deme_params=DEME_PARAMS, grid_size=SPATIAL["grid_size"], structure_radius=0,
    )
    t.grow(n_steps=steps, seed=seed)
    return t


def test_builds_grows_and_materialises_schema():
    t = _count(seed=1, steps=150)
    cd = t.cell_data
    assert {"cell_snv", "cell_cnv", "cell_exp", "cell_crd", "cell_type", "cell_deme", "cell_evo"} <= set(cd)
    n = t.get_tumor_size()
    assert n > 0
    assert cd["cell_snv"].shape == (n, t.n_genes)
    assert list(cd["cell_crd"].columns) == ["row", "col"]
    # every materialised cell maps to a live genotype
    assert set(cd["cell_type"]["cell_id"]).issubset(set(t.genotypes_counts))


def test_reproducible_same_seed():
    def fingerprint(seed):
        t = _count(seed, steps=150)
        h = lambda k: hashlib.md5(np.ascontiguousarray(t.cell_data[k].values)).hexdigest()
        return t.get_tumor_size(), h("cell_snv"), h("cell_cnv"), h("cell_exp"), h("cell_crd")
    assert fingerprint(3) == fingerprint(3)
    assert fingerprint(3) != fingerprint(4)


def test_statistically_equivalent_to_cell_engine():
    seeds, steps = range(10), 150
    cnt = [_count(s, steps, cancer=LOW_DEATH).get_tumor_size() for s in seeds]
    cell = [_cell(s, steps, cancer=LOW_DEATH).get_tumor_size() for s in seeds]

    # both engines have the same low extinction probability (death/birth = 0.1)
    assert sum(s > 0 for s in cnt) >= 6
    assert sum(s > 0 for s in cell) >= 6

    # survivor sizes in the same ballpark (net birth-death drift per step is identical)
    cnt_mean = np.mean([s for s in cnt if s > 0])
    cell_mean = np.mean([s for s in cell if s > 0])
    assert 0.5 < cnt_mean / cell_mean < 2.0
