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
    GENOME_PARAMS, SELECTION_PARAMS, CANCER_CELL_PARAMS, N_SEGMENTS, SEGMENT_SIZE,
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


def test_demes_cap_near_carrying_capacity():
    # DESIGN_crowding.md (Option A): with density-dependent death the tumour SPREADS across demes and
    # each deme caps NEAR carrying_capacity, instead of the old bug where evolved clones outran the
    # absolute death cap and piled 1000s of cells into a few demes.
    K = 8
    cancer = {"division_rate": 0.6, "death_rate": 0.02, "max_birth_rate": 0.9,
              "mutation_rate": 0.3, "dispersal_rate": 0.1}
    t = GenotypeTumor(seed=2, genome_params=GENOME_PARAMS, selection_params=SELECTION_PARAMS,
                      cancer_cell_params=cancer,
                      deme_params={"carrying_capacity": K, "maximum_death_rate": 1.0,
                                   "initial_cancer_cells": 5},
                      spatial_params={"grid_size": 25, "structure_radius": 0},
                      update_mode="tau", tau=1.0)
    t.grow(n_steps=60, seed=2)
    occ = [sum(v for g, v in d.items() if t._is_cancer(g))
           for d in t.demes if any(t._is_cancer(g) for g in d)]
    assert len(occ) > 20                        # the tumour SPREAD across many demes (not one pile)
    assert np.mean(occ) <= 1.5 * K              # mean occupancy caps near K
    assert max(occ) <= 4 * K                    # no deme is a runaway pile (was 100s-1000s x K)


def test_well_mixed_disables_crowding():
    # carrying_capacity=None -> the well-mixed regime: no per-deme ceiling, so a single deme grows
    # unbounded (the role the old carrying_capacity=1 hack played; K=1 now caps at ~1 cell).
    cancer = {**LOW_DEATH, "dispersal_rate": 0.0, "mutation_rate": 0.0}
    t = GenotypeTumor(seed=1, genome_params=GENOME_PARAMS, selection_params=SELECTION_PARAMS,
                      cancer_cell_params=cancer, deme_params={"carrying_capacity": None},
                      spatial_params={"grid_size": 1, "structure_radius": 0},
                      update_mode="tau", tau=1.0)
    assert t._crowding is False
    # ~18 generations is plenty to blow past any finite per-deme K in one deme; 40 here grew to
    # ~4e7 cells and OOM'd when grow() materialised the cell x gene matrix (~13 GB).
    t.grow(n_steps=18, seed=1)
    # one deme, no ceiling -> far more than any finite K would allow
    assert t.get_cancer_size() > 500


def test_engines_agree_on_crowding_death():
    # The count engine's _death_rate and the cell engine's Deme.get_cancer_death_rate implement the
    # SAME density-dependent crowding formula (DESIGN_crowding.md), so they must return identical
    # crowding death for a matched (occupancy, K, division, death, margin) state.
    from iscc.tumor.components.deme import Deme
    from iscc.tumor.components.cell import CancerCell
    K, margin, maxd, div, death = 8, 0.1, 1.0, 0.6, 0.03

    t = GenotypeTumor(seed=1, genome_params=GENOME_PARAMS, selection_params=SELECTION_PARAMS,
                      cancer_cell_params={"division_rate": div, "death_rate": death,
                                          "max_birth_rate": 0.9},
                      deme_params={"carrying_capacity": K, "maximum_death_rate": maxd,
                                   "crowding_margin": margin, "initial_cancer_cells": 1},
                      spatial_params={"grid_size": 1, "structure_radius": 0})
    fid = t.founder_id
    # pin the founder's rates so both engines see identical inputs
    t.genotypes[fid].evolutionary_parameters["division_rate"] = div
    t.genotypes[fid].evolutionary_parameters["death_rate"] = death

    def cell_deme(n):
        first = CancerCell(n_segments=N_SEGMENTS, segment_size=SEGMENT_SIZE)
        first.evolutionary_parameters["division_rate"] = div
        first.evolutionary_parameters["death_rate"] = death
        d = Deme(cell=first, carrying_capacity=K, maximum_death_rate=maxd, crowding_margin=margin)
        for _ in range(n - 1):
            c = CancerCell(n_segments=N_SEGMENTS, segment_size=SEGMENT_SIZE)
            c.evolutionary_parameters["division_rate"] = div
            c.evolutionary_parameters["death_rate"] = death
            d.add_cell(c)
        return d

    for n in (1, 4, 8, 16, 24):
        d_count = t._death_rate(fid, 0, total=n)
        d_cell = cell_deme(n).get_cancer_death_rate(death, division_rate=div,
                                                    immune_cell_fraction=0.0, immune_resistance=0.0)
        assert d_count == pytest.approx(d_cell)
    # and the shared fixed point: death == division at occupancy K/(1+margin)
    assert t._death_rate(fid, 0, total=K / (1 + margin)) == pytest.approx(div)


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
