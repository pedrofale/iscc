"""Tests for the spatiotemporal record on GenotypeTumor: birth records and spatial snapshots.

Every genotype gets a birth record (trace index, step, time, deme) in every run, and
``spatial_snapshot_every > 0`` additionally stores every deme's genotype counts over time. Neither
draws a random number, so the tumour must grow identically with the record on or off; the record
must agree with the traces it sits beside; and the memory budget must thin it evenly.
"""
import os

import numpy as np
import pandas as pd

from conftest import GENOME_PARAMS, SELECTION_PARAMS, CANCER_CELL_PARAMS
from iscc.tumor.models import GenotypeTumor

SPATIAL = {"grid_size": 15, "n_structures": 1, "structure_radius": 0}
LOW_DEATH = {**CANCER_CELL_PARAMS, "death_rate": 0.05}
# A seeded micro-lesion (5 founder cells) so every seed diversifies within a few generations.
DEME = {"carrying_capacity": 20, "initial_cancer_cells": 5}


def _tumor(mode="tau", seed=4, spatial=SPATIAL, **kw):
    return GenotypeTumor(seed=seed, genome_params=GENOME_PARAMS, selection_params=SELECTION_PARAMS,
                         cancer_cell_params=LOW_DEATH, deme_params=DEME, spatial_params=spatial,
                         update_mode=mode, tau=1.0, **kw)


def _grown(mode="tau", n_steps=20, seed=4, **kw):
    t = _tumor(mode=mode, seed=seed, **kw)
    t.grow(n_steps=n_steps, seed=seed)
    return t


def _first_appearance(t):
    """Index of the first trace containing each genotype with a positive count."""
    first = {}
    for i, tr in enumerate(t.traces):
        for gid, c in tr["genotypes_counts"].items():
            if c > 0 and gid not in first:
                first[gid] = i
    return first


def test_births_cover_every_cancer_genotype_and_respect_ancestry():
    t = _grown()
    births = t.genotype_births()
    cancer = [g for g, rep in t.genotypes.items() if rep.type == "cancer"]
    assert len(cancer) > 5, "the fixture must actually diversify"
    assert list(births.index) == cancer                      # every one, in creation order
    founder = births.loc[t.founder_id]
    assert founder.birth_trace == 0 and founder.birth_step == 0 and founder.birth_time == 0.0
    assert founder.birth_deme in t._origin_demes
    for gid, row in births.iterrows():
        parent = row.parent
        if gid == t.founder_id or parent is None or parent not in births.index:
            continue
        # a child arises no earlier than its parent, in time and in creation order
        assert births.loc[parent, "birth_step"] <= row.birth_step
        assert births.loc[parent, "ord"] < row.ord
    placed = births[births.birth_deme >= 0]
    assert placed.birth_deme.between(0, len(t.demes) - 1).all()
    assert set(placed.compartment) == {"primary"}
    assert ((placed.row >= 0) & (placed.row < t.grid_size)).all()


def test_exact_engine_birth_trace_is_the_first_appearance():
    """The exact engine appends a trace after every event, so a genotype is first seen exactly at
    the trace its birth record names."""
    t = _grown(mode="exact", n_steps=1500)
    births = t.genotype_births()
    first = _first_appearance(t)
    seen = [g for g in births.index if g in first]
    assert len(seen) > 5
    for gid in seen:
        assert first[gid] == births.loc[gid, "birth_trace"], gid
    assert births.birth_time.isna().all()                    # the exact engine keeps no time


def test_tau_birth_trace_bounds_the_first_appearance():
    """Under tau-leaping a clone can arise and die within one generation, so it may never appear;
    when it does, it is no earlier than its birth trace."""
    t = _grown(n_steps=16)
    births = t.genotype_births()
    first = _first_appearance(t)
    for gid in births.index:
        if gid in first:
            assert first[gid] >= births.loc[gid, "birth_trace"]
    assert (births.birth_time == births.birth_step.astype(float)).all()   # tau = 1


def _by_ord(t):
    """Traces keyed by creation ordinal: genotype ids come from a process-wide counter, so two
    tumours built in one process get different ids even when they are identical."""
    return [{t.genotypes[g].ord: c for g, c in tr["genotypes_counts"].items()} for tr in t.traces]


def test_recording_does_not_change_growth():
    off = _grown(n_steps=14, seed=7)
    on = _grown(n_steps=14, seed=7, spatial_snapshot_every=1)
    assert off.spatial_history == [] and len(on.spatial_history) == len(on.traces)
    assert _by_ord(off) == _by_ord(on)
    for mat in ("cell_snv", "cell_cnv", "cell_crd"):
        np.testing.assert_array_equal(off.cell_data[mat].values, on.cell_data[mat].values)
    cols = ["ord", "birth_trace", "birth_step", "birth_time", "birth_deme"]
    pd.testing.assert_frame_equal(off.genotype_births()[cols].reset_index(drop=True),
                                  on.genotype_births()[cols].reset_index(drop=True))


def test_snapshots_agree_with_the_traces():
    t = _grown(n_steps=14, spatial_snapshot_every=3)
    snaps = t.spatial_snapshots()
    assert sorted(snaps.trace.unique()) == list(range(0, len(t.traces), 3))
    for ti, sub in snaps.groupby("trace"):
        totals = sub.groupby("genotype")["count"].sum().to_dict()
        expect = {g: c for g, c in t.traces[ti]["genotypes_counts"].items() if c > 0}
        assert totals == expect, ti
    assert (snaps["count"] > 0).all()
    # every cancer genotype in a snapshot has a birth record, so the two can be joined
    cancer = snaps[snaps.cell_type == "cancer"]
    assert set(cancer.genotype) <= set(t.genotype_births().index)
    # positions agree with the deme grid
    coords = np.asarray(t.deme_coords)
    assert (coords[snaps.deme.values, 0] == snaps.row.values).all()


def test_budget_thins_the_record_evenly():
    t = _grown(n_steps=20, spatial_snapshot_every=1, spatial_snapshot_budget=1500)
    assert t.spatial_snapshot_every > 1, "the budget must have bitten"
    kept = [s["trace"] for s in t.spatial_history]
    k = t.spatial_snapshot_every
    assert kept[0] == 0 and all(tr % k == 0 for tr in kept)
    assert len(kept) == 1 or t._spatial_entries <= t.spatial_snapshot_budget
    assert t._spatial_entries == sum(len(s["deme"]) for s in t.spatial_history)


def test_metastatic_deposit_is_recorded_in_its_own_grid():
    spatial = {**SPATIAL, "met_grid_size": 8, "K_met": 8, "host_fill_frac": 0.3,
               "met_seed_kappa": 0.3, "met_transit_floor": 1.0}
    t = _tumor(spatial=spatial, seed=5, spatial_snapshot_every=2)
    t.grow(n_steps=20, seed=5)
    snaps = t.spatial_snapshots()
    met = snaps[snaps.compartment == "met"]
    assert not met.empty and set(met.cell_type) >= {"host"}
    assert met.row.between(0, 7).all() and met.col.between(0, 7).all()
    assert (met.deme >= t.n_primary_demes).all()
    births = t.genotype_births()
    assert set(births.compartment.dropna()) <= {"primary", "met"}


def test_write_emits_the_record(tmp_path):
    on = _grown(n_steps=8, spatial_snapshot_every=2)
    on.write(str(tmp_path / "on"))
    files = set(os.listdir(tmp_path / "on"))
    assert {"genotype_births.csv", "spatial_history.csv.gz"} <= files
    back = pd.read_csv(tmp_path / "on" / "spatial_history.csv.gz")
    assert len(back) == len(on.spatial_snapshots())
    off = _grown(n_steps=8)
    off.write(str(tmp_path / "off"))
    files = set(os.listdir(tmp_path / "off"))
    assert "genotype_births.csv" in files and "spatial_history.csv.gz" not in files
