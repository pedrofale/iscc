"""``grow(materialize=False)`` steps the tumour without rebuilding ``cell_data``.

A caller that steps a tumour one generation at a time only needs the per-cell tables when it samples,
so ``grow(materialize=False)`` skips ``make_cell_data`` and marks the tables stale, and
``ensure_cell_data`` rebuilds them on demand. The guarantee is that skipping changes nothing: every
generator ``make_cell_data`` uses is seeded on its own (the subsample plan, the depth cut, the
passenger overlay, the program activity), and nothing it writes is read by the dynamics. The config
here switches on every layer that draws or writes at materialisation (a ``max_cells`` cap below the
tumour size, coarsened passengers, germline and founder variants, hypoxia, CCI and the program
layer), so a shared stream anywhere in it would show up as a diverging trajectory.
"""
import numpy as np
import pandas as pd
import pytest

from iscc.sample import Resection
from iscc.tumor.components.cell import Cell
from iscc.tumor.models import GenotypeTumor

GENOME = {"n_segments": 6, "segment_size": 40}
SELECTION = {"prop_driver": 0.2, "prop_dispersal": 0.1}
CANCER = {"division_rate": 1.0, "death_rate": 0.02, "mutation_rate": 1.0, "dispersal_rate": 0.3}
DEME = {"carrying_capacity": 20, "initial_cancer_cells": 10}
SPATIAL = {"grid_size": 12, "structure_radius": 3}
MICROENV = {
    "hypoxia": {"strength": 0.8, "n_genes": 10, "o2_source": "perfused", "o2_supply": 0.3},
    "cci": {"strength": 0.6, "n_target_genes": 10, "emitter_type": "cancer", "n_candidate_pairs": 4},
}
EXPRESSION = {
    "program_params": {"n_programs": 4, "n_genes_per_program": 10},
    "activity_params": {"n_active_programs_per_cell": 2, "activity_noise": 0.2},
    "dosage_params": {"dosage_sensitivity_mean": 0.7, "dosage_sensitivity_sd": 0.25,
                      "allele_specific": True},
}
MAX_CELLS = 400
N_CALLS = 30
# One generation per call under tau-leaping. The exact engine runs n_steps - 1 single events per call
# (none at n_steps=1), so it takes 40 per call.
CASES = [("tau", 1), ("exact", 40)]


def _build(mode):
    return GenotypeTumor(seed=7, genome_params=GENOME, selection_params=SELECTION,
                         cancer_cell_params=CANCER, deme_params=DEME, spatial_params=SPATIAL,
                         update_mode=mode, tau=0.5, coarsen_passengers=True, max_cells=MAX_CELLS,
                         microenv_params=MICROENV, expression_params=EXPRESSION,
                         founder_mutations={"n_passenger": 6, "n_driver": 1},
                         germline_params={"het_frac": 0.05, "hom_frac": 0.3})


def _state(t):
    return dict(counts=dict(t.genotypes_counts), demes=[dict(d) for d in t.demes],
                traces=list(t.traces), trace_times=list(getattr(t, "trace_times", [])),
                parents=dict(t.genotypes_parents), step=t.step)


def _run(mode, n_steps, start, **grow_kw):
    # Genotype ids come from a process-wide counter, so each run starts it at the same value and the
    # two runs go one after the other: their ids, and so every table, can then be compared as is.
    Cell._genotype_counter = start
    t = _build(mode)
    states = []
    for _ in range(N_CALLS):
        t.grow(n_steps=n_steps, seed=7, **grow_kw)
        states.append(_state(t))
    return t, states


@pytest.mark.parametrize("mode,n_steps", CASES)
def test_skipping_materialisation_leaves_the_trajectory_unchanged(mode, n_steps):
    start = Cell._genotype_counter
    a, states_a = _run(mode, n_steps, start)
    b, states_b = _run(mode, n_steps, start, materialize=False)
    for k, (sa, sb) in enumerate(zip(states_a, states_b)):
        assert sa == sb, f"diverged at call {k}"
    assert a.get_cancer_size() > 50 and len(a.genotypes) > 50   # a growing, branching tumour
    assert a.get_tumor_size() > MAX_CELLS         # the cap subsampled, so the plan rng was drawn

    b.ensure_cell_data()
    assert set(a.cell_data) == set(b.cell_data)
    for key in a.cell_data:
        pd.testing.assert_frame_equal(a.cell_data[key], b.cell_data[key], check_exact=True)
    assert {"cell_snv_p", "cell_microenv", "cell_program"} <= set(a.cell_data)


def test_stale_tables_are_flagged_and_rebuilt_on_demand():
    t = _build("tau")
    t.grow(n_steps=6, seed=7)
    current = t.cell_data
    assert t.ensure_cell_data() is current        # current tables are returned, not rebuilt

    t.grow(n_steps=1, seed=7, materialize=False)
    assert t.cell_data is current and t._cell_data_stale
    with pytest.raises(ValueError, match="stale"):
        t.truncal_sites()

    rebuilt = t.ensure_cell_data(max_cells=50)
    assert rebuilt is not current and not t._cell_data_stale
    assert len(rebuilt["cell_type"]) <= 60


def test_resection_keeps_the_stale_flag():
    t = _build("tau")
    t.grow(n_steps=6, seed=7)
    old = t.cell_data
    t.grow(n_steps=1, seed=7, materialize=False)
    sample = Resection(t).dissociate(max_cells=10_000)
    assert len(sample["cell_type"]) > 0
    assert t.cell_data is old and t._cell_data_stale


def test_default_grow_never_marks_stale():
    t = _build("tau")
    t.grow(n_steps=3, seed=7)
    assert not t._cell_data_stale
    assert np.isfinite(t.cell_data["cell_exp"].values).all()
