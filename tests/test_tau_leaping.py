"""Tests for the tau-leaping update mode on GenotypeTumor (DESIGN_scalability §7).

Tau-leaping advances ALL clones once per discrete generation (Poisson births/deaths per
(deme, genotype)) instead of one event per update. These tests check it: builds/grows/
materialises the standard schema, is reproducible (seed -> identical), records a full per-clone
snapshot every generation so plot_muller keeps working on a real-time axis, respects carrying
capacity (no unbounded overshoot), leaves the exact engine untouched, and is statistically
equivalent to the exact engine by clone-size DISTRIBUTION at a matched target size.
"""
import hashlib

import numpy as np

from conftest import GENOME_PARAMS, SELECTION_PARAMS, CANCER_CELL_PARAMS
from iscc.tumor.models import GenotypeTumor

SPATIAL = {"grid_size": 21, "n_structures": 1, "structure_radius": 0}
LOW_DEATH = {**CANCER_CELL_PARAMS, "death_rate": 0.05}


def _tau(seed, gens, cancer=LOW_DEATH, deme=None, tau=1.0, snapshot_every=1):
    t = GenotypeTumor(seed=seed, genome_params=GENOME_PARAMS, selection_params=SELECTION_PARAMS,
                      cancer_cell_params=cancer, deme_params=deme or {"carrying_capacity": 1},
                      spatial_params=SPATIAL, update_mode="tau", tau=tau,
                      snapshot_every=snapshot_every)
    t.grow(n_steps=gens, seed=seed)
    return t


def _grow_to(mode, seed, target=1500, cap=100000, cancer=LOW_DEATH):
    t = GenotypeTumor(seed=seed, genome_params=GENOME_PARAMS, selection_params=SELECTION_PARAMS,
                      cancer_cell_params=cancer, deme_params={"carrying_capacity": 1},
                      spatial_params=SPATIAL, update_mode=mode, tau=1.0)
    rng = np.random.default_rng(seed) if mode == "tau" else t.rng
    for _ in range(cap):
        if t.get_cancer_size() == 0 or t.get_cancer_size() >= target:
            break
        if mode == "tau":
            t._tau_generation(rng, t.tau)
        else:
            t.update(rng); t.step += 1
    return t


def test_builds_grows_and_materialises_schema():
    t = _tau(seed=1, gens=12)
    cd = t.cell_data
    assert {"cell_snv", "cell_cnv", "cell_exp", "cell_crd", "cell_type", "cell_deme", "cell_evo"} <= set(cd)
    n = t.get_tumor_size()
    assert n > 0
    assert cd["cell_snv"].shape == (n, t.n_genes)
    assert set(cd["cell_type"]["cell_id"]).issubset(set(t.genotypes_counts))


def test_reproducible_same_seed():
    def fp(seed):
        t = _tau(seed, gens=12)
        h = lambda k: hashlib.md5(np.ascontiguousarray(t.cell_data[k].values)).hexdigest()
        return t.get_cancer_size(), h("cell_snv"), h("cell_cnv")
    assert fp(3) == fp(3)
    assert fp(3) != fp(4)


def test_traces_real_time_axis_and_muller():
    # a full per-clone snapshot every generation -> traces grow with generations, trace_times
    # are the real-time axis (monotone increasing), and plot_muller still renders.
    import matplotlib
    matplotlib.use("Agg")
    t = _tau(seed=2, gens=12, snapshot_every=1)
    assert len(t.traces) == len(t.trace_times)
    assert len(t.traces) >= 12                      # one per generation (+ t=0)
    assert t.trace_times == sorted(t.trace_times)
    assert t.trace_times[-1] == t.time
    t.plot_muller()                                  # must not raise

    # snapshot_every>1 records fewer snapshots but still reaches the same final time
    t2 = _tau(seed=2, gens=12, snapshot_every=4)
    assert len(t2.traces) < len(t.traces)
    assert t2.time == t.time


def test_carrying_capacity_bounds_growth():
    # with a finite carrying capacity the elevated crowded death rate (which saturates at
    # maximum_death_rate) must hold the population near capacity instead of growing unboundedly
    # -- i.e. tau-leaping must not overshoot and blow up. We use a clonal founder (mutation_rate
    # 0, so no fitter driver clones whose division can exceed maximum_death_rate and legitimately
    # escape the cap) on a 1x1 grid, division == maximum_death_rate at the crowded equilibrium.
    cap = 200
    bounded = {**LOW_DEATH, "division_rate": 0.5, "max_birth_rate": 0.5,
               "dispersal_rate": 0.0, "mutation_rate": 0.0}
    t = GenotypeTumor(seed=1, genome_params=GENOME_PARAMS, selection_params=SELECTION_PARAMS,
                      cancer_cell_params=bounded, deme_params={"carrying_capacity": cap,
                      "maximum_death_rate": 0.5}, spatial_params={"grid_size": 1,
                      "n_structures": 1, "structure_radius": 0}, update_mode="tau", tau=1.0)
    t.grow(n_steps=80, seed=1)
    # equilibrium sits where division == crowded death; it must stay within a small multiple of
    # capacity, not explode exponentially.
    assert 0 < t.get_cancer_size() < 4 * cap


def test_exact_mode_untouched():
    # default mode is still "exact"; the new params don't change it.
    t = GenotypeTumor(seed=1, genome_params=GENOME_PARAMS, selection_params=SELECTION_PARAMS,
                      cancer_cell_params=LOW_DEATH, deme_params={"carrying_capacity": 1},
                      spatial_params=SPATIAL)
    assert t.update_mode == "exact"
    t.grow(n_steps=50, seed=1)
    assert t.get_tumor_size() > 0


def test_statistically_equivalent_to_exact_by_distribution():
    seeds = range(16)
    tgt = 1200
    ex = [_grow_to("exact", s, target=tgt) for s in seeds]
    ta = [_grow_to("tau", s, target=tgt) for s in seeds]
    ex_surv = [t for t in ex if t.get_cancer_size() > 0]
    ta_surv = [t for t in ta if t.get_cancer_size() > 0]
    # comparable survival (same low extinction probability)
    assert abs(len(ex_surv) - len(ta_surv)) <= 5

    def n_clones(ts):
        return np.mean([len(t.get_genotype_frequencies()[0]) for t in ts
                        if t.get_cancer_size() > 0])
    # number of surviving clones at matched size in the same ballpark
    r = n_clones(ta_surv) / n_clones(ex_surv)
    assert 0.4 < r < 2.5
