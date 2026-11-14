"""Tau-leaping with critical reactions (``tau_step="cgp_critical"``; Cao, Gillespie and Petzold 2005).

Under the two-valued crowding step (d0 up to K, d1 above), plain tau-leaping lets a full deme overshoot
K and then lose several cells where the exact process loses one. Firing the channels of demes at or
above K one event at a time, as the critical reactions of Cao, Gillespie and Petzold (2005), must hold
a single deme at the exact engine's stationary mean (30.08 at K = 30) and never drive a count negative.
"""
import numpy as np
import pytest

from iscc.tumor.models.count import GenotypeTumor

GENOME = dict(n_segments=2, segment_size=50)
SEL = dict(prop_driver=0.0, prop_dispersal=0.0, prop_immune_resistance=0.0, prop_met_survival=0.0,
           prop_treatment_resistance=0.0, prop_breach=0.0, prop_stromal_survival=0.0)
B, D0, D1 = 0.7, 0.05, 5.0


def build(K=30, n0=30, step="cgp_critical", seed=1, grid=1, disp=0.0, mode="tau"):
    cancer = dict(division_rate=B, death_rate=D0, max_birth_rate=2.0, mutation_rate=0.0,
                  n_snvs_per_allele=0.1, dispersal_rate=disp, cnv_prob=0.0, snv_prob=1.0,
                  wgd_rate=0.0)
    deme = dict(carrying_capacity=K, initial_cancer_cells=n0, maximum_death_rate=D1,
                crowding_overfill=1.5, crowding_d1=D1, tau_deaths="binomial", tau_step=step)
    return GenotypeTumor(seed=seed, genome_params=GENOME, selection_params=SEL,
                         cancer_cell_params=cancer, deme_params=deme,
                         spatial_params=dict(grid_size=grid), update_mode=mode, tau=0.5,
                         snapshot_every=10 ** 9, coarsen_passengers=True)


def _state(T):
    """Every deme's counts keyed by creation ordinal (genotype ids are process-global)."""
    return [sorted((int(T.genotypes[g].ord), int(c)) for g, c in d.items()) for d in T.demes]


def _no_negative(T):
    return all(c > 0 for d in T.demes for c in d.values())


def test_single_deme_holds_K():
    """A single deme at K = 30, time-averaged over 8 seeds x 100 generations after a 20-generation
    burn-in: within 3% of the exact stationary mean (30.08), never negative, never past the slot cap."""
    means = []
    for seed in range(8):
        T = build(seed=seed)
        T.grow(n_steps=20, seed=seed, materialize=False)
        acc = tot = 0.0
        orig = T._tau_substep

        def rec(rng, dt, demes=None, _T=T):
            nonlocal acc, tot
            acc += _T.get_cancer_size() * dt
            tot += dt
            return orig(rng, dt, demes=demes)
        T._tau_substep = rec
        for _ in range(100):
            T._tau_generation(np.random.default_rng([seed, T.step]), T.tau)
            assert _no_negative(T)
            assert T.get_cancer_size() <= 45
        means.append(acc / tot)
    assert abs(np.mean(means) / 30.08 - 1) < 0.03, means


def test_spatial_counts_stay_nonnegative():
    """A small dispersing tumour on a 6 x 6 grid: critical and leaped demes side by side."""
    T = build(K=12, n0=3, grid=6, disp=0.3, seed=4)
    for _ in range(60):
        T._tau_generation(np.random.default_rng([4, T.step]), T.tau)
        assert _no_negative(T)
        for di, d in enumerate(T.demes):
            assert sum(d.values()) <= T._slot_cap_of(di)
    assert T.get_cancer_size() > 0
    assert T._n_crit_events > 0 and T._n_substeps > T._n_crit_events


def test_needs_a_carrying_capacity():
    with pytest.raises(ValueError, match="cgp_critical"):
        GenotypeTumor(seed=1, genome_params=GENOME, selection_params=SEL,
                      deme_params=dict(carrying_capacity=0, initial_cancer_cells=1,
                                       tau_step="cgp_critical"),
                      spatial_params=dict(grid_size=1), update_mode="tau")
