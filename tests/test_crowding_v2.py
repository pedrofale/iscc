"""Tests for the structural-cap crowding law, ``crowding_mode="lottery"``.

The ``"own"`` law scales each clone's crowding death by that clone's OWN evolved division rate, so
the fitness factor cancels out of every statement about equilibrium and relative advantage: a full
deme is a neutral community in which no clone can displace another. ``"lottery"`` splits the two jobs
that term was doing:

* the crowding **death** is demon's two-valued step (d0 at or below K, d1 above it), the same for
  every clone in a deme, so the fitness term cannot cancel;
* the density cap is STRUCTURAL: a deme has ``floor(crowding_overfill * K)`` slots, a birth that
  cannot get one fails, and contested slots go to clones in proportion to the births they drew.

These tests check the step, that the slot cap is never exceeded in either engine, that the lottery
produces competitive exclusion where ``"own"`` produces drift, and that the exact and tau-leaping
paths agree in distribution.
"""
import time

import numpy as np
import pytest

from conftest import GENOME_PARAMS, SELECTION_PARAMS, CANCER_CELL_PARAMS
from iscc.tumor.models import GenotypeTumor

SPATIAL = {"grid_size": 15, "n_structures": 1, "structure_radius": 0}
LOW_DEATH = {**CANCER_CELL_PARAMS, "death_rate": 0.05}


# The paper's lottery settings: slot cap 1.5 K, d1 = 5 above K.
LOTTERY = {"crowding_law": "noble", "crowding_overfill": 1.5, "crowding_d1": 5.0,
           "maximum_death_rate": 5.0}


def _tumour(mode="lottery", seed=1, K=12, cancer=None, deme=None, spatial=None, **kw):
    dp = {"carrying_capacity": K, "initial_cancer_cells": 5, "crowding_mode": mode,
          **(LOTTERY if mode == "lottery" else {})}
    dp.update(deme or {})
    return GenotypeTumor(seed=seed, genome_params=GENOME_PARAMS, selection_params=SELECTION_PARAMS,
                         cancer_cell_params={**LOW_DEATH, **(cancer or {})}, deme_params=dp,
                         spatial_params={**SPATIAL, **(spatial or {})}, **kw)


def _occupancy(t):
    """(cells, K) per non-empty deme."""
    return [(sum(d.values()), t._cap_of(i)) for i, d in enumerate(t.demes) if d]


def _assert_cap(t):
    over = [(i, n, t._slot_cap_of(i)) for i, d in enumerate(t.demes)
            if (n := sum(d.values())) > t._slot_cap_of(i)]
    assert not over, f"structural cap breached in {len(over)} demes, e.g. {over[:3]}"


# --------------------------------------------------------------------------- the default is untouched
def test_default_mode_is_own_and_lottery_is_off():
    # THE gate on this prototype: nothing changes unless the flag is set explicitly, so every
    # existing golden hash, figure and published number stands.
    t = _tumour(mode=None, deme={"crowding_mode": "own"})
    assert t._crowding_mode == "own"
    plain = GenotypeTumor(seed=1, genome_params=GENOME_PARAMS, selection_params=SELECTION_PARAMS,
                          cancer_cell_params=LOW_DEATH,
                          deme_params={"carrying_capacity": 12}, spatial_params=SPATIAL)
    assert plain._crowding_mode == "own"
    assert plain._lottery is False


def test_own_mode_death_rate_is_byte_identical_to_the_shipped_formula():
    # The lottery branch must not perturb the default law by so much as a float: reproduce the
    # shipped Option A expression independently and compare exactly.
    t = _tumour(mode="own", K=10)
    gid = t.founder_id
    ep = t.genotypes[gid].evolutionary_parameters
    steep = 1.0 + t.crowding_margin
    for n in (1, 5, 10, 25):
        t.demes[0].clear()
        t.genotypes_counts[gid] = n
        t.demes[0][gid] = n
        expected = min(ep["death_rate"]
                       + max(0.0, ep["division_rate"] - ep["death_rate"]) * steep * (n / 10),
                       t.maximum_death_rate)
        assert t._death_rate(gid, 0) == expected


def test_well_mixed_regime_ignores_the_flag():
    # carrying_capacity None/0 -> no capacity -> no slots. The single-deme scalability benchmark
    # must stay unbounded whatever crowding_mode says.
    t = _tumour(mode="lottery", deme={"carrying_capacity": None},
                cancer={"dispersal_rate": 0.0, "mutation_rate": 0.0},
                spatial={"grid_size": 1}, update_mode="tau", tau=1.0)
    assert t._crowding is False and t._lottery is False
    t.grow(n_steps=15, seed=1)
    assert t.get_cancer_size() > 500


def test_own_mode_selection_differential_vanishes_at_the_cap():
    # The defect this change exists to fix, asserted directly so the contrast is on the record.
    K = 10
    t = _tumour(mode="own", K=K)
    fast = t.founder_id
    rep = t.genotypes[fast].divide()
    rep.genotype_id = "slow"
    rep.evolutionary_parameters = dict(rep.evolutionary_parameters)
    rep.evolutionary_parameters["division_rate"] = 0.25
    t._register(rep)
    t.demes[0].clear()
    t.genotypes_counts.clear()
    n_fixed = int(round(K / (1.0 + t.crowding_margin)))      # the occupancy where net growth is zero
    t._add(0, fast, n_fixed - 4)
    t._add(0, "slow", 4)
    net_fast = t.genotypes[fast].evolutionary_parameters["division_rate"] - t._death_rate(fast, 0)
    net_slow = 0.25 - t._death_rate("slow", 0)
    assert abs(net_fast - net_slow) < 0.02        # ~0: a full deme is a neutral community


# --------------------------------------------------------------------------- the cap
@pytest.mark.parametrize("update_mode,steps", [("gillespie", 4000), ("tau", 25)])
def test_cap_is_never_exceeded(update_mode, steps):
    K = 12
    t = _tumour(mode="lottery", K=K, seed=4, update_mode=update_mode, tau=1.0,
                cancer={"division_rate": 0.9, "max_birth_rate": 0.95, "dispersal_rate": 0.4})
    rng = np.random.default_rng(4)
    if update_mode == "tau":
        for _ in range(steps):
            t._tau_generation(rng, 1.0)
            _assert_cap(t)
    else:
        for _ in range(steps):
            t.update(rng)
            _assert_cap(t)             # after EVERY event, not just at the end
    assert t.get_cancer_size() > 100   # it actually grew rather than trivially satisfying the cap
    assert max(n for n, _ in _occupancy(t)) > K      # and demes fill past K, into the d1 step


def test_structural_cap_holds_with_immortal_residents_and_eviction_opens_the_wall():
    # A duct WALL deme is seeded FULL of immortal epithelium at K_duct. Cancer enters it only by
    # displacing residents, so the cap must hold and wall demes must still receive cancer.
    spatial = {"grid_size": 21, "structure_radius": 3, "gland_radius": 3, "n_glands": 1,
               "K_duct": 12, "K_stroma": 8, "stroma_fill_frac": 0.3}
    t = _tumour(mode="lottery", K=8, seed=2, update_mode="tau", tau=1.0,
                deme={"initial_cancer_cells": 6}, spatial=spatial, cancer={"dispersal_rate": 0.6})
    t.grow(n_steps=30, seed=2)
    _assert_cap(t)
    wall = [i for i in range(len(t.demes))
            if t.gland_id[i] >= 0 and t.demes[i].get("epithelial", 0)]
    assert sum(sum(c for g, c in t.demes[i].items() if t._is_cancer(g)) for i in wall) > 0


def test_immune_cells_hold_slots_but_are_not_evictable():
    t = _tumour(mode="lottery", K=10, spatial={"immune_density": 0.4}, update_mode="tau", tau=1.0)
    assert t._n_evictable(0) == 0                       # immune only -> nothing displaceable
    t.grow(n_steps=25, seed=1)
    _assert_cap(t)
    # immune cells are static and never displaced, so their count is exactly what was seeded
    assert all(d.get("immune", 0) == 4 for d in t.demes if "immune" in d)


# --------------------------------------------------------------------------- the lottery competes
def _two_clone_deme(mode, seed, b_fast=0.9, b_slow=0.45, K=200, gens=30):
    """One isolated full deme seeded 50/50 with a fast and a 2x-slower clone; returns the fast
    clone's final share and the deme's final occupancy as a fraction of K. No mutation, no dispersal
    — pure within-deme competition. K is large enough that neutral drift cannot fix anything in
    ``gens`` turnovers, so a clone that fixes did so by selection."""
    t = _tumour(mode=mode, K=K, seed=seed, update_mode="tau", tau=1.0,
                cancer={"division_rate": b_fast, "death_rate": 0.05, "max_birth_rate": 0.95,
                        "mutation_rate": 0.0, "dispersal_rate": 0.0},
                deme={"initial_cancer_cells": 1}, spatial={"grid_size": 1})
    fast = t.founder_id
    slow = t.genotypes[fast].divide()
    slow.genotype_id = "slow"
    slow.evolutionary_parameters = dict(slow.evolutionary_parameters)
    slow.evolutionary_parameters["division_rate"] = b_slow
    t._register(slow)
    t.demes[0].clear()
    t.genotypes_counts.clear()
    t._add(0, fast, K // 2)
    t._add(0, "slow", K // 2)
    t.deme_rates = np.array([t._deme_rate(0)], dtype=float)
    rng = np.random.default_rng(seed)
    for _ in range(gens):
        t._tau_generation(rng, 1.0)
    n_fast = t.demes[0].get(fast, 0)
    n_slow = t.demes[0].get("slow", 0)
    n = n_fast + n_slow
    return (n_fast / n if n else float("nan"), n / K)


def test_lottery_produces_competitive_exclusion_where_own_produces_drift():
    # The point of the whole change: inside a FULL deme, a 2x-fitter clone must take over. Under
    # "own" both clones have net growth ~0 at the fixed point, so the composition only random-walks
    # — including AGAINST the fitter clone.
    seeds = (1, 2, 3, 4, 5, 6, 7, 8)
    res = {m: np.array([_two_clone_deme(m, s) for s in seeds]) for m in ("own", "lottery")}
    share = {m: v[:, 0] for m, v in res.items()}
    occ = {m: v[:, 1] for m, v in res.items()}
    assert np.nanmean(share["lottery"]) > 0.95, share["lottery"]      # the fitter clone fixes
    assert share["lottery"].min() > 0.9, share["lottery"]             # in EVERY seed
    assert np.nanmean(share["own"]) < 0.85, share["own"]
    assert share["own"].std() > 0.15, share["own"]                    # a random walk, not a sweep
    assert share["own"].min() < 0.5, share["own"]                     # sometimes the SLOWER clone wins
    # and the deme stays near K and under the slot cap with the lottery (tau-leaping holds a full deme
    # about 10% below K: the d1 deaths of a substep overshoot the excess, a known bias of the
    # integrator), loosely regulated under "own"
    assert 0.85 < occ["lottery"].mean() <= 1.05, occ["lottery"]
    assert occ["lottery"].max() <= 1.5, occ["lottery"]
    assert occ["own"].max() > 1.0 and occ["own"].min() < 0.9


def test_rejected_mutation_births_register_no_genotype():
    # Allocate first, mutate second: a mutation-branch birth that loses the lottery must never reach
    # mutate(), or the registry fills with genotypes that have no cells.
    for mode in ("own", "lottery"):
        t = _tumour(mode=mode, K=10, seed=3, update_mode="tau", tau=1.0,
                    cancer={"mutation_rate": 1.0, "dispersal_rate": 0.0},
                    deme={"initial_cancer_cells": 5}, spatial={"grid_size": 1})
        t.grow(n_steps=30, seed=3)
        live = {g for g, c in t.genotypes_counts.items() if c > 0}
        assert set(t.genotypes_counts) == live       # no zero-count entries either way
        if mode == "lottery":
            _assert_cap(t)


# --------------------------------------------------------------------------- engine agreement
def test_exact_and_tau_agree_in_distribution():
    """The tau allocation is the exact engine's CONDITIONAL law, not an expectation match: the
    births in an interval are a superposition of Poisson processes, so conditional on their counts
    their order is a uniformly random permutation, the exact rule accepts the first S of that order,
    and the composition of the first S items of a random permutation of a multiset IS the
    multivariate hypergeometric. The two engines must therefore agree on occupancy and on the
    selection response at matched parameters."""
    K, gens, seeds = 24, 30, range(12)

    def run(update_mode, seed):
        t = _tumour(mode="lottery", K=K, seed=seed, update_mode=update_mode, tau=1.0,
                    cancer={"division_rate": 0.7, "death_rate": 0.05, "max_birth_rate": 0.95,
                            "mutation_rate": 0.2, "dispersal_rate": 0.4},
                    deme={"initial_cancer_cells": 12}, spatial={"grid_size": 9})
        rng = np.random.default_rng(seed)
        if update_mode == "tau":
            for _ in range(gens):
                t._tau_generation(rng, 1.0)
        else:
            # match the biological time: the exact engine fires one event per update, so run until
            # the population stops changing size systematically (same generations x cells x rate).
            for _ in range(gens * 1400):
                t.update(rng)
        _assert_cap(t)
        interior = [n for i, (n, _) in enumerate(_occupancy(t))
                    if len(t._neighbors(i)) == 4 and all(t.demes[j] for j in t._neighbors(i))]
        mean_div = (sum(c * t.genotypes[g].evolutionary_parameters["division_rate"]
                        for g, c in t.genotypes_counts.items() if t._is_cancer(g))
                    / max(1, t.get_cancer_size()))
        return (np.mean(interior) / K if interior else float("nan"), mean_div)

    tau = np.array([run("tau", s) for s in seeds])
    exact = np.array([run("gillespie", s) for s in seeds])
    # Mean occupancy of the packed interior, as a fraction of K. The exact engine holds a full deme at
    # K. Tau-leaping holds it lower: a substep's d1 deaths are drawn from the pre-step count and so
    # overshoot the excess above K. That is a known bias of the integrator (about 8% here, 12% in a
    # single paper-regime deme), pinned here so a change to it is noticed.
    assert abs(np.nanmean(exact[:, 0]) - 1.0) < 0.03, exact[:, 0]
    assert 0.04 < np.nanmean(exact[:, 0]) - np.nanmean(tau[:, 0]) < 0.15, (tau[:, 0], exact[:, 0])
    # selection response: cell-weighted mean division rate after the same amount of evolution
    assert abs(np.nanmean(tau[:, 1]) - np.nanmean(exact[:, 1])) < 0.06, (tau[:, 1], exact[:, 1])


def test_growth_cost_is_bounded():
    # Births into full demes are drawn and discarded, so the engine does more futile work; against
    # that, rejected mutation-branch births register no genotypes and tau's cost scales with
    # #genotypes x #demes. Guard the net, loosely (this is a wall-clock check, not a benchmark).
    walls = {}
    for mode in ("own", "lottery"):
        t = _tumour(mode=mode, K=20, seed=8, update_mode="tau", tau=1.0,
                    cancer={"dispersal_rate": 0.4}, deme={"initial_cancer_cells": 10},
                    spatial={"grid_size": 17})
        t0 = time.time()
        rng = np.random.default_rng(8)
        while t.get_cancer_size() < 3000 and time.time() - t0 < 60:
            t._tau_generation(rng, 1.0)
        walls[mode] = time.time() - t0
    assert walls["lottery"] < 3.0 * walls["own"], walls


# --------------------------------------------------- the deme-model (Noble) crowding law
def test_noble_law_is_a_two_valued_step_at_the_carrying_capacity():
    """Noble et al. 2022 ("Within-deme dynamics"): the within-deme death rate takes a fixed value d0
    at or below the carrying capacity and a different fixed value d1 above it. NOT a continuous
    function of density — a deme below capacity is not crowded and carries no crowding death."""
    K = 10
    t = _tumour(mode="lottery", K=K,
                deme={"crowding_law": "noble", "crowding_d1": 3.0, "crowding_overfill": 2.0,
                      "maximum_death_rate": 3.5})       # or the clamp clips d1 back
    g = t.founder_id
    base = t.genotypes[g].evolutionary_parameters["death_rate"]

    def death_at(n):
        t.demes[0].clear(); t.genotypes_counts.clear(); t._add(0, g, n)
        return t._death_rate(g, 0)

    for n in (1, 3, 7, K):                       # at or below K -> d0, flat
        assert death_at(n) == pytest.approx(base), f"density-dependent death below K at n={n}"
    for n in (K + 1, K + 4):                     # above K -> d1, flat
        assert death_at(n) == pytest.approx(3.0)


def test_noble_law_rejects_a_d1_below_the_attainable_division_rate():
    """d1 must exceed the largest attainable division rate — that is what makes the cap unbreakable
    however far selection pushes birth rates, so it is checked at construction."""
    with pytest.raises(ValueError, match="largest attainable division rate"):
        _tumour(mode="lottery", K=10, deme={"crowding_d1": 0.1})


def test_noble_law_preserves_the_selection_differential_at_the_cap():
    """The step carries no `div`, so unlike `own` it cannot cancel out of net = div - death: two
    clones in the SAME full deme keep their bare fitness difference."""
    K = 10
    t = _tumour(mode="lottery", K=K,
                deme={"crowding_law": "noble", "crowding_d1": 3.0, "crowding_overfill": 2.0,
                      "maximum_death_rate": 3.5})
    fast = t.genotypes[t.founder_id].divide(); fast.genotype_id = "fast"
    fast.evolutionary_parameters = dict(fast.evolutionary_parameters)
    fast.evolutionary_parameters["division_rate"] = 0.9
    slow = t.genotypes[t.founder_id].divide(); slow.genotype_id = "slow"
    slow.evolutionary_parameters = dict(slow.evolutionary_parameters)
    slow.evolutionary_parameters["division_rate"] = 0.5
    t._register(fast); t._register(slow)
    t.demes[0].clear(); t.genotypes_counts.clear()
    t._add(0, "fast", K // 2); t._add(0, "slow", K // 2)          # deme exactly at K
    net_fast = 0.9 - t._death_rate("fast", 0)
    net_slow = 0.5 - t._death_rate("slow", 0)
    assert net_fast - net_slow == pytest.approx(0.4)              # the bare difference, undiminished


def test_noble_law_rejects_a_d1_clamped_below_the_division_rate_by_maximum_death_rate():
    """A d1 that LOOKS high enough but is clipped back by `maximum_death_rate` cannot bound a deme:
    the check must use the effective (post-clamp) value, not the request."""
    with pytest.raises(ValueError, match="EFFECTIVE d1"):
        _tumour(mode="lottery", K=10, deme={"crowding_d1": 99.0, "maximum_death_rate": 0.5})


def test_lottery_runs_only_with_the_noble_law():
    with pytest.raises(ValueError, match="crowding_law='noble'"):
        _tumour(mode="lottery", deme={"crowding_law": "logistic"})
