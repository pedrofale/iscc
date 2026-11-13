"""Three opt-in engine options: demon's division law, independent random streams, and a confinement
clock that counts generations.

* ``deme_params["dispersal_law"] = "noble"``: demon's invasion rule (Noble et al. 2022). A division
  attempts a mutation with probability ``mutation_rate`` and, independently, one daughter disperses
  with probability ``dispersal_rate``. The default, ``"compete"``, makes each division either mutate
  or disperse, so raising dispersal lowers the mutation supply.
* ``rng_streams = "independent"``: per-run streams keyed by (seed, purpose, step). The default,
  ``"offset"``, adds the step to the seed, so seed s at step k is seed s + 1 at step k - 1.
* ``origin_confinement["clock"] = "generations"``: the tau engine's confinement clock ticks by
  ``tau``, so ``generations`` means generations. The default, ``"steps"``, ticks once per tau step.

Each default must leave growth byte-identical: the digests below were recorded on ``dev`` at
``cb2ae02``, before the options existed.
"""
import hashlib
import json

import numpy as np
import pytest

from iscc.tumor.components.cell import CancerCell
from iscc.tumor.models.count import GenotypeTumor, _STREAM_GROW

GENOME = dict(n_segments=8, segment_size=120)
SELECTION = dict(prop_driver=0.04, prop_dispersal=0.0, prop_immune_resistance=0.02,
                 prop_met_survival=0.0, prop_treatment_resistance=0.0,
                 prop_breach=0.02, breach_effects=2.8,
                 prop_stromal_survival=0.02, stromal_survival_effects=2.2,
                 breach_cost=0.6, stromal_survival_cost=0.6, max_mut_drivers=12,
                 trait_source="mutation")
CANCER = dict(division_rate=0.7, death_rate=0.05, max_birth_rate=2.0, mutation_rate=0.3,
              n_snvs_per_allele=0.1, dispersal_rate=0.9, cnv_prob=0.08, wgd_rate=0.001)
DEMES = dict(carrying_capacity=14, initial_cancer_cells=5, resident_pressure_ref=0.2,
             maximum_death_rate=2.2)
CONFINE = dict(dispersal_factor=0.0, established_at=0.5, generations=30)
SPATIAL = dict(grid_size=20, n_glands=3, structure_radius=4, gland_radius=4, min_gland_sep=9,
               K_duct=14, K_stroma=14, stroma_fill_frac=0.3, cross_gland_kappa=0.06,
               cross_gland_lambda=30, breach_gated_invasion=True, epithelial_barrier=0.0,
               stromal_hazard=0.6, origin_confinement=CONFINE)
LOTTERY = dict(crowding_mode="lottery", crowding_law="noble")
NOBLE = dict(dispersal_law="noble")


def build(mode="tau", deme=None, cancer=None, spatial=None, seed=2, **kw):
    return GenotypeTumor(seed=seed, genome_params=GENOME, selection_params=SELECTION,
                         cancer_cell_params={**CANCER, **(cancer or {})},
                         deme_params={**DEMES, **(deme or {})},
                         spatial_params={**SPATIAL, **(spatial or {})}, update_mode=mode, tau=0.5,
                         snapshot_every=10 ** 9, coarsen_passengers=True, **kw)


def digest(T):
    """Every deme's genotype counts, keyed by creation ordinal (genotype ids are process-global)."""
    parts = [(i, sorted((int(T.genotypes[g].ord), int(c)) for g, c in d.items()))
             for i, d in enumerate(T.demes) if d]
    return hashlib.sha256(json.dumps([T.step, parts], sort_keys=True).encode()).hexdigest()[:16]


# Recorded on dev at cb2ae02, before these options existed.
PINNED = {
    ("tau", False): ("e23a4db7c339a3ee", 60),
    ("tau", True): ("d533a160b067e021", 60),
    ("exact", False): ("5dc832ca9d09ce24", 4000),
    ("exact", True): ("2b974a2bb2ab52fb", 4000),
}


# --------------------------------------------------------------------------- defaults
@pytest.mark.parametrize("mode,lottery", list(PINNED))
def test_defaults_are_byte_identical(mode, lottery):
    want, steps = PINNED[(mode, lottery)]
    T = build(mode, LOTTERY if lottery else None)
    T.grow(steps, materialize=False)
    assert digest(T) == want


@pytest.mark.parametrize("mode,lottery", list(PINNED))
def test_explicit_defaults_match_unset(mode, lottery):
    want, steps = PINNED[(mode, lottery)]
    T = build(mode, {**(LOTTERY if lottery else {}), "dispersal_law": "compete"},
              spatial=dict(origin_confinement={**CONFINE, "clock": "steps"}), rng_streams="offset")
    T.grow(steps, materialize=False)
    assert digest(T) == want


# --------------------------------------------------------------------------- dispersal_law
def _mutation_share(monkeypatch, mode, disp, n_steps):
    """Share of divisions that attempt a mutation, counted on the noble law's division paths."""
    calls = {"div": 0, "mut": 0}
    real_mutate = CancerCell.mutate

    def counting_mutate(self, *a, **k):
        calls["mut"] += 1
        return real_mutate(self, *a, **k)

    monkeypatch.setattr(CancerCell, "mutate", counting_mutate)
    T = build(mode, NOBLE, cancer=dict(dispersal_rate=disp), spatial=dict(origin_confinement=None))
    if mode == "exact":
        real_divide = T._divide_noble

        def counting_divide(*a, **k):
            calls["div"] += 1
            return real_divide(*a, **k)

        T._divide_noble = counting_divide
    else:
        real_split = T._split_births_noble

        def counting_split(rng, di, gid, rep, n, *a, **k):
            calls["div"] += n
            return real_split(rng, di, gid, rep, n, *a, **k)

        T._split_births_noble = counting_split
    T.grow(n_steps, materialize=False)
    return calls["mut"] / calls["div"]


@pytest.mark.parametrize("mode,n_steps", [("exact", 6000), ("tau", 40)])
def test_noble_mutation_supply_does_not_depend_on_dispersal(monkeypatch, mode, n_steps):
    low = _mutation_share(monkeypatch, mode, 0.1, n_steps)
    high = _mutation_share(monkeypatch, mode, 0.9, n_steps)
    # mutation_rate is 0.3; under the default law the shares would be 0.75 and 0.25.
    assert low == pytest.approx(0.3, abs=0.05)
    assert high == pytest.approx(0.3, abs=0.05)


def test_noble_split_matches_the_rule():
    """Births split into the four (mutated, dispersed) classes at mu * d etc., at any dispersal."""
    T = build("tau", NOBLE, spatial=dict(origin_confinement=None))
    gid = T.founder_id
    rep = T.genotypes[gid]
    rng = np.random.default_rng(0)
    for d in (0.1, 0.9):
        rep.evolutionary_parameters["dispersal_rate"] = d
        lists = [[] for _ in range(5)]          # mutants, stayers, dispersals, mut_disp, swaps
        n = 200_000
        T._split_births_noble(rng, 0, gid, rep, n, 1.0, *lists)
        tot = [sum(x[2] for x in lst) for lst in lists]
        mutated = tot[0] + tot[3] + tot[4]
        dispersed = tot[2] + tot[3] + tot[4]
        assert sum(tot) == n
        assert mutated / n == pytest.approx(0.3, abs=0.01)
        assert dispersed / n == pytest.approx(d, abs=0.01)
        # a dispersing mutated birth sends the mutant out half the time
        assert tot[3] / (tot[3] + tot[4]) == pytest.approx(0.5, abs=0.02)


def test_noble_a_dispersed_daughter_can_carry_a_new_mutation():
    T = build("exact", NOBLE, cancer=dict(mutation_rate=1.0, dispersal_rate=1.0),
              spatial=dict(origin_confinement=None))
    rng = np.random.default_rng(3)
    gid = T.founder_id
    rep = T.genotypes[gid]
    left = stayed = 0
    for _ in range(60):
        # divide a founder cell wherever one still is, as the engine only divides cells present
        di = next(i for i, d in enumerate(T.demes) if d.get(gid, 0) > 0)
        before = {i: dict(d) for i, d in enumerate(T.demes)}
        T._divide_noble(di, gid, rep, rng, [])
        for i, d in enumerate(T.demes):
            new = [g for g in d if g not in before[i] and T._is_cancer(g) and g != gid]
            if new:
                left += i != di
                stayed += i == di
    assert left > 0       # the mutant itself moved
    assert stayed > 0     # the parent copy moved and the mutant took its place


def test_noble_rejects_rates_above_one():
    with pytest.raises(ValueError, match="dispersal_rate"):
        build("tau", NOBLE, cancer=dict(dispersal_rate=1.5))
    with pytest.raises(ValueError, match="mutation_rate"):
        build("tau", NOBLE, cancer=dict(mutation_rate=1.2))
    build("tau", cancer=dict(dispersal_rate=1.5))       # the default law still caps silently
    with pytest.raises(ValueError, match="dispersal_law"):
        build("tau", dict(dispersal_law="demon"))


@pytest.mark.parametrize("mode,steps", [("tau", 40), ("exact", 3000)])
def test_noble_keeps_a_closed_acinus_closed(mode, steps):
    T = build(mode, NOBLE, spatial=dict(origin_confinement={**CONFINE, "generations": 10 ** 9}))
    T.grow(steps, materialize=False)
    outside = sum(c for i, d in enumerate(T.demes) if i not in T._origin_demes
                  for g, c in d.items() if T._is_cancer(g))
    assert T.get_cancer_size() > 0 and outside == 0


@pytest.mark.parametrize("mode,lottery,steps", [("tau", False, 60), ("tau", True, 60),
                                                ("exact", False, 4000), ("exact", True, 4000)])
def test_noble_grows_on_every_engine(mode, lottery, steps):
    T = build(mode, {**(LOTTERY if lottery else {}), **NOBLE})
    T.grow(steps, materialize=False)
    assert T.get_cancer_size() > 100
    assert all(c > 0 for d in T.demes for c in d.values())
    if lottery:
        assert all(sum(d.values()) <= T._slot_cap_of(i) for i, d in enumerate(T.demes) if d)


# --------------------------------------------------------------------------- rng_streams
def test_offset_streams_overlap_and_independent_streams_do_not():
    off_a, off_b = build("exact", seed=5), build("exact", seed=6)
    ind_a = build("exact", seed=5, rng_streams="independent")
    ind_b = build("exact", seed=6, rng_streams="independent")
    # seed 5 at step 10 is seed 6 at step 9 under the default scheme
    assert np.array_equal(off_a._run_rng(5, 10, _STREAM_GROW).random(8),
                          off_b._run_rng(6, 9, _STREAM_GROW).random(8))
    assert not np.array_equal(ind_a._run_rng(5, 10, _STREAM_GROW).random(8),
                              ind_b._run_rng(6, 9, _STREAM_GROW).random(8))


def test_independent_streams_change_the_trajectory_and_reject_unknown_values():
    T = build("exact", rng_streams="independent")
    T.grow(2000, materialize=False)
    U = build("exact")
    U.grow(2000, materialize=False)
    assert digest(T) != digest(U)
    with pytest.raises(ValueError, match="rng_streams"):
        build("exact", rng_streams="spawn")


# --------------------------------------------------------------------------- confinement clock
@pytest.mark.parametrize("clock,want", [("steps", 30), ("generations", 60)])
def test_confinement_clock(clock, want):
    """With the acinus already full, `generations: 30` at tau 0.5 lasts 30 tau steps on the default
    clock and 60 tau steps (30 generations) on the generation clock."""
    T = build("tau", spatial=dict(origin_confinement={**CONFINE, "clock": clock}))
    T._origin_age = 0.0                                  # the patch has filled; the countdown runs
    rng = np.random.default_rng(1)
    n = 0
    while T._origin_confined:
        T._tau_generation(rng, T.tau)
        n += 1
    assert n == want
    with pytest.raises(ValueError, match="clock"):
        build("tau", spatial=dict(origin_confinement={**CONFINE, "clock": "time"}))
