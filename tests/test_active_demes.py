"""The cancer-occupied deme index that keeps a tau substep O(occupied demes).

A tau substep used to walk all ``grid_size**2`` demes and pay a composition scan for each of them.
In the ductal field every deme is non-empty (it is pre-filled with normal tissue), so that scan was
paid whatever the lesion's size — and a lesion still confined to its founding acinus paid all 28,900
of them at grid 170 to advance a single occupied deme. ``_active_demes`` iterates the demes that can
actually draw an event instead.

That is a PERFORMANCE change and nothing else, so the tests below pin both halves of it: the index
(``_deme_cancer_n``) stays exactly in step with the demes through births, deaths and resection, and
a tumour grown through the narrowed loop is identical cell for cell to one grown through the full
range.
"""
import hashlib
import json

import numpy as np
import pytest

from iscc.tumor.models.count import GenotypeTumor

GENOME = dict(n_segments=6, segment_size=80)
SELECTION = dict(prop_driver=0.04, prop_dispersal=0.0, prop_immune_resistance=0.02,
                 prop_met_survival=0.02, prop_treatment_resistance=0.0,
                 prop_breach=0.02, breach_effects=2.8,
                 prop_stromal_survival=0.02, stromal_survival_effects=2.2,
                 breach_cost=0.6, stromal_survival_cost=0.6, max_mut_drivers=12,
                 trait_source="mutation")
CANCER = dict(division_rate=0.7, death_rate=0.05, max_birth_rate=2.0, mutation_rate=0.3,
              n_snvs_per_allele=0.1, dispersal_rate=0.9, cnv_prob=0.08, wgd_rate=0.001)
DEMES = dict(carrying_capacity=14, initial_cancer_cells=5, resident_pressure_ref=0.2,
             maximum_death_rate=2.2)
SPATIAL = dict(grid_size=20, n_glands=3, structure_radius=4, gland_radius=4, min_gland_sep=9,
               K_duct=14, K_stroma=14, stroma_fill_frac=0.3, cross_gland_kappa=0.06,
               cross_gland_lambda=30, breach_gated_invasion=True, epithelial_barrier=0.0,
               stromal_hazard=0.6)
CONFINE = dict(dispersal_factor=0.0, established_at=0.5, generations=150)


def make(seed=2, spatial=None, deme=None, mode="tau", confinement=None):
    sp = {**SPATIAL, **(spatial or {})}
    if confinement is not None:
        sp["origin_confinement"] = confinement
    return GenotypeTumor(
        seed=seed, genome_params=GENOME, selection_params=SELECTION, cancer_cell_params=CANCER,
        deme_params={**DEMES, **(deme or {})}, spatial_params=sp,
        update_mode=mode, tau=0.5, snapshot_every=10 ** 9, coarsen_passengers=True)


def advance(T, n, seed=2):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        T._tau_generation(rng, T.tau)
    return T


def state_hash(T):
    """Whole-tumour fingerprint: every deme's genotype counts keyed by creation ordinal WITHIN this
    tumour (``genotype_id`` is a process-global counter and cannot be compared across tumours)."""
    parts = [(i, sorted((int(T.genotypes[g].ord), int(c)) for g, c in d.items()))
             for i, d in enumerate(T.demes) if d]
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()


def recomputed(T):
    """The index rebuilt from scratch out of the demes — what it must always equal."""
    out = {}
    for i, deme in enumerate(T.demes):
        n = sum(c for g, c in deme.items() if T._is_cancer(g))
        if n:
            out[i] = n
    return out


# --------------------------------------------------------------------------- the index
def test_index_matches_the_demes_at_seeding():
    T = make()
    assert T._deme_cancer_n == recomputed(T)
    assert sum(T._deme_cancer_n.values()) == T.get_cancer_size()


@pytest.mark.parametrize("gens", [1, 5, 30])
def test_index_matches_the_demes_through_growth(gens):
    T = advance(make(), gens)
    assert T._deme_cancer_n == recomputed(T)
    assert sum(T._deme_cancer_n.values()) == T.get_cancer_size()


def test_index_survives_the_lottery_crowding_law():
    T = advance(make(deme=dict(crowding_mode="lottery")), 25)
    assert T._deme_cancer_n == recomputed(T)


def test_index_survives_the_exact_engine():
    T = make(mode="exact")
    rng = np.random.default_rng(3)
    for _ in range(3000):
        T.update(rng)
    assert T._deme_cancer_n == recomputed(T)


def test_index_survives_resection():
    """Resection empties a whole compartment — every one of its demes must leave the index."""
    T = advance(make(spatial=dict(met_grid_size=6, K_met=14, host_fill_frac=0.5,
                                  met_seed_kappa=0.4, met_transit_floor=0.5)), 30)
    T._resect("primary")
    assert T._deme_cancer_n == recomputed(T)
    assert all(i >= T.n_primary_demes for i in T._deme_cancer_n), "resected demes still indexed"


def test_a_deme_leaves_the_index_when_its_last_cancer_cell_goes():
    T = make()
    di = T._origin_demes[0]
    gid = T.founder_id
    assert di in T._deme_cancer_n
    T._remove(di, gid, T.demes[di][gid])
    assert di not in T._deme_cancer_n
    T._add(di, gid, 3)
    assert T._deme_cancer_n[di] == 3


def test_normal_cells_never_enter_the_index():
    """Only CANCER cells make a deme live for the substep; a deme full of normal tissue draws
    nothing, which is exactly why it can be skipped."""
    T = make()
    normals = [d for d in range(T.n_primary_demes)
               if T.demes[d] and not any(T._is_cancer(g) for g in T.demes[d])]
    assert normals, "the ductal field should pre-fill demes with normal tissue"
    for d in normals[:20]:
        assert d not in T._deme_cancer_n


# --------------------------------------------------------------------------- the loop
def test_active_demes_is_the_sorted_cancer_occupied_set():
    T = advance(make(), 20)
    assert list(T._active_demes()) == sorted(recomputed(T))


def test_active_demes_falls_back_to_every_deme_under_therapy():
    """Off-target chemo toxicity kills NORMAL cells, so under a dose every deme is live again and
    the loop must widen back to the full range."""
    T = advance(make(), 10)
    T._tx_death_add = {"epithelial": 0.5}
    assert list(T._active_demes()) == list(range(len(T.demes)))


@pytest.mark.parametrize("confinement", [None, CONFINE])
def test_narrowing_the_loop_does_not_change_the_trajectory(confinement, monkeypatch):
    """The skipped demes draw no rng and change no state, so a tumour grown through the narrowed
    loop has to be identical — cell for cell — to one grown through the full range. Checked with
    confinement OFF and ON: confined growth is the case the narrowing was built for (one occupied
    deme out of thousands), and it must not buy its speed with a different tumour."""
    narrowed = advance(make(confinement=confinement), 60)
    monkeypatch.setattr(GenotypeTumor, "_active_demes", lambda self: range(len(self.demes)))
    full = advance(make(confinement=confinement), 60)
    assert state_hash(narrowed) == state_hash(full)
    assert narrowed.get_cancer_size() == full.get_cancer_size()
    assert len(narrowed.genotypes_counts) == len(full.genotypes_counts)
