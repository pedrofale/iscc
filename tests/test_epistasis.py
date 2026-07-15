"""Epistasis / evolutionary-dependency structure in selection (R14, DESIGN_epistasis.md).

Covers:
  * **OFF BY DEFAULT** — no epistasis_params (or n_events=0) => the engine is BIT-IDENTICAL to the
    additive model, and turning epistasis ON does not perturb the gene-role layout;
  * **COMPARABILITY** — the network is part of the SHARED landscape: same config + same layout_seed
    => the identical network across evolution seeds and across every patient of a Cohort (the
    precondition for pooling patients to recover one network at all); a different layout_seed => a
    different network; changing n_interactions does not reshuffle the oncogene/TSG layout;
  * **the fitness term** — beta + E applied multiplicatively over the additive model, cached per
    event set (tau-leap safe), with both gating modes doing what they claim;
  * **recoverability** — in an easy regime the planted network is recovered from a cohort, and the
    empty-network control recovers ~nothing (the false-positive sanity check).
"""
import numpy as np
import pandas as pd
import pytest

from iscc.tumor.models import GenotypeTumor
from iscc.tumor.components.selection import Selection
from iscc.tumor.components.epistasis import EpistasisNetwork, bits_to_events, events_to_bits
from iscc.constants import DEFAULT_LAYOUT_SEED, LAYOUT_OFFSET_EPISTASIS
from iscc.cohort import Cohort
from iscc.integrations import (to_mhn_matrix, to_cell_fraction_matrix, to_mutation_tree,
                               to_treemhn_trees, event_cell_fractions, cooccurrence_scores,
                               top_edges, score_edges, score_order, score_exclusivity)

# prop_driver is high so the small test genome still has enough drivers to carve event modules from
GENOME = {"n_segments": 5, "segment_size": 40}
SELECTION = {"prop_driver": 0.5, "prop_dispersal": 0.1, "prop_immune_resistance": 0.1,
             "prop_treatment_resistance": 0.1, "driver_effects": 1.2}
DEME = {"carrying_capacity": 6, "initial_cancer_cells": 4}
SPATIAL = {"grid_size": 11, "structure_radius": 0}
CANCER = {"division_rate": 0.6, "death_rate": 0.05, "max_birth_rate": 0.95,
          "mutation_rate": 0.9, "dispersal_rate": 0.3}

EPI = {"n_events": 6, "event_size": 5, "n_interactions": 4, "interaction_strength": 0.4,
       "interaction_strength_sd": 0.05}


def _tumor(seed=1, layout_seed=None, epistasis_params=None, dependency_params=None, **sel):
    sp = dict(SELECTION, **sel)
    if epistasis_params is not None:
        sp["epistasis_params"] = epistasis_params
    if dependency_params is not None:
        sp["dependency_params"] = dependency_params
    return GenotypeTumor(seed=seed, layout_seed=layout_seed, genome_params=GENOME,
                         selection_params=sp, cancer_cell_params=CANCER,
                         deme_params=DEME, spatial_params=SPATIAL)


def _grown(seed=1, steps=200, **kw):
    t = _tumor(seed=seed, **kw)
    t.grow(n_steps=steps, seed=seed)
    return t


def _cohort(n=8, epistasis_params=EPI, steps=200, **kw):
    """N patients over ONE shared landscape (shared network, private evolution) — the substrate the
    whole progression benchmark stands on."""
    return Cohort(patient_seeds=list(range(1, n + 1)), genome_params=GENOME,
                  selection_params={**SELECTION, "epistasis_params": epistasis_params, **kw},
                  cancer_cell_params=CANCER, deme_params=DEME, spatial_params=SPATIAL,
                  grow_steps=steps)


def _cohort_tumors(coh):
    return [p.tumor for p in coh.run().patients]


# ================================ off by default (the F8 discipline) ================================
def test_epistasis_is_off_by_default():
    t = _tumor()
    assert t.selection.epistasis is None
    assert t.epistasis_ground_truth() is None
    assert t.event_table().empty


def test_off_is_bit_identical_to_the_additive_model():
    """The whole point of off-by-default: an absent network and an explicitly empty one both
    reproduce the additive engine byte-for-byte at the same seed."""
    base = _grown(seed=3)
    for eq in ({"n_events": 0}, {"n_events": 0, "n_interactions": 99}):
        other = _grown(seed=3, epistasis_params=eq)
        assert np.array_equal(base.cell_data["cell_snv"].values,
                              other.cell_data["cell_snv"].values)
        assert np.array_equal(base.cell_data["cell_cnv"].values,
                              other.cell_data["cell_cnv"].values)


def test_turning_epistasis_on_does_not_perturb_the_gene_layout():
    """The dedicated layout SUB-STREAM at work: the network is drawn from layout_seed +
    LAYOUT_OFFSET_EPISTASIS, so layout_rng is untouched and nothing re-baselines."""
    off, on = _tumor(), _tumor(epistasis_params=EPI)
    for get in ("get_oncogenes", "get_tsgs", "get_dispersal_genes",
                "get_treatment_resistant", "get_immune_resistant"):
        assert list(getattr(off.selection, get)()) == list(getattr(on.selection, get)()), get
    for ct in ("cancer", "epithelial", "stromal", "immune"):
        assert np.array_equal(off.celltype_exps[ct], on.celltype_exps[ct])


def test_network_params_do_not_reshuffle_the_layout():
    """Changing n_interactions / topology changes the NETWORK and nothing else — the independence
    the sub-stream registry exists to guarantee."""
    a = _tumor(epistasis_params=EPI)
    b = _tumor(epistasis_params={**EPI, "n_interactions": 8, "network_topology": "hub"})
    assert list(a.selection.get_oncogenes()) == list(b.selection.get_oncogenes())
    assert np.array_equal(a.celltype_exps["cancer"], b.celltype_exps["cancer"])
    assert not np.array_equal(a.selection.epistasis.E, b.selection.epistasis.E)


# ============================ comparability across patients (the premise) ============================
def test_same_config_different_evolution_seed_shares_the_network():
    """MHN/TreeMHN pool many tumours to fit ONE network; that is only well-posed if every patient
    evolved under the SAME network. Different evolution seed => identical network."""
    a, b = _tumor(seed=1, epistasis_params=EPI), _tumor(seed=2, epistasis_params=EPI)
    assert np.array_equal(a.selection.epistasis.E, b.selection.epistasis.E)
    assert np.array_equal(a.selection.epistasis.beta, b.selection.epistasis.beta)
    assert a.selection.epistasis.true_dag_edges() == b.selection.epistasis.true_dag_edges()
    for ga, gb in zip(a.selection.epistasis.event_genes, b.selection.epistasis.event_genes):
        assert np.array_equal(ga, gb)


def test_different_layout_seed_gives_a_different_network():
    """The replicate-study knob: an explicit different layout_seed re-draws the network."""
    a = _tumor(layout_seed=DEFAULT_LAYOUT_SEED, epistasis_params=EPI)
    b = _tumor(layout_seed=7, epistasis_params=EPI)
    assert not np.array_equal(a.selection.epistasis.E, b.selection.epistasis.E)


def test_network_uses_the_registered_layout_offset():
    """The network comes from the REGISTERED sub-stream, not from layout_rng or the evolution seed."""
    t = _tumor(layout_seed=11, epistasis_params=EPI)
    direct = EpistasisNetwork(
        driver_pool=np.concatenate([t.selection.get_oncogenes(), t.selection.get_tsgs()]),
        seg_offsets=t.selection._seg_offsets, segment_sizes=t.selection.segment_sizes,
        rng=np.random.default_rng(11 + LAYOUT_OFFSET_EPISTASIS), epistasis_params=EPI)
    assert np.array_equal(direct.E, t.selection.epistasis.E)


def test_whole_cohort_shares_one_network():
    """The benchmark is only well-posed if the cohort shares a network — assert it directly."""
    coh = _cohort(n=4, steps=60)
    shared = coh.selection.epistasis           # the cohort's shared landscape
    for t in _cohort_tumors(coh):
        net = t.selection.epistasis
        assert np.array_equal(net.E, shared.E)
        assert np.array_equal(net.beta, shared.beta)
        assert net.true_dag_edges() == shared.true_dag_edges()
        for g, gs in zip(net.event_genes, shared.event_genes):
            assert np.array_equal(g, gs)


# =================================== the network itself ===================================
def _net(**ep):
    sel = Selection(n_segments=5, segment_size=40, rng=np.random.default_rng(DEFAULT_LAYOUT_SEED),
                    layout_seed=DEFAULT_LAYOUT_SEED, epistasis_params={**EPI, **ep},
                    **SELECTION)
    return sel.epistasis


def test_event_modules_are_disjoint_and_drawn_from_drivers():
    sel = Selection(n_segments=5, segment_size=40, rng=np.random.default_rng(DEFAULT_LAYOUT_SEED),
                    layout_seed=DEFAULT_LAYOUT_SEED, epistasis_params=EPI, **SELECTION)
    net = sel.epistasis
    drivers = set(int(g) for g in np.concatenate([sel.get_oncogenes(), sel.get_tsgs()]))
    seen = set()
    for genes in net.event_genes:
        assert len(genes) == EPI["event_size"]
        assert set(int(g) for g in genes) <= drivers      # events are driver genes
        assert not (seen & set(int(g) for g in genes))    # modules are disjoint
        seen |= set(int(g) for g in genes)


def test_E_is_symmetric_zero_diagonal_and_the_right_sparsity():
    net = _net(n_interactions=4)
    assert np.array_equal(net.E, net.E.T)
    assert np.allclose(np.diag(net.E), 0)
    assert len(net.true_edges()) == 4


def test_network_sparsity_when_n_interactions_is_unset():
    net = _net(n_interactions=None, network_sparsity=0.2)     # 6 events -> 15 pairs -> 3 edges
    assert len(net.true_edges()) == 3


@pytest.mark.parametrize("topology,check", [
    ("hub", lambda e: e[0] == 0 or e[1] == 0),     # every edge touches the hub
    ("chain", lambda e: abs(e[1] - e[0]) == 1),    # every edge is a consecutive link
])
def test_topologies(topology, check):
    net = _net(network_topology=topology, n_interactions=5)
    assert all(check(e) for e in net.true_edges())


def test_synergy_and_antagonism_proportions():
    net = _net(n_events=12, event_size=2, n_interactions=40, prop_synergy=1.0)
    assert all(w > 0 for _, _, w in net.true_edges())
    net = _net(n_events=12, event_size=2, n_interactions=40, prop_synergy=0.0)
    assert all(w < 0 for _, _, w in net.true_edges())


def test_mutual_exclusivity_edges_are_strongly_negative_and_distinct():
    net = _net(n_interactions=3, n_exclusive_pairs=2, mutual_exclusivity_strength=2.0)
    excl = net.true_exclusive_pairs()
    assert len(excl) == 2
    for (i, j) in excl:
        assert net.E[i, j] == -2.0
    assert len(net.true_edges()) == 5   # 3 interactions + 2 exclusive, on distinct pairs


def test_invalid_params_are_rejected():
    with pytest.raises(ValueError, match="network_topology"):
        _net(network_topology="banana")
    with pytest.raises(ValueError, match="gating_mode"):
        Selection(n_segments=5, segment_size=40, rng=np.random.default_rng(1),
                  epistasis_params=EPI, dependency_params={"gating_mode": "banana"}, **SELECTION)
    with pytest.raises(ValueError, match="driver genes"):
        _net(n_events=50, event_size=50)   # more event genes than the layout has drivers


# =================================== the fitness term ===================================
def test_fitness_is_beta_plus_E_over_the_event_set():
    net = _net(n_interactions=4, event_effect_mean=0.1, event_effect_sd=0.0)
    bits = events_to_bits([0, 2, 3])
    expected = net.beta[[0, 2, 3]].sum() + net.E[0, 2] + net.E[0, 3] + net.E[2, 3]
    assert net.log_fitness(bits) == pytest.approx(expected)


def test_no_events_is_exactly_neutral():
    """The event-free genotype must map to exactly 1.0 or the network would move the baseline."""
    assert _net().multiplier(0) == 1.0


def test_fitness_caches_per_event_set():
    net = _net()
    bits = events_to_bits([1, 3])
    first = net.log_fitness(bits)
    assert bits in net._fitness_cache
    net._fitness_cache[bits] = 12345.0          # poison the cache
    assert net.log_fitness(bits) == 12345.0     # proves the second call is served from it
    net._fitness_cache[bits] = first
    # the cache is keyed by the event SET, so order of acquisition cannot matter
    assert net.log_fitness(events_to_bits([3, 1])) == first


def test_synergy_beats_the_sum_of_its_parts():
    net = _net(n_interactions=1, network_topology="chain", prop_synergy=1.0)
    (i, j, w) = net.true_edges()[0]
    both = net.log_fitness(events_to_bits([i, j]))
    alone = net.log_fitness(events_to_bits([i])) + net.log_fitness(events_to_bits([j]))
    assert both - alone == pytest.approx(w)
    assert w > 0


def test_fitness_gating_makes_a_child_inert_until_its_parent_arrives():
    net = _net(n_interactions=0, event_effect_mean=0.5, event_effect_sd=0.0)
    net.dag_edges = [(0, 1)]
    net.dag_parents = {i: ((0,) if i == 1 else ()) for i in range(net.n_events)}
    net._parent_mask = np.array([events_to_bits(net.dag_parents[i]) for i in range(net.n_events)],
                                dtype=object)
    net._has_dag, net.gating_mode, net._fitness_cache = True, "fitness", {}
    # event 1 alone: gated off -> contributes nothing
    assert net.log_fitness(events_to_bits([1])) == pytest.approx(0.0)
    # with its parent present it pays out
    assert net.log_fitness(events_to_bits([0, 1])) == pytest.approx(net.beta[0] + net.beta[1])


def test_accessibility_gating_blocks_the_child_from_arising_at_all():
    net = _net(n_interactions=0)
    net.dag_edges = [(0, 1)]
    net.dag_parents = {i: ((0,) if i == 1 else ()) for i in range(net.n_events)}
    net._parent_mask = np.array([events_to_bits(net.dag_parents[i]) for i in range(net.n_events)],
                                dtype=object)
    net._has_dag, net.gating_mode = True, "accessibility"
    net._fitness_cache, net._blocked_cache = {}, {}
    assert net.blocked_events(0) == frozenset({1})           # nothing acquired -> 1 is unreachable
    assert net.blocked_events(events_to_bits([0])) == frozenset()   # parent present -> unblocked
    # under accessibility gating fitness is NOT additionally gated (the event cannot exist anyway)
    assert net.log_fitness(events_to_bits([1])) == pytest.approx(net.beta[1])


def test_dag_is_acyclic_and_respects_branching():
    net = _net(n_interactions=0)
    dp = {"n_constraints": 6, "dag_depth": 3, "dag_branching": 1, "gating_mode": "fitness"}
    sel = Selection(n_segments=5, segment_size=40, rng=np.random.default_rng(DEFAULT_LAYOUT_SEED),
                    layout_seed=DEFAULT_LAYOUT_SEED, epistasis_params=EPI,
                    dependency_params=dp, **SELECTION)
    net = sel.epistasis
    assert all(len(p) <= 1 for p in net.dag_parents.values())   # dag_branching honoured
    # acyclicity: a topological sort must exist
    import itertools
    remaining = dict(net.dag_parents)
    done = set()
    for _ in range(len(remaining) + 1):
        ready = [i for i, ps in remaining.items() if i not in done and set(ps) <= done]
        if not ready:
            break
        done |= set(ready)
    assert done == set(range(net.n_events)), "dependency DAG contains a cycle"


# =============================== the events along a lineage ===============================
def test_events_accumulate_and_order_matches_the_event_set():
    t = _grown(epistasis_params=EPI)
    tbl = t.event_table()
    assert not tbl.empty
    assert tbl["events"].apply(len).max() > 0, "no events acquired: regime is too quiet to test"
    for _, row in tbl.iterrows():
        assert sorted(row["event_order"]) == list(row["events"])   # order is a permutation of the set
        assert len(set(row["event_order"])) == len(row["event_order"])  # monotone: no event twice


def test_child_clones_never_lose_their_parents_events():
    """Events are MONOTONE — the generative assumption MHN/CBN are defined under."""
    t = _grown(epistasis_params=EPI)
    for gid, parent in t.genotypes_parents.items():
        if gid not in t.genotypes or parent not in t.genotypes:
            continue
        child_bits = t.genotypes[gid].event_bits
        parent_bits = t.genotypes[parent].event_bits
        assert child_bits & parent_bits == parent_bits, "a clone lost an ancestor's event"
        assert t.genotypes[gid].event_order[:len(t.genotypes[parent].event_order)] == \
            t.genotypes[parent].event_order


def test_accessibility_gating_yields_exact_order_along_every_lineage():
    """Its whole point: with the mutation vetoed, a child event can NEVER precede its parent."""
    dp = {"n_constraints": 4, "dag_depth": 3, "gating_mode": "accessibility"}
    t = _grown(epistasis_params=EPI, dependency_params=dp)
    net = t.selection.epistasis
    res = score_order(net.true_dag_edges(), t.event_table()["event_groups"])
    assert res["n_scored"] > 0, "no gated events acquired: regime too quiet to test ordering"
    assert res["order_accuracy"] == 1.0


def test_tau_leaping_runs_with_epistasis_and_shares_the_network():
    """The interaction term is a pure function of the event set, so it is tau-leap safe."""
    sp = {**SELECTION, "epistasis_params": EPI}
    t = GenotypeTumor(seed=1, genome_params=GENOME, selection_params=sp, cancer_cell_params=CANCER,
                      deme_params=DEME, spatial_params=SPATIAL, update_mode="tau", tau=0.5)
    t.grow(n_steps=40, seed=1)
    assert np.array_equal(t.selection.epistasis.E, _tumor(epistasis_params=EPI).selection.epistasis.E)
    assert not t.event_table().empty


# =================================== export + scoring ===================================
def test_mhn_matrix_shape_and_content():
    X = to_mhn_matrix(_cohort_tumors(_cohort(n=5)))
    assert X.shape == (5, EPI["n_events"])
    assert set(np.unique(X.values)) <= {0, 1}
    assert list(X.columns) == [f"E{i}" for i in range(EPI["n_events"])]


def test_mutation_tree_is_a_valid_rooted_trie():
    t = _grown(epistasis_params=EPI)
    tree = to_mutation_tree(t)
    root = tree.iloc[0]
    # TreeMHN's convention (its README): root is Node_ID 1, Mutation_ID 0, and its OWN parent.
    assert root["Node_ID"] == 1 and root["Mutation_ID"] == 0 and root["Parent_ID"] == 1
    assert tree["Node_ID"].is_unique
    nodes = set(tree["Node_ID"])
    for _, row in tree.iloc[1:].iterrows():
        assert row["Parent_ID"] in nodes         # every non-root node hangs off a real parent
        assert row["Parent_ID"] < row["Node_ID"]  # parents precede children -> acyclic
        assert 1 <= row["Mutation_ID"] <= EPI["n_events"]


def test_treemhn_trees_cover_every_patient():
    trees = to_treemhn_trees(_cohort_tumors(_cohort(n=4)))
    assert sorted(trees["Patient_ID"].unique()) == [1, 2, 3, 4]


def test_min_clone_freq_prunes_tree_tips():
    t = _grown(epistasis_params=EPI)
    assert len(to_mutation_tree(t, min_clone_freq=0.5)) <= len(to_mutation_tree(t, min_clone_freq=0.0))


def test_event_detection_aggregates_cell_fraction_across_clones():
    """The detection threshold is a per-EVENT cancer-cell fraction, NOT a per-clone size filter.

    This is a regression test for a real bug: filtering CLONES by size and then OR-ing their events
    calls an event undetected whenever its carriers are many small lineages -- which is exactly what
    a favoured combination looks like, since it arises repeatedly. A 60%-cell-fraction event spread
    over 30 clones of 2% each was being reported as ABSENT, silently zeroing the benchmark.
    """
    t = _grown(epistasis_params=EPI)
    net = t.selection.epistasis
    tbl = t.event_table()
    total = tbl["n_cells"].sum()

    frac = event_cell_fractions(t)
    assert frac.shape == (net.n_events,)
    for e in range(net.n_events):
        carried = tbl["events"].apply(lambda ev: e in ev)
        assert frac[e] == pytest.approx(tbl.loc[carried, "n_cells"].sum() / total)
        # the aggregate MUST be >= the largest single carrying clone: that is the whole point
        if carried.any():
            assert frac[e] >= tbl.loc[carried, "n_cells"].max() / total - 1e-12

    # and the binary vector thresholds THAT quantity
    for mf in (0.0, 0.1, 0.5):
        vec = to_mhn_matrix([t], min_freq=mf).values[0]
        expected = (frac > 0).astype(int) if mf == 0 else (frac >= mf).astype(int)
        assert np.array_equal(vec, expected)


def test_cell_fraction_matrix_is_the_continuous_observable():
    tumors = _cohort_tumors(_cohort(n=4))
    F = to_cell_fraction_matrix(tumors)
    assert F.shape == (4, EPI["n_events"])
    assert ((F.values >= 0) & (F.values <= 1)).all()
    # binary presence is exactly the >0 indicator of it
    assert np.array_equal(to_mhn_matrix(tumors).values, (F.values > 0).astype(int))


def test_score_edges_precision_recall():
    truth = [(0, 1), (2, 3)]
    assert score_edges(truth, [(1, 0), (3, 2)])["f1"] == 1.0     # undirected: order within a pair is free
    r = score_edges(truth, [(0, 1), (0, 4)])
    assert r["precision"] == 0.5 and r["recall"] == 0.5
    assert score_edges(truth, [])["recall"] == 0.0
    # sign matters when asked
    assert score_edges([(0, 1, 0.5)], [(0, 1, -0.5)], match_sign=True)["tp"] == 0
    assert score_edges([(0, 1, 0.5)], [(0, 1, 0.9)], match_sign=True)["tp"] == 1


def test_score_order_counts_only_informative_lineages():
    r = score_order([(0, 1)], [(0, 1), (1, 0), (2,)])   # third lineage lacks the child -> skipped
    assert r["n_scored"] == 2 and r["order_accuracy"] == 0.5
    assert score_order([(0, 1)], [(2,)])["n_uninformative"] == 1


def test_score_order_excludes_tied_events():
    """Events acquired in the SAME division have no order to recover; scoring them would credit (or
    penalise) a method for an ordering the simulator never generated."""
    r = score_order([(0, 1)], [((0, 1),)])              # both in one division -> tied
    assert r["n_tied"] == 1 and r["n_scored"] == 0
    r = score_order([(0, 1)], [((0,), (1,)), ((0, 1),)])  # one real order + one tie
    assert r["n_scored"] == 1 and r["n_tied"] == 1 and r["order_accuracy"] == 1.0


# ============================= recoverability + the empty-E control =============================
# A cohort regime where events actually accrue at intermediate frequency: enough mutation for every
# event to be reachable, slow enough growth that clones are not all unique. Selection is left
# NEUTRAL (driver_effects=1.0) so the planted network is the only thing acting — otherwise the
# additive driver-count fitness pins the division rate at max_birth_rate and the clamp eats the
# interaction term entirely (a real trap; see PARAMETERS.md).
RECOVERY_GENOME = {"n_segments": 5, "segment_size": 40}
RECOVERY_SELECTION = {"prop_driver": 0.5, "driver_effects": 1.0}
RECOVERY_CANCER = {"division_rate": 0.2, "death_rate": 0.02, "max_birth_rate": 0.95,
                   "mutation_rate": 0.3, "dispersal_rate": 0.3, "n_snvs_per_allele": 0.15}
RECOVERY_DEME = {"carrying_capacity": 10, "initial_cancer_cells": 4}
RECOVERY_SPATIAL = {"grid_size": 7, "structure_radius": 0}


def _recovery_cohort(n=24, steps=500, **sel):
    coh = Cohort(patient_seeds=list(range(1, n + 1)), genome_params=RECOVERY_GENOME,
                 selection_params={**RECOVERY_SELECTION, **sel},
                 cancer_cell_params=RECOVERY_CANCER, deme_params=RECOVERY_DEME,
                 spatial_params=RECOVERY_SPATIAL, grow_steps=steps)
    # extinct patients contribute an all-zero row and are not patients in a real cohort study
    return [p.tumor for p in coh.run().patients if p.tumor.get_cancer_size() > 0]


def test_planted_dependency_dag_is_recoverable_in_an_easy_regime():
    """The payoff, for the constraint the tools are actually built to see.

    Under ACCESSIBILITY gating the planted constraint is a hard implication — a child event cannot
    arise before its parent — so it survives into the cross-sectional matrix as "no patient carries
    the child without the parent", which is exactly the poset CBN/H-CBN fit, and into the mutation
    trees as an exact order for TreeMHN. Deliberately an EASY regime; where recovery DEGRADES is the
    interesting question and lives in validation/validate_epistasis.py.
    """
    ep = {"n_events": 4, "event_size": 8, "n_interactions": 0,
          "event_effect_mean": 0.0, "event_effect_sd": 0.0}
    dp = {"n_constraints": 2, "dag_depth": 2, "dag_branching": 1, "gating_mode": "accessibility"}
    tumors = _recovery_cohort(epistasis_params=ep, dependency_params=dp)
    net = tumors[0].selection.epistasis
    X = to_mhn_matrix(tumors).values

    informative = violations = 0
    for (parent, child) in net.true_dag_edges():
        violations += int(((X[:, child] == 1) & (X[:, parent] == 0)).sum())
        informative += int(((X[:, child] == 1) & (X[:, parent] == 1)).sum())
    assert informative > 0, "no patient acquired a gated event: regime has no power to test recovery"
    assert violations == 0, "accessibility gating leaked: a child event appeared without its parent"

    order = score_order(net.true_dag_edges(),
                        [g for t in tumors for g in t.event_table()["event_groups"]])
    assert order["n_scored"] > 0
    assert order["order_accuracy"] == 1.0


def test_fitness_epistasis_actually_moves_clone_fitness():
    """The pairwise-E model does what it claims WHERE IT ACTS — on the division rate of the clones
    carrying the pair.

    Note what this does NOT claim: that a cross-sectional method could find it. iscc's E acts on
    FITNESS (how big a clone grows), while MHN/CBN model the RATE OF ACQUISITION; a binary
    "ever acquired" matrix is largely blind to the difference. That gap is the benchmark's headline
    finding, quantified in validation/validate_epistasis.py — not something to paper over here.
    """
    ep = {"n_events": 4, "event_size": 8, "n_interactions": 1, "network_topology": "chain",
          "interaction_strength": 0.5, "interaction_strength_sd": 0.0, "prop_synergy": 1.0,
          "event_effect_mean": 0.0, "event_effect_sd": 0.0}
    tumors = _recovery_cohort(n=8, epistasis_params=ep)
    net = tumors[0].selection.epistasis
    (i, j, w) = net.true_edges()[0]
    both, single = [], []
    for t in tumors:
        for gid in t.genotypes_counts:
            if gid in ("epithelial", "stromal", "immune"):
                continue
            rep = t.genotypes[gid]
            has = (rep.event_bits >> i & 1, rep.event_bits >> j & 1)
            if all(has):
                both.append(rep.evolutionary_parameters["division_rate"])
            elif any(has):
                single.append(rep.evolutionary_parameters["division_rate"])
    assert both and single, "regime produced no informative clones"
    assert np.mean(both) > np.mean(single), "planted synergy did not raise the double-carriers' fitness"


def test_empty_network_control_recovers_nothing():
    """The sanity control (DESIGN_epistasis.md §5): with an EMPTY E, whatever a method reports is a
    FALSE POSITIVE. We can't assert a method finds nothing, but we CAN assert the truth is empty and
    that iscc scores every reported edge against it as a false positive — which is exactly what makes
    the specificity row of the benchmark meaningful."""
    ep = {"n_events": 5, "event_size": 6, "n_interactions": 0}
    tumors = _recovery_cohort(n=8, steps=200, epistasis_params=ep)
    net = tumors[0].selection.epistasis
    assert net.true_edges() == []
    assert np.allclose(net.E, 0)
    X = to_mhn_matrix(tumors)
    pred = top_edges(cooccurrence_scores(X.values), k=3)
    res = score_edges(net.true_edges(), pred)
    assert res["tp"] == 0 and res["fp"] == 3 and res["precision"] == 0.0


def test_exclusivity_sign_recovery_scores_against_the_planted_signs():
    r = score_exclusivity([(0, 1)], np.array([[0, -2.0], [-2.0, 0]]))
    assert r["sign_accuracy"] == 1.0
    r = score_exclusivity([(0, 1)], np.array([[0, 2.0], [2.0, 0]]))
    assert r["sign_accuracy"] == 0.0


# =================================== bit helpers ===================================
def test_bit_helpers_round_trip():
    assert bits_to_events(events_to_bits([0, 3, 7]), 8) == [0, 3, 7]
    assert events_to_bits([]) == 0
    assert bits_to_events(0, 4) == []


# ============================ the REAL tools (own envs; skip if absent) ============================
# Follows the repo's one-env-per-tool convention (validation/README_integration.md): the tool runs in
# `iscc-mhn` / `iscc-treemhn` via subprocess and is SKIPPED when that env is absent. These are
# POSITIVE CONTROLS on the seam -- they assert the export format is one the tool actually accepts and
# that it recovers a signal iscc planted directly in the tool's OWN input. They deliberately do NOT
# assert recovery from a grown tumour: whether the cross-sectional observable carries the signal at
# all is the open question the validation measures, not something to bake into the suite.
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "validation"))


def _ve():
    import validate_epistasis as ve
    return ve


@pytest.mark.skipif(not _ve().mhn_available(), reason="iscc-mhn env not installed")
def test_real_mhn_accepts_our_matrix_and_finds_a_planted_coupling():
    ve = _ve()
    rng = np.random.default_rng(0)
    n = 150
    a = rng.random(n) < 0.5
    X = pd.DataFrame({
        "E0": a.astype(int),
        "E1": np.where(a, rng.random(n) < 0.9, rng.random(n) < 0.1).astype(int),   # E0 -> E1
        "E2": (rng.random(n) < 0.5).astype(int),
        "E3": (rng.random(n) < 0.5).astype(int),
    }, index=[f"P{i}" for i in range(n)])
    theta = ve.run_mhn(X)
    assert theta is not None and theta.shape == (4, 4)
    top = ve.theta_to_edges(theta, k=1)
    assert {top[0][0], top[0][1]} == {0, 1}, f"MHN missed the planted coupling: {top}"


@pytest.mark.skipif(not _ve().treemhn_available(), reason="iscc-treemhn env not installed")
def test_real_treemhn_accepts_our_tree_export():
    """The export contract: TreeMHN's input_tree_df must accept `to_mutation_tree`'s frame as-is
    (root = Node_ID 1, Mutation_ID 0, its OWN parent -- Parent_ID 0 makes its sorter fail)."""
    ve = _ve()
    tumors = _cohort_tumors(_cohort(n=6, steps=150))
    trees = to_treemhn_trees(tumors)
    assert set(["Patient_ID", "Tree_ID", "Node_ID", "Mutation_ID", "Parent_ID"]) <= set(trees.columns)
    roots = trees[trees["Mutation_ID"] == 0]
    assert (roots["Parent_ID"] == roots["Node_ID"]).all()      # root is its own parent
    theta = ve.run_treemhn(trees)
    assert theta is not None
    assert theta.shape == (EPI["n_events"], EPI["n_events"])


def test_treemhn_export_drops_event_free_patients():
    """A patient with no events is a root-only tree, which TreeMHN rejects outright
    ("Tree with ID n does not contain edges from the root"). Regression test: they are dropped and
    the survivors renumbered contiguously (TreeMHN indexes patients by position)."""
    tumors = _cohort_tumors(_cohort(n=6, steps=200))
    trees = to_treemhn_trees(tumors)
    for pid, g in trees.groupby("Patient_ID"):
        assert len(g) >= 2, f"patient {pid} exported with no edges"
        assert (g["Mutation_ID"] == 0).sum() == 1              # exactly one root
    ids = sorted(trees["Patient_ID"].unique())
    assert ids == list(range(1, len(ids) + 1)), "patient IDs must be contiguous from 1"


def test_treemhn_export_raises_when_no_patient_has_events():
    t = _grown(steps=5, epistasis_params={"n_events": 4, "event_size": 1})
    with pytest.raises(ValueError, match="no mutation trees"):
        to_treemhn_trees([t])


# ============================ synthetic lethality (mutual exclusivity) ============================
def test_exclusive_pair_is_lethal_and_actually_removes_the_clone():
    """A strongly negative E is NOT enough to make a pair mutually exclusive, and that is the whole
    reason this knob exists.

    Negative E only suppresses the DIVISION rate, and the crowding law (count.py `_death_rate`) uses
    slope = max(0, div - death_rate): a clone that has stopped dividing therefore takes NO
    density-dependent death at all and is never purged -- it persists as a small clone and the pair
    still reads as co-occurring, the exact opposite of the DISCOVER/MEGSA signal it is meant to plant.
    Synthetic lethality gives the combination `lethal_death_rate` instead, so it is removed.
    """
    ep = {**EPI, "n_interactions": 0, "n_exclusive_pairs": 1,
          "mutual_exclusivity_strength": 8.0, "mutual_exclusivity_lethal": True}
    sel = Selection(n_segments=5, segment_size=40, rng=np.random.default_rng(DEFAULT_LAYOUT_SEED),
                    layout_seed=DEFAULT_LAYOUT_SEED, epistasis_params=ep, **SELECTION)
    net = sel.epistasis
    (i, j) = net.true_exclusive_pairs()[0]
    assert net.is_lethal(events_to_bits([i, j])) is True          # both -> lethal
    assert net.is_lethal(events_to_bits([i])) is False            # one alone -> fine
    assert net.is_lethal(0) is False
    # and it reaches the death rate the engine reads
    assert sel.update_death_rate({}, 0.02, event_bits=events_to_bits([i, j])) == net.lethal_death_rate
    assert sel.update_death_rate({}, 0.02, event_bits=events_to_bits([i])) == 0.02


def test_lethality_can_be_turned_off_leaving_the_soft_fitness_effect():
    ep = {**EPI, "n_interactions": 0, "n_exclusive_pairs": 1,
          "mutual_exclusivity_strength": 8.0, "mutual_exclusivity_lethal": False}
    sel = Selection(n_segments=5, segment_size=40, rng=np.random.default_rng(DEFAULT_LAYOUT_SEED),
                    layout_seed=DEFAULT_LAYOUT_SEED, epistasis_params=ep, **SELECTION)
    net = sel.epistasis
    (i, j) = net.true_exclusive_pairs()[0]
    assert net.is_lethal(events_to_bits([i, j])) is False
    assert sel.update_death_rate({}, 0.02, event_bits=events_to_bits([i, j])) == 0.02
    # the strongly-negative E is still there; it just no longer kills
    assert net.E[i, j] == -8.0


def test_death_rate_is_untouched_when_epistasis_is_off():
    """The death path is newly routed through selection; it must be inert by default."""
    t = _tumor()
    assert t.selection.update_death_rate({}, 0.02, event_bits=0) == 0.02
    assert t.selection.epistasis is None
