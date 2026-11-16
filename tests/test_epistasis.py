"""Epistasis / evolutionary-dependency structure in selection.

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
import functools

from iscc.cohort import Cohort, grow_tumor, shared_selection
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
    grow = functools.partial(
        grow_tumor, genome_params=GENOME,
        selection_params={**SELECTION, "epistasis_params": epistasis_params, **kw},
        cancer_cell_params=CANCER, deme_params=DEME, spatial_params=SPATIAL, grow_steps=steps)
    return Cohort(grow, patient_seeds=list(range(1, n + 1)))


def _cohort_tumors(coh):
    return [p.tumor for p in coh.run().patients]


# ================================ off by default ================================
def test_epistasis_is_off_by_default():
    t = _tumor()
    assert t.selection.epistasis is None
    assert t.epistasis_ground_truth() is None
    assert t.event_table().empty


def test_off_is_bit_identical_to_the_additive_model():
    """An absent network and an explicitly empty one both reproduce the additive engine
    byte-for-byte at the same seed."""
    base = _grown(seed=3)
    for eq in ({"n_events": 0}, {"n_events": 0, "n_interactions": 99}):
        other = _grown(seed=3, epistasis_params=eq)
        assert np.array_equal(base.cell_data["cell_snv"].values,
                              other.cell_data["cell_snv"].values)
        assert np.array_equal(base.cell_data["cell_cnv"].values,
                              other.cell_data["cell_cnv"].values)


def test_turning_epistasis_on_does_not_perturb_the_gene_layout():
    """The network is drawn from its own layout sub-stream (layout_seed + LAYOUT_OFFSET_EPISTASIS),
    so layout_rng is untouched and the other gene roles do not move."""
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
    # the cohort's shared landscape, built without growing anyone
    shared = shared_selection(GENOME, {**SELECTION, "epistasis_params": EPI}).epistasis
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
    X = to_mhn_matrix(_cohort_tumors(_cohort(n=5)), min_freq=0.0)
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

    Filtering CLONES by size and then OR-ing their events would call an event undetected whenever its
    carriers are many small lineages -- which is exactly what a favoured combination looks like, since
    it arises repeatedly. A 60%-cell-fraction event spread over 30 clones of 2% each must be PRESENT.
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
    assert np.array_equal(to_mhn_matrix(tumors, min_freq=0.0).values, (F.values > 0).astype(int))


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
                   "mutation_rate": 0.5, "dispersal_rate": 0.3, "n_snvs_per_allele": 0.15}
RECOVERY_DEME = {"carrying_capacity": 10, "initial_cancer_cells": 4}
RECOVERY_SPATIAL = {"grid_size": 7, "structure_radius": 0}


def _recovery_cohort(n=24, steps=500, **sel):
    grow = functools.partial(
        grow_tumor, genome_params=RECOVERY_GENOME, selection_params={**RECOVERY_SELECTION, **sel},
        cancer_cell_params=RECOVERY_CANCER, deme_params=RECOVERY_DEME,
        spatial_params=RECOVERY_SPATIAL, grow_steps=steps)
    coh = Cohort(grow, patient_seeds=list(range(1, n + 1)))
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
    X = to_mhn_matrix(tumors, min_freq=0.0).values

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
    """The sanity control: with an EMPTY E, whatever a method reports is a
    FALSE POSITIVE. We can't assert a method finds nothing, but we CAN assert the truth is empty and
    that iscc scores every reported edge against it as a false positive — which is exactly what makes
    the specificity row of the benchmark meaningful."""
    ep = {"n_events": 5, "event_size": 6, "n_interactions": 0}
    tumors = _recovery_cohort(n=8, steps=200, epistasis_params=ep)
    net = tumors[0].selection.epistasis
    assert net.true_edges() == []
    assert np.allclose(net.E, 0)
    X = to_mhn_matrix(tumors, min_freq=0.0)
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
    ("Tree with ID n does not contain edges from the root"). They are dropped and the survivors
    renumbered contiguously (TreeMHN indexes patients by position)."""
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
    # the strongly-negative E is still there; it just does not kill
    assert net.E[i, j] == -8.0


def test_death_rate_is_untouched_when_epistasis_is_off():
    """The death path runs through selection; it must be inert when epistasis is off."""
    t = _tumor()
    assert t.selection.update_death_rate({}, 0.02, event_bits=0) == 0.02
    assert t.selection.epistasis is None


# ============================ the planted (explicit) network ============================
# Fig 5's network (vault: decision-fig5-planted-network), with the decision's E1..E6 as events 0..5.
PLANTED = {"effects": 0.05, "interactions": [(0, 1, 0.4)], "exclusive": [(2, 3)],
           "constraints": [(4, 5)]}
PLANTED_EP = {"n_events": 6, "event_size": 5, "lethal_death_rate": 2.5, "planted": PLANTED}
PLANTED_DP = {"gating_mode": "accessibility"}


def _planted_selection(ep=PLANTED_EP, dp=PLANTED_DP, layout_seed=DEFAULT_LAYOUT_SEED):
    return Selection(n_segments=GENOME["n_segments"], segment_size=GENOME["segment_size"],
                     rng=np.random.default_rng(layout_seed), layout_seed=layout_seed,
                     epistasis_params=ep, dependency_params=dp, **SELECTION)


def test_planted_network_is_exactly_as_specified():
    net = _planted_selection().epistasis
    assert net.planted and net.n_events == 6 and net.event_size == 5
    assert np.array_equal(net.beta, np.full(6, 0.05))
    expected_E = np.zeros((6, 6))
    expected_E[0, 1] = expected_E[1, 0] = 0.4
    assert np.array_equal(net.E, expected_E)                  # the one interaction, nothing else
    assert net.true_edges() == [(0, 1, 0.4)]
    assert net.true_exclusive_pairs() == [(2, 3)]
    assert net.true_dag_edges() == [(4, 5)]
    assert net.dag_parents[5] == (4,) and all(net.dag_parents[i] == () for i in range(5))
    assert net.gating_mode == "accessibility"
    assert net.lethal and net.lethal_death_rate == 2.5
    gt = net.ground_truth()
    assert gt["planted"] is True and gt["exclusive_pairs"] == [(2, 3)]
    # per-event effects can be given one by one
    per = _planted_selection({**PLANTED_EP, "planted": {"effects": [0.1, 0, 0, 0, 0, 0.2]}}).epistasis
    assert np.array_equal(per.beta, [0.1, 0, 0, 0, 0, 0.2])
    assert per.true_edges() == [] and per.true_dag_edges() == [] and per.true_exclusive_pairs() == []


def test_planted_fitness_is_beta_plus_the_planted_terms():
    net = _planted_selection().epistasis
    assert net.log_fitness(events_to_bits([0])) == pytest.approx(0.05)
    assert net.log_fitness(events_to_bits([0, 1])) == pytest.approx(0.05 + 0.05 + 0.4)
    assert net.log_fitness(events_to_bits([2, 3])) == pytest.approx(0.1)   # exclusivity has no E term
    assert net.log_fitness(events_to_bits([4, 5])) == pytest.approx(0.1)


def test_planted_modules_are_the_random_modes_modules():
    """Only the event modules are drawn, from the same sub-stream and first: planting the network
    leaves the modules and every other gene role exactly where the random mode puts them."""
    planted = _planted_selection()
    random = _planted_selection({"n_events": 6, "event_size": 5}, None)
    for a, b in zip(planted.epistasis.event_genes, random.epistasis.event_genes):
        assert np.array_equal(a, b)
    for get in ("get_oncogenes", "get_tsgs", "get_dispersal_genes", "get_treatment_resistant"):
        assert list(getattr(planted, get)()) == list(getattr(random, get)())
    # and the modules still follow the layout seed (a replicate network is a different layout seed)
    other = _planted_selection(layout_seed=7)
    assert not all(np.array_equal(a, b) for a, b in
                   zip(planted.epistasis.event_genes, other.epistasis.event_genes))


@pytest.mark.parametrize("ep, dp, match", [
    ({**PLANTED_EP, "n_interactions": 2}, PLANTED_DP, "random-draw"),
    (PLANTED_EP, {"gating_mode": "accessibility", "n_constraints": 1}, "random-draw"),
    ({**PLANTED_EP, "planted": {"interaction": [(0, 1, 0.4)]}}, None, "unknown planted"),
    ({**PLANTED_EP, "planted": {"interactions": [(0, 0, 0.4)]}}, None, "itself"),
    ({**PLANTED_EP, "planted": {"interactions": [(0, 6, 0.4)]}}, None, "out of range"),
    ({**PLANTED_EP, "planted": {"interactions": [(0, 1)]}}, None, "3-tuples"),
    ({**PLANTED_EP, "planted": {"interactions": [(0, 1, 0.4), (1, 0, 0.1)]}}, None, "twice"),
    ({**PLANTED_EP, "planted": {"interactions": [(0, 1, 0.4)], "exclusive": [(1, 0)]}}, None,
     "also an interaction"),
    ({**PLANTED_EP, "planted": {"constraints": [(0, 1), (1, 2), (2, 0)]}}, None, "cycle"),
    ({**PLANTED_EP, "planted": {"effects": [0.1, 0.2]}}, None, "one value per event"),
    ({**PLANTED_EP, "mutual_exclusivity_lethal": False}, PLANTED_DP, "lethality"),
])
def test_planted_network_rejects_what_it_cannot_plant_exactly(ep, dp, match):
    with pytest.raises(ValueError, match=match):
        _planted_selection(ep, dp)


def test_planted_exclusive_carrier_takes_the_lethal_death_rate():
    sel = _planted_selection()
    assert sel.update_death_rate({}, 0.02, event_bits=events_to_bits([2, 3])) == 2.5
    assert sel.update_death_rate({}, 0.02, event_bits=events_to_bits([2])) == 0.02
    assert sel.update_death_rate({}, 0.02, event_bits=events_to_bits([3, 4, 5])) == 0.02


def test_lethal_rate_above_the_death_clamp_is_rejected():
    """Every death rate is clamped to maximum_death_rate, so a lethal rate above it would silently
    become the clamp; the tumour refuses to build instead."""
    with pytest.raises(ValueError, match="maximum_death_rate"):
        _tumor(epistasis_params=PLANTED_EP, dependency_params=PLANTED_DP)       # clamp 1.0 < 2.5
    t = GenotypeTumor(seed=1, genome_params=GENOME,
                      selection_params={**SELECTION, "epistasis_params": PLANTED_EP,
                                        "dependency_params": PLANTED_DP},
                      cancer_cell_params=CANCER, deme_params={**DEME, "maximum_death_rate": 2.5},
                      spatial_params=SPATIAL)
    assert t.selection.epistasis.planted


# A regime where events co-occur often without selection (neutral drivers, slow growth), so a planted
# lethal pair has many would-be carriers to remove.
LETHAL_DEME = {**RECOVERY_DEME, "maximum_death_rate": 2.5}


def _lethal_arm(seed, planted):
    # modules of 25 of the ~100 drivers: both events are common enough to co-occur often unchecked
    ep = {"n_events": 2, "event_size": 25, "lethal_death_rate": 2.5, "planted": planted}
    t = GenotypeTumor(seed=seed, genome_params=RECOVERY_GENOME,
                      selection_params={**RECOVERY_SELECTION, "epistasis_params": ep},
                      cancer_cell_params=RECOVERY_CANCER, deme_params=LETHAL_DEME,
                      spatial_params=RECOVERY_SPATIAL)
    t.grow(n_steps=500, seed=seed, materialize=False)
    return t


def _carriers(t, i=0, j=1):
    both = sum(c for g, c in t.genotypes_counts.items() if t._is_cancer(g)
               and t.genotypes[g].event_bits >> i & 1 and t.genotypes[g].event_bits >> j & 1)
    total = sum(c for g, c in t.genotypes_counts.items() if t._is_cancer(g))
    return both, total


def test_planted_exclusive_pair_carriers_die_in_a_grown_tumour():
    """Same seeds, same modules: with the pair planted exclusive, the cells carrying both events are
    removed (every such genotype dies at the lethal rate), against a control with nothing planted."""
    both_lethal = both_ctl = 0
    for seed in (1, 2, 3, 4, 5):
        lethal, ctl = _lethal_arm(seed, {"exclusive": [(0, 1)]}), _lethal_arm(seed, {})
        for g, rep in lethal.genotypes.items():
            if rep.type == "cancer" and rep.event_bits == 0b11:
                assert rep.evolutionary_parameters["death_rate"] == 2.5
        b, n = _carriers(lethal)
        both_lethal += b
        b, n = _carriers(ctl)
        both_ctl += b
    assert both_ctl > 20, "the control regime has too few double carriers to test removal"
    assert both_lethal < 0.1 * both_ctl


def test_planted_constraint_is_respected_along_every_lineage():
    """E5 accessible only after E4 (accessibility gating): no lineage carries the child without its
    parent, and wherever both are carried the parent came first."""
    ep = {"n_events": 2, "event_size": 10, "planted": {"constraints": [(0, 1)]}}
    tumors = _recovery_cohort(n=6, epistasis_params=ep,
                              dependency_params={"gating_mode": "accessibility"})
    n_child = 0
    for t in tumors:
        for events in t.event_table()["events"]:
            if 1 in events:
                n_child += 1
                assert 0 in events
    assert n_child > 0, "the child event never arose: the regime cannot test the constraint"
    order = score_order([(0, 1)], [g for t in tumors for g in t.event_table()["event_groups"]])
    assert order["constraint_satisfaction"] == 1.0
    assert order["n_scored"] == 0 or order["order_accuracy"] == 1.0


def test_planted_network_is_reachable_through_selection_params():
    """A planted network needs no hook after construction: it rides in selection_params, which is
    all the paper regime's grow passes on (see test_cohort's regime test for the full path)."""
    t = GenotypeTumor(seed=2, genome_params=GENOME,
                      selection_params={**SELECTION, "epistasis_params": PLANTED_EP,
                                        "dependency_params": PLANTED_DP},
                      cancer_cell_params=CANCER, deme_params={**DEME, "maximum_death_rate": 2.5},
                      spatial_params=SPATIAL)
    gt = t.epistasis_ground_truth()
    assert gt["planted"] and gt["edges"] == [(0, 1, 0.4)] and gt["dag_edges"] == [(4, 5)]


# ================================ the share of cells at the cap ================================
def test_division_cap_share():
    from iscc.tumor.diagnostics import division_cap_share
    # neutral drivers and b0 below the cap: nobody can reach it
    t = _grown(seed=1, steps=150, driver_effects=1.0)
    assert division_cap_share(t) == 0.0
    # b0 AT the cap: every cell is there from the start
    at = GenotypeTumor(seed=1, genome_params=GENOME,
                       selection_params={**SELECTION, "driver_effects": 1.0},
                       cancer_cell_params={**CANCER, "division_rate": CANCER["max_birth_rate"]},
                       deme_params=DEME, spatial_params=SPATIAL)
    at.grow(n_steps=60, seed=1, materialize=False)
    assert division_cap_share(at) == 1.0
    # a mixed tumour: the share is the count-weighted fraction whose uncapped rate reaches the cap,
    # and reading it changes nothing
    t = _grown(seed=4, steps=200, driver_effects=1.2)
    counts = dict(t.genotypes_counts)
    n = at_cap = 0
    for g, c in counts.items():
        if not t._is_cancer(g):
            continue
        rep = t.genotypes[g]
        u = rep.baseline_rates["division_rate"] * t.selection.update_division_rate(
            rep.genome_summary, event_bits=rep.event_bits)
        n += c
        at_cap += c * (u >= rep.max_birth_rate - 1e-12)
    share = division_cap_share(t)
    assert share == pytest.approx(at_cap / n)
    assert 0.0 <= share <= 1.0 and dict(t.genotypes_counts) == counts


# =============================== event exports for Fig 5 ===============================
def test_presence_threshold_is_explicit():
    """No default: the old default (any surviving cell) saturates the matrix, so it must be stated."""
    from iscc.integrations import DETECTION_LIMIT
    t = _grown(seed=1, epistasis_params=EPI)
    with pytest.raises(TypeError):
        to_mhn_matrix([t])
    assert DETECTION_LIMIT == 0.05
    frac = event_cell_fractions(t)
    assert np.array_equal(to_mhn_matrix([t], min_freq=DETECTION_LIMIT).values[0],
                          (frac >= 0.05).astype(int))


def test_trie_detection_collapses_nodes_below_the_limit():
    """A node is kept when the cells through it (with its descendants) reach the limit; a dropped
    node's cells go to its deepest kept ancestor."""
    from iscc.integrations.progression import _trie
    paths = [((0, 1), 60), ((0,), 30), ((2,), 4), ((0, 2), 6)]
    full = _trie(paths, total=100)
    assert len(full) == 5                                       # root, 0, 0>1, 2, 0>2
    cut = _trie(paths, total=100, detection=0.05)
    rows = {(r.Parent_ID, r.Mutation_ID): r for r in cut.itertuples()}
    assert set(rows) == {(1, 0), (1, 1), (2, 2), (2, 3)}        # root, 0, 0>1, 0>2; not 2 (4%)
    assert rows[(1, 0)].n_cells == 4                            # the 4% clone merged into the root
    assert cut["n_cells"].sum() == 100
    assert _trie(paths, total=100, detection=0.07)["Mutation_ID"].tolist() == [0, 1, 2]


def test_collapsed_tree_keeps_only_clones_above_the_limit():
    tumors = _cohort_tumors(_cohort(n=4, steps=200))
    for t in tumors:
        tbl = t.event_table()
        total = tbl["n_cells"].sum()
        # without a limit every distinct acquisition path is a node (the export as it always was)
        prefixes = {tuple(o[:k]) for o in tbl["event_order"] for k in range(1, len(o) + 1)}
        assert len(to_mutation_tree(t)) == len(prefixes) + 1
        tree = to_mutation_tree(t, detection=0.05)
        assert tree["n_cells"].sum() == total
        # every kept node's path is carried (as a path prefix) by >= 5% of the cancer cells
        nodes = {int(r.Node_ID): r for r in tree.itertuples()}
        for nid, r in nodes.items():
            if nid == 1:
                continue
            path, cur = [], nid
            while cur != 1:
                path.append(int(nodes[cur].Mutation_ID) - 1)
                cur = int(nodes[cur].Parent_ID)
            path = tuple(reversed(path))
            through = sum(n for order, n in zip(tbl["event_order"], tbl["n_cells"])
                          if tuple(order[:len(path)]) == path)
            assert through / total >= 0.05
    trees = to_treemhn_trees(tumors, detection=0.05)
    ids = sorted(trees["Patient_ID"].unique())
    assert ids == list(range(1, len(ids) + 1))
    # trees stored one patient at a time stack to the same frame
    from iscc.integrations import stack_trees
    stored = [to_mutation_tree(t, detection=0.05) for t in tumors]
    assert stack_trees(stored).equals(trees)


# A tumour whose cells acquire the three dissemination traits by mutation.
TRAITS_SEL = dict(prop_breach=0.05, prop_stromal_survival=0.05, prop_met_survival=0.05,
                  trait_source="mutation")


def _lineage_traits(t, gid, traits):
    """Brute force: the traits any genotype on gid's lineage (itself included) carried."""
    from iscc.integrations.progression import _TRAIT_TALLY
    got, cur = set(), gid
    while cur is not None and cur in t.genotypes and t._is_cancer(cur):
        gs = t.genotypes[cur].genome_summary
        got |= {k for k, name in enumerate(traits) if gs.get(_TRAIT_TALLY[name], 0) > 0}
        cur = t.genotypes_parents.get(cur)
    return got


def test_trait_table_reads_acquisitions_off_the_genealogy():
    from iscc.integrations import TRAIT_EVENTS, trait_table, trait_cell_fractions
    t = _grown(seed=2, steps=250, **TRAITS_SEL)
    tbl = trait_table(t)
    assert tbl.attrs["names"] == list(TRAIT_EVENTS)
    assert tbl["n_cells"].sum() == t.get_cancer_size()
    n_with = 0
    for gid, events, order, groups in zip(tbl["genotype_id"], tbl["events"], tbl["event_order"],
                                          tbl["event_groups"]):
        assert set(events) == _lineage_traits(t, gid, TRAIT_EVENTS)
        assert tuple(e for g in groups for e in g) == order and sorted(order) == sorted(events)
        n_with += bool(events)
    assert n_with > 0, "no trait acquired: the regime cannot test the adapter"
    frac = trait_cell_fractions(t)
    for k, name in enumerate(TRAIT_EVENTS):
        carried = tbl["events"].apply(lambda ev: k in ev)
        assert frac[name] == pytest.approx(tbl.loc[carried, "n_cells"].sum() / tbl["n_cells"].sum())


def test_trait_matrix_and_trees():
    from iscc.integrations import (TRAIT_EVENTS, to_trait_matrix, to_trait_tree, to_trait_trees,
                                   trait_cell_fractions)
    tumors = [_grown(seed=s, steps=250, **TRAITS_SEL) for s in (2, 3, 4)]
    X = to_trait_matrix(tumors)                                   # default: the 5% limit
    assert list(X.columns) == list(TRAIT_EVENTS) and X.shape == (3, 3)
    for i, t in enumerate(tumors):
        frac = trait_cell_fractions(t).values
        assert np.array_equal(X.values[i], (frac >= 0.05).astype(int))
        assert np.array_equal(to_trait_matrix([t], min_freq=0).values[0], (frac > 0).astype(int))
        tree = to_trait_tree(t)
        assert tree.iloc[0].to_dict() == dict(Node_ID=1, Mutation_ID=0, Parent_ID=1,
                                              n_cells=int(tree.iloc[0]["n_cells"]))
        assert set(tree["Mutation_ID"]) <= set(range(len(TRAIT_EVENTS) + 1))
        assert tree["n_cells"].sum() == t.get_cancer_size()
        # a trait kept in the tree is carried by >= 5% of cells, so it is present in the matrix
        for m in set(tree["Mutation_ID"]) - {0}:
            assert X.values[i, m - 1] == 1
    trees = to_trait_trees(tumors)
    assert {"Patient_ID", "Tree_ID", "Node_ID", "Mutation_ID", "Parent_ID"} <= set(trees.columns)


def test_traits_with_events_agree_with_the_event_machinery():
    """Read together with an epistasis network, the event part of the trait adapter is exactly the
    engine's own event_table: same event sets, same acquisition groups."""
    from iscc.integrations import trait_table, to_trait_matrix
    t = _grown(seed=3, steps=250, epistasis_params=EPI, **TRAITS_SEL)
    n = EPI["n_events"]
    tbl = trait_table(t, with_events=True)
    assert tbl.attrs["names"][:n] == t.selection.epistasis.event_names()
    tbl = tbl.set_index("genotype_id")
    ev = t.event_table().set_index("genotype_id")
    for gid, row in ev.iterrows():
        assert tuple(e for e in tbl.loc[gid, "events"] if e < n) == row["events"]
        groups = tuple(g for g in (tuple(e for e in grp if e < n) for grp in tbl.loc[gid, "event_groups"]) if g)
        assert groups == row["event_groups"]
    X = to_trait_matrix([t], with_events=True)
    assert np.array_equal(X.values[0, :n], to_mhn_matrix([t], min_freq=0.05).values[0])
