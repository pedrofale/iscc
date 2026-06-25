"""Noble evolutionary-mode indices (n, D, J1) on the genotype engine (DESIGN_inference.md M0/M0b)."""
import math

import numpy as np

from conftest import GENOME_PARAMS, SELECTION_PARAMS, CANCER_CELL_PARAMS, DEME_PARAMS
from iscc.tumor.models import GenotypeTumor
from iscc.inference import (
    inverse_simpson,
    driver_combination_counts,
    mean_drivers_per_cell,
    clonal_diversity,
    tree_balance_j1,
    clone_tree,
    tree_balance,
    mode_indices,
)

SPATIAL = {"grid_size": 12, "structure_radius": 0}
CANCER = {**CANCER_CELL_PARAMS, "death_rate": 0.02}


# --- pure math: inverse Simpson ---------------------------------------------
def test_inverse_simpson_equal_groups_is_group_count():
    # k equally abundant groups -> effective number = k
    assert inverse_simpson([5, 5, 5, 5]) == 4.0
    assert inverse_simpson([1]) == 1.0


def test_inverse_simpson_dominated_approaches_one():
    # one dominant group dwarfing the rest -> close to 1
    assert inverse_simpson([1000, 1, 1]) < 1.05
    # 3:1 split -> 1 / (0.75^2 + 0.25^2) = 1.6
    assert abs(inverse_simpson([3, 1]) - 1.6) < 1e-9


def test_inverse_simpson_empty_is_zero():
    assert inverse_simpson([]) == 0.0
    assert inverse_simpson([0, 0]) == 0.0


# --- pure math: J1 tree balance (Lemant et al. 2022) ------------------------
# Hand-computed small trees. Sizes default to 0 for any node absent from `sizes`.
def test_j1_single_node_is_nan():
    # no branching mass anywhere -> balance undefined
    assert math.isnan(tree_balance_j1({}, {"a": 1}))
    assert math.isnan(tree_balance_j1({}, {}))


def test_j1_star_tree_is_one():
    # root with k equal-size leaves: one perfectly balanced split -> J1 == 1
    parents = {"a": "r", "b": "r", "c": "r"}
    sizes = {"a": 1, "b": 1, "c": 1}          # root size 0
    assert abs(tree_balance_j1(parents, sizes) - 1.0) < 1e-12


def test_j1_caterpillar_chain_is_zero():
    # pure single-child chain r->a->b: every internal node has outdegree 1 -> J1 == 0
    parents = {"a": "r", "b": "a"}
    sizes = {"r": 1, "a": 1, "b": 1}
    assert tree_balance_j1(parents, sizes) == 0.0


def test_j1_two_unequal_leaves_matches_normalised_entropy():
    # root -> A(1), B(3): single split, J1 = H(1/4, 3/4) / log 2
    parents = {"A": "r", "B": "r"}
    sizes = {"A": 1, "B": 3}
    p = np.array([0.25, 0.75])
    expected = -(p * np.log(p)).sum() / math.log(2)
    assert abs(tree_balance_j1(parents, sizes) - expected) < 1e-12
    assert abs(expected - 0.8112781) < 1e-6      # sanity on the literal value


def test_j1_internal_own_size_excluded_from_own_entropy_but_counts_for_parent():
    # root(0) -> A(leaf 2), I(own size 10); I -> B(1), C(1).
    # I's own 10 must NOT enter I's balance (B,C are equal -> W_I = 1), but it MUST
    # enter S_I = 12 used in root's split. A wrong impl that folds m_i into the entropy
    # would not give W_I = 1 nor this J1.
    parents = {"A": "r", "I": "r", "B": "I", "C": "I"}
    sizes = {"A": 2, "I": 10, "B": 1, "C": 1}
    # root split over S_A=2, S_I=12 (star=14); I split over S_B=1,S_C=1 (star=2, W=1)
    pr = np.array([2.0, 12.0]) / 14.0
    w_root = -(pr * np.log(pr)).sum() / math.log(2)
    expected = (14.0 * w_root + 2.0 * 1.0) / (14.0 + 2.0)
    got = tree_balance_j1(parents, sizes)
    assert abs(got - expected) < 1e-12
    assert abs(got - 0.6427137) < 1e-6


def test_j1_single_child_node_weighs_denominator_only():
    # root -> X (single child), X -> A,B,C equal. The unary root contributes its weight
    # (star = S_X) to the denominator with zero balance, pulling J1 below the X split's 1.
    parents = {"X": "r", "A": "X", "B": "X", "C": "X"}
    sizes = {"A": 1, "B": 1, "C": 1}
    s_x = 3.0                                  # X subtree magnitude
    expected = (s_x * 1.0 + s_x * 0.0) / (s_x + s_x)   # X split (W=1) + unary root (W=0)
    assert abs(tree_balance_j1(parents, sizes) - expected) < 1e-12
    assert abs(tree_balance_j1(parents, sizes) - 0.5) < 1e-12


def test_j1_balanced_beats_imbalanced():
    bal = tree_balance_j1({"a": "r", "b": "r"}, {"a": 1, "b": 1})
    imb = tree_balance_j1({"a": "r", "b": "r"}, {"a": 1, "b": 99})
    assert 0.0 <= imb < bal <= 1.0


# --- grouping by driver combination -----------------------------------------
def _run(seed, steps=300, dispersal=0.2):
    cancer = {**CANCER, "dispersal_rate": dispersal}
    t = GenotypeTumor(
        seed=seed, genome_params=GENOME_PARAMS, selection_params=SELECTION_PARAMS,
        cancer_cell_params=cancer, deme_params=DEME_PARAMS, spatial_params=SPATIAL,
    )
    t.grow(n_steps=steps, seed=seed)
    return t


def test_combination_counts_match_cancer_population():
    t = _run(0)
    combos = driver_combination_counts(t)
    # total over combinations == cancer cell count, and no normal cells leak in
    assert sum(combos.values()) == t.get_cancer_size()
    assert all(isinstance(k, frozenset) for k in combos)


def test_founder_only_population_is_single_clone():
    # one undivided cancer cell -> one combination (the founder's), D == 1, n == its driver load
    t = GenotypeTumor(
        seed=0, genome_params=GENOME_PARAMS, selection_params=SELECTION_PARAMS,
        cancer_cell_params=CANCER, deme_params=DEME_PARAMS, spatial_params=SPATIAL,
    )
    combos = driver_combination_counts(t)
    assert sum(combos.values()) == 1
    assert clonal_diversity(t) == 1.0
    idx = mode_indices(t)
    assert idx["n_clones"] == 1
    assert idx["D"] == 1.0
    # a single clone has no branching, so J1 is undefined (nan)
    parents, sizes = clone_tree(t)
    assert parents == {} and sum(sizes.values()) == 1
    assert math.isnan(idx["J1"]) and math.isnan(tree_balance(t))


def test_mean_drivers_is_count_weighted_combination_size():
    # n must equal the count-weighted mean combination size of the same grouping
    t = _run(1)
    combos = driver_combination_counts(t)
    total = sum(combos.values())
    expected = sum(len(k) * v for k, v in combos.items()) / total
    assert abs(mean_drivers_per_cell(t) - expected) < 1e-9
    assert mean_drivers_per_cell(t) >= 0.0


def test_mode_indices_consistency_and_bounds():
    t = _run(2)
    idx = mode_indices(t)
    n_combos = idx["n_clones"]
    # D is an effective group count: 1 <= D <= number of distinct combinations
    assert 1.0 <= idx["D"] <= n_combos + 1e-9
    assert np.isfinite(idx["n"]) and idx["n"] >= 0.0
    # the one-pass mode_indices agrees with the standalone helpers
    assert abs(idx["D"] - clonal_diversity(t)) < 1e-9
    assert abs(idx["n"] - mean_drivers_per_cell(t)) < 1e-9
    # J1 in [0,1] (or nan if a single clone), and consistent with the standalone helper
    if n_combos > 1:
        assert 0.0 <= idx["J1"] <= 1.0 + 1e-9
        assert abs(idx["J1"] - tree_balance(t)) < 1e-9
    else:
        assert math.isnan(idx["J1"])


def test_clone_tree_is_a_rooted_tree_with_conserved_mass():
    # the contracted clone phylogeny: cell mass is conserved, exactly one root, acyclic
    t = _run(2)
    parents, sizes = clone_tree(t)
    assert sum(sizes.values()) == t.get_cancer_size()
    roots = [n for n in sizes if n not in parents]
    assert len(roots) == 1                       # founder clone
    # every parent edge points to a known clone node; no cycles (walk to root terminates)
    for node in sizes:
        seen, cur = set(), node
        while cur in parents:
            assert cur not in seen
            seen.add(cur)
            cur = parents[cur]
        assert cur == roots[0]


def test_no_cancer_gives_nan_indices():
    t = GenotypeTumor(
        seed=0, genome_params=GENOME_PARAMS, selection_params=SELECTION_PARAMS,
        cancer_cell_params=CANCER, deme_params=DEME_PARAMS, spatial_params=SPATIAL,
    )
    # wipe the founder so no cancer remains
    t.genotypes_counts.clear()
    for d in t.demes:
        for g in list(d):
            if t._is_cancer(g):
                del d[g]
    idx = mode_indices(t)
    assert idx["n_clones"] == 0
    assert np.isnan(idx["n"]) and np.isnan(idx["D"]) and np.isnan(idx["J1"])
