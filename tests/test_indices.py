"""Noble evolutionary-mode indices (n, D) on the genotype engine (DESIGN_inference.md M0)."""
import numpy as np

from conftest import GENOME_PARAMS, SELECTION_PARAMS, CANCER_CELL_PARAMS, DEME_PARAMS
from iscc.tumor.models import GenotypeTumor
from iscc.inference import (
    inverse_simpson,
    driver_combination_counts,
    mean_drivers_per_cell,
    clonal_diversity,
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
    assert np.isnan(idx["n"]) and np.isnan(idx["D"])
