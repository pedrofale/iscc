"""The Muller layout places every clone once, whatever its id.

pymuller's own ordering started a subtree at every identity in sorted string order, so with the
founder at ``"2"`` the clones ``"10"`` to ``"19"``, ``"100"`` to ``"199"``, ... were laid out twice and
their cells counted twice.
"""
from collections import Counter

import numpy as np
import pandas as pd
import pymuller
import pytest

from iscc.tumor import viz

PARENTS = {"10": "2", "11": "2", "3": "2", "12": "10", "100": "12"}


def _adjacency(parents=PARENTS):
    return pd.DataFrame({"Identity": list(parents), "Parent": list(parents.values())})


def test_every_clone_appears_once_as_a_nested_pair():
    order = list(viz._strains_ordering(_adjacency()))
    assert Counter(order) == {s: 2 for s in ["2", "10", "11", "3", "12", "100"]}
    assert order[0] == order[-1] == "2"
    assert order.index("12") > order.index("10") and order.index("100") > order.index("12")


def test_pymuller_uses_the_fixed_ordering():
    assert pymuller.logic._get_strains_ordering is viz._strains_ordering


def test_unnormalised_stack_sums_to_the_population():
    ids = ["2", "10", "11", "3", "12", "100"]
    rng = np.random.default_rng(0)
    counts = pd.DataFrame(rng.integers(1, 50, size=(8, len(ids))), columns=ids)
    pop_df, anc_df, _ = viz._prepare(counts.copy(), pd.DataFrame([PARENTS]))
    y = pymuller.logic._get_y_values(pop_df, anc_df, 1e-6)
    np.testing.assert_allclose(y.sum(axis=1).to_numpy(), counts.sum(axis=1).to_numpy(), rtol=1e-6)


def test_deep_tree_needs_no_recursion():
    parents = {str(i + 1): str(i) for i in range(2, 5000)}
    order = viz._strains_ordering(_adjacency(parents))
    assert len(order) == 2 * 4999 and order[0] == "2"


def test_muller_of_a_tau_tumour_is_drawn_in_generations():
    import matplotlib
    matplotlib.use("Agg")
    from conftest import GENOME_PARAMS, SELECTION_PARAMS, CANCER_CELL_PARAMS, DEME_PARAMS
    from iscc.tumor import GenotypeTumor
    t = GenotypeTumor(seed=3, genome_params=GENOME_PARAMS, selection_params=SELECTION_PARAMS,
                      cancer_cell_params=CANCER_CELL_PARAMS, deme_params=DEME_PARAMS,
                      spatial_params={"grid_size": 10}, update_mode="tau", tau=0.5)
    t.grow(n_steps=40, seed=3)
    ax = t.plot_muller()
    assert ax.get_xlabel() == "Generation"
    assert max(ax.collections[0].get_paths()[0].vertices[:, 0]) == pytest.approx(t.trace_times[-1])
