"""OncoSpot's expression law, as used for the OncoSpot arm of Fig 3f (validation/paper/oncospot_law.py)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "validation", "paper"))
import oncospot_law as OS  # noqa: E402


def test_expression_follows_the_power_law_and_is_zero_at_zero_copies():
    base = np.array([1.0, 2.0, 0.5])
    cn = np.array([[2, 4, 0], [1, 2, 3]])
    mu = OS.expression(base, cn, beta=0.4)
    expected = base[None, :] * np.array([[1.0, 2 ** 0.4, 0.0], [0.5 ** 0.4, 1.0, 1.5 ** 0.4]])
    assert np.allclose(mu, expected)


def test_expression_takes_a_per_gene_beta_and_a_state_multiplier():
    base = np.ones(2)
    cn = np.array([[4, 4]])
    mult = np.array([[2.0, 0.5]])
    mu = OS.expression(base, cn, beta=np.array([1.0, 0.0]), state_multiplier=mult)
    assert np.allclose(mu, [[4.0, 0.5]])


def test_modal_profiles_pick_the_commonest_profile_and_break_ties_by_sort_order():
    clone_of = {"a": "c1", "b": "c1", "c": "c1", "d": "c2", "e": "c2"}
    profile_of = {"a": (2, 3), "b": (2, 3), "c": (2, 2), "d": (3, 2), "e": (2, 1)}
    assert OS.modal_profiles(clone_of, profile_of) == {"c1": (2, 3), "c2": (2, 1)}


def test_permute_rows_keeps_the_rows_and_breaks_their_order():
    state = np.arange(200).reshape(100, 2)
    out = OS.permute_rows(state, np.random.default_rng(0))
    assert sorted(map(tuple, out)) == sorted(map(tuple, state))
    assert not np.array_equal(out, state)


def test_matched_beta_is_one_for_linear_dosage_and_below_one_when_buffered():
    beta = OS.matched_beta(np.array([1.0, 0.5, 0.0]))
    assert np.isclose(beta[0], 1.0)
    assert beta[2] < beta[1] < 1.0
