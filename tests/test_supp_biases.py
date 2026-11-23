"""The supplementary bias figure lists every measurement bias with its default and where it is set.

``validation/paper/supp_biases.py`` names, for each bias, the lines of the package that set its default
and that record its fit. This test runs the figure's own check, so a refactor that moves a default or
a refit that changes one fails here rather than leaving the figure stale.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "validation", "paper"))
import supp_biases as SB  # noqa: E402


def test_every_bias_names_the_lines_that_set_and_fit_its_default():
    assert SB.check() == []
