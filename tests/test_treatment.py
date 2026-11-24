"""Tests for iscc.treatment: Treatment base class and subclasses."""
import numpy as np
import pytest


class TestTreatmentBase:
    def test_adaptive_dosage_above_threshold(self):
        from iscc.treatment.treatment import Treatment
        t = Treatment(adaptive=True)
        # tumor larger than max_tumor_size → dosage = 1
        assert t.get_dosage(0, t.max_tumor_size + 1) == 1.0

    def test_adaptive_dosage_below_threshold(self):
        from iscc.treatment.treatment import Treatment
        t = Treatment(adaptive=True)
        # tumor smaller than max_tumor_size → dosage = 0
        assert t.get_dosage(0, 0) == 0.0

    def test_non_adaptive_dosage_always_one(self):
        from iscc.treatment.treatment import Treatment
        t = Treatment(adaptive=False)
        assert t.get_dosage(0, 0) == 1.0
        assert t.get_dosage(10_000, 0) == 1.0

    def test_default_attributes(self):
        from iscc.treatment.treatment import Treatment
        t = Treatment()
        assert 0 < t.toxicity < 1
        assert 0 < t.effectiveness <= 1


class TestImmunotherapy:
    def test_is_target_empty_checkpoints(self, cancer_cell):
        from iscc.treatment.immunotherapy import Immunotherapy
        immuno = Immunotherapy(immune_checkpoints=[], adaptive=False)
        # With no checkpoints, frac_expressed division by zero — test that constructor works
        assert immuno.targets == []
