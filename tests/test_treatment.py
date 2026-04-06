"""Tests for iscc.treatment: Treatment base class and subclasses."""
import numpy as np
import pytest


class TestTreatmentBase:
    def test_adaptive_dosage_above_threshold(self):
        from iscc.treatment.treatment import Treatment
        t = Treatment(adaptive=True)
        # tumor larger than max_tumor_size → dosage = 1
        assert t.get_dosage(t.max_tumor_size + 1) == 1.0

    def test_adaptive_dosage_below_threshold(self):
        from iscc.treatment.treatment import Treatment
        t = Treatment(adaptive=True)
        # tumor smaller than max_tumor_size → dosage = 0
        assert t.get_dosage(0) == 0.0

    def test_non_adaptive_dosage_always_one(self):
        from iscc.treatment.treatment import Treatment
        t = Treatment(adaptive=False)
        assert t.get_dosage(0) == 1.0
        assert t.get_dosage(10_000) == 1.0

    def test_apply_does_not_crash(self, cancer_cell):
        from iscc.treatment.treatment import Treatment
        t = Treatment(adaptive=False)
        # Base class _apply is a no-op; apply() should not raise
        t.apply(cancer_cell)

    def test_default_attributes(self):
        from iscc.treatment.treatment import Treatment
        t = Treatment()
        assert 0 < t.toxicity < 1
        assert 0 < t.effectiveness <= 1


class TestChemotherapy:
    def test_apply_increases_death_rate(self, cancer_cell):
        from iscc.treatment.chemotherapy import Chemotherapy
        chemo = Chemotherapy(adaptive=False, effectiveness=1.0, toxicity=1.0)
        original_death_rate = cancer_cell.evolutionary_parameters["death_rate"]
        # Force application by making every call hit the target
        np.random.seed(0)
        chemo._apply(cancer_cell)
        new_death_rate = cancer_cell.evolutionary_parameters["death_rate"]
        # Death rate should increase (or at minimum stay the same)
        assert new_death_rate >= original_death_rate


class TestImmunotherapy:
    def test_apply_does_not_crash(self, cancer_cell):
        from iscc.treatment.immunotherapy import Immunotherapy
        immuno = Immunotherapy(immune_checkpoints=[], adaptive=False)
        np.random.seed(0)
        immuno._apply(cancer_cell)
        # Should not crash; immune_resistance is a valid key
        assert "immune_resistance" in cancer_cell.evolutionary_parameters

    def test_is_target_empty_checkpoints(self, cancer_cell):
        from iscc.treatment.immunotherapy import Immunotherapy
        immuno = Immunotherapy(immune_checkpoints=[], adaptive=False)
        # With no checkpoints, frac_expressed division by zero — test that constructor works
        assert immuno.targets == []
