"""Tests for the operating-envelope QC (iscc.tumor.diagnostics / GenotypeTumor.diagnose).

Checks the three contract points from DESIGN_operating_envelope.md:
  * a realistic config passes (no flags);
  * each deliberately-degenerate config trips the RIGHT flag with an actionable hint
    (extinct / monoclonal / well-mixed / no-microenvironment-gradient / hypermutated);
  * ``diagnose()`` is READ-ONLY — it does not alter the tumour's state or growth trajectory.
"""
import copy

import numpy as np
import pytest

from iscc.tumor.models import GenotypeTumor

GENOME = {"n_segments": 3, "segment_size": 60}
SELECTION = {"prop_driver": 0.1, "driver_effects": 1.1}
CANCER = {"max_birth_rate": 0.8, "division_rate": 0.4, "death_rate": 0.02, "mutation_rate": 0.3,
          "dispersal_rate": 0.1, "snv_prob": 0.5, "cnv_prob": 0.5, "n_snvs_per_allele": 0.4,
          "amp_prob": 0.5}
DEME = {"carrying_capacity": 4, "initial_cancer_cells": 5, "maximum_death_rate": 0.5}
SPATIAL = {"grid_size": 16, "structure_radius": 0}


def _grow(cancer=None, deme=None, spatial=None, microenv=None, steps=900, seed=0):
    t = GenotypeTumor(
        genome_params=GENOME, selection_params=SELECTION,
        cancer_cell_params={**CANCER, **(cancer or {})},
        deme_params={**DEME, **(deme or {})},
        spatial_params={**SPATIAL, **(spatial or {})},
        microenv_params=microenv, seed=seed)
    t.grow(n_steps=steps, seed=seed)
    return t


def _flags(diag):
    return {c.name for c in diag.failures}


# --- a good config passes ----------------------------------------------------
def test_good_config_passes():
    diag = _grow().diagnose()
    assert diag.ok, f"realistic config unexpectedly flagged: {_flags(diag)}"
    assert diag.failures == []
    # metrics are populated and sane
    assert diag["n_cancer"] > 50
    assert diag["shannon"] > 1.0
    assert 0.0 <= diag["clone_confinement"] <= 1.0


# --- each degenerate regime trips its flag -----------------------------------
def test_extinct_flag():
    diag = _grow(cancer={"death_rate": 0.5}, deme={"initial_cancer_cells": 1}).diagnose()
    assert "extinct" in _flags(diag)
    hint = [c.hint for c in diag.failures if c.name == "extinct"][0]
    assert "death_rate" in hint  # actionable


def test_monoclonal_low_mutation_flag():
    diag = _grow(cancer={"mutation_rate": 0.0005}).diagnose()
    flags = _flags(diag)
    assert "monoclonal" in flags
    # the hint attributes the culprit to the mutation supply
    hint = [c.hint for c in diag.failures if c.name == "monoclonal"][0]
    assert "mutation_rate" in hint


def test_well_mixed_flag():
    # high dispersal relative to division -> clones smear across the whole lesion (no territories),
    # the regime that breaks the PEtracer & multi-region benchmarks.
    diag = _grow(cancer={"dispersal_rate": 8.0}, steps=1200).diagnose()
    assert "well_mixed" in _flags(diag)
    assert diag["clone_confinement"] < 0.1


def test_no_gradient_flag():
    # hypoxia on but a long O2 diffusion length (large D) on a small tumour -> no core-rim contrast.
    menv = {"hypoxia": {"strength": 1.0, "n_genes": 30, "o2_supply": 0.3, "o2_source": "uniform",
                        "o2_diffusion": 200.0, "o2_consumption": 1.0}}
    diag = _grow(spatial={"grid_size": 8}, deme={"carrying_capacity": 2}, microenv=menv,
                 steps=500).diagnose()
    assert "no_gradient" in _flags(diag)


def test_gradient_present_passes():
    # control for the above: a short diffusion length gives the classic core-rim gradient.
    menv = {"hypoxia": {"strength": 1.0, "n_genes": 30, "o2_supply": 0.3, "o2_source": "uniform",
                        "o2_diffusion": 1.0, "o2_consumption": 1.0}}
    diag = _grow(microenv=menv, steps=1000).diagnose()
    assert "no_gradient" not in _flags(diag)
    assert diag["hypoxia_contrast"] > 0.05


def test_no_gradient_skipped_when_microenv_off():
    # with the microenvironment off the check is not applicable -> skipped, never failed.
    diag = _grow().diagnose()
    ng = [c for c in diag.checks if c.name == "no_gradient"][0]
    assert ng.skipped and ng.ok


def test_hypermutated_flag():
    diag = _grow(cancer={"mutation_rate": 5.0, "n_snvs_per_allele": 8.0}, steps=1200).diagnose()
    assert "hypermutated" in _flags(diag)


# --- thresholds are overridable ----------------------------------------------
def test_thresholds_overridable():
    t = _grow()
    strict = t.diagnose(thresholds={"shannon_min": 100.0})  # impossible -> forces monoclonal flag
    assert "monoclonal" in _flags(strict)
    lenient = t.diagnose()
    assert "monoclonal" not in _flags(lenient)


# --- diagnose() is read-only: it must not change growth ----------------------
def test_diagnose_does_not_alter_state():
    t = _grow(steps=800)
    before_counts = dict(t.genotypes_counts)
    before_size = t.get_cancer_size()
    before_rng = t.rng.bit_generator.state
    t.diagnose(verbose=True)
    assert dict(t.genotypes_counts) == before_counts
    assert t.get_cancer_size() == before_size
    # the tumour's rng was not consumed (read-only readout, like the F8 ground truth)
    assert t.rng.bit_generator.state == before_rng


def test_diagnose_does_not_alter_trajectory():
    # two identical tumours; diagnose one mid-way, then continue both. The engine draws fresh,
    # step-seeded rngs, so if diagnose leaves state untouched the two trajectories stay identical.
    # Genotype ids are process-global (differ between instances), so we compare the label-invariant
    # multiset of clone sizes rather than the id-keyed dict.
    def sizes(t):
        return sorted(t.genotypes_counts.values())

    a = _grow(steps=700, seed=3)
    b = _grow(steps=700, seed=3)
    assert sizes(a) == sizes(b)
    a.diagnose()  # read-only
    a.grow(n_steps=400, seed=3)
    b.grow(n_steps=400, seed=3)
    assert sizes(a) == sizes(b)


def test_report_is_printable():
    diag = _grow().diagnose()
    text = diag.report()
    assert "iscc tumour diagnosis" in text
    assert str(diag) == text


# --- realistic-size advisory (non-failing) -----------------------------------
def test_small_tumor_advisory_is_non_failing():
    # a small but otherwise healthy tumour is NOT degenerate, but gets a size advisory pointing at
    # the levers to reach a realistic (thousands-of-cells) size.
    t = _grow(steps=400)  # few steps -> few hundred cells
    diag = t.diagnose(thresholds={"min_realistic": 1000})
    assert t.get_cancer_size() < 1000
    assert diag.ok  # advisory must not flip ok
    assert any("small tumour" in a for a in diag.advisories)
    assert "for realistic analyses" in diag.report()


def test_no_advisory_for_realistic_size():
    # lower the advisory floor below the achieved size -> no advisory.
    t = _grow(steps=900)
    diag = t.diagnose(thresholds={"min_realistic": t.get_cancer_size() - 1})
    assert diag.advisories == []
