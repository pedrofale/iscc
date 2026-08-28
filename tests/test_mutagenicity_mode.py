"""``Treatment.mutagenicity_mode`` — who gets the therapy-induced mutator phenotype.

The mutator is what makes de novo resistance likely: an exposed clone's mutation rate is multiplied
once and inherited. Under the default "uniform" it is applied to EVERY cancer clone in a treated
compartment regardless of resistance, so a clone taking zero drug still mutates `mutagenicity`x
faster forever. That is the only drug effect in the engine not scaled by (1 - treatment_resistance),
and its main effect on a resistant clone is to multiply the CNA rate that deletes its own resistance
allele — i.e. the mutator added to make resistance ARISE also makes it revert.

"dose" scales the boost by the dose the clone actually receives, exactly as ``_kill_amount`` scales
the hazard. Covered here:

  * the DEFAULT is unchanged — every exposed clone still gets the full multiplier;
  * under "dose" a fully resistant clone gets NO boost, and a partially resistant one gets a
    proportionate share;
  * a fully protected clone is left unflagged, so a descendant that loses resistance is mutagenized
    on its own terms rather than inheriting an exemption;
  * the boost is applied at most ONCE per lineage under both modes (no per-step compounding).
"""
import numpy as np
import pytest

from iscc.treatment.chemotherapy import Chemotherapy


class FakeRep:
    """Minimal stand-in for a genotype representative: the mutagenicity block reads only these."""
    def __init__(self, tr, mutation_rate=0.3):
        self.mutation_rate = mutation_rate
        self.evolutionary_parameters = {"treatment_resistance": tr, "drug_tolerance": 0.0}


def _factor(mutagenicity, mode, tr):
    """The engine's per-clone multiplier (count.py's mutagenicity block)."""
    if mode != "dose":
        return mutagenicity
    return 1.0 + (mutagenicity - 1.0) * (1.0 - min(max(tr, 0.0), 1.0))


def test_default_mode_is_uniform_and_unscaled():
    tx = Chemotherapy(mutagenicity=4.0)
    assert tx.mutagenicity_mode == "uniform"
    for tr in (0.0, 0.5, 1.0):
        assert _factor(tx.mutagenicity, tx.mutagenicity_mode, tr) == 4.0


@pytest.mark.parametrize("tr,expected", [(0.0, 4.0), (0.6429, 2.0713), (1.0, 1.0)])
def test_dose_mode_scales_with_the_dose_received(tr, expected):
    tx = Chemotherapy(mutagenicity=4.0, mutagenicity_mode="dose")
    assert _factor(tx.mutagenicity, tx.mutagenicity_mode, tr) == pytest.approx(expected, abs=1e-3)


def test_fully_resistant_clone_gets_no_mutator_under_dose():
    """A clone the drug cannot touch must not inherit a drug-induced phenotype."""
    tx = Chemotherapy(mutagenicity=4.0, mutagenicity_mode="dose")
    rep = FakeRep(tr=1.0)
    f = _factor(tx.mutagenicity, tx.mutagenicity_mode,
                rep.evolutionary_parameters["treatment_resistance"])
    assert f == 1.0
    assert rep.mutation_rate == 0.3          # untouched


def test_sensitive_clone_still_gets_the_full_boost_under_dose():
    """The whole point of the mutator -- de novo resistance -- must be preserved."""
    tx = Chemotherapy(mutagenicity=4.0, mutagenicity_mode="dose")
    f = _factor(tx.mutagenicity, tx.mutagenicity_mode, 0.0)
    assert f == 4.0


def test_drug_tolerance_also_protects_under_dose():
    """Protection is max(treatment_resistance, drug_tolerance), as everywhere else in the engine."""
    tr = max(0.0, 0.8)
    assert _factor(4.0, "dose", tr) == pytest.approx(1.0 + 3.0 * 0.2, abs=1e-9)


def test_mutagenicity_one_is_a_noop_in_both_modes():
    for mode in ("uniform", "dose"):
        for tr in (0.0, 0.5, 1.0):
            assert _factor(1.0, mode, tr) == 1.0


# ---------------------------------------------------------------------------------------------
# ``Treatment.mutagenicity_target`` — WHAT the mutator accelerates.
#
# `mutagenicity` multiplies mutation_rate, i.e. the chance a division mutates AT ALL; the SNV/CNA
# split happens downstream on snv_prob/cnv_prob, so point mutations and copy-number events scale
# together. That couples the two processes the escape modes turn on: resistance is ACQUIRED by an
# SNV hitting a resistance locus and LOST by a CNA deleting the copy carrying it. Measured, the
# acquisition:reversion ratio is 0.09 at mutagenicity 1.0 AND at 4.0 — the mutator moves the scale,
# never the balance.
#
# target="snv" lowers cnv_prob by exactly the factor that holds the ABSOLUTE per-division CNA rate
# fixed, so the extra mutating divisions all become SNVs. These exercise the real engine block.
# ---------------------------------------------------------------------------------------------
from iscc.tumor.models import GenotypeTumor

_GENOME = {"n_segments": 6, "segment_size": 100}
_DEME = {"carrying_capacity": 5, "initial_cancer_cells": 5}
_CANCER = {"division_rate": 0.4, "death_rate": 0.02, "max_birth_rate": 0.8,
           "mutation_rate": 0.3, "dispersal_rate": 0.9, "cnv_prob": 0.045, "snv_prob": 0.5}
_SEL = {"prop_driver": 0.1, "prop_dispersal": 0.0, "prop_immune_resistance": 0.0,
        "prop_treatment_resistance": 0.0}


# NOTE grow() iterates `range(n_steps - 1)`, so n_steps=1 runs ZERO update steps and never reaches
# the treatment block. Every treated-step test must use n_steps >= 2.
def _tumor(seed=3):
    return GenotypeTumor(seed=seed, genome_params=_GENOME, selection_params=_SEL,
                         cancer_cell_params=_CANCER, deme_params=_DEME,
                         spatial_params={"grid_size": 21, "n_structures": 1,
                                         "structure_radius": 0, "immune_density": 0.0})


def _cna_rate(rep):
    """Absolute per-division probability of a copy-number event, as the engine draws it."""
    m = float(rep.mutation_rate)
    d = rep.evolutionary_parameters["dispersal_rate"]
    mut_prob = m / (m + d) if (m + d) > 0 else 0.0
    c, s = float(rep.cnv_prob), float(rep.snv_prob)
    return mut_prob * (c / (c + s)) if (c + s) > 0 else 0.0


def _snv_rate(rep):
    m = float(rep.mutation_rate)
    d = rep.evolutionary_parameters["dispersal_rate"]
    mut_prob = m / (m + d) if (m + d) > 0 else 0.0
    c, s = float(rep.cnv_prob), float(rep.snv_prob)
    return mut_prob * (s / (c + s)) if (c + s) > 0 else 0.0


def test_target_defaults_to_all():
    assert Chemotherapy(mutagenicity=4.0).mutagenicity_target == "all"


def test_target_all_raises_the_cna_rate_too():
    """The default couples them: turning the mutator up raises copy-number events in lockstep."""
    t = _tumor()
    gid = next(g for g in t.genotypes_counts if t._is_cancer(g))
    before = _cna_rate(t.genotypes[gid])
    t.grow(n_steps=2, treatment=Chemotherapy(mutagenicity=4.0, duration=None))
    after = _cna_rate(t.genotypes[gid])
    assert after > before * 1.5, f"expected the CNA rate to rise, got {before} -> {after}"


def test_target_snv_leaves_the_cna_rate_untouched():
    """A point mutagen: more base substitutions, the SAME chromosomal instability.

    This is the property the whole change exists for -- reversion is a CNA deleting the copy that
    carries the resistance SNV, so holding the CNA rate fixed holds reversion at baseline."""
    t = _tumor()
    gid = next(g for g in t.genotypes_counts if t._is_cancer(g))
    rep = t.genotypes[gid]
    cna_before = _cna_rate(rep)
    n_before = float(rep.n_snvs_per_allele)
    t.grow(n_steps=2, treatment=Chemotherapy(mutagenicity=4.0, duration=None,
                                             mutagenicity_target="snv"))
    assert _cna_rate(rep) == pytest.approx(cna_before, rel=1e-12), "CNA rate must not move at all"
    assert float(rep.n_snvs_per_allele) == pytest.approx(n_before * 4.0, rel=1e-9)
    assert float(rep.cnv_prob) == pytest.approx(_CANCER["cnv_prob"], rel=1e-12), \
        "cnv_prob must be untouched -- the decoupling is structural, not compensated"


def test_target_snv_does_not_touch_the_mutation_fate_probability():
    """`mutation_rate` is the mutate-vs-disperse fate, not a mutation rate; a point mutagen has no
    business changing whether a division mutates INSTEAD of migrating."""
    t = _tumor()
    gid = next(g for g in t.genotypes_counts if t._is_cancer(g))
    rep = t.genotypes[gid]
    m_before = float(rep.mutation_rate)
    t.grow(n_steps=2, treatment=Chemotherapy(mutagenicity=4.0, duration=None,
                                             mutagenicity_target="snv"))
    assert float(rep.mutation_rate) == pytest.approx(m_before, rel=1e-12)


def test_target_all_still_scales_the_fate_probability():
    """The DEFAULT is unchanged: it multiplies mutation_rate and so raises BOTH branches."""
    t = _tumor()
    gid = next(g for g in t.genotypes_counts if t._is_cancer(g))
    rep = t.genotypes[gid]
    m_before, n_before = float(rep.mutation_rate), float(rep.n_snvs_per_allele)
    t.grow(n_steps=2, treatment=Chemotherapy(mutagenicity=4.0, duration=None))
    assert float(rep.mutation_rate) == pytest.approx(m_before * 4.0, rel=1e-9)
    assert float(rep.n_snvs_per_allele) == pytest.approx(n_before, rel=1e-12)
