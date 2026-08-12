"""Regression: a genotype minted MID-STEP must take the active treatment's hazard.

``_apply_treatment`` freezes the per-genotype treatment overrides (``_tx_death_add`` for
chemo/targeted, ``_tx_immune_resist`` for immunotherapy) ONCE at the start of a treated step, from
the genotypes present at that moment. A genotype created DURING the ensuing ``update`` /
``_tau_generation`` (a division-mutation) is absent from those dicts, so the old
``_tx_death_add.get(gid, 0.0)`` / ``_tx_immune_resist.get(gid, base_ir)`` defaults handed the
newborn a DRUG-FREE step: full drug survival, and (under immunotherapy) its untouched baseline
immune resistance. The hole scaled with the mutation rate -- worst exactly in therapy-induced
mutagenesis / resistance-evolution runs -- and cost measurably (~14%) of the nominal kill.

The fix computes each override on demand for an unknown gid (a pure function of the clone's
resistance / tolerance, the active dose, effectiveness and kill_rate) and memoises it, so a newborn
is charged identically to an already-registered sibling. These tests pin that: a mid-step clone
takes the same hazard as its identical-genotype parent, and coverage of live cancer cells is 100%.
"""
import numpy as np
import pytest

from iscc.tumor.models import GenotypeTumor
from iscc.treatment.chemotherapy import Chemotherapy
from iscc.treatment.immunotherapy import Immunotherapy

GENOME = {"n_segments": 6, "segment_size": 100}
DEME = {"carrying_capacity": 5, "initial_cancer_cells": 5}
# High mutation so a treated step actually mints new genotypes (exercises the hole).
CANCER = {"division_rate": 0.4, "death_rate": 0.02, "max_birth_rate": 0.8,
          "mutation_rate": 0.9, "dispersal_rate": 0.2}
# Fully sensitive: no treatment resistance can evolve, so EVERY cancer clone must take the drug
# hazard -- a zero-hazard clone would be a genuine coverage hole, not resistance biology.
SEL_SENS = {"prop_driver": 0.1, "prop_dispersal": 0.0, "prop_immune_resistance": 0.0,
            "prop_treatment_resistance": 0.0, "driver_effects": 1.1, "dispersal_effects": 1.0,
            "treatment_resistant_effects": 1.0, "immune_resistant_effects": 1.0}
# Immune resistance can evolve, so immunotherapy has something to strip.
SEL_IR = {"prop_driver": 0.1, "prop_dispersal": 0.0, "prop_immune_resistance": 0.3,
          "prop_treatment_resistance": 0.0, "driver_effects": 1.1, "dispersal_effects": 1.0,
          "treatment_resistant_effects": 1.0, "immune_resistant_effects": 1.4}


def _build(selection, seed, immune_density=0.0, prob_kill=0.01, update_mode="exact"):
    spatial = {"grid_size": 21, "n_structures": 1, "structure_radius": 0,
               "immune_density": immune_density}
    return GenotypeTumor(
        seed=seed, genome_params=GENOME, selection_params=selection, cancer_cell_params=CANCER,
        deme_params=DEME, spatial_params=spatial, immune_cell_params={"prob_kill": prob_kill},
        update_mode=update_mode)


def _live_cancer(t):
    return [g for g in t.genotypes_counts if t._is_cancer(g) and t.genotypes_counts[g] > 0]


def _deme_of(t, gid):
    for di, deme in enumerate(t.demes):
        if deme.get(gid, 0) > 0:
            return di
    return None


def _mint_clone_after_freeze(t, parent, di):
    """Mint a genotype the way a division-mutation does (Cell.divide -> _register -> _add), but with
    the parent's genotype UNCHANGED and AFTER this step's _apply_treatment has already frozen the
    override dicts. Identical treatment-relevant parameters, distinct genotype_id, no dict entry --
    exactly the state a real mid-step mutant lands in."""
    child = t.genotypes[parent].divide()
    child.set_genotype_id()
    t._register(child)
    t._add(di, child.genotype_id, 1)
    return child


def test_midstep_clone_takes_same_chemo_hazard_as_registered_sibling():
    """A clone minted after the drug dict is frozen must die at the SAME rate as its identical-
    genotype parent in the same deme -- the treatment term is the only thing that could differ, and
    the old ``.get(gid, 0.0)`` default zeroed it for the newborn."""
    t = _build(SEL_SENS, seed=3, update_mode="tau")
    t.grow(n_steps=40, seed=3)                              # establish a multi-clone lesion
    chemo = Chemotherapy(kill_rate=1.5)
    dose = chemo.get_dosage(t.step, t.get_tumor_size())
    assert dose > 0
    t._apply_treatment(chemo, dose)                        # freeze _tx_death_add for the CURRENT gids

    parent = next(g for g in _live_cancer(t) if t._tx_death_add.get(g, 0.0) > 0)
    di = _deme_of(t, parent)
    child = _mint_clone_after_freeze(t, parent, di)

    # the bug's precondition: the frozen dict has no entry for the newborn ...
    assert child.genotype_id not in t._tx_death_add
    # ... yet its death rate must equal its parent's (same deme, same composition): the only term
    # that can differ between two identical genotypes here is the treatment hazard.
    comp = t._deme_comp(t.demes[di])
    total = comp[0]
    d_parent = t._death_rate(parent, di, total, comp=comp)
    d_child = t._death_rate(child.genotype_id, di, total, comp=comp)
    assert d_child == pytest.approx(d_parent)
    # and the applied hazard is the frozen sibling value, not zero
    assert t._tx_death_add[parent] > 0
    assert t._tx_death_add_for(child.genotype_id) == pytest.approx(t._tx_death_add[parent])


@pytest.mark.parametrize("seed", (5, 11))
def test_treated_generation_covers_every_live_cancer_cell(seed):
    """Coverage: after a full treated tau generation mints many clones, 100% of live cancer cells --
    newborns included -- carry a nonzero drug hazard, not the ~86% the frozen-dict-only view saw."""
    t = _build(SEL_SENS, seed=seed, update_mode="tau")
    t.grow(n_steps=40, seed=seed)                          # establish a lesion untreated
    chemo = Chemotherapy(kill_rate=1.2, mutagenicity=4.0)  # mutagenicity => even more mid-step mints
    dose = chemo.get_dosage(t.step, t.get_tumor_size())
    assert dose > 0
    t._apply_treatment(chemo, dose)                        # freeze the dict for the pre-generation gids
    frozen = set(t._tx_death_add)
    rng = np.random.default_rng(seed + t.step)
    t._tau_generation(rng, t.tau)                          # one treated generation: mints newborns

    live = _live_cancer(t)
    assert live, "tumour went extinct before minting; lower kill_rate or raise the lesion size"
    minted = [g for g in live if g not in frozen]
    assert minted, "no mid-step genotype was minted; raise mutation_rate / mutagenicity"

    cells = sum(t.genotypes_counts[g] for g in live)
    covered = sum(t.genotypes_counts[g] for g in live if t._tx_death_add_for(g) > 0)
    assert covered == cells                                # 100% -- the fix
    # the frozen-dict-only view (the OLD applied coverage) misses the newborns: that gap is the bug.
    raw = sum(t.genotypes_counts[g] for g in live if g in frozen)
    assert raw < cells


def test_midstep_clone_takes_stripped_immune_resistance_under_immunotherapy():
    """The same hole in ``_tx_immune_resist``: under immunotherapy a mid-step clone must be stripped
    to its parent's reduced immune resistance, not left sitting at its untouched baseline (which the
    old ``.get(gid, base_ir)`` default handed it -- a free step of full immune evasion)."""
    # Immune killing is irrelevant to what this test pins (immunotherapy strips a clone's resistance
    # param whether or not immune cells sit in its deme), so keep it off and let the lesion grow.
    t = _build(SEL_IR, seed=4, immune_density=0.0, update_mode="tau")
    t.grow(n_steps=40, seed=4)

    parent = _live_cancer(t)[0]
    # give the parent a visibly positive immune resistance so stripping is observable
    t.genotypes[parent].evolutionary_parameters["immune_resistance"] = 0.6
    immuno = Immunotherapy(immune_checkpoints=[])          # broad: every cancer cell is a target
    dose = immuno.get_dosage(t.step, t.get_tumor_size())
    assert dose > 0
    t._apply_treatment(immuno, dose)                       # populate _tx_immune_resist

    assert parent in t._tx_immune_resist
    stripped = t._tx_immune_resist[parent]
    assert stripped < 0.6                                  # immunotherapy actually stripped resistance

    di = _deme_of(t, parent)
    child = _mint_clone_after_freeze(t, parent, di)
    assert child.genotype_id not in t._tx_immune_resist    # bug precondition
    base_ir = child.evolutionary_parameters["immune_resistance"]   # 0.6, its untouched baseline
    # the fix: the newborn is stripped to exactly the parent's value, not left at baseline
    got = t._tx_immune_resist_for(child.genotype_id, child, base_ir)
    assert got == pytest.approx(stripped)
    assert got < base_ir
