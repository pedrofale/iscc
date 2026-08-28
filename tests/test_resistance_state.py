"""Drug-induced resistance STATE — DESIGN_phenotype_plasticity.md §3.3.

A carried, heritable resistance cell-state INDEPENDENT of the genome: acquiring a resistance mutation
sets it (genetic entry / beta_bias), the drug can induce it (plasticity), and it relaxes back off drug
over a configurable timescale (tau_relax) — so, unlike the genomic-only trait, a CNA that deletes the
triggering allele does NOT restore sensitivity. The state feeds the drug-protection term and charges a
proliferation cost. A state transition MINTS A NEW GENOTYPE ID (the state is not derivable from the
genome). OFF BY DEFAULT: with every knob at its inert default ``resistance_state_on`` is False and no
new code path runs, so growth is byte-identical (the golden-hash discipline of test_wgd).

Covered here:
  * off-by-default: explicit-inert == absent (byte-identical stream); no new cell_data / cell_evo schema.
  * genetic entry (n_mut_tr>0 sets the state) + RETENTION when the allele is later deleted (the point).
  * inheritance of the state through divide() and mutate() (allele-loss children keep it).
  * the effect: a state cell takes attenuated drug (folded into the protection max()).
  * the cost: a state cell divides slower, and a twin's rate reflects its OWN state.
  * EXIT is genuinely reachable (relax>0 flips state 1->0) — NOT the rejected latch.
  * a transition mints a NEW genotype id sharing the genome, records the parent, and REUSES the twin.
"""
import hashlib

import numpy as np
import pytest

from conftest import (
    GENOME_PARAMS, SELECTION_PARAMS, CANCER_CELL_PARAMS, N_SEGMENTS, SEGMENT_SIZE, DEME_PARAMS,
)
from iscc.tumor.models import GenotypeTumor
from iscc.tumor.components.cell import CancerCell
from iscc.tumor.components.selection import Selection
from iscc.treatment.chemotherapy import Chemotherapy

SPATIAL = {"grid_size": 15, "n_structures": 1, "structure_radius": 0}
NO_DEATH = {**CANCER_CELL_PARAMS, "death_rate": 0.0}
# A state config that turns every arm on, for the on-path tests.
STATE_ON = {"resistance_state_genetic": True, "resistance_state_effect": 0.99,
            "resistance_state_cost": 0.2, "resistance_state_induction": 0.5,
            "resistance_state_relax": 0.3}


def _count(seed, steps, cancer=NO_DEATH, selection=SELECTION_PARAMS, treatment=None, **kw):
    t = GenotypeTumor(
        seed=seed, genome_params=GENOME_PARAMS, selection_params=selection,
        cancer_cell_params=cancer, deme_params=DEME_PARAMS, spatial_params=SPATIAL,
        update_mode="tau", **kw,
    )
    t.grow(n_steps=steps, seed=seed, treatment=treatment)
    return t


def _snv_hash(t):
    return hashlib.md5(np.ascontiguousarray(t.cell_data["cell_snv"].values)).hexdigest()


def _fresh_cancer(selection, **kw):
    c = CancerCell(
        n_segments=selection.n_segments, segment_size=selection.segment_size,
        n_onc=len(selection.get_oncogenes()), n_tsg=len(selection.get_tsgs()),
        n_disp=len(selection.get_dispersal_genes()), n_ir=len(selection.get_immune_resistant()),
        n_tr=len(selection.get_treatment_resistant()), **kw,
    )
    c.set_genotype_id()
    return c


# --- off by default: byte-identical stream + unchanged schema ---------------------------------------
def test_resistance_state_on_is_false_by_default():
    sel = Selection(n_segments=N_SEGMENTS, segment_size=SEGMENT_SIZE, **SELECTION_PARAMS)
    assert sel.resistance_state_on is False


@pytest.mark.parametrize("knob,val", [
    ("resistance_state_genetic", True), ("resistance_state_effect", 0.9),
    ("resistance_state_cost", 0.2), ("resistance_state_induction", 0.5),
    ("resistance_state_relax", 0.3), ("resistance_state_noise", 0.01),
])
def test_any_knob_turns_the_feature_on(knob, val):
    sel = Selection(n_segments=N_SEGMENTS, segment_size=SEGMENT_SIZE, **{**SELECTION_PARAMS, knob: val})
    assert sel.resistance_state_on is True


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_explicit_inert_state_matches_absent_keys(seed):
    """All state knobs explicitly at their inert defaults must be byte-identical to omitting them —
    the same guarantee test_wgd pins for wgd_rate=0. (off==pre-feature is carried by the existing
    golden-hash suite, which stays green because every new path is gated on resistance_state_on.)"""
    a = _count(seed, steps=120)
    inert = {**SELECTION_PARAMS, "resistance_state_genetic": False, "resistance_state_effect": 0.0,
             "resistance_state_cost": 0.0, "resistance_state_induction": 0.0,
             "resistance_state_relax": 0.0, "resistance_state_noise": 0.0}
    b = _count(seed, steps=120, selection=inert)
    assert a.get_tumor_size() == b.get_tumor_size()
    assert _snv_hash(a) == _snv_hash(b)


def test_off_adds_no_state_columns():
    """With the feature off, neither the per-cell ground-truth frame nor an evo column appears."""
    t = _count(2, steps=120)
    assert "cell_resistance_state" not in t.cell_data
    assert "resistance_state" not in list(t.cell_data["cell_evo"].columns)


def test_on_surfaces_the_ground_truth_frame():
    t = _count(2, steps=60, selection={**SELECTION_PARAMS, **STATE_ON})
    assert "cell_resistance_state" in t.cell_data
    assert set(np.unique(t.cell_data["cell_resistance_state"].values)) <= {0.0, 1.0}


# --- genetic entry + retention (the whole point) ----------------------------------------------------
def _sel(**over):
    return Selection(n_segments=N_SEGMENTS, segment_size=SEGMENT_SIZE, **{**SELECTION_PARAMS, **over})


def test_genetic_entry_sets_state_when_resistance_acquired():
    sel = _sel(resistance_state_genetic=True, resistance_state_effect=0.99)
    c = _fresh_cancer(sel)
    assert c.resistance_state == 0.0
    c.genome_summary["n_mut_tr"] = 1                 # acquired a resistance mutation
    c.update_evolutionary_parameters(sel)
    assert c.resistance_state == 1.0


def test_state_is_retained_when_the_allele_is_deleted():
    """The load-bearing behaviour: lose the resistance allele (n_mut_tr -> 0) and the cell STAYS in the
    state. This is exactly what makes a CNA revertant count as resistant rather than sensitive."""
    sel = _sel(resistance_state_genetic=True, resistance_state_effect=0.99)
    c = _fresh_cancer(sel)
    c.genome_summary["n_mut_tr"] = 1
    c.update_evolutionary_parameters(sel)
    assert c.resistance_state == 1.0
    c.genome_summary["n_mut_tr"] = 0                 # a deletion removed the carrying copy
    c.update_evolutionary_parameters(sel)
    assert c.resistance_state == 1.0                 # NOT cleared -> still resistant


def test_genetic_entry_off_never_sets_state():
    sel = _sel(resistance_state_effect=0.99)          # effect on, genetic OFF
    c = _fresh_cancer(sel)
    c.genome_summary["n_mut_tr"] = 3
    c.update_evolutionary_parameters(sel)
    assert c.resistance_state == 0.0


def test_state_is_inherited_through_divide_and_mutate():
    sel = _sel(resistance_state_genetic=True, resistance_state_effect=0.99)
    parent = _fresh_cancer(sel)
    parent.resistance_state = 1.0
    child = parent.divide()
    assert child.resistance_state == 1.0             # shallow copy carries it
    rng = np.random.default_rng(0)
    child.mutate(rng, sel)                            # a further mutation must not clear it
    assert child.resistance_state == 1.0


# --- the effect: attenuated drug ---------------------------------------------------------------------
def test_state_attenuates_the_drug_hazard():
    """A state cell takes strictly less chemo hazard than an identical-genome sensitive cell, via the
    max(tr, dt, state*effect) protection term."""
    t = _count(3, steps=40, selection={**SELECTION_PARAMS, **STATE_ON})
    chemo = Chemotherapy(kill_rate=1.5, kill_mode="proliferation")
    dose = chemo.get_dosage(t.step, t.get_tumor_size())
    assert dose > 0
    t._apply_treatment(chemo, dose)
    gid = next(g for g in t.genotypes_counts if t._is_cancer(g))
    rep = t.genotypes[gid]
    base = t._kill_amount(rep, dose * chemo.effectiveness, chemo)   # protection 0 -> full hazard
    rep.resistance_state = 1.0
    prot = max(rep.evolutionary_parameters["treatment_resistance"],
               rep.evolutionary_parameters.get("drug_tolerance", 0.0), t._state_protection(rep))
    protected = t._kill_amount(rep, dose * chemo.effectiveness * (1.0 - prot), chemo)
    assert protected < base
    assert t._state_protection(rep) == pytest.approx(0.99)


# --- the cost ---------------------------------------------------------------------------------------
def test_state_charges_a_proliferation_cost():
    sel = _sel(resistance_state_genetic=True, resistance_state_effect=0.99, resistance_state_cost=0.3)
    sensitive = _fresh_cancer(sel)
    sensitive.update_evolutionary_parameters(sel)
    div_sensitive = sensitive.evolutionary_parameters["division_rate"]
    stated = _fresh_cancer(sel)
    stated.resistance_state = 1.0
    stated.update_evolutionary_parameters(sel)
    div_stated = stated.evolutionary_parameters["division_rate"]
    assert div_stated == pytest.approx(div_sensitive * (1.0 - 0.3))
    assert div_stated < div_sensitive


# --- the engine trap: a transition mints a NEW genotype id, reused ----------------------------------
def test_state_twin_mints_a_new_genotype_sharing_the_genome():
    # genetic+effect only (relax=induction=0) so NO transition twins form during growth -> the count
    # assertions below see exactly the one twin this test mints.
    sel = {**SELECTION_PARAMS, "resistance_state_genetic": True, "resistance_state_effect": 0.99}
    t = _count(2, steps=40, selection=sel)
    # a SENSITIVE (state 0) cancer genotype, so requesting its state-1 twin genuinely mints a new one
    # (a genetic-entry genotype already has state 1 and would correctly return itself).
    gid = next(g for g in t.genotypes_counts
               if t._is_cancer(g) and t.genotypes[g].resistance_state == 0.0)
    n_before = len(t.genotypes)
    twin = t._state_twin(gid, 1.0)
    assert twin != gid
    assert twin in t.genotypes
    assert len(t.genotypes) == n_before + 1
    # same genome, different state, genealogy recorded
    assert t.genotypes[twin].genome_summary["n_mut_tr"] == t.genotypes[gid].genome_summary["n_mut_tr"]
    assert t.genotypes[twin].resistance_state == 1.0
    assert t.genotypes_parents[twin] == gid
    # REUSED, not re-minted, on a second request for the same (genome, state)
    assert t._state_twin(gid, 1.0) == twin
    assert len(t.genotypes) == n_before + 1
    # and the two are each other's family members (state 0 twin of the twin is the original)
    assert t._state_twin(twin, 0.0) == gid


# --- EXIT is reachable: this is NOT the rejected latch ----------------------------------------------
def test_exit_is_possible_not_a_latch():
    """With relax high and no drug, a state cell (allele already gone) relaxes back to sensitive within
    a few generations, producing a state-0 twin. The rejected feature was a trait that could NEVER
    decrease; here exit is a first-class, reachable transition."""
    # prop_treatment_resistance=0.0 -> no resistance loci exist -> n_mut_tr is always 0, so every
    # forced-state cell is exit-eligible (no genetic anchor).
    sel_kw = {"prop_treatment_resistance": 0.0,
              "resistance_state_effect": 0.99, "resistance_state_relax": 0.9}
    t = GenotypeTumor(seed=1, genome_params=GENOME_PARAMS,
                      selection_params={**SELECTION_PARAMS, **sel_kw},
                      cancer_cell_params=NO_DEATH, deme_params=DEME_PARAMS,
                      spatial_params=SPATIAL, update_mode="tau")
    t.grow(n_steps=20, seed=1)                        # establish a lesion (no drug)
    # force every live cancer cell into the state (allele-free: n_mut_tr stays 0)
    for g in [g for g in list(t.genotypes_counts) if t._is_cancer(g)]:
        t.genotypes[g].resistance_state = 1.0
    assert any(t.genotypes[g].resistance_state > 0 for g in t.genotypes_counts if t._is_cancer(g))
    rng = np.random.default_rng(0)
    fired = False
    for _ in range(10):
        t._apply_treatment(None, 0.0)                # off drug
        t._apply_state_transitions(rng)
        if any(t._is_cancer(g) and t.genotypes[g].resistance_state == 0.0
               for g in t.genotypes_counts):
            fired = True
            break
    assert fired, "exit never fired at relax=0.9 -> behaves as a latch"


def test_no_exit_while_the_allele_is_present():
    """A cell that still carries the resistance allele keeps its genetic attractor and must not relax
    out (else genetic entry and relaxation would fight each other)."""
    sel = _sel(resistance_state_genetic=True, resistance_state_effect=0.99, resistance_state_relax=0.9)
    t = GenotypeTumor(seed=1, genome_params=GENOME_PARAMS,
                      selection_params={**SELECTION_PARAMS, "resistance_state_genetic": True,
                                        "resistance_state_effect": 0.99, "resistance_state_relax": 0.9},
                      cancer_cell_params=NO_DEATH, deme_params=DEME_PARAMS,
                      spatial_params=SPATIAL, update_mode="tau")
    t.grow(n_steps=20, seed=1)
    for g in [g for g in list(t.genotypes_counts) if t._is_cancer(g)]:
        t.genotypes[g].resistance_state = 1.0
        t.genotypes[g].genome_summary["n_mut_tr"] = 2   # still carries the allele
    rng = np.random.default_rng(0)
    n_before = len(t.genotypes)
    for _ in range(10):
        t._apply_treatment(None, 0.0)
        t._apply_state_transitions(rng)
    assert not any(t._is_cancer(g) and t.genotypes[g].resistance_state == 0.0
                   for g in t.genotypes_counts), "an allele-anchored cell relaxed out"
    assert len(t.genotypes) == n_before               # no exit twins minted
