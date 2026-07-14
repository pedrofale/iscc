"""Treatment on the genotype (count-based) engine, plus the corrected immune-death model.

Covers, on the default GenotypeTumor:
  * the fixed immune-death formula (immune presence *raises* cancer death; immune
    resistance attenuates it) -- the old formula could only lower it and made
    immune-free demes immortal;
  * chemotherapy suppressing growth and selecting for treatment-resistant clones;
  * immunotherapy stripping immune resistance so the local immune cells kill more;
  * adaptive vs continuous dosing;
  * that growth is unchanged when no treatment is given.
"""
import numpy as np

from iscc.tumor.models import GenotypeTumor
from iscc.treatment.chemotherapy import Chemotherapy
from iscc.treatment.immunotherapy import Immunotherapy

GENOME = {"n_segments": 6, "segment_size": 100}
# Seed an established micro-lesion (initial_cancer_cells) so the founder survives crowding +
# immune pressure instead of stochastically dying out (DESIGN_crowding.md founder bottleneck).
DEME = {"carrying_capacity": 5, "initial_cancer_cells": 5}
CANCER = {"division_rate": 0.4, "death_rate": 0.02, "max_birth_rate": 0.8,
          "mutation_rate": 0.5, "dispersal_rate": 0.2}
# resistance can evolve so therapy selects for it; no dispersal/immune unless asked
SEL_TR = {"prop_driver": 0.1, "prop_dispersal": 0.0, "prop_immune_resistance": 0.0,
          "prop_treatment_resistance": 0.05, "driver_effects": 1.1, "dispersal_effects": 1.0,
          "treatment_resistant_effects": 1.2, "immune_resistant_effects": 1.0}
SEL_IR = {"prop_driver": 0.1, "prop_dispersal": 0.0, "prop_immune_resistance": 0.3,
          "prop_treatment_resistance": 0.0, "driver_effects": 1.1, "dispersal_effects": 1.0,
          "treatment_resistant_effects": 1.0, "immune_resistant_effects": 1.4}
# fully sensitive (no resistance can evolve) -> chemo cleanly regresses the tumor
SEL_SENS = {"prop_driver": 0.1, "prop_dispersal": 0.0, "prop_immune_resistance": 0.0,
            "prop_treatment_resistance": 0.0, "driver_effects": 1.1, "dispersal_effects": 1.0,
            "treatment_resistant_effects": 1.0, "immune_resistant_effects": 1.0}


def _build(selection, seed, immune_density=0.0, prob_kill=0.01):
    spatial = {"grid_size": 21, "n_structures": 1, "structure_radius": 0,
               "immune_density": immune_density}
    return GenotypeTumor(
        seed=seed, genome_params=GENOME, selection_params=selection,
        cancer_cell_params=CANCER, deme_params=DEME, spatial_params=spatial,
        immune_cell_params={"prob_kill": prob_kill},
    )


def _mean_param(t, key):
    gids = [g for g in t.genotypes_counts if t._is_cancer(g)]
    w = np.array([t.genotypes_counts[g] for g in gids], float)
    if w.sum() == 0:
        return 0.0
    v = np.array([t.genotypes[g].evolutionary_parameters[key] for g in gids])
    return float((w * v).sum() / w.sum())


# --- corrected immune-death formula (unit) ----------------------------------
def test_immune_presence_raises_cancer_death():
    center = (21 // 2) * 21 + (21 // 2)
    no_immune = _build(SEL_IR, seed=0, immune_density=0.0)
    with_immune = _build(SEL_IR, seed=0, immune_density=1.0, prob_kill=0.5)
    fid = no_immune.founder_id
    d_none = no_immune._death_rate(fid, center)
    d_immune = with_immune._death_rate(with_immune.founder_id, center)
    assert d_immune > d_none            # immune cells increase cancer death
    # no immune -> baseline death PLUS density-dependent crowding (DESIGN_crowding.md), and
    # crucially NOT the old degenerate 0 (immortal cancer). Crowding only adds, so d_none >= baseline.
    assert d_none >= CANCER["death_rate"] > 0


def test_immune_resistance_attenuates_killing():
    center = (21 // 2) * 21 + (21 // 2)
    t = _build(SEL_IR, seed=0, immune_density=1.0, prob_kill=0.5)
    fid = t.founder_id
    t.genotypes[fid].evolutionary_parameters["immune_resistance"] = 0.0
    d_sensitive = t._death_rate(fid, center)
    t.genotypes[fid].evolutionary_parameters["immune_resistance"] = 0.9
    d_resistant = t._death_rate(fid, center)
    assert d_sensitive > d_resistant    # resistance shields the cell from immune killing


def test_immunotherapy_strips_resistance_and_raises_death():
    center = (21 // 2) * 21 + (21 // 2)
    t = _build(SEL_IR, seed=0, immune_density=1.0, prob_kill=0.5)
    fid = t.founder_id
    t.genotypes[fid].evolutionary_parameters["immune_resistance"] = 0.8
    d_before = t._death_rate(fid, center)
    t._apply_treatment(Immunotherapy(immune_checkpoints=[], effectiveness=1.0, toxicity=0.0), dosage=1.0)
    d_after = t._death_rate(fid, center)
    assert t._tx_immune_resist[fid] < 0.8   # resistance stripped
    assert d_after > d_before                # -> more immune killing


# --- chemotherapy (integration) ---------------------------------------------
def test_chemo_suppresses_growth_and_selects_resistance():
    treated_sizes, control_sizes, tr_gain = [], [], []
    for seed in (1, 2, 3, 4):
        t = _build(SEL_TR, seed=seed)
        t.grow(600, seed=seed)
        if t.get_cancer_size() == 0:
            continue
        tr_pre = _mean_param(t, "treatment_resistance")
        t.grow(600, seed=seed + 100,
               treatment=Chemotherapy(adaptive=False, effectiveness=0.95, toxicity=0.05))
        treated_sizes.append(t.get_cancer_size())
        tr_gain.append(_mean_param(t, "treatment_resistance") - tr_pre)

        c = _build(SEL_TR, seed=seed)
        c.grow(600, seed=seed)
        c.grow(600, seed=seed + 100)        # untreated control over the same window
        control_sizes.append(c.get_cancer_size())

    assert len(treated_sizes) >= 2
    assert np.mean(treated_sizes) < np.mean(control_sizes)   # chemo holds the tumor back
    assert np.mean(tr_gain) > 0                              # and selects for resistance


# --- immunotherapy (integration) --------------------------------------------
def test_immunotherapy_reduces_growth():
    # Under density-dependent crowding (DESIGN_crowding.md) immune cells add to a deme's occupancy
    # as well as killing, so strong immune pressure now drives most micro-lesions extinct; the
    # survivors are the lineages that evolved immune resistance. We use a moderate immune density
    # (so a handful of seeds establish) and many seeds, then check immunotherapy — which strips that
    # evolved resistance — reduces the survivors' growth.
    treated, control = [], []
    for seed in range(2, 18):
        t = _build(SEL_IR, seed=seed, immune_density=0.2, prob_kill=0.15)
        t.grow(600, seed=seed)
        if t.get_cancer_size() == 0:
            continue
        t.grow(600, seed=seed + 100,
               treatment=Immunotherapy(immune_checkpoints=[], adaptive=False,
                                       effectiveness=0.95, toxicity=0.05))
        treated.append(t.get_cancer_size())

        c = _build(SEL_IR, seed=seed, immune_density=0.2, prob_kill=0.15)
        c.grow(600, seed=seed)
        c.grow(600, seed=seed + 100)
        control.append(c.get_cancer_size())

    assert len(treated) >= 2
    assert np.mean(treated) < np.mean(control)   # immunotherapy restores immune control


# --- dosing scheduling ------------------------------------------------------
def _small_sensitive(seed):
    # small, fully-sensitive tumor so chemo can move it across a threshold within a
    # short (one-event-per-step) treatment window
    spatial = {"grid_size": 15, "n_structures": 1, "structure_radius": 0}
    return GenotypeTumor(
        seed=seed, genome_params=GENOME, selection_params=SEL_SENS,
        cancer_cell_params=CANCER, deme_params={"carrying_capacity": 3},
        spatial_params=spatial,
    )


def test_adaptive_dosing_controls_tumor_with_less_drug():
    t = _small_sensitive(2)
    t.grow(250, seed=2)
    threshold = t.get_cancer_size() // 2

    cont = Chemotherapy(adaptive=False)
    t.grow(500, seed=300, treatment=cont)
    cont_dose = sum(d for _, d in cont.dosage_trace)

    a = _small_sensitive(2)
    a.grow(250, seed=2)
    adapt = Chemotherapy(adaptive=True, max_tumor_size=threshold)
    a.grow(500, seed=300, treatment=adapt)
    adapt_dose = sum(d for _, d in adapt.dosage_trace)

    # adaptive withholds drug once the tumor is controlled below threshold -> less drug,
    # while keeping the tumor at/above the (smaller) continuous-therapy residual.
    assert adapt_dose < cont_dose
    assert a.get_cancer_size() >= t.get_cancer_size()


# --- no-treatment guard -----------------------------------------------------
def test_no_treatment_matches_plain_grow():
    a = _build(SEL_TR, seed=7)
    a.grow(300, seed=7, treatment=None)
    b = _build(SEL_TR, seed=7)
    b.grow(300, seed=7)
    # genotype_ids are drawn from a process-global counter, so the two runs use different
    # ids; the *multiset of clone sizes* is the engine-state invariant to compare.
    assert sorted(a.genotypes_counts.values()) == sorted(b.genotypes_counts.values())
    assert a.get_cancer_size() == b.get_cancer_size()
