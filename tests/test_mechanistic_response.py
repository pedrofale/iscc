"""``Treatment(response="mechanistic")`` — drug response from the genotype and its expression.

The default ``response="label"`` decides who a drug spares by resistance LABELS: the evolved
``treatment_resistance`` / ``drug_tolerance`` traits and the carried ``resistance_state``. The
escape-mode studies are built on it, so it stays the default and must stay byte-identical.

The mechanistic mode ignores all three. A cancer clone's response comes from what it is:

  * a genotype-targeted drug (``targets``) acts only on clones that CARRY the target — a copy with the
    mutation, or (``amplification_threshold``) more copies of its segment than the threshold — so a
    lineage that deletes the target by CNA stops responding;
  * an expression-targeted drug (``target_genes``) acts in proportion to the clone's expression of the
    target genes (baseline dosage x programs), relative to the founding clone, and not at all below
    ``expression_threshold``;
  * a broad cytotoxic drug with ``kill_mode="proliferation"`` kills in proportion to division rate.
"""
import hashlib

import numpy as np
import pytest

from iscc.tumor.components.cell import Cell
from iscc.tumor.models import GenotypeTumor
from iscc.treatment.chemotherapy import Chemotherapy
from iscc.treatment.immunotherapy import Immunotherapy
from iscc.treatment.targeted import TargetedTherapy

GENOME = {"n_segments": 4, "segment_size": 30}
SELECTION = {"prop_driver": 0.1, "prop_dispersal": 0.0, "prop_immune_resistance": 0.2,
             "prop_treatment_resistance": 0.1, "prop_drug_tolerance": 0.1,
             "drug_tolerance_cost": 0.3, "driver_effects": 1.1,
             "treatment_resistant_effects": 1.3, "immune_resistant_effects": 1.3}
STATE_ON = {"resistance_state_genetic": True, "resistance_state_effect": 0.9,
            "resistance_state_cost": 0.1, "resistance_state_induction": 0.3,
            "resistance_state_relax": 0.2}
CANCER = {"division_rate": 0.6, "death_rate": 0.02, "max_birth_rate": 1.2, "mutation_rate": 0.6,
          "dispersal_rate": 0.2}
DEME = {"carrying_capacity": 10, "initial_cancer_cells": 5}
SPATIAL = {"grid_size": 12, "n_structures": 1, "structure_radius": 0, "immune_density": 0.3}


def _build(seed=3, mode="tau", selection=SELECTION, cancer=CANCER, spatial=SPATIAL, **kw):
    # Genotype ids come from a process-wide counter and the digests include them, so every build
    # starts it at the same value.
    Cell._genotype_counter = 1_000_000
    return GenotypeTumor(seed=seed, genome_params=GENOME, selection_params=selection,
                         cancer_cell_params=cancer, deme_params=DEME, spatial_params=spatial,
                         immune_cell_params={"prob_kill": 0.2}, update_mode=mode, **kw)


def _digest(t):
    blob = repr((t.step, t.traces, sorted(t.genotypes_counts.items()),
                 sorted(t.genotypes_parents.items())))
    return hashlib.md5(blob.encode()).hexdigest()


# Every treatment path the mechanistic mode branches: the death-rate kill (additive and
# proliferation), the dose-scaled mutator, the persister cost, the immunotherapy strip, the
# drug-induced resistance state, and the exact engine.
LABEL_SCENARIOS = {
    "chemo_dose_mutator_persister": dict(
        build={}, steps=30,
        tx=lambda: Chemotherapy(start=8, kill_rate=1.2, mutagenicity=3.0, mutagenicity_mode="dose")),
    "targeted": dict(
        build={}, steps=30,
        tx=lambda: TargetedTherapy(targets=[(0, 1), (2, 5)], start=8, kill_rate=1.2)),
    "immunotherapy": dict(
        build={}, steps=30, tx=lambda: Immunotherapy(immune_checkpoints=[], start=8)),
    "state_induction": dict(
        build=dict(selection={**SELECTION, **STATE_ON}), steps=30,
        tx=lambda: Chemotherapy(start=8, kill_rate=1.2, kill_mode="proliferation")),
    "exact_chemo": dict(
        build=dict(mode="exact"), steps=600, tx=lambda: Chemotherapy(start=200, kill_rate=1.2)),
}


def _run_label(name, **tx_kw):
    sc = LABEL_SCENARIOS[name]
    t = _build(**sc["build"])
    tx = sc["tx"]()
    for k, v in tx_kw.items():
        setattr(tx, k, v)
    t.grow(n_steps=sc["steps"], seed=3, treatment=tx)
    return t


# Recorded on the code BEFORE the mechanistic mode existed (dev @ 3e29c9f).
LABEL_DIGESTS = {
    "chemo_dose_mutator_persister": "58a4c98f370dbeacd25590228c7369e8",
    "targeted": "5c045e91a55268ae1ffbf582ecd93c7c",
    "immunotherapy": "e036d99c7163d531a5978f4c3c56c2b4",
    "state_induction": "31ea0cab8446f7fd3b0b8294262fa5dc",
    "exact_chemo": "d7c6436cc89e2ae96f48459a8e51a619",
}


@pytest.mark.parametrize("name", sorted(LABEL_SCENARIOS))
def test_label_mode_is_byte_identical(name):
    """(a) The default response, and an explicit response="label", reproduce the pre-change runs."""
    assert _digest(_run_label(name)) == LABEL_DIGESTS[name]
    assert _digest(_run_label(name, response="label")) == LABEL_DIGESTS[name]


# --- mechanistic mode -------------------------------------------------------------------------------
# No resistance loci, so nothing but the mechanism under test can spare a clone.
SEL0 = {**SELECTION, "prop_treatment_resistance": 0.0, "prop_drug_tolerance": 0.0,
        "prop_immune_resistance": 0.0}
# The founder carries a handful of passenger SNVs, so every lineage inherits them unless a CNA
# deletes the copy that carries one.
FOUNDER = {"n_passenger": 6}
EXPR = {"program_params": {"n_programs": 4, "n_genes_per_program": 12},
        "activity_params": {"n_active_programs_per_cell": 2, "activity_noise": 0.2},
        "coupling_params": {"phenotype_program_strength": {"division_rate": 3.0, "__default__": 0.5}},
        "dosage_params": {"dosage_sensitivity_mean": 0.8, "dosage_sensitivity_sd": 0.1,
                          "allele_specific": True}}


def _variant(t, gid, edit=None, cells=0, **evo):
    """Register a copy of genotype ``gid`` with some evolutionary parameters overridden (a different
    STATE on the same genome) and/or its genome edited, optionally placing ``cells`` of it."""
    child = t.genotypes[gid].divide()
    child.evolutionary_parameters.update(evo)
    if edit is not None:
        edit(child)
    child.set_genotype_id()
    t._register(child)
    t.genotypes_parents[child.genotype_id] = gid
    if cells:
        deme = next(i for i, d in enumerate(t.demes) if d)
        t._add(deme, child.genotype_id, cells)
    return child.genotype_id


def _edit_copies(seg, drop=0, add=0, drop_carrying=None):
    """A genome edit: remove ``drop`` copies of segment ``seg`` (the ones carrying the SNV at
    ``drop_carrying`` first, when given) and duplicate the first copy ``add`` times."""
    def edit(rep):
        rep.genome = list(rep.genome)
        rep.genome[seg] = {h: list(v) for h, v in rep.genome[seg].items()}
        removed = 0
        if drop_carrying is not None:
            for h in ("p", "m"):
                keep = [c for c in rep.genome[seg][h] if not c[drop_carrying]]
                removed += len(rep.genome[seg][h]) - len(keep)
                rep.genome[seg][h] = keep
        for _ in range(drop - removed):
            hap = "p" if rep.genome[seg]["p"] else "m"
            rep.genome[seg][hap].pop()
        for _ in range(add):
            hap = "p" if rep.genome[seg]["p"] else "m"
            rep.genome[seg][hap].append(rep.genome[seg][hap][0].copy())
        gs = dict(rep.genome_summary)
        gs["seg_cns"] = list(gs["seg_cns"])
        gs["seg_cns"][seg] = len(rep.genome[seg]["p"]) + len(rep.genome[seg]["m"])
        rep.genome_summary = gs
    return edit


def _snv_sites(rep):
    return sorted({(s, int(p)) for s in range(len(rep.genome)) for h in ("p", "m")
                   for c in rep.genome[s][h] for p in np.flatnonzero(c)})


def _carriers(t, site, amp=None):
    """(cells carrying the target at ``site``, cells not carrying it)."""
    car = non = 0
    for g, c in t.genotypes_counts.items():
        if c and t._is_cancer(g):
            if t._carries_target(t.genotypes[g], site, amp):
                car += c
            else:
                non += c
    return car, non


def _grown(seed=5, steps=25):
    t = _build(seed=seed, selection=SEL0, founder_mutations=FOUNDER)
    t.grow(n_steps=steps, seed=seed)
    return t


def _targeted(site, **kw):
    return TargetedTherapy(targets=[site], response="mechanistic", toxicity=0.0, kill_rate=1.5, **kw)


def test_mechanistic_targeted_therapy_kills_carriers_and_spares_a_subclone_without_it():
    """(b) A de novo subclonal mutation as the target: its carriers collapse, the rest grow on."""
    t = _grown()
    founder_sites = set(_snv_sites(t.genotypes[t.founder_id]))
    tot = sum(c for g, c in t.genotypes_counts.items() if t._is_cancer(g))
    subclonal = {s for g, c in t.genotypes_counts.items() if c and t._is_cancer(g)
                 for s in _snv_sites(t.genotypes[g])} - founder_sites
    site = min(sorted(subclonal), key=lambda s: abs(_carriers(t, s)[0] / tot - 0.4))
    car0, non0 = _carriers(t, site)
    assert car0 > 100 and non0 > 100                   # a real subclone, and a real rest

    tx = _targeted(site)
    t._apply_treatment(tx, 1.0)
    carrier = next(g for g, c in t.genotypes_counts.items()
                   if c and t._is_cancer(g) and t._carries_target(t.genotypes[g], site, None))
    spared = next(g for g, c in t.genotypes_counts.items()
                  if c and t._is_cancer(g) and not t._carries_target(t.genotypes[g], site, None))
    assert t._tx_death_add_for(carrier) > 0 and t._tx_death_add_for(spared) == 0

    t.grow(n_steps=5, seed=5, treatment=tx)
    car1, non1 = _carriers(t, site)
    assert car1 < 0.15 * car0                           # carriers are killed...
    assert non1 > non0                                  # ...the subclone without it grows on


def test_a_lineage_that_deletes_the_target_by_cna_stops_responding():
    """(c) A founder SNV as the target: every lineage inherits it, so the only clones that escape
    are those whose CNA deleted the copy carrying it — and those then take no drug."""
    t = _grown()
    founder = t.genotypes[t.founder_id]
    tot = sum(c for g, c in t.genotypes_counts.items() if t._is_cancer(g))
    site = min(_snv_sites(founder), key=lambda s: abs(_carriers(t, s)[1] / tot - 0.15))
    car0, non0 = _carriers(t, site)
    assert non0 > 20

    # each escaping lineage lost the target in a deletion: at the step where it stopped carrying
    # it, the segment lost a copy
    seg = site[0]
    for g, c in t.genotypes_counts.items():
        if not (c and t._is_cancer(g)) or t._carries_target(t.genotypes[g], site, None):
            continue
        child = g
        while not t._carries_target(t.genotypes[t.genotypes_parents[child]], site, None):
            child = t.genotypes_parents[child]
        parent = t.genotypes_parents[child]
        assert (t.genotypes[child].genome_summary["seg_cns"][seg]
                == t.genotypes[parent].genome_summary["seg_cns"][seg] - 1)

    tx = _targeted(site)
    t.grow(n_steps=5, seed=5, treatment=tx)
    car1, non1 = _carriers(t, site)
    assert non0 / (car0 + non0) < 0.25 and non1 / (car1 + non1) > 0.8
    t._apply_treatment(tx, 1.0)
    lost = [g for g, c in t.genotypes_counts.items()
            if c and t._is_cancer(g) and not t._carries_target(t.genotypes[g], site, None)]
    assert all(t._tx_death_add_for(g) == 0 for g in lost)


def test_an_amplification_counts_as_the_target_until_it_is_lost():
    t = _build(selection=SEL0)
    f = t.founder_id
    seg, pos = 1, 4                                     # unmutated: only copy number can make it
    amplified = _variant(t, f, edit=_edit_copies(seg, add=2))          # 4 copies
    reverted = _variant(t, amplified, edit=_edit_copies(seg, drop=1))  # back to 3
    tx = _targeted((seg, pos), amplification_threshold=3)
    t._apply_treatment(tx, 1.0)
    assert t._tx_death_add_for(amplified) > 0
    assert t._tx_death_add_for(f) == 0 and t._tx_death_add_for(reverted) == 0


def test_expression_targeted_kill_drops_for_a_state_with_low_target_expression():
    """(d) The target is a gene of the proliferation program. A slow-cycling STATE of the same
    genome runs that program lower, so it makes less target and takes less drug, in proportion; set
    the threshold between the two and it takes none."""
    t = _build(seed=5, selection=SEL0, expression_params=EXPR)
    P = t.programs
    gene = int(np.argmax(P.dictionary.loading[P.dictionary.program_index["proliferation"]]))
    f = t.founder_id
    slow = _variant(t, f, division_rate=0.8 * t.genotypes[f].evolutionary_parameters["division_rate"])

    tx = Chemotherapy(response="mechanistic", target_genes=[gene], kill_rate=1.5)
    t._apply_treatment(tx, 1.0)
    _, x_slow = t._mech_dose(tx, slow, t.genotypes[slow])
    assert t._mech_dose(tx, f, t.genotypes[f])[1] == 1.0          # the founder sets the nominal dose
    assert 0 < x_slow < 0.8
    h_f, h_slow = t._tx_death_add_for(f), t._tx_death_add_for(slow)
    assert 0 < h_slow < h_f and h_slow / h_f == pytest.approx(x_slow)

    ref = t._target_table(tx)["ref"]
    cut = Chemotherapy(response="mechanistic", target_genes=[gene], kill_rate=1.5,
                       expression_threshold=ref * (1 + x_slow) / 2)
    t._apply_treatment(cut, 1.0)
    assert t._tx_death_add_for(f) == pytest.approx(h_f) and t._tx_death_add_for(slow) == 0


def test_expression_follows_the_target_genes_copy_number():
    """The baseline half of expression: a clone that loses a copy of the target gene's segment makes
    less of it (2 of 3 dosage units in the program-free law), and a nullisomic one none."""
    t = _build(selection=SEL0)
    f = t.founder_id
    one = _variant(t, f, edit=_edit_copies(0, drop=1))
    none = _variant(t, f, edit=_edit_copies(0, drop=2))
    tx = Chemotherapy(response="mechanistic", target_genes=[(0, 3)])
    assert t._mech_dose(tx, one, t.genotypes[one])[1] == pytest.approx(2 / 3)
    assert t._mech_dose(tx, none, t.genotypes[none])[1] == 0.0


def test_mechanistic_cytotoxic_spares_slow_dividing_clones_more_than_fast_ones():
    """(e) Two clones on the same genome, dividing at 0.4 and 1.0, in equal numbers in the same
    demes, no mutation: under a mechanistic cytotoxic drug the hazard is proportional to division
    rate (net = b(1 - kill_rate) - death), so the slow clone keeps far more of its cells. The fast
    clone is labelled 90% treatment-resistant, which would spare it under the label model."""
    t = _build(selection=SEL0, cancer={**CANCER, "mutation_rate": 0.0},
               spatial={**SPATIAL, "immune_density": 0.0})
    f = t.founder_id
    slow = _variant(t, f, division_rate=0.4)
    fast = _variant(t, f, division_rate=1.0, treatment_resistance=0.9)
    for d, deme in enumerate(t.demes):
        if deme.get(f):
            t._remove(d, f, deme[f])
    G, c0 = t.grid_size, t.grid_size // 2
    for d in [(c0 + dr) * G + c0 + dc for dr in range(-2, 3) for dc in range(-2, 3)]:
        t._add(d, slow, 4)
        t._add(d, fast, 4)
    for d in range(len(t.demes)):
        t._refresh_rate(d)

    tx = Chemotherapy(response="mechanistic", kill_mode="proliferation", kill_rate=1.5, toxicity=0.0)
    t._apply_treatment(tx, 1.0)
    assert t._tx_death_add_for(slow) / t._tx_death_add_for(fast) == pytest.approx(0.4)
    t.grow(n_steps=4, seed=3, treatment=tx)
    surv_slow, surv_fast = t.genotypes_counts[slow] / 100, t.genotypes_counts[fast] / 100
    assert surv_slow > 0.2 and surv_slow > 4 * surv_fast


def test_mechanistic_mode_ignores_the_resistance_labels():
    """A clone shielded by every label takes the full mechanistic dose — and pays no persister
    cost — where the label model spares it."""
    t = _build(selection={**SELECTION, **STATE_ON})
    f = t.founder_id
    resistant = _variant(t, f, cells=1, treatment_resistance=1.0)
    tolerant = _variant(t, f, cells=1, drug_tolerance=1.0)
    stated = _variant(t, f, cells=1)
    t.genotypes[stated].resistance_state = 1.0
    shielded = (resistant, tolerant, stated)

    t._apply_treatment(Chemotherapy(kill_rate=1.5), 1.0)
    h_f = t._tx_death_add_for(f)
    assert all(t._tx_death_add_for(g) < 0.2 * h_f for g in shielded)
    assert tolerant in t._tx_div_mult

    t._apply_treatment(Chemotherapy(kill_rate=1.5, response="mechanistic"), 1.0)
    assert all(t._tx_death_add_for(g) == t._tx_death_add_for(f) == h_f for g in shielded)
    assert not t._tx_div_mult


def test_mechanistic_immunotherapy_strips_only_carriers_of_the_checkpoint():
    t = _build(selection=SEL0, founder_mutations=FOUNDER)
    f = t.founder_id
    site = _snv_sites(t.genotypes[f])[0]
    lost = _variant(t, f, edit=_edit_copies(site[0], drop_carrying=site[1]))
    for g in (f, lost):
        t.genotypes[g].evolutionary_parameters["immune_resistance"] = 0.5
    tx = Immunotherapy(immune_checkpoints=[site], response="mechanistic", toxicity=0.0)
    t._apply_treatment(tx, 1.0)
    assert t._tx_immune_resist_for(f, t.genotypes[f], 0.5) == pytest.approx(0.5 * (1 - 0.9))
    assert t._tx_immune_resist_for(lost, t.genotypes[lost], 0.5) == 0.5


def test_mechanistic_dose_scaled_mutator_follows_the_exposure():
    """mutagenicity_mode="dose" scales the mutator by the mechanistic exposure: the full multiplier
    at the founder's target level, a share at 2/3 of it, nothing below the threshold (and no flag,
    so a descendant that regains expression is mutagenized on its own terms)."""
    t = _build(selection=SEL0)
    f = t.founder_id
    one = _variant(t, f, cells=1, edit=_edit_copies(0, drop=1))
    none = _variant(t, f, cells=1, edit=_edit_copies(0, drop=2))
    rate = {g: t.genotypes[g].mutation_rate for g in (f, one, none)}
    tx = Chemotherapy(response="mechanistic", target_genes=[(0, 3)], expression_threshold=1e-12,
                      mutagenicity=3.0, mutagenicity_mode="dose")
    t._apply_treatment(tx, 1.0)
    assert t.genotypes[f].mutation_rate == pytest.approx(3.0 * rate[f])
    assert t.genotypes[one].mutation_rate == pytest.approx((1 + 2 * 2 / 3) * rate[one])
    assert t.genotypes[none].mutation_rate == rate[none]
    assert not getattr(t.genotypes[none], "_tx_mutagenized", False)


def test_mechanistic_knobs_are_validated():
    from iscc.treatment.treatment import Treatment
    from iscc.tumor.tumor import Tumor
    with pytest.raises(ValueError, match="response must be"):
        Treatment(response="bogus")
    with pytest.raises(ValueError, match="only apply with response='mechanistic'"):
        Chemotherapy(target_genes=[0])
    with pytest.raises(ValueError, match="only apply with response='mechanistic'"):
        TargetedTherapy(targets=[(0, 1)], amplification_threshold=3)
    with pytest.raises(ValueError, match="needs target_genes"):
        Chemotherapy(response="mechanistic", expression_threshold=1.0)
    t = _build(selection=SEL0)
    t.grow(n_steps=2, seed=3)
    with pytest.raises(ValueError, match="not on the genome"):
        t.grow(n_steps=2, seed=3,
               treatment=Chemotherapy(response="mechanistic", target_genes=[(9, 0)]))
    # the per-cell engines read the labels in Treatment.apply: refuse rather than run them
    with pytest.raises(NotImplementedError, match="GenotypeTumor only"):
        Tumor.grow(None, treatment=Chemotherapy(response="mechanistic"))
