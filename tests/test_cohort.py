"""Cohort layer + the prerequisite seed-decoupling fix (DESIGN_cohort.md).

Covers:
  * the ENGINE FIX — two GenotypeTumor(same config, different EVOLUTION seed) share IDENTICAL
    driver/oncogene/TSG/dispersal/TR/IR gene sets AND baseline expression (comparability by default),
    while their grown per-cell mutations differ; an explicit distinct layout_seed changes the layout;
  * the Cohort wrapper — N patients over the shared landscape (shared drivers, private evolution),
    subgroups + truncal founder mutations, per-patient private germline markers;
  * patient->batch multiplexing (1:1 and N:1) reusing the scRNA batch machinery, with the
    patient-of-origin / subgroup ground truth on the emitted batches;
  * the cohort ground-truth tables (recurrence, private mutations, shared-vs-private, subgroup response).
"""
import numpy as np
import pytest

from iscc.tumor.models import GenotypeTumor
from iscc.constants import DEFAULT_LAYOUT_SEED
from iscc.cohort import (Cohort, Subgroup, assign_batches, pool_cell_data, run_cohort_batches,
                         concat_cohort_batches, recurrence_table, true_recurrent_drivers,
                         private_mutation_table, shared_private_labels, subgroup_response_table)

GENOME = {"n_segments": 5, "segment_size": 40}
SELECTION = {"prop_driver": 0.12, "prop_dispersal": 0.1, "prop_immune_resistance": 0.1,
             "prop_treatment_resistance": 0.1, "driver_effects": 1.2}
DEME = {"carrying_capacity": 6, "initial_cancer_cells": 4}
SPATIAL = {"grid_size": 11, "structure_radius": 0}
CANCER = {"division_rate": 0.6, "death_rate": 0.05, "max_birth_rate": 0.95,
          "mutation_rate": 0.9, "dispersal_rate": 0.3}


def _tumor(seed, layout_seed=None):
    return GenotypeTumor(seed=seed, layout_seed=layout_seed, genome_params=GENOME,
                         selection_params=SELECTION, cancer_cell_params=CANCER,
                         deme_params=DEME, spatial_params=SPATIAL)


# ============================ the engine seed-decoupling fix =============================
def test_same_config_different_seed_shares_driver_layout():
    """Comparability by default: two runs of the SAME config with DIFFERENT evolution seeds share
    every gene-role identity and the baseline expression (so recurrence/cohort analysis is meaningful)."""
    a, b = _tumor(seed=1), _tumor(seed=2)
    for get in ("get_oncogenes", "get_tsgs", "get_dispersal_genes",
                "get_treatment_resistant", "get_immune_resistant"):
        assert list(getattr(a.selection, get)()) == list(getattr(b.selection, get)()), get
    for ct in ("cancer", "epithelial", "stromal", "immune"):
        assert np.array_equal(a.celltype_exps[ct], b.celltype_exps[ct])


def test_evolution_still_private_across_seeds():
    """The shared landscape must NOT collapse the private dynamics: grown per-cell mutations differ."""
    a, b = _tumor(seed=1), _tumor(seed=2)
    a.grow(n_steps=120, seed=1)
    b.grow(n_steps=120, seed=2)
    # different evolution -> different private mutation matrices (shapes may differ; compare content)
    va, vb = a.cell_data["cell_snv"].values, b.cell_data["cell_snv"].values
    assert va.shape != vb.shape or not np.array_equal(va, vb)


def test_explicit_layout_seed_changes_layout():
    """The layout_seed knob works: a different explicit layout_seed gives a different driver layout."""
    a = _tumor(seed=1, layout_seed=DEFAULT_LAYOUT_SEED)
    c = _tumor(seed=1, layout_seed=7)
    assert list(a.selection.get_oncogenes()) != list(c.selection.get_oncogenes())


def test_default_layout_seed_is_the_constant():
    a = _tumor(seed=999)
    assert a.layout_seed == DEFAULT_LAYOUT_SEED
    b = _tumor(seed=999, layout_seed=DEFAULT_LAYOUT_SEED)
    assert list(a.selection.get_oncogenes()) == list(b.selection.get_oncogenes())


# ================================== the Cohort wrapper ==================================
def _cohort(n=6, **kw):
    return Cohort(patient_seeds=list(range(1, n + 1)), genome_params=GENOME,
                  selection_params=SELECTION, cancer_cell_params=CANCER, deme_params=DEME,
                  spatial_params=SPATIAL, grow_steps=160, **kw)


def test_cohort_runs_shared_landscape_private_evolution():
    co = _cohort().run()
    assert len(co.patients) == 6
    onc = [list(p.tumor.selection.get_oncogenes()) for p in co.patients]
    assert all(o == onc[0] for o in onc)                 # shared driver identities
    snv = [p.cell_data["cell_snv"].values for p in co.patients]
    # not all patients identical (private evolution)
    assert not all(s.shape == snv[0].shape and np.array_equal(s, snv[0]) for s in snv[1:])


def test_subgroups_assignment_and_response():
    subs = [Subgroup("S", {"treatment_resistant_effects": 1.0}, therapy_response=1),
            Subgroup("R", {"treatment_resistant_effects": 3.0}, therapy_response=0)]
    co = _cohort(subgroups=subs)
    assert co.subgroup_assignment == ["S", "R", "S", "R", "S", "R"]
    co.run()
    rt = subgroup_response_table(co)
    assert set(rt["subgroup"]) == {"S", "R"}
    assert rt.loc[0, "therapy_response"] == 1 and rt.loc[1, "therapy_response"] == 0


def test_germline_mutations_in_all_cell_types():
    """A `germline_mutations` variant is present in EVERY cell of the patient — the founder cancer
    cells AND the normal (epithelial/stromal) cells — because germline variants are carried by every
    cell of an individual, not just the tumour (this is the mechanism the private demux markers use)."""
    marks = [3, 11, 27]
    subs = [Subgroup("G", germline_mutations=tuple(marks))]
    spatial = {"grid_size": 9, "structure_radius": 2}     # seeds epithelial/stromal normal cells
    co = Cohort(patient_seeds=[1], genome_params=GENOME, selection_params=SELECTION,
                cancer_cell_params=CANCER, deme_params=DEME, spatial_params=spatial,
                grow_steps=120, subgroups=subs).run()
    snv = co.patients[0].cell_data["cell_snv"].values
    ct = co.patients[0].cell_data["cell_type"].iloc[:, 0].astype(str).values
    genos = co.patients[0].tumor.genotypes
    types = np.array([genos[g].type for g in ct])
    assert (types == "cancer").any() and np.isin(types, ["epithelial", "stromal"]).any()
    # carried by cancer cells AND by normal cells (germline is in every cell of the individual)
    assert (snv[types == "cancer"][:, marks] > 0).mean() > 0.9
    normal = np.isin(types, ["epithelial", "stromal"])
    assert (snv[normal][:, marks] > 0).all()


def test_resistance_EMERGES_and_drives_differential_response():
    """Resistance is NOT seeded — it emerges from mutation + selection. Two runs of the SAME patient
    (same seed/layout, so identical emergent standing variation) that differ ONLY in
    `treatment_resistant_effects` diverge under adjuvant therapy: the high-effect (resistant) subtype
    relapses from the SELECTED emergent resistance mutations while the low-effect (sensitive) subtype is
    eradicated. The emergent resistance mutations are inert in the sensitive run — nothing was imposed."""
    from iscc.treatment.chemotherapy import Chemotherapy
    genome = {"n_segments": 8, "segment_size": 80}
    deme = {"carrying_capacity": 10, "initial_cancer_cells": 8}
    spatial = {"grid_size": 15, "structure_radius": 0}
    cancer = {"division_rate": 0.6, "death_rate": 0.08, "max_birth_rate": 0.98,
              "mutation_rate": 0.9, "dispersal_rate": 0.3}

    def treated(seed, tr_eff):
        sel = {"prop_driver": 0.04, "prop_dispersal": 0.0, "prop_immune_resistance": 0.0,
               "prop_treatment_resistance": 0.06, "driver_effects": 1.15,
               "treatment_resistant_effects": tr_eff}
        co = Cohort(patient_seeds=[seed], genome_params=genome, selection_params=sel,
                    cancer_cell_params=cancer, deme_params=deme, spatial_params=spatial, grow_steps=320)
        chemo = Chemotherapy(start=120, effectiveness=0.95, toxicity=0.01, kill_rate=1.8, rate_multiplier=2.5)
        return co.grow_patient(0, treatment=chemo).get_cancer_size()

    seeds = (1, 2, 3)
    sens = sum(treated(s, 1.0) for s in seeds)
    res = sum(treated(s, 6.0) for s in seeds)
    assert res > 3 * sens        # the resistant subtype carries a much larger post-therapy burden


def test_germline_markers_are_disjoint_and_private():
    co = _cohort(n_germline_markers=20).run()
    marks = [set(co.germline_markers[i].tolist()) for i in range(co.n_patients)]
    # disjoint across patients (private) and none overlap a driver
    for i in range(len(marks)):
        for j in range(i + 1, len(marks)):
            assert marks[i].isdisjoint(marks[j])
    drivers = set(int(x) for x in co.selection.get_oncogenes()) | set(int(x) for x in co.selection.get_tsgs())
    assert all(m.isdisjoint(drivers) for m in marks)


# ================================ patient -> batch mapping ==============================
def test_assign_batches_one_to_one_and_multiplex():
    asg, batches = assign_batches(6, mapping="one_to_one")
    assert asg == {i: i for i in range(6)} and len(batches) == 6
    asg2, batches2 = assign_batches(7, mapping="multiplex", capacity=3)
    assert batches2 == {0: [0, 1, 2], 1: [3, 4, 5], 2: [6]}
    asg3, batches3 = assign_batches(4, explicit={0: 0, 1: 0, 2: 1, 3: 1})
    assert batches3 == {0: [0, 1], 1: [2, 3]}


def test_pool_namespaces_and_labels():
    co = _cohort(n=4).run()
    pooled, meta = pool_cell_data([co.patients[p] for p in range(4)], n_cells_per_patient=20)
    assert all(name.startswith("P") and "::" in name for name in meta.index)
    assert set(meta["patient"]) == {0, 1, 2, 3}
    assert set(meta["cell_type"]) <= {"cancer", "epithelial", "stromal", "immune"}
    # gene columns align across patients (shared genome)
    assert pooled["cell_exp"].shape[1] == co.patients[0].n_genes if hasattr(co.patients[0], "n_genes") \
        else pooled["cell_exp"].shape[1] == co.patients[0].tumor.n_genes


def test_run_cohort_batches_carries_ground_truth():
    co = _cohort(n=6).run()
    assays, batches, asg = run_cohort_batches(co, mapping="multiplex", capacity=3,
                                              n_cells_per_patient=25)
    assert len(assays) == 2                       # 6 patients / capacity 3
    for a in assays:
        assert "patient" in a.obs.columns and "subgroup" in a.obs.columns
    comb = concat_cohort_batches(assays)
    assert set(comb.obs["patient"].astype(int)) == set(range(6))
    assert comb.uns["n_batches"] == 2


# ================================ cohort ground truth ==================================
def test_recurrence_table_annotates_drivers():
    co = _cohort(n=8).run()
    rt = recurrence_table(co)
    assert {"is_oncogene", "is_tsg", "is_driver", "recurrence", "is_recurrent"} <= set(rt.columns)
    # the known recurrent-driver answer key is the onc U tsg gene set
    assert len(true_recurrent_drivers(co)) == int(rt["is_driver"].sum())
    # recurrence is a fraction in [0, 1]
    assert rt["recurrence"].between(0, 1).all()


def test_private_and_shared_private_labels():
    co = _cohort(n=6).run()
    pm = private_mutation_table(co)
    assert set(pm.columns) == {"patient", "gene", "gene_idx", "is_driver"}
    pooled, meta = pool_cell_data([co.patients[p] for p in range(6)], n_cells_per_patient=20)
    sp = shared_private_labels(meta)
    assert "shared_state" in sp.columns and "private_state" in sp.columns
    # shared axis = coarse cell type; private axis = patient
    assert set(sp["shared_state"]) <= {"cancer", "epithelial", "stromal", "immune"}
    assert set(sp["private_state"]) == set(str(i) for i in range(6))
