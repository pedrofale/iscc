"""Cohort layer and the seed decoupling it relies on.

Covers:
  * seed decoupling — two GenotypeTumor(same config, different EVOLUTION seed) share IDENTICAL
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
import functools

from iscc.cohort import (Cohort, Subgroup, assign_batches, pool_cell_data, run_cohort_batches,
                         concat_cohort_batches, recurrence_table, true_recurrent_drivers,
                         private_mutation_table, shared_private_labels, subgroup_response_table,
                         emit_cell_hashtags, demux_hashtags, grow_tumor, shared_selection,
                         assign_subgroups, draw_germline_markers, patient_overrides, clone_key)

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


# ============================ layout vs evolution seed ===================================
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


# --------- layout_seed alone reproduces the layout, without an explicit layout rng ----------
# The engines always hand Selection a seeded ``rng=layout_rng``, but probes/tests/notebooks build a
# Selection directly to inspect the gene layout. Then ``layout_seed`` must be sufficient on its own,
# or an unseeded generator would re-draw the driver/dispersal/TR/IR loci on every call.
def _bare_selection(**kw):
    from iscc.tumor.components.selection import Selection
    return Selection(n_segments=GENOME["n_segments"], segment_size=GENOME["segment_size"],
                     **{**SELECTION, **kw})


def _roles(sel):
    return {name: list(getattr(sel, "get_" + name)())
            for name in ("oncogenes", "tsgs", "dispersal_genes",
                         "treatment_resistant", "immune_resistant")}


def test_layout_seed_alone_reproduces_layout_without_rng():
    """Same layout_seed, no explicit rng -> identical treatment-resistance and driver loci."""
    a, b = _bare_selection(layout_seed=11), _bare_selection(layout_seed=11)
    assert _roles(a) == _roles(b)
    for seg in range(GENOME["n_segments"]):
        assert np.array_equal(a.treatment_resistance[seg], b.treatment_resistance[seg])
        assert np.array_equal(a.drivers[seg], b.drivers[seg])
    # still a real function of the seed, not a layout frozen for every seed
    assert _roles(_bare_selection(layout_seed=12)) != _roles(a)


def test_bare_selection_matches_the_engine_layout():
    """The default rng is the engine's own layout stream, so a directly-constructed Selection sees
    exactly the loci the engine evolves — that is what makes such a probe trustworthy."""
    probe = _bare_selection(layout_seed=DEFAULT_LAYOUT_SEED)
    assert _roles(probe) == _roles(_bare_selection())                      # default == the constant
    assert _roles(probe) == _roles(_tumor(seed=3).selection)               # == what the engine drew


# ================================== the Cohort wrapper ==================================
def _grow(**over):
    """One tumour's growth function at this file's config (what every patient of `_cohort` runs)."""
    kw = dict(genome_params=GENOME, selection_params=SELECTION, cancer_cell_params=CANCER,
              deme_params=DEME, spatial_params=SPATIAL, grow_steps=160)
    return functools.partial(grow_tumor, **{**kw, **over})


def _cohort(n=6, **kw):
    return Cohort(_grow(), patient_seeds=list(range(1, n + 1)), **kw)


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
    per_patient = assign_subgroups(6, subs)
    kw = patient_overrides(6, selection_params=SELECTION, subgroups=per_patient)
    co = _cohort(patient_kwargs=kw, subgroups=per_patient)
    assert co.subgroup_assignment == ["S", "R", "S", "R", "S", "R"]
    # a subgroup is a per-patient override of the shared selection params, nothing more
    assert kw[1]["selection_params"] == {**SELECTION, "treatment_resistant_effects": 3.0}
    co.run()
    assert co.patients[1].tumor.selection.treatment_resistant_effects == 3.0
    rt = subgroup_response_table(co)
    assert set(rt["subgroup"]) == {"S", "R"}
    assert rt.loc[0, "therapy_response"] == 1 and rt.loc[1, "therapy_response"] == 0


def test_germline_mutations_in_all_cell_types():
    """A `germline_mutations` variant is present in EVERY cell of the patient — the founder cancer
    cells AND the normal (epithelial/stromal) cells — because germline variants are carried by every
    cell of an individual, not just the tumour (this is the mechanism the private demux markers use)."""
    marks = [3, 11, 27]
    subs = [Subgroup("G", germline_mutations=tuple(marks))]
    spatial = {"grid_size": 15, "structure_radius": 5}    # seeds epithelial/stromal normal cells
    # Grow a solid gland (real per-deme crowding caps demes, so a tiny duct would hold only a few
    # cancer cells) and keep CNA low: the marks are the germline-INJECTION test, and CNA-driven LOH
    # could otherwise delete the (single-allele) germline variant in a small tumour's dominant clone.
    cancer = {**CANCER, "snv_prob": 0.95, "cnv_prob": 0.05}
    deme = {**DEME, "carrying_capacity": 8}
    co = Cohort(_grow(cancer_cell_params=cancer, deme_params=deme, spatial_params=spatial,
                      grow_steps=300),
                patient_seeds=[1], patient_kwargs=patient_overrides(1, SELECTION, subs),
                subgroups=subs).run()
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
        co = Cohort(functools.partial(grow_tumor, genome_params=genome, selection_params=sel,
                                      cancer_cell_params=cancer, deme_params=deme,
                                      spatial_params=spatial, grow_steps=320),
                    patient_seeds=[seed])
        chemo = Chemotherapy(start=120, effectiveness=0.95, toxicity=0.01, kill_rate=1.8, rate_multiplier=2.5)
        return co.grow_patient(0, treatment=chemo).get_cancer_size()

    seeds = (1, 2, 3)
    sens = sum(treated(s, 1.0) for s in seeds)
    res = sum(treated(s, 6.0) for s in seeds)
    assert res > 3 * sens        # the resistant subtype carries a much larger post-therapy burden


def test_germline_markers_are_disjoint_and_private():
    markers = draw_germline_markers(shared_selection(GENOME, SELECTION), n_patients=6, n_markers=20)
    co = _cohort(patient_kwargs=patient_overrides(6, germline_markers=markers)).run()
    marks = [set(markers[i].tolist()) for i in range(co.n_patients)]
    # disjoint across patients (private) and none overlap a driver
    for i in range(len(marks)):
        for j in range(i + 1, len(marks)):
            assert marks[i].isdisjoint(marks[j])
    drivers = set(int(x) for x in co.selection.get_oncogenes()) | set(int(x) for x in co.selection.get_tsgs())
    assert all(m.isdisjoint(drivers) for m in marks)
    # and each patient's founder actually carries its own markers, and no other patient's
    for i, pr in enumerate(co.patients):
        founder = pr.tumor.genotypes[pr.tumor.founder_id].get_snvs() > 0
        assert founder[markers[i]].all()
        assert not founder[markers[(i + 1) % 6]].any()


# ============================== the thin utility's own contract ===============================
def test_cohort_calls_grow_with_each_seed_and_the_shared_layout_seed():
    """Cohort owns no tumour model: it calls `grow` once per patient with the patient's seed, the
    shared layout seed, and the patient's overrides on top of any call-time overrides."""
    calls = []

    def grow(seed, layout_seed, **kw):
        calls.append((seed, layout_seed, kw))
        return ("tumour", seed)

    co = Cohort(grow, patient_seeds=[11, 12, 13], layout_seed=5,
                patient_kwargs=[{}, {"a": 1}, {"a": 2, "b": 3}])
    assert co.grow_patient(1) == ("tumour", 12)
    assert co.grow_patient(2, b=9)                       # call-time overrides win
    assert calls == [(12, 5, {"a": 1}), (13, 5, {"a": 2, "b": 9})]
    calls.clear()
    co.run()
    assert [c[0] for c in calls] == [11, 12, 13] and {c[1] for c in calls} == {5}
    assert [p.patient_id for p in co.patients] == [0, 1, 2]
    assert co.subgroup_assignment == ["all"] * 3


def test_cohort_patient_equals_the_single_tumour():
    """A cohort grows exactly as its single tumours would: patient i is bit-identical to the same
    growth function called directly with that seed."""
    co = _cohort(n=2)
    a = co.grow_patient(1)
    b = _grow()(seed=2)
    assert np.array_equal(a.cell_data["cell_snv"].values, b.cell_data["cell_snv"].values)
    assert np.array_equal(a.cell_data["cell_cnv"].values, b.cell_data["cell_cnv"].values)


def test_cohort_rejects_inconsistent_inputs():
    with pytest.raises(TypeError, match="grows one tumour"):
        Cohort(None, patient_seeds=[1])
    with pytest.raises(ValueError, match="entries for 2 patients"):
        Cohort(_grow(), patient_seeds=[1, 2], patient_kwargs=[{}])
    with pytest.raises(ValueError, match="may not set"):
        Cohort(_grow(), patient_seeds=[1], patient_kwargs=[{"seed": 3}])
    with pytest.raises(RuntimeError, match="run the cohort first"):
        Cohort(_grow(), patient_seeds=[1]).selection


def test_clone_key_separates_patients_with_the_same_genotype_id():
    """Genotype ids come from a process-wide counter, so two patients grown in separate Slurm tasks
    reuse the same ids. Simulated here by restarting the counter between two patients: their ids
    collide, and the (patient, clone) key keeps them apart."""
    from iscc.constants import normal_names
    from iscc.tumor.components.cell import Cell
    start = Cell._genotype_counter
    a = _grow(grow_steps=60)(seed=1)
    after_a = Cell._genotype_counter
    Cell._genotype_counter = start                       # a fresh process for the second patient
    b = _grow(grow_steps=60)(seed=2)
    Cell._genotype_counter = max(after_a, Cell._genotype_counter)
    ids_a = set(a.genotypes_counts) - set(normal_names)
    ids_b = set(b.genotypes_counts) - set(normal_names)
    assert ids_a & ids_b, "the two tasks should reuse genotype ids"
    keys_a = {clone_key(0, g) for g in ids_a}
    keys_b = {clone_key(1, g) for g in ids_b}
    assert keys_a.isdisjoint(keys_b)
    assert clone_key(3, "17") == "P3::17"


def test_cohort_grows_the_paper_regime_with_a_planted_network():
    """The paper's own growth function is a Cohort's `grow` (through functools.partial), the layout
    seed reaches the tumour, and a planted epistasis network rides in through selection_params."""
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    "validation", "paper"))
    import regime as RG
    tiny = dict(grid_size=24, n_glands=2, min_gland_sep=11,
                origin_confinement=dict(RG.ORIGIN_CONFINEMENT, generations=10))
    ep = {"n_events": 6, "event_size": 10, "lethal_death_rate": 2.5,
          "planted": {"effects": 0.05, "interactions": [(0, 1, 0.4)], "exclusive": [(2, 3)],
                      "constraints": [(4, 5)]}}
    grow = functools.partial(RG.grow, scale="small", target_cancer=300, spatial=tiny,
                             selection={"epistasis_params": ep,
                                        "dependency_params": {"gating_mode": "accessibility"}})
    co = Cohort(grow, patient_seeds=[5, 6])
    a = co.grow_patient(1)
    b = grow(seed=6)                                     # the same tumour, grown on its own
    assert a.reached_target and a.layout_seed == DEFAULT_LAYOUT_SEED
    assert a.get_cancer_size() == b.get_cancer_size()
    assert sorted((int(a.genotypes[g].ord), c) for g, c in a.genotypes_counts.items()) == \
        sorted((int(b.genotypes[g].ord), c) for g, c in b.genotypes_counts.items())
    gt = a.epistasis_ground_truth()
    assert gt["planted"] and gt["edges"] == [(0, 1, 0.4)] and gt["exclusive_pairs"] == [(2, 3)]
    assert gt["dag_edges"] == [(4, 5)] and gt["gating_mode"] == "accessibility"
    # the layout seed is forwarded: another one draws other event modules (built only, not grown)
    other = Cohort(grow, patient_seeds=[5], layout_seed=7).grow_patient(0, max_steps=0)
    assert other.layout_seed == 7
    assert not all(np.array_equal(x, y) for x, y in zip(other.selection.epistasis.event_genes,
                                                        a.selection.epistasis.event_genes))


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


# ============================ RNA-modality demux: cell hashing ==========================
def test_cell_hashing_singlets_assigned_doublets_detected():
    """Cell hashing (the RNA-modality demux): each cell's dominant hashtag recovers its patient
    (near-perfect on singlets), and true doublets score higher on the doublet statistic — the real
    challenge under cell super-loading."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    truth = np.repeat(np.arange(6), 120)                 # 6 patients, 120 cells each
    counts, hashtags, is_doublet = emit_cell_hashtags(truth, ambient_frac=0.08, doublet_rate=0.12, seed=1)
    assert counts.shape == (len(truth), 6)
    assign, doublet_score = demux_hashtags(counts)
    pred = hashtags[assign]
    singlet = ~is_doublet
    # singlets are assigned to the correct patient by their dominant hashtag
    assert (pred[singlet] == truth[singlet]).mean() > 0.98
    # true doublets are detectable (co-dominant second hashtag -> higher doublet score)
    assert 0 < is_doublet.sum() < len(is_doublet)
    assert roc_auc_score(is_doublet.astype(int), doublet_score) > 0.9
