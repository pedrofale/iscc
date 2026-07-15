"""R13 — gene-program expression backbone + dosage/ASE/SNV overlays (DESIGN_expression.md).

The properties that matter here, in order of how much they would hurt if they broke:

1. **Off => unchanged.** `expression_params=None` (the default) leaves the engine on its legacy
   expression path, and growth NEVER reads the program layer, so a tumour is byte-identical with
   programs on or off at a given seed (the F8 discipline: programs are a READOUT).
2. **Comparability.** The program landscape is a property of the GENOME, so two patients sharing a
   config share their programs exactly as they already share their oncogenes — and the sub-streams
   are independent, so changing `n_programs` cannot reshuffle the gene roles or `s_g`.
3. **The coupling routes** do what §3.1 says, and stay sparse.
4. **Alleles split**, so a BAF exists in RNA.
"""
import numpy as np
import pytest

from iscc.tumor.models import GenotypeTumor
from iscc.tumor.programs import ProgramModel

GENOME = {"n_segments": 6, "segment_size": 40}
SELECTION = {"prop_driver": 0.2}
DEME = {"carrying_capacity": 8, "initial_cancer_cells": 8}
SPATIAL = {"grid_size": 12, "structure_radius": 3}
CANCER = {"division_rate": 0.6, "death_rate": 0.03, "mutation_rate": 1.0, "dispersal_rate": 0.4}

PROGRAM_PARAMS = {"n_programs": 6, "n_genes_per_program": 12, "program_overlap": 0.1,
                  "loading_strength": {"mean": 1.0, "sd": 0.3}, "loading_sparsity": 1.0,
                  "program_genomic_scatter": 1.0}
EXPRESSION_PARAMS = {
    "program_params": PROGRAM_PARAMS,
    "activity_params": {"n_active_programs_per_cell": 3, "activity_dist": "lognormal",
                        "activity_mean": 1.0, "activity_sd": 0.5, "activity_noise": 0.2},
    "coupling_params": {"phenotype_program_strength": 0.5, "prop_program_regulator": 0.05,
                        "program_bias_strength": 0.5},
    "dosage_params": {"dosage_sensitivity_mean": 0.7, "dosage_sensitivity_sd": 0.25,
                      "dosage_saturation": 8, "allele_specific": True},
    "snv_effect_params": {"p_lof": 0.1, "p_missense": 0.3, "p_splice": 0.05, "p_silent": 0.55,
                          "nmd_strength": 0.2, "snv_expression_effect": 0.5},
}


def build(seed=1, steps=0, **kw):
    t = GenotypeTumor(seed=seed, genome_params=GENOME, selection_params=SELECTION,
                      cancer_cell_params=CANCER, deme_params=DEME, spatial_params=SPATIAL, **kw)
    if steps:
        t.grow(n_steps=steps, seed=seed)
    return t


def cancer_mask(t):
    ct = t.cell_data["cell_type"]["cell_id"].values
    return np.array([t.genotypes[g].type == "cancer" for g in ct])


# --------------------------------------------------------------- 1. off => unchanged
def test_off_by_default_keeps_base_schema():
    t = build(steps=3000)
    assert t.programs is None
    assert "cell_program" not in t.cell_data
    assert "cell_rna_baf" not in t.cell_data
    assert not hasattr(t, "program_truth")


def test_growth_is_identical_with_programs_on_or_off():
    """Programs are READOUT-ONLY: they must never perturb the evolutionary trajectory."""
    off = build(seed=3, steps=3000)
    on = build(seed=3, steps=3000, expression_params=EXPRESSION_PARAMS)
    assert off.get_tumor_size() == on.get_tumor_size()
    assert off.cell_data["cell_snv"].shape == on.cell_data["cell_snv"].shape
    # the genotype-level truth (who exists, and their genomes) is untouched
    np.testing.assert_array_equal(off.cell_data["cell_snv"].values, on.cell_data["cell_snv"].values)
    np.testing.assert_array_equal(off.cell_data["cell_cnv"].values, on.cell_data["cell_cnv"].values)
    # ... while expression DOES change (otherwise the layer would be doing nothing)
    assert not np.allclose(off.cell_data["cell_exp"].values, on.cell_data["cell_exp"].values)


def test_materialisation_is_reproducible():
    """The per-cell `z` is run-seeded, so re-materialising the same tumour reproduces the cells."""
    t = build(seed=5, steps=3000, expression_params=EXPRESSION_PARAMS)
    first = t.cell_data["cell_program"].values.copy()
    t.make_cell_data()
    np.testing.assert_allclose(first, t.cell_data["cell_program"].values)


# --------------------------------------------------------------- 2. comparability (mirrors cohort)
def test_program_landscape_is_shared_across_evolution_seeds():
    """Two 'patients' (same config, DIFFERENT evolution seed) must get the SAME programs — the same
    requirement the shared driver landscape already meets."""
    a = build(seed=1, expression_params=EXPRESSION_PARAMS)
    b = build(seed=999, expression_params=EXPRESSION_PARAMS)
    np.testing.assert_allclose(a.programs.dictionary.loading, b.programs.dictionary.loading)
    np.testing.assert_allclose(a.programs.dosage_sensitivity, b.programs.dosage_sensitivity)
    assert a.programs.dictionary.gene_program_map() == b.programs.dictionary.gene_program_map()
    assert a.programs.regulator_genes == b.programs.regulator_genes
    # and (already covered elsewhere, asserted here as the sibling property) the gene roles too
    np.testing.assert_array_equal(a.selection.get_oncogenes(), b.selection.get_oncogenes())


def test_different_layout_seed_gives_different_programs():
    a = build(seed=1, expression_params=EXPRESSION_PARAMS)
    b = build(seed=1, expression_params=EXPRESSION_PARAMS, layout_seed=7)
    assert not np.allclose(a.programs.dictionary.loading, b.programs.dictionary.loading)
    assert not np.allclose(a.programs.dosage_sensitivity, b.programs.dosage_sensitivity)


def test_n_programs_does_not_reshuffle_gene_roles_or_sg():
    """The independent-sub-stream property. A shared layout stream would couple these: changing
    `n_programs` would shift every later draw and silently give the tumour different oncogenes,
    breaking comparability between configs that differ only in program parameters."""
    ep_a = dict(EXPRESSION_PARAMS, program_params=dict(PROGRAM_PARAMS, n_programs=6))
    ep_b = dict(EXPRESSION_PARAMS, program_params=dict(PROGRAM_PARAMS, n_programs=11))
    a, b = build(expression_params=ep_a), build(expression_params=ep_b)
    # gene roles come from the untouched BASE layout stream
    np.testing.assert_array_equal(a.selection.get_oncogenes(), b.selection.get_oncogenes())
    np.testing.assert_array_equal(a.selection.get_tsgs(), b.selection.get_tsgs())
    # s_g lives on its own spawned sub-stream, so it is invariant to the program dictionary's size
    np.testing.assert_allclose(a.programs.dosage_sensitivity, b.programs.dosage_sensitivity)
    # ... and the dictionary itself did change
    assert a.programs.dictionary.loading.shape[0] == 6
    assert b.programs.dictionary.loading.shape[0] == 11


def test_f8_niche_programs_are_layout_seeded():
    """F8's hypoxia/CCI gene sets used to draw from the RUN seed, so every patient in a cohort got a
    different hypoxia programme. They are part of the genome, so they follow the layout stream."""
    mp = {"hypoxia": {"n_genes": 10, "strength": 0.5}, "cci": {"n_target_genes": 5, "strength": 0.5}}
    a, b = build(seed=1, microenv_params=mp), build(seed=999, microenv_params=mp)
    np.testing.assert_array_equal(a._hypoxia_genes, b._hypoxia_genes)
    np.testing.assert_array_equal(a._cci_target_genes, b._cci_target_genes)
    c = build(seed=1, microenv_params=mp, layout_seed=7)
    assert not np.array_equal(a._hypoxia_genes, c._hypoxia_genes)


# --------------------------------------------------------------- 3. the coupling routes (§3.1)
def _model(**coupling):
    ep = {"program_params": {"n_programs": 6, "n_genes_per_program": 10},
          "coupling_params": coupling}
    return ProgramModel(n_genes=100, segment_sizes=[50, 50], layout_seed=42, run_seed=0,
                        expression_params=ep)


def test_route1_phenotype_drives_its_program_and_nothing_else():
    P = _model(phenotype_program_strength=0.5)
    k = P.dictionary.program_index
    base = {"division_rate": 0.3, "dispersal_rate": 0.1}
    wt = {"division_rate": 0.3, "dispersal_rate": 0.1,
          "treatment_resistance": 0.0, "immune_resistance": 0.0}
    snv = np.zeros(100)
    # a wild-type clone sits at the origin
    np.testing.assert_allclose(P.clone_drive(wt, base, snv), 0.0)
    # a clone whose division rate doubled: proliferation = strength * (0.6/0.3 - 1) = 0.5
    fit = dict(wt, division_rate=0.6)
    d = P.clone_drive(fit, base, snv)
    assert d[k["proliferation"]] == pytest.approx(0.5)
    # sparse: no other program moved
    assert np.count_nonzero(d) == 1
    # resistance is already normalised to [0,1), so it maps straight through
    res = dict(wt, treatment_resistance=0.8)
    assert P.clone_drive(res, base, snv)[k["drug_resistance"]] == pytest.approx(0.4)


def test_route1_strength_zero_decouples_fitness_from_expression():
    P = _model(phenotype_program_strength=0.0)
    base = {"division_rate": 0.3, "dispersal_rate": 0.1}
    fit = {"division_rate": 0.6, "dispersal_rate": 0.1,
           "treatment_resistance": 0.9, "immune_resistance": 0.9}
    np.testing.assert_allclose(P.clone_drive(fit, base, np.zeros(100)), 0.0)


def test_route2_regulator_shifts_z_without_fitness():
    P = _model(phenotype_program_strength=0.0, prop_program_regulator=0.1,
               program_bias_strength=0.5, n_programs_per_regulator=1)
    assert P.regulator_genes, "expected some regulators at prop_program_regulator=0.1"
    g, targets = next(iter(P.regulator_genes.items()))
    snv = np.zeros(100)
    snv[g] = 1.0
    base = {"division_rate": 0.3, "dispersal_rate": 0.1}
    wt = {"division_rate": 0.3, "dispersal_rate": 0.1,
          "treatment_resistance": 0.0, "immune_resistance": 0.0}
    d = P.clone_drive(wt, base, snv)
    for prog, sign in targets:
        assert d[prog] == pytest.approx(0.5 * sign)


def test_route3_niche_field_drives_its_program():
    P = _model(niche_program_map={"hypoxia": "hypoxia"}, niche_program_strength=1.0)
    k = P.dictionary.program_index
    D = P.niche_drive({"hypoxia": np.array([0.0, 0.5, 1.0]), "cci": np.zeros(3)})
    np.testing.assert_allclose(D[:, k["hypoxia"]], [0.0, 0.5, 1.0])
    # only the mapped program moved
    assert np.count_nonzero(D) == 2


# --------------------------------------------------------------- 4. alleles / dosage / SNV
def test_alleles_split_and_baf_is_emitted():
    t = build(seed=1, steps=6000, expression_params=EXPRESSION_PARAMS)
    cd = t.cell_data
    can = cancer_mask(t)
    assert can.sum() > 0, "test tumour went extinct — no cancer cells to check"
    # the two allele layers reconstruct total expression
    np.testing.assert_allclose(cd["cell_exp_p"].values + cd["cell_exp_m"].values,
                               cd["cell_exp"].values)
    # normals are balanced by construction; cancer carries real allelic imbalance
    assert np.allclose(cd["cell_rna_baf"].values[~can], 0.5)
    baf_cancer = cd["cell_rna_baf"].values[can]
    assert (np.abs(baf_cancer - 0.5) > 1e-9).mean() > 0.05


def test_dosage_sensitivity_zero_makes_copy_number_invisible():
    """s_g = 0 is a fully buffered gene: expression must not track copy number. This is the knob that
    stops the forward model from BEING the linear-dosage law inferCNV/clonealign assume."""
    t = build(seed=1, steps=6000, expression_params=EXPRESSION_PARAMS)
    rep = next(t.genotypes[g] for g in t.genotypes_counts if t.genotypes[g].type == "cancer")
    rep.baseline_exp = t.celltype_exps["cancer"]
    n = t.n_genes
    ep0, em0 = rep.get_exp_alleles(dosage_sensitivity=np.zeros(n))
    ep1, em1 = rep.get_exp_alleles(dosage_sensitivity=np.ones(n))
    base = rep.baseline_exp / 2.0
    # buffered: every non-deleted haplotype contributes exactly base/2, whatever its copy number
    for out, hap in ((ep0, "p"), (em0, "m")):
        present = np.concatenate([np.full(t.selection.segment_sizes[s],
                                          len(rep.genome[s][hap]) > 0) for s in range(t.n_segments)])
        np.testing.assert_allclose(out[present], base[present])
    # linear: contribution scales with that haplotype's copy count
    for out, hap in ((ep1, "p"), (em1, "m")):
        cn = np.concatenate([np.full(t.selection.segment_sizes[s], len(rep.genome[s][hap]))
                             for s in range(t.n_segments)])
        np.testing.assert_allclose(out, base * cn)


def test_nullisomic_gene_is_silent_even_when_buffered():
    """No amount of dosage buffering rescues a gene with zero copies."""
    t = build(seed=1, expression_params=EXPRESSION_PARAMS)
    rep = t.genotypes[t.founder_id]
    rep.baseline_exp = t.celltype_exps["cancer"]
    rep.genome[0]["p"] = []
    rep.genome[0]["m"] = []
    ep, em = rep.get_exp_alleles(dosage_sensitivity=np.zeros(t.n_genes))
    lo, hi = rep._seg_offsets[0], rep._seg_offsets[1]
    assert np.all(ep[lo:hi] == 0) and np.all(em[lo:hi] == 0)


def test_snv_classes_are_drawn_and_separate_from_fitness():
    t = build(seed=1, expression_params=EXPRESSION_PARAMS)
    P = t.programs
    assert P.snv_class.shape == (t.n_genes,)
    assert set(np.unique(P.snv_class)) <= {0, 1, 2, 3}
    # silent/missense leave expression alone; LoF is NMD-degraded; splice is shifted
    np.testing.assert_allclose(P.snv_exp_effect[P.snv_class == 0], 1.0)   # silent
    np.testing.assert_allclose(P.snv_exp_effect[P.snv_class == 1], 1.0)   # missense
    np.testing.assert_allclose(P.snv_exp_effect[P.snv_class == 2], 0.5)   # splice
    np.testing.assert_allclose(P.snv_exp_effect[P.snv_class == 3], 0.2)   # lof -> NMD
    # the expression effect is NOT the fitness effect (they were one knob before R13)
    flat_fitness = np.concatenate(t.selection.mut_effects)
    assert not np.allclose(P.snv_exp_effect, flat_fitness)


# --------------------------------------------------------------- 5. the dictionary itself
def test_program_genomic_scatter_controls_positional_clustering():
    """`program_genomic_scatter` operationalises programs |= CNAs: at 1.0 a program is scattered
    genome-wide (functional); at 0.0 it is a contiguous block that MIMICS a copy-number segment."""
    def spread(scatter):
        P = ProgramModel(n_genes=240, segment_sizes=[40] * 6, layout_seed=42, run_seed=0,
                         expression_params={"program_params": {
                             "n_programs": 4, "n_genes_per_program": 12,
                             "program_overlap": 0.0, "program_genomic_scatter": scatter}})
        # mean gene-index span of a program, relative to the genome
        return np.mean([(g.max() - g.min()) / 240 for g in P.dictionary.program_genes])
    assert spread(0.0) < 0.2, "scatter=0 should give positionally CLUSTERED programs"
    assert spread(1.0) > 0.5, "scatter=1 should give programs spread genome-wide"


def test_dosage_sensitivity_is_bounded():
    t = build(expression_params=EXPRESSION_PARAMS)
    s = t.programs.dosage_sensitivity
    assert s.shape == (t.n_genes,)
    assert s.min() >= 0.0 and s.max() <= 1.0


# --------------------------------------------------------------- 5b. the COHORT layer (§4.3)
def _cohort(n=3, steps=3000, **kw):
    from iscc.cohort import Cohort
    return Cohort(patient_seeds=list(range(1, n + 1)), genome_params=GENOME,
                  selection_params=SELECTION, cancer_cell_params=CANCER, deme_params=DEME,
                  spatial_params=SPATIAL, grow_steps=steps, **kw).run()


def test_cohort_forwards_expression_params():
    co = _cohort(expression_params=EXPRESSION_PARAMS)
    for pr in co.patients:
        assert pr.tumor.programs is not None
        assert "cell_program" in pr.tumor.cell_data


def test_cohort_without_expression_params_is_unchanged():
    co = _cohort()
    for pr in co.patients:
        assert pr.tumor.programs is None
        assert "cell_program" not in pr.tumor.cell_data


def test_cohort_shares_programs_but_not_cnas():
    """The ground truth the cohort program benchmark rests on (DESIGN_expression.md §4.3): patients
    SHARE the program dictionary (it is a property of the genome, drawn from the layout stream) while
    their CNA landscapes are PRIVATE (private evolution) — so 'shared program' vs 'patient-specific
    biology' is a known split, not an assumption."""
    co = _cohort(n=3, steps=6000, expression_params=EXPRESSION_PARAMS)
    ts = [pr.tumor for pr in co.patients]
    # SHARED: dictionary, s_g, gene roles
    for t in ts[1:]:
        np.testing.assert_allclose(ts[0].programs.dictionary.loading, t.programs.dictionary.loading)
        np.testing.assert_allclose(ts[0].programs.dosage_sensitivity, t.programs.dosage_sensitivity)
        np.testing.assert_array_equal(ts[0].selection.get_oncogenes(), t.selection.get_oncogenes())
    # PRIVATE: the per-patient copy-number profile must actually differ
    cn = [t.cell_data["cell_cnv"].values.mean(0) for t in ts]
    assert any(not np.allclose(cn[0], c) for c in cn[1:]), "patients must have private CNA landscapes"


def test_cohort_forwards_microenv_params():
    """Sibling gap closed at the same time: F8 was never reachable through the cohort layer either."""
    mp = {"hypoxia": {"n_genes": 8, "strength": 0.5}}
    co = _cohort(n=2, expression_params=None, microenv_params=mp)
    for pr in co.patients:
        assert len(pr.tumor._hypoxia_genes) > 0
    # and the niche programme is SHARED across patients (the layout stream), as F8's fix requires
    np.testing.assert_array_equal(co.patients[0].tumor._hypoxia_genes,
                                  co.patients[1].tumor._hypoxia_genes)


# --------------------------------------------------------------- 6. the diagnose() check
def _clone_is_state(t):
    d = t.diagnose()
    return next(c for c in d.checks if c.name == "clone_is_state"), d


def test_diagnose_flags_clone_equals_state():
    """`z` collapsing to the per-clone drive makes clone-vs-state clustering trivially easy. It needs
    ALL of activity_sd=0, activity_noise=0 and no activation mask — any one gives per-cell spread."""
    ep = {"program_params": PROGRAM_PARAMS,
          "activity_params": {"activity_sd": 0.0, "activity_noise": 0.0},
          "coupling_params": {"phenotype_program_strength": 0.5}}
    t = build(seed=1, steps=6000, expression_params=ep)
    check, d = _clone_is_state(t)
    assert not check.ok, "expected the clone==state regime to be flagged"
    assert d.metrics["program_within_clone_frac"] == pytest.approx(0.0, abs=1e-9)


def test_diagnose_passes_with_within_clone_heterogeneity():
    t = build(seed=1, steps=6000, expression_params=EXPRESSION_PARAMS)
    check, d = _clone_is_state(t)
    assert check.ok
    assert d.metrics["program_within_clone_frac"] > 0.05


def test_diagnose_skips_program_check_when_programs_are_off():
    t = build(seed=1, steps=6000)
    check, _ = _clone_is_state(t)
    assert check.skipped


def test_program_truth_is_surfaced():
    t = build(seed=1, steps=3000, expression_params=EXPRESSION_PARAMS)
    tr = t.program_truth
    assert tr["loading"].shape == (6, t.n_genes)
    assert len(tr["program_names"]) == 6
    assert tr["dosage_sensitivity"].shape == (t.n_genes,)
    assert tr["snv_class"].shape == (t.n_genes,)
    assert t.cell_data["cell_program"].shape[1] == 6
