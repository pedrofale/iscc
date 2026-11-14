"""Tests for the genotype-level (count-based) engine, GenotypeTumor (DESIGN phase 3b).

Validates: builds/grows/materialises the standard schema, is reproducible (seed -> identical
matrices), and is *statistically equivalent* to the cell-level GlandularTumor (it is NOT
byte-identical — it draws different random variables — so we check survival and survivor-size
distributions match, not exact values).
"""
import hashlib

import numpy as np
import pytest

from conftest import (
    GENOME_PARAMS, SELECTION_PARAMS, CANCER_CELL_PARAMS, N_SEGMENTS, SEGMENT_SIZE,
    EPITHELIAL_CELL_PARAMS, STROMAL_CELL_PARAMS, IMMUNE_CELL_PARAMS, DEME_PARAMS,
)
from iscc.tumor.models import GenotypeTumor, GlandularTumor

SPATIAL = {"grid_size": 15, "n_structures": 1, "structure_radius": 0}
# death_rate 0 -> the founder can never stochastically die out, so schema/reproducibility
# checks are deterministic. The equivalence test uses a small death rate on purpose.
NO_DEATH = {**CANCER_CELL_PARAMS, "death_rate": 0.0}
LOW_DEATH = {**CANCER_CELL_PARAMS, "death_rate": 0.05}


def _count(seed, steps, cancer=NO_DEATH):
    t = GenotypeTumor(
        seed=seed, genome_params=GENOME_PARAMS, selection_params=SELECTION_PARAMS,
        cancer_cell_params=cancer, deme_params=DEME_PARAMS, spatial_params=SPATIAL,
    )
    t.grow(n_steps=steps, seed=seed)
    return t


def _cell(seed, steps, cancer=NO_DEATH):
    t = GlandularTumor(
        seed=seed, genome_params=GENOME_PARAMS, selection_params=SELECTION_PARAMS,
        cancer_cell_params=cancer, epithelial_cell_params=EPITHELIAL_CELL_PARAMS,
        stromal_cell_params=STROMAL_CELL_PARAMS, immune_cell_params=IMMUNE_CELL_PARAMS,
        deme_params=DEME_PARAMS, grid_size=SPATIAL["grid_size"], structure_radius=0,
    )
    t.grow(n_steps=steps, seed=seed)
    return t


def test_builds_grows_and_materialises_schema():
    t = _count(seed=1, steps=150)
    cd = t.cell_data
    assert {"cell_snv", "cell_cnv", "cell_exp", "cell_crd", "cell_type", "cell_deme", "cell_evo"} <= set(cd)
    n = t.get_tumor_size()
    assert n > 0
    assert cd["cell_snv"].shape == (n, t.n_genes)
    assert list(cd["cell_crd"].columns) == ["row", "col"]
    # every materialised cell maps to a live genotype
    assert set(cd["cell_type"]["cell_id"]).issubset(set(t.genotypes_counts))


def test_reproducible_same_seed():
    def fingerprint(seed):
        t = _count(seed, steps=150)
        h = lambda k: hashlib.md5(np.ascontiguousarray(t.cell_data[k].values)).hexdigest()
        return t.get_tumor_size(), h("cell_snv"), h("cell_cnv"), h("cell_exp"), h("cell_crd")
    assert fingerprint(3) == fingerprint(3)
    assert fingerprint(3) != fingerprint(4)


def test_demes_cap_near_carrying_capacity():
    # DESIGN_crowding.md (Option A): with density-dependent death the tumour SPREADS across demes and
    # each deme caps NEAR carrying_capacity, instead of the old bug where evolved clones outran the
    # absolute death cap and piled 1000s of cells into a few demes.
    K = 8
    cancer = {"division_rate": 0.6, "death_rate": 0.02, "max_birth_rate": 0.9,
              "mutation_rate": 0.3, "dispersal_rate": 0.1}
    t = GenotypeTumor(seed=2, genome_params=GENOME_PARAMS, selection_params=SELECTION_PARAMS,
                      cancer_cell_params=cancer,
                      deme_params={"carrying_capacity": K, "maximum_death_rate": 1.0,
                                   "initial_cancer_cells": 5},
                      spatial_params={"grid_size": 25, "structure_radius": 0},
                      update_mode="tau", tau=1.0)
    t.grow(n_steps=60, seed=2)
    occ = [sum(v for g, v in d.items() if t._is_cancer(g))
           for d in t.demes if any(t._is_cancer(g) for g in d)]
    assert len(occ) > 20                        # the tumour SPREAD across many demes (not one pile)
    assert np.mean(occ) <= 1.5 * K              # mean occupancy caps near K
    assert max(occ) <= 4 * K                    # no deme is a runaway pile (was 100s-1000s x K)


def test_well_mixed_disables_crowding():
    # carrying_capacity=None -> the well-mixed regime: no per-deme ceiling, so a single deme grows
    # unbounded (the role the old carrying_capacity=1 hack played; K=1 now caps at ~1 cell).
    cancer = {**LOW_DEATH, "dispersal_rate": 0.0, "mutation_rate": 0.0}
    t = GenotypeTumor(seed=1, genome_params=GENOME_PARAMS, selection_params=SELECTION_PARAMS,
                      cancer_cell_params=cancer, deme_params={"carrying_capacity": None},
                      spatial_params={"grid_size": 1, "structure_radius": 0},
                      update_mode="tau", tau=1.0)
    assert t._crowding is False
    # ~18 generations is plenty to blow past any finite per-deme K in one deme; 40 here grew to
    # ~4e7 cells and OOM'd when grow() materialised the cell x gene matrix (~13 GB).
    t.grow(n_steps=18, seed=1)
    # one deme, no ceiling -> far more than any finite K would allow
    assert t.get_cancer_size() > 500


def test_statistically_equivalent_to_cell_engine():
    seeds, steps = range(10), 150
    cnt = [_count(s, steps, cancer=LOW_DEATH).get_tumor_size() for s in seeds]
    cell = [_cell(s, steps, cancer=LOW_DEATH).get_tumor_size() for s in seeds]

    # both engines have the same low extinction probability (death/birth = 0.1)
    assert sum(s > 0 for s in cnt) >= 6
    assert sum(s > 0 for s in cell) >= 6

    # survivor sizes in the same ballpark (net birth-death drift per step is identical)
    cnt_mean = np.mean([s for s in cnt if s > 0])
    cell_mean = np.mean([s for s in cell if s > 0])
    assert 0.5 < cnt_mean / cell_mean < 2.0


# --- viability limits (CINner max_ploidy / max_cn / max_nullisomy / max_mut_drivers) ---------
# These were enforced ONLY in the cell-level engine (Deme.sample_event) and were never ported to
# this one, so they were silently inert here: genotypes breaching them stayed alive and dividing.
# See GenotypeTumor._is_viable for the reject-at-birth semantics. Both engines now use that rule
# (the cell engine's counterpart is exercised further down, under "the CELL engine enforces the
# same limits, the same way").

# A genome that mutates hard and amplifies/deletes aggressively, so the limits are actually
# REACHED within a short run (the shipped defaults never get near them -- see
# test_default_limits_are_a_noop).
VIAB_GENOME = {"n_segments": 10, "segment_size": 100}
VIAB_CANCER = {"division_rate": 0.3, "death_rate": 0.02, "max_birth_rate": 0.8,
               "mutation_rate": 0.8, "dispersal_rate": 0.2,
               "cnv_prob": 0.95, "snv_prob": 0.05, "amp_prob": 0.7}
VIAB_DEME = {"carrying_capacity": 12, "initial_cancer_cells": 5}
VIAB_SPATIAL = {"grid_size": 15, "structure_radius": 0}


def _grow_counting_rejections(selection_params, seed=1, steps=3000, cancer=None, mode="exact"):
    """Grow a tumour and report how many daughters the viability gate rejected.

    The rejection count keeps the invariant assertions honest: a limit that is never reached
    would satisfy "no survivor breaches it" vacuously.
    """
    rejected = []
    real = GenotypeTumor._is_viable

    def spy(self, rep):
        ok = real(self, rep)
        if not ok:
            rejected.append(rep.genome_summary)
        return ok

    GenotypeTumor._is_viable = spy
    try:
        t = GenotypeTumor(seed=seed, genome_params=VIAB_GENOME, selection_params=selection_params,
                          cancer_cell_params=cancer or VIAB_CANCER, deme_params=VIAB_DEME,
                          spatial_params=VIAB_SPATIAL, update_mode=mode)
        t.grow(n_steps=steps, seed=seed)
    finally:
        GenotypeTumor._is_viable = real
    return t, rejected


def _living_cancer_summaries(t):
    return [t.genotypes[g].genome_summary
            for g, c in t.genotypes_counts.items() if c > 0 and t._is_cancer(g)]


def test_ploidy_limit_binds():
    limits = {"prop_driver": 0.1, "driver_effects": 1.2, "max_ploidy": 2.5}
    t, rejected = _grow_counting_rejections(limits)
    assert rejected, "max_ploidy was never reached -- the test config no longer exercises it"
    assert all(gs["ploidy"] <= 2.5 for gs in _living_cancer_summaries(t))


def test_cn_limit_binds():
    limits = {"prop_driver": 0.1, "driver_effects": 1.2, "max_cn": 3}
    t, rejected = _grow_counting_rejections(limits)
    assert rejected, "max_cn was never reached -- the test config no longer exercises it"
    assert all(gs["highest_cn"] <= 3 for gs in _living_cancer_summaries(t))


def test_nullisomy_limit_binds():
    # deletion-dominated (amp_prob 0.1) so segments actually reach zero copies
    cancer = {**VIAB_CANCER, "amp_prob": 0.1}
    limits = {"prop_driver": 0.1, "driver_effects": 1.2, "max_nullisomy": 0}
    t, rejected = _grow_counting_rejections(limits, cancer=cancer)
    assert rejected, "max_nullisomy was never reached -- the test config no longer exercises it"
    assert all(gs["nullisomy_count"] == 0 for gs in _living_cancer_summaries(t))


def test_gate_rejects_a_breach_of_every_limit():
    """The engine's gate must reject on ALL FOUR limits, not just the ones a run happens to reach."""
    t = GenotypeTumor(seed=1, genome_params=VIAB_GENOME,
                      selection_params={"max_ploidy": 6, "max_cn": 12, "max_nullisomy": 2,
                                        "max_mut_drivers": 1000},
                      cancer_cell_params=VIAB_CANCER, deme_params=VIAB_DEME,
                      spatial_params=VIAB_SPATIAL)
    rep = t.genotypes[t.founder_id]
    assert t._is_viable(rep)  # the untouched diploid founder is viable

    for key, breach in (("ploidy", 6.5), ("highest_cn", 13),
                        ("nullisomy_count", 3), ("n_mutated_drivers", 1001)):
        probe = rep.divide()
        probe.genome_summary = dict(rep.genome_summary)
        probe.genome_summary[key] = breach
        assert not t._is_viable(probe), f"gate ignores a {key} breach"


def test_max_mut_drivers_limit_binds():
    """``max_mut_drivers`` caps the number of distinct mutated driver genes, in BOTH engines.

    It was inert until ``Cell._recount_seg_drivers`` was added: ``n_mutated_drivers`` was
    initialised to 0 in ``Cell.__init__`` and written by no code path, so ``update_viability``'s
    ``n_mutated_drivers > max_mut_drivers`` was ``0 > 1000`` forever. The gate always read the
    limit correctly (test_gate_rejects_a_breach_of_every_limit); only its input was missing.
    """
    cancer = {**VIAB_CANCER, "cnv_prob": 0.05, "snv_prob": 0.95}   # SNV-heavy: drivers get mutated
    limits = {"prop_driver": 0.3, "driver_effects": 1.2, "max_mut_drivers": 2}
    t, rejected = _grow_counting_rejections(limits, cancer=cancer)
    assert rejected, "max_mut_drivers was never reached -- the test config no longer exercises it"
    assert all(gs["n_mutated_drivers"] <= 2 for gs in _living_cancer_summaries(t))


def test_n_mutated_drivers_counts_distinct_genes_not_mutated_copies():
    """The gate's input is DISTINCT mutated driver genes -- the modelling decision behind the knob.

    This is why the copy-weighted ``n_mut_onc``/``n_mut_tsg`` could not be reused as the input:
    ``update_genome_summary_cnv`` scales them by ``sign`` as copies come and go, so one mutated
    oncogene at copy number 4 contributes 4 to them but must contribute 1 here -- amplifying a
    driver is not the same event as mutating three more.
    """
    from iscc.tumor.components.cell import Cell
    from iscc.tumor.components.selection import Selection

    # prop_driver=1.0 -> every position is a driver (onc or tsg), so the arithmetic below is exact
    sel = Selection(n_segments=1, segment_size=20, prop_driver=1.0, rng=np.random.default_rng(0))
    cell = Cell(n_segments=1, segment_size=20, n_onc=len(sel.get_oncogenes()),
                n_tsg=len(sel.get_tsgs()))
    seg, pos = 0, int(sel.drivers[0][0])
    mut_bits = np.zeros(20, dtype=bool)
    mut_bits[pos] = True

    assert cell.genome_summary["n_mutated_drivers"] == 0     # untouched diploid founder

    # one SNV on one copy of one driver gene -> one distinct mutated driver
    cell.genome[seg]["p"][0][pos] = True
    cell.update_genome_summary_mutation(sel, mut_bits, seg)
    assert cell.genome_summary["n_mutated_drivers"] == 1

    # the SAME gene mutated on the other haplotype is still ONE mutated gene
    cell.genome[seg]["m"][0][pos] = True
    cell.update_genome_summary_mutation(sel, mut_bits, seg)
    assert cell.genome_summary["n_mutated_drivers"] == 1

    # amplifying a mutated copy raises the copy-weighted counts but NOT the distinct gene count
    copies_before = cell.genome_summary["n_mut_onc"] + cell.genome_summary["n_mut_tsg"]
    cell.genome[seg]["p"].append(cell.genome[seg]["p"][0].copy())
    cell.update_genome_summary_cnv(sel, cell.genome[seg]["p"][0], seg, 1)
    assert cell.genome_summary["n_mutated_drivers"] == 1
    assert cell.genome_summary["n_mut_onc"] + cell.genome_summary["n_mut_tsg"] > copies_before

    # a second, DIFFERENT driver gene does move it
    pos2 = int(sel.drivers[0][1])
    mut_bits2 = np.zeros(20, dtype=bool)
    mut_bits2[pos2] = True
    cell.genome[seg]["p"][0][pos2] = True
    cell.update_genome_summary_mutation(sel, mut_bits2, seg)
    assert cell.genome_summary["n_mutated_drivers"] == 2

    # deleting every copy carrying them un-mutates both genes again
    for hap in ("p", "m"):
        while cell.genome[seg][hap]:
            bits = cell.genome[seg][hap][0].copy()
            del cell.genome[seg][hap][0]
            cell.update_genome_summary_cnv(sel, bits, seg, -1)
    assert cell.genome_summary["n_mutated_drivers"] == 0


def test_tau_engine_also_enforces_viability():
    """The tau path has its own birth/mutate loop -- it must gate the daughter too."""
    limits = {"prop_driver": 0.1, "driver_effects": 1.2, "max_ploidy": 2.5, "max_cn": 3}
    t, rejected = _grow_counting_rejections(limits, steps=40, mode="tau")
    assert rejected, "limits never reached under tau -- the test config no longer exercises them"
    for gs in _living_cancer_summaries(t):
        assert gs["ploidy"] <= 2.5 and gs["highest_cn"] <= 3


def test_reproducer_tight_ploidy_actually_binds():
    """Regression: the exact reported reproducer. Before the fix this grew to ploidy 3.30 /
    highest_cn 8 with 83% of cells non-viable-but-dividing."""
    limits = {"prop_driver": 0.1, "driver_effects": 1.2,
              "max_ploidy": 2.5, "max_cn": 3, "max_nullisomy": 0}
    t, _ = _grow_counting_rejections(limits)
    living = _living_cancer_summaries(t)
    assert living, "tumour died out -- the reproducer no longer exercises the limits"
    assert max(gs["ploidy"] for gs in living) <= 2.5
    assert max(gs["highest_cn"] for gs in living) <= 3
    assert max(gs["nullisomy_count"] for gs in living) == 0
    # and every living genotype is viable by the selection model's own verdict
    assert all(t.selection.update_viability(gs) == 1 for gs in living)


def test_default_limits_are_a_noop_for_the_exact_engine():
    """At the shipped defaults the limits (ploidy 6 / cn 12 / nullisomy 2 / drivers 1000) are far
    out of reach for the exact engine, so the gate never fires and no existing baseline can move.

    This does NOT hold for tau-leaping, which reaches ~3x the size in the same number of steps and
    does occasionally delete >2 of the 5 default segments outright -- see
    test_default_nullisomy_can_bind_under_tau.
    """
    rejected = []
    real = GenotypeTumor._is_viable

    def spy(self, rep):
        ok = real(self, rep)
        if not ok:
            rejected.append(rep.genome_summary)
        return ok

    GenotypeTumor._is_viable = spy
    try:
        for seed in range(3):
            t = GenotypeTumor(seed=seed, genome_params={"n_segments": 5, "segment_size": 200},
                              selection_params={}, cancer_cell_params={},
                              deme_params={"carrying_capacity": 8, "initial_cancer_cells": 5},
                              spatial_params={"grid_size": 15, "structure_radius": 0})
            t.grow(n_steps=2000, seed=seed)
            assert t.get_cancer_size() > 0
    finally:
        GenotypeTumor._is_viable = real
    assert rejected == [], f"viability gate fired at shipped defaults: {rejected}"
