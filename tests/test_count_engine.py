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


def test_engines_agree_on_crowding_death():
    # The count engine's _death_rate and the cell engine's Deme.get_cancer_death_rate implement the
    # SAME density-dependent crowding formula (DESIGN_crowding.md), so they must return identical
    # crowding death for a matched (occupancy, K, division, death, margin) state.
    from iscc.tumor.components.deme import Deme
    from iscc.tumor.components.cell import CancerCell
    K, margin, maxd, div, death = 8, 0.1, 1.0, 0.6, 0.03

    t = GenotypeTumor(seed=1, genome_params=GENOME_PARAMS, selection_params=SELECTION_PARAMS,
                      cancer_cell_params={"division_rate": div, "death_rate": death,
                                          "max_birth_rate": 0.9},
                      deme_params={"carrying_capacity": K, "maximum_death_rate": maxd,
                                   "crowding_margin": margin, "initial_cancer_cells": 1},
                      spatial_params={"grid_size": 1, "structure_radius": 0})
    fid = t.founder_id
    # pin the founder's rates so both engines see identical inputs
    t.genotypes[fid].evolutionary_parameters["division_rate"] = div
    t.genotypes[fid].evolutionary_parameters["death_rate"] = death

    def cell_deme(n):
        first = CancerCell(n_segments=N_SEGMENTS, segment_size=SEGMENT_SIZE)
        first.evolutionary_parameters["division_rate"] = div
        first.evolutionary_parameters["death_rate"] = death
        d = Deme(cell=first, carrying_capacity=K, maximum_death_rate=maxd, crowding_margin=margin)
        for _ in range(n - 1):
            c = CancerCell(n_segments=N_SEGMENTS, segment_size=SEGMENT_SIZE)
            c.evolutionary_parameters["division_rate"] = div
            c.evolutionary_parameters["death_rate"] = death
            d.add_cell(c)
        return d

    for n in (1, 4, 8, 16, 24):
        d_count = t._death_rate(fid, 0, total=n)
        d_cell = cell_deme(n).get_cancer_death_rate(death, division_rate=div,
                                                    immune_cell_fraction=0.0, immune_resistance=0.0)
        assert d_count == pytest.approx(d_cell)
    # and the shared fixed point: death == division at occupancy K/(1+margin)
    assert t._death_rate(fid, 0, total=K / (1 + margin)) == pytest.approx(div)


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
# See GenotypeTumor._is_viable for the reject-at-birth semantics and why they differ from the cell
# engine's lazy check.

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
    """The engine's gate must reject on ALL FOUR limits, not just the ones a run happens to reach.

    Driven directly through the gate because ``max_mut_drivers`` is currently unreachable by
    simulation -- see test_max_mut_drivers_is_inert_because_its_input_is_never_computed.
    """
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


def test_max_mut_drivers_is_inert_because_its_input_is_never_computed():
    """SEPARATE PRE-EXISTING BUG (both engines): ``genome_summary['n_mutated_drivers']`` is
    initialised to 0 in Cell.__init__ and never written by any code path -- the summary maintains
    n_mut_onc / n_mut_tsg / n_mut_disp / n_mut_ir / n_mut_tr instead. So
    ``update_viability``'s ``n_mutated_drivers > max_mut_drivers`` is ``0 > 1000`` forever, and
    max_mut_drivers cannot fire in the count engine OR the cell engine.

    Enforcing viability here does not fix that: the gate reads the limit correctly
    (test_gate_rejects_a_breach_of_every_limit), the input is simply never populated. It is not a
    mechanical fix either -- n_mut_onc/n_mut_tsg count mutated COPIES (CNVs scale them), not
    distinct mutated driver genes, so populating n_mutated_drivers is a modelling decision.

    This test pins the current reality; delete it when n_mutated_drivers is actually maintained.
    """
    cancer = {**VIAB_CANCER, "cnv_prob": 0.05, "snv_prob": 0.95}
    limits = {"prop_driver": 0.3, "driver_effects": 1.2, "max_mut_drivers": 0}
    t, rejected = _grow_counting_rejections(limits, cancer=cancer)
    # even with max_mut_drivers=0 and an SNV-heavy genome, nothing is ever rejected for it
    assert rejected == []
    living = _living_cancer_summaries(t)
    assert living and all(gs["n_mutated_drivers"] == 0 for gs in living)
    # ...while the drivers that ARE tracked did accumulate, proving the run mutated drivers
    assert any(gs["n_mut_onc"] + gs["n_mut_tsg"] > 0 for gs in living)


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


def test_default_nullisomy_can_bind_under_tau():
    """The limits are NOT unreachable at the shipped defaults under tau-leaping.

    Pins a finding from the viability fix: tau reaches ~3x the exact engine's size in the same
    step budget, and with the default 5-segment genome it occasionally produces a daughter that
    has deleted >2 segments outright (max_nullisomy=2) -- e.g. seg_cns [0, 2, 0, 1, 0]. Such a
    genotype used to survive and keep dividing with 3/5 of its genome gone. So default-config TAU
    trajectories may shift slightly; default-config EXACT trajectories cannot
    (test_default_limits_are_a_noop_for_the_exact_engine).
    """
    rejected = []
    real = GenotypeTumor._is_viable

    def spy(self, rep):
        ok = real(self, rep)
        if not ok:
            rejected.append(dict(rep.genome_summary))
        return ok

    GenotypeTumor._is_viable = spy
    try:
        for seed in range(5):
            t = GenotypeTumor(seed=seed, genome_params={"n_segments": 5, "segment_size": 200},
                              selection_params={}, cancer_cell_params={},
                              deme_params={"carrying_capacity": 8, "initial_cancer_cells": 5},
                              spatial_params={"grid_size": 15, "structure_radius": 0},
                              update_mode="tau")
            t.grow(n_steps=60, seed=seed)
    finally:
        GenotypeTumor._is_viable = real

    assert rejected, "expected the default max_nullisomy to bind under tau"
    # every rejection is a nullisomy breach (ploidy/cn stay far from 6/12 at defaults)
    assert all(gs["nullisomy_count"] > 2 for gs in rejected)
    assert all(gs["ploidy"] <= 6 and gs["highest_cn"] <= 12 for gs in rejected)


# --- the CELL engine enforces the same limits, the same way -----------------------------------
# GlandularTumor used to check viability LAZILY, in Deme.sample_event: a non-viable daughter was
# added to the deme and only died when it was next *sampled* for an event. While it waited it held
# a carrying-capacity slot, counted towards get_tumor_size, and could be sampled into cell_data --
# so genomes breaching the configured limits leaked into the emitted assay data. It now rejects at
# birth like the count engine (see Deme.apply_event / GenotypeTumor._is_viable), which makes the
# limits a real invariant in BOTH engines. The lazy check remains as a backstop only.


def _grow_cell_counting_rejections(selection_params, seed=1, steps=400, cancer=None):
    """Grow a CELL-level tumour and report how many daughters the viability gate rejected.

    The cell engine has no ``_is_viable`` seam to spy on -- its gate reads the ``viability`` that
    ``Cell.mutate`` already computed. ``Selection.update_viability`` is called only from
    ``update_evolutionary_parameters``, which in turn runs only inside ``Cell.mutate``, so a 0
    verdict during growth is exactly one rejected daughter.
    """
    from iscc.tumor.components.selection import Selection

    rejected = []
    real = Selection.update_viability

    def spy(self, genome_summary, **kwargs):
        v = real(self, genome_summary, **kwargs)
        if v == 0:
            rejected.append(dict(genome_summary))
        return v

    Selection.update_viability = spy
    try:
        t = GlandularTumor(seed=seed, genome_params=VIAB_GENOME, selection_params=selection_params,
                           cancer_cell_params=cancer or VIAB_CANCER,
                           epithelial_cell_params=EPITHELIAL_CELL_PARAMS,
                           stromal_cell_params=STROMAL_CELL_PARAMS,
                           immune_cell_params=IMMUNE_CELL_PARAMS,
                           deme_params=VIAB_DEME, grid_size=VIAB_SPATIAL["grid_size"],
                           structure_radius=0)
        t.grow(n_steps=steps, seed=seed)
    finally:
        Selection.update_viability = real
    return t, rejected


def _living_cancer_cells(t):
    return [c for deme in t.deme_list for c in deme.cells if c.type == "cancer"]


def test_cell_engine_rejects_non_viable_daughters_at_birth():
    """The reported reproducer. Before the fix this ended with worst ploidy 2.75 (limit 2.5),
    worst highest_cn 4 (limit 3) and 72 non-viable cells still alive in the demes."""
    limits = {"prop_driver": 0.1, "driver_effects": 1.2,
              "max_ploidy": 2.5, "max_cn": 3, "max_nullisomy": 0}
    t, rejected = _grow_cell_counting_rejections(limits)
    assert rejected, "limits never reached -- the test config no longer exercises them"
    living = _living_cancer_cells(t)
    assert living, "tumour died out -- the reproducer no longer exercises the limits"

    # the invariant: no living cell breaches any limit, and none is merely awaiting its lazy death
    assert max(c.genome_summary["ploidy"] for c in living) <= 2.5
    assert max(c.genome_summary["highest_cn"] for c in living) <= 3
    assert max(c.genome_summary["nullisomy_count"] for c in living) == 0
    assert all(c.evolutionary_parameters["viability"] == 1 for c in living)
    assert all(t.selection.update_viability(c.genome_summary) == 1 for c in living)


def test_cell_engine_never_emits_a_breaching_cell():
    """The point of rejecting at birth: breaching genomes cannot reach the assay data.

    Under the lazy rule a doomed cell was sampled into cell_data for as long as it happened to
    dwell in the deme, so make_cell_data could emit genomes that breach the configured limits.
    """
    limits = {"prop_driver": 0.1, "driver_effects": 1.2,
              "max_ploidy": 2.5, "max_cn": 3, "max_nullisomy": 0}
    t, rejected = _grow_cell_counting_rejections(limits)
    assert rejected, "limits never reached -- the test config no longer exercises them"
    t.make_cell_data()
    evo = t.cell_data["cell_evo"]
    assert len(evo), "no cells emitted -- the test no longer exercises the emitted data"
    assert (evo["viability"] == 1).all()


def test_cell_engine_gate_fires_on_every_limit():
    """The gate must reject on ALL FOUR limits, not just the ones a run happens to reach.

    The cell engine's gate is the `viability` that Selection.update_viability wrote during
    mutate(), so this pins the same four-limit coverage the count engine's gate has.
    """
    t = GlandularTumor(seed=1, genome_params=VIAB_GENOME,
                       selection_params={"max_ploidy": 6, "max_cn": 12, "max_nullisomy": 2,
                                         "max_mut_drivers": 1000},
                       cancer_cell_params=VIAB_CANCER,
                       epithelial_cell_params=EPITHELIAL_CELL_PARAMS,
                       stromal_cell_params=STROMAL_CELL_PARAMS,
                       immune_cell_params=IMMUNE_CELL_PARAMS,
                       deme_params=VIAB_DEME, grid_size=VIAB_SPATIAL["grid_size"],
                       structure_radius=0)
    founder = _living_cancer_cells(t)[0]
    assert t.selection.update_viability(founder.genome_summary) == 1  # diploid founder is viable

    for key, breach in (("ploidy", 6.5), ("highest_cn", 13),
                        ("nullisomy_count", 3), ("n_mutated_drivers", 1001)):
        gs = dict(founder.genome_summary)
        gs[key] = breach
        assert t.selection.update_viability(gs) == 0, f"gate ignores a {key} breach"


def test_cell_engine_default_limits_are_a_noop():
    """At the shipped defaults (ploidy 6 / cn 12 / nullisomy 2 / drivers 1000) the cell engine never
    reaches a limit, so the birth gate never fires and no cell-level baseline can move.

    This is the guarantee that makes the change safe to land: the gate only READS a value that
    mutate() already computed and draws no randomness, so a gate that never fires leaves the
    trajectory bit-identical. Configs where the limits DO bind shift on purpose -- that is the fix.
    """
    from iscc.tumor.components.selection import Selection

    rejected = []
    real = Selection.update_viability

    def spy(self, genome_summary, **kwargs):
        v = real(self, genome_summary, **kwargs)
        if v == 0:
            rejected.append(dict(genome_summary))
        return v

    Selection.update_viability = spy
    try:
        for seed in range(3):
            t = GlandularTumor(seed=seed, genome_params={"n_segments": 5, "segment_size": 200},
                               selection_params={}, cancer_cell_params={},
                               epithelial_cell_params=EPITHELIAL_CELL_PARAMS,
                               stromal_cell_params=STROMAL_CELL_PARAMS,
                               immune_cell_params=IMMUNE_CELL_PARAMS,
                               deme_params={"carrying_capacity": 8, "initial_cancer_cells": 5},
                               grid_size=15, structure_radius=0)
            t.grow(n_steps=400, seed=seed)
            assert t.get_tumor_size() > 0
    finally:
        Selection.update_viability = real
    assert rejected == [], f"viability gate fired at shipped defaults: {rejected}"
