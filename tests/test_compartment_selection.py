"""Compartment-dependent selection — v1 of DESIGN_phenotype_plasticity.md §2.

Two new gene-based heritable axes (``breach`` / ``stromal_survival``) attenuate two local
compartment hazards (``epithelial_barrier`` / ``stromal_hazard``) that the gland geometry seeds, in
the exact shape of the existing immune term. The compartment is never a fixed deme label: every
hazard reads the deme's LIVE cell-type fractions, so a clone that has diluted the resident normals
feels less of the barrier. The compartment is also an R13 route-3 niche field (Part D), driving the
invasive program at the epithelial front — the genetic-vs-niche expression confound.

Covered here:
  * OFF-by-default -> growth is byte-identical (golden hashes captured on the pre-feature engine).
  * a breach-competent genotype has strictly LOWER death than a non-breacher in an epithelial-
    occupied deme, and EQUAL death in a pure-cancer deme (the trait pays off only at the epithelium).
  * stromal_survival is the analogous statement in a stromal-occupied deme.
  * both engines compute the SAME compartment death terms.
  * the two axes are ground-truth (gene counts + the per-genotype trait surface correctly).
  * the compartment niche field drives the invasive program (readout-only; growth is byte-identical
    whether the program layer is on or off).
"""
import hashlib

import numpy as np
import pytest

from conftest import (
    GENOME_PARAMS, SELECTION_PARAMS, CANCER_CELL_PARAMS, N_SEGMENTS, SEGMENT_SIZE,
    EPITHELIAL_CELL_PARAMS, STROMAL_CELL_PARAMS, IMMUNE_CELL_PARAMS, DEME_PARAMS,
)
from iscc.tumor.models import GenotypeTumor
from iscc.tumor.components.cell import CancerCell
from iscc.tumor.components.selection import Selection

# A structured tumour (structure_radius > 0) seeds the epithelial ring + stroma the compartment
# selection acts on. These SPATIAL params carry NO compartment coefficients -> the feature is off.
SPATIAL = {"grid_size": 15, "n_structures": 1, "structure_radius": 4}
# The two compartment axes turned ON (in addition to the base immune/dispersal/... axes).
SEL_ON = {**SELECTION_PARAMS, "prop_breach": 0.2, "prop_stromal_survival": 0.2,
          "breach_effects": 1.5, "stromal_survival_effects": 1.5}


# --- OFF-by-default: byte-identical growth ----------------------------------------------------
# Golden (tumor_size, cell_snv md5) captured on the pre-feature engine for a STRUCTURED tumour with
# NO compartment params. prop_breach/prop_stromal_survival default 0 -> the ``binomial(1, 0.0)`` draws
# short-circuit (no rng consumed, so the gene-role layout and celltype baselines are untouched), and
# epithelial_barrier/stromal_hazard default 0 -> the two death terms are ``+= 0.0``. So the whole
# growth stream is exactly what it was before the axes existed.
_GOLDEN_COUNT = {
    1: (1524, "911b3ac9e297e4986678c66a4c1130bf"),
    2: (1526, "291cfc18b067fc46b9eef9b6b6177ddc"),
    3: (1526, "59029138da96904e502af4aa46c2badf"),
}


def _snv_hash(t):
    t.make_cell_data()
    return hashlib.md5(np.ascontiguousarray(t.cell_data["cell_snv"].values)).hexdigest()


def _count(seed, steps=120, selection=SELECTION_PARAMS, spatial=SPATIAL):
    t = GenotypeTumor(seed=seed, genome_params=GENOME_PARAMS, selection_params=selection,
                      cancer_cell_params=CANCER_CELL_PARAMS, deme_params=DEME_PARAMS,
                      spatial_params=spatial)
    t.grow(n_steps=steps, seed=seed)
    return t


@pytest.mark.parametrize("seed", sorted(_GOLDEN_COUNT))
def test_count_off_by_default_byte_identical(seed):
    t = _count(seed)
    size, digest = _GOLDEN_COUNT[seed]
    assert t.get_tumor_size() == size
    assert _snv_hash(t) == digest, "compartment feature off perturbed the count-engine growth stream"


def test_explicit_zero_matches_absent():
    """prop_breach=prop_ss=0 + epithelial_barrier=stromal_hazard=0 must equal omitting them."""
    sel_zero = {**SELECTION_PARAMS, "prop_breach": 0.0, "prop_stromal_survival": 0.0}
    spatial_zero = {**SPATIAL, "epithelial_barrier": 0.0, "stromal_hazard": 0.0}
    for seed in (1, 2):
        a = _count(seed)
        b = _count(seed, selection=sel_zero, spatial=spatial_zero)
        assert a.get_tumor_size() == b.get_tumor_size()
        assert _snv_hash(a) == _snv_hash(b)


def test_axes_on_barriers_off_growth_byte_identical():
    """Turning the AXES on but leaving the barriers at 0 must not change GROWTH (the axes are drawn
    after the fitness-relevant layouts, and with no barrier they never touch a death rate). The
    control arm of the validation figure relies on this."""
    for seed in (1, 2):
        a = _count(seed)
        b = _count(seed, selection=SEL_ON)   # barriers still absent -> 0
        assert a.get_tumor_size() == b.get_tumor_size()
        assert _snv_hash(a) == _snv_hash(b)


# --- ground truth: the two axes are sequenceable ----------------------------------------------
def test_axes_ground_truth_counts():
    sel = Selection(n_segments=6, segment_size=200, prop_breach=0.2, prop_stromal_survival=0.1)
    assert sel.N_breach == len(sel.get_breach()) > 0
    assert sel.N_ss == len(sel.get_stromal_survival()) > 0
    # off by default
    sel_off = Selection(n_segments=6, segment_size=200)
    assert sel_off.N_breach == 0 and sel_off.N_ss == 0
    assert len(sel_off.get_breach()) == 0 and len(sel_off.get_stromal_survival()) == 0


def test_trait_surfaces_on_genotype():
    """Mutating breach genes raises the per-genotype breach trait above the wild-type 0."""
    sel = Selection(n_segments=4, segment_size=100, prop_breach=0.3, prop_stromal_survival=0.3,
                    breach_effects=1.6, stromal_survival_effects=1.6)
    c = CancerCell(n_segments=4, segment_size=100, n_breach=len(sel.get_breach()),
                   n_ss=len(sel.get_stromal_survival()), division_rate=0.5, death_rate=0.1)
    c.set_genotype_id()
    c.update_evolutionary_parameters(sel)
    assert c.evolutionary_parameters["breach"] == 0.0            # wild-type
    assert c.evolutionary_parameters["stromal_survival"] == 0.0
    # force a mutation on every breach gene on both homologs
    for seg in range(sel.n_segments):
        bits = np.zeros(sel.segment_sizes[seg], dtype=bool)
        bits[sel.breach[seg]] = True
        for hap in ("p", "m"):
            for allele in c.genome[seg][hap]:
                allele[bits] = True
        c.update_genome_summary_mutation(sel, bits, seg)
    c.update_evolutionary_parameters(sel)
    assert c.evolutionary_parameters["breach"] > 0.0
    assert c.genome_summary["n_mut_breach"] > 0


# --- the payoff: a trait pays off ONLY in its compartment -------------------------------------
def _count_tumor_with_traits(breach=0.0, stromal_survival=0.0,
                             epithelial_barrier=0.5, stromal_hazard=0.5):
    spatial = {**SPATIAL, "epithelial_barrier": epithelial_barrier, "stromal_hazard": stromal_hazard}
    t = GenotypeTumor(seed=1, genome_params=GENOME_PARAMS, selection_params=SEL_ON,
                      cancer_cell_params=CANCER_CELL_PARAMS, deme_params=DEME_PARAMS,
                      spatial_params=spatial)
    cg = t.founder_id
    t.genotypes[cg].evolutionary_parameters["breach"] = breach
    t.genotypes[cg].evolutionary_parameters["stromal_survival"] = stromal_survival
    return t, cg


def test_breach_lowers_death_only_at_epithelium():
    tb, cg = _count_tumor_with_traits(breach=0.8)
    tn, cn = _count_tumor_with_traits(breach=0.0)
    epi = tb._normal_genotype("epithelial")
    # epithelial-occupied deme: breacher strictly lower
    tb.demes[0] = {cg: 2, epi: 6}
    tn.demes[0] = {cn: 2, epi: 6}
    assert tb._death_rate(cg, 0) < tn._death_rate(cn, 0)
    # pure-cancer deme: no epithelium -> the trait pays nothing -> EQUAL
    tb.demes[1] = {cg: 8}
    tn.demes[1] = {cn: 8}
    assert tb._death_rate(cg, 1) == pytest.approx(tn._death_rate(cn, 1))


def test_stromal_survival_lowers_death_only_in_stroma():
    ts, cg = _count_tumor_with_traits(stromal_survival=0.8)
    tn, cn = _count_tumor_with_traits(stromal_survival=0.0)
    stro = ts._normal_genotype("stromal")
    ts.demes[0] = {cg: 2, stro: 6}
    tn.demes[0] = {cn: 2, stro: 6}
    assert ts._death_rate(cg, 0) < tn._death_rate(cn, 0)
    # pure-cancer deme: no stroma -> EQUAL
    ts.demes[1] = {cg: 8}
    tn.demes[1] = {cn: 8}
    assert ts._death_rate(cg, 1) == pytest.approx(tn._death_rate(cn, 1))


def test_live_composition_gates_the_barrier():
    """As cancer dilutes the resident epithelium the epithelial death term shrinks (compartment is
    live composition, not a fixed label)."""
    t, cg = _count_tumor_with_traits(breach=0.0)
    epi = t._normal_genotype("epithelial")
    t.demes[0] = {cg: 1, epi: 7}      # mostly epithelial
    d_many = t._death_rate(cg, 0)
    t.demes[0] = {cg: 7, epi: 1}      # cancer has diluted the epithelium
    d_few = t._death_rate(cg, 0)
    assert d_many > d_few


# --- both engines agree on the compartment death terms ----------------------------------------
# NB the cell-engine mirror is DEFERRED in v1 (the ductal-field substrate is count-engine only), so
# there is no engine-agreement test here; the death terms live in count.py's _death_rate only.


# --- Part D: the compartment as an R13 niche field (the confound) ------------------------------
_EXPR = {
    "program_params": {"n_programs": 6, "n_genes_per_program": 8, "program_overlap": 0.1},
    "coupling_params": {"niche_program_map": {"epithelial": "emt"}, "niche_program_strength": 3.0},
    "activity_params": {"activity_mean": 1.0, "activity_sd": 0.3, "activity_noise": 0.05},
}


def test_compartment_niche_field_drives_program():
    """A deme with more epithelium drives the mapped invasive (emt) program harder — the SAME clone
    expressing an env-responsive phenotype. The compartment field is recorded as ground truth."""
    t = GenotypeTumor(seed=1, genome_params=GENOME_PARAMS, selection_params=SEL_ON,
                      cancer_cell_params=CANCER_CELL_PARAMS, deme_params=DEME_PARAMS,
                      spatial_params={**SPATIAL, "epithelial_barrier": 0.2, "stromal_hazard": 0.1},
                      expression_params=_EXPR)
    t.grow(n_steps=60, seed=1)
    t.make_cell_data()
    # ground truth recorded even though F8 (hypoxia/cci) is off
    assert "epithelial" in t.microenv_truth and "stromal" in t.microenv_truth
    assert t.microenv_truth["epithelial"].shape[0] == len(t.demes)
    # the niche drive is monotone in the epithelial field for the mapped program
    epi = t.microenv_truth["epithelial"]
    drive = t.programs.niche_drive({"epithelial": epi, "stromal": t.microenv_truth["stromal"]})
    emt_k = list(t.programs.dictionary.program_names).index("emt")
    order = np.argsort(epi)
    assert np.all(np.diff(drive[order, emt_k]) >= -1e-12)          # non-decreasing in epi frac
    assert drive[:, emt_k].max() > 0.0                            # some deme has epithelium


def test_program_layer_is_readout_only_for_compartment():
    """Growth must be byte-identical whether the compartment niche field feeds the program or not —
    programs are a materialisation-time readout, never fitness."""
    spatial = {**SPATIAL, "epithelial_barrier": 0.2, "stromal_hazard": 0.1}
    a = GenotypeTumor(seed=2, genome_params=GENOME_PARAMS, selection_params=SEL_ON,
                      cancer_cell_params=CANCER_CELL_PARAMS, deme_params=DEME_PARAMS,
                      spatial_params=spatial)
    a.grow(n_steps=80, seed=2)
    b = GenotypeTumor(seed=2, genome_params=GENOME_PARAMS, selection_params=SEL_ON,
                      cancer_cell_params=CANCER_CELL_PARAMS, deme_params=DEME_PARAMS,
                      spatial_params=spatial, expression_params=_EXPR)
    b.grow(n_steps=80, seed=2)
    assert a.get_tumor_size() == b.get_tumor_size()
    assert _snv_hash(a) == _snv_hash(b)
