"""``Selection.trait_source`` — what a dissemination/niche TRAIT reads off the genome.

The dissemination traits (breach, stromal_survival, met_survival, immune/treatment resistance) each
cost proliferation everywhere via the go-or-grow ``*_cost``, and historically each read the
ploidy-normalised dosage of ALL its genes' copies, mutated or not. Because a trait axis is scattered
uniformly at ``prop_*``, a segment's trait-gene count fluctuates around the genome mean, so almost
every copy-number change moves a trait — and with the costs on, ordinary copy-number drift became a
proliferation tax that made nearly every CNA net-antiproliferative.

``trait_source="mutation"`` makes a trait read only its SNV-MUTATED copies, so an arm-level gain of
unmutated invasion genes no longer makes a cell invasive. Covered here:

  * the DEFAULT ("dosage") is the historical expression, bit-for-bit, at both the formula and the
    grown-tumour level;
  * under "mutation" a pure copy-number event (amplification, deletion, whole-genome duplication)
    leaves every trait exactly where it was, and costs exactly nothing;
  * a trait SNV still switches the trait on — and is worth EXACTLY the same under both modes, so the
    two disagree only about what a copy-number change is worth;
  * the oncogene/TSG DRIVER dosage fitness is byte-identical between the modes (it is deliberately
    untouched: driver fitness is the copy-number model and must stay one).
"""
import hashlib

import numpy as np
import pytest

from conftest import (
    GENOME_PARAMS, SELECTION_PARAMS, CANCER_CELL_PARAMS, DEME_PARAMS, N_SEGMENTS, SEGMENT_SIZE,
)
from iscc.tumor.models import GenotypeTumor
from iscc.tumor.components.cell import CancerCell
from iscc.tumor.components.selection import Selection

# All five trait axes populated, with the go-or-grow costs on, on the tiny shared test genome.
SEL_TRAITS = {**SELECTION_PARAMS,
              "prop_breach": 0.2, "prop_stromal_survival": 0.2, "prop_met_survival": 0.2,
              "breach_effects": 2.8, "stromal_survival_effects": 2.2, "met_survival_effects": 1.5,
              "breach_cost": 0.6, "stromal_survival_cost": 0.6, "met_survival_cost": 0.6,
              "treatment_resistance_cost": 0.6}

# (update method, wild-type count key, mutated count key, axis-size attribute, effect attribute)
TRAIT_AXES = (
    ("update_breach", "n_wt_breach", "n_mut_breach", "N_breach", "breach_effects"),
    ("update_stromal_survival", "n_wt_ss", "n_mut_ss", "N_ss", "stromal_survival_effects"),
    ("update_met_survival", "n_wt_ms", "n_mut_ms", "N_ms", "met_survival_effects"),
    ("update_immune_resistance", "n_wt_ir", "n_mut_ir", "N_ir", "immune_resistant_effects"),
    ("update_treatment_resistance", "n_wt_tr", "n_mut_tr", "N_tr", "treatment_resistant_effects"),
)
TRAIT_PARAMS = ("breach", "stromal_survival", "met_survival",
                "immune_resistance", "treatment_resistance")

# A genome roughly the shape of the shipped realistic one (12 x 500 genes, sparse trait axes), used
# for the population-level statements about single CNAs.
REAL_GENOME = {"n_segments": 12, "segment_size": 500}
REAL_SELECTION = {"prop_driver": 0.04, "prop_dispersal": 0.0, "prop_immune_resistance": 0.02,
                  "prop_met_survival": 0.0, "prop_treatment_resistance": 0.0,
                  "prop_breach": 0.02, "breach_effects": 2.8,
                  "prop_stromal_survival": 0.02, "stromal_survival_effects": 2.2,
                  "breach_cost": 0.6, "stromal_survival_cost": 0.6}
REAL_CANCER = {"division_rate": 0.7, "death_rate": 0.05, "max_birth_rate": 0.95,
               "mutation_rate": 0.3, "n_snvs_per_allele": 0.1, "dispersal_rate": 0.9,
               "cnv_prob": 0.35}


def _selection(trait_source=None, genome=None, selection=None, seed=42):
    genome = genome or GENOME_PARAMS
    params = dict(selection if selection is not None else SEL_TRAITS)
    if trait_source is not None:
        params["trait_source"] = trait_source
    return Selection(n_segments=genome["n_segments"], segment_size=genome["segment_size"],
                     rng=np.random.default_rng(seed), layout_seed=seed, **params)


def _cell(sel, genome=None, cancer=None):
    genome = genome or GENOME_PARAMS
    c = CancerCell(n_segments=genome["n_segments"], segment_size=genome["segment_size"],
                   n_onc=len(sel.get_oncogenes()), n_tsg=len(sel.get_tsgs()),
                   n_disp=len(sel.get_dispersal_genes()), n_ir=len(sel.get_immune_resistant()),
                   n_tr=len(sel.get_treatment_resistant()), n_breach=len(sel.get_breach()),
                   n_ss=len(sel.get_stromal_survival()), n_ms=len(sel.get_met_survival()),
                   **(cancer or CANCER_CELL_PARAMS))
    c.update_evolutionary_parameters(sel)
    return c


# --- the DEFAULT is the historical behaviour ---------------------------------------------------
def test_default_trait_source_is_dosage():
    assert Selection().trait_source == "dosage"
    assert _selection().trait_source == "dosage"


def test_default_reproduces_the_historical_expression_bit_for_bit():
    """Every trait axis under the default must be EXACTLY ``_rel_fitness(n_wt, n_mut, ploidy, N, e,
    e**2)`` — the expression each update_* carried before ``trait_source`` existed."""
    sel = _selection()
    rng = np.random.default_rng(0)
    for _ in range(200):
        gs = {"ploidy": float(rng.uniform(0.5, 6.0))}
        for _m, wt_key, mut_key, _N, _e in TRAIT_AXES:
            gs[wt_key] = int(rng.integers(0, 40))
            gs[mut_key] = int(rng.integers(0, 8))
        for method, wt_key, mut_key, n_attr, eff_attr in TRAIT_AXES:
            e = getattr(sel, eff_attr)
            expected = sel._rel_fitness(gs[wt_key], gs[mut_key], gs["ploidy"],
                                        getattr(sel, n_attr), e, e ** 2)
            assert getattr(sel, method)(gs) == expected      # exact, not approximate


def test_explicit_dosage_matches_absent():
    sel_a, sel_b = _selection(), _selection(trait_source="dosage")
    rng = np.random.default_rng(1)
    for _ in range(50):
        gs = {"ploidy": float(rng.uniform(0.5, 6.0))}
        for _m, wt_key, mut_key, _N, _e in TRAIT_AXES:
            gs[wt_key] = int(rng.integers(0, 40))
            gs[mut_key] = int(rng.integers(0, 8))
        for method, *_ in TRAIT_AXES:
            assert getattr(sel_a, method)(gs) == getattr(sel_b, method)(gs)


def _grow(trait_source=None, seed=1, steps=120):
    selection = dict(SEL_TRAITS)
    if trait_source is not None:
        selection["trait_source"] = trait_source
    t = GenotypeTumor(seed=seed, genome_params=GENOME_PARAMS, selection_params=selection,
                      cancer_cell_params=CANCER_CELL_PARAMS, deme_params=DEME_PARAMS,
                      spatial_params={"grid_size": 15, "n_structures": 1, "structure_radius": 4,
                                      "epithelial_barrier": 0.4, "stromal_hazard": 0.4})
    t.grow(n_steps=steps, seed=seed)
    t.make_cell_data()
    return (t.get_tumor_size(),
            hashlib.md5(np.ascontiguousarray(t.cell_data["cell_snv"].values)).hexdigest())


@pytest.mark.parametrize("seed", (1, 2))
def test_growth_unchanged_when_the_option_is_absent(seed):
    """Engine level: not naming ``trait_source`` grows exactly the tumour ``"dosage"`` does, so an
    existing config is untouched."""
    assert _grow(seed=seed) == _grow(trait_source="dosage", seed=seed)


def test_invalid_trait_source_rejected():
    with pytest.raises(ValueError, match="trait_source"):
        Selection(trait_source="copy-number")


# --- "mutation": a pure copy-number event does not touch the traits ----------------------------
@pytest.mark.parametrize("event", ("cnv", "wgd"))
def test_mutation_mode_pure_copy_number_leaves_every_trait_untouched(event):
    """Amplifications, deletions and whole-genome duplications alike: with no trait gene mutated,
    every trait stays exactly 0 and the go-or-grow cost is exactly 1.0 (no proliferation tax)."""
    sel = _selection("mutation")
    kw = {"snv_prob": 0.0, "cnv_prob": 1.0, "wgd_rate": 1.0 if event == "wgd" else 0.0}
    rng = np.random.default_rng(5)
    n_moved = 0
    for _ in range(200):
        c = _cell(sel)
        assert all(c.evolutionary_parameters[p] == 0.0 for p in TRAIT_PARAMS)
        before = list(c.genome_summary["seg_cns"])
        c.mutate(rng, sel, **kw)
        n_moved += c.genome_summary["seg_cns"] != before
        for p in TRAIT_PARAMS:
            assert c.evolutionary_parameters[p] == 0.0, f"{p} moved on a pure {event}"
        assert sel.proliferation_cost(c.evolutionary_parameters) == 1.0
    assert n_moved == 200                       # every draw really did change a copy number
    if event == "wgd":
        assert c.genome_summary["ploidy"] == 4  # ...and a WGD doubled the whole genome


def test_dosage_mode_pure_copy_number_does_move_the_traits():
    """The contrast that motivates the option: under the default, the SAME copy-number events switch
    traits on and levy the go-or-grow cost, though not one trait gene was mutated."""
    sel = _selection("dosage")
    rng = np.random.default_rng(5)
    switched = costed = 0
    for _ in range(200):
        c = _cell(sel)
        c.mutate(rng, sel, snv_prob=0.0, cnv_prob=1.0, wgd_rate=0.0)
        assert c.genome_summary["n_mut_breach"] == 0                 # nothing was ever mutated
        switched += any(c.evolutionary_parameters[p] > 0.0 for p in TRAIT_PARAMS)
        costed += sel.proliferation_cost(c.evolutionary_parameters) < 1.0
    assert switched > 100 and costed > 100      # a majority of pure CNAs, on a genome with no SNVs


# --- "mutation": a trait SNV still switches the trait on ---------------------------------------
def _mutate_one_gene(cell, sel, seg, pos):
    """Put a single SNV on one copy of one gene, through the ordinary summary seam."""
    cell.genome = [{"p": [a.copy() for a in cell.genome[s]["p"]],
                    "m": [a.copy() for a in cell.genome[s]["m"]]}
                   for s in range(cell.n_segments)]
    cell.genome_summary = dict(cell.genome_summary)
    cell.genome_summary["seg_cns"] = list(cell.genome_summary["seg_cns"])
    cell.genome_summary["seg_mut_drivers"] = list(cell.genome_summary["seg_mut_drivers"])
    cell.genome[seg]["p"][0][pos] = True
    bits = np.zeros(cell.segment_sizes[seg], dtype=bool)
    bits[pos] = True
    cell.update_genome_summary_mutation(sel, bits, seg)
    cell.update_evolutionary_parameters(sel)


@pytest.mark.parametrize("axis,getter", [("breach", "get_breach"),
                                         ("stromal_survival", "get_stromal_survival"),
                                         ("met_survival", "get_met_survival")])
def test_mutation_mode_trait_snv_still_switches_the_trait_on(axis, getter):
    """A trait must still be ACQUIRABLE by mutation under ``"mutation"`` — that is what keeps breach
    a real, earnable gate on invasion — and be worth EXACTLY what it is worth under ``"dosage"``."""
    values = {}
    for mode in ("dosage", "mutation"):
        sel = _selection(mode)
        gene = int(getattr(sel, getter)()[0])
        seg = int(np.searchsorted(sel._seg_offsets, gene, side="right") - 1)
        c = _cell(sel)
        assert c.evolutionary_parameters[axis] == 0.0
        _mutate_one_gene(c, sel, seg, gene - int(sel._seg_offsets[seg]))
        assert c.evolutionary_parameters[axis] > 0.0, f"{axis} not acquirable by SNV under {mode}"
        # and it costs proliferation, which is the whole point of the go-or-grow trade-off
        assert sel.proliferation_cost(c.evolutionary_parameters) < 1.0
        values[mode] = c.evolutionary_parameters[axis]
    assert values["dosage"] == pytest.approx(values["mutation"], rel=1e-12)


# --- the driver (oncogene/TSG) dosage fitness is untouched -------------------------------------
def test_driver_dosage_fitness_identical_in_both_modes():
    """The CINner oncogene/TSG copy-number fitness — and the dispersal axis — must be byte-identical
    between the modes. ``trait_source`` only ever changes what the TRAIT axes read."""
    traces = {}
    for mode in ("dosage", "mutation"):
        sel = _selection(mode)
        rng = np.random.default_rng(9)      # identical stream -> identical genomes
        div, disp, ploidy = [], [], []
        for _ in range(150):
            c = _cell(sel)
            for _ in range(4):
                c.mutate(rng, sel)
            div.append(sel.update_division_rate(c.genome_summary))
            disp.append(sel.update_dispersal_rate(c.genome_summary))
            ploidy.append(c.genome_summary["ploidy"])
        traces[mode] = (np.array(div), np.array(disp), np.array(ploidy))
    for a, b in zip(traces["dosage"], traces["mutation"]):
        assert a.tobytes() == b.tobytes()   # same genomes, same driver fitness, bit-for-bit
    assert traces["dosage"][0].std() > 0    # the comparison is not vacuous


# --- the population statement: the copy-number proliferation tax is gone ------------------------
def test_mutation_mode_removes_the_copy_number_proliferation_tax():
    """On a realistic-shaped genome, a single CNA under ``"mutation"`` moves the net division rate by
    the DRIVER fitness alone: ``net == min(baseline * driver_fitness, max_birth_rate)`` exactly. Under
    ``"dosage"`` that identity fails for a large share of CNAs, because the incidental trait switch
    takes its cut first."""
    base, cap = REAL_CANCER["division_rate"], REAL_CANCER["max_birth_rate"]
    frac_up = {}
    for mode in ("dosage", "mutation"):
        sel = _selection(mode, genome=REAL_GENOME, selection=REAL_SELECTION)
        rng = np.random.default_rng(3)
        clean = up = 0
        n = 300
        for _ in range(n):
            c = _cell(sel, genome=REAL_GENOME, cancer=REAL_CANCER)
            c.mutate(rng, sel, snv_prob=0.0, cnv_prob=1.0, wgd_rate=0.0)
            raw = sel.update_division_rate(c.genome_summary)
            net = c.evolutionary_parameters["division_rate"]
            clean += abs(net - min(base * raw, cap)) < 1e-12
            up += net > base + 1e-12
        frac_up[mode] = up / n
        if mode == "mutation":
            assert clean == n            # every CNA: driver fitness only, no trait tax
        else:
            assert clean < 0.6 * n       # most CNAs pay a tax they never earned
    # the headline: the share of CNAs that RAISE the net division rate goes from a few percent to the
    # share the driver model actually calls beneficial
    assert frac_up["dosage"] < 0.15
    assert frac_up["mutation"] > 0.30
