"""The paper-1 simulation regime: one canonical tumour, declared variants of it, and two scales.

Every result in paper 1 comes from the canonical breach-gated ductal field of
``notebooks/example_config.yaml`` (through :mod:`realistic_regime`) or from a declared variant of it,
so no two figures grow different tumours by accident. This module adds what the paper needs on top
of the tutorial config:

* **Expression and microenvironment on.** The program layer (calibrated in :mod:`programs_common`)
  with both niche arms — the epithelial wall drives the invasive ``emt`` program (the genotype–niche
  confound) and the hypoxia field drives the ``hypoxia`` program — allele-specific dosage for the
  allele-aware callers, and the cohort's hypoxia field.
* **Invasion that tells the DCIS-to-invasion story.** See :data:`SELECTION`.
* **Two tiers.** ``ISCC_PAPER_TIER=preview`` (the default) runs locally: the showcase tumour at the
  mid scale (grid 96) and sweeps at the small scale (grid 48). ``full`` is the HPC version: the
  showcase at the cm scale (grid 170), sweeps at the mid scale, more seeds. Nothing else changes.
"""
import copy
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import programs_common as PC  # noqa: E402
import realistic_regime as RR  # noqa: E402
from iscc.tumor.models import GenotypeTumor  # noqa: E402
from iscc.tumor.programs import DEFAULT_SEEDED_PROGRAMS  # noqa: E402

TIER = os.environ.get("ISCC_PAPER_TIER", "preview")
TIERS = {
    "preview": {"showcase_scale": "mid", "sweep_scale": "small", "seeds": 5,
                "showcase_target": 60_000, "sweep_target": 15_000, "max_cells": 4_000},
    "full": {"showcase_scale": "cm", "sweep_scale": "mid", "seeds": 10,
             "showcase_target": 200_000, "sweep_target": 60_000, "max_cells": 8_000},
}
T = TIERS[TIER]

# Invasion. The tutorial config's breach target (prop_breach 0.02) was set when the genome had 600
# genes (~12 breach genes); at today's 6,000 genes it is 120, and one heterozygous hit gives a 64%
# chance of crossing the wall per attempt (breach_effects 2.8). Breach then arises within a couple of
# generations inside the founding acinus and the tumour invades as soon as the acinus opens: there is
# no in-situ phase to speak of (measured on the small field). The values
# below restore the intended target size (12 breach genes) and make a single hit a weak advantage at
# the wall (a 1% crossing chance per attempt). Measured over 3 seeds on the small field: breach
# carriers are ~0% when the acinus opens (the early breach clones are purged by their cost instead
# of fixing), and every first stromal invasion is by breach-carrying cells. The in-situ phase is still
# short at this scale (10-55 generations from release to invasion).
# Stromal survival is selected where it pays: its proliferation cost is 0.05 (the tutorial's 0.6 cost a third of
# the division rate for one hit, far more than it saved in the stroma, so it was never selected), one
# hit protects more (effect 4.0, level 0.75, against 2.2 and 0.545), and the stromal hazard it offsets
# is 0.9 (STROMA below). Its target stays 0.2% of genes: at three times that the trait, nearly neutral
# at this cost, spreads in every tumour whether or not the stroma selects it.
# Driver effect 1.03 in CINner's dosage law (the package default is 1.1): at 1.1 the division rate of
# almost every cell reaches max_birth_rate during the in-situ phase, so after the release clones in a
# full deme no longer differ in division rate (99.8-100% of cells at the cap in the full-tier
# registry). At 1.03 most cells stay below the cap and the between-clone spread of division rates is
# about 7%, and the regime still meets every registered tuning criterion.
SELECTION = {"prop_breach": 0.002, "prop_stromal_survival": 0.002, "breach_effects": 1.01,
             "driver_effects": 1.03, "stromal_survival_cost": 0.05, "stromal_survival_effects": 4.0}
# The hazard resident stromal cells add to cancer cells, sigma * (stromal share of the deme) * (1 -
# stromal survival); with the effect of 4.0 per hit, 0.9 makes stromal survival enriched about 5.6-fold
# in the stroma over tumours grown without the hazard (median of ten seeds; 20% of stromal cancer cells
# carry it) and not in the gland, within every registered criterion; at 1.2 genome doublings sweep most
# tumours.
STROMA = {"stromal_hazard": 0.9}

# The founding acinus stays closed for 200 generations after it is half full: long enough for a
# truncal copy-number layer, short enough that the in-situ phase is not one deme for most of the run.
# The clock counts generations (the default counts tau steps, so the earlier `generations: 400` at tau
# 0.5 was this same window).
ORIGIN_CONFINEMENT = dict(RR._CFG["spatial_params"]["origin_confinement"], generations=200,
                          clock="generations")

# The cell law. Division follows demon's invasion
# rule: a division attempts a mutation with probability `mutation_rate` and, independently, one of its
# daughters disperses with probability `dispersal_rate`. Crowding is demon's two-valued death step,
# the same for every clone in a deme: the baseline death rate up to the capacity K, and `crowding_d1`
# above it, so clones in a full deme keep competing through their division rates. The slot cap sits at
# `crowding_overfill` x K, which has to exceed 1 for a deme to pass K and meet the step at all, and
# `maximum_death_rate` clamps every death rate, so it is raised with `crowding_d1`.
# The values were chosen on preview runs against criteria registered before any run: overfill 1.5 and d1 5 keep the in-situ phase, breach-gated invasion and the
# cost; d1 20 doubles the cost (tau substeps scale with it) and lets genome doubling sweep. Selection
# now acts inside full demes, so the same mutation supply alters more of the genome: `mutation_rate`
# 0.15 (the ER+ preset is 0.30) brings the fraction of genome altered back to the old law's.
CROWDING_OVERFILL, CROWDING_D1 = 1.5, 5.0
# `wgd_rate` 0.01 (the ER+ preset is 0.001): with driver effect 1.03 on the corrected tau-leaping, 30%
# of tumours carry a whole-genome duplication at 0.01 (the lower edge of PCAWG's 30-50%), against 15% at
# 0.005 (20 tumours per rate at the sweep scale). It is an effective rate for this lesion size, whose
# lineages are a few doublings shallower than a real lesion's (an earlier version of the regime used
# 0.005).
CANCER = {"mutation_rate": 0.15, "wgd_rate": 0.01}
# Tau-leaping: binomial deaths
# (Tian and Burrage 2004; Chatterjee et al. 2005) and each leap chosen by the epsilon-bounded rule of Cao,
# Gillespie and Petzold (2006) at epsilon 0.01. The earlier fixed sub-step let a full deme lose about a
# quarter of its cells in one leap under the d1 step, and its capped Poisson deaths favoured small clones.
DEME = {"crowding_overfill": CROWDING_OVERFILL, "crowding_d1": CROWDING_D1,
        "maximum_death_rate": CROWDING_D1,
        "tau_deaths": "binomial", "tau_step": "cgp", "tau_epsilon": 0.01}

# The metastatic deposit (Fig 1a, 2a, 2e): the keys of configs/landing.yaml that
# add the heritable met-survival trait and a second lattice reached through a vessel deme, without the
# landing's treatment resistance. A variant passes MET_SELECTION as `selection` and met_spatial(scale)
# as `spatial`. The trait stays rare in the primary (1-5% of cells in a smoke run) and seeding
# follows the release of the acinus by 30-200 generations.
MET_SELECTION = {"prop_met_survival": 0.015, "met_survival_effects": 3.0, "met_survival_cost": 0.15}


# Immune cells (a declared variant; the canonical regime has none). Static T and NK cells, the cells
# that kill in iscc's immune model, seeded in the stroma at 0.2 of K_stroma (6 per stroma deme, beside
# its 9 fibroblasts: twice the T/NK-to-fibroblast ratio of the normal-breast atlas, 0.37, for the
# infiltrate of a carcinoma; not fitted) and in the duct wall at 0.02 of K_duct (one per wall deme, the
# intraepithelial lymphocytes). A variant passes IMMUNE_SPATIAL as `spatial`, IMMUNE_SELECTION as
# `selection` and IMMUNE as `immune`.
IMMUNE_SPATIAL = {"immune_density": 0.2, "immune_duct_density": 0.02}
# Immune resistance at 0.2% of genes, the target breach and stromal survival were cut to (14 genes
# instead of the example config's 116). At 2% point mutations hit the trait so often that it spreads
# without any immune cells (traits follow trait_source "mutation", so only mutated copies count): in an
# immune-free tumour of the small field 74% of cancer cells carry a mutated resistance copy, which leaves
# no room for immune pressure to select it (exp-immune-variant). One hit protects 75% (effect 4.0,
# stromal survival's): at the example config's 1.1 a hit protects 9%, and immune pressure selected no
# resistance until it was strong enough to double the growth time. Escape events in tumours (HLA or
# B2M loss) protect substantially with one event. Neither change moves another gene's role.
IMMUNE_SELECTION = {"prop_immune_resistance": 0.002, "immune_resistant_effects": 4.0}
# The kill rate: a cancer cell dies at prob_kill x (the immune share of its deme) x (1 - its immune
# resistance). Picked on preview tumours (stage immune_grid3 of tune_regime_v2, five seeds per rate)
# as the largest of 0.5-16 at which the regime still met its criteria and immune resistance was
# selected: at 8, growth to 15,000 cancer cells takes 1.3 times as long as without immune cells and
# the mean immune resistance of cancer cells is 13 times as high; at 16 the cost is 5 times
# (exp-immune-variant). A wild-type cancer cell entering a stroma deme then dies of immune killing at
# about 3 per generation, so invading the stroma takes immune resistance as well as breach.
IMMUNE = {"prob_kill": 8.0}
# The interferon response of cancer cells near T and NK cells (readout only): the immune density of
# each deme, smoothed over one deme (the IFN-gamma they release reaches the cells around them), drives
# an `interferon_response` program, which the human genome layer loads on the Hallmark IFN-gamma
# response set. The gain of 10 gives a drive of 2 at the stromal density, against 3 for a full duct
# wall and full hypoxia; a choice, not a fit.
IMMUNE_SIGNAL = {"field": "immune", "program": "interferon_response", "strength": 10.0, "lengthscale": 1.0}


def met_spatial(scale):
    """The metastatic lattice at ``scale``: as large as the primary field, with the landing's host
    tissue, vessel entry and transit floor."""
    return {"met_grid_size": RR.SCALES[scale]["grid_size"], "K_met": 60, "host_fill_frac": 0.5,
            "met_seed_kappa": 0.03, "met_hazard": 3.5, "met_transit_floor": 0.001}


# Visium spot swapping, fitted to 10x's V1_Breast_Cancer_Block_A_Section_1: SpotClean's model with a bleeding rate of 0.31, a
# distal share of 0.30 (the slide-wide floor, so no in-spot ambient soup) and a kernel sd of 3.25 spot
# pitches, and the edge falloff measured from the tissue's outermost ring.
VISIUM = {"bleed_rate": 0.31, "bleed_distal": 0.30, "bleed_sigma": 6.5, "ambient_frac": 0.0,
          "edge_sigma": 0.29, "edge_reference": "tissue"}

# Oxygen. The calibrated programs config supplies O2 uniformly at a rate (0.3) far below consumption
# (1.5), so every dense deme, normal or cancer, sits near the same equilibrium and the hypoxia field
# is flat: 90% of cancer cells between 0.78 and 0.93, and normal tissue at 0.6. A hypoxia gene then
# varies by a few percent across the tumour, and there is no gradient for anything to follow. The
# paper regime uses the perfused source instead: vessels live in the stroma, so O2 is supplied only
# where there is no cancer and diffuses into cancer-filled ducts and invasive fronts. A deme is 50-60
# um (the tutorial config's ~50 um; 1 cm over the 170-deme cm scale), and O2 penetrates 100-150 um of
# tissue, so the diffusion length sqrt(D / k) is 2 demes (D 4, k 1); supply is ten times consumption so that normal tissue is
# normoxic. Measured on the preview tumours: normal tissue at hypoxia 0.03, cancer cells 0.16-0.88
# (5th-95th percentile). The field is readout-only, so registry.load applies it to stored tumours.
O2 = {"o2_source": "perfused", "o2_supply": 10.0, "o2_consumption": 1.0, "o2_diffusion": 4.0}
MICROENV = copy.deepcopy(PC.MICROENV)
# The bound on a program's log-fold, c in c * tanh(x / c): without it the paper's program settings
# raise a gene up to e^32-fold in a few cells. Readout only;
# registry.load sets it on stored tumours.
PROGRAM_MAX_LOG_FOLD = 5.0
MICROENV["hypoxia"].update(O2)


# Per-gene dosage, calibrated on TCGA breast tumours (validation/paper/supp_dosage.py, the fit in
# validation/data/dosage_fit.json): the "total" law buffers gains and
# losses alike, and s_g ~ clip(N(0.86, 0.61), 0, 1) matches the per-gene slopes of gains and of losses
# together. The previous prior (0.7, 0.25) under the "haplotype" law could not buffer losses.
DOSAGE = {"dosage_law": "total", "dosage_sensitivity_mean": 0.86, "dosage_sensitivity_sd": 0.61}


def expression(immune=False):
    """The program layer with both niche arms and allele-specific dosage (a fresh copy per call).
    ``immune`` adds the interferon response to nearby immune cells (:data:`IMMUNE_SIGNAL`), for the
    immune variant."""
    e = PC.expression_params(scatter=1.0)
    e["dosage_params"]["allele_specific"] = True
    # Cancer cells start from the baseline of the epithelium they transformed from, and differ from it
    # only by what evolution did. Readout only.
    e["cell_of_origin"] = "epithelial"
    cp = e["coupling_params"]
    cp["niche_program_map"] = {"epithelial": "emt", "hypoxia": "hypoxia"}
    cp["niche_program_strength"] = 3.0
    cp["phenotype_program_strength"] = {"breach": 1.0, "__default__": 0.5}
    e["dosage_params"].update(DOSAGE)
    # The bound on a program's log-fold, set here so that a
    # tumour grown by `grow` carries it; `registry.load` still sets it on tumours stored before.
    e["program_params"]["max_log_fold"] = PROGRAM_MAX_LOG_FOLD
    if immune:
        # The program takes the first background slot: its genes and loadings are that slot's draws,
        # so the rest of the program layer is the same as without it.
        e["program_params"]["seeded_programs"] = [*DEFAULT_SEEDED_PROGRAMS, IMMUNE_SIGNAL["program"]]
        cp["niche_program_map"][IMMUNE_SIGNAL["field"]] = IMMUNE_SIGNAL["program"]
        cp["niche_program_strength"] = {"__default__": cp["niche_program_strength"],
                                        IMMUNE_SIGNAL["field"]: IMMUNE_SIGNAL["strength"]}
        cp["immune_signal_lengthscale"] = IMMUNE_SIGNAL["lengthscale"]
    return e


def config(scale, selection=None, cancer=None, deme=None, spatial=None, genome=None, immune=None,
           update_mode="tau", tau=RR.TAU, coarsen=RR.COARSEN, germline=None):
    """Every :class:`GenotypeTumor` keyword of the paper regime at ``scale`` except the seed and the
    readout layers, with a variant's overrides on top.

    This is the regime's one source: :func:`grow` builds from it, and so must any script that grows a
    paper-regime tumour another way (``supp_tau.py`` runs the exact engine through ``update_mode``).
    ``coarsen`` sets ``coarsen_passengers`` (R2 runs it off). ``germline`` replaces the germline
    parameters of the example config (heterozygous at 0.5% of the neutral genes); it changes no growth
    statistic, only the density of the germline sites the DNA assays read. ``immune`` is the
    ``immune_cell_params`` of a variant with immune cells (:data:`IMMUNE`)."""
    sp = {"origin_confinement": dict(ORIGIN_CONFINEMENT), **STROMA, **(spatial or {})}
    blocks = RR.config_for(scale, genome=genome, selection={**SELECTION, **(selection or {})},
                           cancer={**CANCER, **(cancer or {})}, deme={**DEME, **(deme or {})},
                           spatial=sp)
    if immune is not None:
        blocks["immune_cell_params"] = dict(immune)
    return dict(blocks, update_mode=update_mode, tau=tau, coarsen_passengers=coarsen,
                founder_mutations=RR.FOUNDER_MUTATIONS,
                germline_params=RR.GERMLINE if germline is None else germline)


def grow(seed, scale, target_cancer, spatial_snapshot_every=0, max_steps=3_000, chunk=20,
         expression_params="default", microenv_params="default", coarsen=RR.COARSEN,
         on_chunk=None, layout_seed=None, **overrides):
    """Grow one paper-regime tumour to ``target_cancer`` cancer cells (counts only).

    Returns the tumour without materialised cell data; sample it with :mod:`realistic_regime`'s
    ``resect`` / ``dissociated`` / ``section`` or ``t.make_cell_data(max_cells=...)``. Growth stops
    early, with ``t.reached_target`` False, if the cancer dies out or ``max_steps`` pass. 3,000 steps
    is 1,500 generations: the acinus opens 200 generations after it is half full, and invasion
    normally follows within tens of generations, so a tumour still below target by then has stayed in
    situ, which the manifest records.
    Growth stops at the first event (exact) or tau leap that reaches ``target_cancer``, not at the end
    of a chunk, so ``t.get_cancer_size()`` is the size reached, just past the target, and ``t.time``
    the time it took.
    ``overrides`` are :func:`config`'s (the variant's blocks, ``update_mode``, ``tau``); a variant
    with immune cells (``immune``) reads out with the interferon response, ``expression(immune=True)``.
    ``on_chunk(t)``, if given, is called after every chunk, for monitoring.
    ``layout_seed`` is :class:`GenotypeTumor`'s (``None``, the default, is the package's
    ``DEFAULT_LAYOUT_SEED``), so ``functools.partial(grow, scale=..., target_cancer=...)`` is the
    growth function of an :class:`iscc.cohort.Cohort`."""
    cfg = config(scale, coarsen=coarsen, **overrides)
    expr = (expression(immune=overrides.get("immune") is not None) if expression_params == "default"
            else expression_params)
    menv = copy.deepcopy(MICROENV) if microenv_params == "default" else microenv_params
    t = GenotypeTumor(seed=seed, layout_seed=layout_seed, max_cells=1, expression_params=expr,
                      microenv_params=menv, spatial_snapshot_every=spatial_snapshot_every, **cfg)
    steps = 0
    while 0 < t.get_cancer_size() < target_cancer and steps < max_steps:
        t.grow(n_steps=chunk, seed=seed, target_cancer=target_cancer)
        steps += chunk
        if on_chunk is not None:
            on_chunk(t)
    t.reached_target = t.get_cancer_size() >= target_cancer
    t.max_cells = T["max_cells"]
    t.cell_data = None
    return t
