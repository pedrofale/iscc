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
# no in-situ phase to speak of (measured on the small field, paper1 probe, 2026-10-31). The values
# below restore the intended target size (12 breach genes) and make a single hit a weak advantage at
# the wall (a 1% crossing chance per attempt). Measured over 3 seeds on the small field: breach
# carriers are ~0% when the acinus opens (the early breach clones are purged by their cost instead
# of fixing), and every first stromal invasion is by breach-carrying cells. The in-situ phase is still
# short at this scale (10-55 generations from release to invasion).
# Stromal survival is kept rare: at 0.02 it hitchhikes to fixation in some seeds, and in no setting
# tried was it enriched in stroma over ducts, because the permissive stroma (stromal_hazard 0.6)
# gives it little to do. Selecting it after invasion needs a stromal-hazard study; it is recorded as
# an open modelling item, and the figures do not claim it.
SELECTION = {"prop_breach": 0.002, "prop_stromal_survival": 0.002, "breach_effects": 1.01}

# The founding acinus stays closed for 200 generations after it is half full: long enough for a
# truncal copy-number layer, short enough that the in-situ phase is not one deme for most of the run.
# The clock counts generations (the default counts tau steps, so the earlier `generations: 400` at tau
# 0.5 was this same window; issues note N8).
ORIGIN_CONFINEMENT = dict(RR._CFG["spatial_params"]["origin_confinement"], generations=200,
                          clock="generations")

# The cell law (vault: decision-cell-law-from-reference-simulators). Division follows demon's invasion
# rule: a division attempts a mutation with probability `mutation_rate` and, independently, one of its
# daughters disperses with probability `dispersal_rate`. Crowding is demon's two-valued death step,
# the same for every clone in a deme: the baseline death rate up to the capacity K, and `crowding_d1`
# above it, so clones in a full deme keep competing through their division rates. The slot cap sits at
# `crowding_overfill` x K, which has to exceed 1 for a deme to pass K and meet the step at all, and
# `maximum_death_rate` clamps every death rate, so it is raised with `crowding_d1`.
# The values were chosen on preview runs against criteria registered before any run (vault:
# exp-regime-v2-tuning): overfill 1.5 and d1 5 keep the in-situ phase, breach-gated invasion and the
# cost; d1 20 doubles the cost (tau substeps scale with it) and lets genome doubling sweep. Selection
# now acts inside full demes, so the same mutation supply alters more of the genome: `mutation_rate`
# 0.15 (the ER+ preset is 0.30) brings the fraction of genome altered back to the old law's.
CROWDING_OVERFILL, CROWDING_D1 = 1.5, 5.0
# `force_snv` off: an SNV event whose Poisson draws all come up 0 leaves the genome unchanged, as in
# demon and CINner, instead of placing one SNV anyway (vault: paper1-sprint, decisions).
CANCER = {"mutation_rate": 0.15, "force_snv": False}
DEME = {"dispersal_law": "noble", "crowding_mode": "lottery", "crowding_law": "noble",
        "crowding_overfill": CROWDING_OVERFILL, "crowding_d1": CROWDING_D1,
        "maximum_death_rate": CROWDING_D1}
# Per-run random streams keyed by (seed, purpose, step), so neighbouring seeds never share a stream.
RNG_STREAMS = "independent"

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
MICROENV["hypoxia"].update(O2)


# Per-gene dosage, calibrated on TCGA breast tumours (validation/paper/supp_dosage.py, the fit in
# validation/data/dosage_fit.json; vault: decision-dosage-total-law): the "total" law buffers gains and
# losses alike, and s_g ~ clip(N(0.86, 0.61), 0, 1) matches the per-gene slopes of gains and of losses
# together. The previous prior (0.7, 0.25) under the "haplotype" law could not buffer losses.
DOSAGE = {"dosage_law": "total", "dosage_sensitivity_mean": 0.86, "dosage_sensitivity_sd": 0.61}


def expression():
    """The program layer with both niche arms and allele-specific dosage (a fresh copy per call)."""
    e = PC.expression_params(scatter=1.0)
    e["dosage_params"]["allele_specific"] = True
    # Cancer cells start from the baseline of the epithelium they transformed from, and differ from it
    # only by what evolution did (vault: decision-cancer-cell-of-origin-baseline). Readout only.
    e["cell_of_origin"] = "epithelial"
    cp = e["coupling_params"]
    cp["niche_program_map"] = {"epithelial": "emt", "hypoxia": "hypoxia"}
    cp["niche_program_strength"] = 3.0
    cp["phenotype_program_strength"] = {"breach": 1.0, "__default__": 0.5}
    e["dosage_params"].update(DOSAGE)
    return e


def config(scale, selection=None, cancer=None, deme=None, spatial=None, genome=None):
    """The parameter blocks of the paper regime at ``scale``, with a variant's overrides on top."""
    sp = {"origin_confinement": dict(ORIGIN_CONFINEMENT), **(spatial or {})}
    return RR.config_for(scale, genome=genome, selection={**SELECTION, **(selection or {})},
                         cancer={**CANCER, **(cancer or {})}, deme={**DEME, **(deme or {})}, spatial=sp)


def grow(seed, scale, target_cancer, spatial_snapshot_every=0, max_steps=3_000, chunk=20,
         expression_params="default", microenv_params="default", coarsen=RR.COARSEN,
         rng_streams=RNG_STREAMS, on_chunk=None, **overrides):
    """Grow one paper-regime tumour to ``target_cancer`` cancer cells (counts only).

    Returns the tumour without materialised cell data; sample it with :mod:`realistic_regime`'s
    ``resect`` / ``dissociated`` / ``section`` or ``t.make_cell_data(max_cells=...)``. Growth stops
    early, with ``t.reached_target`` False, if the cancer dies out or ``max_steps`` pass. 3,000 steps
    is 1,500 generations: the acinus opens 200 generations after it is half full, and invasion
    normally follows within tens of generations, so a tumour still below target by then has stayed in
    situ, which the manifest records.
    ``on_chunk(t)``, if given, is called after every chunk, for monitoring."""
    cfg = config(scale, **overrides)
    expr = expression() if expression_params == "default" else expression_params
    menv = copy.deepcopy(MICROENV) if microenv_params == "default" else microenv_params
    t = GenotypeTumor(seed=seed, update_mode="tau", tau=RR.TAU, coarsen_passengers=coarsen,
                      max_cells=1, founder_mutations=RR.FOUNDER_MUTATIONS,
                      germline_params=RR.GERMLINE, expression_params=expr, microenv_params=menv,
                      spatial_snapshot_every=spatial_snapshot_every, rng_streams=rng_streams, **cfg)
    steps = 0
    while 0 < t.get_cancer_size() < target_cancer and steps < max_steps:
        t.grow(n_steps=chunk, seed=seed)
        steps += chunk
        if on_chunk is not None:
            on_chunk(t)
    t.reached_target = t.get_cancer_size() >= target_cancer
    t.max_cells = T["max_cells"]
    t.cell_data = None
    return t
