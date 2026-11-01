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

# The founding acinus stays closed this many generations after it is half full: long enough for a
# truncal copy-number layer, short enough that the in-situ phase is not one deme for most of the run.
ORIGIN_CONFINEMENT = dict(RR._CFG["spatial_params"]["origin_confinement"], generations=400)

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


def expression():
    """The program layer with both niche arms and allele-specific dosage (a fresh copy per call)."""
    e = PC.expression_params(scatter=1.0)
    e["dosage_params"]["allele_specific"] = True
    cp = e["coupling_params"]
    cp["niche_program_map"] = {"epithelial": "emt", "hypoxia": "hypoxia"}
    cp["niche_program_strength"] = 3.0
    cp["phenotype_program_strength"] = {"breach": 1.0, "__default__": 0.5}
    return e


def config(scale, selection=None, cancer=None, deme=None, spatial=None, genome=None):
    """The parameter blocks of the paper regime at ``scale``, with a variant's overrides on top."""
    sp = {"origin_confinement": dict(ORIGIN_CONFINEMENT), **(spatial or {})}
    return RR.config_for(scale, genome=genome, selection={**SELECTION, **(selection or {})},
                         cancer=cancer, deme=deme, spatial=sp)


def grow(seed, scale, target_cancer, spatial_snapshot_every=0, max_steps=3_000, chunk=20,
         expression_params="default", microenv_params="default", coarsen=RR.COARSEN, **overrides):
    """Grow one paper-regime tumour to ``target_cancer`` cancer cells (counts only).

    Returns the tumour without materialised cell data; sample it with :mod:`realistic_regime`'s
    ``resect`` / ``dissociated`` / ``section`` or ``t.make_cell_data(max_cells=...)``. Growth stops
    early, with ``t.reached_target`` False, if the cancer dies out or ``max_steps`` pass. 3,000 steps
    is 1,500 generations: the acinus opens at 400 and invasion normally follows within ~50, so a
    tumour still below target by then has stayed in situ, which the manifest records."""
    cfg = config(scale, **overrides)
    expr = expression() if expression_params == "default" else expression_params
    menv = copy.deepcopy(MICROENV) if microenv_params == "default" else microenv_params
    t = GenotypeTumor(seed=seed, update_mode="tau", tau=RR.TAU, coarsen_passengers=coarsen,
                      max_cells=1, founder_mutations=RR.FOUNDER_MUTATIONS,
                      germline_params=RR.GERMLINE, expression_params=expr, microenv_params=menv,
                      spatial_snapshot_every=spatial_snapshot_every, **cfg)
    steps = 0
    while 0 < t.get_cancer_size() < target_cancer and steps < max_steps:
        t.grow(n_steps=chunk, seed=seed)
        steps += chunk
    t.reached_target = t.get_cancer_size() >= target_cancer
    t.max_cells = T["max_cells"]
    t.cell_data = None
    return t
