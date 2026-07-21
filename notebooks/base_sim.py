"""Shared spatial substrate for the iscc *science* showcase notebooks.

Every science notebook in ``notebooks/`` imports this helper so they all analyse the **same**
deterministic, spatially-structured tumour (and the same 5-tumour cohort). Nothing is cached to disk
— the tumour is small enough to re-grow in ~40-60 s, which keeps the notebooks self-contained and the
ground truth exactly reproducible.

Design choices baked in here (see the per-notebook narratives for why they matter):

* **Spatial structure + microenvironment.** ``structure_radius > 0`` seeds a diploid
  epithelial/stromal field with a cancer focus growing inside it, so every notebook analyses a
  *mixture* (malignant + normal), never a pre-filtered cancer-only matrix.
* **CINner-style selection** (``prop_driver``) drives clonal evolution.
* **WGD on** (``wgd_rate = 0.05``) — whole-genome duplication at a realistic PCAWG-like prevalence.
* **Allele-specific expression on** and the **gene-program layer on** (reusing the vetted
  ``validation/programs_common.expression_params``), so the expression notebooks have real structure.
* **Shared landscape across the cohort.** ``grow_base_tumor`` never passes ``layout_seed``, so every
  seed falls back to ``DEFAULT_LAYOUT_SEED`` and shares the SAME gene roles, program dictionary and
  epistasis landscape — the comparability guarantee the cohort notebooks recover.
"""
import os
import sys

import numpy as np

# reuse the vetted expression_params() from the validation harness
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "validation"))
import programs_common as PC  # noqa: E402

from iscc.tumor.models import GenotypeTumor  # noqa: E402

BASE_SEED = 3
COHORT_SEEDS = [3, 4, 5, 6, 7]  # 5 tumours sharing the landscape

GENOME = {"n_segments": 12, "segment_size": 50}
# CINner selection tuned so the founder is near-neutral (small, distributed fitness) rather than
# hyper-polyclonal or instantly saturated: weak drivers (`driver_effects` just above 1) + a low
# mutation rate keep most clones sub-maximal so there is a real fitness GRADIENT to see. Dispersal
# mutations are rare but strong (a motility switch). No resistance genes (untreated primary).
SELECTION = {"prop_driver": 0.15, "prop_dispersal": 0.003,
             "prop_immune_resistance": 0.0, "prop_treatment_resistance": 0.0,
             "driver_effects": 1.03, "dispersal_effects": 12.0}
CANCER = {"division_rate": 0.4, "death_rate": 0.03, "max_birth_rate": 0.98,
          "mutation_rate": 0.02, "dispersal_rate": 0.025, "cnv_prob": 0.15,
          "wgd_rate": 0.05}  # low baseline division/dispersal so growth fills the lumen, WGD on
# Gland geometry: an epithelial ring at `structure_radius` (a duct cross-section), an empty lumen the
# cancer grows into, stroma outside. `resident_pressure_ref` + `crowding_mode="own"` make a cancer
# cell in a normal-occupied deme need enough fitness to survive the crowding there, so breaching the
# ring/stroma is fitness-gated (DCIS -> microinvasion). K per deme = 8.
DEME = {"carrying_capacity": 8, "initial_cancer_cells": 4,
        "resident_pressure_ref": 0.55, "crowding_mode": "own"}
# R=18 duct holds ~13k cells, so growing to ~10k fills it ~3/4 -> the ring stays visible and only a
# modest fraction microinvades (a contained DCIS), rather than a blob that overgrows the gland.
SPATIAL = {"grid_size": 54, "structure_radius": 18}
# Comedo-type hypoxia: O2 is supplied ONLY by the perfused stroma/empty tissue (`o2_source`), so the
# cancer-packed lumen core starves and the hypoxia program lights up there (route 3, niche->program).
# `o2_supply` must EXCEED `o2_consumption` (default 1) or even the stroma reads hypoxic (a flat field);
# `o2_diffusion` sets the oxygen penetration depth (tuned so the gradient spans the duct radius).
MICROENV = {"hypoxia": {"strength": 0.8, "o2_diffusion": 25.0, "o2_supply": 8.0,
                        "o2_source": "perfused", "n_genes": 8}}


def EXPR():
    """Program layer ON (via the vetted validation params) + allele-specific dosage ON.

    Two couplings are configured on top of the vetted params so the program gradients track their
    drivers to a comparable degree:

    * ``phenotype_program_strength`` is a PER-PHENOTYPE dict. Division's fold-change is bounded by
      ``max_birth_rate`` (it saturates ~1.5x) whereas dispersal's is unbounded (tens x), so a single
      shared gain would bury the proliferation signal under activity noise. A larger division gain
      restores parity, so proliferation tracks division about as well as EMT tracks dispersal.
    * ``niche_program_map`` routes the hypoxia FIELD (from ``MICROENV``) to the hypoxia program.
    """
    e = PC.expression_params(scatter=1.0)
    e["dosage_params"]["allele_specific"] = True
    cp = e["coupling_params"]
    cp["phenotype_program_strength"] = {"__default__": 0.5, "division_rate": 10.0}
    cp["niche_program_map"] = {"hypoxia": "hypoxia"}
    cp["niche_program_strength"] = 8.0  # lift the field signal above per-cell activity noise
    return e


def _n_cancer(t):
    """Number of live cancer cells across the genotype counts."""
    return int(sum(c for g, c in t.genotypes_counts.items() if c > 0 and t._is_cancer(g)))


def grow_base_tumor(seed=BASE_SEED, target_cancer=10000, cancer_params=None, expression=None,
                    selection_params=None, verbose=False):
    """Grow one spatially-structured tumour to >= ``target_cancer`` cancer cells and materialise it.

    Never passes ``layout_seed`` -> defaults to ``DEFAULT_LAYOUT_SEED`` so every seed shares the SAME
    gene roles / program dictionary / epistasis landscape (the cohort-comparability guarantee).

    Parameters
    ----------
    seed : int
        Growth seed (also the cohort member id).
    target_cancer : int
        Grow until at least this many cancer cells exist.
    cancer_params : dict, optional
        Overrides merged into the base ``CANCER`` config (e.g. ``{"wgd_rate": 0.0}`` for the WGD-off
        comparison in the allele-CNA notebook).
    expression : dict, optional
        Overrides the default ``EXPR()`` expression params.
    selection_params : dict, optional
        Overrides merged into the base ``SELECTION`` config (e.g. adding an ``epistasis_params``
        network for the MHN/TreeMHN cohort notebook — the gene roles stay shared via ``layout_seed``).
    """
    cancer = dict(CANCER)
    if cancer_params:
        cancer.update(cancer_params)
    selection = dict(SELECTION)
    if selection_params:
        selection.update(selection_params)
    t = GenotypeTumor(seed=seed, genome_params=GENOME, selection_params=selection,
                      cancer_cell_params=cancer, deme_params=DEME, spatial_params=SPATIAL,
                      expression_params=(expression if expression is not None else EXPR()),
                      microenv_params=MICROENV, update_mode="tau", tau=1.0)
    # Adaptive stepping: coarse (10) while far from the target, fine (2) near it, so the tumour lands
    # close to `target_cancer` instead of overshooting by ~50% in one big leap and overgrowing the duct.
    while True:
        ncan = _n_cancer(t)
        if ncan >= target_cancer:
            break
        n_steps = 10 if ncan < 0.7 * target_cancer else 2
        t.grow(n_steps=n_steps, seed=seed)
        if verbose:
            print(f"  seed={seed}: {_n_cancer(t)} cancer cells", flush=True)
    t.make_cell_data()
    return t


def grow_cohort(seeds=COHORT_SEEDS, target_cancer=10000, **kwargs):
    """Generator over ``(seed, tumour)`` — grow one at a time so RAM never holds 5 full tumours.

    Each materialised base tumour is ~1-2 GB; the cohort notebooks must assay each tumour to a
    COMPACT matrix and drop the tumour before advancing (that is what makes this a generator).
    """
    for s in seeds:
        yield s, grow_base_tumor(seed=s, target_cancer=target_cancer, **kwargs)


# ------------------------------------------------------------------ shared per-cell helpers
def cell_types(t):
    """Per-cell coarse type array ('cancer'/'epithelial'/'stromal'/...) aligned to ``cell_data``."""
    gid = t.cell_data["cell_type"]["cell_id"].astype(str).values
    g = t.genotypes
    return np.array([g[x].type if x in g else "?" for x in gid])


def cell_rate(t, rate="division_rate"):
    """Per-cell evolutionary rate ('division_rate','dispersal_rate',...) aligned to ``cell_data``.

    NaN for any cell whose genotype does not carry the rate. ``division_rate`` is the CINner fitness
    (baseline x driver multipliers, capped at ``max_birth_rate``); ``dispersal_rate`` is the motility.
    """
    gid = t.cell_data["cell_type"]["cell_id"].astype(str).values
    g = t.genotypes
    return np.array([g[x].evolutionary_parameters.get(rate, np.nan) if x in g else np.nan
                     for x in gid])


def cell_program(t, name):
    """Per-cell activity of a named gene program ('proliferation','emt','hypoxia',...), or NaN col."""
    Z = t.cell_data.get("cell_program")
    if Z is None or name not in Z.columns:
        import numpy as _np
        return _np.full(len(t.cell_data["cell_type"]), _np.nan)
    return Z[name].values


def gland_ring(ax, color="k", ls="--", lw=0.9, alpha=0.6):
    """Draw the epithelial-ring circle (radius ``structure_radius``) on a spatial axes, for context."""
    R = SPATIAL["structure_radius"]
    c = SPATIAL["grid_size"] / 2.0
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(c + R * np.cos(th), c + R * np.sin(th), color=color, ls=ls, lw=lw, alpha=alpha)


def cell_radius(t):
    """Per-cell distance from the duct centre (for scoring core/rim gradients like hypoxia)."""
    crd = t.cell_data["cell_crd"]
    c = SPATIAL["grid_size"] / 2.0
    return np.sqrt((crd["row"].values - c) ** 2 + (crd["col"].values - c) ** 2)


def cancer_mask(t):
    """Boolean mask selecting the malignant cells within the mixed ``cell_data``."""
    return cell_types(t) == "cancer"


def segment_offsets(t):
    """Per-segment start indices into the per-gene copy-number matrix (len n_segments+1)."""
    sizes = np.asarray(t.selection.segment_sizes)
    return np.concatenate([[0], np.cumsum(sizes)]).astype(int)


def segment_cn(t):
    """(cells x n_segments) TOTAL copy number (first gene of each segment) + per-gene->segment map."""
    sizes = np.asarray(t.selection.segment_sizes)
    offs = segment_offsets(t)
    cnv = t.cell_data["cell_cnv"].values
    seg = np.stack([cnv[:, offs[s]] for s in range(t.n_segments)], axis=1).astype(float)
    gene_seg = np.concatenate([np.full(sizes[s], s) for s in range(t.n_segments)]).astype(int)
    return seg, gene_seg
