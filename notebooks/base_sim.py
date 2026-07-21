"""Shared spatial substrate for the iscc *science* showcase notebooks — the **ductal field**.

Every science notebook in ``notebooks/`` imports this helper so they all analyse the **same**
deterministic, spatially-structured tumour (and the same 5-tumour cohort). Nothing is cached to disk
— the tumour re-grows in ~15-30 s, which keeps the notebooks self-contained and the ground truth
exactly reproducible.

The substrate is a **multi-focal DCIS→IDC lesion on a ductal field** (``DESIGN_ductal_field.md`` +
``DESIGN_phenotype_plasticity.md`` §2): a FIELD of many small epithelial-ring glands scattered in
moderate-density stroma (a population-genetics *island model*), grown from ONE cancer founder. The
design choices baked in here (see the per-notebook narratives for why they matter):

* **Ductal-field substrate + microenvironment.** ``n_glands`` small epithelial rings in stroma
  (``stroma_fill_frac`` real stromal cells) — every notebook analyses a *mixture* (malignant +
  epithelial + stromal), never a pre-filtered cancer-only matrix. A single founder colonises the
  other glands via low-rate **cross-gland (island) dispersal** (``cross_gland_kappa``), so a section
  shows several *clonally related* foci.
* **Compartment-dependent selection (DCIS→IDC).** Two sequenceable heritable traits, ``breach`` and
  ``stromal_survival``, attenuate two live-cell-fraction death hazards (``epithelial_barrier`` at the
  gland wall, ``stromal_hazard`` in the stroma). A lumen founder is confined (DCIS) until a subclone
  evolves the escape traits and invades the stroma (IDC).
* **CINner-style selection** (``prop_driver``) drives clonal evolution; **WGD on** (``wgd_rate=0.05``,
  PCAWG-like); **allele-specific expression on** and the **gene-program layer on** (via the vetted
  ``validation/programs_common.expression_params``).
* **The genetic-vs-niche emt confound.** The invasive ``emt`` program is driven by BOTH ``breach``
  (route-1 genetic arm, in ``DEFAULT_PHENOTYPE_PROGRAM_MAP``) AND the epithelial compartment field
  (route-3 niche arm, ``niche_program_map={"epithelial": "emt"}``). ``prop_dispersal=0``, so the legacy
  ``dispersal_rate→emt`` map is inert; ``breach`` gets a dedicated program gain.
* **Shared landscape across the cohort.** ``grow_base_tumor`` never passes ``layout_seed``, so every
  seed falls back to ``DEFAULT_LAYOUT_SEED`` and shares the SAME gene roles, program dictionary,
  gland-field layout and epistasis landscape — the comparability guarantee the cohort notebooks recover.

Parameters are scaled up from ``validation/validate_compartment_selection.py`` /
``validate_ductal_field.py`` so the confined lesion still reaches **≥10 000 cancer cells** with the
compartment barriers ON (a bigger field + more glands, not weaker barriers).
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
NORMALS = ("epithelial", "stromal", "immune")

GENOME = {"n_segments": 12, "segment_size": 50}
# CINner selection on the ductal field. `prop_driver` gives clonal fitness variation; the two
# compartment axes (`prop_breach`, `prop_stromal_survival`, with their `*_effects`) are the DCIS→IDC
# escape traits. `prop_dispersal=0` (so cross-gland spread rate is a constant × dispersal_rate, and the
# legacy dispersal→emt program route is inert); a small immune-resistance axis keeps the immune
# microenvironment part of the data. Proportions × 600 genes all round to a non-empty gene set (verified
# via t.selection.get_breach()/get_stromal_survival()).
SELECTION = {"prop_driver": 0.04, "prop_dispersal": 0.0, "prop_immune_resistance": 0.02,
             "prop_treatment_resistance": 0.02, "prop_breach": 0.03, "prop_stromal_survival": 0.03,
             "breach_effects": 2.2, "stromal_survival_effects": 2.2}
CANCER = {"division_rate": 0.7, "death_rate": 0.05, "max_birth_rate": 0.95,
          "mutation_rate": 0.6, "dispersal_rate": 0.35, "cnv_prob": 0.15,
          "wgd_rate": 0.05}  # WGD on (PCAWG-like)
DEME = {"carrying_capacity": 20, "initial_cancer_cells": 8, "resident_pressure_ref": 0.2}
# The ductal FIELD: 4 small epithelial-ring glands (radius 4) scattered ≥12 apart on a 40² grid, in
# moderate-density stroma (stroma_fill_frac=0.3) — a legible multi-focal section (like
# validate_ductal_field.py / validate_compartment_selection.py, 4 glands). K_duct/K_stroma are MODERATE
# (a 2D deme stands for a 3D column through the duct/stroma depth, DESIGN_ductal_field.md §3); K_duct is
# larger here so the 4 glands still hold enough of a confined DCIS mass to reach ≥10k cancer WITH the
# barriers on (fewer glands ⇒ bigger ducts, not weaker barriers). cross_gland_kappa is the low
# island-dispersal rate that seeds one gland's lumen from another's (multi-focal spread, no breach).
# epithelial_barrier / stromal_hazard are the two compartment hazards ON (tuned so the lesion shows a
# DCIS→IDC transition AND still reaches ≥10k cancer cells).
SPATIAL = {"grid_size": 40, "structure_radius": 4, "n_glands": 4, "gland_radius": 4,
           "min_gland_sep": 12, "K_duct": 60, "K_stroma": 40, "stroma_fill_frac": 0.3,
           "cross_gland_kappa": 0.06, "cross_gland_lambda": None,
           "epithelial_barrier": 1.2, "stromal_hazard": 0.7}


def EXPR():
    """Program layer ON (via the vetted validation params) + allele-specific dosage ON + the
    genetic-vs-niche emt confound coupling.

    * ``niche_program_map={"epithelial": "emt"}`` routes the epithelial compartment field to the
      invasive/emt program — the **niche arm** of the confound (the same clone expresses emt more at
      the epithelial interface than in the stroma).
    * ``phenotype_program_strength={"breach": 1.0, "__default__": 0.5}`` sizes the **genetic arm**:
      ``breach → emt`` is already in ``DEFAULT_PHENOTYPE_PROGRAM_MAP``; ``breach`` sweeps to
      near-fixation, so it needs a dedicated gain (larger than the ``[0,1)``-scaled default) for a
      substantial genetic signal.
    """
    e = PC.expression_params(scatter=1.0)
    e["dosage_params"]["allele_specific"] = True
    cp = e["coupling_params"]
    cp["niche_program_map"] = {"epithelial": "emt"}
    cp["niche_program_strength"] = 3.0
    cp["phenotype_program_strength"] = {"breach": 1.0, "__default__": 0.5}
    return e


def _n_cancer(t):
    """Number of live cancer cells across the genotype counts."""
    return int(sum(c for g, c in t.genotypes_counts.items() if c > 0 and t._is_cancer(g)))


def grow_base_tumor(seed=BASE_SEED, target_cancer=10000, cancer_params=None, expression=None,
                    selection_params=None, spatial_params=None, verbose=False):
    """Grow one ductal-field tumour to >= ``target_cancer`` cancer cells and materialise it.

    Never passes ``layout_seed`` -> defaults to ``DEFAULT_LAYOUT_SEED`` so every seed shares the SAME
    gene roles / program dictionary / gland-field layout / epistasis landscape (the cohort-comparability
    guarantee).

    Parameters
    ----------
    seed : int
        Growth seed (also the cohort member id).
    target_cancer : int
        Grow until at least this many cancer cells exist.
    cancer_params, selection_params, spatial_params : dict, optional
        Overrides merged into the base ``CANCER`` / ``SELECTION`` / ``SPATIAL`` configs. e.g.
        ``cancer_params={"wgd_rate": 0.0}`` (WGD-off contrast, ``wgd_allele_cna``),
        ``spatial_params={"epithelial_barrier": 0.0, "stromal_hazard": 0.0}`` (the barrier-OFF DCIS→IDC
        control, ``compartment_selection_confound``), or an ``epistasis_params`` network in
        ``selection_params`` (``cohort_mhn_recurrence``).
    expression : dict, optional
        Overrides the default ``EXPR()`` expression params.
    """
    cancer = dict(CANCER)
    if cancer_params:
        cancer.update(cancer_params)
    selection = dict(SELECTION)
    if selection_params:
        selection.update(selection_params)
    spatial = dict(SPATIAL)
    if spatial_params:
        spatial.update(spatial_params)
    t = GenotypeTumor(seed=seed, genome_params=GENOME, selection_params=selection,
                      cancer_cell_params=cancer, deme_params=DEME, spatial_params=spatial,
                      expression_params=(expression if expression is not None else EXPR()),
                      update_mode="tau", tau=1.0)
    # Adaptive stepping: coarse while far from the target, fine near it, so the lesion lands close to
    # `target_cancer` (an invasive mass grows fast, so a single coarse leap would overshoot badly).
    while True:
        ncan = _n_cancer(t)
        if ncan >= target_cancer:
            break
        ratio = ncan / target_cancer
        n_steps = 6 if ratio < 0.12 else (2 if ratio < 0.6 else 1)
        t.grow(n_steps=n_steps, seed=seed)
        if verbose:
            print(f"  seed={seed}: {_n_cancer(t)} cancer cells", flush=True)
    t.make_cell_data()
    return t


def grow_cohort(seeds=COHORT_SEEDS, target_cancer=10000, **kwargs):
    """Generator over ``(seed, tumour)`` — grow one at a time so RAM never holds 5 full tumours.

    Each materialised ductal-field tumour is ~1-3 GB; the cohort notebooks must assay each tumour to a
    COMPACT matrix and drop the tumour before advancing (that is what makes this a generator).
    """
    for s in seeds:
        yield s, grow_base_tumor(seed=s, target_cancer=target_cancer, **kwargs)


# ------------------------------------------------------------------ shared per-cell helpers
def cell_types(t):
    """Per-cell coarse type array ('cancer'/'epithelial'/'stromal'/'immune') aligned to ``cell_data``."""
    gid = t.cell_data["cell_type"]["cell_id"].astype(str).values
    g = t.genotypes
    return np.array([g[x].type if x in g else "?" for x in gid])


def cancer_mask(t):
    """Boolean mask selecting the malignant cells within the mixed ``cell_data``."""
    return cell_types(t) == "cancer"


def cell_gland(t):
    """Per-cell gland id aligned to ``cell_data`` (>=0 = gland index, -1 = stroma)."""
    return t.cell_data["cell_gland"]["gland_id"].values


def cell_rate(t, rate="division_rate"):
    """Per-cell evolutionary rate ('division_rate','breach','stromal_survival',...) aligned to
    ``cell_data``. NaN for any cell whose genotype does not carry the rate. ``division_rate`` is the
    CINner fitness (baseline × driver multipliers, capped at ``max_birth_rate``)."""
    gid = t.cell_data["cell_type"]["cell_id"].astype(str).values
    g = t.genotypes
    return np.array([g[x].evolutionary_parameters.get(rate, np.nan) if x in g else np.nan
                     for x in gid])


def cell_trait(t, name):
    """Per-cell value of a compartment/evo trait from ``cell_evo`` ('breach','stromal_survival',...)."""
    return t.cell_data["cell_evo"][name].values


def cell_program(t, name):
    """Per-cell activity of a named gene program ('proliferation','emt','hypoxia',...), or NaN col."""
    Z = t.cell_data.get("cell_program")
    if Z is None or name not in Z.columns:
        return np.full(len(t.cell_data["cell_type"]), np.nan)
    return Z[name].values


def cell_epithelial_fraction(t):
    """Per-cell epithelial fraction of the cell's deme — the niche arm of the emt confound.

    ``microenv_truth["epithelial"]`` is the per-deme live epithelial fraction (populated by
    ``make_cell_data`` whenever the gland structure is on), indexed here by each cell's deme."""
    demes = t.cell_data["cell_deme"]["deme_id"].values
    return t.microenv_truth["epithelial"][demes]


# ------------------------------------------------------------------ spatial-viz helpers (ductal field)
def draw_glands(ax, t, color="k", ls="--", lw=0.8, alpha=0.5):
    """Overlay every gland's epithelial-ring circle (centre + ``gland_radius``) on a spatial axes."""
    if t.gland_centers is None:
        return
    th = np.linspace(0, 2 * np.pi, 120)
    for (cr, cc) in t.gland_centers:
        ax.plot(cc + t.gland_radius * np.cos(th), cr + t.gland_radius * np.sin(th),
                color=color, ls=ls, lw=lw, alpha=alpha)


def gland_layout_grid(t):
    """(grid × grid) per-deme gland id (>=0), NaN where stroma — the static field layout."""
    grid = np.full((t.grid_size, t.grid_size), np.nan)
    for di in range(len(t.demes)):
        if t.gland_id[di] >= 0:
            r, c = t.deme_coords[di]
            grid[r, c] = t.gland_id[di]
    return grid


def expanded_tissue_rgb(t, section_frac=0.4, seed=0):
    """Cell-resolution tissue image (each deme a block of its individual cells, a ~40% section slice):
    green epithelial wall, pink stroma, blue immune, red cancer. Returns ``(rgb, s)``.

    This is the fast, pymuller-free analogue of ``tumor.plot_grid(expand_demes=True, color=["cell_type"])``
    with a fixed cancer colour: the engine's expand view builds a per-CLONE Muller colormap over EVERY
    cancer genotype (``viz.cell_type_colors`` → ``pymuller``), which is O(#genotypes) and hangs at the
    ~10^4 genotypes an infinite-sites tumour carries — even though the cancer here is drawn a single
    colour. We colour by coarse cell TYPE directly, so it is instant and returns only a small RGB array
    (letting the grid series keep a section image per snapshot instead of a full ~1 GB ``cell_data``)."""
    from collections import defaultdict
    cd = t.cell_data
    demes = cd["cell_deme"]["deme_id"].values
    crd = cd["cell_crd"].values
    ty = cell_types(t)
    rng = np.random.default_rng(seed)
    deme_cells, deme_rc = defaultdict(list), {}
    for i in range(len(ty)):
        d = int(demes[i]); deme_cells[d].append(i); deme_rc[d] = (int(crd[i, 0]), int(crd[i, 1]))
    max_sec = max((max(1, int(round(section_frac * len(v)))) for v in deme_cells.values()), default=1)
    s = max(1, int(np.ceil(np.sqrt(max_sec))))
    color = {"cancer": (0.84, 0.15, 0.16), "epithelial": (0.17, 0.55, 0.24),
             "stromal": (0.98, 0.80, 0.86), "immune": (0.40, 0.40, 0.90)}
    img = np.ones((t.grid_size * s, t.grid_size * s, 3))
    for d, cells in deme_cells.items():
        n_sec = min(s * s, int(round(section_frac * len(cells))))
        sample = rng.choice(cells, size=n_sec, replace=False) if (cells and n_sec) else []
        r, c = deme_rc[d]
        block = np.ones((s * s, 3))
        if len(sample):
            for p, ci in zip(rng.choice(s * s, size=len(sample), replace=False), sample):
                block[p] = color.get(ty[ci], (0.84, 0.15, 0.16))
        img[r * s:(r + 1) * s, c * s:(c + 1) * s] = block.reshape(s, s, 3)
    return img, s


def cancer_frac_grid(t):
    """(grid × grid) per-deme fraction of cells that are cancer, NaN where the deme is empty."""
    grid = np.full((t.grid_size, t.grid_size), np.nan)
    for di, deme in enumerate(t.demes):
        total = sum(deme.values())
        if total == 0:
            continue
        cancer = sum(c for g, c in deme.items() if t._is_cancer(g))
        r, c = t.deme_coords[di]
        grid[r, c] = cancer / total
    return grid


def cancer_gland_grid(t):
    """(grid × grid) per-deme gland id of demes that CONTAIN cancer (>=0 = confined DCIS focus,
    -1 = stroma-invaded IDC), NaN where no cancer — the multi-focal + breakout map."""
    grid = np.full((t.grid_size, t.grid_size), np.nan)
    for di, deme in enumerate(t.demes):
        if any(t._is_cancer(g) for g in deme):
            r, c = t.deme_coords[di]
            grid[r, c] = t.gland_id[di]
    return grid


def glands_colonised(t):
    """Set of gland indices that currently contain any cancer (multi-focal spread from one founder)."""
    return {int(t.gland_id[di]) for di, deme in enumerate(t.demes)
            if t.gland_id[di] >= 0 and any(t._is_cancer(g) for g in deme)}


def stroma_cancer_pct(t):
    """Percent of cancer cells that sit in the stroma (gland_id == -1) — the DCIS→IDC readout."""
    stroma = ingland = 0
    for di, deme in enumerate(t.demes):
        cc = sum(c for g, c in deme.items() if t._is_cancer(g))
        if t.gland_id[di] == -1:
            stroma += cc
        else:
            ingland += cc
    tot = stroma + ingland
    return 100.0 * stroma / tot if tot else 0.0


# ------------------------------------------------------------------ segment / assay helpers
def segment_offsets(t):
    """Per-segment start indices into the per-gene copy-number matrix (len n_segments+1)."""
    sizes = np.asarray(t.selection.segment_sizes)
    return np.concatenate([[0], np.cumsum(sizes)]).astype(int)


def segment_cn(t):
    """(cells × n_segments) TOTAL copy number (first gene of each segment) + per-gene->segment map."""
    sizes = np.asarray(t.selection.segment_sizes)
    offs = segment_offsets(t)
    cnv = t.cell_data["cell_cnv"].values
    seg = np.stack([cnv[:, offs[s]] for s in range(t.n_segments)], axis=1).astype(float)
    gene_seg = np.concatenate([np.full(sizes[s], s) for s in range(t.n_segments)]).astype(int)
    return seg, gene_seg


def thin_section(t, per_deme_cap=8, seed=0):
    """A depth-subsampled ``cell_data`` view for the 2D-section spatial assay (Visium).

    iscc's ``Visium`` currently pools ALL cells within a spot's radius (the section-slice sampling of
    ``DESIGN_ductal_field.md`` §3.1 is a pending ENGINE TODO, NOT built here), so with the moderate K a
    single spot over the duct over-fills far past a realistic per-spot count. As a stand-in we sample at
    most ``per_deme_cap`` cells per deme UNIFORMLY at random (uniform keeps the composition
    representative) and hand Visium that thinned section. This is a NOTEBOOK-side workaround, not an
    engine change. Returns a dict with the same keys as ``cell_data``, each frame reindexed to the kept
    cells."""
    rng = np.random.default_rng(seed)
    demes = t.cell_data["cell_deme"]["deme_id"].values
    ids = np.asarray(t.cell_data["cell_type"].index)
    from collections import defaultdict
    by_deme = defaultdict(list)
    for i, d in enumerate(demes):
        by_deme[int(d)].append(i)
    keep_pos = []
    for d, rows in by_deme.items():
        if len(rows) <= per_deme_cap:
            keep_pos.extend(rows)
        else:
            keep_pos.extend(int(r) for r in rng.choice(rows, size=per_deme_cap, replace=False))
    keep_pos = np.sort(np.asarray(keep_pos))
    keep_ids = ids[keep_pos]
    return {k: (v.loc[keep_ids] if hasattr(v, "loc") else v) for k, v in t.cell_data.items()}
