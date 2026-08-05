"""Shared spatial substrate for the iscc *science* showcase notebooks — the **ductal field**.

Every science notebook in ``notebooks/`` imports this helper so they all analyse the **same**
deterministic, spatially-structured tumour (and the same 5-tumour cohort) — now the SAME realistic,
cm-scale, breach-gated ductal field the tutorials ship (``notebooks/example_config.yaml`` via
``validation/realistic_regime.py``), not a smaller stand-in. Nothing is cached to disk; the tumour
re-grows from counts in a couple of minutes at cm-scale and only a ``max_cells`` subsample is ever
materialised, which keeps the notebooks self-contained and the ground truth exactly reproducible.

The substrate is a **multi-focal DCIS→IDC lesion on a ductal field** (``DESIGN_ductal_field.md`` +
``DESIGN_phenotype_plasticity.md`` §2): a FIELD of many small epithelial-ring glands scattered in
moderate-density stroma (a population-genetics *island model*), grown from ONE cancer founder. The
design choices baked in here (see the per-notebook narratives for why they matter):

* **Ductal-field substrate + microenvironment.** ``n_glands`` small epithelial rings in stroma
  (``stroma_fill_frac`` real stromal cells) — every notebook analyses a *mixture* (malignant +
  epithelial + stromal), never a pre-filtered cancer-only matrix. A single founder colonises the
  other glands via low-rate **cross-gland (island) dispersal** (``cross_gland_kappa``), so a section
  shows several *clonally related* foci.
* **Compartment-dependent selection (DCIS→IDC), breach-GATED.** Crossing the basement membrane
  (duct→stroma) REQUIRES the ``breach`` trait (``breach_gated_invasion``): a lumen founder is confined
  (DCIS) until a subclone evolves ``breach`` and invades the permissive stroma (IDC), where
  ``stromal_survival`` is then selected. This gate replaces the older leaky ``epithelial_barrier``
  death-hazard; a go-or-grow cost keeps each trait net-favoured only in its own compartment.
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

The config blocks are the realistic regime (``realistic_regime.py``) with a few science-notebook
overrides (WGD/CNV bumped for the copy-number demos; the ``EXPR()`` program layer on). The lesion
grows to a confluent cm-scale IDC-with-DCIS (tens of thousands of cancer cells) with the breach gate
ON — realistic scale AND structure, matching the shipped tutorials.
"""
import os
import sys

import numpy as np

# reuse the vetted expression_params() from the validation harness + the SHARED realistic regime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "validation"))
import programs_common as PC  # noqa: E402
import realistic_regime as RR  # noqa: E402  the one grid-170 breach-gated ductal-field regime

from iscc.tumor.models import GenotypeTumor  # noqa: E402

BASE_SEED = 3
COHORT_SEEDS = [3, 4, 5, 6, 7]  # 5 tumours sharing the landscape
NORMALS = ("epithelial", "stromal", "immune")

# The science notebooks now grow the SAME realistic, cm-scale, breach-gated DCIS→IDC ductal field the
# tutorials ship (``notebooks/example_config.yaml`` via ``validation/realistic_regime.py``) — grid 170,
# 8 ducts, K_duct 60 / K_stroma 30, breach-gated invasion — instead of the old grid-40 mini-field. The
# blocks below are that regime, with the few SCIENCE-notebook overrides these demos specifically need.
GENOME = dict(RR.GENOME)                       # 12 × 50 = 600 genes (same as the tutorials)
# Realistic breach-gated selection (met/treatment axes OFF; MET_SELECTION turns them on for the arc).
SELECTION = dict(RR.SELECTION)
# WGD turned UP to PCAWG-like: the tutorial config keeps WGD rare for a clean single-clone story, but
# the science notebooks (wgd_allele_cna especially) need frequent WGD + copy-number churn to have a
# signal, so bump wgd_rate and cnv_prob above the tutorial defaults.
CANCER = {**RR.CANCER, "wgd_rate": 0.05, "cnv_prob": 0.15}
DEME = dict(RR.DEME)
# The realistic cm-scale ductal field (grid 170, 8 breach-gated ducts, K_duct 60 / K_stroma 30). Same
# structure and scale as the tutorials, so every science notebook analyses a cm-scale IDC-with-DCIS
# section. ``cross_gland`` (intraductal spread) + the breach gate give the multi-focal → confluent
# invasion; a deme's ``K`` stands for its 3-D column (DESIGN_ductal_field.md §3).
SPATIAL = {**RR.SPATIAL, **RR.SCALES["cm"]}


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


def grow_base_tumor(seed=BASE_SEED, target_cancer=80000, cancer_params=None, expression=None,
                    selection_params=None, spatial_params=None, max_cells=RR.MAX_CELLS,
                    scale="cm", coarsen=RR.COARSEN, verbose=False):
    """Grow one realistic ductal-field tumour to >= ``target_cancer`` cancer cells, then materialise a
    ``max_cells`` representative subsample (a biopsy of the one tissue).

    Delegates to :func:`realistic_regime.grow_realistic` (the shared, materialisation-efficient grower)
    with the science-notebook overrides layered on: WGD/CNV bumped (``CANCER``) and the default
    ``EXPR()`` program layer. Never passes ``layout_seed`` -> defaults to ``DEFAULT_LAYOUT_SEED`` so
    every seed shares the SAME gene roles / program dictionary / gland-field layout / epistasis
    landscape (the cohort-comparability guarantee).

    Parameters
    ----------
    seed : int
        Growth seed (also the cohort member id).
    target_cancer : int
        Grow until at least this many cancer cells exist (~1–2 min at cm-scale).
    cancer_params, selection_params, spatial_params : dict, optional
        Overrides merged into the base ``CANCER`` / ``SELECTION`` / ``SPATIAL`` configs. e.g.
        ``cancer_params={"wgd_rate": 0.0}`` (WGD-off contrast, ``wgd_allele_cna``),
        ``spatial_params={"breach_gated_invasion": False}`` (the no-gate DCIS→IDC control,
        ``compartment_selection_confound``), or an ``epistasis_params`` network in ``selection_params``
        (``cohort_mhn_recurrence``).
    expression : dict, optional
        Overrides the default ``EXPR()`` expression params.
    max_cells : int, optional
        Cap on the materialised per-cell tables (the cm-scale tumour is far larger — sample it).
    scale : {"cm", "mid", "small"}, optional
        Field-size preset (:data:`realistic_regime.SCALES`). ``cm`` (default) is the full grid-170
        regime the tutorials ship.
    coarsen : bool, optional
        Passenger coarsening (default on — required at cm-scale). Turn OFF for lineage-resolved SNVs
        (``tree_inference_dna`` phylogenetics), which forces a smaller ``scale``/``target_cancer`` since
        every SNV then spawns a genotype. See :func:`realistic_regime.grow_realistic`.
    """
    return RR.grow_realistic(
        seed=seed, target_cancer=target_cancer, scale=scale, coarsen=coarsen,
        selection=selection_params,
        cancer={"wgd_rate": 0.05, "cnv_prob": 0.15, **(cancer_params or {})},
        spatial=spatial_params,
        expression=(expression if expression is not None else EXPR()),
        max_cells=max_cells, materialise=True, verbose=verbose)


def new_tumor(seed=BASE_SEED, cancer_params=None, selection_params=None, spatial_params=None,
              expression=None, max_cells=RR.MAX_CELLS):
    """An UNGROWN, cm-safe realistic ductal-field tumour — same config as :func:`grow_base_tumor` but
    NOT grown, for the notebooks that step it themselves to build a DCIS→IDC growth movie.

    Carries ``coarsen_passengers`` + ``max_cells`` so it stays tractable at grid 170 (without them a
    cm-scale tumour explodes the genotype count and materialises millions of cells). Call ``.grow(...)``
    yourself; each ``grow`` materialises a ``max_cells`` subsample. ``spatial_params={"breach_gated_
    invasion": False}`` gives the no-gate DCIS→IDC control (``compartment_selection_confound``)."""
    cfg = RR.config_for("cm", selection=selection_params,
                        cancer={"wgd_rate": 0.05, "cnv_prob": 0.15, **(cancer_params or {})},
                        spatial=spatial_params)
    return GenotypeTumor(seed=seed, update_mode="tau", tau=RR.TAU, coarsen_passengers=RR.COARSEN,
                         max_cells=max_cells,
                         expression_params=(expression if expression is not None else EXPR()), **cfg)


def grow_cohort(seeds=COHORT_SEEDS, target_cancer=80000, **kwargs):
    """Generator over ``(seed, tumour)`` — grow one at a time so RAM never holds 5 full tumours.

    Each cm-scale tumour is materialised only to a ``max_cells`` subsample, but the cohort notebooks
    should still assay each to a COMPACT matrix and drop the tumour before advancing (that is what
    makes this a generator).
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


# ==================================================================================================
# Metastasis module (R9) — OPT-IN. The 8 DCIS->IDC science notebooks use grow_base_tumor (no met) and
# are unchanged; grow_metastasis_tumor() adds a second (met) grid, the heritable met-establishment
# axis, and runs the full clinical arc:  grow -> seed -> resect -> chemo -> relapse.
# ==================================================================================================

# On top of the base DCIS->IDC config: the met deposit + establishment axis, and RARE + STRONG
# treatment resistance (one resistance mutation flips survival) so systemic chemo cleanly partitions
# susceptible vs resistant clones instead of leaving a graded, half-resistant smear.
MET_SELECTION = {"prop_met_survival": 0.06, "met_survival_effects": 2.5,
                 "prop_treatment_resistance": 0.005, "treatment_resistant_effects": 4.0}
# A LOWER mutation load for the arc (than the base >=10k config): resistance genes still exist in the
# layout, but a clone rarely ACQUIRES one, so the met is mostly SUSCEPTIBLE before chemo — chemo then
# crashes it to a rare resistant remnant that relapses (instead of a met that is already resistant).
# It also keeps the clone count legible for the Muller.
MET_CANCER = {"mutation_rate": 0.35, "n_snvs_per_allele": 0.15}
# The arc is a legibility demo, so it grows a SMALLER field than the >=10k base config (fewer/smaller
# glands, lower seeding rate for a cleaner founder read) — identical mechanism, just quicker.
MET_SPATIAL = {"grid_size": 24, "structure_radius": 3, "gland_radius": 3, "min_gland_sep": 7,
               "K_duct": 36, "K_stroma": 22,
               "met_grid_size": 11, "K_met": 16, "host_fill_frac": 0.35,
               "met_seed_kappa": 0.04, "met_hazard": 0.45, "met_transit_floor": 0.02}


def compartment_cancer(t):
    """(primary_cancer, met_cancer) live cancer-cell counts."""
    p = sum(c for i in range(t.n_primary_demes) for g, c in t.demes[i].items() if t._is_cancer(g))
    m = sum(c for i in range(t.n_primary_demes, len(t.demes)) for g, c in t.demes[i].items() if t._is_cancer(g))
    return p, m


def met_cancer(t, resistant=None):
    """Met cancer-cell count; ``resistant=True/False`` filters by treatment_resistance >= 0.5."""
    n = 0
    for i in range(t.n_primary_demes, len(t.demes)):
        for g, c in t.demes[i].items():
            if t._is_cancer(g):
                if resistant is None:
                    n += c
                elif (t.genotypes[g].evolutionary_parameters["treatment_resistance"] >= 0.5) == resistant:
                    n += c
    return n


def normal_totals(t):
    """Per-snapshot total NORMAL (host/epithelial/stromal/immune) cell counts (primary, met), read from
    the per-compartment traces — for the chemo-toxicity-on-normal-tissue plot."""
    from iscc.constants import normal_names
    prim, met = [], []
    for tr in t.traces:
        pc, mc = tr.get("primary_counts", tr["genotypes_counts"]), tr.get("met_counts", {})
        prim.append(sum(v for k, v in pc.items() if k in normal_names))
        met.append(sum(v for k, v in mc.items() if k in normal_names))
    return np.array(prim), np.array(met)


def grow_metastasis_tumor(seed=BASE_SEED, met_target=220, primary_cap=3200, max_chemo_gens=40,
                          chemo_toxicity=0.1, verbose=False):
    """Grow the DCIS->IDC ductal field WITH a metastatic deposit and run the full clinical arc:

        grow (DCIS->IDC)  ->  seed the met  ->  resect the primary  ->  systemic chemo (which also
        kills off-target normal tissue) run UNTIL the susceptible clones are eliminated  ->  relapse of
        the surviving resistant clones.

    Reuses the base GENOME / CANCER / DEME / SELECTION + the compartment axes, adding MET_SELECTION and
    MET_SPATIAL. Returns ``(tumour, marks)`` with ``marks`` a dict of snapshot indices
    {seeding, resection, chemo_start, chemo_end} for annotating the Muller plots."""
    from iscc.tumor.models import GenotypeTumor
    from iscc.treatment import Surgery, Chemotherapy
    spatial = {**SPATIAL, **MET_SPATIAL}
    selection = {**SELECTION, **MET_SELECTION}
    t = GenotypeTumor(seed=seed, genome_params=GENOME, selection_params=selection,
                      cancer_cell_params={**CANCER, **MET_CANCER}, deme_params=DEME,
                      spatial_params=spatial, update_mode="tau", tau=1.0)
    marks = {}
    # per-deme spatial snapshots at each stage of the arc (the traces only hold COUNTS, so the spatial
    # composition has to be captured as we go) -> attached as t.arc_snapshots for plot_clone_grid_series.
    snaps = {}

    def _snap(label):
        snaps[label] = [dict(d) for d in t.demes]

    while True:
        p, m = compartment_cancer(t)
        if m >= met_target or p + m >= primary_cap:
            break
        t.grow(2, seed=seed)
        if "seeding" not in marks and met_cancer(t) > 0:
            marks["seeding"] = len(t.traces)
            _snap("seeding")
        if verbose:
            print(f"  grow: primary={p} met={m}", flush=True)
    marks["resection"] = len(t.traces)
    _snap("pre_resection")
    t.grow(6, seed=seed, treatment=Surgery(start=t.step, site="primary"))
    # systemic chemo run until the SUSCEPTIBLE met clones are gone (or a generation cap / cure)
    marks["chemo_start"] = len(t.traces)
    _snap("pre_chemo")
    chemo = Chemotherapy(start=t.step, duration=10 ** 6, kill_rate=1.6, effectiveness=0.98,
                         toxicity=chemo_toxicity, sites="both")
    for k in range(max_chemo_gens):
        if met_cancer(t) == 0:
            break                                             # cured
        if k > 2 and met_cancer(t, resistant=False) == 0:
            break                                             # susceptible eliminated
        t.grow(2, seed=seed, treatment=chemo)
        if verbose:
            print(f"  chemo {k}: sensitive={met_cancer(t, False)} resistant={met_cancer(t, True)}", flush=True)
    marks["chemo_end"] = len(t.traces)
    _snap("chemo_end")
    t.grow(14, seed=seed)                                     # relapse of the surviving resistant clones
    _snap("end")
    t.arc_snapshots = snaps
    t.make_cell_data()
    return t, marks
