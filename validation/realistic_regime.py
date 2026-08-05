"""Single source of the realistic, structured, cm-scale tumour regime for EVERY iscc benchmark and
science notebook — the grid-170 breach-gated DCIS→IDC ductal field of ``notebooks/example_config.yaml``.

Why this module exists
----------------------
The paper's thesis is *structure-misleads-inference* and the engine is advertised as reaching
"millions of cells" with a "structured microenvironment". But historically each ``validate_*.py`` and
each science notebook grew its OWN small ad-hoc sim (grid 13–50, thousands of cells) that predates the
scalability engine — so the evidence undersold the claims and no two benchmarks shared a tumour. This
module makes the shipped tutorials, the science notebooks and the paper benchmarks all grow from the
SAME realistic tumour (structure AND cm-scale), by tracing every config back to the ONE canonical YAML.

The cm-scale discipline
-----------------------
A grid-170 tumour is ~10^6-capacity / ~0.5M realised cells; materialising every cell across a dozen
dense ``(n_cells × n_genes)`` frames is tens of GB. So the rule — followed here and in the tutorials —
is **grow ONE big tumour, then SAMPLE a tractable piece per tool**:

* ``grow_realistic(...)`` grows the field and by default does **not** materialise ``cell_data`` (the
  memory-safe default); it advances only per-clone counts (tau-leaping), so growth cost scales with
  #clones × #demes, not #cells.
* ``resect(t)`` returns a :class:`iscc.sample.Resection` — cut the specimen with ``bisect`` then
  ``dissociate`` (a sequencing sample, full depth) or ``slice`` (a thin imaging/Visium section), each
  materialising only the sampled piece. ``dissociated(t, max_cells=…)`` /
  ``section(t, max_cells=…)`` are the one-call shortcuts most benchmarks want.
* ``t.make_cell_data(max_cells=N)`` is the non-spatial shortcut: a representative Binomial subsample of
  the whole tumour (spatial + clonal + cell-type proportions preserved) — a biopsy of one tissue.

See ``handoffs/realistic_sim_for_paper.md`` and ``DESIGN_scalability.md`` §8–9.
"""
import os

import numpy as np
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(REPO, "notebooks", "example_config.yaml")

# Coarse cell-type names that are NOT cancer (the mixed ductal-field microenvironment).
NORMALS = ("epithelial", "stromal", "immune", "host")


def load_config(path=CONFIG_PATH):
    """The canonical realistic config as a plain dict (the same YAML the tutorials load)."""
    with open(path) as f:
        return yaml.safe_load(f)


_CFG = load_config()
# The canonical parameter blocks — exposed so a benchmark can override JUST what its effect needs
# (e.g. ``cancer={"wgd_rate": 0.05}`` to turn WGD on) while inheriting the realistic structure/scale.
GENOME = dict(_CFG["genome_params"])              # 12 × 50 = 600 genes
SELECTION = dict(_CFG["selection_params"])         # breach-gated DCIS→IDC, go-or-grow costs
CANCER = dict(_CFG["cell_params"]["cancer"])
DEME = dict(_CFG["deme_params"])
SPATIAL = dict(_CFG["spatial_params"])             # grid 170, 8 ducts, K_duct 60 / K_stroma 30
TAU = _CFG.get("tau", 0.5)
MAX_CELLS = _CFG.get("max_cells", 50000)
COARSEN = bool(_CFG.get("coarsen_passengers", True))

# Scale presets. ``cm`` is the full grid-170 paper/tutorial regime. ``mid`` / ``small`` shrink the
# FIELD (fewer, closer ducts on a smaller grid) for cheap CI / interactive runs while keeping the
# breach-gated ductal biology IDENTICAL — same genes, same selection, same gate. Use ``cm`` for any
# figure that goes in the paper; use the smaller presets only for tests and quick iteration.
SCALES = {
    "cm": {"grid_size": 170, "n_glands": 8, "min_gland_sep": 14},
    "mid": {"grid_size": 96, "n_glands": 5, "min_gland_sep": 13},
    "small": {"grid_size": 48, "n_glands": 3, "min_gland_sep": 11},
}


def config_for(scale="cm", genome=None, selection=None, cancer=None, deme=None, spatial=None):
    """The five nested-dict parameter blocks for a realistic tumour at ``scale``, with overrides
    merged on top. Returned as a dict of dicts so callers can inspect or hand them to
    :class:`GenotypeTumor` directly."""
    if scale not in SCALES:
        raise ValueError(f"scale must be one of {sorted(SCALES)}, got {scale!r}")
    return {
        "genome_params": {**GENOME, **(genome or {})},
        "selection_params": {**SELECTION, **(selection or {})},
        "cancer_cell_params": {**CANCER, **(cancer or {})},
        "deme_params": {**DEME, **(deme or {})},
        "spatial_params": {**SPATIAL, **SCALES[scale], **(spatial or {})},
    }


def grow_realistic(seed=2, target_cancer=150_000, scale="cm", genome=None, selection=None,
                   cancer=None, deme=None, spatial=None, expression=None, max_cells=MAX_CELLS,
                   materialise=False, tau=TAU, coarsen=COARSEN, verbose=False):
    """Grow one realistic breach-gated ductal-field tumour to ``target_cancer`` cancer cells.

    Memory-safe by default: ``materialise=False`` leaves ``cell_data`` empty (the tumour is per-clone
    counts). Sample it with :func:`resect` / :func:`dissociated` / :func:`section`, or pass
    ``materialise=True`` to get a ``max_cells``-bounded representative subsample of the whole tumour.

    ``GenotypeTumor.grow`` force-materialises ``cell_data`` at the end of EVERY call, so the growth
    loop here keeps the cap at 1 cell (a near-free materialisation) while stepping and only expands the
    real ``max_cells`` sample once, at the end — otherwise a cm-scale run would rebuild a 50k-cell
    table on each of ~20 iterations.

    Parameters
    ----------
    seed : int
        Growth seed (tutorials use 2).
    target_cancer : int
        Grow until at least this many cancer cells exist. The grid-170 field reaches ~190k in ~20
        tau-generations; a smaller target (or scale) is much faster for iteration.
    scale : {"cm", "mid", "small"}
        Field size preset (see :data:`SCALES`). ``cm`` for paper figures.
    genome, selection, cancer, deme, spatial : dict, optional
        Overrides merged onto the canonical blocks (e.g. ``cancer={"wgd_rate": 0.05}`` for WGD on,
        ``selection={"epistasis_params": net}`` for a cohort landscape).
    expression : dict, optional
        Gene-program / dosage expression params (off by default — the tutorials don't need them; the
        program notebooks pass one).
    max_cells : int or None
        Assay-memory cap used when ``materialise=True`` (defaults to the config's 50k).
    """
    cfg = config_for(scale, genome, selection, cancer, deme, spatial)
    from iscc.tumor.models import GenotypeTumor
    # Cap at 1 during the growth loop so grow()'s forced make_cell_data stays ~free; the real cap is
    # restored for the single materialisation at the end.
    t = GenotypeTumor(seed=seed, expression_params=expression, update_mode="tau", tau=tau,
                      coarsen_passengers=coarsen, max_cells=1, **cfg)
    # Adaptive stepping: coarse leaps while far from the target, finer near it, so an invasive mass
    # (which grows fast once it breaches) lands close to `target_cancer` without a huge overshoot. Only
    # the last ~10% single-steps — at cm-scale a single grid-170 step over thousands of clones is the
    # expensive unit, and a small overshoot is harmless (``max_cells`` bounds the assay regardless).
    while t.get_cancer_size() < target_cancer:
        ratio = t.get_cancer_size() / target_cancer
        t.grow(n_steps=(8 if ratio < 0.15 else 3 if ratio < 0.6 else 2 if ratio < 0.9 else 1), seed=seed)
        if verbose:
            print(f"  seed={seed}: {t.get_cancer_size()} cancer cells", flush=True)
    t.max_cells = max_cells
    if materialise:
        t.make_cell_data(max_cells=max_cells)
    else:
        t.cell_data = None  # counts only — sample via resect()/dissociated()/section()
    return t


def resect(t, compartment="primary"):
    """A :class:`iscc.sample.Resection` over ``t`` — the memory-safe way to cut a cm-scale specimen
    into samples (``bisect`` → ``dissociate`` / ``slice``)."""
    from iscc.sample import Resection
    return Resection(t, compartment=compartment)


def dissociated(t, max_cells=MAX_CELLS, frac=1.0, axis="x", compartment="primary", seed=0):
    """A dissociated sequencing sample (full depth) as a ``cell_data`` dict, bounded to ``max_cells``.

    ``frac < 1`` first bisects the specimen and dissociates only that in-plane piece (a regional
    sample); ``frac == 1`` dissociates the whole specimen. This is the tractable input for the
    dissociated assays / tools (scDNA, scRNA, clonealign, inferCNV, Numbat, …)."""
    spec = resect(t, compartment=compartment)
    region = None
    if frac < 1.0:
        region, _ = spec.bisect(frac=frac, axis=axis)
    return spec.dissociate(region=region, max_cells=max_cells)


def section(t, depth_frac=0.5, max_cells=MAX_CELLS, frac=1.0, axis="x", compartment="primary"):
    """A thin 2-D imaging/Visium SECTION as a ``cell_data`` dict (structure kept, each deme-column
    thinned to ~``depth_frac`` of its ``K``), bounded to ``max_cells`` — the tractable input for the
    spatial tools (Visium deconvolution, niche/spatial diagnostics)."""
    spec = resect(t, compartment=compartment)
    region = None
    if frac < 1.0:
        region, _ = spec.bisect(frac=frac, axis=axis)
    return spec.slice(region=region, depth_frac=depth_frac, max_cells=max_cells)


# ------------------------------------------------------------------ shared readouts (count-space)
def n_cancer(t):
    """Live cancer-cell count across the per-deme genotype counts (no materialisation needed)."""
    return t.get_cancer_size()


def stroma_cancer_pct(t):
    """Percent of cancer cells sitting in the stroma (``gland_id == -1``) — the DCIS→IDC invasion
    readout, straight from the counts."""
    stroma = ingland = 0
    for di, deme in enumerate(t.demes):
        cc = sum(c for g, c in deme.items() if t._is_cancer(g))
        if t.gland_id[di] == -1:
            stroma += cc
        else:
            ingland += cc
    tot = stroma + ingland
    return 100.0 * stroma / tot if tot else 0.0


def glands_colonised(t):
    """Set of gland indices that currently contain any cancer (multi-focal spread from one founder)."""
    return {int(t.gland_id[di]) for di, deme in enumerate(t.demes)
            if t.gland_id[di] >= 0 and any(t._is_cancer(g) for g in deme)}
