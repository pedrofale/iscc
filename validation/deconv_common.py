"""Shared data generation for the spatial-deconvolution benchmark (cell2location + RCTD) — FLAGSHIP.

`iscc` is (as far as we know) the only simulator that emits BOTH a Visium section AND an scRNA
reference FROM THE SAME TUMOUR, with the TRUE per-spot cell-type/clone composition of the section
*and* the true composition of the reference sample. Real spatial-deconvolution benchmarks must borrow
a reference from a different piece of tissue (or another patient) and can never measure the resulting
reference mismatch, let alone separate its causes. `iscc` can dial each cause independently and knows
the truth for both ends.

WHAT IS GENERATED (mirroring how deconvolution is actually done)
    1. ONE tumour grown with the F8 microenvironment and the R13 program layer ON, so the four cell
       types (cancer / epithelial / stromal / immune) have DISTINCT, co-expressed marker programmes
       (a deconvolution reference is only informative if the types differ transcriptionally — R13 is
       what supplies that, non-circularly, from program combinations rather than hand-drawn markers).
    2. A **Visium section**: the real `Visium` assay over a window centred on the tumour, with the
       true per-spot composition recomputed from `spot_members` under any labelling (coarse type or
       CNA clone).
    3. An **scRNA reference** taken from a SEPARATE biopsy (F1) of the SAME tumour, optionally pushed
       through **dissociation** (F2) and emitted with a chosen **assay/protocol** (F3). The reference
       is always same-tumour but a different sample — exactly the realistic case.

THE HEADLINE — what an imperfect reference costs, split into its three real, separable causes:
    * **regional mismatch** — the reference biopsy is from a different region, so its clone / niche /
      cell-type composition differs from the section's (`reference_mode="regional"`, swept by offset);
    * **dissociation bias** (F2) — the reference passes through dissociation (which the section does
      not), distorting cell-type composition (`dissociation_strength`);
    * **assay / batch** (F3) — reference and section are different technologies / batches
      (`reference_protocol`, `section vs reference batch seeds`).
An **oracle** reference built from the EXACT cells in the section (no regional mismatch, no
dissociation, matched assay) is the ceiling; degrading toward realism and attributing the error to
each source is the deliverable — not a raw accuracy number.

Each external tool runs in its OWN dedicated conda env (`iscc-cell2location`, `iscc-rctd`) per
`README_integration.md`; this module stays in the core `iscc` env and never imports them. Data crosses
the env boundary as files (AnnData / CSV) via thin runner scripts (`cell2location_runner.py`,
`rctd_runner.R`) shelled out with `subprocess`.
"""
import os
import subprocess

import numpy as np
import pandas as pd

import integration_common as C   # segment_cn / define_clones / cell_types live here

REPO = C.REPO
HOME = C.HOME

# Dedicated conda envs holding the external tools (kept out of the core `iscc` env).
CELL2LOCATION_PYTHON = os.environ.get(
    "ISCC_CELL2LOCATION_PYTHON", os.path.join(HOME, "miniconda3/envs/iscc-cell2location/bin/python"))
RCTD_RSCRIPT = os.environ.get(
    "ISCC_RCTD_RSCRIPT", os.path.join(HOME, "miniconda3/envs/iscc-rctd/bin/Rscript"))

CELLTYPES = ("cancer", "epithelial", "stromal", "immune")


def cell2location_available():
    return os.path.exists(CELL2LOCATION_PYTHON)


def rctd_available():
    return os.path.exists(RCTD_RSCRIPT)


# --------------------------------------------------------------------------------------------------
# The shared tumour: a solid lesion (cancer) inside an epithelial gland, embedded in stroma, with a
# sparse immune infiltrate seeded in every deme — so a tissue section is a genuine mixture of the four
# types and a biopsy from a different region has a different mixture (the regional mismatch). The
# program layer is ON with a per-cell-type program bias, giving each type a distinct co-expressed
# marker set; allele-specific dosage is on so the clone axis carries an (emergent) expression signal.
# --------------------------------------------------------------------------------------------------
GENOME = {"n_segments": 10, "segment_size": 30}                       # 300 genes over 10 segments
SELECTION = {"prop_driver": 0.25, "prop_dispersal": 0.05, "prop_immune_resistance": 0.15,
             "prop_treatment_resistance": 0.0}
DEME = {"carrying_capacity": 10, "initial_cancer_cells": 8}
SPATIAL = {"grid_size": 26, "structure_radius": 10, "immune_density": 0.12}
CANCER = {"division_rate": 0.85, "death_rate": 0.03, "max_birth_rate": 0.98,
          "mutation_rate": 1.0, "dispersal_rate": 0.2}

N_PROGRAMS = 6
# Each coarse cell type is biased toward its own program, so the reference carries distinct,
# CO-EXPRESSED marker sets (what a deconvolution reference needs). Programs are functional gene sets
# scattered across the genome (`program_genomic_scatter=1`), ORTHOGONAL to the contiguous CNA segments
# — which is exactly why a CNA clone is NOT a cell type (the clone-vs-cell-type confound below).
# loading_strength / loading_sparsity kept MODEST on purpose: a heavy-tailed loading (high sparsity)
# combined with a strong per-type bias makes exp(z·loading) blow one marker gene up to ~90% of all
# counts (an unrealistic pileup that also starves every other gene). These values give each type a
# distinct, co-expressed marker set (pairwise signature correlation ~0.6) with no single gene above
# ~10% of the library — realistic scRNA/Visium expression.
PROGRAM_PARAMS = {"n_programs": N_PROGRAMS, "n_genes_per_program": 20, "program_overlap": 0.1,
                  "loading_strength": {"mean": 0.5, "sd": 0.2}, "loading_sparsity": 0.5,
                  "program_genomic_scatter": 1.0}
_B = 1.2
CELLTYPE_PROGRAM_BIAS = {"cancer":     [_B, 0.0, 0.0, 0.0, 0.0, 0.0],
                         "epithelial": [0.0, 0.0, 0.0, 0.0, _B, 0.0],
                         "stromal":    [0.0, _B, 0.0, 0.0, 0.0, 0.0],
                         "immune":     [0.0, 0.0, 0.0, _B, 0.0, 0.0]}
ACTIVITY_PARAMS = {"n_active_programs_per_cell": 3, "activity_dist": "lognormal",
                   "activity_mean": 1.0, "activity_sd": 0.5, "activity_noise": 0.2,
                   "celltype_program_bias": CELLTYPE_PROGRAM_BIAS}
COUPLING_PARAMS = {"phenotype_program_strength": 0.5}


def expression_params(allele_specific=True):
    return {"program_params": dict(PROGRAM_PARAMS),
            "activity_params": dict(ACTIVITY_PARAMS),
            "coupling_params": dict(COUPLING_PARAMS),
            "dosage_params": {"dosage_sensitivity_mean": 0.7, "dosage_sensitivity_sd": 0.25,
                              "dosage_saturation": 8, "allele_specific": bool(allele_specific)}}


def grow_tumor(seed=3, steps=2500, dispersal_rate=None, cnv_prob=None, amp_prob=None,
               immune_density=None, genome=None, spatial=None, deme=None):
    """Grow the shared four-type spatial tumour with the R13 program layer on."""
    from iscc.tumor.models import GenotypeTumor
    cancer = dict(CANCER)
    if dispersal_rate is not None:
        cancer["dispersal_rate"] = dispersal_rate
    if cnv_prob is not None:
        cancer["cnv_prob"] = cnv_prob
        cancer["snv_prob"] = 1.0 - cnv_prob
    if amp_prob is not None:
        cancer["amp_prob"] = amp_prob
    spatial = dict(spatial or SPATIAL)
    if immune_density is not None:
        spatial["immune_density"] = immune_density
    t = GenotypeTumor(seed=seed, genome_params=genome or GENOME, selection_params=SELECTION,
                      cancer_cell_params=cancer, deme_params=deme or DEME, spatial_params=spatial,
                      expression_params=expression_params(), microenv_params={"hypoxia_strength": 1.0})
    t.grow(n_steps=steps, seed=seed)
    t.make_cell_data()
    return t


# --------------------------------------------------------------------------------------------------
# Per-cell labels: coarse type (the deconvolution target) and CNA clone (the confound axis).
# --------------------------------------------------------------------------------------------------
def coarse_types(tumor):
    """Per-cell coarse type ('cancer'/'epithelial'/'stromal'/'immune') as a pandas Series."""
    idx = tumor.cell_data["cell_type"].index
    return pd.Series(C.cell_types(tumor), index=idx, name="cell_type")


def clone_labels(tumor, n_clones=4):
    """Per-CANCER-cell CNA clone id (0..K-1) as a Series over cancer cells, plus the clone consensus."""
    types = C.cell_types(tumor)
    seg, _ = C.segment_cn(tumor)
    cancer = types == "cancer"
    idx = np.asarray(tumor.cell_data["cell_type"].index)
    labels, consensus = C.define_clones(seg[cancer], n_clones=n_clones)
    return pd.Series(labels, index=idx[cancer], name="clone"), consensus


def population_labels(tumor, n_clones=4):
    """Per-cell 'population' label used for the clone-vs-cell-type confound: normal cells keep their
    coarse type, cancer cells are split into their CNA clones ('clone0'…). A deconvolution reference
    built on these labels asks the tool to separate program-defined cell types from CNA-defined clones
    — only the former are a program combination, so the latter should be much harder."""
    types = coarse_types(tumor)
    clones, _ = clone_labels(tumor, n_clones=n_clones)
    lab = types.copy()
    lab.loc[clones.index] = ["clone%d" % c for c in clones.values]
    return lab.rename("population")


def cancer_centroid(tumor):
    """(row, col) centroid of the cancer cells — where a tumour-focused section is centred."""
    types = C.cell_types(tumor)
    crd = tumor.cell_data["cell_crd"][["row", "col"]].values.astype(float)
    cancer = types == "cancer"
    return crd[cancer].mean(0) if cancer.sum() else crd.mean(0)


# --------------------------------------------------------------------------------------------------
# The Visium section + its true per-spot composition.
# --------------------------------------------------------------------------------------------------
def _spot_composition(members, label_by, categories):
    """True per-spot proportions over `categories` from each spot's member cell ids."""
    K = len(categories)
    cat_idx = {c: i for i, c in enumerate(categories)}
    comp = np.zeros((len(members), K))
    for s, m in enumerate(members):
        if len(m) == 0:
            continue
        for c in m:
            lab = label_by.get(c)
            j = cat_idx.get(lab)
            if j is not None:
                comp[s, j] += 1.0
    tot = comp.sum(1, keepdims=True)
    return np.divide(comp, tot, out=np.zeros_like(comp), where=tot > 0)


def build_section(tumor, spot_radius=0.9, spot_pitch=1.5, section_radius=9.0, mu_counts=3000,
                  seed=7, n_clones=4, min_cells=1):
    """Realize the Visium section and its ground-truth per-spot composition.

    Returns a dict: ``spot_counts`` (spots x genes DataFrame), ``coords`` (spots x 2), ``members``
    (member cell ids per spot), ``n_cells`` (per spot), ``true_type`` (spots x 4 type proportions),
    ``true_clone`` (spots x population proportions incl. clones), ``type_categories``,
    ``clone_categories``, and the raw ``Visium`` object.
    """
    from iscc.data import Visium
    cd = tumor.cell_data
    gs = int(tumor.grid_size)
    vis = Visium(seed=seed, spot_pitch=spot_pitch, spot_radius=spot_radius,
                 mu_counts=mu_counts).run(cd, grid_side=gs)

    # crop to a tumour-centred window (a realistic tissue section spans the lesion + its margin)
    coords = vis.spot_coords
    if section_radius is not None:
        ctr = cancer_centroid(tumor)
        d = np.sqrt(((coords - ctr) ** 2).sum(1))
        keep = (d <= section_radius) & (vis.obs["n_cells"].values >= min_cells)
    else:
        keep = vis.obs["n_cells"].values >= min_cells
    keep = np.where(keep)[0]

    members = [vis.spot_members[s] for s in keep]
    type_by = coarse_types(tumor).to_dict()
    pop_by = population_labels(tumor, n_clones=n_clones).to_dict()
    type_cats = list(CELLTYPES)
    clone_cats = [c for c in CELLTYPES if c != "cancer"] + ["clone%d" % c for c in range(n_clones)]

    true_type = _spot_composition(members, type_by, type_cats)
    true_clone = _spot_composition(members, pop_by, clone_cats)

    spot_names = [vis.spot_names[s] for s in keep]
    spot_counts = vis.spot_counts.iloc[keep]
    return dict(spot_counts=spot_counts, coords=coords[keep], members=members,
                n_cells=vis.obs["n_cells"].values[keep], spot_names=spot_names,
                true_type=true_type, true_clone=true_clone,
                type_categories=type_cats, clone_categories=clone_cats, visium=vis)


# --------------------------------------------------------------------------------------------------
# The scRNA reference (oracle / regional, +/- dissociation, chosen assay), and its true composition.
# --------------------------------------------------------------------------------------------------
def _reference_cells(tumor, section, mode, offset, radius, seed):
    """Pick the reference cells: the exact section cells (oracle) or an offset punch (regional)."""
    from iscc.sample import Biopsy
    if mode == "oracle":
        cells = sorted({c for m in section["members"] for c in m})
        return np.array(cells), np.zeros(len(cells))
    ctr = cancer_centroid(tumor)
    # move the biopsy centre `offset` grid units away from the section centre (toward the margin)
    direction = np.array([1.0, 1.0]) / np.sqrt(2.0)
    bctr = ctr + direction * float(offset)
    bx = Biopsy(tumor.cell_data, rng=np.random.default_rng(seed), grid_size=int(tumor.grid_size))
    idx, _, geom = bx.sample(biopsy_type="punch", center=tuple(bctr), radius=radius)
    return np.asarray(idx), None


def build_reference(tumor, section, mode="regional", offset=8.0, radius=6.0,
                    dissociation_strength=0.0, protocol="10x", n_cells=400, seed=11,
                    n_clones=4, label_by="type"):
    """Build an scRNA reference and its per-cell labels + measured composition.

    Parameters
    ----------
    mode : "oracle" | "regional"
        oracle = the exact section cells (the ceiling); regional = a punch offset from the section.
    offset : float
        Regional biopsy distance (grid units) from the section centre — the regional-mismatch knob.
    dissociation_strength : float in [0, 1]
        F2 dissociation bias strength; 0 = no dissociation. Scales the per-type UNDER-recovery of the
        fragile types (immune most, then stromal) relative to cancer/epithelial.
    protocol : "10x" | "smartseq3"
        F3 assay/batch for the reference (the section is always Visium) — the assay-mismatch knob.
    label_by : "type" | "population"
        Cell-type reference (coarse types) or the confound reference (normal types + CNA clones).

    Returns a dict: ``counts`` (cells x genes DataFrame), ``labels`` (per-cell label array),
    ``categories`` (label order matching the section's), ``composition`` (measured label fractions),
    ``n_input``/``n_recovered`` and the F2 ``composition_shift`` when dissociation is on.
    """
    from iscc.data import scRNA
    from iscc.sample.dissociation.dissociation import Dissociation

    ref_cells, _ = _reference_cells(tumor, section, mode, offset, radius, seed)

    # F2 dissociation: subset the reference cells by cell-type-dependent recovery. The default recovery
    # (cancer .9 / epi .8 / stroma .6 / immune .4) is interpolated toward 1.0 by (1-strength), so
    # strength=0 recovers everyone and strength=1 is the full default bias.
    diss_shift = None
    if dissociation_strength > 0:
        base = {"cancer": 0.9, "epithelial": 0.8, "stromal": 0.6, "immune": 0.4}
        recov = {k: 1.0 - dissociation_strength * (1.0 - v) for k, v in base.items()}
        sub_cd = {k: (v.loc[ref_cells] if hasattr(v, "loc") else v)
                  for k, v in tumor.cell_data.items()}
        diss = Dissociation(sub_cd, rng=np.random.default_rng(seed + 1), recovery_probs=recov)
        kept_idx, meta, _ = diss.run()
        ref_cells = np.asarray(kept_idx)
        diss_shift = meta["composition_shift"]

    ref_cells = list(ref_cells)
    if n_cells is not None and len(ref_cells) > n_cells:
        rng = np.random.default_rng(seed + 2)
        ref_cells = list(rng.choice(ref_cells, size=n_cells, replace=False))

    rna = scRNA(n_cells=len(ref_cells), protocol=protocol, seed=seed + 3).run(
        tumor.cell_data, cell_subset=ref_cells)
    counts = rna.observed_counts.astype(float)
    ref_cells = list(counts.index)

    if label_by == "population":
        lab_series = population_labels(tumor, n_clones=n_clones)
        categories = section["clone_categories"]
    else:
        lab_series = coarse_types(tumor)
        categories = section["type_categories"]
    labels = lab_series.reindex(ref_cells).astype(str).values

    present, cnt = np.unique(labels, return_counts=True)
    comp = {c: 0.0 for c in categories}
    for c, n in zip(present, cnt):
        if c in comp:
            comp[c] = float(n / cnt.sum())
    return dict(counts=counts, labels=labels, categories=list(categories), composition=comp,
                n_recovered=len(ref_cells), dissociation_shift=diss_shift, mode=mode, offset=offset)


# --------------------------------------------------------------------------------------------------
# Scoring: true vs inferred per-spot proportions.
# --------------------------------------------------------------------------------------------------
def _jsd(p, q, eps=1e-12):
    p = np.asarray(p, float) + eps
    q = np.asarray(q, float) + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    def _kl(a, b):
        return np.sum(a * np.log(a / b))
    return float(0.5 * _kl(p, m) + 0.5 * _kl(q, m))


def score_proportions(true_comp, inferred, categories, inferred_categories=None):
    """Score inferred per-spot proportions against the truth.

    ``inferred`` is (spots x K) aligned to ``inferred_categories`` (defaults to ``categories``); it is
    reindexed to ``categories`` (missing types -> 0, so a reference that lacks a type is penalised) and
    renormalised. Returns mean per-spot JSD / RMSE, the flattened Pearson r, per-type r and per-type
    recall (mean inferred where the type is truly the majority).
    """
    T = np.asarray(true_comp, float)
    P = np.asarray(inferred, float)
    inferred_categories = list(inferred_categories or categories)
    # align inferred columns to `categories`
    col = {c: i for i, c in enumerate(inferred_categories)}
    A = np.zeros((P.shape[0], len(categories)))
    for j, c in enumerate(categories):
        if c in col:
            A[:, j] = P[:, col[c]]
    s = A.sum(1, keepdims=True)
    A = np.divide(A, s, out=np.full_like(A, 1.0 / A.shape[1]), where=s > 0)

    n = min(T.shape[0], A.shape[0])
    T, A = T[:n], A[:n]
    jsd = float(np.mean([_jsd(T[i], A[i]) for i in range(n)]))
    rmse = float(np.sqrt(np.mean((T - A) ** 2)))
    flat_r = float(np.corrcoef(T.ravel(), A.ravel())[0, 1]) if T.std() > 0 and A.std() > 0 else float("nan")
    per_type_r, per_type_recall = {}, {}
    for j, c in enumerate(categories):
        if T[:, j].std() > 0 and A[:, j].std() > 0:
            per_type_r[c] = float(np.corrcoef(T[:, j], A[:, j])[0, 1])
        maj = T.argmax(1) == j
        if maj.sum():
            per_type_recall[c] = float(A[maj, j].mean())
    return dict(jsd=jsd, rmse=rmse, flat_r=flat_r, per_type_r=per_type_r,
                per_type_recall=per_type_recall,
                mean_type_r=float(np.mean(list(per_type_r.values()))) if per_type_r else float("nan"),
                n_spots=int(n))


# --------------------------------------------------------------------------------------------------
# Runners into the dedicated envs (data crosses as files).
# --------------------------------------------------------------------------------------------------
def _write_ref_and_spatial(ref, section, work_dir):
    import anndata as ad
    os.makedirs(work_dir, exist_ok=True)
    genes = list(ref["counts"].columns)
    ref_ad = ad.AnnData(X=ref["counts"].values.astype("float32"),
                        obs=pd.DataFrame({"cell_type": ref["labels"]}, index=ref["counts"].index),
                        var=pd.DataFrame(index=genes))
    sp = section["spot_counts"][genes]
    sp_ad = ad.AnnData(X=sp.values.astype("float32"),
                       obs=pd.DataFrame(index=sp.index), var=pd.DataFrame(index=genes))
    sp_ad.obsm["spatial"] = np.asarray(section["coords"], float)
    ref_path = os.path.join(work_dir, "ref.h5ad")
    sp_path = os.path.join(work_dir, "spatial.h5ad")
    ref_ad.write_h5ad(ref_path)
    sp_ad.write_h5ad(sp_path)
    return ref_path, sp_path, list(ref["categories"])


def run_cell2location(ref, section, work_dir, epochs=250, epochs_sp=2000, n_cells_per_spot=None,
                      seed=0):
    """Run the REAL cell2location (in `iscc-cell2location`) and return (props DataFrame, categories).

    props is spots x cell-types (proportions, rows sum to 1), row-aligned to the section spots.
    """
    if not cell2location_available():
        raise RuntimeError(f"cell2location env not found at {CELL2LOCATION_PYTHON}")
    ref_path, sp_path, cats = _write_ref_and_spatial(ref, section, work_dir)
    out_csv = os.path.join(work_dir, "c2l_props.csv")
    if n_cells_per_spot is None:
        n_cells_per_spot = float(np.mean(section["n_cells"]))
    runner = os.path.join(REPO, "validation", "cell2location_runner.py")
    subprocess.run([CELL2LOCATION_PYTHON, runner, ref_path, sp_path, out_csv,
                    str(int(epochs)), str(int(epochs_sp)), str(float(n_cells_per_spot)),
                    str(int(seed))], check=True, capture_output=True, text=True)
    props = pd.read_csv(out_csv, index_col=0)
    return props, list(props.columns)


def _write_csvs(ref, section, work_dir):
    """Write RCTD inputs as CSVs (R can't read h5ad without heavy deps): reference counts
    (genes x cells) + per-cell labels, spatial counts (genes x spots) + spot coords."""
    os.makedirs(work_dir, exist_ok=True)
    genes = list(ref["counts"].columns)
    ref_c = ref["counts"][genes].T                         # genes x cells (int counts)
    ref_c.round().astype(int).to_csv(os.path.join(work_dir, "ref_counts.csv"))
    pd.Series(ref["labels"], index=ref["counts"].index, name="cell_type").to_csv(
        os.path.join(work_dir, "ref_labels.csv"))
    sp = section["spot_counts"][genes].T                   # genes x spots
    sp.round().astype(int).to_csv(os.path.join(work_dir, "sp_counts.csv"))
    pd.DataFrame(np.asarray(section["coords"], float), index=section["spot_names"],
                 columns=["x", "y"]).to_csv(os.path.join(work_dir, "sp_coords.csv"))
    return work_dir


def run_rctd(ref, section, work_dir, mode="full", seed=0):
    """Run the REAL RCTD (spacexr, in `iscc-rctd`) and return (props DataFrame, categories)."""
    if not rctd_available():
        raise RuntimeError(f"RCTD env not found at {RCTD_RSCRIPT}")
    _write_csvs(ref, section, work_dir)
    out_csv = os.path.join(work_dir, "rctd_props.csv")
    runner = os.path.join(REPO, "validation", "rctd_runner.R")
    subprocess.run([RCTD_RSCRIPT, runner, work_dir, out_csv, mode, str(int(seed))],
                   check=True, capture_output=True, text=True)
    props = pd.read_csv(out_csv, index_col=0)
    return props, list(props.columns)
