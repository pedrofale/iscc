"""Shared data generation for the multi-modal integration benchmarks (clonealign + inferCNV).

Both demos run a REAL downstream integration method on iscc-simulated data and score it against
iscc's known ground truth (the scMultiSim/SISTEM "we provide the ground truth" convention; see the
PEtracer section of the manuscript). The point is *non-circularity*: the copy-number -> expression
coupling that clonealign and inferCNV both rely on is NOT hand-imposed here — it EMERGES from the
engine's per-allele dosage model (``CancerCell.get_exp``: a neutral gene's expression is
baseline*(1 + copy_number), plus selection) and from the microenvironment (F8). A bolt-on simulator
would have to impose exactly the dosage law these methods assume; iscc does not, so this is a fair
test — and iscc additionally carries the true per-cell clone label and per-gene copy number that no
real dataset provides.

This module grows ONE multi-clone tumour with distinct segmental CNAs and a normal (epithelial /
stromal) compartment, defines clones from the per-segment copy-number profile, and runs scDNA + scRNA
on the SAME cells so the DNA<->RNA link carries ground truth. The two validation scripts
(:mod:`validate_clonealign`, :mod:`validate_infercnv`) consume the products here and hand them to the
real tool living in its own dedicated conda env (``iscc-clonealign`` / ``iscc-infercnv``), keeping the
core ``iscc`` env free of the heavy R+TensorFlow / infercnvpy dependencies.
"""
import os
import subprocess

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOME = os.path.expanduser("~")

# Dedicated conda envs holding the external tools (kept out of the core `iscc` env).
CLONEALIGN_RSCRIPT = os.environ.get(
    "ISCC_CLONEALIGN_RSCRIPT", os.path.join(HOME, "miniconda3/envs/iscc-clonealign/bin/Rscript"))
INFERCNV_PYTHON = os.environ.get(
    "ISCC_INFERCNV_PYTHON", os.path.join(HOME, "miniconda3/envs/iscc-infercnv/bin/python"))


def clonealign_available():
    return os.path.exists(CLONEALIGN_RSCRIPT)


def infercnv_available():
    return os.path.exists(INFERCNV_PYTHON)

# A solid tumour: several segments so distinct segmental CNAs can accumulate, a modest normal
# compartment (structure_radius>0 seeds diploid epithelial/stromal cells — the inferCNV reference),
# and enough evolutionary time that a few subclones with distinct gains/losses reach appreciable size.
GENOME = {"n_segments": 12, "segment_size": 50}          # 600 genes across 12 "chromosomes"
SELECTION = {"prop_driver": 0.2, "prop_dispersal": 0.0, "prop_immune_resistance": 0.0,
             "prop_treatment_resistance": 0.0}
DEME = {"carrying_capacity": 8, "initial_cancer_cells": 4}
SPATIAL = {"grid_size": 20, "structure_radius": 5}
CANCER = {"division_rate": 0.6, "death_rate": 0.03, "max_birth_rate": 0.98,
          "mutation_rate": 1.1, "dispersal_rate": 0.5}


def grow_tumor(seed=3, steps=750, genome=None, spatial=None, cancer=None, deme=None):
    """Grow the shared multi-clone tumour (cancer + diploid normal compartment)."""
    from iscc.tumor.models import GenotypeTumor
    t = GenotypeTumor(seed=seed,
                      genome_params=genome or GENOME, selection_params=SELECTION,
                      cancer_cell_params=cancer or CANCER, deme_params=deme or DEME,
                      spatial_params=spatial or SPATIAL)
    t.grow(n_steps=steps, seed=seed)
    t.make_cell_data()
    return t


def cell_types(tumor):
    """Per-cell coarse type ('cancer' / 'epithelial' / 'stromal' / 'immune') aligned to cell_data."""
    gid = tumor.cell_data["cell_type"].iloc[:, 0].astype(str).values
    g = tumor.genotypes
    return np.array([g[x].type if x in g else "?" for x in gid])


def segment_cn(tumor):
    """Per-cell per-segment copy number (n_cells x n_segments): the CN is constant within a segment,
    so we take the first gene of each segment. Also returns the per-gene->segment index map."""
    n_seg = tumor.n_segments
    sizes = tumor.selection.segment_sizes
    offs = np.concatenate([[0], np.cumsum(sizes)]).astype(int)
    cnv = tumor.cell_data["cell_cnv"].values
    seg = np.stack([cnv[:, offs[s]] for s in range(n_seg)], axis=1).astype(float)
    gene_seg = np.concatenate([np.full(sizes[s], s) for s in range(n_seg)]).astype(int)
    return seg, gene_seg


def define_clones(seg_cn_cancer, n_clones=4, min_frac=0.05, random_state=0):
    """Group cancer cells into ``n_clones`` clones by their per-segment CN profile (the standard
    clonealign setup: clones + their integer CN profiles come from the DNA modality). Returns
    (labels, consensus) where ``consensus`` is the integer per-segment CN of each clone.

    Agglomerative clustering on the discrete segment-CN vectors recovers the dominant CN states; the
    consensus (rounded mean) is the clone's copy-number profile. Clones below ``min_frac`` of cells
    are merged into their nearest surviving clone (by CN L1 distance) so every clone is well-sampled.
    """
    from sklearn.cluster import AgglomerativeClustering
    n = len(seg_cn_cancer)
    k = min(n_clones, max(1, len(np.unique(seg_cn_cancer, axis=0))))
    lab = AgglomerativeClustering(n_clusters=k).fit_predict(seg_cn_cancer)
    consensus = np.stack([np.rint(seg_cn_cancer[lab == c].mean(0)) for c in range(k)])
    # merge tiny clones into nearest surviving clone
    counts = np.bincount(lab, minlength=k)
    keep = np.where(counts >= max(1, int(min_frac * n)))[0]
    if len(keep) < k:
        drop = [c for c in range(k) if c not in keep]
        for c in drop:
            d = np.abs(consensus[keep] - consensus[c]).sum(1)
            lab[lab == c] = keep[int(np.argmin(d))]
    # relabel 0..K-1 and recompute consensus
    uniq = sorted(set(lab.tolist()))
    remap = {old: i for i, old in enumerate(uniq)}
    lab = np.array([remap[x] for x in lab])
    consensus = np.stack([np.rint(seg_cn_cancer[lab == c].mean(0)) for c in range(len(uniq))])
    return lab, consensus.astype(int)


def informative_segments(consensus):
    """Segments whose integer CN differs across clones (where the dosage signal lives)."""
    return np.where(consensus.std(axis=0) > 0)[0]


# --------------------------------------------------------------------------------------------------
# clonealign inputs: scRNA counts (cells x genes) + clone CN gene profiles (genes x clones)
# --------------------------------------------------------------------------------------------------
def build_clonealign_inputs(tumor, n_clones=4, protocol="10x", seed=0, dna_breadth="wgs"):
    """Assemble the clonealign benchmark on the cancer cells of ``tumor``.

    Returns a dict with: ``Y`` (scRNA counts, cells x genes DataFrame), ``L`` (clone CN per gene,
    genes x clones DataFrame), ``labels`` (true clone per RNA cell), ``consensus`` (clones x segments
    int CN), ``gene_seg`` (per-gene segment index), ``dna_concordance`` (how well the scDNA-derived
    clone CN matches the true consensus — DNA is genuinely in the loop), ``clone_names``.
    """
    from iscc.data import scRNA

    types = cell_types(tumor)
    seg, gene_seg = segment_cn(tumor)
    cancer = types == "cancer"
    cancer_cells = list(np.asarray(tumor.cell_data["cell_exp"].index)[cancer])

    labels, consensus = define_clones(seg[cancer], n_clones=n_clones)
    K = consensus.shape[0]
    clone_names = [f"clone{c}" for c in range(K)]
    lab_by_cell = {c: labels[i] for i, c in enumerate(cancer_cells)}

    # scRNA on the cancer cells -> counts for clonealign (raw counts; clonealign models NB + size).
    rna = scRNA(n_cells=len(cancer_cells), protocol=protocol, seed=100 + seed).run(
        tumor.cell_data, cell_subset=cancer_cells)
    Y = rna.observed_counts.astype(float)
    genes = list(Y.columns)
    gseg = np.array([gene_seg[list(tumor.cell_data["cell_exp"].columns).index(g)] for g in genes])
    rna_cells = list(Y.index)
    labels_rna = np.array([lab_by_cell[c] for c in rna_cells])

    # clone CN gene profiles: broadcast the (integer) per-segment consensus to genes. In the standard
    # clonealign workflow the clone CN profiles are GIVEN (from a separate DNA clonal analysis); we
    # supply the integer states a clean DNA caller yields and separately confirm scDNA recovers them.
    L = pd.DataFrame(consensus[:, gseg].T, index=genes, columns=clone_names).astype(float)

    dna_concordance = _scdna_concordance(tumor, cancer_cells, lab_by_cell, consensus, gene_seg,
                                         breadth=dna_breadth, seed=200 + seed)

    return dict(Y=Y, L=L, labels=labels_rna, consensus=consensus, gene_seg=gseg,
                clone_names=clone_names, rna=rna, cancer_cells=cancer_cells,
                dna_concordance=dna_concordance)


def _scdna_concordance(tumor, cancer_cells, lab_by_cell, consensus, gene_seg, breadth="wgs", seed=0):
    """Run scDNA on the same cancer cells and check the per-clone CN it recovers matches the true
    consensus (so the clone CN profiles clonealign consumes are ones the DNA modality supports).
    Returns the fraction of (clone, segment) entries whose rounded scDNA CN equals the true state."""
    from iscc.data import scDNA
    n_seg = consensus.shape[1]
    dna = scDNA(n_cells=len(cancer_cells), breadth=breadth, seed=seed).run(
        tumor.cell_data, cell_subset=cancer_cells)
    seg_ids = np.array([int(g.split("_")[1]) for g in dna.genes])
    cov = dna.coverage.values.astype(float)
    seg_cov = np.stack([cov[:, seg_ids == s].mean(1) for s in range(n_seg)], axis=1)
    med = np.median(seg_cov, axis=1, keepdims=True)
    obs_cn = 2.0 * seg_cov / np.where(med > 0, med, 1.0)
    dna_lab = np.array([lab_by_cell[c] for c in dna.cells])
    K = consensus.shape[0]
    recovered = np.stack([np.rint(np.median(obs_cn[dna_lab == c], axis=0)) for c in range(K)])
    return float((recovered == consensus).mean())


def write_clonealign_inputs(inp, out_dir):
    """Write Y (cells x genes) and L (genes x clones) as CSVs for the R clonealign runner."""
    os.makedirs(out_dir, exist_ok=True)
    inp["Y"].to_csv(os.path.join(out_dir, "Y.csv"))
    inp["L"].to_csv(os.path.join(out_dir, "L.csv"))
    pd.Series(inp["labels"], index=inp["Y"].index, name="true_clone").to_csv(
        os.path.join(out_dir, "truth.csv"))
    return out_dir


def run_clonealign(Y, L, work_dir, max_iter=200, n_repeats=3, seed=1):
    """Run the REAL clonealign (in the ``iscc-clonealign`` env) on counts ``Y`` (cells x genes) and
    copy number ``L`` (genes x clones). Returns the (cells x clones) assignment-probability DataFrame.

    Shells out to :mod:`clonealign_runner` (R) so the heavy R+TensorFlow stack stays in its own env.
    """
    if not clonealign_available():
        raise RuntimeError(f"clonealign env not found at {CLONEALIGN_RSCRIPT}")
    in_dir = os.path.join(work_dir, "in")
    out_dir = os.path.join(work_dir, "out")
    os.makedirs(in_dir, exist_ok=True)
    Y.to_csv(os.path.join(in_dir, "Y.csv"))
    L.to_csv(os.path.join(in_dir, "L.csv"))
    runner = os.path.join(REPO, "validation", "clonealign_runner.R")
    subprocess.run([CLONEALIGN_RSCRIPT, runner, in_dir, out_dir,
                    str(int(max_iter)), str(int(n_repeats)), str(int(seed))],
                   check=True, capture_output=True, text=True)
    probs = pd.read_csv(os.path.join(out_dir, "clone_probs.csv"), index_col=0)
    return probs.reindex(Y.index)


def score_assignment(true_labels, probs):
    """Score a clone assignment (probabilities, cells x clones) against integer ``true_labels``.

    Matches predicted clone columns to true clone ids by the best (Hungarian) permutation, then
    reports accuracy, chance/majority baselines, ARI, and one-vs-rest AUC per clone (and its mean).
    """
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import roc_auc_score, adjusted_rand_score

    P = np.asarray(probs.values, dtype=float)
    true = np.asarray(true_labels, dtype=int)
    K = P.shape[1]
    pred = P.argmax(1)
    conf = np.array([[((pred == a) & (true == b)).sum() for b in range(K)] for a in range(K)])
    row, col = linear_sum_assignment(-conf)
    pred_to_true = {row[i]: col[i] for i in range(len(row))}
    true_to_pred = {v: k for k, v in pred_to_true.items()}
    pred_matched = np.array([pred_to_true[p] for p in pred])
    acc = float((pred_matched == true).mean())
    per_auc = {}
    for b in range(K):
        y = (true == b).astype(int)
        if 0 < y.sum() < len(y) and b in true_to_pred:
            per_auc[b] = float(roc_auc_score(y, P[:, true_to_pred[b]]))
    return dict(
        accuracy=acc,
        chance=1.0 / K,
        majority=float(np.bincount(true).max() / len(true)),
        ari=float(adjusted_rand_score(true, pred)),
        per_clone_auc=per_auc,
        mean_auc=float(np.mean(list(per_auc.values()))) if per_auc else float("nan"),
        pred_matched=pred_matched,
        n_clones=K,
    )


# --------------------------------------------------------------------------------------------------
# inferCNV inputs: an AnnData of malignant + normal cells with genomic ordering + true CN
# --------------------------------------------------------------------------------------------------
def build_infercnv_inputs(tumor, n_cancer=None, n_normal=200, protocol="10x", seed=0):
    """Assemble an AnnData for infercnvpy: malignant (cancer) + normal (epithelial/stromal) cells,
    with per-gene genomic coordinates (segment -> chromosome, position within segment) and the true
    per-cell per-segment CN carried in ``obs``/``uns`` for scoring.
    """
    from iscc.data import scRNA

    types = cell_types(tumor)
    seg, gene_seg = segment_cn(tumor)
    idx = np.asarray(tumor.cell_data["cell_exp"].index)
    rng = np.random.default_rng(seed)

    cancer_cells = list(idx[types == "cancer"])
    if n_cancer is not None and len(cancer_cells) > n_cancer:
        cancer_cells = list(rng.choice(cancer_cells, size=n_cancer, replace=False))
    normal_pool = list(idx[np.isin(types, ("epithelial", "stromal"))])
    n_normal = min(n_normal, len(normal_pool))
    normal_cells = list(rng.choice(normal_pool, size=n_normal, replace=False)) if n_normal else []

    cells = cancer_cells + normal_cells
    rna = scRNA(n_cells=len(cells), protocol=protocol, seed=300 + seed).run(
        tumor.cell_data, cell_subset=cells)
    adata = rna.to_anndata()

    # cell-type label (malignant vs normal reference)
    tmap = {c: t for c, t in zip(idx, types)}
    adata.obs["cell_type"] = [("malignant" if tmap[c] == "cancer" else "normal") for c in adata.obs_names]
    adata.obs["fine_type"] = [tmap[c] for c in adata.obs_names]

    # genomic coordinates: segment -> chromosome, position within segment -> start (infercnvpy needs
    # chromosome + start + end in var, and genes ordered along the genome).
    genes = list(adata.var_names)
    gi = {g: i for i, g in enumerate(list(tumor.cell_data["cell_exp"].columns))}
    seg_of = np.array([gene_seg[gi[g]] for g in genes])
    pos_of = np.array([int(g.split("_")[2]) for g in genes])
    adata.var["chromosome"] = [f"chr{s}" for s in seg_of]
    adata.var["start"] = (pos_of * 100).astype(int)
    adata.var["end"] = (pos_of * 100 + 100).astype(int)

    # true per-cell per-segment CN (for scoring) -> obsm, plus segment order in uns
    seg_by_cell = pd.DataFrame(seg, index=idx, columns=[f"chr{s}" for s in range(tumor.n_segments)])
    adata.obsm["true_seg_cn"] = seg_by_cell.reindex(adata.obs_names).values.astype(float)
    adata.uns["segments"] = [f"chr{s}" for s in range(tumor.n_segments)]
    adata.uns["n_segments"] = int(tumor.n_segments)
    return adata


def run_infercnv(adata, work_dir, window_size=20, step=2):
    """Run the REAL infercnvpy (in the ``iscc-infercnv`` env) on ``adata`` and return the inferred
    per-segment CNV score (cells x segments), aggregating infercnvpy's genomic windows by chromosome.

    Returns a dict with ``seg_score`` (aligned to ``adata.uns['segments']``), ``x_cnv`` (raw windows),
    ``cell_type``, ``obs_names`` (order matches ``adata``), and ``segments``.
    """
    if not infercnv_available():
        raise RuntimeError(f"infercnv env not found at {INFERCNV_PYTHON}")
    os.makedirs(work_dir, exist_ok=True)
    in_h5ad = os.path.join(work_dir, "in.h5ad")
    out_npz = os.path.join(work_dir, "out.npz")
    adata.write_h5ad(in_h5ad)
    runner = os.path.join(REPO, "validation", "infercnv_runner.py")
    subprocess.run([INFERCNV_PYTHON, runner, in_h5ad, out_npz, str(window_size), str(step)],
                   check=True, capture_output=True, text=True)

    d = np.load(out_npz, allow_pickle=True)
    X = d["x_cnv"]
    chrom_names = [str(c) for c in d["chrom_names"]]
    starts = [int(s) for s in d["chrom_starts"]]
    order = np.argsort(starts)
    chrom_names = [chrom_names[i] for i in order]
    starts = [starts[i] for i in order]
    bounds = starts + [X.shape[1]]
    per_chrom = {chrom_names[i]: X[:, bounds[i]:bounds[i + 1]].mean(1) for i in range(len(chrom_names))}

    segments = list(adata.uns["segments"])
    seg_score = np.stack([per_chrom.get(s, np.zeros(X.shape[0])) for s in segments], axis=1)
    obs_names = [str(x) for x in d["obs_names"]]
    return dict(seg_score=seg_score, x_cnv=X, cell_type=np.array([str(c) for c in d["cell_type"]]),
                obs_names=obs_names, segments=segments)


def score_infercnv(res, true_seg_cn, clone_labels=None):
    """Score infercnvpy output against iscc's true per-cell per-segment copy number.

    ``res`` is :func:`run_infercnv`'s output (rows aligned to ``true_seg_cn``); ``clone_labels`` (per
    malignant cell) enables the denoised clone-level recovery. Returns per-segment single-cell
    correlations, the malignant-vs-normal AUC (by CNV magnitude), and — when clone labels are given —
    the clone x segment inferred/true matrices and their correlation on CN-varying segments.
    """
    from sklearn.metrics import roc_auc_score

    seg = res["seg_score"]
    mal = res["cell_type"] == "malignant"
    tr = np.asarray(true_seg_cn, dtype=float)
    n_seg = seg.shape[1]

    per_seg_r = {}
    for j in range(n_seg):
        t = tr[mal, j]
        if t.std() > 0:
            per_seg_r[j] = float(np.corrcoef(seg[mal, j], t)[0, 1])
    mag = np.abs(res["x_cnv"]).mean(1)
    mn_auc = (float(roc_auc_score(mal.astype(int), mag))
              if 0 < mal.sum() < len(mal) else float("nan"))

    out = dict(per_segment_r=per_seg_r,
               mean_segment_r=float(np.mean(list(per_seg_r.values()))) if per_seg_r else float("nan"),
               malignant_normal_auc=mn_auc, n_malignant=int(mal.sum()), n_normal=int((~mal).sum()))

    if clone_labels is not None:
        lab = np.asarray(clone_labels)
        K = int(lab.max()) + 1
        clone_inf = np.stack([seg[mal][lab == k].mean(0) for k in range(K)])
        clone_true = np.stack([tr[mal][lab == k].mean(0) for k in range(K)])
        info = np.where(clone_true.std(0) > 0.1)[0]
        # centre each segment (remove the shared tumour-vs-normal baseline offset) to expose the
        # clone-specific copy-number signal, then correlate inferred vs true across clones x segments.
        ci = clone_inf - clone_inf.mean(0, keepdims=True)
        ct = clone_true - clone_true.mean(0, keepdims=True)
        rc = (float(np.corrcoef(ci[:, info].ravel(), ct[:, info].ravel())[0, 1])
              if len(info) else float("nan"))
        out.update(clone_inferred=clone_inf, clone_true=clone_true, informative_segments=info,
                   clone_level_r=rc, n_clones=K)
    return out
