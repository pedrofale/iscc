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
NUMBAT_RSCRIPT = os.environ.get(
    "ISCC_NUMBAT_RSCRIPT", os.path.join(HOME, "miniconda3/envs/iscc-numbat/bin/Rscript"))


def numbat_available():
    return os.path.exists(NUMBAT_RSCRIPT)


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


# ---------------------------------------------------------------------------------------------
# SUBSTRATE: the toy rig above, or the REALISTIC ductal field the rest of iscc is calibrated to.
#
# The constants above are a TOY: grid 20x20, K=8 (<=3,200 cells), one epithelial ring, and 12 x 50 =
# 600 genes. `validation/realistic_regime.py` — the regime the notebooks (via notebooks/base_sim.py),
# the sweeps and the tests all use — is grid 48/96/170, a ductal FIELD of 3/5/8 glands, 12 x 500 =
# 6,000 genes. Every real-tool benchmark here ran on the toy one, so the paper's integration results
# were REAL TOOLS on UNREALISTIC DATA. 600 genes matters most for the gene x cell tools (scDEF, cNMF,
# inferCNV, cell2location), where real matrices carry ~20k.
#
# The switch is OPT-IN so the migration can go one benchmark at a time and nothing changes silently:
# pass regime="realistic" (or set ISCC_INTEGRATION_REGIME=realistic to flip a whole run). `scale` is
# the cost dial — "small" keeps the ductal field and the 6,000-gene genome while staying affordable;
# "cm" is the paper-scale field.
REGIME = os.environ.get("ISCC_INTEGRATION_REGIME", "toy")
# Cancer-cell targets per scale. "small" is deliberately close to the toy rig's ~3k cells, so the
# FIRST thing that changes is the substrate (field + gene count), not the sample size — otherwise a
# score shift cannot be attributed.
SCALE_TARGETS = {"small": 4_000, "mid": 20_000, "cm": 150_000}
# Clone-detection threshold. 5% was tuned on the toy rig; on the realistic ductal field it merges the
# smaller clones away (mid-scale: 3 clones at 5%, but 4 at 2% — sizes 994/254/211/68). 2% is not a
# fitted convenience: it is the SAME detection limit at which iscc's clonal-diversity index matches
# real multi-region phylogenies (median D 4.11 vs the empirical 3.96, hull coverage 5% -> 53%; see
# mode4_scratch/evomode_threshold_test.py). Two independent lines of evidence land on ~2%, which is
# also what multi-region bulk can actually resolve.
MIN_CLONE_FRAC = 0.02 if REGIME == "realistic" else 0.05


def grow_tumor(seed=3, steps=750, genome=None, spatial=None, cancer=None, deme=None,
               regime=None, scale="mid", target_cancer=None, max_cells=8_000):
    """Grow the shared multi-clone tumour (cancer + diploid normal compartment).

    ``regime="toy"`` (default) is the historical small rig. ``regime="realistic"`` grows the
    calibrated breach-gated ductal field instead, via ``realistic_regime.grow_realistic`` — same
    return contract (``cell_data`` materialised), so every helper below and every caller works
    unchanged. ``steps`` is ignored under the realistic regime, which grows to a cancer-cell TARGET
    rather than for a fixed number of steps.
    """
    regime = regime or REGIME
    if regime == "realistic":
        import realistic_regime as RR
        return RR.grow_realistic(
            seed=seed, scale=scale,
            target_cancer=target_cancer or SCALE_TARGETS[scale],
            genome=genome, cancer=cancer, deme=deme, spatial=spatial,
            materialise=True, max_cells=max_cells)
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


def segment_allele_cn(tumor):
    """Per-cell per-segment ALLELE-SPECIFIC copy number ``(p_cn, m_cn)`` from the genotype genome —
    the ground truth for the WGD allele-state benchmark. ``cell_cnv`` only carries TOTAL CN (p+m), so
    it cannot express allelic imbalance (a 4+0 segment reads as total 4, identical to a balanced 2+2);
    the per-homolog counts here can. Returns ``cell_id -> (n_seg, 2) int array`` (columns = p, m); a
    cell whose genotype has no materialised genome maps to ``None``.

    A segment ``(p, m)`` is *allelically imbalanced* when ``p != m`` (BAF != 0.5 — the signal the allele
    layer reads); it is *allele-only detectable* (copy-neutral-LOH-like) when it is imbalanced yet its
    total ``p+m`` is even, so total copy number alone cannot distinguish it from a balanced state. WGD
    is what populates these high-CN even-total imbalanced states (the doubling+loss signature).
    """
    gid = tumor.cell_data["cell_type"].iloc[:, 0].astype(str).values
    idx = np.asarray(tumor.cell_data["cell_type"].index)
    g = tumor.genotypes
    cache, out = {}, {}
    for cell, gg in zip(idx, gid):
        if gg not in cache:
            rep = g.get(gg)
            cache[gg] = (np.array([(len(s["p"]), len(s["m"])) for s in rep.genome], dtype=int)
                         if rep is not None and hasattr(rep, "genome") else None)
        out[cell] = cache[gg]
    return out


def define_clones(seg_cn_cancer, n_clones=4, min_frac=None, random_state=0):
    """Group cancer cells into ``n_clones`` clones by their per-segment CN profile (the standard
    clonealign setup: clones + their integer CN profiles come from the DNA modality). Returns
    (labels, consensus) where ``consensus`` is the integer per-segment CN of each clone.

    Agglomerative clustering on the discrete segment-CN vectors recovers the dominant CN states; the
    consensus (rounded mean) is the clone's copy-number profile. Clones below ``min_frac`` of cells
    are merged into their nearest surviving clone (by CN L1 distance) so every clone is well-sampled.

    ``min_frac=None`` (default) resolves to :data:`MIN_CLONE_FRAC`, which tracks the regime — 5% on
    the toy rig, 2% on the realistic field, where 5% merges away clones a real multi-region study
    would resolve.
    """
    if min_frac is None:
        min_frac = MIN_CLONE_FRAC
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
    # NOTE this reconstruction is only valid for a NEAR-DIPLOID tumour. `2 * cov / median(cov)`
    # assumes the median segment is CN 2, and scDNA coverage is per-cell library-size normalised, so
    # on the WGD+ realistic ductal field (clones at CN 4 across nearly every segment) it returns ~1
    # where the truth is 4 and concordance collapses to 0.00 — for a set of clone profiles that are
    # perfectly correct. A diploid NORMAL reference is not sufficient either: a tetraploid cell
    # spreads the same read budget over twice the genome, so per-copy coverage FALLS. Recovering
    # absolute CN here needs ploidy-aware renormalisation. Until then this diagnostic is meaningful
    # ONLY on the toy regime; it does not affect `L`, which is built from the TRUE per-segment CN.
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
    r = subprocess.run([CLONEALIGN_RSCRIPT, runner, in_dir, out_dir,
                        str(int(max_iter)), str(int(n_repeats)), str(int(seed))],
                       capture_output=True, text=True)
    if r.returncode != 0:
        # check=True raises CalledProcessError with R's stderr buried in an attribute nothing prints,
        # so a failure surfaces as a bare "non-zero exit status 1" and the real R message is lost.
        # Put it in the exception text.
        raise RuntimeError(
            f"clonealign (R) failed with exit {r.returncode}. Inputs: Y {Y.shape}, L {L.shape}.\n"
            f"--- R stderr ---\n{r.stderr[-2000:]}")
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


# ==================================================================================================
# Numbat (allele-aware CNA-from-expression) — DEMO 2
# --------------------------------------------------------------------------------------------------
# Numbat is the allele-aware successor to inferCNV: on top of the expression signal it reads B-allele
# frequencies (BAF) at phased SNP sites and a phylogeny prior. The clean question iscc can answer is
# whether the allele layer actually HELPS, and by how much, in a HEAD-TO-HEAD against expression-only
# inferCNV on the SAME cells.
#
# INPUT-INTERFACE SCOPING (the documented main risk). Numbat normally builds its allele dataframe from
# cellsnp-lite pileups on a BAM + a POPULATION PHASING PANEL (Eagle/1000G) — machinery an abstract
# genome has none of. We take the route the handoff prefers: feed Numbat the allele counts DIRECTLY.
# iscc tracks the two homologs (`p`/`m`) explicitly, so it already knows the true PHASE — the very thing
# a phasing panel only approximates. We therefore treat each gene as one phased heterozygous marker with
# GT = "1|0" (the `p` homolog is haplotype 0 everywhere, globally consistent by construction), the ALT
# (B-allele) count = reads from the `m` homolog, drawn at the gene's UMI depth from iscc's per-allele
# expression (`cell_rna_baf`), and iscc's segments mapped onto Numbat's chromosomes via a custom `gtf`.
# Bypassing pileup + phasing this way — "we fed allele counts directly because the population phasing
# panel does not apply to an abstract genome, and iscc's homolog labels ARE the ground-truth phasing" —
# is itself the honest answer to the scoping question, and worth a sentence in the paper.

# The Numbat tumour reuses the shared multi-clone GENOME/SELECTION/etc. above but is grown with the R13
# allele layer ON (allele_specific dosage) so `cell_rna_baf` exists. Programs stay OFF: this keeps the
# expression regime identical to the inferCNV benchmark (the head-to-head must be fair), while the
# allelic imbalance that Numbat exploits still emerges from the per-homolog dosage of the CNAs.
NUMBAT_EXPR = {"dosage_params": {"dosage_sensitivity_mean": 0.85, "dosage_sensitivity_sd": 0.15,
                                 "dosage_saturation": 8, "allele_specific": True}}


def grow_tumor_alleles(seed=3, steps=750, genome=None, spatial=None, cancer=None, deme=None):
    """Grow the shared multi-clone tumour with the allele-resolved expression layer ON (so
    ``cell_rna_baf`` / ``cell_exp_p`` / ``cell_exp_m`` are materialised for Numbat)."""
    from iscc.tumor.models import GenotypeTumor
    t = GenotypeTumor(seed=seed,
                      genome_params=genome or GENOME, selection_params=SELECTION,
                      cancer_cell_params=cancer or CANCER, deme_params=deme or DEME,
                      spatial_params=spatial or SPATIAL, expression_params=NUMBAT_EXPR)
    t.grow(n_steps=steps, seed=seed)
    t.make_cell_data()
    return t


def build_cna_inputs(tumor, n_cancer=None, n_normal=150, n_clones=4, protocol="10x", seed=0):
    """Shared substrate for the Numbat-vs-inferCNV head-to-head: pick malignant (cancer) + normal
    (epithelial/stromal) cells, emit ONE scRNA realisation, and return everything BOTH tools consume so
    they see identical counts. Returns a dict with:

      ``adata``      — infercnvpy-ready AnnData (genomic coords in ``var``, true per-cell per-segment CN
                       in ``obsm['true_seg_cn']``, malignant/normal label in ``obs``) — the inferCNV input;
      ``counts``     — cells x genes DataFrame (the emitted UMI counts);
      ``cell_ids``   — malignant + normal cell order (matches ``adata`` / ``counts``);
      ``is_malignant`` — bool per cell; ``clone_labels`` — per-malignant-cell clone id (0..K-1);
      ``true_seg_cn`` — cells x segments true CN; ``consensus`` — clones x segments int CN;
      ``gene_seg``   — per-gene segment index; ``n_segments``.
    """
    adata = build_infercnv_inputs(tumor, n_cancer=n_cancer, n_normal=n_normal, protocol=protocol,
                                  seed=seed)
    types = cell_types(tumor)
    seg, gene_seg = segment_cn(tumor)
    idx_all = np.asarray(tumor.cell_data["cell_type"].index)
    tmap = {c: t for c, t in zip(idx_all, types)}

    cell_ids = list(adata.obs_names)
    is_mal = np.array([tmap[c] == "cancer" for c in cell_ids])
    counts = pd.DataFrame(np.asarray(adata.X), index=cell_ids, columns=list(adata.var_names))

    # clones on the malignant cells (the same CN-profile clustering used elsewhere)
    seg_by_cell = pd.DataFrame(seg, index=idx_all, columns=[f"seg{s}" for s in range(tumor.n_segments)])
    mal_cells = [c for c in cell_ids if tmap[c] == "cancer"]
    labels, consensus = define_clones(seg_by_cell.loc[mal_cells].values, n_clones=n_clones)
    clone_by_cell = {c: int(l) for c, l in zip(mal_cells, labels)}

    return dict(adata=adata, counts=counts, cell_ids=cell_ids, is_malignant=is_mal,
                clone_labels=np.array([clone_by_cell[c] for c in mal_cells]), mal_cells=mal_cells,
                true_seg_cn=np.asarray(adata.obsm["true_seg_cn"]), consensus=consensus,
                gene_seg=gene_seg, n_segments=int(tumor.n_segments))


def build_numbat_inputs(tumor, shared, work_dir, depth_frac=1.0, seed=0):
    """Materialise Numbat's file inputs from the shared substrate (route (i): allele counts directly).

    Writes to ``work_dir``:
      ``count_mat.csv``  — genes x cells integer UMI matrix (Numbat's expression input);
      ``ref_counts.csv`` — genes x NORMAL cells (for the ``lambdas_ref`` expression reference);
      ``df_allele.csv``  — the phased allele dataframe (cell, snp_id, CHROM, POS, cM, REF, ALT, AD, DP,
                           GT, gene) with ALT = m-homolog counts drawn at UMI depth from ``cell_rna_baf``;
      ``gtf.csv``        — gene -> (CHROM = segment 1..S, gene_start, gene_end) custom annotation.
    Returns the paths + the per-cell truth needed to score.
    """
    os.makedirs(work_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    cell_ids = shared["cell_ids"]
    genes = list(shared["counts"].columns)
    counts = shared["counts"]                              # cells x genes
    gene_seg = shared["gene_seg"]
    seg_of = {g: int(gene_seg[i]) for i, g in enumerate(list(tumor.cell_data["cell_exp"].columns))}

    # per-gene genomic layout: segment -> CHROM (1..S), position within segment -> POS (bp), cM ~ POS.
    pos_in_seg = {g: int(g.split("_")[2]) for g in genes}
    chrom = np.array([seg_of[g] + 1 for g in genes])
    pos = np.array([pos_in_seg[g] * 1000 + 1 for g in genes])    # 1 kb gene spacing
    gtf = pd.DataFrame({"gene": genes, "gene_start": pos, "gene_end": pos + 500,
                        "CHROM": chrom, "gene_length": 500})   # numbat gtf schema
    gtf.to_csv(os.path.join(work_dir, "gtf.csv"), index=False)

    # count matrix (genes x cells) and the normal-cell reference
    cmat = counts.T.round().astype(int)                    # genes x cells
    cmat.to_csv(os.path.join(work_dir, "count_mat.csv"))
    normal_cells = [c for c, m in zip(cell_ids, shared["is_malignant"]) if not m]
    cmat[normal_cells].to_csv(os.path.join(work_dir, "ref_counts.csv"))

    # allele dataframe: ALT(B) = m-homolog fraction (1 - BAF); DP = UMI depth; AD ~ Binomial(DP, mfrac).
    baf = tumor.cell_data["cell_rna_baf"].loc[cell_ids, genes].values      # cells x genes (p fraction)
    mfrac = 1.0 - baf
    umi = counts.values                                    # cells x genes
    dp = np.rint(umi * float(depth_frac)).astype(int)
    ad = rng.binomial(np.maximum(dp, 0), np.clip(mfrac, 0.0, 1.0))
    ci, gi = np.where(dp > 0)                               # only covered (cell, gene) sites
    gpos = {g: (int(chrom[j]), int(pos[j])) for j, g in enumerate(genes)}
    gname = np.array(genes)
    rows = pd.DataFrame({
        "cell": np.array(cell_ids)[ci],
        "gene": gname[gi],
        "CHROM": chrom[gi],
        "POS": pos[gi],
        "AD": ad[ci, gi],
        "DP": dp[ci, gi],
    })
    rows["snp_id"] = rows["CHROM"].astype(str) + "_" + rows["POS"].astype(str)
    rows["cM"] = rows["POS"] / 1e6                          # ~1 cM / Mb; phase is exact so this only
    rows["REF"] = "A"                                       # sets the HMM's phase-switch length scale
    rows["ALT"] = "B"
    rows["GT"] = "1|0"                                      # p = hap0, m = hap1, globally consistent
    rows = rows[["cell", "snp_id", "CHROM", "POS", "cM", "REF", "ALT", "AD", "DP", "GT", "gene"]]
    rows.to_csv(os.path.join(work_dir, "df_allele.csv"), index=False)

    return dict(work_dir=work_dir, cell_ids=cell_ids, genes=genes,
                is_malignant=shared["is_malignant"], clone_labels=shared["clone_labels"],
                mal_cells=shared["mal_cells"], true_seg_cn=shared["true_seg_cn"],
                consensus=shared["consensus"], n_segments=shared["n_segments"])


def run_numbat(numbat_in, work_dir, min_cells=20, ncores=1, max_iter=1, t=1e-5, seed=0, min_llr=5.0):
    """Run the REAL Numbat (in ``iscc-numbat``) on the file inputs; return its per-cell outputs.

    Shells out to :mod:`numbat_runner` (R). Returns a dict with ``clone`` (per-cell clone id),
    ``p_aneuploid`` (P(cell is aneuploid) — the malignant score), ``seg_cn`` (cells x segments inferred
    total CN, aligned to ``numbat_in['cell_ids']`` order where present), and ``cells`` (row order).
    ``min_llr`` is Numbat's pseudobulk CNV-filtering threshold (default 5); the noisier WGD regime uses
    a lower value so real-but-weak CNVs are not all filtered out (its own "reduce min_LLR" advice).
    """
    if not numbat_available():
        raise RuntimeError(f"numbat env not found at {NUMBAT_RSCRIPT}")
    out_dir = os.path.join(work_dir, "numbat_out")
    os.makedirs(out_dir, exist_ok=True)
    runner = os.path.join(REPO, "validation", "numbat_runner.R")
    subprocess.run([NUMBAT_RSCRIPT, runner, numbat_in["work_dir"], out_dir,
                    str(int(min_cells)), str(int(ncores)), str(int(max_iter)),
                    str(float(t)), str(int(seed)), str(float(min_llr))],
                   check=True, capture_output=True, text=True)

    clone = pd.read_csv(os.path.join(out_dir, "clone.csv"))            # cell, clone, p_aneuploid
    seg = pd.read_csv(os.path.join(out_dir, "cell_seg_cn.csv"), index_col=0)  # cells x CHROM
    cells = list(clone["cell"])
    n_seg = numbat_in["n_segments"]

    def _grid(fname):
        """Read a cells x CHROM numbat output into a (cells x n_seg) array aligned to `cells`."""
        path = os.path.join(out_dir, fname)
        if not os.path.exists(path):
            return None
        m = pd.read_csv(path, index_col=0)
        arr = np.full((len(cells), n_seg), np.nan)
        for j in range(n_seg):
            col = str(j + 1)
            if col in m.columns:
                arr[:, j] = m.reindex(cells)[col].values
        return arr

    seg_cn = _grid("cell_seg_cn.csv")
    # allele-state layer (the point of Numbat over inferCNV): P(allelic imbalance) and P(copy-neutral
    # LOH) per (cell, segment) — present only if the runner emitted them (older runs won't have them).
    seg_imbalance = _grid("cell_seg_imbalance.csv")
    seg_loh = _grid("cell_seg_loh.csv")
    return dict(clone=clone.set_index("cell")["clone"].to_dict(),
                p_aneuploid=clone.set_index("cell")["p_aneuploid"].to_dict(),
                seg_cn=seg_cn, seg_imbalance=seg_imbalance, seg_loh=seg_loh, cells=cells)


def score_numbat(res, numbat_in):
    """Score Numbat against iscc truth: malignant-vs-normal AUC (by P(aneuploid)), per-segment single-
    cell CN correlation, clone-assignment ARI, and the denoised clone-level CN correlation — the same
    quantities :func:`score_infercnv` reports, so the two tools are compared like-for-like."""
    from sklearn.metrics import roc_auc_score, adjusted_rand_score

    cells = res["cells"]
    id_pos = {c: i for i, c in enumerate(numbat_in["cell_ids"])}
    is_mal_all = numbat_in["is_malignant"]
    mal = np.array([bool(is_mal_all[id_pos[c]]) for c in cells])
    tr_all = numbat_in["true_seg_cn"]
    tr = np.stack([tr_all[id_pos[c]] for c in cells]).astype(float)   # cells x segments (true CN)

    pan = np.array([res["p_aneuploid"].get(c, np.nan) for c in cells])
    ok = ~np.isnan(pan)
    mn_auc = (float(roc_auc_score(mal[ok].astype(int), pan[ok]))
              if ok.sum() and 0 < mal[ok].sum() < ok.sum() else float("nan"))

    seg = res["seg_cn"]
    per_seg_r = {}
    for j in range(seg.shape[1]):
        col = seg[:, j]
        m = mal & ~np.isnan(col)
        if m.sum() > 2 and tr[m, j].std() > 0 and np.nanstd(col[m]) > 0:
            per_seg_r[j] = float(np.corrcoef(col[m], tr[m, j])[0, 1])

    # clone assignment ARI (malignant cells only), against the true clone labels
    mal_cells = numbat_in["mal_cells"]
    clone_true = {c: int(l) for c, l in zip(mal_cells, numbat_in["clone_labels"])}
    common = [c for c in cells if c in clone_true and c in res["clone"]]
    ari = (float(adjusted_rand_score([clone_true[c] for c in common],
                                     [res["clone"][c] for c in common]))
           if len(common) > 2 else float("nan"))

    # clone-level CN correlation (denoise per inferred clone, centre, correlate on CN-varying segments)
    out = dict(malignant_normal_auc=mn_auc, per_segment_r=per_seg_r,
               mean_segment_r=float(np.mean(list(per_seg_r.values()))) if per_seg_r else float("nan"),
               clone_ari=ari, n_cells=len(cells), n_malignant=int(mal.sum()))
    if len(common) > 2:
        lab = np.array([res["clone"][c] for c in common])
        pos = np.array([cells.index(c) for c in common])
        segc = seg[pos]
        trc = tr[pos]
        uk = [k for k in np.unique(lab) if (lab == k).sum() >= 2]
        if len(uk) >= 2:
            ci = np.stack([np.nanmean(segc[lab == k], 0) for k in uk])
            ct = np.stack([trc[lab == k].mean(0) for k in uk])
            info = np.where(ct.std(0) > 0.1)[0]
            cin = ci - np.nanmean(ci, 0, keepdims=True)
            ctn = ct - ct.mean(0, keepdims=True)
            good = info[~np.isnan(cin[:, info]).any(0)] if len(info) else info
            out["clone_level_r"] = (float(np.corrcoef(cin[:, good].ravel(), ctn[:, good].ravel())[0, 1])
                                    if len(good) else float("nan"))
    return out


def _stratified_auc(total, y, score):
    """AUC of ``score`` for ``y`` (0/1), computed WITHIN each total-CN stratum and pooled (size-
    weighted). Controlling for total CN isolates the *allelic* signal: an inferCNV-style total-CN score
    is chance (~0.5) here by construction, because 4+0 and 2+2 have the same total; the allele layer is
    not. Strata lacking both classes or with a constant score are skipped."""
    from sklearn.metrics import roc_auc_score
    total, y, score = np.asarray(total), np.asarray(y, int), np.asarray(score, float)
    ok = ~np.isnan(score)
    total, y, score = total[ok], y[ok], score[ok]
    aucs, wts = [], []
    for t in np.unique(total):
        m = total == t
        yt, st = y[m], score[m]
        if 0 < yt.sum() < len(yt) and np.std(st) > 0:
            aucs.append(roc_auc_score(yt, st)); wts.append(len(yt))
    return float(np.average(aucs, weights=wts)) if aucs else float("nan"), int(np.sum(wts))


def score_numbat_imbalance(res, numbat_in, allele_cn, infercnv=None):
    """Score allelic-imbalance-STATE recovery — the capability the allele layer has and inferCNV does
    not. For every malignant (cell, segment) present in Numbat's output, the ground truth is whether
    the segment is allelically imbalanced (``p != m``, from :func:`segment_allele_cn`); Numbat's score
    is ``P(imbalance)`` from its allele posterior. inferCNV, if given, is scored on the SAME segments
    using its total-CN magnitude — the only signal it has.

    Returns AUCs both marginally and *controlling for total CN* (the honest test: at a fixed total,
    only the allele layer can see the split), plus the segment-class fractions for the figure.
    """
    from sklearn.metrics import roc_auc_score

    cells = res["cells"]
    nb_imb = res.get("seg_imbalance")
    if nb_imb is None:
        return None                                    # runner predates the allele-state output
    id_pos = {c: i for i, c in enumerate(numbat_in["cell_ids"])}
    is_mal_all = numbat_in["is_malignant"]
    mal_set = set(numbat_in["mal_cells"])

    inf_lookup = None
    if infercnv is not None:
        inf_seg = np.abs(infercnv["seg_score"])        # |relative CNV| per (cell, seg)
        inf_lookup = {c: inf_seg[i] for i, c in enumerate(infercnv["obs_names"])}

    gt_imb, total, nb_s, nb_l, inf_s = [], [], [], [], []
    for ci, c in enumerate(cells):
        if c not in mal_set or c not in id_pos or allele_cn.get(c) is None:
            continue
        acn = allele_cn[c]
        for s in range(numbat_in["n_segments"]):
            p, m = int(acn[s, 0]), int(acn[s, 1])
            gt_imb.append(int(abs(p - m) >= 1))
            total.append(p + m)
            nb_s.append(nb_imb[ci, s])
            nb_l.append(res["seg_loh"][ci, s] if res.get("seg_loh") is not None else np.nan)
            inf_s.append(inf_lookup[c][s] if (inf_lookup is not None and c in inf_lookup) else np.nan)
    gt_imb = np.array(gt_imb); total = np.array(total)
    nb_s = np.array(nb_s, float); nb_l = np.array(nb_l, float); inf_s = np.array(inf_s, float)

    def _auc(y, sc):
        ok = ~np.isnan(sc)
        return (float(roc_auc_score(y[ok], sc[ok]))
                if ok.sum() and 0 < y[ok].sum() < ok.sum() and np.std(sc[ok]) > 0 else float("nan"))

    even = total % 2 == 0                               # even total -> total CN can't reveal imbalance
    out = dict(
        n_pairs=int(len(gt_imb)),
        frac_imbalanced=float(gt_imb.mean()) if len(gt_imb) else float("nan"),
        frac_allele_only=float((gt_imb & even).mean()) if len(gt_imb) else float("nan"),
        # marginal AUCs
        numbat_auc=_auc(gt_imb, nb_s),
        numbat_auc_even=_auc(gt_imb[even], nb_s[even]),
        infercnv_auc_even=_auc(gt_imb[even], inf_s[even]) if infercnv is not None else float("nan"),
    )
    # the rigorous, total-CN-controlled AUCs (the money numbers)
    out["numbat_auc_ctrl"], out["n_ctrl"] = _stratified_auc(total, gt_imb, nb_s)
    if infercnv is not None:
        out["infercnv_auc_ctrl"], _ = _stratified_auc(total, gt_imb, inf_s)
    return out
