"""Fig 3's bias attribution: a tool's error with each measurement bias switched off in turn.

Registered before any run in the vault (``exp-paper1-bias-attribution``). For each R14 tumour (R11's design
with immune cells) that reached its target, the tumour is cut through its middle row; one dissociation of
the tumour piece of the upper half (its cancer demes and a rim of one deme, about 3,000 cells) and one thin
section of a 64-deme window of the lower half around its cancer are measured under every condition: every bias on, each bias
switched off in turn, and every bias off. The tumour, the samples and the seeds are the same in every
condition, so the conditions differ only by the bias. inferCNV (infercnvpy) runs on the scRNA with the
microenvironment as reference, and RCTD (spacexr) on the Visium with the scRNA as reference, each in its
own conda environment, as the tutorials run them. Each tool is scored twice: against the truth of the
cells it was given (per-cell malignancy AUC, per-spot proportion error), and against the ground truth of
the tumour region the sample came from, read off its demes (the malignant fraction and the copy-number
profile a study would report, and the cancer share of the section); sampling biases show only in the
second, since they change which cells are measured rather than how well each is classified.

    python validation/paper/bias_attribution.py run [--run R14] [--tumours seed1,disp0.9_seed1] [--conditions all_on,all_off]
    python validation/paper/bias_attribution.py run --pregrown ductal --conditions all_on,all_off   # smoke
    python validation/paper/bias_attribution.py collect      # -> <out>/bias_attribution.csv

``ISCC_INFERCNV_PYTHON`` and ``ISCC_RCTD_RSCRIPT`` name the tools' interpreters (default: the
``iscc-infercnv`` and ``iscc-rctd`` conda environments under ``~/miniconda3/envs``).
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import regime as RG  # noqa: E402
import registry as REG  # noqa: E402

OUT = os.path.join(os.environ.get("ISCC_PAPER_OUT") or os.path.join(REG.REPO, "analysis_data", "paper"),
                   RG.TIER, "bias_attribution")
ENVS = os.path.expanduser("~/miniconda3/envs")
INFERCNV_PYTHON = os.environ.get("ISCC_INFERCNV_PYTHON", os.path.join(ENVS, "iscc-infercnv", "bin", "python"))
RCTD_RSCRIPT = os.environ.get("ISCC_RCTD_RSCRIPT", os.path.join(ENVS, "iscc-rctd", "bin", "Rscript"))
N_CELLS, WINDOW, SEED = 3000, 64, 1


def atlas_stress():
    """The mean dissociation stress over the tumours of Wu et al. 2021 (the atlas median is about 0)."""
    from importlib import resources
    with resources.files("iscc.data").joinpath("reference", "wu2021_scrna_batches.tsv").open() as f:
        return float(pd.read_csv(f, sep="\t", comment="#")["stress_strength"].mean())


# Every bias on: the package's 10x preset and default recovery, the atlas stress, and the paper regime's
# Visium (spot swapping and capture field fitted to the 10x breast section).
ON = dict(recovery="default", stress=None, ambient_frac=0.05, doublet_rate=0.05, dispersion=0.35,
          sigma_lib=0.35, spot_swapping=True, capture_field=True)
OFF = dict(recovery=None, stress=0.0, ambient_frac=0.0, doublet_rate=0.0, dispersion=0.0, sigma_lib=0.0,
           spot_swapping=False, capture_field=False)
CONDITIONS = {"all_on": {}, **{f"{k}_off": {k: OFF[k]} for k in ON}, "all_off": dict(OFF)}


def settings(condition):
    s = dict(ON, **CONDITIONS[condition])
    if s["stress"] is None:
        s["stress"] = atlas_stress()
    return s


def visium_kwargs(s):
    kw = dict(RG.VISIUM)
    if not s["spot_swapping"]:
        kw["bleed_rate"] = 0.0
    if not s["capture_field"]:
        kw.update(edge_sigma=0.0, field_sigma=0.0)
    return kw


def _tumours(pregrown=None, run_name="R14"):
    if pregrown:
        import iscc.pregrown
        return [(f"pregrown_{pregrown}", lambda: getattr(iscc.pregrown, pregrown)())]
    out = []
    for task in REG.tasks(run_name):
        rec = os.path.splitext(REG.task_path(run_name, task))[0] + ".json"
        if os.path.exists(rec) and json.load(open(rec)).get("reached_target"):
            out.append((f"{run_name}_{task}", lambda task=task: REG.load(run_name, task)))
    return out


def _region_truth(t, layer, demes):
    """The ground truth of a region of the tumour, read off its demes (every cell, not a sample): the
    share of cancer cells, and the count-weighted mean copy number of its cancer cells on each arm,
    relative to the ploidy of each cell (a Series over arm names)."""
    from iscc.integrations import biological_types
    counts = {}
    for d in demes:
        for g, c in t.demes[d].items():
            counts[g] = counts.get(g, 0) + c
    gids = list(counts)
    n = np.array([counts[g] for g in gids], dtype=float)
    cancer = np.asarray(biological_types(gids)) == "cancer"
    cn_p, cn_m = layer.homolog_cn([g for g, k in zip(gids, cancer) if k], t)
    cn = (cn_p + cn_m).astype(float)
    rel = cn / cn.mean(axis=1, keepdims=True)
    profile = pd.Series((rel * n[cancer][:, None]).sum(axis=0) / n[cancer].sum(), index=list(layer.arms))
    return dict(cancer_frac=float(n[cancer].sum() / n.sum()), profile=profile)


def _samples(t):
    """The fixed samples of one tumour: the tumour piece of the upper half to dissociate, the section, the
    shared stress genes. The tumour is cut through its middle row (the median row of its cancer cells),
    as a pathologist halves a resected tumour, so both halves hold cancer; the dissociated piece is the
    upper half's cancer demes with a rim of one deme (``Resection.cancer_region``), not the whole half,
    which is mostly normal tissue."""
    from iscc.sample import Resection
    specimen = Resection(t)
    G = t.grid_size
    n_cancer = np.array([sum(c for g, c in t.demes[d].items() if t._is_cancer(g)) for d in range(G * G)],
                        dtype=float)
    rows = np.repeat(np.arange(G), G)
    order = np.argsort(rows, kind="stable")
    cum = np.cumsum(n_cancer[order])
    middle = int(rows[order][np.searchsorted(cum, cum[-1] / 2)])
    upper, lower = specimen.bisect(frac=(middle + 0.5) / G, axis="y")
    upper = sorted(set(specimen.cancer_region(margin=1)) & set(upper))
    lower_set = set(lower)
    w = np.array([sum(c for g, c in t.demes[d].items() if t._is_cancer(g)) for d in lower], dtype=float)
    rc = np.array([t.deme_coords[d] for d in lower], dtype=float)
    centre = tuple(int(round(x)) for x in (rc * w[:, None]).sum(axis=0) / max(w.sum(), 1.0))
    lesion = sorted(set(t.primary_window(side=WINDOW, center=centre)) & lower_set)
    section = specimen.slice(lesion, depth_frac=0.5, max_cells=100_000, layers="visium")
    genes = specimen.dissociate(upper, max_cells=10, seed=0)["cell_exp"].columns
    stress_genes = list(np.random.default_rng(0).choice(genes, size=round(0.05 * len(genes)), replace=False))
    return specimen, upper, section, stress_genes, lesion


INFERCNV_SCRIPT = '''
import sys
import numpy as np, pandas as pd, scanpy as sc, infercnvpy as cnv

def main(d):
    adata = sc.read_10x_h5(f"{d}/scrna/outs/filtered_feature_bc_matrix.h5")
    adata.var_names = adata.var["gene_ids"]
    genes = pd.read_csv(f"{d}/scrna/clonalscope/genes.bed", sep="\\t", index_col="geneID")
    for k, col in (("chromosome", "chr"), ("start", "start"), ("end", "end")):
        adata.var[k] = genes.loc[adata.var_names, col].to_numpy()
    ref = pd.read_csv(f"{d}/reference_barcodes.txt", header=None)[0]
    adata.obs["annotation"] = np.where(adata.obs_names.isin(ref), "microenvironment", "unknown")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    cnv.tl.infercnv(adata, reference_key="annotation", reference_cat="microenvironment")
    cnv.tl.pca(adata); cnv.pp.neighbors(adata); cnv.tl.leiden(adata); cnv.tl.cnv_score(adata)
    adata.write_h5ad(f"{d}/infercnv.h5ad")

if __name__ == "__main__":
    main(sys.argv[1])
'''

RCTD_SCRIPT = '''
suppressMessages({library(spacexr); library(Matrix)})
d <- commandArgs(trailingOnly = TRUE)[1]
read_10x <- function(dir) {
  counts <- readMM(file.path(dir, "matrix.mtx.gz"))
  rownames(counts) <- make.unique(read.delim(file.path(dir, "features.tsv.gz"), header = FALSE)[[2]])
  colnames(counts) <- read.delim(file.path(dir, "barcodes.tsv.gz"), header = FALSE)[[1]]
  as(counts, "CsparseMatrix")
}
ref_counts <- read_10x(file.path(d, "scrna/outs/filtered_feature_bc_matrix"))
annotation <- read.csv(file.path(d, "reference_cell_types.csv"), row.names = 1)
ref_counts <- ref_counts[, rownames(annotation)]
cell_types <- factor(annotation$cell_type, levels = sort(unique(annotation$cell_type)))
names(cell_types) <- rownames(annotation)
reference <- Reference(ref_counts, cell_types)
spot_counts <- read_10x(file.path(d, "visium/outs/filtered_feature_bc_matrix"))
positions <- read.csv(file.path(d, "visium/outs/spatial/tissue_positions.csv"), row.names = 1)
coords <- positions[colnames(spot_counts), c("pxl_col_in_fullres", "pxl_row_in_fullres")]
colnames(coords) <- c("x", "y")
rctd <- run.RCTD(create.RCTD(SpatialRNA(coords, spot_counts), reference, max_cores = 1), doublet_mode = "full")
write.csv(as.matrix(normalize_weights(rctd@results$weights)), file.path(d, "rctd_proportions.csv"))
'''


def _score_infercnv(d, layer, region):
    import anndata as ad
    from sklearn.metrics import roc_auc_score
    from iscc.integrations import biological_types
    truth = pd.read_csv(f"{d}/scrna/truth/cells.tsv", sep="\t", index_col="barcode", dtype={"genotype": str})
    truth = truth[~truth["is_doublet"]]
    kind = pd.Series(biological_types(truth["genotype"]), index=truth.index)
    result = ad.read_h5ad(f"{d}/infercnv.h5ad")
    score = result.obs["cnv_score"].reindex(truth.index)
    epi = kind.isin(["cancer", "epithelial"])
    auc = float(roc_auc_score(kind[epi] == "cancer", score[epi]))
    # inferCNV's windows of 100 genes moved by 10, each assigned to the arm of its middle gene
    genes = layer.genes.assign(chromosome=layer.genes["chrom"].str[3:].astype(int)).sort_values(["chromosome", "start"])
    window_arm = np.concatenate([g["arm"].to_numpy()[np.arange(0, len(g) - 99, 10) + 50]
                                 for _, g in genes.groupby("chromosome")])
    arms = pd.unique(genes["arm"])
    inferred = pd.DataFrame(result.obsm["X_cnv"].toarray(), index=result.obs_names)
    inferred = inferred.T.groupby(window_arm).mean().T[arms].reindex(truth.index)
    seg = pd.read_csv(f"{d}/scrna/truth/cn_segments.tsv", sep="\t", dtype={"genotype": str})
    true_cn = seg.pivot(index="genotype", columns="arm", values="cn").loc[truth["genotype"], arms]
    true_cn.index = truth.index
    cancer = kind == "cancer"
    rel = true_cn[cancer].div(true_cn[cancer].mean(axis=1), axis=0)
    r = float(np.corrcoef(inferred[cancer].to_numpy().ravel(), rel.to_numpy().ravel())[0, 1])
    # what a study reports about the tumour, scored against the region's truth: the share of barcodes
    # (doublets included) called malignant, a cell scoring above the 95th percentile of the reference;
    # and the mean profile of the called cells against the region's count-weighted true profile
    ref = pd.read_csv(f"{d}/reference_barcodes.txt", header=None)[0]
    every = result.obs["cnv_score"]
    called = every > every.reindex(ref).quantile(0.95)
    allx = pd.DataFrame(result.obsm["X_cnv"].toarray(), index=result.obs_names)
    allx = allx.T.groupby(window_arm).mean().T[arms]
    prof = allx[called.to_numpy()].mean()
    truth_prof = region["profile"].reindex(arms)
    return dict(infercnv_auc=auc, infercnv_cn_r=r, infercnv_called_frac=float(called.mean()),
                true_cancer_frac=region["cancer_frac"],
                infercnv_frac_error=float(called.mean() - region["cancer_frac"]),
                infercnv_profile_r=float(np.corrcoef(prof.to_numpy(), truth_prof.to_numpy())[0, 1]))


def _score_rctd(d, region):
    from iscc.integrations import biological_types
    spots = pd.read_csv(f"{d}/visium/truth/spots.tsv", sep="\t", index_col="barcode", dtype={"genotypes": str})
    spot_cells = spots["genotypes"].dropna().str.split(",").explode()
    true_prop = pd.crosstab(spot_cells.index, biological_types(spot_cells), normalize="index")
    rctd = pd.read_csv(f"{d}/rctd_proportions.csv", index_col=0)
    true_prop = true_prop.reindex(index=rctd.index, columns=rctd.columns, fill_value=0.0)
    from scipy.io import mmread
    import gzip
    mtx = f"{d}/visium/outs/filtered_feature_bc_matrix"
    with gzip.open(f"{mtx}/matrix.mtx.gz") as f:
        umis = np.asarray(mmread(f).sum(axis=0)).ravel()
    with gzip.open(f"{mtx}/barcodes.tsv.gz", "rt") as f:
        umis = pd.Series(umis, index=[line.strip() for line in f]).reindex(rctd.index)
    est = float(np.average(rctd["cancer"], weights=umis)) if "cancer" in rctd else float("nan")
    return dict(rctd_mae=float((rctd - true_prop).abs().mean().mean()), rctd_spots=int(len(rctd)),
                rctd_cancer_frac=est, true_section_cancer_frac=region["cancer_frac"],
                rctd_frac_error=est - region["cancer_frac"])


def run_one(name, t, samples, layer, expression, condition, regions):
    from iscc.data import Visium, scRNA
    from iscc.data.human import write_scrna, write_visium
    from iscc.integrations import biological_types
    specimen, upper, section, stress_genes, _ = samples
    s = settings(condition)
    d = tempfile.mkdtemp(prefix="bias_attr_")
    t0 = time.time()
    try:
        cells = specimen.dissociate(upper, max_cells=N_CELLS, recovery_probs=s["recovery"],
                                    stress_strength=s["stress"], stress_genes=stress_genes, seed=SEED)
        rna = scRNA(protocol="10x", n_cells=len(cells["cell_exp"]), seed=SEED, ambient_frac=s["ambient_frac"],
                    doublet_rate=s["doublet_rate"], dispersion=s["dispersion"], sigma_lib=s["sigma_lib"]
                    ).run(cells, gene_lift=expression)
        write_scrna(f"{d}/scrna", rna, layer, cells, tumor=t)
        own = {k: v.copy() for k, v in section.items()}            # every condition reads the same section
        vis = Visium(seed=SEED, section_frac=1.0, **visium_kwargs(s)).run(own, grid_side=WINDOW,
                                                                          gene_lift=expression)
        write_visium(f"{d}/visium", vis, layer, tumor=t)
        truth = pd.read_csv(f"{d}/scrna/truth/cells.tsv", sep="\t", index_col="barcode", dtype={"genotype": str})
        truth = truth[~truth["is_doublet"]]
        kind = pd.Series(biological_types(truth["genotype"]), index=truth.index)
        pd.Series(kind.index[kind.isin(["stromal", "immune"])]).to_csv(f"{d}/reference_barcodes.txt",
                                                                       index=False, header=False)
        kind.rename("cell_type").to_frame().to_csv(f"{d}/reference_cell_types.csv")
        with open(f"{d}/run_infercnv.py", "w") as f:
            f.write(INFERCNV_SCRIPT)
        with open(f"{d}/run_rctd.R", "w") as f:
            f.write(RCTD_SCRIPT)
        env = dict(os.environ, MPLBACKEND="Agg", TQDM_DISABLE="1", PYTHONWARNINGS="ignore")
        subprocess.run([INFERCNV_PYTHON, f"{d}/run_infercnv.py", d], check=True, cwd=d, env=env,
                       stdout=subprocess.DEVNULL, stderr=open(f"{d}/infercnv.log", "w"))
        subprocess.run([RCTD_RSCRIPT, f"{d}/run_rctd.R", d], check=True, cwd=d, env=env,
                       stdout=subprocess.DEVNULL, stderr=open(f"{d}/rctd.log", "w"))
        row = dict(tumour=name, condition=condition, **{k: (v if not isinstance(v, dict) else "default")
                                                        for k, v in s.items()},
                   n_cells=int(len(truth)), **_score_infercnv(d, layer, regions["upper"]),
                   **_score_rctd(d, regions["section"]),
                   wall_s=round(time.time() - t0, 1))
    finally:
        shutil.rmtree(d, ignore_errors=True)
    os.makedirs(os.path.join(OUT, "rows"), exist_ok=True)
    with open(os.path.join(OUT, "rows", f"{name}_{condition}.json"), "w") as f:
        json.dump(row, f)
    print(json.dumps(row), flush=True)


def run(tumours=None, conditions=None, pregrown=None, force=False, run_name="R14"):
    from iscc.data.human import HumanGenomeLayer
    conds = conditions or list(CONDITIONS)
    for name, get in _tumours(pregrown, run_name):
        if tumours and name.split("_", 1)[1] not in tumours:
            continue
        todo = [c for c in conds if force or not os.path.exists(os.path.join(OUT, "rows", f"{name}_{c}.json"))]
        if not todo:
            continue
        t = get()
        layer = HumanGenomeLayer(t)
        expression = layer.expression(t)
        samples = _samples(t)
        regions = {"upper": _region_truth(t, layer, samples[1]), "section": _region_truth(t, layer, samples[4])}
        for c in todo:
            run_one(name, t, samples, layer, expression, c, regions)


def collect():
    rows = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(OUT, "rows", "*.json")))]
    df = pd.DataFrame(rows)
    path = os.path.join(OUT, "bias_attribution.csv")
    df.to_csv(path, index=False)
    print(f"{len(df)} rows -> {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["run", "collect"])
    ap.add_argument("--run", default="R14", help="the registry run (default R14, R11's design with immune cells)")
    ap.add_argument("--tumours", help="comma-separated tasks of the run (default: every one that reached its target)")
    ap.add_argument("--conditions", help=f"comma-separated, of {list(CONDITIONS)}")
    ap.add_argument("--pregrown", help="a pregrown tumour instead of the run (smoke)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.tumours.split(",") if a.tumours else None, a.conditions.split(",") if a.conditions else None,
            a.pregrown, a.force, a.run)
    else:
        collect()


if __name__ == "__main__":
    main()
