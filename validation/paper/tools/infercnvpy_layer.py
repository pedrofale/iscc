"""infercnvpy on a human-gene dataset written by iscc.data.human.write_visium or write_scrna.

Run with the image's base environment (``bin/python-infercnv``)::

    python infercnvpy_layer.py <dataset_dir> <out_dir> [--window 100] [--n-clones 4] [--seed 0]

Places the genes with ``<dataset>/clonalscope/genes.bed``, takes the barcodes ``celltype.tsv`` labels
``normal`` as the reference (as Clonalscope does), and runs ``infercnvpy.tl.infercnv``. Writes the
common output of this directory (see calls.py): calls.tsv (p_malignant = the CNV score, ranked against
the reference: the share of reference barcodes it exceeds; malignant above the reference's 95th
percentile; clone = k-means on the CNV profile of the non-reference barcodes), cn_arm.tsv (the mean
CNV signal over each arm's windows, scale "log2ratio"), run.json, and the CNV matrix in native/.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import calls as CALLS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("out")
    ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--n-clones", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    import infercnvpy as cnv
    import scanpy as sc
    from sklearn.cluster import KMeans

    ad = sc.read_10x_h5(os.path.join(a.dataset, "outs", "filtered_feature_bc_matrix.h5"), gex_only=True)
    ad.var_names = ad.var["gene_ids"].astype(str)
    bed = pd.read_csv(os.path.join(a.dataset, "clonalscope", "genes.bed"), sep="\t").set_index("geneID")
    ad = ad[:, ad.var_names.isin(bed.index)].copy()
    ad.var["chromosome"] = bed.loc[ad.var_names, "chr"].to_numpy()
    ad.var["start"] = bed.loc[ad.var_names, "start"].to_numpy()
    ad.var["end"] = bed.loc[ad.var_names, "end"].to_numpy()
    ct_path = os.path.join(a.dataset, "clonalscope", "celltype.tsv")
    ct = pd.read_csv(ct_path, sep="\t").set_index("barcode").iloc[:, 0] if os.path.exists(ct_path) else None
    ad.obs["reference"] = ("normal" if ct is None else ct.reindex(ad.obs_names).fillna("tumor").to_numpy())
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    ref = ["normal"] if (ad.obs["reference"] == "normal").any() else None
    step = 10
    cnv.tl.infercnv(ad, reference_key="reference" if ref else None, reference_cat=ref,
                    window_size=a.window, step=step)
    X = ad.obsm["X_cnv"]
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    score = np.abs(X).mean(axis=1)
    is_ref = (ad.obs["reference"] == "normal").to_numpy()
    if is_ref.any():
        ref_scores = np.sort(score[is_ref])
        p = np.searchsorted(ref_scores, score, side="right") / len(ref_scores)
        malignant = (score > np.quantile(ref_scores, 0.95)).astype(int)
    else:
        p, malignant = np.full(len(score), np.nan), np.full(len(score), np.nan)
    clone = np.full(len(score), "", dtype=object)
    tum = ~is_ref
    if tum.sum() >= a.n_clones:
        km = KMeans(n_clusters=a.n_clones, n_init=10, random_state=a.seed).fit(X[tum])
        clone[tum] = km.labels_.astype(str)
    calls = pd.DataFrame({"barcode": list(ad.obs_names), "malignant": malignant, "p_malignant": p,
                          "clone": clone})
    # Window coordinates: infercnvpy slides windows of `window` genes by `step` genes along each
    # chromosome's genes in start order; uns["cnv"]["chr_pos"] holds each chromosome's first window.
    order = sorted(ad.uns["cnv"]["chr_pos"].items(), key=lambda kv: kv[1])
    win_chr = np.empty(X.shape[1], dtype=object)
    win_pos = np.zeros(X.shape[1])
    for i, (ch, s0) in enumerate(order):
        s1 = order[i + 1][1] if i + 1 < len(order) else X.shape[1]
        g = np.sort(ad.var.loc[ad.var["chromosome"] == ch, "start"].to_numpy())
        win_chr[s0:s1] = ch
        if len(g):
            win_pos[s0:s1] = g[np.minimum(np.arange(s1 - s0) * step + a.window // 2, len(g) - 1)]
    arm = CALLS.arm_of(win_chr, win_pos, CALLS.arms(a.dataset))
    df = pd.DataFrame(X, index=ad.obs_names)
    per_arm = df.T.groupby(arm).mean().drop("", errors="ignore")
    cn_arm = per_arm.stack().rename("value").reset_index()
    cn_arm.columns = ["arm", "barcode", "value"]
    os.makedirs(os.path.join(a.out, "native"), exist_ok=True)
    np.save(os.path.join(a.out, "native", "X_cnv.npy"), X)
    pd.Series(arm).to_csv(os.path.join(a.out, "native", "window_arm.csv"), index=False)
    CALLS.write(a.out, calls, cn_arm.assign(scale="log2ratio"), tool="infercnvpy", version=cnv.__version__,
                window=a.window, n_clones=a.n_clones, seed=a.seed, n_barcodes=int(ad.n_obs),
                n_reference=int(is_ref.sum()), wall_s=round(time.time() - t0, 1))
    print(f"infercnvpy done: {int(np.nansum(malignant))} of {len(calls)} called malignant", flush=True)


if __name__ == "__main__":
    main()
