"""Fit the per-gene Visium platform factor (``visium_platform_factor.tsv.gz``) on Wu et al. 2021's paired tumours.

Visium captures each gene with a sensitivity of its own relative to droplet scRNA-seq (cell2location's
per-gene technology sensitivity; RCTD's platform effect). Four tumours of Wu et al. 2021 (Nat Genet) have
both assays: CID4290 (scRNA CID4290A), CID4465, CID4535 and CID44971. For tumour ``p`` the Visium
pseudobulk share of gene ``g`` is modelled, as in RCTD's bulk mode, as proportional to
``(sum_t w_pt mu_ptg) exp(gamma_g)``: ``mu_ptg`` the mean share of ``g`` in the scRNA cells of type ``t``
(Wu's major types), ``w_pt >= 0`` per tumour, ``gamma_g`` shared. The fit alternates non-negative least
squares for ``w_p`` with ``gamma_g += mean_p log(v_pg / s_pg)``. Genes with at least ``--min-umis`` UMIs in
both pseudobulks of every tumour get a factor, centred at a median of 0; the others are left out (0).
The registration is the vault note ``exp-visium-platform-factor``.

Data (CC-BY 4.0): scRNA GEO GSE176078 (``GSE176078_Wu_etal_2021_BRCA_scRNASeq.tar.gz``, read as a stream);
Visium Zenodo 4739739 (``filtered_count_matrices``). Genes are matched to the GRCh38-2020-A genes by
symbol; a symbol shared by two genes is dropped.

Usage::

    python validation/data/fit_visium_platform_factor.py --scrna-tar <GSE176078 tar.gz> \\
        --visium <filtered_count_matrices dir> --out <visium_platform_factor.tsv.gz> [--report <csv>]
"""
import argparse
import io
import os
import tarfile

import numpy as np
import pandas as pd
import scipy.io
from scipy.optimize import nnls

from iscc.data.human import load_reference

PAIRS = {"CID4290": "CID4290A", "CID4465": "CID4465", "CID4535": "CID4535", "CID44971": "CID44971"}
MEMBER = "Wu_etal_2021_BRCA_scRNASeq/"


def scrna_type_sums(tar_path, chunk=20_000_000):
    """Per (tumour, major type): UMI sums over genes and the number of cells, streamed from the archive."""
    with tarfile.open(tar_path, "r:gz") as tf:
        meta = pd.read_csv(tf.extractfile(MEMBER + "metadata.csv"), index_col=0)
        genes = pd.read_csv(tf.extractfile(MEMBER + "count_matrix_genes.tsv"), header=None)[0].astype(str)
        bcs = pd.read_csv(tf.extractfile(MEMBER + "count_matrix_barcodes.tsv"), header=None)[0].astype(str)
        meta = meta.reindex(bcs)
        keep = meta["orig.ident"].isin(PAIRS.values()).to_numpy()
        groups = pd.Index(sorted(set(zip(meta.loc[keep, "orig.ident"], meta.loc[keep, "celltype_major"]))))
        gid = np.full(len(bcs), -1)
        gid[keep] = groups.get_indexer(list(zip(meta.loc[keep, "orig.ident"], meta.loc[keep, "celltype_major"])))
        sums = np.zeros(len(groups) * len(genes))
        f = io.TextIOWrapper(tf.extractfile(MEMBER + "count_matrix_sparse.mtx"))
        line = f.readline()
        while line.startswith("%"):
            line = f.readline()
        n_genes, n_cells, _ = map(int, line.split())
        assert (n_genes, n_cells) == (len(genes), len(bcs)), (n_genes, n_cells)
        for part in pd.read_csv(f, sep=" ", header=None, names=["g", "c", "x"], chunksize=chunk,
                                dtype={"g": np.int32, "c": np.int32, "x": np.float32}):
            g = gid[part["c"].to_numpy() - 1]
            on = g >= 0
            flat = g[on].astype(np.int64) * len(genes) + part["g"].to_numpy()[on] - 1
            sums += np.bincount(flat, weights=part["x"].to_numpy()[on], minlength=sums.size)
    n_cells = pd.Series(gid[gid >= 0]).value_counts().reindex(range(len(groups)), fill_value=0).to_numpy()
    sums = sums.reshape(len(groups), len(genes))
    return pd.DataFrame(sums, index=pd.MultiIndex.from_tuples(groups, names=["tumour", "type"]), columns=genes), n_cells


def visium_sums(vis_dir, sample):
    d = os.path.join(vis_dir, f"{sample}_filtered_count_matrix")
    genes = pd.read_csv(os.path.join(d, "features.tsv.gz"), header=None, compression=None)[0].astype(str)
    with open(os.path.join(d, "matrix.mtx.gz"), "rb") as f:            # plain text despite the name
        X = scipy.io.mmread(f).tocsr()                                   # genes x spots
    return pd.Series(np.asarray(X.sum(axis=1)).ravel(), index=genes), X.shape[1]


def fit(mu, v, floor, n_iter=500, tol=1e-4):
    """``mu[p]``: types x genes shares, ``v[p]``: gene shares; returns gamma and the type weights.

    The weights are fitted by non-negative least squares with each gene weighted by the inverse square
    root of its share, so the few most expressed genes do not set them alone. A predicted share is
    floored at ``floor[p]`` (one UMI of the pseudobulk), so a gene the fitted types do not express
    gets a finite factor. Type weights and the factors of a type's marker genes trade off freely, so
    gamma is kept free of any type-level part: after each update the part of gamma explained by each
    gene's type mix (the share of its predicted expression from each type, averaged over tumours) is
    removed. In iscc the type composition of a spot comes from the simulated tissue, not the factor."""
    n_genes = len(next(iter(v.values())))
    gamma = np.zeros(n_genes)
    for it in range(n_iter):
        logr, weights, mix = [], {}, []
        for p in v:
            A = (mu[p] * np.exp(gamma)[None, :]).T
            sw = 1.0 / np.sqrt(v[p] + floor[p])
            w, _ = nnls(A * sw[:, None], v[p] * sw)
            weights[p] = w
            parts = A * w[None, :]
            pred = parts.sum(axis=1)
            mix.append(np.divide(parts, pred[:, None], out=np.zeros_like(parts), where=pred[:, None] > 0))
            pred = pred / pred.sum()
            logr.append(np.log((v[p] + floor[p]) / (pred + floor[p])))
        new = gamma + np.mean(logr, axis=0)
        pi = np.mean(mix, axis=0)                       # genes x types (types aligned across tumours)
        beta, *_ = np.linalg.lstsq(pi, new, rcond=None)
        new = new - pi @ beta
        new -= np.median(new)
        step = np.abs(new - gamma).max()
        gamma = new
        if step < tol:
            break
    return gamma, weights, it + 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scrna-tar", required=True)
    ap.add_argument("--visium", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", default=None, help="CSV of the per-tumour fit summaries")
    ap.add_argument("--min-umis", type=float, default=20.0)
    ap.add_argument("--min-cells", type=int, default=20)
    a = ap.parse_args()

    ref = load_reference(snps=False).genes
    sym = ref["gene_name"].astype(str)
    unique_sym = sym[~sym.duplicated(keep=False)]
    cache = os.path.splitext(a.out)[0] + ".scrna_sums.pkl"
    if os.path.exists(cache):
        sc, n_cells = pd.read_pickle(cache)
    else:
        sc, n_cells = scrna_type_sums(a.scrna_tar)
        pd.to_pickle((sc, n_cells), cache)
    print(f"scRNA: {len(sc)} tumour-type groups, {sc.shape[1]:,} genes", flush=True)
    vis = {}
    for p in PAIRS:
        vis[p], n_spots = visium_sums(a.visium, p)
        print(f"Visium {p}: {n_spots:,} spots, {len(vis[p]):,} genes, {vis[p].sum():,.0f} UMIs", flush=True)

    common = pd.Index(unique_sym.to_numpy()).intersection(sc.columns[~sc.columns.duplicated(keep=False)])
    for p in PAIRS:
        common = common.intersection(vis[p].index[~vis[p].index.duplicated(keep=False)])
    ok = np.ones(len(common), bool)
    for p, s in PAIRS.items():
        ok &= (sc.loc[s].sum(axis=0).reindex(common).to_numpy() >= a.min_umis)
        ok &= (vis[p].reindex(common).to_numpy() >= a.min_umis)
    genes = common[ok]
    print(f"genes: {len(common):,} matched by symbol, {len(genes):,} with >= {a.min_umis:g} UMIs in every "
          f"pseudobulk", flush=True)

    mu, v, types, floor = {}, {}, {}, {}
    all_types = sorted(set(sc.index.get_level_values("type")))
    for p, s in PAIRS.items():
        block = sc.loc[s]
        cells = pd.Series(n_cells[[sc.index.get_loc((s, t)) for t in block.index]], index=block.index)
        block = block[cells.to_numpy() >= a.min_cells][genes].reindex(all_types).fillna(0.0)
        types[p] = all_types
        m = block.to_numpy()
        mu[p] = np.divide(m, m.sum(axis=1, keepdims=True), out=np.zeros_like(m), where=m.sum(axis=1, keepdims=True) > 0)
        x = vis[p].reindex(genes).to_numpy()
        v[p] = x / x.sum()
        floor[p] = 1.0 / x.sum()
    gamma, weights, n_it = fit(mu, v, floor)
    print(f"converged after {n_it} iterations; gamma sd {gamma.std():.3f}, "
          f"5-95% {np.percentile(gamma, 5):.2f} to {np.percentile(gamma, 95):.2f}", flush=True)
    rows = []
    for p in PAIRS:
        A = (mu[p] * np.exp(gamma)[None, :]).T
        pred = A @ weights[p]
        pred /= pred.sum()
        sw = 1.0 / np.sqrt(v[p] + floor[p])
        w0, _ = nnls(mu[p].T * sw[:, None], v[p] * sw)
        pred0 = mu[p].T @ w0
        pred0 /= pred0.sum()
        lv = np.log10(v[p] + floor[p])
        rows.append(dict(tumour=p, r_without=np.corrcoef(lv, np.log10(pred0 + floor[p]))[0, 1],
                         r_with=np.corrcoef(lv, np.log10(pred + floor[p]))[0, 1],
                         **{f"w_{t}": round(float(w / weights[p].sum()), 4) for t, w in zip(types[p], weights[p])}))
    report = pd.DataFrame(rows)
    print(report.round(3).to_string(index=False), flush=True)
    if a.report:
        report.to_csv(a.report, index=False)
    ids = ref.set_index("gene_name").loc[genes, "gene_id"]
    out = pd.DataFrame({"gene_id": ids.to_numpy(), "gene_name": genes, "log_factor": np.round(gamma, 4)})
    out.sort_values("gene_id").to_csv(a.out, sep="\t", index=False)
    mt = out["gene_name"].str.startswith("MT-")
    print(f"wrote {len(out):,} genes to {a.out}; MT- genes {mt.sum()}, mean log factor "
          f"{out.loc[mt, 'log_factor'].mean():.2f}", flush=True)
    order = out.sort_values("log_factor")
    print("lowest:", ", ".join(f"{n} {x:.1f}" for n, x in zip(order["gene_name"][:10], order["log_factor"][:10])))
    print("highest:", ", ".join(f"{n} {x:.1f}" for n, x in zip(order["gene_name"][-10:], order["log_factor"][-10:])))


if __name__ == "__main__":
    main()
