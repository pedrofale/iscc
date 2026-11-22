"""Calibrate the human programme strength (``HUMAN_PROGRAMME_SCALE``, ``HUMAN_PROGRAMME_MAX_LOG_FOLD``).

How strongly a programme moves its genes, measured the same way in real and simulated cancer cells: for
each of iscc's Hallmark programmes, cells are scored by the mean z-scored log1p(CP10k) of the programme's
genes and split into the top and bottom 10%; for each programme gene detected in at least 5% of the
cells, the log2 fold change of mean CP10k between the two (pseudocount 0.1). The median and the 90th
percentile over genes summarise a programme. Real: the cancer epithelial cells of every Wu et al. 2021
tumour with at least 300 of them, thinned binomially to the simulated median depth; the target is the
median over tumours. Simulated: the cancer cells of a dissociated sample of an iscc tumour (10x preset)
at each (scale, bound) of a grid. The pair with the smallest squared distance to the target is the
default. The registration is the vault note ``exp-programme-fold-changes``.

Usage::

    python validation/data/fit_programme_strength.py --scrna-tar <GSE176078 tar.gz> --tumour <.iscc> \\
        --out <csv> [--scales 0.1 0.2 ...] [--bounds 1.5 2.5 5]
"""
import argparse
import io
import os
import tarfile

import numpy as np
import pandas as pd
import scipy.sparse as sp

MEMBER = "Wu_etal_2021_BRCA_scRNASeq/"
PROGRAMMES = ("proliferation", "emt", "hypoxia", "interferon_response", "drug_resistance")


def programme_stats(X, lib, members, min_detect=0.05, decile=0.1, pseudo=0.1):
    """Per programme: the median and 90th percentile of the top-against-bottom-decile log2 fold
    changes of its genes. ``X`` cells x genes counts (dense), ``lib`` the cells' UMI totals,
    ``members`` programme -> column indices."""
    cp = X / lib[:, None] * 1e4
    lg = np.log1p(cp)
    out = {}
    for name, cols in members.items():
        cols = np.asarray(cols)
        z = lg[:, cols]
        sd = z.std(axis=0)
        ok = sd > 0
        if ok.sum() < 5:
            continue
        score = ((z[:, ok] - z[:, ok].mean(axis=0)) / sd[ok]).mean(axis=1)
        lo, hi = np.quantile(score, [decile, 1 - decile])
        top, bot = score >= hi, score <= lo
        det = (X[:, cols] > 0).mean(axis=0) >= min_detect
        if det.sum() < 5:
            continue
        fc = np.log2((cp[top][:, cols[det]].mean(axis=0) + pseudo) / (cp[bot][:, cols[det]].mean(axis=0) + pseudo))
        out[name] = dict(median=float(np.median(fc)), p90=float(np.percentile(fc, 90)), n_genes=int(det.sum()))
    return out


def real_cells(tar_path, symbols, min_cells=300, chunk=20_000_000):
    """Wu et al.'s cancer epithelial cells of every tumour with ``min_cells`` of them: counts over
    ``symbols`` (cells x genes, sparse), each cell's UMI total over all genes, and its tumour."""
    with tarfile.open(tar_path, "r:gz") as tf:
        meta = pd.read_csv(tf.extractfile(MEMBER + "metadata.csv"), index_col=0)
        genes = pd.read_csv(tf.extractfile(MEMBER + "count_matrix_genes.tsv"), header=None)[0].astype(str)
        bcs = pd.read_csv(tf.extractfile(MEMBER + "count_matrix_barcodes.tsv"), header=None)[0].astype(str)
        meta = meta.reindex(bcs)
        cancer = (meta["celltype_major"] == "Cancer Epithelial").to_numpy()
        n = meta.loc[cancer, "orig.ident"].value_counts()
        keep = cancer & meta["orig.ident"].isin(n.index[n >= min_cells]).to_numpy()
        row = np.full(len(bcs), -1)
        row[keep] = np.arange(keep.sum())
        col = pd.Index(symbols).get_indexer(genes)                    # Wu gene -> column, -1 if not wanted
        lib = np.zeros(keep.sum())
        rr, cc, xx = [], [], []
        f = io.TextIOWrapper(tf.extractfile(MEMBER + "count_matrix_sparse.mtx"))
        line = f.readline()
        while line.startswith("%"):
            line = f.readline()
        for part in pd.read_csv(f, sep=" ", header=None, names=["g", "c", "x"], chunksize=chunk,
                                dtype={"g": np.int32, "c": np.int32, "x": np.float32}):
            r = row[part["c"].to_numpy() - 1]
            on = r >= 0
            g = part["g"].to_numpy()[on] - 1
            x = part["x"].to_numpy()[on]
            lib += np.bincount(r[on], weights=x, minlength=lib.size)
            c = col[g]
            w = c >= 0
            rr.append(r[on][w]); cc.append(c[w]); xx.append(x[w])
    X = sp.csr_matrix((np.concatenate(xx), (np.concatenate(rr), np.concatenate(cc))), shape=(keep.sum(), len(symbols)))
    return X, lib, meta.loc[keep, "orig.ident"].to_numpy()


def thin(X, lib, target, rng):
    """Binomial thinning of each cell to ``target`` UMIs (cells below it kept as they are)."""
    p = np.minimum(1.0, target / np.maximum(lib, 1.0))
    Y = X.tocoo(copy=True)
    Y.data = rng.binomial(Y.data.astype(np.int64), p[Y.row]).astype(float)
    return Y.tocsr(), np.round(lib * p)


def simulated(tumour_path, scales, bounds, n_cells, seed):
    import sys
    sys.path.insert(0, "validation")
    from iscc.data import scRNA
    from iscc.data.human import HumanGenomeLayer, load_hallmark_programs
    from iscc.integrations import biological_types
    from iscc.sample import Resection
    from iscc.tumor.models import GenotypeTumor
    t = GenotypeTumor.load(tumour_path)
    layer = HumanGenomeLayer(t)
    spec = Resection(t)
    cells = spec.dissociate(spec.cancer_region(margin=1), max_cells=n_cells, seed=seed)
    kind = pd.Series(biological_types(cells["cell_type"]["cell_id"]), index=cells["cell_type"].index)
    cancer = kind.index[kind == "cancer"]
    hall = load_hallmark_programs()
    ids = layer.genes["gene_id"].to_numpy()
    sym = layer.genes["gene_name"].astype(str).to_numpy()
    members = {p: np.flatnonzero(np.isin(ids, hall.loc[hall["programme"] == p, "gene_id"])) for p in PROGRAMMES}
    rows = []
    for s in scales:
        for c in bounds:
            expr = layer.expression(t, programme_scale=s, programme_max_log_fold=c)
            sc = scRNA(protocol="10x", n_cells=len(cancer), seed=seed).run(cells, cell_subset=cancer, gene_lift=expr)
            X = sc.counts_csr
            lib = np.asarray(X.sum(axis=1)).ravel()
            dense = X[:, np.unique(np.concatenate(list(members.values())))].toarray()
            remap = {p: np.searchsorted(np.unique(np.concatenate(list(members.values()))), m) for p, m in members.items()}
            st = programme_stats(dense.astype(float), lib.astype(float), remap)
            for p, v in st.items():
                rows.append(dict(source="iscc", scale=s, bound=c, programme=p, median_lib=float(np.median(lib)), **v))
            print(f"scale {s:g} bound {c:g}: median UMIs {np.median(lib):,.0f}; " +
                  ", ".join(f"{p} {v['median']:.2f}/{v['p90']:.2f}" for p, v in st.items()), flush=True)
    return pd.DataFrame(rows), sym, members


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scrna-tar", required=True)
    ap.add_argument("--tumour", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scales", type=float, nargs="+", default=[0.1, 0.2, 0.35, 0.5, 0.75, 1.0])
    ap.add_argument("--bounds", type=float, nargs="+", default=[1.5, 2.5, 5.0])
    ap.add_argument("--n-cells", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    sim, sym, members = simulated(a.tumour, a.scales, a.bounds, a.n_cells, a.seed)
    target_depth = float(sim["median_lib"].median())
    want = pd.Index(sorted(set(sym[np.concatenate(list(members.values()))])))
    cache = os.path.splitext(a.out)[0] + ".wu_cancer.pkl"
    if os.path.exists(cache):
        X, lib, tumour = pd.read_pickle(cache)
    else:
        X, lib, tumour = real_cells(a.scrna_tar, want)
        pd.to_pickle((X, lib, tumour), cache)
    rng = np.random.default_rng(a.seed)
    Xt, libt = thin(X, lib, target_depth, rng)
    real_members = {p: want.get_indexer(sym[m]) for p, m in members.items()}
    real_members = {p: m[m >= 0] for p, m in real_members.items()}
    rows = []
    for tu in np.unique(tumour):
        on = tumour == tu
        st = programme_stats(Xt[on].toarray(), libt[on], real_members)
        for p, v in st.items():
            rows.append(dict(source="wu", tumour=tu, programme=p, median_lib=float(np.median(libt[on])), **v))
    real = pd.DataFrame(rows)
    target = real.groupby("programme")[["median", "p90"]].median()
    print(f"real: {len(np.unique(tumour))} tumours, {len(tumour):,} cancer cells thinned to {target_depth:,.0f} UMIs")
    print(target.round(3).to_string())
    loss = []
    for (s, c), g in sim.groupby(["scale", "bound"]):
        g = g.set_index("programme").reindex(target.index)
        loss.append(dict(scale=s, bound=c, loss=float(((g[["median", "p90"]] - target) ** 2).to_numpy().sum())))
    loss = pd.DataFrame(loss).sort_values("loss")
    print(loss.round(4).to_string(index=False))
    best = loss.iloc[0]
    edge = best["scale"] in (min(a.scales), max(a.scales)) or best["bound"] in (min(a.bounds), max(a.bounds))
    print(f"best: scale {best['scale']:g}, bound {best['bound']:g}" + (" (on the grid's edge)" if edge else ""))
    pd.concat([real, sim, loss.assign(source="loss")], ignore_index=True).to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
