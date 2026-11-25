"""One tumour of the breast cancer atlas of Wu et al. 2021 as an AnnData, the scRNA reference of Figure 3.

Source: GEO GSE176078, ``GSE176078_Wu_etal_2021_BRCA_scRNASeq.tar.gz`` (Wu et al. 2021, Nat Genet
53:1334-1347, doi:10.1038/s41588-021-00911-1): 100,064 cells of 26 tumours, each its own 10x run, with
the authors' cell types (``celltype_major``). Figure 3 compared iscc's scRNA with PBMC3k until
2026-11-25; a breast tumour is the like-for-like reference for a simulated breast tumour.

The tumour is the one with the most malignant cells (``celltype_major == "Cancer Epithelial"``),
CID4290A (ER+, 4,053 of 5,789 cells), unless ``--tumour`` names another. The matrix is read in chunks
and only the tumour's cells are kept, so the 2.4 GB file is never held in memory.

Usage::

    python validation/paper/tools/wu_tumour.py --tar GSE176078_Wu_etal_2021_BRCA_scRNASeq.tar.gz \\
        --out data/wu2021_CID4290A.h5ad
    python validation/paper/tools/wu_tumour.py --dir <extracted Wu_etal_2021_BRCA_scRNASeq> --out ...
"""
import argparse
import os
import shutil
import subprocess
import tempfile

import numpy as np
import pandas as pd
import scipy.sparse as sp

MALIGNANT = "Cancer Epithelial"
CHUNK = 5_000_000         # entries of the matrix read at a time


def pick_tumour(meta):
    """The tumour with the most malignant cells."""
    n = meta.loc[meta["celltype_major"] == MALIGNANT, "orig.ident"].value_counts()
    return str(n.index[0])


def extract(d, tumour=None):
    """The counts (cells x genes CSR), genes and metadata of one tumour of the extracted atlas ``d``."""
    genes = pd.read_csv(os.path.join(d, "count_matrix_genes.tsv"), sep="\t", header=None)[0].astype(str).to_numpy()
    bcs = pd.read_csv(os.path.join(d, "count_matrix_barcodes.tsv"), sep="\t", header=None)[0].astype(str).to_numpy()
    meta = pd.read_csv(os.path.join(d, "metadata.csv"), index_col=0).reindex(bcs)
    if meta["orig.ident"].isna().any():
        raise ValueError("the metadata misses some barcodes of the matrix")
    tumour = tumour or pick_tumour(meta)
    keep = (meta["orig.ident"] == tumour).to_numpy()
    if not keep.any():
        raise ValueError(f"no cells of tumour {tumour!r}")
    col = np.full(len(bcs), -1, np.int64)
    col[keep] = np.arange(keep.sum())
    path = os.path.join(d, "count_matrix_sparse.mtx")
    skip = 0
    with open(path) as f:
        for line in f:
            skip += 1
            if not line.startswith("%"):
                n_rows, n_cols, _ = (int(x) for x in line.split())
                break
    if (n_rows, n_cols) != (len(genes), len(bcs)):
        raise ValueError(f"matrix {n_rows} x {n_cols} is not genes x barcodes ({len(genes)} x {len(bcs)})")
    rows, cols, vals = [], [], []
    for ch in pd.read_csv(path, sep=" ", header=None, skiprows=skip, chunksize=CHUNK,
                          dtype={0: np.int32, 1: np.int32, 2: np.float32}, engine="c"):
        c = col[ch[1].to_numpy() - 1]
        on = c >= 0
        rows.append(ch[0].to_numpy()[on] - 1)
        cols.append(c[on])
        vals.append(ch[2].to_numpy()[on])
    X = sp.csr_matrix((np.concatenate(vals), (np.concatenate(cols), np.concatenate(rows))),
                      shape=(int(keep.sum()), len(genes)))
    return X, genes, meta.loc[keep].copy(), tumour


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--tar", help="the GEO archive")
    src.add_argument("--dir", help="the archive, already extracted (Wu_etal_2021_BRCA_scRNASeq)")
    ap.add_argument("--tumour", default=None, help="default: the tumour with the most malignant cells")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import anndata as ad
    work = None
    try:
        if a.tar:
            work = tempfile.mkdtemp(dir=os.environ.get("TMPDIR"))
            subprocess.run(["tar", "xzf", a.tar, "-C", work], check=True)
            d = os.path.join(work, "Wu_etal_2021_BRCA_scRNASeq")
        else:
            d = a.dir
        X, genes, meta, tumour = extract(d, a.tumour)
    finally:
        if work:
            shutil.rmtree(work, ignore_errors=True)
    obs = meta[["orig.ident", "subtype", "celltype_major", "celltype_minor"]].astype(str)
    adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=pd.Index(genes, name="gene")))
    adata.var_names_make_unique()
    adata.uns["source"] = f"Wu et al. 2021, GEO GSE176078, tumour {tumour}"
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    adata.write_h5ad(a.out, compression="gzip")
    print(f"{tumour}: {adata.n_obs} cells x {adata.n_vars} genes -> {a.out}")
    print(obs["celltype_major"].value_counts().to_string())


if __name__ == "__main__":
    main()
