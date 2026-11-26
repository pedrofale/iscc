"""Figure 3 against held-out references: does iscc, fitted to one tumour and one section, sit within the
spread of other real tumours and sections read the same way?

scRNA: iscc's statistics in Figure 3D-E (the showcase tumour over the human genes, fitted to CID4290A of the
breast cancer atlas of Wu et al. 2021), against the eight other tumours of the atlas with the most malignant
cells (GEO GSE176078; the matrix streamed out of the archive once, only those tumours kept), each read with
``fig2_realism._count_stats``: the coefficient of variation of the library size, and the zero fraction and
the variance against the mean (binned curves). The distance of two curves is the largest difference in
zero fraction, and the median difference in log10 variance, over the means both cover.

Visium: iscc's spot depth in Figure 3F (fitted to the 10x breast section), against the six sections of Wu et
al. (Zenodo 4739739: filtered matrices and tissue positions), each read as Figure 3F reads the 10x section:
the mean, standard deviation and skewness of log10 UMIs per in-tissue spot, and the Moran's I of the spot
depth (``iscc.data.morans_i`` on pixel coordinates over the median spacing of neighbouring spots).

    python validation/paper/fig3_holdout.py --wu-tar GSE176078_Wu_etal_2021_BRCA_scRNASeq.tar.gz \\
        --wu-visium <dir with filtered_count_matrices/ and spatial/> --fig3 <derived>/fig3 \\
        --out <derived>/fig3/holdout.json
"""
import argparse
import gzip
import io
import json
import os
import subprocess
import sys
import tarfile

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.io import mmread
from scipy.stats import skew

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import fig2_realism as F2  # noqa: E402

MALIGNANT = "Cancer Epithelial"
N_TUMOURS = 9                  # CID4290A (the fitted reference, as a check) and eight held out
CHUNK = 5_000_000
INNER = "Wu_etal_2021_BRCA_scRNASeq"


def _curve_distance(a, b):
    """Largest difference in zero fraction and median difference in log10 variance of two binned
    curves (``_count_stats``), over the means both cover."""
    lo, hi = max(min(a["curve_m"]), min(b["curve_m"])), min(max(a["curve_m"]), max(b["curve_m"]))
    grid = np.linspace(lo, hi, 25)
    z = np.abs(np.interp(grid, a["curve_m"], a["curve_z"]) - np.interp(grid, b["curve_m"], b["curve_z"]))
    v = np.abs(np.interp(grid, a["curve_m"], a["curve_v"]) - np.interp(grid, b["curve_m"], b["curve_v"]))
    return dict(zero_max=float(z.max()), var_median=float(np.median(v)))


def wu_tumours(tar):
    """The counts (cells x genes CSR) of the tumours with the most malignant cells, one pass over the matrix."""
    with tarfile.open(tar, "r:gz") as tf:
        meta = pd.read_csv(tf.extractfile(f"{INNER}/metadata.csv"), index_col=0)
        bcs = pd.read_csv(tf.extractfile(f"{INNER}/count_matrix_barcodes.tsv"), sep="\t", header=None)[0].astype(str).to_numpy()
        genes = pd.read_csv(tf.extractfile(f"{INNER}/count_matrix_genes.tsv"), sep="\t", header=None)[0].astype(str).to_numpy()
    meta = meta.reindex(bcs)
    top = meta.loc[meta["celltype_major"] == MALIGNANT, "orig.ident"].value_counts().index[:N_TUMOURS]
    tum = meta["orig.ident"].to_numpy()
    which = {t: np.flatnonzero(tum == t) for t in top}
    col = np.full(len(bcs), -1, np.int64)
    owner = np.full(len(bcs), -1, np.int64)
    for k, t in enumerate(top):
        col[which[t]] = np.arange(len(which[t]))
        owner[which[t]] = k
    parts = {k: ([], [], []) for k in range(len(top))}
    # the matrix (2.4 GB of text) streamed out of the archive, never written to disk
    p = subprocess.Popen(["tar", "-xzOf", tar, f"{INNER}/count_matrix_sparse.mtx"], stdout=subprocess.PIPE)
    stream = io.TextIOWrapper(p.stdout)
    header = 0
    for line in stream:
        header += 1
        if not line.startswith("%"):
            break
    for ch in pd.read_csv(stream, sep=" ", header=None, chunksize=CHUNK,
                          dtype={0: np.int32, 1: np.int32, 2: np.float32}, engine="c"):
        c = ch[1].to_numpy() - 1
        o = owner[c]
        for k in range(len(top)):
            on = o == k
            if on.any():
                parts[k][0].append(col[c[on]])
                parts[k][1].append(ch[0].to_numpy()[on] - 1)
                parts[k][2].append(ch[2].to_numpy()[on])
    p.wait()
    out = {}
    for k, t in enumerate(top):
        r, g, v = (np.concatenate(x) for x in parts[k])
        out[str(t)] = sp.csr_matrix((v, (r, g)), shape=(len(which[t]), len(genes)))
    return out


def _median_nn(xy):
    from scipy.spatial import cKDTree
    d, _ = cKDTree(xy).query(xy, k=2)
    return float(np.median(d[:, 1]))


def wu_sections(root):
    """Per section: log10 depth statistics and Moran's I of the depth, as Figure 3F reads the 10x section."""
    from iscc.data import morans_i
    out = {}
    fdir = os.path.join(root, "filtered_count_matrices")
    for name in sorted(os.listdir(fdir)):
        s = name.replace("_filtered_count_matrix", "")
        d = os.path.join(fdir, name)

        def _open(f):
            with open(f, "rb") as fh:
                gz = fh.read(2) == b"\x1f\x8b"
            return gzip.open(f, "rb") if gz else open(f, "rb")
        X = mmread(_open(os.path.join(d, "matrix.mtx.gz"))).tocsc()        # genes x spots
        bcs = pd.read_csv(_open(os.path.join(d, "barcodes.tsv.gz")), header=None)[0].astype(str).to_numpy()
        pos = pd.read_csv(os.path.join(root, "spatial", f"{s}_spatial", "tissue_positions_list.csv"), header=None,
                          index_col=0)
        pos = pos.reindex(bcs)
        tot = np.asarray(X.sum(axis=0), float).ravel()
        ok = (tot > 0) & pos[1].fillna(0).to_numpy().astype(bool)
        xy = pos[[4, 5]].to_numpy(float)[ok]
        xy = xy / _median_nn(xy)
        lt = np.log10(tot[ok])
        out[s] = dict(n_spots=int(ok.sum()), mean=float(lt.mean()), sd=float(lt.std()), skew=float(skew(lt)),
                      moran=float(morans_i(tot[ok], xy)), loglib=lt.tolist())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--wu-tar", required=True)
    ap.add_argument("--wu-visium", required=True)
    ap.add_argument("--fig3", required=True, help="Figure 3's cache directory (d.json, e.json)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = json.load(open(os.path.join(a.fig3, "d.json")))["left"]
    e = json.load(open(os.path.join(a.fig3, "e.json")))["left"]
    iscc_rna, ref_rna = d["iscc"], d["real"]
    res = dict(scrna=dict(fitted_reference=d.get("reference"), iscc=dict(cv_lib=iscc_rna["cv_lib"]), tumours={}),
               visium=dict(iscc=dict(mean=float(np.mean(e["tots"])), sd=float(np.std(e["tots"])),
                                     skew=float(skew(e["tots"])), moran=e["moran_iscc"]),
                           fitted_section=dict(mean=float(np.mean(e["toto"])), sd=float(np.std(e["toto"])),
                                               skew=float(skew(e["toto"])), moran=e["moran_real"]),
                           sections={}))
    for t, X in wu_tumours(a.wu_tar).items():
        st = F2._count_stats(X)
        res["scrna"]["tumours"][t] = dict(n_cells=int(X.shape[0]), cv_lib=st["cv_lib"],
                                          iscc_distance=_curve_distance(iscc_rna, st),
                                          fitted_reference_distance=_curve_distance(ref_rna, st),
                                          curve_m=st["curve_m"], curve_z=st["curve_z"], curve_v=st["curve_v"])
        print("scrna", t, json.dumps({k: v for k, v in res["scrna"]["tumours"][t].items()
                                      if not k.startswith("curve")}), flush=True)
    res["visium"]["sections"] = wu_sections(a.wu_visium)
    for s, v in res["visium"]["sections"].items():
        print("visium", s, json.dumps({k: x for k, x in v.items() if k != "loglib"}), flush=True)
    print("visium iscc", json.dumps(res["visium"]["iscc"]), "fitted", json.dumps(res["visium"]["fitted_section"]))
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
