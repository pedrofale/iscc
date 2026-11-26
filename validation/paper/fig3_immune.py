"""S10 Fig: immune and stromal signal in tumour-rich spots, iscc against seven real breast cancer sections.

Each section is read through marker-gene shares of the counts of its spots: the share of the T-cell, all
immune (PTPRC) and fibroblast markers in the quarter of spots with the most epithelial signal, relative to
the quarter with the least. Real: the 10x breast section (V1_Breast_Cancer_Block_A_Section_1, the reference
of Figure 3) and the six sections of Wu et al. 2021 (Zenodo 4739739). iscc: the pregrown tumour with immune
cells (``iscc.pregrown.immune``, ``$ISCC_PREGROWN_DIR``), four windows of 70 demes through the regime's
Visium over the human genes, pooled.

The analysis of iscc-markdown experiments/data/library-content-probe/immune_in_tumour.py and
immune_in_tumour_wu.py (2026-11-26), in one script; it writes ``<derived>/fig3/immune.json``.

    ISCC_PREGROWN_DIR=... python validation/paper/fig3_immune.py --wu-visium <dir> --out <derived>/fig3/immune.json
"""
import argparse
import gzip
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import scipy.io as sio

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

SETS = {"epithelial": ["EPCAM", "KRT8", "KRT18", "KRT19", "KRT7", "CDH1"],
        "T": ["CD3D", "CD3E", "CD2", "TRAC", "CD8A"],
        "pan_immune": ["PTPRC"],
        "fibroblast": ["COL1A1", "COL1A2", "COL3A1", "DCN", "LUM"]}
WU = ("1142243F", "1160920F", "CID4290", "CID4465", "CID44971", "CID4535")


def ratios(counts, names):
    """Each marker share in the epithelium-richest quarter of spots over the poorest quarter."""
    import scipy.sparse as sp
    X = sp.csr_matrix(counts)
    tot = np.asarray(X.sum(1)).ravel().astype(float)
    on = tot > 0
    sh = {k: np.asarray(X[on][:, np.isin(names, g)].sum(1)).ravel() / tot[on] for k, g in SETS.items()}
    e = sh["epithelial"]
    hi, lo = e >= np.quantile(e, 0.75), e <= np.quantile(e, 0.25)
    return {k: float(sh[k][hi].mean() / max(sh[k][lo].mean(), 1e-12)) for k in SETS if k != "epithelial"}


def _open(path):
    with open(path, "rb") as f:
        gz = f.read(2) == b"\x1f\x8b"
    return gzip.open(path, "rt") if gz else open(path, "rt")


def wu_sections(root):
    out = {}
    for s in WU:
        d = os.path.join(root, "filtered_count_matrices", f"{s}_filtered_count_matrix")
        X = sio.mmread(_open(os.path.join(d, "matrix.mtx.gz"))).tocsr().T.tocsr()
        feats = pd.read_csv(_open(os.path.join(d, "features.tsv.gz")), sep="\t", header=None)
        names = (feats[1] if feats.shape[1] > 1 else feats[0]).astype(str).to_numpy()
        out[s] = ratios(X, names)
    return out


def tenx_section():
    sys.path.insert(0, os.path.join(REPO, "validation"))
    import validate_visium as VV
    cwd = os.getcwd()
    os.chdir(REPO)
    try:
        real = VV.load_real_visium()
    finally:
        os.chdir(cwd)
    return ratios(real.X, np.asarray(real.var_names))


def iscc_immune():
    import iscc.pregrown
    import regime as RG
    from iscc.data import Visium
    from iscc.data.human import HumanGenomeLayer
    from iscc.sample import Resection
    t = iscc.pregrown.immune()
    layer = HumanGenomeLayer(t)
    expr = layer.expression(t)
    name_of = dict(zip(layer.genes["gene_id"], layer.genes["gene_name"]))
    q = t.grid_size // 4
    spec = Resection(t)
    parts, names = [], None
    for k, (r, c) in enumerate([(q, q), (q, 3 * q), (3 * q, q), (3 * q, 3 * q)]):
        sec = spec.slice(t.primary_window(side=70, center=(r, c)), depth_frac=0.5, max_cells=10 ** 7, layers="visium")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            v = Visium(seed=1 + k, section_frac=1.0, **RG.VISIUM).run(sec, grid_side=70, gene_lift=expr)
        on = (v.obs["n_cells"] > 0).to_numpy()
        if names is None:
            names = np.array([name_of.get(g, g) for g in v.spot_counts.columns])
        parts.append(v.spot_counts.to_numpy()[on])
        del v, sec
    return ratios(np.vstack(parts), names)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--wu-visium", required=True, help="dir with filtered_count_matrices/")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = dict(statistic="marker share in the epithelium-richest quarter of spots over the poorest quarter",
               real={"10x section": tenx_section(), **wu_sections(a.wu_visium)})
    print("real", json.dumps(res["real"]), flush=True)
    res["iscc"] = iscc_immune()
    print("iscc", json.dumps(res["iscc"]), flush=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
