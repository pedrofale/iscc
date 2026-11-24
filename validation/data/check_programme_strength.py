"""Held-out check of the human programme strength on 10x's V1_Breast_Cancer_Block_A_Section_1.

A section of an iscc tumour is measured with Visium at one seed for each programme setting
(``--settings scale:bound ...``), and each is compared with the real section: the share of UMIs on the
Hallmark programme genes, the share held by the 50 most expressed genes, the genes detected per spot
at 5,000-7,000 UMIs, the correlation of log10 gene shares and the median absolute log2 ratio of gene
shares over the genes above 1e-5 in the real section and in every setting. The registration is the
vault note ``exp-programme-fold-changes``.

Usage::

    python validation/data/check_programme_strength.py --real-h5 <filtered_feature_bc_matrix.h5> \\
        --tumour <.iscc> --settings 1:5 0.35:2.5 [--out <csv>]
"""
import argparse
import sys

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, "validation")
import realistic_regime as RR  # noqa: E402
from iscc.data import Visium  # noqa: E402
from iscc.data.human import HumanGenomeLayer, load_hallmark_programs  # noqa: E402
from iscc.tumor.models import GenotypeTumor  # noqa: E402


def spot_stats(C, prog):
    C = sp.csr_matrix(C)
    umi = np.asarray(C.sum(axis=1)).ravel()
    det = np.diff(C.indptr)
    mid = (umi > 5000) & (umi < 7000)
    tot = np.asarray(C.sum(axis=0)).ravel().astype(float)
    share = tot / tot.sum()
    return share, dict(n_spots=int(len(umi)), median_umis=float(np.median(umi)),
                       genes_at_5k_7k=float(np.median(det[mid])) if mid.any() else np.nan,
                       programme_share=float(share[prog].sum()),
                       top50_share=float(np.sort(share)[::-1][:50].sum()))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--real-h5", required=True)
    ap.add_argument("--tumour", required=True)
    ap.add_argument("--settings", nargs="+", required=True, help="scale:bound pairs")
    ap.add_argument("--max-cells", type=int, default=60_000)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    t = GenotypeTumor.load(a.tumour)
    layer = HumanGenomeLayer(t)
    genes = layer.genes["gene_id"].to_numpy()
    hall = load_hallmark_programs()
    prog = np.isin(genes, hall["gene_id"])
    with h5py.File(a.real_h5, "r") as f:
        m = f["matrix"]
        X = sp.csc_matrix((m["data"][()], m["indices"][()], m["indptr"][()]), shape=tuple(m["shape"][()]))
        ids = [b.decode() for b in m["features/id"][()]]
    pos = pd.Index(ids).get_indexer(genes)
    R = X.T.tocsr()[:, np.maximum(pos, 0)].tocsc()
    R = R @ sp.diags((pos >= 0).astype(float))                      # genes the section lacks: 0
    R.eliminate_zeros()
    real, row = spot_stats(R, prog)
    rows = [dict(setting="real", **row)]
    shares = {}
    sec = RR.section(t, depth_frac=0.5, max_cells=a.max_cells)
    for setting in a.settings:
        s, c = (float(x) for x in setting.split(":"))
        vz = Visium(seed=a.seed).run(sec, gene_lift=layer.expression(t, programme_scale=s, programme_max_log_fold=c))
        C = sp.csr_matrix(vz.spot_counts.to_numpy()[np.asarray(vz.obs["n_cells"]) > 0])
        del vz
        shares[setting], row = spot_stats(C, prog)
        rows.append(dict(setting=setting, **row))
        print(rows[-1], flush=True)
    common = real > 1e-5
    for sh in shares.values():
        common &= sh > 1e-5
    det = real > 0
    for r in rows[1:]:
        sh = shares[r["setting"]]
        r["r_log_share"] = float(np.corrcoef(np.log10(real[det] + 1e-7), np.log10(sh[det] + 1e-7))[0, 1])
        r["median_abs_log2"] = float(np.median(np.abs(np.log2(sh[common] / real[common]))))
        r["n_genes_common"] = int(common.sum())
    out = pd.DataFrame(rows)
    print(out.round(4).to_string(index=False))
    if a.out:
        out.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
