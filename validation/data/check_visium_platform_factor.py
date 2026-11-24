"""Held-out check of the Visium platform factor on 10x's V1_Breast_Cancer_Block_A_Section_1.

A section of an iscc immune-variant tumour is measured with Visium twice at the same seed, with the
packaged platform factor and without it, and each pseudobulk is compared with the real section's
(another tumour and another lab than the Wu et al. 2021 tumours the factor is fitted on). Metrics over
genes: the Pearson correlation of log10(share + 1e-7) over the genes the real section detects, the
median absolute log2 ratio over the genes above a share of 1e-5 in both, and the chrM share. The
registration is the vault note ``exp-visium-platform-factor``.

Usage::

    python validation/data/check_visium_platform_factor.py --real-h5 <filtered_feature_bc_matrix.h5> \\
        --tumour <registry .iscc file> [--out <csv>]
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
from iscc.data.human import HumanGenomeLayer  # noqa: E402
from iscc.tumor.models import GenotypeTumor  # noqa: E402


def real_shares(path, genes):
    with h5py.File(path, "r") as f:
        m = f["matrix"]
        X = sp.csc_matrix((m["data"][()], m["indices"][()], m["indptr"][()]), shape=tuple(m["shape"][()]))
        ids = [b.decode() for b in m["features/id"][()]]
    tot = pd.Series(np.asarray(X.sum(axis=1)).ravel(), index=ids)
    tot = tot[~tot.index.duplicated()].reindex(genes).fillna(0.0)
    return tot / tot.sum(), X.shape[1]


def metrics(real, sim, chrm):
    det = real > 0
    lr, ls = np.log10(real[det] + 1e-7), np.log10(sim[det] + 1e-7)
    both = (real > 1e-5) & (sim > 1e-5)
    return dict(r_log_share=float(np.corrcoef(lr, ls)[0, 1]),
                median_abs_log2=float(np.median(np.abs(np.log2(sim[both] / real[both])))),
                n_genes_r=int(det.sum()), n_genes_ratio=int(both.sum()), chrM_share=float(sim[chrm].sum()))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--real-h5", required=True)
    ap.add_argument("--tumour", required=True)
    ap.add_argument("--max-cells", type=int, default=60_000)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    t = GenotypeTumor.load(a.tumour)
    layer = HumanGenomeLayer(t)
    genes = layer.genes["gene_id"].to_numpy()
    chrm = (layer.genes["chrom"] == "chrM").to_numpy()
    real, n_real = real_shares(a.real_h5, genes)
    real = real.to_numpy()
    sec = RR.section(t, depth_frac=0.5, max_cells=a.max_cells)
    rows = [dict(arm="real", n_spots=n_real, chrM_share=float(real[chrm].sum()))]
    shares = {}
    for arm, platform in (("without", None), ("with", "default")):
        vz = Visium(seed=a.seed).run(sec, gene_lift=layer.expression(t, visium_platform=platform))
        tissue = np.asarray(vz.obs["n_cells"]) > 0
        tot = vz.spot_counts.to_numpy()[tissue].sum(axis=0).astype(float)
        shares[arm] = tot / tot.sum()
        rows.append(dict(arm=arm, n_spots=int(tissue.sum()), **metrics(real, shares[arm], chrm)))
        print(rows[-1], flush=True)
    # sensitivity: the ratio over one gene set for both arms (above 1e-5 in the real and both simulated)
    common = (real > 1e-5) & (shares["without"] > 1e-5) & (shares["with"] > 1e-5)
    for row in rows[1:]:
        sim = shares[row["arm"]]
        row["median_abs_log2_common"] = float(np.median(np.abs(np.log2(sim[common] / real[common]))))
        row["n_genes_common"] = int(common.sum())
    out = pd.DataFrame(rows)
    print(out.round(4).to_string(index=False))
    if a.out:
        out.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
