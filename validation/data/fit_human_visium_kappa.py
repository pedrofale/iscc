"""Fit the Dirichlet-multinomial concentration of Visium counts over the human genes (HUMAN_VISIUM_KAPPA).

``estimate_visium`` (count_model "dm") on the in-tissue spots of 10x's V1_Breast_Cancer_Block_A_Section_1
(Space Ranger raw matrix and tissue positions, the section iscc's Visium defaults are calibrated on),
over the GRCh38-2020-A genes the human genome layer emits. Also reports the fit over all 36,601 genes.

Usage::

    python validation/data/fit_human_visium_kappa.py --raw-h5 <raw_feature_bc_matrix.h5> \\
        --positions <tissue_positions_list.csv>
"""
import argparse

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

from iscc.data import estimate_visium
from iscc.data.human import HumanGenomeLayer, arm_genome, load_reference


def read_10x_h5(path):
    with h5py.File(path, "r") as f:
        m = f["matrix"]
        shape = tuple(m["shape"][()])
        X = sp.csc_matrix((m["data"][()], m["indices"][()], m["indptr"][()]), shape=shape)
        bcs = [b.decode() for b in m["barcodes"][()]]
        ids = [b.decode() for b in m["features/id"][()]]
    return X.T.tocsr(), bcs, ids                     # barcodes x features


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw-h5", required=True)
    ap.add_argument("--positions", required=True)
    args = ap.parse_args()
    X, bcs, ids = read_10x_h5(args.raw_h5)
    pos = pd.read_csv(args.positions, header=None,
                      names=["barcode", "in_tissue", "array_row", "array_col", "pxl_row", "pxl_col"])
    pos = pos.set_index("barcode").reindex(bcs)
    tissue = pos["in_tissue"].to_numpy() == 1
    coords = pos.loc[tissue, ["array_row", "array_col"]].to_numpy(float)
    Xt = X[tissue]
    print(f"spots: {len(bcs)} on the slide, {tissue.sum()} in tissue; features {len(ids)}; "
          f"median UMIs per in-tissue spot {np.median(np.asarray(Xt.sum(1)).ravel()):.0f}")
    ref = load_reference(snps=False)
    layer = HumanGenomeLayer(genome_params=arm_genome(genes_per_segment=4, reference=ref)["genome_params"],
                             reference=ref)
    emitted = pd.Index(ids).get_indexer(layer.genes["gene_id"])
    for label, cols in (("all features", np.arange(len(ids))), ("the layer's genes", emitted[emitted >= 0])):
        est = estimate_visium(Xt[:, cols].toarray(), count_model="dm", coords=coords)
        print(f"{label} ({len(cols)} genes): kappa = {est.hypers.kappa:,.0f}; "
              f"mu_counts {est.hypers.mu_counts:,.0f}, sigma_counts {est.hypers.sigma_counts:.3f}")


if __name__ == "__main__":
    main()
