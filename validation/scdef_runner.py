"""Run the REAL scDEF on an iscc AnnData and write its factors back for scoring.

Lives on the far side of the dedicated ``iscc-scdef`` conda env so the core ``iscc`` env never
carries scDEF (jax + a heavy Bayesian stack). The iscc validation script (`validate_programs.py`)
writes an AnnData of scRNA counts; this script fits scDEF's hierarchical Bayesian factor model and
writes the inferred gene signatures + per-cell factor activities back, to be scored against iscc's
TRUE program `loading` matrix and per-cell `z` (`program_truth`) — the ground truth no real dataset
provides.

What we pull out of the fitted model (all posterior means, `scDEF.pmeans`):
  * ``L0W`` (n_factors x n_genes) — the gene LOADINGS of the finest layer. Scored against the true
    `loading` by cosine similarity, and thresholded into a gene set for Jaccard/AUPRC.
  * ``L0z`` (n_cells x n_factors) — per-cell factor ACTIVITIES. Scored against the true per-cell `z`.
  * ``L1W`` (n_L1 x n_L0) — the HIERARCHY: how the coarse layer composes the fine factors. Lets the
    benchmark ask whether scDEF's inferred hierarchy LEVEL matches the true program granularity,
    which is the thing scDEF offers over a flat factor model like cNMF.

`batch_key` (optional) hands scDEF its OWN batch-correction path: the column of `adata.obs` holding
the batch label, which scDEF models internally. That is what a practitioner would actually do, and it
is the honest way to test batch correction here — hand-rolling a log-space centering and exponentiating
back to counts mangles the matrix a Poisson-gamma model expects (negatives clip to zero).

Usage:  python scdef_runner.py <in.h5ad> <out.npz> [n_factors] [n_epoch] [seed] [batch_key]
"""
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import anndata as ad
import scdef


def main():
    in_h5ad, out_npz = sys.argv[1], sys.argv[2]
    n_factors = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    n_epoch = int(sys.argv[4]) if len(sys.argv) > 4 else 200
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    batch_key = sys.argv[6] if len(sys.argv) > 6 and sys.argv[6] not in ("", "none") else None

    adata = ad.read_h5ad(in_h5ad)
    # scDEF wants raw counts; the validation script writes them in X (and, defensively, `counts`).
    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()

    kw = {}
    if batch_key is not None:
        if batch_key not in adata.obs:
            raise RuntimeError(f"batch_key {batch_key!r} not in adata.obs ({list(adata.obs)})")
        kw["batch_key"] = batch_key
    model = scdef.scDEF(adata, counts_layer="counts", n_factors=n_factors, seed=seed, **kw)
    model.fit(n_epoch=n_epoch)

    pm = model.pmeans
    loadings = np.asarray(pm["L0W"], dtype=np.float32)      # (K, genes)
    activities = np.asarray(pm["L0z"], dtype=np.float32)    # (cells, K)
    # The coarse layer's composition of the fine factors (the hierarchy). Absent if only one layer.
    hierarchy = np.asarray(pm.get("L1W", np.zeros((0, loadings.shape[0]))), dtype=np.float32)

    np.savez_compressed(
        out_npz,
        loadings=loadings,
        activities=activities,
        hierarchy=hierarchy,
        layer_sizes=np.asarray(getattr(model, "layer_sizes", []), dtype=int),
        var_names=np.array(adata.var_names, dtype=object),
        obs_names=np.array(adata.obs_names, dtype=object),
    )
    print(f"scDEF done: {activities.shape[0]} cells x {loadings.shape[0]} factors "
          f"x {loadings.shape[1]} genes -> {out_npz}")


if __name__ == "__main__":
    main()
