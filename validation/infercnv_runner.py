"""Run the REAL infercnvpy (scverse's Python inferCNV) on an iscc AnnData.

Lives on the far side of the dedicated ``iscc-infercnv`` conda env so the core ``iscc`` env never
carries infercnvpy. The iscc validation script writes an AnnData (malignant + normal cells, genes
annotated with chromosome/start/end); this script runs infercnvpy's running-window smoothing and
writes the inferred per-window CNV matrix back for scoring against iscc's true copy number.

Usage:  python infercnv_runner.py <in.h5ad> <out.npz> [window_size] [step]
"""
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import scanpy as sc
import anndata as ad
import infercnvpy as cnv


def main():
    in_h5ad, out_npz = sys.argv[1], sys.argv[2]
    window_size = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    step = int(sys.argv[4]) if len(sys.argv) > 4 else 2

    adata = ad.read_h5ad(in_h5ad)

    # log-normalise the counts (inferCNV operates on log-normalised expression).
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    has_normal = "normal" in set(adata.obs["cell_type"])
    cnv.tl.infercnv(
        adata,
        reference_key="cell_type" if has_normal else None,
        reference_cat=["normal"] if has_normal else None,
        window_size=window_size,
        step=step,
        exclude_chromosomes=None,
    )
    cnv.tl.cnv_score(adata, groupby="cell_type") if has_normal else None

    X = adata.obsm["X_cnv"]
    X = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)
    chr_pos = adata.uns["cnv"]["chr_pos"]                 # {chromosome: start col index}
    chrom_names = np.array(list(chr_pos.keys()))
    chrom_starts = np.array([chr_pos[c] for c in chrom_names], dtype=int)

    np.savez_compressed(
        out_npz,
        x_cnv=X.astype(np.float32),
        obs_names=np.array(adata.obs_names, dtype=object),
        cell_type=np.array(adata.obs["cell_type"], dtype=object),
        chrom_names=chrom_names,
        chrom_starts=chrom_starts,
        n_windows=np.array([X.shape[1]]),
    )
    print(f"infercnvpy done: {X.shape[0]} cells x {X.shape[1]} windows, "
          f"{len(chrom_names)} chromosomes -> {out_npz}")


if __name__ == "__main__":
    main()
