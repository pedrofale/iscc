"""Run the REAL cNMF (consensus NMF) on an iscc AnnData — the comparator for the scDEF benchmark.

Lives on the far side of the dedicated ``iscc-cnmf`` conda env, per the one-env-per-external-tool
convention (`validation/README_integration.md`). cNMF (Kotliar et al., eLife 2019) is the
field-standard gene-expression-program method: it runs NMF many times over and keeps the consensus
components, so it is the natural FLAT baseline against scDEF's HIERARCHICAL Bayesian factor model.

Output mirrors `scdef_runner.py` so `validate_programs.py` can score both through one path:
  * ``loadings`` (K x genes) — the consensus spectra, scored against the true `loading`.
  * ``activities`` (cells x K) — the usage matrix, scored against the true per-cell `z`.
cNMF has no hierarchy, so `hierarchy` is empty — that asymmetry IS part of the comparison.

NOTE on genes: cNMF ordinarily fits on a high-variance gene subset (`num_highvar_genes=2000`). We
pass the FULL gene set, because the benchmark scores the inferred loadings against the true loading
matrix in full gene space — an HVG filter would silently drop the weakly-loaded tail of every true
program (which `loading_sparsity` deliberately creates) and flatter the recovery scores.

Usage:  python cnmf_runner.py <in.h5ad> <out.npz> [K] [n_iter] [seed]
"""
import os
import shutil
import sys
import tempfile
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import anndata as ad
from cnmf import cNMF

# cNMF's consensus step drops outlier NMF replicates beyond this density; 2.0 is permissive (keeps
# essentially every replicate), which is what we want for a small, clean benchmark matrix.
DENSITY_THRESHOLD = 2.0


def _align(spectra, var_names, K, n_genes):
    """Return a (K x n_genes) float32 matrix aligned to `var_names`, whichever way `spectra` is laid out.

    cNMF's `load_results` hands back the spectra as a DataFrame of (genes x K) — gene names on the
    INDEX, program ids on the COLUMNS — which is the transpose of the (K x genes) the name suggests.
    Rather than hard-code an orientation (getting it wrong silently yields an all-zero matrix and a
    loading cosine of exactly 0, which reads as "the tool failed" instead of "the scorer is broken"),
    detect it by asking which axis actually carries the gene names.
    """
    gene_index = {g: i for i, g in enumerate(var_names)}
    idx_hits = sum(1 for g in spectra.index if g in gene_index)
    col_hits = sum(1 for g in spectra.columns if g in gene_index)
    if idx_hits >= col_hits:
        genes_axis, values = list(spectra.index), np.asarray(spectra.values)          # (genes, K)
    else:
        genes_axis, values = list(spectra.columns), np.asarray(spectra.values).T      # (genes, K)
    if max(idx_hits, col_hits) == 0:
        raise RuntimeError("cNMF spectra carry no recognisable gene names on either axis "
                           f"(index[:3]={list(spectra.index[:3])}, columns[:3]={list(spectra.columns[:3])})")

    out = np.zeros((K, n_genes), dtype=np.float32)
    for j, g in enumerate(genes_axis):
        i = gene_index.get(g)
        if i is not None:
            out[:, i] = values[j, :]
    return out


def main():
    in_h5ad, out_npz = sys.argv[1], sys.argv[2]
    K = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    n_iter = int(sys.argv[4]) if len(sys.argv) > 4 else 20
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0

    adata = ad.read_h5ad(in_h5ad)
    var_names_full = list(adata.var_names)               # alignment target: the FULL gene space
    n_genes = len(var_names_full)

    # Drop genes with zero counts in every cell. cNMF normalises to TPM, so an all-zero gene divides
    # by zero -> "NaNs in normalized counts matrix" -> factorize() dies. Such genes are unassignable
    # by ANY method (no signal at all), and iscc produces them routinely: a nullisomic segment is
    # genuinely silent, and oncogenes sit at baseline 0.01 so Poisson rounds them to zero. They are
    # dropped from the FIT and left at zero loading in the aligned output.
    expressed = np.asarray(adata.X.sum(axis=0)).ravel() > 0
    fit_adata = adata[:, expressed].copy()
    n_fit = int(expressed.sum())
    if n_fit < K:
        raise RuntimeError(f"only {n_fit} expressed genes; cannot fit K={K} programs")

    workdir = tempfile.mkdtemp(prefix="iscc_cnmf_")
    try:
        # cNMF reads counts from a file of its own; hand it the expressed-gene subset.
        counts_fn = os.path.join(workdir, "counts.h5ad")
        fit_adata.write_h5ad(counts_fn)

        obj = cNMF(output_dir=workdir, name="iscc")
        obj.prepare(counts_fn=counts_fn, components=[K], n_iter=n_iter, seed=seed,
                    num_highvar_genes=n_fit)             # full expressed gene space — see docstring
        obj.factorize(worker_i=0, total_workers=1)
        obj.combine()
        obj.consensus(k=K, density_threshold=DENSITY_THRESHOLD, show_clustering=False,
                      close_clustergram_fig=True)

        usage, spectra_scores, spectra_tpm, top_genes = obj.load_results(
            K=K, density_threshold=DENSITY_THRESHOLD)

        # Which matrix is the "loading"? `spectra_tpm` is the program's ABSOLUTE expression in TPM,
        # so its magnitude is dominated by each gene's baseline abundance rather than by what the
        # program DOES to that gene — cosine against a true log-fold loading would mostly measure the
        # baseline. `spectra_scores` is cNMF's z-scored gene signature (the matrix it ranks genes by),
        # which is the right analogue of the true loading. We emit scores as `loadings` and keep TPM
        # alongside so the scorer can report both.
        loadings = _align(spectra_scores, var_names_full, K, n_genes)
        loadings_tpm = _align(spectra_tpm, var_names_full, K, n_genes)

        activities = np.asarray(usage.values, dtype=np.float32)  # (cells, K)

        np.savez_compressed(
            out_npz,
            loadings=loadings,
            loadings_tpm=loadings_tpm,
            activities=activities,
            hierarchy=np.zeros((0, K), dtype=np.float32),   # cNMF is flat, by construction
            layer_sizes=np.asarray([K], dtype=int),
            var_names=np.array(var_names_full, dtype=object),
            obs_names=np.array(usage.index, dtype=object),
        )
        print(f"cNMF done: {activities.shape[0]} cells x {K} programs x {n_fit}/{n_genes} "
              f"expressed genes -> {out_npz}")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
