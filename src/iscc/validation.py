"""Validation utilities: compare simulated data against real data.

Computes the standard single-cell count-realism summary statistics used to benchmark
scRNA-seq simulators (library size, mean expression, overdispersion, dropout/zero fraction)
and overlays a simulated dataset on a real reference. Designed to take any two cells x genes
count matrices (numpy arrays or AnnData), so it works for both the genotype and cell engines
and for any real reference dataset.
"""
import numpy as np


def _counts(x):
    """Accept an AnnData, DataFrame, or ndarray and return a dense cells x genes array."""
    if hasattr(x, "X"):
        x = x.X
    if hasattr(x, "toarray"):
        x = x.toarray()
    if hasattr(x, "values"):
        x = x.values
    return np.asarray(x, dtype=float)


def summary_stats(counts):
    """Per-cell and per-gene count-realism statistics."""
    c = _counts(counts)
    lib = c.sum(axis=1)                              # library size per cell
    gene_mean = c.mean(axis=0)                       # mean expression per gene
    gene_var = c.var(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        gene_cv2 = np.where(gene_mean > 0, gene_var / gene_mean ** 2, np.nan)  # overdispersion
    gene_zero = (c == 0).mean(axis=0)               # dropout per gene
    cell_zero = (c == 0).mean(axis=1)               # zero fraction per cell
    return dict(lib_size=lib, gene_mean=gene_mean, gene_cv2=gene_cv2,
                gene_zero=gene_zero, cell_zero=cell_zero,
                median_lib=float(np.median(lib)),
                cv_lib=float(np.std(lib) / (np.mean(lib) + 1e-9)),
                mean_zero_frac=float((c == 0).mean()))


def compare_plot(sim, real, out_path=None, sim_label="iscc (simulated)", real_label="real"):
    """Overlay simulated vs real count-realism statistics; returns the matplotlib figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s, r = summary_stats(sim), summary_stats(real)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    axes[0].hist(np.log1p(s["lib_size"]), bins=40, alpha=0.5, density=True, label=sim_label)
    axes[0].hist(np.log1p(r["lib_size"]), bins=40, alpha=0.5, density=True, label=real_label)
    axes[0].set_xlabel("log1p(library size)"); axes[0].set_ylabel("density")
    axes[0].set_title("Library size"); axes[0].legend(fontsize=8)

    axes[1].scatter(np.log1p(s["gene_mean"]), np.log1p(s["gene_cv2"]), s=6, alpha=0.4, label=sim_label)
    axes[1].scatter(np.log1p(r["gene_mean"]), np.log1p(r["gene_cv2"]), s=6, alpha=0.4, label=real_label)
    axes[1].set_xlabel("log1p(mean)"); axes[1].set_ylabel("log1p(CV$^2$)")
    axes[1].set_title("Mean–variance (overdispersion)"); axes[1].legend(fontsize=8)

    axes[2].scatter(np.log1p(s["gene_mean"]), s["gene_zero"], s=6, alpha=0.4, label=sim_label)
    axes[2].scatter(np.log1p(r["gene_mean"]), r["gene_zero"], s=6, alpha=0.4, label=real_label)
    axes[2].set_xlabel("log1p(mean)"); axes[2].set_ylabel("zero fraction")
    axes[2].set_title("Dropout curve"); axes[2].legend(fontsize=8)

    axes[3].hist(s["cell_zero"], bins=40, alpha=0.5, density=True, label=sim_label)
    axes[3].hist(r["cell_zero"], bins=40, alpha=0.5, density=True, label=real_label)
    axes[3].set_xlabel("zero fraction per cell"); axes[3].set_ylabel("density")
    axes[3].set_title("Cell sparsity"); axes[3].legend(fontsize=8)

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
    return fig, dict(sim=s, real=r)
