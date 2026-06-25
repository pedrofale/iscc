"""Validation utilities: compare simulated data against real data.

Computes the standard single-cell count-realism summary statistics used to benchmark
scRNA-seq simulators (library size, mean expression, overdispersion, dropout/zero fraction)
and overlays a simulated dataset on a real reference. Designed to take any two cells x genes
count matrices (numpy arrays or AnnData), so it works for both the genotype and cell engines
and for any real reference dataset.
"""
import numpy as np


# --- evolutionary-mode metrics ----------------------------------------------
def clone_diversity(counts):
    """Shannon diversity of clone (genotype) frequencies."""
    c = np.asarray([v for v in counts], dtype=float)
    c = c[c > 0]
    if c.sum() == 0:
        return 0.0
    p = c / c.sum()
    return float(-(p * np.log(p)).sum())


def spatial_assortment(labels_by_deme, grid_size):
    """Fraction of adjacent occupied demes that share the same dominant clone label.

    A high value means clones occupy spatially coherent patches (clonal expansion); a low
    value means lineages are finely intermixed in space. `labels_by_deme` maps a linear
    deme index (row-major over a grid_size x grid_size grid) to its dominant label.
    """
    match = total = 0
    for idx, lab in labels_by_deme.items():
        r, c = divmod(idx, grid_size)
        for nr, nc in [(r, c + 1), (r + 1, c)]:  # right & down neighbours (each pair once)
            if 0 <= nr < grid_size and 0 <= nc < grid_size:
                j = nr * grid_size + nc
                if j in labels_by_deme:
                    total += 1
                    match += labels_by_deme[j] == lab
    return match / total if total else float("nan")


# --- SNV site-frequency-spectrum metrics ------------------------------------
def population_vaf(tumor, cancer_only=True):
    """Bulk variant allele frequencies across the tumor's cancer cell population.

    For every genomic site, VAF = (mutated allele copies summed over cells) /
    (total allele copies summed over cells), i.e. exactly what a bulk WGS assay
    of the (cancer) cells would report. Works on the genotype-count engine by
    weighting each genotype's per-site mutated-copy count by its cell count.
    """
    n_sites = tumor.n_segments * tumor.segment_size
    num = np.zeros(n_sites)
    den = np.zeros(n_sites)
    for gid, cnt in tumor.genotypes_counts.items():
        if cancer_only and not tumor._is_cancer(gid):
            continue
        rep = tumor.genotypes[gid]
        for seg in range(rep.n_segments):
            copies = rep.genome[seg]["p"] + rep.genome[seg]["m"]
            cn = rep.genome_summary["seg_cns"][seg]
            sl = slice(seg * rep.segment_size, (seg + 1) * rep.segment_size)
            den[sl] += cnt * cn
            if copies and cn > 0:
                num[sl] += cnt * np.sum(copies, axis=0)
    return np.divide(num, den, out=np.zeros_like(num), where=den > 0)


def site_frequency_spectrum(freqs, f_min=0.1, f_max=0.5, n_bins=50):
    """Cumulative number of mutations with VAF >= f, over a grid of f in [f_min, f_max].

    Returns (f_grid, M) where M[i] is the count of sites with frequency >= f_grid[i].
    """
    f = np.asarray(freqs, dtype=float)
    f = f[f > 0]
    grid = np.linspace(f_min, f_max, n_bins)
    M = np.array([(f >= g).sum() for g in grid], dtype=float)
    return grid, M


def neutral_sfs_rsq(freqs, f_min=0.1, f_max=0.5, n_bins=50):
    """Goodness-of-fit (R^2) of the cumulative SFS to the neutral 1/f power law.

    Under neutral exponential growth (Williams et al. 2016, Nat Genet) the cumulative
    number of subclonal mutations is M(f) = (mu/beta)(1/f - 1/f_max), i.e. linear in 1/f.
    A high R^2 indicates neutral dynamics; selection (clonal sweeps / subclones) distorts
    the spectrum and lowers it. Returns (rsq, slope) with slope ~ mutation rate / mu_beta.
    """
    grid, M = site_frequency_spectrum(freqs, f_min, f_max, n_bins)
    x = 1.0 / grid
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, M, rcond=None)
    pred = A @ coef
    ss_res = float(((M - pred) ** 2).sum())
    ss_tot = float(((M - M.mean()) ** 2).sum())
    rsq = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(rsq), float(coef[0])


# --- copy-number-alteration metrics -----------------------------------------
def segment_copy_numbers(tumor, cancer_only=True):
    """Population mean copy number per genome segment (count-weighted over genotypes)."""
    num = np.zeros(tumor.n_segments)
    den = 0.0
    for gid, cnt in tumor.genotypes_counts.items():
        if cancer_only and not tumor._is_cancer(gid):
            continue
        seg_cns = tumor.genotypes[gid].genome_summary["seg_cns"]
        num += cnt * np.asarray(seg_cns, dtype=float)
        den += cnt
    return num / den if den > 0 else num


def segment_driver_content(tumor):
    """Net oncogenic content per segment: (#oncogene positions) - (#TSG positions)."""
    dt = tumor.selection.driver_types
    return np.array([int((dt[s] == 1).sum()) - int((dt[s] == -1).sum())
                     for s in range(tumor.n_segments)], dtype=float)


def cna_amplification_signature(tumors):
    """Pooled relationship between segment copy number and net oncogene content.

    Cancer CNA landscapes recurrently amplify oncogenes and delete tumor suppressors
    (Beroukhim et al. 2010; Davoli et al. 2013). Under the CINner-style selection model this
    should appear as a positive correlation between a segment's net oncogene content and its
    population copy number; under neutral drift there is no such correlation. Pools (net, CN)
    pairs over a list of tumors (each may have a different random driver layout).

    Returns (net, cn, slope, corr) where net/cn are the pooled per-segment arrays.
    """
    net, cn = [], []
    for t in tumors:
        if t.get_cancer_size() == 0:
            continue
        net.extend(segment_driver_content(t))
        cn.extend(segment_copy_numbers(t))
    net, cn = np.asarray(net), np.asarray(cn)
    if net.size < 2 or net.std() == 0 or cn.std() == 0:
        return net, cn, float("nan"), float("nan")
    slope = np.polyfit(net, cn, 1)[0]
    corr = float(np.corrcoef(net, cn)[0, 1])
    return net, cn, float(slope), corr


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
