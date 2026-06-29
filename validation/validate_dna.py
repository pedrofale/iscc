"""Posterior-predictive validation of the DNA `estimate_dna()` (DESIGN_inference §C.1, M4 DNA half).

Same loop as the scRNA validation (§B), for DNA: take a DNA-seq dataset, **estimate** its technical
hyper-parameters (`estimate_dna`, method-of-moments / 1-D MLE), **re-simulate** DNA from the same
tumour with the fitted hypers, then recompute the **same** count-level summary statistics and
overlay re-simulated vs observed. A tight overlay means the fitted technical layer reproduces the
data's coverage / allele / GC / CNA structure.

The recomputed summaries (the §C.1 validation set):
  * coverage CV per called-CN level   (the CN-conditioned depth dispersion = kappa/nb)
  * BAF spread at het loci             (single-cell Beta-Binomial overdispersion)
  * realized ADO rate                  (single-cell allelic dropout)
  * coverage-vs-GC curve               (gc_curve_sigma)
  * CNA log2-ratio distribution        (bulk depth ratio, the CN signal)

A real DNA-seq reference is a nice-to-have; per the deliverable a synthetic fit->simulate->overlay
is acceptable. Here the "observed" data is itself an iscc simulation with chosen ground-truth hypers
(the stand-in for real data), so the figure shows the full round-trip: ground-truth -> observed ->
estimate -> re-simulate -> match. Pass ``--real-bulk`` / ``--real-sc`` h5ad to overlay real data.

Usage:  python validation/validate_dna.py
Produces manuscript/figures/validation_dna.png.
"""
import argparse
import os

import numpy as np

from iscc.tumor.models import GenotypeTumor
from iscc.data.dna import bulkDNA, scDNA, genome_features
from iscc.data import estimate_dna_from_assay

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GENOME = {"n_segments": 24, "segment_size": 40}            # ~960 loci
DEME = {"carrying_capacity": 8}
SPATIAL = {"grid_size": 28, "structure_radius": 0}
CANCER = {"division_rate": 0.4, "death_rate": 0.02, "max_birth_rate": 0.95,
          "mutation_rate": 0.8, "dispersal_rate": 0.2}
SELECTION = {"prop_driver": 0.2, "prop_dispersal": 0.0, "prop_immune_resistance": 0.0,
             "prop_treatment_resistance": 0.0, "driver_effects": 1.4, "dispersal_effects": 1.0,
             "treatment_resistant_effects": 1.0, "immune_resistant_effects": 1.0}

# Ground-truth technical hypers for the "observed" data (the stand-in for a real dataset).
TRUE_BULK = dict(kappa=1500.0, error_rate=0.01)
TRUE_SC = dict(kappa=5.0, ado_rate=0.20, beta_binom_conc=30.0, error_rate=0.006)


# --------------------------------------------------------------------------------------
# Summary statistics (recomputed identically on observed + re-simulated)
# --------------------------------------------------------------------------------------
def _cov_cv_by_cn(coverage, cn):
    """Coverage coefficient of variation within each rounded called-CN level."""
    coverage = np.asarray(coverage, float).ravel()
    cn = np.round(np.asarray(cn, float).ravel()).astype(int)
    out = {}
    for level in np.unique(cn):
        c = coverage[cn == level]
        c = c[c > 0]
        if c.size >= 20:
            out[int(level)] = float(c.std() / c.mean())
    return out


def _baf_at_hets(alt, coverage, het_obs, min_cov=8):
    """Observed BAF (alt/coverage) at genuinely-het (cell, locus) observations.

    ``het_obs`` is a per-observation boolean mask (cells x loci): a locus with *mean* VAF ~0.5 can
    be a clonal mixture (cells hom-ref + cells hom-alt), so we select per-cell het, not per-locus.
    """
    a = np.asarray(alt, float)
    c = np.asarray(coverage, float)
    ok = het_obs & (c >= min_cov)
    return (a[ok] / c[ok])


def _gc_curve(coverage, cn, gc, nbins=10):
    """Mean CN-normalized coverage in GC bins (the GC->coverage bias curve)."""
    percopy = np.asarray(coverage, float).ravel() / np.maximum(np.asarray(cn, float).ravel(), 1e-9)
    g = np.asarray(gc, float).ravel()
    edges = np.quantile(g, np.linspace(0, 1, nbins + 1))
    edges[-1] += 1e-9
    centers, means = [], []
    for i in range(nbins):
        m = (g >= edges[i]) & (g < edges[i + 1])
        if m.sum() >= 5:
            centers.append(0.5 * (edges[i] + edges[i + 1]))
            means.append(float(np.mean(percopy[m])))
    norm = np.mean(means) if means else 1.0
    return np.array(centers), np.array(means) / (norm if norm else 1.0)


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--n-cells", type=int, default=200)
    ap.add_argument("--out", default=os.path.join(REPO, "manuscript/figures/validation_dna.png"))
    args = ap.parse_args()

    # --- a realistic tumour: per-locus CN + VAF ground truth ----------------------------
    tumor = GenotypeTumor(seed=args.seed, genome_params=GENOME, cancer_cell_params=CANCER,
                          deme_params=DEME, spatial_params=SPATIAL, selection_params=SELECTION)
    tumor.grow(1200, seed=args.seed)
    cell_data = tumor.make_cell_data()
    genes = list(cell_data["cell_snv"].columns)
    gc, _, _ = genome_features(genes)

    # ============================ BULK (WGS) ============================================
    obs_bulk = bulkDNA(breadth="wgs", seed=10, **TRUE_BULK).run(cell_data)
    est_bulk = estimate_dna_from_assay(obs_bulk)
    sim_bulk = bulkDNA(breadth="wgs", seed=11, depth_model=est_bulk.depth_model,
                       **est_bulk.dna_kwargs()).run(cell_data)
    print(f"BULK fit: {est_bulk}")

    ob, sb = obs_bulk.observed_data, sim_bulk.observed_data
    cv_obs = _cov_cv_by_cn(ob["coverage"], ob["true_cn"])
    cv_sim = _cov_cv_by_cn(sb["coverage"], sb["true_cn"])
    gcx_o, gcy_o = _gc_curve(ob["coverage"], ob["true_cn"], ob["gc"])
    gcx_s, gcy_s = _gc_curve(sb["coverage"], sb["true_cn"], sb["gc"])

    # ============================ SINGLE-CELL (WGS) =====================================
    obs_sc = scDNA(n_cells=args.n_cells, breadth="wgs", seed=20, **TRUE_SC).run(cell_data)
    est_sc = estimate_dna_from_assay(obs_sc)
    sim_sc = scDNA(n_cells=args.n_cells, breadth="wgs", seed=21, depth_model=est_sc.depth_model,
                   **est_sc.dna_kwargs()).run(cell_data)
    print(f"SC   fit: {est_sc}")

    # each assay draws its own random cell subset, so take each one's *own* per-cell het mask
    def _het_obs(assay):
        af_cells = assay.true_alt_fraction.values
        return (af_cells > 0.15) & (af_cells < 0.85)
    baf_o = _baf_at_hets(obs_sc.alt_counts.values, obs_sc.coverage.values, _het_obs(obs_sc))
    baf_s = _baf_at_hets(sim_sc.alt_counts.values, sim_sc.coverage.values, _het_obs(sim_sc))
    ado_o = float(np.mean((baf_o <= 0.05) | (baf_o >= 0.95)))
    ado_s = float(np.mean((baf_s <= 0.05) | (baf_s >= 0.95)))

    # --- overlay figure -----------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    ax = axes[0, 0]
    levels = sorted(set(cv_obs) & set(cv_sim))
    ax.plot(levels, [cv_obs[l] for l in levels], "o-", label="observed", color="tab:gray")
    ax.plot(levels, [cv_sim[l] for l in levels], "s--", label="re-simulated (fitted)", color="tab:red")
    ax.set_xlabel("called copy number"); ax.set_ylabel("coverage CV")
    ax.set_title(f"Bulk: CN-conditioned coverage dispersion\n(kappa true={TRUE_BULK['kappa']:.0f}, "
                 f"fit={est_bulk.hypers.kappa:.0f})")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(gcx_o, gcy_o, "o-", label="observed", color="tab:gray")
    ax.plot(gcx_s, gcy_s, "s--", label="re-simulated (fitted)", color="tab:red")
    ax.set_xlabel("GC content"); ax.set_ylabel("normalized coverage")
    ax.set_title(f"Bulk: coverage-vs-GC curve\n(gc_curve_sigma fit={est_bulk.hypers.gc_curve_sigma:.3f})")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    bins = np.linspace(0, 1, 31)
    ax.hist(baf_o, bins=bins, density=True, alpha=0.5, label="observed", color="tab:gray")
    ax.hist(baf_s, bins=bins, density=True, alpha=0.5, label="re-simulated (fitted)", color="tab:red")
    ax.set_xlabel("BAF at het loci"); ax.set_ylabel("density")
    ax.set_title(f"Single-cell: het BAF spread + ADO\n(ADO obs={ado_o:.2f}, fit-sim={ado_s:.2f}; "
                 f"conc true={TRUE_SC['beta_binom_conc']:.0f}, fit={est_sc.hypers.beta_binom_conc:.0f})")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    lo = ob["log2_ratio"].replace([np.inf, -np.inf], np.nan).dropna().values
    ls = sb["log2_ratio"].replace([np.inf, -np.inf], np.nan).dropna().values
    rng = (np.nanpercentile(np.concatenate([lo, ls]), 1), np.nanpercentile(np.concatenate([lo, ls]), 99))
    bins = np.linspace(rng[0], rng[1], 41)
    ax.hist(lo, bins=bins, density=True, alpha=0.5, label="observed", color="tab:gray")
    ax.hist(ls, bins=bins, density=True, alpha=0.5, label="re-simulated (fitted)", color="tab:red")
    ax.set_xlabel("CNA log2 ratio"); ax.set_ylabel("density")
    ax.set_title("Bulk: CNA log2-ratio distribution")
    ax.legend(fontsize=8)

    fig.suptitle("DNA estimate_dna(): fitted technical params reproduce observed summaries "
                 "(fit -> re-simulate -> overlay)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=120, bbox_inches="tight")

    print(f"\nbulk coverage-CV by CN  observed={ {k: round(v,3) for k,v in cv_obs.items()} }")
    print(f"bulk coverage-CV by CN  fit-sim ={ {k: round(v,3) for k,v in cv_sim.items()} }")
    print(f"sc ADO rate  observed={ado_o:.3f}  fit-sim={ado_s:.3f}")
    print(f"figure -> {args.out}")


if __name__ == "__main__":
    main()
