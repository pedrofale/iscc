"""Posterior-predictive validation of the DNA `estimate_dna()` (DESIGN_inference §C.1, M4 DNA half).

Same loop as the scRNA (§B) / Visium (§C.2) validations, for DNA: take a DNA-seq dataset,
**estimate** its technical hyper-parameters (`estimate_dna`, method-of-moments / 1-D MLE),
**re-simulate** DNA from a tumour with the fitted hypers, then recompute the **same** count-level
summary statistics and overlay re-simulated vs observed. A tight overlay means the fitted technical
layer reproduces the data's coverage / allele / CN structure.

The recomputed summaries (the §C.1 validation set):
  * coverage CV per called-CN level   (the CN-conditioned depth dispersion = kappa/nb)
  * per-locus coverage distribution    (mu_depth + dispersion)
  * BAF spread at het loci + ADO       (single-cell Beta-Binomial overdispersion / allelic dropout)

DEFAULTS TO REAL DATA — the DNA analogue of scRNA fitting PBMC3k and Visium fitting `visium_sge`.
One real dataset per DNA regime (built by `validation/data/build_dna_reference.py`, reduced to the
small coverage/alt/called-CN inputs `estimate_dna` consumes; raw sources are never required):

  * BULK WGS         — **GIAB HG002** (NIST GiaB; the WGS technical gold standard).
  * SC PANEL         — **Mission Bio Tapestri** (targeted scDNA; deep, clear het calls).   [--sc-source tapestri]
  * SC WGS (default) — **DLP+ OV2295** (Shah lab; the scWGS copy-number benchmark).         [--sc-source dlp]

`estimate_dna` is fit on the real reduction (so the technical magnitudes are LEARNED), then the
technical layer is re-simulated on a synthetic tumour and the summaries overlaid (the biology
differs; the TECHNICAL summaries — depth distribution, CN-conditioned dispersion, het BAF — are what
is validated, exactly as scRNA fits PBMC3k then simulates on a tumour). The fitted `mu_depth` is a
per-locus magnitude, so it transfers across the differing locus counts directly (HG002 ~hundreds x
pooled, Tapestri ~tens x, DLP ~hundreds-of-reads/bin) — the re-sim depth matches the fit by
construction (the DNA analogue of the Visium coord-unit normalization). `--synthetic` runs the
offline ground-truth round-trip instead (also the automatic fallback when a reference is absent).

Usage:  python validation/validate_dna.py [--sc-source {dlp,tapestri}] [--synthetic]
Produces manuscript/figures/validation_dna.png.
"""
import argparse
import os
import sys

import numpy as np

from iscc.tumor.models import GenotypeTumor
from iscc.data.dna import bulkDNA, scDNA, genome_features
from iscc.data import estimate_dna, estimate_dna_from_assay

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "validation", "data"))
import build_dna_reference as B  # noqa: E402  (reducer + cache loader)

GENOME = {"n_segments": 24, "segment_size": 40}            # ~960 loci
DEME = {"carrying_capacity": 8}
SPATIAL = {"grid_size": 28, "structure_radius": 0}
CANCER = {"division_rate": 0.4, "death_rate": 0.02, "max_birth_rate": 0.95,
          "mutation_rate": 0.8, "dispersal_rate": 0.2}
SELECTION = {"prop_driver": 0.2, "prop_dispersal": 0.0, "prop_immune_resistance": 0.0,
             "prop_treatment_resistance": 0.0, "driver_effects": 1.4, "dispersal_effects": 1.0,
             "treatment_resistant_effects": 1.0, "immune_resistant_effects": 1.0}

# Ground-truth technical hypers for the synthetic round-trip (the offline stand-in for real data).
TRUE_BULK = dict(kappa=1500.0, error_rate=0.01)
TRUE_SC = dict(kappa=5.0, ado_rate=0.20, beta_binom_conc=30.0, error_rate=0.006)


# --------------------------------------------------------------------------------------
# Summary statistics (recomputed identically on observed + re-simulated)
# --------------------------------------------------------------------------------------
def _cov_cv_by_cn(coverage, cn, min_n=20):
    """CN-conditioned coverage CV — the dispersion kappa models, measured the way it is FIT.

    `estimate_dna` fits the depth dispersion **within each cell** (each cell is its own
    compositional draw), so the matching summary is the *within-cell* coverage CV at each rounded
    CN level, then the median across cells. Pooling all cells together instead would conflate the
    technical per-locus dispersion with cross-cell depth / cross-region biology the model does not
    represent (and would make a real-vs-re-sim overlay unfair). 1D (bulk) input is pooled directly.
    """
    coverage = np.asarray(coverage, float)
    cn = np.round(np.asarray(cn, float)).astype(int)
    if coverage.ndim == 1:
        coverage, cn = coverage[None, :], cn[None, :]
    per_cell = {}
    for X, C in zip(coverage, cn):
        for level in np.unique(C):
            c = X[C == level]
            c = c[c > 0]
            if c.size >= min_n and c.mean() > 0:
                per_cell.setdefault(int(level), []).append(float(c.std() / c.mean()))
    return {lvl: float(np.median(v)) for lvl, v in per_cell.items() if len(v) >= 1}


def _baf_at_hets(alt, coverage, het_obs, min_cov=8):
    """Observed BAF (alt/coverage) at genuinely-het observations (a per-element boolean mask)."""
    a = np.asarray(alt, float).ravel()
    c = np.asarray(coverage, float).ravel()
    h = np.asarray(het_obs, bool).ravel()
    ok = h & (c >= min_cov)
    return (a[ok] / c[ok]) if ok.any() else np.array([])


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


def _ado_frac(baf, extreme=0.05):
    return float(np.mean((baf <= extreme) | (baf >= 1.0 - extreme))) if baf.size else float("nan")


def make_tumor(seed):
    tumor = GenotypeTumor(seed=seed, genome_params=GENOME, cancer_cell_params=CANCER,
                          deme_params=DEME, spatial_params=SPATIAL, selection_params=SELECTION)
    tumor.grow(1200, seed=seed)
    return tumor.make_cell_data()


# --------------------------------------------------------------------------------------
# REAL mode: fit estimate_dna on the cached real references, re-simulate, overlay
# --------------------------------------------------------------------------------------
def _fit_reference(ref):
    """Run `estimate_dna` on a loaded reduced reference dict."""
    return estimate_dna(ref["coverage"], ref["alt"], ref["cn"], modality=ref["modality"],
                        breadth=ref["breadth"], depth_model=ref["depth_model"],
                        het_mask=ref["het_mask"], variant_mask=ref["variant_mask"])


def _het_obs_assay_sc(assay, lo=0.15, hi=0.85):
    af = assay.true_alt_fraction.values
    return (af > lo) & (af < hi)


def _het_obs_assay_bulk(assay, lo=0.15, hi=0.85):
    af = assay.observed_data["true_alt_fraction"].values
    return (af > lo) & (af < hi)


def run_real(args, bulk_ref, sc_ref):
    """Fit on real HG002 (bulk) + the chosen sc reference, re-simulate, overlay the summaries."""
    cell_data = make_tumor(args.seed)

    # --- BULK: fit HG002 -> re-simulate bulkDNA on the synthetic tumour --------------------
    est_bulk = _fit_reference(bulk_ref)
    sim_bulk = bulkDNA(breadth=est_bulk.breadth, seed=11, depth_model=est_bulk.depth_model,
                       **est_bulk.dna_kwargs()).run(cell_data)
    sb = sim_bulk.observed_data
    print(f"BULK fit (HG002): {est_bulk}")

    # --- SC: fit the chosen sc reference -> re-simulate scDNA -----------------------------
    est_sc = _fit_reference(sc_ref)
    sim_sc = scDNA(n_cells=args.n_cells, breadth=est_sc.breadth, seed=21,
                   depth_model=est_sc.depth_model, **est_sc.dna_kwargs()).run(cell_data)
    print(f"SC   fit ({args.sc_source}): {est_sc}")

    # coverage-CV-by-CN (the CN-conditioned dispersion): the sc source carries many CN states.
    cv_obs = _cov_cv_by_cn(sc_ref["coverage"], sc_ref["cn"])
    cv_sim = _cov_cv_by_cn(sim_sc.coverage.values, sim_sc.true_cn.values)

    # per-locus coverage distribution: HG002 bulk real vs re-sim.
    cov_obs = np.asarray(bulk_ref["coverage"], float).ravel()
    cov_obs = cov_obs[cov_obs > 0]
    cov_sim = sb["coverage"].values.astype(float)
    cov_sim = cov_sim[cov_sim > 0]

    # het BAF + ADO: prefer the sc source if it carries het reads (Tapestri); else HG002 bulk hets.
    if sc_ref["het_mask"] is not None and np.asarray(sc_ref["het_mask"]).any():
        baf_o = _baf_at_hets(sc_ref["alt"], sc_ref["coverage"], sc_ref["het_mask"])
        baf_s = _baf_at_hets(sim_sc.alt_counts.values, sim_sc.coverage.values,
                             _het_obs_assay_sc(sim_sc))
        baf_src = f"{args.sc_source} (single-cell; ADO)"
    else:
        # HG002 hets are germline (true VAF == 0.5), so compare the technical allele-sampling spread
        # around 0.5: select re-sim bulk loci near 0.5 too (subclonal VAFs would be biology, not the
        # Binomial/error layer this panel validates).
        baf_o = _baf_at_hets(bulk_ref["alt"], bulk_ref["coverage"], bulk_ref["het_mask"])
        baf_s = _baf_at_hets(sb["alt_counts"].values, sb["coverage"].values,
                             _het_obs_assay_bulk(sim_bulk, lo=0.43, hi=0.57))
        baf_src = "HG002 bulk (germline hets; no ADO)"

    sources = {"HG002 (bulk-WGS)": (est_bulk, bulk_ref["source"]),
               f"{args.sc_source} ({est_sc.breadth}-{'panel' if est_sc.breadth=='panel' else 'WGS'})":
                   (est_sc, sc_ref["source"])}
    _make_real_figure(cv_obs, cv_sim, cov_obs, cov_sim, baf_o, baf_s, baf_src, sources,
                      args.out, args.sc_source)
    print(f"sc coverage-CV by CN  observed={ {k: round(v,3) for k,v in cv_obs.items()} }")
    print(f"sc coverage-CV by CN  fit-sim ={ {k: round(v,3) for k,v in cv_sim.items()} }")
    print(f"het BAF ADO  observed={_ado_frac(baf_o):.3f}  fit-sim={_ado_frac(baf_s):.3f}  [{baf_src}]")
    print(f"figure -> {args.out}")


def _make_real_figure(cv_o, cv_s, cov_o, cov_s, baf_o, baf_s, baf_src, sources, out, sc_source):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    ax = axes[0, 0]
    levels = sorted(set(cv_o) & set(cv_s))
    if levels:
        ax.plot(levels, [cv_o[l] for l in levels], "o-", label="real", color="tab:gray")
        ax.plot(levels, [cv_s[l] for l in levels], "s--", label="re-simulated (fitted)",
                color="tab:red")
    else:  # no shared CN levels — show each separately
        ax.plot(sorted(cv_o), [cv_o[l] for l in sorted(cv_o)], "o-", label="real", color="tab:gray")
        ax.plot(sorted(cv_s), [cv_s[l] for l in sorted(cv_s)], "s--",
                label="re-simulated (fitted)", color="tab:red")
    ax.set(xlabel="called copy number", ylabel="coverage CV",
           title=f"CN-conditioned coverage dispersion\n({sc_source} sc; kappa fit)")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    lo = min(cov_o.min(), cov_s.min()) if cov_o.size and cov_s.size else 0
    hi = np.percentile(np.concatenate([cov_o, cov_s]), 99) if cov_o.size and cov_s.size else 1
    bins = np.linspace(lo, hi, 41)
    ax.hist(cov_o, bins=bins, density=True, alpha=0.5, label="real", color="tab:gray")
    ax.hist(cov_s, bins=bins, density=True, alpha=0.5, label="re-simulated (fitted)", color="tab:red")
    ax.set(xlabel="per-locus coverage", ylabel="density",
           title="Bulk (HG002): coverage distribution\n(mu_depth + dispersion)")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    bins = np.linspace(0, 1, 31)
    if baf_o.size:
        ax.hist(baf_o, bins=bins, density=True, alpha=0.5, label="real", color="tab:gray")
    if baf_s.size:
        ax.hist(baf_s, bins=bins, density=True, alpha=0.5, label="re-simulated (fitted)",
                color="tab:red")
    ax.set(xlabel="BAF at het loci", ylabel="density",
           title=f"Het BAF spread + ADO\n[{baf_src}]  "
                 f"(ADO real={_ado_frac(baf_o):.2f}, sim={_ado_frac(baf_s):.2f})")
    ax.legend(fontsize=8)

    ax = axes[1, 1]; ax.axis("off")
    lines = ["fit -> re-simulate -> overlay (real DNA references)", ""]
    candidates = ("gc_curve_sigma", "error_rate", "ado_rate", "beta_binom_conc", "doublet_rate")
    for label, (est, src) in sources.items():
        h = est.hypers
        disp = (f"kappa={h.kappa:.0f}" if est.depth_model == "dm"
                else f"nb_disp={h.nb_dispersion:.3f}")
        prior_only = [c for c in candidates if c not in est.fitted]
        lines += [f"{label}:",
                  f"  source: {src[:52]}",
                  f"  mu_depth={h.mu_depth:.1f}  {disp}",
                  f"  ado={h.ado_rate:.2f} conc={h.beta_binom_conc:.1f} "
                  f"cap={h.capture_sigma:.2f} err={h.error_rate:.1e}",
                  f"  fitted: {est.fitted}",
                  f"  prior-only: {prior_only}",
                  ""]
    ax.text(0.0, 0.5, "\n".join(lines), fontsize=7.5, family="monospace", va="center")

    fig.suptitle("DNA estimate_dna(): fitted technical params reproduce REAL DNA summaries "
                 "(GIAB HG002 + Tapestri/DLP+)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")


# --------------------------------------------------------------------------------------
# SYNTHETIC mode: the offline ground-truth round-trip (also the auto-fallback)
# --------------------------------------------------------------------------------------
def run_synthetic(args):
    cell_data = make_tumor(args.seed)
    genes = list(cell_data["cell_snv"].columns)
    gc, _, _ = genome_features(genes)

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

    obs_sc = scDNA(n_cells=args.n_cells, breadth="wgs", seed=20, **TRUE_SC).run(cell_data)
    est_sc = estimate_dna_from_assay(obs_sc)
    sim_sc = scDNA(n_cells=args.n_cells, breadth="wgs", seed=21, depth_model=est_sc.depth_model,
                   **est_sc.dna_kwargs()).run(cell_data)
    print(f"SC   fit: {est_sc}")

    baf_o = _baf_at_hets(obs_sc.alt_counts.values, obs_sc.coverage.values, _het_obs_assay_sc(obs_sc))
    baf_s = _baf_at_hets(sim_sc.alt_counts.values, sim_sc.coverage.values, _het_obs_assay_sc(sim_sc))
    ado_o, ado_s = _ado_frac(baf_o), _ado_frac(baf_s)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    ax = axes[0, 0]
    levels = sorted(set(cv_obs) & set(cv_sim))
    ax.plot(levels, [cv_obs[l] for l in levels], "o-", label="observed", color="tab:gray")
    ax.plot(levels, [cv_sim[l] for l in levels], "s--", label="re-simulated (fitted)", color="tab:red")
    ax.set(xlabel="called copy number", ylabel="coverage CV")
    ax.set_title(f"Bulk: CN-conditioned coverage dispersion\n(kappa true={TRUE_BULK['kappa']:.0f}, "
                 f"fit={est_bulk.hypers.kappa:.0f})")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(gcx_o, gcy_o, "o-", label="observed", color="tab:gray")
    ax.plot(gcx_s, gcy_s, "s--", label="re-simulated (fitted)", color="tab:red")
    ax.set(xlabel="GC content", ylabel="normalized coverage")
    ax.set_title(f"Bulk: coverage-vs-GC curve\n(gc_curve_sigma fit={est_bulk.hypers.gc_curve_sigma:.3f})")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    bins = np.linspace(0, 1, 31)
    ax.hist(baf_o, bins=bins, density=True, alpha=0.5, label="observed", color="tab:gray")
    ax.hist(baf_s, bins=bins, density=True, alpha=0.5, label="re-simulated (fitted)", color="tab:red")
    ax.set(xlabel="BAF at het loci", ylabel="density")
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
    ax.set(xlabel="CNA log2 ratio", ylabel="density")
    ax.set_title("Bulk: CNA log2-ratio distribution")
    ax.legend(fontsize=8)

    fig.suptitle("DNA estimate_dna(): fitted technical params reproduce observed summaries "
                 "(synthetic round-trip)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=120, bbox_inches="tight")

    print(f"bulk coverage-CV by CN  observed={ {k: round(v,3) for k,v in cv_obs.items()} }")
    print(f"bulk coverage-CV by CN  fit-sim ={ {k: round(v,3) for k,v in cv_sim.items()} }")
    print(f"sc ADO rate  observed={ado_o:.3f}  fit-sim={ado_s:.3f}")
    print(f"figure -> {args.out}")


# --------------------------------------------------------------------------------------
def _load(name, override):
    path = override if override else B.reference_path(name)
    return B.load_reference(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sc-source", choices=["dlp", "tapestri"], default="dlp",
                    help="Single-cell real reference: DLP+ scWGS (default) or Tapestri sc-panel.")
    ap.add_argument("--real-bulk", default=None, help="Override path to the bulk (HG002) .npz.")
    ap.add_argument("--real-sc", default=None, help="Override path to the single-cell .npz.")
    ap.add_argument("--synthetic", action="store_true",
                    help="Skip the real references; run the synthetic ground-truth round-trip.")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--n-cells", type=int, default=200)
    ap.add_argument("--out", default=os.path.join(REPO, "manuscript/figures/validation_dna.png"))
    args = ap.parse_args()

    if args.synthetic:
        return run_synthetic(args)

    bulk_ref = _load("hg002", args.real_bulk)
    sc_ref = _load(args.sc_source, args.real_sc)
    if bulk_ref is None or sc_ref is None:
        missing = [n for n, r in [("HG002 bulk", bulk_ref), (f"{args.sc_source} sc", sc_ref)]
                   if r is None]
        print(f"[real DNA reference(s) missing: {', '.join(missing)} — build with "
              f"validation/data/build_dna_reference.py]\n -> falling back to synthetic round-trip")
        return run_synthetic(args)
    run_real(args, bulk_ref, sc_ref)


if __name__ == "__main__":
    main()
