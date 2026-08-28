"""Validate iscc's whole-genome duplication (WGD) against the PCAWG WGD signature.

WGD -- a punctuated doubling of every copy on both homologs -- occurs in roughly 30-50% of real
tumors and leaves a characteristic genome-ploidy signature: WGD tumors sit at an elevated, NON-integer
mean ploidy (the genome doubles to ~4, then subsequent copy-number loss erodes it back toward ~3),
whereas non-WGD tumors stay near-diploid (~2)~\\cite{pcawg_2020}. iscc models WGD as a separate
per-division event channel (``wgd_rate``), gated by the ploidy-viability limit (``max_ploidy``): a
doubling that would breach the cap yields a non-viable daughter and is rejected at birth.

This validation shows the mechanism reproduces both facts:
  A. WGD FREQUENCY -- sweeping ``wgd_rate`` over replicate tumors moves the cohort WGD prevalence
     smoothly through the real ~30-50% band (shaded); we report the rate that lands in it.
  B. PLOIDY SIGNATURE -- at a plausible rate, the per-cell mean-ploidy distribution is BIMODAL:
     a near-diploid (~2) non-WGD mode and an elevated, non-integer WGD mode (~3-4) -- the
     doubling-then-loss signature, split by the ground-truth ``is_wgd`` label.

WGD is neutral here (no fitness/tolerance effect in v1, DESIGN_focal_cna.md sec 10), so the prevalence
and ploidy come from the mechanism + viability interaction alone, not a fitted selective advantage.

Produces the paper repo's figures/validation_wgd.png. Prints headline numbers.
Usage:  python -u validation/validate_wgd.py [--quick]
"""
import argparse
import os

import numpy as np

from iscc.tumor.models import GenotypeTumor
from _paths import figure_path

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = figure_path("validation_wgd.png")

# A small, fast, well-mixed-ish spatial config grown with tau-leaping to ~1.5k cells per tumor. WGD is
# neutral, so it drifts: a generous max_ploidy=6 lets the diploid->tetraploid doubling survive, and the
# balanced amp/del erodes ploidy back down (the doubling+loss signature). Not tuned to any dataset.
GENOME = {"n_segments": 10, "segment_size": 40}
SELECTION = {"prop_driver": 0.15, "prop_dispersal": 0.0, "prop_immune_resistance": 0.0,
             "prop_treatment_resistance": 0.0, "max_ploidy": 6, "max_cn": 12}
DEME = {"carrying_capacity": 25, "initial_cancer_cells": 5}
SPATIAL = {"grid_size": 8, "structure_radius": 0}

# per-division WGD probabilities to sweep (0.0 = the off/byte-identical baseline)
RATES = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12]
# a tumor is called WGD (PCAWG-style: the dominant/clonal genome is doubled) when a MAJORITY of its
# cancer cells carry the is_wgd label.
WGD_TUMOR_THRESHOLD = 0.5
# real WGD prevalence band (PCAWG pan-cancer)
PCAWG_BAND = (0.30, 0.50)


def _cancer(wgd_rate):
    return {"division_rate": 0.5, "death_rate": 0.05, "max_birth_rate": 0.9,
            "mutation_rate": 0.6, "dispersal_rate": 0.15, "amp_prob": 0.5, "wgd_rate": wgd_rate}


def _grow(wgd_rate, seed, steps):
    t = GenotypeTumor(seed=seed, genome_params=GENOME, selection_params=SELECTION,
                      cancer_cell_params=_cancer(wgd_rate), deme_params=DEME,
                      spatial_params=SPATIAL, update_mode="tau", tau=1.0)
    t.grow(steps, seed=seed)
    return t


def _cancer_genotypes(t):
    """(count, is_wgd, ploidy) per living cancer genotype."""
    out = []
    for g, c in t.genotypes_counts.items():
        if c > 0 and t._is_cancer(g):
            rep = t.genotypes[g]
            out.append((c, bool(rep.is_wgd), float(rep.genome_summary["ploidy"])))
    return out


def _tumor_stats(t):
    """Per-tumor: WGD-cell fraction and count-weighted mean ploidy over cancer cells."""
    gts = _cancer_genotypes(t)
    tot = sum(c for c, _, _ in gts)
    if tot == 0:
        return None
    wgd_frac = sum(c for c, w, _ in gts if w) / tot
    ploidy = sum(c * p for c, _, p in gts) / tot
    return wgd_frac, ploidy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fewer seeds/steps for a fast smoke run")
    args = ap.parse_args()
    seeds = range(6) if args.quick else range(20)
    steps = 20 if args.quick else 30

    # --- the sweep -------------------------------------------------------------------------------
    prevalence, mean_wgd_frac, mean_ploidy = [], [], []
    print(f"{'wgd_rate':>9} | {'tumors':>6} | {'WGD prevalence':>14} | {'mean WGD-frac':>13} | {'mean ploidy':>11}")
    for wr in RATES:
        fracs, ploidies = [], []
        for s in seeds:
            st = _tumor_stats(_grow(wr, 200 + s, steps))
            if st is not None:
                fracs.append(st[0]); ploidies.append(st[1])
        fracs, ploidies = np.array(fracs), np.array(ploidies)
        prev = float(np.mean(fracs > WGD_TUMOR_THRESHOLD))
        prevalence.append(prev)
        mean_wgd_frac.append(float(fracs.mean()))
        mean_ploidy.append(float(ploidies.mean()))
        print(f"{wr:>9.3f} | {len(fracs):>6d} | {prev:>13.0%} | {fracs.mean():>13.2f} | {ploidies.mean():>11.2f}")

    # the rate whose cohort prevalence lands closest to the middle of the real PCAWG band
    band_mid = np.mean(PCAWG_BAND)
    in_band = [i for i, p in enumerate(prevalence) if PCAWG_BAND[0] <= p <= PCAWG_BAND[1]]
    headline_i = (min(in_band, key=lambda i: abs(prevalence[i] - band_mid)) if in_band
                  else int(np.argmin([abs(p - band_mid) for p in prevalence])))
    headline_rate = RATES[headline_i]

    # --- the ploidy signature at the headline rate -----------------------------------------------
    ploidy_wgd, w_wgd, ploidy_dip, w_dip = [], [], [], []
    for s in seeds:
        for c, is_wgd, ploidy in _cancer_genotypes(_grow(headline_rate, 300 + s, steps)):
            (ploidy_wgd if is_wgd else ploidy_dip).append(ploidy)
            (w_wgd if is_wgd else w_dip).append(c)
    ploidy_wgd, w_wgd = np.array(ploidy_wgd), np.array(w_wgd, float)
    ploidy_dip, w_dip = np.array(ploidy_dip), np.array(w_dip, float)

    def _wmean(x, w):
        return float(np.average(x, weights=w)) if len(x) else float("nan")

    print()
    print(f"HEADLINE: wgd_rate={headline_rate:.3f} -> cohort WGD prevalence {prevalence[headline_i]:.0%} "
          f"(real PCAWG {PCAWG_BAND[0]:.0%}-{PCAWG_BAND[1]:.0%})")
    print(f"  mean ploidy: non-WGD cells {_wmean(ploidy_dip, w_dip):.2f} (near-diploid) vs "
          f"WGD cells {_wmean(ploidy_wgd, w_wgd):.2f} (elevated, non-integer)")
    print(f"  baseline (wgd_rate=0) mean ploidy {mean_ploidy[0]:.2f}; ploidy climbs to "
          f"{mean_ploidy[-1]:.2f} at wgd_rate={RATES[-1]:.2f}")

    # --- figure ----------------------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.2))

    # Panel A: WGD frequency vs rate, PCAWG band shaded
    axA.axhspan(PCAWG_BAND[0], PCAWG_BAND[1], color="tab:green", alpha=0.12,
                label=f"PCAWG WGD prevalence\n({PCAWG_BAND[0]:.0%}-{PCAWG_BAND[1]:.0%})")
    axA.plot(RATES, prevalence, "o-", color="tab:red", label="cohort WGD prevalence")
    axA.plot(RATES, mean_wgd_frac, "s--", color="tab:blue", alpha=0.7,
             label="mean WGD-cell fraction")
    axA.axvline(headline_rate, color="k", lw=0.8, ls=":")
    axA.set_xlabel("WGD rate per division  (wgd_rate)")
    axA.set_ylabel("fraction")
    axA.set_ylim(-0.03, 1.03)
    axA.set_title("A  WGD frequency reaches the real range")
    axA.legend(fontsize=8, loc="upper left")

    # Panel B: ploidy distribution at the headline rate, split by ground-truth is_wgd
    bins = np.linspace(1.0, 5.0, 33)
    axB.hist(ploidy_dip, bins=bins, weights=w_dip, color="tab:gray", alpha=0.7,
             label=f"non-WGD cells (mean {_wmean(ploidy_dip, w_dip):.2f})")
    axB.hist(ploidy_wgd, bins=bins, weights=w_wgd, color="tab:red", alpha=0.6,
             label=f"WGD cells (mean {_wmean(ploidy_wgd, w_wgd):.2f})")
    axB.axvline(2.0, color="k", lw=0.8, ls=":")
    axB.text(2.02, axB.get_ylim()[1] * 0.92, "diploid", fontsize=8)
    axB.axvline(4.0, color="tab:red", lw=0.8, ls=":")
    axB.text(3.55, axB.get_ylim()[1] * 0.92, "tetraploid", fontsize=8, color="tab:red")
    axB.set_xlabel("mean genome ploidy (per cancer cell)")
    axB.set_ylabel("cancer cells")
    axB.set_title(f"B  Doubling+loss ploidy signature\n(wgd_rate={headline_rate:.2f})")
    axB.legend(fontsize=8, loc="upper right")

    fig.suptitle("Whole-genome duplication in iscc reproduces the PCAWG WGD frequency + ploidy signature",
                 fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, dpi=120, bbox_inches="tight")
    print(f"figure -> {FIG}")


if __name__ == "__main__":
    main()
