"""Validate iscc's copy-number alterations against the cancer CNA selection signature.

Cancer genomes recurrently amplify oncogenes and delete tumour suppressors -- copy-number
alterations are under selection, not just drift (Beroukhim et al. 2010, Nature; Davoli et al.
2013, Cell). Under iscc's CINner-style selection model, a segment's population copy number
should therefore rise with its net oncogenic content (oncogene minus TSG positions), whereas
under neutral drift there should be no such relationship.

We grow tumours with selection on vs off, pool the per-segment (net oncogene content, mean copy
number) pairs over replicates, and compare the relationship. The effect size is modest because
copy-number events act per segment while drivers are distributed within segments (amplifying a
segment helps its oncogenes but also its TSGs); the neutral-vs-selection contrast is the point.

Produces manuscript/figures/validation_cna.png.
Usage:  python validation/validate_cna.py
"""
import os

import numpy as np

from iscc.tumor.models import GenotypeTumor
from iscc.validation import cna_amplification_signature

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS = range(8)
STEPS = 1500

GENOME = {"n_segments": 16, "segment_size": 60}
DEME = {"carrying_capacity": 6}
SPATIAL = {"grid_size": 25, "structure_radius": 0}
CANCER = {"division_rate": 0.4, "death_rate": 0.02, "max_birth_rate": 0.95,
          "mutation_rate": 0.7, "dispersal_rate": 0.2}


def _selection(driver_effects):
    return {"prop_driver": 0.2, "prop_dispersal": 0.0, "prop_immune_resistance": 0.0,
            "prop_treatment_resistance": 0.0, "driver_effects": driver_effects,
            "dispersal_effects": 1.0, "treatment_resistant_effects": 1.0,
            "immune_resistant_effects": 1.0}


def _grow(driver_effects, seed):
    t = GenotypeTumor(seed=seed, genome_params=GENOME, selection_params=_selection(driver_effects),
                      cancer_cell_params=CANCER, deme_params=DEME, spatial_params=SPATIAL)
    t.grow(STEPS, seed=seed)
    return t


def main():
    regimes = {"neutral": 1.0, "selection": 1.5}
    results = {}
    print(f"{'regime':>10} | {'corr(CN, net-onc)':>17} | {'slope':>8} | n_seg")
    for name, eff in regimes.items():
        tumors = [_grow(eff, s) for s in SEEDS]
        net, cn, slope, corr = cna_amplification_signature(tumors)
        results[name] = (net, cn, slope, corr)
        print(f"{name:>10} | {corr:>17.3f} | {slope:>8.4f} | {net.size}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, name in zip(axes, ["neutral", "selection"]):
        net, cn, slope, corr = results[name]
        # jitter net for visibility; show mean CN per net level
        ax.scatter(net + np.random.default_rng(0).normal(0, 0.05, net.size), cn,
                   s=10, alpha=0.3, color="tab:gray")
        levels = np.unique(net)
        means = [cn[net == L].mean() for L in levels]
        ax.plot(levels, means, "o-", color="tab:red", label="mean CN")
        xs = np.linspace(net.min(), net.max(), 50)
        ax.plot(xs, np.polyval(np.polyfit(net, cn, 1), xs), "--", color="tab:blue",
                label=f"fit (r={corr:.2f})")
        ax.axhline(2.0, color="k", lw=0.6, ls=":")
        ax.set_xlabel("net oncogenic content per segment\n(#oncogenes - #TSGs)")
        ax.set_title(name)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("population mean copy number")
    fig.suptitle("Copy number rises with oncogenic content under selection, not drift (iscc)")
    fig.tight_layout()
    out = os.path.join(REPO, "manuscript/figures/validation_cna.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"figure -> {out}")


if __name__ == "__main__":
    main()
