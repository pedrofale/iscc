"""Validation of the F1/F2 sampling layer (`isccsample`).

Three panels demonstrating that the sampling biases behave as designed:

  (a) **Multi-region heterogeneity** — a single biopsy region recovers far fewer
      clones than the whole tumour, but the UNION of multiple disjoint regions
      approaches the true clone count (the substrate for multi-region phylogeny).
  (b) **Dissociation composition bias** — cell-type-dependent recovery shifts the
      sampled cell-type fractions away from the true tissue fractions, in the
      direction set by the recovery probabilities (immune/stromal down,
      cancer/epithelial up).
  (c) **Liquid-biopsy enrichment** — circulating cells drawn by the liquid biopsy
      are enriched for high-dispersal clones vs a uniform sample.

Usage:  python validation/validate_sampling.py
Produces the paper repo's figures/validation_sampling.png.
"""
import argparse
import os

import numpy as np

from iscc.tumor.models import GenotypeTumor
from iscc.sample.biopsy.biopsy import Biopsy
from iscc.sample.dissociation.dissociation import Dissociation, biological_type, DEFAULT_RECOVERY
from _paths import figure_path

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GENOME = {"n_segments": 6, "segment_size": 20}
CANCER = {"division_rate": 0.5, "death_rate": 0.02, "max_birth_rate": 0.95,
          "mutation_rate": 0.6, "dispersal_rate": 0.25}
SELECTION = {"prop_driver": 0.2, "prop_dispersal": 0.3,
             "prop_immune_resistance": 0.1, "prop_treatment_resistance": 0.1}
SPATIAL = {"grid_size": 28, "structure_radius": 5, "immune_density": 0.2}
DEME = {"carrying_capacity": 10, "initial_cancer_cells": 5}


def build_mixed_tumor(seed, steps):
    """Tumour with an immune/stromal/epithelial microenvironment (for the
    composition-bias panel — needs multiple biological types present)."""
    t = GenotypeTumor(
        seed=seed, genome_params=GENOME, cancer_cell_params=CANCER,
        epithelial_cell_params={"division_rate": 0.0, "death_rate": 0.02},
        stromal_cell_params={"division_rate": 0.0, "death_rate": 0.02},
        immune_cell_params={"division_rate": 0.0, "death_rate": 0.02, "prob_kill": 0.0},
        deme_params=DEME, spatial_params=SPATIAL, selection_params=SELECTION)
    t.grow(steps, seed=seed)
    return t.make_cell_data()


def build_cancer_tumor(seed, steps):
    """Cancer-only tumour with finer spatial spread (for the heterogeneity and
    liquid-biopsy panels — so regions land on spatially-structured clones)."""
    t = GenotypeTumor(
        seed=seed, genome_params=GENOME,
        cancer_cell_params={**CANCER, "mutation_rate": 0.5, "dispersal_rate": 0.45},
        deme_params={"carrying_capacity": 4, "initial_cancer_cells": 5},
        spatial_params={"grid_size": 50, "structure_radius": 0, "immune_density": 0.0},
        selection_params=SELECTION)
    t.grow(steps, seed=seed)
    return t.make_cell_data()


def cancer_mask(cell_data):
    ct = cell_data["cell_type"]["cell_id"].astype(str)
    return ~ct.isin(("immune", "stromal", "epithelial"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--out", default=figure_path("validation_sampling.png"))
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cancer_cd = build_cancer_tumor(args.seed, max(args.steps, 1500))
    cct = cancer_cd["cell_type"]["cell_id"].astype(str)
    total_clones = cct.nunique()

    # ---------------------------------------------------------------- (a) heterogeneity
    bx = Biopsy(cancer_cd, rng=np.random.default_rng(args.seed + 3))
    extent = bx._extent()
    radius = max(1.5, 0.06 * extent)   # small local cores

    K = 8
    chosen, region, geom = bx.sample("multiregion", n_regions=K, radius=radius)
    region_sets = []
    for lab in region.unique():
        members = region.index[region.values == lab]
        region_sets.append(set(cct.loc[members].unique()))
    # accumulate the union over regions added smallest-first (cleaner "approaches truth")
    region_sets.sort(key=len)
    per_region_clones, union_clones, seen = [], [], set()
    for s in region_sets:
        per_region_clones.append(len(s))
        seen |= s
        union_clones.append(len(seen))

    ax = plt.figure(figsize=(15, 4.5)).add_subplot(1, 3, 1)
    fig = ax.figure
    xs = np.arange(1, len(union_clones) + 1)
    ax.plot(xs, union_clones, "o-", color="tab:blue", label="union of regions")
    ax.axhline(total_clones, ls="--", color="k", label=f"whole tumour ({total_clones})")
    ax.bar(xs, per_region_clones, color="tab:orange", alpha=0.5, label="single region")
    ax.set_xlabel("number of disjoint regions pooled")
    ax.set_ylabel("distinct cancer clones")
    ax.set_title("(a) Multi-region heterogeneity\nsingle region << truth; union approaches it")
    ax.legend(fontsize=8)

    # ---------------------------------------------------------------- (b) composition bias
    mixed_cd = build_mixed_tumor(args.seed, args.steps)
    diss = Dissociation(mixed_cd, rng=np.random.default_rng(args.seed + 1))  # DEFAULT_RECOVERY
    _, dmeta, _ = diss.run()
    types = ["cancer", "epithelial", "stromal", "immune"]
    inp = {t: dmeta["input_composition"].get(t, 0.0) for t in types}
    samp = {t: dmeta["sampled_composition"].get(t, 0.0) for t in types}
    # enrichment = sampled / input fraction; >1 over-recovered, <1 under-recovered.
    # (Cleaner than absolute fractions when one type dominates the tissue.)
    enrich = [samp[t] / inp[t] if inp[t] > 0 else np.nan for t in types]

    ax = fig.add_subplot(1, 3, 2)
    colors = ["tab:red" if e >= 1 else "tab:blue" for e in enrich]
    x = np.arange(len(types))
    ax.bar(x, enrich, color=colors, alpha=0.7)
    ax.axhline(1.0, ls="--", color="k", lw=1)
    for i, t in enumerate(types):
        ax.text(i, enrich[i] + 0.02, f"p={DEFAULT_RECOVERY[t]:.1f}\n{inp[t]:.0%}→{samp[t]:.0%}",
                ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(types, rotation=20)
    ax.set_ylabel("sampled / true fraction (enrichment)")
    ax.set_title("(b) Dissociation composition bias\nimmune/stromal under-recovered, cancer/epi over")
    ax.set_ylim(0, max(enrich) * 1.25)

    # ---------------------------------------------------------------- (c) liquid enrichment
    evo = cancer_cd["cell_evo"]
    signal = "n_mut_disp" if "n_mut_disp" in evo.columns else "dispersal_rate"
    cancer_idx = cct.index
    liq_vals, unif_vals = [], []
    for s in range(30):
        b = Biopsy(cancer_cd, rng=np.random.default_rng(1000 + s))
        ch, _, _ = b.sample("liquid", n_liquid=30)
        liq_vals.append(evo.loc[ch, signal].mean())
        u = np.random.default_rng(2000 + s).choice(cancer_idx, 30, replace=False)
        unif_vals.append(evo.loc[u, signal].mean())
    liq_m, unif_m = np.mean(liq_vals), np.mean(unif_vals)

    ax = fig.add_subplot(1, 3, 3)
    ax.hist(unif_vals, bins=15, alpha=0.5, color="tab:gray", label=f"uniform (mean {unif_m:.2f})")
    ax.hist(liq_vals, bins=15, alpha=0.5, color="tab:green", label=f"liquid (mean {liq_m:.2f})")
    ax.axvline(unif_m, ls="--", color="tab:gray")
    ax.axvline(liq_m, ls="--", color="tab:green")
    ax.set_xlabel(f"mean per-cell dispersal signal ({signal})")
    ax.set_ylabel("count (over 30 draws)")
    ax.set_title(f"(c) Liquid biopsy enrichment\n+{100*(liq_m-unif_m)/max(unif_m,1e-9):.0f}% vs uniform")
    ax.legend(fontsize=8)

    fig.suptitle("isccsample F1/F2: spatial heterogeneity, dissociation composition bias, "
                 "liquid-biopsy dispersal enrichment", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=120, bbox_inches="tight")

    print(f"(a) whole-tumour clones={total_clones}; single-region mean="
          f"{np.mean(per_region_clones):.1f}; union(K={K})={union_clones[-1]}")
    print(f"(b) composition shift: " + ", ".join(
        f"{t} {dmeta['composition_shift'].get(t, 0):+.3f}" for t in types))
    print(f"(c) liquid dispersal mean={liq_m:.3f} vs uniform={unif_m:.3f}")
    print(f"figure -> {args.out}")


if __name__ == "__main__":
    main()
