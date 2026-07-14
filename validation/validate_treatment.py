"""Validate iscc's treatment module: therapy response and adaptive dosing.

Grows a tumor to an established burden, then continues it under three regimens and tracks
the cancer-cell burden over time:

  * no treatment       -> the tumor keeps growing;
  * continuous chemo   -> the (sensitive) tumor regresses;
  * adaptive chemo     -> dosing pauses below a burden threshold, controlling the tumor
                          with less total drug (West et al. 2023).

Demonstrates that treatment is wired into the default genotype engine and that the
corrected death model lets therapy actually regress a tumor.

Produces manuscript/figures/validation_treatment.png.
Usage:  python validation/validate_treatment.py
"""
import os

import numpy as np

from iscc.tumor.models import GenotypeTumor
from iscc.treatment.chemotherapy import Chemotherapy
from iscc.constants import normal_names

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS = range(4)
BURN_IN = 300
TREAT = 900

GENOME = {"n_segments": 6, "segment_size": 100}
# K>=~6: with density-dependent crowding (DESIGN_crowding.md) the death ramp is steep at very small
# K, so K=3 would drive even the untreated tumour extinct; K=8 gives a persistent tumour to treat.
DEME = {"carrying_capacity": 8, "initial_cancer_cells": 5}
SPATIAL = {"grid_size": 15, "n_structures": 1, "structure_radius": 0}
# fully sensitive tumor so the therapy response is clean
SELECTION = {"prop_driver": 0.1, "prop_dispersal": 0.0, "prop_immune_resistance": 0.0,
             "prop_treatment_resistance": 0.0, "driver_effects": 1.1, "dispersal_effects": 1.0,
             "treatment_resistant_effects": 1.0, "immune_resistant_effects": 1.0}
CANCER = {"division_rate": 0.4, "death_rate": 0.02, "max_birth_rate": 0.8,
          "mutation_rate": 0.5, "dispersal_rate": 0.2}


def _build(seed):
    return GenotypeTumor(seed=seed, genome_params=GENOME, selection_params=SELECTION,
                         cancer_cell_params=CANCER, deme_params=DEME, spatial_params=SPATIAL)


def _cancer_burden(traces):
    return np.array([sum(c for g, c in tr["genotypes_counts"].items() if g not in normal_names)
                     for tr in traces], dtype=float)


def _run(regimen, seed):
    t = _build(seed)
    t.grow(BURN_IN, seed=seed)
    start = len(t.traces)
    treatment = None
    if regimen == "continuous":
        treatment = Chemotherapy(adaptive=False)
    elif regimen == "adaptive":
        threshold = max(1, t.get_cancer_size() // 2)
        treatment = Chemotherapy(adaptive=True, max_tumor_size=threshold)
    t.grow(TREAT, seed=seed + 1000, treatment=treatment)
    burden = _cancer_burden(t.traces)[start:]
    dose = sum(d for _, d in treatment.dosage_trace) if treatment is not None else 0.0
    return burden, dose


def main():
    regimens = ["none", "continuous", "adaptive"]
    curves, doses = {}, {}
    for reg in regimens:
        runs = [_run(reg, s) for s in SEEDS]
        n = min(len(b) for b, _ in runs)
        curves[reg] = np.vstack([b[:n] for b, _ in runs])
        doses[reg] = np.mean([d for _, d in runs])
    print(f"{'regimen':>11} | {'final burden':>12} | {'total drug (dosed steps)':>24}")
    for reg in regimens:
        print(f"{reg:>11} | {curves[reg][:, -1].mean():>12.0f} | {doses[reg]:>24.0f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    colors = {"none": "tab:gray", "continuous": "tab:red", "adaptive": "tab:blue"}
    for reg in regimens:
        m = curves[reg].mean(0)
        sd = curves[reg].std(0)
        x = np.arange(len(m))
        axes[0].plot(x, m, color=colors[reg], label=reg)
        axes[0].fill_between(x, m - sd, m + sd, color=colors[reg], alpha=0.15)
    axes[0].set_xlabel("steps after treatment start")
    axes[0].set_ylabel("cancer-cell burden")
    axes[0].set_title("Tumor burden under therapy")
    axes[0].legend(fontsize=8)

    axes[1].bar(["continuous", "adaptive"],
                [doses["continuous"], doses["adaptive"]],
                color=[colors["continuous"], colors["adaptive"]])
    axes[1].set_ylabel("total drug delivered (dosed steps)")
    axes[1].set_title("Adaptive dosing uses less drug")

    fig.suptitle("iscc treatment module: therapy response and adaptive dosing")
    fig.tight_layout()
    out = os.path.join(REPO, "manuscript/figures/validation_treatment.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"figure -> {out}")


if __name__ == "__main__":
    main()
