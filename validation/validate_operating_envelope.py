"""Layer 2 of the operating envelope (DESIGN_operating_envelope.md): the reported operating ranges.

Produces ``manuscript/figures/validation_operating_envelope.png`` — the phase-diagram figure the
supplementary "Operating regimes" section (the robustness/sensitivity analysis, in the spirit of
scMultiSim and CINner) is built on. Panels:

  A. mutation_rate x dispersal_rate — regime map (realistic vs monoclonal / well-mixed / hypermutated)
  B. selection strength (driver_effects) x mutation_rate — regime map (sweep vs neutral-diverse)
  C. tumour size x O2 diffusion length (o2_diffusion D) — hypoxia core-rim contrast + no-gradient zone
  D-F. 1-D slices from the characterisation sweep CSV: survival vs death_rate; diversity vs
       mutation_rate; clonal-territory confinement vs dispersal_rate — the degenerate thresholds drawn on.

The 2-D phase panels are computed here on a coarse grid (small tumours, one seed) so the figure is
self-contained; the 1-D panels reuse ``analysis/characterize_regimes.csv`` (built if missing). Same
read-only ``diagnose()`` as the built-in QC, so the reported map and the runtime warning agree.

Usage:  python validation/validate_operating_envelope.py
"""
import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "analysis"))
import characterize_regimes as cr  # noqa: E402  (baseline builder + sweep)
from iscc.tumor.diagnostics import diagnose, DEFAULT_THRESHOLDS  # noqa: E402

OUT = os.path.join(REPO, "manuscript/figures/validation_operating_envelope.png")

# Regime categories -> (label, colour). Priority order used when several flags fire.
REGIMES = [
    ("extinct", "extinct", "#3b3b3b"),
    ("hypermutated", "hypermutated", "#8c2d04"),
    ("monoclonal", "monoclonal", "#d94801"),
    ("low_mutation", "monoclonal", "#d94801"),
    ("well_mixed", "well-mixed", "#6a51a3"),
]
GOOD = ("realistic", "#31a354")


def classify(tumor):
    """Map a grown tumour to a regime category index into ``CATS`` (0 == realistic)."""
    d = diagnose(tumor)
    failed = {c.name for c in d.failures}
    for i, (flag, _lab, _col) in enumerate(REGIMES):
        if flag in failed:
            return i + 1
    return 0


CATS = [GOOD] + [(lab, col) for _flag, lab, col in REGIMES]


def _phase_grid(xs, ys, make_over, n_steps, seed=0):
    """Regime category on the (xs, ys) grid. ``make_over(x, y)`` -> baseline-override dict."""
    Z = np.zeros((len(ys), len(xs)), dtype=int)
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            t = cr._build(make_over(x, y))
            t.grow(n_steps=n_steps, seed=seed)
            Z[iy, ix] = classify(t)
    return Z


def _contrast_grid(sizes, Ds, n_steps, seed=0):
    """Hypoxia core-rim contrast over (tumour size, o2_diffusion D)."""
    C = np.full((len(Ds), len(sizes)), np.nan)
    for iy, D in enumerate(Ds):
        for ix, g in enumerate(sizes):
            over = {"spatial_params": {"grid_size": g}, "deme_params": {"carrying_capacity": 3},
                    "microenv_params": cr._hypoxia(D)}
            t = cr._build(over)
            t.grow(n_steps=n_steps, seed=seed)
            C[iy, ix] = diagnose(t)["hypoxia_contrast"]
    return C


def _draw_phase(ax, xs, ys, Z, xlabel, ylabel, title, xlog=False, ylog=False):
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap([c for _, c in CATS])
    norm = BoundaryNorm(np.arange(-0.5, len(CATS) + 0.5), cmap.N)
    ax.pcolormesh(np.arange(len(xs) + 1), np.arange(len(ys) + 1), Z, cmap=cmap, norm=norm,
                  edgecolors="white", linewidth=0.5)
    ax.set_xticks(np.arange(len(xs)) + 0.5)
    ax.set_yticks(np.arange(len(ys)) + 0.5)
    ax.set_xticklabels([f"{x:g}" for x in xs], rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels([f"{y:g}" for y in ys], fontsize=8)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--quick", action="store_true", help="coarser grids for a fast smoke run")
    args = ap.parse_args()

    # 1-D slices from the sweep CSV (build it if absent).
    import pandas as pd
    if not os.path.exists(cr.OUT_CSV):
        cr.run(quick=args.quick)
    df = pd.read_csv(cr.OUT_CSV)

    n = 800 if args.quick else 1200
    step = 2 if args.quick else 1

    # A. mutation_rate x dispersal_rate
    mrs = [0.005, 0.05, 0.2, 1.0, 4.0][::step]
    disps = [0.05, 0.2, 1.0, 4.0, 10.0][::step]
    Za = _phase_grid(mrs, disps,
                     lambda mr, d: {"cancer_cell_params": {"mutation_rate": mr, "dispersal_rate": d}},
                     n_steps=n)
    # B. driver_effects x mutation_rate (selection strength vs mutation supply)
    des = [1.0, 1.3, 1.8, 2.5, 3.5][::step]
    mrs2 = [0.005, 0.03, 0.1, 0.5, 2.0][::step]
    Zb = _phase_grid(des, mrs2,
                     lambda de, mr: {"selection_params": {"driver_effects": de, "prop_driver": 0.2},
                                     "cancer_cell_params": {"mutation_rate": mr}},
                     n_steps=n)
    # C. tumour size x O2 diffusion length
    sizes = [8, 12, 18, 26][::step]
    Ds = [0.3, 2.0, 15.0, 120.0][::step]
    C = _contrast_grid(sizes, Ds, n_steps=n)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(2, 3, figsize=(17, 10))

    _draw_phase(ax[0, 0], mrs, disps, Za, "mutation_rate", "dispersal_rate",
                "A. mutation x dispersal")
    _draw_phase(ax[0, 1], des, mrs2, Zb, "driver_effects (selection)", "mutation_rate",
                "B. selection strength x mutation")

    # C. contrast heatmap with the no-gradient zone outlined
    im = ax[0, 2].pcolormesh(np.arange(len(sizes) + 1), np.arange(len(Ds) + 1), C,
                             cmap="inferno_r", vmin=0, edgecolors="white", linewidth=0.5)
    ax[0, 2].set_xticks(np.arange(len(sizes)) + 0.5); ax[0, 2].set_xticklabels(sizes, fontsize=8)
    ax[0, 2].set_yticks(np.arange(len(Ds)) + 0.5)
    ax[0, 2].set_yticklabels([f"{d:g}" for d in Ds], fontsize=8)
    ax[0, 2].set(xlabel="tumour size (grid_size)", ylabel="O2 diffusion length (o2_diffusion D)",
                 title="C. hypoxia core-rim contrast")
    for iy in range(len(Ds)):
        for ix in range(len(sizes)):
            if not np.isnan(C[iy, ix]) and C[iy, ix] < DEFAULT_THRESHOLDS["contrast_min"]:
                ax[0, 2].add_patch(plt.Rectangle((ix, iy), 1, 1, fill=False, edgecolor="#00d0ff",
                                                 lw=2.5))
    fig.colorbar(im, ax=ax[0, 2], fraction=0.046, label="core - rim hypoxia")

    # D. survival vs death_rate
    _slice(ax[1, 0], df, "death_rate", "n_cancer", "death_rate (division_rate = 0.3)",
           "cancer cells", "D. survival vs death rate", hline=DEFAULT_THRESHOLDS["min_cancer"],
           hlabel="extinct threshold", axvspan=(0.3, 0.36))
    ax[1, 0].annotate("extinction\n(death >= division)", (0.31, ax[1, 0].get_ylim()[1] * 0.6),
                      fontsize=8, color="#3b3b3b")

    # E. clonal diversity vs mutation_rate
    _slice(ax[1, 1], df, "mutation_rate", "shannon", "mutation_rate",
           "clonal Shannon diversity", "E. diversity vs mutation rate",
           hline=DEFAULT_THRESHOLDS["shannon_min"], hlabel="monoclonal threshold", xlog=True)

    # F. clonal-territory confinement vs dispersal_rate
    _slice(ax[1, 2], df, "dispersal_rate", "clone_confinement", "dispersal_rate",
           "clone spatial confinement", "F. territories vs mixing",
           hline=DEFAULT_THRESHOLDS["confinement_min"], hlabel="well-mixed threshold", xlog=True)

    handles = [Patch(facecolor=col, label=lab) for lab, col in CATS]
    ax[0, 1].legend(handles=handles, loc="upper left", bbox_to_anchor=(1.28, 1.0), fontsize=9,
                    title="regime", frameon=False)

    fig.suptitle("iscc operating envelope: which parameter ranges yield realistic tumours "
                 "(degenerate regimes labelled)", fontsize=13, y=1.0)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print("figure ->", args.out)


def _slice(ax, df, axis, metric, xlabel, ylabel, title, hline=None, hlabel="", xlog=False,
           axvspan=None):
    """Plot a metric-vs-knob slice (mean +/- s.d. over seeds) from the sweep CSV."""
    sub = df[df.axis == axis]
    g = sub.groupby("value")[metric]
    x = np.array(sorted(sub.value.unique()), dtype=float)
    mean = g.mean().reindex(sorted(sub.value.unique())).values
    sd = g.std().reindex(sorted(sub.value.unique())).fillna(0).values
    ax.plot(x, mean, "o-", color="#31a354")
    ax.fill_between(x, mean - sd, mean + sd, color="#31a354", alpha=0.2)
    if axvspan is not None:
        ax.axvspan(*axvspan, color="#3b3b3b", alpha=0.12)
    if hline is not None:
        ax.axhline(hline, ls="--", color="#d94801", lw=1.3, label=hlabel)
        ax.legend(fontsize=8)
    if xlog:
        ax.set_xscale("log")
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)


if __name__ == "__main__":
    main()
