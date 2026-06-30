"""Schematic overview figure for the iscc manuscript (Fig. 1).

DRAFT schematic drawn with matplotlib (no iscc dependency) so it always builds; intended to be
replaced by a designed vector figure before submission. Three panels:
  A. tumor growth (spatial, multi-clone, under selection + treatment) -> ground truth
  B. sampling (biopsy / dissociation / liquid biopsy)
  C. assays (bulk+sc DNA, scRNA, Visium) -> matrices -> reads

Run:  python manuscript/figures/make_overview.py   (writes overview.png next to this file)
"""
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

BLUE, GREEN, ORANGE, RED, GREY = "#3b6fb6", "#4f9d69", "#e08a3c", "#c0413b", "#8a8f99"
INK = "#222428"


def box(ax, x, y, w, h, text, fc="white", ec=INK, fs=9, lw=1.3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.03",
                                fc=fc, ec=ec, lw=lw, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, zorder=4)


def arrow(ax, x0, y0, x1, y1, lw=1.6, color=INK):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14,
                                 lw=lw, color=color, zorder=2))


def panel_title(ax, x, y, letter, title):
    ax.text(x, y, letter, fontsize=15, fontweight="bold", color=INK, ha="left", va="top")
    ax.text(x + 0.05, y, title, fontsize=11, fontweight="bold", color=INK, ha="left", va="top")


def main():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    # ---- Panel A: tumor growth ------------------------------------------------
    axA = axes[0]
    panel_title(axA, 0.0, 1.0, "A", "Tumor growth")
    rng = np.random.default_rng(7)
    # a clonal "tumor": clusters of colored cells, one resistant patch (red)
    centers = [(0.32, 0.55, BLUE, 130), (0.5, 0.62, GREEN, 90),
               (0.45, 0.42, ORANGE, 80), (0.6, 0.5, RED, 45)]
    for cx, cy, col, n in centers:
        pts = rng.normal([cx, cy], 0.055, size=(n, 2))
        axA.scatter(pts[:, 0], pts[:, 1], s=8, c=col, alpha=0.8, edgecolors="none", zorder=2)
    axA.add_patch(Circle((0.46, 0.52), 0.205, fill=False, ec=GREY, lw=1.0, ls=(0, (4, 3)), zorder=1))
    axA.text(0.46, 0.30, "spatial demes · multi-clone\ncopy-number + driver selection",
             ha="center", va="top", fontsize=8.5, color=INK)
    # treatment marker
    arrow(axA, 0.78, 0.62, 0.66, 0.55, color=RED)
    axA.text(0.80, 0.64, "therapy", fontsize=8.5, color=RED, ha="left")
    box(axA, 0.06, 0.04, 0.88, 0.13,
        "ground truth: per-cell genotype · CNA · expression · coordinates · lineage",
        fc="#f3f5f8", fs=8.5)

    # ---- Panel B: sampling ----------------------------------------------------
    axB = axes[1]
    panel_title(axB, 0.0, 1.0, "B", "Sampling")
    box(axB, 0.30, 0.80, 0.40, 0.12, "simulated tumor", fc="#eef2f7")
    box(axB, 0.04, 0.50, 0.28, 0.13, "biopsy\n(needle / region)", fc="white")
    box(axB, 0.36, 0.50, 0.28, 0.13, "dissociation\n(composition bias)", fc="white")
    box(axB, 0.68, 0.50, 0.28, 0.13, "liquid biopsy\n(phenotype bias)", fc="white")
    for x in (0.18, 0.50, 0.82):
        arrow(axB, 0.50, 0.80, x, 0.635)
    box(axB, 0.28, 0.22, 0.44, 0.13, "sampled cells\n(state preserved)", fc="#f3f5f8")
    for x in (0.18, 0.50, 0.82):
        arrow(axB, x, 0.50, 0.50, 0.355)

    # ---- Panel C: assays ------------------------------------------------------
    axC = axes[2]
    panel_title(axC, 0.0, 1.0, "C", "Molecular & spatial data")
    box(axC, 0.34, 0.84, 0.32, 0.11, "sampled cells", fc="#f3f5f8")
    labels = ["bulk &\nsc DNA", "scRNA", "spatial\n(Visium)"]
    cols = [BLUE, GREEN, ORANGE]
    xs = [0.06, 0.39, 0.72]
    for x, lab, col in zip(xs, labels, cols):
        arrow(axC, 0.50, 0.84, x + 0.11, 0.70)
        box(axC, x, 0.57, 0.22, 0.13, lab, fc="white", ec=col, lw=1.6)
        arrow(axC, x + 0.11, 0.57, x + 0.11, 0.45)
        box(axC, x, 0.32, 0.22, 0.12, "count /\ncoverage", fc="#f3f5f8", fs=8)
    # small visium grid glyph inside the spatial branch
    for i in range(3):
        for j in range(3):
            axC.add_patch(Circle((0.745 + i * 0.045, 0.605 + j * 0.028), 0.008,
                                  fc=ORANGE, ec="none", alpha=0.7, zorder=5))
    arrow(axC, 0.50, 0.32, 0.50, 0.20)
    box(axC, 0.20, 0.05, 0.60, 0.12, "aligned reads (FASTQ / BAM)\nconsistent across modalities",
        fc="#eef2f7", fs=8.5)

    fig.suptitle(r"iscc: from tumor evolution to multi-modal molecular data",
                 fontsize=13, fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(os.path.dirname(__file__), "overview.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
