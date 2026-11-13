"""S5 Fig: published tools already run on iscc output, scored against the ground truth.

One row per score: the tool's result on iscc output (filled dot, in its modality's colour), the
reference it has to beat where one exists (open circle: chance, the majority clone, equal
proportions, the best decoy pair or a random ranking), and the per-clone or per-type spread where the
tutorial printed one (a bar behind the dot). Every value is copied from a printed output of the
executed tool tutorials of August 2026 into ``supp_tools_metrics.json``, which records the notebook,
cell and line of each; nothing here re-runs a tool. Those tutorials predate the paper regime, so the
figure is labelled as pre-regime evidence and its caption carries the caveats.

Writes ``<paper>/figures/paper1/supp_tools.pdf`` (and a PNG preview).
Run:  python validation/paper/supp_tools.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import style as S  # noqa: E402

METRICS = os.path.join(HERE, "supp_tools_metrics.json")
ORDER = ("DNA", "RNA", "spatial")          # the canonical modality order of the paper
HEADER = {"DNA": "Single-cell and bulk DNA", "RNA": "Single-cell RNA", "spatial": "Spatial transcriptomics"}

ROW_MM, HEAD_MM, TOP_MM, BOTTOM_MM = 4.6, 5.2, 9.0, 10.0
LABEL_MM, PLOT_MM = 72.0, 66.0


def load(path=METRICS):
    with open(path) as fh:
        return json.load(fh)["rows"]


def layout(rows):
    """The y position (mm from the top of the plot area) of every group header and row."""
    y, heads, ys = 0.0, [], []
    for mod in ORDER:
        group = [r for r in rows if r["modality"] == mod]
        if not group:
            continue
        heads.append((mod, y))
        y += HEAD_MM
        for r in group:
            ys.append((r, y + ROW_MM / 2))
            y += ROW_MM
    return heads, ys, y


def fmt(v):
    """Two decimals, or three below 0.1 so that a small error is not rounded to one digit."""
    return f"{v:.3f}" if v < 0.1 else f"{v:.2f}"


def label(r):
    s = f"{r['tool']}: {r['metric']}"
    return s + " (lower is better)" if r.get("lower_is_better") else s


def draw(rows):
    S.setup()
    heads, ys, plot_h = layout(rows)
    fig = S.figure("double", TOP_MM + plot_h + BOTTOM_MM)
    ax = S.ax_mm(fig, LABEL_MM, TOP_MM, PLOT_MM, plot_h)
    ax.set_xlim(0, 1)
    ax.set_ylim(plot_h, 0)                 # mm, top to bottom, so rows sit where layout() put them
    ax.set_yticks([])
    S.despine(ax, left=False)
    ax.set_xticks(np.linspace(0, 1, 5))
    ax.set_xlabel("score against the ground truth")
    ax.axvline(0.5, color=S.BG_CELLS, lw=0.4, zorder=0)

    for mod, y in heads:
        S.text_mm(fig, 2.0, TOP_MM + y + HEAD_MM * 0.62, HEADER[mod], fontsize=6.5, fontweight="bold",
                  va="center")
    for r, y in ys:
        colour = S.MODALITY[r["modality"]]
        S.text_mm(fig, 4.0, TOP_MM + y, label(r), fontsize=5.5, va="center")
        if "range" in r:
            lo, hi = r["range"]
            ax.plot([lo, hi], [y, y], color=colour, lw=2.2, alpha=0.3, solid_capstyle="round", zorder=1)
        if "baseline" in r:
            ax.plot([r["baseline"], r["value"]], [y, y], color=S.GREY, lw=0.5, zorder=2)
            ax.scatter([r["baseline"]], [y], s=9, facecolor="white", edgecolor=S.GREY, linewidth=0.6,
                       zorder=3, clip_on=False)
        ax.scatter([r["value"]], [y], s=11, color=colour, zorder=4, clip_on=False)
        S.text_mm(fig, LABEL_MM + PLOT_MM + 3.0, TOP_MM + y, fmt(r["value"]), fontsize=5.5, va="center")
        if "baseline" in r:
            S.text_mm(fig, LABEL_MM + PLOT_MM + 9.0, TOP_MM + y,
                      f"{r['baseline_label']} {fmt(r['baseline'])}", fontsize=5.0, va="center",
                      color=S.GREY)

    # Key, above the plot: what the three marks mean.
    kx, ky = LABEL_MM, 3.2
    kax = S.ax_mm(fig, kx, ky - 2.0, 4.0, 4.0)
    kax.set_xlim(0, 1); kax.set_ylim(0, 1); kax.axis("off")
    kax.scatter([0.5], [0.5], s=11, color=S.INK)
    S.text_mm(fig, kx + 4.5, ky, "tool on iscc output, coloured by modality", fontsize=5.5, va="center")
    kax2 = S.ax_mm(fig, kx + 46.0, ky - 2.0, 4.0, 4.0)
    kax2.set_xlim(0, 1); kax2.set_ylim(0, 1); kax2.axis("off")
    kax2.scatter([0.5], [0.5], s=9, facecolor="white", edgecolor=S.GREY, linewidth=0.6)
    S.text_mm(fig, kx + 50.5, ky, "reference to beat", fontsize=5.5, va="center")
    kax3 = S.ax_mm(fig, kx + 73.0, ky - 2.0, 6.0, 4.0)
    kax3.set_xlim(0, 1); kax3.set_ylim(0, 1); kax3.axis("off")
    kax3.plot([0.1, 0.9], [0.5, 0.5], color=S.INK, lw=2.2, alpha=0.3, solid_capstyle="round")
    S.text_mm(fig, kx + 79.5, ky, "range over clones or cell types", fontsize=5.5, va="center")
    S.text_mm(fig, 2.0, TOP_MM + plot_h + BOTTOM_MM - 1.5,
              "Tool tutorials executed 25-27 August 2026, before the paper regime; values from their printed outputs.",
              fontsize=4.8, color=S.GREY, va="bottom")
    return fig


def main():
    rows = load()
    fig = draw(rows)
    path = S.save(fig, "supp_tools", stamp=False)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
