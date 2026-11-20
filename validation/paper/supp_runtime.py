"""S4 Fig: iscc against demon, OncoSpot, stCNASim, CINner and SISTEM at matched size, one core (A, B),
and where iscc's time goes (C).

Reads ``comparisons/results/runtime.csv``, one row per (tool, variant, size, seed) as written by
``comparisons/runtime_collect.py`` from the cluster logs: wall time and peak memory measured by
``/usr/bin/time -v`` around each tool's whole run, on identical Arrhenius CPU nodes. The x axis is
the number of cancer cells requested (for OncoSpot, half its cells, its tumour fraction being 0.5),
so every tool is compared at the tumour size it simulated. stCNASim simulates spots, not cells: it
is placed at the cancer cells OncoSpot puts under the same number of spots (1,461 spots for 7,500
cancer cells). Its own benchmark size (``n_cancer`` 0, 1,600 spots) is in the CSV but not plotted.

The tools do different work in a run, which the legend states for each; the panels show what a run
costs, not which tool is faster at the same work, and the figure says so. demon grows the tumour only
(the demon match's selection settings: K 16, invasion at 0.1, drivers only).

iscc is shown as the paper runs it on regime v4 (variant ``v4``: binomial tau-leaping with the
Cao-Gillespie-Petzold step, the tumour grown, resected, sectioned and put through Visium;
``comparisons/iscc_runtime_v4.py``). The earlier iscc variants (v3 and those timed on the old tau-leaping) stay in
the CSV and are not drawn. Failed attempts are left out; they are kept in the CSV with ``ok`` False.
A run cancelled before it finished is drawn in the wall-time panel as a hollow upward triangle at the
lower bound of its wall time (``wall_lower_s``). Two of demon's five 75,000-cell runs stopped on demon's
own check that no birth rate reaches the crowding death rate (its driver effects are unbounded) and
are left out; the three that finished are drawn.

Panel C splits the iscc runs of A into their steps, as iscc's JSON line in each log times them
(``stage_grow_s``, ``stage_section_s``, ``stage_visium_s``): growth, sectioning and Visium. These are the
runs the earlier Fig 2G drew from ``$ISCC_DATA_DIR/logs/runtime-v4``; ``runtime_collect.py`` read the
same logs into the CSV, so A and B already hold that panel's wall time and peak memory point for point.
The steps sum to the run less the interpreter's start (about 2 s).

Drawn at 165 mm, the text width, so that its text prints at its true size.

Usage::

    python validation/paper/supp_runtime.py
"""
import os
import sys

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import style as S  # noqa: E402

RES = os.path.join(HERE, "comparisons", "results")
# Fixed colours for the simulators (Okabe-Ito, distinct under colour blindness); iscc keeps the
# house blue. The order is the legend order. Markers are a second cue beside colour (the two blues,
# the two oranges).
SIM = {"iscc": S.SOURCE["iscc"], "demon": "#E69F00", "OncoSpot": "#D55E00", "stCNASim": "#56B4E9",
       "CINner": "#009E73", "SISTEM": "#CC79A7"}
MARKER = {"stCNASim": "s", "demon": "D"}
LABEL = {tool: tool for tool in SIM}
ISCC_VARIANT = "v4"
# iscc's steps (C): one colour, iscc's, told apart by marker and line style
STEPS = (("stage_grow_s", "growth", "o", "-"), ("stage_section_s", "section", "s", "--"),
         ("stage_visium_s", "Visium", "^", ":"))
W_MM, H_MM = 165, 62


def _series(ax, sub, col, colour, **kw):
    g = sub.groupby("n_cancer")[col]
    x = np.array(sorted(g.groups))
    ax.plot(x, g.median().loc[x], color=colour, **kw)
    ax.vlines(x, g.min().loc[x], g.max().loc[x], color=colour, lw=0.8, alpha=kw.get("alpha", 1.0))


def _log_sizes(ax, sizes, ylab):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{n:,}" for n in sizes])
    ax.minorticks_off()
    ax.set_xlabel("Cancer cells")
    ax.set_ylabel(ylab)


def main():
    S.setup()
    runs = pd.read_csv(os.path.join(RES, "runtime.csv"))
    runs = runs[(runs.tool != "iscc") | (runs.variant == ISCC_VARIANT)]
    df = runs[runs.ok.astype(bool) & (runs.n_cancer > 0)].copy()   # failed attempts and stCNASim's own size
    df["peak_gb"] = df.peak_mb / 1000.0           # GB as the paper states it (743.9 MB is 0.74 GB)
    lower = runs["wall_lower_s"] if "wall_lower_s" in runs else pd.Series(np.nan, index=runs.index)
    unfinished = runs[lower.notna()]
    sizes = sorted(df.n_cancer.unique())
    fig = S.figure(W_MM, H_MM)
    axes = [S.ax_mm(fig, 12, 5, 42, 38), S.ax_mm(fig, 67, 5, 42, 38), S.ax_mm(fig, 121, 5, 42, 38)]
    handles, bounds = [], []
    for ax, col, ylab in zip(axes, ("wall_s", "peak_gb"), ("Wall time (s)", "Peak memory (GB)")):
        for tool, colour in SIM.items():
            sub = df[df.tool == tool]
            if sub.empty:
                continue
            _series(ax, sub, col, colour, marker=MARKER.get(tool, "o"), ms=3)
            if col == "wall_s":
                handles.append(Line2D([], [], color=colour, marker=MARKER.get(tool, "o"), ms=3,
                                      label=LABEL[tool]))
            lb = unfinished[unfinished.tool == tool].groupby("n_cancer").wall_lower_s.min()
            if col == "wall_s" and len(lb):
                ax.plot(lb.index, lb.values, ls="none", marker="^", ms=4, mfc="white", mec=colour)
                bounds.append(Line2D([], [], ls="none", marker="^", ms=4, mfc="white", mec=colour,
                                     label=f"{tool}, cancelled unfinished"))
        _log_sizes(ax, sizes, ylab)
    # C: the iscc runs of A, step by step
    ax = axes[2]
    isc = df[df.tool == "iscc"]
    steps = []
    for col, lab, mk, ls in STEPS:
        _series(ax, isc, col, SIM["iscc"], marker=mk, ms=3, ls=ls)
        steps.append(Line2D([], [], color=SIM["iscc"], marker=mk, ms=3, ls=ls, label=lab))
    _log_sizes(ax, sizes, "Wall time (s)")
    ax.set_title("iscc by step")
    # one legend for A and B, below them, so that it covers no series; C's own below C
    fig.legend(handles=handles + bounds, loc="upper left", bbox_to_anchor=(12 / W_MM, 1 - 52 / H_MM), ncol=4)
    fig.legend(handles=steps, loc="upper left", bbox_to_anchor=(121 / W_MM, 1 - 52 / H_MM), ncol=3)
    S.panel_letter(fig, "A", 1, 1)
    S.panel_letter(fig, "B", 56, 1)
    S.panel_letter(fig, "C", 110, 1)
    S.save(fig, "supp_runtime", stamp=False)
    med = isc.groupby("n_cancer")[["wall_s"] + [c for c, *_ in STEPS]].median().round(2)
    print("iscc medians per step (s):")
    print(med.to_string())


if __name__ == "__main__":
    main()
