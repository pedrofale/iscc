"""Runtime panel: iscc against demon, OncoSpot, stCNASim, CINner and SISTEM at matched size, one core.

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


def _series(ax, sub, col, colour, **kw):
    g = sub.groupby("n_cancer")[col]
    x = np.array(sorted(g.groups))
    ax.plot(x, g.median().loc[x], color=colour, **kw)
    ax.vlines(x, g.min().loc[x], g.max().loc[x], color=colour, lw=0.8, alpha=kw.get("alpha", 1.0))


def main():
    S.setup()
    runs = pd.read_csv(os.path.join(RES, "runtime.csv"))
    runs = runs[(runs.tool != "iscc") | (runs.variant == ISCC_VARIANT)]
    df = runs[runs.ok.astype(bool) & (runs.n_cancer > 0)].copy()   # failed attempts and stCNASim's own size
    df["peak_gb"] = df.peak_mb / 1000.0           # GB as the paper states it (743.9 MB is 0.74 GB)
    lower = runs["wall_lower_s"] if "wall_lower_s" in runs else pd.Series(np.nan, index=runs.index)
    unfinished = runs[lower.notna()]
    fig = S.figure("double", 72)
    axes = [S.ax_mm(fig, 14, 6, 70, 40), S.ax_mm(fig, 104, 6, 70, 40)]
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
        ax.set_xscale("log")
        ax.set_yscale("log")
        sizes = sorted(df.n_cancer.unique())
        ax.set_xticks(sizes)
        ax.set_xticklabels([f"{n:,}" for n in sizes])
        ax.minorticks_off()
        ax.set_xlabel("Cancer cells")
        ax.set_ylabel(ylab)
    # one legend for both panels, below them, so that it covers no series
    fig.legend(handles=handles + bounds, loc="upper left", bbox_to_anchor=(14 / 183, 1 - 55 / 72), ncol=4,
               fontsize=5.5)
    S.panel_letter(fig, "a", 2, 2)
    S.panel_letter(fig, "b", 92, 2)
    S.save(fig, "supp_runtime", stamp=False)


if __name__ == "__main__":
    main()
