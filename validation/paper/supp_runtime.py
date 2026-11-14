"""Runtime panel: iscc against OncoSpot, stCNASim, CINner and SISTEM at matched size, one core.

Reads ``comparisons/results/runtime.csv``, one row per (tool, variant, size, seed) as written by
``comparisons/runtime_collect.py`` from the cluster logs: wall time and peak memory measured by
``/usr/bin/time -v`` around each tool's whole run, on identical Arrhenius CPU nodes. The x axis is
the number of cancer cells requested (for OncoSpot, half its cells, its tumour fraction being 0.5),
so every tool is compared at the tumour size it simulated. stCNASim simulates spots, not cells: it
is placed at the cancer cells OncoSpot puts under the same number of spots (1,461 spots for 7,500
cancer cells). Its own benchmark size (``n_cancer`` 0, 1,600 spots) is in the CSV but not plotted.

iscc is shown as the paper regime runs it (``make_cell_data(layers="visium")`` in low-memory mode,
variant ``visium-low``). The runs before chunked materialisation (variant ``before``) are drawn
faintly in the memory panel only, since its wall time did not change. Failed attempts are left out;
they are kept in the CSV with ``ok`` False. A run cancelled before it finished is drawn in the wall-time
panel as a hollow upward triangle at the lower bound of its wall time (``wall_lower_s``).

Usage::

    python validation/paper/supp_runtime.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import style as S  # noqa: E402

RES = os.path.join(HERE, "comparisons", "results")
# Fixed colours for the simulators (Okabe-Ito, distinct under colour blindness); iscc keeps the
# house blue. The order is the legend order.
SIM = {"iscc": S.SOURCE["iscc"], "OncoSpot": "#D55E00", "stCNASim": "#56B4E9", "CINner": "#009E73",
       "SISTEM": "#CC79A7"}
MARKER = {"stCNASim": "s"}            # a second cue beside colour for the two blues
LABEL = {"iscc": "iscc (grow, section, Visium)", "OncoSpot": "OncoSpot (end to end)",
         "stCNASim": "stCNASim (spots, counts, reads)",
         "CINner": "CINner (grow, sample, phylogeny)", "SISTEM": "SISTEM (grow, sample, read counts)"}
ISCC_VARIANT, ISCC_BEFORE = "visium-low", "before"


def _series(ax, sub, col, colour, **kw):
    g = sub.groupby("n_cancer")[col]
    x = np.array(sorted(g.groups))
    ax.plot(x, g.median().loc[x], color=colour, **kw)
    ax.vlines(x, g.min().loc[x], g.max().loc[x], color=colour, lw=0.8, alpha=kw.get("alpha", 1.0))


def main():
    S.setup()
    runs = pd.read_csv(os.path.join(RES, "runtime.csv"))
    df = runs[runs.ok.astype(bool) & (runs.n_cancer > 0)]   # failed attempts and stCNASim's own size
    lower = runs["wall_lower_s"] if "wall_lower_s" in runs else pd.Series(np.nan, index=runs.index)
    unfinished = runs[lower.notna()]
    fig = S.figure("double", 58)
    axes = [S.ax_mm(fig, 14, 8, 62, 40), S.ax_mm(fig, 96, 8, 62, 40)]
    for ax, col, ylab in zip(axes, ("wall_s", "peak_mb"), ("wall time (s)", "peak memory (MB)")):
        for tool, colour in SIM.items():
            sub = df[df.tool == tool]
            if tool == "iscc":
                if col == "peak_mb":
                    _series(ax, sub[sub.variant == ISCC_BEFORE], col, colour, ls="--", alpha=0.35,
                            label="iscc before chunked cell tables")
                sub = sub[sub.variant == ISCC_VARIANT]
            if sub.empty:
                continue
            _series(ax, sub, col, colour, marker=MARKER.get(tool, "o"), label=LABEL[tool])
            lb = unfinished[unfinished.tool == tool].groupby("n_cancer").wall_lower_s.min()
            if col == "wall_s" and len(lb):
                ax.plot(lb.index, lb.values, ls="none", marker="^", ms=4, mfc="white", mec=colour,
                        label=f"{tool} not finished: at least")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("cancer cells")
        ax.set_ylabel(ylab)
    axes[0].legend(loc="upper left")
    axes[1].legend(loc="upper left", handles=[h for h in axes[1].get_legend_handles_labels()[0]
                                              if h.get_label().startswith("iscc before")])
    S.panel_letter(fig, "a", 2, 2)
    S.panel_letter(fig, "b", 84, 2)
    S.save(fig, "supp_runtime", stamp=False)


if __name__ == "__main__":
    main()
