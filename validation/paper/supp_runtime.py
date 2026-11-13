"""Runtime panel: iscc against OncoSpot, CINner and SISTEM at matched tumour size, one core.

Reads ``comparisons/results/runtime.csv``, one row per (tool, size, seed) as written by
``comparisons/runtime_collect.py`` from the cluster logs: wall time and peak memory measured by
``/usr/bin/time -v`` around each tool's whole run, on identical Arrhenius CPU nodes. The x axis is
the number of cancer cells requested (for OncoSpot, half its cells, its tumour fraction being 0.5),
so every tool is compared at the tumour size it simulated. Failed attempts are left out; they are
kept in the CSV with ``ok`` False.

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
SIM = {"iscc": S.SOURCE["iscc"], "OncoSpot": "#D55E00", "CINner": "#009E73", "SISTEM": "#CC79A7"}
LABEL = {"iscc": "iscc (grow, section, Visium)", "OncoSpot": "OncoSpot (end to end)",
         "CINner": "CINner (grow, sample, phylogeny)", "SISTEM": "SISTEM (grow, sample, read counts)"}


def main():
    S.setup()
    df = pd.read_csv(os.path.join(RES, "runtime.csv"))
    df = df[df.ok.astype(bool)]        # a failed attempt (OncoSpot's CNAsim, once) is retried, not timed
    fig = S.figure("double", 58)
    axes = [S.ax_mm(fig, 14, 8, 62, 40), S.ax_mm(fig, 96, 8, 62, 40)]
    for ax, col, ylab in zip(axes, ("wall_s", "peak_mb"), ("wall time (s)", "peak memory (MB)")):
        for tool, colour in SIM.items():
            sub = df[df.tool == tool]
            if sub.empty:
                continue
            g = sub.groupby("n_cancer")[col]
            x = np.array(sorted(g.groups))
            med, lo, hi = g.median().loc[x], g.min().loc[x], g.max().loc[x]
            ax.plot(x, med, color=colour, marker="o", label=LABEL[tool])
            ax.vlines(x, lo, hi, color=colour, lw=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("cancer cells")
        ax.set_ylabel(ylab)
    axes[0].legend(loc="upper left")
    S.panel_letter(fig, "a", 2, 2)
    S.panel_letter(fig, "b", 84, 2)
    S.save(fig, "supp_runtime", stamp=False)


if __name__ == "__main__":
    main()
