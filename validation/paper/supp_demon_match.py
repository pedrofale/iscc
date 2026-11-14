"""Supplementary figure: iscc with the tissue off reproduces demon (the demon match).

Reads the per-run results that ``comparisons/demon_match.py`` wrote on the cluster and that are
committed, small, under ``comparisons/results/``:

* ``demon_match.csv``: one row per run (model, condition, seed, n, D, J1, clones, size, time);
* ``demon_match_curves.csv``: growth curves (generations, cancer cells, occupied demes);
* ``demon_match_clones.csv``: driver-clone sizes at the end.

Panels: (a) growth in time, (b) occupied demes against size, (c) the clone-size distribution and
(d) Noble's indices n, D and J1 per seed, each for neutral growth and for selection (s = 0.1). The
selection row adds iscc with its fixed driver effect set to demon's realised one (about 0.2, since
demon draws 1 + s·Exp(1) and the larger effects are the ones that establish).
The comparison does not depend on the paper's scale tier, so the figure carries no preview stamp.

Usage::

    python validation/paper/supp_demon_match.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import style as S  # noqa: E402

RES = os.path.join(HERE, "comparisons", "results")
# Fixed colours for the external simulators (Okabe-Ito, which stays distinct under colour blindness);
# iscc keeps the house blue.
DEMON = "#E69F00"
# Each row's series: (model, condition in the results, label, colour).
ROWS = {
    "neutral (s = 0)": [("demon", "neutral", "demon", DEMON), ("iscc", "neutral", "iscc", S.SOURCE["iscc"])],
    "selection (s = 0.1)": [("demon", "selection", "demon", DEMON),
                            ("iscc", "selection", "iscc", S.SOURCE["iscc"]),
                            ("iscc", "realised", "iscc, s = 0.2", "#9DB7E0")],
}


def main():
    S.setup()
    runs = pd.read_csv(os.path.join(RES, "demon_match.csv"))
    curves = pd.read_csv(os.path.join(RES, "demon_match_curves.csv"))
    clones = pd.read_csv(os.path.join(RES, "demon_match_clones.csv"))
    fig = S.figure("double", 106)
    x0, w, gap = 12, 32, 10           # three panels, then three 11-mm index panels: 183 mm
    for row, (title, series) in enumerate(ROWS.items()):
        y = 6 + row * 50
        S.text_mm(fig, x0, y - 1, title, fontsize=6.5, va="bottom")
        a = S.ax_mm(fig, x0, y + 2, w, 30)
        b = S.ax_mm(fig, x0 + w + gap, y + 2, w, 30)
        c = S.ax_mm(fig, x0 + 2 * (w + gap), y + 2, w, 30)
        for model, cond, label, colour in series:
            sub = curves[(curves.model == model) & (curves.cond == cond)]
            for _, g in sub.groupby("seed"):
                a.plot(g.t, g.cells, color=colour, lw=0.5, alpha=0.6)
                b.plot(g.cells, g.demes, color=colour, lw=0.5, alpha=0.6)
            cs = clones[(clones.model == model) & (clones.cond == cond)]["size"].to_numpy(float)
            cs = np.sort(cs)[::-1]
            if len(cs):
                c.plot(cs, np.arange(1, len(cs) + 1) / len(cs), color=colour, lw=0.8, label=label)
        a.set_yscale("log")
        a.set_xlabel("generations")
        a.set_ylabel("cancer cells")
        b.set_xlabel("cancer cells")
        b.set_ylabel("occupied demes")
        c.set_xscale("log")
        c.set_yscale("log")
        c.set_xlabel("clone size (cells)")
        c.set_ylabel("fraction of clones this large or larger")
        c.legend(loc="lower left")
        # (d) the indices: one dot per seed, models side by side
        for k, ind in enumerate(("n", "D", "J1")):
            ax = S.ax_mm(fig, x0 + 3 * (w + gap) + k * 15, y + 2, 11, 30)
            for j, (model, cond, label, colour) in enumerate(series):
                v = runs[(runs.model == model) & (runs.cond == cond)][ind].to_numpy(float)
                xs = j + np.linspace(-0.15, 0.15, len(v)) if len(v) > 1 else [j]
                ax.scatter(xs, v, s=3, color=colour)
                if len(v):
                    ax.hlines(np.nanmedian(v), j - 0.3, j + 0.3, color=S.INK, lw=0.6)
            ax.set_xlim(-0.6, len(series) - 0.4)
            ax.set_xticks(range(len(series)))
            ax.set_xticklabels([lab for _, _, lab, _ in series], rotation=90)
            ax.set_title(ind)
        if row == 0:
            for letter, x in zip("abcd", (x0, x0 + w + gap, x0 + 2 * (w + gap), x0 + 3 * (w + gap))):
                S.panel_letter(fig, letter, x - 9, 2)
    S.save(fig, "supp_demon_match", stamp=False)


if __name__ == "__main__":
    main()
