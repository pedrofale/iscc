"""Supplementary figure: iscc with the tissue off against demon (the demon match and clonal territories).

Reads the per-run results that ``comparisons/demon_match.py`` wrote on the cluster and that are
committed, small, under ``comparisons/results/``:

* ``demon_match.csv``: one row per run (model, engine, condition, dispersal, driver rate, seed, n, D,
  J1, clones, territory score, size, time);
* ``demon_match_curves.csv``: growth curves of the match (generations, cancer cells, occupied demes);
* ``demon_match_clones.csv``: driver-clone sizes at the end of the match;
* ``demon_match_maps.csv``: dominant-clone maps of the territory sweep's first seed.

Rows 1 and 2, the match (K 16, dispersal 0.1, 2e-3 drivers per division, 20,000 cancer cells, ten
seeds), for neutral growth and for selection (s = 0.1): (a) growth in time, (b) occupied demes against
size, (c) the clone-size distribution and (d) Noble's indices n, D and J1 per seed. iscc runs on both
of its engines, the exact one (Gillespie's direct method, demon's algorithm) and the paper's
tau-leaping (binomial, Cao-Gillespie-Petzold step at epsilon 0.01). The selection row adds iscc with
its fixed driver effect set to demon's realised one (about 0.2, since demon draws 1 + s·Exp(1) and the
larger effects are the ones that establish).
Row 3, clonal territories against dispersal (neutral marks at 2e-2 per division, five seeds per
rate): (e) the dominant clone of every deme in demon and in iscc's tau-leaping at the lowest and the
highest dispersal (seed 1), (f) the share of neighbouring demes with the same dominant clone beside its
expectation under random placement (each mean and standard deviation over the seeds), and (g) the clones
it is computed over.
The comparison does not depend on the paper's scale tier, so the figure carries no preview stamp.

Usage::

    python validation/paper/supp_demon_match.py
"""
import os
import sys

import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import style as S  # noqa: E402

RES = os.path.join(HERE, "comparisons", "results")
MU_MATCH, MU_TERR, DISP_MATCH = 1e-3, 1e-2, 0.1
# Fixed colours for the external simulators (Okabe-Ito, which stays distinct under colour blindness);
# iscc keeps the house blue, its exact engine dashed, its tau-leaping (the paper's engine) solid.
DEMON = "#E69F00"
ISCC, REALISED = S.SOURCE["iscc"], "#9DB7E0"
LS = {"gillespie": "-", "exact": (0, (2.5, 1.2)), "tau": "-"}
# Each row's series: (model, engine, condition in the results, label, colour).
ROWS = {
    "neutral": [("demon", "gillespie", "neutral", "demon", DEMON),
                ("iscc", "exact", "neutral", "iscc exact", ISCC),
                ("iscc", "tau", "neutral", "iscc τ-leap", ISCC)],
    "selection": [("demon", "gillespie", "selection", "demon", DEMON),
                  ("iscc", "exact", "selection", "iscc exact", ISCC),
                  ("iscc", "tau", "selection", "iscc τ-leap", ISCC),
                  ("iscc", "exact", "realised", "iscc exact, s = 1.2", REALISED),
                  ("iscc", "tau", "realised", "iscc τ-leap, s = 1.2", REALISED)],
}
TERR = [("demon", "gillespie", "demon", DEMON), ("iscc", "exact", "iscc exact", ISCC),
        ("iscc", "tau", "iscc τ-leap", ISCC)]
# clone colours by size rank (largest first), ancestral in light grey
CLONE_RANK = [fam[i] for i in (1, 2) for fam in S.CLONE_FAMILIES[:-1]] + ["#BBBBBB"] * 30
ANCESTRAL = "#D0D0D0"


def _sel(df, model, engine, cond=None, mu=MU_MATCH, disp=DISP_MATCH):
    m = (df.model == model) & (df.engine == engine) & np.isclose(df.mu, mu) & np.isclose(df.migration, disp)
    return df[m & (df.cond == cond)] if cond is not None else df[m]


def _map_rgb(sub, half=26):
    """RGB image of one run's dominant clones, centred on the tumour's centroid."""
    r0, c0 = int(round(sub.row.mean())), int(round(sub.col.mean()))
    img = np.ones((2 * half + 1, 2 * half + 1, 3))
    for r, c, k in zip(sub.row, sub.col, sub.clone):
        i, j = r - r0 + half, c - c0 + half
        if 0 <= i < img.shape[0] and 0 <= j < img.shape[1]:
            img[i, j] = to_rgb(ANCESTRAL if k < 0 else CLONE_RANK[k])
    return img


def main():
    S.setup()
    runs = pd.read_csv(os.path.join(RES, "demon_match.csv"))
    curves = pd.read_csv(os.path.join(RES, "demon_match_curves.csv"))
    clones = pd.read_csv(os.path.join(RES, "demon_match_clones.csv"))
    maps = pd.read_csv(os.path.join(RES, "demon_match_maps.csv"))
    fig = S.figure("double", 180)
    x0, w, gap = 12, 28, 12
    xi, wi, gi = 130, 13, 3.5            # the three index panels
    for row, (title, series) in enumerate(ROWS.items()):
        y = 6 + row * 52
        S.text_mm(fig, x0, y - 1, title, fontsize=6.5, va="bottom", fontweight="bold")
        a = S.ax_mm(fig, x0, y + 2, w, 30)
        b = S.ax_mm(fig, x0 + w + gap, y + 2, w, 30)
        c = S.ax_mm(fig, x0 + 2 * (w + gap), y + 2, w, 30)
        for model, engine, cond, label, colour in series:
            sub = _sel(curves, model, engine, cond)
            for _, g in sub.groupby("seed"):
                a.plot(g.t, g.cells, color=colour, lw=0.5, alpha=0.6, ls=LS[engine])
                b.plot(g.cells, g.demes, color=colour, lw=0.5, alpha=0.6, ls=LS[engine])
            cs = np.sort(_sel(clones, model, engine, cond)["size"].to_numpy(float))[::-1]
            if len(cs):
                c.plot(cs, np.arange(1, len(cs) + 1) / len(cs), color=colour, lw=0.8, ls=LS[engine],
                       label=label)
        a.set_yscale("log")
        a.set_xlabel("Generations")
        a.set_ylabel("Cancer cells")
        b.set_xlabel("Cancer cells")
        b.set_ylabel("Occupied demes")
        c.set_xscale("log")
        c.set_yscale("log")
        c.set_xlabel("Clone size (cells)")
        c.set_ylabel("Fraction of clones\nthis large or larger")
        c.legend(loc="lower left", handlelength=1.8)
        # (d) the indices: one dot per seed, models side by side, the median as a bar
        for k, ind in enumerate(("n", "D", "J1")):
            ax = S.ax_mm(fig, xi + k * (wi + gi), y + 2, wi, 30)
            for j, (model, engine, cond, label, colour) in enumerate(series):
                v = _sel(runs, model, engine, cond)[ind].to_numpy(float)
                xs = j + np.linspace(-0.18, 0.18, len(v)) if len(v) > 1 else [j]
                ax.scatter(xs, v, s=2.5, color=colour, marker="s" if engine == "exact" else "o")
                if len(v):
                    ax.hlines(np.nanmedian(v), j - 0.32, j + 0.32, color=S.INK, lw=0.6)
            ax.set_xlim(-0.6, len(series) - 0.4)
            ax.set_xticks(range(len(series)))
            ax.set_xticklabels([s[3] for s in series], rotation=90)
            ax.set_title({"J1": "$J^1$"}.get(ind, ind))
        if row == 0:
            for letter, x in zip("abcd", (x0, x0 + w + gap, x0 + 2 * (w + gap), xi)):
                S.panel_letter(fig, letter, x - 9, 2)

    # --- row 3: territories against dispersal ----------------------------------------------------
    y = 114
    S.text_mm(fig, x0, y - 1, "clonal territories against dispersal", fontsize=6.5, va="bottom",
              fontweight="bold")
    terr = runs[np.isclose(runs.mu, MU_TERR) & (runs.cond == "neutral")].copy()
    disps = sorted(terr.migration.unique())
    lo, hi = (disps[0], disps[-1]) if disps else (None, None)
    for i, (model, engine, label) in enumerate([("demon", "gillespie", "demon"), ("iscc", "tau", "iscc τ-leap")]):
        for jd, d in enumerate((lo, hi)):
            ax = S.ax_mm(fig, x0 - 6 + jd * 25, y + 5 + i * 25, 22, 22)
            S.despine(ax, False, False)
            sub = (maps[(maps.model == model) & (maps.engine == engine) & np.isclose(maps.migration, d)
                        & (maps.seed == 1)] if d is not None else maps.iloc[:0])
            if len(sub):
                ax.imshow(_map_rgb(sub), interpolation="nearest")
            if i == 0 and d is not None:
                ax.set_title(f"dispersal {d:g}", fontsize=6)
            if jd == 0:
                ax.set_ylabel(label, fontsize=6)
    S.panel_letter(fig, "e", x0 - 9, y + 2)
    f = S.ax_mm(fig, 76, y + 6, 40, 38)
    g = S.ax_mm(fig, 134, y + 6, 40, 38)
    # (f) the observed share of same-clone neighbours (filled) beside its expectation under random
    # placement (open), (g) the clones above 4% of cells: mean and standard deviation over the seeds
    from matplotlib.lines import Line2D
    handles = []
    for model, engine, label, colour in TERR:
        sub = terr[(terr.model == model) & (terr.engine == engine)]
        if sub.empty:
            continue
        mkr = "s" if engine == "exact" else "o"
        for ax, col, kw in ((f, "assortment", {}), (f, "assortment_expected", dict(mfc="white", lw=0.6)),
                            (g, "n_territory_clones", {})):
            st = sub.groupby("migration")[col]
            ax.errorbar(st.mean().index, st.mean().values, yerr=st.std().values, color=colour,
                        ls=LS[engine], marker=mkr, ms=2.5, capsize=1.5, **{"lw": 0.8, **kw})
        handles.append(Line2D([], [], color=colour, ls=LS[engine], marker=mkr, ms=2.5, lw=0.8, label=label))
    handles += [Line2D([], [], color=S.GREY, ls="none", marker="o", ms=2.5, label="observed"),
                Line2D([], [], color=S.GREY, ls="none", marker="o", ms=2.5, mfc="white",
                       label="random placement")]
    f.set_ylim(0, 1.05)
    for ax in (f, g):
        ax.set_xscale("log")
        ax.set_xlabel("Dispersal per division")
        if disps:
            ax.set_xticks(disps)
            ax.set_xticklabels([f"{d:g}" for d in disps])
        ax.minorticks_off()
    f.set_ylabel("Share of neighboring demes\nwith the same clone")
    g.set_ylabel("Clones above 4% of cells")
    f.legend(handles=handles, loc="lower right", handlelength=1.8)
    S.panel_letter(fig, "f", 64, y + 2)
    S.panel_letter(fig, "g", 122, y + 2)
    S.save(fig, "supp_demon_match", stamp=False)


if __name__ == "__main__":
    main()
