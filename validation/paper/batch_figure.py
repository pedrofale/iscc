"""S13 Fig, the batch validation, from validation.json (``batch_validation.py``).

``supp_batch_validation``: the real A-B contrast of Ding et al., ``iscc``'s and Splatter's, in one
figure, 165 mm wide (the text width) so that it is included at its true size:

* A, B (S1): the mean and the spread of the per-gene log2 fold change by mean expression (the median
  over cell types of each decile), real against every seed of iscc and Splatter;
* C (S3): the correlation of the fold changes between pairs of cell types, every pair of every seed;
* D-G (S0, S1, S2, S4): one small axis per registered summary, real (diamond and dashed line)
  against the five seeds of iscc and of Splatter (dots) and their median (bar).

Splatter's tuning curves stay in validation.json; the caption gives the values chosen.

The panels A-C were Fig 5's panel B; ``draw_s1``, ``draw_s3``, ``COL``, ``LABEL`` and ``_versions``
are shared with ``fig5_figure.py``. Written through ``style.save`` into ``<paper>/figures/paper1/``
(``$ISCC_PAPER_DIR``, see ``validation/_paths.py``).

    PYTHONPATH=src python validation/paper/batch_figure.py $D/batchval/validation.json
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.dirname(HERE), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

import style as S  # noqa: E402

# One colour per entity for the whole paper: iscc and real from style.SOURCE; Splatter's is the
# data-simulator supplement's (style.TOOL["Splatter"]).
COL = {"real": S.SOURCE["real"], "iscc": S.SOURCE["iscc"], "splatter": S.TOOL.get("Splatter", "#A50026"),
       "splatteriscc": S.TOOL.get("Splatter", "#A50026")}
# Labels follow the paper's style for text inside figures (no parentheticals or settings: the caption
# names the data set and how Splatter was matched).
LABEL = {"real": "real", "iscc": "iscc", "splatter": "Splatter", "splatteriscc": "Splatter at iscc's size"}
TICK = {"real": "real", "iscc": "iscc", "splatter": "Splatter", "splatteriscc": "Splatter at\niscc's size"}


def _versions(v):
    names = list(v["versions_stats"])
    groups = {"real": ["real"]}
    for pre in ("iscc", "splatter", "splatteriscc"):
        g = sorted(n for n in names if n.startswith(pre + "_s"))
        if g:
            groups[pre] = g
    return groups


def _decile_curve(st, key):
    """Median over the scored cell types of each decile's (log mean, value)."""
    per = [r["curve"] for r in st["per_type"].values() if "curve" in r]
    x = np.median(np.array([c["log_mean"] for c in per]), axis=0)
    y = np.median(np.array([c[key] for c in per]), axis=0)
    return x / np.log(10), y                               # log10 of the mean for the axis


S1_YLABEL = {"mean_log2fc": "mean log$_2$ fold change", "sd_log2fc": "s.d. of log$_2$ fold change"}


def draw_s1(ax, v, key):
    """S1 on ``ax``: the median over cell types of each mean-expression decile's ``key``
    ("mean_log2fc" or "sd_log2fc"), real against every seed of iscc and Splatter. Shared with
    ``fig5_figure.py``; the summary statistics are in validation.json and the text, not on the axes."""
    vs, groups = v["versions_stats"], _versions(v)
    for pre in ("splatteriscc", "splatter", "iscc"):
        for i, n in enumerate(groups.get(pre, [])):
            x, y = _decile_curve(vs[n], key)
            ax.plot(x, y, color=COL[pre], lw=0.6, alpha=0.75, ls=("--" if pre == "splatteriscc" else "-"),
                    label=(LABEL[pre] if i == 0 else None))
    x, y = _decile_curve(vs["real"], key)
    ax.plot(x, y, color=S.INK, lw=1.2, marker="o", ms=2.2, label=LABEL["real"], zorder=5)
    ax.set_xlabel("log$_{10}$ mean expression")
    ax.set_ylabel(S1_YLABEL[key])
    if key == "mean_log2fc":
        ax.axhline(0, color=S.GREY, lw=0.4, zorder=0)


def draw_s3(ax, v):
    """S3 on ``ax``: the correlation of the log2 fold changes between pairs of cell types, every
    pair of every seed, with the median of the seeds' medians. Shared as :func:`draw_s1`."""
    vs, groups = v["versions_stats"], _versions(v)
    order = [g for g in ("real", "iscc", "splatter", "splatteriscc") if g in groups]
    rng = np.random.default_rng(0)
    for xi, g in enumerate(order):
        vals = [c for n in groups[g] for c in vs[n]["S3_pairs"].values()]
        meds = [vs[n]["summary"]["S3_median_corr"] for n in groups[g]]
        jitter = rng.uniform(-0.18, 0.18, len(vals))
        ax.scatter(np.full(len(vals), xi) + jitter, vals, s=2.0, color=(S.INK if g == "real" else COL[g]),
                   alpha=(0.8 if g == "real" else 0.35), linewidths=0)
        ax.hlines(np.median(meds), xi - 0.3, xi + 0.3, color=S.INK, lw=1.0, zorder=5)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([TICK[g] for g in order])
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylabel("correlation of log$_2$\nfold changes")


# ------------------------------------------------------------------ S13 Fig
# The registered summaries, one small axis each: (key, title, x label). Titles name what is measured;
# the caption maps them to S0-S4 (style guide, "Text inside figures").
ROWS = [
    ("D", [("S0", "Size", "s.d. of log$_2$ fold change")]),
    ("E", [("S1_spearman", "Dependence on mean", "Spearman correlation of\nfold change and mean"),
           ("S1_spread_ratio", "Spread by mean", "fold change s.d., lowest over\nhighest tercile of mean")]),
    ("F", [("S2_slope_low", "Lowest tercile of mean", "slope of detection change\non log$_2$ fold change"),
           ("S2_slope_mid", "Middle tercile of mean", "slope of detection change\non log$_2$ fold change"),
           ("S2_slope_high", "Highest tercile of mean", "slope of detection change\non log$_2$ fold change")]),
    ("G", [("silhouette_batch", "Batches", "silhouette"),
           ("silhouette_celltype", "Cell types", "silhouette"),
           ("ilisi_batch", "Batches", "iLISI"),
           ("clisi_celltype", "Cell types", "cLISI")]),
]
# Drawn at the text width, so that it is included at its true size (no text below 5.5 pt) and leaves
# room on the page for the caption.
W_MM, LEFT, RIGHT = 165, 24, 163


def _value(st, key):
    x = st["summary"].get(key) if key in st["summary"] else st["S4"].get(key)
    return np.nan if x is None else float(x)


def _strip(ax, v, key, order, labels):
    """One summary: categories as rows (real on top), the statistic on the x axis."""
    vs, groups = v["versions_stats"], _versions(v)
    real = _value(vs["real"], key)
    ax.axvline(real, color=S.GREY, lw=0.5, ls="--", zorder=0)
    for i, g in enumerate(order):
        vals = np.array([_value(vs[n], key) for n in groups[g]])
        if g == "real":
            ax.scatter(vals, [i], marker="D", s=9, color=S.INK, zorder=5)
            continue
        ax.scatter(vals, i + np.linspace(-0.18, 0.18, len(vals)), s=6, linewidths=0.6,
                   facecolors=("none" if g == "splatteriscc" else COL[g]), edgecolors=COL[g], zorder=4)
        ax.vlines(np.nanmedian(vals), i - 0.32, i + 0.32, color=S.INK, lw=1.0, zorder=5)
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([TICK[g] for g in order] if labels else [])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)


def supp_panel(v):
    S.setup()
    groups = _versions(v)
    order = [g for g in ("real", "iscc", "splatter", "splatteriscc") if g in groups]
    strip_h = 4.0 * len(order) + 1.0
    y_ac, h_ac = 7, 30                               # A-C, then the rows D-E, F and G
    y_rows = [y_ac + h_ac + 20]
    for _ in range(2):
        y_rows.append(y_rows[-1] + strip_h + 19)
    fig = S.figure(W_MM, y_rows[-1] + strip_h + 10)

    # A-C: the former Fig 5B
    axA, axB, axC = S.ax_mm(fig, 13, y_ac, 38, h_ac), S.ax_mm(fig, 64, y_ac, 38, h_ac), S.ax_mm(fig, 117, y_ac, 46, h_ac)
    draw_s1(axA, v, "mean_log2fc")
    draw_s1(axB, v, "sd_log2fc")
    hl = dict(zip(*axA.get_legend_handles_labels()[::-1]))
    names = [LABEL[g] for g in order if LABEL[g] in hl]
    axA.legend([hl[n] for n in names], names, loc="best")
    draw_s3(axC, v)
    for letter, x in (("A", 1), ("B", 53), ("C", 104)):
        S.panel_letter(fig, letter, x, y_ac - 4)

    # D-G: the registered summaries; D and E share a row
    def row(y, items, w, gap, letters):
        for j, ((key, title, xlab), letter) in enumerate(zip(items, letters)):
            x = LEFT + j * (w + gap)
            ax = S.ax_mm(fig, x, y, w, strip_h)
            _strip(ax, v, key, order, labels=(j == 0))
            ax.set_title(title)
            ax.set_xlabel(xlab)
            if letter:
                S.panel_letter(fig, letter, (1 if j == 0 else x - 6), y - 5)
    w3, g3 = (RIGHT - LEFT - 2 * 8) / 3, 8
    w4, g4 = (RIGHT - LEFT - 3 * 6) / 4, 6
    row(y_rows[0], ROWS[0][1] + ROWS[1][1], w3, g3, ["D", "E", None])
    row(y_rows[1], ROWS[2][1], w3, g3, ["F", None, None])
    row(y_rows[2], ROWS[3][1], w4, g4, ["G", None, None, None])
    return S.save(fig, "supp_batch_validation")


def main(path):
    with open(path) as f:
        v = json.load(f)
    out = supp_panel(v)
    print(f"[batch figure] {out}")
    return out


if __name__ == "__main__":
    main(sys.argv[1])
