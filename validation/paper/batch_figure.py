"""Figures of Fig 5's batch panel, from validation.json (``batch_validation.py run``).

* ``fig5_batch_validation``: the panel before b-c, S1 and S3 for the real A-B contrast, ``iscc``'s
  and Splatter's: the signed and the spread of the per-gene log2 fold change by mean expression
  (the median over cell types of each decile), and the correlation of the fold changes between
  pairs of cell types.
* ``supp_batch_validation``: S0-S4, one small axis per summary (real; iscc's and Splatter's five
  seeds and their median), and Splatter's two tuning curves.

Written through ``style.save`` into ``<paper>/figures/paper1/`` (``$ISCC_PAPER_DIR``, see
``validation/_paths.py``).

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
LABEL = {"real": "real", "iscc": "iscc", "splatter": "Splatter", "splatteriscc": "Splatter matched to iscc"}


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
    ("mean_log2fc" or "sd_log2fc"), real against every seed of iscc and Splatter. Shared by
    :func:`main_panel` and Fig 5's panel B (``fig5_figure.py``); the summary statistics are in
    validation.json and the text, not printed on the axes."""
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
    ax.set_xticklabels([LABEL[g].replace(" matched", "\nmatched") for g in order])
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylabel("correlation of log$_2$\nfold changes")


def main_panel(v):
    S.setup()
    fig = S.figure("double", 52)
    axes = [S.ax_mm(fig, 14, 7, 46, 36), S.ax_mm(fig, 74, 7, 46, 36), S.ax_mm(fig, 136, 7, 42, 36)]
    draw_s1(axes[0], v, "mean_log2fc")
    draw_s1(axes[1], v, "sd_log2fc")
    axes[0].legend(loc="best")
    draw_s3(axes[2], v)
    for letter, x in (("A", 2), ("B", 63), ("C", 125)):
        S.panel_letter(fig, letter, x, 2)
    return S.save(fig, "fig5_batch_validation")


# The supplementary panels: (key, title, y label). Titles are the statistics' names, defined in the
# Methods; the y label is the measured quantity (style guide, "Text inside figures").
SUPP = [("S0", "S0", "s.d. of log$_2$ fold change"),
        ("S1_spearman", "S1", "Spearman correlation of\nfold change and mean"),
        ("S1_spread_ratio", "S1", "fold change s.d., lowest\nover highest tercile"),
        ("S2_slope_low", "S2 in the lowest tercile", "slope of detection\nchange on fold change"),
        ("S2_slope_mid", "S2 in the middle tercile", "slope of detection\nchange on fold change"),
        ("S2_slope_high", "S2 in the highest tercile", "slope of detection\nchange on fold change"),
        ("S3_median_corr", "S3", "median correlation\nof fold changes"),
        ("silhouette_batch", "S4", "silhouette of batches"),
        ("silhouette_celltype", "S4", "silhouette of cell types"),
        ("ilisi_batch", "S4", "iLISI"),
        ("clisi_celltype", "S4", "cLISI")]
SUPP_TICK = {"real": "real", "iscc": "iscc", "splatter": "Splatter", "splatteriscc": "Splatter\nmatched"}
TUNING = (("de_tuning", "de_facLoc", "differential expression\nfactor location", "silhouette of cell types"),
          ("batch_tuning", "batch_facScale", "batch factor scale", "s.d. of log$_2$ fold change"))


def supp_panel(v):
    """S0-S4, one small axis per statistic (real: diamond; iscc's and Splatter's five seeds: dots; their
    median: bar), and Splatter's two tuning curves (the real value: dashed; the chosen value: solid).
    The encodings are stated in the caption, not in the figure."""
    S.setup()
    vs, groups = v["versions_stats"], _versions(v)
    order = [g for g in ("real", "iscc", "splatter", "splatteriscc") if g in groups]
    fig = S.figure("double", 112)
    cols, w, h, x0, gx, gy = 5, 25, 22, 11, 11, 14
    letters = "ABCDEFGHIJKLMNOP"

    def place(k):
        r, c = divmod(k, cols)
        x, y = x0 + c * (w + gx), 7 + r * (h + gy)
        S.panel_letter(fig, letters[k], x - 9, y - 4.5)
        return S.ax_mm(fig, x, y, w, h)
    for k, (key, title, ylab) in enumerate(SUPP):
        ax = place(k)
        for xi, g in enumerate(order):
            vals = []
            for n in groups[g]:
                st = vs[n]
                vals.append(st["summary"].get(key, st["S4"].get(key)) if key in st["summary"] else st["S4"].get(key))
            vals = np.array([np.nan if x is None else x for x in vals], float)
            if g == "real":
                ax.scatter([xi], vals, marker="D", s=9, color=S.INK, zorder=5)
            else:
                ax.scatter(xi + np.linspace(-0.15, 0.15, len(vals)), vals, s=5,
                           facecolors=("none" if g == "splatteriscc" else COL[g]), edgecolors=COL[g],
                           linewidths=0.5)
                ax.hlines(np.nanmedian(vals), xi - 0.28, xi + 0.28, color=S.INK, lw=0.8)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([SUPP_TICK[g] for g in order], fontsize=5.0)
        ax.set_xlim(-0.6, len(order) - 0.4)
        ax.set_title(title)
        ax.set_ylabel(ylab, fontsize=5.5)
    sp = v.get("splatter", {})
    k = len(SUPP)
    for key, chosen_key, xlab, ylab in TUNING:
        tr = sp.get(key)
        if not tr:
            continue
        ax = place(k)
        k += 1
        pts = sorted(tr["trace"], key=lambda d: d["value"])
        ax.plot([d["value"] for d in pts], [d["stat"] for d in pts], color=COL["splatter"], lw=0.7, marker="o", ms=1.8)
        ax.axhline(tr["target"], color=S.GREY, lw=0.5, ls="--")
        chosen = sp.get(chosen_key)
        if chosen is not None:
            ax.axvline(chosen, color=S.INK, lw=0.5)
        ax.set_xlabel(xlab, fontsize=5.5)
        ax.set_ylabel(ylab, fontsize=5.5)
        ax.set_title("Splatter tuning")
    return S.save(fig, "supp_batch_validation")


def main(path):
    with open(path) as f:
        v = json.load(f)
    a = main_panel(v)
    b = supp_panel(v)
    print(f"[batch figure] {a}\n[batch figure] {b}")
    return a, b


if __name__ == "__main__":
    main(sys.argv[1])
