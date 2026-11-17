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
LABEL = {"real": "real (Ding A vs B)", "iscc": "iscc", "splatter": "Splatter",
         "splatteriscc": "Splatter, iscc's size"}


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


def main_panel(v):
    S.setup()
    vs, groups = v["versions_stats"], _versions(v)
    fig = S.figure("double", 52)
    axes = [S.ax_mm(fig, 14, 7, 46, 36), S.ax_mm(fig, 74, 7, 46, 36), S.ax_mm(fig, 136, 7, 42, 36)]
    for ax, key, ylab in ((axes[0], "mean_log2fc", "mean log$_2$ FC (B / A)"),
                          (axes[1], "sd_log2fc", "s.d. of log$_2$ FC (B / A)")):
        for pre in ("splatteriscc", "splatter", "iscc"):
            for i, n in enumerate(groups.get(pre, [])):
                x, y = _decile_curve(vs[n], key)
                ax.plot(x, y, color=COL[pre], lw=0.6, alpha=0.75, ls=("--" if pre == "splatteriscc" else "-"),
                        label=(LABEL[pre] if i == 0 else None))
        x, y = _decile_curve(vs["real"], key)
        ax.plot(x, y, color=S.INK, lw=1.2, marker="o", ms=2.2, label=LABEL["real"], zorder=5)
        ax.set_xlabel("log$_{10}$ mean expression (decile)")
        ax.set_ylabel(ylab)
        if key == "mean_log2fc":
            ax.axhline(0, color=S.GREY, lw=0.4, zorder=0)
    axes[0].legend(loc="best")
    s_r = v["versions_stats"]["real"]["summary"]
    S.stat(axes[0], f"real Spearman {s_r['S1_spearman']:.2f}", "lower right")
    S.stat(axes[1], f"real low/high ratio {s_r['S1_spread_ratio']:.2f}", "upper right")
    ax = axes[2]
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
    ax.set_xticklabels([LABEL[g].replace(" (Ding A vs B)", "").replace(", ", ",\n") for g in order])
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylabel("correlation of log$_2$ FC\nbetween cell-type pairs")
    for letter, x in (("a", 2), ("b", 63), ("c", 125)):
        S.panel_letter(fig, letter, x, 2)
    return S.save(fig, "fig5_batch_validation")


SUPP = [("S0", "S0: s.d. of log$_2$ FC"), ("S1_spearman", "S1: Spearman(FC, mean)"),
        ("S1_spread_ratio", "S1: spread low / high"), ("S2_slope_low", "S2: slope, low tercile"),
        ("S2_slope_mid", "S2: slope, mid tercile"), ("S2_slope_high", "S2: slope, high tercile"),
        ("S3_median_corr", "S3: median pair correlation"), ("silhouette_batch", "S4: silhouette, batch"),
        ("silhouette_celltype", "S4: silhouette, cell type"), ("ilisi_batch", "S4: iLISI, batch"),
        ("clisi_celltype", "S4: cLISI, cell type")]


def supp_panel(v):
    S.setup()
    vs, groups = v["versions_stats"], _versions(v)
    order = [g for g in ("real", "iscc", "splatter", "splatteriscc") if g in groups]
    fig = S.figure("double", 118)
    cols, w, h, gx, gy = 5, 27, 24, 9.5, 13
    for k, (key, title) in enumerate(SUPP):
        r, c = divmod(k, cols)
        ax = S.ax_mm(fig, 10 + c * (w + gx), 6 + r * (h + gy), w, h)
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
        ax.set_xticklabels([{"real": "real", "iscc": "iscc", "splatter": "Splat.", "splatteriscc": "Splat.*"}[g]
                            for g in order])
        ax.set_xlim(-0.6, len(order) - 0.4)
        ax.set_title(title)
    sp = v.get("splatter", {})
    for k, (key, xlab, target_lab) in enumerate((("de_tuning", "de.facLoc", "real silhouette"),
                                                  ("batch_tuning", "batch.facScale", "real S0"))):
        tr = sp.get(key)
        if not tr:
            continue
        idx = len(SUPP) + k
        r, c = divmod(idx, cols)
        ax = S.ax_mm(fig, 10 + c * (w + gx), 6 + r * (h + gy), w, h)
        pts = sorted(tr["trace"], key=lambda d: d["value"])
        ax.plot([d["value"] for d in pts], [d["stat"] for d in pts], color=COL["splatter"], lw=0.7, marker="o", ms=1.8)
        ax.axhline(tr["target"], color=S.GREY, lw=0.5, ls="--")
        chosen = sp.get("de_facLoc" if key == "de_tuning" else "batch_facScale")
        if chosen is not None:
            ax.axvline(chosen, color=S.INK, lw=0.5)
        ax.set_xlabel(xlab)
        ax.set_title(f"Splatter tuning ({target_lab})")
    S.text_mm(fig, 10, 116, "Diamond: real; dots: five seeds; bar: median. Splat.* = Splatter at iscc's S0 "
              "(only when iscc's S0 differs from the real one by more than 1.5x).", fontsize=5.0, color=S.GREY)
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
