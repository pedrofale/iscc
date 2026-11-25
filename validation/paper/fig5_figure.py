"""Fig 5: one atlas-sized cohort of 50 patients, and what tools read from it.

Draws ``figures/paper1/fig5_cohort.pdf`` at the text width (panels A, D, E and F below, lettered A to D
in the paper), and panel C, which the half-page layout moved out, as ``supp_cohort_embeddings.pdf``
(S14 Fig), through ``validation/_paths.py`` (``ISCC_PAPER_DIR`` picks the paper checkout), from
summaries only. Panel B, the batch validation, became S13 Fig and is drawn by ``batch_figure.py``.


* ``collect_summary.json`` (``fig5_atlas.py collect``; panels A, E, F) and
  ``integrate/level_<L>.json`` (``batch_cohort.py integrate``; panels C, D) under ``$ISCC_FIG5_OUT``
  (or ``--out``), at the levels of ``batch_levels.LEVELS``; summaries of the first design's 10x /
  intermediate / strong levels, if present, are ignored.

A panel whose summaries are missing is drawn as a placeholder that says so (a preview only: the
figure goes into the paper once its summaries exist). Text inside the figure follows the paper's
style: bold uppercase panel letters, short panel titles, no settings, parentheticals, code names or
possessives, and every axis on the measured quantity's own scale, compared values side by side.
Everything else is in the caption.

A. The cohort: 50 patients, each grown to its own size; the two pooling designs (confounded,
   balanced); the simulated network of event modules E0-E6 with each module's size (genes) and the
   share of patients carrying it at 5% of the cancer cells.
C. Embeddings of the pooled cells at the fitted batch level, without integration and after Harmony
   and scVI, in both designs: microenvironment cells coloured by batch, cancer cells by patient.
D. Integration at the fitted and shared levels and with one technical source at a time: iLISI of the
   microenvironment cells, and the share of neighbours from the same patient and from the same clone
   among cancer cells, each beside its value in the embedding without batch effect (grey diamond).
E. The simulated network from the DNA: each tool's estimate for the simulated relations against the six
   pairs of E0 (grey band: their magnitudes); filled, a relation recovered (simulated sign and a
   magnitude above every E0 pair; for the order, a larger effect of E5 on E6 than of E6 on E5, above
   every E0 pair); the rank among the 21 pairs beside each column; CBN as the edges it returns.
F. Architecture against the mix of stages: the order of breach and the survival traits in the true
   genealogies, and each tool's dependencies among the traits against those between E0 and each trait.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch  # noqa: E402

import batch_levels as BL  # noqa: E402
import fig5_atlas as A  # noqa: E402
import style as S  # noqa: E402

# Colours: the paper's fixed tool colours (style.TOOL), and the series of panel B from batch_figure
# (real: ink; iscc: style.SOURCE["iscc"]; Splatter: batch_figure.COL). Relations in E and F are
# coloured by relation, not by tool (the tool is the column).
TOOL_COLOUR = {"none": S.GREY, "harmony": S.TOOL["Harmony"], "scvi": S.TOOL["scVI"]}
TOOL_LABEL = {"none": "no integration", "harmony": "Harmony", "scvi": "scVI", "cooc": "co-occurrence",
              "mhn": "MHN", "treemhn": "TreeMHN", "cbn": "CBN"}
DESIGNS = ("confounded", "balanced")
LEVEL_LABEL = {"fitted": "fitted", "shared": "shared", "depth": "depth", "dispersion": "dispersion",
               "ambient": "ambient RNA", "dropout": "dropout", "doublets": "doublets",
               "stress": "dissociation stress"}
EMBED_LEVEL = "fitted"
SYNERGY, REDUNDANCY, ORDER = "#2F6D3A", "#C44E52", S.INK
REFERENCE = "#9E9E9E"
FS, FS_SMALL = 5.5, 4.8


def palette50():
    """50 distinguishable colours for 50 batches or patients: they show mixing against separation,
    not identity."""
    import matplotlib as mpl
    cols = []
    for name in ("tab20", "tab20b", "tab20c"):
        cols += [mpl.colors.to_hex(c) for c in mpl.colormaps[name].colors]
    return cols[:50]


def load(out):
    def read(p):
        if not p or not os.path.exists(p):
            return None
        with open(p) as f:
            return json.load(f)
    d = os.path.join(out, "integrate")
    levels = {}
    if os.path.isdir(d):
        for fn in sorted(os.listdir(d)):
            if fn.startswith("level_") and fn.endswith(".json"):
                r = read(os.path.join(d, fn))
                if r and r.get("level") in BL.LEVELS:          # the batch model's levels only
                    levels[r["level"]] = r
    return dict(collect=read(os.path.join(out, "collect_summary.json")), levels=levels)


def placeholder(ax, what):
    S.despine(ax, False, False)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color(S.BG_CELLS)
    ax.text(0.5, 0.5, f"{what}\nnot yet run", ha="center", va="center", fontsize=FS, color=S.GREY,
            transform=ax.transAxes)


def heading(fig, letter, title, x, y):
    """A bold uppercase panel letter and the panel's short title beside it."""
    S.panel_letter(fig, letter, x, y)
    S.text_mm(fig, x + 4.2, y + 0.55, title, fontsize=6.5, va="top")


# ------------------------------------------------------------------ A: the cohort and the network
def panel_a(fig, x, y, coll, w=176):
    h = 31
    ax = S.ax_mm(fig, x, y, w, h)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    pal = palette50()
    # the patients, ordered by their size (dot area follows the size)
    targets = (np.asarray(coll["cohort"]["targets"]) if coll is not None
               else np.asarray(A.cohort_targets(len(A.COHORT_SEEDS))))
    n = len(targets)
    lo, hi = A.TARGET_RANGE
    ax.text(0, 0.5, f"{n} patients", fontsize=FS, va="top")
    for k, i in enumerate(np.argsort(targets)):
        r = 0.35 + 0.65 * np.sqrt(np.clip((targets[i] - lo) / (hi - lo), 0, 1))
        ax.add_patch(Circle((1.2 + k * (66 / max(n - 1, 1)), 6.2), r, fc=pal[i % len(pal)], ec="none"))
    ax.text(0, 8.6, f"{lo:,}", fontsize=FS_SMALL, color=S.GREY, va="top")
    ax.text(68, 8.6, f"{hi:,} cancer cells", fontsize=FS_SMALL, color=S.GREY, va="top", ha="right")
    # the two pooling designs of the same cells
    for k, (name, mix) in enumerate((("confounded", False), ("balanced", True))):
        y0 = 15.0 + 8.4 * k
        ax.text(0, y0 + 3.4, name, fontsize=FS, va="center")
        for b in range(6):
            bx = 13.5 + 6.0 * b
            ax.add_patch(FancyBboxPatch((bx, y0 + 0.9), 4.8, 5.0, boxstyle="round,pad=0.15", fc="white",
                                        ec=S.INK, lw=0.4))
            for q in range(6):
                p = (6 * b + 7 * q) % 50 if mix else b
                ax.add_patch(Circle((bx + 0.9 + (q % 3) * 1.5, y0 + 2.4 + (q // 3) * 2.0), 0.5,
                                    fc=pal[p], ec="none"))
        ax.text(49.6, y0 + 3.4, "…", fontsize=FS, va="center")
        ax.text(53.5, y0 + 3.4, "50 batches", fontsize=FS_SMALL, color=S.GREY, va="center")
    # the simulated network, right-aligned in the panel so that E6 stays inside it
    x0 = w - 89
    cohort = (coll or {}).get("cohort", {})
    sizes = cohort.get("sizes")
    freq = (cohort.get("frequency") or {}).get("0.05")
    pos = {0: x0 + 12, 1: x0 + 26, 2: x0 + 37, 3: x0 + 51, 4: x0 + 62, 5: x0 + 75, 6: x0 + 86}
    yc = 10.5
    for name, evs in (("control", [0]), ("synergy", [1, 2]), ("redundancy", [3, 4]), ("order", [5, 6])):
        ax.text(np.mean([pos[e] for e in evs]), 3.2, name, fontsize=FS, ha="center", va="top")
    for e, px in pos.items():
        ax.add_patch(Circle((px, yc), 2.8, fc="white", ec=S.INK, lw=0.5, zorder=3))
        ax.text(px, yc, f"E{e}", ha="center", va="center", fontsize=FS, zorder=4)
        ax.text(px, yc + 4.0, str(sizes[e]) if sizes else "–", ha="center", va="top", fontsize=FS_SMALL,
                color=S.GREY)
        ax.text(px, yc + 6.8, f"{freq[f'E{e}']:.0%}" if freq else "–", ha="center", va="top",
                fontsize=FS_SMALL, color=S.GREY)
    ax.plot([pos[1] + 2.8, pos[2] - 2.8], [yc, yc], color=SYNERGY, lw=1.2)
    ax.plot([pos[3] + 2.8, pos[4] - 2.8], [yc, yc], color=REDUNDANCY, lw=1.2, ls=(0, (2, 1)))
    ax.add_patch(FancyArrowPatch((pos[5] + 2.8, yc), (pos[6] - 2.8, yc), arrowstyle="-|>", mutation_scale=5,
                                 lw=0.8, color=S.INK))
    ax.text(x0 + 7, yc + 4.0, "genes", fontsize=FS_SMALL, color=S.GREY, va="top", ha="right")
    ax.text(x0 + 7, yc + 6.8, "patients", fontsize=FS_SMALL, color=S.GREY, va="top", ha="right")


# ------------------------------------------------------------------ C: embeddings at the fitted level
def panel_c(fig, x, y, levels):
    w, gap, dgap = 12.2, 0.8, 2.6
    methods = ("none", "harmony", "scvi")
    pal = palette50()
    col_x = {(d, m): x + di * (3 * w + 2 * gap + dgap) + mi * (w + gap)
             for di, d in enumerate(DESIGNS) for mi, m in enumerate(methods)}
    for di, d in enumerate(DESIGNS):
        cx = x + di * (3 * w + 2 * gap + dgap) + (3 * w + 2 * gap) / 2
        S.text_mm(fig, cx, y - 4.4, d, ha="center", fontsize=FS)
        for m in methods:
            S.text_mm(fig, col_x[(d, m)] + w / 2, y - 1.0, TOOL_LABEL[m], ha="center", fontsize=FS_SMALL)
    emb = (levels.get(EMBED_LEVEL) or {}).get("embeddings")
    for ri, how in enumerate(("batch", "patient")):
        yy = y + ri * (w + gap)
        S.text_mm(fig, x - 1.0, yy + w / 2, how, rotation=90, ha="center", va="center", fontsize=FS_SMALL,
                  color=S.GREY)
        for d in DESIGNS:
            for m in methods:
                ax = S.ax_mm(fig, col_x[(d, m)], yy, w, w)
                ax.set_xticks([])
                ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(True)
                    sp.set_color(S.BG_CELLS)
                    sp.set_linewidth(0.4)
                key = f"{d}/{m}"
                if not emb or key not in emb["xy"]:
                    ax.text(0.5, 0.5, "not yet\nrun", ha="center", va="center", fontsize=FS_SMALL,
                            color=S.GREY, transform=ax.transAxes)
                    continue
                xy = np.asarray(emb["xy"][key])
                cancer = np.asarray(emb["labels"]["cell_type"]) == "cancer"
                if how == "batch":
                    lab = np.asarray([int(str(b).replace("batch", "")) for b in emb["labels"]["batch"][d]])
                    fg = ~cancer
                else:
                    lab = np.asarray([int(p) for p in emb["labels"]["patient"]])
                    fg = cancer
                ax.scatter(xy[~fg, 0], xy[~fg, 1], s=0.12, c=S.BG_CELLS, rasterized=True, linewidths=0)
                ax.scatter(xy[fg, 0], xy[fg, 1], s=0.12, c=[pal[v % len(pal)] for v in lab[fg]],
                           rasterized=True, linewidths=0)


# ------------------------------------------------------------------ D: integration, every level
D_COLUMNS = (("ilisi_batch_noncancer", "iLISI", "microenvironment"),
             ("patient_purity_cancer", "same patient", "cancer cells"),
             ("clone_purity_cancer", "same clone", "cancer cells"))
D_OFFSET = {"none": -0.26, "harmony": 0.0, "scvi": 0.26}


def panel_d(fig, x, y, levels):
    """A row per level (fitted, shared, then one technical source at a time), a line within it per
    tool, and a column per measure on its own scale: filled for the confounded design, hollow for the
    balanced one, and the value of the embedding without batch effect beside them (grey diamond)."""
    rows = ("fitted", "shared") + BL.ARMS
    lab_w, w, gap, row_h = 19, 16, 2.6, 3.3
    h = row_h * len(rows)
    axes = [S.ax_mm(fig, x + lab_w + k * (w + gap), y, w, h) for k in range(len(D_COLUMNS))]
    for k, (ax, (key, xlabel, title)) in enumerate(zip(axes, D_COLUMNS)):
        ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([LEVEL_LABEL[r] for r in rows] if k == 0 else [], fontsize=FS_SMALL)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", labelsize=FS_SMALL)
        ax.axhline(1.5, color=S.BG_CELLS, lw=0.5)             # fitted and shared above the sources
        ax.set_title(title, fontsize=FS_SMALL, pad=2)
        ax.set_xlabel(xlabel if k == 0 else f"neighbors from the\n{xlabel}", fontsize=FS_SMALL, labelpad=1)
        if key.startswith("ilisi"):                           # scaled to 50 batches: every value is
            ax.set_xlim(0.045, 0.09)                          # 0.05-0.09, so the shared 0-1 scale hides
            ax.set_xticks([0.05, 0.07, 0.09])                 # what integration does to it
            ax.set_xticklabels(["0.05", "0.07", "0.09"])
        else:
            ax.set_xlim(-0.03, 1.03)
            ax.set_xticks([0, 0.5, 1])
            ax.set_xticklabels(["0", "0.5", "1"])
        S.despine(ax, left=False)
        for ri, lv in enumerate(rows):
            rec = levels.get(lv)
            if not rec:
                continue
            for d in DESIGNS:
                ref = ((rec.get("truth") or {}).get(d) or {}).get(key)
                if ref is not None:
                    ax.scatter([ref], [ri], s=9, marker="D", facecolors=REFERENCE if d == "confounded" else "white",
                               edgecolors=REFERENCE, linewidths=0.55, zorder=2)
            for run, r in rec["runs"].items():
                d, m = run.split("/")
                v = (r.get("metrics") or {}).get(key)
                if v is None or m not in D_OFFSET:
                    continue
                ax.scatter([v], [ri + D_OFFSET[m]], s=7, marker="o",
                           facecolors=TOOL_COLOUR[m] if d == "confounded" else "white",
                           edgecolors=TOOL_COLOUR[m], linewidths=0.55, zorder=3)
    if not any(lv in levels for lv in rows):
        S.text_mm(fig, x + lab_w + 1.5 * w + gap, y + h / 2, "integration\nnot yet run", ha="center",
                  va="center", fontsize=FS, color=S.GREY)
    handles = ([Line2D([], [], marker="o", ls="", mfc=TOOL_COLOUR[m], mec=TOOL_COLOUR[m], markersize=3,
                       label=TOOL_LABEL[m]) for m in ("none", "harmony", "scvi")]
               + [Line2D([], [], marker="D", ls="", mfc=REFERENCE, mec=REFERENCE, markersize=2.6,
                         label="no batch effect"),
                  Line2D([], [], marker="o", ls="", mfc=S.INK, mec=S.INK, markersize=3, label="confounded"),
                  Line2D([], [], marker="o", ls="", mfc="white", mec=S.INK, markersize=3, label="balanced")])
    axes[0].legend(handles=handles, loc="upper left", ncol=3, fontsize=FS_SMALL, handletextpad=0.2,
                   columnspacing=0.9, bbox_to_anchor=(-lab_w / w, -0.36), borderaxespad=0)


# ------------------------------------------------------------------ E, F: tools against E0
def _forest(ax, rows, band, tool, show_labels, ranks=False):
    """One tool's column: per row the estimates (grey dots for the reference pairs, a coloured marker
    per relation, filled when recovered) over the band of the reference magnitudes."""
    vals = [v for r in rows for v in r["values"] if v is not None and np.isfinite(v)]
    lim = max([abs(v) for v in vals] + [band or 0, 0.1]) * 1.18
    if band:
        ax.axvspan(-band, band, color="#EBEBEB", zorder=0, lw=0)
    ax.axvline(0, color=S.BG_CELLS, lw=0.5, zorder=1)
    for k, r in enumerate(rows):
        yk = r.get("y", k)
        for v in r["values"]:
            if v is None or not np.isfinite(v):
                continue
            if r.get("reference"):
                ax.scatter([v], [yk], s=6, color=S.GREY, zorder=3, linewidths=0)
            else:
                ax.scatter([v], [yk], s=13, marker=r.get("marker", "o"),
                           facecolors=r["colour"] if r.get("filled") else "white",
                           edgecolors=r["colour"], linewidths=0.7, zorder=4)
        if r.get("note"):
            # on the side away from the row's own marker, so a large estimate does not cover it
            left = any(v is not None and np.isfinite(v) and v > 0 for v in r["values"])
            ax.text(0.03 if left else 0.97, yk, r["note"], transform=ax.get_yaxis_transform(),
                    fontsize=FS_SMALL - 0.6, color=S.GREY, ha="left" if left else "right", va="center")
        if ranks and r.get("rank") is not None:
            ax.text(1.05, yk, f"{r['rank']}", transform=ax.get_yaxis_transform(), fontsize=FS_SMALL,
                    color=S.GREY, va="center", ha="left")
    n = sum(1 for r in rows if not r.get("overlay"))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_yticks(range(n))
    ax.set_yticklabels([r["label"] for r in rows if not r.get("overlay")] if show_labels else [],
                       fontsize=FS_SMALL)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=FS_SMALL)
    ax.set_title(TOOL_LABEL[tool], fontsize=FS, pad=1.5)
    if ranks:
        ax.text(1.05, -0.62, "rank", transform=ax.get_yaxis_transform(), fontsize=FS_SMALL - 0.6,
                color=S.GREY, va="bottom", ha="left")
    S.despine(ax, left=False)


def _edges(ax, rows, tool, show_labels):
    """CBN's column: per row whether the edge is returned (right) or not (left)."""
    for k, r in enumerate(rows):
        for j, present in enumerate(r["edges"]):
            dy = (j - (len(r["edges"]) - 1) / 2) * 0.13 if len(r["edges"]) > 1 else 0
            if r.get("reference"):
                ax.scatter([1 if present else 0], [k + dy], s=6, color=S.GREY, zorder=3, linewidths=0)
            else:
                ax.scatter([1 if present else 0], [k], s=13, marker=r.get("marker", "o"),
                           facecolors=r["colour"] if (present and r.get("fill_if_present")) else "white",
                           edgecolors=r["colour"], linewidths=0.7, zorder=4)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["label"] for r in rows] if show_labels else [], fontsize=FS_SMALL)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["no", "edge"], fontsize=FS_SMALL)
    ax.tick_params(axis="y", length=0)
    ax.set_title(TOOL_LABEL[tool], fontsize=FS, pad=1.5)
    S.despine(ax, left=False)


def _signed(p):
    return p["sign"] * p["strength"]


E_LABELS = ("E0 pairs", "E1–E2 synergy", "E3–E4 redundancy", "effect of E5 on E6", "effect of E6 on E5")


def _e_rows(sc, fit):
    if sc is None or sc.get("kind") != "theta":
        return None, None
    pairs = fit["pairs"]
    rel = sc["relations"]
    rows = [dict(label=E_LABELS[0], reference=True, values=[_signed(pairs[f"{i},{j}"]) for i, j in A.E0_PAIRS]),
            dict(label=E_LABELS[1], values=[_signed(rel["synergy"])], colour=SYNERGY,
                 filled=rel["synergy"]["recovered"], rank=rel["synergy"]["rank"]),
            dict(label=E_LABELS[2], values=[_signed(rel["redundancy"])], colour=REDUNDANCY,
                 filled=rel["redundancy"]["recovered"], rank=rel["redundancy"]["rank"])]
    o = rel["order"]
    if fit.get("symmetric"):
        # an undirected score: one value for the pair, between the two directed rows
        rows += [dict(label=E_LABELS[3], values=[None]), dict(label=E_LABELS[4], values=[None]),
                 dict(label="", values=[_signed(o)], colour=ORDER, filled=False, rank=o["rank"], y=3.5,
                      overlay=True, note="undirected")]
    else:
        rows += [dict(label=E_LABELS[3], values=[o["ji"]], colour=ORDER, filled=o["recovered"], marker=">",
                      rank=o["rank"]),
                 dict(label=E_LABELS[4], values=[o["ij"]], colour=ORDER, filled=False, marker="<")]
    return rows, sc["e0_max_strength"]


def _e_cbn_rows(sc):
    edges = sc["edges"]
    return [dict(label=E_LABELS[0], reference=True,
                 edges=[A._cbn_has(edges, "E0", f"E{j}") or A._cbn_has(edges, f"E{j}", "E0") for j in range(1, 7)]),
            dict(label=E_LABELS[1], edges=[sc["synergy_edge"]], colour=SYNERGY),
            dict(label=E_LABELS[2], edges=[sc["redundancy_edge"]], colour=REDUNDANCY),
            dict(label=E_LABELS[3], edges=[sc["order_edge"]], colour=ORDER, marker=">", fill_if_present=True),
            dict(label=E_LABELS[4], edges=[sc["reverse_edge"]], colour=ORDER, marker="<")]


def panel_e(fig, x, y, coll):
    tools = ("cooc", "mhn", "treemhn")
    lab_w, w, gap, h, wc = 22, 12.2, 4.2, 19, 8           # 79 mm: beside panel B
    res = None
    if coll:
        res = next((r for r in coll["results"] if r["panel"] == "d" and r["threshold"] == A.DETECTION), None)
    if res is None:
        placeholder(S.ax_mm(fig, x + lab_w, y, 3 * (w + gap) + wc, h), "network scores")
        return
    for ti, tool in enumerate(tools):
        ax = S.ax_mm(fig, x + lab_w + ti * (w + gap), y, w, h)
        fit = res["fits"].get(tool) or {}
        rows, band = _e_rows(res["scores"].get(tool), fit)
        if rows is None:
            placeholder(ax, TOOL_LABEL[tool])
            continue
        _forest(ax, rows, band, tool, show_labels=(ti == 0), ranks=True)
        ax.set_xlabel("log odds" if tool == "cooc" else "Θ", fontsize=FS_SMALL, labelpad=1)
    ax = S.ax_mm(fig, x + lab_w + 3 * (w + gap), y, wc, h)
    cbn = res["scores"].get("cbn")
    if cbn and cbn.get("kind") == "cbn":
        _edges(ax, _e_cbn_rows(cbn), "cbn", show_labels=False)
    else:
        placeholder(ax, "CBN")


F_TRAITS = ("breach", "stromal", "met")
F_NAME = {"breach": "breach", "stromal": "stromal survival", "met": "metastatic survival"}
F_LABELS = ("E0 and each trait", "breach–stromal survival", "breach–metastatic survival",
            "stromal–metastatic survival", "breach on metastatic survival", "metastatic survival on breach")


def _f_rows(sc, fit):
    if sc is None or sc.get("kind") != "theta":
        return None, None
    rows = [dict(label=F_LABELS[0], reference=True, values=[_signed(p) for p in sc["e0_trait"].values()])]
    for lab, key in zip(F_LABELS[1:4], ("breach-stromal", "breach-met", "stromal-met")):
        p = sc["trait_pairs"][key]
        rows.append(dict(label=lab, values=[_signed(p)], colour=S.INK, filled=p["above_e0"]))
    o = sc["order"]["breach<met"]
    if fit.get("symmetric"):
        rows += [dict(label=F_LABELS[4], values=[None]), dict(label=F_LABELS[5], values=[None]),
                 dict(label="", values=[None], y=4.5, overlay=True, note="undirected")]
    else:
        rows += [dict(label=F_LABELS[4], values=[o["theta_t_breach"]], colour=ORDER,
                      filled=bool(o["order_ok"] and o["above_e0"]), marker=">"),
                 dict(label=F_LABELS[5], values=[o["theta_breach_t"]], colour=ORDER, filled=False, marker="<")]
    return rows, sc["e0_max_strength"]


def _f_cbn_rows(sc):
    edges = sc["edges"]

    def either(a, b):
        return A._cbn_has(edges, a, b) or A._cbn_has(edges, b, a)
    return [dict(label=F_LABELS[0], reference=True, edges=[either("E0", t) for t in F_TRAITS]),
            dict(label=F_LABELS[1], edges=[either("breach", "stromal")], colour=S.INK),
            dict(label=F_LABELS[2], edges=[either("breach", "met")], colour=S.INK),
            dict(label=F_LABELS[3], edges=[either("stromal", "met")], colour=S.INK),
            dict(label=F_LABELS[4], edges=[sc["breach_to_met"]], colour=ORDER, marker=">", fill_if_present=True),
            dict(label=F_LABELS[5], edges=[sc["met_to_breach"]], colour=ORDER, marker="<")]


def panel_f(fig, x, y, coll):
    tools = ("cooc", "mhn", "treemhn")
    lab_w, w, gap, h, wc = 29, 11.8, 2.4, 21, 8
    res = None
    if coll:
        res = next((r for r in coll["results"] if r["panel"] == "e" and r["threshold"] == A.DETECTION), None)
    # the truth: which trait the genealogy of each tumour acquired first
    # the truth's bars on the left, the tools on the right: one row along the bottom of the figure
    ax = S.ax_mm(fig, x + lab_w, y + 5, 24, 6.5)
    orders = ((coll or {}).get("truth") or {}).get("orders")
    if not orders:
        placeholder(ax, "genealogies")
    else:
        show = [("breach<met_survival", "breach before metastatic survival"),
                ("breach<stromal_survival", "breach before stromal survival")]
        for k, (key, lab) in enumerate(show):
            o = orders.get(key) or {}
            v = o.get("share_first")
            nn = (o.get("tumours_first") or 0) + (o.get("tumours_second") or 0)
            if v is not None:
                ax.barh([k], [v], color=S.BG_CELLS, height=0.62)
                ax.text(1.03, k, f"{v:.0%} of {nn}", va="center", fontsize=FS_SMALL,
                        transform=ax.get_yaxis_transform())
        ax.axvline(0.5, color=S.GREY, lw=0.5, ls=":")
        ax.set_xlim(0, 1)
        ax.set_ylim(1.5, -0.5)
        ax.set_yticks([0, 1])
        ax.set_yticklabels([lab for _, lab in show], fontsize=FS_SMALL)
        ax.tick_params(axis="y", length=0)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(["0", "0.5", "1"], fontsize=FS_SMALL)
        ax.set_xlabel("share of tumors", fontsize=FS_SMALL, labelpad=1)
    yt, xf = y, x + 68
    if res is None:
        placeholder(S.ax_mm(fig, xf + lab_w, yt, 3 * (w + gap) + wc, h), "trait scores")
        return
    for ti, tool in enumerate(tools):
        ax = S.ax_mm(fig, xf + lab_w + ti * (w + gap), yt, w, h)
        fit = res["fits"].get(tool) or {}
        rows, band = _f_rows(res["scores"].get(tool), fit)
        if rows is None:
            placeholder(ax, TOOL_LABEL[tool])
            continue
        _forest(ax, rows, band, tool, show_labels=(ti == 0))
        ax.set_xlabel("log odds" if tool == "cooc" else "Θ", fontsize=FS_SMALL, labelpad=1)
    ax = S.ax_mm(fig, xf + lab_w + 3 * (w + gap), yt, wc, h)
    cbn = res["scores"].get("cbn")
    if cbn and cbn.get("kind") == "cbn":
        _edges(ax, _f_cbn_rows(cbn), "cbn", show_labels=False)
    else:
        placeholder(ax, "CBN")


def main():
    """Fig 5 at the text width (A cohort, B integration, C simulated network, D architecture), and the two
    panels the half-page layout moved out: S13 Fig (the batch effect against real data) and S14 Fig
    (the embeddings at the fitted level)."""
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=None, help="summary directory (as fig5_atlas.py)")
    ap.add_argument("--name", default="fig5_cohort")
    args = ap.parse_args()
    out = args.out or os.environ.get("ISCC_FIG5_OUT") or os.path.join(A.REG.OUT, "fig5-atlas")
    data = load(out)
    S.setup()
    fig = S.figure(165, 136)
    heading(fig, "A", "Cohort and simulated network", 2, 2)
    panel_a(fig, 4, 8, data["collect"], w=158)
    heading(fig, "B", "Integration", 2, 43)
    panel_d(fig, 4, 52, data["levels"])
    heading(fig, "C", "Simulated network", 84, 43)
    panel_e(fig, 84, 54, data["collect"])
    heading(fig, "D", "Architecture", 2, 98)
    panel_f(fig, 2, 107, data["collect"])
    path = S.save(fig, args.name)
    fig14 = S.figure(92, 36)
    panel_c(fig14, 6, 8, data["levels"])
    path14 = S.save(fig14, "supp_cohort_embeddings")
    print(f"[fig5] {path}, {path14} (levels: {sorted(data['levels'])}, collect: "
          f"{data['collect'] is not None})")


if __name__ == "__main__":
    main()
