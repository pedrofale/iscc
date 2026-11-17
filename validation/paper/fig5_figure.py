"""Fig 5: one atlas-sized cohort of 50 patients, and what tools read from it.

Draws ``figures/paper1/fig5_cohort.pdf`` (through ``validation/_paths.py``; ``ISCC_PAPER_DIR`` picks
the paper checkout) from the summaries :mod:`fig5_atlas` writes under ``$ISCC_FIG5_OUT`` (or
``--out``), and from nothing else: ``collect_summary.json`` (panels a, d, e) and
``integrate/level_<L>.json`` (panels b, c). A panel whose summaries are missing is drawn as a
placeholder that says so. The design and its expectations were registered before the run. (The
earlier design's figure, from ``fig5_cohort.py``'s summaries, is in this file's history at 38937d4.)

a. The cohort and the planted network: 50 patients over one shared landscape, each grown to its own
   size; their dissociated cells pooled two ways (one batch per patient; patients spread over 50
   batches); the network of event modules E0-E6 with each module's size and the share of patients
   carrying it at 5% detection.
b. Embeddings of the pooled cells at the 10x and strong batch levels, without integration and after
   Harmony and scVI, in both designs: microenvironment cells coloured by batch (mixing) and cancer
   cells by patient (patient-specific tumour biology), the other population in grey.
c. What integration removes: microenvironment batch mixing (iLISI) against the cancer cells' patient
   neighbourhoods and, within each patient, clone neighbourhoods, each as a share of its value in the
   batch-free embedding.
d. The planted network from the DNA: each tool's estimate for the planted relations against the six
   pairs of the control E0 (the grey band spans their magnitudes); a filled marker is a relation
   recovered by the registered rule (planted sign, magnitude above every E0 pair; for the order,
   Theta[E6,E5] above Theta[E5,E6] and above every E0 pair); the rank among the 21 pairs at the right.
e. Architecture against the mix of stages: the order of breach and the survival traits in the true
   genealogies, and each tool's trait-trait dependencies against the E0-trait ones.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch  # noqa: E402

import fig5_atlas as A  # noqa: E402
import style as S  # noqa: E402

TOOL_COLOUR = {"none": S.GREY, "harmony": S.TOOL["Harmony"], "scvi": S.TOOL["scVI"],
               "cooc": "#4D4D4D", "mhn": S.TOOL["MHN"], "treemhn": S.TOOL["TreeMHN"]}
TOOL_LABEL = {"none": "no integration", "harmony": "Harmony", "scvi": "scVI", "cooc": "co-occurrence",
              "mhn": "MHN", "treemhn": "TreeMHN", "cbn": "CBN"}
DESIGN_LABEL = {"confounded": "one batch per patient", "balanced": "patients across batches"}
LEVEL_LABEL = {"10x": "10x preset", "strong": "strong", "intermediate": "intermediate"}
LEVEL_MARKER = {"10x": "o", "intermediate": "s", "strong": "^"}
SYNERGY, REDUNDANCY, TRAIT = "#2F6D3A", "#C44E52", "#4C72B0"
FS, FS_SMALL = 5.5, 4.8


def palette50():
    """50 distinguishable colours for 50 batches or patients: they show mixing (confetti) against
    separation (patches), not identity."""
    import matplotlib as mpl
    cols = []
    for name in ("tab20", "tab20b", "tab20c"):
        cols += [mpl.colors.to_hex(c) for c in mpl.colormaps[name].colors]
    return cols[:50]


def load(out):
    def read(p):
        if not os.path.exists(p):
            return None
        with open(p) as f:
            return json.load(f)
    d = os.path.join(out, "integrate")
    levels = {}
    if os.path.isdir(d):
        for fn in sorted(os.listdir(d)):
            if fn.startswith("level_") and fn.endswith(".json"):
                r = read(os.path.join(d, fn))
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


# ------------------------------------------------------------------ a: the cohort and the network
def panel_a(fig, x, y, coll):
    w, h = 176, 38
    ax = S.ax_mm(fig, x, y, w, h)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    pal = palette50()
    # the cohort: its patients ordered by their target size, the dot area following the size
    if coll is not None:
        targets = np.asarray(coll["cohort"]["targets"])
    else:
        targets = np.asarray(A.cohort_targets(len(A.COHORT_SEEDS)))
    n = len(targets)
    lo, hi = A.TARGET_RANGE
    ax.text(0, 1, f"{n} patients over one shared landscape, each grown to its own size",
            fontsize=FS, va="top")
    for k, i in enumerate(np.argsort(targets)):
        r = 0.35 + 0.65 * np.sqrt(np.clip((targets[i] - lo) / (hi - lo), 0, 1))
        ax.add_patch(Circle((1.2 + k * (66 / max(n - 1, 1)), 7.5), r, fc=pal[i % len(pal)], ec="none"))
    ax.text(0, 10.2, f"{lo:,}", fontsize=FS_SMALL, color=S.GREY, va="top")
    ax.text(68, 10.2, f"{hi:,} cancer cells (in situ to metastatic)", fontsize=FS_SMALL, color=S.GREY,
            va="top", ha="right")
    # the two pooling designs of the same cells
    for k, mix in enumerate((False, True)):
        y0 = 17.5 + 10 * k
        ax.text(0, y0, "patients across 50 batches" if mix else "one batch per patient (50 batches)",
                fontsize=FS, va="bottom")
        for b in range(6):
            bx = 0.5 + 6.0 * b
            ax.add_patch(FancyBboxPatch((bx, y0 + 0.9), 4.8, 5.0, boxstyle="round,pad=0.15", fc="white",
                                        ec=S.INK, lw=0.4))
            for q in range(6):
                p = (6 * b + 7 * q) % 50 if mix else b
                ax.add_patch(Circle((bx + 0.9 + (q % 3) * 1.5, y0 + 2.4 + (q // 3) * 2.0), 0.5,
                                    fc=pal[p], ec="none"))
        ax.text(36.6, y0 + 3.4, "…", fontsize=FS, va="center")
    ax.text(40.5, 18.4, "1,000 primary cells per\npatient by scRNA, with\nthe 10x preset or a\nstrong "
                        "batch effect;\nthe same cells without\none: the reference",
            fontsize=FS_SMALL, color=S.GREY, va="top", linespacing=1.15)
    # the planted network
    x0 = 77
    cohort = (coll or {}).get("cohort", {})
    sizes = cohort.get("sizes")
    freq = (cohort.get("frequency") or {}).get("0.05")
    net = ((coll or {}).get("truth") or {}).get("network") or A.NETWORK
    eff = net["effects"]
    inter = {(i, j): v for i, j, v in net["interactions"]}
    ax.text(x0, 1, "planted network: modules of driver genes (events)", fontsize=FS, va="top")
    ax.text(x0, 4.6, f"own effect {eff[0]:+g} on division each; E6's {eff[6]:+g} counts only with E5 "
                     "(fitness gating)", fontsize=FS_SMALL, color=S.GREY, va="top")
    pos = {0: x0 + 9, 1: x0 + 25, 2: x0 + 38, 3: x0 + 54, 4: x0 + 67, 5: x0 + 82, 6: x0 + 95}
    yc = 18.5
    for name, evs in (("control", [0]), ("synergy", [1, 2]), ("redundancy", [3, 4]),
                      ("order by selection", [5, 6])):
        ax.text(np.mean([pos[e] for e in evs]), 10.5, name, fontsize=FS, ha="center", va="top")
    for e, px in pos.items():
        ax.add_patch(Circle((px, yc), 2.8, fc="white", ec=S.INK, lw=0.5, zorder=3))
        ax.text(px, yc, f"E{e}", ha="center", va="center", fontsize=FS, zorder=4)
        lab = (f"{sizes[e]} gene" + ("s" if sizes[e] != 1 else "")) if sizes else "size: pilot"
        ax.text(px, yc + 4.0, lab, ha="center", va="top", fontsize=FS_SMALL, color=S.GREY)
        if freq:
            ax.text(px, yc + 6.6, f"{freq[f'E{e}']:.0%}", ha="center", va="top", fontsize=FS_SMALL, color=S.GREY)
    ax.plot([pos[1] + 2.8, pos[2] - 2.8], [yc, yc], color=SYNERGY, lw=1.2)
    ax.text((pos[1] + pos[2]) / 2, yc - 0.9, f"{inter.get((1, 2), 0):+g}", ha="center", va="bottom",
            fontsize=FS_SMALL, color=SYNERGY)
    ax.plot([pos[3] + 2.8, pos[4] - 2.8], [yc, yc], color=REDUNDANCY, lw=1.2, ls=(0, (2, 1)))
    ax.text((pos[3] + pos[4]) / 2, yc - 0.9, f"{inter.get((3, 4), 0):+g}", ha="center", va="bottom",
            fontsize=FS_SMALL, color=REDUNDANCY)
    ax.add_patch(FancyArrowPatch((pos[5] + 2.8, yc), (pos[6] - 2.8, yc), arrowstyle="-|>", mutation_scale=5,
                                 lw=0.8, color=S.INK))
    ax.text(x0 + 3.5, yc + 4.0, "module", fontsize=FS_SMALL, color=S.GREY, va="top", ha="right")
    if freq:
        ax.text(x0 + 3.5, yc + 6.6, "patients", fontsize=FS_SMALL, color=S.GREY, va="top", ha="right")
    ax.text(x0, 33.5, "panel e adds three traits with no interaction planted: breach, stromal survival, "
                      "metastatic survival", fontsize=FS_SMALL, color=S.GREY, va="top")


# ------------------------------------------------------------------ b: embeddings
def panel_b(fig, x, y, levels):
    w, gap, dgap, lgap = 11.5, 0.8, 2.6, 2.4
    designs, methods = ("confounded", "balanced"), ("none", "harmony", "scvi")
    rows = [(lv, how) for lv in A.EMBED_LEVELS for how in ("batch", "patient")]
    pal = palette50()
    col_x = {(d, m): x + di * (3 * w + 2 * gap + dgap) + mi * (w + gap)
             for di, d in enumerate(designs) for mi, m in enumerate(methods)}
    for di, d in enumerate(designs):
        cx = x + di * (3 * w + 2 * gap + dgap) + (3 * w + 2 * gap) / 2
        S.text_mm(fig, cx, y - 4.4, DESIGN_LABEL[d], ha="center", fontsize=FS)
        for m in methods:
            S.text_mm(fig, col_x[(d, m)] + w / 2, y - 1.0, "none" if m == "none" else TOOL_LABEL[m],
                      ha="center", fontsize=FS_SMALL)
    for ri, (lv, how) in enumerate(rows):
        yy = y + ri * (w + gap) + (ri // 2) * lgap
        if how == "batch":
            S.text_mm(fig, x - 4.6, yy + w + gap / 2, LEVEL_LABEL[lv], rotation=90, ha="center", va="center",
                      fontsize=FS)
        S.text_mm(fig, x - 1.0, yy + w / 2, how, rotation=90, ha="center", va="center", fontsize=FS_SMALL,
                  color=S.GREY)
        emb = (levels.get(lv) or {}).get("embeddings")
        for d in designs:
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
    yb = y + 4 * (w + gap) + lgap + 0.6
    note = "batch: microenvironment cells by batch; patient: cancer cells by patient; others grey"
    rec = next((levels[lv]["embeddings"] for lv in A.EMBED_LEVELS
                if (levels.get(lv) or {}).get("embeddings")), None)
    if rec:
        note += f"\n{rec['n_shown']:,} of {rec['n_cells_total']:,} cells shown, {rec.get('projection', '')}"
    S.text_mm(fig, x, yb, note, fontsize=FS_SMALL, color=S.GREY, va="top", linespacing=1.15)


# ------------------------------------------------------------------ c: what integration removes
def panel_c(fig, x, y, levels):
    w, h, gap = 33, 20, 3.0
    axes = [S.ax_mm(fig, x, y, w, h), S.ax_mm(fig, x, y + h + gap, w, h)]
    keys = [("patient_purity_cancer_retention", "cancer cells'\npatient kept"),
            ("clone_purity_cancer_retention", "clones within\npatients kept")]
    if not levels:
        for ax in axes:
            placeholder(ax, "integration")
        return
    for ax, (key, lab) in zip(axes, keys):
        for lv, rec in levels.items():
            for run, r in rec["runs"].items():
                d, m = run.split("/")
                ret = r.get("retention") or {}
                xv, yv = ret.get("ilisi_batch_noncancer_retention"), ret.get(key)
                if xv is None or yv is None:
                    continue
                ax.scatter([xv], [yv], s=10, marker=LEVEL_MARKER.get(lv, "o"),
                           facecolors=TOOL_COLOUR[m] if d == "confounded" else "white",
                           edgecolors=TOOL_COLOUR[m], linewidths=0.6, zorder=3)
        ax.axhline(1, color=S.BG_CELLS, lw=0.5, zorder=0)
        ax.axvline(0.8, color=S.BG_CELLS, lw=0.5, ls=":", zorder=0)
        ax.set_ylabel(lab, fontsize=FS)
        ax.set_xlim(-0.05, 1.12)
    axes[0].axhline(0.6, color=S.BG_CELLS, lw=0.5, ls=":", zorder=0)
    axes[0].set_xticklabels([])
    axes[1].set_xlabel("microenvironment batch mixing (iLISI)", fontsize=FS)
    S.text_mm(fig, x, y + 2 * h + gap + 6.5, "each as a share of its value in the batch-free embedding",
              fontsize=FS_SMALL, color=S.GREY, va="top")
    handles = ([Line2D([], [], marker="o", ls="", mfc=TOOL_COLOUR[m], mec=TOOL_COLOUR[m], markersize=3,
                       label=TOOL_LABEL[m]) for m in ("none", "harmony", "scvi")]
               + [Line2D([], [], marker="o", ls="", mfc=S.INK, mec=S.INK, markersize=3,
                         label=DESIGN_LABEL["confounded"]),
                  Line2D([], [], marker="o", ls="", mfc="white", mec=S.INK, markersize=3,
                         label=DESIGN_LABEL["balanced"])]
               + [Line2D([], [], marker=LEVEL_MARKER[lv], ls="", mfc="white", mec=S.GREY, markersize=3,
                         label=LEVEL_LABEL[lv] + ("\n(exploratory)" if lv == "intermediate" else ""))
                  for lv in ("10x", "strong", "intermediate") if lv in levels])
    axes[0].legend(handles=handles, loc="upper left", bbox_to_anchor=(1.04, 1.0), fontsize=FS_SMALL,
                   handletextpad=0.2, borderaxespad=0, labelspacing=0.45)


# ------------------------------------------------------------------ d, e: tools against E0
def _forest(ax, rows, band, tool, show_labels, ranks=False):
    """One tool's column: per row the estimates (grey reference dots, coloured relation markers,
    filled when recovered) over the band of the reference magnitudes."""
    vals = [v for r in rows for v in r["values"] if v is not None and np.isfinite(v)]
    lim = max([abs(v) for v in vals] + [band or 0, 0.1]) * 1.18
    if band:
        ax.axvspan(-band, band, color="#EBEBEB", zorder=0, lw=0)
    ax.axvline(0, color=S.BG_CELLS, lw=0.5, zorder=1)
    for k, r in enumerate(rows):
        shown = [v for v in r["values"] if v is not None and np.isfinite(v)]
        for v in shown:
            if r.get("reference"):
                ax.scatter([v], [k], s=6, color=S.GREY, zorder=3, linewidths=0)
            else:
                ax.scatter([v], [k], s=13, marker=r.get("marker", "o"),
                           facecolors=r["colour"] if r.get("filled") else "white",
                           edgecolors=r["colour"], linewidths=0.7, zorder=4)
        if not shown and not r.get("blank"):
            ax.text(0, k, "n/a", fontsize=FS_SMALL, color=S.GREY, ha="center", va="center")
        if ranks and r.get("rank") is not None:
            ax.text(1.04, k, f"{r['rank']}", transform=ax.get_yaxis_transform(), fontsize=FS_SMALL,
                    color=S.GREY, va="center", ha="left")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["label"] for r in rows] if show_labels else [], fontsize=FS_SMALL)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=FS_SMALL)
    ax.set_title(TOOL_LABEL[tool], fontsize=FS, pad=1.5)
    S.despine(ax, left=False)


def _signed(p):
    return p["sign"] * p["strength"]


def _d_rows(sc, fit):
    if sc is None or sc.get("kind") != "theta":
        return None, None
    pairs = fit["pairs"]
    rel = sc["relations"]
    rows = [dict(label="E0 pairs (6)", reference=True, values=[_signed(pairs[f"{i},{j}"]) for i, j in A.E0_PAIRS]),
            dict(label="E1–E2 synergy (+)", values=[_signed(rel["synergy"])], colour=SYNERGY,
                 filled=rel["synergy"]["recovered"], rank=rel["synergy"]["rank"]),
            dict(label="E3–E4 redundancy (−)", values=[_signed(rel["redundancy"])], colour=REDUNDANCY,
                 filled=rel["redundancy"]["recovered"], rank=rel["redundancy"]["rank"])]
    o = rel["order"]
    if fit.get("symmetric"):
        rows += [dict(label="E5 on E6", values=[_signed(o)], colour=S.INK, filled=False, rank=o["rank"]),
                 dict(label="E6 on E5", values=[None], blank=True)]
    else:
        rows += [dict(label="E5 on E6, Θ[E6,E5]", values=[o["ji"]], colour=S.INK,
                      filled=o["recovered"], marker=">", rank=o["rank"]),
                 dict(label="E6 on E5, Θ[E5,E6]", values=[o["ij"]], colour=S.INK, filled=False, marker="<")]
    return rows, sc["e0_max_strength"]


def panel_d(fig, x, y, coll):
    tools = ("cooc", "mhn", "treemhn")
    lab_w, w, gap, h = 24, 17, 5.0, 19
    res = None
    if coll:
        res = next((r for r in coll["results"] if r["panel"] == "d" and r["threshold"] == A.DETECTION), None)
    if res is None:
        placeholder(S.ax_mm(fig, x + lab_w, y, 3 * w + 2 * gap, h), "panel d (collect)")
        return
    for ti, tool in enumerate(tools):
        ax = S.ax_mm(fig, x + lab_w + ti * (w + gap), y, w, h)
        fit = res["fits"].get(tool) or {}
        rows, band = _d_rows(res["scores"].get(tool), fit)
        if rows is None:
            placeholder(ax, TOOL_LABEL[tool])
            continue
        _forest(ax, rows, band, tool, show_labels=(ti == 0), ranks=True)
        ax.set_xlabel("log odds" if tool == "cooc" else "Θ", fontsize=FS_SMALL, labelpad=1)
    S.text_mm(fig, x + lab_w + 3 * w + 2 * gap + 0.5, y - 0.3, "rank\nof 21", fontsize=FS_SMALL,
              color=S.GREY, va="bottom", ha="left", linespacing=1.0)
    cbn = res["scores"].get("cbn")
    if cbn and cbn.get("kind") == "cbn":
        def yn(b):
            return "yes" if b else "no"
        txt = (f"CBN edges: E5→E6 {yn(cbn['order_edge'])}, E6→E5 {yn(cbn['reverse_edge'])}, "
               f"E1–E2 {yn(cbn['synergy_edge'])}, E3–E4 {yn(cbn['redundancy_edge'])}; "
               f"{len(cbn['edges'])} edges in all, {len(cbn['e0_edges'])} with E0")
    else:
        txt = "CBN: " + str(list((res["fits"].get("cbn") or {"not run": 1}))[0])
    S.text_mm(fig, x, y + h + 6.5, txt, fontsize=FS_SMALL, color=S.GREY, va="top")
    S.text_mm(fig, x, y + h + 9.3, f"{res['n']} patients, events at 5%; grey band: the six E0 pairs' "
                                   "magnitudes; filled: recovered", fontsize=FS_SMALL, color=S.GREY, va="top")


def _e_rows(sc, fit):
    if sc is None or sc.get("kind") != "theta":
        return None, None
    rows = [dict(label="E0–trait (3)", reference=True, values=[_signed(p) for p in sc["e0_trait"].values()])]
    lab = {"breach-stromal": "breach–stromal", "breach-met": "breach–metastatic",
           "stromal-met": "stromal–metastatic"}
    for k, p in sc["trait_pairs"].items():
        rows.append(dict(label=lab.get(k, k), values=[_signed(p)], colour=TRAIT, filled=p["above_e0"]))
    o = sc["order"]["breach<met"]
    if fit.get("symmetric"):
        rows += [dict(label="breach on met", values=[None], blank=True),
                 dict(label="met on breach", values=[None], blank=True)]
    else:
        rows += [dict(label="breach on met, Θ[M,B]", values=[o["theta_t_breach"]], colour=S.INK,
                      filled=bool(o["order_ok"] and o["above_e0"]), marker=">"),
                 dict(label="met on breach, Θ[B,M]", values=[o["theta_breach_t"]], colour=S.INK,
                      filled=False, marker="<")]
    return rows, sc["e0_max_strength"]


def panel_e(fig, x, y, coll):
    tools = ("cooc", "mhn", "treemhn")
    lab_w, w, gap, h = 27, 15, 3.0, 21
    res = None
    if coll:
        res = next((r for r in coll["results"] if r["panel"] == "e" and r["threshold"] == A.DETECTION), None)
    # the truth: which trait each tumour's genealogy acquired first
    ax = S.ax_mm(fig, x + lab_w, y, 30, 7)
    orders = ((coll or {}).get("truth") or {}).get("orders")
    if not orders:
        placeholder(ax, "genealogy")
    else:
        show = [("breach<met_survival", "breach before met"), ("breach<stromal_survival", "breach before stromal")]
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
        ax.set_xticklabels(["0", "", "1"], fontsize=FS_SMALL)
        ax.set_title("true genealogies: tumours carrying both", fontsize=FS_SMALL, pad=1.5)
    yt = y + 15
    if res is None:
        placeholder(S.ax_mm(fig, x + lab_w, yt, 3 * w + 2 * gap, h), "panel e (collect)")
        return
    for ti, tool in enumerate(tools):
        ax = S.ax_mm(fig, x + lab_w + ti * (w + gap), yt, w, h)
        fit = res["fits"].get(tool) or {}
        rows, band = _e_rows(res["scores"].get(tool), fit)
        if rows is None:
            placeholder(ax, TOOL_LABEL[tool])
            continue
        _forest(ax, rows, band, tool, show_labels=(ti == 0))
        ax.set_xlabel("log odds" if tool == "cooc" else "Θ", fontsize=FS_SMALL, labelpad=1)
    S.text_mm(fig, x, yt + h + 6.5, "grey band: the E0–trait pairs' magnitudes; filled: above them "
                                    "(order: also Θ[M,B] > Θ[B,M])", fontsize=FS_SMALL, color=S.GREY,
              va="top")


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=None, help="summary directory (as fig5_atlas.py)")
    ap.add_argument("--name", default="fig5_cohort")
    args = ap.parse_args()
    out = args.out or os.environ.get("ISCC_FIG5_OUT") or os.path.join(A.REG.OUT, "fig5-atlas")
    data = load(out)
    S.setup()
    fig = S.figure("double", 172)
    S.panel_letter(fig, "a", 2, 2)
    panel_a(fig, 6, 3, data["collect"])
    S.panel_letter(fig, "b", 2, 44)
    panel_b(fig, 13, 52, data["levels"])
    S.panel_letter(fig, "c", 98, 44)
    panel_c(fig, 110, 50, data["levels"])
    S.panel_letter(fig, "d", 2, 116)
    panel_d(fig, 4, 124, data["collect"])
    S.panel_letter(fig, "e", 98, 116)
    panel_e(fig, 96, 120, data["collect"])
    path = S.save(fig, args.name)
    print(f"[fig5] {path} (levels: {sorted(data['levels'])}, collect: {data['collect'] is not None})")


if __name__ == "__main__":
    main()
