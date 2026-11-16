"""Fig 5: what batch integration removes from a cohort, and the dependencies progression models infer.

Draws ``figures/paper1/fig5_cohort.pdf`` (through ``validation/_paths.py``; ``ISCC_PAPER_DIR`` picks
the paper checkout) from the summaries :mod:`fig5_cohort` writes, and from nothing else:
``part1/cohort<c>.json``, ``part2_summary.json`` and ``part3_summary.json`` under ``$ISCC_FIG5_OUT``
(else ``<registry OUT>/fig5``). A panel whose summaries are missing is drawn as a placeholder that
says so. Registered design and expectations: vault note exp-paper1-fig5-cohort.

a. The design: patients over one shared landscape, pooled into batches two ways (confounded, one
   batch per patient; balanced, patients spread over batches), and the planted network of Part II.
b. Embeddings of one cohort at the strong batch level, without integration and after Harmony and
   scVI, under the two designs, coloured by batch and by each patient's private clones.
c. What integration removes: batch mixing among microenvironment cells (iLISI as a share of its
   batch-free value) against cancer cells' patient-specific neighbourhoods kept (patient purity as a
   share of the batch-free embedding's), per design, tool, level and cohort.
d. The planted network: the rank (of 15 pairs, by |Theta| or |log odds|) of the synergistic pair
   E1-E2 and the exclusive pair E3-E4 against cohort size, per tool (filled where the sign is right),
   and how often the order E5 -> E6 is read in the right direction.
e. Tissue architecture without epistasis: the strength of the trait dependencies (breach, stromal and
   metastatic survival) in the ductal and neutral arms at 300 tumours, against the empty network's
   null (the 95th percentile of the top strength over its resamples).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch  # noqa: E402

import fig5_cohort as F  # noqa: E402
import style as S  # noqa: E402

TOOL_COLOUR = {"none": S.GREY, "harmony": S.TOOL["Harmony"], "scvi": S.TOOL["scVI"],
               "cooc": "#4D4D4D", "mhn": S.TOOL["MHN"], "treemhn": S.TOOL["TreeMHN"]}
TOOL_LABEL = {"none": "no integration", "harmony": "Harmony", "scvi": "scVI", "cooc": "co-occurrence",
              "mhn": "MHN", "treemhn": "TreeMHN"}
DESIGN_LABEL = {"confounded": "one batch per patient", "balanced": "patients across batches"}
LEVEL_MARKER = {"10x": "o", "intermediate": "s", "strong": "^"}


def load(out):
    """Every summary present under ``out`` (None where a part has not run)."""
    def read(p):
        if not os.path.exists(p):
            return None
        with open(p) as f:
            return json.load(f)
    d1 = os.path.join(out, "part1")
    p1 = [read(os.path.join(d1, fn)) for fn in sorted(os.listdir(d1))] if os.path.isdir(d1) else []
    return dict(part1=[x for x in p1 if x], part2=read(os.path.join(out, "part2_summary.json")),
                part3=read(os.path.join(out, "part3_summary.json")))


def placeholder(ax, what):
    S.despine(ax, False, False)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color(S.BG_CELLS)
    ax.text(0.5, 0.5, f"{what}\nnot yet run", ha="center", va="center", fontsize=5.5, color=S.GREY,
            transform=ax.transAxes)


# ------------------------------------------------------------------ a: the design
def panel_a(fig, x, y):
    ax = S.ax_mm(fig, x, y, 86, 52)
    ax.set_xlim(0, 86)
    ax.set_ylim(52, 0)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((1, 2), 30, 7, boxstyle="round,pad=0.4", fc=S.PASTEL["grey"], ec="none"))
    ax.text(16, 5.5, "one landscape (layout seed)", ha="center", va="center", fontsize=5.5)
    pal = [fam[1] for fam in S.CLONE_FAMILIES[:6]]
    for p in range(6):
        cx = 4 + 5 * p
        ax.add_patch(FancyArrowPatch((16, 9.5), (cx, 14.5), arrowstyle="-", lw=0.4, color=S.GREY))
        ax.add_patch(Circle((cx, 17), 2.2, fc=pal[p], ec="none"))
        ax.text(cx, 17, str(p + 1), ha="center", va="center", fontsize=4.5, color="white")
    ax.text(16, 22.5, "patients: own seed, private clones", ha="center", fontsize=5, color=S.GREY)
    for k, (name, mix) in enumerate((("confounded", False), ("balanced", True))):
        y0 = 29 + 12 * k
        ax.text(1, y0 - 1.5, DESIGN_LABEL[name], fontsize=5.5)
        for b in range(6):
            bx = 1 + 5 * b
            ax.add_patch(FancyBboxPatch((bx, y0), 4, 6, boxstyle="round,pad=0.2", fc="white",
                                        ec=S.INK, lw=0.4))
            members = range(6) if mix else [b]
            for q, p in enumerate(members):
                ax.add_patch(Circle((bx + 1 + (q % 2) * 2, y0 + 1 + (q // 2) * 2), 0.7, fc=pal[p], ec="none"))
    # the planted network (Part II)
    ax.text(62, 4, "planted network (Part II)", ha="center", fontsize=5.5)
    pos = {0: (46, 14), 1: (58, 14), 2: (46, 28), 3: (58, 28), 4: (70, 14), 5: (82, 14)}
    for e, (px, py) in pos.items():
        ax.add_patch(Circle((px, py), 3.2, fc="white", ec=S.INK, lw=0.5))
        ax.text(px, py, f"E{e + 1}", ha="center", va="center", fontsize=5)
    ax.plot([49.2, 54.8], [14, 14], color="#2F6D3A", lw=1.2)
    ax.text(52, 11.5, "+0.4", ha="center", fontsize=4.5, color="#2F6D3A")
    ax.plot([49.2, 54.8], [28, 28], color="#C44E52", lw=1.2, ls=(0, (2, 1)))
    ax.text(52, 25.5, "lethal", ha="center", fontsize=4.5, color="#C44E52")
    ax.add_patch(FancyArrowPatch((73.2, 14), (78.8, 14), arrowstyle="-|>", mutation_scale=5, lw=0.8,
                                 color=S.INK))
    ax.text(76, 11.5, "order", ha="center", fontsize=4.5)
    ax.text(64, 36, "own effect 0.05 each; modules of 10 driver genes\n"
                    "Part III: breach, stromal and metastatic survival,\nno interaction planted",
            ha="center", va="top", fontsize=4.8, color=S.GREY)


# ------------------------------------------------------------------ b: embeddings
def _clone_colour(key):
    """One hue family per patient, one shade per clone ("P<p>::clone <k>"; the ancestral backbone
    lightest)."""
    patient, label = key.split("::", 1)
    fam = S.CLONE_FAMILIES[int(patient[1:]) % (len(S.CLONE_FAMILIES) - 1)]
    k = int(label.split()[-1]) - 1 if label.startswith("clone") else len(fam) - 1
    return fam[min(k, len(fam) - 1)]


def panel_b(fig, x, y, part1):
    rec = next((r for r in part1 if r.get("embeddings")), None)
    designs, methods = ("confounded", "balanced"), ("none", "harmony", "scvi")
    w, h, gap = 16, 16, 1.5
    if rec is None:
        placeholder(S.ax_mm(fig, x, y, 6 * w + 5 * gap, 2 * h + gap), "Part I embeddings")
        return
    emb = rec["embeddings"]
    patients = np.array(emb["labels"]["patient"])
    clone = np.array(emb["labels"]["clone"])
    is_cancer = np.array(emb["labels"]["cell_type"]) == "cancer"
    uniq = sorted(set(clone[is_cancer]))
    ccol = {c: _clone_colour(c) for c in uniq}
    for di, d in enumerate(designs):
        batch = np.array(emb["labels"]["batch"][d])
        bcol = {b: c for b, c in zip(sorted(set(batch)), ["#4C72B0", "#DD8452", "#55A868", "#C44E52",
                                                          "#8172B3", "#937860", "#DA8BC3", "#8C8C8C"])}
        for mi, m in enumerate(methods):
            key = f"{d}/{emb['level']}/{m}"
            for row, how in enumerate(("batch", "clone")):
                ax = S.ax_mm(fig, x + (3 * di + mi) * (w + gap), y + row * (h + gap), w, h)
                S.despine(ax, False, False)
                if key not in emb["xy"]:
                    placeholder(ax, TOOL_LABEL[m])
                    continue
                xy = np.asarray(emb["xy"][key])
                if how == "batch":
                    ax.scatter(xy[:, 0], xy[:, 1], s=0.3, c=[bcol[b] for b in batch], rasterized=True)
                else:
                    ax.scatter(xy[~is_cancer, 0], xy[~is_cancer, 1], s=0.3, c=S.BG_CELLS, rasterized=True)
                    ax.scatter(xy[is_cancer, 0], xy[is_cancer, 1], s=0.3,
                               c=[ccol[c] for c in clone[is_cancer]], rasterized=True)
                if row == 0:
                    ax.set_title(TOOL_LABEL[m], fontsize=5.5, pad=1.5)
        S.text_mm(fig, x + 3 * di * (w + gap) + 1.5 * (w + gap) - gap / 2, y - 4.5, DESIGN_LABEL[d],
                  ha="center", fontsize=5.5)
    S.text_mm(fig, x - 1.5, y + h / 2, "batch", rotation=90, ha="right", va="center", fontsize=5)
    S.text_mm(fig, x - 1.5, y + h + gap + h / 2, "clone", rotation=90, ha="right", va="center", fontsize=5)
    S.text_mm(fig, x, y + 2 * h + 2 * gap + 2.5,
              f"cohort {rec['cohort'] + 1}, {emb['level']} batch effect, {len(patients)} cells "
              f"({emb.get('projection', '')})", fontsize=4.8, color=S.GREY)


# ------------------------------------------------------------------ c: what integration removes
def panel_c(fig, x, y, part1):
    ax = S.ax_mm(fig, x, y, 40, 34)
    if not part1:
        placeholder(ax, "Part I")
        return
    for rec in part1:
        for key, r in rec["runs"].items():
            d, lev, m = key.split("/")
            ret = r.get("retention") or {}
            xv, yv = ret.get("ilisi_batch_noncancer_retention"), ret.get("patient_purity_cancer_retention")
            if xv is None or yv is None:
                continue
            ax.scatter([xv], [yv], s=9, marker=LEVEL_MARKER.get(lev, "o"),
                       facecolors=TOOL_COLOUR[m] if d == "confounded" else "white",
                       edgecolors=TOOL_COLOUR[m], linewidths=0.6)
    ax.axhline(1, color=S.BG_CELLS, lw=0.5, zorder=0)
    ax.axvline(0.8, color=S.BG_CELLS, lw=0.5, ls=":", zorder=0)
    ax.axhline(0.6, color=S.BG_CELLS, lw=0.5, ls=":", zorder=0)
    ax.set_xlabel("microenvironment batch mixing\n(iLISI, share of batch-free)")
    ax.set_ylabel("cancer cells' patient\nneighbourhood kept (share)")
    handles = ([Line2D([], [], marker="o", ls="", color=TOOL_COLOUR[m], markersize=3, label=TOOL_LABEL[m])
                for m in ("none", "harmony", "scvi")]
               + [Line2D([], [], marker="o", ls="", mfc=S.INK, mec=S.INK, markersize=3,
                         label="one batch per patient"),
                  Line2D([], [], marker="o", ls="", mfc="white", mec=S.INK, markersize=3,
                         label="patients across batches"),
                  Line2D([], [], marker="o", ls="", mfc="white", mec=S.GREY, markersize=3, label="10x preset"),
                  Line2D([], [], marker="^", ls="", mfc="white", mec=S.GREY, markersize=3, label="strong"),
                  Line2D([], [], marker="s", ls="", mfc="white", mec=S.GREY, markersize=3,
                         label="intermediate (exploratory,\nadded after the smoke test)")])
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=4.8)


# ------------------------------------------------------------------ d: the planted network
def _by(fits, arm_prefix, tool, kind="nested", thr=F.DETECTION):
    out = {}
    for f in fits:
        if f["arm"].startswith(arm_prefix) and f["kind"] == kind and f["threshold"] == thr:
            fit = f["fits"].get(tool) or {}
            if "pairs" in fit:
                out.setdefault(f["n"], []).append((f["arm"], fit))
    return out


def panel_d(fig, x, y, part2):
    tools = ("cooc", "mhn", "treemhn")
    w, gap = 24, 4
    if part2 is None:
        placeholder(S.ax_mm(fig, x, y, 3 * w + 2 * gap, 30), "Part II")
        return
    pairs = {"E1-E2": ("0,1", +1), "E3-E4": ("2,3", -1)}
    for ti, tool in enumerate(tools):
        ax = S.ax_mm(fig, x + ti * (w + gap), y, w, 24)
        data = _by(part2["fits"], "planted", tool)
        if not data:
            placeholder(ax, TOOL_LABEL[tool])
            continue
        ns = sorted(data)
        for pi, (lab, (key, sign)) in enumerate(pairs.items()):
            col = ["#2F6D3A", "#C44E52"][pi]
            for n in ns:
                for arm, fit in data[n]:
                    p = fit["pairs"][key]
                    ax.scatter([n * (1 + 0.04 * (pi - 0.5))], [p["rank"]], s=7,
                               facecolors=col if p["sign"] == sign else "white", edgecolors=col,
                               linewidths=0.6)
            med = [np.median([fit["pairs"][key]["rank"] for _, fit in data[n]]) for n in ns]
            ax.plot(ns, med, color=col, lw=0.7, label=lab)
        ax.set_xscale("log")
        ax.set_xticks(ns)
        ax.set_xticklabels([str(n) for n in ns])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlim(min(ns) / 1.3, max(ns) * 1.3)
        ax.set_ylim(15.5, 0.5)
        ax.set_yticks([1, 5, 10, 15])
        ax.set_title(TOOL_LABEL[tool], fontsize=5.5, pad=1.5)
        if ti == 0:
            ax.set_ylabel("rank of the planted pair\n(of 15)")
            ax.legend(loc="lower right", fontsize=4.8)
        ax.set_xlabel("patients")
        if tool == "cooc":
            cbn = {}
            for f in part2["fits"]:
                c = f["fits"].get("cbn") or {}
                if f["arm"].startswith("planted") and f["kind"] == "nested" and "edges" in c \
                        and f["threshold"] == F.DETECTION:
                    # evam merges events with identical columns into one node ("E2_E1_E5")
                    cbn.setdefault(f["n"], []).append(any("E4" in a.split("_") and "E5" in b.split("_")
                                                          for a, b in c["edges"]))
            if cbn:
                ax.text(0.03, 0.03, "CBN E5\u2192E6 edge: " + ", ".join(
                    f"{np.mean(cbn[n]):.0%}" for n in sorted(cbn)), transform=ax.transAxes, fontsize=4.5,
                    color=S.GREY)
        # the order E5 -> E6: share of replicates with Theta[E6, E5] > Theta[E5, E6]
        if tool != "cooc":
            right = [np.mean([fit["pairs"]["4,5"]["ji"] > fit["pairs"]["4,5"]["ij"] for _, fit in data[n]])
                     for n in ns]
            ax.text(0.03, 0.03, "E5→E6 right: " + ", ".join(f"{r:.0%}" for r in right),
                    transform=ax.transAxes, fontsize=4.5, color=S.GREY)


# ------------------------------------------------------------------ e: architecture without epistasis
def _null95(part2, tool, n):
    if part2 is None:
        return None
    tops = [f["fits"][tool]["top_strength"] for f in part2["fits"]
            if f["arm"] in part2.get("null_arms", ["empty"]) and f["kind"] == "bootstrap" and f["n"] == n
            and f["threshold"] == F.DETECTION and "top_strength" in f["fits"].get(tool, {})]
    return float(np.percentile(tops, 95)) if tops else None


def panel_e(fig, x, y, part3, part2):
    tools = ("cooc", "mhn", "treemhn")
    ax = S.ax_mm(fig, x, y, 50, 30)
    if part3 is None:
        placeholder(ax, "Part III")
        return
    names = next(iter(part3["cohorts"].values()))["names"]
    idx = {n: i for i, n in enumerate(names)}
    show = [("breach", "stromal_survival"), ("breach", "met_survival"), ("stromal_survival", "met_survival"),
            ("E0", "E1")]
    short = {"breach": "B", "stromal_survival": "S", "met_survival": "M", "E0": "C1", "E1": "C2"}
    n = max(f["n"] for f in part3["fits"])
    xt, labels = [], []
    for pi, (a, b) in enumerate(show):
        key = f"{min(idx[a], idx[b])},{max(idx[a], idx[b])}"
        for ti, tool in enumerate(tools):
            for ai, arm in enumerate(("ductal", "neutral")):
                vals = [f["fits"][tool]["pairs"][key]["strength"] for f in part3["fits"]
                        if f["arm"] == arm and f["n"] == n and f["kind"] == "nested"
                        and f["threshold"] == F.DETECTION and "pairs" in f["fits"].get(tool, {})]
                if not vals:
                    continue
                xp = pi * 4 + ti + 0.3 * (ai - 0.5)
                ax.scatter([xp], vals[:1], s=10, marker="o",
                           facecolors=TOOL_COLOUR[tool] if arm == "ductal" else "white",
                           edgecolors=TOOL_COLOUR[tool], linewidths=0.6)
        xt.append(pi * 4 + 1)
        labels.append(f"{short[a]}–{short[b]}")
    for ti, tool in enumerate(tools):
        nl = _null95(part2, tool, n)
        if nl is not None:
            ax.axhline(nl, color=TOOL_COLOUR[tool], lw=0.5, ls=":", zorder=0)
    ax.set_xticks(xt)
    ax.set_xticklabels(labels)
    ax.set_ylabel("|Θ| or |log odds|")
    ax.set_title(f"{n} tumours; filled ductal, hollow neutral; dotted: empty-network null (95%)",
                 fontsize=4.8, pad=1.5)


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=None, help="summary directory (as fig5_cohort.py)")
    ap.add_argument("--name", default="fig5_cohort")
    args = ap.parse_args()
    out = args.out or os.environ.get("ISCC_FIG5_OUT") or os.path.join(F.REG.OUT, "fig5")
    data = load(out)
    S.setup()
    fig = S.figure("double", 150)
    S.panel_letter(fig, "a", 2, 3)
    panel_a(fig, 6, 6)
    S.panel_letter(fig, "c", 98, 3)
    panel_c(fig, 112, 8, data["part1"])
    S.panel_letter(fig, "b", 2, 60)
    panel_b(fig, 8, 68, data["part1"])
    S.panel_letter(fig, "d", 2, 112)
    panel_d(fig, 16, 118, data["part2"])
    S.panel_letter(fig, "e", 110, 112)
    panel_e(fig, 126, 118, data["part3"], data["part2"])
    path = S.save(fig, args.name)
    print(f"[fig5] {path} (part1 cohorts: {len(data['part1'])}, part2: {data['part2'] is not None}, "
          f"part3: {data['part3'] is not None})")


if __name__ == "__main__":
    main()
