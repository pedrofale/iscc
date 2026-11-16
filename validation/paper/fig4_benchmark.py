"""Figure 4: a shared answer key lets integration be scored link by link.

Draws the five panels of the figure plan ("Fig 4") from the summaries the Fig 4 jobs keep; it computes
nothing heavy and reads no tumour:

  (a) the benchmark, a schematic: one tumour measured by WGS, scDNA, scRNA and Visium, the tools on each
      modality and how they chain across modalities, every link scored against the answer key;
  (b) single-modality calls against the truth by niche (duct wall, hypoxic core, stroma): the malignancy
      AUC and the per-arm copy-number correlation per tool and assay, one dot per tumour, the bar the
      mean (``fig4_tools.py``: ``$FIG4_DEV/b/<tumour>/scores.csv``);
  (c) clonealign's balanced accuracy on scRNA given the true clones, the clones called from simulated
      scDNA and the true profiles of only the clones scDNA found, with a correlation classifier on the true
      profiles as the reference; the gap (true minus called) is marked where the truth arm is above chance
      (``fig4_score.CHANCE_MARGIN``); ``fig4_chain.py`` and ``fig4_diag.py`` outputs in ``$FIG4_OUT`` and
      ``$FIG4_DIAG``;
  (d) the per-spot clone-fraction error of Clonalscope with and without the DNA prior, and of CalicoST
      (``$FIG4_OUT/d``);
  (e) structure misleads a lineage method: (i) one ductal-field tumour's clones and hypoxia field, (ii) the
      field's spatial against lineage autocorrelation per tumour, (iii) its lineage share by architecture
      and selection (``fig4_lineage.py``: ``$FIG4_DEV/e``).

Writes ``figures/paper1/fig4_benchmark.{pdf,png}`` through ``_paths`` (``ISCC_PAPER_DIR`` picks the
checkout) and every number it draws to ``$FIG4_DEV/fig4_benchmark_numbers.json``.

    FIG4_OUT=.../fig4-v4 FIG4_DIAG=.../fig4-v4-diag FIG4_DEV=.../fig4-dev/v4 ISCC_PAPER_DIR=.../fig4-dev/v4 \\
        python validation/paper/fig4_benchmark.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import style as S  # noqa: E402

D = "/nobackup/proj/disk/naiss2025-22-1610/personal/pedrof/iscc"
OUT = os.environ.get("FIG4_OUT", f"{D}/fig4-v4")
DIAG = os.environ.get("FIG4_DIAG", f"{D}/fig4-v4-diag")
DEV = os.environ.get("FIG4_DEV", f"{D}/fig4-dev/v4")
TUMOURS = os.environ.get("FIG4_TUMOURS", "seed1,seed2,disp0.1_seed1,disp0.9_seed1,disp0.9_seed2").split(",")

# niche colours, validated with the dataviz palette check (all pairs, light): CVD and normal vision pass
NICHE = {"duct_wall": ("duct wall", "#7570B3"), "hypoxic_core": ("hypoxic core", "#D95F02"),
         "stroma": ("stroma", "#1B9E77")}
TOOL_D = {"Clonalscope": "#EB6834", "Clonalscope, DNA prior": S.MODALITY["DNA"], "CalicoST": "#1BAF7A"}
# the tools that take a normal reference are given the sample's non-epithelial normals (Visium: the spots of
# such normals only), and every tool is scored on the epithelial-lineage barcodes (fig4_tools.py)
ROWS_B = [("scrna", "copykat", "CopyKAT"), ("scrna", "copykat-ref", "CopyKAT, normal ref."),
          ("scrna", "numbat", "Numbat"), ("scrna", "infercnvpy", "inferCNV"),
          ("visium", "copykat", "CopyKAT"), ("visium", "copykat-ref", "CopyKAT, normal ref."),
          ("visium", "calicost", "CalicoST"), ("visium", "clonalscope", "Clonalscope"),
          ("visium", "cancerfinder", "Cancer-Finder")]


def _label(task, swept=False):
    if task.startswith("disp"):
        d, s = task[4:].split("_seed")
        lab = f"disp. {d}, s{s}"
    else:
        lab = f"showcase, s{task.replace('seed', '')}"
    return lab + (" (WGD)" if swept else "")


def _overlay(fig):
    W, H = fig.get_size_inches() / S.MM
    ax = fig.add_axes([0, 0, 1, 1], facecolor="none", zorder=-1)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    return ax


def _box(ax, x, y, w, h, text, fc, ec="none", tc=None, fs=5.2, bold=False):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.0", fc=fc, ec=ec, lw=0.5))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc or S.INK,
            fontweight="bold" if bold else "normal", linespacing=1.15)


def _arrow(ax, x0, y0, x1, y1, color="#6F6F6F", lw=0.6, ls="-"):
    from matplotlib.patches import FancyArrowPatch
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=5, color=color, lw=lw,
                                 ls=ls, shrinkA=0, shrinkB=0))


def panel_a(fig, ov):
    """The benchmark as a schematic: modalities, tools, the chains across modalities, the answer key."""
    S.panel_letter(fig, "a", 2, 3)
    DNA, RNA, SP = S.MODALITY["DNA"], S.MODALITY["RNA"], S.MODALITY["spatial"]
    pale = {DNA: "#E4E1F0", RNA: "#FBE0D1", SP: "#D9EEDD"}
    # the tumour
    from matplotlib.patches import Circle
    cx, cy = 16, 25
    ov.add_patch(Circle((cx, cy), 9.5, fc=S.TISSUE, ec=S.GREY, lw=0.4))
    rng = np.random.default_rng(4)
    cols = [S.CLONE_FAMILIES[0][1], S.CLONE_FAMILIES[1][1], S.CLONE_FAMILIES[2][1], S.CLONE_FAMILIES[3][1]]
    for k in range(70):
        r, th = 8.2 * np.sqrt(rng.random()), 2 * np.pi * rng.random()
        x, y = cx + r * np.cos(th), cy + r * np.sin(th)
        ov.add_patch(Circle((x, y), 0.75, fc=cols[int((np.arctan2(y - cy, x - cx) + np.pi) / (np.pi / 2)) % 4],
                            ec="white", lw=0.2))
    ov.text(cx, cy + 12.5, "one tumour", ha="center", va="top", fontsize=5.2)
    # the four modalities
    mods = [("WGS", DNA, 6.0), ("scDNA", DNA, 16.0), ("scRNA", RNA, 26.0), ("Visium", SP, 36.0)]
    for name, col, y in mods:
        _box(ov, 33, y, 16, 7, name, pale[col], ec=col, bold=True)
        _arrow(ov, 26, cy, 33, y + 3.5, lw=0.45)
    # tools on each modality alone (b) and the chains across modalities (c, d)
    _box(ov, 55, 15.5, 25, 8, "HMMcopy (DLP+)\n+ clustering", pale[DNA], ec=DNA)
    _arrow(ov, 49, 19.5, 55, 19.5)
    _box(ov, 85, 15.5, 16, 8, "DNA clones\nand profiles", "white", ec=DNA)
    _arrow(ov, 80, 19.5, 85, 19.5)
    _box(ov, 55, 25.5, 25, 8, "CopyKAT, Numbat,\ninferCNV", pale[RNA], ec=RNA)
    _arrow(ov, 49, 29.5, 55, 29.5)
    _box(ov, 55, 35.5, 25, 8, "CopyKAT, CalicoST,\nClonalscope, Cancer-Finder", pale[SP], ec=SP, fs=4.6)
    _arrow(ov, 49, 39.5, 55, 39.5)
    _box(ov, 108, 25.5, 22, 8, "clonealign\n(scRNA cells to clones)", pale[RNA], ec=RNA, fs=4.8)
    _arrow(ov, 101, 21.5, 108, 27.5, color=DNA)
    _arrow(ov, 49, 28.0, 108, 29.0, color="#BDBDBD", lw=0.4)
    _box(ov, 108, 35.5, 22, 8, "Clonalscope with\na DNA prior (spots)", pale[SP], ec=SP, fs=4.8)
    _arrow(ov, 101, 22.5, 108, 37.5, color=DNA)
    _box(ov, 108, 5.5, 22, 8, "lineage vs space\n(hypoxia field)", "white", ec=S.GREY, fs=4.8)
    _arrow(ov, 26, 18, 108, 9.5, color="#BDBDBD", lw=0.4)
    for x, y, lab in ((81, 29.5, "b"), (81, 39.5, "b"), (131.5, 29.5, "c"), (131.5, 39.5, "d"), (131.5, 9.5, "e")):
        ov.text(x, y, lab, fontsize=5.5, fontweight="bold", color=S.GREY, va="center")
    # the answer key, scored against every link
    _box(ov, 145, 9, 36, 31, "", "#F4F4F4", ec=S.INK)
    ov.text(163, 12, "answer key", ha="center", va="top", fontsize=5.6, fontweight="bold")
    ov.text(163, 17.5, "true clones and genealogy\ncopy number per cell and arm\nmalignant or normal per cell\n"
            "clone fractions per spot\nniche of every cell\nthe environment's fields", ha="center", va="top",
            fontsize=4.7, linespacing=1.4, color=S.INK)
    for y in (9.5, 19.5, 29.5, 39.5):
        ov.plot([132 if y != 19.5 else 101, 145], [y, min(max(y, 12), 37)], color=S.INK, lw=0.35, ls=(0, (1.5, 1.5)))
    ov.text(138.5, 44, "every link scored", ha="center", va="top", fontsize=4.6, color=S.GREY)


def panel_b(fig, ov, nums):
    S.panel_letter(fig, "b", 2, 52)
    files = [os.path.join(DEV, "b", f"R11_{t}", "scores.csv") for t in TUMOURS]
    have = [pd.read_csv(f).assign(tumour=os.path.basename(os.path.dirname(f))) for f in files if os.path.exists(f)]
    df = pd.concat(have, ignore_index=True) if have else pd.DataFrame()
    nums["b"] = {"tumours": sorted(set(df["tumour"])) if len(df) else []}
    for col, (metric, xlab, lim) in enumerate((("auc", "Malignancy AUC", (0.4, 1.0)),
                                               ("cn_r", "Per-arm copy-number correlation", (-0.2, 1.0)))):
        x0 = 36 + col * 77
        ax = S.ax_mm(fig, x0, 56, 52, 42)
        n = len(ROWS_B)
        for i, (assay, tool, lab) in enumerate(ROWS_B):
            y = n - 1 - i
            if i in (0, 4):
                pass
            sub = df[(df["assay"] == assay) & (df["tool"] == tool)] if len(df) else df
            vals = sub[sub["niche"].isin(NICHE)][metric] if len(sub) else pd.Series(dtype=float)
            if not len(sub) or not np.isfinite(vals.astype(float)).any():
                failed = len(sub) and "status" in sub and sub["status"].astype(str).str.startswith("failed").all()
                note = "pending" if not len(sub) else ("failed in every tumour" if failed else "no call")
                ax.text(lim[0] + 0.02 * (lim[1] - lim[0]), y, note, fontsize=4.4, color=S.GREY, va="center")
                continue
            for k, (niche, (nlab, ncol)) in enumerate(NICHE.items()):
                v = sub[sub["niche"] == niche][metric].astype(float).dropna()
                yy = y + (1 - k) * 0.24
                ax.scatter(v, np.full(len(v), yy), s=4, color=ncol, edgecolors="white", linewidths=0.2, zorder=3)
                if len(v):
                    m = float(v.mean())
                    ax.plot([m, m], [yy - 0.11, yy + 0.11], color=ncol, lw=1.2, zorder=4)
                    nums["b"].setdefault(f"{assay}:{tool}", {})[f"{niche}:{metric}"] = dict(
                        mean=round(m, 3), min=round(float(v.min()), 3), max=round(float(v.max()), 3), n=int(len(v)))
        ax.axhline(4.5, color=S.GREY, lw=0.4)
        ax.set_yticks(range(n))
        ax.set_yticklabels([lab for _, _, lab in ROWS_B][::-1], fontsize=4.8)
        ax.set_ylim(-0.6, n - 0.4)
        ax.set_xlim(*lim)
        ax.set_xlabel(xlab)
        ax.axvline(0.5 if metric == "auc" else 0.0, color=S.INK, lw=0.4, ls=":")
        if col == 0:
            ax.text(-0.42, (n - 1 + 5) / 2 / n, "scRNA", transform=ax.transAxes, rotation=90, va="center",
                    fontsize=5.2, color=S.MODALITY["RNA"], fontweight="bold")
            ax.text(-0.42, 2.0 / n, "Visium", transform=ax.transAxes, rotation=90, va="center", fontsize=5.2,
                    color=S.MODALITY["spatial"], fontweight="bold")
        else:
            ax.set_yticklabels([])
    h = [Line2D([], [], marker="o", ls="none", mfc=c, mec="white", ms=3.4, label=l) for l, c in NICHE.values()]
    h.append(Line2D([], [], color=S.INK, lw=1.2, label="mean over tumours"))
    S.text_mm(fig, 36, 53.0, "Reference: the non-epithelial normals (Visium: their spots). AUC on epithelial-lineage "
              "barcodes, cancer vs. not (spot: half its cells cancer), where a niche holds \u226510 normal epithelial",
              fontsize=4.2, color=S.GREY, va="top")
    fig.legend(handles=h, loc="upper left", bbox_to_anchor=(36 / 183, 1 - 48.6 / FIG_H), ncol=4, fontsize=4.8,
               frameon=False)


def panel_c(fig, ov, c, nums):
    S.panel_letter(fig, "c", 2, 106)
    ax = S.ax_mm(fig, 13, 119, 78, 30)
    DNA = S.MODALITY["DNA"]
    nums["c"] = {}
    for i, task in enumerate(TUMOURS):
        r = c[c["tumour"] == f"R11_{task}"]
        if not len(r):
            continue
        r = r.iloc[0]
        ax.plot([i - 0.32, i + 0.32], [r.chance] * 2, color=S.GREY, lw=0.6, zorder=1)
        ax.plot(i + 0.2, r.bal_corr, marker="D", ms=3.0, mfc="#BDBDBD", mec="white", mew=0.3, ls="none", zorder=2)
        ax.plot(i - 0.14, r.bal_oracle_set, "o", ms=3.4, mfc="white", mec=S.INK, mew=0.6, zorder=2)
        ax.plot(i, r.bal_true, "o", ms=4, mfc=S.INK, mec="white", mew=0.4, zorder=3)
        ax.plot(i, r.bal_dna, "o", ms=4, mfc=DNA, mec="white", mew=0.4, zorder=3)
        if bool(r.above_chance):
            ax.plot([i, i], [r.bal_dna, r.bal_true], color=S.GREY, lw=0.6, zorder=1)
            ax.text(i + 0.08, (r.bal_true + r.bal_dna) / 2, f"{r.bal_dna - r.bal_true:+.2f}".replace("-", "−"),
                    fontsize=4.6, va="center", ha="left")
        else:
            ax.text(i, 0.97, "true clones\nat chance", fontsize=4.2, color=S.GREY, ha="center", va="top",
                    linespacing=1.1)
        nums["c"][task] = {k: (round(float(r[k]), 3) if isinstance(r[k], (float, np.floating)) else r[k])
                           for k in ("bal_true", "bal_dna", "bal_oracle_set", "bal_corr", "chance", "above_chance",
                                     "acc_true", "acc_dna", "majority", "n_true_clones", "n_called_clones",
                                     "hmmcopy_concordance_scaled")}
    ax.set_xticks(range(len(TUMOURS)))
    ax.set_xticklabels([_label(t) for t in TUMOURS], fontsize=4.6, rotation=25, ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.6, len(TUMOURS) - 0.4)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("clonealign balanced accuracy")
    h = [Line2D([], [], marker="o", ls="none", mfc=S.INK, mec="white", ms=4, label="true clones and profiles"),
         Line2D([], [], marker="o", ls="none", mfc=DNA, mec="white", ms=4, label="clones called from scDNA"),
         Line2D([], [], marker="o", ls="none", mfc="white", mec=S.INK, mew=0.6, ms=3.4,
                label="true profiles of the clones scDNA found"),
         Line2D([], [], marker="D", ls="none", mfc="#BDBDBD", mec="white", ms=3.0,
                label="reference: correlation with the true profiles"),
         Line2D([], [], color=S.GREY, lw=0.6, label="chance")]
    fig.legend(handles=h, loc="upper left", bbox_to_anchor=(12 / 183, 1 - 105.5 / FIG_H), fontsize=4.4,
               frameon=False, ncol=2, columnspacing=1.0, handletextpad=0.3)


def panel_d(fig, ov, d, nums):
    S.panel_letter(fig, "d", 100, 106)
    ax = S.ax_mm(fig, 111, 119, 70, 30)
    nums["d"] = {}
    names = list(TOOL_D)
    w = 0.22
    sub_all = d[d["resolution"] == "fine"] if len(d) else d
    for i, task in enumerate(TUMOURS):
        g = sub_all[sub_all["tumour"] == f"R11_{task}"] if len(sub_all) else sub_all
        if not len(g):
            continue
        base = g[g["tool"] == "one clone everywhere"]["error"]
        if len(base):
            ax.plot([i - 0.4, i + 0.4], [base.iat[0]] * 2, color=S.GREY, lw=0.6, ls=":", zorder=1)
        rec = {"largest_clone_everywhere": round(float(base.iat[0]), 3) if len(base) else None,
               "mixed_spots": round(float(g["mixed_spots"].iat[0]), 3)}
        for j, n in enumerate(names):
            v = g[g["tool"] == n]["error"]
            if len(v):
                ax.plot(i + (j - 1) * w, v.iat[0], "o", ms=4, mfc=TOOL_D[n], mec="white", mew=0.4, zorder=3)
                rec[n] = round(float(v.iat[0]), 3)
        nums["d"][task] = rec
    ax.set_xticks(range(len(TUMOURS)))
    ax.set_xticklabels([_label(t) for t in TUMOURS], fontsize=4.6, rotation=25, ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.6, len(TUMOURS) - 0.4)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("per-spot clone-fraction error")
    h = [Line2D([], [], marker="o", ls="none", mfc=TOOL_D[n], mec="white", ms=4, label=n) for n in names]
    h.append(Line2D([], [], color=S.GREY, lw=0.6, ls=":", label="largest clone everywhere"))
    fig.legend(handles=h, loc="upper left", bbox_to_anchor=(110 / 183, 1 - 105.5 / FIG_H), fontsize=4.4,
               frameon=False, ncol=2, columnspacing=1.0, handletextpad=0.3)


def panel_e(fig, ov, nums):
    import fig4_lineage as FL
    S.panel_letter(fig, "e", 2, 162)
    p = os.path.join(DEV, "e", "panel_i.npz")
    if os.path.exists(p):
        z = np.load(p, allow_pickle=True)
        for k, (key, title) in enumerate((("clones", "clones"), ("hypoxia", "hypoxia field"))):
            ax = S.ax_mm(fig, 6 + k * 24, 168, 22, 22)
            if key == "clones":
                ax.imshow(z[key], interpolation="nearest")
            else:
                im = ax.imshow(z[key], cmap="YlOrRd", vmin=0, vmax=1, interpolation="nearest")
                cax = S.ax_mm(fig, 6 + k * 24 + 22.6, 168, 1.3, 22)
                fig.colorbar(im, cax=cax).ax.tick_params(labelsize=4.3, length=1.5)
            S.despine(ax, False, False)
            ax.set_title(title, fontsize=5.2, pad=2)
        S.text_mm(fig, 6, 193.5, f"(i) {z['run']} {z['task']}, ductal field", fontsize=4.5, color=S.GREY, va="top")
    df0 = FL.load(DEV)
    if len(df0) and "stalled" in df0 and df0["stalled"].any():
        names = ", ".join(f"{r} {t.replace('_seed', ' s').replace('_', ' ')}" for r, t in
                          zip(df0.run[df0.stalled], df0.task[df0.stalled]))
        S.text_mm(fig, 6, 197.0, f"(ii, iii) open: {int(df0.stalled.sum())} tumours that stalled before their target "
                  f"size\n({names})", fontsize=4.2, color=S.GREY, va="top", linespacing=1.2)
    df = FL.load(DEV)
    ARCH = {"ductal": ("ductal field", S.CLONE_FAMILIES[0][1]), "free": ("without glands", S.CLONE_FAMILIES[2][1])}
    if len(df):
        ax = S.ax_mm(fig, 68, 168, 30, 26)
        stalled = df["stalled"].astype(bool) if "stalled" in df else pd.Series(False, index=df.index)
        for arch, (lab, col) in ARCH.items():
            sub = df[(df.arch == arch) & ~stalled]
            ax.scatter(sub.field_sp, sub.field_lin, s=5, color=col, edgecolors=S.INK, linewidths=0.2,
                       label=f"{lab} ({int((df.arch == arch).sum())})", zorder=3)
            st = df[(df.arch == arch) & stalled]
            ax.scatter(st.field_sp, st.field_lin, s=7, facecolors="white", edgecolors=col, linewidths=0.6, zorder=4)
        if stalled.any():
            ax.scatter([], [], s=7, facecolors="white", edgecolors=S.INK, linewidths=0.6,
                       label=f"stalled in situ ({int(stalled.sum())})")
        lim = max(1.0, 1.05 * float(np.nanmax(df[["field_sp", "field_lin"]].to_numpy())))
        ax.plot([0, lim], [0, lim], color=S.INK, lw=0.5, ls=":")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel("Spatial autocorrelation")
        ax.set_ylabel("Lineage autocorrelation")
        ax.legend(fontsize=4.4, loc="upper left", frameon=False)
        S.text_mm(fig, 68, 164.5, "(ii)", fontsize=5, va="top")
        ax = S.ax_mm(fig, 118, 168, 62, 26)
        groups = [("R1", "ductal", None, "ductal\nsweep (R1)"), ("R3", "ductal", "selected", "ductal\nselected"),
                  ("R3", "ductal", "neutral", "ductal\nneutral"), ("R3", "free", "selected", "no glands\nselected"),
                  ("R3", "free", "neutral", "no glands\nneutral")]
        for k, (run, arch, sel, lab) in enumerate(groups):
            sub = df[(df.run == run) & (df.arch == arch) & (df.selection.astype(str) == str(sel))]
            x = k + np.random.default_rng(k).uniform(-0.17, 0.17, len(sub))
            st = sub["stalled"].astype(bool).to_numpy() if "stalled" in sub else np.zeros(len(sub), bool)
            ax.scatter(x[~st], sub.share[~st], s=4, color=ARCH[arch][1], alpha=0.9 if sel != "neutral" else 0.55,
                       edgecolors="none", zorder=3)
            ax.scatter(x[st], sub.share[st], s=6, facecolors="white", edgecolors=ARCH[arch][1], linewidths=0.6,
                       zorder=4)
            if len(sub):
                ax.plot([k - 0.3, k + 0.3], [sub.share.mean()] * 2, color=S.INK, lw=0.9, zorder=4)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([g[3] for g in groups], fontsize=4.5)
        ax.set_ylim(0, max(0.6, 1.1 * float(df.share.max())))
        ax.set_ylabel("Lineage share of the field")
        S.text_mm(fig, 118, 164.5, "(iii)", fontsize=5, va="top")
        nums["e"] = FL.summary(DEV)
    else:
        nums["e"] = {}


FIG_H = 205


def main():
    import fig4_score as FS
    S.setup()
    fig = S.figure("double", FIG_H)
    ov = _overlay(fig)
    nums = {"sources": dict(fig4_out=OUT, diag=DIAG, dev=DEV, tumours=TUMOURS,
                            chance_margin=FS.CHANCE_MARGIN)}
    panel_a(fig, ov)
    panel_b(fig, ov, nums)
    c, d = FS.collect(OUT, DIAG)
    c = c[c["resolution"] == "fine"] if len(c) else c
    panel_c(fig, ov, c, nums)
    panel_d(fig, ov, d, nums)
    panel_e(fig, ov, nums)
    path = S.save(fig, "fig4_benchmark")
    os.makedirs(DEV, exist_ok=True)
    with open(os.path.join(DEV, "fig4_benchmark_numbers.json"), "w") as f:
        json.dump(nums, f, indent=1, default=str)
    print("wrote", path, "and", os.path.join(DEV, "fig4_benchmark_numbers.json"))


if __name__ == "__main__":
    main()
