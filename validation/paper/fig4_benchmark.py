"""Figure 4: a shared ground truth lets integration be scored link by link.

Draws the five panels of Fig 4 at half a page (165 by 114 mm) from the summaries the Fig 4 jobs keep; it
computes nothing heavy and reads no tumour:

  (a) the benchmark, a schematic: one tumour measured by WGS, scDNA, scRNA and Visium, the tools on each
      modality, the chain from the scDNA clones into scRNA, the lineage analysis on the truth, and the
      ground truth every link is scored against;
  (b) single-modality calls against the truth by niche (duct wall, hypoxic core, stroma): the malignancy
      AUC and the per-arm copy-number correlation per tool and assay, one dot per tumour, the bar the
      mean (``fig4_tools.py``: ``$FIG4_DEV/b/<tumour>/scores.csv``);
  (c) clonealign's balanced accuracy on scRNA given the true clones, the clones called from simulated
      scDNA and the true profiles of only the clones scDNA found, with a correlation classifier on the true
      profiles as the reference; a line joins true and called where the truth arm is above chance
      (``fig4_score.CHANCE_MARGIN``); ``fig4_chain.py`` and ``fig4_diag.py`` outputs in ``$FIG4_OUT`` and
      ``$FIG4_DIAG``;
  (e) copy-number tree methods against the true genealogy, confined against free growth (a placeholder
      until ``trees.csv`` of exp-paper1-cn-trees-architecture exists, ``$FIG4_TREES``);
  (d) structure misleads a lineage method: the hypoxia field's lineage and spatial autocorrelation
      side by side by architecture and selection, and the share of tumours each test calls heritable
      (``fig4_lineage.py``: ``$FIG4_DEV/e``); the scatter of one against the other, a separate panel
      before 2026-11-25, repeated the same numbers.

The per-spot clone-fraction error of Clonalscope with and without the DNA prior, and the map of one
ductal-field tumour (Fig 4d and 4e(i) before the half-page layout), are S12 Fig, drawn by
``supp_benchmark_extra.py`` from the same inputs.

Writes ``figures/paper1/fig4_benchmark.{pdf,png}`` through ``_paths`` (``ISCC_PAPER_DIR`` picks the
checkout) and every number it draws to ``$FIG4_DEV/fig4_benchmark_numbers.json``. The stalled tumours of
(d) come from the registry records under ``$ISCC_PAPER_OUT``; without it none is found.

    FIG4_OUT=.../fig4-v4 FIG4_DIAG=.../fig4-v4-diag FIG4_DEV=.../fig4-dev/v4 ISCC_PAPER_OUT=.../paper-v4 \\
        ISCC_PAPER_DIR=... python validation/paper/fig4_benchmark.py

A ``FIG4_*`` variable that is not set falls back to the same directory under ``$ISCC_DATA_DIR``.
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

import _paths as P  # noqa: E402
import style as S  # noqa: E402

# The Fig 4 job outputs: each from its own variable, else under $ISCC_DATA_DIR (``_paths.data_dir``).
OUT = os.environ.get("FIG4_OUT") or P.data_dir("fig4-v4")
DIAG = os.environ.get("FIG4_DIAG") or P.data_dir("fig4-v4-diag")
DEV = os.environ.get("FIG4_DEV") or P.data_dir("fig4-dev", "v4")
# The tumours of Fig 4 are R14, R11's design and seeds with the immune variant (the figures about measured
# data and tools use tumours with immune cells; Pedro, 2026-11-22). Every one of the six reaches its target.
RUN = os.environ.get("FIG4_RUN", "R14")
TUMOURS = os.environ.get("FIG4_TUMOURS",
                         "seed1,seed2,disp0.1_seed1,disp0.1_seed2,disp0.9_seed1,disp0.9_seed2").split(",")

# Half a page at full text width: 165 mm by at most 114 mm, included at its true size, so no text is
# smaller than SMALL (5.5 pt) and the panel letters are 8 pt.
W_MM, FIG_H = 165, 168                  # 114 before panel E (2026-11-25), 156 before (d) took the control
SMALL = 5.5

# niche colours, validated with the dataviz palette check (all pairs, light): CVD and normal vision pass
NICHE = {"duct_wall": ("duct wall", "#E7298A"), "hypoxic_core": ("hypoxic core", "#A6761D"),
         "stroma": ("stroma", "#666666")}      # hues no modality uses (the modalities are purple, orange, green)
# the tools that take a normal reference are given the sample's non-epithelial normals (Visium: the spots of
# such normals only), and every tool is scored on the epithelial-lineage barcodes (fig4_tools.py)
# (CopyKAT runs twice: "copykat" without a reference, "copykat-ref" given the microenvironment normals)
ROWS_B = [("scrna", "copykat", "CopyKAT, no reference"), ("scrna", "copykat-ref", "CopyKAT, reference"),
          ("scrna", "numbat", "Numbat"), ("scrna", "infercnvpy", "inferCNV"),
          ("visium", "copykat", "CopyKAT, no reference"), ("visium", "copykat-ref", "CopyKAT, reference"),
          ("visium", "calicost", "CalicoST"), ("visium", "clonalscope", "Clonalscope"),
          ("visium", "cancerfinder", "Cancer-Finder")]
# the architectures of (d): ductal field and growth without glands
ARCH = {"ductal": ("ductal field", S.CLONE_FAMILIES[0][1]), "free": ("without glands", S.CLONE_FAMILIES[2][1])}

# Layout in mm from the top-left corner. Row 1: (a). Row 2: (b) and (c). Row 3: (d).
A_ROWS, A_BH = (1.0, 8.2, 15.4, 22.6), 5.2        # the four modality rows of (a) and their box height
# (b) is clonealign on the left, (c) the single-modality calls on the right (swapped on 2026-11-25, so the
# chain from DNA into RNA comes first, as in the text)
B_TOP, B_H, B_W, B_X = 35.0, 29.5, 36.0, (86.0, 127.0)
C_X, C_TOP, C_W, C_H = 12.0, 41.0, 45.5, 21.0
R3_LETTER, R3_TOP, R3_H = 74.5, 84.0, 30.0       # (d) with its legends above the axes
R4_LETTER, R4_TOP, R4_H = 127.5, 132.0, 26.0      # (e): copy-number trees


def _group(task):
    return f"dispersal {task[4:].split('_seed')[0]}" if task.startswith("disp") else "showcase"


def tumour_axis(ax, tasks, h_mm):
    """Label the benchmark tumours in two rows: the seed under each tumour, and its regime (the showcase or
    a dispersal) under a bracket spanning its seeds; ``h_mm`` is the height of the axes."""
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels([f"seed {t.split('seed')[-1]}" for t in tasks], fontsize=SMALL)
    groups = {}
    for i, t in enumerate(tasks):
        groups.setdefault(_group(t), []).append(i)
    tr = ax.get_xaxis_transform()
    y = -4.0 / h_mm
    for lab, idx in groups.items():
        ax.plot([min(idx) - 0.3, max(idx) + 0.3], [y, y], transform=tr, color=S.GREY, lw=0.4, clip_on=False)
        ax.text((min(idx) + max(idx)) / 2, y - 0.4 / h_mm, lab, transform=tr, ha="center", va="top",
                fontsize=SMALL)


def _overlay(fig):
    W, H = fig.get_size_inches() / S.MM
    ax = fig.add_axes([0, 0, 1, 1], facecolor="none", zorder=-1)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    return ax


def _box(ax, x, y, w, h, text, fc, ec="none", tc=None, fs=SMALL, bold=False):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.0", fc=fc, ec=ec, lw=0.5))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc or S.INK,
            fontweight="bold" if bold else "normal", linespacing=1.1)


def _arrow(ax, x0, y0, x1, y1, color="#6F6F6F", lw=0.6, ls="-"):
    from matplotlib.patches import FancyArrowPatch
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=5, color=color, lw=lw,
                                 ls=ls, shrinkA=0, shrinkB=0))


def panel_a(fig, ov):
    """The benchmark as a schematic, one row per modality: the tools on each, the chain from the scDNA
    clones into scRNA, the lineage analysis on the truth, and the ground truth every link is scored against."""
    S.panel_letter(fig, "a", 0.5, 0.5)
    DNA, RNA, SP = S.MODALITY["DNA"], S.MODALITY["RNA"], S.MODALITY["spatial"]
    pale = {DNA: "#E4E1F0", RNA: "#FBE0D1", SP: "#D9EEDD"}
    mid = [y + A_BH / 2 for y in A_ROWS]
    # the tumour
    from matplotlib.patches import Circle
    cx, cy, rad = 13.5, (mid[0] + mid[-1]) / 2, 7.5
    ov.add_patch(Circle((cx, cy), rad, fc=S.TISSUE, ec=S.GREY, lw=0.4))
    rng = np.random.default_rng(4)
    cols = [S.CLONE_FAMILIES[k][1] for k in range(4)]
    for _ in range(55):
        r, th = (rad - 1.1) * np.sqrt(rng.random()), 2 * np.pi * rng.random()
        x, y = cx + r * np.cos(th), cy + r * np.sin(th)
        ov.add_patch(Circle((x, y), 0.6, fc=cols[int((np.arctan2(y - cy, x - cx) + np.pi) / (np.pi / 2)) % 4],
                            ec="white", lw=0.2))
    ov.text(cx, cy + rad + 0.8, "one tumor", ha="center", va="top", fontsize=SMALL)

    def rim(x, y):
        d = np.array([x - cx, y - cy])
        return np.array([cx, cy]) + rad * d / np.hypot(*d)

    # the four modalities
    MX = 26.0
    for (name, col), y, m in zip((("WGS", DNA), ("scDNA", DNA), ("scRNA", RNA), ("Visium", SP)), A_ROWS, mid):
        _box(ov, MX, y, 14, A_BH, name, pale[col], ec=col, fs=6, bold=True)
        _arrow(ov, *rim(MX, m), MX, m, lw=0.45)
    # the lineage analysis reads the truth of the tumour: a gray line out between the WGS and scDNA rows
    g = (A_ROWS[0] + A_BH + A_ROWS[1]) / 2
    ov.plot([rim(MX, g)[0], MX], [rim(MX, g)[1], g], color="#BDBDBD", lw=0.4)
    _arrow(ov, MX, g, 95, mid[0], color="#BDBDBD", lw=0.4)
    # clonealign reads the scRNA cells (gray, behind the scRNA tools) and the DNA clones (purple)
    TX, TW = 45.0, 34.0
    _arrow(ov, MX + 14, mid[2] + 1.6, 106, mid[2] + 1.6, color="#BDBDBD", lw=0.4)
    # the tools on each modality alone (b)
    _box(ov, TX, A_ROWS[1], TW, A_BH, "HMMcopy, then clustering", pale[DNA], ec=DNA)
    _box(ov, TX, A_ROWS[2], TW, A_BH, "CopyKAT, Numbat, inferCNV", pale[RNA], ec=RNA)
    _box(ov, TX, A_ROWS[3], TW, A_BH, "CopyKAT, CalicoST,\nClonalscope, Cancer-Finder", pale[SP], ec=SP)
    for m in mid[1:]:
        _arrow(ov, MX + 14, m, TX, m)
    # the chain from DNA into RNA (c) and the lineage analysis (d)
    _box(ov, 84, A_ROWS[1], 18, A_BH, "DNA clones\nand profiles", "white", ec=DNA)
    _arrow(ov, TX + TW, mid[1], 84, mid[1])
    _box(ov, 106, A_ROWS[2], 17, A_BH, "clonealign", pale[RNA], ec=RNA)
    _arrow(ov, 102, mid[1] + 1.2, 106, mid[2] - 0.6, color=DNA)
    _box(ov, 95, A_ROWS[0], 28, A_BH, "lineage and space\nin the hypoxia field", "white", ec=S.GREY)
    for x, m, lab in ((TX + TW + 1.2, mid[2], "C"), (TX + TW + 1.2, mid[3], "C"), (124.2, mid[2], "B"),
                      (124.2, mid[0], "D")):
        ov.text(x, m, lab, fontsize=6, fontweight="bold", color=S.GREY, va="center")
    # the ground truth, scored against every link
    KX, KW = 133.0, 31.5
    _box(ov, KX, A_ROWS[0], KW, A_ROWS[3] + A_BH - A_ROWS[0], "", "#F4F4F4", ec=S.INK)
    ov.text(KX + KW / 2, A_ROWS[0] + 1.2, "ground truth", ha="center", va="top", fontsize=6, fontweight="bold")
    ov.text(KX + KW / 2, A_ROWS[0] + 4.6, "true clones and genealogy\ncopy number per cell\nand chromosome arm\n"
            "malignant or normal per cell\nclone fractions per spot\nniche of every cell\nfields of the environment",
            ha="center", va="top", fontsize=SMALL, linespacing=1.3, color=S.INK)
    dot = dict(color=S.INK, lw=0.35, ls=(0, (1.5, 1.5)))
    for x0, m in ((129.5, mid[0]), (102.0, mid[1]), (127.0, mid[2]), (TX + TW + 3.5, mid[3])):
        ov.plot([x0, KX], [m, m], **dot)


def _numbers_b_all(df, nums):
    """Beside the drawn niches: each tool's scores over all its scored barcodes (niche ``all``: the mean, range
    and per-tumour values), and per tumour the scored barcodes it left undefined, which count as
    non-malignant calls (score_tools.malignancy_score)."""
    if not len(df):
        return
    al = df[df["niche"] == "all"]
    for assay, tool, _ in ROWS_B:
        sub = al[(al["assay"] == assay) & (al["tool"] == tool)]
        if not len(sub):
            continue
        rec = nums["b"].setdefault(f"{assay}:{tool}", {})
        for metric in ("auc", "cn_r"):
            v = sub.set_index("tumour")[metric].astype(float).dropna()
            if len(v):
                rec[f"all:{metric}"] = dict(mean=round(float(v.mean()), 3), min=round(float(v.min()), 3),
                                            max=round(float(v.max()), 3), n=int(len(v)),
                                            per_tumour={k: round(float(x), 3) for k, x in v.items()})
        und = sub[sub["n_undefined"].notna()] if "n_undefined" in sub else sub.iloc[:0]
        if len(und):
            rec["undefined"] = {r.tumour: dict(n_undefined=int(r.n_undefined),
                                               n_undefined_malignant=int(r.n_undefined_malignant),
                                               n_scored=int(r.n_scored), n_malignant=int(r.n_malignant),
                                               share=round(float(r.n_undefined) / max(int(r.n_scored), 1), 4))
                                for r in und.itertuples()}


def panel_b(fig, ov, nums):
    S.panel_letter(fig, "c", 59.5, 30.0)
    files = [os.path.join(DEV, "b", f"{RUN}_{t}", "scores.csv") for t in TUMOURS]
    have = [pd.read_csv(f).assign(tumour=os.path.basename(os.path.dirname(f))) for f in files if os.path.exists(f)]
    df = pd.concat(have, ignore_index=True) if have else pd.DataFrame()
    nums["b"] = {"tumours": sorted(set(df["tumour"])) if len(df) else []}
    _numbers_b_all(df, nums)
    n = len(ROWS_B)
    # the AUC axis reaches below the lowest niche AUC (0.403) and past 1, so no dot is cut at an edge
    for col, (metric, xlab, lim, ticks) in enumerate(
            (("auc", "Malignancy AUC", (0.35, 1.02), (0.4, 0.6, 0.8, 1.0)),
             ("cn_r", "Copy number correlation\nper chromosome arm", (-0.2, 1.0), (0.0, 0.5, 1.0)))):
        ax = S.ax_mm(fig, B_X[col], B_TOP, B_W, B_H)
        for i, (assay, tool, lab) in enumerate(ROWS_B):
            y = n - 1 - i
            sub = df[(df["assay"] == assay) & (df["tool"] == tool)] if len(df) else df
            vals = sub[sub["niche"].isin(NICHE)][metric] if len(sub) else pd.Series(dtype=float)
            if not len(sub) or not np.isfinite(vals.astype(float)).any():
                failed = len(sub) and "status" in sub and sub["status"].astype(str).str.startswith("failed").all()
                note = "pending" if not len(sub) else ("failed in every tumor" if failed else "no call")
                ax.text(lim[0] + 0.02 * (lim[1] - lim[0]), y, note, fontsize=SMALL, color=S.GREY, va="center")
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
        ax.set_yticklabels([lab for _, _, lab in ROWS_B][::-1], fontsize=SMALL)
        ax.set_ylim(-0.6, n - 0.4)
        ax.set_xlim(*lim)
        ax.set_xticks(ticks)
        ax.set_xlabel(xlab)
        ax.axvline(0.5 if metric == "auc" else 0.0, color=S.INK, lw=0.4, ls=":")
        if col == 0:
            # the modality of each block of rows, left of the tool names
            tr = ax.get_yaxis_transform()
            x = -25.4 / B_W                    # the modality, left of the tools' names
            ax.text(x, 6.5, "scRNA", transform=tr, rotation=90, ha="center", va="center", fontsize=6,
                    color=S.MODALITY["RNA"], fontweight="bold")
            ax.text(x, 2.0, "Visium", transform=tr, rotation=90, ha="center", va="center", fontsize=6,
                    color=S.MODALITY["spatial"], fontweight="bold")
        else:
            ax.set_yticklabels([])
    h = [Line2D([], [], marker="o", ls="none", mfc=c, mec="white", ms=3.4, label=l) for l, c in NICHE.values()]
    # the means are short vertical ticks in each niche's colour, so the key is a tick too
    h.append(Line2D([], [], marker="|", ls="none", color="#6F6F6F", ms=4.5, mew=1.2, label="mean over tumors"))
    fig.legend(handles=h, loc="upper left", bbox_to_anchor=(B_X[0] / W_MM, 1 - 30.2 / FIG_H), ncol=4,
               fontsize=SMALL, frameon=False, borderaxespad=0, handletextpad=0.3, columnspacing=1.2)


def panel_c(fig, ov, c, nums):
    S.panel_letter(fig, "b", 0.5, 30.0)
    ax = S.ax_mm(fig, C_X, C_TOP, C_W, C_H)
    DNA = S.MODALITY["DNA"]
    nums["c"] = {}
    for i, task in enumerate(TUMOURS):
        r = c[c["tumour"] == f"{RUN}_{task}"]
        if not len(r):
            continue
        r = r.iloc[0]
        ax.plot([i - 0.32, i + 0.32], [r.chance] * 2, color=S.GREY, lw=0.6, zorder=1)
        ax.plot(i + 0.2, r.bal_corr, marker="D", ms=3.0, mfc="#BDBDBD", mec="white", mew=0.3, ls="none", zorder=2)
        ax.plot(i - 0.14, r.bal_oracle_set, "o", ms=3.4, mfc="white", mec=S.INK, mew=0.6, zorder=2)
        if np.isfinite(r.get("bal_proportional", np.nan)):     # the true clones, every gene proportional
            ax.plot(i + 0.12, r.bal_proportional, "s", ms=3.2, mfc="white", mec=S.INK, mew=0.6, zorder=2)
        ax.plot(i, r.bal_true, "o", ms=4, mfc=S.INK, mec="white", mew=0.4, zorder=3)
        ax.plot(i, r.bal_dna, "o", ms=4, mfc=DNA, mec="white", mew=0.4, zorder=3)
        if bool(r.above_chance):
            # the two values are drawn side by side; the line joins them, without printing their difference
            ax.plot([i, i], [r.bal_dna, r.bal_true], color=S.GREY, lw=0.6, zorder=1)
        else:
            ax.text(i, 0.99, "true clones\nat chance", fontsize=SMALL, color=S.GREY, ha="center", va="top",
                    linespacing=1.05)
        nums["c"][task] = {k: (round(float(r[k]), 3) if isinstance(r[k], (float, np.floating)) else r[k])
                           for k in ("bal_true", "bal_dna", "bal_oracle_set", "bal_proportional", "bal_corr", "chance",
                                     "above_chance",
                                     "acc_true", "acc_dna", "majority", "n_true_clones", "n_called_clones",
                                     "hmmcopy_concordance_scaled")}
    tumour_axis(ax, TUMOURS, C_H)
    ax.set_xlim(-0.6, len(TUMOURS) - 0.4)
    ax.set_ylim(0, 1.02)
    ax.set_yticks((0, 0.5, 1.0))
    ax.set_ylabel("Balanced accuracy")
    # two columns, filled column by column: the three short labels, then the two long ones
    h = [Line2D([], [], marker="o", ls="none", mfc=S.INK, mec="white", ms=4, label="true clones"),
         Line2D([], [], marker="o", ls="none", mfc=DNA, mec="white", ms=4, label="clones from scDNA"),
         Line2D([], [], color=S.GREY, lw=0.6, label="chance"),
         Line2D([], [], marker="o", ls="none", mfc="white", mec=S.INK, mew=0.6, ms=3.4,
                label="true profiles of scDNA clones"),
         Line2D([], [], marker="D", ls="none", mfc="#BDBDBD", mec="white", ms=3.0,
                label="best-correlated true profile"),
         Line2D([], [], marker="s", ls="none", mfc="white", mec=S.INK, mew=0.6, ms=3.2,
                label="true clones, proportional dosage")]
    fig.legend(handles=h, loc="upper left", bbox_to_anchor=(4.0 / W_MM, 1 - 30.2 / FIG_H), fontsize=SMALL,
               frameon=False, ncol=2, borderaxespad=0, columnspacing=0.8, handletextpad=0.3)


def _rs_frame(df):
    """The row-standardised statistics of each cached tumour as columns (caches from 2026-11-26 on)."""
    if "rs" not in df:
        return df.iloc[0:0]
    df = df[df["rs"].map(lambda r: isinstance(r, dict))].copy()
    for k in ("field_lin", "field_sp", "control_lin", "control_sp", "field_perm_p", "control_perm_p"):
        df[f"{k}_rs"] = df["rs"].map(lambda r, k=k: r[k])
    return df


PETRACER = os.environ.get("FIG4_PETRACER") or os.path.join(DEV, "petracer", "summary.json")


def panel_de(fig, ov, nums):
    """(d) Lineage autocorrelation is no evidence of heritability in spatially structured tumours. Left:
    per architecture and selection, the lineage and spatial autocorrelation (row-standardised Moran's I,
    ``fig4_lineage``) of the hypoxia field, environmental by construction, and of a trait heritable by
    construction (Brownian motion along the true genotype tree), one marker per tumour, the line the mean
    of the tumours that reached their size. Middle: the same in a real tumour, the 4T1 tumour traced with
    PEtracer (``fig4_petracer.py``): the distance to the tumour and to the lung boundary, which no cell
    inherits, and a trait simulated along its real tree, one marker per section. Right: the share of
    tumours each test calls heritable, a permutation test of the lineage autocorrelation alone and the
    comparison of the lineage with the spatial autocorrelation, for the field and for the control."""
    import fig4_lineage as FL
    df = _rs_frame(FL.load(DEV))
    S.panel_letter(fig, "d", 0.5, R3_LETTER)
    if not len(df):
        nums["lineage"] = {}
        return
    ax = S.ax_mm(fig, 11.0, R3_TOP, 74.0, R3_H)
    groups = [("R1", "ductal", None, "dispersal\nsweep"), ("R3", "ductal", "selected", "selected"),
              ("R3", "ductal", "neutral", "neutral"), ("R3", "free", "selected", "selected"),
              ("R3", "free", "neutral", "neutral")]
    MEASURE = (("field_lin_rs", "o", -0.27, True), ("field_sp_rs", "^", -0.09, True),
               ("control_lin_rs", "o", 0.09, False), ("control_sp_rs", "^", 0.27, False))
    for k, (run, arch, sel, _) in enumerate(groups):
        sub = df[(df.run == run) & (df.arch == arch) & (df.selection.astype(str) == str(sel))]
        st = sub["stalled"].astype(bool).to_numpy() if "stalled" in sub else np.zeros(len(sub), bool)
        for m, (key, mk, off, field) in enumerate(MEASURE):
            x = k + off + np.random.default_rng(10 * k + m).uniform(-0.05, 0.05, len(sub))
            v = sub[key].to_numpy(dtype=float)
            if field:
                ax.scatter(x, v, s=4, marker=mk, color=ARCH[arch][1], edgecolors="none", zorder=3,
                           alpha=0.9 if sel != "neutral" else 0.55)
            else:
                ax.scatter(x, v, s=4.5, marker=mk, facecolors="white", edgecolors=S.GREY, linewidths=0.5,
                           zorder=3)
            if (~st).any():
                ax.plot([k + off - 0.08, k + off + 0.08], [float(np.nanmean(v[~st]))] * 2, color=S.INK, lw=0.9,
                        zorder=4)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[3] for g in groups], fontsize=SMALL)
    tr = ax.get_xaxis_transform()
    y = -6.5 / R3_H
    for lab, idx in ((ARCH["ductal"][0], (0, 2)), (ARCH["free"][0], (3, 4))):
        ax.plot([idx[0] - 0.35, idx[1] + 0.35], [y, y], transform=tr, color=S.GREY, lw=0.4, clip_on=False)
        ax.text(sum(idx) / 2, y - 0.4 / R3_H, lab, transform=tr, ha="center", va="top", fontsize=SMALL)
    ax.set_xlim(-0.55, len(groups) - 0.45)
    ax.set_ylim(-0.1, 1.05)
    ax.set_yticks([0, 0.5, 1])
    ax.set_ylabel("Autocorrelation")
    h = [Line2D([], [], marker="o", ls="none", mfc="#6F6F6F", mec="none", ms=3.2, label="lineage"),
         Line2D([], [], marker="^", ls="none", mfc="#6F6F6F", mec="none", ms=3.2, label="spatial"),
         Line2D([], [], marker="s", ls="none", mfc="#6F6F6F", mec="none", ms=3.2, label="hypoxia field"),
         Line2D([], [], marker="s", ls="none", mfc="white", mec=S.GREY, mew=0.6, ms=3.2, label="heritable trait"),
         Line2D([], [], color=S.INK, lw=0.9, label="mean")]
    ax.legend(handles=h, fontsize=SMALL, loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=5, frameon=False,
              borderaxespad=0, handletextpad=0.2, columnspacing=0.8)
    # the real tumour
    axr = S.ax_mm(fig, 99.0, R3_TOP, 27.0, R3_H)
    axr.set_title("real tumor", fontsize=SMALL, pad=2)
    real = json.load(open(PETRACER)) if os.path.exists(PETRACER) else None
    cats = (("tumor_boundary_dist", "tumor\nboundary", True), ("lung_boundary_dist", "lung\nboundary", True),
            ("heritable_control", "heritable\ntrait", False))
    nums["petracer"] = {}
    if real:
        for ci, (key, _, field) in enumerate(cats):
            for si, (sec, rec) in enumerate(real["sections"].items()):
                f = rec["fields"][key]
                for off, val, mk in ((-0.14, f["lin_rs"], "o"), (0.14, f["sp_rs"], "^")):
                    if field:
                        axr.scatter([ci + off + (si - 1) * 0.04], [val], s=5, marker=mk, color=S.INK,
                                    edgecolors="none", zorder=3)
                    else:
                        axr.scatter([ci + off + (si - 1) * 0.04], [val], s=5.5, marker=mk, facecolors="white",
                                    edgecolors=S.GREY, linewidths=0.5, zorder=3)
            nums["petracer"][key] = {sec: rec["fields"][key] for sec, rec in real["sections"].items()}
    else:
        axr.text(0.5, 0.5, "pending", ha="center", va="center", transform=axr.transAxes, fontsize=SMALL,
                 color=S.GREY)
    axr.set_xticks(range(len(cats)))
    axr.set_xticklabels([c[1] for c in cats], fontsize=SMALL)
    axr.set_xlim(-0.5, len(cats) - 0.5)
    axr.set_ylim(-0.1, 1.05)
    axr.set_yticks([0, 0.5, 1])
    axr.set_yticklabels([])
    # what each test concludes, for the field and for the control
    ax2 = S.ax_mm(fig, 137.0, R3_TOP, 26.0, R3_H)
    arch_groups = [("ductal", "ductal\nfield"), ("free", "without\nglands")]
    bars = (("field_perm_p_rs", "perm", True, S.INK, "permutation test"),
            ("control_perm_p_rs", "perm", False, "white", None),
            ("field", "compare", True, "#BDBDBD", "lineage above spatial"),
            ("control", "compare", False, "white", None))
    called = {}
    for gi, (arch, _) in enumerate(arch_groups):
        sub = df[df.arch == arch]
        if not len(sub):
            continue
        n = len(sub)
        rec = {}
        for bi, (key, test, field, col, _) in enumerate(bars):
            if test == "perm":
                c = int((sub[key] < 0.05).sum())
            else:
                c = int((sub[f"{key}_lin_rs"] > sub[f"{key}_sp_rs"]).sum())
            rec[f"{'field' if field else 'control'}_{test}"] = c
            x = gi + (bi - 1.5) * 0.2
            ax2.bar(x, c / n, width=0.18, color=col, edgecolor=S.INK if col == "white" else col, lw=0.5)
            ax2.text(x, c / n + 0.02, f"{c}/{n}", ha="center", va="bottom", fontsize=SMALL - 0.5, rotation=90)
        called[arch] = dict(n=n, **rec)
    ax2.set_xticks(range(len(arch_groups)))
    ax2.set_xticklabels([lab for _, lab in arch_groups], fontsize=SMALL)
    ax2.set_xlim(-0.55, len(arch_groups) - 0.45)
    ax2.set_ylim(0, 1.32)
    ax2.set_yticks([0, 0.5, 1])
    ax2.set_ylabel("Called heritable", labelpad=1)
    ax2.legend(handles=[Line2D([], [], marker="s", ls="none", mfc=col, mec=col, ms=3.2, label=lab)
                        for _, _, _, col, lab in bars if lab]
               + [Line2D([], [], marker="s", ls="none", mfc="white", mec=S.INK, mew=0.5, ms=3.2,
                         label="open: heritable trait")],
               fontsize=SMALL, loc="lower left", bbox_to_anchor=(-0.35, 1.01), frameon=False, borderaxespad=0,
               handletextpad=0.2, labelspacing=0.15)
    nums["lineage"] = FL.summary(DEV)
    nums["lineage"]["called_heritable"] = called


# beside the Fig 4 outputs unless given: <parent of $FIG4_DEV>/cn-trees/trees.csv
TREES = os.environ.get("FIG4_TREES") or os.path.join(os.path.dirname(os.path.normpath(DEV)), "cn-trees", "trees.csv")


def panel_trees(fig, nums):
    """(e) copy-number tree methods against the true genealogy, confined against free growth, from the
    true and the called copy numbers (``trees.csv`` of exp-paper1-cn-trees-architecture: one row per
    tumour, method and input, with ``nrf``). A placeholder until the file exists."""
    S.panel_letter(fig, "e", 0.5, R4_LETTER)
    ax = S.ax_mm(fig, 11.0, R4_TOP, 152.0, R4_H)
    if not os.path.exists(TREES):
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color(S.BG_CELLS)
        ax.text(0.5, 0.5, "pending: copy-number trees against the true genealogy, confined against free growth,\n"
                "from true and called copy number (Harald, Marzie; exp-paper1-cn-trees-architecture)",
                ha="center", va="center", fontsize=SMALL, color=S.GREY, transform=ax.transAxes)
        nums["trees"] = None
        return
    df = pd.read_csv(TREES)
    groups = [(a, sel) for a in ("confined", "free") for sel in ("selected", "neutral")]
    methods = sorted(df.method.unique())
    inputs = [i for i in ("true", "called") if i in set(df.input)]
    width = 0.8 / max(1, len(methods) * len(inputs))
    for gi, (a, sel) in enumerate(groups):
        for mi, m in enumerate(methods):
            for ii, inp in enumerate(inputs):
                v = df[(df.arch == a) & (df.selection == sel) & (df.method == m) & (df.input == inp)].nrf
                x = gi - 0.4 + width * (mi * len(inputs) + ii + 0.5)
                ax.scatter(np.full(len(v), x), v, s=4, color=S.CLONE_FAMILIES[mi][1],
                           alpha=1.0 if inp == "true" else 0.5, linewidths=0, zorder=3)
                if len(v):
                    ax.plot([x - width * 0.4, x + width * 0.4], [v.mean()] * 2, color=S.INK, lw=0.9, zorder=4)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{sel}\n{'ductal field' if a == 'confined' else 'without glands'}" for a, sel in groups],
                       fontsize=SMALL)
    ax.set_ylabel("Tree distance to the truth")
    nums["trees"] = df.groupby(["arch", "selection", "method", "input"]).nrf.mean().round(3).to_dict()


def main():
    import fig4_score as FS
    S.setup()
    fig = S.figure(W_MM, FIG_H)
    ov = _overlay(fig)
    nums = {"sources": dict(fig4_out=OUT, diag=DIAG, dev=DEV, tumours=TUMOURS,
                            chance_margin=FS.CHANCE_MARGIN)}
    panel_a(fig, ov)
    panel_b(fig, ov, nums)
    c, _ = FS.collect(OUT, DIAG)
    c = c[c["resolution"] == "fine"] if len(c) else c
    panel_c(fig, ov, c, nums)
    panel_de(fig, ov, nums)
    panel_trees(fig, nums)
    path = S.save(fig, "fig4_benchmark")
    os.makedirs(DEV, exist_ok=True)
    with open(os.path.join(DEV, "fig4_benchmark_numbers.json"), "w") as f:
        json.dump(nums, f, indent=1, default=str)
    print("wrote", path, "and", os.path.join(DEV, "fig4_benchmark_numbers.json"))


if __name__ == "__main__":
    main()
