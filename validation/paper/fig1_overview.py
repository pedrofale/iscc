"""Figure 1: what iscc is, and where it sits.

(a) The pipeline on one tumour, as an Input -> iscc -> Output banner with real thumbnails of the
showcase tumour (registry run R0): the grown ductal field coloured by clone, where it is sampled, and
what each assay returns, all coloured by the same clones; the answer key that travels with the data.
(b) The positioning: the evolution front and the data front, the partial bridges, and the join.

Run:  python validation/paper/fig1_overview.py      (reads R0/seed1; writes figures/paper1/fig1)
"""
import os
import sys

import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402
import registry as REG  # noqa: E402
import style as S  # noqa: E402

RUN, TASK = "R0", "seed1"
BANNER = {"Input": "#8C8C8C", "iscc": S.SOURCE["iscc"], "Output": "#5E8C61"}


def box(fig, x, y, w, h, fc, ec="#9A9A9A", lw=0.4, r=1.2):
    """A flat rounded box at a mm position (top-left origin)."""
    W, H = fig.get_size_inches() / S.MM
    p = FancyBboxPatch((x / W, 1 - (y + h) / H), w / W, h / H, transform=fig.transFigure,
                       boxstyle=f"round,pad=0,rounding_size={r / W}", fc=fc, ec=ec, lw=lw)
    fig.add_artist(p)
    return p


def arrow(fig, x0, y0, x1, y1, color="#6F6F6F", lw=0.6):
    W, H = fig.get_size_inches() / S.MM
    fig.add_artist(FancyArrowPatch((x0 / W, 1 - y0 / H), (x1 / W, 1 - y1 / H),
                                   transform=fig.transFigure, arrowstyle="-|>", mutation_scale=5,
                                   color=color, lw=lw, shrinkA=0, shrinkB=0))


def banner(fig, x, y, w, label):
    box(fig, x, y, w, 4.2, BANNER[label], ec="none", r=0.8)
    S.text_mm(fig, x + w / 2, y + 2.1, label, ha="center", va="center", color="white",
              fontsize=7, fontweight="bold")


def umap_coords(counts, seed=0):
    import scanpy as sc
    import anndata as ad
    a = ad.AnnData(counts.values.astype(np.float32))
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, n_top_genes=1500)
    a = a[:, a.var.highly_variable].copy()
    sc.pp.pca(a, n_comps=30, random_state=seed)
    sc.pp.neighbors(a, random_state=seed)
    sc.tl.umap(a, random_state=seed)
    return a.obsm["X_umap"]


def main():
    S.setup()
    t = REG.load(RUN, TASK)
    cl = C.Clones(t, min_frac=0.02)          # the overview shows finer clones than the sweeps
    A = C.assays(RUN, TASK)
    lab = {c: (cl.of(A["genotype"][c]) if A["is_cancer"][c] else A["cell_type"][c]) for c in A["cells"]}
    normal_colour = {"stromal": "#CDBFA8", "epithelial": S.COMPARTMENT["epithelium"], "immune": "#9E9E9E"}

    fig = S.figure("double", 106)
    S.panel_letter(fig, "a", 1, 1)

    # --- banners ---
    banner(fig, 5, 2, 36, "Input")
    banner(fig, 46, 2, 78, "iscc")
    banner(fig, 129, 2, 53, "Output")

    # --- Input: what the user specifies ---
    items = [("Tissue", "ductal field: glands,\nepithelial wall, stroma"),
             ("Evolution", "selection on drivers,\ncopy number, traits"),
             ("Sampling and assays", "biopsy, dissociation, section;\nDNA, RNA, spatial, reads"),
             ("Niche", "hypoxia field, epithelial wall,\nimmune cells")]
    for i, (h, s) in enumerate(items):
        y = 9 + i * 12.5
        box(fig, 5, y, 36, 11, S.PASTEL["grey"])
        S.text_mm(fig, 7, y + 2.4, h, fontsize=6, fontweight="bold", va="center")
        S.text_mm(fig, 7, y + 6.6, s, fontsize=5, va="center", color=S.GREY, linespacing=1.15)
    arrow(fig, 42, 30, 45.5, 30)

    # --- iscc: grow, sample, assay on one tumour ---
    G = t.grid_size
    tm = C.tissue_map(t, cl)
    ax = S.ax_mm(fig, 48, 11, 24, 24)
    ax.imshow(tm, interpolation="nearest")
    S.despine(ax, False, False)
    S.text_mm(fig, 60, 37.5, "Grow", ha="center", fontsize=6.5, fontweight="bold")
    S.text_mm(fig, 60, 41.2, "a spatial tumour under\nselection, with its history", ha="center",
              fontsize=5, color=S.GREY, va="top", linespacing=1.15)
    arrow(fig, 73, 23, 76.5, 23)

    ax = S.ax_mm(fig, 78, 11, 24, 24)
    ax.imshow(tm, interpolation="nearest", alpha=0.55)
    half = G // 2
    ax.add_patch(Rectangle((-0.5, -0.5), half, G, fc="none", ec=S.MODALITY["DNA"], lw=0.7, ls="--"))
    ax.add_patch(Rectangle((half - 0.5, -0.5), G - half, G, fc="none", ec=S.MODALITY["spatial"], lw=0.7))
    S.despine(ax, False, False)
    S.text_mm(fig, 90, 37.5, "Sample", ha="center", fontsize=6.5, fontweight="bold")
    S.text_mm(fig, 90, 41.2, "dissociate one part,\nsection another", ha="center", fontsize=5,
              color=S.GREY, va="top", linespacing=1.15)
    arrow(fig, 103, 23, 106.5, 23)

    for i, (m, txt) in enumerate((("DNA", "bulk, single cell"), ("RNA", "single cell"),
                                  ("spatial", "Visium, imaging"))):
        y = 11 + i * 8.4
        box(fig, 108, y, 15, 7, "white", ec=S.MODALITY[m], lw=0.7)
        S.text_mm(fig, 115.5, y + 2.6, m, ha="center", fontsize=6, color=S.MODALITY[m], fontweight="bold")
        S.text_mm(fig, 115.5, y + 5.1, txt, ha="center", fontsize=4.5, color=S.GREY)
    S.text_mm(fig, 115.5, 37.5, "Assay", ha="center", fontsize=6.5, fontweight="bold")
    S.text_mm(fig, 115.5, 41.2, "counts or\naligned reads", ha="center", fontsize=5, color=S.GREY,
              va="top", linespacing=1.15)
    arrow(fig, 124.5, 23, 128, 23)

    # --- Output: real data, coloured by the same clones ---
    cells = A["dna_cells"]
    cov = A["dna_coverage"].values
    seg_ids = np.array([int(g.split("_")[1]) for g in A["dna_coverage"].columns])
    nseg = seg_ids.max() + 1
    segcov = np.stack([cov[:, seg_ids == s].mean(1) for s in range(nseg)], axis=1)
    ratio = segcov / np.median(segcov, axis=1, keepdims=True)
    order = np.argsort([cl.labels.index(lab[c]) if lab[c] in cl.labels else 99 for c in cells], kind="stable")
    ax = S.ax_mm(fig, 133, 10, 19, 22)
    calls = np.clip(np.rint(2 * ratio[order]), 0, 6)       # integer calls, as after a CN caller
    ax.imshow(calls, aspect="auto", cmap=S.DIVERGING, norm=S.cn_norm(6), interpolation="nearest")
    S.despine(ax, False, False)
    ax.set_title("scDNA: copy number", fontsize=5.5, pad=1.5)
    axb = S.ax_mm(fig, 130.6, 10, 1.6, 22)
    colours = [cl.colour.get(lab[cells[i]], S.BG_CELLS) for i in order]
    axb.imshow(np.array([C._hex(c) for c in colours])[:, None, :], aspect="auto", interpolation="nearest")
    S.despine(axb, False, False)

    xy = umap_coords(A["rna_counts"])
    ax = S.ax_mm(fig, 157, 10, 22, 22)
    cols = [cl.colour.get(lab[c], normal_colour.get(lab[c], S.BG_CELLS)) for c in A["cells"]]
    ax.scatter(xy[:, 0], xy[:, 1], s=0.25, c=cols, rasterized=True, linewidths=0)
    S.despine(ax, False, False)
    ax.set_title("scRNA: cells", fontsize=5.5, pad=1.5)

    ax = S.ax_mm(fig, 133, 35, 19, 19)
    sg = A["section_genotype"]
    spot_rgb, spot_alpha = [], []
    for members in A["visium_members"]:
        labs = [cl.of(sg[m]) for m in members if sg.get(m) in t.genotypes and t._is_cancer(sg[m])]
        frac = len(labs) / max(len(members), 1)
        if labs:
            top = max(set(labs), key=labs.count)
            spot_rgb.append(cl.colour[top])
        else:
            spot_rgb.append(S.COMPARTMENT["stroma"])
        spot_alpha.append(0.25 + 0.75 * frac)
    xyv = A["visium_coords"]
    ax.scatter(xyv[:, 1], -xyv[:, 0], s=0.9, c=spot_rgb, alpha=None, rasterized=True, linewidths=0)
    ax.set_aspect("equal")
    S.despine(ax, False, False)
    ax.set_title("Visium: spots", fontsize=5.5, pad=1.5)

    box(fig, 156, 35.5, 26, 19, S.PASTEL["blue"], ec="none")
    S.text_mm(fig, 158, 38, "Answer key", fontsize=6, fontweight="bold", va="center")
    keys = ("per cell and read: clone,\nlineage, genotype, allele-\nspecific copy number,\n"
            "niche, program activity;\nper clone: where and\nwhen it arose")
    S.text_mm(fig, 158, 40.5, keys, fontsize=4.8, va="top", linespacing=1.2, color=S.INK)

    # clone legend under the output
    for i, l in enumerate(cl.labels):
        x = 130 + (i % 4) * 13
        y = 57.5 + (i // 4) * 3
        box(fig, x, y - 0.9, 1.8, 1.8, cl.colour[l], ec="none", r=0.3)
        S.text_mm(fig, x + 2.6, y, l, fontsize=4.8, va="center")

    # --- b: positioning ---
    S.panel_letter(fig, "b", 1, 66)
    y0, h = 69, 21
    fronts = [(5, 62, S.PASTEL["yellow"], "Evolution front: model the process",
               "CINner, demon, HAL, PhysiCell; CellCoal, CNAsim, SISTEM",
               "selection, spatial growth and phylogenies;\nlittle or no molecular data (DNA at most)"),
              (120, 62, S.PASTEL["green"], "Data front: reproduce the measurements",
               "Splatter, scDesign2/3, SRTsim, scCube",
               "realistic counts and tissue layouts;\ncells without a genome, tissue drawn")]
    for x, w, fc, title, tools, gist in fronts:
        box(fig, x, y0, w, h, fc, ec="none")
        S.text_mm(fig, x + 2, y0 + 3, title, fontsize=6.5, fontweight="bold", va="center")
        S.text_mm(fig, x + 2, y0 + 8, tools, fontsize=5.5, va="center")
        S.text_mm(fig, x + 2, y0 + 12.5, gist, fontsize=5, va="top", color=S.GREY, linespacing=1.25)
    box(fig, 70, y0, 47, h, "white", ec="#BDBDBD", lw=0.4)
    S.text_mm(fig, 93.5, y0 + 3, "Partial bridges", fontsize=6, fontweight="bold", ha="center", va="center")
    S.text_mm(fig, 93.5, y0 + 6.5, "stCNASim: planted clones, real reads, no genealogy\n"
              "scMultiSim: several modalities from an input tree\n"
              "J-SPACE: spatial evolution into DNA reads",
              fontsize=4.9, ha="center", va="top", linespacing=1.35)
    box(fig, 55, 94, 77, 10.5, S.SOURCE["iscc"], ec="none")
    S.text_mm(fig, 93.5, 97, "iscc: the join", fontsize=7, fontweight="bold", ha="center", va="center",
              color="white")
    S.text_mm(fig, 93.5, 101.3, "one grown tumour into DNA, RNA and spatial data, layers coupled, full answer key",
              fontsize=5.2, ha="center", va="center", color="white")
    arrow(fig, 36, y0 + h + 0.5, 56, 99)
    arrow(fig, 151, y0 + h + 0.5, 131.5, 99)

    path = S.save(fig, "fig1_overview")
    print("wrote", path)


if __name__ == "__main__":
    main()
