"""House style for the paper-1 figures: one place for sizes, fonts, colours and export.

The spec is measured from Pe'er-lab Nature Biotechnology figures (Spectra, SEACells, ENVI) and the
LN's t-test Figure 1; the reasoning is in the research vault's ``roadmap/paper1-style-guide``. Every
``fig*.py`` script calls :func:`setup` first and saves through :func:`save`, so a style change is
made here once and every figure follows.

Two rules matter more than any single value:

* **One colour per entity for the whole paper.** A tool, a modality or a data source keeps its
  colour in every panel of every figure (Spectra's own figures drift, which is the mistake to avoid).
  Look colours up in :data:`TOOL`, :data:`MODALITY` and :data:`SOURCE`; never pick one inline.
* **Fixed physical width.** Figures are exported as PDF at 89 or 183 mm and never with
  ``bbox_inches="tight"``, which silently changes the width and therefore every font size.

``ISCC_PAPER_TIER`` (``preview`` by default, or ``full``) says which simulation scale produced a
figure, and :func:`save` stamps preview figures so they cannot be mistaken for final ones.
"""
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paths import figure_path  # noqa: E402

TIER = os.environ.get("ISCC_PAPER_TIER", "preview")
GRID = False                       # the Pe'er figures use none; the LN's t-test figure uses light grids

MM = 1 / 25.4
WIDTH_MM = {"single": 89, "onehalf": 136, "double": 183}
MAX_HEIGHT_MM = 247

# --- colours -------------------------------------------------------------------------------------
INK = "#1A1A1A"                    # text, truth
GREY = "#818181"                   # secondary text, schematic captions
BG_CELLS = "#D9D9D9"               # context cells, background
TISSUE = "#E3E1DC"                 # normal tissue drawn as background
PASTEL = {"blue": "#C9E1F3", "green": "#C3E7C3", "yellow": "#EDEDAB", "grey": "#EDEDED"}

SOURCE = {"iscc": "#4C72B0", "real": "#9E9E9E", "truth": INK}
MODALITY = {"DNA": "#756BB1", "RNA": "#E6550D", "spatial": "#31A354"}
COMPARTMENT = {"lumen": "#FFFFFF", "epithelium": "#8F7BAE", "stroma": "#F1E9DD", "met": "#C9DDEB"}

# One colour per third-party tool, fixed for the whole paper (seaborn "deep" order after iscc's blue).
TOOL = {
    "clonealign": "#DD8452", "inferCNV": "#55A868", "Numbat": "#C44E52", "scDEF": "#8172B3",
    "cNMF": "#937860", "HMMcopy": "#DA8BC3", "SCITE": "#CCB974", "neighbour-joining": "#8C8C8C",
    "MEDICC2": "#64B5CD", "Harmony": "#2F6D3A", "scVI": "#B07AA1", "MHN": "#A6761D",
    "TreeMHN": "#E7298A", "RCTD": "#1B9E77", "cell2location": "#7570B3", "PyClone-VI": "#66A61E",
}

# Clone colours: one hue family per founding lineage, subclones from dark to light.
CLONE_FAMILIES = [
    ["#08519C", "#3182BD", "#6BAED6", "#BDD7E7"],
    ["#006D2C", "#31A354", "#74C476", "#BAE4B3"],
    ["#A63603", "#E6550D", "#FD8D3C", "#FDBE85"],
    ["#54278F", "#756BB1", "#9E9AC8", "#CBC9E2"],
    ["#7A0177", "#C51B8A", "#F768A1", "#FBB4B9"],
    ["#4D4D4D", "#878787", "#BABABA", "#E0E0E0"],
]

DIVERGING = LinearSegmentedColormap.from_list(
    "iscc_div", ["#2F479E", "#7485B1", "#BCC1CE", "#FCF7F7", "#E3B2AC", "#C66263", "#A81B20"])
SEQUENTIAL = "viridis"
EXPRESSION = "afmhot_r"


def cn_norm(vmax=6):
    """Copy-number colour scale: white at 2, never below 0 (blue loss, red gain)."""
    return TwoSlopeNorm(vcenter=2.0, vmin=0.0, vmax=float(vmax))


# --- setup and layout ----------------------------------------------------------------------------
def setup():
    """Apply the house rcParams. Call once at the top of every figure script."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": 6.5, "axes.titlesize": 6.5, "axes.titleweight": "normal", "axes.titlepad": 3,
        "axes.labelsize": 6.5, "axes.labelpad": 2,
        "xtick.labelsize": 5.5, "ytick.labelsize": 5.5,
        "legend.fontsize": 5.5, "legend.title_fontsize": 6.5, "legend.frameon": False,
        "legend.handlelength": 1.0, "legend.handletextpad": 0.4, "legend.labelspacing": 0.3,
        "legend.columnspacing": 0.8, "legend.borderaxespad": 0.2,
        "axes.linewidth": 0.5, "axes.edgecolor": INK, "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": GRID, "grid.color": "#E5E5E5", "grid.linewidth": 0.4,
        "axes.prop_cycle": cycler(color=list(TOOL.values())),
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 2.0, "ytick.major.size": 2.0,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.minor.size": 1.2, "ytick.minor.size": 1.2,
        "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
        "lines.linewidth": 1.0, "lines.markersize": 2.5, "lines.markeredgewidth": 0.4,
        "patch.linewidth": 0.5, "scatter.edgecolors": "none", "image.cmap": SEQUENTIAL,
        "boxplot.medianprops.color": INK, "boxplot.flierprops.markersize": 1.5,
        "mathtext.default": "regular", "axes.unicode_minus": True,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.dpi": 300,
    })


def figure(width="double", height_mm=120):
    """A figure of exact physical size; ``width`` is "single", "onehalf", "double" or mm."""
    w = WIDTH_MM.get(width, width)
    return plt.figure(figsize=(w * MM, min(height_mm, MAX_HEIGHT_MM) * MM))


def ax_mm(fig, x, y, w, h, **kw):
    """Axes at an exact mm box, measured from the figure's top-left corner."""
    W, H = fig.get_size_inches() / MM
    return fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H], **kw)


def panel_letter(fig, letter, x_mm, y_mm, size=8):
    """A bold lowercase panel letter; reuse one x per column and one y per row."""
    W, H = fig.get_size_inches() / MM
    fig.text(x_mm / W, 1 - y_mm / H, letter, fontsize=size, fontweight="bold",
             ha="left", va="top", color=INK)


def text_mm(fig, x_mm, y_mm, s, **kw):
    """Figure-level text at a mm position from the top-left corner."""
    W, H = fig.get_size_inches() / MM
    kw.setdefault("color", INK)
    return fig.text(x_mm / W, 1 - y_mm / H, s, **kw)


def despine(ax, left=True, bottom=True):
    """Keep only the requested spines; embeddings and tissue maps use ``despine(ax, False, False)``."""
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)
    if not (left or bottom):
        ax.set_xticks([])
        ax.set_yticks([])


def stat(ax, s, loc="lower right"):
    """Print a statistic inside the axes ("r = 0.69"), the house alternative to a legend."""
    xy = {"lower right": (0.97, 0.04, "right", "bottom"), "upper left": (0.03, 0.97, "left", "top"),
          "upper right": (0.97, 0.97, "right", "top"), "lower left": (0.03, 0.04, "left", "bottom")}[loc]
    ax.text(xy[0], xy[1], s, transform=ax.transAxes, ha=xy[2], va=xy[3], fontsize=5.5, color=INK)


def clone_colours(clones, parents, founder):
    """A colour per clone: one hue family per lineage founded by a child of ``founder``, darker
    for the lineage founder and lighter with depth. ``clones`` is the list to colour (e.g. the shown
    clones after merging small ones); ``parents`` maps clone -> parent over the full genealogy."""
    def lineage_and_depth(c):
        depth, cur = 0, c
        while cur in parents and parents[cur] not in (None, founder) and cur != founder:
            cur, depth = parents[cur], depth + 1
        return cur, depth
    roots, out = {}, {}
    for c in clones:
        if c == founder:
            out[c] = CLONE_FAMILIES[-1][1]
            continue
        root, depth = lineage_and_depth(c)
        fam = CLONE_FAMILIES[roots.setdefault(root, len(roots) % (len(CLONE_FAMILIES) - 1))]
        out[c] = fam[min(depth, len(fam) - 1)]
    return out


def save(fig, name, stamp=True):
    """Write ``figures/paper1/<name>.pdf`` (vector, fixed width) plus a PNG preview beside it.

    Preview-tier figures carry a small grey stamp in the bottom-right corner."""
    if stamp and TIER != "full":
        fig.text(0.995, 0.004, "preview at local scale; regenerate at full scale", ha="right", va="bottom",
                 fontsize=4.5, color=GREY)
    path = figure_path(os.path.join("paper1", f"{name}.pdf"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    fig.savefig(path[:-4] + ".png", dpi=200)
    plt.close(fig)
    return path
