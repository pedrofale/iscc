"""Presentation figures for the four modes of therapeutic escape (validate_escape_modes.py).

Drawn from the SAME four runs as the paper figure -- the same PANELS and the same panel cache -- for
slides rather than the page:

  slides/escape_modes_summary.{png,pdf}   all four modes on one log axis, sensitive vs resistant
  slides/escape_mode_<K>_growth.png       one per mode: metastasis sections over time, above the
                                          met's Muller plot of clone frequencies

They go to figures/slides/ in the paper repository: slide material, kept apart from the figures
paper.tex includes.

The summary uses LOG y with sensitive and resistant as separate lines, where the paper figure stacks
them on a linear axis. On the linear axis the pre-therapy composition (what separates I from II)
and the birth of the relapsing lineage (what separates III from IV) are flattened against zero.
The growth figures' Muller is in clone FREQUENCIES for the same reason: the deposit at the first
dose is ~3% of its final size, so a count Muller shows only the relapse. Size is carried by the
sections instead, each drawn on one shared crop and labelled with its cell count.

The growth figures need each panel's growth record (validate_escape_modes._growth_record). A panel
cached without one is re-run for it -- ~6-10 min each -- and the re-run is checked against the
cached curves before anything is drawn from it.

Usage:  python validation/escape_modes_slides.py [--only I,IV] [--cache DIR]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, Patch, Rectangle
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator, PercentFormatter
from matplotlib.transforms import offset_copy
from scipy.ndimage import gaussian_filter1d

import validate_escape_modes as VEM
from iscc.constants import normal_names
from iscc.tumor import viz
from _paths import figure_path

# make_analysis_data.py's copy of the panels: git-ignored, and where they already are on most
# machines. validate_escape_modes.py's own default (validation/.cache_escape_modes) is not ignored.
CACHE = os.path.join(VEM.REPO, "analysis_data", "escape_modes")

# The paper's red; the paper's blue (#4c78a8) raised in chroma, which it fails as a slide colour.
# The pair passes the dataviz palette checks: CVD dE 16.6 (protan), normal-vision 26.0, >= 3:1.
SENS, RES = "#3a6fb5", "#c1344e"
BAND = "#2f6d3a"
TISSUE = "#e3e1dc"
INK, INK2, MUTED, RULE = "#1a1a1a", "#4a4a4a", "#7a7a7a", "#d9d9d6"

# (title, outcome line, origin line, mark the relapsing lineage's birth). Origins are the lineage
# walk-up in validate_escape_modes._simulate: in I and II resistance entered in the PRIMARY (735 and
# 154 generations before the first dose), so there is no birth to mark on the met's axis.
TEXT = {
    "I": ("Born resistant", "no response", "deposit founded by resistant cells", False),
    "II": ("Sweeps pre-therapy", "no response", "resistance fixes before dosing", False),
    "III": ("Rare clone selected", "responds {depth}, then relapses",
            "relapse clone born pre-therapy", True),
    "IV": ("Arises on drug", "responds {depth}, then relapses", "relapse clone born on drug", True),
}

# Growth figures: section times in generations from the first dose (each snaps to the nearest
# captured frame), the met clone basis, and how the sections and the Muller are drawn.
SNAP_GENS = (-30, -12, 0, 25, 60, 120)
MIN_FREQ = VEM.RIG["min_freq"]
SEED = "seed"                # the synthetic root above the tumour founder; see _clone_basis
BLOCK = 3                    # a deme is drawn as BLOCK x BLOCK pixels; see _section
MIN_CELLS = 25               # the Muller starts once the deposit first holds this many cells
MULLER_SMOOTH = 1.5          # Gaussian width in generations: removes per-step wobble, keeps the cliff

plt.rcParams.update({
    "font.family": ["Arial", "DejaVu Sans"],      # Helvetica Neue's .ttc exposes no bold face
    "font.size": 13,
    "axes.edgecolor": RULE, "axes.linewidth": 1.0,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "xtick.labelsize": 12, "ytick.labelsize": 12,
    "svg.fonttype": "none", "pdf.fonttype": 42,
})


def _rel_gen(r):
    """Each trace row's generation, relative to the first dose."""
    g = r["trace_gen"].astype(float)
    return g - g[int(r["c0"])]


def _depth(r):
    """Burden at the first dose over the on-drug nadir -- the paper's response depth."""
    s, res = r["sens"].astype(float), r["res"].astype(float)
    burden = (s + res)[int(r["c0"]):int(r["c1"]) + 1]
    return burden[0] / max(burden.min(), 1.0)


def _header(ax, fig, key, depth, dy=(47, 26, 7)):
    """Title + outcome line + origin line, stacked at fixed point offsets above ``ax``."""
    title, outcome, origin, mark = TEXT[key]

    def above(y, x=0):
        return offset_copy(ax.transAxes, fig=fig, x=x, y=y, units="points")

    ax.text(0, 1, f"{key}  {title}", transform=above(dy[0]), ha="left", va="bottom",
            fontsize=15.5, fontweight="bold", color=INK)
    ax.text(0, 1, outcome.format(depth=f"{depth:.0f}×"), transform=above(dy[1]), ha="left",
            va="bottom", fontsize=12.5, color=INK2)
    if mark:
        ax.plot([0], [1], transform=above(dy[2] + 6, 4.5), marker="o", ms=8, mfc=RES, mec="none",
                clip_on=False)
    ax.text(0, 1, origin, transform=above(dy[2], 13 if mark else 0), ha="left", va="bottom",
            fontsize=12.5, color=INK2)


def _birth_dot(ax, r, y):
    """The relapsing lineage starts as one cell: mark its birth generation at height ``y``."""
    o = int(r["origin_row"])
    if o >= 0:
        ax.plot([_rel_gen(r)[o]], [y], marker="o", ms=10, mfc=RES, mec="white", mew=2, zorder=6,
                clip_on=False)


# --- the summary: all four modes, log y --------------------------------------------------------

def _smooth_counts(y, w=5):
    """Centred running mean over ``w`` trace rows (~3-5 generations), normalised by the window's
    actual length so the ends are not dragged toward the zero padding. Under one cell is not drawn:
    on a log axis single-cell flicker reads as signal."""
    k = np.ones(w)
    y = np.convolve(y, k, mode="same") / np.convolve(np.ones_like(y), k, mode="same")
    return np.where(y >= 1, y, np.nan)


def _cells(v, _pos):
    return {1: "1", 10: "10", 100: "100", 1e3: "1K", 1e4: "10K", 1e5: "100K"}.get(v, "")


def _summary_figure(results):
    fig, axes = plt.subplots(1, len(VEM.PANELS), figsize=(13.33, 5.0), sharey=True)
    for ax, p in zip(axes, VEM.PANELS):
        key, r = p["key"], results[p["key"]]
        x = _rel_gen(r)
        c1 = x[int(r["c1"])]
        ax.axvspan(0, c1, color=BAND, alpha=0.09, lw=0, zorder=0)
        ax.text(c1 / 2, 2.4e5, "therapy", ha="center", va="center", fontsize=11, color=MUTED)
        for y, c in ((r["sens"], SENS), (r["res"], RES)):
            ax.plot(x, _smooth_counts(y.astype(float)), color=c, lw=2.2, solid_joinstyle="round",
                    solid_capstyle="round", zorder=3)
        if TEXT[key][3]:
            _birth_dot(ax, r, 1)
        _header(ax, fig, key, _depth(r))
        ax.set_yscale("log")
        ax.set_ylim(0.6, 4e5)
        ax.yaxis.set_major_locator(FixedLocator([1, 10, 100, 1e3, 1e4, 1e5]))
        ax.yaxis.set_major_formatter(FuncFormatter(_cells))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_xlim(-165, 210)
        ax.set_xticks([-100, 0, 100, 200])
        ax.grid(axis="y", color="#ececea", lw=0.8, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=0, pad=6)
    axes[0].set_ylabel("cells in the metastasis", fontsize=13, color=INK2)
    fig.supxlabel("generations since first dose", fontsize=13, color=INK2, y=0.02)
    handles = [plt.Line2D([], [], color=SENS, lw=2.6), plt.Line2D([], [], color=RES, lw=2.6)]
    fig.legend(handles, ["sensitive cells", "resistant cells"], loc="upper right",
               bbox_to_anchor=(0.995, 1.0), ncol=2, frameon=False, fontsize=13, handlelength=1.6,
               labelcolor=INK2)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.15, top=0.74, wspace=0.13)
    return fig


# --- the growth figures: sections over time above a Muller -------------------------------------

def _shade(hex_color, t):
    """``hex_color`` mixed toward white (t > 0) or black (t < 0) by |t|."""
    c = np.asarray(matplotlib.colors.to_rgb(hex_color))
    return tuple(c + (1 - c) * t) if t >= 0 else tuple(c * (1 + t))


# Clones of one status take one hue; these steps keep a child legible against its parent band
# without the darkest reading as an outline.
SHADE_STEPS = (0.0, 0.38, 0.18, 0.55, -0.12, 0.28, 0.46)


def _clone_basis(g):
    """The met's display clones, each split into its sensitive and resistant cells.

    The clones are the hero animation's: functional clones (driver signature) merged below
    MIN_FREQ. That merge folds every clone smaller than MIN_FREQ into its nearest shown ancestor
    WHATEVER its resistance, so a sensitive clone collects the many small resistant subclones it
    spawns -- on III, clone 1017 is 99% sensitive at the first dose and 89% resistant over the run.
    One colour per clone would repaint the sensitive deposit red. Instead each clone is drawn as
    two halves, the one its founding genotype belongs to and the other nested inside it, so the red
    area at every generation is exactly the resistant share.

    Returns (per-row counts over the halves, {half -> parent half}, {gid -> clone}, resistant(gid)).
    A half is named ``<clone>|S`` or ``<clone>|R``."""
    # Lineages seeded separately from the primary meet only IN the primary, so on the met's counts
    # alone every seeding is its own root -- and a root is never merged, however small (II has 186).
    # A zero-count SEED above the tumour founder gives them one root to fold into.
    full_parents = dict(g["parents"])
    for gid in g["stage"]:                          # the record's clones, closed over ancestors
        full_parents.setdefault(gid, SEED)
    traces = [{"genotypes_counts": {**mc, SEED: 0}} for mc in g["met_counts"]]
    _, parents, gmap, cols = viz._display_basis(traces, full_parents,
                                                {**g["driver_map"], SEED: (-1,)}, MIN_FREQ)
    clone_parent = {str(c): str(parents[c].iloc[0]) for c in parents.columns}
    gmap = {str(k): str(v) for k, v in gmap.items()}

    def resistant(gid):
        return g["stage"].get(gid) == "resistance"

    n = len(g["met_counts"])
    acc = {}
    for i, mc in enumerate(g["met_counts"]):
        for gid, c in mc.items():
            h = f"{gmap[gid]}|{'R' if resistant(gid) else 'S'}"
            acc.setdefault(h, np.zeros(n))[i] += c

    def half(clone, prefer_r):
        """The clone's half of the preferred status, or its other half if that one never had cells."""
        a, b = (f"{clone}|R", f"{clone}|S") if prefer_r else (f"{clone}|S", f"{clone}|R")
        return a if a in acc else (b if b in acc else None)

    edges = {}
    for c in cols:
        own = half(c, resistant(c))
        if own is None:
            continue
        if c in clone_parent:
            p = clone_parent[c]
            parent_half = half(p, resistant(p))
            if parent_half is not None:
                edges[own] = parent_half
        other = f"{c}|{'S' if own.endswith('|R') else 'R'}"
        if other in acc:
            edges[other] = own
    return pd.DataFrame(acc), edges, gmap, resistant


def _muller_layout(pivot, edges):
    """Nested Muller y-values: each clone wraps its children (parent, children..., parent), leaves
    doubled into one band. pymuller's ordering is not used: it lists only clones that appear in an
    edge, so a founder with no shown children would vanish from the plot."""
    children = {}
    for c, p in edges.items():
        children.setdefault(p, []).append(c)
    first = {c: int(np.argmax(pivot[c].to_numpy() > 0)) for c in pivot.columns}

    def inner(c):
        kids = sorted(children.get(c, []), key=lambda k: (first.get(k, 0), k))
        if not kids:
            return [c, c]
        return [c] + sum((inner(k) for k in kids), []) + [c]

    order = []
    for root in sorted((c for c in pivot.columns if c not in edges), key=lambda c: (first[c], c)):
        order += inner(root)
    Y = pivot[order].to_numpy(float) / 2
    cols, keep = list(order), [0]
    for i in range(1, len(cols)):
        if cols[i] == cols[i - 1]:
            Y[:, i] *= 2
            keep.pop()
        keep.append(i)
    return Y[:, keep], [cols[i] for i in keep]


def _band_of(gid, gmap, parents):
    """A genotype's display clone; one the trace never kept walks up to its nearest kept ancestor,
    and one with no kept ancestor belongs to the SEED root."""
    x = gid
    while x is not None and x not in gmap:
        x = parents.get(x)
    return gmap[x] if x is not None else gmap.get(SEED, SEED)


def _section(frame, coords, gsz, k_max, cell_color, rng):
    """One met section. A deme is a BLOCK x BLOCK patch whose filled share is its tumour density
    (cancer cells / ``k_max``, at least one pixel when it holds any) and whose filled pixels take
    the colours of cells sampled from it: density reads as fill, a rare clone as the odd pixel.
    At one pixel per CELL a cell is under half a point wide on a slide.

    Normal cells are not drawn: the whole grid is tissue. That also hides the holes chemotherapy's
    toxicity leaves in the host -- real, but beside the point here, and they read as render faults."""
    s = BLOCK
    tissue = np.asarray(matplotlib.colors.to_rgb(TISSUE))
    img = np.tile(tissue, (gsz * s, gsz * s, 1))
    for i, d in frame["demes"].items():
        cells = [gid for gid, n in d.items() if gid not in normal_names for _ in range(n)]
        if not cells:
            continue
        k = min(s * s, max(1, int(round(s * s * len(cells) / k_max))))
        block = np.tile(tissue, (s * s, 1))
        pick = rng.choice(len(cells), size=k, replace=False)
        block[rng.choice(s * s, size=k, replace=False)] = [cell_color(cells[j]) for j in pick]
        r, c = coords[i]
        img[r * s:(r + 1) * s, c * s:(c + 1) * s] = block.reshape(s, s, 3)
    return img


def _compact(n):
    return f"{n / 1e3:.1f}K" if n >= 1e3 else f"{n:.0f}"


def _growth_figure(key, r, g):
    rel = _rel_gen(r)
    counts, edges, gmap, resistant = _clone_basis(g)

    # Colours: a hue per status, and one shade step per clone (its two halves share it), stepped in
    # order of appearance so the founder takes the base hue.
    first = {}
    for h in counts.columns:
        c = h.rsplit("|", 1)[0]
        first[c] = min(first.get(c, 10 ** 9), int(np.argmax(counts[h].to_numpy() > 0)))
    step = {c: SHADE_STEPS[i % len(SHADE_STEPS)]
            for i, c in enumerate(sorted(first, key=lambda c: (first[c], c)))}
    colors = {h: _shade(RES if h.endswith("|R") else SENS, step[h.rsplit("|", 1)[0]])
              for h in counts.columns}

    # One Muller column per generation (the LAST trace row of it), so x is generations, not rows.
    # Counts are smoothed first and normalised after, so the near-empty early deposit does not
    # flicker between compositions.
    gen = r["trace_gen"].astype(int)
    last = np.r_[np.nonzero(np.diff(gen))[0], len(gen) - 1]
    pivot = counts.iloc[last].reset_index(drop=True)
    pivot = pivot.apply(lambda col: gaussian_filter1d(col.to_numpy(float), MULLER_SMOOTH,
                                                      mode="nearest")).clip(lower=0)
    total = pivot.sum(axis=1).to_numpy()
    pivot = pivot.div(np.where(total > 0, total, 1.0), axis=0)
    # Blank until the deposit first holds MIN_CELLS -- below that its composition is noise -- and
    # never again after, or an early dip back under it cuts a slit through the plot.
    pivot.loc[~np.maximum.accumulate(total >= MIN_CELLS)] = 0.0
    Y, order = _muller_layout(pivot, edges)
    x = rel[last]

    fig = plt.figure(figsize=(13.33, 6.0))
    left, width = 0.085, 0.895
    axm = fig.add_axes([left, 0.10, width, 0.26])
    axm.stackplot(x, Y.T, colors=[colors[c] for c in order], lw=0, zorder=2)
    # The bands fill the axes, so therapy is a bar ABOVE them, plus its two edges drawn across them.
    c1 = rel[int(r["c1"])]
    top = axm.get_xaxis_transform()
    for edge in (0, c1):
        axm.axvline(edge, color="white", lw=1.4, zorder=3)
    axm.add_patch(Rectangle((0, 1.02), c1, 0.11, transform=top, facecolor=BAND, alpha=0.18, lw=0,
                            clip_on=False))
    axm.text(c1 / 2, 1.075, "therapy", transform=top, ha="center", va="center", fontsize=10.5,
             color=INK2)
    if TEXT[key][3]:
        _birth_dot(axm, r, 0)
    axm.set_ylim(0, 1)
    axm.set_xlim(-110, 170)
    axm.yaxis.set_major_locator(FixedLocator([0, 0.5, 1]))
    axm.yaxis.set_major_formatter(PercentFormatter(1.0))
    axm.set_xticks([-100, -50, 0, 50, 100, 150])
    axm.spines[["top", "right"]].set_visible(False)
    axm.tick_params(length=0, pad=6)
    axm.set_ylabel("clone frequency", fontsize=13, color=INK2)
    axm.set_xlabel("generations since first dose", fontsize=13, color=INK2)

    # Sections: the nearest captured frame to each target time, all on ONE crop and one density
    # scale, so size and density compare across time. The crop is the union of the tumour's extent
    # over the sections that do not already fill the grid -- otherwise the relapse, which fills it,
    # sets the window and every earlier section is a speck; the relapse then overflows the window.
    frame_gen = np.array([rel[f["row"]] for f in g["frames"]])
    picks = [int(np.argmin(np.abs(frame_gen - t))) for t in SNAP_GENS]
    gsz, coords = g["met_grid_size"], g["met_coords"]
    boxes, k_max = [], 1
    for k in picks:
        occ = []
        for i, d in g["frames"][k]["demes"].items():
            n_cancer = sum(n for gid, n in d.items() if gid not in normal_names)
            if n_cancer:
                occ.append(coords[i])
                k_max = max(k_max, n_cancer)
        occ = np.array(occ)
        boxes.append((occ.min(0), occ.max(0) + 1))
    inside = [b for b in boxes if (b[1] - b[0]).max() < 0.8 * gsz] or boxes
    lo, hi = np.min([b[0] for b in inside], axis=0), np.max([b[1] for b in inside], axis=0)
    half = (hi - lo).max() / 2 + 1
    cr, cc = (lo + hi) / 2

    rgb = {}

    def cell_color(gid):
        # the cell's OWN status picks the half; its clone picks the shade
        if gid not in rgb:
            half = f"{_band_of(gid, gmap, g['parents'])}|{'R' if resistant(gid) else 'S'}"
            rgb[gid] = colors.get(half, matplotlib.colors.to_rgb(RES if resistant(gid) else SENS))
        return rgb[gid]

    rng = np.random.default_rng(0)
    met_cells = (r["sens"] + r["res"]).astype(int)
    n, size = len(picks), 0.27
    w = size * 6.0 / 13.33
    for j, k in enumerate(picks):
        f = g["frames"][k]
        cx = left + (j + 0.5) * width / n
        ax = fig.add_axes([cx - w / 2, 0.57, w, size])
        ax.imshow(_section(f, coords, gsz, k_max, cell_color, rng), extent=(0, gsz, gsz, 0),
                  interpolation="nearest")
        ax.set_xlim(cc - half, cc + half)
        ax.set_ylim(cr + half, cr - half)
        ax.axis("off")
        t = frame_gen[k]
        when = "first dose" if abs(t) < 1 else f"{t:+.0f}".replace("-", "−")
        ax.text(0.5, -0.04, when, transform=ax.transAxes, ha="center", va="top", fontsize=12,
                color=INK2)
        ax.text(0.5, -0.155, f"{_compact(met_cells[f['row']])} cells", transform=ax.transAxes,
                ha="center", va="top", fontsize=10.5, color=MUTED)
        fig.add_artist(ConnectionPatch(xyA=(0.5, -0.31), coordsA=ax.transAxes, xyB=(t, 1.17),
                                       coordsB=top, color=RULE, lw=1.0, zorder=1))
        if j == 0:
            _header(ax, fig, key, _depth(r), dy=(50, 30, 11))

    fig.legend(handles=[Patch(color=SENS, label="sensitive clones"),
                        Patch(color=RES, label="resistant clones"),
                        Patch(color=TISSUE, label="normal tissue")],
               loc="upper right", bbox_to_anchor=(0.995, 1.0), ncol=3, frameon=False, fontsize=13,
               handlelength=1.0, handleheight=1.0, labelcolor=INK2)
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma-separated panel keys for the growth figures")
    ap.add_argument("--cache", default=CACHE, help="the panel cache (default: analysis_data/escape_modes)")
    args = ap.parse_args()
    wanted = [k.strip() for k in args.only.split(",") if k.strip()] or [p["key"] for p in VEM.PANELS]

    out_dir = os.path.dirname(figure_path(os.path.join("slides", "x")))
    os.makedirs(out_dir, exist_ok=True)

    results = {}
    for p in VEM.PANELS:
        if p["key"] in wanted:
            results[p["key"]], growth = VEM._load(p, cache_dir=args.cache, growth=True)
            fig = _growth_figure(p["key"], results[p["key"]], growth)
            out = os.path.join(out_dir, f"escape_mode_{p['key']}_growth.png")
            fig.savefig(out, dpi=300, facecolor="white")
            plt.close(fig)
            print(f"figure -> {out}")
        else:
            results[p["key"]] = VEM._load(p, cache_dir=args.cache)

    fig = _summary_figure(results)
    for ext in ("png", "pdf"):
        out = os.path.join(out_dir, f"escape_modes_summary.{ext}")
        fig.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"figure -> {os.path.join(out_dir, 'escape_modes_summary.{png,pdf}')}")


if __name__ == "__main__":
    sys.exit(main())
