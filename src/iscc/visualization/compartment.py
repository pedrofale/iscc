"""Render the primary + metastasis COMPOSITE animation (``isccgif --compartment``) from an isccsim
compartment trajectory: two cell-resolution deme-grids on the LEFT, their own-compartment centered
("fish"/symmetric) Mullers on the RIGHT, a shared tumour-time axis, and ONE shared clone colormap so a
clone is the same colour in the grid and the Muller.

This promotes ``notebooks/landing_animation.py``'s ``build_figure`` / centered-Muller / GIF logic into
the CLI. It is ENGINE-FREE: everything is read from the trajectory pickle written by ``isccsim`` (see
``iscc.tumor.arc``), so the GIF renders offline.

``splash=True`` is the minimal HERO variant (no legend, "Primary"/"Metastasis" titles only, no axis
numbers, per-panel event labels with a legible semi-opaque box); ``splash=False`` is the fully-labelled
general variant (clone legend, cell counts, per-phase caption, Muller y labels).
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
import imageio.v2 as imageio

from ..tumor import viz


# --- dark hero aesthetic ------------------------------------------------------------------------
BG, PANEL, INK, SUBINK, GRID_LINE = "#0d1117", "#161b22", "#e6edf3", "#9aa4b2", "#30363d"
# neutral grey in-GIF text for the SPLASH (no blue cast, unlike INK/SUBINK)
SPLASH_TITLE, SPLASH_LABEL = "#e0e0e0", "#b3b3b3"

# NEUTRAL-grey chrome anchors reserved in every palette so the small (size-budgeted) palette can't
# blue-shift or dim the grey titles/labels.
_CHROME_ANCHORS = [(13, 17, 23), (224, 224, 224), (179, 179, 179), (255, 255, 255),
                   (150, 150, 150), (96, 96, 96), (48, 48, 48)]


def frame_cell_data(demes, deme_coords, n_primary_demes, gid_ord):
    """Rebuild the minimal per-cell frames a grid render needs (``cell_deme`` / ``cell_crd`` /
    ``cell_type`` / ``cell_compartment``) from a deme-count snapshot, in the EXACT order the engine's
    ``make_cell_data`` materialises cells — deme index ascending, and within a deme by genotype creation
    ordinal — so the section-sampled grid is identical to the engine's own render."""
    d_id, rows, cols, cids, comp = [], [], [], [], []
    for di, deme in enumerate(demes):
        if not deme:
            continue
        r, c = deme_coords[di]
        cmp = 1 if di >= n_primary_demes else 0
        for gid in sorted(deme.keys(), key=lambda g: gid_ord.get(str(g), 0)):
            cnt = int(deme[gid])
            if cnt <= 0:
                continue
            d_id.extend([di] * cnt); rows.extend([r] * cnt); cols.extend([c] * cnt)
            cids.extend([gid] * cnt); comp.extend([cmp] * cnt)
    idx = pd.RangeIndex(len(cids))
    return {
        "cell_deme": pd.DataFrame({"deme_id": d_id}, index=idx),
        "cell_crd": pd.DataFrame({"row": rows, "col": cols}, index=idx),
        "cell_type": pd.DataFrame({"cell_id": cids}, index=idx),
        "cell_compartment": pd.DataFrame({"compartment": comp}, index=idx),
    }


def build_figure(traj, splash=False):
    """Persistent, LANDSCAPE figure: each cell-resolution grid on the LEFT, its own-compartment Muller
    on the RIGHT, in a 2x2 layout so the two Mullers stack in the right column and SHARE one tumour-time
    x-axis, each aligned with its grid's row. The Mullers are drawn ONCE (semantic clone colours,
    ``min_freq`` collapses the passenger rainbow so the sweeps read as bands); each frame only moves the
    reveal cursor over them. Returns ``(fig, axp, axm, ax_mp, ax_mm, xmax)``."""
    traces = traj["traces"]
    parents = traj["genotypes_parents"]
    colors = traj["colors"]
    marks = traj["marks"]
    fig = plt.figure(figsize=(12.6, 7.1), dpi=150 if splash else 170)
    fig.patch.set_facecolor(BG)
    if splash:
        # Manual, SYMMETRIC placement: two EQUAL square grids in the left column, two wide Mullers in
        # the right column, the whole block centered in the canvas (equal margins).
        gh = 0.40
        gw = gh * 7.1 / 12.6
        mw, gap = 0.52, 0.045
        lw = (1.0 - gw - gap - mw) / 2.0
        mx, y_top, y_bot = lw + gw + gap, 0.525, 0.075
        axp = fig.add_axes([lw, y_top, gw, gh])
        ax_mp = fig.add_axes([mx, y_top, mw, gh])
        axm = fig.add_axes([lw, y_bot, gw, gh])
        ax_mm = fig.add_axes([mx, y_bot, mw, gh])
    else:
        gs = GridSpec(2, 2, width_ratios=[1.0, 1.62], height_ratios=[1, 1], hspace=0.22, wspace=0.075,
                      left=0.028, right=0.988, top=0.865, bottom=0.085)
        axp = fig.add_subplot(gs[0, 0]); ax_mp = fig.add_subplot(gs[0, 1])
        axm = fig.add_subplot(gs[1, 0]); ax_mm = fig.add_subplot(gs[1, 1], sharex=ax_mp)

    viz.plot_muller_compartments(traces, parents, axes=(ax_mp, ax_mm),
                                 driver_map=traj["driver_map"], min_freq=traj["min_freq"],
                                 clone_colors=colors, mark_generations=marks, star_genotype=None,
                                 # smoothing_std is a Gaussian width in GENERATIONS: the pymuller default
                                 # (0.1) is ~no smoothing, so the stacked band edges kink at every step.
                                 # 3.0 is small vs the ~230-generation arc — it removes the per-step wobble
                                 # while KEEPING the sharp chemo bottleneck, the resection mark, the
                                 # multi-clone bands, and the DCIS-fills-before-invasion ordering readable.
                                 smoothing_std=3.0,
                                 centered=splash)
    xmax = len(traces) - 1
    # splash: make the clinical-event lines PANEL-SPECIFIC to where the intervention acts — resection
    # removes the primary (primary panel only); chemo's visible effect is on the deposit (met panel
    # only); seeding is the shared branch point (both). The general path keeps all marks on both.
    keep_marks = {"primary": {"seeding", "resection"},
                  "metastasis": {"seeding", "chemo start", "chemo end"}}
    for ax, band in ((ax_mp, "primary"), (ax_mm, "metastasis")):
        ax.set_facecolor(BG)
        ax.set_xlim(0, xmax)
        if splash:
            for txt in list(ax.texts):
                if txt.get_text() not in keep_marks[band]:
                    xpos = txt.get_position()[0]
                    for ln in list(ax.lines):
                        xd = ln.get_xdata()
                        if len(xd) and abs(float(xd[0]) - xpos) < 1e-6:
                            ln.remove()
                    txt.remove()
        for ln in ax.lines:
            ln.set_color(SPLASH_LABEL if splash else SUBINK); ln.set_alpha(0.75)
        for txt in ax.texts:
            txt.set_color(SPLASH_LABEL if splash else INK); txt.set_fontsize(9 if splash else 8.5)
            if splash:
                txt.set_bbox(dict(facecolor=PANEL, alpha=0.82, edgecolor="none", pad=1.8))
                txt.set_zorder(4)
        if splash:
            ax.axis("off")
        else:
            for side, sp in ax.spines.items():
                sp.set_visible(side in ("left", "bottom")); sp.set_color(GRID_LINE)
            ax.tick_params(colors=SUBINK, labelsize=8, length=2.5)
            ax.set_ylabel(f"{band}\ncells", color=SUBINK, fontsize=9.5)
    if not splash:
        ax_mp.set_xlabel(""); ax_mp.tick_params(labelbottom=False)
        ax_mm.set_xlabel("tumour time (snapshot) — seeding · resection · chemotherapy annotated",
                         color=SUBINK, fontsize=9.5)
    if ax_mp.legend_:
        ax_mp.legend_.remove()
    return fig, axp, axm, ax_mp, ax_mm, xmax


def draw_grid(ax, sub, gsz, colors, title, title_color=INK):
    """Render one cell-resolution grid. ``title`` is used verbatim; ``title_color`` lets the splash use
    a neutral grey (no blue cast). The empty background == the page colour so the grid has no card/seam:
    only the cells are non-page pixels and the tumour reads as growing directly on the splash."""
    viz.plot_grid(sub, gsz, None, None, color=["cell_type"], ax=ax, expand_demes=True,
                  section_frac=0.85, cancer_color=None, type_cmap=colors, empty_color=BG)
    if ax.legend_:
        ax.legend_.remove()
    ax.set_title(title, color=title_color, fontsize=12, pad=5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_visible(False)


def _save_gif_pillow(gif_path, imgs, per, budget_mb=5.95, candidates=(48, 32, 24, 20)):
    """Write ``imgs`` as an optimised GIF via Pillow with ONE global palette = neutral chrome anchors
    (so grey text stays grey) + an adaptive fill from a montage of representative frames. Tries palette
    sizes largest-first and keeps the largest whose file is <= ``budget_mb``. Returns the written MB."""
    from PIL import Image
    idxs = sorted({0, len(imgs) // 3, 2 * len(imgs) // 3, len(imgs) - 1})
    montage = np.concatenate([imgs[i] for i in idxs], axis=1)
    best = None
    for ncol in candidates:
        n_fill = max(1, ncol - len(_CHROME_ANCHORS))
        base = Image.fromarray(montage).convert("P", palette=Image.ADAPTIVE, colors=n_fill)
        pal = base.getpalette()[:3 * n_fill]
        for c in _CHROME_ANCHORS:
            pal += list(c)
        pal += [0] * (3 * 256 - len(pal))
        palimg = Image.new("P", (1, 1)); palimg.putpalette(pal)
        pframes = [Image.fromarray(a).quantize(palette=palimg, dither=Image.Dither.NONE) for a in imgs]
        pframes[0].save(gif_path, save_all=True, append_images=pframes[1:], duration=per, loop=0,
                        optimize=True, disposal=2)
        mb = os.path.getsize(gif_path) / 1e6
        best = (ncol, mb)
        if mb <= budget_mb:
            break
    return best[1]


def render_animation(traj, out_path, splash=False, poster=True):
    """Render the compartment trajectory to an animated GIF at ``out_path``.

    ``splash=True`` -> the minimal HERO variant (label-light: no clone legend, no cell counts, no phase
    caption, no Muller y labels/ticks; keeps the event vlines + annotations), size-budgeted via a Pillow
    global palette. ``splash=False`` -> the fully-labelled general animation. Returns the written MB."""
    frames = traj["frames"]
    colors = traj["colors"]
    deme_coords = traj["deme_coords"]
    n_primary = traj["n_primary_demes"]
    gid_ord = traj["gid_ord"]
    grid_size, met_grid_size = traj["grid_size"], traj["met_grid_size"]

    fig, axp, axm, ax_mp, ax_mm, xmax = build_figure(traj, splash=splash)

    caption = None
    if not splash:
        legend_handles = [Patch(facecolor=col, edgecolor="none", label=f"{i+1}. {lbl}")
                          for i, (lbl, col) in enumerate(traj["stage_legend"])]
        fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.985),
                   ncol=5, fontsize=9.5, frameon=False, labelcolor=INK, handlelength=1.1,
                   columnspacing=1.6, handletextpad=0.5)
        caption = fig.text(0.5, 0.918, "", ha="center", va="top", color=SUBINK, fontsize=10.5)

    grid_titles = {"primary": "Primary", "metastasis": "Metastasis"} if splash else \
                  {"primary": "primary duct field", "metastasis": "metastasis"}
    phase_caption = traj.get("phase_caption", {})
    cursor_artists = []

    def render(fi):
        frame = frames[fi]
        cd = frame_cell_data(frame["demes"], deme_coords, n_primary, gid_ord)
        comp = cd["cell_compartment"]["compartment"].to_numpy()
        for ax in (axp, axm):
            ax.clear()
        for ax, val, gsz, key in ((axp, 0, grid_size, "primary"),
                                  (axm, 1, met_grid_size, "metastasis")):
            mask = comp == val
            sub = {k: df[mask] for k, df in cd.items()}
            title = grid_titles[key] if splash else f"{grid_titles[key]}   ({int(mask.sum()):,} cells)"
            draw_grid(ax, sub, gsz, colors, title, title_color=SPLASH_TITLE if splash else INK)
        tl = frame["cursor"]
        for a in cursor_artists:
            a.remove()
        cursor_artists.clear()
        for ax in (ax_mp, ax_mm):
            cursor_artists.append(ax.axvspan(tl, xmax, color=BG, alpha=0.82, zorder=5))
            cursor_artists.append(ax.axvline(tl, color="#c9d1d9", lw=1.1, alpha=0.55, zorder=6))
        if caption is not None:
            caption.set_text(phase_caption.get(frame["phase"], ""))
        fig.canvas.draw()
        return np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()

    imgs = [render(fi) for fi in range(len(frames))]

    if splash:
        LOOP_MS = 17000
        per = [max(40, round(LOOP_MS / len(imgs)))] * len(imgs)
    else:
        per = [170] * len(imgs)
        per[0] = 900
        per[-1] = 1900

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    if splash:
        size_mb = _save_gif_pillow(out_path, imgs, per, budget_mb=5.95)
    else:
        imageio.mimsave(out_path, imgs, format="GIF", duration=per, loop=0)
        size_mb = os.path.getsize(out_path) / 1e6

    if poster:
        poster_path = os.path.splitext(out_path)[0] + "_poster.png"
        imageio.imwrite(poster_path, imgs[-1])
    plt.close(fig)
    return size_mb
