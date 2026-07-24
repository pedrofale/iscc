"""Docs landing-page growth ANIMATION (the moving version of ``metastasis_demo.py``).

Renders the full clinical arc of one clonally-linked primary + metastasis on ONE shared clone
colormap, laid out as two CELL-RESOLUTION deme-grids on top and the full-width 2-band Muller below,
and writes an animated GIF (+ a poster PNG and a few representative frames) under ``docs/assets/``.

The four selective sweeps of the metastatic arc each read, both as spatial progression in the grids
and as bands in the Muller, with the clinical events annotated:
  1. escape the duct     (breach)              — cancer crosses the epithelial wall into the stroma
  2. survive the stroma  (stromal_survival)    — cancer traverses the hostile stroma
  3. establish the met   (met_survival)         — a founder seeds and grows the second grid
  4. escape chemo        (treatment resistance) — after resection + systemic chemo, resistant cells relapse

Colour is by the STAGE-DOMINANT selective trait of each cancer clone (the same clone -> same colour in
the primary grid, the met grid, and both Muller bands); normal tissue is muted so the cancer reads.

Run:  python notebooks/landing_animation.py
"""
import os
import copy
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
import imageio.v2 as imageio

from iscc.tumor.models import GenotypeTumor
from iscc.tumor import viz
from iscc.treatment import Surgery, Chemotherapy
from iscc.constants import normal_names

HERE = os.path.dirname(__file__)
ASSETS = os.path.abspath(os.path.join(HERE, "..", "docs", "assets"))
FRAMES = os.path.join(HERE, "example_out", "landing_frames")
os.makedirs(ASSETS, exist_ok=True)
os.makedirs(FRAMES, exist_ok=True)

# ---- config: metastasis_demo.py's arc, scaled up a touch for the story -------------------------
GENOME = {"n_segments": 12, "segment_size": 50}
# breach is rarer + the epithelial barrier is HIGH, so the ducts pack full (DCIS) for a good stretch
# before a breach subclone escapes into the stroma (IDC); met seeding is delayed (met_seed_kappa) so it
# follows stromal invasion. On-screen order: ducts fill -> breach -> stroma -> met seeding -> chemo -> relapse.
# treatment_resistance is made RARE (one site, prop 0.001) and STRONG (effects 2.8 -> resistant
# treatment_resistance ~0.64) so that at chemo time only a SMALL subclone of the metastasis is resistant.
# Strong chemo (see grow_arc) then kills the sensitive bulk -> the met visibly regresses to that one
# resistant subclone, which relapses after chemo as essentially one clone (one colour, red).
SELECTION = {"prop_driver": 0.04, "prop_dispersal": 0.0, "prop_immune_resistance": 0.02,
             "prop_treatment_resistance": 0.001, "prop_breach": 0.012, "prop_stromal_survival": 0.03,
             "prop_met_survival": 0.05, "breach_effects": 2.2, "stromal_survival_effects": 2.2,
             "met_survival_effects": 2.5, "treatment_resistant_effects": 2.8}
CANCER = {"division_rate": 0.7, "death_rate": 0.05, "max_birth_rate": 0.95, "mutation_rate": 0.6,
          "dispersal_rate": 0.35, "cnv_prob": 0.15, "wgd_rate": 0.05}
DEME = {"carrying_capacity": 20, "initial_cancer_cells": 8, "resident_pressure_ref": 0.2}
# met_grid_size == grid_size and K_met == K_duct so the two grids have IDENTICAL cell granularity: same
# number of demes AND the same per-deme carrying capacity -> the same expanded block size, so neither
# grid looks coarser/finer than the other. (A met is still a smaller deposit; only the resolution matches.)
SPATIAL = {"grid_size": 26, "structure_radius": 3, "n_glands": 4, "gland_radius": 3, "min_gland_sep": 8,
           "K_duct": 40, "K_stroma": 26, "stroma_fill_frac": 0.3, "cross_gland_kappa": 0.06,
           "epithelial_barrier": 6.0, "stromal_hazard": 0.7,
           "met_grid_size": 26, "K_met": 40, "host_fill_frac": 0.35,
           "met_seed_kappa": 0.02, "met_hazard": 0.45, "met_transit_floor": 0.02}

MIN_FREQ = 0.05   # clone size-merge threshold — MUST match between story_colors() and the Muller draw

# ---- dark hero aesthetic -----------------------------------------------------------------------
BG, PANEL, INK, SUBINK, GRID_LINE = "#0d1117", "#161b22", "#e6edf3", "#9aa4b2", "#30363d"
# the four selective sweeps + proliferation + none (the shared clone colormap)
STAGE_COL = {
    "none":       (0.50, 0.53, 0.60, 1.0),   # grey
    "prolif":     (0.23, 0.51, 0.84, 1.0),   # blue   — proliferation (onc/TSG)
    "breach":     (0.98, 0.62, 0.09, 1.0),   # orange — escape the duct (breach)
    "stromal":    (0.18, 0.75, 0.44, 1.0),   # green  — survive the stroma
    "met":        (0.68, 0.40, 0.86, 1.0),   # purple — establish the metastasis
    "resistance": (0.94, 0.24, 0.28, 1.0),   # red    — escape chemo (resistance)
}
STAGE_ORDER = ["prolif", "breach", "stromal", "met", "resistance"]
STAGE_LABEL = {"prolif": "proliferation", "breach": "duct escape", "stromal": "stromal survival",
               "met": "met establishment", "resistance": "chemo resistance"}
# muted normal tissue: barely above the page colour so the cancer pops and the grids blend seamlessly
# into the splash (no filled-square seam). The duct epithelium stays a touch brighter as the DCIS anchor.
NORMAL_MUTED = {"epithelial": (0.15, 0.29, 0.23, 1.0), "stromal": (0.18, 0.15, 0.18, 1.0),
                "host": (0.13, 0.16, 0.21, 1.0), "immune": (0.24, 0.22, 0.11, 1.0)}


def stage_of(rep, B=0.70, S=0.80, M=0.92, R=0.55):
    """The stage-dominant selective trait of a cancer clone: the LAST sweep in the metastatic arc it
    has activated (saturated). ``None`` for normal cells.

    The fitness effects are spatially gated (``GenotypeTumor._death_rate``): ``breach`` only lifts the
    epithelial-barrier hazard at the duct wall, ``stromal_survival`` only the stromal hazard, and
    ``met_survival`` only the met host hazard. ``met_survival`` therefore drifts up neutrally in the
    primary, so it needs a high threshold (near-saturation) to count as the met stage — otherwise it
    would mislabel every invasion clone. The invasion axes are selected earlier and get lower
    thresholds, so the duct-escape (orange) and stromal (green) fronts surface before the primary turns
    met-capable (purple)."""
    if getattr(rep, "type", None) != "cancer":
        return None
    ep = rep.evolutionary_parameters
    base_div = getattr(rep, "baseline_rates", {}).get("division_rate", ep["division_rate"])
    if ep.get("treatment_resistance", 0) > R:
        return "resistance"
    if ep.get("met_survival", 0) > M:
        return "met"
    if ep.get("stromal_survival", 0) > S:
        return "stromal"
    if ep.get("breach", 0) > B:
        return "breach"
    if ep["division_rate"] > base_div + 1e-9:
        return "prolif"
    return "none"


def story_colors(t):
    """{gid -> rgba}: ONE shared clone colormap for the grids AND the Mullers.

    The bug this fixes: the cell-grids used to colour each cancer cell by its OWN genotype's stage,
    while the Muller colours a band by the stage of its FUNCTIONAL-CLONE founder — two different
    groupings, so the same cell got different colours in the grid and the Muller. Here every cancer
    genotype is instead routed through the SAME display basis the Muller draws on (functional clone via
    ``driver_map``, size-merged at ``MIN_FREQ``) and takes ITS BAND's colour. A band's colour is the
    stage-dominant sweep of the band's founder — exactly what the Muller keys on — so a clone is
    literally the same colour in the primary grid, the met grid, and both Muller bands. Normals keep
    their muted type colour."""
    by_str = {str(gid): rep for gid, rep in t.genotypes.items()}
    driver_map = t._functional_signatures()
    _, _, gmap, basis_cols = viz._display_basis(t.traces, t.genotypes_parents,
                                                driver_map=driver_map, min_freq=MIN_FREQ)
    # each display band -> the stage colour of its founder genotype (what the Muller already keys on)
    band_col = {}
    for b in basis_cols:
        st = stage_of(by_str.get(str(b)))
        band_col[str(b)] = STAGE_COL["none" if st is None else st]
    out = {}
    for sgid, rep in by_str.items():
        if getattr(rep, "type", None) != "cancer":
            out[sgid] = NORMAL_MUTED.get(getattr(rep, "type", None), (0.7, 0.7, 0.7, 1.0))
            continue
        b = str(gmap.get(sgid, sgid))                       # this genotype's display band
        st = stage_of(rep)
        out[sgid] = band_col.get(b, STAGE_COL["none" if st is None else st])
    for n in normal_names:
        out[n] = NORMAL_MUTED[n]
    return out


def compartment_cancer(t):
    p = sum(c for i in range(t.n_primary_demes) for g, c in t.demes[i].items() if t._is_cancer(g))
    m = sum(c for i in range(t.n_primary_demes, len(t.demes)) for g, c in t.demes[i].items()
            if t._is_cancer(g))
    return p, m


def grow_arc():
    """Grow the whole arc, capturing a (demes, genotype_counts, trace_len, phase) frame at each step,
    plus the clinical-event snapshot indices."""
    # tau=0.5 (half a generation per captured frame) -> ~2x the frames of tau=1.0, so the growth reads
    # as a smooth sweep instead of discrete jumps. Treatment windows are in STEP units, so the fixed
    # phases below scale their step counts (and the chemo duration) by 1/tau to keep the clinical arc's
    # real-time length: #steps * tau stays constant vs the tau=1.0 tuning.
    TAU = 0.5
    # seed=2 is the reproducible draw where treatment resistance stays a SMALL late subclone of the met
    # (~10% at chemo time) rather than sweeping early, so strong chemo can visibly wipe the sensitive bulk
    # and leave that one resistant clone to relapse (see grow_arc header / SELECTION).
    SEED = 2
    t = GenotypeTumor(seed=SEED, genome_params=GENOME, selection_params=SELECTION, cancer_cell_params=CANCER,
                      deme_params=DEME, spatial_params=SPATIAL, update_mode="tau", tau=TAU)
    frames = []

    def capture(phase):
        frames.append((copy.deepcopy(t.demes), dict(t.genotypes_counts), len(t.traces), phase))

    capture("growth")
    seeding_snap = None
    while True:
        p, m = compartment_cancer(t)
        if m >= 260 or p + m >= 5600:
            break
        t.grow(n_steps=1, seed=SEED)            # fine cadence -> a smooth DCIS -> IDC -> seeding arc
        if seeding_snap is None and compartment_cancer(t)[1] > 0:
            seeding_snap = len(t.traces)
        capture("growth" if seeding_snap is None else "metastatic seeding")

    resect_snap = len(t.traces)
    t.grow(n_steps=1, seed=SEED, treatment=Surgery(start=t.step, site="primary"))
    capture("resection")
    for _ in range(3):                          # settle after resection (2 -> 4 frames at TAU=0.5)
        t.grow(n_steps=1, seed=SEED)
        capture("resection")

    # STRONG systemic chemo over a long window: kill_rate (2.8) is well above the birth cap and
    # effectiveness ~1, so every SENSITIVE clone regresses hard; the lone resistant subclone
    # (treatment_resistance ~0.64) is only partly hit, dips, then rebounds once chemo stops. 20 steps
    # (vs 12) give the regression room to read on-screen as a dramatic shrink-to-one-clone.
    chemo_snap = len(t.traces)
    chemo = Chemotherapy(start=t.step, duration=20, kill_rate=2.8, effectiveness=0.98,
                         toxicity=0.12, sites="both")
    for _ in range(20):
        t.grow(n_steps=1, seed=SEED, treatment=chemo)
        capture("chemotherapy")
    chemo_end_snap = len(t.traces)
    for _ in range(24):                         # relapse: the single resistant clone re-expands (red)
        t.grow(n_steps=1, seed=SEED)
        capture("relapse")

    marks = [(seeding_snap, "seeding"), (resect_snap, "resection"),
             (chemo_snap, "chemo start"), (chemo_end_snap, "chemo end")]
    p, m = compartment_cancer(t)
    print(f"arc grown: primary={p} met={m}  frames={len(frames)}  traces={len(t.traces)}  marks={marks}")
    return t, frames, marks


def build_figure(t, marks, colors, splash=False):
    """Persistent, LANDSCAPE figure: each cell-resolution grid on the LEFT, its own-compartment Muller
    on the RIGHT, in a 2x2 gridspec so the two Mullers stack in the right column and SHARE one
    tumour-time x-axis, each aligned with its grid's row:

        [0,0] primary grid      | [0,1] primary Muller
        [1,0] metastasis grid   | [1,1] metastasis Muller  (shares x with the primary Muller)

    ``splash=False`` (the DEFAULT, general path) keeps the full labelling — per-band y-axis labels and
    tick counts, a descriptive x label. ``splash=True`` is the minimal HERO variant: no y labels/ticks,
    an x label of just "Time", and (in ``main``) no clone legend / cell counts / phase caption; only the
    event vertical lines + their annotations remain. The Mullers are drawn ONCE; each frame only moves
    the reveal cursor over them (no band flicker)."""
    fig = plt.figure(figsize=(12.6, 7.1), dpi=170)          # ~2140 x 1210 px -> crisp when shown large
    fig.patch.set_facecolor(BG)
    if splash:
        # Manual, SYMMETRIC placement: two EQUAL square grids in the left column, two wide Mullers in
        # the right column, the whole block centered in the canvas (equal left/right & top/bottom
        # margins). The grid axes are square and IDENTICAL in size, so both grids render the same size.
        gh = 0.40                                           # grid height (figure fraction)
        gw = gh * 7.1 / 12.6                                # equal width -> square axes (=> equal grids)
        mw, gap = 0.52, 0.045                               # Muller width, grid<->Muller gap
        lw = (1.0 - gw - gap - mw) / 2.0                    # equal L/R margin -> content centered at x=0.5
        mx, y_top, y_bot = lw + gw + gap, 0.525, 0.075      # rows centered at y=0.5
        axp = fig.add_axes([lw, y_top, gw, gh])             # primary grid (top-left)
        ax_mp = fig.add_axes([mx, y_top, mw, gh])           # primary Muller (top-right)
        axm = fig.add_axes([lw, y_bot, gw, gh])             # metastasis grid (bottom-left)
        ax_mm = fig.add_axes([mx, y_bot, mw, gh])           # metastasis Muller (bottom-right)
    else:
        gs = GridSpec(2, 2, width_ratios=[1.0, 1.62], height_ratios=[1, 1], hspace=0.22, wspace=0.075,
                      left=0.028, right=0.988, top=0.865, bottom=0.085)
        axp = fig.add_subplot(gs[0, 0]); ax_mp = fig.add_subplot(gs[0, 1])
        axm = fig.add_subplot(gs[1, 0]); ax_mm = fig.add_subplot(gs[1, 1], sharex=ax_mp)

    # draw the full-arc 2-band Muller once (semantic clone colours; min_freq collapses the passenger
    # rainbow so the sweeps read as bands). Founder-coloured bands -> a band's colour is its clone's.
    # splash uses the CENTERED (Noble "fish"/stream) baseline — bands grow outward from a midline.
    viz.plot_muller_compartments(t.traces, t.genotypes_parents, axes=(ax_mp, ax_mm),
                                 driver_map=t._functional_signatures(), min_freq=MIN_FREQ,
                                 clone_colors=colors, mark_generations=marks, star_genotype=None,
                                 centered=splash)
    xmax = len(t.traces) - 1
    for ax, band in ((ax_mp, "primary"), (ax_mm, "metastasis")):
        ax.set_facecolor(BG)                     # axes background == the page colour (no card/seam)
        ax.set_xlim(0, xmax)
        for ln in ax.lines:                      # event vlines -> visible dashed grey on the dark page
            ln.set_color(SUBINK); ln.set_alpha(0.75)
        for txt in ax.texts:                     # recolour the event labels for the dark theme
            txt.set_color(INK); txt.set_fontsize(9 if splash else 8.5)
        if splash:
            ax.axis("off")                       # NO axis chrome whatsoever (no spines/ticks/labels);
            #                                      the bands + event vlines + their text still render.
        else:
            for side, sp in ax.spines.items():
                sp.set_visible(side in ("left", "bottom")); sp.set_color(GRID_LINE)
            ax.tick_params(colors=SUBINK, labelsize=8, length=2.5)
            ax.set_ylabel(f"{band}\ncells", color=SUBINK, fontsize=9.5)
    if not splash:
        ax_mp.set_xlabel(""); ax_mp.tick_params(labelbottom=False)   # only the bottom Muller carries x
        ax_mm.set_xlabel("tumour time (snapshot) — seeding · resection · chemotherapy annotated",
                         color=SUBINK, fontsize=9.5)
    if ax_mp.legend_:
        ax_mp.legend_.remove()
    return fig, axp, axm, ax_mp, ax_mm, xmax


def draw_grid(ax, sub, gsz, colors, title):
    """Render one cell-resolution grid. ``title`` is used verbatim (the caller formats it — with a cell
    count for the general path, or just "Primary"/"Metastasis" for the splash)."""
    # empty background == the page colour so the grid has NO card/seam: only the cells are non-page
    # pixels and the tumour reads as growing directly on the splash.
    viz.plot_grid(sub, gsz, None, None, color=["cell_type"], ax=ax, expand_demes=True,
                  section_frac=0.85, cancer_color=None, type_cmap=colors, empty_color=BG)
    if ax.legend_:
        ax.legend_.remove()
    ax.set_title(title, color=INK, fontsize=12, pad=5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor(BG)
    for sp in ax.spines.values():          # no frame around the grid
        sp.set_visible(False)


def _save_gif_pillow(gif_path, imgs, per, budget_mb=5.95, candidates=(48, 32, 24, 20)):
    """Write ``imgs`` as an optimised GIF via Pillow with ONE global adaptive palette (see caller).
    Tries palette sizes largest-first and keeps the largest whose file is <= ``budget_mb`` (falls back
    to the smallest if none fit). Returns the written size in MB."""
    from PIL import Image
    idxs = sorted({0, len(imgs) // 3, 2 * len(imgs) // 3, len(imgs) - 1})
    montage = np.concatenate([imgs[i] for i in idxs], axis=1)   # palette sees the whole arc + chrome
    best = None
    for ncol in candidates:
        pal = Image.fromarray(montage).convert("P", palette=Image.ADAPTIVE, colors=ncol)
        pframes = [Image.fromarray(a).quantize(palette=pal, dither=Image.Dither.NONE) for a in imgs]
        pframes[0].save(gif_path, save_all=True, append_images=pframes[1:], duration=per, loop=0,
                        optimize=True, disposal=2)
        mb = os.path.getsize(gif_path) / 1e6
        print(f"  palette={ncol}: {mb:.2f} MB")
        best = (ncol, mb)
        if mb <= budget_mb:
            break
    return best[1]


def main(splash=False):
    """Render the arc to a GIF. ``splash=True`` -> the minimal HERO variant written to
    ``docs/assets/landing_hero.gif`` (label-light: no clone legend, no cell counts, no phase caption,
    no Muller y labels/ticks, "Time" x label; keeps the event vlines + annotations). ``splash=False``
    (default) -> the fully-labelled general animation written to ``notebooks/example_out/``."""
    t0 = time.time()
    t, frames, marks = grow_arc()
    colors = story_colors(t)

    fig, axp, axm, ax_mp, ax_mm, xmax = build_figure(t, marks, colors, splash=splash)

    caption = None
    if not splash:
        # the shared clone legend + a per-phase caption (the general/default path keeps all labels)
        legend_handles = [Patch(facecolor=STAGE_COL[s], edgecolor="none", label=f"{i+1}. {STAGE_LABEL[s]}")
                          for i, s in enumerate(STAGE_ORDER)]
        fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.985),
                   ncol=5, fontsize=9.5, frameon=False, labelcolor=INK, handlelength=1.1,
                   columnspacing=1.6, handletextpad=0.5)
        caption = fig.text(0.5, 0.918, "", ha="center", va="top", color=SUBINK, fontsize=10.5)

    grid_titles = {"primary": "Primary", "metastasis": "Metastasis"} if splash else \
                  {"primary": "primary duct field", "metastasis": "metastasis"}
    cursor_artists = []
    representative = {}  # phase -> frame index, for saved PNGs

    def render(fi):
        demes, gcounts, tl, phase = frames[fi]
        t.demes = copy.deepcopy(demes); t.genotypes_counts = dict(gcounts)
        t.cell_data = None
        t.make_cell_data()
        cd = t.cell_data
        comp = cd["cell_compartment"]["compartment"].to_numpy()
        for ax in (axp, axm):
            ax.clear()
        for ax, val, gsz, key in ((axp, 0, t.grid_size, "primary"),
                                  (axm, 1, t.met_grid_size, "metastasis")):
            mask = comp == val
            sub = {k: df[mask] for k, df in cd.items()}
            title = grid_titles[key] if splash else f"{grid_titles[key]}   ({int(mask.sum()):,} cells)"
            draw_grid(ax, sub, gsz, colors, title)
        # progressive reveal cursor over the static Muller
        for a in cursor_artists:
            a.remove()
        cursor_artists.clear()
        for ax in (ax_mp, ax_mm):
            cursor_artists.append(ax.axvspan(tl, xmax, color=BG, alpha=0.82, zorder=5))
            cursor_artists.append(ax.axvline(tl, color="#c9d1d9", lw=1.1, alpha=0.55, zorder=6))
        if caption is not None:
            caption.set_text({
                "growth": "① proliferation in the ducts (DCIS)",
                "metastatic seeding": "①→② breach the duct wall, survive the stroma, seed the met (IDC)",
                "resection": "the primary is surgically resected",
                "chemotherapy": "systemic chemotherapy — the sensitive metastasis regresses",
                "relapse": "④ resistant clones relapse the metastasis"}[phase])
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        representative.setdefault(phase, fi)
        return buf

    # build all frames; hold the DCIS start and the relapsed end a little longer so the loop reads
    imgs = []
    for fi in range(len(frames)):
        imgs.append(render(fi))
        if fi % 10 == 0:
            print(f"  frame {fi+1}/{len(frames)}  ({time.time()-t0:.0f}s)")
    if splash:
        # UNIFORM cadence, ~17 s loop regardless of frame count: per-frame = loop / frame-count, so more
        # frames -> shorter per-frame -> smoother playback (NOT a longer loop). Every frame the same
        # duration so the loop is seamless — no dwell at the DCIS start or the relapsed end.
        LOOP_MS = 17000
        per = [max(40, round(LOOP_MS / len(imgs)))] * len(imgs)
    else:
        # general path: ~6 fps with a short hold on the first/last frames so the loop reads.
        per = [170] * len(imgs)
        per[0] = 900
        per[-1] = 1900

    stem = "landing_hero" if splash else "landing_full"
    out_dir = ASSETS if splash else os.path.join(HERE, "example_out")
    gif_path = os.path.join(out_dir, f"{stem}.gif")
    if splash:
        # imageio's GIF writer ignores palettesize/subrectangles, so quantise with Pillow directly to
        # keep the finer-cadence (many-frame) GIF small WITHOUT dropping frames or lowering the pixel
        # size. ONE global adaptive palette is built from a montage of representative frames (early
        # rainbow primary + late single-clone met + the neutral titles/labels present in every frame),
        # so the chrome stays crisp and every arc colour is represented; dithering is OFF (it only adds
        # incompressible noise here). Pick the LARGEST palette that stays within the size budget.
        size_mb = _save_gif_pillow(gif_path, imgs, per, budget_mb=5.95)
    else:
        imageio.mimsave(gif_path, imgs, format="GIF", duration=per, loop=0)
        size_mb = os.path.getsize(gif_path) / 1e6
    print(f"wrote {gif_path}  ({size_mb:.2f} MB, {len(imgs)} frames, ~{sum(per)/1000.0:.1f}s, "
          f"{'splash/minimal' if splash else 'full-labelled'})")

    # poster PNG (final state) + representative frames for the report
    poster = os.path.join(out_dir, f"{stem}_poster.png")
    imageio.imwrite(poster, imgs[-1])
    print("wrote", poster)
    for phase, fi in representative.items():
        p = os.path.join(FRAMES, f"{stem}_{fi:02d}_{phase.replace(' ', '_')}.png")
        imageio.imwrite(p, imgs[fi])
        print("  representative:", p)
    plt.close("all")
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splash", action="store_true",
                    help="render the minimal HERO variant to docs/assets/landing_hero.gif "
                         "(default: the fully-labelled general animation to notebooks/example_out/)")
    main(splash=ap.parse_args().splash)
