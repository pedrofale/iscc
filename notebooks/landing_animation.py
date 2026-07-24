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
SELECTION = {"prop_driver": 0.04, "prop_dispersal": 0.0, "prop_immune_resistance": 0.02,
             "prop_treatment_resistance": 0.03, "prop_breach": 0.03, "prop_stromal_survival": 0.03,
             "prop_met_survival": 0.05, "breach_effects": 2.2, "stromal_survival_effects": 2.2,
             "met_survival_effects": 2.5}
CANCER = {"division_rate": 0.7, "death_rate": 0.05, "max_birth_rate": 0.95, "mutation_rate": 0.6,
          "dispersal_rate": 0.35, "cnv_prob": 0.15, "wgd_rate": 0.05}
DEME = {"carrying_capacity": 20, "initial_cancer_cells": 8, "resident_pressure_ref": 0.2}
SPATIAL = {"grid_size": 26, "structure_radius": 3, "n_glands": 4, "gland_radius": 3, "min_gland_sep": 8,
           "K_duct": 40, "K_stroma": 26, "stroma_fill_frac": 0.3, "cross_gland_kappa": 0.06,
           "epithelial_barrier": 1.2, "stromal_hazard": 0.7,
           "met_grid_size": 12, "K_met": 18, "host_fill_frac": 0.35,
           "met_seed_kappa": 0.07, "met_hazard": 0.45, "met_transit_floor": 0.02}

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
# muted normal tissue (close to the panel so the cancer pops, but faintly readable as anatomy)
NORMAL_MUTED = {"epithelial": (0.16, 0.30, 0.24, 1.0), "stromal": (0.28, 0.22, 0.27, 1.0),
                "host": (0.22, 0.25, 0.30, 1.0), "immune": (0.32, 0.30, 0.14, 1.0)}


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
    """{gid -> rgba} colouring every cancer clone by its stage-dominant sweep and every normal by its
    (muted) type — ONE map, so a clone is the same colour in every panel."""
    out = {}
    for gid, rep in t.genotypes.items():
        st = stage_of(rep)
        out[str(gid)] = NORMAL_MUTED.get(getattr(rep, "type", None), (0.7, 0.7, 0.7, 1.0)) \
            if st is None else STAGE_COL[st]
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
    t = GenotypeTumor(seed=3, genome_params=GENOME, selection_params=SELECTION, cancer_cell_params=CANCER,
                      deme_params=DEME, spatial_params=SPATIAL, update_mode="tau", tau=1.0)
    frames = []

    def capture(phase):
        frames.append((copy.deepcopy(t.demes), dict(t.genotypes_counts), len(t.traces), phase))

    capture("growth")
    seeding_snap = None
    while True:
        p, m = compartment_cancer(t)
        if m >= 260 or p + m >= 5600:
            break
        t.grow(n_steps=1, seed=3)               # fine cadence -> a smooth DCIS -> IDC -> seeding arc
        if seeding_snap is None and compartment_cancer(t)[1] > 0:
            seeding_snap = len(t.traces)
        capture("growth" if seeding_snap is None else "metastatic seeding")

    resect_snap = len(t.traces)
    t.grow(n_steps=1, seed=3, treatment=Surgery(start=t.step, site="primary"))
    capture("resection")
    t.grow(n_steps=1, seed=3)
    capture("resection")

    chemo_snap = len(t.traces)
    chemo = Chemotherapy(start=t.step, duration=6, kill_rate=1.4, effectiveness=0.95,
                         toxicity=0.12, sites="both")
    for _ in range(6):
        t.grow(n_steps=1, seed=3, treatment=chemo)
        capture("chemotherapy")
    chemo_end_snap = len(t.traces)
    for _ in range(10):
        t.grow(n_steps=1, seed=3)
        capture("relapse")

    marks = [(seeding_snap, "seeding"), (resect_snap, "resection"),
             (chemo_snap, "chemo start"), (chemo_end_snap, "chemo end")]
    p, m = compartment_cancer(t)
    print(f"arc grown: primary={p} met={m}  frames={len(frames)}  traces={len(t.traces)}  marks={marks}")
    return t, frames, marks


def build_figure(t, marks, colors):
    """Persistent figure: two cell-resolution grids on top, the static 2-band Muller full-width below.
    The Muller is drawn ONCE; each frame only moves the reveal cursor over it (no band flicker)."""
    fig = plt.figure(figsize=(7.4, 6.9), dpi=115)
    fig.patch.set_facecolor(BG)
    gs = GridSpec(3, 2, height_ratios=[1.5, 0.6, 0.6], hspace=0.32, wspace=0.10,
                  left=0.035, right=0.985, top=0.815, bottom=0.065)
    axp = fig.add_subplot(gs[0, 0])
    axm = fig.add_subplot(gs[0, 1])
    ax_mp = fig.add_subplot(gs[1, :])
    ax_mm = fig.add_subplot(gs[2, :], sharex=ax_mp)

    # draw the full-arc 2-band Muller once (semantic clone colours; min_freq collapses the passenger
    # rainbow so the sweeps read as bands). Founder-coloured bands -> a band's colour is its clone's.
    viz.plot_muller_compartments(t.traces, t.genotypes_parents, axes=(ax_mp, ax_mm),
                                 driver_map=t._functional_signatures(), min_freq=0.05,
                                 clone_colors=colors, mark_generations=marks, star_genotype=None)
    xmax = len(t.traces) - 1
    for ax, band in ((ax_mp, "primary"), (ax_mm, "metastasis")):
        ax.set_facecolor(PANEL)
        for sp in ax.spines.values():
            sp.set_color(GRID_LINE)
        ax.tick_params(colors=SUBINK, labelsize=6.5, length=2)
        ax.set_xlim(0, xmax)
        ax.set_ylabel(f"{band}\ncells", color=SUBINK, fontsize=7.5)
        for txt in ax.texts:                     # recolour the event labels for the dark theme
            txt.set_color(INK); txt.set_fontsize(6.5)
    ax_mp.set_xlabel("")
    ax_mm.set_xlabel("snapshot (tumour time)", color=SUBINK, fontsize=7.5)
    if ax_mp.legend_:
        ax_mp.legend_.remove()
    return fig, axp, axm, ax_mp, ax_mm, xmax


def draw_grid(ax, sub, gsz, colors, title, n_cells):
    viz.plot_grid(sub, gsz, None, None, color=["cell_type"], ax=ax, expand_demes=True,
                  section_frac=0.85, cancer_color=None, type_cmap=colors, empty_color=PANEL)
    if ax.legend_:
        ax.legend_.remove()
    ax.set_title(f"{title}   ({n_cells:,} cells)", color=INK, fontsize=9, pad=4)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_color(GRID_LINE)


def main():
    t0 = time.time()
    t, frames, marks = grow_arc()
    colors = story_colors(t)
    mark_x = {lbl: x for x, lbl in marks}

    fig, axp, axm, ax_mp, ax_mm, xmax = build_figure(t, marks, colors)

    # a single stage legend + a phase caption, drawn on the figure
    legend_handles = [Patch(facecolor=STAGE_COL[s], edgecolor="none", label=f"{i+1}. {STAGE_LABEL[s]}")
                      for i, s in enumerate(STAGE_ORDER)]
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.935),
               ncol=5, fontsize=6.8, frameon=False, labelcolor=INK, handlelength=1.0,
               columnspacing=1.1, handletextpad=0.4)
    title = fig.text(0.5, 0.986, "iscc  —  one tumour: a primary seeds a clonally-linked metastasis",
                     ha="center", va="top", color=INK, fontsize=11.5, weight="bold")
    caption = fig.text(0.5, 0.882, "", ha="center", va="top", color=SUBINK, fontsize=8.5)

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
        for ax, val, gsz, ttl in ((axp, 0, t.grid_size, "primary duct field"),
                                  (axm, 1, t.met_grid_size, "metastasis")):
            mask = comp == val
            sub = {k: df[mask] for k, df in cd.items()}
            draw_grid(ax, sub, gsz, colors, ttl, int(mask.sum()))
        # progressive reveal cursor over the static Muller
        for a in cursor_artists:
            a.remove()
        cursor_artists.clear()
        for ax in (ax_mp, ax_mm):
            cursor_artists.append(ax.axvspan(tl, xmax, color=BG, alpha=0.82, zorder=5))
            cursor_artists.append(ax.axvline(tl, color="#f0f6fc", lw=1.3, alpha=0.95, zorder=6))
        cap = {"growth": "① proliferation in the ducts (DCIS)",
               "metastatic seeding": "①→② breach the duct wall, survive the stroma, seed the met (IDC)",
               "resection": "the primary is surgically resected",
               "chemotherapy": "systemic chemotherapy — the sensitive metastasis regresses",
               "relapse": "④ resistant clones relapse the metastasis"}[phase]
        caption.set_text(cap)
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
    per = [170] * len(imgs)
    per[0] = 900          # linger on the DCIS start
    per[-1] = 1900        # linger on the relapsed metastasis before looping

    gif_path = os.path.join(ASSETS, "landing_hero.gif")
    imageio.mimsave(gif_path, imgs, format="GIF", duration=per, loop=0)  # ~6 fps + holds
    size_mb = os.path.getsize(gif_path) / 1e6
    dur = sum(per) / 1000.0
    print(f"wrote {gif_path}  ({size_mb:.2f} MB, {len(imgs)} frames, ~{dur:.1f}s)")

    # poster PNG (final state) + representative frames for the report
    poster = os.path.join(ASSETS, "landing_hero_poster.png")
    imageio.imwrite(poster, imgs[-1])
    print("wrote", poster)
    for phase, fi in representative.items():
        p = os.path.join(FRAMES, f"frame_{fi:02d}_{phase.replace(' ', '_')}.png")
        imageio.imwrite(p, imgs[fi])
        print("  representative:", p)
    plt.close("all")
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
