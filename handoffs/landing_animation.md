# Handoff prompt — the docs landing-page growth animation (primary + metastasis grids + Muller)

> SUPERSEDED (historical). The landing hero now renders from the ordinary CLI —
> `isccsim --sim-config configs/landing.yaml` + `isccgif --compartment --splash` (seed 2; render logic
> in `iscc.visualization.compartment` / `iscc.tumor.arc`). The standalone `notebooks/landing_animation.py`
> described below was removed.


Saved 2026-07-24. Copy the block below into a fresh session. **Docs/viz work.** Deliverable: the animation
that replaces the placeholder hero on the docs landing page — the **primary tumour grid**, the
**metastasis grid**, and the **Muller plot(s)** growing over time on one shared clone colormap, laid out as two
**cell-resolution** grids on top and a full-width Muller below (see LAYOUT). Branch from `dev`.

Context you need:
- The landing page is a Material "splash": `docs/index.md` uses `overrides/home.html`, whose hero is a single
  `<img class="mdx-hero__image" src="{{ 'assets/landing_hero_placeholder.png' | url }}">`. **That `<img>` is the
  swap point.** Today it points at the static 3-panel placeholder `docs/assets/landing_hero_placeholder.png`
  (primary | metastasis | Muller, "animation coming soon"). Replace it with the real animation.
- The metastasis feature (commits 66c74e6 / 3acb9e9 / b34a6b8 / 712f61b) added a **second, clonally-linked
  deme-grid** (the metastatic deposit) + per-site treatment/resection, and viz for it. `notebooks/metastasis_demo.py`
  is the end-to-end demo (READ IT FIRST) — it grows the whole clinical arc and produces the two static
  acceptance figures the animation is the moving version of.

---

```
Build the docs landing-page growth ANIMATION for iscc: the primary tumour grid, the metastasis grid, and the
Muller plot(s) growing over time on ONE shared clone colormap, laid out as two CELL-RESOLUTION grids on top and a
full-width Muller below (see LAYOUT), to replace the static hero placeholder. READ notebooks/metastasis_demo.py
FIRST (the static version of exactly this scene). Branch from `dev`.

REPO & ENV
- Repo: /Users/pedroferreira/projects/iscc/repo (branch `dev`). Python: ~/miniconda3/envs/iscc/bin/python.
- Conventions: commit on `dev` WITH `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`; be honest; do not
  break the test suite (this is viz/docs — you should not need to touch engine code; if you do, keep it green).

THE SCENE (what animates, per frame)
A dark-themed hero (match the placeholder aesthetic: bg ~#0d1117, panel ~#161b22, light labels) laid out in TWO
ROWS (see LAYOUT), three panels growing together over the tumour's life:
  - PRIMARY tumour grid (the ductal field) and METASTASIS grid — both CELL-RESOLUTION: each deme expanded to a
    block of its INDIVIDUAL CELLS, NOT the deme-consensus / one-colour-per-deme view. Coloured by clone
    (functional clone / driver combo) on ONE shared colormap, so a clone is the same colour in both grids (the met
    founder inherits its primary clone's colour). The met grid is empty until the seeding event, then grows.
  - MULLER — the clonal dynamics revealed progressively up to the current time (a growing Muller, or the full
    2-band primary-over-metastasis Muller with a moving time cursor). Same clone colours as the grids.

LAYOUT (2 rows — use a matplotlib gridspec, NOT plot_grid_compartments' built-in side-by-side layout):
  Row 1 — TWO COLUMNS: primary grid (left) | metastasis grid (right), each cell-resolution, equal size.
  Row 2 — FULL WIDTH, spanning both columns: the Muller (the 2-band primary-over-met Muller may stack its two
    bands within this row).
  i.e. gridspec(2, 2): grids at [0,0] and [0,1]; Muller at [1, :]. Render each grid into its own top-row axis.
  This replaces the old single-row banner; the hero <img> scales to whatever aspect you render (roughly 3:2 / 4:3
  — two grids on top, wide Muller below). Keep it legible at hero size.

THE STORY THE ANIMATION MUST TELL (non-negotiable — this IS the deliverable, not decoration). The four
selection episodes of the metastatic arc must each be clearly legible, both as spatial progression in the grids
AND as selective sweeps in the Muller:
  1. ESCAPE THE DUCT — a `breach` subclone sweeps and cancer crosses the epithelial wall out of the gland lumen
     into the stroma (the DCIS -> IDC transition).
  2. SURVIVE THE STROMA — a `stromal_survival` subclone sweeps as cancer traverses the hostile stroma.
  3. ESTABLISH IN THE METASTASIS — a `met_survival` (prop_met_survival) subclone seeds the second grid and grows
     the metastatic deposit.
  4. ESCAPE CHEMO — after the primary is resected and systemic chemo regresses the met, a treatment-resistant /
     persister subclone survives and the met relapses.
Annotate all four events on the timeline (seeding, resection, chemo start/end), as metastasis_demo.py does. If
any of the four is not visibly readable in the finished animation, it has failed its purpose — tune the config
(selection strengths, event timing, size, grow duration) until all four read.

API (all in metastasis_demo.py + src/iscc/tumor/viz.py — reuse, don't reinvent)
- Grow with the demo's config: GenotypeTumor(spatial_params=SPATIAL, ...) where SPATIAL carries BOTH the ductal
  field (grid_size, n_glands, K_duct, ...) AND the met deposit (met_grid_size, K_met, host_fill_frac,
  met_seed_kappa, met_hazard, met_transit_floor); SELECTION adds prop_met_survival / met_survival_effects. Copy
  metastasis_demo.py's GENOME/SELECTION/CANCER/DEME/SPATIAL verbatim as the starting point (seed=3, tau-leaping).
- t.demes: the first t.n_primary_demes demes are the PRIMARY grid; the rest are the MET grid. metastasis_demo.py's
  compartment_cancer(t) splits primary/met cancer counts — use it to detect the seeding frame (met cancer first >0).
- t.grow(n_steps, seed, treatment=...) grows and appends to t.traces; Surgery(site="primary") and
  Chemotherapy(...) drive the resection/chemo phases (see the demo's arc).
- Grids (CELL-RESOLUTION, REQUIRED — show cells, not deme consensus): expand each deme into a block of its
  individual cells. viz.py's `_expanded_cell_grid` / `plot_grid(ax=..., expand_demes=True, section_frac=...)` do
  exactly this (section-sampled); use it, coloured by clone. If that path is currently wired only for cell-TYPE
  colouring, extend it to clone colouring (viz.py edits are fine — keep the suite green). Because the layout is the
  custom 2-row gridspec (LAYOUT), render EACH grid into its OWN top-row axis (ax=...); use plot_grid_compartments /
  compartment_cancer only as the reference for how primary-vs-met demes are split (the first t.n_primary_demes
  demes are primary, the rest are met). For a MOVING grid, grow in small increments and re-render the current
  cell-resolution grid each increment.
- Muller: viz.plot_muller_compartments(traces, genotypes_parents, by_drivers=True, min_freq=0.05,
  mark_generations=marks) draws the 2-band primary-over-met Muller across the whole arc. min_freq is REQUIRED
  (infinite-sites => thousands of clones); by_drivers=True colours by functional clone so sweeps are legible.
- SHARED COLORMAP is the whole point: build ONE clone->colour map and pass it to all three panels so a clone reads
  the same everywhere. metastasis_demo.py already does this for the two static figures — mirror it.

FRAME CAPTURE (recommended approach)
Grow the arc in small steps; at each step render one combined figure (the 3 panels via subplots on the shared
colormap) and capture it as a frame. Concretely: build the config -> loop { t.grow(n_steps=1 or 2); t.make_cell_data();
render the 2-row layout (LAYOUT) — the two CELL-RESOLUTION grids on the top row (two columns) and the full-width
Muller-so-far below — into one figure; append the frame }. Cover the arc through seeding, growth, resection,
chemo, relapse (as in the demo). Hold a few frames on the final state so the loop reads.

SIZE FOR THE STORY, NOT FOR SPEED. The tumour must be large enough and run long enough that all FOUR selective
sweeps above are clearly legible — that is the whole point (showing iscc's power and a plausible metastatic-
evolution story). Start from metastasis_demo.py's config and SCALE UP (grid_size / met_grid_size, K, grow
duration, and the selection strengths that make each sweep visible) as needed until the four sweeps read; do NOT
shrink the biology to save render time. Manage the WEB asset's weight instead through frame count, fps,
resolution, and encoding (see OUTPUT below) — it is a one-off OFFLINE render, not interactive, so a larger tumour
over more frames is fine. If render time becomes painful, coarsen the frame cadence (grow more steps per frame),
not the tumour.

OUTPUT FORMAT (decide, then wire it in)
Two good options — pick one and update overrides/home.html accordingly:
  (A) SIMPLE — animated GIF (or APNG) at docs/assets/landing_hero.gif; the hero stays an <img>, just repoint the
      src. Easiest; watch the file size (target < ~3-4 MB: cap frames/fps/resolution, quantize the palette).
  (B) BETTER quality/size — an MP4 (H.264) + WebM (VP9) pair + a poster PNG; change the hero <img> in
      overrides/home.html to <video class="mdx-hero__image" autoplay loop muted playsinline poster="...">
      <source ...></video>. Sharper and smaller for the same length, but needs ffmpeg and the template edit.
Assemble frames with imageio (GIF, no ffmpeg needed) or matplotlib.animation / ffmpeg (mp4). Whichever you choose,
the asset MUST live under docs/assets/ (mkdocs copies it into the site) and the page MUST still build.

WIRE-IN + VERIFY
- Put the asset in docs/assets/. Update the hero in overrides/home.html to point at it (repoint the <img src>, or
  switch to <video> for option B). You can leave landing_hero_placeholder.png in place or delete it once unused.
- Verify: `~/miniconda3/envs/iscc/bin/mkdocs build` is clean, then `mkdocs serve` and CONFIRM the animation plays
  on the Home page (the splash hero). In your FINAL REPORT, embed a couple of representative frames (and note the
  file size + duration) so the user can see it without running anything.

DELIVERABLES: a generator script (e.g. notebooks/landing_animation.py or docs-tooling, reusing metastasis_demo.py's
config + the compartment viz); the animation asset(s) under docs/assets/; the overrides/home.html hero repointed
at it; a one-line note in BACKLOG.md. Full suite green; commit on `dev`.

HONEST NOTES: the shared clone colormap across primary grid + met grid + Muller is the single most important thing
— if a clone is a different colour in different panels the scene is misleading. Size the tumour for the STORY: all
four selective sweeps (duct escape, stromal survival, met establishment, chemo escape) must be legible — that,
not a small file, is the deliverable; manage web weight via encoding / frames / resolution instead. If a seamless
loop is hard, hold on the final frame instead. Keep the asset self-contained under
docs/assets/ (no external URLs — the site is offline-buildable). Report the real file size; if a GIF blows past a
few MB, switch to option B rather than shipping a huge GIF.
```
