# Handoff prompt — the docs landing-page growth animation (primary + metastasis grids + Muller)

Saved 2026-07-24. Copy the block below into a fresh session. **Docs/viz work.** Deliverable: the animation
that replaces the placeholder hero on the docs landing page — the **primary tumour grid**, the
**metastasis grid**, and the **Muller plot(s)** growing over time, side by side, on one shared clone
colormap. Branch from `dev`.

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
Muller plot(s) growing over time, side by side on ONE shared clone colormap, to replace the static hero
placeholder. READ notebooks/metastasis_demo.py FIRST (the static version of exactly this scene). Branch from `dev`.

REPO & ENV
- Repo: /Users/pedroferreira/projects/iscc/repo (branch `dev`). Python: ~/miniconda3/envs/iscc/bin/python.
- Conventions: commit on `dev` WITH `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`; be honest; do not
  break the test suite (this is viz/docs — you should not need to touch engine code; if you do, keep it green).

THE SCENE (what animates, per frame)
A wide, dark-themed hero (match the placeholder aesthetic: bg ~#0d1117, panel ~#161b22, light labels) with three
panels growing together over the tumour's life:
  1. PRIMARY tumour deme-grid (the ductal field) — coloured by clone (functional clone / driver combo).
  2. METASTASIS deme-grid — SAME clone colormap, so a clone is the same colour in both grids (the met founder
     inherits its primary clone's colour). It appears only after the seeding event, then grows.
  3. MULLER — the clonal dynamics revealed progressively up to the current time (a growing Muller, or the full
     2-band primary-over-metastasis Muller with a moving time cursor). Same clone colours as the grids.
Optionally annotate the key events over time (seeding -> resection -> chemo -> relapse), as metastasis_demo.py does.

API (all in metastasis_demo.py + src/iscc/tumor/viz.py — reuse, don't reinvent)
- Grow with the demo's config: GenotypeTumor(spatial_params=SPATIAL, ...) where SPATIAL carries BOTH the ductal
  field (grid_size, n_glands, K_duct, ...) AND the met deposit (met_grid_size, K_met, host_fill_frac,
  met_seed_kappa, met_hazard, met_transit_floor); SELECTION adds prop_met_survival / met_survival_effects. Copy
  metastasis_demo.py's GENOME/SELECTION/CANCER/DEME/SPATIAL verbatim as the starting point (seed=3, tau-leaping).
- t.demes: the first t.n_primary_demes demes are the PRIMARY grid; the rest are the MET grid. metastasis_demo.py's
  compartment_cancer(t) splits primary/met cancer counts — use it to detect the seeding frame (met cancer first >0).
- t.grow(n_steps, seed, treatment=...) grows and appends to t.traces; Surgery(site="primary") and
  Chemotherapy(...) drive the resection/chemo phases (see the demo's arc).
- Grids: viz.plot_grid_compartments(cell_data, primary_grid_size, met_grid_size, traces, genotypes_parents,
  color=["cancer_frac" or a clone key]) draws the two grids side by side (t.plot_grid_compartments(...) is the
  bound form). For a MOVING grid you need the per-timepoint deme composition — grow in small increments and render
  the current grid each increment (plot_clone_grid_series / _expanded_cell_grid in viz.py show how a grid is drawn
  from deme state; expand_demes=True gives the cell-resolution look).
- Muller: viz.plot_muller_compartments(traces, genotypes_parents, by_drivers=True, min_freq=0.05,
  mark_generations=marks) draws the 2-band primary-over-met Muller across the whole arc. min_freq is REQUIRED
  (infinite-sites => thousands of clones); by_drivers=True colours by functional clone so sweeps are legible.
- SHARED COLORMAP is the whole point: build ONE clone->colour map and pass it to all three panels so a clone reads
  the same everywhere. metastasis_demo.py already does this for the two static figures — mirror it.

FRAME CAPTURE (recommended approach)
Grow the arc in small steps; at each step render one combined figure (the 3 panels via subplots on the shared
colormap) and capture it as a frame. Concretely: build the config -> loop { t.grow(n_steps=1 or 2); t.make_cell_data();
render primary grid + met grid + Muller-so-far into a 16:5-ish figure; append the frame }. Cover the arc through
seeding, growth, resection, chemo, relapse (as in the demo). Hold a few frames on the final state so the loop reads.
Keep the tumour SMALL enough to render many frames fast (the demo uses grid_size=26, met_grid_size=12 — good for a
web hero; do NOT scale to 10k-cell notebook sizes here, this is a visual, not a benchmark).

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
— if a clone is a different colour in different panels the scene is misleading. Keep the tumour small (web hero,
fast render). If a seamless loop is hard, hold on the final frame instead. Keep the asset self-contained under
docs/assets/ (no external URLs — the site is offline-buildable). Report the real file size; if a GIF blows past a
few MB, switch to option B rather than shipping a huge GIF.
```
