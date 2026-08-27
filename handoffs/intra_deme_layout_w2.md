# Handoff prompt — W2: informative intra-deme cell layout

Saved 2026-08-27. Copy the block below as the opening message of a fresh session.
Design reference: `DESIGN_cci_spatial.md` section **W2** (and **W0** for what it feeds).

**Ordering corrected 2026-08-27: W2 is NOT a prerequisite for W3.** W3's receptor-dependence works at
deme resolution (the receptor term is already per-cell), and Visium spot coordinates are exact and
independent of cell jitter, so a Visium-resolution CCI benchmark ships without this. W2 is required
only for a SINGLE-CELL SPATIAL (imaging) benchmark, where co-deme cells currently sit at identical
coordinates. See `DESIGN_cci_spatial.md` for the corrected chain. W1 is SUPERSEDED — do not start it.

---

```
Implement W2 — INFORMATIVE INTRA-DEME CELL LAYOUT in iscc. Replace the uniform jitter that places a
deme's cells with a labelled point process conditioned on the deme's composition, and route BOTH
spatial assays through it. Design: DESIGN_cci_spatial.md section W2.

REPO & ENV
- Repo: /Users/pedroferreira/projects/iscc/repo  (branch `dev`).
- Python/pytest: ~/miniconda3/envs/iscc/bin/python  (`~/miniconda3/envs/iscc/bin/python -m pytest`).
- `conda run` is BLOCKED; call env binaries by absolute path.
- USE `git -C /Users/pedroferreira/projects/iscc/repo ...` FOR EVERY GIT COMMAND. The shell cwd
  resets between calls in this environment; a `cd` at the start of a command is not enough and has
  already caused one commit to land in the wrong repository.
- Conventions: commit on `dev`; `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>` trailer;
  keep the FULL suite green (currently 908 passed, 1 skipped); honest comments, no overselling.
- ASK BEFORE RUNNING ANY SIMULATION, and run them one at a time (16 GB machine).

WHAT EXISTS NOW
- `src/iscc/sample/section.py::spatialize(cell_data, section_frac=1.0, jitter=0.5, seed=0)` is the
  single canonical placement step. It scatters each deme's cells UNIFORMLY in the deme's unit cell
  (integer coord +/- jitter) and its docstring calls the intra-deme layout "cosmetic". It uses
  `cell_deme` (deme id) and `cell_crd` (integer deme coords) and carries every per-cell table over.
- Only ONE caller in `src/`: `data/visium.py:263`, inside the Visium assay.
- **VERIFIED GAP, and it widens this job:** the single-cell spatial (imaging) assay does NOT go
  through `spatialize` at all. `data/imaging.py:157` reads `cell_crd` verbatim, so every cell in a
  deme is emitted at the IDENTICAL integer coordinate — `obsm["spatial"]` has co-deme cells stacked
  at one point, with zero sub-deme resolution and pairwise distances of exactly 0. Any
  nearest-neighbour, Ripley's-K or contact-based analysis on the MERFISH/Xenium-like output is
  currently degenerate. W2 must route imaging through the placement layer too.

MEASUREMENT THAT MOTIVATES THIS (pure geometry, Visium v1: spot_radius 0.55, pitch 2.0 demes)
    cells/spot 11.2      distinct demes/spot 1.62 (max 4)
    fraction of a spot's cells from its DOMINANT deme: 0.94
    re-jitter only, identical deme composition:
      cells/spot correlation 0.10    mean |change| 1.5 of 11.2    deme-set Jaccard 0.61
Read it carefully: spot COMPOSITION is deme-resolution and robust, so do not expect it to move much.
What IS currently noise is spot DEPTH and — the point of this work — WHICH CELL TYPES CO-OCCUR inside
a spot, presently a random draw from the deme's composition. With 75% of spots multi-type in the
`rctd` dataset, the mixtures the deconvolution benchmark measures are unstructured today.

THE MODEL
A labelled point process evaluated at materialisation, conditioned on the deme's composition (counts
by clone and by coarse type — the engine already tracks these; clone from `cell_data["cell_type"]`,
coarse type via `tumor.genotypes[gid].type`, deme from `cell_data["cell_deme"]`):
  - a HARDCORE radius (cells may not overlap),
  - SAME-CLONE ATTRACTION (clonal patches within a deme),
  - a TYPE-PAIR INTERACTION MATRIX (attraction/inhibition; e.g. immune at the gland-stroma interface).
Sequential (dart-throwing) placement or a Gibbs/Strauss process. Cost is not a concern: it is O(n^2)
per deme and n is carrying-capacity-sized (8-59). This is sCCIgen's model applied INSIDE a deme,
conditioned on a composition that came from evolution — their spatial texture, our provenance, engine
untouched.

CONTRACT CHANGE — STATE IT, DO NOT HIDE IT
Composition stays EXACT; the layout is no longer cosmetic, and `spatialize`'s docstring must stop
saying it is. Keep the uniform mode behind a flag and DEFAULT TO CURRENT BEHAVIOUR so published
numbers do not move until someone re-runs deliberately.

WHAT A RE-RUN WILL RESTATE (budget it; do not do it casually)
- `analysis_data/rctd` + `notebooks/tool_rctd_R.ipynb` (currently MAE 0.058, mean per-type r 0.974).
- `validation/validate_deconvolution.py`, `validate_visium.py`, `validate_spatial_diagnostic.py`,
  and possibly `validate_petracer.py` (spatial autocorrelation). Check each before assuming.

TESTS / DONE
- Composition is preserved EXACTLY per deme (counts by clone and type unchanged).
- Hardcore radius respected; no two cells closer than the minimum.
- Same-clone nearest-neighbour fraction measurably above the uniform-placement baseline, and
  type-pair attraction/inhibition moves in the configured direction.
- Default-off is bit-identical to today.
- Imaging output no longer has co-deme cells at identical coordinates.
- Full suite green. Update `DESIGN_cci_spatial.md` W2 with what was built, and `BACKLOG.md`.

WHY IT MATTERS DOWNSTREAM (corrected)
This is what makes a SINGLE-CELL-RESOLUTION CCI benchmark possible. It is NOT required for a
Visium-resolution one: W3's effect is `strength x ligand_available x receiver receptor expression`,
and the receptor term is per-cell already, so per-cell response heterogeneity exists without per-cell
positions. What W2 adds is geometry below the deme — which contact-range methods and any
imaging-based analysis need, and which the imaging assay currently lacks entirely.
```
