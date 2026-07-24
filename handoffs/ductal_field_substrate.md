# Handoff prompt — ductal-field spatial substrate (multi-focal DCIS, island model)

Saved 2026-07-18. Copy the block below into a fresh session. **Prerequisite for the revised
compartment-selection v1** (`handoffs/compartment_selection_v1.md`). Full design in `DESIGN_ductal_field.md`
(READ FIRST). This is a `GenotypeTumor` (count-engine) spatial-substrate change ONLY — no selection axes here.
OFF-BY-DEFAULT / byte-identical when off. Branch from current `dev`.

---

```
Build the ductal-field spatial substrate for iscc per DESIGN_ductal_field.md (READ FIRST). Count engine
(GenotypeTumor / count.py) only. This replaces the single central-ring geometry with MANY small glands at 2D
positions (an island model), so the structured case is anatomically reasonable and usable for spatial
transcriptomics. NO selection axes in this handoff (that is handoffs/compartment_selection_v1.md, built on
top). Keep it OFF-BY-DEFAULT and byte-identical when off.

REPO & ENV
- Repo: /Users/pedroferreira/projects/iscc/repo (branch `dev`). Python/pytest: ~/miniconda3/envs/iscc/bin/python.
- Conventions: commit on `dev` WITH `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`; keep the FULL
  suite green; OFF-BY-DEFAULT / byte-identical-when-off (verify with a test); be honest.

WHY (DESIGN_ductal_field.md §0–§1): real breast is a branching ductal tree of many small ducts; multi-focal
DCIS is ONE clone spreading through connected glands, so a section shows several EVOLUTIONARILY RELATED foci.
We model this as a population-genetics ISLAND MODEL: many small glands, fast local growth, and a LOW cross-
gland migration rate standing in for intraductal spread through the (out-of-plane) ductal tree — no drawn
graph, the rate IS the connectivity. This is also the ST-realistic layout (separate foci in stroma, connecting
ducts out of plane and not drawn).

PART 1 — GEOMETRY (rewrite _seed_structure, count.py:360)
Today _seed_structure seeds ONE ring at the grid centre. Generalise to N glands:
- Config (spatial_params): n_glands (default 1), gland_radius (default = current structure_radius),
  min_gland_sep, K_duct, K_stroma, stroma_fill_frac (∈(0,1], default 1.0 = current behaviour). n_glands=1 +
  stroma_fill_frac=1.0 MUST reproduce the current single-structure seeding byte-identically.
- Place n_glands centres in the grid interior (use self.rng — the LAYOUT rng, for cohort comparability),
  rejection-sampled to be ≥ min_gland_sep apart and to fit (radius+margin) inside the grid.
- Each gland = a small epithelial ring: bresenham_circumference(center, gland_radius) seeded to K_duct with
  the "epithelial" normal genotype (reuse _normal_genotype/_add as now); the INTERIOR (get_inside) is the
  lumen — leave empty (DCIS-growable). Multiple demes per gland (ring + lumen) ⇒ within-gland neighbour
  dispersion.
- Stroma: every deme not belonging to a gland gets the "stromal" genotype at round(stroma_fill_frac*K_stroma)
  cells — MODERATE density (stroma_fill_frac≈0.3–0.5; stroma is less dense than epithelium but NOT empty — real
  stromal cells that the assays capture AND that carry the stromal hazard), leaving headroom for invasion.
- Founder: ONE cancer founder in ONE gland's lumen (pick a lumen deme of gland 0).

PART 2 — PER-DEME GLAND LABELS (ground truth + targeting)
- Record self.gland_id: np.array over demes, = gland index for that gland's ring+lumen demes, -1 for stroma.
  Also keep self.gland_lumen_demes (list per gland) and self.gland_centers (for distance weighting). Surface
  gland_id per cell in cell_data (e.g. cell_data["cell_gland"], like cell_microenv) so downstream benchmarks +
  the ST/phylogeography analyses can use it. Gate it on n_glands being set / structure on, mirroring the F8
  discipline (absent otherwise → base schema unchanged).

PART 3 — CROSS-GLAND (ISLAND) DISPERSAL (count.py dispersal branch, ~line 734, and the tau path ~line 576)
Today a dispersing daughter goes to a uniformly-random NEIGHBOUR (no capacity gate; crowding is via death).
Add a second channel:
- Config: cross_gland_kappa (default 0.0 → OFF → byte-identical), cross_gland_lambda (distance kernel).
- For a clone in a GLAND deme, split its dispersing daughters: a fraction ~ kappa go CROSS-GLAND, the rest
  stay local (neighbour). Concretely, the daughter disperses at the existing dispersal_rate; of those, draw
  Binomial(n_disp, kappa/(1+kappa)) [or scale cross_gland_rate = kappa·dispersal_rate as its own Poisson
  channel] to be cross-gland. A cell in STROMA (gland_id==-1) has NO cross-gland channel (intraductal spread
  is gland→gland only).
- A cross-gland daughter lands in a LUMEN deme of ANOTHER gland, chosen distance-weighted over gland centres
  (prob ∝ exp(-d/cross_gland_lambda); d from the source gland's centre) — or uniform if lambda is None. It is
  CONFINED (lumen→lumen), so it needs NO breach (breach lives in the selection handoff and gates only local
  escape into stroma). Use _add to the chosen lumen deme; genotypes_parents already records the lineage, so
  the cross-gland hops become the inter-gland spread tree (ground truth).
- Do this in BOTH the exact-update dispersal resolution and the tau-leaping path so the two engines agree.

PART 4 — CARRYING CAPACITY PER COMPARTMENT (K captures 3D depth — NOT a handful)
- A 2D deme stands for a 3D COLUMN (cross-section × duct/stroma depth), so K is a real subpopulation size,
  MODERATE-TO-LARGE — do NOT shrink it to a handful. Duct demes: K_duct moderate (captures depth), with a duct
  = a small ring of a few demes (lumen + wall) for within-duct structure. Stroma demes: K_stroma
  similar/modestly higher, seeded at MODERATE density (stroma_fill_frac≈0.3–0.5). Honour a PER-DEME K in the crowding death law (_death_rate /
  _resident_ref) — today carrying_capacity is a scalar; add a per-deme capacity array.
- ST resolution is a GRID-SPACING constraint, NOT a K bound (corrects DESIGN's earlier note): a Visium spot
  must cover SEVERAL demes to be a mixture — set by grid density vs spot_radius, INDEPENDENT of K. Keep the
  grid fine relative to the spot and K moderate-large; keep TOTAL cells sane via field size / n_glands. Large K
  is nearly free at growth (genotype COUNTS per deme); it only costs at make_cell_data / assay. n_glands=1 +
  uniform K = current single-structure run, byte-identical.

OFF-BY-DEFAULT / BYTE-IDENTICAL (critical)
n_glands=1, cross_gland_kappa=0, stroma_fill_frac=1.0, uniform K = the CURRENT single-structure spatial run,
EXACTLY. Add a byte-identical test (hash cell_snv/cell_cnv vs a pre-change baseline, like
test_wgd_off_is_byte_identical).

PARAMETERS.md: document n_glands, gland_radius, min_gland_sep, K_duct, K_stroma, stroma_fill_frac,
cross_gland_kappa, cross_gland_lambda (spatial section), with defaults = off/current.

VALIDATION (validation/validate_ductal_field.py -> manuscript/figures/validation_ductal_field.png):
Grow a multi-gland field (n_glands≈10–30, small gland_radius, moderate K, moderate-density stroma, cross_gland_kappa>0)
from ONE founder. Show:
  (A) multi-focal structure: plot_grid coloured by gland_id + cancer presence — several glands colonised from
      one founder; the fraction of glands invaded rises over time.
  (B) RELATEDNESS: the inter-gland spread is a tree — reconstruct which gland seeded which from genotypes_parents
      + gland_id (the ground-truth phylogeography); show foci are clonally related (shared truncal mutations)
      with between-focus divergence (island bottleneck).
  (C) scale sanity: per-gland cancer cell counts are realistic (tens–low-hundreds), total is sequenceable.
Print headline numbers.

GRID PLOTS ARE MANDATORY (the user reviews these to evaluate your work). The validation MUST produce a
tumor.plot_grid(...) GROWTH TIME-SERIES — a row of snapshots at several steps (e.g. 4–6 timepoints from
seeding to final) — saved as a PNG, showing the spatial tumour: colour by gland_id AND (a second row / panel)
by clone/cancer-vs-normal, so the multi-focal spread from one founder across glands is visually obvious. Save
to manuscript/figures/ and, in your FINAL REPORT, embed/display the saved image(s) (read them back) so the
user can actually SEE the growth — do not just report numbers. A run that prints stats but shows no grid plot
is INCOMPLETE.

TESTS (tests/test_ductal_field.py):
- OFF byte-identical (n_glands=1, kappa=0, fill=1, uniform K -> identical to the pre-change single structure).
- n_glands>1 seeds N disjoint rings + moderate-density stroma; gland_id labels + gland_lumen_demes correct; one founder.
- cross_gland_kappa>0: cancer reaches a DIFFERENT gland than the founder's (a cross-gland hop occurs), and its
  lineage traces back to the founder (genotypes_parents); a STROMA cell never initiates a cross-gland hop.
- per-deme K honoured (duct demes cap at K_duct, stroma at K_stroma).
- both engines (exact + tau) agree that cross-gland dispersal reaches other glands.

DELIVERABLES: multi-gland _seed_structure + gland_id/labels + cell_data["cell_gland"]; cross-gland island
dispersal (both engine paths); per-compartment K; PARAMETERS.md; validate_ductal_field.py + figure; tests; a
short manuscript note (ductal-field substrate: multi-focal DCIS via an island model, ST-usable); flip a
BACKLOG item. Full suite green; commit on `dev`.

HONEST NOTES: keep it to the SUBSTRATE — no breach/stromal selection here (that's the next handoff). If
n_glands=1 off-by-default is NOT byte-identical, STOP and fix (a new label/param is perturbing an existing
path). Do NOT build an explicit ductal graph or drawn connecting ducts — the cross-gland RATE replaces them
(DESIGN §6). 2D-section distance is only a proxy for ductal proximity, so distance-weighting is approximate;
uniform targeting is the zero-extra-parameter fallback — note whichever you use.
```
