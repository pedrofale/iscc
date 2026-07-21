# Handoff prompt — migrate the showcase notebook suite onto the DUCTAL FIELD + add the DCIS→IDC / genetic–niche confound story

Saved 2026-07-21. Copy the block below into a fresh session. **Docs/pedagogy work, not an engine change.**
This EVOLVES the existing executed notebook suite (built by `663cb00`, on the OLD single central-ring
geometry) onto the **ductal-field substrate + compartment-dependent selection** that landed afterwards, and
adds the flagship **genetic-vs-niche expression confound** as its own notebook. Branch from `dev`.

**Read `handoffs/showcase_notebooks.md` FIRST** — it is the parent handoff (the shared-substrate discipline,
the MEMORY/RAM notes, the API cheat-sheet, the mixed-microenvironment rule, the acceptance criteria). This
handoff only says what CHANGES; do not re-derive the parts that carry over.

## What changed since the notebooks were built (so you don't rediscover it)
The notebooks predate these — they run on a single central ring and know nothing about glands/compartments:
- **Ductal-field substrate** (`58fccf8`, `DESIGN_ductal_field.md`): the structured case is now a FIELD of many
  small epithelial-ring glands at 2D positions in **moderate-density stroma** (island model). New
  `spatial_params`: `n_glands`, `gland_radius`, `min_gland_sep`, `K_duct`, `K_stroma`, `stroma_fill_frac`
  (0.3–0.5 = real stromal cells that carry the stromal hazard), `cross_gland_kappa` (island dispersal:
  lumen→lumen, bypasses the wall), `cross_gland_lambda`. `n_glands=1 + stroma_fill_frac=1.0` = the old single
  ring, byte-identical. Per-cell ground truth `cell_data["cell_gland"]` (gland index, −1 = stroma).
- **Compartment-dependent selection v1** (`be6f427`, `DESIGN_phenotype_plasticity.md` §2): two new gene-based,
  sequenceable traits — `prop_breach`, `prop_stromal_survival` (+ `breach_effects`, `stromal_survival_effects`)
  — attenuate two LIVE-fraction death terms (`epithelial_barrier·epithelial_fraction·(1−breach)`,
  `stromal_hazard·stromal_fraction·(1−stromal_survival)`; `epithelial_barrier`/`stromal_hazard` in
  `spatial_params`, default 0). DCIS→IDC emerges: a lumen founder is confined by the wall + hostile stroma,
  spreads between glands as related foci, and invades only once a subclone evolves the escape traits.
- **The genetic–niche emt confound**: the invasive `emt` program is driven by BOTH `breach` (route-1 genetic
  arm, now in `DEFAULT_PHENOTYPE_PROGRAM_MAP`) AND the epithelial compartment field (route-3 niche arm, via
  `coupling_params["niche_program_map"] = {"epithelial": "emt"}`). `prop_dispersal=0`, so the legacy
  `dispersal_rate→emt` map is inert — do NOT rely on it. `breach`∈[0,1) sweeps to near-fixation, so give it a
  dedicated program gain (`phenotype_program_strength={"breach": ~1.0, "__default__": 0.5}`) for a substantial
  genetic arm.
- **Cell-resolution `plot_grid`** (`6d879d3`): `tumor.plot_grid(..., expand_demes=True, section_frac=…)` draws
  each deme as a block of its individual cells (a section slice); it also colours by gland/program. Use it.
- **Spatial diagnostics** (`299d5b3`): `validation/validate_spatial_diagnostic.py` (clonal territories,
  selection, CNA, programs) is a good pattern source.

**Reference configs that already work (copy their parameters, don't reinvent):**
- `validation/validate_ductal_field.py` — the substrate: `n_glands=4, gland_radius=3, min_gland_sep=8,
  K_duct=25, K_stroma=25, stroma_fill_frac=0.35, cross_gland_kappa=0.04` on `GRID=20`. Multi-focal spread from
  ONE founder + the inter-gland phylogeography (relatedness) panel.
- `validation/validate_compartment_selection.py` — the compartment + confound: `prop_breach=0.03,
  prop_stromal_survival=0.03, breach_effects=2.2, stromal_survival_effects=2.2, prop_dispersal=0`,
  `epithelial_barrier=1.2, stromal_hazard=0.7`, and the EXPR block (`niche_program_map={"epithelial":"emt"}`,
  `niche_program_strength=3.0`, `phenotype_program_strength={"breach":1.0,"__default__":0.5}`). Its three
  panels (DCIS→IDC grid series; escape-trait selection at the front; the 65% niche / 35% genetic confound with
  genotype-controlled r≈0.40) are the science NB3 below turns into a mixed-tumour notebook.

---

```
Migrate the iscc showcase notebook suite in notebooks/ onto the DUCTAL-FIELD substrate + compartment-dependent
selection, and add a new notebook for the genetic-vs-niche expression confound. The notebooks were built by
663cb00 on the OLD single central-ring geometry; the ductal field (58fccf8) and compartment selection (be6f427)
landed afterwards. READ handoffs/showcase_notebooks.md FIRST (the parent handoff: shared-substrate discipline,
RAM notes, API cheat-sheet, mixed-microenvironment rule, acceptance) — this task only changes the SUBSTRATE and
adds ONE notebook; everything else in that handoff still holds. Also read DESIGN_ductal_field.md and
DESIGN_phenotype_plasticity.md §2. Branch from `dev`.

REPO & ENV (same as parent handoff)
- Repo: /Users/pedroferreira/projects/iscc/repo (branch `dev`). Python/Jupyter: ~/miniconda3/envs/iscc/bin/python.
- Execute every notebook end-to-end in the core env (jupyter nbconvert --to notebook --execute --inplace) so
  outputs are saved; NO errors. Commit on `dev` WITH `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
  Do NOT touch mkdocs.yml or docs/tutorials/ (notebooks/ only). Do NOT change engine code — notebooks + base_sim.py
  only. Keep the test suite green (you only add/execute notebooks). Be honest about weak reconstructions.

TASK 1 — migrate notebooks/base_sim.py to the DUCTAL FIELD (this propagates to every notebook)
Rewrite the shared substrate so all notebooks inherit the ductal field + compartment selection + the confound.
Keep the parent handoff's structure (grow_base_tumor / grow_cohort, tau-leaping, shared layout_seed, EXPR() via
programs_common). Change the config to a multi-gland field with compartment selection ON. Start from the
validate_compartment_selection.py parameters and SCALE UP to reach the hard target (≥10k cancer cells):
  - SPATIAL/FIELD: a bigger grid + MORE glands than the GRID=20/4-gland validation (e.g. grid_size≈40–60,
    n_glands≈8–20, gland_radius≈3, min_gland_sep so glands don't touch), K_duct/K_stroma MODERATE (captures the
    duct's 3D depth; ~25–60), stroma_fill_frac≈0.35, cross_gland_kappa>0 (island spread), cross_gland_lambda.
  - SELECTION: prop_driver, prop_breach>0, prop_stromal_survival>0 (+ *_effects), prop_dispersal=0,
    prop_immune_resistance small (immune is part of the microenvironment). VERIFY breach/stromal-survival genes
    actually EXIST at your genome size (t.selection.get_breach()/get_stromal_survival() non-empty) — bump the
    prop or n_segments if a proportion rounds to 0 genes.
  - Compartment barriers ON: epithelial_barrier>0, stromal_hazard>0 — but TUNE so the tumour BOTH shows a DCIS→
    IDC transition (some subclone breaches into stroma) AND still reaches ≥10k cancer cells. Barriers confine
    growth (the validation got ~6k with barriers on a small grid), so a bigger field / more glands / longer grow
    is how you hit ≥10k WITH the barriers on. If you truly cannot get both, prefer a larger field over weakening
    the barriers, and say what you did.
  - EXPR(): keep programs + allele-specific ON, and ADD the confound coupling:
    coupling_params["niche_program_map"]={"epithelial":"emt"}, niche_program_strength≈3.0,
    phenotype_program_strength={"breach":1.0,"__default__":0.5}. (breach→emt is already in the default phenotype
    map; the niche map adds the epithelial arm.)
Sanity-print in a scratch run: cancer≥10k, purity, #glands colonised from the one founder, mean breach at the
wall vs control, #breach/#stromal-survival genes, that cell_gland and cell_program exist.

TASK 2 — rewrite notebooks/base_simulation.ipynb for the MULTI-FOCAL DUCTAL FIELD
The base notebook must now SHOW the ductal field, not a single ring. Cells:
  1. MD intro: the shared tumour is a multi-focal DCIS→IDC lesion on a ductal field (many small glands in
     moderate-density stroma); list features on (island substrate + microenvironment, compartment selection,
     CINner drivers, WGD, allele-specific expression, gene programs, the emt confound); ≥10k cancer cells.
  2. grow_base_tumor(); print malignant/normal counts, purity, WGD fraction, #genotypes, #glands invaded from
     ONE founder; assert cancer≥10000.
  3. MANDATORY GROWTH GRID TIME-SERIES: tumor.plot_grid at 4–6 timepoints from seeding to final, coloured by
     cell type (cancer vs epithelial vs stromal) AND a second row coloured by cell_gland — so the multi-focal
     spread from one founder across glands, then breakout into stroma (DCIS→IDC), is visually obvious. Use
     expand_demes=True for a cell-resolution view. (The user reviews these grid plots to evaluate the work — a
     run that shows no growth grid series is INCOMPLETE.)
  4. clonal dynamics: tumor.plot_muller(ax=..., min_freq=0.03) (min_freq is REQUIRED — infinite-sites → tens of
     thousands of clones; optionally by_drivers=True to colour by driver combinations).
  5. the ground-truth matrices (cell_snv/cell_cnv/cell_exp heatmaps) + one line each on what downstream
     notebooks read; note the new cell_gland and compartment-selection traits.
  6. MD: table mapping each downstream notebook to the aspect it uses; Next pointers.

TASK 3 — NEW notebook notebooks/compartment_selection_confound.ipynb (the flagship new science)
The pedagogical, mixed-tumour version of validate_compartment_selection.py — DCIS→IDC selection + the
genetic-vs-niche emt confound, analysed as a real study would (never pre-filter to cancer). Cells:
  1. MD: two normal compartments (gland wall, stroma) each impose a hazard a cancer clone must evolve a trait to
     survive; iscc knows the genotype AND the niche, so it can expose a confound no real dataset can.
  2. grow_base_tumor(); spatial viz — plot_grid coloured by cell_gland + compartment, and by the breach /
     stromal_survival traits, showing lumen-confined DCIS foci vs a stroma-invading subclone.
  3. DCIS→IDC: fraction of cancer in the stroma over time (essentially zero while confined) vs a barrier-OFF
     control grown with epithelial_barrier=stromal_hazard=0 (invades immediately). Mean breach at the wall vs in
     the control; mean stromal_survival in invaded stroma vs control (the escape traits are selected where their
     barrier acts).
  4. THE CONFOUND: run scRNA on the MIXTURE; take the invasive (emt) program activity for MALIGNANT cells; show
     it rises with the epithelial fraction of a cell's niche EVEN controlling for genotype (partial correlation),
     so "invasive expression ⇒ invasive genotype" is confounded by location. Because iscc generates both
     contributions, quantify the genetic-vs-niche variance split (as the validation does). Pointer to
     validation/validate_compartment_selection.py.

TASK 4 — update the SPATIAL notebooks to exploit the ductal field
- notebooks/scrna_visium_integration.ipynb: the Visium section now straddles MULTIPLE glands + stroma (multi-
  focal). Show spots that mix malignant + epithelial + stromal at gland/stroma boundaries; deconvolve as before,
  but now the true per-spot composition spans cell types AND foci. GOTCHA: iscc's Visium currently pools ALL
  cells within spot_radius (no depth/section slice — DESIGN_ductal_field.md §3.1 is a pending ENGINE TODO, not
  built), so with large K a spot over-fills. Keep K moderate and/or subsample cells to a realistic per-spot
  count before/at the assay, and NOTE this limitation in the notebook. Do NOT try to fix the engine here.
- notebooks/tree_inference_dna.ipynb: ADD the phylogeography angle the ductal field enables — the inter-gland
  colonisation is a TRUE spread tree (reconstruct which gland seeded which from cell_gland + genotypes_parents),
  and the multi-focal foci are clonally RELATED (shared truncal mutations, between-focus divergence from the
  island bottleneck). Show foci relatedness alongside the existing bulk/scDNA clone-tree reconstruction; keep
  the tree methods lightweight (scipy/sklearn/Bio.Phylo). Pointer to validate_multiregion_phylo.py.

TASK 5 — re-execute the remaining notebooks on the new substrate (they inherit it via base_sim)
combining_scdna_scrna, wgd_allele_cna, gene_programs, cohort_shared_programs, cohort_mhn_recurrence: their
SCIENCE is unchanged, but they must re-run on the ductal field and update their spatial viz to the multi-focal
picture (colour by cell_gland where a spatial panel appears). Two small additions:
  - gene_programs.ipynb: now that the emt program has a niche arm, add a short cell/section showing the invasive
    program is driven by BOTH genotype (breach) and niche (epithelial fraction) — a mini version of the Task-3
    confound — so the programs notebook reflects the new coupling. Keep the NMF recovery story.
  - cohort_*: unchanged thesis (shared programs / recurrent drivers across 5 patients); just grow on the ductal
    field and keep the shared-layout_seed comparability. Guarded external-tool envs as in the parent handoff.

HARD REQUIREMENTS (carry over from the parent handoff, still binding)
- ≥10k cancer cells per single-tumour sim (and per cohort member); spatial structure ON; the MICROENVIRONMENT is
  part of the data — analyse the MIXTURE, never pre-filter to cancer (ground truth only scores/colours). No
  pseudobulk anywhere — use multi-cell count observations. Self-contained in the core env for single-tumour
  notebooks; guarded external-tool envs for the two cohort notebooks. Every notebook executed with outputs.
- GRID PLOTS ARE MANDATORY and are how the user evaluates the work: base_simulation AND
  compartment_selection_confound MUST each show a plot_grid GROWTH TIME-SERIES (several timepoints, coloured by
  cell type + cell_gland/compartment), read back and displayed in the final report. Numbers without grid plots =
  incomplete.

GOTCHAS / HONEST NOTES
- Barriers vs scale: strong barriers confine DCIS and fight the ≥10k target — reach scale with a bigger field /
  more glands, not by neutering the barriers. If both are impossible, enlarge the field and say so.
- Breach genes must exist at your genome size (proportions can round to 0 genes) — verify get_breach()/
  get_stromal_survival() are non-empty.
- Visium pools all cells in a spot (no section slice yet) — keep K moderate / subsample; note it; don't fix the
  engine.
- prop_dispersal=0, so dispersal_rate→emt is inert; the emt genetic arm is breach (already in the default map)
  and needs the phenotype_program_strength breach gain to be visible.
- RAM: a materialised ductal-field tumour with all layers is ~1–2 GB; for the cohort grow→assay→release one at
  a time (parent handoff MEMORY note).

DELIVERABLES: rewritten base_sim.py (ductal field + compartment selection + confound coupling); rewritten
base_simulation.ipynb (multi-focal, growth grid series); NEW compartment_selection_confound.ipynb; updated
scrna_visium_integration.ipynb + tree_inference_dna.ipynb (multi-focal / phylogeography); re-executed
combining_scdna_scrna, wgd_allele_cna, gene_programs (+ confound cell), cohort_shared_programs,
cohort_mhn_recurrence — all EXECUTED, ≥10k cancer cells, mixed microenvironment, spatial viz. A one-line
BACKLOG.md note. Commit on `dev` with the Co-Authored-By trailer. In the FINAL REPORT, embed the base_simulation
and compartment_selection_confound growth grid images so the user can SEE the multi-focal DCIS→IDC growth.
```
