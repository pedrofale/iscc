# iscc BACKLOG — tracked threads

Living task tracker (started 2026-06-30). Durable across sessions. Each item: status, one-line
description, and a pointer to the design doc / handoff that scopes it. Status legend:
**NOW** (active priority) · **NEXT** (queued) · **LATER** (deferred) · **DONE** (kept briefly for context).

> Companion planning docs: `DESIGN_features.md` (F1–F8), `DESIGN_inference.md`, `DESIGN_recommender.md`
> (2nd paper), `RESEARCH_QUESTIONS.md` (R1–R11), `handoffs/` (copy-pasteable session prompts).

---

## Paper 1 — the core iscc software paper (PLoS Comp Biol)

**Thesis:** the first *internally consistent*, end-to-end tumor simulator (one selection-driven,
spatial, treatable tumor → bulk/sc DNA, scRNA, spatial, to reads), calibrated from real data, whose
consistency reveals failure modes single-modality benchmarks cannot. Claim = integration +
calibration + consistency-enabled insight. Do **not** claim "first spatial-evo→data" (CHESS/J-SPACE),
"first multimodal" (scMultiSim), or "richest CNA mechanism" (CINner/SISTEM). Full positioning in
memory `iscc-paper-positioning.md`.

**Benchmarking strategy = mirror scMultiSim's playbook** (user decision 2026-07-02; scMultiSim is
the primary Camp-2 exemplar): realism vs real + a **suite of ground-truth downstream-method
benchmarks** + capability breadth. NOT a KS-distance bake-off vs count sims (Camp-1 turf, we'd tie).
Pitch: "as scMultiSim provides ground truth for cell-state/regulatory tasks, iscc provides it for the
full tumor-evolution / cancer-genomics task spectrum." Do NOT build GRN/scATAC (their turf; deferred).

- **NOW (separate session) — the ground-truth benchmark suite** (run real methods on iscc data,
  score vs known truth). **Active next: `clonealign` + `inferCNV`** (handoff:
  `handoffs/clonealign_infercnv.md`):
  - **Flagship — `clonealign`** (CNA-dosage DNA↔RNA clone assignment; works today, non-circular). AUC vs true clone.
  - `inferCNV`/`copyKAT` (CNA-from-expression) — correlation vs true per-cell CNA.
  - Multi-batch **integration/correction** on F3 batches (Harmony/scVI) — iLISI/ARI vs known clones/types.
    *(subsumed by / graduates into the Multi-patient cohort milestone below.)*
  - Cell-type/clone **clustering** — ARI; **spatial deconvolution** (cell2location/Tangram) — per-spot accuracy.
  - Already in-paper (count as suite members): **selection/rate inference** (ABC recovery), **sampling/
    experimental-design**, realism-vs-real, **PEtracer** (done). Niche/**CCI** inference builds on F8.
  - Manuscript: validation reframed as *"iscc as a benchmarking substrate"* — DONE (PEtracer + microenv sections).
- **DONE — PEtracer validation** (flagship ground-truth + real-data benchmark). New
  `iscc.integrations` seam (`to_lineage_tree`/`to_newick`/`to_anndata`/`decompose_lineage_spatial`;
  19 tests) built on F8 (extrinsic) + F9 (single-cell spatial) + `genotypes_parents` lineage.
  **Tier 1 (self-contained, `validation/validate_petracer.py`):** the Hotspot-style lineage-vs-spatial
  autocorrelation decomposition on iscc data with a KNOWN split, EXPOSING the lineage-space CONFOUND.
  **Headline finding:** under clonal territories (low `dispersal_rate`) the purely-environmental
  hypoxia field acquires LINEAGE autocorrelation (0.16) that EXCEEDS its spatial autocorrelation
  (0.03), so a tree-based method mis-classifies **100%** of extrinsic genes as heritable; as clones
  intermix (high dispersal) the field correctly reads as spatial (0.11 > 0.04) and the
  mis-classification drops to ~40–55%, while genuinely-intrinsic genes stay 100% heritable throughout.
  iscc uniquely reveals this — real data has no ground truth. Figure `validation_petracer.png`.
  **Tier 2 (real data, `validation/data/build_petracer_reference.py` + `--real`) — DONE & RUN:**
  DNA-reference cache pattern. The real PEtracer M2 tumour (Figshare 10.6084/m9.figshare.28473866,
  `M2_tumor_tracing.h5td`, 553 MB — `.h5td` = TreeData; fetched via the ndownloader API with a
  browser UA, since only the web UI blocks bots) was reduced (3 largest per-clone lineage trees,
  `treedata` + a self-contained Newick/networkx tree reader; no ete3/cassiopeia). **The confound is
  CONFIRMED in real data without ground truth:** the ground-truth-free signature corr(I_lineage,
  I_spatial) across genes is +0.42/+0.51 in the two territorial/moderate trees (coord-lineage
  coupling +0.39/+0.11) and +0.02 in the intermixed tree (coupling −0.03); iscc's `dispersal_rate`
  sweep spans and brackets all three (`validation_petracer_real.png`). Reduced `.npz` committed
  (~6 KB, aggregate stats only); raw `.h5td` git-ignored. Caveats: MERFISH↔F9 ok; mouse metastasis
  (multi-site = R9) → validated PER-CLONE-TREE.
- **NEXT — "Multi-region trees are not phylogenies"** (benchmark-suite member; thematic SIBLING of the
  PEtracer confound — the "spatial structure misleads inference" cluster). Reproduce + extend Alves,
  Prieto & Posada, *Multiregional Tumor Trees Are Not Phylogenies* (PMC5549612; Posada = CellCoal
  author, already cited). Claim: bulk multi-region samples are ADMIXED (each region = a clone mixture),
  so a "sample tree" from regional bulk VAF/mutation profiles reflects similarity not lineage →
  spurious parallel mutations, biased divergence, reversed ordering. **iscc is ideal** (TRUE phylogeny
  via `genotypes_parents`/`to_newick` + REAL spatial admixture via clonal territories + F1 multiregion
  biopsy) and goes BEYOND their illustrative cases with a quantitative ground-truth sweep: (a) NJ
  "sample tree" from regional bulk VAF → RF distance / #spurious parallelisms vs the true clone tree;
  (b) MORE REGIONS DOESN'T FIX IT (admixture, not sampling density, is the problem); (c) sweep admixture
  via `dispersal_rate` (territories↔intermixed); (d) clonal deconvolution first (Clomial-style) recovers
  the truth (their fix). Mostly feasible now (multiregion biopsy + `bulkDNA` + `iscc.integrations.to_newick`
  exist); new: NJ-from-VAF + RF distance (+ optional deconvolution). Fits alongside/after clonealign+inferCNV.
- **DONE — Capability/feature matrix** (Table 1, `paper.tex`; commit 4ecc7ab). Verify a few competitor
  cells (SISTEM spatial/reads, CINner DNA, J-SPACE selection/sampling, scMultiSim DNA/consistency).
- **OPTIONAL (lean) — one realism head-to-head** iscc vs Splatter vs real on the 8 scRNA summary stats,
  framed as *parity* (preempt "is expression as realistic as a count sim?"). Skip to stay lean.
- **DONE (recent)** — manuscript caught up to F6/F7/M4; positioning + coupled-vs-bolt-on framing +
  experimental-design Outlook; overview schematic + `arxiv.sty` + build README; comparison Table 1;
  spine/assay notebooks re-executed and feature-complete (estimation/tau/reads demos).
- **LATER — bib hygiene**: verify the "auto-added — verify" entries in `references.bib`.

## F8 — microenvironment-driven expression (the integration keystone) ✅ DONE
- **DONE** — `models/count.py`: per-deme × gene modifier at materialisation (hypoxia `_o2_field` +
  CCI `_cci_field`), OPTIONAL via `microenv_params`, OFF ⇒ bit-identical, growth byte-identical
  on/off (readout only). Ground truth surfaced (`microenv_truth` + `cell_microenv`). 12 tests;
  `validation/validate_microenvironment.py` → figure. Unblocks spatial-niche/CCI benchmarks + the
  PEtracer intrinsic-vs-extrinsic decomposition. **Future extension:** microenvironment→FITNESS
  coupling (hypoxia slowing division) — deferred; v1 is expression-only. **Next (optional):** demo
  notebook; a manuscript niche/microenvironment paragraph + the figure.

## F9 — single-cell spatial assay (imaging-based) ✅ DONE
- **DONE** — `data/imaging.py` `scSpatial` (`ASSAYS["scspatial"]`): per-cell panel counts at
  `cell_crd`, NB/DM, transcriptome-coverage (`panel`/`n_panel_genes`) + data-distribution knobs
  (`IMAGING_PRESETS`), no spot aggregation, coords retained. 14 tests; wired into `isccdata` CLI
  (`-a scspatial`). Unblocks the PEtracer expression comparison. **Next (optional):** a demo notebook.

## Multi-patient cohort — ground truth for cohort analysis & personalized medicine (NEW, plan-first)
- **NEXT (after clonealign+inferCNV; own milestone, design-first like F8).** A `Cohort` layer that
  runs a DIFFERENT tumour per patient (own EVOLUTION seed → private clones, private passenger
  mutations, patient-specific CNAs + spatial structure) over a **SHARED, config-determined
  specification** — common driver genes / recurrent oncogenes+TSGs, shared selection landscape + gene
  panel — so recurrence is meaningful. Emit patients into batches via a **flexible patient→batch
  mapping** (below). Surface **cohort ground truth**: recurrent-vs-private drivers, per-patient private
  mutations, patient-of-origin, true **shared-vs-private cell-state** labels.
- **PREREQUISITE ENGINE FIX — comparability by default (user, 2026-07-03).** Two runs with the SAME
  config must use the SAME driver genes (else recurrence/cohort analysis is meaningless). Today they
  do NOT: `Selection.make_drivers()` (`components/selection.py:83`) draws driver/oncogene/TSG
  POSITIONS from `self.rng`, which is seeded by the RUN seed → different seed = different layout.
  Fix: DECOUPLE the **layout seed** (config-determined — a fixed default or a `genome_seed`/config
  hash, shared across patients) from the **evolution seed** (per-run, private). Then any two same-config
  runs are comparable by construction (same driver identities; different stochastic evolution). This is
  a foundational fix beyond cohorts — it makes ALL same-config runs comparable. (Real-genome mode
  already has a fixed shared `genome_spec`, so it's comparable already; this fixes ABSTRACT mode.)
- **Flexible patient→batch multiplexing (user, 2026-07-03).** The user picks how patients map to
  sequencing batches: **1:1** (each patient its own batch) OR **N:1** (multiplex/pool several patients
  into one batch, up to a technical capacity — as done with cell hashing / genetic multiplexing to
  save cost). This drives the technical batch structure AND unlocks a **demultiplexing benchmark**
  (assign pooled cells back to patient-of-origin, à la souporcell/demuxlet/vireo — ground truth = the
  patient label, which real data lacks). Reuses the scRNA "confounded"/multi-batch machinery.
- **Why it MUST be shown (user, 2026-07-03):** no simulator gives cohort-level ground truth for
  shared-vs-private structure — exactly what real cohorts can never provide. Unlocks flagship
  benchmarks: (1) **multi-patient batch integration** (Harmony/scVI/scANVI/LIGER) — score whether a
  method aligns SHARED states while preserving PRIVATE ones (the central over/under-correction failure
  mode, no real ground truth); (2) **recurrence / driver detection** (MutSigCV/dNdScv-style — recover
  recurrent drivers vs private passengers); (3) shared-vs-private states / cross-patient progression /
  consensus subtypes.
- **The personalized-medicine story (user, 2026-07-03):** the cohort is also an illustration of the
  NEED FOR PERSONALIZED MEDICINE. Make patient **subgroups** (molecular subtypes, e.g. distinct
  driver/resistance profiles) that respond DIFFERENTLY to therapy — so a treatment benefits one
  subgroup but not another. Coupled to the existing **treatment module**, this yields ground truth for
  **patient stratification / treatment-response prediction / biomarker discovery**: which patients
  benefit from which therapy, with a known answer. "Every tumour is different" made concrete + scorable.
- **Feasibility (mostly plumbed):** real-genome mode already gives a FIXED shared genome + arm-level
  selection landscape (loop seeds over one `genome_spec` → shared recurrence + private evolution); the
  scRNA batch machinery already has the **"confounded" design = different tumours → different batches**
  (`run_scrna_batches` docstring = the multi-patient case). New pieces: share the ABSTRACT-mode driver
  layout across patients; a `Cohort` wrapper; the shared-vs-private + subgroup/therapy-response ground
  truth bookkeeping. Plan design-first, then handoff.

## External-simulator adapters (recipe; additive `iscc.integrations` seam)
- **STARTED** — the `iscc.integrations` seam now exists (added by the PEtracer validation):
  `to_newick(tumor)` / `to_lineage_tree(tumor)` (lineage export) + `to_anndata(cell_data)`
  (coords/identity/expression). 19 tests.
- **LATER** — the per-simulator ADAPTERS (`adapters/scmultisim.py`, …) that feed the export to
  external sims (scMultiSim ATAC/GRN, SymSim, simATAC, SRTsim) so they generate their modality *on
  iscc's tumor*. Design: `DESIGN_features.md` §I. Honest limit: structure-conditioned, NOT
  genome-consistent (external tool doesn't see iscc's CNAs/SNVs). Gives ATAC/GRN support without
  building the native regulatory layer.

## Pedagogy notebook track (teaching tumor data analysis on iscc data)
- **NOW (plan) / NEXT (build)** — curriculum in `notebooks/TUTORIALS_PLAN.md`. Running example: a
  spatial tumor grown → non-adaptive chemo when large → partial kill → evolved resistance; every
  modality used to detect/characterize the resistant subclone, with ground truth revealed to grade
  each method. Spatial-niche/CCI lessons depend on F8.

## Paper 2 — the experimental-design RECOMMENDER (separate paper)
- **LATER** — full scoping in `DESIGN_recommender.md`. Two modes (generative + recommender); fit
  technical priors from a pilot, then score which designs recover a stated goal under budget.
  Positioned beside scPower/PoweREST. Depends on paper 1 being strong first.

## Scientific "hypothesis-generator" vignettes (paper 1 tail or standalone)
- **LATER — Axis A (Graham/Sottoriva): neutral-evolution detectability map** — confusion map of
  (true selection × depth × spatial sampling) → inferred neutral vs selected (adjudicates the 1/f
  debate). Mostly built (have 1/f validation + tunable selection/depth + spatial sampling).
- **LATER — Axis B (West/Anderson): value of information in adaptive therapy** — how sparse/biased a
  monitoring assay can be before the adaptive-vs-continuous advantage collapses. Flagship for the
  recommender paper; modest extension (feed the controller from a simulated assay readout).

## Engine / inference follow-ups
- **LATER — M3b HPC rerun** — fewer, bigger tumors (~800 × 8000-cell, tau-leaped) for the canonical
  Charm/PCAWG figure; HPC-bound. The flagship real-genome figure.
- **LATER — `estimate_visium` kappa-on-sparse-data refinement** — fit on expressed genes; minor
  estimator robustness, doesn't affect defaults.
- **LATER — fill the 5 analysis-stub notebooks** — `dna_mhn`, `visium_niches`, `scrna_batch_effects`,
  `combining_scdna_scrna`, `real_data_comparison` (currently title-only roadmap stubs).
- **LATER — RESEARCH_QUESTIONS R1–R11** — 3D (R1), AI histology (R2), metastasis/multi-site (R9),
  focal CNAs/WGD (R10), recommender (R11), etc. Each a substantial new track; promote when committed.

## Housekeeping
- **NOW — commit** the uncommitted work on `dev`: manuscript catch-up + positioning edits, the
  re-executed notebooks, `DESIGN_recommender.md`, `DESIGN_features.md` F8, `RESEARCH_QUESTIONS.md` R11,
  `handoffs/F8_*.md`, this `BACKLOG.md`, and `notebooks/TUTORIALS_PLAN.md`.
