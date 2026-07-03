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

- **NOW — the ground-truth benchmark suite** (the load-bearing comparison; run real methods on iscc
  data, score vs known truth):
  - **Flagship — `clonealign`** (CNA-dosage DNA↔RNA clone assignment; works today, non-circular). AUC vs true clone.
  - `inferCNV`/`copyKAT` (CNA-from-expression) — correlation vs true per-cell CNA.
  - Multi-batch **integration/correction** on F3 batches (Harmony/scVI) — iLISI/ARI vs known clones/types.
  - Cell-type/clone **clustering** — ARI; **spatial deconvolution** (cell2location/Tangram) — per-spot accuracy.
  - Already in-paper (count as suite members): **selection/rate inference** (ABC recovery), **sampling/
    experimental-design**, realism-vs-real. Niche/**CCI** inference waits on F8.
  - Manuscript: reframe validation as *"iscc as a benchmarking substrate"* organized as this suite.
- **NEXT (separate session) — PEtracer validation** (flagship real-data + ground-truth benchmark).
  Handoff saved: `handoffs/PEtracer_validation.md`. F8 (extrinsic) + F9 (single-cell spatial) + the
  engine's `genotypes_parents` lineage now make the intrinsic-vs-extrinsic decomposition possible.
  **Tier 1 (self-contained):** run the Hotspot-style lineage-vs-spatial autocorrelation decomposition
  on iscc data with a KNOWN split, and **expose the lineage-space CONFOUND** — under clonal
  territories a purely environmental (hypoxia) signal gets lineage-autocorrelation, so a tree-based
  method MIS-CLASSIFIES it as heritable; iscc uniquely reveals this (real data can't). Tunable via
  `dispersal_rate` (territories vs intermixed). **The headline finding.** **Tier 2 (real data):**
  reduce PEtracer (Figshare 10.6084/m9.figshare.28473866 + GEO GSE290975; DNA-reference cache
  pattern), compare lineage/spatial-autocorrelation + clone-territory + tree stats. Caveats: MERFISH↔F9
  ok; mouse metastasis (multi-site = R9) → validate per-tumour. User wants BOTH tiers.
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

## External-simulator adapters (recipe; additive `iscc.integrations` seam)
- **LATER** — let iscc be the evolutionary+spatial substrate; export lineage tree (Newick) + AnnData
  (coords/identity) so external sims (scMultiSim ATAC/GRN, SymSim, simATAC, SRTsim) generate their
  modality *on iscc's tumor*. Design: `DESIGN_features.md` §I. Honest limit: structure-conditioned,
  NOT genome-consistent (external tool doesn't see iscc's CNAs/SNVs). Gives ATAC/GRN support without
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
