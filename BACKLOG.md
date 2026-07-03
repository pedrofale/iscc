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
  score vs known truth). **Active next: multi-batch integration / clustering / deconvolution:**
  - **DONE — `clonealign`** (CNA-dosage DNA↔RNA clone assignment; non-circular). The GENUINE R
    `clonealign` (kieranrcampbell/clonealign, in the dedicated `iscc-clonealign` env) assigns scRNA
    cells to their true clone at mean AUC 0.84 (acc 0.58 vs 0.25 chance, 4 clones), driven by the
    EMERGENT dosage coupling (accuracy rises with the CN-informative-gene fraction).
    `validation/validate_clonealign.py` + `validation/clonealign_runner.R` + fig `validation_clonealign.png`.
  - **DONE — `inferCNV`/`copyKAT`** (CNA-from-expression). The GENUINE `infercnvpy` (scverse, in the
    dedicated `iscc-infercnv` env) separates malignant vs normal at AUC 0.99 and recovers the clonal
    CNA structure (clone-level r 0.86 vs true per-cell CN). `validation/validate_infercnv.py` +
    `validation/infercnv_runner.py` + fig `validation_infercnv.png`. Shared data-gen +
    run/score helpers in `validation/integration_common.py`; `tests/test_integration.py` (8 tests,
    real tools guarded by dedicated-env availability); manuscript §"non-circular ground truth for
    multi-modal integration". Setup: `validation/README_integration.md`.
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
- **DONE — "Multi-region trees are not phylogenies"** (benchmark-suite member; thematic SIBLING of the
  PEtracer confound — the "spatial structure misleads inference" cluster). Reproduced + extended Alves,
  Prieto & Posada, *Multiregional Tumor Trees Are Not Phylogenies* (PMC5549612; `alves_multiregion_2017`,
  auto-added—verify). New self-contained analysis seam `iscc.integrations.multiregion` (NJ, Fitch
  parsimony, Robinson–Foulds — numpy-only, no ete3/dendropy): `true_origin_counts` (the ANSWER KEY —
  per-locus #independent origins via Fitch on the true clone tree pruned to observed clones;
  single-origin loci are the clean substrate), `region_bulk_profiles` (deep `bulkDNA` per multiregion
  region = admixed VAF), `oracle_clone_profiles` (per-cell clone truth as oracle deconvolution = the
  fix's achievable bound), `count_spurious_parallel` (Hamming-NJ tree + Fitch; spurious = truly
  single-origin yet reconstruction infers ≥2 origins), `multiregion_phylogeny` (high-level: naive vs
  fix + RF). **Headline numbers** (18k-loci SNV-only tumour, clonal territories, K=8, 3 seeds): naive
  region sample tree spurious-parallelism rate **~24%** + ordering-reversal **~23%**; oracle clone
  deconvolution **~0.4%** with **~93%** true-clone-split recall — despite MORE leaves than the region
  tree (⇒ admixture, not tree size). MORE REGIONS DOESN'T FIX IT (naive rate 9%→29% as K:4→10, never →0;
  fix stays ~0). Error scales with measured per-region admixture (`dispersal_rate` sweep, corr **~0.7**).
  `validation/validate_multiregion_phylo.py` (~23s) → `manuscript/figures/validation_multiregion_phylo.png`;
  `tests/test_multiregion_phylo.py` (10); manuscript §"multi-region bulk sample trees are not phylogenies"
  beside the PEtracer section (`fig:multiregion`). Full suite 407 green.
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

## Multi-patient cohort — ground truth for cohort analysis & personalized medicine ✅ DONE (2026-07-03)
- **DONE** — design `DESIGN_cohort.md`; all pieces shipped:
  1. **Prerequisite engine fix (comparability by default):** a config-determined `layout_seed`
     (default `constants.DEFAULT_LAYOUT_SEED = 42`) drives the Selection gene-role layout + shared
     per-cell-type baseline expression, decoupled from the per-run evolution seed, in BOTH engines
     (`tumor/tumor.py`, `tumor/models/count.py`). Two same-config runs now share driver identities by
     construction (Jaccard 1.0) and differ only in evolution; byte-identical to the old plumbing at the
     default seed (verified: full suite green). `tests/test_cohort.py` covers the guarantee.
  2. **`iscc.cohort`** package — `Cohort`/`Subgroup`/`PatientResult` (`cohort.py`), patient→batch
     multiplexing 1:1 & N:1 + pooled scRNA emission (`batch.py`), cohort ground-truth tables
     (`groundtruth.py`). Subgroups differ by EFFECT scalars (shared landscape); resistance is
     **subclonal by default** (a seeded pre-existing resistant subclone, selected under therapy) with
     a truncal option; per-patient private germline markers for demux.
  3. **`validation/validate_cohort.py`** (+ `cohort_common.py`, `harmony_runner.py`) →
     `manuscript/figures/validation_cohort.png` + Results subsection `sec:cohort`, `fig:cohort`. The
     4 benchmarks: recurrence-enablement (shared Jaccard 1.0 vs unshared 0.04), personalized-medicine
     stratification (therapy differential + single-cell subclone biomarker AUC 1.0), multi-patient
     integration (shared-state iLISI 1.9→5.3, ARI preserved), demultiplexing (patient-of-origin 1.0 vs
     chance 0.12). External integration/demux tools wired behind `iscc-harmony`/`iscc-scvi`/`iscc-demux`
     env guards (clonealign/inferCNV convention); the figure is self-contained.
  - **Honest finding:** per-gene SNV recurrence enrichment is modest in abstract mode (fitness depends
    on the COUNT of mutated drivers, so per-gene convergence is weak + passengers hitchhike in sweeps);
    the real-genome arm model is the sharper substrate. The shared-vs-unshared *enablement* contrast is
    the strong result there.

<details><summary>original spec</summary>

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
</details>

## Operating envelope — parameter→phenotype atlas & built-in QC ✅ DONE (2026-07-03)
- **DONE** — design `DESIGN_operating_envelope.md`; the three layers all shipped:
  1. `analysis/characterize_regimes.py` (+ cached `analysis/characterize_regimes.csv`, 201 rows) —
     coarse sweep over every axis, all metrics per run.
  2. `validation/validate_operating_envelope.py` → `manuscript/figures/validation_operating_envelope.png`
     (phase diagrams + 1-D slices) and the supplementary subsection *"Operating regimes: which
     parameters yield realistic tumors"* + defaults/valid-ranges table (`sec:envelope`, `tab:envelope`).
  3. `iscc.tumor.diagnostics` + `GenotypeTumor.diagnose()` (read-only QC, overridable thresholds,
     actionable hints) + opt-out auto-warn from `isccsim` (`--no-diagnose`). Tests:
     `tests/test_diagnostics.py`.
  - Honest findings recorded in the design doc: the well-mixed metric uses per-clone spatial
     confinement (a naive clone-label Moran's I is confounded because higher dispersal also reduces
     clone count); fraction-genome-altered **saturates** (viability-capped) rather than running away.

<details><summary>original spec</summary>
- **Goal (user):** know & REPORT which parameter ranges produce which tumour features, so users don't
  run the simulator and end up with "crappy tumours" — extinct, monoclonal, hypermutated mush,
  well-mixed with no clonal territories, or no microenvironment gradient. Two audiences: (a) reviewers
  — a robustness/sensitivity analysis (standard for simulator papers: scMultiSim, CINner both ship one);
  (b) users — documented sane defaults + valid ranges + a runtime warning. NB the "well-mixed" and
  "no-gradient" degenerate zones are exactly the regimes that would silently BREAK the PEtracer and
  multi-region benchmarks, so this also protects the headline results.
- **Three deliverable layers (design-first):**
  1. **Characterization sweep** (`analysis/characterize_regimes.py`): grid over key axes → phenotype
     metrics per run. AXES (real knobs, `notebooks/example_config.yaml` + `Selection`): `mutation_rate`
     & `n_snvs_per_allele` (SNV load); `division_rate` vs `death_rate` & `initial_cancer_cells`
     (survival); `dispersal_rate` (× `division_rate`) (spatial mixing); `prop_driver` × `driver_effects`
     (selection strength); `grid_size` × `carrying_capacity` (tumour size / #demes); `amp_prob` &
     `max_cn` (CNA burden); microenv hypoxia `D`/`k`/`s` × tumour size (gradient). METRICS: N cells /
     P(extinction); clonal Shannon diversity & #subclones; VAF 1/f neutral-tail fit; TMB (muts/cell);
     positive-selection detectability; Moran's I of clone labels (territories↔mixed);
     fraction-genome-altered / ploidy; hypoxia core–rim contrast.
  2. **Reported operating ranges**: a manuscript SUPPLEMENTARY "operating regimes / robustness"
     section — phase-diagram figure(s) with the realistic region highlighted and the degenerate regimes
     labelled (extinction / monoclonal-sweep / low-mutation-monoclonal / hypermutated / well-mixed /
     no-gradient) + a defaults-and-valid-ranges table.
  3. **Built-in QC diagnostic** (`tumor.diagnose()` / a small report): after growth, flag degenerate
     output against thresholds (extinct; diversity < X; no spatial structure; no O2 gradient; TMB out of
     range) with ACTIONABLE hints ("raise `mutation_rate`", "lower `dispersal_rate`", "grow larger").
     The direct answer to "so users don't end up with crappy tumours."
- **Links:** feeds the RECOMMENDER (paper 2 — a prior over valid designs); complements CALIBRATION
  (fitting to real data should land you IN the good region; this lets us verify that). Handoff:
  `handoffs/operating_envelope.md`.
</details>

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
