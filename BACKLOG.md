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

## Therapeutic escape + the manuscript build ✅ DONE (2026-08-25)
- **All four Kane/Maley escape modes reproduce**, one kill model and schedule, resistance genetics the
  only difference between panels. Figure `validation_escape_modes.png`, manuscript §"reproduces the
  four modes of therapeutic escape", citation `kane_fitness_2026` (v2, posted 2026-08-04, verified at
  source). Reproducible from the repo: `validation/validate_escape_modes.py`, whose four panels come
  out **byte-identical** to the published runs. Runtime ~25-40 min, per-panel npz cache, `--only`/`--force`.
- **Drug-induced resistance STATE** (`ad86f23`, plasticity §3.3) verified: 908 passed / 1 skipped,
  749 → 0 sensitive on the original rig. §3.4 records the 2x2 decomposition — **exit is the
  load-bearing knob, not the cost** — and that **tau→∞ is NOT required**: pricing resistance once
  instead of twice gives 97.7% resistant at relax=0.02 with the clean start intact.
- **A mutagen shared across all four panels is IMPOSSIBLE, and that is a result.** III and IV differ
  only in the relapsing lineage's origin, so the drug that manufactures IV's de novo clone also
  manufactures one in III and overwrites the pre-existing clone that defines it. Do not retune it away.
- **The manuscript had NEVER been compiled.** Three pre-existing bugs fixed (`4083d9d`): a
  `\providecommand{\affiliations}` that guaranteed the collision it was meant to prevent; 21
  `\Cref{sec:...}` pointing at STARRED sections (body sections now numbered — reversible, but then
  those refs need rewriting); and **18 bib entries with an inline `% auto-added` comment after the
  key**, which BibTeX parsed as the first field name — 17 references were rendering with no author,
  title, journal or year. Now 35 pages, zero undefined refs, zero BibTeX warnings.
  **Build with `tectonic -X compile paper.tex`** (self-contained, runs BibTeX). BasicTeX needs sudo;
  conda `texlive-core` is a plain-TeX stub with no `latex.ltx`.
- **Orphaned figures triaged.** `validation_ductal_field.png` + `validation_spatial_diagnostic.png`
  wired in (`1b855c8`) with honest captions — the ductal-field divergence panel reads BACKWARDS for an
  island bottleneck (within-focus 0.108 > between-focus 0.044, likely a focus-means artefact) and the
  Moran's I contrast is only 0.70 vs 0.66. **NEEDS REVIEW.** The other two deliberately left out:
  `programs_cohort` (n=1, already deferred by the user) and `evolution_modes` (a NEGATIVE result —
  "5% of real tumours in iscc hull"; check like-for-likeness first, since iscc's D is full ground
  truth and Noble's is sparse multi-region sequencing).

## F8 — microenvironment-driven expression (the integration keystone) ✅ DONE
- **DONE** — `models/count.py`: per-deme × gene modifier at materialisation (hypoxia `_o2_field` +
  CCI `_cci_field`), OPTIONAL via `microenv_params`, OFF ⇒ bit-identical, growth byte-identical
  on/off (readout only). Ground truth surfaced (`microenv_truth` + `cell_microenv`). 12 tests;
  `validation/validate_microenvironment.py` → figure. Unblocks spatial-niche/CCI benchmarks + the
  PEtracer intrinsic-vs-extrinsic decomposition. **Future extension:** microenvironment→FITNESS
  coupling (hypoxia slowing division) — deferred; v1 is expression-only. **Next (optional):** demo
  notebook; a manuscript niche/microenvironment paragraph + the figure.

## CCI + spatial (DESIGN_cci_spatial.md) — W0 + W3 ✅ DONE (2026-08-27, Visium resolution)
- **DONE — W0 the L-R database.** ONE new parameter `microenv_params['cci']['n_candidate_pairs']`
  (default 1). iscc emits its OWN candidate ligand-receptor database over its own abstract gene ids:
  row 0 wired, the rest unwired decoys, drawn from the layout stream (disjoint from the readout target
  genes). The clone-correlated class is MEASURED (`iscc.integrations.clone_correlation`), not planted.
  `cci_database` / `write_cci_database` emit the four `Update-CellChatDB` CSVs and assert the
  `geneInfo` whitelist is complete; `validation/cellchat_runner.R` re-asserts it in R (round trip of
  `validation/README_cellchat.md`).
- **DONE — W3 receptor-dependence.** CCI effect is now
  `1 + strength·ligand_avail[deme]·receptor[genotype]` on the target genes (per-(deme,genotype), both
  total + allele layers). Ligand-weighted smoothed emitter field; per-cell received signal in
  `cell_microenv['cci_level']`. NORMALISATION keeps the calibrated `strength` valid (each term ÷ its
  population mean). OFF bit-identical; growth byte-identical ON. W0/W3 tests in
  `tests/test_microenvironment.py`.
- **DONE — recoverability check** `validation/validate_cci.py` (+ `cellchat_runner.R`, `iscc-cellchat`
  env): runs real CellChat on a planted Visium section, ranks pairs by `prob`. CAVEAT (stated, not
  discovered): F8 modulates TARGETS and reads L/R while CellChat scores L/R group means, and the
  deme-scale signal sits below the ~55 µm spot — so the wired pair is not expected to separate at
  Visium. See the run report + `manuscript/figures/validation_cci.png`.
- **NEXT (separate handoff) — W4** the `dispersal_rate` confound sweep (interaction vs clonal
  relatedness), and **W2** intra-deme layout for single-cell spatial resolution. Both scoped in
  `DESIGN_cci_spatial.md`; W4 needs this landed, and needs a target-aware method or W2's resolution.

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
     (`groundtruth.py`). Subgroups differ by EFFECT scalars over the shared landscape; **resistance is
     NOT seeded — it EMERGES** from mutation + selection (a subtype differs only in
     `treatment_resistant_effects`; standing resistance mutations arise in an untreated burn-in and
     adjuvant therapy selects them). Inherited `germline_mutations` (+ per-patient private demux markers)
     are applied to EVERY cell of the patient — tumour AND normal, as real germline variants are — never
     for acquired resistance.
  3. **`validation/validate_cohort.py`** (+ `cohort_common.py`, `harmony_runner.py`) →
     `manuscript/figures/validation_cohort.png` + Results subsection `sec:cohort`, `fig:cohort`. The
     4 benchmarks: recurrence-enablement (shared Jaccard 1.0 vs unshared 0.04), personalized-medicine
     (emergent differential response; recovery AUC — baseline non-predictive ~0.5, emergent relapse
     signature ~0.8, response readout 1.0), multi-patient integration (shared-state iLISI 1.9→5.3, ARI
     preserved), demultiplexing PER MODALITY — DNA: genetic demux on germline SNPs (all cells, cancer +
     normal, acc 1.0); RNA: cell hashing (`hashing.py`, HTO/MULTI-seq) with near-perfect singlets and
     doublets as the failure mode (naive acc falls with doublet rate, doublets detectable). External
     integration/demux tools wired behind `iscc-harmony`/`iscc-scvi`/`iscc-demux` env guards
     (clonealign/inferCNV convention); the figure is self-contained.
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
  - **Realistic size + perf:** documented that size ≈ #steps (exact) / use `update_mode="tau"`;
     non-failing `small tumour` advisory; and a ~9× count-engine speed-up (skip immune scan when no
     immune cells; deme occupancy once per substep) — 134k cells in ~20s, byte-identical output.
  - **Closed the loop** (user, 2026-07-03): `validation/validate_calibration_envelope.py` →
     `manuscript/figures/validation_calibration_envelope.png` + `sec:envelope`/`fig:calibration`.
     ABC-rf recovery estimates land 100% inside the good ranges and regrow non-degenerate; the prior
     audit CAUGHT + fixed a founder-extinction leak in the inference base config
     (`initial_cancer_cells=5`), taking prior non-degenerate coverage 98%→100%. Tests in
     `test_diagnostics.py`.

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

## Integration benchmark suite — the paper thesis (tool matrix + two prerequisites)
- **THESIS (2026-07-06):** iscc = the ground-truth substrate for DATA-INTEGRATION methods at two scales
  — cross-modal *within* a tumour, and cohort-wide *across* patients (biological patient effects
  confounded with technical batch effects). Abstract + intro of `paper.tex` redrafted along this arc
  (UNCOMMITTED, pending review). Positioning: memory `iscc-paper-positioning`.
- **Tool matrix** (representative, not exhaustive; ✅ done / ◑ partial / ⬜ new):
  - DNA→RNA clone assignment, CNA dosage: **clonealign** ✅
  - DNA→RNA clone assignment, SNVs: **cardelino** ⬜ (needs SNV→expr realism, R13)
  - bulk-DNA + scRNA clone tree: **PhylEx** ⬜
  - CNA-from-expression: inferCNV/CopyKAT ✅; **Numbat** ✅ (allele-aware, R13; fed allele counts DIRECTLY
    — iscc's p/m homologs ARE the phasing, no population panel; head-to-head vs inferCNV = ~tie on total
    CN + AUC, since iscc CNAs mostly change total CN and cnLOH is rare — an honest, reported negative)
  - scRNA→Visium deconvolution: **cell2location, RCTD** ✅ — FLAGSHIP (paired same-tumour ref + true
    per-spot comp; matched-vs-mismatched reference decomposed into regional/dissociation/assay — regional
    under-sampling dominates; clone-vs-cell-type confound: types r≈0.99, CNA clones r≈0.4–0.8)
  - cell–cell communication / niche: CellChat / CellPhoneDB / COMMOT ⬜ (F8 ground truth)
  - scRNA cohort integration: **Harmony, scVI/scANVI** ✅ (+ Scanorama, LIGER; scIB metrics)
  - multi-sample Visium integration: GraphST / STAligner ⬜ (2D cross-patient; PASTE 3D out of scope, R1)
  - DNA cohort progression: **MHN, TreeMHN**, CBN/H-CBN, REVOLVER ◑ — R14 (epistasis) DONE, so the
    planted network + the scoring seam (`iscc.integrations.progression`: `to_mhn_matrix` /
    `to_treemhn_trees` / `score_edges` / `score_order`) now EXIST and the built-in co-occurrence
    baseline runs (`validation/validate_epistasis.py`). **DONE — the REAL tools now run** in
    `iscc-mhn` (MHN 1.2.3, Schill et al. 2020) and `iscc-treemhn` (TreeMHN, Luo/Kuipers/Beerenwinkel
    2023) via `validation/mhn_runner.py` + `validation/treemhn_runner.R`, scored in the manuscript
    §"which planted evolutionary dependencies can cohort progression models recover?". This line said
    "what remains" until 2026-08-25 and was stale — the work had already shipped.
    **The finding: the OBSERVABLE, not the cohort size, decides recovery.** iscc's pairwise `E` acts
    on FITNESS (how large the carrying clones grow), not on the rate at which events arise, so a
    binary "did this patient ever acquire event i" matrix — MHN's input — saturates and is nearly
    blind to it. TreeMHN reads tree TOPOLOGY, so what it adds is ORDER, not frequency; fitness
    epistasis generates no order, so its advantage is orthogonal and it cannot beat MHN on pairwise
    `E` for STRUCTURAL reasons rather than want of data.
  - pooled demultiplexing: vireo/souporcell (DNA), cell-hashing + scDblFinder (RNA) ✅
  - subclonal deconvolution (multi-region bulk): PyClone-VI / Pairtree / Clomial ◑ (oracle deconv done)
  - **DONE — gene-program / GEP inference: `scDEF` FLAGSHIP + `cNMF` comparator** (Hotspot still
    optional). The REAL tools in dedicated `iscc-scdef` / `iscc-cnmf` envs, scored against the true
    `loading` + per-cell `z` (R13's `program_truth`) across the SNV/CNA-burden sweep.
    `validation/validate_programs.py` (+ `programs_common.py`, `scdef_runner.py`, `cnmf_runner.py`)
    → `manuscript/figures/validation_programs.png`; manuscript §"copy-number burden manufactures
    spurious expression programs", beside PEtracer + multi-region in the *structure misleads
    inference* arc.
    **The hypothesis was only HALF right, and the honest half is the more interesting one** (3 seeds):
    true-program recovery is largely ROBUST to CNA burden — over FGA 0→0.55, scDEF's loading cosine
    only 0.74→0.69 and cNMF's 0.51→0.40 — so "recovery degrades with FGA" is weak (cNMF) to absent
    (scDEF). What CNA burden does instead is **manufacture spurious positionally-clustered factors**:
    matched factors stay at the scattered null (0.20–0.22) at every burden while SPURIOUS ones climb
    to **0.76** (cNMF) / **0.37** (scDEF) vs a null 95th pct of 0.28, and cNMF invents more of them
    (7→9 of 12). The **SNV sweep is the control that makes it attributable to CONTIGUITY**: flat
    recovery, statistic never leaves the null. So the mechanism is *invention of artefacts*, not
    destruction of signal — a genuine PEtracer sibling.
    **Caveats:** the diagnostic detects POSITION not artefact (the `program_genomic_scatter=0` control
    makes correctly-matched factors score 0.37/0.42 too — a flag for review, not a verdict); magnitude
    depends on the per-gene dosage buffering `s_g`. **Unplanned:** scDEF wins on loadings + fewer/cleaner
    spurious factors, but cNMF recovers per-cell ACTIVITIES far better (r≈0.80 vs 0.44).
    **RESOLVED 2026-07-15 (user decision):** `iscc-scdef` is now **pinned to scDEF 0.6.1** from PyPI
    (rebuilt from scratch — no longer an editable clone of `~/projects/scDEF`, which had reported
    0.4.8 while executing 0.6.1 source). `__version__`, the dist metadata and the load path all agree,
    and both runner paths (plain + `batch_key`) were re-verified against the released package. The
    version of record is stated in `validation/README_integration.md` and in the manuscript Results
    (scDEF v0.6.1, cNMF v1.7.1) — **update both together if it ever changes**. `ferreira_scdef_2024`
    now cites the **bioRxiv preprint** (v2, 2024-01-04, doi 10.1101/2022.10.15.512383), confirmed
    against the bioRxiv API; no journal DOI exists yet, so re-check at submission.
  - **DONE (with a caveat) — cohort-level: shared vs patient-specific programs** (§4.3,
    `validation/validate_programs_cohort.py` → `manuscript/figures/validation_programs_cohort.png`).
    Cohort now forwards `expression_params` + `microenv_params` (it forwarded NEITHER — F8 was
    unreachable through the cohort layer too). Truth: patients SHARE the program dictionary (layout
    stream) but have PRIVATE CNA landscapes. **A CNA-driven factor is real biology that is not a
    program** — which flips §4.2's verdict on the same positional statistic (nuisance there,
    signal-to-preserve here). **Result (6 patients, 24 factors):** pooled/demux, scDEF finds 5
    patient-specific factors at positional clustering 0.74 (null 0.24) = genuine CNA biology, plus the
    best shared recovery (0.68). One-batch-per-patient: factors double to 11 while clustering collapses
    to 0.37 (batch masquerading as patient biology). `batch_key` prunes back to 6 and restores mixing,
    but survivors sit AT the null (0.28) and `cna_retention` never recovers (0.44 → 0.27 → 0.27).
    ⇒ **demultiplexing preserves patient-specific biology that no batch correction recovers.**
    Data-level, tool-independent: corr(CN dev, expr dev) 0.49 pooled → 0.26 confounded.
    **CAVEAT — DO NOT PUT THIS FIGURE IN THE PAPER AS-IS:** every arm is **n=1** (one cohort, one batch
    draw, one tool fit; `build_cohort` hardcodes `patient_seeds=1..N`, `run_tool` uses `seed=0`, and
    `emit()`'s `seed` arg is DEAD — never threaded to `run_cohort_batches`). The arms are *paired* on
    one cohort, which is the right structure, but there is no spread, so small gaps
    (`shared_recovery` 0.68/0.67/0.65) are almost certainly noise while the large ones
    (`cna_retention`, clustering) are only *probably* real. The quick-vs-full swing
    (`n_patient_specific` 0 → 5) proves these statistics are configuration-sensitive. **To promote:**
    thread a rep seed through those three places, loop, report mean ± sd (~15 fits + 3 cohort growths,
    ≈30 min). A manuscript subsection is also NOT yet written for this one. Deferred by user 2026-07-15.
- **Two PREREQUISITES — NOW IN PAPER 1 (decision 2026-07-14), design-first, not built. Handoffs ready:
  `handoffs/expression_programs.md` (R13) and `handoffs/epistasis.md` (R14).**
  - **Expression realism** — `DESIGN_expression.md` (R13) ✅ **DONE 2026-07-15**. Expression is now a
    GENE-PROGRAM backbone (`iscc/tumor/programs.py`: `ProgramDictionary` + the per-cell `z` sampler —
    **the shared R12 cell-state model, built once**) with the gene-level overlays on top: per-gene
    dosage sensitivity `s_g` + saturation (axis A), allele-resolved expression + **BAF in RNA**
    (`cell_exp_p`/`cell_exp_m`/`cell_rna_baf` — the `p`/`m` alleles are no longer summed; unlocks
    Numbat/CalicoST), and per-SNV functional classes LoF→NMD / missense / splice / silent with the
    expression effect **decoupled from the fitness `mut_effect`** (unlocks cardelino/PhylEx). All three
    genotype→program routes of §3.1 are wired: **route 1 (phenotype-mediated, the default)** reads the
    evolved per-clone phenotype, so CINner drivers — and R14's epistasis multiplier — reach the
    programs by construction; route 2 (direct regulators, no fitness change); route 3 (niche→program,
    generalising F8). Off by default and **readout-only** (growth is byte-identical on/off — pinned by
    `test_programs.py`); the landscape draws from the layout stream so programs are **comparable across
    patients**. Ground truth on `tumor.program_truth`; `expression_params` documented in
    `PARAMETERS.md`; `diagnose()` gained a `clone_is_state` check. Suite 490→511.
    - **Benchmark** (`validation/validate_programs.py`, envs `iscc-scdef` + `iscc-cnmf`): see the
      program-recovery item above.
  - **Epistasis** — `DESIGN_epistasis.md` (R14, **paper 1**) ✅ **DONE**. Pairwise `E`, a conjunctive
    dependency DAG (`fitness` | `accessibility` gating) and mutual exclusivity are plantable in
    `Selection` (off by default → bit-identical; cached per event set → tau-leap safe), drawn from the
    layout stream (`LAYOUT_OFFSET_EPISTASIS`) so a whole cohort shares ONE network. Ground truth via
    `tumor.epistasis_ground_truth()` / `tumor.event_table()`; benchmark in
    `validation/validate_epistasis.py` → `manuscript/figures/validation_epistasis.png`; Results
    section `sec:epistasis`.
    **Headline: the OBSERVABLE decides recovery.** `E` acts on FITNESS (how large the carrying clones
    grow) while MHN/CBN model the RATE of event ACQUISITION. Paired sweep (only `E` differs): `E` 0→1.5
    expands the clones carrying the pair **3.5% → 42%** of the tumour (12x), plateauing at 59% past the
    fitness clamp `log(b_max/b_0)`; P(the pair ever AROSE) barely moves (0.03→0.09 — a mutation
    property). But a BINARY "event present" matrix registers almost none of that, because RECURRENT
    MUTATION saturates presence (the combination arises many times independently, so it is already
    present at `E=0`; selection changes only how much of the tumour it occupies, which presence discards).
    **Real tools, each in its own env** (`iscc-mhn`, `iscc-treemhn`; both pass positive controls):
    * **MHN** (binary presence) retains SOME signal — ranks the planted pair 1st in 4/5 network draws
      vs 1/5 for the empty-E control (mean rank 1.4 vs 2.0 of 6 pairs). Suggestive, NOT significant
      (Fisher p=0.21, n=5 draws). Note the control's own rank 2.0 ≫ chance 3.5 = a real false-positive
      tendency — without the empty-E arm this would have looked like recovery.
    * **TreeMHN** does NOT beat chance on pairwise `E` (rank 3.4 vs chance 3.5) — but NOT for want of
      data, and NOT because "trees retain clone sizes" (they do **not**: `input_tree_df` accepts ONLY
      Patient_ID/Tree_ID/Node_ID/Mutation_ID/Parent_ID and rejects anything else; its `weights` is a
      per-TREE weight for tree uncertainty). **TreeMHN's gain over MHN is event ORDER, not frequency**,
      and fitness epistasis produces NO order — only frequency. Its extra information is orthogonal to
      the planted signal. **Falsifiable prediction, tested and CONFIRMED:** give the same two tools an
      ORDER signal (accessibility-gated DAG) and TreeMHN wins decisively — TreeMHN rank **1.80**
      (top-1 0.6) vs MHN **4.00** (top-1 0.0, worse than chance) vs floor 5.40. The single draw where
      TreeMHN fails is the one where 1/40 patients carried the gated event (no signal to read).
      **So: each tool recovers exactly the signal its input encodes.** That 2x2 (tool x signal-type)
      is the benchmark's real result.
    Conjunctive constraints under **accessibility** gating are recovered perfectly (1.00, true AND
    reconstructed trees) though at low power (~8 child-carrying lineages/draw); the same DAG under
    **fitness** gating leaves no trace (conjunction 0.14).
    **Known limitation, stated in the paper:** the cohort tumours are ~130 cells, so a clone arising
    late has no time to expand and the frequency signal never fully develops — the absolute recovery
    numbers are a FLOOR for this regime, not an estimate for real cohorts. Scaling the benchmark
    (tau-leaping, ~10⁴-cell tumours) is the obvious next step and is what would test whether the
    frequency observable recovers `E` once selection has room to act.
    **Corrected 2026-07-15 (was wrong in the first commit):** the earlier claim "recovered at chance
    regardless of cohort size" was an artifact of (a) a bug — `min_freq` filtered CLONES by size then
    OR'd them, so an event carried by dozens of small lineages was called ABSENT (now a per-event
    cell-fraction threshold, `event_cell_fractions`; regression test added), and (b) sweeping cohort
    size when the binding axis was the observable. Do not re-cite the retracted numbers.
- **Priority (new work):** deconvolution (cell2location/RCTD, flagship) → Numbat → cardelino/PhylEx →
  MHN/TreeMHN (after epistasis) → multi-Visium → CCI. Each external tool in its own `iscc-<tool>` env.
- **QUEUED — "cash in R13": handoff `handoffs/deconvolution_numbat.md`** (BLOCKED until R13 lands). Runs
  the two tools R13 was built for: **deconvolution** (needs the program layer; headline = the cost of a
  REALISTIC reference — same tumour, *different sample*: Visium section vs an scRNA reference from a
  separate F1 biopsy of another region, after F2 dissociation — decomposed into its three real sources:
  regional mismatch, dissociation bias, assay/batch. iscc alone can dial each and know the truth for all;
  ties to the existing "biopsy and dissociation shape the sampled data" section) and **Numbat**
  (needs ASE/BAF; headline = does the allele layer beat expression-only inferCNV?). Known risk: Numbat's
  allele input normally comes from cellsnp-lite + a phasing panel, which an abstract genome lacks — scope
  the interface first (likely feed allele counts directly).

## Ductal-field spatial substrate (DESIGN_ductal_field.md)
- **DONE (2026-07-21) — the island-model ductal field (count engine).** `count.py:_seed_structure`
  rewritten from ONE central ring to a FIELD of `n_glands` small epithelial-ring glands (ring wall +
  empty DCIS-growable lumen) at 2D positions in **moderate-density** stroma (`stroma_fill_frac`≈0.3–0.5;
  real stromal cells that carry the compartment stromal hazard), grown from ONE
  founder in gland 0's lumen. New spatial_params: `n_glands`, `gland_radius`, `min_gland_sep`,
  `K_duct`/`K_stroma` (per-deme capacity — a deme is a 3D column so K is moderate-to-large, honoured
  via a per-deme `_deme_capacity` array in the crowding law), `stroma_fill_frac`, `cross_gland_kappa`,
  `cross_gland_lambda`. **Cross-gland (island) dispersal** (`κ·dispersal_rate`, distance-weighted or
  uniform) seeds one gland's lumen from another's (lumen→lumen, bypasses the wall → confined DCIS, no
  breach); a stroma cell never hops; wired into BOTH the exact and tau dispersal paths. Per-deme
  `gland_id` (−1 = stroma) + `gland_lumen_demes` + `gland_centers` ground truth, surfaced as
  `cell_data["cell_gland"]`; `viz.plot_grid` gained `color=["gland_id"]` and `["cancer_frac"]`.
  **OFF-BY-DEFAULT & byte-identical** (n_glands=1 + κ=0 + fill=1.0 + uniform K = the old single ring;
  golden hashes). One founder → many clonally-related foci (the inter-gland hops are the spread tree,
  a DCIS phylogeography). Ships: `tests/test_ductal_field.py` (12), `validate_ductal_field.py` →
  `validation_ductal_field.png` (grid growth time-series + a CELL-resolution 2D-section row via the new
  `viz.plot_grid(expand_demes=True, section_frac=…)`), `PARAMETERS.md`. Handoff:
  `handoffs/ductal_field_substrate.md`. **Prerequisite for the revised compartment-selection v1.**
- **DONE (2026-07-21) — spatial diagnostics validation.** `validation/validate_spatial_diagnostic.py`
  → `validation_spatial_diagnostic.png` (+ `tests/test_spatial_diagnostic.py`): at a mid generation
  (before takeover) shows CLONAL TERRITORIES (per-deme genetic PC1 + Moran's I — the structure a
  genotype-id map hides under infinite-sites; lower dispersal tends to sharpen it), spatial SELECTION
  (breach/stromal_survival trait maps), spatial CNA (per-deme ploidy), and NICHE-vs-GENOTYPE expression
  (emt tracks the epithelial wall, corr ≈ 0.95; proliferation flatter). Verifies the spatial layer end
  to end.

## Genotype→phenotype in the structured setting (DESIGN_phenotype_plasticity.md)
- **DONE (2026-07-21) — v1 compartment-dependent selection (COUNT engine, on the ductal field).** Two
  gene-based heritable axes (`prop_breach`, `prop_stromal_survival` + `_effects`, off by default)
  attenuate two compartment hazards (`spatial_params.{epithelial_barrier, stromal_hazard}`, default 0),
  in the EXACT shape of the existing immune term:
  `death += barrier · compartment_fraction(deme) · (1 − trait)`. **BOTH keyed to the deme's LIVE
  cell fraction** (epithelial wall / stromal cells — symmetric with immune; the stroma is seeded at
  MODERATE density by the substrate so its live fraction is meaningful); pressure tracks where the
  resident normals still are and softens as cancer dilutes them. Normals are never cleared (barrier
  selects on their *presence*, the `DESIGN_crowding.md` immortal-normal invariant). On the ductal-field
  substrate this yields **DCIS→IDC**: a lumen-founded lesion is confined by the wall + stroma (DCIS),
  spreads multi-focally via the confined cross-gland route, and invades the stroma (IDC) only once a
  subclone evolves `breach` (cross the wall) + `stromal_survival` (survive the stroma); each trait a
  mutation → sequenceable → recoverable. **COUNT engine only** (`count._death_rate`); the cell-engine
  mirror is DEFERRED with the count-only substrate. OFF-by-default & **byte-identical** (golden hashes;
  and on the multi-gland field; `binomial(1,0.0)` short-circuits so the layout stream is untouched).
  The compartment is also an R13 route-3 **niche field** (`epithelial`/`stromal` in `microenv_truth`
  → `niche_program_map`), so the SAME clone expresses the invasive/emt program more at the epithelial
  front — the genetic-vs-niche confound (validation: genotype-controlled r≈0.46, niche ≈72% of emt
  variance), present with ZERO carried epistate. Readout-only. Ships:
  `tests/test_compartment_selection.py` (14), `validate_compartment_selection.py` →
  `validation_compartment.png` (DCIS→IDC grid growth series), `PARAMETERS.md`, manuscript paragraph.
  Handoff: `handoffs/compartment_selection_v1.md`.
- **LATER — v2 the plastic epistate** (a memoried + noisy state `s` + the `(τ, σ, β)` identifiability
  map; DESIGN_phenotype_plasticity.md §3). Build ONLY once a benchmark needs the carried state — the
  three things a post-growth, memoryless readout cannot represent: memory / hysteresis, selection
  acting on the phenotype, and env-independent persisters. The basic confound is NOT a reason (v1 has
  it). This is where the hard-to-tune knobs live; keep them out of v1.

## Cell-state trajectories & differentiation (R12; plan-first, LATER)
- **LATER (design-first, like F8; whiteboard stage).** Add a CONTINUOUS cell-state / differentiation
  axis that EMERGES from the evolving genome (not read off a fixed input tree, unlike
  scMultiSim/SymSim/PROSSTT/dyngen). Abstraction: cell state = a point `z` in a low-dim space of
  expression PROGRAMS on a Waddington landscape set by (a) a differentiation hierarchy (baseline +
  pseudotime), (b) the GENOTYPE (deforms the landscape — block / de-differentiation / new program /
  plasticity, via a new **differentiation-regulator** gene role in `Selection`), and (c) the
  microenvironment (reuse the F8 niche modifier). Expression composes multiplicatively:
  `expr ∝ base · CNA-dosage · exp(Σ z·loading) · niche` — same pattern as CNA dosage + F8.
- **Three tiers (mirror F8):** v1 readout (draw `z` at materialization; off-by-default, bit-identical);
  v2 inherited-and-drifting `z` along the lineage (TRUE trajectories + pseudotime + RNA-velocity ground
  truth + the pseudotime-confound benchmark); v3 state→fitness coupling (shares R8b's engine
  constraints). Build v1 first; the science is in v2.
- **Payoff:** a ground-truth trajectory/RNA-velocity benchmark no fixed-tree simulator can produce —
  and the SAME "structure misleads inference" story as PEtracer / multi-region trees (clonal
  territories entangle lineage + state + space → when does pseudotime read the true differentiation
  axis vs get hijacked?). New benchmark-suite member; mechanistic analogue of LARRY clone+state+fate.
- **Do NOT build** a mechanistic GRN/SDE engine (dyngen/SymSim/scMultiSim turf; GRN/ATAC deferred).
- **Open decisions before coding** (see doc §7): mechanistic depth, v1-vs-v2 staging, first phenomenon
  to anchor (EMT is attractive — genotype AND niche), #programs, and the v2/v3 count-engine question
  (carry per-cell `z` without breaking reproducibility/caching/tau-leaping = R8b). Design doc:
  `DESIGN_celltrajectory.md`; research framing: `RESEARCH_QUESTIONS.md` R12.

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
- **DONE (2026-07-18) — three executed SCIENCE showcase notebooks** (in `notebooks/`, not yet in
  mkdocs): `combining_scdna_scrna.ipynb` (emergent CN→expression dosage; reconstruct clones from RNA
  via DNA CN profiles), `wgd_allele_cna.ipynb` (WGD + the allele layer: allele-only-detectable states
  rise with WGD; BAF imbalance localises to malignant CNAs), `gene_programs.ipynb` (R13 programs vs the
  positional CNA confound; self-contained sklearn-NMF recovery). Each grows a `structure_radius>0`
  spatial gland, analyses the MIXED tumour (malignant + subsampled microenvironment, ~50% purity),
  visualises the spatial structure, is self-contained in the core env, and points to the full
  external-tool benchmark under `validation/`.
- **DONE (2026-07-22) — migrated the showcase notebook SUITE onto the DUCTAL FIELD + compartment
  selection**, and added the flagship `compartment_selection_confound.ipynb` (DCIS→IDC + the
  genetic-vs-niche `emt` confound; partial r≈0.4). `base_sim.py` now grows the multi-focal ductal field
  (12 glands, compartment barriers ON, `breach→emt` + `epithelial→emt` coupling) to ≥10k cancer; every
  notebook re-executed on it. `base_simulation` + `compartment_selection_confound` carry the mandatory
  `plot_grid` growth-grid time-series; `tree_inference_dna` gains the inter-gland phylogeography (spread
  tree + island-bottleneck divergence); `scrna_visium_integration` spans the multi-focal section with a
  `thin_section` workaround for the Visium all-cells-pooling limitation (`DESIGN_ductal_field.md` §3.1
  engine TODO, not fixed here). `handoffs/showcase_notebooks_ductal.md`.
- **DONE (2026-07-22) — 4-gland field + results-quality fixes (user follow-up).** Dropped `n_glands`
  12→**4** (legible multi-focal section, bigger ducts: `grid_size=40, gland_radius=4, K_duct=60`).
  Fixed three degenerate results: (1) `combining_scdna_scrna` is now **allele-aware** — clones defined on
  total CN + `segment_allele_cn`, reconstructed with the `cell_rna_baf` signal (the Numbat idea on the
  RNA side); the allele lever lifts clone-recovery ARI ~0.01 (dosage) → ~0.14, and the nan/constant-input
  correlations are guarded. (2) `compartment_selection_confound` samples the confound + escape-trait
  panel **mid-transition** (breach ~0.72, still segregating) instead of the swept end-state → variance
  split niche ~83 % / genetic ~17 % (was 95/5), partial r≈0.43, a real in-gland-vs-stroma breach gradient.
  (3) `gene_programs` NMF on **log1p(CPM)** (0.38→0.45). Also added `base_sim.expanded_tissue_rgb` — a
  fast, pymuller-free cell-resolution grid (the `plot_grid(expand_demes)` per-clone Muller colormap is
  O(#genotypes) and hangs at ~10^4 genotypes).
- **DONE (2026-07-24) — docs landing-page growth ANIMATION** (`handoffs/landing_animation.md`).
  The landing hero renders the full metastatic arc (DCIS → breach → stromal → seeding →
  resection → chemo → resistant relapse) as the docs Home hero. It is now reproducible from the ordinary
  CLI — `isccsim --sim-config configs/landing.yaml` + `isccgif --compartment --splash` (seed 2, the
  tutorial's tumour + met + treatment; render logic in `iscc.visualization.compartment` /
  `iscc.tumor.arc`). The earlier bespoke `notebooks/landing_animation.py` generator was removed.
  Layout = a 2×2 gridspec: cell-resolution
  deme-grids on the LEFT (primary top, metastasis bottom), each compartment's Muller on the RIGHT sharing
  one tumour-time axis. ALL panels share one clone colormap keyed by the stage-dominant selective trait
  (blue proliferation / orange duct-escape / green stromal / purple met / red chemo resistance); the four
  sweeps read spatially + as Muller bands, events annotated. Two render modes: `main(splash=False)` (the
  default general path — full legend/labels/counts) and `--splash` → the minimal HERO variant at
  `docs/assets/landing_hero.gif` (no legend/labels/counts, "Time" x-axis, CENTERED Noble "fish"/stream
  Muller, seamless `#0d1117` background, ~2140×1210, ~17 s/loop). Hero `<img>` in `overrides/home.html`
  repointed off the placeholder. viz.py additions (all back-compatible, default byte-identical):
  `plot_grid`/`_expanded_cell_grid` gained `empty_color`; `plot_muller`/`_draw_muller_panel`/
  `plot_muller_compartments` gained `centered` (symmetric `baseline="sym"` stack).

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

## Engine bug — carrying capacity not enforced (DONE 2026-07-14; spatial realism)
- **DONE — `carrying_capacity` is now a real per-deme cap** (Option A, `DESIGN_crowding.md`). The old
  crowding death was a FIXED absolute rate `min(death_rate·K, max_death_rate)` (≤0.5), but selection
  raises each clone's `division_rate` up to `max_birth_rate` (0.8) — measured 0.30→0.78 over 30 gens —
  so once evolved div > death cap, growth was positive regardless of occupancy → demes overfilled
  (~1,200–4,200 cells/deme at nominal K=10), a dense PILE not a spatial spread.
- **Fix shipped:** density-dependent death RELATIVE to each clone's own evolved division rate,
  `death = death_rate + (div − death_rate)·(1+margin)·occupancy/K`, clamped at `maximum_death_rate`
  raised to **≥ `max_birth_rate`** (default 1.0). Applied to BOTH engines (count + cell), verified
  identical by an engine-agreement test. `carrying_capacity=None/0` → well-mixed (unbounded) regime,
  the explicit replacement for the old K=1 hack; the single-deme SISTEM benchmark re-measured (~5M in
  ~2.5 min, still < 3 min). Shipped configs re-tuned (example grid 50 / K 10 / structure 20 → ~10k
  cells spread across ~1,260 demes; glandular/mixed K 5). Full suite re-baselined + green.
- **Consequences resolved:** (1) the honest spatial scaling story is documented (a genuinely spatial
  tumour at millions of cells is HPC-bound ∝ occupied demes; the sub-3-min claim is well-mixed/single-
  deme only, and stays). (2) `carrying_capacity` semantics fixed in `PARAMETERS.md`. (3) `diagnose()`
  gained a "demes over-filling" check (now passing). (4) PEtracer confound (100%→32%) and multi-region
  spurious parallelism (0.219 vs 0.004) both re-confirmed after re-baselining; figures regenerated.

## Engine bug — F8 gene programs are not comparable across patients (DONE 2026-07-15; with R13)
- **DONE — F8's program designation now draws from the LAYOUT stream.** `prog_rng` was
  `default_rng(self.seed + 9973)` — the per-run EVOLUTION seed — so `_hypoxia_genes` /
  `_cci_target_genes` DIFFERED between patients in a cohort and "the hypoxia program" was not
  comparable across tumours (F8 predated the cohort's `layout_seed` decoupling and was never migrated;
  the dedicated-stream intent was right, the seed source was wrong). Now
  `default_rng(layout_seed + LAYOUT_OFFSET_F8_PROGRAMS)` (`constants.py` registry), so two patients
  sharing a config share their niche programs exactly as they already share their oncogenes.
  Pinned by `test_programs.py::test_f8_niche_programs_are_layout_seeded`. **The full suite stayed
  green with no test changes** — `test_microenvironment.py`'s assertions are statistical, not pinned
  to particular gene indices — confirming the "results statistically unchanged" expectation.
- **General rule (user, 2026-07-15):** anything that is a property of the GENOME/landscape (gene→program
  map, `loading`, dosage sensitivities `s_g`, the epistasis network) comes from the **layout stream**;
  event-level draws stay on the run seed. Use **independent sub-streams per component**
  (`SeedSequence(layout_seed).spawn(n)`) so changing e.g. `n_programs` doesn't reshuffle the oncogene/TSG
  layout.

## CNA-mechanism parity (R10) — focal / WGD / whole-chromosome (design-first)
- **Closes the one honest capability gap vs CINner/SISTEM** (iscc is arm-resolution). Design +
  difficulty decomposition in `DESIGN_focal_cna.md`. Key insight: the current per-segment genome makes
  WGD **cheap**, whole-chromosome **moderate**, focal **the refactor** (needs allele-specific interval /
  run-length CN — recommended over fine-binning: O(#breakpoints) not O(#bins), and shared plumbing with
  R13's ASE). Fitness + viability already handle it (the 2026-07-14 viability fix makes `max_ploidy` bind).
- **DONE — v1 WGD** (2026-07-17, handoff `handoffs/wgd.md`). `wgd_rate` is a separate per-division event
  channel in `CancerCell.mutate` (off by default → byte-identical when off, verified vs the pre-WGD
  baseline); WGD duplicates all copies on both homologs via the existing `update_genome_summary_cnv`
  seam, so the reject-at-birth viability gate (`max_ploidy`/`max_cn`) drops non-viable doublings for
  free. Ground truth `is_wgd` (per genotype, monotone) surfaced as `cell_data["cell_wgd"]` in both
  engines. `tests/test_wgd.py`; `validation/validate_wgd.py` shows cohort WGD prevalence sweeping through
  the real ~30–50% PCAWG band and the doubling+loss ploidy signature (near-diploid ~2 vs WGD ~3–3.5).
  Table-1 WGD cell added.
- **DONE — WGD allele-state axis** (2026-07-18, `validate_numbat.py --wgd-rate`). Measurement corrected
  the naive premise: a **pure** doubling (the diploid 1+1 -> 2+2) is *unidentifiable* from relative
  expression + BAF (it is
  allelically balanced and cancels under per-cell normalisation → both inferCNV AND Numbat infer ~2n for
  WGD cells; iscc reproduces the limit). What WGD *does* create as the doubled genome erodes is high-copy
  **allelic imbalance** — even-total states (4+0, 3+1) whose total CN matches a balanced 2+2, so only the
  allele layer can see them. The axis scores allelic-imbalance-STATE recovery *controlling for total CN*
  (the total-CN benchmark had collapsed Numbat's `loh` state into `neu`=2, discarding the signal). WGD
  raises the allele-only-detectable segment fraction ~4-6× (≈1%→≈4-6%); Numbat recovers imbalance at
  AUC ≈0.7-0.8 where inferCNV sits at chance. New: `segment_allele_cn`/`score_numbat_imbalance`,
  `numbat_runner.R` emits per-seg `cnv_state`/P(imbalance)/P(loh) + degrades to neutral (never crashes)
  when Numbat finds nothing, `tests/test_integration.py::TestWgdAlleleState`, `fig:numbat_wgd`.
- **LATER — v2 whole-chromosome** (chromosome→segment grouping) and **v3 focal** (the interval refactor;
  gate on whether the CNA-caller/Numbat results need GISTIC-peak resolution).

## Engine / inference follow-ups
- **NOW — two chemotherapy laws are in use across the paper's figures.** `kill_mode="proliferation"`
  is set in exactly ONE place, `validation/validate_escape_modes.py:66` (with `kill_rate=1.0`,
  `chemo_steps=120`). Every other treatment in the repo takes the `"additive"` default:
  `configs/landing.yaml` (the docs hero), `notebooks/base_sim.py` (the metastasis notebook),
  `validation/cohort_common.py:165` (the personalized-medicine cohort),
  `validation/validate_treatment.py`, and `notebooks/02_tumor_growth.ipynb`. So the four-escape-modes
  figure models cytotoxic chemo under one law and every other treatment result under another.
  They are not equivalent: `count.py::_kill_amount` states that the additive mode "SELECTS for the
  fast clones and the deposit regrows mid-course" and that "one absolute number cannot serve clones
  whose birth rates differ threefold" — the proliferation mode was written in `05cda0c` to fix exactly
  that, then never propagated past the file it was written for.
  MEASURED on the hero arc (seed 2, grow phase byte-identical across all three, so these are clean
  A/Bs): additive 1.5 -> the deposit is ERADICATED, no relapse, `diagnose()` reports `[FAIL] extinct`;
  additive 1.5 with `treatment_resistant_effects` raised to 6.0 -> resistance expands DURING the
  course (70 -> 42,128), no nadir, and the relapse decays 98.9% -> 58.6% resistant; proliferation 1.0
  -> a clean nadir (3,860 -> ~230) and a relapse, but only 10.9% resistant at 36 steps because ~82
  sensitive survivors out-grow the cost-bearing resistant clone. Both additive failures are what the
  docstring predicts, at each end of the range.
  DECIDE which law each result should use. `cohort_common` and `validate_treatment` feed published
  figures, so this is not a mechanical find-and-replace. Found 2026-08-27 while re-rendering the hero.
  RESOLVED FOR THE HERO 2026-08-27 (`fe6e498`): `configs/landing.yaml` adopts the escape-modes drug
  wholesale, so the split is now 4-vs-2. `cohort_common`, `validate_treatment` and
  `02_tumor_growth.ipynb` are still on additive.
- **NOW — the "How treatment acts on cells" tutorial teaches the OTHER engine's formula.**
  `notebooks/02_tumor_growth.ipynb` (cells 11-13) tells the reader that chemotherapy "raises a cancer
  cell's death rate by a factor `rate_multiplier ** (1 - treatment_resistance)`" and plots that curve
  with `rate_multiplier=5.0`. That is the CELL engine's law (`iscc/treatment/chemotherapy.py::_apply`).
  Every tutorial in the docs runs `mode: genotype` (`notebooks/example_config.yaml:38`), and the count
  engine never reads `rate_multiplier` at all — it uses `kill_rate` through `_kill_amount`, whose own
  docstring says "the CELL engine is a THIRD model, not either of these ... The two engines therefore
  do NOT agree under treatment."
  So the one page that explains how treatment works explains a mechanism the reader's own simulations
  do not use, and tunes a parameter they ignore. The notebook does not actually dose its tumour (cell
  13 says so), which is why no output ever contradicted the text.
  Fix depends on the decision above — teach whichever law the genotype engine is settled on — and
  needs a re-execution. Found 2026-08-27 during the notebook consistency pass.
- **NEXT — the driver layout is uniform across arms, but `s_arm` is inferred against arm content
  that is not.** `selection.py` draws driver roles with
  `rng.choice([-1,0,1], p=[prop_driver/2, 1-prop_driver, prop_driver/2])` — the SAME rate on every
  segment — so simulated per-arm oncogene/TSG counts are binomial noise around a constant. Real arms
  are strongly uneven (17p TSG-rich, 8q oncogene-rich), and `inference/genome.py::GenomeSpec` already
  carries those COSMIC/Davoli counts and uses them for the `s_arm` prior and the Charm comparison. So
  the engine's driver layout contradicts the annotation the real-genome inference is scored against.
  Found 2026-08-27 while assessing a (rejected) gene-naming layer; it is independent of naming and
  matters for the flagship Charm/PCAWG figure. Decide whether `prop_driver` should become per-arm and
  content-informed, or whether the arm-CN model's `prop_driver=0` scope already sidesteps it.
- **LATER — M3b HPC rerun** — fewer, bigger tumors (~800 × 8000-cell, tau-leaped) for the canonical
  Charm/PCAWG figure; HPC-bound. The flagship real-genome figure. **When we target the spatial-scale /
  runtime issue, the apt comparison is Noble's `demon`** (deme-based spatial sim, `noble_spatial_2022`;
  robjohnnoble/demon_model — verify software cite), NOT SISTEM (that was the well-mixed comparison). demon
  is deme-structured like iscc's engine → the right cells/s + time-to-N head-to-head for a genuinely
  spatial tumour; iscc already validates evolutionary-mode indices against Noble (M3a). See
  `DESIGN_crowding.md`. (user 2026-07-17)
- **LATER — `estimate_visium` kappa-on-sparse-data refinement** — fit on expressed genes; minor
  estimator robustness, doesn't affect defaults.
- **LATER — RESEARCH_QUESTIONS R1–R11** — 3D (R1), AI histology (R2), metastasis/multi-site (R9),
  focal CNAs/WGD (R10), recommender (R11), etc. Each a substantial new track; promote when committed.

## Engine/docs bug — WGD saturates at 100% ✅ RESOLVED 2026-08-26 (it was a CATEGORY ERROR)
- **RESOLVED.** Not a calibration problem and not an engine regression: the tutorial compared a
  WITHIN-TUMOUR cell fraction against PCAWG's COHORT statistic (fraction of *tumours* that are WGD).
  100% is correct behaviour — WGD is early and clonal, so a WGD+ tumour is WGD throughout. Lowering
  `wgd_rate` to force 33% would have fitted a number to the wrong denominator AND put the tutorial at
  odds with `validate_wgd.py`, which already does it correctly (majority-of-cells => tumour is WGD,
  PCAWG band applied to cohort prevalence). Fixed in `c0328e1`: prose reads the number at the right
  level, the printed line is labelled a within-tumour fraction, and the mean-ploidy line guards the
  now-empty non-WGD set (was printing `nan`).
- **Paper verified, no edit needed.** Full 20-seed sweep: prevalence moves through the PCAWG 30-50%
  band, calibrated rate **0.05 -> 45%**, ploidy bimodal (2.01 vs 4.01). NOTE a `--quick` (6-seed) run
  reports 0.08 -> 33%, which is NOISE — at 20 seeds 0.08 gives 85%. **Do not pick a rate from
  `--quick`.** `wgd_allele_cna.ipynb` uses 0.05, matching the calibrated rate.
- Original report kept below for context.
- **WAS:** Re-executing the tutorials against the current engine moved WGD from a third of malignant
  cells to **all** of them:
      base_simulation        WGD fraction (malig)  32.8% -> 100.0%
      wgd_allele_cna         33% of malignant WGD  ->    100%; mean ploidy "malig non-WGD" = **nan**
      combining_scdna_scrna  allelic imbalance     21%   ->  65%
  This **contradicts `wgd_allele_cna`'s own markdown** ("roughly a third of human tumours (PCAWG)
  carry one"), which the old 33% appeared to confirm, and it leaves the WGD-vs-non-WGD contrast with
  no non-WGD cells — hence the `nan`.
- **It is not a WGD code change.** No commit since 2026-08-09 touches WGD meaningfully (≤3 diff
  lines each). `wgd_rate` is 0.05 per **mutating division** and `Cell.is_wgd` is inherited and never
  reset (`cell.py:128`), so the fraction is a function of divisions-per-lineage and drifts toward
  saturation. **The old 33% was never pinned to the PCAWG anchor** — it was where that run length
  happened to land. Most likely tipped by `2af92d3` (layout rng reseed) pushing a near-binary WGD
  sweep over the line; NOT `de3ccb7`, whose `crowding_law` defaults to "logistic" (off).
- **Fix before the next `publish-main.sh`:** recalibrate so the tutorial lands near 1/3 at its own run
  length (or pin the anchor another way, e.g. report WGD at a fixed generation). A published tutorial
  must not print `nan` or contradict its own text. Check `manuscript/figures/validation_wgd.png` and
  `validation/validate_wgd.py` at the same time — the same drift may affect the paper figure.

## Notebook execution — PATH gotcha (2026-08-25)
- `reads.ipynb` shells out to `dwgsim` / `bwa` / `samtools`. Executed via nbconvert **without the env's
  `bin` on `PATH`**, `shutil.which` returns `None` for all of them and the notebook silently degrades
  to a no-op that still exits 0 — committing that pass would publish a broken tutorial. Re-run with
  `PATH="$CONDA_PREFIX/bin:$PATH"`. Only `reads.ipynb` is affected (checked across all 17).
- Runtimes for the two slow ones, so future caps are not set too low: `base_simulation` 22 min,
  `compartment_selection_confound` 43 min. Full pass ≈ 1h50m, serial.

## Evolution modes vs real tumours ✅ SOLVED 2026-08-26 — the "5% in hull" was an ARTEFACT
- `validate_evolution_modes.py` reports only **5% of real tumours inside iscc's (n, D) hull**, which
  reads as "iscc is far more clonally diverse than real tumours". **It is an observation-model
  mismatch, not a modelling failure.** iscc's `D` is inverse-Simpson over the FULL genotype census
  (no detection limit); the empirical `D` comes from published multi-region phylogenies with a
  handful of RESOLVED clones — **median 7, max 22** over the 43 tumours. Inverse-Simpson over k
  categories is bounded by k, and `D <= n_clones` holds for **all 43** rows (observed max D 11.25).
  A real tumour in that dataset *cannot* score D=30; its phylogeny has ~7 nodes.
- **Measured (`mode4_scratch/evomode_threshold_test.py`, same 64-tumour sweep, pooling — not
  dropping — every driver combination below a cancer-cell-fraction threshold):**

        CCF thresh   sim D median   sim clones med   D in real band   hull coverage
          0.00 (census)    26.98            59              17%             5%
          0.01              9.31            26              73%            51%
          0.02              4.11            12             100%            53%
          0.05              1.64             4              75%            37%
          0.10              1.25             2              60%            23%

  At a **2% detection limit** — what multi-region bulk actually resolves — iscc's median D is **4.11
  against the real median 3.96**, every simulated tumour lands in the real D range, and hull coverage
  goes **5% -> 53%**. The control behaves exactly as predicted: `n` is a per-cell mean, is
  threshold-independent (0.89-16.41), and agreed with the real 0.48-12.90 all along. Coverage PEAKS
  at ~2% and falls at 5-10% (over-thresholding collapses D below the real median) — and ~2% is the
  limit that yields clone counts near the empirical median of 7. Self-consistent.
- **To promote:** add a matched-observation panel to `validate_evolution_modes.py` (report hull
  coverage at census AND at a 2% CCF limit, stating why the census comparison is not like-for-like),
  then the figure becomes publishable — it currently sits unused precisely because the census number
  looks like a failure.

## Integration benchmarks: REAL tools, but NOT realistic data (found 2026-08-26)
- **Tools: genuine, verified.** `clonealign_runner.R` -> `library(clonealign)`, `numbat_runner.R` ->
  `library(numbat)`, `rctd_runner.R` -> `library(spacexr)`, `treemhn_runner.R` -> `library(TreeMHN)`;
  Python side scDEF 0.6.1 / cNMF 1.7.1 / mhn 1.2.3 / cell2location / harmony / infercnvpy, each in its
  own `iscc-<tool>` env. Versions agree between the paper, `README_integration.md` and what is installed.
- **Data: a TOY substrate.** Every real-tool benchmark grows its tumour from `integration_common.py`:
  **grid 20x20, K=8 (<=3,200 cells), `structure_radius=5` (ONE ring), 12 x 50 = 600 genes, 750 steps.**
  `validation/realistic_regime.py` — built, tested, and the regime everything else is calibrated to —
  is **grid 48/96/170, a ductal FIELD of 3/5/8 glands, 12 x 500 = 6,000 genes, max_cells 50,000**.
  So the benchmarks carrying the paper's central integration thesis run at ~1/10 the genes and a
  single ring instead of the field. `realistic_regime` is imported by `sweep_calibration.py`,
  `sweep_score.py`, `notebooks/base_sim.py` and four tests — **by no integration benchmark**.
- **No rationale is recorded.** `integration_common.py`'s comment justifies the STRUCTURE (segments
  for CNAs, a normal compartment for inferCNV's reference, time for subclones) but never mentions
  scale, and neither does `README_integration.md`. 600 genes is a real concern for the gene x cell
  tools specifically (scDEF, cNMF, inferCNV, cell2location) where real data is ~20k genes.
- **Decide before submission:** either migrate the integration benchmarks onto `realistic_regime`
  (the plan already in `realistic-regime-migration`), or state the substrate and defend it explicitly
  in the manuscript. Right now the paper implies realistic data and does not say otherwise.

## Real tools in the notebooks ✅ MOSTLY DONE (2026-08-26)

**Architecture (user decision):** analysis notebooks LOAD pre-generated data; they never simulate.
`validation/make_analysis_data.py` does the `iscc` half once and writes plain tables to
`analysis_data/` (GITIGNORED — numbat alone is ~80 MB). This is what makes an R-kernel notebook
possible at all: a notebook has one kernel, and growing a tumour is Python.

**Four R notebooks, each on its own kernel, all succeeding:**

    tool_clonealign_R.ipynb   accuracy 0.54 (chance 0.25, majority 0.49), ARI 0.26, AUC 0.88
                              (was 0.81/0.47/0.94 while L came from iscc's TRUE copy number; it
                              now comes from HMMcopy's call, which is the whole point)
    tool_numbat_R.ipynb       malignant-vs-normal AUC 0.977 over 1,677 cells
    tool_rctd_R.ipynb         MAE 0.058 vs 0.277 flat baseline, mean per-type r 0.974
    tool_treemhn_R.ipynb      both planted DAG edges the TOP 2 of 12, on SCITE-inferred trees
    tool_scite_trees.ipynb    ancestor-descendant recall 0.91, precision 0.80
    tool_hmmcopy_R.ipynb      within 1 copy on ~2/3 of (clone, segment) entries
    tool_pyclonevi.ipynb      prevalence vs true CCF r 0.96, clustering ARI 0.73
    tool_mhn_bulk.ipynb       planted edge rank 1 of 12 at every CV lambda; control finds none
    tool_scdef_cohort.ipynb   shared-program recovery mean matched cosine 0.50

IRkernel is installed in all four `iscc-<tool>` envs; kernelspecs `ir-clonealign` / `ir-numbat` /
`ir-rctd` / `ir-treemhn` are written directly (`IRkernel::installspec` shells out to `jupyter`, which
is not on those envs' PATH). `notebooks/r_preamble.R` carries the RETICULATE_PYTHON fix — without it
reticulate searches a cached uv interpreter and reports "Valid installation of TensorFlow not found"
for a correctly installed package.

**Cohort integration FIXED — Harmony and scDEF were never actually running.** Four stacked faults:
harmonypy not installed; `sc.external.pp.harmony_integrate` incompatible with harmonypy 2.0 (it
transposes Z_corr, right for 1.x, wrong for 2.x — Harmony converges and the result is discarded, so
call `harmonypy.run_harmony` directly); Jupyter's `MPLBACKEND=module://matplotlib_inline...` inherited
by the subprocess, which the tool env cannot import, killing `import scdef` (run_tool now strips
MPLBACKEND/PYTHONPATH — this covers cNMF too); and `n_epoch=100`, which is validate_programs'
`--quick` setting and under-trains scDEF to noise. Now: Harmony runs, scDEF mean cosine 0.08 -> 0.39.

**THE STRUCTURAL LESSON, worth more than any individual bug:** every one of these failures produced
plausible output and exit code 0. `reads.ipynb` silently produced nothing when binaries were off PATH;
`cohort_shared_programs` silently substituted per-batch centring and in-core NMF. The handler also
truncated exceptions to `str(e)[:200]`, so every scDEF failure read as a bare "import scdef". Fixed
structurally: untruncated errors, and every fallback prints `!! <TOOL> DID NOT RUN`.

### Outstanding
- **cNMF clean recovery (DEFERRED by user 2026-08-26).** cNMF now runs in `gene_programs.ipynb` but
  scores 0.07 mean cosine vs sklearn NMF's 0.30 — not the ~0.5 validate_programs reports. Cause is
  the TUMOUR, not the tool: `base_sim.EXPR()` layers a strong niche arm (`niche_program_strength=3.0`,
  epithelial -> emt) because that notebook demonstrates the genetic-vs-niche CONFOUND, so much of the
  variance is compartment-driven and cNMF's consensus keeps the niche structure. Ruled out by
  measurement: gene misalignment, normal-cell dilution (0.02->0.05), the dropout matrix (0.05->0.07).
  Fix would be a separate small cNMF notebook over a dataset built with the validation expression
  config — keeps `gene_programs` as the confound demo it was designed to be.
- **rctd dataset is NOT on the realistic ductal field.** It grows `deconv_common`'s grid-26 / one-ring
  / 300-gene substrate because `build_section`'s spot radius / pitch / section radius are tuned to that
  grid; repointing them at grid-96 would silently change how many cells land in a spot, which is the
  quantity the deconvolution benchmark measures. Migrate the GEOMETRY with it.
- **The integration benchmarks' realistic regime is opt-in, not default.** `regime="realistic"` /
  `ISCC_INTEGRATION_REGIME=realistic`, default still "toy". Switching the default restates published
  numbers, so it needs a deliberate pass per benchmark.

## Housekeeping
- **DONE — commit the backlog of uncommitted `dev` work.** The tree is clean; that item had gone
  stale. (Was: manuscript catch-up + positioning edits, re-executed notebooks, `DESIGN_recommender.md`,
  `DESIGN_features.md` F8, `RESEARCH_QUESTIONS.md` R11, `handoffs/F8_*.md`, `notebooks/TUTORIALS_PLAN.md`.)
- **DONE (2026-08-25) — `../STATUS.md` rewritten.** It still described the predecessor project
  `tumorevo` and was dated Jan 2025.

---

## Ground-truth leaks in the analysis datasets (found 2026-08-26)

- **DONE 2026-08-26 — TreeMHN's trees are now inferred.** `_treemhn` genotypes each patient
  single-cell (500 cells, fd 0.01, ad 0.20) and reconstructs the tree with **SCITE**, which is
  TreeMHN's own assumed upstream. iscc's true trees are still written as `truth_trees.csv`, held
  back. SCITE is built from source into `iscc-scite` (`validation/scite_common.py`;
  `notebooks/tool_scite_trees.ipynb`). Measured: SCITE edge recall 0.84 / precision 0.94, 22 of 36
  trees exactly right; TreeMHN then recovers both planted edges at ranks 1-2 (Theta +1.64, +0.02,
  against +5.09/+4.44 on the true trees — the cost of inferring through dropout).
  Gotcha worth remembering: `Tree_ID` must be unique per patient or TreeMHN reads the cohort as one
  tree and dies in `remove_duplicates`.
- **DONE 2026-08-26 — clonealign's `L` is now HMMcopy's call.** `_clonealign` reads the per-clone
  profiles from `analysis_data/dna/hmmcopy/clone_cn_called.csv` instead of iscc's true per-segment
  CN. Cost of closing it: accuracy 0.81 -> 0.54 (majority 0.49), mean AUC 0.94 -> 0.88. The two
  small clones stay near-perfect; the two large ones, which truly differ in ONE segment of twelve,
  do not survive depth-based calling on a WGD genome. Swap HMMcopy for SCICoNE when it is packaged
  (author is doing that separately; recommendation: bioconda/conda-forge recipe + a `shutil.which`
  fallback for `binary_path`, then scikit-build-core/cibuildwheel wheels, then pybind11 to drop the
  subprocess entirely).
- **DONE 2026-08-26 — `tree_inference_dna` retired**, replaced by three notebooks on ONE tumour:
  `tool_hmmcopy_R` (CN from depth), `tool_scite_trees` (tree from single-cell genotypes),
  `tool_pyclonevi` (clones from bulk).

### Gotchas from that build, worth not rediscovering

- **HMMcopy needs UNIFORM amplification.** scDNA's MDA/MALBAC defaults (kappa=5) leave the median
  locus at zero reads and `correctReadcount`'s GC loess dies with "span is too small". Use
  `kappa=500, mu_depth=60` (DLP+-like) and aggregate loci into bins.
- `HMMsegment(NULL, getparam=TRUE)` returns a template with `mu = NA`; passing it gives "missing
  value where TRUE/FALSE needed". Let `HMMsegment` derive parameters from the corrected data.
- `correctReadcount(mappability = 0.6)`, not the 0.9 default — iscc's mappability spans ~0.33-1.0.
- `plotBias` needs `KernSmooth`.
- **Ploidy is not identifiable from depth**, even with diploid cells in the run: a fixed read budget
  makes a tetraploid cell's total depth look diploid. Anchoring pins the reference to exactly 2 and
  leaves the tumour at ~2 when the truth is 4. Recovering it needs allele fractions (ichorCNA-style
  purity/ploidy search). This is the same wall `_scdna_concordance` documents.
- **PyClone-VI's truth must be the CARRIER SET**, not "the clone that carries it most" — the latter
  splits truncal mutations across labels and gives ARI 0.00 on a fit that is actually fine.
- Bulk on the raw field is 19% purity and only the truncal cluster separates; macro-dissect to ~50%.

## MHN needed its own cohort — why (settled 2026-08-26)

`notebooks/tool_mhn_bulk.ipynb` is in the nav and works, on `analysis_data/mhn/`. Two findings, both
measured against the real tool rather than assumed:

- **iscc's pairwise fitness `E` is invisible to a presence/absence method.** It sets how large the
  carrying clones GROW, not how often a combination arises, so a planted-E cohort and a matched
  zero-E control have *identical* presence marginals. This confirms validate_epistasis' headline
  finding against the real tool.
- **Accessibility gating does survive into presence** (a child cannot arise before its parent — a
  hard zero in the joint). But TreeMHN's cohort, grown 500 steps at event_size=8, has every parent
  fixed at 1.00: no variance, Theta collapses to zero at every lambda from 3e-3 to 0.3. **Time is
  the lever, not module size** — at 150 steps the parents sit at 0.78/0.83 while the gated children
  still reach 0.06/0.25. `MHN_EVENT_SIZE=7, MHN_STEPS=150, MHN_N_PATIENTS=300`.

Result: `E3->E2` is the strongest promoting entry of twelve at every CV-selected lambda, and the
no-DAG control never reports it. `E0->E1` sits at the power limit (E1 in 6% of patients) and the
notebook says so with a lambda-sensitivity table. Checked and rejected as fixes: more patients
(makes E1 rarer), eight different network draws (none gives two common children), and dropping the
1-SE rule (the control then produces false positives).
