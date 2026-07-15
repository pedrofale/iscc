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
  - CNA-from-expression: inferCNV/CopyKAT ✅; **Numbat** ⬜ (needs allele-specific expr/ASE, R13)
  - scRNA→Visium deconvolution: **cell2location, RCTD** ⬜ — FLAGSHIP (paired ref + true per-spot comp;
    can also test matched-vs-mismatched reference)
  - cell–cell communication / niche: CellChat / CellPhoneDB / COMMOT ⬜ (F8 ground truth)
  - scRNA cohort integration: **Harmony, scVI/scANVI** ✅ (+ Scanorama, LIGER; scIB metrics)
  - multi-sample Visium integration: GraphST / STAligner ⬜ (2D cross-patient; PASTE 3D out of scope, R1)
  - DNA cohort progression: **MHN, TreeMHN**, CBN/H-CBN, REVOLVER ◑ — R14 (epistasis) DONE, so the
    planted network + the scoring seam (`iscc.integrations.progression`: `to_mhn_matrix` /
    `to_treemhn_trees` / `score_edges` / `score_order`) now EXIST and the built-in co-occurrence
    baseline runs (`validation/validate_epistasis.py`). Running the REAL tools in their own
    `iscc-mhn` / `iscc-treemhn` envs is what remains.
  - pooled demultiplexing: vireo/souporcell (DNA), cell-hashing + scDblFinder (RNA) ✅
  - subclonal deconvolution (multi-region bulk): PyClone-VI / Pairtree / Clomial ◑ (oracle deconv done)
  - **gene-program / GEP inference: `scDEF` ⬜ FLAGSHIP** (+ cNMF comparator, Hotspot optional) — ground
    truth = the true `loading` matrix + per-cell `z` (needs the R13 program layer). Scored across a
    **SNV/CNA-burden sweep**; hypothesis: contiguous CNA dosage induces *positional* pseudo-programs that
    factor models absorb ⇒ recovery degrades with FGA (a sibling of the PEtracer confound). Handoff:
    `handoffs/expression_programs.md`.
- **Two PREREQUISITES — NOW IN PAPER 1 (decision 2026-07-14), design-first, not built. Handoffs ready:
  `handoffs/expression_programs.md` (R13) and `handoffs/epistasis.md` (R14).**
  - **Expression realism** — `DESIGN_expression.md` (R13). **Expression is modelled as GENE PROGRAMS**
    (the backbone) with two gene-level genotype overlays: dosage (CNA, contiguous) and cis-SNV
    (gene-level). Genotype couples at 3 levels — program activity (driver→program = R12 deformation),
    dosage, single-gene cis. **programs ⟂ CNAs** (functional/scattered vs positional/contiguous) is what
    makes the benchmarks non-circular. **R12 and R13 SHARE one program/`z` implementation** (R12 = `z`
    dynamics, R13 = `z`→counts + overlays) → the program layer is on the paper-1 critical path. Gates
    clonealign/inferCNV *fairness*, **Numbat/CalicoST** (ASE), **cardelino/PhylEx** (SNV). Hard engine
    prerequisite: **stop summing the `p`/`m` alleles** (ASE/BAF), shared with R10.
  - **Epistasis** — `DESIGN_epistasis.md` (R14, **paper 1**) ✅ **DONE**. Pairwise `E`, a conjunctive
    dependency DAG (`fitness` | `accessibility` gating) and mutual exclusivity are plantable in
    `Selection` (off by default → bit-identical; cached per event set → tau-leap safe), drawn from the
    layout stream (`LAYOUT_OFFSET_EPISTASIS`) so a whole cohort shares ONE network. Ground truth via
    `tumor.epistasis_ground_truth()` / `tumor.event_table()`; benchmark in
    `validation/validate_epistasis.py` → `manuscript/figures/validation_epistasis.png`; Results
    section `sec:epistasis`.
    **Honest headline (a NEGATIVE result, and the interesting one):** iscc's `E` acts on FITNESS
    (clone size) while MHN/CBN model the RATE of event ACQUISITION, and a cross-sectional
    event-presence matrix cannot tell them apart — so pairwise `E` is recovered at CHANCE regardless
    of cohort size (10→80) or interaction strength (0.25→2.0). Conjunctive constraints under
    **accessibility** gating are recovered perfectly (1.00 in every network draw, true AND
    reconstructed trees); the same DAG under **fitness** gating leaves no trace (conjunction holds in
    ~23% of lineages; apparent order swings 0.07–0.85 across draws, tracking which event is
    intrinsically faster). Next: run the REAL MHN/TreeMHN in their own envs against this same answer
    key — the seam and metrics are already in place.
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

## Engine bug — F8 gene programs are not comparable across patients (NEXT; small)
- **F8's program designation uses the RUN seed, not the layout seed.** `prog_rng =
  np.random.default_rng(self.seed + 9973)` (`count.py:142`) ⇒ `_hypoxia_genes` / `_cci_target_genes`
  DIFFER between patients in a cohort, so "the hypoxia program" isn't comparable across tumours. F8
  predates the cohort's `layout_seed` decoupling and was never migrated (the dedicated-stream intent was
  right; the seed source is wrong). **Fix:** draw them from a `layout_seed`-derived sub-stream (see
  `layout_rng`, `count.py:43-55`). Changes which arbitrary genes are hypoxia-responsive ⇒ regenerate the
  F8/PEtracer figures (results should be statistically unchanged) + re-check `test_microenvironment.py`.
  **Folded into `handoffs/expression_programs.md`** (R13 builds the general program layer and must use the
  layout stream anyway) — do it there, or standalone if R13 slips.
- **General rule (user, 2026-07-15):** anything that is a property of the GENOME/landscape (gene→program
  map, `loading`, dosage sensitivities `s_g`, the epistasis network) comes from the **layout stream**;
  event-level draws stay on the run seed. Use **independent sub-streams per component**
  (`SeedSequence(layout_seed).spawn(n)`) so changing e.g. `n_programs` doesn't reshuffle the oncogene/TSG
  layout.

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
