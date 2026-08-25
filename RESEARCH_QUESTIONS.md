# RESEARCH QUESTIONS: open problems behind iscc's feature goals

Status: living doc (started 2026-06-25).

This is **not** an implementation plan — the DESIGN docs cover that (`DESIGN_features.md` = feature
generation, `DESIGN_inference.md` = validation/estimation, `DESIGN_scalability.md` = engine). This
file collects the **open, research-grade questions** that each feature goal raises: things we don't
yet know how to do well, several of which are publishable in their own right. Some (3D tumors, AI
histology) are aspirational goals *beyond* the current F1–F7 roadmap.

Each entry: the question, why it matters, candidate approaches / sub-questions, and the feature it
serves.

---

## Theme 1 — Spatial dimensionality (2D → 3D)

### R1. How do we realistically extend a 2D ABM tumor to 3D without simulating the full 3D tumor?
**Why it matters.** Real tumors are 3D; biopsy needle paths, spatial-assay sections, histology, and
even the evolutionary-mode indices (Noble's "structure governs mode") all depend on 3D geometry. But
a full 3D ABM is far more expensive — cell count scales as L³ vs L², compounding the §7 size/cost
problem — so brute-force 3D may be infeasible at realistic sizes.

**Candidate approaches / sub-questions.**
- *Statistical lifting from sections.* Treat the 2D simulation as one section and generatively sample
  a stack of mutually-consistent parallel sections (consistent clone fields / cell-type maps along
  z). What conditional model preserves clone spatial coherence across z?
- *Multi-resolution.* A cheap coarse 3D deme lattice for global geometry + fine 2D refinement where a
  section/biopsy is taken (couple deme counts in 3D to within-deme 2D detail).
- *Boundary-only 3D.* Growth is boundary-driven, so explicitly simulate the 3D surface and treat the
  interior statistically.
- *Learned lifting.* Train a generative model (e.g. diffusion) on 3D references (organoid imaging,
  cleared-tissue volumes) to lift 2D fields to 3D.
- *What must be preserved?* Surface-to-volume ratio, 3D clone-patch-size distribution, and the
  evolutionary indices computed on 3D-realistic samples — define the invariants a "good enough" 3D
  extension must match.

**Serves:** spatial realism, sampling geometry (R3), histology (R2), the Noble validation under 3D.

---

## Theme 2 — Generative histology & imaging

### R2. Given a 2D ABM tumor, how do we use AI to generate a histology that matches it? And how does that extend to 3D?
**Why it matters.** Histology (H&E) is the most abundant real tumour data. A simulator that emits a
*matched* histology image turns the ABM ground truth (cell types, positions, density, architecture)
into a **labeled benchmark** for digital pathology — nuclei segmentation, tissue classification,
cell-type inference, and "omics-from-H&E" methods — with perfect ground truth that real data lacks.

**Candidate approaches / sub-questions.**
- *Layout-conditioned synthesis.* Map the ABM's cell-type / density / architecture maps → an H&E
  image with a conditional generative model (conditional GAN or diffusion; cf. layout-to-image and
  pathology-image-synthesis work). The cell-type/nuclei layout is the conditioning signal.
- *Faithfulness as the crux.* The generated image is only useful if what a method *infers from it*
  (nuclei positions, cell types, density gradients, tumour boundary) matches the ABM ground truth.
  How do we measure and enforce that fidelity (cycle-consistency to a segmentation, ground-truth
  recovery error)?
- *Beyond H&E.* Condition IHC/IF channels on the simulated marker expression (`cell_exp`); virtual
  multiplex staining.
- *3D extension.* Generate a **z-stack of mutually consistent serial sections** from the 3D tumour
  (R1) — consistency and registration across z — toward volumetric histology.
- *Inverse problem.* Can we infer ABM / tissue parameters *from* real histology? This would calibrate
  the simulator directly from H&E (links the inference layer, R6).

**Validation.** A trained classifier (or pathologist) cannot distinguish sim from real; downstream
segmentation/classification methods perform comparably on simulated vs real, and recover the known
labels.

**Serves:** a new modality (histology) beyond the current assay set; benchmark generation for digital
pathology; calibration-from-images.

---

## Theme 3 — Sampling & biopsy realism

### R3. How does biopsy geometry through a (3D) tumor bias observed heterogeneity?
**Why it matters.** Multi-region phylogenetics and clinical genomics are dominated by *what the
biopsy saw*. A needle core through a 3D tumour samples a 1D transect of a 3D clone structure — very
different from a 2D punch. Quantifying this bias is both a feature (F1) and a question.

**Sub-questions.** Optimal multi-region sampling to recover the true clone tree? How much does
needle-path geometry change inferred dN/dS, diversity, or the Noble indices? Calibrate dissociation
composition bias against paired single-cell + histology data.

**Serves:** F1/F2 (sampling layer); depends on R1 for a 3D substrate.

---

## Theme 4 — Assays, batch effects & benchmark transfer

### R4. Do iscc-generated benchmarks *predict* real-world method performance?
**Why it matters.** The whole value proposition is that iscc benchmarks (batch correction, CNA
calling, integration) generalise. This is an empirical, testable claim, not an assumption.
**Sub-questions.** Does a batch-correction method's ranking on iscc-simulated batches (known ground
truth) match its ranking on real multi-batch data? What realism is *necessary* for transfer — full
distributional match, or only the few statistics methods are sensitive to? **Serves:** the batch
model (F3) and `estimate()` (M2).

### R5. How faithfully can an abstract-genome read simulator stand in for real reads?
**Why it matters.** iscc's genome is synthetic; emitting reads (F7) needs a synthetic reference. Can
benchmarks of variant/CNA callers on iscc reads transfer to real genomes, or does the lack of real
sequence context (repeats, mappability, GC structure) break it? **Serves:** read emission (C/F7),
breadth-aware DNA (WGS/WES/panel).

---

## Theme 5 — Inference & identifiability

### R6. Which real-data statistics break the selection ↔ CNA-rate identifiability, and do inferred parameters transfer?
**Why it matters.** CINner had to *fix* a parameter to identify selection. We want to (a) find summary
statistics (e.g. SNV + CNA jointly, multi-region structure) that resolve the degeneracy, and (b) test
whether parameters inferred on iscc-simulated data recover known values *and* generalise to real
cohorts (PCAWG/TCGA). **Serves:** M1/M3b inference; the §A.3 identifiability analysis.

---

## Theme 6 — Scale & multi-resolution

### R7. What is the minimal agent representation that preserves the statistics methods care about?
**Why it matters.** The deme abstraction and tau-leaping (§7) trade fidelity for size. When does
tau-leaping diverge from exact dynamics? What is the coarsest deme/agent representation that still
reproduces the SFS, CNA landscape, and spatial indices a downstream method consumes? **Serves:** the
§7 size/cost assessment; makes realistic-scale (10⁶–10⁹) tumours feasible.

---

## Theme 7 — Microenvironment & treatment

### R8. How do we model immune dynamics (recruitment / migration / exhaustion) tractably at genotype-count resolution?
**Why it matters.** The current immune compartment is static; realistic immunotherapy and immune
predation need dynamics that still fit the count-based engine. **Sub-questions.** A spatial
recruitment field coupled to tumour burden? Do iscc-simulated adaptive-therapy schedules transfer to
predict good schedules in vivo? **Serves:** treatment realism; immune milestone.

### R8b. Should the microenvironment couple to FITNESS, not just expression? (F8 future extension)
**Why it matters.** F8 (microenvironment-driven expression, `DESIGN_features` §H) modulates the
expression **readout** only — the hypoxia and cell-cell-communication fields do not change division
or death, so growth is byte-identical F8 on/off. Real hypoxia slows proliferation, drives necrosis
and selects for hypoxia-tolerant clones; paracrine signalling can be pro- or anti-proliferative. The
open question is how to feed the F8 deme fields back into the per-genotype rates **without breaking
reproducibility, the genotype-count caching, or tau-leaping**, and whether the resulting
eco-evolutionary feedback (niche construction) reproduces known patterns (necrotic cores, invasive
fronts). **Serves:** F8 realism; ties to the treatment/immune milestone (R8).

---

## Theme 8 — Metastasis & multi-site dissemination

### R9. How do we extend iscc's single-tumour spatial model to multi-site / metastatic disease, tractably?
**Why it matters.** iscc currently grows one spatially structured tumour; real disease (and much
of clinical genomics) is **multi-site** — primary plus metastases linked by cell migration. SISTEM
(Weiner & Bansal 2025) shows this is feasible at clonal resolution with organotropic, genotype-
dependent migration and a migration-graph ground truth — but it's DNA-seq-only. The question is how
to add multi-site dissemination to iscc *while keeping the multi-modal + inference machinery*.

**Sub-questions / approaches.**
- A small number of coupled deme-grids (sites) linked by a migration process; per-site selection
  landscapes (a metastasis is a different microenvironment) reusing the per-arm/region selection of
  the real-genome mode.
- **Organotropism**: migration probability as a function of genotype–site fitness compatibility
  (seed-and-soil), vs a fixed pairwise-distance model.
- Ground truth to emit: the **migration graph** and per-site clonal composition — a labelled
  benchmark for metastatic-seeding / clonal-origin inference methods.
- How does multi-region/multi-site **sampling** (R3) interact — can iscc recover the true seeding
  topology from sampled data?
- Cost: multi-site multiplies the §7 scale problem; the generation-based clonal update (SISTEM /
  §7 tau-leaping) is the enabling primitive.

**Serves:** a new disease axis (metastasis); positions iscc against SISTEM with multi-modality +
inference. Not in F1–F7 — a substantial new track if promoted.

---

## Theme 9 — Genomic resolution & CNA mechanism richness

### R10. How do we add focal (sub-arm) CNAs, WGD, and whole-chromosome events without blowing up the count engine?
**Why it matters.** iscc's CNA model only does **amplify/delete a whole segment** — at its finest
(real-genome mode) that is **arm resolution**. Both competitors are richer: **CINner and SISTEM both
have the full mechanism set** — *focal (sub-arm) amplifications & deletions, chromosome-arm
missegregation, whole-chromosome missegregation, and whole-genome duplication (WGD)*. So iscc is at
parity for *arm-level* CNA landscapes (what M3b fits to PCAWG; the Davoli/Charm story), but **cannot
represent a tight focal amplicon (*MYC*, *EGFR*), a focal TSG deletion, genome doubling, or
whole-chromosome gains/losses** as distinct events — a real CNA-realism deficit vs *both* tools.
Focal CNAs are biologically central (recurrent oncogene amplicons / TSG focal deletions, GISTIC
peaks). The mechanism is *known* (both competitors implement it); the genuinely open part is doing
it cheaply.

**KEY INSIGHT (2026-07-16) — this is not one job.** The current genome (`cell.py`) is per-segment:
a segment is `{'p':[copy,…],'m':[copy,…]}`, each copy a whole-segment SNV bitset, so **copy number is
segment-granular** while SNVs are position-resolved. That splits the three events by difficulty:
- **WGD — cheap** in the current rep: duplicate every copy in `p` and `m` once. Newly well-behaved
  because the 2026-07-14 viability fix makes `max_ploidy` actually bind.
- **Whole-chromosome gain/loss — moderate:** needs a chromosome→segment grouping (real-genome/arm mode
  already groups segments into arms), then act on the whole set on one homolog.
- **Focal (sub-arm) amp/del — the real refactor:** the per-segment rep cannot express sub-segment CN.

**The representation decision (the crux) for focal.** Two ways to get sub-segment CN:
- *Fine-binning* (smaller segments): simple but memory/work scale **O(#bins)**, multiplied by the
  genotype-count caching — expensive at gene resolution.
- *Allele-specific interval / run-length CN* (**recommended**): each homolog's CN is a piecewise-constant
  function — a sorted list of `(start,end,copy_state)` intervals, `copy_state` carrying the SNV bitset.
  A CNA of ANY span (focal/arm/chromosome) is an interval op (split at breakpoints, change CN over
  `[start,end]`). Memory is **O(#breakpoints)** (tens–hundreds, not thousands) — more general AND more
  scalable than fine-binning, and essentially how CINner represents allele-specific CN. It subsumes
  segments/arms as special cases; it also shares the "per-homolog CN" plumbing with R13's ASE work.

**What you DON'T need to build:** fitness (a focal oncogene amp already boosts `division_rate` via the
existing dosage/driver pathway; `s_arm` scores arm/chromosome events) and viability (`max_ploidy`/
`max_cn`/`max_nullisomy` already bind post-fix). Only the event *geometry* is new. Each event type is a
mechanism with its own rate, **ABC-estimable** exactly as CINner fits its mechanism probabilities
(extends §A.0's `mut_prob`/`cnv_prob`). Must stay compatible with the genotype-count engine and
tau-leaping (§7).

**Staged plan → `DESIGN_focal_cna.md`:** v1 WGD (cheap; ships alongside the Numbat benchmark, which
detects WGD via BAF), v2 whole-chromosome, v3 focal (the interval refactor). WGD handoff:
`handoffs/wgd.md`.

- *Validation*: recurrent **focal** amplification of oncogenes / focal deletion of TSGs (the focal
  analogue of the arm-level Davoli/Charm result); focal-CNA frequency spectra vs real GISTIC peaks;
  WGD frequency (~30–50% of tumours) + the doubling+loss ploidy distribution vs PCAWG.

**Serves:** CNA-mechanism parity with CINner/SISTEM; richer DNA-assay ground truth (F4/F5 already
derive coverage from `seg_cns`, so focal events would yield realistic focal CNA profiles); more
mechanism-rate targets for the inference layer. Lives in the tumour engine (`mutate`/`Selection` +
real-genome mode), not F1–F7.

---

## Theme 10 — Experimental design ("recommender mode")

### R11. Can iscc recommend an experimental design, not just generate data?
**Why it matters.** Existing power/design tools (scPower, powsimR, POWSC for scRNA; PoweREST for
spatial; Tarabichi/Boutros for subclone reconstruction) are all single-modality and single-task
(usually DE/eQTL or subclone detection) and statistical rather than biology-generative. iscc's full
tumor→sample→assay→(reads)→treatment chain, scored against a known ground truth, could turn the
generative engine into a **goal-agnostic, multi-modal, biology-grounded design engine**: fit
technical/batch priors from a pilot (`estimate*`), then for a stated goal + budget assess which
analyses are feasible and which design is optimal — including cross-modal allocation and adaptive-
therapy monitoring (sequential design / value of information, the West–Anderson tie-in). **This is a
second mode and likely a separate paper.** **Serves / see:** `DESIGN_recommender.md` (full scoping);
operationalizes R3 (sampling), R4 (benchmark transfer), R8 (treatment-schedule transfer).

---

## Theme 11 — Cell-state dynamics & differentiation trajectories

### R12. How do we model continuous cell state / differentiation trajectories that EMERGE from the evolving genome — rather than being read off a fixed input tree — while staying tractable and multimodally consistent?
**Why it matters.** iscc's cell states are currently discrete (cell type × clone) plus the F8
microenvironment expression readout; there is no continuous differentiation axis. Real cells occupy a
*spectrum* of states set by (a) their position in a differentiation hierarchy (stem → progenitor →
differentiated — the source of pseudotime) and (b), for cancer cells, mutations that reshape which
states are reachable (differentiation block, de-differentiation, aberrant programs, plasticity). No
simulator couples a continuous cell-state trajectory to an EVOLVING, spatially-explicit clonal genome
under selection: scMultiSim / SymSim / PROSSTT / dyngen all take a differentiation tree (or a GRN) as
*input* and carry no genome evolution. Trajectory that *emerges from* clonal evolution + microenvironment
is iscc's white space — and the same non-circularity argument as clonealign applies.

**Framing (the abstraction).** Cell state = a point `z` in a low-dimensional space of expression
*programs* on a Waddington-style landscape whose attractors/barriers are set by three inputs: the
differentiation hierarchy (baseline), the genotype (which DEFORMS the landscape), and the
microenvironment (F8). Empirically supported by the limited set of recurrent expression meta-programs
across tumours (Gavish & Tirosh, Nature 2023) — a handful of program axes, not a full GRN, suffices.

**Candidate approaches / sub-questions.**
- *State→expression map.* Program-loading model: `expr_g ∝ … · exp(Σ_k z_k · loading[k,g])`, composing
  multiplicatively with CNA dosage + the F8 niche modifier (same pattern). How many programs; which to
  seed from known signatures (cell cycle, EMT)?
- *Genotype→landscape coupling.* Add a **differentiation-regulator** gene role to `Selection` (beside
  driver/TSG/dispersal/resistance); a clone's landscape = cell-of-origin baseline deformed by the
  diff-regulator mutations/CNAs it carries. Four deformation modes: block / de-differentiation / new
  program / plasticity. What is the minimal parameterisation?
- *Dynamics tier.* v1 readout (draw `z` at materialization — like F8, off-by-default, bit-identical);
  v2 inherited-and-drifting `z` along the lineage (true trajectories); v3 state→fitness coupling (stem
  divides, differentiated doesn't; therapy selects a state). The open engine question for v2/v3 is the
  same as R8b: feed a per-cell dynamical state into the genotype-count engine WITHOUT breaking
  reproducibility, the per-genotype expression caching, or tau-leaping.
- *The confound (the scientific payoff).* Under v2, pseudotime, clone and space become entangled
  (clonal territories co-locate lineage + state) — the SAME "structure misleads inference" theme as
  PEtracer and multi-region trees. When does a trajectory method (Slingshot/Monocle/PAGA/CellRank/
  scVelo) recover the TRUE differentiation axis vs get hijacked by clonal/CNA/spatial structure? iscc
  knows the true state, the true RNA velocity AND the confounders — a ground-truth trajectory/velocity
  benchmark no fixed-tree simulator can produce (mechanistic analogue of LARRY clone+state+fate data).
- *What NOT to build.* Not a mechanistic GRN/SDE engine (dyngen/SymSim/scMultiSim turf; GRN/ATAC already
  deferred). The latent-program model gets continuous states + pseudotime + velocity at a fraction of the cost.

**Validation.** Emit ground-truth pseudotime/velocity/state + program labels; score trajectory-inference
recovery vs truth; quantify the confound across the dispersal/territoriality sweep (parallels PEtracer);
check expression realism (recovered meta-programs resemble the Gavish/Tirosh set).

**Serves:** a continuous cell-state axis for the scRNA/spatial readout; a new benchmark-suite member
(trajectory + RNA-velocity inference) in the "iscc as a benchmarking substrate" arc; ties to F8
(niche→program), R8b (state→fitness), and treatment (therapy-driven lineage plasticity/resistance).
Design-first plan in `DESIGN_celltrajectory.md`. Not in F1–F7 — a substantial new track if promoted.

---

## Theme 12 — Genotype→expression realism (CNA & SNV coupling)

### R13. How should copy number and point mutations map to expression, so the DNA↔RNA integration benchmarks are non-circular and allele/SNV-based tools are testable?
**Why it matters.** The DNA↔RNA integration tools (clonealign, inferCNV/CopyKAT, Numbat, cardelino,
PhylEx) all *invert* a genotype→expression relationship; a benchmark is only fair if iscc's forward
model is not the inverse model the tool assumes. Today it largely is: CNA→expression is ~linear additive
dosage (`get_exp`, `cell.py`), the SNV→expression effect reuses the *fitness* parameter (entangled, no
functional classes), alleles are summed (no allele-specific expression / BAF), and genes are
independent (no co-expression structure). So the benchmarks risk being too easy / partly circular, and
the allele-based tools (Numbat, CalicoST) cannot be tested at all.
**Candidate approaches / sub-questions.** (A) per-gene **dosage sensitivity** + saturation (partial
buffering) so the CN law ≠ the assumption; (B) **allele-specific expression / BAF** by keeping the p/m
split through to expression (the genome is already allele-resolved); (C) **SNV functional classes**
(LoF→NMD, missense, splice, silent) decoupled from fitness, incl. the TSG two-hit with CNA loss; (D)
**co-expression / program** structure (= R12) for realistic covariance. How much is needed for transfer
(R4)? Where do sensitivities / SNV classes come from (calibrate from real CN–expression pairs; abstract
role-based vs real-genome annotation)?
**Serves:** the credibility of the whole DNA↔RNA integration benchmark suite; unlocks allele-based
(Numbat/CalicoST) and SNV-based (cardelino/PhylEx) tools. Design: `DESIGN_expression.md`. Ties to F8
(niche), R12 (programs), R10 (allele/focal CNA).

## Theme 13 — Epistasis & evolutionary-dependency structure

### R14. How do we encode epistasis / ordered dependencies in the selection model so cohort progression models have a known network to recover?
**Why it matters.** Cohort DNA-integration / progression tools (MHN, TreeMHN, CBN/H-CBN, REVOLVER,
RECAP) recover a network of promoting/inhibiting/ordering dependencies between events. iscc's fitness is
**additive** (driver *count* in abstract mode; per-arm in real-genome), so the true network is ~empty
and a benchmark would only measure false-positive rate. To make this a rich benchmark we must plant a
**known** dependency structure and show the method recovers it.
**Candidate approaches / sub-questions.** Pairwise epistasis `E_{ij}` (synergy/antagonism); conjunctive
/ ordered constraints (CBN-style DAG: B beneficial or accessible only after A → temporal order); mutual
exclusivity / synthetic lethality (negative interactions). Fitness-gating vs accessibility-gating for
order. Sparsity/topology/magnitudes that are detectable yet realistic. Must stay compatible with the
genotype-count engine + tau-leaping (interaction is a pure function of a genotype's event set → caches
per genotype). **Serves:** the DNA cohort-progression benchmark row (MHN/TreeMHN/CBN/REVOLVER); pairs
with the cohort milestone. Design: `DESIGN_epistasis.md`. Ties to R6 (identifiability), R10 (CNA events).

### R15. Does polyploidization→depolyploidization leave a detectable signature, and have real tumours been through it?

**The question, aside from iscc entirely.** Therapy- (or oncogene-) induced arrest → endoreplication →
unequal/multipolar division → aneuploid progeny is a plausible route by which cancer genomes acquire
their karyotypes (`DESIGN_senescence_escape.md`). If it leaves a **signature**, and that signature is
present in real tumours, that is evidence they passed through the process — most interestingly during
INITIATION rather than under therapy.

**Transcriptional vs genomic answer different questions.** A transcriptional signature marks cells
*currently* doing it — senescence markers, SASP, and a reported re-expression of **meiosis /
spermatogenesis** programs during these atypical divisions (mechanistically apt: neosis is a reductive
division borrowing a germline program). Useful for catching the act in a treated biopsy, useless for
history, because expression does not record the past. **Only the genome records a past event.**

**The predicted genomic event shape.** WGD, then **near-random WHOLE-CHROMOSOME missegregation with
frequent nullisomies** — many chromosomes redistributed in ONE division rather than lost one at a time.

**The discriminating statistic is TIMING/CORRELATION, not the events.** Gradual CIN loses chromosomes
independently over many divisions, so losses spread across phylogenetic branches; a neosis event dumps
them on ONE branch simultaneously. This is exactly how **chromothripsis** is identified (clustered
breakpoints, oscillating CN, randomness of fragment order — one catastrophe vs slow accumulation).
**Neosis would be to whole-chromosome aneuploidy what chromothripsis is to rearrangements.**

**The observation may already exist.** Navin's **punctuated copy-number evolution** (Gao et al.,
*Nat Genet* 2016, ng.3641): in TNBC most CN alterations are acquired **at the earliest stages in short
punctuated bursts**, followed by stable clonal expansion, with phylogenetics + modelling rejecting
gradual accumulation. That is the exact shape this mechanism predicts — burst, early, then stasis — and
nobody appears to have connected it mechanistically to polyploidization/depolyploidization.

**But consistent is not diagnostic.** Chromothripsis, breakage-fusion-bridge, or a single catastrophic
mitosis WITHOUT polyploidy all give punctuated bursts. Two tests separate them:
1. **4n-by-loss vs 2n-by-gain.** Neosis goes THROUGH a polyploid intermediate, so the burst should be
   preceded by WGD and the karyotype better explained as subtraction from tetraploid than addition to
   diploid. Copy-number timing methods can pose this.
2. **Homozygous losses unsurvivable from 2n.** THE SHARPER TEST. Multipolar division produces frequent
   nullisomies, and a 4n cell can lose regions that would be immediately lethal from 2n. A tumour
   carrying homozygous deletions of loci essential in a diploid context **needed a polyploid
   intermediate to get there** — a footprint of the ROUTE, not just the endpoint.

**Limitation: selection overwrites.** A tumour that did this at initiation and then grew for years has
had its karyotype reshaped and the randomness fingerprint degrades toward whatever was selected. Argues
for **early lesions** (DCIS, adenomas) and tumours where WGD is CLONAL, i.e. demonstrably early.

**Where iscc and the cell lines come in.** The cell-line system (`DESIGN_senescence_escape.md` §1) gives
the segregation kernel and the FRESH signature, before selection touches it — the calibration for what
to look for in the messy case. iscc could then run the forward model: apply a measured segregation
kernel, evolve under selection, and ask how long the signature survives and which statistics retain
power. **Blocked on the same gap as everything else here: `Cell.divide()` produces exactly equal
daughters, so iscc cannot currently simulate the event whose signature this question is about.**
**Serves:** a distinct paper from the escape-mode work; ties to R10 (WGD / whole-chromosome events).

---

## How to use this file
- These are **questions, not scheduled work** — promote one to a DESIGN doc / milestone when we
  commit to building it.
- 3D (R1) and histology (R2) are the current aspirational headliners and are *not* yet in
  `DESIGN_features.md` F1–F7; they would each be a substantial new track.
