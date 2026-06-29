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

## How to use this file
- These are **questions, not scheduled work** — promote one to a DESIGN doc / milestone when we
  commit to building it.
- 3D (R1) and histology (R2) are the current aspirational headliners and are *not* yet in
  `DESIGN_features.md` F1–F7; they would each be a substantial new track.
