# iscc tutorials — a course in tumor data analysis on simulated data

Status: plan (2026-06-30). A pedagogy track of notebooks that teaches the standard concepts of tumor
data analysis (DNA, RNA, spatial) the way a bioinformatics student would learn them — but on **iscc
simulated data instead of real data**. The pedagogical superpower: because the data are simulated, we
can **reveal the ground truth** in every lesson and *grade* how well the standard method recovered it
— something real data never allows.

These are distinct from the existing notebooks: `01/02/03` (pipeline/tumor/data overview) and
`assay_*`/`reads` (per-module deep-dives, for *users of iscc*). The tutorials are for *learners of
tumor data analysis* and follow one story end-to-end. Proposed home: `notebooks/tutorials/`.

---

## The running example (one tumor, told across the whole course)

A **spatially structured tumor** is grown from a single transformed clone. It accumulates copy-number
events and mutations under selection and forms a few spatial clones. When it reaches a large burden it
receives **non-adaptive (continuous) chemotherapy**, which kills most sensitive cells but **does not
eradicate it**: a pre-existing/early **resistant subclone** survives the bottleneck and **expands**,
so the tumor **relapses as a resistant tumor**. We assay it at **two timepoints — pre-treatment and
relapse** — with all four modalities (bulk DNA, scDNA, scRNA, Visium).

This single arc motivates every analysis: *find the clones, find the resistant one, characterize its
genome and expression, locate it in space, and watch treatment select for it.*

**Deliverable:** `notebooks/tutorials/generate_course_tumor.py` (config + fixed seed) that grows the
tumor, applies the chemo schedule, and writes the multi-timepoint multi-modal dataset **plus the
ground-truth answer key** (per-cell clone/genotype/CNA/SNV, spatial coords, the treatment timeline,
and the label of which clone is resistant) under `notebooks/tutorials/course_data/` (git-ignored,
regenerated on first run). Every tutorial loads from there.

---

## Curriculum

Each notebook: **motivate the concept → run the standard analysis on iscc data → reveal the ground
truth → grade the method → discuss when it works and when it breaks.**

### Part 0 — Setup
- **T0 · The case study & the ground truth.** Build/load the running example. Visualize the growth
  curve with the treatment timeline (grow → treat → partial response → resistant relapse). Show the
  answer key (clones, the resistant clone, the two timepoints, the four assays). Establish the
  "reveal-and-grade" pattern used throughout. *Concepts: ground truth, why simulation enables it.*

### Part 1 — Tumor evolution (the biology behind the data)
- **T1 · Clonal evolution & selection.** Muller plot of clone frequencies through growth and
  treatment; spatial clone map; fitness; drivers vs passengers; **treatment as a selective pressure**
  (sensitive clones crash, the resistant clone sweeps). *Concepts: selection, clonal sweep, bottleneck,
  cost/benefit of resistance.*

### Part 2 — DNA sequencing
- **T2 · Bulk DNA-seq.** VAF distributions; clonal vs subclonal variants; the neutral 1/f expectation
  and deviations; purity/ploidy intuition; **detecting the resistant subclone as a VAF shift from
  pre-treatment to relapse.** *Concepts: VAF, CCF, mutation calling, neutral vs selected.*
- **T3 · Single-cell DNA-seq.** Per-cell copy-number profiles; allelic dropout and noise; clustering
  cells into clones; **building the clonal phylogeny** and reading the resistant clade's CNAs/mutations.
  Grade the inferred tree vs the true lineage. *Concepts: scDNA, CNA calling, ADO, clonal phylogenetics.*

### Part 3 — RNA sequencing
- **T4 · scRNA-seq fundamentals.** QC, normalization, HVGs, PCA/UMAP, clustering, cell-type annotation
  (malignant vs stromal/immune). *Concepts: counts, normalization, dropout, clustering, annotation.*
- **T5 · CNV-from-expression & clone mapping.** Infer CNAs from expression (infercnvpy/copyKAT idea);
  identify malignant cells; **map RNA cells to DNA clones via the copy-number→expression dosage
  relationship** (the clonealign concept). Grade clone assignment vs truth; show *why* it works
  (dosage) and where it fails (low dosage effect). *Concepts: inferCNV, malignant-cell ID, RNA↔DNA
  linkage, the dosage assumption.*
- **T6 · Differential expression & the resistance program.** Resistant vs sensitive, and pre- vs
  relapse; find the resistance signature; pathway interpretation. **Show the scRNA variant-calling
  limitation** (a mutation in a lowly expressed gene is invisible in RNA) using iscc's read-consistent
  variants — a lesson real data can't teach cleanly. *Concepts: DE, signatures, expression vs genotype.*

### Part 4 — Spatial transcriptomics
- **T7 · Spatial (Visium) fundamentals.** Spot QC; spatial domains; spatial DE; Moran's I; **locate
  the resistant clone's territory** in the tissue at relapse. *Concepts: spots, spatial domains,
  spatial autocorrelation.*
- **T8 · Spatial deconvolution & clonal architecture.** Deconvolve spots into cell types
  (cell2location/Tangram idea); **map clones in space** and watch the resistant niche emerge from
  pre-treatment to relapse. Grade deconvolution vs the true per-spot composition. *Concepts:
  deconvolution, spatial clonal architecture.*
- **T9 · Microenvironment & cell–cell communication** *(depends on F8 — see `DESIGN_features.md` §H).*
  Hypoxia program in the tumor core; ligand–receptor crosstalk at clone/cell-type boundaries; niche
  analysis. **Flagged as forthcoming until F8 lands.** *Concepts: niches, CCI, hypoxia.*

### Part 5 — Integration & synthesis
- **T10 · Multi-modal integration capstone.** Tie DNA + RNA + spatial together: **trace the resistant
  clone across all three modalities from one ground truth.** What each modality sees, where each is
  blind, and why a *consistent* multi-modal benchmark matters (the non-circularity point). *Concepts:
  multi-omic integration, the value of consistency.*
- **T11 · (Optional) Measuring evolution rigorously.** Phylogenetics, dN/dS, the site-frequency
  spectrum, selection inference; **how treatment bent the evolutionary trajectory.** *Concepts:
  cancer evolution measurement.*

---

## Design notes
- **Audience/level:** intro bioinformatics / computational-biology graduate students; assume Python +
  basic statistics. Each notebook is self-contained but builds on the shared example.
- **Tooling:** Python-first — `scanpy`, `squidpy`, `anndata`, `infercnvpy`, `scikit-learn`. Heavy R
  tools (`clonealign`, `copyKAT`, `inferCNV`, `Numbat`) are introduced conceptually with a light
  Python stand-in for the lesson; an "advanced" appendix can call the real R tools. List/add deps to
  the env as the track is built.
- **Format conventions:** every notebook ends with a "grade the method" cell (recovered vs truth) and
  a "when does this break?" discussion — the consistent pedagogical beat.
- **Reproducibility:** one config + seed; `generate_course_tumor.py` regenerates the dataset; the
  answer key is loaded, never recomputed, so lessons can't drift.
- **Relationship to the paper:** several tutorials double as the *integration-method demonstrations*
  on the paper-1 backlog (T5 clonealign-style mapping, T8 deconvolution, T9/T10 niche+integration) —
  build once, use for both teaching and the paper figure.
- **Sequencing of work:** T0–T8 + T10 can be built now; T9 waits on F8; T11 is optional/advanced.
