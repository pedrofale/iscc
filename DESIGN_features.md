# DESIGN: simulation features (sampling, reads, assay batch effects)

Status: design / scoping (2026-06-25). Complements `DESIGN_inference.md` (that doc is the
*validation/estimation* roadmap; this one is the *feature-generation* roadmap). The two meet at
one point: the batch/technical parameters defined here are exactly what the Splatter-style
`estimate()` (DESIGN_inference §B / M2) fits from real data.

## Guiding principle: separate biology from technical; make technical hierarchical & instance-keyed

Every observed dataset = **biological ground truth** (tumor → sample) passed through an **assay
instance** = (protocol, batch realization keyed by a seed).

```
tumor (true per-cell state)  ──>  sample (biopsy/dissociation: biological/prep bias)
                                      └──> Assay instance = protocol × Batch(seed)  ──> observed data
```

Same tumor + two assay instances ⇒ **same biology, two technical signatures**. Because iscc
keeps the ground truth (clone, cell type, spatial coords) and labels every observation with its
`batch`, a multi-batch dataset is a *labeled* benchmark for batch correction / data integration /
replicate-concordance methods — the cell↔cell correspondence across batches is known exactly,
which real data can never provide. Support both **balanced** (one tumor split across batches) and
**confounded** (different tumors → different batches) designs, since confounding is the hard case
integration methods must handle.

## A. Sampling layer (`isccsample`)

Currently a uniform random subset; biopsy/dissociation/slice are stubs. Plan:

- **Solid biopsy — spatial region selection** over the deme grid:
  - *needle core*: a narrow strip of demes; *punch*: a disk; *multi-region*: k disjoint regions,
    each tagged with a `region` label. This reproduces multi-region heterogeneity (you only see
    the clones present in the sampled region) — the substrate for phylogeny studies.
- **Liquid biopsy (blood / CTC)**: sample a small number of circulating cells, biased toward
  high-dispersal / invasive clones; very low counts. Models liquid biopsy.
- **Dissociation biases** (sample-prep, *biological* not sequencing-technical):
  - cell-type-dependent dissociation efficiency → composition bias (e.g. immune/stromal vs
    epithelial recover differently); doublet formation; dissociation-stress signature; ambient
    release.
- **Slicing** (for spatial): select the 2D section presented to the spatial assay. iscc is 2D so
  the section = grid or sub-grid; a true 3D tumor + arbitrary cut plane is future work.
- Output: subset of cells + metadata (region labels, method) preserving per-cell ground truth.

## B. Assay batch-effect model (the core new piece)

### B.1 Hierarchy
`hyper-parameters` (protocol-typical magnitudes) → `per-batch realization` (one seed = one
reaction/plate/run) → `per-cell / per-gene observation`. The batch effect is a **property of the
assay instance**, applied consistently across all cells of that instance, and reproducible from
its seed.

### B.2 scRNA model
For cell *c*, gene *g*, batch *b*, with biological expression `λ_cg` (from the tumor's `cell_exp`
ground truth):

- **per-gene batch factor** `β_gb ~ LogNormal(0, σ_batch²)` — *shared across all cells of batch b*,
  varies by gene. `σ_batch` is the batch-strength hyper-parameter (Splatter's `batch.facScale`).
  This is the canonical batch effect: a per-gene fold-change common to a run.
- **per-batch depth** `μ_lib,b`; **per-cell library factor** `ℓ_c ~ LogNormal(μ_lib,b, σ_lib²)`.
- **per-batch dispersion** `φ_b`, **dropout**, **ambient profile**, **doublet rate**.
- Counts: `y_cg ~ NB(ℓ_c · β_gb · λ_cg, φ_b)`, then add ambient counts (from a batch ambient
  profile) and doublets (merge two cells' profiles at the batch doublet rate).

**Protocol presets** set which components dominate:
- *10x (droplet, UMI, 3′)*: significant ambient RNA + doublets; moderate depth; UMI suppresses
  amplification noise.
- *Smart-seq3 (plate, full-length, 5′ UMIs)*: higher sensitivity and lower dropout than 10x;
  the 5′ UMIs suppress amplification noise on UMI-counted molecules (unlike Smart-seq2), with
  full-length non-UMI coverage available for allele/isoform-level signal; extra nesting level —
  **plate** (= batch) **+ well** (per-cell position) effects.

### B.3 Running batches
Two `Assay` instances with identical hyper-parameters but different seeds → two batches; write
each as its own AnnData with `batch` in `.obs`, concatenate for integration benchmarking. The
biological signal (`λ`) is shared; only the `β`, depth, ambient draws differ.

## C. Read simulation (FASTQ / BAM)

Phased, because the abstract genome has no nucleotide sequence:
- **C1 — allele-specific *count* realism**: per-locus coverage (depth) + binomial VAF sampling
  from the bitset genome with an error rate → coverage/VAF tables. Extends `data/dna.py`.
- **C2 — read sequences (FASTQ)**: generate a synthetic reference sequence per segment/gene once,
  apply each cell's SNVs/CNAs, sample reads under a coverage + error model → FASTQ.
- **C3 — BAM**: align reads to the synthetic reference (optional; for tools that consume BAMs).
Gate C2/C3 behind C1 being validated; this is the heaviest feature.

## D. Per-modality technical / batch considerations

| Modality | Batch unit | Dominant technical factors | iscc model |
|---|---|---|---|
| **scRNA 10x** | reaction / lane | ambient RNA, doublets, depth, per-gene batch factor, capture efficiency | §B.2 |
| **scRNA Smart-seq3** | plate (+ well) | plate + well effects; 5′ UMIs (low amplification noise) + full-length coverage; higher sensitivity / lower dropout than 10x | §B.2 + plate/well nesting |
| **bulk DNA** (WGS / WES / panel) | library / run (+ capture kit) | coverage depth, **GC-bias curve**, mappability, PCR duplication, sequencing error, FFPE C>T deamination; breadth-dependent capture bias (see below) | per-batch GC→coverage curve + depth + error; per-bin CNA log-ratio noise |
| **single-cell DNA** (WGS / WES / panel; DLP, 10x-CNV, Tapestri…) | chip / run (+ per-cell amplification) | very low coverage, **allelic dropout (ADO)**, MDA/MALBAC uneven amplification (per-cell), GC bias, doublets; same breadth axis | per-cell amplification profile nested in run-batch; ADO rate |
| **spatial (Visium)** | slide / section | per-spot capture efficiency (**spatially autocorrelated**, edge effects), spot library size, lateral mRNA diffusion/bleed, spot→cell aggregation (~1–10 cells / 55µm), ambient, per-gene batch factor, **section plane** | spatially-correlated capture field + spot aggregation + diffusion kernel + §B.2 per-gene factor |

Key modality-specific notes:
- **Capture breadth (WGS / WES / targeted panel) is orthogonal to bulk-vs-single-cell** — both
  bulk and single-cell DNA can be any of the three. Breadth sets three things: (i) the **observed
  locus set** — WGS = all genome positions; WES = an exon/gene subset; panel = a small designated
  set (e.g. the driver genes); (ii) the **depth regime** — WGS low, WES moderate on-target, panel
  very high; (iii) the **dominant capture bias** — WGS GC bias, WES per-target/probe capture
  variability, panel per-amplicon bias. In iscc this is a `breadth` + `target_genes` parameter on
  the DNA assay (reusing `dna.py`'s existing `target_genes`), applied identically to bulk and sc.
  Downstream implication: panel → deep VAF on few drivers but poor genome-wide CNA; WGS → good CNA,
  shallow VAF; WES → coding SNVs + coarse CNA.
- **Bulk/sc DNA**: the batch effect that matters for *CNA calling* is the GC/coverage bias and the
  per-bin log-ratio noise level (what panel-of-normals normalization targets); for *VAF* it is
  depth-driven binomial noise + error rate. Model batch = GC-curve + depth + error (+ ADO for sc).
- **scDNA** has a *nested* technical structure: a per-cell amplification profile (the dominant
  noise) sits inside the run-level batch — analogous to Smart-seq3's well-in-plate nesting.
- **Spatial**: the distinctive requirement is that the technical noise is **spatially correlated**
  (capture efficiency varies smoothly across the slide), and that a "batch" also includes which
  2D **section** of the tumor was placed on the slide — part technical, part biological sampling.

## E. Architecture / API

```
src/iscc/data/
  assay.py        # Assay base: protocol + batch hyper-params + seed -> Batch realization
  batch.py        # Batch: per-gene factor draw, depth, dispersion, ambient, doublets (per modality)
  rna.py          # scRNA: protocol presets 10x / smart-seq3, applies Batch
  dna.py          # bulk + single-cell DNA × {WGS, WES, panel} breadth; GC/coverage/ADO, applies Batch
  visium.py       # spatial: capture field + aggregation, applies Batch
  reads.py        # C1 counts -> (C2 FASTQ -> C3 BAM)
```
CLI: `isccdata --protocol 10x --batches 2 [--batch-seed ...]` writes one labelled AnnData per
batch. Batch hyper-parameters are exactly what `estimate()` (DESIGN_inference §B) fits from real
data — so realistic defaults can be *learned*, not guessed.

## F. Milestones (features)
- **F1** — spatial solid biopsy (needle / punch / multi-region) + region labels.
- **F2** — dissociation biases (composition bias, doublets) + liquid biopsy.
- **F3** — **scRNA batch model + 10x / Smart-seq3 presets** (the user's headline ask); multi-batch
  output with ground-truth labels.
- **F4** — bulk DNA technical (depth, GC bias, error, VAF/coverage tables) across **WGS/WES/panel**
  breadth = read-counts C1.
- **F5** — single-cell DNA (ADO + per-cell amplification, nested in run-batch), same WGS/WES/panel breadth.
- **F6** — spatial batch (spatially-correlated capture field, diffusion, section plane).
- **F7** — read emission C2 (FASTQ) → C3 (BAM).

## Validation hooks (per DESIGN_inference conventions)
- Batch model: *recover* injected batch params via `estimate()`; show two same-tumor batches share
  biology but differ technically (e.g. kBET / iLISI-style mixing only after correction).
- Benchmark utility: integrate two batches, measure recovery of the *known* clone/type labels.
- DNA/spatial: validate coverage/GC and spatial-autocorrelation distributions vs real references.
```
