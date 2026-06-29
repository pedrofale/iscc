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
- **The count emission is a pluggable step** — keep everything above (biology → library → batch)
  shared and swap only the final draw. **Default `count_model="nb"`** (field standard: Splatter/
  DESeq2/edgeR, composes cleanly with the multiplicative factors, what `estimate()` fits). Optional
  **`"dm"` (Dirichlet-Multinomial)**: total `N_c = ℓ_c`, proportions `p_c ~ Dirichlet(β_gb·λ_cg·κ_b)`
  — the compositional model (fixed library total, gene–gene competition for reads) for later
  robustness studies. (A scDesign-style copula over NB marginals is a third future option for
  realistic gene–gene correlation.)

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

## C. Read simulation (read-count matrices → FASTQ / BAM)

Follow the standard 3-stage DNA-seq pipeline (as SISTEM, Weiner 2025): **per-cell reference →
coverage model → third-party read simulator**. Only the coverage model is bespoke; do NOT
reimplement the sequencer error model — delegate stage 3 to a validated tool (**ART**, DWGSIM,
pIRS, InSilicoSeq). Phased:

- **C1 — allele-specific *count / coverage* realism** (highest value; needs NO nucleotide sequence):
  per-locus coverage scaled by **copy number** (amplified region → proportionally more reads — how
  CNA callers work), depth dispersion, GC/mappability, + binomial VAF sampling with an error rate
  → **read-count / coverage / VAF matrices**. This is what most CNA/SNV callers actually consume
  (SISTEM emits these *separately* from raw reads), and it works on the abstract bitset genome
  today. **Breadth-aware** (WGS/WES/panel, §D); for single-cell, apply ADO + amplification bias
  first. Extends `data/dna.py` (already has a `data_mode='reads'` stub).
- **C2 — raw reads (FASTQ)**: needs a nucleotide reference. Two options:
  - *(A) synthesise* a random per-segment reference once, apply each cell's SNVs/CNAs → per-cell
    FASTA. Works today, but lacks real sequence context (repeats, GC, mappability) — whether
    caller-benchmarks transfer is open (RESEARCH_QUESTIONS R5).
  - *(B) anchor to the real genome* via the M3b real-genome mode (segments = chromosome arms →
    hg38 coordinates) and use the **real reference sequence**. Better long-term path for read-level
    realism; the preferred option once real-genome mode is mature.
  Then build the per-cell reference + coverage (C1) and call a **third-party read simulator** → FASTQ.
- **C3 — BAM**: align the FASTQ back to the chosen reference (for tools that consume BAMs).

Gate C2/C3 behind C1; C1 alone serves the many methods that take count/coverage matrices, not reads.

**Read emission is per-modality** — the count matrix is the universal interface (what most methods
consume *and* the input these read simulators take), but the read tool + reference differ:
- **DNA** (genome reference): ART / DWGSIM / pIRS / InSilicoSeq.
- **scRNA, droplet/10x** (count matrix → reads, barcodes+UMIs): **scReadSim** (Yan 2023, learns from
  real data; current best) ≫ minnow (Sarkar 2019, count→reads but poor coverage realism).
- **scRNA full-length / Smart-seq3** (5′ UMI + full-length): less standardised — polyester-style bulk
  RNA read sim + a UMI layer. **Known gap**, not a blocker.
- **spatial (Visium)**: read-level rarely needed; count-level (F6) suffices for nearly all
  spatial-method benchmarking (reads would be a 10x-style sim with spatial barcodes).

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
- **F3 — DONE** — **scRNA batch model + 10x / Smart-seq3 presets** (the user's headline ask);
  multi-batch output with ground-truth labels. `data/batch.py` (`BatchHyperParams` + `Batch`:
  per-gene factor β_gb, per-batch depth/dispersion, ambient soup, doublets, dropout; **pluggable
  count emission** `COUNT_MODELS={"nb",...}`, `"dm"` Dirichlet-Multinomial stubbed as a drop-in
  seam). `data/rna.py` rewritten: `PROTOCOL_PRESETS` (10x: ambient+doublets+dropout; smartseq3:
  deeper, low dispersion, well-in-plate nesting, dropout off), Splatter-style composition
  (β reshapes per-gene, library sets total), AnnData output with clone/coords/batch in `.obs`,
  `run_scrna_batches` (shared / split designs) + `concat_batches`. CLI: `isccdata -a scrna
  --protocol {10x,smartseq3} --batches N --design {shared,split} --count-model nb` → one labelled
  h5ad per batch + `scrna_all_batches.h5ad`. `load_cell_data` now also loads `cell_type` (clone
  ground truth). Tests: `tests/test_assay_scrna.py`. Demo: `notebooks/assay_scrna.ipynb`
  (sweeps depth/dispersion/dropout/batch-strength/#batches/protocol → counts, UMAP, library-size
  dist, batch mixing). Hyper-param names = M2 `estimate()` targets.
- **F4** — bulk DNA technical (depth, GC bias, error, VAF/coverage tables) across **WGS/WES/panel**
  breadth = read-counts C1.
- **F5** — single-cell DNA (ADO + per-cell amplification, nested in run-batch), same WGS/WES/panel breadth.
- **F6** — spatial batch (spatially-correlated capture field, diffusion, section plane).
- **F7** — read emission: **C1 count/coverage matrices** (copy-number-scaled, breadth-aware) first,
  then C2 FASTQ (synthetic or real-genome-anchored reference → third-party read simulator) → C3 BAM.

## Validation hooks (per DESIGN_inference conventions)
- Batch model: *recover* injected batch params via `estimate()`; show two same-tumor batches share
  biology but differ technically (e.g. kBET / iLISI-style mixing only after correction).
- Benchmark utility: integrate two batches, measure recovery of the *known* clone/type labels.
- DNA/spatial: validate coverage/GC and spatial-autocorrelation distributions vs real references.

## G. Deliverables — per-module demo notebooks (required for every feature)

Every feature module ships, alongside its **code + tests**, a **dedicated demo notebook** under
`notebooks/` that (a) shows how to use the module and (b) demonstrates the **impact of its key
parameters**, starting from a *realistic upstream input*. These complement the end-to-end pipeline
notebooks (`01_pipeline_walkthrough`, `02_tumor_growth`, `03_data_overview`), which show the whole
chain; these are single-module deep-dives. (They also become the reference for "what each knob does"
in the eventual web platform — see `DESIGN_platform.md`.)

Each notebook should: take a realistic input from the previous stage (e.g. the scRNA notebook starts
from a tumour **dissociation/sample**, not a raw tumour); **sweep the module's key parameters** and
visualize their effect on the output; and surface the ground truth (clone / type / coords) so the
parameter effect is interpretable.

| Notebook | Module (milestone) | Sweeps / shows |
|---|---|---|
| `sampling_biopsy.ipynb` | biopsy/sampling (F1) | region geometry (needle/punch/multi-region) → observed clonal heterogeneity |
| `dissociation.ipynb` | dissociation (F2) | composition bias, doublet rate → cell-type proportions vs ground truth |
| `assay_scrna.ipynb` | scRNA assay (F3) | **given a dissociation:** depth, dispersion, dropout, **batch strength / #batches**, protocol (10x vs Smart-seq3) → counts, UMAP, library-size dist, batch mixing |
| `assay_dna.ipynb` | bulk + sc DNA (F4/F5) | **breadth (WGS/WES/panel)**, depth, GC bias, ADO → coverage / VAF / CNA profiles |
| `assay_spatial.ipynb` | Visium (F6) | capture field, diffusion, spot size, section → spot counts + spatial autocorrelation |
| `treatment.ipynb` | treatment (shipped) | chemo/targeted/immuno, dose schedule, adaptive vs continuous → burden over time (backfill) |
| `reads.ipynb` | read emission (F7) | coverage, error → FASTQ / VAF tables |
