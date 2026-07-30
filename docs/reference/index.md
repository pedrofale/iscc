# API Reference

`iscc`'s Python API mirrors the CLI pipeline — **grow → sample → assay**, with **treatment**
applied during growth and optional **inference** of parameters from real data. Import the package
as `import iscc`; see the [Overview](../overview.md) for how the stages connect.

## Tumor

Grow a selection-driven tumor on a grid of **demes** — small tissue patches, each holding up to a
carrying capacity of cells. Both engines are spatially explicit at the deme level and simulate the
same process (birth, death, mutation, and dispersal under a CINner fitness model); they differ only
in how the cell population is represented.

| | |
|---|---|
| [`iscc.tumor.GenotypeTumor`](GenotypeTumor.md) | Represents each deme as genotype **counts**. Fast and scalable — the default. |
| [`iscc.tumor.GlandularTumor`](GlandularTumor.md) | Represents each cell as its own **object**. Exact, but does not scale to large tumors. |

## Treatment

Therapies applied to the tumor during growth.

| | |
|---|---|
| [`iscc.treatment.Chemotherapy`](Chemotherapy.md) | Cytotoxic therapy acting on dividing cells. |
| [`iscc.treatment.TargetedTherapy`](TargetedTherapy.md) | Therapy against a driver-defined subclone. |
| [`iscc.treatment.Immunotherapy`](Immunotherapy.md) | Immune-cell–mediated killing in the microenvironment. |
| [`iscc.treatment.Surgery`](Surgery.md) | Resect a compartment (e.g. the primary). |

## Sample

Draw cells from the grown tumor before assaying.

| | |
|---|---|
| [`iscc.sample.Biopsy`](Biopsy.md) | Spatially-localized sample of cells. |
| [`iscc.sample.Dissociation`](Dissociation.md) | Whole-tumor dissociation into a cell suspension. |

## Data

Generate single-cell and bulk molecular data from the sampled cells.

| | |
|---|---|
| [`iscc.data.scDNA`](scDNA.md) | Single-cell DNA (copy number + SNV). |
| [`iscc.data.bulkDNA`](bulkDNA.md) | Bulk DNA-seq. |
| [`iscc.data.scRNA`](scRNA.md) | Single-cell RNA expression. |
| [`iscc.data.Visium`](Visium.md) | 10x Visium spatial transcriptomics. |

## Reads

Generate sequencing reads (FASTQ, optionally aligned BAM) from the assayed counts. DNA reads
use DWGSIM/ART; RNA reads use a synthetic 10x transcriptome or the scReadSim template. These
call external read simulators, so the corresponding binaries must be installed.

| | |
|---|---|
| [`iscc.data.reads.emit_dna_reads`](emit_dna_reads.md) | Bulk / single-cell DNA reads (→ FASTQ, optional BAM). |
| [`iscc.data.reads.emit_scrna_reads`](emit_scrna_reads.md) | Mutation-aware single-cell RNA reads. |
| [`iscc.data.reads.emit_visium_reads`](emit_visium_reads.md) | Spatial (Visium) reads. |

## Inference

Fit the **tumor's evolutionary parameters** (division / mutation / dispersal / selection rates)
to real data by Approximate Bayesian Computation over the growth simulator.

| | |
|---|---|
| [`iscc.inference.ABC`](ABC.md) | Approximate Bayesian computation over evolutionary rates. |
| [`iscc.inference.Prior`](Prior.md) | Product prior over the named parameters to infer. |
| [`iscc.inference.Posterior`](Posterior.md) | Result of an ABC run — samples and point estimates. |

## Calibration

Fit each **assay's technical parameters** (library size, dispersion, dropout, capture field, …)
to a real reference dataset, so simulated data matches a target platform.

| | |
|---|---|
| [`iscc.data.estimate_rna`](estimate_rna.md) | Calibrate the scRNA assay from a real count matrix. |
| [`iscc.data.estimate_dna`](estimate_dna.md) | Calibrate the DNA assay from coverage / allele statistics. |
| [`iscc.data.estimate_visium`](estimate_visium.md) | Calibrate the Visium assay from a real section. |
