# API Reference

`iscc`'s Python API mirrors the CLI pipeline — **grow → sample → assay**, with **treatment**
applied during growth and optional **inference** of parameters from real data. Import the package
as `import iscc`; see the [Overview](../overview.md) for how the stages connect.

## Tumor

Grow a spatially-structured, selection-driven tumor.

| | |
|---|---|
| [`iscc.tumor.GenotypeTumor`](GenotypeTumor.md) | Fast genotype-level (count-based) engine — the default. |
| [`iscc.tumor.GlandularTumor`](GlandularTumor.md) | Cell-level engine with explicit glandular structure. |

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

## Inference

Fit `iscc`'s parameters to real data.

| | |
|---|---|
| [`iscc.inference.ABC`](ABC.md) | Approximate Bayesian computation over evolutionary rates. |
