# API Reference

`iscc`'s Python API mirrors the CLI pipeline — **grow → sample → assay**, with **treatment**
applied during growth and optional **inference** of parameters from real data. Import the package
as `import iscc`; see the [Overview](../overview.md) for how the stages connect.

## Tumor

Grow a spatially-structured, selection-driven tumor.

| | |
|---|---|
| [`GenotypeTumor`](GenotypeTumor.md) | Fast genotype-level (count-based) engine — the default. |
| [`GlandularTumor`](GlandularTumor.md) | Cell-level engine with explicit glandular structure. |

## Treatment

Therapies applied to the tumor during growth.

| | |
|---|---|
| [`Chemotherapy`](Chemotherapy.md) | Cytotoxic therapy acting on dividing cells. |
| [`TargetedTherapy`](TargetedTherapy.md) | Therapy against a driver-defined subclone. |
| [`Immunotherapy`](Immunotherapy.md) | Immune-cell–mediated killing in the microenvironment. |
| [`Surgery`](Surgery.md) | Resect a compartment (e.g. the primary). |

## Sample

Draw cells from the grown tumor before assaying.

| | |
|---|---|
| [`Biopsy`](Biopsy.md) | Spatially-localized sample of cells. |
| [`Dissociation`](Dissociation.md) | Whole-tumor dissociation into a cell suspension. |

## Data

Generate single-cell and bulk molecular data from the sampled cells.

| | |
|---|---|
| [`scDNA`](scDNA.md) | Single-cell DNA (copy number + SNV). |
| [`bulkDNA`](bulkDNA.md) | Bulk DNA-seq. |
| [`scRNA`](scRNA.md) | Single-cell RNA expression. |
| [`Visium`](Visium.md) | 10x Visium spatial transcriptomics. |

## Inference

Fit `iscc`'s parameters to real data.

| | |
|---|---|
| [`ABC`](ABC.md) | Approximate Bayesian computation over evolutionary rates. |
