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
| [`iscc.tumor.Tumor`](Tumor.md) | Shared base of the two engines — the common construction parameters. |

## Treatment

Therapies applied to the tumor during growth.

| | |
|---|---|
| [`iscc.treatment.Chemotherapy`](Chemotherapy.md) | Cytotoxic therapy acting on dividing cells. |
| [`iscc.treatment.TargetedTherapy`](TargetedTherapy.md) | Therapy against a driver-defined subclone. |
| [`iscc.treatment.Immunotherapy`](Immunotherapy.md) | Immune-cell–mediated killing in the microenvironment. |
| [`iscc.treatment.Surgery`](Surgery.md) | Resect a compartment (e.g. the primary). |
| [`iscc.treatment.Treatment`](Treatment.md) | Shared base of the therapies — dosing schedule + per-cell effect. |

## Sample

Draw cells from the grown tumor before assaying.

| | |
|---|---|
| [`iscc.sample.Resection`](Resection.md) | Cut a resected specimen into samples — bisect (in-plane cut) / dissociate / slice (depth cut). |
| [`iscc.sample.Biopsy`](Biopsy.md) | Spatially-localized sample of cells. |
| [`iscc.sample.Dissociation`](Dissociation.md) | Whole-tumor dissociation into a cell suspension. |
| [`iscc.sample.spatialize`](spatialize.md) | Place a deme-based tumor's cells at sub-deme positions and thin to a section (for spatial assays). |
| [`iscc.sample.tissue_image`](tissue_image.md) | Rasterize placed cell positions into an H&E-like morphology image (a spatial-slide background). |

## Data

Generate single-cell and bulk molecular data from the sampled cells.

| | |
|---|---|
| [`iscc.data.bulkDNA`](bulkDNA.md) | Bulk DNA-seq. |
| [`iscc.data.scDNA`](scDNA.md) | Single-cell DNA (copy number + SNV). |
| [`iscc.data.scRNA`](scRNA.md) | Single-cell RNA expression. |
| [`iscc.data.Visium`](Visium.md) | 10x Visium spatial transcriptomics. |
| [`iscc.data.DNA`](DNA.md) | Shared base of the DNA assays — breadth + coverage core. |

Each assay page also lists **technology presets** — the parameter settings that approximate named
platforms (MALBAC / DLP / Tapestri for scDNA, WGS / WES / panel for bulk DNA, and 10x / Smart-seq3
for scRNA). The assays' technical defaults live in these hyper-parameter containers (the targets of the
[Calibration](#calibration) estimators, overridable per assay):

| | |
|---|---|
| [`iscc.data.DNABatchHyperParams`](DNABatchHyperParams.md) | DNA technical parameters. |
| [`iscc.data.RNABatchHyperParams`](RNABatchHyperParams.md) | scRNA technical parameters. |
| [`iscc.data.VisiumBatchHyperParams`](VisiumBatchHyperParams.md) | Visium technical parameters. |

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
| [`iscc.data.estimate_dna`](estimate_dna.md) | Calibrate the DNA assay from coverage / allele statistics. |
| [`iscc.data.estimate_rna`](estimate_rna.md) | Calibrate the scRNA assay from a real count matrix. |
| [`iscc.data.estimate_visium`](estimate_visium.md) | Calibrate the Visium assay from a real section. |

## CN evolution analysis

Analyse a **grown** tumor's copy-number evolution against its own ground truth. Seven co-equal
questions, each with its own metric set — none is an input to another. `iscc.cnevo` reads
`tumor.traces` (exact per-generation clone counts) and the genotype registry (which retains every
genotype ever created), so ancestral copy number and the CNA event log are *recovered*, not inferred.

Grow with `trace_occupancy=True` to make the r/K split and the colonisation curve measurable.

| | |
|---|---|
| [`iscc.cnevo.sweep_metrics`](sweep_metrics.md) | Clonal dynamics: coalescent depth, sweeps, selection. |
| [`iscc.cnevo.diversity_trajectory`](diversity_trajectory.md) | Diversity per generation — the path through Noble's `(n, D, J1)` mode space. |
| [`iscc.cnevo.growth_phase`](growth_phase.md) | r- vs K-phase demography, from the crowding law. |
| [`iscc.cnevo.cn_landscape`](cn_landscape.md) | FGA, ploidy, LOH, WGD and recurrence over time. |
| [`iscc.cnevo.data_quality`](data_quality.md) | Is a sampled clone set a usable CN benchmark? |
| [`iscc.cnevo.reconstruction_potential`](reconstruction_potential.md) | How recoverable is the true clone tree from CN alone? |
| [`iscc.cnevo.spatial_structure`](spatial_structure.md) | Multi-focality and invasion (glandular runs only). |
| [`iscc.cnevo.select_clones`](select_clones.md) | Sample clones — one representative cell each. |
| [`iscc.cnevo.cna_event_table`](cna_event_table.md) | The derived, lossless CNA event log. |
