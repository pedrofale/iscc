# API Reference

The Python API follows the stages of the CLI pipeline, **grow → sample → assay**, with
**treatment** applied during growth and the **estimation** of parameters from real data. Import the
package as `import iscc`, and see the [Overview](../overview.md) for how the stages connect.

## Tumor

Grow a selection-driven tumor on a grid of **demes** — small tissue patches, each holding up to a
carrying capacity of cells. Each deme is held as genotype counts, and the process (birth, death,
mutation and dispersal under a CINner fitness model) runs either event by event (exact) or by
tau-leaping.

| | |
|---|---|
| [`iscc.tumor.GenotypeTumor`](GenotypeTumor.md) | The tumour engine: per-deme genotype **counts**, exact or tau-leaping. |

## Cohort

Grow several patients over one shared gene layout, so that they share their driver genes, baseline
expression and programs and differ in their evolution.

| | |
|---|---|
| [`iscc.cohort.Cohort`](Cohort.md) | A cohort of patient tumours grown over one shared gene layout. |
| [`iscc.cohort.grow_tumor`](grow_tumor.md) | Grow one patient tumour for a fixed number of steps. |

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
platforms (DLP+ / Tapestri for scDNA, WGS / WES / panel for bulk DNA, and 10x / Smart-seq3
for scRNA). The assays' technical defaults live in these hyper-parameter containers (the targets of the
[Calibration](#calibration) estimators, overridable per assay):

| | |
|---|---|
| [`iscc.data.DNABatchHyperParams`](DNABatchHyperParams.md) | DNA technical parameters. |
| [`iscc.data.RNABatchHyperParams`](RNABatchHyperParams.md) | scRNA technical parameters. |
| [`iscc.data.VisiumBatchHyperParams`](VisiumBatchHyperParams.md) | Visium technical parameters. |
| [`iscc.data.visium_background.bleed`](visium_bleed.md) | Visium spot swapping: counts bleeding between spots, as SpotClean models it. |

## Human genome layer

Lift the coarse growth genome onto the human genome when cells are measured: real GRCh38 genes, a
germline of real heterozygous SNPs, haplotype allele counts, and the input files of copy-number
tools (CopyKAT, CalicoST, Clonalscope, Cancer-Finder). Growth never reads it.

| | |
|---|---|
| [`iscc.data.human.arm_genome`](arm_genome.md) | A growth genome with one segment per chromosome arm, at any resolution. |
| [`iscc.data.human.HumanGenomeLayer`](HumanGenomeLayer.md) | The gene map, the germline, and the allele counts of an assay. |
| [`iscc.data.human.HumanExpression`](HumanExpression.md) | Human-gene expression from atlas baselines per cell type and Hallmark programmes, for the assays' `gene_lift`. |
| [`iscc.data.human.GeneLift`](GeneLift.md) | The map from growth genes to human genes that the assays take as `gene_lift`. |
| [`iscc.data.human.AlleleCounts`](AlleleCounts.md) | Per-haplotype UMI counts at heterozygous SNPs. |
| [`iscc.data.human.write_visium`](write_visium.md) | Write a Visium assay as Space Ranger output, CalicoST's `snp_dir` and the truth. |
| [`iscc.data.human.write_scrna`](write_scrna.md) | Write an scRNA assay as Cell Ranger output and the truth. |
| [`iscc.data.human.check_tool_ready`](check_tool_ready.md) | Test a dataset against each tool's input thresholds. |
| [`iscc.data.human.HumanReference`](HumanReference.md) | The genes, arms and SNPs of the human reference. |
| [`iscc.data.human.load_reference`](load_reference.md) | Load the packaged reference (and the SNPs, if prepared). |
| [`iscc.data.human.fetch_snps`](fetch_snps.md) | Download and prepare the 1000 Genomes common SNPs. |
| [`iscc.data.human.load_atlas_baselines`](load_atlas_baselines.md) | The packaged per-cell-type baselines from the Human Breast Cell Atlas. |
| [`iscc.data.human.load_hallmark_programs`](load_hallmark_programs.md) | The packaged Hallmark gene sets that seed the named programmes. |

## Reads

Generate sequencing reads (FASTQ, optionally aligned BAM) from the assayed counts. DNA reads
use DWGSIM/ART, and RNA reads use a synthetic 10x transcriptome or the scReadSim template. These
call external read simulators, so the corresponding binaries must be installed. The paper describes
only the synthetic references, and the real-reference and scReadSim paths are not part of it.

| | |
|---|---|
| [`iscc.data.reads.emit_dna_reads`](emit_dna_reads.md) | Bulk / single-cell DNA reads (→ FASTQ, optional BAM). |
| [`iscc.data.reads.emit_scrna_reads`](emit_scrna_reads.md) | Mutation-aware single-cell RNA reads. |
| [`iscc.data.reads.emit_visium_reads`](emit_visium_reads.md) | Spatial (Visium) reads. |

## Integrations

Hand the ground truth over in the forms that other tools read.

| | |
|---|---|
| [`iscc.integrations.clones_from_clades`](clones_from_clades.md) | Clone labels of sampled cells, as clades of the true genealogy with enough sampled cells. |
| [`iscc.integrations.clone_summary`](clone_summary.md) | One row per clone: its size, representative genotype and mutations. |
| [`iscc.integrations.to_anndata`](to_anndata.md) | A tumour or a sample as one AnnData, with coordinates and ground truth. |

## Inference

Estimate the **evolutionary parameters of the tumor** (division, mutation, dispersal and selection
rates) from real data by approximate Bayesian computation over the growth simulator. The paper does
not use this module.

| | |
|---|---|
| [`iscc.inference.ABC`](ABC.md) | Approximate Bayesian computation over evolutionary rates. |
| [`iscc.inference.Prior`](Prior.md) | Product prior over the named parameters to infer. |
| [`iscc.inference.Posterior`](Posterior.md) | Result of an ABC run — samples and point estimates. |

## Calibration

Estimate the **technical parameters of each assay** (library size, dispersion, capture field, …)
from a real reference dataset, so that simulated data match a target platform. The scRNA dropout
curve is fitted only on request (`fit_dropout=True`).

| | |
|---|---|
| [`iscc.data.estimate_dna`](estimate_dna.md) | Calibrate the DNA assay from coverage / allele statistics. |
| [`iscc.data.estimate_rna`](estimate_rna.md) | Calibrate the scRNA assay from a real count matrix. |
| [`iscc.data.estimate_visium`](estimate_visium.md) | Calibrate the Visium assay from a real section. |
