# API Reference

`iscc`'s Python API mirrors the CLI pipeline — **grow → sample → assay**, with **treatment** applied
during growth and optional **inference** of parameters from real data. See the
[Overview](overview.md) for how the stages fit together.

## Grow a tumor

::: iscc.tumor.models.count.GenotypeTumor
::: iscc.tumor.models.glandular.GlandularTumor

## Apply treatment

::: iscc.treatment.chemotherapy.Chemotherapy
::: iscc.treatment.targeted.TargetedTherapy
::: iscc.treatment.immunotherapy.Immunotherapy
::: iscc.treatment.surgery.Surgery

## Sample

::: iscc.sample.biopsy.biopsy.Biopsy
::: iscc.sample.dissociation.dissociation.Dissociation

## Generate data

::: iscc.data.dna.scDNA
::: iscc.data.dna.bulkDNA
::: iscc.data.rna.scRNA
::: iscc.data.visium.Visium

## Infer parameters from real data

::: iscc.inference.abc.ABC
