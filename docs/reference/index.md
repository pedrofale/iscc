# API Reference

`iscc`'s Python API mirrors the CLI pipeline — **grow → sample → assay**, with **treatment**
applied during growth and optional **inference** of parameters from real data. Each stage has its
own page:

- [**Tumor**](tumor.md) — grow a spatially-structured tumor (`GenotypeTumor`, `GlandularTumor`)
- [**Treatment**](treatment.md) — chemotherapy, targeted, immunotherapy, surgery
- [**Sample**](sample.md) — biopsy and dissociation
- [**Data**](data.md) — generate single-cell and bulk DNA / RNA / spatial assays
- [**Inference**](inference.md) — fit assay and evolutionary parameters from real data

See the [Overview](../overview.md) for how the stages connect, or the CLI entry points
`isccsim` / `isccsample` / `isccdata`.
