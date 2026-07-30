::: iscc.data.bulkDNA

## Technology presets

Bulk DNA sequencing differs by **capture breadth** — set `breadth`:

| Assay | `breadth` | Depth (`mu_depth`) | Loci | Dominant capture bias |
|---|---|---|---|---|
| Whole-genome (WGS) | `"wgs"` | ~30× | all | GC curve |
| Whole-exome (WES) | `"wes"` | ~120× | exome (~30% of loci) | per-target |
| Targeted panel | `"panel"` | ~1500× | a small gene panel | per-amplicon |
