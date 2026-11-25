::: iscc.data.bulkDNA

## Technology presets

Bulk DNA sequencing differs by **capture breadth** — set `breadth`:

| Assay | `breadth` | Depth (`mu_depth`) | Loci | Dominant capture bias |
|---|---|---|---|---|
| Whole-genome (WGS) | `"wgs"` | ~30× | all | GC curve |
| Whole-exome (WES) | `"wes"` | ~120× | exome (~30% of loci) | per-target |
| Targeted panel | `"panel"` | ~1500× | a small gene panel | per-amplicon |

At every breadth, the coverage spread is the one we fitted with [`estimate_dna`](estimate_dna.md) on
the GIAB HG002 benchmark genome (`kappa` 230,823 on 20,000 sites, coefficient of variation 0.30 at two
copies and 298×, from the allele depths over all the datasets of the benchmark).
On a 6,000-locus genome at 30×, with the GC curve on top, the coefficient of variation at two copies is
about 0.38.
