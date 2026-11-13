# Provenance of the human reference tables

Rebuilt by `validation/data/build_human_reference.py` (2026-11-13); nothing here is hand-edited.

| File | Source | Licence | Download checked by SHA-256 |
|---|---|---|---|
| `grch38_2020A_genes.tsv.gz` | GENCODE release 32, primary assembly GTF, filtered as 10x Genomics documents for its `GRCh38-2020-A` reference (allowed gene and transcript biotypes, no read-through transcripts, no PAR copies): 36,601 genes, identical in IDs and coordinates to Clonalscope's `hg38.genes.bed` | GENCODE data are freely available without restriction | `0c36c37abb0c88637b04a8e401c93f0199a65bbdd0702c9e13bdfbadc053fe1e` (`gencode.v32.primary_assembly.annotation.gtf.gz`) |
| `grch38_arms.tsv`, `grch38_chrom_sizes.tsv` | UCSC `cytoBand.txt.gz` (hg38) for centromeres and chromosome ends; the 39 arms and their Cancer Gene Census counts from `validation/data/realgenome_arms.csv` (Davoli's arm set, as in CINner) | UCSC annotation tables, derived from public NCBI data | `e514e48b0a12a516fd5231b44241a20a282f290bdd6131cadcddc6276ee26082` (`cytoBand.txt.gz`) |

Not shipped: the 1000 Genomes phase-3 common SNPs (allele frequency above 5%) on GRCh38, as
distributed by cellsnp-lite (`genome1K.phase3.SNP_AF5e2.chr1toX.hg38.vcf.gz`, 92 MB, SHA-256
`88e22ea3c4d416aed925e5b3efeb371ef023c954599a325dcb334dffcf6ad707`). 1000 Genomes data are released
without restriction. `iscc.data.human.fetch_snps()` downloads it, checks the checksum, and keeps the
4,035,522 SNPs inside the 35,061 modelled genes.

## Cell-type baselines and programmes (L3)

Rebuilt by `validation/data/build_atlas_baselines.py` (2026-11-13).

| File | Source | Licence | Download checked by SHA-256 |
|---|---|---|---|
| `hbca_type_baselines.tsv.gz`, `hbca_types.tsv` | Human Breast Cell Atlas (Kumar et al., Nature 2023, doi:10.1038/s41586-023-06252-9), CELLxGENE dataset "scRNA-seq data - all cells" (`842c6f5d-4a94-4eef-8510-8c792d1124bc`, version `296226b4-4916-44a3-97cc-60bdfb95f2e1`, 714,331 cells, 126 women). Mean per-cell composition over the 10x 3' v3 cells of each type: luminal epithelium (`LummHR-*`, `Lumsec-*`; 110,374 cells), fibroblasts (195,127), T and NK cells (71,288), each from 107 donors. 4,778 GRCh38-2020-A genes the atlas does not measure are 0 | CC BY 4.0 (CELLxGENE) | `0ccf028b0d3509851f8412ead5a8cf28409b4c93fda3ee0fafef519139f30d96` (the h5ad, 6.9 GB, not shipped) |
| `hallmark_programs.tsv` | MSigDB Hallmark gene sets v2025.1 (Liberzon et al., Cell Systems 2015), matched to GRCh38-2020-A symbols: proliferation (E2F targets + G2M checkpoint, 321 genes), EMT (200), hypoxia (200), drug resistance (xenobiotic metabolism, a proxy, 196) | CC BY 4.0 (MSigDB v2022.1 and later) | `f22066af72e215ccb7b89d88e492c07e1eef17534c2ca7b0f9902cfecbbdd8e9` (`h.all.v2025.1.Hs.symbols.gmt`) |
