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
