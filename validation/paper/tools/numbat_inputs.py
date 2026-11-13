"""Write Numbat's allele table and gene positions for a human-gene scRNA dataset (write_scrna).

    python numbat_inputs.py <dataset_dir>

Writes ``<dataset>/numbat/df_allele.tsv.gz`` (cell, snp_id, CHROM, POS, cM, REF, ALT, AD, DP, GT, gene)
from ``snp_dir`` and ``gtf.tsv`` (gene, gene_start, gene_end, CHROM, gene_length: the columns of
Numbat's own ``gtf_hg38``) from the gene BED. iscc knows each
heterozygous SNP's phase exactly, so ``GT`` is "1|0" when ALT sits on haplotype 1 (homolog p, the
``A`` matrix) and "0|1" otherwise, and ``AD`` counts the ALT haplotype's UMIs. Genetic position is
taken as 1 cM per Mb, which only sets Numbat's phase-switch prior (the phase has no switch errors).
"""
import gzip
import os
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp


def main(ds):
    snp_dir = os.path.join(ds, "snp_dir")
    A = sp.load_npz(os.path.join(snp_dir, "cell_snp_Aallele.npz")).tocoo()
    B = sp.load_npz(os.path.join(snp_dir, "cell_snp_Ballele.npz")).tocsr()
    snps = pd.read_csv(os.path.join(snp_dir, "snps.tsv"), sep="\t")
    bcs = pd.read_csv(os.path.join(snp_dir, "barcodes.txt"), header=None)[0].to_numpy()
    with gzip.open(os.path.join(ds, "outs", "filtered_feature_bc_matrix", "features.tsv.gz"), "rt") as f:
        feat = pd.read_csv(f, sep="\t", header=None, names=["gene_id", "gene", "type"])
    sym = dict(zip(feat["gene_id"], feat["gene"]))
    D = (sp.load_npz(os.path.join(snp_dir, "cell_snp_Aallele.npz")) + B).tocoo()   # depth per entry
    A = A.tocsr()
    rows, cols, dp = D.row, D.col, D.data
    a = np.asarray(A[rows, cols]).ravel()
    alt_on_p = (snps["alt_hap"].to_numpy() == "p")[cols]
    ad = np.where(alt_on_p, a, dp - a)
    out = os.path.join(ds, "numbat")
    os.makedirs(out, exist_ok=True)
    s = snps.iloc[cols]
    df = pd.DataFrame({"cell": bcs[rows], "snp_id": s["snp_id"].to_numpy(),
                       "CHROM": s["chrom"].str.replace("chr", "", regex=False).astype(int).to_numpy(),
                       "POS": s["pos"].to_numpy(), "cM": s["pos"].to_numpy() / 1e6,
                       "REF": s["ref"].to_numpy(), "ALT": s["alt"].to_numpy(), "AD": ad.astype(int),
                       "DP": dp.astype(int), "GT": np.where(alt_on_p, "1|0", "0|1"),
                       "gene": [sym.get(g, g) for g in s["gene_id"]]})
    df.sort_values(["CHROM", "POS", "cell"]).to_csv(os.path.join(out, "df_allele.tsv.gz"), sep="\t", index=False)
    bed = pd.read_csv(os.path.join(ds, "clonalscope", "genes.bed"), sep="\t")
    gtf = pd.DataFrame({"gene": [sym.get(g, g) for g in bed["geneID"]], "gene_start": bed["start"],
                        "gene_end": bed["end"],
                        "CHROM": bed["chr"].str.replace("chr", "", regex=False).astype(int),
                        "gene_length": bed["end"] - bed["start"]}).drop_duplicates("gene")
    gtf.to_csv(os.path.join(out, "gtf.tsv"), sep="\t", index=False)
    print(f"numbat inputs: {len(df)} allele entries over {df['snp_id'].nunique()} SNPs and "
          f"{df['cell'].nunique()} cells; {len(gtf)} genes", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
