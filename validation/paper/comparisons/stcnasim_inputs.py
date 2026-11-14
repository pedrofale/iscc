"""Seed inputs for stCNASim from a public Visium run, for the runtime comparison.

    python stcnasim_inputs.py whitelist <tissue_positions_list.csv> <out.txt>
    python stcnasim_inputs.py seed <tissue_positions_list.csv> <n_spots> <out cell_anno.tsv>
    python stcnasim_inputs.py features <gencode.gtf.gz> <out.tsv>
    python stcnasim_inputs.py phase <het SNP VCF> <seed> <out.tsv>
    python stcnasim_inputs.py clones <n_normal> <n_tumour1> <n_tumour2> <out clone_anno.tsv>

stCNASim's own seed (600 normal spots of the HCC-3N Visium section, GSA-Human HRA000437) is under
controlled access. The stand-in is GSE211956 / GSM6506111 (SRR21230125, a public HCC Visium
section, also OncoSpot's depth reference series): its 4,992-barcode positions file is the barcode
whitelist, and the seed is the spatially contiguous panel of in-tissue spots nearest the tissue's
centre, as stCNASim's seed is a contiguous panel. Phasing is random, so it times stCNASim faithfully
but is not a realistic phase. The phased SNPs are written as a VCF: stCNASim 0.6.0's TSV path fails
(``utils/gsnp.py`` ``check_dup_snp`` passes an undefined ``sep`` to ``load_snps``).
"""
import gzip
import sys

import numpy as np
import pandas as pd


def _positions(path):
    return pd.read_csv(path, header=None, names=["barcode", "in_tissue", "row", "col", "px_row", "px_col"])


def whitelist(pos_fn, out):
    _positions(pos_fn).barcode.str.replace("-1", "", regex=False).to_csv(out, index=False, header=False)


def seed(pos_fn, n, out):
    p = _positions(pos_fn)
    p = p[p.in_tissue == 1].copy()
    # Visium array rows are offset by one column on alternate rows; columns step by 2.
    y, x = p.row.to_numpy(float), p.col.to_numpy(float) / 2.0
    d = (y - y.mean()) ** 2 + (x - x.mean()) ** 2
    pick = p.iloc[np.argsort(d)[:int(n)]]
    # STARsolo writes CB without Cell Ranger's "-1" suffix
    pd.DataFrame({"cell": pick.barcode.str.replace("-1", "", regex=False), "cell_type": "normal"}).to_csv(
        out, sep="\t", index=False, header=False)


def features(gtf_fn, out):
    rows = []
    with gzip.open(gtf_fn, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            c = line.split("\t")
            if c[2] != "gene" or c[0] not in {f"chr{i}" for i in range(1, 23)}:
                continue
            a = dict(kv.strip().split(" ", 1) for kv in c[8].strip().rstrip(";").split(";") if kv.strip())
            if a.get("gene_type", "").strip('"') not in ("protein_coding", "lncRNA"):
                continue
            rows.append((c[0], int(c[3]), int(c[4]), a["gene_name"].strip('"'), c[6]))
    pd.DataFrame(rows).to_csv(out, sep="\t", index=False, header=False)


def phase(vcf_fn, seed_, out):
    """Heterozygous SNPs with a random phase, as a one-sample VCF with GT 0|1 or 1|0."""
    rng = np.random.default_rng(int(seed_))
    opener = gzip.open if vcf_fn.endswith(".gz") else open
    with opener(vcf_fn, "rt") as f, open(out, "w") as o:
        o.write("##fileformat=VCFv4.2\n")
        o.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
        o.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tseed\n")
        for line in f:
            if line.startswith("#"):
                continue
            c = line.split("\t")
            gt = "0|1" if rng.integers(0, 2) else "1|0"
            o.write(f"{c[0]}\t{c[1]}\t.\t{c[3]}\t{c[4]}\t.\tPASS\t.\tGT\t{gt}\n")


def clones(n0, n1, n2, out):
    pd.DataFrame([("normal", "normal", int(n0)), ("tumor1", "normal", int(n1)), ("tumor2", "normal", int(n2))]
                 ).to_csv(out, sep="\t", index=False, header=False)


if __name__ == "__main__":
    {"whitelist": whitelist, "seed": seed, "features": features, "phase": phase,
     "clones": clones}[sys.argv[1]](*sys.argv[2:])
