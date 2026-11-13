"""Build the human reference tables the human genome layer ships with (``iscc.data.human``).

Two small derived tables are written into the package; the large SNP list stays outside it
(``iscc.data.human.fetch_snps`` downloads it and checks its checksum).

Sources (all GRCh38):
  1. **GENCODE release 32, primary assembly** — the annotation 10x Genomics built its
     ``GRCh38-2020-A`` Cell Ranger / Space Ranger reference from. We apply the same filter 10x
     documents for that build (allowed gene and transcript biotypes, no read-through transcripts,
     no PAR copies), which yields its 36,601 features. GENCODE data are freely available
     (https://www.gencodegenes.org/pages/data_access.html).
     https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_32/gencode.v32.primary_assembly.annotation.gtf.gz
  2. **UCSC cytoBand (hg38)** — chromosome lengths and centromere positions, giving each arm's
     interval. The arm set and the per-arm Cancer Gene Census counts are the 39 arms of
     ``validation/data/realgenome_arms.csv`` (Davoli's canonical set, as in CINner).
     https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/cytoBand.txt.gz

Outputs (under ``src/iscc/data/human/reference/``):
  * ``grch38_2020A_genes.tsv.gz`` — gene_id, gene_name, chrom, start, end, strand, gene_type, in the
    order of the 10x features file (GTF order). Coordinates are GTF ones: 1-based, inclusive.
  * ``grch38_arms.tsv`` — arm, chrom, start, end (0-based, half-open), n_onc, n_tsg.
  * ``grch38_chrom_sizes.tsv`` — chrom, length (chr1-22, X, Y).

Usage::

    python validation/data/build_human_reference.py --ref-dir <dir with the two downloads>

``--check-bed`` compares the gene table with another GRCh38-2020-A gene BED (for example
Clonalscope's ``data-raw/hg38.genes.bed``) and reports any difference in gene IDs or coordinates.
"""
import argparse
import csv
import gzip
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "src", "iscc", "data", "human", "reference")
GTF = "gencode.v32.primary_assembly.annotation.gtf.gz"
CYTOBAND = "cytoBand.hg38.txt.gz"

# The biotypes 10x keeps for GRCh38-2020-A (its "Build notes for reference packages").
BIOTYPES = {
    "protein_coding", "lncRNA",
    "IG_C_gene", "IG_D_gene", "IG_J_gene", "IG_LV_gene", "IG_V_gene",
    "IG_V_pseudogene", "IG_J_pseudogene", "IG_C_pseudogene",
    "TR_C_gene", "TR_D_gene", "TR_J_gene", "TR_V_gene", "TR_V_pseudogene", "TR_J_pseudogene",
}
_ATTR = re.compile(r'(\S+) "([^"]*)"')


def _attrs(field):
    out, tags = {}, set()
    for k, v in _ATTR.findall(field):
        if k == "tag":
            tags.add(v)
        else:
            out[k] = v
    return out, tags


def _strip_version(gene_id):
    # 10x strips the version suffix (ENSG00000243485.5 -> ENSG00000243485). PAR_Y copies keep their
    # suffix and are removed by the PAR filter anyway.
    m = re.match(r"^(ENSG\d+)\.\d+$", gene_id)
    return m.group(1) if m else gene_id


def build_genes(gtf_path):
    """Return the GRCh38-2020-A gene rows, in GTF order."""
    allow, genes = set(), []
    with gzip.open(gtf_path, "rt") as f:
        for ln in f:
            if ln.startswith("#"):
                continue
            chrom, _src, feat, start, end, _score, strand, _frame, attr = ln.rstrip("\n").split("\t")
            if feat not in ("gene", "transcript"):
                continue
            a, tags = _attrs(attr)
            if feat == "transcript":
                if (a.get("gene_type") in BIOTYPES and a.get("transcript_type") in BIOTYPES
                        and "readthrough_transcript" not in tags and "PAR" not in tags):
                    allow.add(a["gene_id"])
            else:
                genes.append((a["gene_id"], a.get("gene_name", ""), chrom, int(start), int(end),
                              strand, a.get("gene_type", "")))
    rows = [(_strip_version(g[0]),) + g[1:] for g in genes if g[0] in allow]
    return rows


def build_arms(cytoband_path, arms_csv):
    cent, chrom_end = {}, {}
    with gzip.open(cytoband_path, "rt") as f:
        for ln in f:
            chrom, s, e, _name, stain = ln.rstrip("\n").split("\t")
            s, e = int(s), int(e)
            chrom_end[chrom] = max(chrom_end.get(chrom, 0), e)
            if stain == "acen":
                lo, hi = cent.get(chrom, (s, e))
                cent[chrom] = (min(lo, s), max(hi, e))
    rows = []
    with open(arms_csv) as f:
        for r in csv.DictReader(f):
            arm = r["arm"]
            chrom = "chr" + arm[:-1]
            c_lo, c_hi = cent[chrom]
            start, end = (0, c_lo) if arm.endswith("p") else (c_hi, chrom_end[chrom])
            rows.append((arm, chrom, start, end, int(float(r["n_onc"])), int(float(r["n_tsg"]))))
    sizes = [(c, chrom_end[c]) for c in [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]]
    return rows, sizes


def check_bed(genes, bed_path):
    ours = {g[0]: (g[2], g[3], g[4]) for g in genes}
    theirs = {}
    with open(bed_path) as f:
        next(f)
        for ln in f:
            chrom, s, e, gid = ln.split()[:4]
            theirs[gid] = (chrom, int(s), int(e))
    only_ours, only_theirs = set(ours) - set(theirs), set(theirs) - set(ours)
    moved = [g for g in set(ours) & set(theirs) if ours[g] != theirs[g]]
    print(f"genes: ours {len(ours)}, bed {len(theirs)}; only ours {len(only_ours)}, "
          f"only bed {len(only_theirs)}, coordinates differ {len(moved)}")
    return not (only_ours or only_theirs or moved)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ref-dir", required=True, help="directory holding the GTF and cytoBand downloads")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--check-bed", default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    genes = build_genes(os.path.join(args.ref_dir, GTF))
    print(f"GRCh38-2020-A genes: {len(genes)}")
    with gzip.open(os.path.join(args.out, "grch38_2020A_genes.tsv.gz"), "wt", compresslevel=9) as f:
        f.write("gene_id\tgene_name\tchrom\tstart\tend\tstrand\tgene_type\n")
        for g in genes:
            f.write("\t".join(map(str, g)) + "\n")

    arms, sizes = build_arms(os.path.join(args.ref_dir, CYTOBAND),
                             os.path.join(HERE, "realgenome_arms.csv"))
    with open(os.path.join(args.out, "grch38_arms.tsv"), "w") as f:
        f.write("arm\tchrom\tstart\tend\tn_onc\tn_tsg\n")
        for r in arms:
            f.write("\t".join(map(str, r)) + "\n")
    with open(os.path.join(args.out, "grch38_chrom_sizes.tsv"), "w") as f:
        f.write("chrom\tlength\n")
        for c, n in sizes:
            f.write(f"{c}\t{n}\n")
    print(f"arms: {len(arms)}; chromosomes: {len(sizes)}")
    if args.check_bed:
        ok = check_bed(genes, args.check_bed)
        print("matches the BED" if ok else "DIFFERS from the BED")


if __name__ == "__main__":
    main()
