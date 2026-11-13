"""The human reference of the genome layer: genes, chromosome arms and common germline SNPs.

The gene table is 10x Genomics' ``GRCh38-2020-A`` feature set (36,601 genes from GENCODE release 32,
filtered as 10x documents), and the arms are the 39 autosomal arms of the real-genome mode (the five
acrocentric short arms excluded), with their Cancer Gene Census counts. Both ship with the package;
``validation/data/build_human_reference.py`` rebuilds them from the sources.

The SNP list is too large to ship. :func:`fetch_snps` downloads the 1000 Genomes phase-3 list of
common SNPs (allele frequency above 5%, on GRCh38) that cellsnp-lite distributes, checks it, and
keeps only the SNPs inside the modelled genes; :func:`load_reference` finds the result through the
``ISCC_HUMAN_REF`` environment variable or an explicit path.
"""
import gzip
import hashlib
import os
import urllib.request
from dataclasses import dataclass, field
from importlib import resources

import numpy as np
import pandas as pd

SNP_URL = ("https://sourceforge.net/projects/cellsnp/files/SNPlist/"
           "genome1K.phase3.SNP_AF5e2.chr1toX.hg38.vcf.gz/download")
SNP_FILE = "genome1K.phase3.SNP_AF5e2.chr1toX.hg38.vcf.gz"
SNP_SHA256 = "88e22ea3c4d416aed925e5b3efeb371ef023c954599a325dcb334dffcf6ad707"
SNP_CACHE = "snps_in_genes.npz"
ENV_VAR = "ISCC_HUMAN_REF"
_BASES = np.array(list("ACGT"))


def _table(name, **kw):
    with resources.files("iscc.data.human").joinpath("reference", name).open("rb") as f:
        if name.endswith(".gz"):
            f = gzip.open(f)
        return pd.read_csv(f, sep="\t", **kw)


@dataclass
class HumanReference:
    """The human genome the layer lifts a growth genome onto.

    Attributes
    ----------
    genes : pandas.DataFrame
        The ``GRCh38-2020-A`` genes in 10x feature order: ``gene_id`` (Ensembl, unversioned),
        ``gene_name``, ``chrom``, ``start``, ``end`` (1-based, inclusive, as in the GTF), ``strand``,
        ``gene_type``.
    arms : pandas.DataFrame
        The 39 modelled arms in genome order: ``arm``, ``chrom``, ``start``, ``end`` (0-based,
        half-open), ``n_onc``, ``n_tsg`` (Cancer Gene Census oncogenes and suppressors on the arm).
    chrom_sizes : pandas.DataFrame
        ``chrom``, ``length`` for chr1-22, X and Y.
    snps : pandas.DataFrame or None
        Common SNPs inside the modelled genes: ``chrom``, ``pos`` (1-based), ``ref``, ``alt``, ``af``
        and ``gene`` (row index into ``genes``). None until :func:`fetch_snps` has prepared them;
        expression needs no SNPs, allele counts do.
    """
    genes: pd.DataFrame
    arms: pd.DataFrame
    chrom_sizes: pd.DataFrame
    snps: pd.DataFrame = field(default=None)

    def gene_arm(self):
        """Index into ``arms`` of the arm holding each gene's midpoint, or -1 for a gene on no
        modelled arm (chrX, chrY, chrM, an acrocentric short arm, a centromere, a scaffold)."""
        mid = (self.genes["start"].to_numpy() + self.genes["end"].to_numpy()) / 2.0
        chrom = self.genes["chrom"].to_numpy()
        out = np.full(len(self.genes), -1, dtype=int)
        for a, r in enumerate(self.arms.itertuples(index=False)):
            out[(chrom == r.chrom) & (mid >= r.start) & (mid < r.end)] = a
        return out


def load_reference(snps=None):
    """Load the packaged human reference, with its SNPs if they have been prepared.

    Parameters
    ----------
    snps : str, pandas.DataFrame or False, optional
        The SNPs: a path to the ``snps_in_genes.npz`` that :func:`fetch_snps` writes, or a table with
        the columns of :attr:`HumanReference.snps`. Default None looks for ``snps_in_genes.npz`` in the
        directory named by the ``ISCC_HUMAN_REF`` environment variable and leaves ``snps`` empty if
        there is none; False never loads them.

    Returns
    -------
    HumanReference
    """
    genes = _table("grch38_2020A_genes.tsv.gz", dtype={"chrom": str})
    arms = _table("grch38_arms.tsv")
    sizes = _table("grch38_chrom_sizes.tsv")
    if snps is None and os.environ.get(ENV_VAR):
        cand = os.path.join(os.environ[ENV_VAR], SNP_CACHE)
        snps = cand if os.path.exists(cand) else None
    if isinstance(snps, (str, os.PathLike)):
        snps = read_snps(snps)
    elif snps is False:
        snps = None
    return HumanReference(genes=genes, arms=arms, chrom_sizes=sizes, snps=snps)


def load_atlas_baselines():
    """The packaged per-cell-type baselines from the Human Breast Cell Atlas.

    Returns
    -------
    (pandas.DataFrame, pandas.DataFrame)
        The baselines, indexed by ``gene_id`` with one column per iscc type (``epithelial``,
        ``stromal``, ``immune``), each summing to 1 over the GRCh38-2020-A genes; and per type the atlas
        labels, cells, donors, mean counts and ``rna_content`` (mean counts relative to the epithelium).
    """
    return _table("hbca_type_baselines.tsv.gz").set_index("gene_id"), _table("hbca_types.tsv")


def load_hallmark_programs():
    """The packaged Hallmark gene sets that seed iscc's named programmes: ``programme``,
    ``hallmark_set``, ``gene_id``, ``gene_name`` (MSigDB v2025.1, CC BY 4.0)."""
    return _table("hallmark_programs.tsv")


def read_snps(path):
    """Read a ``snps_in_genes.npz`` written by :func:`prepare_snps` into a table."""
    z = np.load(path)
    return pd.DataFrame({
        "chrom": np.char.add("chr", z["chrom"].astype(str)),
        "pos": z["pos"].astype(np.int64),
        "ref": _BASES[z["ref"]],
        "alt": _BASES[z["alt"]],
        "af": z["af"].astype(float),
        "gene": z["gene"].astype(np.int64),
    })


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_snps(dest=None, url=SNP_URL, sha256=SNP_SHA256):
    """Download the 1000 Genomes common-SNP list, check it, and prepare the in-gene subset.

    Writes ``genome1K.phase3.SNP_AF5e2.chr1toX.hg38.vcf.gz`` (92 MB) and ``snps_in_genes.npz`` into
    ``dest``. Point ``ISCC_HUMAN_REF`` at ``dest`` afterwards and :func:`load_reference` finds them.

    Parameters
    ----------
    dest : str, optional
        Target directory; default the ``ISCC_HUMAN_REF`` directory.
    url : str, optional
        Source of the list (cellsnp-lite's distribution of the 1000 Genomes phase-3 SNPs with allele
        frequency above 5%, lifted to GRCh38).
    sha256 : str, optional
        Expected checksum of the download; a mismatch raises.

    Returns
    -------
    str
        Path of the prepared ``snps_in_genes.npz``.
    """
    dest = dest or os.environ.get(ENV_VAR)
    if not dest:
        raise ValueError(f"give dest or set {ENV_VAR}")
    os.makedirs(dest, exist_ok=True)
    vcf = os.path.join(dest, SNP_FILE)
    if not os.path.exists(vcf):
        urllib.request.urlretrieve(url, vcf)
    got = _sha256(vcf)
    if sha256 and got != sha256:
        raise ValueError(f"{vcf}: checksum {got} does not match the expected {sha256}")
    out = os.path.join(dest, SNP_CACHE)
    prepare_snps(vcf, out)
    return out


def prepare_snps(vcf_path, out_path, reference=None):
    """Keep the SNPs of a VCF that fall inside a modelled gene, and write them compactly.

    A SNP inside several overlapping genes is assigned to the first of them in gene-table order.
    Only biallelic single-base substitutions on chr1-22 are kept.

    Parameters
    ----------
    vcf_path : str
        A (gzipped) VCF with an ``AF`` entry in INFO, chromosomes named with or without ``chr``.
    out_path : str
        The ``.npz`` to write.
    reference : HumanReference, optional
        Default the packaged one.
    """
    ref = reference or load_reference(snps=False)
    arm_of = ref.gene_arm()
    code = {b: i for i, b in enumerate("ACGT")}
    chroms, poss, refs, alts, afs = [], [], [], [], []
    opener = gzip.open if str(vcf_path).endswith(".gz") else open
    with opener(vcf_path, "rt") as f:
        for ln in f:
            if ln.startswith("#"):
                continue
            c, pos, _id, r, a, _q, _flt, info = ln.split("\t", 8)[:8]
            c = c[3:] if c.startswith("chr") else c
            if not c.isdigit() or len(r) != 1 or len(a) != 1 or r not in code or a not in code:
                continue
            af = next((float(x[3:]) for x in info.split(";") if x.startswith("AF=")), np.nan)
            chroms.append(int(c)); poss.append(int(pos)); refs.append(code[r]); alts.append(code[a])
            afs.append(af)
    chroms, poss = np.asarray(chroms, np.int8), np.asarray(poss, np.int64)
    gene = np.full(len(poss), -1, dtype=np.int64)
    g = ref.genes
    for c in range(1, 23):
        sel = np.flatnonzero(chroms == c)
        if not sel.size:
            continue
        order = sel[np.argsort(poss[sel], kind="stable")]
        p = poss[order]
        rows = np.flatnonzero((g["chrom"].to_numpy() == f"chr{c}") & (arm_of >= 0))
        for gi in rows:                      # gene-table order: the first gene wins an overlap
            lo = np.searchsorted(p, g["start"].iat[gi], side="left")
            hi = np.searchsorted(p, g["end"].iat[gi], side="right")
            if hi > lo:
                tgt = order[lo:hi]
                free = gene[tgt] < 0
                gene[tgt[free]] = gi
    keep = gene >= 0
    np.savez_compressed(out_path, chrom=chroms[keep], pos=poss[keep].astype(np.int32),
                        ref=np.asarray(refs, np.int8)[keep], alt=np.asarray(alts, np.int8)[keep],
                        af=np.asarray(afs, np.float32)[keep], gene=gene[keep].astype(np.int32))
    return int(keep.sum())
