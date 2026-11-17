"""Mutation-aware scRNA read/count emission.

Real scRNA reads carry a cell's *expressed* somatic mutations, so iscc can show that **SNV
calling from scRNA is intrinsically hard**. This adapter reuses the shared `variants.inject`
seam (total-preserving) and the expression count matrix, under two hard constraints:

  1. **UMI totals are conserved.** scRNA reads are DRIVEN by the expression count matrix — we
     do NOT invent coverage. For each cell-gene the UMI total is fixed by the matrix; this layer
     only adds SEQUENCE CONTENT, splitting those UMIs into alt-carrying vs ref-carrying. So
     `alt + ref == expression count`, always, and an unexpressed gene (0 UMIs) is uncallable.
  2. **One abstraction parameter.** A single scalar `obs_fidelity ∈ [0, 1]` folds monoallelic
     expression, transcriptional bursting, RNA editing and RT error into ONE knob mapping the
     true RNA-VAF to the observed one: `v_obs = v_true · obs_fidelity` (each mutant transcript is
     faithfully reported with prob `obs_fidelity`, else reverts to reference — the dominant
     under-detection failure mode). It is NOT a mechanistic ASE model, by design.

`v_true` is the engine's `cell_rna_vaf` — the allele-expression-weighted RNA-VAF (`(v·e)/(v·e +
1-v)`, copy number cancels); see `make_cell_data`. The benchmark is the observed RNA-VAF (what a
caller sees) vs the true DNA-VAF (`cell_snv`): expression gating + `obs_fidelity` distortion make
the scRNA VAF drop out / diverge from the genomic truth.

The count-level observed alt/ref UMI matrix (`observed_allele_counts`) already demonstrates the
effect and is what most scRNA variant callers consume. Actual reads (scReadSim, on a REAL
template) are an optional realism layer with the SAME totals — stubbed behind `ScReadSimAdapter`
and skipped gracefully when the binary is absent (CI stays light).
"""
from __future__ import annotations

import gzip
import os
from collections import namedtuple

import numpy as np
import pandas as pd

from .base import find_binary
from .dna import _BASES, _ALT_OF
from ..dna import genome_bases
from . import variants


#: Count-level mutation-aware scRNA output. All matrices are cells x genes, aligned to `total`;
#: ``alt + ref == total`` exactly (UMI totals conserved). ``obs_vaf`` = alt/total (0 where total=0).
ScrnaAlleleCounts = namedtuple("ScrnaAlleleCounts", ["alt", "ref", "total", "obs_vaf"])


def distort_vaf(rna_vaf, obs_fidelity):
    """Map true RNA-VAF -> observed RNA-VAF with the single fidelity knob.

    ``v_obs = v_true · obs_fidelity`` (clipped to [0,1]): each mutant transcript is faithfully
    reported with prob `obs_fidelity`, else reverts to reference. One parameter abstracting
    monoallelic expression / bursting / editing / RT error — `obs_fidelity=1` is faithful,
    smaller values under-detect the mutation (the dominant scRNA failure mode).
    """
    v = np.asarray(rna_vaf, dtype=float)
    return np.clip(v * float(obs_fidelity), 0.0, 1.0)


#: Where a protocol's reads fall along a transcript, for the variant-site coverage model
#: (`site_coverage`): a 3' protocol reads only the last `THREE_PRIME_WINDOW` bases before the
#: poly(A) tail (10x 3' and Visium); a full-length protocol (Smart-seq3) reads the whole
#: transcript. Within reach, a molecule's reads cover a given site with probability
#: `SITE_COVERAGE` (about read length over the window). Both values are those the human genome
#: layer uses for germline SNPs (`HumanGenomeLayer(snp_window=1000, snp_coverage=0.1)`), so
#: germline and somatic sites are read by one model.
PROTOCOL_COVERAGE = {"10x": "3prime", "visium": "3prime", "smartseq3": "full"}
THREE_PRIME_WINDOW = 1000           # bases from the 3' end that a 3' protocol reads
SITE_COVERAGE = 0.1                 # probability that a molecule within reach is read at the site
TRANSCRIPT_MEDIAN = 2500            # median transcript length (bases), about that of human mRNAs
TRANSCRIPT_SDLOG = 0.7              # log-normal spread of transcript lengths


def site_geometry(genes, seed=20240601):
    """Each gene's transcript length and the distance of its variant site from the 3' end.

    A genome property like `iscc.data.dna.genome_bases`: drawn from a gene-name-seeded RNG at a
    FIXED genome seed, so it is the same in every run, batch and modality. The length is
    log-normal around `TRANSCRIPT_MEDIAN`; the site sits at a uniformly random position along the
    transcript. Returns ``(length, dist_from_3prime)`` as float arrays aligned to `genes`.
    """
    from ..dna import _gene_base_seed
    genes = list(genes)
    length = np.empty(len(genes))
    dist = np.empty(len(genes))
    for i, g in enumerate(genes):
        # offset the stream so the geometry is independent of the gene's ref/alt draw
        rng = np.random.default_rng(_gene_base_seed(g, seed) + 7919)
        length[i] = TRANSCRIPT_MEDIAN * np.exp(TRANSCRIPT_SDLOG * rng.standard_normal())
        dist[i] = length[i] * rng.random()
    return length, dist


def site_coverage(genes, protocol, window=THREE_PRIME_WINDOW, coverage=SITE_COVERAGE, seed=20240601):
    """Probability that a counted molecule of each gene is read at the gene's variant site.

    The simplest model of where reads fall: a 3' protocol (``"10x"``, ``"visium"``) can read the
    site only if it lies within `window` bases of the 3' end; a full-length protocol
    (``"smartseq3"``) can read it anywhere; within reach, a molecule is read at the site with
    probability `coverage`. So the same SNV can be visible to Smart-seq3 and invisible to 10x,
    which, with depth, is why full-length protocols call more SNVs.
    Returns a float array aligned to `genes` (`coverage` or 0 under this model).
    """
    key = str(protocol).lower().replace("-", "").replace("_", "")
    kind = PROTOCOL_COVERAGE.get(key)
    if kind is None:
        raise ValueError(f"no coverage model for protocol {protocol!r}; known: {sorted(PROTOCOL_COVERAGE)}")
    genes = list(genes)
    c = float(coverage)
    if kind == "full":
        return np.full(len(genes), c)
    _, dist = site_geometry(genes, seed=seed)
    return np.where(dist <= float(window), c, 0.0)


def observed_allele_counts(expr_counts, rna_vaf, obs_fidelity=1.0, error_rate=0.001, rng=None,
                           site_cover=None):
    """Split each cell-gene's expression UMIs into alt/ref counts, conserving the total.

      expr_counts   cells x genes UMI matrix (the scRNA count matrix; fixes the totals).
      rna_vaf       cells x genes true RNA-VAF (`cell_rna_vaf`); reindexed to `expr_counts`.
      obs_fidelity  the single distortion knob (see `distort_vaf`).
      error_rate    per-base RNA error/editing floor handed to `variants.inject`.
      site_cover    optional per-gene probability that a molecule is read at the variant site
                    (`site_coverage`), aligned to the columns. When given, each cell-gene's UMIs
                    are first thinned to those covering the site, ``Binomial(UMIs, site_cover)``,
                    and only those are split. ``None`` (default) reads every molecule at the site.

    Returns a `ScrnaAlleleCounts` of cells x genes DataFrames. `alt + ref == total`, where `total`
    is the expression count (``site_cover=None``) or the molecules covering the site, never more;
    `obs_vaf = alt / total` (0 where no molecule covers the site, i.e. uncallable).
    """
    if rng is None:
        rng = np.random.default_rng()
    expr = expr_counts if isinstance(expr_counts, pd.DataFrame) else pd.DataFrame(expr_counts)
    vaf = pd.DataFrame(rna_vaf).reindex(index=expr.index, columns=expr.columns).fillna(0.0)

    total = expr.values.astype(np.int64)
    if site_cover is not None:
        p = np.clip(np.asarray(site_cover, dtype=float).ravel(), 0.0, 1.0)
        if p.size != total.shape[1]:
            raise ValueError("site_cover must have one entry per gene column")
        if np.all((p == 0.0) | (p == 1.0)):          # all-or-nothing coverage needs no draw
            total = total * p.astype(np.int64)[None, :]
        else:
            total = rng.binomial(total, np.broadcast_to(p[None, :], total.shape))
    v_obs = distort_vaf(vaf.values, obs_fidelity)
    split = variants.inject(total, v_obs, error_rate=error_rate, rng=rng)
    alt, ref = split.alt, split.ref
    with np.errstate(invalid="ignore", divide="ignore"):
        obs_vaf = np.where(total > 0, alt / np.maximum(total, 1), 0.0)

    mk = lambda a: pd.DataFrame(a, index=expr.index, columns=expr.columns)
    return ScrnaAlleleCounts(mk(alt), mk(ref), mk(total), mk(obs_vaf))


class SyntheticTranscriptome:
    """Per-gene cDNA sequences for self-contained scRNA read emission (synthetic genome).

    Each gene gets a deterministic random cDNA of `read_length` bases with the variant site at a
    fixed interior position carrying the CANONICAL per-locus ref base; an alt-carrying molecule
    substitutes the CANONICAL alt base there. Mirrors the DNA `SyntheticReference`: works on the
    abstract genome with no external reference, and crucially shares the SAME canonical (ref, alt)
    map (`iscc.data.dna.genome_bases`) so a somatic SNV emits identical alleles in scRNA/Visium and
    DNA reads. Only the variant SITE must match across modalities; the surrounding transcript
    context legitimately differs from genomic context, so it stays independently random. All emitted
    reads cover the variant site — real 3'-bias coverage gaps (a real reason scRNA SNV calling is
    hard) are folded into `obs_fidelity`; the scReadSim path is for real-template realism.
    """

    def __init__(self, genes, seed=20240601, read_length=90, genome_seed=20240601):
        self.genes = list(genes)
        self.read_length = int(read_length)
        self.var_pos = self.read_length // 2
        rng = np.random.default_rng(seed)
        # Canonical per-locus (ref, alt) from the FIXED genome_seed — shared with DNA/Visium reads.
        ref_b, alt_b = genome_bases(self.genes, seed=genome_seed)
        self.seq = {}
        self.ref_base = {}
        self.alt_base = {}
        for gi, g in enumerate(self.genes):
            chars = list(_BASES[rng.integers(0, 4, size=self.read_length)].tolist())
            chars[self.var_pos] = str(ref_b[gi])   # canonical ref base at the variant site
            self.seq[g] = "".join(chars)
            self.ref_base[g] = str(ref_b[gi])
            self.alt_base[g] = str(alt_b[gi])       # canonical alt base for this locus


def write_scrna_fastq(alt, ref, transcriptome, outdir, barcode_len=16, umi_len=12,
                      error_rate=0.001, seed=42, prefix="scrna"):
    """Write 10x-style paired FASTQ from the observed alt/ref UMI matrices (self-contained).

    One read PER UMI (the count matrix is UMI-deduplicated), so the total read count equals the
    total UMIs — totals are conserved end to end. R1 = per-cell barcode (16 bp) + UMI (12 bp);
    R2 = the gene's cDNA covering the variant, with the alt base on alt-carrying molecules and a
    per-base sequencing-error floor. Returns the FASTQ paths, read count and the cell barcodes.
    """
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(seed)
    cells, genes = list(alt.index), list(alt.columns)
    barcodes = {c: "".join(_BASES[rng.integers(0, 4, size=barcode_len)].tolist()) for c in cells}
    r1_path = os.path.join(outdir, f"{prefix}_R1.fastq.gz")
    r2_path = os.path.join(outdir, f"{prefix}_R2.fastq.gz")
    q1, q2 = "I" * (barcode_len + umi_len), "I" * transcriptome.read_length
    altv, refv = alt.values, ref.values
    vp = transcriptome.var_pos
    n_reads = 0
    with gzip.open(r1_path, "wt") as f1, gzip.open(r2_path, "wt") as f2:
        for ci, c in enumerate(cells):
            bc = barcodes[c]
            for gi, g in enumerate(genes):
                na, nr = int(altv[ci, gi]), int(refv[ci, gi])
                if na + nr == 0:
                    continue
                base_seq = transcriptome.seq[g]
                alt_at_vp = transcriptome.alt_base[g]   # canonical alt (shared across modalities)
                for carries_alt, n in ((True, na), (False, nr)):
                    for _ in range(n):
                        seq = list(base_seq)
                        if carries_alt:
                            seq[vp] = alt_at_vp
                        if error_rate > 0:
                            for p in np.where(rng.random(len(seq)) < error_rate)[0]:
                                seq[p] = _ALT_OF.get(seq[p], "A")
                        umi = "".join(_BASES[rng.integers(0, 4, size=umi_len)].tolist())
                        rid = f"@{c}:{g}:{n_reads}"
                        f1.write(f"{rid}\n{bc}{umi}\n+\n{q1}\n")
                        f2.write(f"{rid}\n{''.join(seq)}\n+\n{q2}\n")
                        n_reads += 1
    return {"R1": r1_path, "R2": r2_path, "n_reads": n_reads, "barcodes": barcodes}


def spot_clone_mixture_vaf(spot_members, cell_exp, cell_rna_vaf, genes):
    """Spot-level clone-mixture RNA-VAF: the expression-weighted mixture of member cells' RNA-VAF.

    A Visium spot pools several cells of possibly different clones, so its observed RNA-VAF at gene g
    is NOT a single cell's value but the expression-weighted average over its members (a cell that
    expresses g more contributes more mutant/ref transcripts):

        spot_rna_vaf[s, g] = sum_{c in s} cell_exp[c, g] * cell_rna_vaf[c, g]
                             / sum_{c in s} cell_exp[c, g]                    (0 where unexpressed)

    This clone-mixture signal is exactly what spatial deconvolution must untangle. `spot_members` is
    the per-spot list of member cell ids (the Visium assay records it as ``Visium.spot_members``).
    Returns a (n_spots x n_genes) DataFrame aligned to `genes`.
    """
    exp = pd.DataFrame(cell_exp).reindex(columns=list(genes))
    vaf = pd.DataFrame(cell_rna_vaf).reindex(index=exp.index, columns=list(genes)).fillna(0.0)
    exp_v = exp.values.astype(float)
    vaf_v = vaf.values.astype(float)
    row_of = {cid: i for i, cid in enumerate(exp.index)}
    n_spots, n_genes = len(spot_members), len(genes)
    out = np.zeros((n_spots, n_genes))
    for s, mem in enumerate(spot_members):
        rows = [row_of[m] for m in (mem.tolist() if hasattr(mem, "tolist") else mem) if m in row_of]
        if not rows:
            continue
        e = exp_v[rows]
        num = (e * vaf_v[rows]).sum(axis=0)
        den = e.sum(axis=0)
        out[s] = np.divide(num, den, out=np.zeros(n_genes), where=den > 0)
    spot_names = [f"S{i}" for i in range(n_spots)]
    return pd.DataFrame(out, index=spot_names, columns=list(genes))


def emit_visium_reads(cell_data, grid_side=None, obs_fidelity=0.5, error_rate=0.001, seed=42,
                      cell_subset=None, emit_fastq=False, read_length=90, transcriptome=None,
                      outdir="visium_reads_out", count_model="dm", genome_seed=20240601,
                      site_coverage_model=False, **visium_kwargs):
    """End-to-end mutation-aware Visium reads: spot counts -> alt/ref UMIs.

    Visium reads ARE scRNA reads barcoded by SPOT instead of cell, so this reuses the scRNA
    read/count seam unchanged on the spot x gene axis (`observed_allele_counts` / `write_scrna_fastq` /
    `SyntheticTranscriptome`) — the per-"cell" barcode index simply becomes the SPATIAL barcode.

    Runs the Visium assay to get the spot x gene expression count matrix (fixing UMI totals) and
    the spot->cell membership it records, computes the spot-level clone-mixture RNA-VAF (the
    expression-weighted mixture of member cells' `cell_rna_vaf`; see `spot_clone_mixture_vaf`), then
    splits each spot-gene's UMIs into alt/ref via `observed_allele_counts`. The SAME invariants as
    scRNA hold: spot UMI totals conserved exactly (alt + ref == count), and per-base sequencing error
    is the SINGLE read-error source (the molecular split is error-free, error enters only in the
    FASTQ), so the FASTQ's non-reference proportion == `error_rate` (not doubled).

    Reads (``emit_fastq=True``): a self-contained 10x-style paired FASTQ on a synthetic
    transcriptome, R1 = a per-SPOT barcode + UMI, one read per UMI so totals are conserved.

    Returns a dict: ``alt``/``ref``/``total``/``obs_vaf`` (spots x genes), ``spot_rna_vaf`` (true
    clone-mixture), ``clone_frac`` (dominant-clone fraction per spot), ``dominant_clone``,
    ``n_cells``, ``obs_fidelity``, ``status``, ``fastq`` paths and ``n_reads``.
    """
    from ..visium import Visium

    if "cell_rna_vaf" not in cell_data:
        raise KeyError("cell_data lacks 'cell_rna_vaf'; call make_cell_data() on a current engine "
                       "(make_cell_data adds it). The mutation-aware Visium path needs the true RNA-VAF.")
    if "cell_crd" not in cell_data:
        raise KeyError("Visium needs spatial coordinates (cell_crd).")
    os.makedirs(outdir, exist_ok=True)
    if grid_side is None:
        grid_side = int(cell_data["cell_crd"].max().max()) + 1

    assay = Visium(seed=seed, count_model=count_model, **visium_kwargs).run(
        cell_data, grid_side=grid_side)
    spot_counts = assay.spot_counts                                # spots x genes, fixed UMI totals
    genes = spot_counts.columns
    spot_rna_vaf = spot_clone_mixture_vaf(assay.spot_members, cell_data["cell_exp"],
                                          cell_data["cell_rna_vaf"], genes)
    spot_rna_vaf.index = spot_counts.index

    # Visium is a 3' protocol: with the site-coverage model on, only molecules whose variant site
    # lies near the 3' end are read there (`site_coverage`); off (default), every molecule is.
    cover = site_coverage(genes, "visium", seed=genome_seed) if site_coverage_model else None
    rng = np.random.default_rng(seed + 7)
    allele = observed_allele_counts(spot_counts, spot_rna_vaf, obs_fidelity=obs_fidelity,
                                    error_rate=error_rate, rng=rng, site_cover=cover)

    result = {
        "alt": allele.alt, "ref": allele.ref, "total": allele.total, "obs_vaf": allele.obs_vaf,
        "spot_rna_vaf": spot_rna_vaf, "clone_frac": assay.obs["clone_frac"].copy(),
        "dominant_clone": assay.obs["clone"].copy(), "n_cells": assay.obs["n_cells"].copy(),
        "obs_fidelity": float(obs_fidelity), "spot_coords": assay.spot_coords,
        "fastq": [], "n_reads": 0, "status": "counts-only",
    }

    # self-contained synthetic-transcriptome reads (spot-barcoded; no external binary).
    if emit_fastq:
        if transcriptome is None:
            transcriptome = SyntheticTranscriptome(genes, seed=seed, read_length=read_length,
                                                   genome_seed=genome_seed)
        # Molecular content WITHOUT the sequencing-error floor (error_rate=0): the per-base error in
        # write_scrna_fastq is the SINGLE read-error source (otherwise it would be applied twice,
        # doubling the FASTQ's error reads — same single-source contract as scRNA reads).
        mol = observed_allele_counts(spot_counts, spot_rna_vaf, obs_fidelity=obs_fidelity,
                                     error_rate=0.0, rng=np.random.default_rng(seed + 9),
                                     site_cover=cover)
        fq = write_scrna_fastq(mol.alt, mol.ref, transcriptome, outdir, error_rate=error_rate,
                               seed=seed + 11, prefix="visium")
        result["fastq"] = [fq["R1"], fq["R2"]]
        result["n_reads"] = fq["n_reads"]
        result["barcodes"] = fq["barcodes"]                        # one spatial barcode per spot
        result["status"] = "emitted:synthetic-fastq"
    return result


class ScReadSimAdapter:
    """scReadSim adapter (optional realism layer). Count -> barcoded/UMI reads on a REAL template.

    scReadSim emulates real scRNA reads by learning from a template BAM + real reference, matching
    a given count matrix — so it CONSERVES the per-cell-gene UMI totals we hand it. The somatic
    mutation content (the alt/ref split above) is injected at the expressed-variant loci. Requires
    a real reference/template even though iscc's counts are synthetic; left as a documented seam
    and skipped when the binary is absent (CI stays light). DWGSIM/ART (DNA) do not apply here.
    """

    binary = "scReadSim"

    def build_command(self, count_matrix_path, reference, template_bam, out_prefix, extra=None):
        cmd = ["--count", count_matrix_path, "--ref", reference,
               "--bam", template_bam, "--out", out_prefix]
        if extra:
            cmd.extend(extra)
        return [str(a) for a in cmd]


def emit_scrna_reads(cell_data, obs_fidelity=0.5, protocol="10x", error_rate=0.001,
                     seed=42, cell_subset=None, emit_fastq=False, read_length=90,
                     transcriptome=None, reference=None, template_bam=None,
                     outdir="scrna_reads_out", genome_seed=20240601, site_coverage_model=False,
                     **scrna_kwargs):
    """End-to-end mutation-aware scRNA: counts -> observed alt/ref UMI matrices -> reads.

    Runs the scRNA assay to get the expression count matrix (fixing UMI totals), pulls the
    engine's `cell_rna_vaf`, and produces the count-level observed allele matrices via
    `observed_allele_counts`. Surfaces the true DNA-VAF (`cell_snv`) alongside, for the
    scDNA-vs-scRNA benchmark.

    Reads (``emit_fastq=True``): a self-contained 10x-style paired FASTQ on a synthetic
    transcriptome (`write_scrna_fastq`) — barcodes + UMIs, one read per UMI so totals are
    conserved, the alt base carried on alt UMIs. No external binary (mirrors the DNA
    SyntheticReference). Supplying a real `reference` + `template_bam` selects the scReadSim seam
    instead (realism on real sequence), skipped gracefully when the binary is absent.

    Parameters
    ----------
    cell_data : dict
        Materialised per-cell tables of a tumour or a sample; must hold ``cell_rna_vaf``.
    obs_fidelity : float, optional
        Probability that a mutant transcript is reported as mutant (default 0.5), so the observed
        RNA-VAF is ``cell_rna_vaf * obs_fidelity``. One scalar for monoallelic expression,
        bursting, RNA editing and reverse-transcription error.
    protocol : {"10x", "smartseq3"}, optional
        The scRNA protocol of the assay (default ``"10x"``), which also sets the read window when
        ``site_coverage_model`` is on.
    error_rate : float, optional
        Per-base error rate of the reads (default 0.001).
    seed : int, optional
        Seed of the assay and of the allele split (default 42).
    cell_subset : sequence, optional
        Cells to assay (default all).
    emit_fastq : bool, optional
        Write paired FASTQ on a synthetic transcriptome (default False, counts only).
    read_length : int, optional
        Length of the cDNA read of the synthetic transcriptome (default 90).
    transcriptome : SyntheticTranscriptome, optional
        A synthetic transcriptome to reuse (default None, built from the genes and ``seed``).
    reference, template_bam : str, optional
        A real reference and a template BAM, which select the scReadSim path instead of the
        synthetic transcriptome. It is skipped when the ``scReadSim`` binary is absent.
    outdir : str, optional
        Output directory (default ``"scrna_reads_out"``).
    genome_seed : int, optional
        Seed of the transcript lengths and variant-site positions, shared by every run and modality
        (default 20240601).
    site_coverage_model : bool, optional
        Read a variant site only where the protocol reaches it (default False). A 3' protocol
        (10x) reads a site only within the last 1,000 bases of its transcript and a full-length
        protocol (Smart-seq3) anywhere, and a molecule within reach is read at the site with
        probability 0.1, as in the paper. When False, every molecule is read at the site.
    **scrna_kwargs
        Passed to [`scRNA`][iscc.data.scRNA].

    Returns
    -------
    dict
        ``alt``/``ref``/``total``/``obs_vaf`` (cells x genes), ``dna_vaf`` (true, aligned),
        ``obs_fidelity``, ``protocol``, ``site_cover``, ``status``, ``fastq`` (paths when emitted)
        and ``n_reads``.
    """
    from ..rna import scRNA

    if "cell_rna_vaf" not in cell_data:
        raise KeyError("cell_data lacks 'cell_rna_vaf'; call make_cell_data() on a current engine "
                       "(make_cell_data adds it). The mutation-aware scRNA path needs the true RNA-VAF.")
    os.makedirs(outdir, exist_ok=True)

    assay = scRNA(protocol=protocol, seed=seed, **scrna_kwargs).run(cell_data, cell_subset=cell_subset)
    expr = assay.observed_counts                                   # cells x genes, fixed UMI totals
    # With the site-coverage model on, a 3' protocol (10x) reads a molecule at its variant site only
    # if the site lies near the 3' end, a full-length one (Smart-seq3) always (`site_coverage`).
    cover = (site_coverage(expr.columns, protocol, seed=genome_seed) if site_coverage_model else None)
    rng = np.random.default_rng(seed + 7)
    allele = observed_allele_counts(expr, cell_data["cell_rna_vaf"], obs_fidelity=obs_fidelity,
                                    error_rate=error_rate, rng=rng, site_cover=cover)
    dna_vaf = pd.DataFrame(cell_data["cell_snv"]).reindex(index=expr.index, columns=expr.columns)

    result = {
        "alt": allele.alt, "ref": allele.ref, "total": allele.total, "obs_vaf": allele.obs_vaf,
        "dna_vaf": dna_vaf, "obs_fidelity": float(obs_fidelity), "protocol": protocol,
        "site_cover": (None if cover is None else pd.Series(cover, index=expr.columns)),
        "fastq": [], "n_reads": 0, "status": "counts-only",
    }

    # real-template realism path (scReadSim) when a real reference + template BAM are supplied.
    if reference is not None and template_bam is not None:
        adapter = ScReadSimAdapter()
        if find_binary(adapter.binary) is None:
            result["status"] = f"counts-only:{adapter.binary}-absent"
            return result
        count_path = os.path.join(outdir, "counts.csv")
        expr.to_csv(count_path)
        result["command"] = adapter.build_command(count_path, reference, template_bam,
                                                  os.path.join(outdir, "sim"))
        result["status"] = "reads-seam:screadsim"
        return result

    # self-contained synthetic-transcriptome reads (no external binary).
    if emit_fastq:
        if transcriptome is None:
            transcriptome = SyntheticTranscriptome(expr.columns, seed=seed, read_length=read_length,
                                                   genome_seed=genome_seed)
        # The molecular content handed to the writer is the TRUE alt/ref after obs_fidelity but
        # WITHOUT the sequencing-error floor (error_rate=0): the per-base error in
        # write_scrna_fastq is the SINGLE read-error source. Otherwise the floor would be applied
        # twice (here AND in the count split), doubling the FASTQ's error reads at the variant
        # site. With one source, the FASTQ's non-ref proportion == error_rate and its observed VAF
        # == the count matrix's p_eff = v_obs*(1-e)+(1-v_obs)*e (see test_error_rate_single_source).
        mol = observed_allele_counts(expr, cell_data["cell_rna_vaf"], obs_fidelity=obs_fidelity,
                                     error_rate=0.0, rng=np.random.default_rng(seed + 9),
                                     site_cover=cover)
        fq = write_scrna_fastq(mol.alt, mol.ref, transcriptome, outdir,
                               error_rate=error_rate, seed=seed + 11)
        result["fastq"] = [fq["R1"], fq["R2"]]
        result["n_reads"] = fq["n_reads"]
        result["barcodes"] = fq["barcodes"]
        result["status"] = "emitted:synthetic-fastq"
    return result
