"""Mutation-aware scRNA read/count emission (DESIGN_features §C, F7b).

Real scRNA reads carry a cell's *expressed* somatic mutations, so iscc can show that **SNV
calling from scRNA is intrinsically hard**. This adapter reuses the shared `variants.inject`
seam (total-preserving) and the F3 expression count matrix, under two hard constraints:

  1. **UMI totals are conserved.** scRNA reads are DRIVEN by the F3 expression count matrix — we
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


def observed_allele_counts(expr_counts, rna_vaf, obs_fidelity=1.0, error_rate=0.001, rng=None):
    """Split each cell-gene's expression UMIs into alt/ref counts, conserving the total.

      expr_counts   cells x genes UMI matrix (the F3 scRNA count matrix; fixes the totals).
      rna_vaf       cells x genes true RNA-VAF (`cell_rna_vaf`); reindexed to `expr_counts`.
      obs_fidelity  the single distortion knob (see `distort_vaf`).
      error_rate    per-base RNA error/editing floor handed to `variants.inject`.

    Returns a `ScrnaAlleleCounts` of cells x genes DataFrames. `alt + ref == expr_counts`
    element-wise (UMI totals conserved); `obs_vaf = alt / total` (0 where the gene is unexpressed,
    i.e. uncallable).
    """
    if rng is None:
        rng = np.random.default_rng()
    expr = expr_counts if isinstance(expr_counts, pd.DataFrame) else pd.DataFrame(expr_counts)
    vaf = pd.DataFrame(rna_vaf).reindex(index=expr.index, columns=expr.columns).fillna(0.0)

    total = expr.values.astype(np.int64)
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
    substitutes the CANONICAL alt base there. Mirrors the DNA `SyntheticReference`: works on today's
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
    the per-spot list of member cell ids (the F6 assay records it as ``Visium.spot_members``).
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
                      **visium_kwargs):
    """End-to-end mutation-aware Visium reads (F6 reads, reuses F7b): spot counts -> alt/ref UMIs.

    Visium reads ARE scRNA reads barcoded by SPOT instead of cell, so this reuses the F7b seam
    unchanged on the spot x gene axis (`observed_allele_counts` / `write_scrna_fastq` /
    `SyntheticTranscriptome`) — the per-"cell" barcode index simply becomes the SPATIAL barcode.

    Runs the F6 Visium assay to get the spot x gene expression count matrix (fixing UMI totals) and
    the spot->cell membership it records, computes the NEW spot-level clone-mixture RNA-VAF (the
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
                       "(F7b adds it). The mutation-aware Visium path needs the true RNA-VAF.")
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

    rng = np.random.default_rng(seed + 7)
    allele = observed_allele_counts(spot_counts, spot_rna_vaf, obs_fidelity=obs_fidelity,
                                    error_rate=error_rate, rng=rng)

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
                                     error_rate=0.0, rng=np.random.default_rng(seed + 9))
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
                     outdir="scrna_reads_out", genome_seed=20240601, **scrna_kwargs):
    """End-to-end mutation-aware scRNA: F3 counts -> observed alt/ref UMI matrices -> reads.

    Runs the F3 scRNA assay to get the expression count matrix (fixing UMI totals), pulls the
    engine's `cell_rna_vaf`, and produces the count-level observed allele matrices via
    `observed_allele_counts`. Surfaces the true DNA-VAF (`cell_snv`) alongside, for the
    scDNA-vs-scRNA benchmark.

    Reads (``emit_fastq=True``): a self-contained 10x-style paired FASTQ on a synthetic
    transcriptome (`write_scrna_fastq`) — barcodes + UMIs, one read per UMI so totals are
    conserved, the alt base carried on alt UMIs. No external binary (mirrors the DNA
    SyntheticReference). Supplying a real `reference` + `template_bam` selects the scReadSim seam
    instead (realism on real sequence), skipped gracefully when the binary is absent.

    Returns a dict: ``alt``/``ref``/``total``/``obs_vaf`` (cells x genes), ``dna_vaf`` (true,
    aligned), ``obs_fidelity``, ``status``, ``fastq`` (paths when emitted) and ``n_reads``.
    """
    from ..rna import scRNA

    if "cell_rna_vaf" not in cell_data:
        raise KeyError("cell_data lacks 'cell_rna_vaf'; call make_cell_data() on a current engine "
                       "(F7b adds it). The mutation-aware scRNA path needs the true RNA-VAF.")
    os.makedirs(outdir, exist_ok=True)

    assay = scRNA(protocol=protocol, seed=seed, **scrna_kwargs).run(cell_data, cell_subset=cell_subset)
    expr = assay.observed_counts                                   # cells x genes, fixed UMI totals
    rng = np.random.default_rng(seed + 7)
    allele = observed_allele_counts(expr, cell_data["cell_rna_vaf"], obs_fidelity=obs_fidelity,
                                    error_rate=error_rate, rng=rng)
    dna_vaf = pd.DataFrame(cell_data["cell_snv"]).reindex(index=expr.index, columns=expr.columns)

    result = {
        "alt": allele.alt, "ref": allele.ref, "total": allele.total, "obs_vaf": allele.obs_vaf,
        "dna_vaf": dna_vaf, "obs_fidelity": float(obs_fidelity), "protocol": protocol,
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
                                     error_rate=0.0, rng=np.random.default_rng(seed + 9))
        fq = write_scrna_fastq(mol.alt, mol.ref, transcriptome, outdir,
                               error_rate=error_rate, seed=seed + 11)
        result["fastq"] = [fq["R1"], fq["R2"]]
        result["n_reads"] = fq["n_reads"]
        result["barcodes"] = fq["barcodes"]
        result["status"] = "emitted:synthetic-fastq"
    return result
