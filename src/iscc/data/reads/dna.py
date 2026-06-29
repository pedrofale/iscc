"""DNA read-emission adapter (DESIGN_features §C C2/C3, §E `reads/dna.py`, F7).

SISTEM-faithful (Weiner & Bansal 2025): per-cell full reference -> copy-number-scaled
coverage -> a third-party short-read simulator -> (optional) BAM. Concretely:

  1. A pluggable `Reference` (synthetic default / real-genome seam) gives a nucleotide
     sequence per genome segment.
  2. `build_cell_fasta` applies a cell's **CNAs** (duplicate / delete segment sequence) and
     **SNVs** (substitute bases at mutated loci, allele-aware) -> a per-cell FASTA. Copy-number
     scaling is thus baked into the reference multiplicity exactly as SISTEM does it.
  3. Coverage REUSES the C1 model in `data/dna.py` (`coverage_budget`) for the per-segment
     read budget; the allele split is the shared `variants.inject` seam.
  4. A read-simulator adapter (default **DWGSIM**, **ART** alternative) writes the FASTA +
     coverage, shells out via `run_binary`, collects FASTQ. C3 aligns back (bwa + samtools).

Allele-layer consistency with the count model (`data/dna.py`/`batch.py`, the model `estimate_dna`
inverts). The coverage-layer hypers (`mu_depth`, `kappa`, GC, capture, …) are already shared via
`coverage_budget`. The allele/base-layer hypers are wired here too so reads reproduce the same
statistics the estimate is fit to:
  * **single-cell ADO + allele overdispersion** — `effective_alt_fraction` applies the SAME
    `DNABatch.apply_ado` + `DNABatch.allele_balance` the count path uses, then bakes the observed
    fraction into per-copy alt multiplicity (so het loci drop to 0/1 and allele balance is lumpy).
  * **sequencing error** — `error_rate` is wired into the simulator's per-base error (DWGSIM `-e`),
    the read-level base-error floor (ART uses its sequencing-system profile instead).
  * **FFPE C>T (`ffpe_ct_rate`)** is NOT yet reproduced in reads (count-level only) — a documented
    gap (needs per-locus C-site context in the reference).

The external binaries are OPTIONAL: everything up to the shell-out (FASTA, coverage, command
construction) runs with no tools installed, and `emit_reads` reports a graceful skip.
"""
from __future__ import annotations

import os
from collections import OrderedDict

import numpy as np

from .base import collect_fastq, find_binary, run_binary
from . import variants

_BASES = np.array(list("ACGT"))
# Deterministic alt base for each ref base (a fixed transition/transversion), so an injected
# SNV is always a real substitution (alt != ref) and reproducible.
_ALT_OF = {"A": "G", "G": "A", "C": "T", "T": "C", "N": "A"}


def _segment_of(gene):
    """Parse the segment index from an engine gene name `G_{seg}_{pos}` (else -1)."""
    s = str(gene).split("_")
    if len(s) >= 3 and s[0] == "G":
        try:
            return int(s[1])
        except ValueError:
            return -1
    return -1


# ======================================================================================
# Reference backends (pluggable; SISTEM's "pass one or multiple reference genomes" model)
# ======================================================================================
class Reference:
    """Interface: a nucleotide sequence per genome segment + the base index of each locus.

    Subclasses expose:
      * ``seg_ids``           — ordered segment ids present in the genome.
      * ``segments[seg]``     — global locus indices belonging to that segment.
      * ``base_seq[seg]``     — the segment's reference nucleotide string.
      * ``locus_local_pos[gl]`` — the base offset (within its segment seq) of locus `gl`,
                                  i.e. the site an SNV at that locus substitutes.
    """

    seg_ids: list
    segments: dict
    base_seq: dict
    locus_local_pos: dict

    def write_fasta(self, path, sequences=None):
        """Write `sequences` (default = the bare reference) as ``>seg{id}`` records."""
        seqs = sequences if sequences is not None else {
            f"seg{seg}": self.base_seq[seg] for seg in self.seg_ids
        }
        with open(path, "w") as fh:
            for name, seq in seqs.items():
                fh.write(f">{name}\n")
                for i in range(0, len(seq), 70):
                    fh.write(seq[i:i + 70] + "\n")
        return path


class SyntheticReference(Reference):
    """Backend (A) — a random per-segment FASTA, deterministic in `seed` (the DEFAULT).

    Works on today's abstract genome (genes `G_{seg}_{pos}`): each locus owns a fixed window
    of `locus_length` random bases; the segment sequence is those windows concatenated. The
    lack of real sequence context (benchmark transfer) is RESEARCH_QUESTIONS R5 — handled by
    the real-genome backend below, behind the same interface.
    """

    def __init__(self, genes, seed=20240601, locus_length=60):
        self.genes = list(genes)
        self.locus_length = int(locus_length)
        rng = np.random.default_rng(seed)
        seg_to_loci = OrderedDict()
        for gl, g in enumerate(self.genes):
            seg_to_loci.setdefault(_segment_of(g), []).append(gl)
        self.seg_ids = list(seg_to_loci.keys())
        self.segments = {seg: loci for seg, loci in seg_to_loci.items()}
        self.base_seq = {}
        self.locus_local_pos = {}
        for seg, loci in self.segments.items():
            chars = _BASES[rng.integers(0, 4, size=len(loci) * self.locus_length)]
            self.base_seq[seg] = "".join(chars.tolist())
            for j, gl in enumerate(loci):
                # substitute at the window midpoint (a stable, well-inside-read site).
                self.locus_local_pos[gl] = j * self.locus_length + self.locus_length // 2


class RealGenomeReference(Reference):
    """Backend (B) — a user-supplied FASTA anchored to the M3b real-genome mode (seam).

    Segments = chromosome arms -> hg38 coordinates (`iscc.inference.genome.GenomeSpec`). The
    SEAM is real: it ingests a FASTA and maps each arm's loci to coordinates, so a mature
    real-genome wiring drops in unchanged. The coordinate spreading is a thin default (evenly
    spaced within each contig) pending the full M3b locus->bp map; supply `locus_coords` to
    override. Requires a real per-segment FASTA, hence not a synthetic fallback.
    """

    def __init__(self, fasta_path, genes, seg_to_contig=None, locus_coords=None):
        self.fasta_path = fasta_path
        self.genes = list(genes)
        self._contigs = self._read_fasta(fasta_path)
        seg_to_loci = OrderedDict()
        for gl, g in enumerate(self.genes):
            seg_to_loci.setdefault(_segment_of(g), []).append(gl)
        self.seg_ids = list(seg_to_loci.keys())
        self.segments = {seg: loci for seg, loci in seg_to_loci.items()}
        # map each segment id to a contig name in the FASTA (default: name order).
        names = list(self._contigs.keys())
        self.seg_to_contig = seg_to_contig or {
            seg: names[i % len(names)] for i, seg in enumerate(self.seg_ids)
        }
        self.base_seq = {seg: self._contigs[self.seg_to_contig[seg]] for seg in self.seg_ids}
        self.locus_local_pos = {}
        for seg, loci in self.segments.items():
            L = len(self.base_seq[seg])
            for j, gl in enumerate(loci):
                if locus_coords is not None and gl in locus_coords:
                    self.locus_local_pos[gl] = int(locus_coords[gl])
                else:  # thin default: evenly spaced sites within the contig.
                    self.locus_local_pos[gl] = int((j + 0.5) / max(len(loci), 1) * L)

    @staticmethod
    def _read_fasta(path):
        contigs = OrderedDict()
        name, chunks = None, []
        with open(path) as fh:
            for line in fh:
                line = line.rstrip("\n")
                if line.startswith(">"):
                    if name is not None:
                        contigs[name] = "".join(chunks)
                    name, chunks = line[1:].split()[0], []
                elif name is not None:
                    chunks.append(line.strip())
        if name is not None:
            contigs[name] = "".join(chunks)
        return contigs


REFERENCES = {"synthetic": SyntheticReference, "real": RealGenomeReference}


# ======================================================================================
# Per-cell reference construction (CNAs + allele-aware SNVs -> FASTA)
# ======================================================================================
def _segment_cn(cnv_row, loci):
    """Integer copy number for a segment = rounded CN at its loci (constant within a segment)."""
    vals = np.asarray(cnv_row, dtype=float)[loci]
    return int(round(float(np.median(vals)))) if len(vals) else 2


def build_cell_fasta(reference, cnv_row, af_row, path, name="cell"):
    """Apply one cell's CNAs + SNVs to `reference` -> a per-cell FASTA on disk.

    CNAs: a segment with copy number `cn` is emitted as `cn` sequence copies (amplification =
    repeated sequence; deletion `cn=0` = omitted) — so a downstream simulator at uniform
    coverage yields reads ∝ CN. SNVs: at each locus with alt-fraction `v>0`, the alt base is
    substituted on ``round(v*cn)`` of the `cn` copies (**allele-aware**: a het `v=0.5`, `cn=2`
    mutates exactly one copy; a homozygous `v=1` mutates all).

    `af_row` is whatever alt fraction the caller wants encoded: the TRUE fraction for bulk, or the
    OBSERVED fraction (after `effective_alt_fraction`'s ADO + allele overdispersion) for single
    cells — so single-cell allelic dropout shows up as copies collapsing to all-ref / all-alt.

    Returns the OrderedDict of ``contig_name -> sequence`` that was written.
    """
    cnv_row = np.asarray(cnv_row, dtype=float)
    af_row = np.asarray(af_row, dtype=float)
    records = OrderedDict()
    for seg in reference.seg_ids:
        loci = reference.segments[seg]
        cn = _segment_cn(cnv_row, loci)
        if cn <= 0:
            continue  # homozygous deletion -> segment absent from this cell's genome
        base = reference.base_seq[seg]
        # which copies carry the alt allele at each mutated locus (allele-aware).
        mutated = [(reference.locus_local_pos[gl], int(round(af_row[gl] * cn)))
                   for gl in loci if af_row[gl] > 0]
        for cp in range(cn):
            seq = list(base)
            for pos, n_alt in mutated:
                if cp < n_alt:  # this copy carries the alt allele at this locus
                    seq[pos] = _ALT_OF.get(seq[pos], "A")
            records[f"{name}_seg{seg}_cp{cp}"] = "".join(seq)
    reference.write_fasta(path, records)
    return records


def effective_alt_fraction(af_row, batch=None, modality="bulk"):
    """Map a cell's TRUE per-locus alt fraction to the OBSERVED fraction the reads should carry.

    Bulk: returns the true fraction unchanged — population mixing of many cells produces the bulk
    VAF, and single-cell amplification artifacts do not apply.

    Single-cell: applies the SAME allele model the count path uses (`DNABatch.apply_ado` then
    `DNABatch.allele_balance`), so reads reproduce the dominant scDNA artifacts — allelic dropout
    (het loci collapse to 0/1 with prob `ado_rate`) and Beta-Binomial allele-balance overdispersion
    (`beta_binom_conc`). The balance is taken WITHOUT the sequencing-error floor
    (`apply_error=False`): this is the TRUE molecular content, and the per-base read error (DWGSIM
    `-e`, wired from `error_rate`) is the SINGLE read-error source — otherwise the floor would be
    applied twice (here AND by the simulator). `build_cell_fasta` then quantises at copy-number
    resolution.
    """
    af = np.asarray(af_row, dtype=float)
    if modality != "sc" or batch is None:
        return af
    af_obs, _ = batch.apply_ado(af)
    return np.asarray(batch.allele_balance(af_obs, apply_error=False), dtype=float)


# ======================================================================================
# Coverage (REUSE the C1 copy-number-scaled model in data/dna.py)
# ======================================================================================
def coverage_budget(cell_data, breadth="wgs", modality="bulk", seed=42,
                    cell_subset=None, **dna_kwargs):
    """Per-segment read budget from the C1 model (copy-number-scaled, breadth-aware).

    Runs the existing `bulkDNA` / `scDNA` coverage engine and aggregates per-locus coverage to
    a per-segment ``total`` (∝ copy number) — the read budget a simulator must place. Returns
    ``(per_segment: dict seg->reads, assay)`` where `assay` carries the per-locus coverage and
    true alt fraction (for `variants.inject`).
    """
    from ..dna import bulkDNA, scDNA

    if modality == "bulk":
        assay = bulkDNA(breadth=breadth, seed=seed, **dna_kwargs).run(
            cell_data, cell_subset=cell_subset)
        od = assay.observed_data
        per_seg = od.groupby("segment")["coverage"].sum().to_dict()
    elif modality == "sc":
        assay = scDNA(breadth=breadth, seed=seed, **dna_kwargs).run(
            cell_data, cell_subset=cell_subset)
        cov = assay.coverage  # cells x loci
        segs = np.array([_segment_of(g) for g in cov.columns])
        per_seg = {}
        for s in np.unique(segs):
            per_seg[int(s)] = int(cov.values[:, segs == s].sum())
    else:
        raise ValueError(f"modality must be 'bulk' or 'sc', got {modality!r}")
    return {int(k): int(v) for k, v in per_seg.items()}, assay


# ======================================================================================
# Read-simulator adapters (default DWGSIM, ART alternative)
# ======================================================================================
class DwgsimAdapter:
    """DWGSIM adapter (SISTEM's default). Builds ``dwgsim [-C cov] ref.fa prefix``."""

    binary = "dwgsim"

    def build_command(self, ref_fa, out_prefix, coverage, read_length=150,
                      error_rate=0.001, extra=None):
        cmd = [
            "-C", float(coverage),
            "-1", int(read_length), "-2", int(read_length),
            "-e", float(error_rate), "-E", float(error_rate),
        ]
        if extra:
            cmd.extend(extra)
        cmd.extend([ref_fa, out_prefix])
        return [str(a) for a in cmd]

    def expected_fastq(self, out_prefix):
        # dwgsim writes <prefix>.bwa.read1.fastq.gz / read2.fastq.gz
        return [out_prefix + ".bwa.read1.fastq.gz", out_prefix + ".bwa.read2.fastq.gz"]

    def emit(self, ref_fa, out_prefix, coverage, **opts):
        args = self.build_command(ref_fa, out_prefix, coverage, **opts)
        run_binary(self.binary, args)
        return collect_fastq(out_prefix) or self.expected_fastq(out_prefix)


class ArtAdapter:
    """ART (`art_illumina`) adapter (alternative). Builds a paired-end Illumina command."""

    binary = "art_illumina"

    def build_command(self, ref_fa, out_prefix, coverage, read_length=150,
                      seqsys="HS25", frag_mean=200, frag_sd=10, error_rate=None, extra=None):
        # ART draws base errors from the empirical sequencing-system profile (`-ss`), so the
        # scalar `error_rate` does not map to a flag here; accepted for a uniform adapter API
        # and ignored (DWGSIM is the error-rate-driven backend).
        cmd = [
            "-ss", seqsys, "-sam", "-p",
            "-i", ref_fa, "-l", int(read_length), "-f", float(coverage),
            "-m", int(frag_mean), "-s", int(frag_sd), "-o", out_prefix,
        ]
        if extra:
            cmd.extend(extra)
        return [str(a) for a in cmd]

    def expected_fastq(self, out_prefix):
        return [out_prefix + "1.fq", out_prefix + "2.fq"]

    def emit(self, ref_fa, out_prefix, coverage, **opts):
        args = self.build_command(ref_fa, out_prefix, coverage, **opts)
        run_binary(self.binary, args)
        return collect_fastq(out_prefix) or self.expected_fastq(out_prefix)


SIMULATORS = {"dwgsim": DwgsimAdapter, "art": ArtAdapter}


# ======================================================================================
# C3 — alignment to BAM (bwa + samtools); gated behind C2 (needs FASTQ)
# ======================================================================================
def align_to_bam(ref_fa, fastqs, out_bam, threads=1):
    """Align FASTQ back to `ref_fa` (bwa mem) + samtools sort/index -> sorted, indexed BAM.

    Gated behind C2: requires the FASTQ produced by a simulator. Raises `MissingBinaryError`
    (via `run_binary`) if bwa/samtools are absent.
    """
    if not fastqs:
        raise ValueError("align_to_bam needs FASTQ input (run the C2 simulator first)")
    sam = out_bam + ".sam"
    run_binary("bwa", ["index", ref_fa])
    run_binary("bwa", ["mem", "-t", threads, ref_fa, *fastqs], stdout_path=sam)
    run_binary("samtools", ["sort", "-@", threads, "-o", out_bam, sam])
    run_binary("samtools", ["index", out_bam])
    return out_bam


# ======================================================================================
# High-level entry point
# ======================================================================================
def _select_cells(cell_data, modality, n_cells, cell_subset, seed):
    cells = np.asarray(cell_data["cell_snv"].index)
    if cell_subset is not None:
        return np.asarray(list(cell_subset))
    if modality == "bulk":
        return cells
    n = min(int(n_cells), len(cells))
    rng = np.random.default_rng(seed)
    return rng.choice(cells, size=n, replace=False)


def emit_reads(cell_data, reference=None, simulator="dwgsim", breadth="wgs",
               modality="bulk", outdir="reads_out", seed=42, n_cells=20,
               cell_subset=None, mean_coverage=None, emit_bam=False,
               read_length=150, **dna_kwargs):
    """End-to-end DNA read emission: cell_data -> per-cell FASTA -> FASTQ (+ BAM).

      reference     a `Reference` instance, or None for the SyntheticReference default.
      simulator     "dwgsim" (default) or "art".
      modality      "bulk" (pool cells into one reference) or "sc" (per-cell FASTA + FASTQ).
      mean_coverage uniform fold coverage handed to the simulator (CN scaling already baked
                    into the FASTA); defaults from the C1 budget if None.
      emit_bam      also align the FASTQ to a sorted/indexed BAM (needs bwa + samtools).

    Returns a result dict with the FASTA path(s), the per-segment coverage budget, the chosen
    `mean_coverage`, the true/observed allele split (from `variants.inject`), and — when the
    binary is installed — ``fastq``/``bam`` paths. ``status`` is "emitted" or "skipped:<tool>"
    so the count-level path and CI stay green without the optional binaries.
    """
    os.makedirs(outdir, exist_ok=True)
    snv = cell_data["cell_snv"]
    genes = list(snv.columns)
    if reference is None:
        reference = SyntheticReference(genes, seed=seed)
    if simulator not in SIMULATORS:
        raise ValueError(f"unknown simulator {simulator!r}; choose from {sorted(SIMULATORS)}")
    adapter = SIMULATORS[simulator]()

    cells = _select_cells(cell_data, modality, n_cells, cell_subset, seed)
    per_seg, assay = coverage_budget(cell_data, breadth=breadth, modality=modality,
                                     seed=seed, cell_subset=cells, **dna_kwargs)
    if mean_coverage is None:
        mean_coverage = float(assay.hypers.mu_depth)

    # per-cell FASTA (sc: one per cell; bulk: pool the cells into one reference). For single
    # cells, map the true alt fraction to the OBSERVED fraction (ADO + allele overdispersion,
    # via the count model's allele methods) so reads carry the same scDNA artifacts the
    # estimate is fit to; bulk uses the true fraction (population mixing gives the bulk VAF).
    cnv_df = cell_data.get("cell_cnv")
    batch = getattr(assay, "batch", None)
    fasta_paths = []
    pool_records = OrderedDict()
    for c in cells:
        cnv_row = (cnv_df.loc[c, genes].values if cnv_df is not None
                   else np.full(len(genes), 2.0))
        af_row = effective_alt_fraction(snv.loc[c, genes].values, batch=batch, modality=modality)
        if modality == "sc":
            p = os.path.join(outdir, f"{c}.fa")
            build_cell_fasta(reference, cnv_row, af_row, p, name=str(c))
            fasta_paths.append(p)
        else:  # bulk: accumulate records into one pooled FASTA
            recs = build_cell_fasta(reference, cnv_row, af_row,
                                    os.path.join(outdir, "_tmp.fa"), name=str(c))
            pool_records.update(recs)
    if modality == "bulk":
        p = os.path.join(outdir, "pooled.fa")
        reference.write_fasta(p, pool_records)
        fasta_paths = [p]
        try:
            os.remove(os.path.join(outdir, "_tmp.fa"))
        except OSError:
            pass

    # shared variant seam: split each segment's read budget into alt/ref counts (count-level
    # allele content; the FASTA carries the same VAF via copy multiplicity for the simulator).
    rng = np.random.default_rng(seed + 1)
    seg_loci = reference.segments
    seg_af = {}
    for seg, loci in seg_loci.items():
        af_seg = snv.loc[cells, [genes[gl] for gl in loci]].values
        seg_af[seg] = float(af_seg.mean()) if af_seg.size else 0.0
    totals = np.array([per_seg.get(int(s), 0) for s in reference.seg_ids])
    fracs = np.array([seg_af[s] for s in reference.seg_ids])
    split = variants.inject(totals, fracs, error_rate=float(assay.hypers.error_rate), rng=rng)

    result = {
        "fasta": fasta_paths,
        "per_segment_coverage": per_seg,
        "mean_coverage": mean_coverage,
        "simulator": simulator,
        "modality": modality,
        "n_cells": len(cells),
        "allele_split": {int(s): (int(a), int(r)) for s, a, r in
                         zip(reference.seg_ids, split.alt, split.ref)},
        "command": adapter.build_command(fasta_paths[0],
                                         os.path.join(outdir, "sim"),
                                         mean_coverage, read_length=read_length,
                                         error_rate=float(assay.hypers.error_rate)),
        "fastq": [],
        "bam": None,
        "status": "skipped",
    }

    if find_binary(adapter.binary) is None:
        result["status"] = f"skipped:{adapter.binary}-absent"
        return result

    fastqs = []
    for fa in fasta_paths:
        prefix = os.path.splitext(fa)[0] + ".sim"
        fastqs.extend(adapter.emit(fa, prefix, mean_coverage, read_length=read_length,
                                   error_rate=float(assay.hypers.error_rate)))
    result["fastq"] = fastqs
    result["status"] = "emitted"

    if emit_bam:
        if find_binary("bwa") is None or find_binary("samtools") is None:
            result["status"] = "emitted:fastq-only(bwa/samtools-absent)"
        else:
            bam = os.path.join(outdir, "reads.sorted.bam")
            align_to_bam(fasta_paths[0], fastqs, bam)
            result["bam"] = bam
    return result


class DNAReadEmitter:
    """`ReadEmitter`-conforming wrapper around `emit_reads` (DESIGN_features §E protocol)."""

    def __init__(self, simulator="dwgsim", breadth="wgs", modality="bulk", **kwargs):
        self.simulator = simulator
        self.breadth = breadth
        self.modality = modality
        self.kwargs = kwargs

    def emit(self, matrix, reference, outdir):
        return emit_reads(matrix, reference=reference, simulator=self.simulator,
                          breadth=self.breadth, modality=self.modality, outdir=outdir,
                          **self.kwargs)
