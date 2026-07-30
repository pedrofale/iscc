"""Read emission (DESIGN_features §C C2/C3, F7): count/coverage matrices -> FASTQ / BAM.

Per-modality read-realism ADAPTERS over shared seams:
  * `base`     — the `ReadEmitter` protocol + `run_binary` (PATH detect / skip-if-absent).
  * `variants` — the modality-agnostic variant-injection seam (`inject`, total-preserving).
  * `dna`      — DWGSIM(default)/ART: Reference{synthetic|real} -> per-cell FASTA -> C1
                 coverage -> variants.inject -> reads -> bwa/samtools BAM.
  * `rna`      — mutation-aware scRNA (F7b): F3 count matrix (UMI totals conserved) ->
                 variants.inject(observed RNA-VAF) -> alt/ref UMIs -> self-contained 10x-style
                 FASTQ (synthetic transcriptome) or the scReadSim seam (real template).
"""
from .base import ReadEmitter, MissingBinaryError, find_binary, run_binary
from .variants import inject, AlleleSplit
from .dna import (
    Reference, SyntheticReference, RealGenomeReference, REFERENCES,
    build_cell_fasta, effective_alt_fraction, coverage_budget, emit_dna_reads, DNAReadEmitter,
    DwgsimAdapter, ArtAdapter, SIMULATORS, align_to_bam,
)
from .rna import (
    distort_vaf, observed_allele_counts, emit_scrna_reads, ScReadSimAdapter,
    ScrnaAlleleCounts, SyntheticTranscriptome, write_scrna_fastq,
    emit_visium_reads, spot_clone_mixture_vaf,
)

__all__ = [
    "ReadEmitter", "MissingBinaryError", "find_binary", "run_binary",
    "inject", "AlleleSplit",
    "Reference", "SyntheticReference", "RealGenomeReference", "REFERENCES",
    "build_cell_fasta", "effective_alt_fraction", "coverage_budget", "emit_dna_reads",
    "DNAReadEmitter", "DwgsimAdapter", "ArtAdapter", "SIMULATORS", "align_to_bam",
    "distort_vaf", "observed_allele_counts", "emit_scrna_reads", "ScReadSimAdapter",
    "ScrnaAlleleCounts", "SyntheticTranscriptome", "write_scrna_fastq",
    "emit_visium_reads", "spot_clone_mixture_vaf",
]
