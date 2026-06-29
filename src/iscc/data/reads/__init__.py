"""Read emission (DESIGN_features §C C2/C3, F7): count/coverage matrices -> FASTQ / BAM.

Per-modality read-realism ADAPTERS over shared seams:
  * `base`     — the `ReadEmitter` protocol + `run_binary` (PATH detect / skip-if-absent).
  * `variants` — the modality-agnostic variant-injection seam (`inject`, total-preserving).
  * `dna`      — DWGSIM(default)/ART: Reference{synthetic|real} -> per-cell FASTA -> C1
                 coverage -> variants.inject -> reads -> bwa/samtools BAM.

A later `rna` adapter (scReadSim) reuses `base` + `variants` unchanged.
"""
from .base import ReadEmitter, MissingBinaryError, find_binary, run_binary
from .variants import inject, AlleleSplit
from .dna import (
    Reference, SyntheticReference, RealGenomeReference, REFERENCES,
    build_cell_fasta, coverage_budget, emit_reads, DNAReadEmitter,
    DwgsimAdapter, ArtAdapter, SIMULATORS, align_to_bam,
)

__all__ = [
    "ReadEmitter", "MissingBinaryError", "find_binary", "run_binary",
    "inject", "AlleleSplit",
    "Reference", "SyntheticReference", "RealGenomeReference", "REFERENCES",
    "build_cell_fasta", "coverage_budget", "emit_reads", "DNAReadEmitter",
    "DwgsimAdapter", "ArtAdapter", "SIMULATORS", "align_to_bam",
]
