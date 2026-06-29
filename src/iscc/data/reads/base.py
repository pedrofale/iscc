"""Shared read-emission seams (DESIGN_features §C C2/C3, §E `reads/base.py`).

Modality-agnostic glue between iscc's count/coverage matrices and the third-party
short-read simulators (DWGSIM / ART) + aligners (bwa / samtools):

  * `ReadEmitter` — the protocol every per-modality adapter implements:
    ``emit(matrix, reference, outdir) -> {"fastq": ..., "bam": ...}``.
  * `find_binary` / `run_binary` — detect an external tool on ``$PATH`` and shell out.
    A missing tool raises `MissingBinaryError` **only when actually asked to emit** — the
    bespoke layers (per-cell FASTA, coverage, variant injection) need no binary, so CI and
    the count-level path stay green without DWGSIM/ART/samtools/bwa installed.

The DNA adapter lives in `reads/dna.py`; a later scRNA adapter (`reads/rna.py`) reuses the
same protocol + `run_binary`, and the shared variant seam in `reads/variants.py`.
"""
from __future__ import annotations

import glob
import os
import shutil
import subprocess
from typing import Protocol, runtime_checkable


class MissingBinaryError(RuntimeError):
    """Raised when an external read tool is required to emit but absent from ``$PATH``."""


@runtime_checkable
class ReadEmitter(Protocol):
    """A per-modality read adapter (DNA: DWGSIM/ART; scRNA later: scReadSim).

    `matrix` is the upstream count/coverage interface (the universal input both DNA and RNA
    read simulators ingest); `reference` is the sequence backend (DNA only); `outdir` is where
    FASTQ (+ BAM) are written. Returns a dict of output paths (at least ``"fastq"``).
    """

    def emit(self, matrix, reference, outdir) -> dict: ...


def find_binary(name: str) -> str | None:
    """Return the absolute path to `name` on ``$PATH``, or ``None`` if not installed."""
    return shutil.which(name)


def run_binary(name, args, cwd=None, capture=True, stdout_path=None, check=True):
    """Run external tool `name` with `args`, raising `MissingBinaryError` if absent.

    This is the single choke-point that touches the shell: detect the binary, build the
    ``[binary, *args]`` command, run it, and surface a clear error. Callers that only build
    the bespoke layers never reach here, so an uninstalled tool is a no-op until emission is
    genuinely requested.

      name         tool name (looked up on $PATH).
      args         list of string arguments (already rendered; numbers -> str by caller).
      cwd          working directory for the subprocess.
      stdout_path  if given, the subprocess stdout is redirected to this file (e.g. SAM).
      check        raise CalledProcessError on a non-zero exit.

    Returns the `subprocess.CompletedProcess`.
    """
    path = find_binary(name)
    if path is None:
        raise MissingBinaryError(
            f"external tool {name!r} not found on $PATH; install it to emit reads/BAM "
            f"(DNA needs dwgsim>=0.1.13 or art_illumina, plus bwa + samtools for BAM). "
            f"The count/coverage matrices and per-cell FASTA do not require it."
        )
    cmd = [path] + [str(a) for a in args]
    if stdout_path is not None:
        with open(stdout_path, "w") as fh:
            return subprocess.run(cmd, cwd=cwd, stdout=fh, check=check)
    return subprocess.run(cmd, cwd=cwd, capture_output=capture, text=True, check=check)


def collect_fastq(prefix_or_dir):
    """Collect FASTQ files produced under a prefix or directory (sorted, gz-aware)."""
    pats = []
    if os.path.isdir(prefix_or_dir):
        base = os.path.join(prefix_or_dir, "*")
    else:
        base = prefix_or_dir + "*"
    for ext in (".fastq", ".fq", ".fastq.gz", ".fq.gz"):
        pats.extend(glob.glob(base + ext))
    return sorted(set(pats))
