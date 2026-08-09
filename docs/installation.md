# Installation

```bash
pip install insilico-cancer-center
```

Graph drawing uses `pygraphviz`, which needs the system **Graphviz** library installed first
(e.g. `brew install graphviz`, `apt-get install graphviz graphviz-dev`, or `conda install -c conda-forge graphviz`).

## From source

```bash
git clone https://github.com/pedrofale/iscc
cd iscc
poetry install
```

## Non-Python dependencies

Growth, sampling and the DNA/RNA/spatial assays need nothing beyond the Python stack. Two features
shell out to external binaries, which iscc finds on `$PATH`:

| Feature | Needs |
|---|---|
| Graph drawing (`pygraphviz`) | `graphviz` |
| FASTQ emission (`emit_dna_reads`) | `dwgsim` (program version ≥ 0.1.13) or `art_illumina` |
| BAM alignment (`emit_dna_reads(emit_bam=True)`) | `bwa` + `samtools` |

`environment.yml` in the repository root declares them all:

```bash
conda env create -f environment.yml
```

Then `poetry install` into that environment for the Python packages. Without these binaries the read
emitter still builds the reference, the per-cell FASTA, the coverage budget and the exact simulator
command — it returns `status="skipped:<tool>"` instead of writing FASTQ, so the rest of the pipeline
is unaffected.

## Command-line tools

Installing the package provides the pipeline entry points:

| Tool | Stage |
|---|---|
| `isccsim` | grow a tumor |
| `isccsample` | biopsy / dissociate into a sample |
| `isccdata` | generate DNA/RNA/spatial assay data |
| `isccfig`, `isccgif` | figures and animations |

Once installed, see the [Overview](overview.md) for the pipeline and a quickstart.
