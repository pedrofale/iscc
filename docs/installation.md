# Installation

```bash
pip install insilico-cancer-center
```

`iscc` needs Python 3.10, 3.11 or 3.12.

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

## Running the tutorials

The tutorials are notebooks in the `notebooks` folder of the repository, next to
`tutorial_config.yaml`, the configuration they grow from. Clone the repository, or download a notebook
and the configuration into one folder, and install the packages that the tutorials use for analysis
and plots:

```bash
pip install scanpy squidpy jupyterlab
```

## Environments for the analysis-tool tutorials

Each [analysis-tool tutorial](tutorials/tool_ascat_pyclonevi.ipynb) runs its tool from its own conda
environment, never from the environment of `iscc`. The commands below create each environment with
the version of the tool that the published tutorials ran. The notebooks call the tools at
`~/miniconda3/envs/<environment>`, so change that path in a notebook if conda is installed elsewhere.

```bash
# ASCAT 3.2.0 and PyClone-VI 0.2.0 (bulk WGS)
conda create -n iscc-ascat -c conda-forge -c bioconda r-base=4.5 r-remotes r-data.table r-doparallel \
    r-foreach r-rcolorbrewer bioconductor-genomicranges bioconductor-iranges bioconductor-s4vectors
conda run -n iscc-ascat Rscript -e 'remotes::install_github("VanLoo-lab/ascat/ASCAT")'
conda create -n iscc-pyclonevi -c conda-forge -c bioconda pyclone-vi=0.2.0

# HMMcopy 1.52 (single-cell DNA)
conda create -n iscc-hmmcopy -c conda-forge -c bioconda bioconductor-hmmcopy r-data.table r-kernsmooth

# clonealign 1.99.3 on TensorFlow 2.15 (scRNA cells to scDNA clones)
conda create -n iscc-clonealign -c conda-forge -c bioconda python=3.10 r-base r-remotes r-reticulate \
    r-tensorflow r-glue r-dplyr r-tidyr r-ggplot2 r-cowplot r-matrixstats r-r.utils r-progress \
    bioconductor-singlecellexperiment
conda run -n iscc-clonealign pip install tensorflow==2.15.0 tensorflow-probability==0.23.0
conda run -n iscc-clonealign Rscript -e 'remotes::install_github("kieranrcampbell/clonealign")'

# inferCNV through infercnvpy 0.6.1 (copy number from scRNA)
conda create -n iscc-infercnv -c conda-forge python=3.10
conda run -n iscc-infercnv pip install infercnvpy==0.6.1

# RCTD through spacexr 2.2.1 (Visium deconvolution)
conda create -n iscc-rctd -c conda-forge r-base=4.3 compilers make r-remotes r-matrix r-doparallel \
    r-foreach r-quadprog r-rcpp r-rcppeigen r-dplyr r-reshape2 r-ggplot2 r-pals r-data.table r-mgcv \
    r-fields r-readr r-knitr r-rmarkdown r-rfast r-locfdr r-metafor r-compquadform r-tibble
conda run -n iscc-rctd Rscript -e 'remotes::install_github("dmcable/spacexr", build_vignettes = FALSE)'

# Harmony and scVI (batch integration)
conda create -n iscc-integration -c conda-forge python=3.11
conda run -n iscc-integration pip install scanpy==1.11.5 harmonypy==0.0.10 scvi-tools==1.4.2
```

Once installed, see the [Overview](overview.md) for the pipeline and a quickstart.
