# Multi-modal integration benchmarks (clonealign + inferCNV)

Two demonstrations that run a **real downstream integration method** on `iscc`-simulated data and
score it against `iscc`'s known ground truth (the scMultiSim / SISTEM "we provide the ground truth"
convention; pairs with the PEtracer section of the paper).

- **clonealign** (`validate_clonealign.py`) — DNA↔RNA clone-of-origin assignment. Scores accuracy /
  AUC vs the true clone, and shows the assignment is driven by the *emergent* copy-number → expression
  dosage coupling (accuracy rises with the fraction of CN-informative genes).
- **inferCNV** (`validate_infercnv.py`) — copy-number-from-expression. Scores malignant-vs-normal
  separation and the recovery of the clonal CNA structure vs the true per-cell copy number.

**Why it's non-circular:** the CN→expression coupling both methods invert is *not* imposed — it
emerges from the engine's per-allele dosage model (`CancerCell.get_exp`) and selection. A bolt-on
simulator would have to hand-impose exactly the dosage law the method assumes.

## Architecture

Each external tool is heavy and dependency-conflicting, so it lives in its **own dedicated conda
env**, and the core `iscc` env stays clean:

```
iscc            data generation (grow tumour, scDNA + scRNA) + scoring + figures
  │  writes inputs to a temp dir, subprocess ↓, reads results back
  ├── iscc-clonealign   R + TensorFlow + kieranrcampbell/clonealign   (clonealign_runner.R)
  └── iscc-infercnv     infercnvpy (scverse)                          (infercnv_runner.py)
```

`integration_common.py` holds the shared data generation and the run/score helpers. The validation
scripts and `tests/test_integration.py` skip gracefully when a dedicated env is absent (override the
interpreters with `ISCC_CLONEALIGN_RSCRIPT` / `ISCC_INFERCNV_PYTHON`).

## Building the dedicated envs

```bash
# --- inferCNV (Python) ---
conda create -y -n iscc-infercnv -c conda-forge python=3.10
~/miniconda3/envs/iscc-infercnv/bin/pip install infercnvpy scanpy anndata

# --- clonealign (R + TensorFlow) ---
conda create -y -n iscc-clonealign -c conda-forge python=3.10 \
  r-base r-remotes r-biocmanager r-reticulate \
  r-glue r-dplyr r-ggplot2 r-matrixstats r-r.utils r-progress r-jsonlite
# Bioconductor + R-tensorflow as precompiled binaries (avoids source compilation):
conda install -y -n iscc-clonealign -c conda-forge -c bioconda \
  bioconductor-summarizedexperiment r-tensorflow r-tfruns
# TensorFlow + tensorflow_probability (clonealign uses tf.compat.v1 graph mode):
~/miniconda3/envs/iscc-clonealign/bin/pip install "tensorflow==2.15.0" "tensorflow-probability==0.23.0"
# the clonealign package itself:
RETICULATE_PYTHON=~/miniconda3/envs/iscc-clonealign/bin/python \
  ~/miniconda3/envs/iscc-clonealign/bin/R --no-echo \
  -e 'remotes::install_github("kieranrcampbell/clonealign", upgrade="never", dependencies=FALSE)'
```

`clonealign_runner.R` points `reticulate` at its own env's Python (where `tensorflow` /
`tensorflow_probability` live), so no extra configuration is needed.

## Running

```bash
python -u validation/validate_clonealign.py   # -> manuscript/figures/validation_clonealign.png
python -u validation/validate_infercnv.py      # -> manuscript/figures/validation_infercnv.png
```
