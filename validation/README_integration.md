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

## Convention — one dedicated env per external tool (applies to ALL validations)

**Any `iscc` validation that shells out to an external tool MUST run that tool in its own dedicated
`iscc-<tool>` conda env — never install it into the core `iscc` env.** The core env stays limited to
`iscc` + its scientific stack; heavy or dependency-conflicting tools (R packages, TensorFlow,
scvi-tools, …) are isolated so a broken or conflicting install can never poison the simulator itself,
and each benchmark is reproducible from a pinned env.

The pattern for each tool (see `integration_common.py`):

1. A module-level interpreter path, env-var-overridable:
   `<TOOL> = os.environ.get("ISCC_<TOOL>_PYTHON", "~/miniconda3/envs/iscc-<tool>/bin/python")`
   (or `.../bin/Rscript`).
2. A `<tool>_available()` guard so the validation **and its test** SKIP gracefully when the env is absent.
3. A thin runner script (`<tool>_runner.{py,R}`) executed via `subprocess`; data crosses the env
   boundary as files (CSV / AnnData) in a temp dir. The core-env script never imports the external tool.

Document each new env's build recipe in the "Building the dedicated envs" section below.

**Current envs:** `iscc-clonealign`, `iscc-infercnv`, `iscc-scdef`, `iscc-cnmf`.
**Planned (build the same way when the benchmark lands):** cohort/batch integration — `iscc-scvi`
(scvi-tools / scANVI), `iscc-harmony` (harmonypy); demultiplexing — `iscc-demux` (vireo / souporcell);
spatial deconvolution — `iscc-cell2location` / `iscc-tangram`. Self-contained validations (pure
numpy/scipy, e.g. the multi-region NJ trees) need **no** extra env.

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

```bash
# --- cNMF (Python) — the flat GEP comparator for the R13 program benchmark ---
conda create -y -n iscc-cnmf -c conda-forge python=3.10
~/miniconda3/envs/iscc-cnmf/bin/pip install cnmf          # pulls scanpy/anndata/scikit-learn

# --- scDEF (Python + jax) — the flagship program-inference tool ---
# Built by cloning an existing working scDEF env (fastest reliable route on this machine):
conda create -y -n iscc-scdef --clone scdef
# From scratch instead:
#   conda create -y -n iscc-scdef -c conda-forge python=3.10
#   ~/miniconda3/envs/iscc-scdef/bin/pip install scdef
```

!!! warning "`iscc-scdef` is NOT pinned — resolve before submission"
    The cloned env installs scDEF **editable**, pointing at the working checkout at
    `~/projects/scDEF/src`. Three consequences, all bad for a reproducible benchmark:

    * `scdef.__version__` and the dist metadata report **0.4.8**, while the checkout's
      `pyproject.toml` declares **0.6.1** — so the version a paper would record is not the code that
      ran.
    * Editing that checkout silently changes iscc's benchmark numbers.
    * scDEF was under review while this benchmark was written, and its authors' response letter notes
      that they *"identified issues with the initial version of our model… fixed these problems, and
      re-ran all the analyses"* — so 0.4.8 vs 0.6.1 may straddle a model fix.

    Before publishing, decide which scDEF is the benchmark target and pin it
    (`pip install scdef==<version>` into a fresh `iscc-scdef`), then record that version in the
    manuscript. Override the interpreter meanwhile with `ISCC_SCDEF_PYTHON`.

`clonealign_runner.R` points `reticulate` at its own env's Python (where `tensorflow` /
`tensorflow_probability` live), so no extra configuration is needed.

### Gene programs (scDEF / cNMF) — R13

`validate_programs.py` (+ `programs_common.py`, `scdef_runner.py`, `cnmf_runner.py`) scores program
recovery against iscc's true `loading` / per-cell `z` across an SNV/CNA-burden sweep. Both tools are
optional — the script prints `SKIPPING (env absent)` and carries on. Override the interpreters with
`ISCC_SCDEF_PYTHON` / `ISCC_CNMF_PYTHON`.

```bash
~/miniconda3/envs/iscc/bin/python validation/validate_programs.py --quick   # smoke test
~/miniconda3/envs/iscc/bin/python validation/validate_programs.py --reps 3  # the paper figure
```

## Running

```bash
python -u validation/validate_clonealign.py   # -> manuscript/figures/validation_clonealign.png
python -u validation/validate_infercnv.py      # -> manuscript/figures/validation_infercnv.png
python -u validation/validate_epistasis.py     # -> manuscript/figures/validation_epistasis.png
```

## Cohort progression models (MHN / TreeMHN / CBN / REVOLVER) — R14

`validate_epistasis.py` plants a known event×event network (`DESIGN_epistasis.md`) and scores recovery
against it. It runs **core-env-only today**: the "method" is the built-in pairwise log-odds baseline
(`iscc.integrations.cooccurrence_scores` — the DISCOVER/MEGSA statistic), so the figure renders without
R. That baseline is the **floor of the benchmark, not a stand-in for MHN**: it cannot separate a direct
interaction from one induced through a shared ancestor, which is exactly what MHN's regularized fit is
built to remove.

Wiring the real tools in is deliberately cheap, because the seam already exists —
`iscc.integrations.progression` emits each tool's input shape and scores any tool's output:

| tool | env | input (already emitted) | scored with |
|---|---|---|---|
| MHN | `iscc-mhn` | `to_mhn_matrix(tumors)` — patients × events binary | `score_edges` (vs the planted `E`) |
| TreeMHN | `iscc-treemhn` | `to_treemhn_trees(tumors)` — per-patient mutation trees (`Patient_ID`, `Node_ID`, `Mutation_ID`, `Parent_ID`) | `score_edges` + `score_order` |
| CBN / H-CBN | `iscc-cbn` | `to_cbn_poset(tumors)` | `score_order` (the conjunction) |
| REVOLVER | `iscc-revolver` | `to_treemhn_trees(tumors)` | `score_edges` + `score_order` |

Follow the one-env-per-tool convention above: add `<TOOL> = os.environ.get("ISCC_<TOOL>_RSCRIPT", ...)`,
a `<tool>_available()` skip guard, and a thin `<tool>_runner.R` that reads the emitted CSV and writes
back an edge/order list.

**Read the result carefully before running anything.** The benchmark's finding is that iscc's pairwise
`E` acts on *fitness* while MHN/CBN model the *rate of acquisition*, so a cross-sectional matrix is
near-blind to it — recovery sits at chance for any cohort size or interaction strength, while
conjunctive constraints under `gating_mode: accessibility` are recovered perfectly. A real MHN run is
expected to reproduce that contrast, not overturn it; if it does overturn it, that is the interesting
result and worth chasing.
