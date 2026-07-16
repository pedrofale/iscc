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

**Current envs:** `iscc-clonealign`, `iscc-infercnv`, `iscc-scdef`, `iscc-cnmf`, `iscc-mhn`,
`iscc-treemhn`, `iscc-cell2location`, `iscc-rctd`, `iscc-numbat`.
**Planned (build the same way when the benchmark lands):** cohort/batch integration — `iscc-scvi`
(scvi-tools / scANVI), `iscc-harmony` (harmonypy); demultiplexing — `iscc-demux` (vireo / souporcell).
Self-contained validations (pure numpy/scipy, e.g. the multi-region NJ trees) need **no** extra env.

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

# --- scDEF (Python + jax) — the flagship program-inference tool. PINNED: see the note below ---
conda create -y -n iscc-scdef -c conda-forge python=3.10
~/miniconda3/envs/iscc-scdef/bin/pip install "scdef==0.6.1"     # pulls jax / scanpy / anndata
```

!!! note "`iscc-scdef` is pinned to scDEF **0.6.1** — keep it that way"
    **The benchmark's scDEF version of record is 0.6.1** (PyPI), verified with
    `python -c "import scdef; print(scdef.__version__)"` → `0.6.1`, loading from the env's own
    `site-packages`.

    This is deliberate, and the earlier state is worth remembering as a cautionary tale: the env was
    first built by cloning a local `scdef` env, which turned out to be an **editable install pointing
    at the working checkout** `~/projects/scDEF/src`. It reported version **0.4.8** (both
    `__version__` and the dist metadata) while executing the checkout's **0.6.1** source — so the
    version a paper would have recorded was not the code that ran, and any edit to that checkout
    would have silently changed iscc's published numbers. scDEF was also mid-revision at the time,
    with a model fix between those versions, so the discrepancy was not cosmetic.

    Do not rebuild this env by cloning a development env. Pin the version, and update the number here
    **and in the manuscript Methods** together if it ever changes. Override the interpreter with
    `ISCC_SCDEF_PYTHON` for local experiments only.

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

### Spatial deconvolution (cell2location / RCTD) — the flagship, R13

`validate_deconvolution.py` (+ `deconv_common.py`, `cell2location_runner.py`, `rctd_runner.R`) grows ONE
four-type tumour (cancer / epithelial / stromal / immune, made transcriptionally distinct by the R13
program layer), emits a **Visium section** and an **scRNA reference from a SEPARATE biopsy of the SAME
tumour** (F1 biopsy → F2 dissociation → F3 assay), and scores per-spot composition against iscc's true
per-spot composition. The headline is the **matched-vs-mismatched reference decomposition**: an oracle
reference (the exact section cells) is the ceiling, and the cost of a realistic reference is attributed
to regional mismatch / dissociation / assay — all with ground truth iscc uniquely has. Both tools are
optional (the script prints a note and carries on); override with `ISCC_CELL2LOCATION_PYTHON` /
`ISCC_RCTD_RSCRIPT`.

```bash
# --- cell2location (Python; scvi-tools / pyro / torch, CPU is fine for this small benchmark) ---
conda create -y -n iscc-cell2location -c conda-forge python=3.10
~/miniconda3/envs/iscc-cell2location/bin/pip install cell2location scanpy anndata

# --- RCTD / spacexr (R). spacexr is GitHub-only and compiles C++, so install its CRAN deps as
#     PRECOMPILED conda-forge packages, then compile spacexr itself inside the ACTIVATED env (a bare
#     `Rscript -e install.packages(...)` cannot find the conda toolchain -> "C compiler cannot create
#     executables" / make Error 127). ---
conda create -y -n iscc-rctd -c conda-forge r-base=4.3 compilers make
conda install -y -n iscc-rctd -c conda-forge \
  r-matrix r-doparallel r-foreach r-quadprog r-rcpp r-rcppeigen r-dplyr r-reshape2 r-ggplot2 \
  r-pals r-data.table r-mgcv r-irlba r-fields r-readr r-knitr r-rmarkdown r-rfast r-locfdr \
  r-metafor r-compquadform r-tibble r-plyr
git clone --depth 1 https://github.com/dmcable/spacexr.git /tmp/spacexr
source ~/miniconda3/etc/profile.d/conda.sh && conda activate iscc-rctd
R CMD INSTALL --no-multiarch /tmp/spacexr      # compiles against the conda-forge deps
```

### Numbat (allele-aware CNA-from-expression) — R13

`validate_numbat.py` (+ the Numbat section of `integration_common.py`, `numbat_runner.R`) reuses the
multi-clone tumour with the R13 **allele-resolved** expression layer ON, and runs Numbat **head-to-head
against inferCNV** on the SAME cells: the clean question is whether the allele layer helps over
expression-only CNA calling. **Input-interface note (the scoped main risk):** Numbat normally builds its
allele table from a cellsnp-lite pileup + a population phasing panel — machinery an abstract genome lacks.
We instead feed Numbat the allele counts DIRECTLY: iscc tracks the two homologs, so it knows the true
phase (each gene is one phased marker, `GT="1|0"`, ALT = m-homolog reads drawn at UMI depth from
`cell_rna_baf`), and iscc's segments map onto Numbat chromosomes via a custom `gtf`. iscc's homolog
labels ARE the ground-truth phasing a real panel only approximates. Override with `ISCC_NUMBAT_RSCRIPT`.

```bash
# --- Numbat (R + Bioconductor). Same rule as RCTD: install everything available from conda-forge /
#     bioconda PRECOMPILED, then compile the three source-only C++ packages inside the ACTIVATED env.
#     r-paralleldist is CRAN-only (NOT on conda-forge), so it joins the source-compiled set. ---
conda create -y -n iscc-numbat -c conda-forge r-base r-remotes r-biocmanager compilers make
conda install -y -n iscc-numbat -c conda-forge -c bioconda \
  r-ape r-phangorn r-vegan r-ggraph r-tidygraph r-graphlayouts r-igraph r-vcfr r-rcppparallel \
  r-rhpcblasctl r-roptim r-reshape2 r-plyr r-ggforce r-ggrepel r-memoise r-cachem r-fastmap r-mass \
  r-nlme r-mgcv r-digest r-fastmatch r-quadprog r-catools r-data.table r-dplyr r-tidyr r-stringr \
  r-glue r-purrr r-zoo r-matrix r-optparse r-scales r-tibble r-magrittr r-r.utils r-logger \
  bioconductor-ggtree bioconductor-genomicranges bioconductor-iranges
source ~/miniconda3/etc/profile.d/conda.sh && conda activate iscc-numbat
R -e 'install.packages(c("parallelDist","scistreer","hahmmr","numbat"), repos="https://cloud.r-project.org", dependencies=FALSE)'
```

```bash
~/miniconda3/envs/iscc/bin/python validation/validate_deconvolution.py --quick   # smoke (RCTD only)
~/miniconda3/envs/iscc/bin/python validation/validate_deconvolution.py           # the paper figure
~/miniconda3/envs/iscc/bin/python validation/validate_numbat.py --quick          # smoke
~/miniconda3/envs/iscc/bin/python validation/validate_numbat.py                  # the paper figure
```

## Running

```bash
python -u validation/validate_clonealign.py   # -> manuscript/figures/validation_clonealign.png
python -u validation/validate_infercnv.py      # -> manuscript/figures/validation_infercnv.png
python -u validation/validate_epistasis.py     # -> manuscript/figures/validation_epistasis.png
```

## Cohort progression models (MHN / TreeMHN / CBN / REVOLVER) — R14

`validate_epistasis.py` plants a known event×event network (`DESIGN_epistasis.md`) and scores recovery
against it with the **real tools**, each in its own env. `iscc.integrations.progression` is the seam:
it emits each tool's input shape and scores any tool's output.

| tool | env | input | scored with |
|---|---|---|---|
| **MHN** (Schill et al. 2020) ✅ | `iscc-mhn` | `to_mhn_matrix(tumors, min_freq=…)` — patients × events binary | `score_edges` vs the planted `E` |
| **TreeMHN** (Luo et al. 2023) ✅ | `iscc-treemhn` | `to_treemhn_trees(tumors)` — per-patient mutation trees | `score_edges` + `score_order` |
| CBN / H-CBN ⬜ | `iscc-cbn` | `to_cbn_poset(tumors)` | `score_order` (the conjunction) |
| REVOLVER ⬜ | `iscc-revolver` | `to_treemhn_trees(tumors)` | `score_edges` + `score_order` |

The built-in co-occurrence log-odds (`cooccurrence_scores`, the DISCOVER/MEGSA statistic) is kept as
the **floor** — it cannot separate a direct interaction from one induced through a shared ancestor,
which is exactly what MHN's regularized fit removes. The script SKIPS any tool whose env is absent
(override with `ISCC_MHN_PYTHON` / `ISCC_TREEMHN_RSCRIPT`) and always renders the floor.

`Theta` is **not** `E`: MHN/TreeMHN estimate a *rate* modifier, iscc plants a *fitness/selection*
coefficient. The benchmark therefore scores recovered **edges** (symmetrised by the larger |Theta| of
the two directions — iscc's `E` is symmetric and defines no direction), never values.

### Reading the result before you run it

**Each tool recovers exactly the signal its input encodes.** iscc's `E` acts on fitness (how large the
carrying clones grow); a binary "event present" matrix is saturated by **recurrent mutation** (a
favoured combination arises many times independently, so it is already present at `E=0`) and discards
the frequency the signal lives in.

**TreeMHN does NOT read clone sizes** — a natural assumption, and wrong: `input_tree_df` accepts only
`Patient_ID`/`Tree_ID`/`Node_ID`/`Mutation_ID`/`Parent_ID` and errors on any other column (`weights`
is a per-TREE weight for tree uncertainty). What it adds over MHN is **event ORDER**. Fitness
epistasis produces no order, so TreeMHN is structurally blind to it — and wins decisively once the
planted signal IS an order:

| | FREQUENCY signal (pairwise `E`) | ORDER signal (accessibility DAG) |
|---|---|---|
| co-occurrence floor | 4.20 | 5.40 |
| MHN (presence) | 1.60 (control 2.00 — largely false positives) | 4.00 |
| TreeMHN (tree topology) | 3.40 ≈ chance | **1.80** |

(mean rank of the planted pair of 6; chance 3.5; lower is better.) Recovering fitness epistasis needs
a frequency-aware observable, which neither tool consumes.

Note the two halves need **different event alphabets**: `event_size=2` keeps binary presence from
saturating (panels B/C), but a gated child then reaches only 0–2 of 40 patients, so the DAG panels use
`event_size=8`. Always check the printed child-carrying patient count before reading panel D.

**Caveat:** the cohort tumours are ~130 cells — a clone arising late has no time to expand, so the
absolute recovery rates are a floor for this regime, not an estimate for real cohorts.

**Detection semantics matter** (this was a real bug): `min_freq` in `to_mhn_matrix` is a per-**event
cancer-cell fraction** threshold aggregated ACROSS clones (`event_cell_fractions`), *not* a per-clone
size filter. Filtering clones and OR-ing them calls a 60%-cell-fraction event ABSENT whenever its
carriers are dozens of small lineages — which is precisely what a favoured combination looks like.
`min_clone_freq` (in `clone_events` / `to_mutation_tree`) is the genuinely clone-level filter, for
pruning tree tips.

```bash
# --- MHN (Python) ---
conda create -y -n iscc-mhn -c conda-forge python=3.11
~/miniconda3/envs/iscc-mhn/bin/pip install mhn        # spang-lab/LearnMHN; pulls numpy/pandas/scipy

# --- TreeMHN (R + Rcpp/OpenMP) ---
conda create -y -n iscc-treemhn -c conda-forge r-base=4.3 r-remotes r-matrix r-rcpp \
  r-rcpparmadillo r-igraph r-dplyr r-tidyr r-ggplot2 r-reshape2 r-mass
# TreeMHN's own deps, which install_github does NOT resolve on conda-forge R:
conda install -y -n iscc-treemhn -c conda-forge r-gtools r-diagrammer r-ggm r-ggpubr r-mnormt
# it has compiled code, so the env needs a toolchain:
conda install -y -n iscc-treemhn -c conda-forge compilers make
cat > ~/miniconda3/envs/iscc-treemhn/lib/R/etc/Makevars.site <<'MV'
CC=$(ls ~/miniconda3/envs/iscc-treemhn/bin/*-clang | head -1)
CXX=$(ls ~/miniconda3/envs/iscc-treemhn/bin/*-clang++ | head -1)
CXX11=$CXX
CXX14=$CXX
CXX17=$CXX
MV
# Two upstream patches are needed (both filed under "known install quirks" below):
git clone https://github.com/cbg-ethz/TreeMHN.git && cd TreeMHN
#  1. src/Makevars hardcodes -lstdc++fs (a GCC-only shim); clang/libc++ ships std::filesystem:
printf 'CXX_STD = CXX17\nPKG_CXXFLAGS = -fopenmp $(SHLIB_OPENMP_CXXFLAGS)\nPKG_LIBS = -lgomp $(SHLIB_OPENMP_CFLAGS) $(LAPACK_LIBS) $(BLAS_LIBS) $(FLIBS)\n' > src/Makevars
#  2. build CLEAN -- a stale object tree silently yields a .dylib without the Rcpp symbols
#     ("object '_TreeMHN_get_augmented_trees' not found" at runtime):
~/miniconda3/envs/iscc-treemhn/bin/R CMD INSTALL --preclean --clean .
```

!!! note "TreeMHN input convention"
    Its README: the root node is **its own parent** (`Node_ID = 1`, `Mutation_ID = 0`,
    `Parent_ID = 1`). `Parent_ID = 0` makes `sort_one_tree_df` fail. `to_mutation_tree` emits the
    correct convention; `tests/test_epistasis.py` pins it.
