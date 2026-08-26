# iscc example notebooks

The notebooks published under **Tutorials** on the docs site, in three groups: how the simulator
works, what data it emits, and what real analysis tools do with that data.

`docs/tutorials/*.ipynb` are symlinks to this folder, so editing here is editing the site. Every
notebook is committed **with its outputs** — re-execute with `jupyter nbconvert --execute --inplace`
and then run `python scripts/clean_notebook_outputs.py notebooks/*.ipynb` to drop progress-bar noise.

## Overview

- **`01_pipeline_walkthrough.ipynb`** — one tumour, end to end: grow → sample → assay → inspect,
  with the Python API and the equivalent CLI (`isccsim`/`isccsample`/`isccdata`).

## Tumor evolution

- **`02_tumor_growth.ipynb`** — the tumour itself: spatial structure, the CINner-style selection
  model, clonal dynamics.
- **`base_simulation.ipynb`** — the shared multi-focal DCIS→IDC lesion on a ductal field that the
  science notebooks are built on, and the ground-truth matrices each one reads.
- **`compartment_selection_confound.ipynb`** — the DCIS→IDC transition (a lumen founder confined by
  the gland wall + stroma until escape traits evolve) and the **genetic-vs-niche** expression
  confound.
- **`treatment_escape_modes.ipynb`** — the four routes to therapy escape (Kane & Maley), all
  reproduced on one treatment model.
- **`metastasis.ipynb`** — seeding, dormancy and outgrowth at a distant site.

## Data generation

Per-modality demos of the assay layer's realism (counts/coverage, and how parameters are fit to real
references). Generated from the matching `_build_*.py` scripts.

- **`assay_dna.ipynb`** — bulk + single-cell DNA: copy-number-dependent and GC-biased coverage, het
  allele-fraction spread with allelic dropout, `estimate_dna` round-trips.
- **`assay_scrna.ipynb`** — scRNA counts: variable library size, negative-binomial overdispersion,
  dropout, batch/technical effects.
- **`assay_spatial.ipynb`** — 10x Visium: per-spot aggregation, spatial autocorrelation,
  `estimate_visium` fit to a real section.
- **`reads.ipynb`** — read-level output: per-cell reference → coverage budget → the shared variant
  seam → FASTQ/BAM (DWGSIM/ART), degrading gracefully without the binaries.

## Data analysis examples

Each of these runs a **real, published tool** on `iscc` output and scores it against ground truth the
tool never sees. Two rules hold throughout:

1. **No simulation in the notebook.** The datasets are generated once by
   `python validation/make_analysis_data.py` into `analysis_data/<tool>/` (git-ignored). The
   notebook loads files; `r_preamble.R`'s `analysis_dir()` errors with the regeneration command if
   they are missing.
2. **No subprocesses.** Each notebook runs in the tool's *own* kernel — `R (iscc-clonealign)`,
   `Python (iscc-scdef)`, and so on — so the fitted object is inspectable and the tool's own
   plotting functions work. Plots come from the tool wherever the tool ships one; hand-drawn figures
   are only for scoring against `iscc` truth, which no tool ships a plotter for.

- **`analysis_ground_truth.ipynb`** — the shared analysis dataset and everything `iscc` knows about
  it, including the true clone phylogeny. The other notebooks reference it.
- **`tool_clonealign_R.ipynb`** — clonealign: assign scRNA cells to scDNA copy-number clones.
- **`tool_numbat_R.ipynb`** — Numbat: call copy number from scRNA with an allele layer.
- **`tool_hmmcopy_R.ipynb`** — HMMcopy: call copy number from single-cell read depth; feeds clonealign.
- **`tool_scite_trees.ipynb`** — SCITE: reconstruct the mutation tree from single-cell genotypes.
- **`tool_pyclonevi.ipynb`** — PyClone-VI: clonal clusters and cancer-cell fractions from bulk DNA.
- **`tool_rctd_R.ipynb`** — RCTD: deconvolve Visium spots against an scRNA reference.
- **`tool_treemhn_R.ipynb`** — TreeMHN: recover planted precedence constraints from mutation trees.
- **`tool_mhn_bulk.ipynb`** — MHN: the same cohort read cross-sectionally, order discarded.
- **`tool_scdef_cohort.ipynb`** — scDEF: recover programs shared across a five-patient cohort
  despite batch effects.

## Shared example data

`01` and `02` read a small shared dataset under `example_out/` (git-ignored), produced by:

```bash
python notebooks/generate_example.py          # ~2s; skips if already present
python notebooks/generate_example.py --force  # regenerate
```

The notebooks regenerate it automatically on first run if it is missing. `example_config.yaml` is
the small tumour config they share.

The production-tool benchmarks these examples point to live under `validation/`.
