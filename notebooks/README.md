# iscc example notebooks

A guided tour of the iscc pipeline. Run them in order; each is self-contained and executes
top-to-bottom in the `iscc` environment.

## Spine notebooks

1. **`01_pipeline_walkthrough.ipynb`** — how to use iscc end-to-end: grow a tumor → sample it →
   run assays, with the Python API and the equivalent CLI (`isccsim`/`isccsample`/`isccdata`).
2. **`02_tumor_growth.ipynb`** — explore the *tumor*: spatial structure, the CINner-style
   selection model, clonal dynamics, and how treatment acts on cells.
3. **`03_data_overview.ipynb`** — explore the *data*: bulk DNA, single-cell DNA, scRNA (scanpy),
   and Visium spatial transcriptomics (squidpy).

## Assay deep-dives

Per-modality demos of the data module's realism (counts/coverage and how parameters are fit to
real references). They are generated from the matching `_build_*.py` scripts (re-run the script,
then execute the notebook):

- **`assay_dna.ipynb`** — bulk + single-cell DNA: copy-number-dependent and GC-biased coverage,
  het allele-fraction spread with allelic dropout, and `estimate_dna` round-trips.
- **`assay_scrna.ipynb`** — scRNA counts: variable library size, negative-binomial
  overdispersion, dropout, and batch/technical effects.
- **`assay_spatial.ipynb`** — 10x Visium: per-spot aggregation, spatial autocorrelation, and
  `estimate_visium` fit to a real section.
- **`reads.ipynb`** — read-level output: per-cell reference → coverage budget → the shared
  variant seam → FASTQ/BAM (DWGSIM/ART), degrading gracefully without the binaries.

## Shared example data

`02` and `03` read a small shared dataset under `example_out/` (git-ignored). It is produced by:

```bash
python notebooks/generate_example.py          # ~2s; skips if already present
python notebooks/generate_example.py --force   # regenerate
```

The notebooks regenerate it automatically on first run if it's missing.
`example_config.yaml` is the small tumor config they all use.

## Science showcase (one shared simulation)

Where the spine + assay notebooks cover the *mechanics*, these show the *science*: what each modality
lets you recover, scored against ground truth. They are all built on **one** spatially-structured
tumour — a breast **DCIS** duct (~10k cancer cells filling a lumen inside an epithelial ring, with
fitness-gated microinvasion; WGD + allele-specific expression + phenotype-coupled gene programs on),
re-grown deterministically by the importable helper `base_sim.py` — so every notebook shares the
identical substrate. Start with `base_simulation`.

- **`base_simulation.ipynb`** — shows the shared tumour: the duct + microinvasion, the distributed
  fitness gradient, the programs tracking their drivers (proliferation↔division, EMT↔dispersal,
  hypoxia↔core), and the ground-truth matrices each notebook reads.
- **`combining_scdna_scrna.ipynb`** — scDNA + scRNA from the same mixed tumour: reconstruct subclones
  from expression via DNA-defined copy-number profiles (the clonealign idea), scored vs truth.
- **`wgd_allele_cna.ipynb`** — whole-genome doubling + the allele layer: the ploidy signature, why
  total copy number misses a doubling, and the allele-only-detectable states (`cell_rna_baf`).
- **`gene_programs.ipynb`** — expression programs vs the contiguous-CNA confound; NMF recovery on the
  mixture with the positional-clustering diagnostic.
- **`tree_inference_dna.ipynb`** — the clonal phylogeny from bulk DNA-seq and from scDNA-seq, scored
  against iscc's true clone tree.
- **`scrna_visium_integration.ipynb`** — deconvolve Visium spots with an scRNA reference (NNLS),
  scored against the true per-spot composition.
- **`cohort_shared_programs.ipynb`** — a 5-patient cohort sharing one landscape: recover the shared
  gene programs across batches (scDEF, or an in-core fallback).
- **`cohort_mhn_recurrence.ipynb`** — recurrent mutations and their interactions/order across the
  cohort (MHN + TreeMHN, or an in-core fallback).

The production-tool benchmarks these point to live under `validation/`.

## Backlog (analysis demos — not yet implemented)

These are title-only stubs, kept as a roadmap:

- `dna_mhn.ipynb` — Mutual Hazard Networks from DNA-seq (see `cohort_mhn_recurrence.ipynb`)
- `visium_niches.ipynb` — niche identification from Visium
- `scrna_batch_effects.ipynb` — technical/biological batch effects
- `real_data_comparison.ipynb` — simulated vs real data

> The legacy `tumor_growth.ipynb` and `data_overview.ipynb` (old `tumorevo` API) are superseded
> by `02_`/`03_` and can be removed.
