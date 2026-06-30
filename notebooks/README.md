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

## Backlog (analysis demos — not yet implemented)

These are title-only stubs, kept as a roadmap. They depend on the data-realism milestone and
will be filled in afterwards:

- `dna_mhn.ipynb` — Mutual Hazard Networks from DNA-seq
- `visium_niches.ipynb` — niche identification from Visium
- `scrna_batch_effects.ipynb` — technical/biological batch effects
- `combining_scdna_scrna.ipynb` — joint scDNA + scRNA analysis
- `real_data_comparison.ipynb` — simulated vs real data

> The legacy `tumor_growth.ipynb` and `data_overview.ipynb` (old `tumorevo` API) are superseded
> by `02_`/`03_` and can be removed.
