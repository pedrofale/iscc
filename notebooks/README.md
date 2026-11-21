# iscc tutorials

The notebooks published under **Tutorials** on the docs site.
`docs/tutorials/*.ipynb` are symlinks to this folder, so editing here edits the site.
Every notebook is committed with its outputs; re-execute one with `jupyter nbconvert --execute --inplace`.

## Using iscc

- **`01_quickstart.ipynb`**: one tumor and its four data sets in a few calls, and the answer key (clones, genealogy, copy number, cell types, niches).
- **`02_shape_the_tumor.ipynb`**: what each part of the configuration does to the tumor (selection, mutation, tissue architecture, niche selection, the metastatic site, treatment), shown by contrasting runs.
- **`03_shape_the_data.ipynb`**: the assays and their technology presets, the measurement biases, fitting the assay parameters to a real data set, and batches.
- **`04_cohort.ipynb`**: many patients over one configuration, pooled into sequencing batches, with the answer key across patients.

## Analysis tools

One notebook per main use case, each running one standard tool as its documentation shows and scoring it against the answer key.
Each tool runs in its own conda environment (`iscc-<tool>`), never in the environment of `iscc`.

- **`tool_scrna_cnv.ipynb`**: copy number and malignancy from scRNA with inferCNV (infercnvpy).
- **`tool_rctd.ipynb`**: cell-type deconvolution of Visium spots with RCTD.
- **`tool_integration.ipynb`**: batch integration of a cohort with Harmony and scVI.

## Configuration

- **`tutorial_config.yaml`**: the configuration the tutorials grow from, the paper's regime at its sweep scale (a 96 by 96 field) without the metastatic site or treatment.
- **`example_config.yaml`**: the base configuration of the paper's regime, which `validation/realistic_regime.py` reads.

`archive/` holds the notebooks of earlier releases, which are no longer published.
