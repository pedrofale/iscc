# iscc tutorials

The notebooks published under **Tutorials** on the docs site.
`docs/tutorials/*.ipynb` are symlinks to this folder, so editing here edits the site.
Every notebook is committed with its outputs; re-execute one with `jupyter nbconvert --execute --inplace`.

## Using iscc

- **`01_quickstart.ipynb`**: one tumor and its four data sets in a few calls, and the ground truth (clones, genealogy, copy number, cell types, niches).
- **`02_shape_the_tumor.ipynb`**: what each part of the configuration does to the tumor (selection, mutation, tissue architecture, niche selection, the metastatic site, treatment), shown by contrasting runs.
- **`03_shape_the_data.ipynb`**: the assays and their technology presets, the measurement biases, fitting the assay parameters to a real data set, and batches.
- **`04_cohort.ipynb`**: many patients over one configuration, pooled into sequencing batches, with the ground truth across patients.
- **`05_simulated_next_to_real.ipynb`**: the data of `iscc` next to the real data they stand for, a breast tumor in scRNA-seq (Wu et al. 2021) and a breast cancer section in Visium, by cell type, marker genes, composition and depth.

## Analysis tools

One notebook per main use case, each running one standard tool as its documentation shows and scoring it against the ground truth.
Each tool runs in its own conda environment (`iscc-<tool>`), never in the environment of `iscc`.

- **`tool_ascat_pyclonevi.ipynb`**: purity, ploidy and allele-specific copy number from bulk WGS with ASCAT, then clones with PyClone-VI.
- **`tool_hmmcopy.ipynb`**: copy number from single-cell DNA with HMMcopy at the settings of the DLP+ pipeline.
- **`tool_clonealign.ipynb`**: matching scRNA cells to the clones of the scDNA with clonealign.
- **`tool_scrna_cnv.ipynb`**: copy number and malignancy from scRNA with inferCNV (infercnvpy).
- **`tool_rctd.ipynb`**: cell-type deconvolution of Visium spots with RCTD.
- **`tool_integration.ipynb`**: batch integration of a cohort with Harmony and scVI.

## Configuration

- **`tutorial_config.yaml`**: the configuration the tutorials grow from, the paper's regime at its sweep scale (a 96 by 96 field) with its immune variant (T and NK, myeloid and B and plasma cells in the stroma, the T and NK cells killing) and a stroma of fibroblasts, endothelial cells and perivascular cells, without the metastatic site or treatment.
- **`example_config.yaml`**: the base configuration of the paper's regime, which `validation/realistic_regime.py` reads.

`archive/` holds the notebooks of earlier releases, which are no longer published.
