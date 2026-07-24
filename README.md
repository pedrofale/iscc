# iscc — in silico cancer center

[![Tests](https://github.com/pedrofale/iscc/actions/workflows/main.yaml/badge.svg)](https://github.com/pedrofale/iscc/actions/workflows/main.yaml)
[![PyPI version](https://img.shields.io/pypi/v/insilico-cancer-center)](https://pypi.org/project/insilico-cancer-center/)
[![Python versions](https://img.shields.io/pypi/pyversions/insilico-cancer-center)](https://pypi.org/project/insilico-cancer-center/)

Simulate tumor growth, treatment, and multi-modal molecular data generation under different
evolutionary models. `iscc` aims to be a standard data simulator for computational
tumor-evolution methods: grow a spatially structured tumor under a CINner-style
copy-number/driver selection model, optionally treat it, sample it (biopsy/dissociation), and
generate single-cell and bulk DNA/RNA and spatial (Visium-like) data — all with shared
ground truth.

`iscc` is the rename and expansion of the earlier `tumorevo` package.

📖 **Documentation & tutorials:** https://pedrofale.github.io/iscc/ — built from `docs/` with
MkDocs Material. Build locally with `pip install -r docs/requirements.txt && mkdocs serve`.

## Installation

```bash
$ pip install insilico-cancer-center    # installs the `iscc` package; or: poetry install
```

## Pipeline

`iscc` is a four-stage pipeline, each stage a command-line tool:

| Stage | CLI | Does |
|---|---|---|
| 1. Grow | `isccsim` | grow a tumor; write ground-truth genotypes/CNVs/expression/spatial state |
| 2. Sample | `isccsample` | biopsy / dissociate the tumor into a sample of cells |
| 3. Assay | `isccdata` | generate single-cell & bulk DNA/RNA and spatial-transcriptomics data |
| viz | `isccfig`, `isccgif` | Muller plot, 2D slice, clone tree, and growth animations |

### 1. Simulate tumor growth

The default engine is the fast genotype-level (count-based) `genotype` model; a cell-level
`glandular` engine is also available. Parameters come from a YAML config:

```bash
$ isccsim --sim-config config.yaml --steps 2000 --random-seed 0 -o sim_out
```

The shipped defaults produce a realistic multi-clone tumour. **See `PARAMETERS.md`** for each knob's
default, valid range, and what going out of range does — plus `tumor.diagnose()`, which flags a
degenerate tumour after growth and tells you which knob to turn.

This writes (see `SCHEMA.md` for the full layout): `cell_data/` (per-cell ground truth),
`trace_counts.csv`, `parents.csv`, `genotypes.csv`, `gene_data/`, and `grid.csv` (spatial modes).

### Applying treatment

Chemotherapy, targeted therapy and immunotherapy can be applied during growth, optionally as an
adaptive regimen that doses only above a tumor-burden threshold:

```bash
$ isccsim --sim-config config.yaml --treatment chemo --steps 4000 -o sim_out
$ isccsim --sim-config config.yaml --treatment chemo --adaptive --max-tumor-size 5000 -o sim_out
```

(Immunotherapy requires immune cells in the microenvironment: set `spatial_params.immune_density`
in the config.)

### 2. Sample, then 3. generate data

```bash
$ isccsample sim_out --method biopsy --fraction 0.2 -o sample_out
$ isccdata   sample_out -o data_out
```

## Validation

`iscc` ships reproducible validation scripts under `validation/` that benchmark each module
against established results:

- `validate_evolution.py` — dispersal governs the mode of evolution (Noble et al. 2022)
- `validate_snv.py` — neutral SNV site-frequency spectrum follows the 1/f law (Williams et al. 2016)
- `validate_cna.py` — copy number tracks oncogenic content under selection (Beroukhim 2010; Davoli 2013)
- `validate_treatment.py` — therapy response and adaptive dosing
- `validate_scrna.py` — scRNA count realism vs a real 10x dataset (PBMC3k)
- `validate_inference_recovery.py` — ABC recovers known CNA/SNV rates from synthetic tumours (parameter recovery)

Run the test suite with `python -m pytest`.

## Status & roadmap

Working: spatial growth (glandular), CINner-style selection, treatment, multi-cell observation,
single-cell/bulk DNA & RNA and Visium-like data, and the validation suite above. In progress
(see `DESIGN_inference.md`): a parameter-estimation/inference layer (ABC for the tumor module;
Splatter-style `estimate()` for the assays). Not yet implemented: read-level (FASTQ/BAM) output,
non-spatial/boundary spatial modes, and realistic biopsy/dissociation noise (see `AUDIT.md`).

## License

MIT.
