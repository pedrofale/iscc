<div align="left">
  <img src="https://github.com/pedrofale/iscc/raw/main/docs/assets/logo_text.svg#only-light" alt="in silico cancer center" width="340px">
  <img src="https://github.com/pedrofale/iscc/raw/main/docs/assets/logo_text_dark.svg#only-dark" alt="in silico cancer center" width="340px">
</div>
<p></p>

[![PyPI](https://img.shields.io/pypi/v/insilico-cancer-center.svg?style=flat)](https://pypi.org/project/insilico-cancer-center/)
[![Tests](https://github.com/pedrofale/iscc/actions/workflows/main.yaml/badge.svg)](https://github.com/pedrofale/iscc/actions/workflows/main.yaml)
[![Docs](https://github.com/pedrofale/iscc/actions/workflows/docs.yml/badge.svg)](https://pedrofale.github.io/iscc/)

`iscc` (in silico cancer center) is a multi-modal tumor evolution data simulator. It grows one
selection-driven, spatially-structured tumor, optionally treats it, samples it, and generates
single-cell and bulk DNA, RNA, and spatial data, down to sequencing reads. Because every run knows
the true clones, mutations, copy numbers, cell states, spatial niches, and lineages, `iscc` provides
a realistic ground truth for benchmarking computational tumor evolution methods.

## Main features

- **Multi-modal data generation.** DNA, RNA, and spatial data are emitted from the same evolving cells, so cross-modal relationships are coherent by construction.
- **Ground truth for benchmarking.** Every run knows the true clones, copy numbers, mutations, cell
  states, spatial niches, and lineage.
- **Calibrated to real data.** Assay parameters can be fit from a pilot dataset, and the defaults were pre-tuned to fit real data.

## The pipeline

`iscc` is a pipeline with a command-line tool associated with the main steps:

| Stage | CLI | Does |
|---|---|---|
| 1. Grow | `isccsim` | grow (and treat) a tumor; write ground-truth genotypes / CNVs / expression / spatial state |
| 2. Sample | `isccsample` | biopsy / dissociate the tumor into a sample of cells |
| 3. Assay | `isccdata` | generate single-cell & bulk DNA/RNA and spatial-transcriptomics data |
| 4. Visualize | `isccfig`, `isccgif` | Muller plot, 2D slice, clone tree, and growth animations |

## Quickstart

```bash
isccsim --sim-config config.yaml --steps 2000 --random-seed 0 -o sim_out
```

The shipped defaults produce a realistic multi-clone tumor, and `tumor.diagnose()` flags a degenerate
tumor after growth and tells you which knob to turn.

## Reproducing the landing animation

On the home page we show one simulated tumor carried through a clinical arc on a field of 96 by 96
demes. The lesion fills the ducts (DCIS), invades the stroma after a breach of the duct wall (IDC) and
seeds a metastasis. We then resect the primary and give chemotherapy, under which the metastasis shrinks
to about a fifth of its size and then regrows from a resistant subclone. It is reproducible from the
ordinary `iscc` CLI with the provided config file:

```bash
# 1. run the whole clinical arc (grow -> resect the primary -> chemo -> relapse) and write the
#    compartment trajectory the renderer consumes. --no-tables writes ONLY that trajectory, which
#    is all the renderer reads; drop it if you also want to run isccsample / isccdata on the result.
isccsim --sim-config configs/landing_hero.yaml -o out --no-tables

# 2. fully-labelled version (use --splash to drop the annotations as in the home page).
#    --skip-confinement starts the animation where the lesion first breaks out of its founding duct:
#    the founder is sealed in one acinus for the first ~90% of the run, so without it most frames
#    show a duct where nothing moves.
isccgif out --compartment --skip-confinement -o landing_full.gif
```

`configs/landing_hero.yaml` holds the exact parameters (genome, selection, ductal-field, metastasis and
treatment settings) and the timed treatment `schedule:` (the multi-phase arc, including the surgical
resection). It is `configs/landing.yaml`, the tutorial tumor with a metastasis compartment and
treatment, at the sweep scale of the paper. We use the invasion and stromal-survival settings of the
paper's regime and a shorter course of chemotherapy, and list every changed value in the header of
the config. Clones are coloured by their dominant selective trait and have the same colour in the
grids and the Muller plots:

| Colour | Trait |
|---|---|
| blue | proliferation |
| orange | duct escape |
| green | stromal survival |
| purple | metastasis establishment |
| red | chemotherapy resistance |
