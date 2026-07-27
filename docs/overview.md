<div align="left">
  <img src="https://github.com/pedrofale/iscc/raw/main/docs/assets/logo_text.svg" width="340px">
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

The animation on the home page is one simulated tumor carried through the whole clinical arc: a
primary ductal lesion fills the ducts (DCIS), a subclone breaches into the stroma (IDC), it seeds a
metastasis, and a resistant clone relapses after chemotherapy. It is reproducible from the ordinary
`iscc` CLI by using the provided config file:

```bash
# 1. run the whole clinical arc (grow -> resect the primary -> chemo -> relapse) and write the
#    compartment trajectory the renderer consumes
isccsim --sim-config configs/landing.yaml -o out

# 2. fully-labelled version (use --splash to drop the annotations as in the home page)
isccgif out --compartment -o landing_full.gif
```

`configs/landing.yaml` holds the exact parameters (genome, selection, ductal-field, metastasis and
treatment settings) and the timed treatment `schedule:` (the multi-phase arc, including the surgical
resection). Clones are coloured by their dominant selective trait and have the same colour in the grids
and the Muller plots:

| Colour | Trait |
|---|---|
| blue | proliferation |
| orange | duct escape |
| green | stromal survival |
| purple | metastasis establishment |
| red | chemotherapy resistance |
