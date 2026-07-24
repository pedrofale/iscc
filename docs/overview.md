# Overview

`iscc` is an internally-consistent, multi-modal **tumor-evolution data simulator**. It grows one
selection-driven, spatially-structured tumor, optionally treats it, samples it (biopsy /
dissociation), and generates single-cell and bulk **DNA / RNA / spatial** data — down to sequencing
reads — all sharing the *same* ground truth.

## Why iscc

- **One consistent tumor, every modality.** DNA, RNA, and spatial data are emitted from the *same*
  evolving cells, so cross-modal relationships (copy-number → expression, clone → niche) are coherent
  rather than bolted on.
- **Ground truth for benchmarking.** Every run knows the true clones, copy numbers, mutations, cell
  states, spatial niches, and lineage — the labels real data can never provide.
- **Calibrated to real data.** Assay parameters can be fit from a pilot dataset, and the shipped
  defaults sit inside a characterized *operating envelope* of realistic tumors.

## The pipeline

`iscc` is a staged pipeline, each stage a command-line tool:

| Stage | CLI | Does |
|---|---|---|
| 1. Grow | `isccsim` | grow a tumor; write ground-truth genotypes / CNVs / expression / spatial state |
| 2. Sample | `isccsample` | biopsy / dissociate the tumor into a sample of cells |
| 3. Assay | `isccdata` | generate single-cell & bulk DNA/RNA and spatial-transcriptomics data |
| viz | `isccfig`, `isccgif` | Muller plot, 2D slice, clone tree, and growth animations |

## Quickstart

```bash
isccsim --sim-config config.yaml --steps 2000 --random-seed 0 -o sim_out
```

The shipped defaults produce a realistic multi-clone tumor, and `tumor.diagnose()` flags a degenerate
tumor after growth and tells you which knob to turn.

## Reproducing the landing animation

The animation on the home page is one simulated tumor carried through the whole clinical arc — a
primary ductal lesion fills the ducts (DCIS), a subclone breaches into the stroma (IDC), it seeds a
metastasis, and a resistant clone relapses after chemotherapy. It is reproducible from the ordinary
iscc CLI — a committed config plus the two standard commands:

```bash
# 1. run the whole clinical arc (grow -> resect the primary -> chemo -> relapse) and write the
#    compartment trajectory the renderer consumes
isccsim --sim-config configs/landing.yaml -o out

# 2a. minimal hero variant -> docs/assets/landing_hero.gif
isccgif out --compartment --splash -o docs/assets/landing_hero.gif

# 2b. fully-labelled version (clone legend + axes + per-phase caption)
isccgif out --compartment -o landing_full.gif
```

`configs/landing.yaml` holds the exact parameters (genome, selection, ductal-field, metastasis and
treatment settings) and the timed treatment `schedule:` (the multi-phase arc, including the surgical
resection). Clones are coloured by their **dominant selective trait** — the same colour in the grids
and the Muller plots:

| Colour | Trait |
|---|---|
| blue | proliferation (driver mutations) |
| orange | duct escape (`breach`) |
| green | stromal survival |
| purple | metastasis establishment |
| red | chemotherapy resistance |
