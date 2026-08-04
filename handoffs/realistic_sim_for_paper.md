# Handoff: put the paper's evidence on the realistic structured simulation

## Goal
Move the paper's benchmark suite, the extended notebooks, and the manuscript figures onto the same
**realistic, structured, cm-scale** simulation the tutorials now use — the ductal-field,
breach-gated DCIS→IDC tumour (`notebooks/example_config.yaml`, grid 170, K_duct 60 / K_stroma 30,
seed 2, ~190k cancer / ~460k total). Today the evidence is generated on much **smaller, per-benchmark
ad-hoc sims** that predate the cm-scale engine, which undersells the paper's own claims.

## Why (the gap, measured 2026-08-04)
The paper's thesis is explicitly *structure-misleads-inference* and it claims the engine "reaches
millions of cells" with a "structured microenvironment" (`manuscript/paper.tex` §abstract/intro).
But the evidence is produced at grid **13–50** (thousands of cells):

| where | sim | scale |
|---|---|---|
| **Tutorials** (shipped, v0.2.0) | `example_config.yaml` — full ductal field, breach-gated, cm-scale | **grid 170**, ~460k cells |
| `notebooks/base_sim.py` (extended notebooks) | 4 glands + breach/DCIS→IDC — *same biology* | grid 40, ~10k cancer |
| `validation/integration_common.py` (clonealign, inferCNV, Numbat) | structure_radius 5 | grid 20 |
| `validation/deconv_common.py` (cell2location/RCTD — FLAGSHIP) | structure_radius 10 | grid 26 |
| `validation/programs_common.py` (scDEF/cNMF) | structure_radius 0, cancer-only | grid 16 |
| `validation/cohort_common.py` (Harmony/scVI, demux) | radius 0–5 | grid 13–17 |

So the benchmarks are mostly *not flat* — they use ductal-field biology — but they're **small and
fragmented** (each `*_common.py` has its own grid/params). The cm-scale engine + `Resection` +
`max_cells` (all new in 0.2.0) now make it feasible to grow ONE realistic cm-scale tumour and sample
a **tractable** piece per external tool, exactly as the tutorials do. Demonstrating the failure
modes on that realistic tumour (glands, breach-gated invasion, cm-scale, millions of cells) is far
more compelling — and internally consistent with what `iscc` now ships and showcases.

## Plan
1. **A shared realistic benchmark regime.** Derive one config/helper from `example_config.yaml` (the
   ductal-field cm-scale biology) that every benchmark grows from — replacing the per-benchmark
   `SPATIAL`/`GENOME`/`CANCER` dicts in `integration_common.py`, `deconv_common.py`,
   `programs_common.py`, `cohort_common.py`, and `base_sim.py`. Keep it seeded/reproducible.
2. **Keep the external tools tractable via sampling, not shrinking the tumour.** Grow the big
   realistic tumour once, then `Resection`/biopsy + `make_cell_data(max_cells=…)` to hand each tool
   (clonealign, inferCNV, Numbat, cell2location/RCTD, scDEF/cNMF, Harmony/scVI, MHN/TreeMHN, …) a
   sample it can run on — the tutorials' pattern. Watch `make_cell_data` memory at cm-scale.
3. **Per-benchmark judgement (don't blindly switch everything):**
   - *Structure-thesis benchmarks* (deconvolution, Visium niches, spatial diagnostics, PEtracer,
     niche/CCI) → **must** use the realistic structured tumour; that IS the point.
   - *Molecular-isolation benchmarks* (gene programs, MHN progression, clonealign SNV/CNA) →
     may keep a controlled variant to isolate the effect, but should **also confirm the finding on
     the realistic tumour** so it isn't a toy artefact.
   - Note where a benchmark genuinely needs a special structure (cohort = multi-patient; demux =
     immune_density) and parameterise the shared regime rather than forking it.
4. **Regenerate + re-run.** Re-produce every `manuscript/figures/validation_*.png` from the
   realistic sims, re-running the REAL tools in their dedicated `iscc-<tool>` conda envs (see memory
   `iscc-dedicated-envs-per-tool`; `validation/README_integration.md`). Update the paper's numbers,
   sentences, and any "millions of cells / realistic structure" claims to match what's actually shown.
5. **Re-run the extended notebooks** (`base_sim.py` consumers: `combining_scdna_scrna`, `dna_mhn`,
   `visium_niches`, `scrna_batch_effects`, `real_data_comparison`) on the scaled-up realistic regime;
   fix any broken recap cross-links.

## Constraints / gotchas
- **Tool cost at scale:** cell2location, Numbat, scDEF are slow/memory-heavy — sample down (step 2),
  don't feed them cm-scale directly. Each runs in its own env, never the core env.
- **Effects may shift with realism/scale:** e.g. the deconvolution regional-confound and the
  scDEF/cNMF "CNA manufactures spurious factors" result should be *re-measured* — the numbers in the
  paper today are from small sims and may move. That's expected and fine; report the realistic ones.
- **Determinism:** seed everything; `Date.now()`/`random` are unavailable in workflow scripts.
- **Big effort, multi-session:** likely one session to build the shared regime + convert 2–3 flagship
  benchmarks (deconvolution, programs, clonealign), then further sessions per tool + the paper edit.

## Key files & references
- `notebooks/example_config.yaml` (the realistic regime to align to); `notebooks/base_sim.py`.
- `validation/integration_common.py`, `deconv_common.py`, `programs_common.py`, `cohort_common.py`,
  `validate_*.py` (write the figures), `README_integration.md`.
- `manuscript/paper.tex`, `manuscript/figures/`.
- Memory: `iscc-integration-benchmark-suite`, `iscc-paper-positioning` (consistency-not-firsts),
  `iscc-dedicated-envs-per-tool`, `spatial-visium-scale-audit`, `iscc-project-vision`.
