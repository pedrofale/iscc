# Handoff prompt — operating envelope (parameter→phenotype atlas + built-in QC)

Saved 2026-07-03. Copy the block below into a fresh session. A DESIGN-FIRST milestone: characterise
which parameter ranges produce which tumour features, REPORT them (robustness/sensitivity section, as
scMultiSim/CINner do), and add a runtime QC so users don't generate "crappy tumours". Companion:
`BACKLOG.md` ("Operating envelope" section = the spec), memory `iscc-operating-envelope.md`,
`iscc-paper-positioning.md`. Branch from current `dev`. Do this independently of the benchmark suite.

---

```
Design-first, then build the OPERATING ENVELOPE for iscc: a characterisation of which parameter ranges
produce which tumour features, REPORTED so users pick sane settings, plus a built-in QC that warns when
a run produced a degenerate ("crappy") tumour. Two audiences: reviewers (a robustness/sensitivity
analysis — standard for simulator papers) and users (documented defaults + valid ranges + a runtime
warning). This also PROTECTS the headline benchmarks: the "well-mixed" and "no-gradient" degenerate
regimes silently break the PEtracer and multi-region-phylogeny demos, so it must flag them.

REPO & ENV
- Repo: /Users/pedroferreira/projects/iscc/repo (branch `dev`).
- Python/pytest: ~/miniconda3/envs/iscc/bin/python.
- Conventions: commit on `dev` WITH `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`; keep the
  FULL pytest suite green; be honest (report which regimes are genuinely bad, don't paper over them).
  Match surrounding style.

STEP 0 — DESIGN FIRST. Read the knob set and where each enters the dynamics: the example config
`notebooks/example_config.yaml` (the knobs users actually set), `tumor/models/count.py` __init__
(deme_params/spatial_params/genome_params/cell_params/microenv_params; `_death_rate` crowding; grow),
`tumor/components/selection.py` (prop_driver/driver_effects/prop_dispersal/dispersal_effects, gene vs
arm mode). Write a short `DESIGN_operating_envelope.md` scoping the axes, metrics, thresholds, and the
degenerate-regime taxonomy before coding.

THE DEGENERATE REGIMES (the taxonomy to detect & report), each with its metric and culprit knob(s):
- EXTINCT / too small — N cells, P(extinction) — death_rate >= division_rate; initial_cancer_cells=1
  (~7% founder extinction); tiny carrying_capacity.
- MONOCLONAL (no mutations) — low clonal Shannon diversity / #subclones — mutation_rate or
  n_snvs_per_allele too low.
- MONOCLONAL (selective sweep) — low diversity + one high-freq driver clone — driver_effects /
  prop_driver too high.
- HYPERMUTATED MUSH — TMB out of range, broken 1/f tail, every cell unique — mutation_rate too high.
- WELL-MIXED (no territories) — Moran's I of clone labels ~ 0 — dispersal_rate too high vs division_rate.
  *(breaks PEtracer + multi-region demos — must flag.)*
- NO MICROENVIRONMENT GRADIENT — hypoxia core–rim contrast ~ 0 — tumour small vs O2 diffusion length
  (D/k); grid_size x carrying_capacity too small. *(F8 microenv_params on.)*
- CNA RUNAWAY / TRIVIAL GENOME — fraction-genome-altered saturated / too few genes — amp_prob & max_cn
  too high; n_segments x segment_size too small.

=== LAYER 1: characterisation sweep ===
`analysis/characterize_regimes.py`: sweep the key AXES and compute the METRICS below, one row per run
(save a tidy CSV; keep tumours small + seeds fixed so it's fast and reproducible; a coarse grid + a few
seeds per cell is enough — this is a map, not an inference).
- AXES: mutation_rate & n_snvs_per_allele; division_rate vs death_rate (& initial_cancer_cells);
  dispersal_rate (relative to division_rate); prop_driver x driver_effects; grid_size x
  carrying_capacity; amp_prob & max_cn; microenv hypoxia D/k/s x tumour size.
- METRICS (reuse existing code where it exists — 1/f fit, Moran's I from iscc.integrations, TMB from
  cell_snv, diversity from genotype counts, hypoxia from microenv_truth): N cells / P(extinction);
  clonal Shannon diversity & #subclones above a freq threshold; VAF 1/f neutral-tail goodness-of-fit;
  TMB (mutations/cell); positive-selection detectability (driver enrichment vs neutral); Moran's I of
  clone labels; fraction-genome-altered / mean ploidy; hypoxia core–rim contrast.

=== LAYER 2: reported operating ranges (manuscript) ===
- `validation/validate_operating_envelope.py` → `manuscript/figures/validation_operating_envelope.png`:
  phase-diagram panels (e.g. mutation_rate x dispersal_rate; selection strength x mutation_rate;
  tumour-size x hypoxia D/k) with the REALISTIC region highlighted and each degenerate zone labelled.
- A SUPPLEMENTARY Results subsection "Operating regimes: which parameters yield realistic tumours"
  (framed as the robustness/sensitivity analysis scMultiSim & CINner provide) + a defaults-and-valid-
  ranges TABLE (per knob: default, valid range, what going out of range does). Keep cites/refs resolving.

=== LAYER 3: built-in QC diagnostic ===
- Add `tumor.diagnose()` (or a `iscc/tumor/diagnostics.py` helper) that, after growth, computes the
  metrics and returns/prints a structured report flagging each failed check against a threshold, with
  an ACTIONABLE hint ("monoclonal: raise mutation_rate"; "well-mixed: lower dispersal_rate"; "no O2
  gradient: grow larger / raise k"; "extinct: lower death_rate or raise initial_cancer_cells").
  Thresholds live in the design doc; make them overridable. Optionally auto-warn from the isccdata /
  grow entrypoint when a run is degenerate (opt-out flag). MUST NOT change simulation output — it is a
  read-only readout (like F8's ground truth). Keep it cheap.

DELIVERABLES: `DESIGN_operating_envelope.md`; `analysis/characterize_regimes.py` (+ CSV);
`validation/validate_operating_envelope.py` → figure; the manuscript supp subsection + defaults/ranges
table; `tumor.diagnose()` + tests `tests/test_diagnostics.py` (a deliberately-bad config trips the right
flag; a good config passes; diagnose() does not alter growth). Flip the BACKLOG item to DONE. Run the
full suite; commit on `dev`.

FEASIBILITY: most metrics already have code (1/f validation, Moran's I in iscc.integrations, diversity/
TMB from cell_data, hypoxia from microenv_truth). New work = the sweep harness, the phase-diagram
figure, the QC helper + thresholds, and the manuscript section. Keep the sweep coarse and cached.
```
