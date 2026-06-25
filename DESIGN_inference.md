# DESIGN: parameter estimation & inference layer

Status: design / scoping (2026-06-25). No inference code exists yet.

## Progress (one milestone ≈ one Claude session; cold-start by reading this doc)
- [ ] **M0** — Noble `(n, D)` indices + recast `validate_evolution` (§D.1)
- [ ] **M0b** — `J₁` tree-balance index + unit tests on small trees (§D.1)
- [ ] **M1** — expose CNA/SNV rates (§A.0) + ABC engine (§A.1–2) + parameter recovery (§A.4.1)
- [ ] **M2** — scRNA `estimate()` + fitted PBMC3k comparison (§B)
- [ ] **M3a** — real-tumour `(n, D, J₁)` overlay from Noble's published CSV (§D.2)
- [ ] **M3b** — real-genome mode + fit-to-real PCAWG + Charm correlation (§A.5, A.4.2–3)
- [ ] **M4** — DNA/Visium estimation (§C)

Per-session ritual: `/clear` → "read DESIGN_inference.md and do M<x>" → implement + run *targeted*
tests → commit → tick the box above + update memory → stop (don't roll into the next milestone).
Use Sonnet for mechanical milestones (M2, M3a, M0 indices); Opus for reasoning-heavy ones (M0b
J₁ correctness, M1 ABC/identifiability, M3b genome mapping).

## Why this exists

For a simulator paper in PLOS Computational Biology (CINner's venue), qualitative
"recapitulate a known law" checks are *supporting* evidence, not the bar. The bar set by
CINner (Dinh et al. 2025) and the Splatter-class single-cell simulators is **parameter
estimation from real data**, demonstrated by:

1. **Parameter recovery** — infer back known parameters from synthetic ("ground-truth")
   data (needs no external dataset).
2. **Fit-to-real** — fit the model to a real dataset and show the *fitted* simulator
   reproduces that dataset's summary statistics.
3. **Orthogonal correlation** — inferred parameters correlate with independent biology
   (CINner: per-arm selection vs Davoli Charm scores).

iscc currently has **five qualitative validations** (dispersal→mode/Noble, SNV 1/f/Williams,
scRNA realism/PBMC3k, treatment response, CNA selection direction/Davoli) and **zero**
parameter estimation. This document scopes the estimation layer that closes that gap, module
by module, against each module's leading competitor.

## What CINner does (reference)

- **Inference:** ABC with random forests (ABC-rf). Sample 10k parameter sets from priors →
  simulate → summary statistics = per-chromosome-arm gain/loss frequencies → posterior fit to
  **PCAWG**; MAP point estimate.
- **Identifiability:** selection + CNA probabilities are jointly non-identifiable in bulk DNA;
  they *fix* whole-chromosome missegregation prob (0.01) to identify selection.
- **Priors:** missegregation Beta(1,10); per-arm selection Uniform(0.1, 2.0).
- **Validation:** fitted sim reproduces real cancer-type CNA landscapes; inferred selection
  correlates with Davoli 2013 Charm scores; ABC recovers posteriors on synthetic single-cell
  data.

## Per-module target (competitor convention → iscc plan)

| Module | Leading sim & its convention | iscc gap | Estimation target |
|---|---|---|---|
| Tumor + CNA | CINner / CNAsim: ABC-infer selection + CNA rates from PCAWG; fit real gain/loss; recover on synthetic | no inference, no fit-to-real | ABC engine + CNA/SNV summary stats; recovery → fit-to-real |
| Tumor evolution mode | Noble 2022: characterize tumours by `(n, D, J₁)` indices; overlay real tumour phylogenies in index space | qualitative dispersal→mode with non-Noble metrics | Noble indices (§D.1) + real-tumour overlay (§D.2), single structure |
| scRNA | Splatter `splatEstimate`, scDesign2/3: estimate NB mean/dispersion/dropout/library-size from real data | params hardcoded | `rna.estimate(real_adata)` → fitted PBMC3k comparison |
| DNA assay | CellCoal / CNAsim: realistic coverage, error, allelic dropout | counts-only, fixed fpr/fnr | estimate coverage+error from real bulk/sc DNA |
| Spatial (Visium) | SRTsim / scDesign3: estimate spot params from real Visium; validate spatial autocorrelation | basic aggregation | estimate spots/capture; validate Moran's I |

## A. Tumor / CNA inference (flagship — CINner-equivalent)

### A.0 Prerequisite: expose the rates as parameters
`CancerCell.mutate(..., n_events=5, mut_prob=.1, cnv_prob=.1)` currently hardcodes the SNV/CNA
event probabilities. Inference requires them as configurable parameters (per-cell or selection
params) threaded through `GenotypeTumor`. Likewise `driver_effects` (selection strength) is
already a param. **First task: lift `mut_prob`, `cnv_prob` (and amp/del split, event length)
into the config and the engine.**

### A.1 ABC engine (`src/iscc/inference/abc.py`)
Generic, dependency-light (numpy + scikit-learn only; **no JAX**, no R):
- `priors`: dict of name → scipy/numpy sampler.
- `simulate(theta) -> summary_vector` callback (wraps a short GenotypeTumor run).
- **Rejection ABC** first (transparent): sample N θ, simulate, keep the ε-closest by a scaled
  distance on summary stats. Then **regression-adjustment / RF-ABC** via
  `sklearn.ensemble.RandomForestRegressor` mapping summaries→params for tighter posteriors
  (the Python analogue of CINner's ABC-rf).
- Returns posterior samples + MAP.
- Parallelise simulation with `multiprocessing` (each sim is independent).

### A.2 Summary statistics (`src/iscc/inference/summaries.py`)
- Per-segment **gain frequency** (P(CN>2)) and **loss frequency** (P(CN<2)) — the CINner analogue.
- **Fraction genome altered (FGA)**, ploidy distribution.
- SNV: site-frequency-spectrum features (reuse `validation.neutral_sfs_rsq`, VAF quantiles).
Reuse existing `validation.segment_copy_numbers` / `population_vaf`.

### A.3 Identifiability
Replicate CINner's approach: jointly inferring selection + CNA rate is likely non-identifiable
from CN frequencies alone. Plan: fix one (e.g. total CNA rate) and infer selection, or add an
SNV-based statistic that breaks the degeneracy. Report a posterior-correlation/identifiability
analysis as part of the results.

### A.4 Deliverables (in priority order)
1. **Parameter recovery** (`validation/validate_inference_recovery.py`): choose θ_true →
   simulate → ABC → show posterior concentrates near θ_true; report coverage of credible
   intervals over replicate truths. **Abstract genome is fine — no external data needed.**
   This is the single most important, lowest-risk methods-paper figure.
2. **Fit-to-real** — needs the real-genome mode (A.5). Fit to a real cancer-type per-arm
   gain/loss profile (PCAWG/TCGA); show fitted sim reproduces it.
3. **Orthogonal** — inferred per-segment selection vs Davoli Charm scores (needs A.5).

### A.5 Real-genome mode (needed only for fit-to-real / Charm correlation)
iscc's genome is abstract (synthetic segments, random driver layout). To fit real PCAWG data
and correlate with Charm scores, add a configuration where **segments = chromosome arms** and
**drivers = known oncogenes/TSGs** (COSMIC/Davoli lists). Parameter *recovery* (A.4.1) does not
need this; fit-to-real does. Scope this as its own milestone.

## B. scRNA estimation (Splatter-style)

`src/iscc/data/estimate.py` (or `RNA.estimate`):
- **Library size:** fit lognormal to per-cell totals → (μ, σ) → feeds `lib_size_sigma`, depth.
- **Mean expression:** fit gamma/empirical to per-gene means.
- **Dispersion:** fit the mean–variance trend (NB: var = μ + μ²/size) → per-gene or global
  `dispersion`.
- **Dropout:** optional logistic dropout-vs-mean (Splatter) if NB zeros are insufficient.
Then:
- Upgrade `validation/validate_scrna.py` to a **fitted** comparison: estimate on PBMC3k → simulate
  → overlay (vs the current default-params overlay).
- **Estimation-recovery test:** simulate with known params → estimate → recover.

## C. DNA & Visium estimation (later)
- **DNA:** estimate coverage (mean + overdispersion) and error/ADO rates from a real bulk/sc DNA
  dataset; validate VAF accuracy and coverage distribution. (Couples to the deferred read-level
  emission.)
- **Visium:** estimate spots-per-tissue, counts-per-spot, capture efficiency from a real Visium
  dataset; validate spatial autocorrelation (Moran's I) and spot count distribution.

## D. Tumour-evolution-mode validation (Noble-style indices + real-data overlay)

This upgrades our current `validate_evolution.py` (Shannon diversity + a custom spatial
assortment over a dispersal sweep) to the metrics and fit-to-real that Noble et al. 2022 use,
**while keeping a single spatial structure** (the glandular/invasive engine). Noble's *central*
result — that distinct spatial *structures* occupy distinct regions of index space (silhouette
0.60) — requires multiple structures and is therefore **explicitly out of scope for now**
(deferred with the non-spatial/boundary/mixed engines). What a single structure *can*
legitimately do is exactly Noble's best-fit-model result (their Fig. 3d–e): the invasive
glandular model alone, at driver fitness ≈ 0.2, matches the empirical tumours. So our single
structure should (a) be characterized in Noble's indices and (b) be shown to cover the region of
index space where real tumours lie.

### D.1 (#1) Noble's evolutionary-mode indices (`src/iscc/inference/indices.py`)
Computed from the genotype tree (`genotypes_parents`) and counts; reuse where possible.
- **Clonal diversity D** = inverse Simpson, `D = 1 / Σ pᵢ²`, where `pᵢ` is the frequency of the
  i-th *driver-mutation combination*. Group cancer genotypes by their set of mutated driver
  positions (from `get_snvs` on oncogene/TSG indices, already used in `make_cell_data`), then
  compute frequencies over those groups. (Replaces our Shannon `clone_diversity`.)
- **Mean drivers per cell n** = `Σ i·pᵢ` = count-weighted mean number of driver mutations per
  cell (tree depth).
- **Tree-balance index J₁** ∈ [0,1] — Noble's weighted, Shannon-entropy-based balance index on
  the clone phylogeny (definition in Noble 2022 / the `treebalance` literature). Implement
  faithfully from `genotypes_parents` + clone sizes; this is the most involved metric and should
  be unit-tested against a couple of hand-computed small trees.
- (Optional) **clonal turnover Θ̄** and turnover time from `self.traces`.

Deliverable: recast `validate_evolution.py` to report `(n, D, J₁)` over the parameter sweep
(dispersal and/or driver fitness), tracing the model's trajectory through index space.

### D.2 (#3) Real-tumour overlay (fit-to-real for the evolution module)
Overlay empirical tumours in the same `(n, D, J₁)` space and show iscc's single structure covers
the region where real tumours fall (Noble's Fig. 3b/3d).
- **Data source (preferred):** reuse Noble et al.'s *published* per-tumour index values (their
  Fig. 3 Source Data / the `robjohnnoble` repo) — 35 tumours across 6 cancer types (ccRCC, NSCLC,
  breast, mesothelioma, uveal melanoma, AML). This avoids re-deriving phylogenies and matches
  their exact `(n, D, J₁)` definitions. Ship the small CSV under `validation/data/`.
- **Fallback:** compute `(n, D, J₁)` ourselves from a couple of public multi-region/single-cell
  trees if the Source Data is not directly usable.
- Deliverable: `validation/validate_evolution_modes.py` → figure overlaying the empirical points
  on the iscc parameter-sweep cloud; report the fraction of real tumours inside the simulated
  envelope and the best-matching driver-fitness value (cf. Noble's ≈0.2).
- **Honest framing:** with one structure this is a *coverage / best-fit* claim ("iscc reproduces
  the empirical index range"), not Noble's structure-discrimination claim.

### Scope note
- IN: Noble indices `(n, D, J₁)` (#1); real-tumour overlay (#3), single structure.
- OUT (deferred): multiple spatial structures and the silhouette structure-separation result
  (#2) — revisit when non-spatial/boundary/mixed engines exist.

## Code layout
```
src/iscc/inference/
  __init__.py
  abc.py          # priors, rejection + RF-ABC, parallel simulate
  summaries.py    # CNA gain/loss, FGA, SFS features
  indices.py      # Noble evolutionary-mode indices: n, D (inv-Simpson), J1 tree balance
  tumor.py        # GenotypeTumor -> summary wrapper + prior config
src/iscc/data/estimate.py   # estimate() for RNA (then DNA, Visium)
validation/
  data/noble_empirical_indices.csv   # published (n, D, J1) for 35 real tumours
  validate_inference_recovery.py
  validate_evolution_modes.py        # (n,D,J1) sweep + real-tumour overlay
  validate_scrna.py                  # upgraded to fitted comparison
```

## Sequencing (milestones)
- **M0 — Noble evolutionary-mode indices** (#1, §D.1). `(n, D, J₁)` from the genotype tree;
  recast `validate_evolution.py` to use them. No external data, single structure, small —
  good warm-up that also strengthens an existing validation.
- **M1 — ABC engine + tumor parameter recovery** (no external data). Expose CNA/SNV rates
  (A.0), build `abc.py` + `summaries.py`, deliver the recovery figure (A.4.1). *Highest value,
  lowest risk; demonstrable end-to-end without any download.*
- **M2 — scRNA `estimate()`** + fitted PBMC3k comparison + recovery test (B). Small, self-contained.
- **M3a — real-tumour evolution overlay** (#3, §D.2). Overlay Noble's published `(n, D, J₁)`
  for 35 real tumours on the iscc sweep; needs only a small CSV, not a genome-mapping layer.
- **M3b — real-genome mode + fit-to-real PCAWG/TCGA + Charm correlation** (A.5, A.4.2–3).
  The CINner-parity CNA result; needs external data + a genome-mapping layer.
- **M4 — DNA/Visium estimation** (C). Couples to read-level emission and a real Visium dataset.

## Risks
- **Compute:** ABC needs thousands of sims. The genotype engine is fast (~seconds for a small
  tumor) but 10k × seconds = hours; mitigate with small inference-sized sims, RF-ABC (fewer
  draws), and `multiprocessing`. No JAX (project constraint) — stick to numpy/sklearn.
- **Identifiability:** selection vs CNA-rate degeneracy (A.3); plan to fix-one-and-infer like
  CINner, and report the analysis honestly.
- **Abstract genome:** blocks fit-to-real and Charm correlation until M3; recovery (M1) is
  unaffected.
- **Summary-stat stochasticity:** noisy per-θ summaries; use replicate sims per θ or accept
  noise and lean on RF-ABC's averaging.

## How existing validations fit
The five current validations remain as qualitative model-behaviour checks. The estimation layer
adds the quantitative pillar (recovery + fit-to-real) that the venue expects, reusing existing
metrics as summary statistics (`segment_copy_numbers`, `population_vaf`, `neutral_sfs_rsq`,
`summary_stats`).
