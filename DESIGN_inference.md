# DESIGN: parameter estimation & inference layer

Status: design / scoping (2026-06-25). No inference code exists yet.

Companion: `DESIGN_features.md` (feature generation: sampling, reads, assay **batch-effect**
model). The assay `estimate()` here (§B/§C) fits that doc's batch model from real data — they
share the same technical parameters.

## Progress (one milestone ≈ one Claude session; cold-start by reading this doc)
- [x] **M0** — Noble `(n, D)` indices + recast `validate_evolution` (§D.1)
- [x] **M0b** — `J₁` tree-balance index + unit tests on small trees (§D.1)
- [x] **M1** — expose CNA/SNV rates (§A.0) + ABC engine (§A.1–2) + parameter recovery (§A.4.1)
- [x] **M2** — scRNA `estimate()` + fitted PBMC3k comparison (§B). *Built: `src/iscc/data/
  estimate.py` fits the F3 `BatchHyperParams` from a real count matrix — library-size lognormal
  (`mu_lib`,`sigma_lib`), NB `dispersion` via the mean-variance trend **de-biased for the
  library-size CV** (`phi=(S-c2)/(1+c2)`; pooled trend would inflate it), per-gene mean-expression
  gamma (Splatter table), protocol-aware logistic dropout (on for 10x, off for Smart-seq3), and —
  with ≥2 batches — `sigma_batch` (two-way-centred cross-batch log-fold-change) + `depth_batch_sigma`.
  Ambient/doublet are counts-only-unidentifiable → carried from the protocol preset, flagged in
  `fitted`. Recovery test (`tests/test_estimate_scrna.py`): simulate known params → recover mu_lib
  (rel 5%), sigma_lib/dispersion/sigma_batch (abs ~0.05); depth_batch_sigma tracks the **realized**
  per-batch spread (the gap to the hyper-param is genuine finite-batch sampling). `validate_scrna.py`
  upgraded to a FITTED PBMC3k overlay (mu_lib/sigma_lib/dispersion/dropout learned, not hardcoded);
  fitted depth+dispersion match real, but PBMC's cross-cell-type heterogeneity (median CV²≈99) stays
  out of reach of a clonal tumour's technical params — honest partial.*
- [x] **M3a** — real-tumour `(n, D, J₁)` overlay from Noble's published trees (§D.2). *Finding:
  iscc partially covers the real index region (best at scarce driver loci, prop_driver≈0.05); the
  abstract genome couples n and D, so it under-reaches the high-n deep-sweep tumours — motivates M3b.*
- [x] **M3b** — real-genome mode + fit-to-real PCAWG + Charm correlation (§A.5, A.4.2–3). *Built:
  real-genome mode (39 human arms; `GenomeSpec` from cytoBand+COSMIC CGC+Davoli; per-arm CN
  selection `s_arm` as an alternative `Selection` mode reading `seg_cns`; variable per-arm segment
  sizes — abstract mode byte-identical). Data generator downloads+cites cytoBand/COSMIC/Davoli/PCAWG
  → `validation/data/realgenome_*.csv`. Estimator: the arm-CN fitness factorises, so a **pooled
  per-arm RF** `(gain[arm],loss[arm])→s[arm]` (not a 39-D joint map, which extrapolated to the
  wrong sign) fits `s_arm` to PCAWG; cohort summary = mean over independent tumours. **Now
  tau-leaped (DESIGN_scalability §7):** `RealGenomeSimulator` grows each cohort tumour with
  tau-leaping (`update_mode="tau"`, the default) and bounds growth by **size** (`grow_to_size`,
  default 600 cells) — bounded per-sim cost, no exponential-growth stragglers. At a *matched* size
  tau and exact give a statistically identical per-arm summary (bias ≈ +0.003; tau-vs-exact
  agreement ≈ the exact-vs-exact Monte-Carlo noise floor: r≈0.47 vs floor≈0.48, meanAbsDiff 0.030
  vs 0.027 — `tests/test_realgenome_tau.py`, validation `--equivalence` panel), and tau is faster
  at every size with the gap **widening with size** (~1.3× @ 600 cells, ~1.9× @ 1500, ~6.2× @ 4000,
  where exact hits ~14 s/cohort-tumour and was the HPC bottleneck). The scaled in-session fit
  (4000×8, grow-to-600 ≈ 26× the prior smoke) reproduces the result: **fit-to-real r=0.78**
  (RMSE 0.25) and **s_arm↑ with Charm** (Pearson r=0.33, p=0.04; Spearman ρ=0.32) and with
  oncogene−TSG content (r=0.38, p=0.02) — oncogene-dense arms inferred amplification-favoured,
  TSG-dense deletion-favoured. HPC settings for the canonical figure live in the validation
  docstring (larger `--target-size`, where the tau speedup compounds).*
- [x] **M4** — DNA/Visium estimation (§C). **DNA half DONE** (`estimate_dna`, §C.1: MoM/MLE fit of
  `DNABatchHyperParams`, recovery + posterior-predictive `validate_dna`). **Visium half DONE** (§C.2):
  `estimate_visium` fits `VisiumBatchHyperParams` (per-spot library + the spatial/nugget split of the
  log-library residual via an SE autocorrelation fit → `field_sigma`/`field_lengthscale`; count
  overdispersion), recovery + posterior-predictive `validate_visium`.

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
| scRNA | Splatter `splatEstimate`, scDesign2/3: estimate NB mean/dispersion/dropout/library-size (+ batch) from real data | params hardcoded | `rna.estimate(real_adata)` → fitted PBMC3k; also fit batch model (DESIGN_features §B) |
| DNA assay | CellCoal / CNAsim: realistic coverage, error, allelic dropout | counts-only, fixed fpr/fnr | breadth-aware (WGS/WES/panel × bulk/sc) estimate of coverage/GC/ADO or panel VAF |
| Spatial (Visium) | SRTsim / scDesign3: estimate spot params from real Visium; validate spatial autocorrelation | **done**: spatially-autocorrelated capture field + aggregation + diffusion; `estimate_visium` fits spots/library/field; validate Moran's I | estimate spots/capture; validate Moran's I |

## A. Tumor / CNA inference (flagship — CINner-equivalent)

### A.0 Prerequisite: expose the rates as parameters
`CancerCell.mutate(..., n_snvs_per_allele=0.5, mut_prob=.1, cnv_prob=.1)` currently hardcodes the
SNV/CNA event probabilities. Inference requires them as configurable parameters (per-cell or selection
params) threaded through `GenotypeTumor`. Likewise `driver_effects` (selection strength) is
already a param. **First task: lift `mut_prob`, `cnv_prob` (and amp/del split, per-allele SNV rate)
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
iscc's genome is abstract (synthetic segments, random driver layout, single global
`driver_effects`). Real-genome mode is **mostly a configuration of the genome + selection layer,
plus one engine generalization** — the engine already tracks per-segment copy number and applies
CNAs per segment.

**The three mappings** (replace the random/global choices with real ones):

| | abstract mode (now) | real-genome mode |
|---|---|---|
| segments | arbitrary `n_segments` | **= human chromosome arms** (~39 autosomal p/q arms) |
| segment size | uniform `segment_size` | **∝ arm length** (cytoband table) |
| driver layout | random per `prop_driver` | **real per-arm oncogene/TSG content** (COSMIC CGC / Davoli) |
| selection strength | scalar `driver_effects` | **per-arm coefficient `s_arm`** (a vector) |

**The one real engine change — per-arm selection coefficients (CINner arm model).** Generalize
`Selection.driver_effects` from a scalar to a per-segment vector `s[seg]`, and add an arm-CN
fitness term `∏_seg s[seg]^(seg_cns[seg] − baseline)` reading the per-segment copy numbers iscc
already maintains (`genome_summary['seg_cns']`). `s>1` ⇒ amplification beneficial (oncogene-
dominated arm); `s<1` ⇒ deletion beneficial (TSG arm). The current fitness already responds to
arm CN indirectly (amplifying an oncogene segment raises `n_wt_onc`); this is the clean, direct
reformulation. **Add it as an alternative selection mode** so the abstract gene-driver model
keeps working.

**Reused unchanged:** the CNA mechanics (`mutate()` amp/del; `seg_cns`); `inference/summaries.py`
(per-segment gain/loss frequency = per-arm gain/loss, directly comparable to PCAWG);
`inference/abc.py` (just infers the higher-dimensional `s_arm` vector).

**Construction:** a `GenomeSpec` (built once from data) carrying arm names, per-arm lengths →
`segment_size`s, per-arm oncogene/TSG counts, and the `s_arm` vector. `GenotypeTumor(
genome_mode="real", genome_spec=…)` wires `n_segments = #arms`, arm-sized segments, and per-arm
`s` into `Selection` instead of the random `prop_driver` path.

**Data (download + cite, mirror M3a):** UCSC cytoBand (arm lengths); COSMIC CGC / Davoli
oncogene+TSG lists (gene→arm); Davoli 2013 Charm scores (orthogonal target). Store under
`validation/data/`.

**Caveats:** identifiability (§A.3) — fix the CNA rate, infer `s_arm`; ~39 arm coefficients is a
much larger ABC than M1's recovery demo → RF-ABC + `multiprocessing`, and the one-event-per-step
engine cost (`DESIGN_scalability.md` §7) bites here; SNV drivers can be off in v1 (arm-CN model
is what fits PCAWG; hybrid gene+arm is a later refinement). Parameter *recovery* (A.4.1) does not
need this mode; fit-to-real (A.4.2) and Charm (A.4.3) do.

## B. scRNA estimation (Splatter-style)

`estimate()` fits the parameters of the `DESIGN_features.md` §B assay/batch model from a real
dataset, so realistic technical magnitudes are *learned, not guessed*. **Depends on the batch
model existing (DESIGN_features F3); make `estimate()` protocol-aware (10x vs Smart-seq3) since
the active noise components differ.**

`src/iscc/data/estimate.py` (or `RNA.estimate`):
- **Library size:** fit lognormal to per-cell totals → (μ, σ) → feeds `lib_size_sigma`, depth.
- **Mean expression:** fit gamma/empirical to per-gene means.
- **Dispersion:** fit the mean–variance trend (NB: var = μ + μ²/size) → per-gene or global
  `dispersion`.
- **Dropout:** optional logistic dropout-vs-mean (Splatter) if NB zeros are insufficient.
- **Batch hyper-parameters** (needs ≥2 real batches): per-gene batch-factor scale `σ_batch`,
  per-batch depth/dispersion shifts, ambient and doublet rates — the technical parameters of the
  §B batch model in `DESIGN_features.md`.
Then:
- Upgrade `validation/validate_scrna.py` to a **fitted** comparison: estimate on PBMC3k → simulate
  → overlay (vs the current default-params overlay).
- **Estimation-recovery test:** simulate with known params → estimate → recover (incl. batch params).

## C. DNA & Visium estimation

Fit the per-modality technical parameters defined in `DESIGN_features.md` §D. The DNA half (C.1)
is **unblocked** — F4/F5 already give a generative model with named parameters
(`DNABatchHyperParams` in `data/batch.py`). The Visium half (C.2) is **DONE** (landed with F6):
`estimate_visium` inverts the `VisiumBatchHyperParams` generative model.

### C.1 DNA estimation (M4 DNA half)

**This is method-of-moments / MLE, not ABC.** The flagship tumour layer (§A) needs ABC because the
map parameters→observables is implicit (only reachable by simulating). The DNA *assay* parameters
are the opposite: each maps **directly to a marginal summary statistic** of a real coverage /
allele-count matrix. So DNA estimation is closed-form / 1-D MLE on summary statistics — the same
machinery as the scRNA `estimate()` (§B), fast and simulation-free. ABC stays reserved for the
biology.

**Target & shape.** Fit a `DNABatchHyperParams` (the `estimate()` targets are already documented on
that dataclass). Add a `DNAEstimate` dataclass + `estimate_dna(...)` in `data/estimate.py` (or a
sibling `estimate_dna.py`), mirroring `RNAEstimate`/`estimate()`: it returns fitted hypers ready to
drive `DNABatch` / the F4–F5 runners, plus a `.fitted` map flagging which fields were learned vs
carried from a preset. **Breadth-aware** (WGS / WES / panel) and **modality-aware** (bulk vs
single-cell), exactly as §B is protocol-aware (10x vs Smart-seq3) — the active noise components
differ by breadth and modality.

**Conditional on called copy number (the one wrinkle).** Our depth model is copy-number-scaled
(coverage ∝ CN × efficiency). Real data does not hand you true CN, so estimation is a **two-pass**:
(1) run a CN caller on the real data (ASCAT/Sequenza for bulk; Ginkgo/HMMcopy for single-cell) to
get per-locus CN, then (2) estimate the *residual technical noise given that CN*. We are explicit
that the technical params are fit **conditional on the caller's CN** — which is precisely the CN
regime we re-inject when simulating. This is the DNA analogue of §B's library-size de-biasing of the
dispersion: separate the technical layer from the part that is really biology.

**Per-field estimators (the `DNABatchHyperParams` targets).**

| Field | Estimated from real data by | Needs |
|---|---|---|
| `mu_depth` | mean per-locus coverage | counts only |
| `kappa` (DM) / `nb_dispersion` (NB) | coverage variance **after** dividing out CN×efficiency → MoM on the residual dispersion | counts + called CN |
| `gc_curve_sigma` | LOESS regress log-coverage on per-locus GC; curvature of the fit | counts + reference GC |
| `capture_sigma` | LogNormal sd of per-target (WES) / per-amplicon (panel) coverage after removing GC+CN (ideally vs a panel-of-normals) | counts + CN |
| `error_rate` | alt fraction at non-variant / hom-ref loci (the noise floor) | counts + variant mask |
| `depth_batch_sigma` | sd of log mean-depth across samples/batches | ≥2 batches |
| `ado_rate` (sc) | fraction of **known-het** loci collapsing to ~0 or ~1 BAF | single-cell + het sites |
| `beta_binom_conc` (sc) | BAF overdispersion at het loci beyond Binomial | single-cell + het sites |
| `ffpe_ct_rate` | excess alt at C>T-eligible sites vs the error floor | counts + variant context |
| `doublet_rate` | **not identifiable from counts** → prior-only (carried from preset, flagged not-fit) | demultiplexing |
| `breadth`, `depth_model` | **chosen, not estimated** (you know the assay) | — |

The `_PRIOR_ONLY` escape that §B's `estimate()` already uses (ambient/doublet) carries the genuinely
unidentifiable fields honestly instead of pretending to fit them.

**Bulk vs single-cell.** Same `DNABatchHyperParams`; fit shared fields (depth, dispersion, GC,
capture, error) from either modality. `ado_rate` / `beta_binom_conc` / `doublet_rate` are the
per-cell amplification/dropout layer — fittable **only from single-cell** data. `kappa` is the
amplification regime: large from pooled bulk, small from lumpy MDA/MALBAC single-cell; it falls out
of the residual coverage dispersion but means different things per modality.

**Validation (posterior-predictive, same loop as §B).** Fit hypers from a real DNA-seq dataset →
simulate DNA from a synthetic iscc tumour with those hypers → recompute the **same** summary stats
(coverage CV per CN level, BAF spread at hets, realized ADO rate, GC curve, CNA log-ratio
distribution) → overlay vs real. Plus an **estimation-recovery test**: simulate with known hypers →
estimate → recover (the DNA analogue of §B's recovery test). Targeted-panel validation focuses on
VAF accuracy + per-amplicon bias (no genome-wide CNA); WGS/WES on coverage + GC + CNA log-ratio.

**Deliverables.** `estimate_dna()` + `DNAEstimate`; a recovery test; a fitted real-data overlay
(`validation/validate_dna.py`); honest `.fitted` flags. Couples to read-level emission
(`DESIGN_features.md` C / F4–F5) but does **not** require it — counts-level estimation stands alone.

### C.2 Visium estimation (M4 Visium half — DONE, landed with F6)
`estimate_visium()` mirrors `estimate()`/`estimate_dna()` (MoM/curve-fit, `.fitted` map,
`_PRIOR_ONLY` escape): fits a `VisiumBatchHyperParams` from a Visium AnnData (per-spot counts +
`obsm["spatial"]`). **Spots-per-tissue** (on-tissue spot count / occupied fraction) and
**counts-per-spot** (`mu_counts` = mean per-spot total). The **headline decomposition**: the
variance of `log(per-spot library)` splits into a SPATIAL part (the capture field, recovered as the
zero-lag amplitude `a` of an SE autocorrelation fit → `field_sigma`) and a NON-SPATIAL nugget (the
i.i.d. depth noise → `sigma_counts`), with **`field_lengthscale`** the SE range of that same
correlogram fit on the per-spot library-size residual — the spatial analogue of M2's library-size
de-biasing. Count overdispersion → `kappa` (DM) / `nb_dispersion` (NB). The **per-gene batch
factor** (`sigma_batch`) is identifiable only across **≥2 sections** (constant within one section,
so confounded with biology — like M2 needing ≥2 batches); on a single section it (and
`ambient_frac`/`edge_sigma`/`diffusion_sigma`) is carried `_PRIOR_ONLY`. Recovery (simulate known
hypers → estimate → recover `mu_counts`/`sigma_counts`/`field_lengthscale`) in
`tests/test_estimate_visium.py`. Validation `validation/validate_visium.py`: posterior-predictive
overlay — fit → re-simulate → match **Moran's I** (spatial autocorrelation), the spot-count
distribution, and spots-per-tissue (`manuscript/figures/validation_visium.png`). **Defaults to REAL
data** (spatial analogue of scRNA/PBMC3k): `scanpy.datasets.visium_sge`
(`V1_Breast_Cancer_Block_A_Section_1`); real coords are normalized to spot-pitch units so the fitted
`field_lengthscale` is dimensionless and transfers to the synthetic grid. `--synthetic` is the
offline ground-truth round-trip + automatic fallback.

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
- **M2 — scRNA `estimate()`** + fitted PBMC3k comparison + recovery test (B). Fits the
  DESIGN_features §B batch model, so it **depends on that batch model (DESIGN_features F3)**;
  schedule M2 after F3 (or scope M2's first cut to the non-batch params only).
- **M3a — real-tumour evolution overlay** (#3, §D.2). Overlay Noble's published `(n, D, J₁)`
  for 35 real tumours on the iscc sweep; needs only a small CSV, not a genome-mapping layer.
- **M3b — real-genome mode + fit-to-real PCAWG/TCGA + Charm correlation** (A.5, A.4.2–3).
  The CINner-parity CNA result; needs external data + a genome-mapping layer.
- **M4 — DNA/Visium estimation** (C). **DNA half (C.1) is unblocked** (F4/F5 done): MoM/MLE fit of
  `DNABatchHyperParams`, two-pass CN-conditional, breadth-aware (WGS/WES/panel × bulk/sc) — a small
  matrix of breadth-specific estimate/validate cases, counts-level (no read emission required).
  **Visium half (C.2) DONE** (landed with F6): `estimate_visium` fits the per-spot library + the
  spatial/nugget split of the capture field (`field_sigma`/`field_lengthscale`) + count
  overdispersion from per-spot counts + coords; recovery + `validate_visium` posterior-predictive.

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
