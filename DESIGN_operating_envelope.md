# DESIGN: Operating envelope — parameter→phenotype atlas & built-in QC

**Status:** design (this doc) → implemented (`analysis/characterize_regimes.py`,
`validation/validate_operating_envelope.py`, `iscc.tumor.diagnostics`, `tests/test_diagnostics.py`).
Backlog item: *Operating envelope — parameter→phenotype atlas & built-in QC* (BACKLOG.md).

## Why

Users set the knobs in `notebooks/example_config.yaml` and can easily land in a **degenerate**
regime — a tumour that is extinct, monoclonal, hypermutated mush, spatially well-mixed with no
clonal territories, or with no microenvironment gradient. Such "crappy tumours" are not just an
aesthetic problem: the *well-mixed* and *no-gradient* regimes silently **break** the headline
benchmarks — the PEtracer lineage–space confound (needs clonal territories) and the multi-region
"sample trees are not phylogenies" result (needs regional clone structure). So we need to

1. **know & report** which parameter ranges yield realistic tumours (a robustness / sensitivity
   analysis, the kind scMultiSim and CINner ship), and
2. **warn at runtime** when a specific run came out degenerate, with an actionable hint.

Two audiences: reviewers (the sweep + phase diagram + supplementary section) and users (documented
defaults + valid ranges + `tumor.diagnose()`).

## Scope / non-goals

- The diagnostic is a **read-only readout** (like the F8 microenvironment ground truth): it MUST NOT
  change simulation output. It reads the grown state (`genotypes_counts`, `demes`, `selection`,
  the O2 field) and computes metrics; it never draws from `self.rng` or mutates the tumour.
- Targets the default **genotype-count engine** (`GenotypeTumor`), which the whole
  inference/sampling/analysis stack uses. The cell-level engines expose the same
  `genotypes_counts` / `demes` interface, so the helper works there too, but the tests exercise the
  count engine.
- The sweep is a **coarse map, not an inference**: small tumours, few seeds per cell, cached CSV.
  It characterises the axes qualitatively (where the boundaries are), not to publication precision.

## Where each knob enters the dynamics

(From `tumor/models/count.py`, `tumor/components/selection.py`, `tumor/components/cell.py`.)

| Knob | Set in | Enters via | Degenerate failure when… |
|---|---|---|---|
| `mutation_rate` | `cell_params.cancer` | `mut_prob = mutation_rate/(mutation_rate+dispersal_rate)` in `update` | too low → monoclonal (no clones); too high → hypermutated mush |
| `n_snvs_per_allele` | `cell_params.cancer` | Poisson mean new SNVs per allele in `mutate` | too low → no SNV diversity; too high → mush |
| `division_rate` vs `death_rate` | `cell_params.cancer` | event rate `count*(div+death)`, death share `death/(div+death)`; crowding multiplies death by `carrying_capacity` when a deme is full (`_death_rate`) | `death ≥ div` → extinction |
| `initial_cancer_cells` | `deme_params` | founder seed count (`_add`) | `1` → ~`death/div` founder-extinction prob |
| `dispersal_rate` | `cell_params.cancer` | dispersal vs mutation branch; disperse places a daughter in a neighbour deme | too high vs `division_rate` → well-mixed (no territories) |
| `prop_driver` × `driver_effects` | `selection_params` | `Selection.update_division_rate` (CINner oncogene/TSG fitness) | too high → selective sweep (monoclonal) |
| `grid_size` × `carrying_capacity` | `spatial_params`/`deme_params` | number of demes × cells per deme = max tumour size | too small → tumour too small for a gradient / few clones |
| `amp_prob` × `max_cn` | `cell_params.cancer` / `selection` | CNA amplification vs deletion; viability caps `highest_cn ≤ max_cn` | too high → CNA runaway / saturated FGA |
| `n_segments` × `segment_size` | `genome_params` | `n_genes = Σ segment_size`; the SNV/CNA substrate | too small → trivial genome (nothing to fit) |
| hypoxia `o2_diffusion` `D`, `o2_consumption` `k`, `o2_supply` `s` × tumour size | `microenv_params.hypoxia` | `_o2_field` steady-state O2; hypoxia = 1−O2 | diffusion length ≫ tumour, or tumour too small → no core–rim gradient |

## Metrics (Layer 1 sweep + Layer 3 QC)

Reuse existing code wherever it exists.

| Metric | Definition | Source |
|---|---|---|
| `n_cancer` / P(extinct) | `tumor.get_cancer_size()`; extinct if `== 0` | engine |
| `shannon` | Shannon diversity of cancer genotype frequencies | `iscc.validation.clone_diversity` |
| `n_subclones` | # cancer genotypes with frequency ≥ `subclone_freq` (default 0.01) | new (this module) |
| `vaf_1f_rsq` | R² of cumulative SFS to the neutral 1/f law | `iscc.validation.population_vaf` + `neutral_sfs_rsq` |
| `tmb` | mean # mutated sites per cancer cell (count-weighted over genotypes) | new (`rep.get_snvs()>0`) |
| `tmb_frac` | `tmb / n_genes` (fraction of the genome mutated per cell) | new |
| `driver_enrichment` | mean population VAF at driver sites / mean VAF at passenger sites (>1 ⇒ positive selection detectable) | new (`population_vaf` + `selection.get_oncogenes/get_tsgs`) |
| `clone_confinement` | `1 −` size-weighted mean of (per-subclone spatial spread / whole-tumour spread) over subclones (territories ⇒ high; well-mixed ⇒ ~0) | new |
| `fga` | fraction of genes with copy number ≠ 2 (count-weighted) | new (`seg_cns`) |
| `mean_ploidy` | count-weighted mean genome ploidy | new (`genome_summary['ploidy']`) |
| `hypoxia_contrast` | mean hypoxia in the inner (core) demes − mean in the outer (rim) demes, over occupied demes; only when `microenv_params` hypoxia is on | new (`tumor._o2_field`) |

**On the well-mixed metric (honesty note).** The regime taxonomy names "Moran's I of clone labels
~ 0". In this engine a naive clone-label Moran's I is *confounded*: raising `dispersal_rate` lowers
the mutation-branch probability `mut/(mut+dispersal)`, so it also **reduces the number of clones** —
fewer, larger clones can read as *more* spatially autocorrelated, so label-Moran's I is
non-monotonic in dispersal (verified empirically). We therefore measure **per-clone spatial
confinement** (spread of each subclone relative to the whole tumour) instead, which isolates mixing
cleanly and is monotonic in dispersal: `clone_confinement` falls from ~0.5 (territories) toward 0
(each clone smeared across the whole lesion) as dispersal rises. It is only meaningful with ≥2
subclones, so the QC gates the *well-mixed* check on `n_subclones ≥ 2` and a monoclonal tumour is
reported as *monoclonal*, not *well-mixed*.

## Degenerate-regime taxonomy & default thresholds

Thresholds live in `iscc.tumor.diagnostics.DEFAULT_THRESHOLDS` and are **overridable** per call
(`tumor.diagnose(thresholds={...})`). Defaults are deliberately lenient (flag the clearly broken,
not the merely unusual) and were sanity-checked against the sweep CSV.

| Flag | Metric & test | Default threshold | Culprit knob(s) → hint |
|---|---|---|---|
| `extinct` | `n_cancer < min_cancer` | `min_cancer = 25` | `death_rate ≥ division_rate`, tiny `initial_cancer_cells`/`carrying_capacity` → *lower death_rate or raise initial_cancer_cells* |
| `monoclonal` | `shannon < shannon_min` | `shannon_min = 0.5` | if `tmb < tmb_min`: *raise mutation_rate / n_snvs_per_allele*; elif `driver_enrichment > sweep_enrichment`: *lower driver_effects / prop_driver (selective sweep)*; else *raise mutation_rate* |
| `hypermutated` | `tmb_frac > tmb_frac_max` | `tmb_frac_max = 0.5` | *lower mutation_rate / n_snvs_per_allele* |
| `low_mutation` | `tmb < tmb_min` | `tmb_min = 1.0` | *raise mutation_rate / n_snvs_per_allele* |
| `well_mixed` | `n_subclones ≥ 2` and `clone_confinement < confinement_min` | `confinement_min = 0.1` | *lower dispersal_rate (relative to division_rate)* — **breaks PEtracer + multi-region demos** |
| `no_gradient` | microenv on and `hypoxia_contrast < contrast_min` | `contrast_min = 0.05` | *grow a larger tumour or raise o2_consumption k / lower o2_diffusion D* |
| `cna_runaway` | `fga > fga_max` | `fga_max = 0.95` | *lower amp_prob / max_cn* |
| `trivial_genome` | `n_genes < min_genes` | `min_genes = 100` | *raise n_segments / segment_size* |

Checks that don't apply (e.g. `no_gradient` when the microenvironment is off) are reported as
**skipped**, not failed.

## Realistic tumour size (thousands of cancer cells)

Users expect to grow tumours to a realistic size — **at least ~10³, up to ~10⁴ cancer cells** — for
both **structured** (`structure_radius > 0`, a glandular duct) and **unstructured**
(`structure_radius = 0`) simulations. Two levers control this:

- **Capacity.** The soft ceiling is `grid_size² × carrying_capacity` (crowding death switches on once
  a deme exceeds `carrying_capacity`). The shipped `grid_size = 25`, `carrying_capacity = 5` holds
  ~3×10³; scale either up (e.g. `grid_size = 40`, `carrying_capacity = 10` → ~10⁴) for bigger tumours.
- **Time.** In the **exact** engine each step is one birth/death event, so tumour size grows roughly
  **one cell per step** — reaching thousands needs thousands of steps (measured on the default
  config: ~710 cells at 1000 steps, **2174 at 3000**, 4440 at 6000; unstructured reaches ~2280
  similarly). For large tumours use **`update_mode="tau"`** (tau-leaping), which advances all clones
  per generation so wall-time scales with #clones × #generations rather than #cells.

Both structured and unstructured configs reach thousands of cells this way. The diagnostic emits a
**non-failing `small tumour` advisory** (threshold `min_realistic = 1000`) pointing the user at these
levers; it does not mark a small-but-healthy tumour as degenerate (only genuine extinction,
`n_cancer < min_cancer = 25`, is a hard fail).

## Layer 1 — characterisation sweep (`analysis/characterize_regimes.py`)

Sweeps the axes above one at a time around a realistic baseline (small tumour, fixed seeds), one row
per `(axis, value, seed)` run, computing all metrics. Writes a tidy CSV
`analysis/characterize_regimes.csv`. `--quick` shrinks it for CI/smoke. Coarse grid + a few seeds —
this is a map, deterministic and cached.

## Layer 2 — reported operating ranges (`validation/validate_operating_envelope.py`)

Produces `manuscript/figures/validation_operating_envelope.png`: phase-diagram panels
(mutation_rate × dispersal_rate; selection strength × mutation_rate; tumour-size × hypoxia D/k) with
the realistic region highlighted and each degenerate zone labelled, plus a metric-vs-knob panel. The
manuscript supplementary subsection *"Operating regimes: which parameters yield realistic tumours"*
(framed as the robustness/sensitivity analysis scMultiSim/CINner provide) plus a
defaults-and-valid-ranges table cite this figure.

## Layer 3 — built-in QC (`iscc.tumor.diagnostics` + `GenotypeTumor.diagnose()`)

`diagnose(tumor, thresholds=None, verbose=False) -> TumorDiagnosis`. `TumorDiagnosis` holds the
metric values and a list of `Check(name, ok, value, threshold, hint)`; `.ok` is True iff no check
failed; `__str__`/`.report()` prints a structured, colour-free readout. `GenotypeTumor.diagnose()`
is a thin method. The `isccdata`/`grow` entrypoint auto-warns when a run is degenerate, with an
opt-out flag (`--no-diagnose`). Cheap (O(#genotypes + #demes), one O2 solve if hypoxia on) and
read-only — never touches `self.rng` or the counts.
