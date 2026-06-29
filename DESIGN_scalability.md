# Design note: scalability & the genotype-level architecture

_Status: proposal (2026-06-23). Companion to `AUDIT.md`. Motivated by the scalability
review of the agent-based core._

## 1. Where we are

iscc is a **cell-level agent-based model**. Each cell is a Python object holding a nested
genome (`list[segment]` of `{hap: [allele_set, …]}`), a `genome_summary`, and
`evolutionary_parameters`. Demes hold an (insertion-ordered) collection of cells; the tumor
holds a grid of demes plus rate vectors and trace snapshots. Growth is a Gillespie-style loop:
pick a deme ∝ rate, pick cells ∝ rate, sample birth/death/dispersal, apply.

### Measured costs (10×1000-gene genome, ~3k cells)

| | original | + incremental counts | + index sampling |
|---|---|---|---|
| throughput | ~155 cells/s | ~390 cells/s | **~1490 cells/s** |
| 3000-step wall | 19.9 s | 7.8 s | **2.0 s** |

Two behavior-preserving wins so far (both verified byte-identical / RNG-stream preserved):

1. **Incremental counts** — removed the per-step O(demes × genotypes) `Counter` rebuild
   (`Tumor.register_birth/death/parent`). Was ~65% of runtime.
2. **Index sampling** — `rng.choice(deme_list, p=…)` over a list of objects rebuilt an
   object ndarray every call (~71% after #1). Sampling an integer index instead
   (`rng.choice(len(deme_list), p=…)` then index in) draws the *identical* random numbers
   ~60× faster. Dropped `choice` from 3.87 s → 0.24 s over a 3000-step run.

Profile now (3000 steps, ~2.6 s total) is **flat** — no dominant per-step cost:

| Cost | Note |
|---|---|
| `make_cell_data` (dense matrices) | ~1.5 s, **one-time** O(n_cells × n_genes) output build — addressed by lazy materialization (§2) |
| `deepcopy` on division | ~0.5 s — addressed by copy-on-write genotypes (§5 phase 2) |
| small `np.array` allocs in event loop | minor |
| `Generator.choice` | ~0.24 s (was the bottleneck) |
| CINner fitness math | ~0% |

Two structural ceilings remain:

- **Time:** rate-proportional sampling is O(n_demes) (and O(n_cells_in_deme)) *per event*; the
  whole run is O(events) with a large per-event constant.
- **Memory:** one Python object per cell, and dense `cell × gene` output matrices —
  O(n_cells × n_genes). At 10⁵ cells × 10⁴ genes that's ~8 GB per matrix.

Note (already settled): **JAX is not the answer.** The hot path is object/dict sampling and
bookkeeping, not vectorizable array math; the CINner math is ~0% of runtime. Scaling is an
algorithm/data-structure problem, addressed below with plain numpy.

## 2. Target architecture: genotype-level state

**Key observation:** cells are highly redundant — many share an identical genotype (a clone).
The model already tracks `genotypes_counts`. The redesign makes the *genotype* the unit of
state and represents a deme as counts over genotypes, not a bag of cell objects.

### Data model

- **`Genotype`** (interned, shared, copy-on-write): compact genome + summary + fitness
  parameters + `parent_id`. Created once per distinct clone and never mutated in place.
  - Compact genome: a per-segment **copy-number vector** (`int16[n_segments]`) plus **SNVs**
    as a sorted array / per-segment bitset of mutated positions. A child copies the parent's
    arrays and applies a small delta (optionally stored as a delta against the parent for a
    persistent/immutable genotype tree → low memory).
  - `evolutionary_parameters` computed once at creation (the CINner math), then reused by
    every cell of that clone.
- **`Deme`**: `dict[genotype_id -> count]` (+ per-type counts) and an aggregate rate. No
  per-cell objects.
- **`Tumor`**: a genotype **registry** `{id -> Genotype}`, the deme grid, and rate structures.

### Events (per-event cost → O(log n), no genome copy on the common path)

- **Birth:** pick deme ∝ rate → pick genotype in deme ∝ `count × division_rate` → with prob
  `p_mut` create a *new* child `Genotype` (copy + delta, recompute summary/fitness, register
  parent), else increment the existing genotype's count. Genome copy happens **only when a new
  genotype is born** (rare vs. total births), not on every division.
- **Death:** decrement a genotype's count in the deme.
- **Dispersal/migration:** move count between demes.

### Outputs

Emit **`genotype × gene`** matrices + **per-deme genotype counts** (already have the latter).
The dense `cell × gene` view is materialized lazily — and ideally only for the cells a
**biopsy/assay actually captures** (the sampling stage), which is where per-cell stochasticity
(expression noise, dropout) naturally belongs anyway. This removes the O(n_cells × n_genes)
memory wall from the growth stage.

## 3. Faster event sampling

The object→index fix above already removed most of the `choice` cost without changing the
algorithm (still O(n) per draw, but ~60× smaller constant). The remaining O(n) only bites once
the grid/clone counts get *much* larger; at that point replace `np.random.choice(p=…)` with a
**Fenwick/alias structure** over deme rates (and, within a deme, over genotype rates):

- Maintain a cumulative-rate tree; sampling an event is O(log n), updating a rate after an
  event is O(log n).
- This is the standard Gillespie-with-binary-search / composition-rejection approach and pairs
  naturally with the genotype-count demes. Deferred until profiling shows `choice` re-emerging.

## 4. Expected scaling

| | current (cell-level) | target (genotype-level) |
|---|---|---|
| state memory | O(n_cells) objects | O(n_genotypes) + O(n_demes) |
| per-event time | O(n_demes + n_cells_in_deme) | O(log n_demes + log n_genotypes) |
| genome copy | every division (deepcopy) | only on a new genotype |
| output memory | dense O(n_cells × n_genes) | O(n_genotypes × n_genes), cells materialized on demand |

Since `n_genotypes ≪ n_cells` under clonal growth, this is a large constant- and
order-reduction in both axes.

## 5. Migration path (incremental, API-stable)

Keep the public surface (`GlandularTumor`, `grow`, `write`, `cell_data`, `isccsim`) stable;
`cell_data` becomes a materialized view. Phased so each step is shippable and testable:

1. **Done:** incremental tumor-level counts/parents (removed the per-step Counter rebuild)
   **and** index-based sampling (removed the object-array `choice` cost). ~9.6× overall.
2. **Done — Copy-on-write genotypes:** clonal cells share one `genome`/`genome_summary`
   (shallow share in `divide`); `mutate` copies-first before diverging. Removes the
   per-division `deepcopy` (now only on actual mutations) and collapses genome memory to
   ~one object per genotype (measured: 3998 cells → 11 genome objects, 363×), while every
   cell keeps its own coordinates/deme/genotype label and per-cell `evolutionary_parameters`.
   Output byte-identical.
3. **Done — genotype-count engine (`GenotypeTumor`, `models/count.py`).** Demes are
   `dict[genotype_id -> count]` over a shared genotype registry; no per-cell objects.
   Cells are materialized on demand in `make_cell_data`. Built **alongside** the cell
   engine (additive — all existing tests stay green) and selectable via `isccsim
   --mode genotype`; writes the canonical layout so `isccsample`/`isccdata` consume it
   unchanged. **Validated** as statistically equivalent to `GlandularTumor` (matched
   survival 11/12 vs 11/12, survivor-size means 399 vs 398, comparable clone counts) and
   reproducible (deterministic creation-ordinal ordering, seeded RNG). **~8.5× faster**
   than the optimized cell engine on a matched 10k-gene run (14.7k vs 1.7k cells/s).
   Not byte-identical by design (different random variables); validated by distribution.
   Treatment (chemo/targeted/immuno) and the corrected additive immune-death model now run on
   this engine, with optional `immune_density` seeding (see `count.py::grow`/`_apply_treatment`/
   `_death_rate`); immune *dynamics* (recruitment/migration) remain future work.

   **Now the default (3c — parity & integration).** `GenotypeTumor` has full feature
   parity: shared `plot_muller`/`plot_grid` (factored into `tumor/viz.py`), structures +
   normal cells (epithelial/stromal seeded as static counts, reusing glandular's geometry),
   and complete `cell_data` incl. `cell_evo`. It is the **default `isccsim` mode**, the
   example config/`generate_example.py` and notebooks 01–03 all use it, and the cell engine
   (`glandular`) is retained as the validation reference / opt-in mode.
4. **Done — compact array genome (bitset SNVs).** Each allele copy is now a boolean
   bitset of length `segment_size` (was a Python `set`); copy number is the number of
   allele copies, tracked explicitly. `mutate`/`get_snvs`/`get_cnvs`/`get_exp`/`expresses`
   and the summary updaters all operate on bitsets (vectorized via numpy indexing /
   `np.sum`). Used by **both** engines. This also fixes the CNV-amplification aliasing
   (amplified copies are now independent `.copy()`s) and is the **prerequisite for
   read/FASTQ-level and allele-specific assays** (explicit per-position alleles). Validated
   by the full suite + statistical-equivalence/reproducibility tests (not byte-identical to
   the pre-phase-4 set order — `np.where(~bits)` is sorted). Modest growth-speed trade
   (~14.7k→~10.8k cells/s at 10k genes) for vectorizable materialization and reads-readiness.
   NOTE: emitting reads itself is future assay-realism work; phase 4 only lays the
   representation foundation.
5. **Alias/Fenwick event sampling:** O(log n) deme/genotype selection — only once `choice`
   re-emerges as a bottleneck at much larger scale.

## 6. Constraints to preserve

- **Reproducibility:** iterate genotype ids in a deterministic (insertion/sorted) order and
  keep threading the seeded `numpy.Generator` (see `AUDIT.md` reproducibility entry). The
  alias structure must be built/updated deterministically.
- **Spatial models:** demes/grid stay; only the *intra-deme* representation changes.
- **Treatment & selection:** act per genotype (fitness is already a genotype property), so they
  map cleanly onto the new model.
- **Per-cell stochasticity** (expression noise, sequencing dropout) moves to the
  materialization/assay stage, consistent with the pipeline's biology → lab → assay split.

## 7. Tumor-size realism & computational cost — REQUIRED ASSESSMENT — **DONE (2026-06-29)**

> **Outcome:** benchmarked the exact engine, implemented **tau-leaping** as an alternative update
> mode on `GenotypeTumor` (the exact one-event engine is preserved as the validation reference),
> validated it against the exact engine by distribution, and re-benchmarked. iscc now reaches
> **Noble parity (10⁶ cells) in ~15–50 s locally** and projects **diagnosis scale (10⁹) in ~4 h**,
> where the exact engine was effectively capped at ~10⁴ cells/run. See **§7.1** for the numbers.


We have not yet established how large — and therefore how *realistic* — a tumor iscc can grow,
nor the wall-time cost of reaching realistic sizes. This is a prerequisite for credible "realistic
data" claims and it gates the ABC inference (DESIGN_inference §A: ~10^4 sims, each needing a
plausibly-sized tumor, is infeasible if a single sim is slow).

**Reference population sizes to compare against:**
- Real tumour at diagnosis: ~10⁹–10¹⁰ cells (~1 cm³).
- **Noble et al. 2022**: simulate to **~10⁶ cells**, with demes/glands holding **512–8,192 cells**
  each (within-deme Moran process; between-deme fission/migration).
- **Jeffrey West / HAL** 2D agent-based models: typically **~10⁴–10⁶ agents** (on-lattice; HAL is
  built for efficiency + real-time viz). See HAL (Bravo et al. 2020) and the "seven-step guide to
  spatial ABM of tumour evolution" (arXiv:2311.03569).
- **SISTEM** (Weiner et al. 2025, PMC12701798): an agent-based, **clonal-level** DNA-seq simulator
  that reaches **5×10⁵–5×10⁷ cells across 1–5 sites in 0.46–25 min on commodity hardware** — by
  advancing **all clones once per discrete generation** (a birth–death–migration step per clone),
  not one event per update. **External precedent that the fix below is right and achievable**: the
  generation-based, clone-batched update *is* the tau-leaping approach. iscc's M3b "publication-scale
  fit is HPC-bound" is exactly the cost SISTEM's design avoids.

**Two distinct bottlenecks (keep separate):**
1. **Size ← Θ(N) sequential events.** Even with O(log n) event *sampling* (§3), the engine does
   **one birth/death per `update()`**, so reaching N cells needs ~N events (more with turnover).
   This — not per-event cost — is what caps achievable size. **Fix: tau-leaping** (as in CINner):
   per time-step τ, draw `Poisson(rate · count · τ)` births/deaths per (deme, genotype) and apply
   in batch, so wall-time scales with #clones × #timesteps rather than #cells. This is the main
   lever for realistic size; needs care at carrying capacity and to preserve reproducibility.
   *SISTEM does the simplest concrete version: one synchronous birth–death(–migration) step per
   clone per discrete generation — a good template to follow for iscc's genotype-count demes.*
2. **Per-event cost ← #demes / #genotypes** (the §3 Fenwick/alias item). Independent of (1);
   matters once grids get large.

**REQUIREMENT — growth-over-time visualization must be preserved (and improved).** Tau-leaping
advances by *time intervals*, not by skipping events: at the end of every τ (or every generation)
the engine still records a **full per-clone count snapshot** into `self.traces`, so `plot_muller`
/ `plot_grid` (which read `traces` + `genotypes_parents`) keep working unchanged — no trajectory
is lost. It is in fact better: the trace x-axis becomes **real simulation time / generations**
(currently it is the event/step index, a lumpy proxy for time), and snapshot resolution is a free
parameter (record every leap, or every k). Keep the exact one-event engine as a **reference** and
validate tau-leaping against it by distribution (clone-size distribution, Muller structure), as we
did when introducing the genotype engine. Use small/adaptive τ (smaller when populations are tiny)
to stay accurate and avoid carrying-capacity overshoot.

**Deme abstraction lever.** Setting `carrying_capacity` to gland scale (Noble's 512–8,192) lets
each deme/agent represent thousands of biological cells, so #agents ≪ #cells. But *growing into*
those cells still costs events unless (1) is addressed — the abstraction reduces agent count, not
the number of birth events.

**Deliverable for the assessment session:**
- Benchmark the genotype engine: cells/s and events/s, **max feasible tumor size**, and whether
  throughput degrades as #demes/#genotypes grow (isolating bottleneck 2).
- Project wall-time to 10⁶ (Noble parity) and 10⁹ (diagnosis) cells, with and without tau-leaping.
- Decide: implement tau-leaping (and/or alias sampling) and re-benchmark; document the realistic
  size/time envelope vs Noble & West.
- (A benchmark script was drafted in this session but deliberately not run; start there.)

## 7.1 Results (2026-06-29)

Benchmark: `validation/benchmark_scalability.py` (10×1000 = 10k-gene genome, `carrying_capacity=1`
so growth is unbounded — the size/time envelope; low death rate so the founder survives). Machine:
local Mac, single process, numpy only.

### Exact one-event engine — the §7 baseline (the binding cost M3b flagged)

| metric | value |
|---|---|
| reached | ~19k cells in 20 s (grid 64², mut=0.01) |
| throughput | **degrades 20×** as the tumour grows: **9421 ev/s → 451 ev/s** over the run |
| naïve projection (final 931 cells/s) | 10⁶ ≈ 18 min, 10⁹ ≈ 12 d |
| honest projection | **worse than naïve** — per-event cost keeps rising with #genotypes (see below), so a single run is effectively capped at ~10⁴ cells locally; 10⁶ is HPC-bound |

The throughput collapse **confirms bottleneck 2**: each `update()` does
`sorted(cancer_gids)` + a per-genotype weight array + `rng.choice` over genotypes in the deme, so
per-event cost grows with #genotypes/deme (169 → 1754 genotypes over the run). This is the §3
Fenwick/alias item and is **separate** from the size problem — but it means the exact engine's size
ceiling is even lower than the Θ(N)-events argument alone implies.

### Tau-leaping engine (this session)

One synchronous generation advances **all** clones: per (deme, genotype, count c) draw
`Poisson(division·c·τ)` births and `Poisson(death·c·τ)` deaths, split births into a mutation branch
(in-place division → new genotype) and a dispersal branch (daughter to a neighbour) by
`Binomial(births, mut_prob)`, apply in batch. Adaptive sub-stepping keeps `rate·dt ≤ 0.34` (accurate
Poisson regime; prevents carrying-capacity overshoot since death rates are re-read each substep).

| regime | reached | rate | projection |
|---|---|---|---|
| mut=0.01, grid 96² | 508k cells, 33 gen, 25 s | 20k cells/s | 10⁶ ≈ 49 s |
| mut=0.001 (clonal), grid 128² | **1.27M cells (> Noble 10⁶), 41 gen, 18.5 s** | 69k cells/s | **10⁶ ≈ 15 s, 10⁹ ≈ 4 h** |

Tau-leaping makes cost scale with **#clones × #generations, not #cells** (guarantee a): a clone of
size 10⁵ advances in one Poisson draw per generation. The residual slowdown at very large N is
**genotype creation** (a deepcopy per *new* mutant clone) — intrinsic to the infinite-sites model
and **identical in cost to the exact engine**; it is the §3 concern, not a tau-leaping cost, and it
shrinks with a realistically small `mutation_rate` (the clonal regime above). Alias/Fenwick sampling
(§3/§5.5) is the next lever if even larger grids are needed.

### Validation (`validation/validate_tau_leaping.py`, `tests/test_tau_leaping.py`)

Validated against the exact engine **by distribution** (as the genotype engine was validated vs the
cell engine — it is NOT byte-identical, different random variables), grown to a matched size over
40 seeds:

- clone-size summaries agree within ~18% — size 1.18×, #clones 1.18×, top-clone-fraction 0.93×,
  inverse-Simpson diversity 1.18× — **inside the genotype-vs-cell engine's accepted 0.5–2.0 band**.
- **Definitive correctness signature:** the (small) clone-count bias **converges to the exact
  process as τ → 0** — #clones ratio 1.154 (τ=1) → 1.092 (τ=0.5) → 1.054 (τ=0.25).
- **Growth-over-time preserved (guarantee b):** a full per-clone snapshot is recorded every
  `snapshot_every` generations into `self.traces`, so `plot_muller`/`plot_grid` work unchanged —
  now on a **real-time x-axis** (`self.trace_times`). Figure:
  `manuscript/figures/validate_tau_leaping.png` (growth curve · Muller · clone-size ECDF · τ→0).
  Fixed a latent `viz` bug (dangling ancestry edges for clones born-and-lost within a snapshot
  interval) that the higher clone counts exposed; the fix is engine-agnostic and a no-op for the
  exact engine.

### Envelope vs the literature

iscc + tau-leaping now sits with **Noble 2022 (10⁶)** and the upper end of **West/HAL (10⁴–10⁶)**
for local single runs, and reaches **SISTEM's 5×10⁵–5×10⁷ range** in minutes — exactly the
generation-batched clonal update SISTEM uses. Diagnosis scale (10⁹) is an HPC run, not infeasible.
This unblocks publication-scale ABC: a reference table of plausibly-sized tumours is now minutes,
not the HPC-only cost M3b reported.

### How to use

`GenotypeTumor(..., update_mode="tau", tau=1.0, snapshot_every=1)`; `grow(n_steps=G)` advances `G`
generations. Default remains `update_mode="exact"` (unchanged reference engine). Reproducible
(seeded `numpy.Generator`, creation-ordinal genotype ordering). No JAX.
