# iscc — Full Module Audit & Gap Report

_Date: 2026-06-23. Scope: every module under `src/iscc`, the test suite, and packaging.
Goal context: turn iscc into the standard end-to-end tumor-evolution data simulator
(growth → biopsy → lab prep → sequencing/spatial → treatment), with a CINner-style
selection model._

## TL;DR

The architecture is the right shape and a surprising amount **works**: spatial growth,
the CINner-style division-rate fitness, and the four sequencing assays (bulkDNA, scDNA,
scRNA, Visium) are implemented and covered by passing tests (65 pass / 3 fail; the 3
failures are stale `Treatment.get_dosage` tests, not engine bugs).

But the **agent core has several confirmed correctness bugs that corrupt the
genome/fitness state every downstream stage depends on**, the **sampling layer
(biopsy/slice/dissociation) is essentially unimplemented**, and **all five CLI entry
points are broken** (wrong target or empty module). These must be fixed before any
realism work, because simulated data inherits them silently.

Maturity by pipeline stage:

| Stage | Module(s) | State |
|---|---|---|
| Growth (spatial ABM) | `tumor/tumor.py`, `models/glandular.py`, `components/deme.py` | **Works**, with core bugs below |
| Selection (CINner) | `components/selection.py` | **Works**, with counting/mapping bugs |
| Genome representation | `components/genome.py` | **Empty stub** |
| Treatment | `treatment/*` | Partial; `expresses()`-based targeting broken |
| Biopsy / slice / dissociation | `sample/*` | **Stubs / empty files** |
| Sequencing + spatial | `data/dna.py,rna.py,visium.py` | **Implemented + tested** |
| CLIs / packaging | `pyproject.toml`, `*/main.py` | **All entry points broken** |
| Docs | `README.md` | Stale (describes old `tumorevo`) |

---

## Fix log — 2026-06-23 (engine-correctness pass)

Sections **A and B are fully resolved**, plus one bug discovered while smoke-testing the
spatial paths. Test suite: **78 passing** (was 65/3-failing); added
`tests/test_engine_regressions.py` (one test per bug) and repaired the 3 stale
`Treatment.get_dosage` tests to the current `(step, tumor_size)` API.

- A1 genome aliasing — fixed (`cell.py`, per-segment comprehension) + regression test.
- A2 driver counting `sum`→`len` — fixed in both `update_genome_summary_mutation` and
  `update_genome_summary_cnv`; CNV wild-type counts now use `len(selection.<cat>[seg])`
  instead of `segment_size` + regression test.
- A3 `nullisomy_count` now counts zero-copy segments + regression test.
- A4 `make_celltype_exps` indexes driver genes directly + regression test.
- A5 `death_rate` mapped to `update_death_rate` + regression test.
- A6 `gene_to_pos` splits on `_` and returns ints + regression test.
- B7 `expresses()` indexing fixed (flat `seg*size+pos`, `seg_mut_effects[seg][pos]`,
  set-membership for mutation status) and made self-sufficient if `baseline_exp` unset +
  regression test.
- B8 immune-cell dispersal: uses `cell`, fixes `'divison_rate'` typo, and now *migrates*
  (removes from source, updates both deme rates) + regression test.
- B9 `Deme.get_genotype_frequencies` rewritten to use `genotypes_counts`.
- B10 `MixedTumor` `super()` fixed (still incomplete: needs single-deme grid — deferred).
- B11 `get_neighboring_demes` off-by-one (`0 <= idx < grid_size`).
- B12 stromal placement: flatten structure points into an `occupied` set.
- **B14 (new)** `glandular.py` created Epithelial/Stromal cells without `segment_size`,
  so structured tumors crashed in `make_cell_data` with a non-default segment size; now
  propagated, and per-deme `deme_rate` accumulates with `+=`. Covered by an end-to-end
  structured-growth regression test.

## Fix log — 2026-06-23 (step 2: CLIs + I/O schema)

Section **C is resolved** and a canonical I/O schema is locked (see `SCHEMA.md`). The
pipeline now runs end-to-end as three composable stages: `isccsim → isccsample → isccdata`.

- All five `pyproject.toml` entry points repointed to `package.module:main`.
- `isccsim` (`tumor/main.py`) rewritten against the config-driven `GlandularTumor` API
  (old stale `MODE_LIST`/positional-`Tumor` version deleted).
- `Tumor.write()` bug fixed — it called a non-existent `get_cell_data()`; now uses
  `make_cell_data()`/`self.cell_data`.
- `isccdata` (`data/main.py`) implemented — loads a sample's `cell_data/` and runs an
  assay (scrna/bdna/scdna/visium); Visium grid side inferred from coordinates.
- `isccsample` (`sample/main.py`) rewritten as a functional sampler (subset + meta);
  removed the broken `ASSAYS`/`sample_regions` references (assays now live in `isccdata`).
- `Visium` constructor params renamed to match `visium.yaml` (`n_spots_x`/`n_spots_y`).
- Added `tests/test_pipeline_cli.py` (sim → sample → data for all four assays). Suite: **84 passing**.

> Note: the package is installed editable, so refresh the console scripts with
> `pip install -e .` (or `poetry install`) to pick up the new entry points; the tests
> invoke the Click commands directly and pass without reinstalling.

## Fix log — 2026-06-23 (performance)

Two behavior-preserving optimizations (output verified **byte-identical**, all tests pass);
see `DESIGN_scalability.md` for the full analysis. ~155 → ~1490 cells/s (**~9.6×**) on a
10×1000-gene, ~3k-cell run (3000-step wall: 19.9 s → 2.0 s).

- **Incremental genotype bookkeeping**: `Tumor.register_birth/death/parent` maintain the
  tumor-level counts/parents on each event, replacing the per-step O(demes × genotypes)
  `Counter` rebuild in `Tumor.update`. ~2.5×.
- **Index-based sampling**: `rng.choice` over the `deme_list`/cell lists rebuilt an object
  ndarray every call (the dominant cost). Sampling an integer index and indexing in draws the
  *identical* random stream ~60× faster. `choice` dropped 3.87 s → 0.24 s.

- **Copy-on-write genotypes** (DESIGN phase 2): clonal cells share one
  `genome`/`genome_summary`; `mutate` copies-first before diverging. `deepcopy` now happens
  only on actual mutations, and genome memory collapses to ~one object per genotype
  (measured 3998 cells → 11 genome objects, **363×**). Per-cell coordinates/deme/label and
  `evolutionary_parameters` stay per-cell. Output byte-identical; 2 COW isolation tests added.

- **Genotype-count engine** (DESIGN phase 3b): new `GenotypeTumor` (`models/count.py`)
  represents demes as `dict[genotype_id -> count]` over a shared genotype registry — no
  per-cell objects; cells are materialized on demand. Built alongside the cell engine
  (additive; all tests stay green), selectable via `isccsim --mode genotype`, writes the
  canonical layout for `isccsample`/`isccdata`. Statistically equivalent to `GlandularTumor`
  (survival 11/12 vs 11/12, survivor sizes 399 vs 398) and reproducible; **~8.5× faster** on
  a matched 10k-gene run (14.7k vs 1.7k cells/s). Not byte-identical by design.

Overall growth throughput: ~155 cells/s (original cell engine) → ~1.7k (optimized cell
engine) → ~14.7k (genotype engine), i.e. ~95× end-to-end.

- **Compact bitset genome** (DESIGN phase 4): each allele copy is a boolean bitset of length
  `segment_size` (was a Python set); copy number is the explicit number of copies. Genome
  ops are vectorized; used by both engines. Fixes the CNV-amplification aliasing and is the
  representation foundation for read/FASTQ-level and allele-specific assays (emitting reads
  is still future work). Modest growth-speed trade for vectorizable materialization.

- **Genotype engine is now the default** (parity work): shared `plot_muller`/`plot_grid`
  (`tumor/viz.py`), structures + normal cells, full `cell_data` incl `cell_evo`; it is the
  default `isccsim` mode and is used by the example config, `generate_example.py`, and
  notebooks 01–03. `GlandularTumor` (cell engine) is retained as the validation reference
  and the opt-in `mode: glandular`.

## Fix log — 2026-06-23 (reproducibility)

`config + seed` now produces a **bit-identical tumor** (verified across separate processes:
matching `cell_snv`/`cell_cnv`/`cell_exp`/`cell_crd` and driver layout). Previously two
`seed=0` runs gave different ground truth.

- `Selection` takes a seeded `rng` (drivers/dispersal/resistance layout); `Tumor` builds
  `self.rng = np.random.default_rng(seed)` and threads it into `Selection`,
  `make_celltype_exps`, and `GlandularTumor`'s structure seeding — replacing global
  `np.random` calls.
- `Deme.cells` changed from a `set` to an insertion-ordered `dict`. A `set` of Cell
  objects iterates in object-id order, which varies across processes, so the seeded RNG
  was sampling cells in a different order each run — the main determinism leak.
- `cell.set_baseline_exp()` still uses global `np.random`, but only as a fallback in
  `expresses()` (treatment-time, not part of the written dataset); deferred to the
  treatment-realism milestone.
- Added `tests/test_reproducibility.py` (same-seed identical, different-seed differs).
  Suite: **86 passing**.

Remaining open items below are Sections E: empty `genome.py`, realistic sampling layer
(spatial biopsy / slicing / dissociation dropout+doublets), assay realism
(NB/dropout RNA, allele-specific DNA, spatial mixing), a validation-against-real-data
harness, and updated docs (README still describes `tumorevo`).

---

## A. Confirmed critical bugs in the agent core

These were reproduced by running code, not just reading it. They silently corrupt the
genome and the fitness summary, so every CNV/SNV/expression/fitness output is suspect.

1. **Genome segments are aliased to one shared object.**
   `cell.py:32` — `self.genome = [{'p':[set()], 'm':[set()]}] * n_segments`. The `* n`
   replicates the *reference*, so `genome[0] is genome[1]` is `True` (verified). A
   mutation or CNV on one segment mutates all segments. This breaks the entire
   per-segment CNV/SNV model. Fix: build with a comprehension creating fresh
   dicts/lists/sets per segment.

2. **Driver counts use `sum()` of positions instead of `len()`.**
   `cell.py` `update_genome_summary_mutation` (lines ~72–90) and
   `update_genome_summary_cnv` (lines ~95–118) do `sum(muts.intersection(selection.onc[seg]))`.
   `muts` is a set of position indices, so `sum` adds the indices together rather than
   counting them (verified: `sum=11` vs `len=2`). Every `n_mut_*`/`n_wt_*` tally — and
   therefore the CINner fitness — is wrong. Fix: `len(...)`.

3. **`nullisomy_count` is set to the max copy number.**
   `cell.py:124` — `self.genome_summary['nullisomy_count'] = np.max(seg_cns)`. Should be
   the count of zero-copy segments, e.g. `int(np.sum(np.asarray(seg_cns) == 0))`. As
   written, the viability check `nullisomy_count > max_nullisomy` is meaningless.

4. **`make_celltype_exps` indexes with `np.where` on an index array.**
   `tumor.py:84-85` — `exp[np.where(self.selection.get_tsgs())]`. `get_tsgs()` already
   returns gene indices; `np.where` of that returns `0..k-1` (verified), so the wrong
   genes get the TSG/oncogene baseline expression. Fix: index directly,
   `exp[self.selection.get_tsgs()] = ...`. (Also revisit the values: TSGs are set to 0.8
   and oncogenes to 0.01, which looks biologically inverted — confirm intent.)

5. **`death_rate` evolutionary parameter is driven by the division-rate fitness.**
   `selection.py:44` maps `'death_rate': self.update_division_rate` in `update_dict`.
   `update_cell_evolutionary_parameters` then overwrites each cell's `death_rate` with
   the division fitness (which ignores the passed `param`). A correct
   `update_death_rate` exists but is unused. Fix: map `'death_rate'` to
   `self.update_death_rate`.

6. **`gene_to_pos` splits on the wrong delimiter.**
   `selection.py:277` splits gene names on `'-'`, but `get_gene_names` builds them with
   `'_'` (`G_{seg}_{pos}`). Verified to raise `ValueError`. Any code path that maps a
   gene name back to coordinates is broken.

---

## B. Other bugs (by inspection)

7. **`expresses()` uses wrong genomic indexing and indexes a set by position.**
   `cell.py:201-218` reads `self.baseline_exp[seg+pos]` / `seg_mut_effects[seg+pos]`
   (should be `seg*segment_size + pos`) and does `self.genome[seg][hap][all][pos]` where
   the allele is a *set* of mutated positions (not position-indexable). This is the
   targeting predicate for **TargetedTherapy** and **Immunotherapy** (`is_target` →
   `cell.expresses(...)`), so molecularly-targeted treatment is currently broken.

8. **`Deme.apply_event` dispersal branch references an undefined `new_cell`** and the
   misspelled key `'divison_rate'` (`deme.py:132-137`). Immune-cell dispersal will raise.

9. **`Deme.get_genotype_frequencies` reads `cell.snv`** (`deme.py:185`), an attribute
   that doesn't exist on `Cell`. Dead/broken helper.

10. **`MixedTumor` skips its own constructor.** `models/mixed.py:9` calls
    `super(Tumor, self).__init__(...)`, which runs `object.__init__` and bypasses
    `Tumor.__init__`. Should be `super(MixedTumor, self)` (or `super()`). MixedTumor is
    non-functional; note `Deme.apply_event` has a `tumor.type == 'mixed'` branch that
    therefore can't be reached correctly.

11. **`GlandularTumor.get_neighboring_demes` off-by-one.** `glandular.py:160-161` uses
    `tup[0] > 0 and tup[0] < self.grid_size`, excluding index 0 and including no guard at
    `grid_size-1` correctly; should be `0 <= tup[0] < grid_size`. Edge demes lose
    neighbors.

12. **Stromal placement membership test is against a list of lists.**
    `glandular.py:140` checks `(row,col) not in structure_circles`, but
    `structure_circles` is a list of per-structure point lists, so the test is always
    True. Stromal cells are mis-placed.

13. **Shallow-copy of `evolutionary_parameters` on divide.** `cell.py:131-136` `divide()`
    does `copy(self)` and deep-copies only `genome`/`genome_summary`; the
    `evolutionary_parameters` dict is shared between parent and child until reassigned.
    Worth making the copy explicit.

---

## C. Broken CLIs / packaging

All five `pyproject.toml` scripts are currently non-functional:

- `isccsim = "iscc.tumor:main"` → `main` lives in `iscc/tumor/main.py`, not exported by
  `iscc/tumor/__init__.py`; the target should be `iscc.tumor.main:main`. Moreover
  `tumor/main.py` is **stale**: it references undefined `simulate_nonspatial`/`MODE_LIST`
  symbols, uses the old positional `Tumor(cancer_cell, selection, ...)` signature (the
  current `Tumor` is config-driven), and has `os.mkdir(exists_ok=True)` (invalid kwarg,
  line 152). It needs a rewrite against the current `Tumor`/`GlandularTumor` API.
- `isccsample = "iscc.sample:main"` → same export problem (`main` is in
  `sample/main.py`), **and** `sample/main.py` references `ASSAYS`/`ASSAY_NAMES`/`sample`
  which live in `iscc.data`, not `iscc.sample`. It also reads `cell_snv.csv` etc. from
  the path root, whereas the tumor writer emits them under a `cell_data/` subdir —
  output/input schema mismatch.
- `isccdata = "iscc.data:main"` → `data/main.py` is **empty** (0 lines). Either implement
  it or drop the entry point. There is conceptual overlap to resolve: assay/data
  generation currently lives behind `isccsample`, while `isccdata` is the
  empty-but-advertised command.
- `isccfig = "iscc.visualization.draw:main"` and `isccgif = "...animate:main"` — these
  point at the right module path and look usable (the `draw.py` Click command is intact).

---

## D. What's genuinely solid (build on these)

- **Spatial growth loop**: `Tumor.grow` / `Deme.update` / Gillespie-ish rate-proportional
  deme selection works and is tested (`test_tumor_growth.py`, `grown_tumor` fixture).
- **CINner-style fitness shape**: `update_division_rate` implements
  `tsg_effect^(2·n/ploidy) · og_effect^(2·n/ploidy)` — the right functional form; it just
  consumes corrupted counts (bug #2) and needs validation.
- **Sequencing assays**: `bulkDNA`, `scDNA` (binary + counts with FPR/FNR), `scRNA`
  (multinomial UMIs), and `Visium` (spot-radius cell pooling + multinomial) are
  implemented and pass `test_data.py`. These are the natural foundation for the realism
  upgrades (negative-binomial/dropout for RNA, allele-specific coverage for DNA, spot
  mixing/segmentation for spatial).
- **Visualization**: Muller plots, spatial grid plots, clone trees in `draw.py`/`util.py`.

---

## E. Missing for the "standard simulator" goal

- **`genome.py` is empty** — no sequence-level representation (referenced in the project's
  own STATUS notes as unfinished gene-segment/haplotype/allele model).
- **Sampling layer is stubs**: `biopsy/solid.py` and `biopsy/blood.py` are empty;
  `biopsy.py`, `slice/slice.py`, `dissociation/dissociation.py` are signatures only.
  This is the missing bridge between the spatial tumor and the assays (regional biopsy,
  physical slicing, dissociation-induced dropout/doublets).
- **RNA realism**: current scRNA is a plain multinomial — no overdispersion
  (negative-binomial), dropout, library-size variation, ambient RNA, or doublets.
- **DNA realism**: no allele-specific copy number, no read/FASTQ mode (raises
  `NotImplementedError`), uniform coverage model.
- **Treatment realism**: dosing/scheduling is thin; pre/post longitudinal output isn't a
  first-class artifact; targeting is broken (bug #7).
- **No validation harness** comparing simulated summary statistics to real data, and **no
  stable, documented config + output schema** for external method developers to depend on.
- **Docs/packaging**: README still documents `tumorevo`/`tumorsim`/`tumorfig`.

---

## Fix log — selection model & evolutionary dynamics

Building the evolutionary-dynamics validation exposed three coupled bugs in the CINner-style
selection model; all fixed (suite: 92 tests).

- **Fitness blow-up.** `effect^(2·n/ploidy)` used the raw count of category gene *copies*
  (hundreds), so e.g. `dispersal_rate = 1.1^196 ≈ 1.3e8`, driving the mutation probability to
  ~0 (no diversification) and overwriting the configured baseline rates. Fixed by making
  fitness **relative to the all-wild-type diploid baseline** (computed in log space): the
  baseline genome is neutral (factor 1) and only deviations shift it; division/dispersal are
  applied as a clamped multiplier on the configured baseline rates, and resistances map a
  relative fitness ≥1 into [0,1).
- **Genotype-id collisions.** `genotype_id = str(id(self))` recycled ids after dead cells were
  garbage-collected, producing "genotype is its own parent" once mutations became frequent.
  Replaced with a process-unique monotonic counter.
- **No-op mutations.** `mutate()` returned early (without a new id) when an allele was
  infinite-sites-saturated, leaving the child with the parent's id. `mutate()` now returns
  whether it created a new genotype; callers treat a no-op as a same-genotype division.

Validated by `validation/validate_evolution.py` (+ `tests/test_evolution.py`): sweeping
dispersal reproduces the Noble (2022) mode gradient — diversity Shannon 5.8→2.7 and spatial
assortment 0.21→0.61 as dispersal increases.

## Known modeling issues (deferred to the realism milestones)

- **Copy number could go negative.** ✅ FIXED (2026-06-29). `CancerCell.mutate` deletion branch
  had a `len==1` special case that *kept* the last allele (zeroed it) yet still passed `sign=-1`
  to `update_genome_summary_cnv`, so `seg_cns` drifted **below 0** under repeated deletions.
  Fixed by always removing the copy (allowing nullisomy at CN 0, which the viability check
  handles), keeping `seg_cns` equal to the actual allele count. Also added a guard so `mutate`
  returns `False` (no-op) on a fully-deleted genome instead of dividing by an all-zero
  segment-selection weight (`0/0 → NaN` in `rng.choice`). Surfaced only once the example tumour
  actually produced CNAs (below); previously masked.

- **Example config produced a cancer-free tumour.** ✅ FIXED (2026-06-29). The shipped
  `notebooks/example_config.yaml` had `carrying_capacity=1` + a `structure_radius=4` duct, so the
  single cancer founder was boxed in by normal cells and went extinct (or could not expand) —
  runs yielded ~588 cells but **`cancer=0`** (pure normal tissue, no CNAs/SNVs), which is why the
  F4/F5 assay notebook saw a diploid, mutation-free sample. Root causes: (a) a single founder has
  `P(extinction) ≈ death/division ≈ 7%` regardless of `cc`; (b) `cc=1` left no room to grow. Fixed
  by seeding an established micro-lesion (new additive engine param `deme_params.initial_cancer_cells`,
  default 1) and updating the example config to `cc=5`, `structure_radius=5`,
  `initial_cancer_cells=5`, `mutation_rate=0.4`. Now produces ~1000 cancer cells with CNAs (CN
  0–7) and SNVs across all seeds (incl. seed 42), within the duct. The engine's evolutionary
  dynamics were already validated — but only in free-growth (`structure_radius=0`, `cc≥5`) configs;
  the duct + `cc=1` demo path had never been exercised end-to-end.

- **Immune-effect formula is degenerate/inverted.** ✅ FIXED (2026-06-25). The old
  `cell_death_rate * (immune_cell_fraction ** immune_resistance)` could only *lower* the death
  rate and gave `0 ** r = 0` (immortal cancer) in immune-free demes. Replaced in both engines
  (`Deme.get_cancer_death_rate` and `GenotypeTumor._death_rate`) with **additive contact
  pressure**: `death = min(base*crowd, max) + prob_kill * immune_fraction * (1 - immune_resistance)`.
  More local immune cells now raise cancer death, attenuated by immune resistance; `ImmuneCell.prob_kill`
  is now used. The no-immune case reduces to baseline death (so the evolution/SNV validations,
  which seed no immune cells, are unchanged). The genotype engine also gained optional immune
  seeding (`spatial_params.immune_density`) so immunotherapy has a substrate. Immune *dynamics*
  (recruitment/migration/exhaustion) remain future work — the compartment is currently static.

- **Treatment now runs on the default genotype engine.** ✅ DONE (2026-06-25). `GenotypeTumor.grow`
  accepts `treatment=`; per-step `_apply_treatment` turns the cell-level stochastic effect into a
  dose-scaled, per-clone rate modifier (`_tx_death_add` / `_tx_immune_resist`) that never mutates
  the shared genotype. Chemo/targeted add a kill hazard (default `kill_rate=1.5`, above the
  `max_birth_rate` cap so even driver-amplified clones regress); immunotherapy strips immune
  resistance. Also fixed `TargetedTherapy.__init__` (`super(Treatment,...)` skipped init) and the
  inverted `Immunotherapy._apply` (multiplied resistance up; now strips it down). Exposed via
  `isccsim --treatment {chemo,targeted,immuno} [--adaptive ...]`. Validated by
  `validation/validate_treatment.py` + `tests/test_treatment_engine.py`.

## Suggested fix ordering

1. **Engine correctness (Section A + B)** — start here; everything inherits these.
   Add regression tests for genome independence, driver counting, nullisomy, and the
   death-rate mapping.
2. **Unbreak CLIs + lock an I/O schema (Section C)** — makes the pipeline runnable
   end-to-end and gives a contract to test against.
3. **Sampling layer (Section E)** — implement biopsy/slice/dissociation as the
   tumor→assay bridge.
4. **Assay realism (Section E)** — NB/dropout RNA, allele-specific DNA, spatial mixing.
5. **Validation + docs/packaging** — the adoption layer.
