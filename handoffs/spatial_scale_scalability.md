# Handoff: simulate tumors at spatial-assay scale (cm-scale tissue, ≤5 min)

## Goal
Grow a tumor **bigger than a Visium capture area** (≥ 6.5 mm ≈ **grid 260–325** at deme 20–25 µm)
in **≤ 5 minutes**, so that spatial (Visium) and molecular (scRNA / scDNA / bulk) assays each sample
**subsets** of one tissue — even if a biopsy removes the whole tumor. It must be **one shared
simulation** used by the landing-page GIF *and* every tutorial (no special-case sim for the spatial
demo), it must still show the **selective pressures** we care about (CINner clonal fitness), and all
assays on it must be realistic.

## Why this is blocked today (measured this session)
The count-based `GenotypeTumor`'s per-step work scales with the number of **distinct genotypes**,
which explodes under infinite-sites + high mutation:

| grid | tumor @25µm | cells | distinct genotypes | grow time |
|---|---|---|---|---|
| 45 | 1.1 mm | 15,690 | **7,095 (~45% of cells)** | 12 s |
| 60 | 1.5 mm | 54 k | — | 42 s |
| 90 | 2.25 mm | 175 k | ~80 k (est.) | **497 s (8.3 min)** |

Growth is **super-linear**: 1.5× grid → 12× time. A >6.5 mm tumor (grid ~260) would take **hours**.
`example_config.yaml` already uses tau-leaping (`update_mode: tau, tau 0.5`), so this is *with* the
fast path. Root cause: `mutation_rate 0.6` makes almost every cell a unique (passenger-differentiated)
genotype, and the hot loop is per-genotype-per-deme.

## Key insight / recommended direction
**Birth/death rates depend on FITNESS (drivers, copy number), not on neutral passenger mutations.**
So the *dynamics* only need to distinguish **fitness classes** (few), while the thousands of
passenger-differentiated genotypes are neutral labels. Decouple them:
- Run the hot loop (deme rates, birth/death genotype selection) over **fitness classes**, not raw
  genotypes → the loop size stops growing with the tumor.
- Track passenger identity **lazily** (attach on birth of a *new* fitness class, or reconstruct from
  the genealogy at assay time) so per-cell genotype output is still available for `make_cell_data`.

Complementary levers (profile first to rank them):
- **Prune** extinct / sub-threshold genotypes from per-deme structures each step.
- **Coarsen**: bin genotypes by fitness class for the dynamics.
- **Fenwick / alias sampling** over deme and within-deme rates (already in `DESIGN_scalability.md` §3).
- Sparse per-deme genotype structures.

Start by **profiling grid 90** to confirm the hot loop is per-genotype iteration (not `choice`,
not deme count) before choosing the fix.

## Constraints / gotchas
- **≤ 5 min** to grow a ≥ 6.5 mm tumor.
- **Assay memory at scale**: `make_cell_data` materializes per-cell tables. 175 k cells is already
  ~100 MB; cm-scale is millions of cells → assays (Visium section-slice, scRNA/scDNA `n_cells`,
  bulk pool) must subsample **without materializing all cells**. This is a second scalability front.
- **Physical scale** (audited this session, from `DESIGN_ductal_field.md` §3): a deme is a 3-D
  column, K is the depth population, the grid must be **fine relative to the 55 µm spot** (a spot
  covers *several* demes), a gland = a few demes. Anchor **deme ≈ 20–25 µm** → cell ~12 µm, gland
  ~120–150 µm, section = thin slice → **~7 cells/spot** (a handful). Visium capture 6.5 mm =
  ~260–325 demes. This audit is *why* cm-scale grids are needed.
- Preserve **selection**: after speeding up, confirm clonal selection / driver sweeps are still
  visible (don't kill diversity so hard that selection disappears — the "trade diversity for size"
  shortcut was explicitly *not* chosen).
- **One shared config**: re-scale `notebooks/example_config.yaml` (shared with the landing GIF via
  `isccgif` and all tutorials); nothing tutorial-specific.
- Commit on `dev` with the `Co-Authored-By: Claude Opus 4.8` trailer.

## Already done (this session, on `dev`, UNCOMMITTED — the *assay* side is ready)
The Visium/spatial assay is already spatial-realistic; the milestone is purely the **engine** (grow
big fast) + **assay memory at scale**:
- `src/iscc/sample/section.py` (new): `spatialize()` (place deme cells + take a thin section) and
  `tissue_image()` (H&E-like morphology raster). Shared utility — MERFISH/`imaging.py` and the
  landing GIF's `_expanded_cell_grid` can adopt it (imaging.py currently stacks cells per deme).
- `src/iscc/data/visium.py`: `section_frac` (thin-slice sampling per §3.1), fixed **Visium v1 slide**
  (78×64 = 4,992 spots, `run(grid_side=None)`), `section_image()` preview, squidpy-native
  `to_anndata` (obsm (x,y), `uns['spatial']` scalefactors + tissue image, `in_tissue`, `total_counts`).
- `src/iscc/tumor/viz.py`: cell_type-panel fix (colour demes by dominant *coarse* type).
- `notebooks/03_data_overview.ipynb`: Visium section rewritten (fixed slide, spot-level, pure squidpy)
  — currently on the small 0.5 mm tumor; it will produce a rich slide once the engine can grow big.
- **Decide before starting**: commit this assay-side work to `dev` first so the engine session builds
  on it cleanly. (Also delete the temp `mkdocs.preview.yml` at repo root — a local preview artifact.)

## Deliverables
1. Profile grid 90 → confirm the per-genotype hot loop.
2. Engine change decoupling dynamics from #genotypes (fitness-class dynamics + lazy passengers, or
   pruning/coarsening) → grow **≥ 6.5 mm in ≤ 5 min** with selection intact.
3. `make_cell_data` / assays memory-safe at cm-scale (subsample, don't materialize all cells).
4. Re-scale the shared `example_config.yaml` (deme ≈ 20–25 µm, grid ≥ 260) and re-run the landing GIF
   + all tutorials; verify the Visium tutorial now yields a well-covered slide.
5. Tests + validation (growth correctness, selection still visible, assay outputs).

## Key files & references
- `src/iscc/tumor/models/count.py` — `GenotypeTumor` (the count engine + tau-leaping). Hot path here.
- `DESIGN_scalability.md` — §2 genotype-level state, §3 Fenwick/alias sampling, §4 expected scaling.
- `DESIGN_ductal_field.md` §3 / §3.1 — the deme-column physical model + section-slice assay.
- `notebooks/example_config.yaml` — shared config (grid_size, mutation_rate 0.6, carrying_capacity 20,
  K_duct 40, K_stroma 26, update_mode tau).
- Memory: `iscc-scalability-decisions` (no JAX; genotype-count engine default ~95×; bitset genome;
  copy-on-write; tau-leaping §7), `iscc scalability` roadmap.
