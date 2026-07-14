# DESIGN — carrying-capacity / crowding is not enforced (engine bug + fix) [design-first]

Status: **diagnosed 2026-07-09, fix not yet built.** A real engine bug found while measuring iscc's
spatial scaling vs SISTEM: `carrying_capacity` does not actually cap a deme's occupancy, so at scale
the tumour is a **dense pile in a few demes** rather than a spatially spread tumour. This undermines the
spatial-structure realism and any spatial-scalability comparison. Companion: `PARAMETERS.md` (caveat),
`BACKLOG.md`, the operating-envelope docs (a QC check).

## Symptom (measured)
Grow a tumour on a large grid (K=10, grid 400²=160k demes). At millions of cells the tumour occupies
only ~900–1,400 demes → **~1,200–4,200 cells per deme** (nominal capacity 10). Occupied demes grow
far slower than cell count, i.e. cells pile in place instead of spreading. Same with the step-function
crowding and with a logistic-death prototype.

## Root cause (verified, not theory)
The crowding death is a **fixed absolute rate**:
```
crowd = carrying_capacity if total > carrying_capacity else 1.0     # step, not continuous
death = min(death_rate * crowd, maximum_death_rate)                  # cell.py / count.py _death_rate
```
so above capacity `death = min(death_rate·K, max_death_rate)` — a constant ≤ `max_death_rate` (0.5 by
default; 0.3 with the shipped `death_rate=0.03`, K=10).

But **selection raises each clone's division rate** (CINner fitness) up to `max_birth_rate` (0.8).
Measured: population-mean `division_rate` climbs 0.30 → **0.78** over 30 generations (99% of cells
above 0.30), while the death cap stays 0.30. Once evolved `division_rate > death cap`, net growth is
**positive regardless of occupancy** → unbounded overfill, worsening as the tumour evolves. (The
births-vs-deaths draw ratio measured 0.37 = 0.30/0.80 exactly — deaths are drawn correctly at the cap;
births are simply higher.)

**Why the obvious patches fail:** a logistic death `base + (div−base)·(N/K)` and a larger `death_rate`
were **both clamped by `max_death_rate` (0.5) below the evolved division rate (0.8)** — the clamp
defeats them. A carrying capacity expressed as an absolute death rate cannot regulate a population
whose birth rate evolves above it.

## The fix (options)
Density-dependent death is the right mechanism (Noble / deme models) — it just has to be **relative to
each clone's own, evolving division rate, and not clamped below it**:

- **A — relative logistic death (recommended, minimal).** Above capacity make death exceed the clone's
  *current* division rate with a restoring slope, e.g. `death = base + (division_rate(clone) − base +
  margin)·(N/K)`, and **raise/remove `maximum_death_rate`** so it can reach ≥ `max_birth_rate`. Gives a
  stable per-deme fixed point near K even for fast-evolved clones. One-line-ish change in `_death_rate`
  + a config default change (`maximum_death_rate ≥ max_birth_rate`).
- **B — hard occupancy / boundary-driven (most spatially faithful).** A cell divides only into a deme
  with free space (in-place or a non-full neighbour), else the birth is blocked. This is the classic
  spatial-tumour-ABM mechanism (Waclaw 2015, Noble 2022) that yields boundary-driven growth and crisp
  clonal territories. Bigger change (rework the birth/dispersal branches in `update` + `_tau_substep`).

Recommend **A** first (repairs the death regulation the user identified, least invasive), with **B** as
the more faithful longer-term option. Either way: **off-by-default / gated** and re-validate, because
changing crowding changes every spatial output.

## Impact (why it matters)
- **Spatial realism:** with a real cap the tumour spreads (occupied demes ∝ cells/K) → genuine
  territories + boundary-driven growth. Strengthens the spatial-evolution story.
- **Scalability claims:** the defensible SISTEM comparison is **well-mixed only** — in the single-deme
  regime iscc grows 5M cells in **under 3 min** (`occ-demes = 1`; `benchmark_scalability.py --tau-grid 1`),
  reaching SISTEM's scale (5M in ~20 min); this claim is in the manuscript scalability section. Do
  **not** claim a *spatial* speed advantage: the "millions of cells in minutes" numbers were the
  non-spreading pile, and a genuinely spatial tumour at that scale is HPC-bound (cost ∝ occupied demes).
- **`carrying_capacity` semantics:** the config comment ("cells per deme") is not what the dynamics do.
- **QC:** add a `tumor.diagnose()` check for *demes over-filling* (mean cells/deme ≫ `carrying_capacity`).

## Existing results
Likely **unaffected**: PEtracer and multi-region run on small grids where the tumour fills the lattice
and territories come from *low dispersal* keeping clones spatially coherent (a real mechanism,
independent of the density bug). **Re-check** those small-scale runs after the fix. Not fatal, but the
density/`carrying_capacity`/size-realism aspects and all large-scale spatial fidelity need the fix.

## Prototype validation (2026-07-14) — Option A confirmed
A scratch prototype of Option A (density-dependent death `base + (division_rate(clone) − base)·(N/K)`,
with `maximum_death_rate = 3.0 ≥ max_birth_rate`) on the spatial config (grid 400², K=10, dispersal 0.1):
demes **cap near K** (mean **8.9**, max ~20 cells/deme, vs 1,200–4,200 broken) and the tumour **spreads**
(occupied demes ∝ cells/K: **15,346 demes at 137k cells**, vs ~900 broken). This confirms the death-based
fix works once death is relative to the clone's evolved division rate and the clamp is raised. It also
pins the true spatial cost: **~137k cells in 150 s (~900 cells/s, decreasing)** — a spatial 5M-cell
tumour is hours (HPC-bound), which is the honest scaling story.

## Validation when built
Demes cap near K (mean cells/deme ≈ K); occupied demes ∝ cells/K; boundary-driven growth (rim divides,
interior static); PEtracer/multi-region confounds still reproduce; add the `diagnose()` over-fill check.
