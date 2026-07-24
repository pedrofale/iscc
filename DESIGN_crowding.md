# DESIGN — carrying-capacity / crowding is not enforced (engine bug + fix) [DONE]

Status: **DONE 2026-07-14 (Option A shipped, all results re-baselined).** A real engine bug found while
measuring iscc's spatial scaling vs SISTEM: `carrying_capacity` did not actually cap a deme's
occupancy, so at scale the tumour was a **dense pile in a few demes** rather than a spatially spread
tumour. Fixed by making crowding death density-dependent **relative to each clone's own evolved
division rate** (Option A), with `maximum_death_rate` raised to ≥ `max_birth_rate`. See
**§ Implemented (2026-07-14)** at the bottom for the final formula, the well-mixed option, and the
re-validation outcomes. Companion: `PARAMETERS.md`, `BACKLOG.md`, `DESIGN_operating_envelope.md`
(the over-fill QC check).

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
- **The right SPATIAL-regime runtime comparison is Noble's `demon`** (deme-based spatial tumour simulator,
  `noble_spatial_2022`; robjohnnoble/demon_model — verify the software citation), NOT SISTEM. demon is
  deme-structured like iscc's engine, so it is the apt head-to-head for cells-per-second in a genuinely
  *spatial* (many-occupied-deme) tumour — whereas SISTEM was the well-mixed comparison. iscc already
  validates its evolutionary-mode indices against Noble (M3a), so demon is the natural runtime baseline
  too. **TODO when we target the spatial-scale / HPC issue** (BACKLOG "M3b HPC rerun"): benchmark iscc
  vs demon on the same spatial config (grid, K, dispersal), report cells/s and time-to-N, and state the
  regime honestly. User request 2026-07-17.
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

## Implemented (2026-07-14) — Option A shipped
**Final formula** (both engines: `count.py::_death_rate` and `components/deme.py::get_cancer_death_rate`,
verified identical by `test_engines_agree_on_crowding_death`):
```
slope = max(0, division_rate(clone) − death_rate) · (1 + crowding_margin)      # crowding_margin default 0.1
death = death_rate + slope · (occupancy / carrying_capacity)                    # occupancy = total cells in deme
death = min(death, maximum_death_rate)                                          # maximum_death_rate default 1.0 (≥ max_birth_rate)
# then the existing immune + treatment terms are added on top, unchanged
```
Death rises **relative to the clone's own evolved division rate**, so the per-deme fixed point sits at
`occupancy = K/(1+margin)` (death == division there) and death > division above it — a restoring force to
the cap for **any** evolved division rate. The small `crowding_margin` (0.1) steepens the slope so the cap
is firm rather than marginally stable exactly at K, and puts the mean occupancy just below K. The
`maximum_death_rate` clamp is now **≥ `max_birth_rate`** (default raised 0.5 → **1.0**) so the clamp can
never sit below an evolved clone's division rate (which re-opened the bug).

**Well-mixed option (preserves the SISTEM benchmark).** `carrying_capacity = None` (or `0`) sets
`self._crowding = False` → no crowding term → **unbounded growth** in a deme. This is the explicit
replacement for the old `carrying_capacity = 1` "no ceiling" hack (which now genuinely caps a deme at
~1 cell). `benchmark_scalability.py` uses `carrying_capacity=None`; the single-deme SISTEM claim
**re-measured 2026-07-14: ~5M cells in ~2.5 min** (`--tau-grid 1`, mut 0.01) — still under 3 minutes.

**Founder seeding.** Because crowding death ramps up from occupancy 0, a lone founder in a small-K deme
is extinction-prone; both engines now seed `initial_cancer_cells` identical founder clones (capped by K),
and the cell engine mirrors this (`GlandularTumor._seed_founders`).

**Re-tuned shipped defaults.** `notebooks/example_config.yaml` → grid 50, K 10, `structure_radius` 20,
`maximum_death_rate` 1.0: grown with tau it fills the gland to **~10,300 cancer cells across ~1,260 demes,
mean 8.2 cells/deme**, `diagnose()` passes (not extinct, clonal diversity present, not over-filling).
`tumorconfigs/{glandular,mixed}.yaml` → K 5 (K=1 would now go extinct), `maximum_death_rate` 1.0.
Inference configs → `maximum_death_rate` 1.0.

**Measured cap** (Option A, K=10 grid 25, tau): mean **~8** cells/deme, no runaway pile (was
1,200–4,200/deme broken); tumour spreads (occupied demes ∝ cells/K).

**Re-validation outcomes.**
- Full `pytest` suite green (growth-dependent fixtures deliberately re-baselined; added
  `test_demes_cap_near_carrying_capacity`, `test_well_mixed_disables_crowding`,
  `test_engines_agree_on_crowding_death`).
- **PEtracer confound HOLDS** (`validate_petracer.py`): extrinsic genes mis-called heritable **100%** at
  low dispersal (clonal territories) → **32%** at high dispersal (resolved); clean monotonic sweep.
- **Multi-region spurious parallelism HOLDS** (`validate_multiregion_phylo.py`): naive region-tree
  spurious rate **0.219** vs deconvolved **0.004**; "more regions doesn't fix it"; admixture-correlated.
- **Operating envelope** phase diagrams regenerated (`validate_operating_envelope.py`); added the
  `diagnose()` **over-fill** check (mean cells/deme ≫ K) — now a passing check across the map.

**QC.** `tumor.diagnose()` gained a `deme_occupancy` metric and an `overfilled` check (fails if mean
cells/occupied-deme > 3×K; skipped in the well-mixed regime) — see `DESIGN_operating_envelope.md`.

---

# ADDENDUM — resident-pressure invasion gate + the "finger" regime [2026-07-20]

Status: **engine change shipped, full `pytest` green (569).** Option A made the equilibrium deme
density *independent of fitness* (its whole point: cap overfill for any evolved division rate). A side
effect surfaced while trying to get **fitness-gated invasion of a gland**: because crowding death is
`base + (div − base)(1+m)·occupancy`, the survival condition `net = div − death > 0` reduces to
`occupancy < K/(1+m)` — the `(div − base)` factor **cancels**, so *survival under crowding is
fitness-independent*. A cancer cell dispersing into a gland deme (occupancy ≥ 1 from the immortal
epithelial residents) therefore has `net < 0` **regardless of fitness**, and invasion is by dispersal
pressure alone, not selection (measured: invaded cells no fitter than core, Mann–Whitney p = 0.76).

## The fix (two-term crowding death)

Split the crowding death by the crowder type — keep Option A for cancer-on-cancer, add a **fixed-
reference** term for the immortal normal residents:

```
death = base
      + (div − base)·(1+m)·(n_cancer / K)     # Option A: caps cancer density, fitness-independent, no overfill
      + (ref − base)·(1+m)·(n_normal / K)      # resident pressure: FIXED ref (not div) => a fitness threshold
```

Because the resident term uses a fixed `ref` (default = the founder cancer division rate,
`deme_params["resident_pressure_ref"]`), it does **not** cancel in the survival condition:

```
net > 0  ⇔  div > base + (ref − base)(1+m)(n_normal / K)
```

a genuine **fitness threshold** rising with the resident count. Only fitter cancer survives (and then
disperses onward from) a normal-occupied gland deme; **normal cells never die** (still static). A
cancer-only deme has `n_normal = 0`, so the term vanishes and every `structure_radius = 0` result is
**byte-identical** to Option A (verified against the stashed original). Implemented identically in both
engines (`count.py _death_rate`, `deme.py get_cancer_death_rate`); `resident_pressure_ref` also plumbed
through `Deme.__init__`. Verified: invaded cells now significantly fitter than core (p < 1e-4).

## Localized invasion needs RARE dispersal, not just the gate

With the gate on, the *whole* gland ring still dissolves at once, because CINner + spatial selection
drives the **entire front to `max_birth_rate`** (fitness saturates ~0.98), so every boundary clone
clears the gate. Localization comes from making **dispersal rare** (a clone crosses only if it also
acquires a dispersal mutation): **low baseline `dispersal_rate` + small `prop_dispersal` + low
`mutation_rate` + strong `dispersal_effects`**. Then only the rare clones that acquire dispersal punch
**localized fingers**; the low-dispersal bulk is held inside the ring.

Caveat (known, accepted): dispersal is **undirected diffusion** (daughter → uniformly-random
neighbour) and is crowding-neutralised inside the packed tumour, so a high-dispersal clone
random-walks around **wherever it arose** (often central) rather than migrating to the front; the
fingers are its escapees across the ring. Directional (free-space-biased) dispersal is a possible
future change, deliberately NOT made here.

## Rule of thumb — population size vs `structure_radius` (the finger regime)

The confined bulk fills the gland interior — a disk of radius `R = structure_radius`, ≈ `π·R²` demes,
each holding ≈ `c·K` cancer cells (dispersal influx over-packs the cap by `c ≈ 1.6`, measured; `c`
drifts ~1.9→1.6 as K goes 4→16). So the gland's cancer capacity is

  **N_gland ≈ π · R² · c · K   (c ≈ 1.6)  ⟹  R ≈ √( N / (π · c · K) )**

This is **linear in the deme size `K`** (verified: at fixed `R=20`, confined cells 8,070 / 15,440 /
31,984 for K = 4 / 8 / 16 — doubling with K). `K` therefore sets **spatial resolution, not physical
size**: a deme holding `K` cells is `≈ √K` cell-diameters across, so the physical gland radius is
`r ≈ R·√K·d_cell` and the physical cell count `N ≈ π·(r/d_cell)²` is *independent* of `K`. The same
physical DCIS duct (radius `r`, `N` cells) is reproduced by any `(K, R)` with `R·√K = r/d_cell`
(equivalently `π·R²·K = N`) — a genuine degeneracy. The DCIS mapping: a distended duct ~1–2 mm across,
breast cells ~10–20 µm ⟹ `N ≈ 4,000–15,000` cross-section cancer cells; `base_sim` uses `K=8, R=20`
(≈ a ~1.8 mm duct).

- `R ≪ R*` : gland can't contain N → demes overfill, overflow is large → **broad, whole-ring** invasion.
- `R ≈ R*` : bulk fills the gland (~K–1.3K/deme) with **modest overflow → localized fingers**.
- `R ≫ R*` : tumour never fills/pressures the ring → **no breach**.

Verified (target N ≈ 10 000, K = 8, so `R* ≈ √(10000/(π·8)) ≈ 20`):

| `R` | `π·R²·K` | invaded % | cells / gland-deme |
|----:|--------:|----------:|-------------------:|
| 12  | 3,619   | 19%       | 18.0 (overfull)    |
| 16  | 6,433   | 13%       | 11.4               |
| 20  | 10,053  | 9%        | 12.3 (fills nicely)|
| 24  | 14,476  | 9%        | 10.3               |
| 28  | 19,704  | 10%       | 10.8               |

Below `R*` the interior is overfull (18 vs K=8) and overflow doubles (19%); at/above `R*` the bulk is
contained and overflow settles to ~9–10% (localized fingers). This `R ≈ √(N/(π·c·K))` sizing is the
design rule for the showcase `base_sim` (N ≈ 10k, K = 8 → `structure_radius = 20`).
