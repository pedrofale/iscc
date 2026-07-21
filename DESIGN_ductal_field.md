# DESIGN — ductal-field spatial substrate (multi-focal DCIS→IDC, island model) [design-first]

Status: **DESIGN-FIRST (2026-07-18), NOT built.** The spatial substrate the compartment-selection model
(`DESIGN_phenotype_plasticity.md` §2) runs on. Motivated by making the structured case anatomically
reasonable AND usable for spatial transcriptomics. **No engine code until sign-off.**

## 0. The problem with the current geometry
`count.py:_seed_structure` seeds ONE central epithelial ring — a single "gland" that, at realistic deme
counts, becomes thousands of cells in a cross-section, far larger than a real duct. Real breast tissue is a
**branching ductal tree of many small ducts** in stroma; multi-focal DCIS is a **single clone spreading
through connected glands**, so a section shows multiple *evolutionarily related* foci. We need that, and we
need it to look like a real section because spatial transcriptomics (ST) is a target — and **ST is the one
modality where you cannot abstract space away** (space is the signal).

## 1. The model: glands as islands (option C)
Represent the ductal system as a **population-genetics island model**: many small glands at 2D positions,
fast *local* growth, and a *low* cross-gland migration rate standing in for intraductal spread through the
(out-of-plane) ductal tree. Two spread routes, cleanly separated:

- **Intraductal (DCIS, confined):** a low **cross-gland dispersal** rate seeds one gland's lumen from
  another's — no stroma, **no breach**. This produces the related multi-focal foci. The connecting ducts are
  out of plane and simply *not drawn*, which is exactly what a real multi-focal section looks like (separate
  foci, no visible in-plane connection). No graph, no explicit edges — the *rate* is the connectivity.
- **Invasive (IDC):** the existing **cross-deme** (neighbour) dispersal; crossing the epithelial wall into
  stroma is gated by the `breach` trait, surviving/traversing stroma by `stromal_survival`
  (`DESIGN_phenotype_plasticity.md` §2).

A **single founder** in one gland then yields related foci for free; the cross-gland hops *are* the
inter-gland spread tree (ground truth), giving a phylogeography benchmark (reconstruct the spread history /
focus relationships from a section) with a realistic truncal-plus-divergence clonal structure (island-model
bottleneck at each hop). This is the multi-region-phylogeny confound in a DCIS setting with a known answer.

## 2. Geometry (seeding)
- **N glands** at 2D positions (random scatter, min-distance apart, grid interior). Each gland is a **small
  epithelial ring** (radius ~2–3 demes: lumen inside, epithelial wall on the circumference), seeded to
  `K_duct` on the wall; lumen demes start empty (DCIS-growable). Multiple demes per gland ⇒ within-gland
  neighbour dispersion (the requested intra-duct spatial effect).
- **Stroma** fills the rest, seeded **sparse** (low baseline occupancy of a higher-headroom deme — normal
  stroma is acellular; invasion fills it into a dense mass).
- **Per-deme `gland_id`** label (which gland, or stroma) recorded at seeding — needed for cross-gland
  targeting and for the compartment hazards / ground truth.
- **One cancer founder** in one gland's lumen.

## 3. Carrying capacity (K captures the duct's 3D depth — NOT a handful)
A 2D deme stands for a 3D **column** of tissue (a cross-section point through the duct's/stroma's depth), so
its K is the number of cells in that column — a real subpopulation size (Noble's deme-K), **moderate-to-large,
not a handful**. Each duct is therefore *a few demes* (2D within-duct structure) × *a moderate K* (the depth) ≈
a realistic **3D** duct volume; the resulting "thousands of cells per duct" is correct for a 3D duct, not the
cross-section over-count it would be if K were the whole cell budget.
- `K_duct`: **moderate** (captures depth); a duct is a small ring of a FEW demes (lumen + wall) so within-duct
  neighbour dispersion still works. Not a handful of cells per deme.
- `K_stroma`: similar / modestly higher (bulk), seeded **sparse** (acellular normal stroma; headroom for a
  dense invasive mass).
- **ST resolution is a GRID-SPACING constraint, NOT a K bound** (this corrects an earlier note): a Visium spot
  must cover **several demes** to be a mixture (for deconvolution / sub-spot structure), set by grid density vs
  spot radius — *independent of per-deme K*. K is the depth population the assay samples; large K just gives
  more (representative) cells per location. So: grid fine relative to the spot, K moderate-large for depth +
  sequenceable per-location populations, TOTAL kept sane via field size. Large K is nearly free at growth (the
  count engine tracks genotype COUNTS per deme); it only costs at `make_cell_data` / assay, where you subsample.

## 4. Dispersal (engine)
One free dispersal event as today (a fraction of divisions send a daughter elsewhere; no capacity gate —
crowding is via death). Split the dispersing daughters into two channels:
- **cross-deme (local):** uniformly-random neighbour deme (unchanged, `count.py:734`).
- **cross-gland (island):** with weight `κ` (config, <1), a cell **currently in a gland** instead seeds a
  **lumen deme of another gland**, chosen distance-weighted (prob ∝ e^(−d/λ) over gland centres; λ one knob)
  or uniformly. Confined ⇒ lands in lumen, no breach. Rate scales with the heritable dispersal trait:
  `cross_gland_rate = κ · dispersal_rate` — **no new gene axis**, one global knob (κ; plus λ if distance-
  weighted). *(2D-section distance is only a proxy for ductal-tree proximity — the tree winds through 3D — so
  distance-weighting is itself approximate; uniform is the zero-extra-parameter fallback.)*

## 5. Interaction with the compartment hazards (`DESIGN_phenotype_plasticity.md` §2)
- **Epithelial barrier = live cell fraction** (`epithelial_fraction(deme)·(1−breach)`): the wall *is* the
  epithelial cells; it dynamically dilutes as cancer crosses (the composition-based, no-fixed-label
  requirement). Cross-gland (island) dispersal **bypasses** the wall (lumen→lumen), so it needs no breach —
  the wall confines *local* escape into stroma, not intraductal spread. That is exactly DCIS biology.
- **Stromal hazard = a stromal-region FIELD**, not the live stromal-cell fraction: normal stroma is seeded
  sparse, so a fraction-based term would be too weak; the stromal hostility is environmental (immune/hypoxia/
  ECM/no niche support), so key it to the stromal region (F8-style), `(1−stromal_survival)`.

## 6. Scope / staging
- **v1:** the geometry (multi-gland island field), the cross-gland dispersal channel + `gland_id` labels, the
  low-K / sparse-stroma capacities. The `GenotypeTumor` (count) engine only (default). Off-by-default: N=1
  gland + κ=0 recovers the current single-structure behaviour (byte-identical when the field is off).
- **Not planned:** an explicit ductal graph / drawn connecting ducts (option B — the island rate replaces it);
  a full 3D engine; normal-cell clearance (`DESIGN_phenotype_plasticity.md` §2 — the barrier selects on
  presence, not removal).

## 7. Open decisions
- Cross-gland targeting: distance-weighted (λ) vs uniform (recommend distance-weighted, small λ).
- Whether a gland is a ring-of-demes (lumen+wall, resolves intra-gland structure) or a single low-K deme
  (simpler, no within-gland space) — **recommend ring-of-demes** (the user wants within-gland spatial
  effects, and it cleanly separates lumen=DCIS-growable from wall=barrier).
- Realistic N, K_duct, K_stroma, gland radius — pick so a gland cross-section is ~tens–low-hundreds of cells
  and a section holds several glands, within the ST-spot bound (§3).
