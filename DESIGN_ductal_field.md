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
- **Stroma** fills the rest, seeded at a **realistic moderate density** (stroma is less dense than epithelium
  but NOT empty — it has fibroblasts/immune/endothelial cells; `stroma_fill_frac` ≈ 0.3–0.5 of `K_stroma`),
  leaving headroom for an invasive mass to fill. These stromal cells are real cells (captured by scRNA/Visium)
  AND the source of the stromal hazard (§5).
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
- `K_stroma`: similar / modestly higher (bulk), seeded at **moderate density** (`stroma_fill_frac`≈0.3–0.5 —
  real stromal cells, meaningful fraction for the hazard, headroom for a dense invasive mass).
- **ST resolution is a GRID-SPACING constraint, NOT a K bound** (this corrects an earlier note): a Visium spot
  must cover **several demes** to be a mixture (for deconvolution / sub-spot structure), set by grid density vs
  spot radius — *independent of per-deme K*. K is the depth population the assay samples; large K just gives
  more (representative) cells per location. So: grid fine relative to the spot, K moderate-large for depth +
  sequenceable per-location populations, TOTAL kept sane via field size. Large K is nearly free at growth (the
  count engine tracks genotype COUNTS per deme); it only costs at `make_cell_data` / assay, where you subsample.

### 3.1 The 2D-section assay samples the depth (not the growth model)
The full-depth K lives in the GROWTH model; the depth-subsampling lives in the ASSAY. A 2D-section spatial
assay (Visium, F9 single-cell-spatial) must NOT drop a deme's whole K into a pixel — the thin cross-section
holds only a *slice* of the deme's 3D column. So at each spot the assay **samples a realistic per-pixel number
of cells** (≈ `section_thickness/column_depth · K`, e.g. a handful per Visium spot) **uniformly at random from
each covered deme**, and records the per-spot ground truth (n_cells, clone/type fractions) from those *sampled*
cells (uniform sampling keeps the composition representative). **Dissociated assays (scRNA/scDNA) have no such
issue** — dissociation mixes all depths, so they sample the whole per-location population directly. So growth =
the full 3D population per location; each assay takes the appropriate *view* (section slice for Visium, whole
population for dissociation). NOTE: iscc's current `Visium` pools *all* cells in a spot — the ST-generation
work must switch it to this section-slice sampling so large K doesn't over-fill spots.

### 3.2 Concrete physical scale + the shipped DCIS→IDC example (BUILT 2026-07-31)

This pins §3/§3.1 to concrete numbers, decided with Pedro while building the spatial-assay-scale example
(`notebooks/example_config.yaml`). **The anchor is that a deme is a PATCH of tissue, not a cell**, and its
`carrying_capacity` K is the cell population of the 3-D column it stands for.

| quantity | value | reasoning |
|---|---|---|
| cell | ~12 µm | breast epithelial cell |
| **deme** | **~50 µm** in-plane | a PATCH of ~4 cells across — deliberately NOT ~1 cell wide; it stands for a 3-D column |
| **K** (`carrying_capacity`) | **K_duct 60, K_stroma 30** | the column's 3-D cell population (a denser duct, looser stroma) — "captures the depth" (§3) |
| **duct** | `gland_radius` 4 → **9 demes ≈ 450 µm** | a realistic DCIS-expanded duct spanning MANY demes (§3: a few demes × moderate K = a 3-D duct) |
| **tissue** | grid 170 → **~8.5 mm** | LARGER than a 6.5 mm Visium capture, so the slide samples a subset; total ≈ **10⁶ cells** |
| **Visium v1** | 100 µm pitch = **2 demes**, 55 µm spot ≈ **1 deme** | see the deconvolution note below |

**2-D views are thin SECTIONS (§3.1), not the whole column.** A duct deme has K_duct = 60 cells in its
3-D column, but a cell-level 2-D view — an H&E image, a Visium spot, an imaging assay — is ONE ~12 µm slice,
so it shows only ~one layer: **~17 packed cells per duct deme** (a 50 µm patch of 12 µm cells), fewer in
stroma. Materialising/plotting the WHOLE column into a 2-D pixel over-crowds it; the assay/plot must take a
thin section (~`section_thickness/column_depth · K`). Dissociated assays (scRNA/scDNA) instead mix all
depths and sample the whole per-location population.

**Deconvolution tradeoff (refines §3's "spot covers several demes").** A Visium spot is 55 µm ≈ ONE deme at
this scale, so the shipped example resolves ST at ~deme (50 µm) resolution and each spot's cells are a
representative *within-deme* mixture. Sub-spot deconvolution across FINER structure would need a finer grid
(deme < spot) — but a cell is ~12 µm and a spot only ~4–5 cells across, so "deme ≪ spot" forces ~cell-sized
demes, which is the "1-cell-wide" deme we rejected. This is an inherent physical tension: pick deme ≈ spot
(this example — cleaner biology, deme-resolution ST) OR deme ≈ ½–¼ spot (sub-spot deconvolution, ~cell-sized
demes). K (depth) is orthogonal to both.

**The key scalability consequence.** ~10⁶ cells falls straight out of the correct physics (grid² × K), NOT
from an arbitrary "make it huge" — so the scalability engine (`DESIGN_scalability.md §8`) is *required*, not
optional. And it is affordable: tau-leaping advances a whole clone (all K cells of a deme) in one Poisson
draw, so growth cost scales with #clones × #demes, **not #cells** — a big K is nearly free at growth and
only costs at `make_cell_data`/assay, where `max_cells` subsamples (§9).

**Biology of the shipped example (single clonal founder; DCIS → IDC, BREACH-LIMITED).** One founder in ONE
duct grows confined (DCIS, behind a high `epithelial_barrier`); its descendants spread along the ductal tree
via `cross_gland` dispersal (the 3-D-connected ducts, §1/§4), filling ~8 duct cross-sections that are ALL one
clone (the Muller has a single founder with derived subclones). Invasion is **breach-limited**, decided with
Pedro: *"breaching is VERY unlikely but once it's breached, invasion and dispersal through the stroma should
be easy."* So crossing the duct wall (`breach`) is RARE (`prop_breach` ~1e-3) and STRONG, but the stroma is
**PERMISSIVE** (low `stromal_hazard` ~0.6) — once a clone is out it disperses and grows readily.
`stromal_survival` is a minor add-on, NOT a required second gate (contrast the earlier 2-hit hostile-stroma
design, which kept invasion focal by making the stroma lethal; that flip-flopped and was dropped because at
8 ducts a strict-enough second gate meant *nothing* invaded while a looser one meant *all* ducts invaded in
parallel — the breach-limited single-rare-event mechanism gives one clone escaping and then spreading,
which is the correct DCIS→IDC picture). Because the breach clone also reaches connected ducts via
`cross_gland`, invasion starts near several ducts and **coalesces** into a confluent invasive region with
residual DCIS ducts embedded — matching the Janesick 10x Visium breast sample (several DCIS ducts + a
dominant invasive region). `grow(n_steps=…)` sets how far the invasion has coalesced (the shipped tutorials
grow to ~150: confluent invasive region(s) + DCIS ducts + residual stroma). Selection is intact under
passenger coarsening (the cell-weighted division rate evolves above baseline; driver sweeps and the DCIS→IDC
breach remain visible).

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
- **Stromal hazard = the live stromal-cell fraction** (`stromal_fraction(deme)·(1−stromal_survival)`),
  symmetric with the epithelial barrier and the immune term — the stromal cells (fibroblasts etc.) *are* the
  hostile microenvironment, and keying to their live fraction is composition-based (no fixed label) and
  consistent. This requires seeding stroma at a **realistic moderate density** (§2) so the fraction is
  meaningful (stroma is less dense than epithelium but NOT empty); the (immortal) stromal cells then dilute as
  cancer invades, so the hazard is strongest when a clone first arrives in virgin stroma and softens as it
  establishes. *(Optional variant, not the default: a persistent stromal-region FIELD if you want
  `stromal_survival` under selection everywhere even after cancer takes over — less consistent, use only if a
  benchmark needs it.)*
- **The invasive / EMT expression program is driven by `breach` (route 1) + the compartment field (route 3).**
  `breach` — the heritable trait a clone evolves to cross the epithelial wall — *is* the invasion phenotype, so
  the R13 phenotype→program map drives the `emt` program from it (`DEFAULT_PHENOTYPE_PROGRAM_MAP` now includes
  `breach → emt`, alongside the compartment-field route-3 map `epithelial → emt`). This gives the invasive
  program a **genetic arm** (breach genotype) AND a **niche arm** (epithelial interface) — the two halves of the
  genetic-vs-niche expression confound (`DESIGN_phenotype_plasticity.md` §2). **Do NOT rely on the legacy
  `dispersal_rate → emt` map for the genetic arm:** `prop_dispersal = 0` in the structured configs, so
  dispersal never varies and that route is inert (zero fold-change ⇒ zero drive). `breach → emt` is the correct,
  always-live genetic driver; `dispersal_rate → emt` is retained only for the day `prop_dispersal > 0`.

## 6. Scope / staging
- **v1:** the geometry (multi-gland island field), the cross-gland dispersal channel + `gland_id` labels, the
  per-compartment K + moderate-density stroma. The `GenotypeTumor` (count) engine only (default). Off-by-default: N=1
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

## 8. Follow-ups (future additions, NOT v1)
- **Make immune infiltration spatial (currently uniform + static).** iscc seeds immune cells UNIFORMLY in every
  deme at a fixed density (`immune_density`), static — no recruitment/migration/depletion — and the kill term is
  `immune_prob_kill·immune_fraction(deme)·(1−immune_resistance)` (`count.py`). For DCIS→IDC, real immune activity
  is **peri-ductal and concentrated at the invasive front / in the stroma**, not a flat backdrop. Follow-up: key
  immune density to the **stromal region and/or the tumour boundary** (a front-concentrated field, F8-style) —
  optionally active recruitment dynamics — so immune surveillance actually shapes *where* invasion succeeds.
- **Immune-interaction expression program in cancer cells (R13 route-3).** A cancer cell that is NOT
  immune-resistant and sits among immune cells is under immune attack and should EXPRESS that interaction — an
  interferon / immune-response program (IFN response, antigen presentation, stress) — whereas a resistant cell
  shows immune-evasion instead. Follow-up: add a niche→program coupling (`DESIGN_expression.md` R13 route-3, the
  same mechanism the compartment niche uses to drive the invasive program) that drives an "immune-response"
  program in cancer cells as a function of the **local immune fraction × (1 − immune_resistance)**, so the
  interaction is visible in scRNA/ST. This is both realism (real cancer cells at an immune front carry an IFN
  signature) AND a ground-truth confound (is the immune signature driven by genotype / resistance status vs the
  local immune niche?).
