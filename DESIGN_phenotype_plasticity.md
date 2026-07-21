# DESIGN — genotype→phenotype in the structured setting: compartment selection (v1) + plastic epistate (v2) [design-first]

Status: **DESIGN-FIRST (2026-07-18), NOT built.** Supersedes the ad-hoc "compartment × payoff" plan from the
other session. Extends R12 (cell-state trajectory) and R13 route-3 (niche→program); builds on F8
(microenvironment) and the existing gene-based `Selection`. **No engine code until sign-off.**

## 0. The problem, and the three papers that frame it
iscc's engine currently has **no phenotype layer**: a genotype's rates (`division/death/dispersal/immune_
resistance`) are a deterministic function of its genome, cached per genotype (`count.py` `_death_rate`,
`selection.update_*`). The microenvironment modulates only the *payoff* of a fixed phenotype (the immune term,
`_death_rate:453`). We want iscc to be the **ground-truth engine for genotype-vs-phenotype links** in the
structured (normal-cells + microenvironment) setting. Three papers set the frame:

- **Gatenbee, West, …, Graham, Anderson 2019** (bioRxiv 594598, spatial EGT of ductal-carcinoma
  immunoediting: pioneer–engineer, public goods, "space is the game changer"). The biology target — AND the
  cautionary tale: its **payoff matrix (`bG, bN, cC, …`) is unmeasurable**, so it validates only a qualitative
  prediction and stalls. The trap to avoid *and* the gap iscc fills.
- **Whiting, …, Sottoriva, Graham** (*Trends in Cancer*, opinion): a phenotype has three sources —
  **genetic**, **plasticity** (non-genetic, *environment-responsive*, reversible), and **noise** (non-genetic,
  *environment-independent*, stochastic). Separating them from data is the field's open problem.
- **Yanai lab, Nat Genet 2022** (s41588-022-01141-9): recurrent cancer cell-state modules across 9 cancer
  types, **causally linked to TME cell types**, with invasive/pEMT states **at the leading edge** — i.e.
  microenvironment-driven, spatially patterned, reversible states. The validation target.

### The disciplining insight (do not skip — it defines the whole v1/v2 split)
Separate two layers, because the confound lives in only one of them.

- **Selection dynamics (rates).** Modelling rates already gives *process stochasticity*; compartment payoffs
  already give *context-dependent selection* (env reads a **fixed** phenotype differently). A plastic state
  that **equilibrates instantly to the current compartment is observationally identical to context-dependent
  selection** — it buys nothing for the dynamics.
- **The observable phenotype (expression).** iscc materialises `cell_exp` **after** growth, as a function of
  (genotype, niche) (R13 dosage/SNV effects + route-3 niche→program, F8 fields). So the **same clone in two
  compartments already expresses differently** — env-responsive phenotype in Graham's sense — and the
  **genetic-vs-environmental attribution confound is present in the data with ZERO epistate and zero new
  dynamics parameters.** It is memoryless (reflects the *final* location only), but the confound does not
  require memory. This is how the engine already works; v1 only has to feed the *compartment* into the
  niche→program input.

So the confound — the thing the framing papers are actually about — is **free in v1**. A carried, memoried
epistate is needed ONLY for the three things a post-growth, memoryless readout cannot represent:
  1. **memory / hysteresis** — phenotype reflecting the *trajectory*, not just the final location;
  2. **selection acting on the phenotype** — evolutionary feedback (the plastic state itself under selection),
     vs v1 where the phenotype is a downstream readout that does not affect fitness;
  3. **persistent env-independent noise** — heritable-with-decay stochastic states, i.e. **drug-tolerant
     persisters** that arise without a mutation and are transiently inherited.
None of the three is needed for the confound. So: **v1 = context-dependent selection + context-dependent
phenotype-at-materialisation** (shows the confound, no epistate, no new dynamics knobs). **v2 = the minimal
memoried + noisy epistate**, justified only by (1)–(3). v1 is a strict limit of v2 (τ→0, phenotype→readout),
so v2 always degenerates back to it.

## 1. What already exists (reuse, do not rebuild)
- **Gland geometry = lumen / epithelial ring / stroma.** `_seed_structure` (`count.py:345`) seeds an
  epithelial ring at `structure_radius`, the cancer founder just inside (lumen), stroma outside. The
  compartments are already spatial.
- **Four gene-based heritable axes** (division onc/TSG, dispersal, immune-resistance, treatment-resistance),
  each = mutations at designated genes (`selection.make_*`), fitness via `_rel_fitness`. Sequenceable →
  recoverable. This is *why* iscc can benchmark selection inference; do not replace it with an abstract vector.
- **Context-dependent selection, already implemented, for immune:** `_death_rate` adds
  `immune_prob_kill · immune_fraction(deme) · (1 − immune_resistance)` — a heritable trait that pays off only
  where immune cells are. Treatment resistance is likewise context-gated. This is the template v1 generalises.
- **F8 microenvironment fields** (hypoxia O2, CCI) + per-deme modifiers — the substrate that drives env effects.
- **R13 program activity `z`** (the per-cell phenotype vector) + **route-3 (niche→program)** design — the
  natural home for the v2 epistate. R12 is the plastic-landscape framing.

## 2. v1 — compartment-dependent selection (GENETIC; the identifiable floor)
**Principle (one mechanic for every compartment):** a compartment contributes a local **hazard** to cancer
death, attenuated by a **matching heritable resistance trait** — exactly the existing immune term, generalised.

- **Two new gene-based axes** (mirror `make_immune_resistant`): `prop_breach`, `prop_stromal_survival`, with
  `breach_effects` / `stromal_survival_effects`, `N_breach`/`N_ss` counts, `update_breach`/`update_stromal_
  survival` returning `_rel_fitness` multipliers. They flow into `cell_data` + DNA-seq for free.
- **`_death_rate` gains two terms** (same shape as immune):
  ```
  death += epithelial_barrier · epithelial_frac(deme) · (1 − breach)
  death += stromal_hazard     · stromal_frac(deme)    · (1 − stromal_survival)
  ```
- **Normals stay immortal** (the crowding-fix invariant, `DESIGN_crowding.md`). "Breach" = crossing a barrier
  the trait removes, NOT Gatenbee clearance of mortal normals (that is v3, §4).
- **Emergent behaviour (selection):** sequential invasion — lumen → breach the epithelial ring → survive the
  stroma → resist immune where present. Each barrier selects a *different* heritable trait, each recoverable by
  sequencing. The **selected traits are genetic** (permanent, additive, cannot switch off): who *survives* a
  compartment is genotype × context.
- **Context-dependent phenotype = the confound, for free.** `cell_exp` is materialised **after** growth as
  `f(genotype, niche)`, so v1 must feed the **compartment identity into the niche→program input** (R13 route-3
  — a few existing program parameters, NOT an epistate). Then the *same clone* expresses an invasive-ish
  program at the epithelial front and a different program in the stroma: **env-responsive phenotype, and the
  genetic-vs-environmental attribution confound, present in the data with zero carried state.** This phenotype
  is memoryless (final location only) and is a **readout** — it does not feed back into fitness in v1.
- **Parameters (~4 selection knobs, all ground-truth):** `epithelial_barrier`, `stromal_hazard`, plus
  `prop_`/`_effects` for the two axes. The phenotype confound reuses existing R13/F8 niche→program knobs — **no
  new dynamics parameters.**
- **v1 validation:** (a) spatial invasion dynamics vs `structure_radius`/hazard coefficients; (b) selection
  benchmark — can DNA-seq + a selection-inference method **recover the breach / stromal-survival drivers** and
  the compartment each was selected in? (c) **the confound benchmark** — given scRNA state + spatial data, can
  a method attribute an invasive expression state to genotype vs compartment, when iscc knows both? (d) the
  "invasion requires a driver mutation" regime (the genetic baseline the v2 plastic regime is contrasted with).

**v1 is NOT:** memoried, phenotype-selected, or persister-bearing (the §0 (1)–(3) list). It *is* env-responsive
in its observable phenotype — enough for the confound — and it ships the spatial-invasion story on ~4 knobs.

## 3. v2 — the plastic epistate (heritable-with-decay + noise; the flagship)
The **single** new object, kept deliberately minimal to dodge the Gatenbee trap. **Justified only by the three
things v1's post-growth, memoryless phenotype cannot represent** (§0): (1) memory/hysteresis, (2) selection
acting on the phenotype, (3) persistent env-independent noise (persisters). The basic genetic-vs-environmental
confound is NOT a reason to build v2 — v1 already has it.

- **A low-dimensional epistate `s`.** Reuse R13's seeded programs — e.g. `s ∈ {invasive, proliferative,
  quiescent}` activities — NOT a free matrix. Discretised to a few levels for engine tractability.
- **Three dynamics terms, one per Graham category:**
  1. **genetic bias** `β_bias`: the genotype sets `s`'s attractor / lowers the barrier to a state (a driver
     makes the invasive state *more accessible*, not mandatory) — "the genetic determinant of a non-genetic
     phenotype".
  2. **plasticity** `τ_relax`: the local compartment/niche (F8) pulls `s` toward a target with relaxation
     timescale `τ_relax`. **This timescale is the entire difference from v1** (τ→0 ⇒ instantaneous ⇒ v1).
  3. **noise** `σ_noise`: env-independent stochastic switching of `s`.
- **Inheritance = the memory:** daughters inherit `s` with decay toward the genetic attractor + noise
  (epigenetic-like, not genetic). A cell that left the epithelium stays partly invasive for ~`τ_relax`.
- **Selection reads `s`, not the genotype:** `rates = f(baseline(genotype), s, compartment)`. So breach pays
  off when `s` is invasive **and** the cell is at the epithelium; move to stroma and `s` relaxes → the
  phenotype switches **reversibly** with the same genotype. The breacher→proliferator switch falls out.
- **Engine cost (be honest):** a memoried `s` travels with cells across demes, so it splits each genotype into
  **(genotype × epistate) sublineages**. Discretising `s` to a few levels keeps this a small multiplier on the
  genotype count, not a per-cell state — but it is a real change to the count engine's caching, not a free
  rate modifier. This cost is *why* v2 is only worth it for the attribution question.

### 3.1 Tunability discipline (the concern, addressed directly)
The worry — "a lot of stuff that's hard to tune" — is right, so the design is constrained to make it *not* so:
- `s` is **2–3 axes, few discrete levels** (not a continuum, not a matrix).
- **Exactly three new dynamics parameters**, each individually interpretable and each with a degenerate limit:
  `τ_relax` (memory; **τ→0 recovers v1**), `σ_noise` (noise; **σ→0 ⇒ deterministic plasticity**), `β_bias`
  (genetic pull; **β→∞ ⇒ genetic-only**). So **v2 CONTAINS v1 as a limit** and each knob is testable by
  degeneration — you are never tuning blind.
- These are **ground-truth knobs, never fit.** The deliverable is an **identifiability map over (τ, σ, β)**,
  not a calibration. We never claim the "right" values — we ask what data can recover.
- **The Gatenbee lesson is encoded as a hard scope rule:** no per-interaction payoff coefficients, no
  macrophage public-goods matrix in v2. Those are v3 (§4), gated on a specific benchmark, because each is a new
  unmeasurable parameter.

### 3.2 v2 validation = the flagship identifiability benchmark
The *basic* genetic-vs-environment confound is a v1 benchmark (§2). v2's benchmarks are the ones that need the
carried state — from a **known** `(τ, σ, β)`, ask what inference recovers:
- **Separate plasticity (env-responsive) from noise (env-independent)** — Graham's finer split, which *requires
  memory* to be identifiable (an env-independent state persists against the niche; a plastic one tracks it).
  v1 cannot pose this; v2 can, with ground truth.
- **Detect persisters** — a heritable-with-decay drug-tolerant state arising without a mutation; can a method
  distinguish it from a resistance driver? (needs (3).)
- **Migration-history / hysteresis inference** — does a cell's lagged state reveal where it *came from*? (needs
  (1); impossible for a memoryless readout.)
- **Phenotype-driven evolution** — when the plastic state is under selection ((2)), does the population evolve
  toward a state the genotype alone would not predict?
- **Reproduce Yanai** — recurrent, TME-driven, spatially-patterned states emerge from the niche→state pull
  (partly already in v1; v2 adds the leading-edge *persistence* behind the front).
- **Interrogate Gatenbee** — instantiate the ductal-breach model with known parameters and map which are
  recoverable: the validation layer they lacked. (Likely finding: initial conditions + the genetic
  contribution are recoverable; the plastic/noise split and payoff coefficients are only partially
  identifiable, and not from any single modality — the negative result *is* the contribution.)

## 4. Staging / scope
- **v1 (build first):** ~4 selection params into `Selection` + `_death_rate`, plus feeding compartment into the
  existing R13/F8 niche→program input (no new dynamics knobs). Ships spatial invasion + the
  genetic-selection-recovery benchmark **and the genetic-vs-environment expression confound**; the identifiable
  baseline. No carried epistate.
- **v2 (the flagship; likely paper-2):** the epistate `s` + `(τ, σ, β)` + selection-reads-state; the
  genotype-vs-phenotype identifiability benchmark. Build only after v1 lands **and** a benchmark motivates it.
- **v3 (LATER, optional, each individually gated):** macrophage / public-goods diffusible fields via F8
  (M1-ROS / M2-GF as diffusible public goods), and mortal-normal Gatenbee clearance. Each is a new
  unmeasurable-parameter risk; add only when a specific result demands it.

## 5. Open decisions
- Reuse R13 `z` as the epistate vs a dedicated small state (recommend **reuse** — one phenotype object across
  the engine and the data modalities).
- Epistate discretisation level (tractability vs fidelity) — the main engine-cost dial.
- Does v1 `breach` gate **survival-in** or **dispersal-into** an epithelial-occupied deme? (mechanics detail;
  survival-in matches the immune template most directly.)
- Epithelial ring thickness (currently 1 deme) — may need thickness or just a strong `epithelial_barrier`.

## 6. What this deliberately avoids
- The abstract `(b, d, w)` trait vector (un-sequenceable → breaks iscc's inference-benchmark premise).
- A free-parameter payoff / game matrix (the Gatenbee unvalidatability trap).
- Deterministic genotype→phenotype (the current limitation we are fixing).
- A *carried* epistate without memory or noise (redundant — post-growth `expression = f(genotype, niche)`
  already gives the memoryless env-responsive phenotype and the confound; the §0 insight).
- Building v2 before v1, or v3 before a benchmark needs it.
