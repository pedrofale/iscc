# DESIGN: the experimental-design recommender — "iscc as a study-design engine"

Status: design / scoping (2026-06-30). Forward-looking; **intended as a separate paper** from the
core iscc software paper. No recommender code exists yet.

Companion: `DESIGN_inference.md` (the `estimate*` layer this builds on), `DESIGN_features.md`
(sampling + assays it drives), `RESEARCH_QUESTIONS.md` R3/R4/R8 (the open questions it operationalizes).

---

## 1. Motivation: two modes of iscc

Today iscc is a **generative** engine: parameters → a simulated tumor → sampled, assayed data.
The natural second mode inverts the question an experimentalist actually asks. They don't want
*"give me some data"*; they want *"given what my platform can do and what I'm trying to learn,
is my planned experiment going to work — and if not, what should I change?"*

So iscc gains a second mode:

- **Mode 1 — Generative (current).** `params → data`. Benchmark substrate, ground-truth data.
- **Mode 2 — Recommender (new).** `pilot data + scientific goal + proposed design + budget →
  feasibility assessment + optimal design`. iscc runs the *virtual experiment* end-to-end —
  through a real evolving tumor, realistic sampling, the fitted assay, **and the downstream
  analysis** — and scores the result against the ground truth it alone knows.

The recommender is the generative engine wrapped in a loop that is only possible because iscc
**knows the right answer**: it can grade whether a proposed experiment would have recovered the
truth, for *any* analysis goal, not just the handful that closed-form power formulas cover.

### Worked scenario (the one that motivated this)

> A Smart-seq3 (SS3) biotechnologist hands iscc **one example SS3 dataset**. iscc's `estimate`
> layer fits the SS3 technical/batch profile (sensitivity, library-size variability, dropout,
> overdispersion, batch structure). The user then states a **goal** ("resolve the clonal
> phylogeny of a ~10⁴-cell tumor and detect any subclone above 5% CCF") and a **proposed
> collection** (N cells across 3 regions at SS3 depth, one timepoint, fixed budget). iscc reports:
> *with your SS3 noise profile, this design recovers the tree 0.78 of the time and detects a 5%
> subclone with power 0.55; reallocating to scDNA panel on the same budget raises subclone power
> to 0.9 but loses expression phenotype; goal "spatial niche interaction" is infeasible from
> dissociated SS3 at any allocation — you need a spatial assay.*

The same loop applies to every technology iscc supports (bulk/sc DNA, 10x/SS3 scRNA, Visium), and
— uniquely — to **treatment** goals (see §3, the sequential/value-of-information case).

---

## 2. The recommender loop

```
pilot dataset(s)  ──►  estimate*  ──►  fitted technical + batch priors  (per technology)
                                              │
scientific goal  ──►  estimand g + success criterion (power π, accuracy ε, FDR α)
                                              │
design space D   ──►  {modality, #cells, depth, breadth, #regions, #timepoints, replicates}
   + cost model        under budget B
                                              ▼
        for each candidate design d ∈ D, Monte-Carlo over R simulated tumors:
            grow tumor (ground truth Θ*)  →  sample per d  →  emit data with fitted priors
              →  run the ACTUAL downstream method  →  score estimate ĝ against truth g(Θ*)
                                              ▼
        power/accuracy(d) with uncertainty  →  feasibility verdict + Pareto frontier
              + sensitivity to the assumed priors  →  recommended design d*
```

**Inputs.**
1. *Pilot data* per technology → `estimate_dna` / `estimate` (RNA) / `estimate_visium` → technical
   priors. Optional: no pilot → use the calibrated defaults (§ calibrated-defaults), flagged as
   assumption-driven.
2. *Goal* as a measurable **estimand + success criterion** (§3).
3. *Design space + constraints*: which modalities are available, the knobs that can vary, per-unit
   costs, and a total budget (§4).

**Engine (per candidate design).** Monte-Carlo over simulated tumors and technical draws; for each,
sample → assay → **run the downstream analysis the user would actually run** → score `ĝ` vs the
ground-truth `g(Θ*)`. Aggregate into power (P(success criterion met)) or expected accuracy, with
Monte-Carlo CIs.

**Outputs.**
- A **feasibility verdict** per goal ("achievable / marginal / infeasible under budget B").
- **Power/accuracy curves** over the design knobs (the scPower deliverable, generalized).
- A **Pareto frontier** across competing goals or cost axes (e.g. expression phenotype vs subclone
  resolution; depth vs cell number; cells vs regions).
- The **minimal / cheapest design** that meets the criterion, and the marginal value of the next
  dollar (where to reallocate).
- **Sensitivity**: how the recommendation moves if the fitted priors are off (robustness of the
  advice to the pilot being unrepresentative).

---

## 3. Goal taxonomy (the estimands the recommender can score)

The differentiator vs existing tools is breadth of goal. Because iscc has a full ground truth, the
recommender can score essentially any estimand, including ones with **no closed-form power**:

**Expression / cell-state (overlaps scPower, powsimR, POWSC, PoweREST):**
- Differential expression / marker detection power.
- Cell-type identification; **rare population detection** (smallest detectable fraction).
- Batch-effect correctability: does integration recover the ground-truth cell structure?

**Genome / evolution (overlaps Tarabichi/Boutros subclone-power, but goal-agnostic):**
- **Clone-tree / phylogeny recovery** accuracy (e.g. Robinson–Foulds distance ≤ ε).
- **Subclone detection** at cancer-cell-fraction f with power π (the classic depth × purity ×
  #regions question — but measured, not assumed).
- **Selection-coefficient / evolutionary-rate** estimation accuracy (ties to the ABC layer).
- CNA-calling accuracy (segment-level / focal, once focal CNAs land — RESEARCH_QUESTIONS R10).

**Spatial (overlaps PoweREST for DEG; iscc adds structure):**
- Spatial-niche / cell–cell-interaction detection power.
- Spatial-domain recovery; deconvolution accuracy at a given spot size.

**Cross-modal & allocation (NO existing tool):**
- "For goal g and budget B, which **modality** (or mix) maximizes power?" — e.g. scRNA vs scDNA vs
  Visium vs multiome, the question no single-modality power tool can pose.

**Treatment / sequential design (NO existing tool — the West/Anderson tie-in):**
- **Value of information in adaptive therapy**: how sparse/biased a monitoring assay (ctDNA VAF,
  one biopsy, imaging proxy) can be before the adaptive-vs-continuous advantage collapses.
- **Optimal monitoring schedule**: when to re-biopsy / re-assay to control the tumor — sequential
  experimental design over time, not just a one-shot allocation.

---

## 4. Design space & cost model

Design variables the recommender optimizes over:
- **Modality**: bulk DNA, sc-DNA (panel / WGS), scRNA (10x / SS3 / …), Visium / spatial, multiome.
- **Per-modality depth/breadth**: reads or UMIs per cell; WGS vs WES vs panel (and panel gene set);
  #spots / section area for spatial.
- **Cells / units**: #cells (or #nuclei) per sample; #spots.
- **Sampling geometry**: #biopsy regions, region size & placement, dissociation protocol (the
  composition-bias knob), solid vs liquid biopsy.
- **Replication / longitudinal**: #patients/tumors, #timepoints (for treatment / sequential goals).
- **Budget**: total cost with per-unit cost coefficients (per-cell, per-read, per-section,
  per-sample); the optimizer respects `cost(d) ≤ B`.

This is the same "shallow-many vs deep-few under a fixed budget" tradeoff scPower/POWSC study for
scRNA DE — generalized across modalities, sampling geometry, time, and arbitrary goals.

---

## 5. What's reusable vs new

**Reuse (already in iscc):** the generative engine (tumor → sample → assay → reads), the `estimate*`
layer (pilot → technical priors), the calibrated defaults, the sampling module (multi-region,
dissociation, liquid biopsy), the treatment module, and the validation harness (ground-truth scoring
machinery).

**New (the recommender layer):**
- A **goal/estimand API**: declare an estimand `g`, a downstream analysis `ĝ = method(data)`, and a
  success criterion. A small library of built-in goals (DE, phylogeny, subclone, niche, control).
- A **design-space + cost-model API** and an **optimizer** (grid / Bayesian optimization / bandit
  over designs; the Monte-Carlo inner loop is embarrassingly parallel).
- A **report generator** (power curves, Pareto fronts, feasibility verdict, sensitivity).
- Adapters to **real downstream methods** (call the actual DE/CNA/phylogeny/niche tools so the score
  reflects the method the user will use, not a proxy).

---

## 6. Prior art & positioning

Experimental-design / power tools exist, but each is **single-modality and single-task**, and
**statistical rather than biology-generative** (they model count distributions from pilot data; none
has a tumor, evolution, spatial tissue, or treatment underneath, so they cannot score goals like
"recover the phylogeny", "detect a spatial interaction", or "control the tumor").

| Tool | Modality | Goal(s) it scores | Pilot-data driven | Generative biology | Cross-modal | Treatment |
|---|---|---|---|---|---|---|
| **iscc recommender** | DNA + RNA + spatial (+reads) | **any** (DE, phylogeny, subclone, niche, selection, control) | ✅ `estimate*` | ✅ evolving spatial tumor | ✅ | ✅ |
| **scPower** (Schmid 2021, Nat Commun) | scRNA | DE, eQTL power | ✅ | ⬜ (NB priors) | ⬜ | ⬜ |
| **powsimR** (Vieth 2017) | bulk + scRNA | DE power | ✅ | ⬜ | ⬜ | ⬜ |
| **POWSC** (Su 2020) | scRNA | DE power, sample size | ✅ (sim) | ⬜ | ⬜ | ⬜ |
| **PoweREST** (PLoS CB 2024) | Visium | DEG between conditions | ✅ | ⬜ | ⬜ | ⬜ |
| **Tarabichi/Boutros** (Nat Commun 2020) | bulk/multi-region DNA | subclonal-reconstruction power | data-study | ⬜ (study, not tool) | ⬜ | ⬜ |

**The one-sentence positioning.** *scPower/powsimR/POWSC/PoweREST tell you how many cells/reads/spots
you need to detect differential expression in one modality; the iscc recommender tells you whether
your whole multi-modal study — across sampling geometry, assay choice, and even treatment monitoring
— can recover the biological truth you care about, by running the experiment in silico against a
known answer.* It is to study design what iscc-generative is to benchmarking: the same ground-truth
advantage, pointed at the planning question.

Adjacent / worth citing: scDesign3 (count realism), the multi-task spatial-sim benchmark (Genome
Biol 2025), and active-sampling work (e.g. RL-based spatial sampling, arXiv 2512.13635) for the
sequential-design framing.

---

## 7. Validation (does the recommender's advice transfer? — RESEARCH_QUESTIONS R4)

The recommender is only useful if its predictions hold up. Three validation tiers:
1. **Internal consistency.** A design predicted to reach power π reaches it on held-out simulations.
2. **Recover known results (sanity).** Reproduce scPower's "shallow-many beats deep-few" for scRNA
   DE, and Tarabichi's "≥3 regions + ~1000× → power ≥0.8" for subclone detection — from the
   generative engine, with no formula baked in.
3. **External transfer (the real claim).** On a task with a real benchmark, the recommender's
   *ranking of designs* matches the ranking observed in real data. This is R4 made operational.

---

## 8. Paper plan (separate from the core iscc paper)

- **Venue**: PLoS Computational Biology (methods/software), companion to the core iscc paper; lives
  beside scPower / PoweREST (both design-tool papers) rather than beside CINner/SISTEM/HAL.
- **Framing**: "From simulating data to designing experiments: a ground-truth-grounded, multi-modal
  experimental-design engine for cancer genomics."
- **Core claim**: existing power tools optimize one modality for one statistical task; the
  generative tumor→data chain lets you optimize *any* analysis goal across modalities, sampling and
  time, scored against a known answer.
- **Figure arc**: (1) the two-mode architecture + recommender loop; (2) the SS3 worked scenario
  (pilot → priors → feasibility verdict); (3) cross-modal allocation (scRNA vs scDNA vs Visium for a
  fixed goal/budget) — the figure no competitor can make; (4) sequential design / value-of-
  information for adaptive-therapy monitoring (the West/Anderson tie-in); (5) validation §7
  (recover scPower + Tarabichi results; external transfer).
- **Relationship to the core paper**: the core paper establishes the generative engine + realism +
  inference (it must be accepted/strong first, because the recommender's credibility rests on the
  realism it demonstrates). This paper is the "so what can you *do* with it" follow-up.

---

## 9. Risks & open questions
- **Downstream-method coupling.** Calling real analysis tools in the inner loop is powerful but
  heavy (runtime, dependencies). Need a tiered approach: fast surrogate scores for the optimizer's
  coarse sweep, real methods for the final candidates.
- **Compute.** R tumors × |D| designs × downstream method is expensive; relies on tau-leaping (§7),
  parallelism, and Bayesian-optimization/bandit search rather than full grids.
- **Prior-representativeness.** A single pilot may not represent the new cohort; §6 sensitivity
  analysis must be first-class, and the report must state assumption-dependence honestly.
- **Goal specification UX.** Turning a vague scientific aim into an estimand + criterion is the hard
  human-facing part; a curated goal library lowers the barrier.
- **Scope discipline.** Easy to balloon into "simulate everything"; the paper should ship a small,
  well-validated goal set (DE, phylogeny, subclone, niche, adaptive-monitoring) rather than all of §3.
