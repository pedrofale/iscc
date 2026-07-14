# DESIGN — cell-state trajectories & differentiation (design-first; NOT yet built)

Status: **design-first; the program layer is now PAPER-1 work** (decision 2026-07-14). Companion to
`RESEARCH_QUESTIONS.md` R12, `DESIGN_features.md` (F8 = the precedent this mirrors), memory
`iscc-cell-trajectory-project.md`. Nothing here is implemented yet — this captures the plan; decide the
open questions (§7) before coding. Do it design-first, exactly like F8.

**SHARED WITH R13 (`DESIGN_expression.md`, decision 2026-07-14).** The program-loading model in this doc
(the per-cell program-activity vector `z` and its gene `loading` matrix) **is R13's expression backbone**
— one implementation serves both features. This doc (R12) owns how `z` *moves* — the differentiation
hierarchy, the genotype landscape-deformation, and the F8 niche coupling. R13 owns how `z` *becomes
counts* (the multiplicative expression map) and the gene-level dosage/SNV overlays layered on top. Because
R13 is now paper-1 work, **the program/`z` layer is on the paper-1 critical path** (the trajectory/RNA-
velocity *dynamics* of `z`, v2+, can still follow later). Build the `z` + `loading` machinery once.

## 1. Motivation

iscc's cell states today are **discrete**: cell type (cancer/epithelial/stromal/immune) × clone
(genotype), plus the F8 microenvironment expression *readout*. There is no continuous differentiation
axis. But real cells occupy a **spectrum** of states, set by:
1. their position in a **differentiation hierarchy** (stem → progenitor → differentiated) — the source
   of pseudotime; and
2. for cancer cells, **mutations** that reshape which states are reachable (differentiation block,
   de-differentiation, aberrant programs, plasticity).

No existing simulator couples a continuous cell-state trajectory to an **evolving, spatially-explicit
clonal genome under selection**. scMultiSim / SymSim / PROSSTT / dyngen all take a differentiation tree
(or a GRN) as **input** and have no genome evolution. iscc's trajectory would **emerge** from clonal
evolution + microenvironment — the same non-circularity differentiator as clonealign.

## 2. The abstraction — a genotype-deformed landscape

Cell state = a point **z** in a low-dimensional space of expression **programs** (cell cycle, stemness,
EMT, hypoxia-response, secretory, …) on a Waddington-style landscape. The landscape's attractors and
barriers are set by three inputs:

- **differentiation hierarchy** → the baseline landscape (stem attractor draining to differentiated
  valleys); flow down it = pseudotime;
- **genotype** → *deforms* the landscape (the novel cancer part);
- **microenvironment (F8)** → pushes specific programs (hypoxia → EMT).

Empirical support for the low-dim view: tumours reuse a limited set (a few dozen) of recurrent
continuous expression **meta-programs** (Gavish & Tirosh, Nature 2023) — so a handful of program axes,
not a full GRN, captures most continuous heterogeneity. This is what keeps the model cheap.

## 3. Where it lives in iscc (the clean split)

- **Evolution layer (cells/genotypes):** the state variable `z` and its dynamics — carried by cells,
  deformed by genotype, optionally coupled to fitness. "The landscape" lives here.
- **Expression layer:** the state→expression map (program loadings). `z` multiplies into the existing
  pipeline exactly like the F8 niche modifier.
- **The bridge = the `Selection` gene-role system.** Genes are already typed
  driver/oncogene/TSG/dispersal/resistance. Add one role: **differentiation regulator**. A clone's
  landscape params = the cell-of-origin's baseline deformed by the diff-regulator mutations/CNAs it
  carries. Mutations already carry typed phenotypic effects in iscc — this adds "effect on the state
  landscape" as another typed effect. Same machinery, no new paradigm.

## 4. The expression model (composition)

Multiplicative, matching how CNA dosage and F8 already compose:

```
expr_g(cell) = base[type, g] · dosage_g(CNA) · exp( Σ_k z_k · loading[k, g] ) · niche_g(F8)
```

- `z` = per-cell loadings on a handful of programs.
- `loading[k, g]` = fixed program dictionary (which genes each program moves); seed a couple from known
  signatures (cell cycle, EMT), draw the rest.
- **Normal hierarchy:** each compartment (epithelial/stromal/immune + the cancer cell-of-origin) gets a
  small stem→progenitor→differentiated fate tree; normal cells flow down it → pseudotime + a realistic
  non-malignant reference.

## 5. Genotype→landscape deformation (the novel bit)

Differentiation-regulator hits do one of four things:

| Mode | Landscape effect | z effect | Real anchor |
|---|---|---|---|
| **Block** | raise the progenitor exit barrier | z-mass piles at progenitor | AML differentiation arrest |
| **De-differentiation** | move attractor toward stem | z shifts to stem end | stemness reacquisition (CytoTRACE-high) |
| **New program** | unlock an attractor absent in normal | a z-axis that is 0 in normal turns on | EMT / neuroendocrine transdifferentiation |
| **Plasticity** | flatten the landscape | z variance ↑ (clone explores more) | aggressive high-plasticity clones |

EMT is both genotype- and niche-driven, so the microenvironment coupling here is biologically real —
and half of it (the niche field) already exists via F8.

## 6. Dynamics — three tiers (mirror the F8 rollout)

| Tier | What | Cost / risk | Unlocks |
|---|---|---|---|
| **v1 readout** | draw `z` at materialization from (compartment × genotype-deformed landscape × niche) | cheap; off-by-default; bit-identical on/off (like F8) | continuous states + ground-truth state/program labels; expression realism |
| **v2 inherited drift** | cell *carries* `z`; daughters inherit + step down the landscape | moderate; touches the per-cell state in the engine | TRUE trajectories & pseudotime along the lineage; ground-truth RNA velocity; the pseudotime-confound benchmark |
| **v3 fitness coupling** | `z` feeds division/death (stem divides, differentiated doesn't; therapy selects a state) | engine change; shares R8b's constraints | therapy-driven lineage plasticity / resistance; cancer-stem-cell dynamics |

**Recommendation:** build **v1 first** (readout-only, exactly F8's call) — ground truth + realism with
zero engine risk. The science is in **v2**.

## 7. Open decisions (settle these before coding)

1. **How mechanistic** — latent programs (recommended) vs a mechanistic GRN (rejected: dyngen/SymSim/
   scMultiSim turf; GRN/ATAC already deferred).
2. **v1 readout vs jump to v2 inherited** — stage v1 → v2 (recommended), or go straight to v2 if the
   trajectory benchmark is the point.
3. **Which phenomenon to anchor first** — differentiation block (AML-like) / EMT continuum / CSC
   hierarchy / therapy-driven plasticity. EMT is attractive: couples genotype AND niche, gives a clean
   continuous axis.
4. **How many programs, and which seeded** — start with ~4–8; seed cell-cycle + EMT from known
   signatures.
5. **v2/v3 engine question (= R8b):** how to carry a per-cell dynamical `z` in the genotype-count engine
   without breaking reproducibility, per-genotype expression caching, or tau-leaping. (Genotype-count
   exchangeability assumes cells of the same clone in the same deme are interchangeable — a per-cell
   continuous `z` challenges that; may need a per-(clone, deme) state distribution rather than per-cell,
   or a small number of state-bins.)

## 8. The scientific payoff (why v2 fits the paper)

Under v2, pseudotime, clonal genotype and spatial position become **entangled** (clonal territories
co-locate lineage + state) — the SAME "structure misleads inference" theme as PEtracer (lineage-space)
and multi-region trees. A trajectory method (Slingshot/Monocle/PAGA/CellRank/scVelo) reads a pseudotime
axis; iscc knows the TRUE differentiation state, the TRUE velocity AND the confounders — so we can show
*when pseudotime recovers differentiation vs when it is hijacked by clonal/CNA/spatial structure*. A
ground-truth trajectory/velocity benchmark no fixed-tree simulator can produce; the mechanistic analogue
of LARRY (Weinreb/Klein) clone+state+fate data. Slots into "iscc as a benchmarking substrate" beside
clonealign and PEtracer.

## 9. Validation plan (when built)

- Emit ground truth: per-cell true `z` / program loadings, true pseudotime, true RNA velocity (v2).
- Score trajectory inference (Slingshot/Monocle/PAGA/CellRank/scVelo) vs truth; quantify the confound
  across the dispersal/territoriality sweep (parallels the PEtracer figure).
- Expression realism: recovered meta-programs resemble the Gavish/Tirosh set.
- **Dedicated conda envs** for any external trajectory tool (scVelo/CellRank/etc.), per the
  `validation/README_integration.md` "one dedicated env per external tool" convention.

## 10. Staged plan (when promoted)

1. Design sign-off on §7 decisions → this doc updated.
2. v1 readout: program dictionary + `z` sampler + the multiplicative expression hook (beside the F8
   modifier); differentiation-regulator gene role in `Selection`; ground-truth surfacing; tests
   (off ⇒ bit-identical, like F8); a `validation/validate_celltrajectory.py` figure.
3. v2 inherited drift: per-cell/per-(clone,deme) `z` carried through division; true pseudotime +
   velocity ground truth; the trajectory-inference + confound benchmark.
4. (optional) v3 state→fitness coupling — jointly with R8b.
