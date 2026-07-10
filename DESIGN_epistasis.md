# DESIGN — epistasis / evolutionary-dependency structure in selection [design-first; NOT built]

Status: **whiteboard / design-first** (started 2026-07-09). Companion to `RESEARCH_QUESTIONS.md` R14,
the multi-patient cohort milestone (`DESIGN_cohort.md`, done), `DESIGN_expression.md` (the data-side
sibling). Motivated by the **DNA cohort-integration** row of the benchmark suite: cohort progression
models need a *known dependency network* to recover, which iscc's additive selection does not provide.
Nothing here is built yet.

## 1. Why (the gap)

Cohort DNA-integration / progression tools — **MHN**, **TreeMHN**, **CBN/H-CBN**, **REVOLVER**,
**RECAP** — recover a *network of promoting / inhibiting / ordering dependencies* between evolutionary
events across many patients. iscc's fitness is currently **additive**: abstract mode scores the
*count* of mutated drivers (`n_mutated_drivers`, per-role `n_mut_*` counts in `genome_summary`), and
real-genome mode is per-arm — neither encodes event×event interactions. So the true progression network
is ~empty, and a benchmark would only measure a method's **false-positive rate**. To make this a rich,
publishable benchmark we must be able to **plant a known epistasis / dependency structure** and show the
method recovers it.

## 2. The model — three interaction types (the planted ground truth)

Extend `Selection` so a genotype's fitness depends on *which* events co-occur, not just how many:

- **Pairwise epistasis** `E_{ij}`: event `i`'s fitness effect is modulated by the presence of event
  `j` (log-additive/multiplicative interaction term). `E_{ij} > 0` = co-selection (synergy),
  `E_{ij} < 0` = antagonism.
- **Conjunctive / ordered constraints (CBN-style):** event `B` is only beneficial (or only
  accessible) once `A` is present — a dependency DAG that induces **temporal order** in the lineage.
  This is exactly what CBN/H-CBN and TreeMHN's ordering recover.
- **Mutual exclusivity / synthetic lethality:** strongly negative `E_{ij}` ⇒ co-occurrence is
  deleterious ⇒ the two events appear **mutually exclusive** across the cohort (the DISCOVER/MEGSA
  signal).

The planted `E` / DAG **is the answer key**.

## 3. Engine integration

- Fitness (`Selection.update_division_rate` and the CINner fitness path) already reads a genotype's
  event set (`genome_summary` role counts, or the per-arm CN vector). Add an interaction term computed
  from the event set: `log fitness += Σ_i β_i·x_i + Σ_{i<j} E_{ij}·x_i·x_j` (+ DAG gating for
  conjunctive constraints).
- Must stay compatible with the **genotype-count engine**, **tau-leaping**, and reproducibility — the
  interaction is a pure function of a genotype's event set, so it caches per genotype (like the existing
  fitness). Off-by-default (empty `E` / no DAG ⇒ current additive behaviour, bit-identical).
- Events can be abstract driver roles (v1) or real-genome arms/genes (later).

## 4. Ground truth surfaced
The true interaction matrix `E` / dependency DAG / order constraints and, per patient, the realized
event order along the lineage — the answer key for MHN/TreeMHN/CBN/REVOLVER edge and ordering recovery.

## 5. Benchmark (the payoff, pairs with the cohort milestone)
Plant a network → run the cohort (shared landscape, per-patient private evolution; `DESIGN_cohort.md`)
→ collect per-patient trees (`to_newick`, true) or reconstructed trees → run **TreeMHN / MHN / CBN /
REVOLVER** → score recovered edges/order vs truth (edge precision–recall, order accuracy). Also score
**mutual-exclusivity / co-occurrence** detection against the planted signs. External tools each in
their own env (`iscc-treemhn`, …) per the dedicated-env convention.

## 6. Staged plan
1. **v1 — pairwise `E`** in abstract mode (a sparse, planted interaction matrix over driver roles).
2. **v2 — conjunctive/ordered constraints** (a dependency DAG) for CBN/H-CBN and TreeMHN ordering.
3. **v3 — mutual exclusivity / synthetic lethality** (negative interactions) + real-genome arm events.

## 7. Open decisions
- Network **sparsity/topology** and interaction **magnitudes** (must be detectable yet realistic).
- Abstract driver-role events vs real-genome arm/gene events as the interaction alphabet.
- Whether ordering comes from **fitness gating** (B inert until A) or **accessibility gating** (B cannot
  arise until A) — these are different generative stories the progression models make different
  assumptions about.

## 8. Relation
Cohort milestone (`DESIGN_cohort.md`, done — the substrate) · R6 (identifiability) · R10 (CNA events =
the interaction alphabet) · `DESIGN_expression.md` (data-side sibling prerequisite). Honest note: like
the expression realism doc, this is a **prerequisite that gates a whole benchmark category** (DNA cohort
progression) — without it the MHN/TreeMHN benchmark tests specificity only.
