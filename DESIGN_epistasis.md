# DESIGN — epistasis / evolutionary-dependency structure in selection [BUILT]

Status: **BUILT** (2026-07-15; paper 1). Companion to `RESEARCH_QUESTIONS.md` R14, the multi-patient
cohort milestone (`DESIGN_cohort.md`, done), `DESIGN_expression.md` (R13, the data-side sibling — also
paper 1 now). Motivated by the **DNA cohort-integration** row of the benchmark suite: cohort
progression models need a *known dependency network* to recover, which iscc's additive selection did
not provide.

**Where it lives:** `src/iscc/tumor/components/epistasis.py` (the network) + `Selection`
(`epistasis_params` / `dependency_params`, off by default, drawn from `layout_seed +
LAYOUT_OFFSET_EPISTASIS`) · ground truth `tumor.epistasis_ground_truth()` / `tumor.event_table()` ·
scoring seam `iscc.integrations.progression` · tests `tests/test_epistasis.py` · benchmark
`validation/validate_epistasis.py` · docs `PARAMETERS.md` · Results `sec:epistasis`.

**Decisions made while building** (the §7 open questions, resolved):
- **Event alphabet = disjoint MODULES of driver genes** (`event_size`), not single genes. MHN/TreeMHN
  pool patients to fit ONE network, which needs events to RECUR across patients; a single named gene
  out of ~10⁴ is hit in almost no patient, so a single-gene alphabet gives a cohort with no shared
  events and nothing to recover. `event_size=1` recovers the single-gene case.
- **Both gating modes implemented**, `fitness` the default — and the choice turns out to decide the
  whole benchmark (see below). Stated explicitly in the paper.
- **Events are monotone** (never revert, even if a deletion removes the mutated allele) — the
  generative assumption MHN/CBN/TreeMHN are defined under.
- **Ties are recorded, not broken**: events acquired in the same division form one tied group
  (`event_groups`). Flattening them would rank events by the segment their module sits in — a
  property of the layout, not of the evolution — and silently corrupt every ordering ground truth.

**The headline: the OBSERVABLE decides recovery** (§5, measured in `validate_epistasis.py` with the
REAL MHN and TreeMHN, each in its own env). Pairwise `E` acts on **fitness** — how large the clones
carrying a combination grow — while MHN/CBN model the **rate of event acquisition**.

* **The mechanism is large and verified** (paired sweep, only `E` differs): `E` 0→1.5 expands the
  clones carrying the pair **3.5% → 42%** of the tumour (12x), plateauing at 59% once `|E|` exceeds
  the fitness clamp `log(b_max/b_0)`. P(the pair ever *arose*) barely moves (0.03→0.09) — that is a
  mutation property, and `E` does not touch it.
* **Binary "event present" registers almost none of it**, because **recurrent mutation saturates
  presence**: the combination arises many times independently, so it is already present at `E=0`;
  selection changes only how much of the tumour it occupies, which a presence call discards.
* **Real MHN** (binary presence) still retains some signal: planted pair ranked 1st in 4/5 network
  draws vs 1/5 for the empty-`E` control (mean rank 1.4 vs 2.0 of 6). **Suggestive, not significant**
  (Fisher p=0.21, n=5). The control's rank 2.0 ≫ chance 3.5 is a real **false-positive tendency** —
  without the empty-`E` arm it would have looked like recovery. That arm is why the control exists.
* **Real TreeMHN** does **not** beat chance on pairwise `E` (rank 3.4 vs chance 3.5) — and the reason
  is structural, not power. **TreeMHN never sees clone sizes**: `input_tree_df` accepts ONLY
  Patient_ID/Tree_ID/Node_ID/Mutation_ID/Parent_ID and errors on any other column (`weights` is a
  per-TREE weight for tree uncertainty). Its gain over MHN is **event ORDER**; fitness epistasis
  produces **no order**, only frequency — so its extra information is orthogonal to the planted signal.
  **Prediction, tested and confirmed:** on an ORDER signal (accessibility-gated DAG) TreeMHN wins
  decisively — rank **1.80** (top-1 0.6) vs MHN **4.00** (below chance) vs the co-occurrence floor 5.40.

**The 2x2 is the real result — each tool recovers exactly the signal its input encodes:**

| observable a tool consumes | FREQUENCY signal (pairwise `E`) | ORDER signal (accessibility DAG) |
|---|---|---|
| co-occurrence floor (presence) | 4.20 | 5.40 |
| **MHN** — binary presence | 1.60 *(control 2.00 → mostly false-positive tendency)* | 4.00 |
| **TreeMHN** — tree topology (order, NOT sizes) | 3.40 ≈ chance | **1.80** |

(mean rank of the planted pair among 6; chance 3.5; lower is better; 5 network draws.) Recovering
fitness epistasis needs a **frequency-aware** observable, which NEITHER tool consumes — that is the
gap this benchmark identifies.
* **Conjunctive constraints** under **accessibility** gating are recovered **perfectly** (1.00, true
  and reconstructed trees), though at low power (~8 child-carrying lineages/draw); the identical DAG
  under **fitness** gating leaves **no trace** (conjunction holds in 0.14 of lineages).

So the benchmark's real finding is *which observable* carries *which* planted structure — visible only
because iscc knows the answer, and because the empty-`E` control separates recovery from a tool's
false-positive rate.

**The deepest finding — a LEVEL MISMATCH between the planted truth and MHN's input.** MHN's matrix is
**patient-level** ("did this patient acquire event `i` anywhere?"); iscc's `E` is **genotype-level**
(it fires only in a cell carrying both). Measured over 120 patients: 59.5% read as co-occurring for
MHN, but only 14.7% have any clone carrying both — so **75% of MHN's co-occurring patients have the
two events in DIFFERENT subclones, where `E` never fired at all**. Its input is mostly noise with
respect to the planted mechanism. This is not an iscc artefact: it is the intra-tumour-heterogeneity
confound that MOTIVATES TreeMHN (bulk co-occurrence ≠ same-cell co-occurrence), and iscc can put a
number on it — a benchmark real data cannot run. It belongs beside the PEtracer lineage-vs-space and
multi-region sample-tree confounds.

**Mutual exclusivity: the knob was a NO-OP, now fixed but still not recoverable.** A strongly negative
`E` only suppresses division, and the crowding law's `slope = max(0, div − death)` then exempts the
non-dividing clone from density death entirely — it persists and still reads as co-occurring.
`mutual_exclusivity_lethal` (default on) now gives the combination `lethal_death_rate`. That makes the
knob do what it says, but it does NOT make exclusivity recoverable: killing a *clone* barely moves a
*patient-level* co-occurrence that is 75% parallel subclones.

**REFUTED hypothesis (recorded so it is not retried blind):** we predicted that pruning the mutation
tree to DETECTABLE clones (`min_clone_freq`) would reintroduce the frequency channel into TreeMHN's
topology, since clone size would decide which tips survive. Tested at `min_clone_freq=0.02` on 12
FRESH network draws (the only threshold with usable power): planted rank 3.58 vs empty-E control 3.33,
both at chance, paired Wilcoxon **p = 0.78 — not supported**. Pruning decides *which* tips survive, but
a surviving big clone and a surviving small one are still ONE node each, so the trie discards size
regardless; and at that threshold the trees collapse to ~2.5 nodes over 23/40 patients. An earlier
apparent effect (2.80 vs 3.80) came from choosing the threshold after seeing five — do not re-cite it.

**Known limitation (in the paper):** the cohort tumours are ~130 cells, so a clone arising late has
little time to expand and the frequency signal never fully develops. The absolute recovery rates are a
**floor for this regime**, not an estimate for real cohorts; the mechanism (frequency carries the
signal, presence does not) is structural and should generalise. Scaling to ~10⁴-cell tumours via
tau-leaping is the natural next step.

**Retracted 2026-07-15 (was wrong in the first commit):** an earlier version claimed `E` is
"recovered at chance regardless of cohort size". That was an artifact of a **bug** — `min_freq`
filtered CLONES by size and then OR'd their events, so an event carried by dozens of small lineages
(exactly what a favoured combination looks like) was called ABSENT — compounded by sweeping cohort
size when the binding axis was the observable. Detection is now a per-event **cancer-cell fraction**
threshold (`event_cell_fractions`), with a regression test. Do not re-cite the retracted numbers.

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
