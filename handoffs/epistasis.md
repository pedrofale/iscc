# Handoff prompt — R14: epistasis / evolutionary-dependency structure in selection

Saved 2026-07-15. Copy the block below into a fresh session. **Paper-1 work** (decision 2026-07-14).
Full design in `DESIGN_epistasis.md` — read it first. Sibling handoff: `handoffs/expression_programs.md`
(R13, the data-side prerequisite). Depends on the multi-patient cohort milestone (done) as its substrate.
Branch from current `dev`.

---

```
Build R14 for iscc: add epistasis / ordered-dependency structure to the selection model so cohort
DNA-progression tools have a KNOWN network to recover, and benchmark them against it. Full design in
DESIGN_epistasis.md — READ IT FIRST, don't re-derive. Paper-1 work.

REPO & ENV
- Repo: /Users/pedroferreira/projects/iscc/repo (branch `dev`).
- Python/pytest: ~/miniconda3/envs/iscc/bin/python.
- Conventions: commit on `dev` WITH `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`; keep the
  FULL suite green (451 now); each external tool in its OWN `iscc-<tool>` env (see
  `validation/README_integration.md`); be honest about weak/negative results.

WHY: MHN, TreeMHN, CBN/H-CBN, REVOLVER recover a NETWORK of promoting/inhibiting/ordering dependencies
between evolutionary events across a cohort. iscc's fitness is currently ADDITIVE (driver COUNT in
abstract mode — `n_mutated_drivers` / the per-role `n_mut_*` counts in `genome_summary`; per-arm in
real-genome mode), so the true network is ~EMPTY and a benchmark would only measure a method's
false-positive rate. Plant a known structure ⇒ a real benchmark.

=== THE MODEL (DESIGN_epistasis.md §2) — three interaction types, all planted = ground truth ===
- **Pairwise epistasis `E_ij`**: event i's fitness effect modulated by the presence of event j
  (log-additive/multiplicative). E>0 synergy/co-selection; E<0 antagonism.
- **Conjunctive / ordered constraints (CBN-style)**: event B only beneficial (or only accessible) once A
  is present — a dependency DAG inducing temporal ORDER in the lineage.
- **Mutual exclusivity / synthetic lethality**: strongly negative E ⇒ co-occurrence deleterious ⇒ events
  look mutually exclusive across the cohort (the DISCOVER/MEGSA signal).

=== ENGINE INTEGRATION (DESIGN_epistasis.md §3) ===
Fitness reads a genotype's event set already; add the interaction term:
  log fitness += Σ_i β_i·x_i + Σ_{i<j} E_ij·x_i·x_j     (+ DAG gating for conjunctive constraints)
- Interaction is a PURE FUNCTION of a genotype's event set ⇒ caches per genotype; tau-leaping-safe.
- **OFF-BY-DEFAULT**: empty `E` / no DAG ⇒ current additive behaviour, bit-identical. (The F8 discipline.)
- Decide + document: ordering via **fitness gating** (B inert until A) vs **accessibility gating** (B
  cannot arise until A) — these are different generative stories that the progression models make
  different assumptions about. This choice matters; state it in the paper.

=== EXPOSE THESE PARAMETERS ===
- `epistasis_params`: `n_interactions` / `network_sparsity` (how many non-zero `E_ij`), `network_topology`
  (random / hub / chain), `interaction_strength` (mean/sd — must be detectable yet realistic),
  `prop_synergy` vs `prop_antagonism`, `mutual_exclusivity_strength`.
- `dependency_params` (conjunctive): `n_constraints`, DAG depth/branching, `gating_mode`
  (`fitness` | `accessibility`).
- Event alphabet: abstract driver roles (v1) vs real-genome arms/genes (later).
Document in `PARAMETERS.md` (defaults + valid ranges). Surface ground truth: the true `E` matrix /
dependency DAG / order constraints, plus the realised per-patient event ORDER along each lineage.

=== VALIDATION (pairs with the cohort milestone, DESIGN_cohort.md — done) ===
Plant a network → run a COHORT (shared driver landscape, per-patient private evolution; the `Cohort`
layer + `layout_seed` decoupling already exist) → collect per-patient trees (true via
`iscc.integrations.to_newick`, and/or reconstructed from scDNA) → run the tools → score vs truth.
- **Tools:** **TreeMHN** (flagship — consumes per-patient mutation TREES, which iscc has exactly),
  **MHN** (cross-sectional), + CBN/H-CBN and/or REVOLVER if cheap. Each in its own env (`iscc-treemhn`,
  `iscc-mhn`, …). Add bib entries (flag "auto-added — verify"): MHN (Schill et al. 2020), TreeMHN (Luo,
  Kuipers & Beerenwinkel 2023), CBN, REVOLVER (Caravagna et al. 2018).
- **Metrics:** recovered-edge **precision/recall** vs the planted `E`/DAG; **order accuracy** for
  conjunctive constraints; mutual-exclusivity/co-occurrence sign recovery.
- **Sweeps worth doing:** #patients in the cohort (how many tumours before the network is recoverable?);
  interaction strength; network sparsity; true-trees vs reconstructed-trees (does tree-inference error
  destroy the progression signal? — a nice iscc-only question).
- **Sanity control:** empty `E` ⇒ the tools should recover ~nothing (false-positive rate).

DELIVERABLES: the epistasis/dependency model in `Selection` (off-by-default, cached, tau-leap-safe);
parameters + `PARAMETERS.md` docs; ground truth surfaced; tests (off ⇒ bit-identical; planted network is
recoverable in an easy regime; interaction caching correct); `validation/validate_epistasis.py` → figure
(edge precision/recall vs cohort size & interaction strength; true vs reconstructed trees); a manuscript
Results paragraph in the integration/cohort arc; flip the BACKLOG R14 item. Full suite green; commit `dev`.

HONEST NOTES: this gates a whole benchmark category — without it the MHN/TreeMHN row tests specificity
only. Don't over-tune the planted network to make the tools look good (or bad); the interesting result is
*where* recovery breaks down (cohort size, strength, tree error). Report negatives.
```
