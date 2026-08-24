# HANDOFF — drug-induced resistance STATE (DESIGN_phenotype_plasticity.md §3.3)

Implement a carried, heritable **resistance cell-state** that is independent of the genome, so that
losing the triggering mutation does not make a cell sensitive again.

Read `DESIGN_phenotype_plasticity.md` §3 and **§3.3 in full before writing code** — §3.3 was written
specifically for this task and contains the measurements that motivate it, the failed approaches, and
the engine trap.

---

## 1. Why (the one-paragraph version)

iscc models treatment resistance as an **SNV at a genomic locus**, in a genome undergoing segmental
copy-number change. A deletion of the copy carrying that SNV takes `n_mut_tr` 1 → 0 and the cell is
drug-sensitive again. Measured: **4.3e-4 per division**, and it is *proportional to the deletion
burden*, so it can only be reduced by asserting a lower-CIN tumour. Reversion is not the problem —
**supply** is: an ~80,000-cell resistant clone does ~30,000 divisions/generation, so ~12–17 revertants
appear every generation, and a handful surviving to the end of dosing multiply ~150× across the
drug-free relapse. Escape mode IV therefore relapses at 99.1% resistant with ~749 sensitive cells
instead of a clean 100%.

The fix is not to suppress reversion — it should keep firing at its calibrated rate — but to make
resistance **a cell state rather than a genomic allele**, so there is nothing for a deletion to remove.

## 2. What to build

A resistance state carried on the cancer-cell representative, **not derived from `genome_summary`**.

- **Entry, genetic (`β_bias`).** Acquiring `n_mut_tr > 0` sets the state. The state is then carried
  independently — a later CNA that deletes the triggering allele **leaves the cell in the state**.
  This is the whole point of the feature.
- **Entry, drug-induced (plasticity).** While a dose is active, a per-division probability of entering
  the state, scaled by **the dose the cell actually receives** — i.e. by `(1 - protection)`, the same
  factor `_kill_amount` uses (`models/count.py:_kill_amount`). A cell the drug cannot reach must not be
  reprogrammed by it.
- **Exit (`τ_relax`).** Off drug the attractor returns to sensitive and the state relaxes back over a
  configurable timescale. `τ → ∞` is permanent resistance; short `τ` is a classical persister.
  **Exit must be possible** — a state that can never be left is the latch, which was tried and
  REJECTED (see §4).
- **Effect.** The state contributes to the drug-protection term, alongside the existing
  `max(treatment_resistance, drug_tolerance)` (`count.py:1301-1302`, `1830-1831`).
- **Cost.** Charge proliferation like the other trait costs, so that OFF drug a cell that exits is
  fitter and the state also reverts *by selection*. That reproduces the adaptive-therapy dynamic rather
  than asserting it.

## 3. The engine trap (read this before designing the data structure)

**The state cannot be derived from the genome — that is the entire point — so a state transition must
MINT A NEW GENOTYPE ID.** Two cells with identical genomes and different states are different engine
entities. This is expressible: genotype ids come from a counter, not from genome content
(`components/cell.py:set_genotype_id`), and `Cell.divide()` shallow-copies so a daughter inherits both
the state and any flags (this is how `_tx_mutagenized` already works).

The cost is that it **inflates the append-only genotype registry**, which is iscc's known scalability
wall. §3 of the design already flags "(genotype × epistate) sublineages" as a real change to the count
engine's caching. **Quantify this before committing to a design** — measure genotype-count growth with
the feature on versus off at rig scale, and say plainly what it costs.

## 4. What NOT to do (all tried; do not repeat)

- **A trait LATCH** (`treatment_resistance` can never decrease). Reaches 100% resistant, was implemented
  and then **REJECTED as unbiological and fully reverted** — resistance alleles genuinely are lost.
  The state model is different *because exit remains possible and is governed by its own rate*.
- **Suppressing reversion.** `amp_prob` 0.5→0.8 halves it but inflates ploidy 2.34→3.06, which dilutes
  EVERY graded trait via `effect ** (2*n_mut/ploidy)` (`selection.py:434`).
- **Raising `kill_rate`.** Kills revertants faster but starves the de novo origin mode IV requires;
  and under the graded map a one-hit clone dies above `kill_rate ≈ 1.22` anyway.
- **Turning up `mutagenicity`.** It CANNOT shift the acquisition:reversion balance — measured 0.14 at
  mutagenicity 1.0 and 0.14 at 4.0 — because `mutation_rate` is the mutate-vs-disperse FATE probability
  and scales both branches together. (`mutagenicity_target="snv"`, commit 22899ea, fixes that half.)
- **Scaling `cnv_prob` down to compensate for a raised `mutation_rate`.** Held the CNA rate fixed
  exactly, tests passed, and 0/4 seeds relapsed at all — fewer CNAs lower driver-dosage fitness, and
  under proliferation kill the DIVISION count scales with `b`.

## 5. Acceptance

The target is **exactly 0 sensitive cells at relapse WITH the drug-free tail present.** Keeping the
drug on to the end also gives ~0 and is already known — it is not the goal here.

Baseline to beat (rig scale, `mode4_scratch/`):

    variant                              sensitive @ relapse    % resistant
    shipped CIN, seed 5 (rig/PM5.npz)            749              99.095%
    low-CIN cnv_prob=0.0045, seed 2 (LC2)        215              99.715%

Required, all four simultaneously:
1. **0 sensitive cells** at the final row, with `relapse_steps=90` (the shipped drug-free tail).
2. **Reversion still firing** at its calibrated rate — do NOT reduce `cnv_prob`/`amp_prob` to get there.
   Verify CNA deletions of the resistance allele still occur; the state is what makes them not matter.
3. **Mode IV still passes all four criteria** — `mode4_scratch/accept_mode4.py`.
4. **Default OFF and byte-identical when off**; full suite green with golden hashes unaffected
   (887 passed, 1 skipped, ~12 min).

Working mode IV config to build on (rig, seed 5):
`prop_treatment_resistance=0.00025 treatment_resistant_effects=2.8 treatment_resistance_cost=0.35`
`n_snvs_per_allele=0.02 mutagenicity=20.0 mutagenicity_target=snv kill_mode=proliferation`
`kill_rate=1.0 chemo_steps=120 relapse_steps=90` — driven via `mode4_scratch/trace_mode.py`.

## 6. Operational constraints (hard)

- Repo `/Users/pedroferreira/projects/iscc/repo`, branch **dev**. Commit on dev with a
  `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>` trailer. **NEVER commit to `main`** — main is
  a filtered publish via `scripts/publish-main.sh`, and internal docs (DESIGN_*.md, handoffs/, BACKLOG.md,
  analysis/, scripts/) must never reach it.
- Python: `/Users/pedroferreira/miniconda3/envs/iscc/bin/python`. **`conda run` is BLOCKED.**
- **ASK PERMISSION BEFORE RUNNING ANY SIMULATION.** The user is explicit about this. Run them **strictly
  one at a time** — 16 GB machine, swap has repeatedly been within a few hundred MB of exhaustion.
  Scratch runs go through `mode4_scratch/run1.sh` (swap watchdog + time cap).
- Never run `git restore/checkout/stash/clean/reset`.
- Scratch scripts and all escape-mode artefacts live OUTSIDE the repo in
  `/Users/pedroferreira/projects/iscc/mode4_scratch/`.

## 7. Engine gotchas that will cost you an hour each

- **`grow(n_steps=1)` runs ZERO update steps** (`for local_step in range(n_steps - 1)`) and never
  reaches the treatment block. Treated-step tests need `n_steps >= 2`.
- **`Selection(layout_seed=...)` without an explicit `rng=`** was non-reproducible until commit 2af92d3;
  engines pass `rng=self.layout_rng`, standalone probes did not.
- **`_tx_mutagenized` is inherited through `divide()`'s shallow copy**, which is why a dose-scaled
  mutator is a no-op for a de novo clone (it inherits the mutator from its drug-exposed sensitive
  ancestor). Expect the same inheritance semantics for the state — that is desirable here, but be
  deliberate about it.
- **`prop_treatment_resistance: 0.0001` draws ZERO resistance genes** and silently disables the axis.
  0.00025 gives 2 loci — and in the shipped layout **both land on the same segment**, so one deletion
  clears both. Relevant if you compare against genomic resistance.

## 8. Context worth having

- Whiting, Househam, Baker, Sottoriva & Graham, *Phenotypic noise and plasticity in cancer evolution*,
  **Trends in Cell Biology 34:451-464 (2024)** — splits non-genetic variation into **plasticity**
  (environment-responsive) and **noise** (cell-intrinsic stochastic), separated operationally by lineage
  tracing. All four of iscc's current escape modes are **genetic**; this feature is what gives iscc the
  other two categories, and iscc has lineage tracing as ground truth.
- França, …, Yanai, *Drug-induced adaptation along a resistance continuum in cancer cells*
  (bioRxiv 2022.06.21.496830; Nature 2024) — gradual dose escalation walks cells up a continuum of
  increasingly resistant states via transcriptional/epigenetic reprogramming. The behaviour to reproduce.
- Sharma et al., *Cell* 141:69 (2010) — the chromatin-mediated reversible drug-tolerant state. This is
  the biology that makes "maintained phenotype" defensible where "indelible allele" was not.
- Memory `maley-escape-modes` carries the full measurement history.
