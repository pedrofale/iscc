# Handoff prompt — multi-patient cohort (ground truth for cohort analysis & personalized medicine)

Saved 2026-07-03. Copy the block below into a fresh session. This is a real ENGINE milestone — do it
**DESIGN-FIRST** (like F8): read the code, write a short design doc, get the seed-decoupling
prerequisite right, THEN build. Do this AFTER clonealign+inferCNV and the multi-region-phylogeny
benchmark. Companion: `BACKLOG.md` (the "Multi-patient cohort" section is the spec), memory
`iscc-paper-positioning.md`. Branch from current `dev`.

---

```
Design-first, then build the multi-patient COHORT layer for iscc: run many patients (tumours)
separately over a SHARED driver landscape, pool them into batches flexibly, and surface cohort-level
ground truth — for benchmarking multi-patient integration, demultiplexing, recurrence/driver
detection, and (coupled to the treatment module) PERSONALIZED MEDICINE / patient stratification. No
simulator provides this ground truth; it is exactly what real cohorts can never give.

REPO & ENV
- Repo: /Users/pedroferreira/projects/iscc/repo (branch `dev`).
- Python/pytest: ~/miniconda3/envs/iscc/bin/python.
- Conventions: commit on `dev` WITH `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`; keep
  the FULL pytest suite green (this touches the engine); be honest. Match surrounding style.

STEP 0 — DESIGN FIRST. Read the relevant engine code (`tumor/models/count.py` __init__ / Selection
construction; `tumor/components/selection.py`; `data/rna.py` `run_scrna_batches` incl. its
"confounded" design; `data/batch.py`). Write a short `DESIGN_cohort.md` (design/scoping) before
coding. Confirm the two decisions below against the code, then implement.

=== PREREQUISITE ENGINE FIX (do first; it is foundational, not cohort-specific) ===
Comparability by default: two runs with the SAME config MUST use the SAME driver genes (else
recurrence/cohort analysis is meaningless). Today they do NOT — `Selection.make_drivers()`
(`components/selection.py:83`) draws driver/oncogene/TSG POSITIONS from `self.rng`, which is seeded by
the RUN seed, so a different seed → a different driver layout.
- Fix: DECOUPLE a config-determined **layout seed** (a fixed default, or a `genome_seed` / config-hash
  read in `count.py`) that seeds the Selection gene-role layout, from the per-run **evolution seed**
  that drives the stochastic dynamics. Two same-config runs then share driver identities and differ
  only in evolution.
- Keep it BACKWARD-COMPATIBLE: choose the default so existing single-tumour tests still pass (e.g. the
  layout seed defaults to something that reproduces current fixtures, or gate the new behaviour). Add a
  test: two GenotypeTumor(same config, different seed) have IDENTICAL oncogene/TSG/driver index sets.
- Note: the real-genome mode (`genome_spec`) already has a fixed shared landscape — this fixes ABSTRACT
  mode to match that guarantee.

=== THE COHORT LAYER ===
- `Cohort` (e.g. `iscc/cohort/…` or `iscc/tumor/cohort.py`): given ONE shared config + a list of
  per-patient evolution seeds (and optional per-patient config deltas for SUBGROUPS), run N tumours,
  each private in its evolution but sharing the driver landscape.
- **Subgroups (personalized medicine):** allow patient subgroups defined by distinct driver/resistance
  profiles (e.g. subgroup A carries a targeted-therapy-sensitising driver, subgroup B a resistant one),
  so that — coupled to the existing TREATMENT module — a therapy benefits one subgroup and not another.
  This yields ground truth for stratification / treatment-response prediction / biomarker discovery:
  "which patients benefit from which therapy," with a known answer.
- **Flexible patient→batch mapping:** a user-specified assignment of patients to sequencing batches —
  1:1 (each patient its own batch) or N:1 (multiplex/pool several patients into one batch up to a
  technical capacity, cell-hashing / genetic-multiplexing style). Emit per-batch data by reusing the
  scRNA multi-batch / "confounded" machinery (`run_scrna_batches`) with the patient→batch grouping, so
  each batch carries both biological (different patients) and technical variation.
- **Cohort ground truth surfaced:** recurrent-vs-private drivers (recurrence per gene across patients),
  per-patient private mutations, per-cell **patient-of-origin**, true **shared-vs-private cell-state**
  labels, and per-patient subgroup + true therapy-response.

BENCHMARKS TO DEMONSTRATE (pick the strongest 2–3 for the first pass):
- **Multi-patient batch integration** (Harmony/scVI/scANVI/LIGER on the pooled cohort) — score whether
  the method aligns SHARED cell states across patients while preserving PRIVATE ones (iLISI/ARI vs the
  shared-vs-private ground truth). The central over/under-correction failure mode, with no real truth.
- **Demultiplexing** (only meaningful under N:1 pooling) — assign pooled cells back to patient-of-origin
  (souporcell/demuxlet/vireo-style, using per-patient private variants) — accuracy vs the true patient
  label.
- **Recurrence / driver detection** — recover recurrent drivers vs private passengers across the cohort
  (MutSigCV/dNdScv-style) — precision/recall vs the true recurrent set.
- **Personalized-medicine / stratification** — recover the therapy-responsive subgroup from molecular
  profiles; show a therapy helps one subgroup not another (a survival/burden readout via treatment).

DELIVERABLES: `DESIGN_cohort.md`; the engine seed-decoupling fix (+ comparability test); the `Cohort`
layer (+ tests); `validation/validate_cohort.py` → figure(s) for the chosen benchmarks; a manuscript
Results subsection ("iscc provides cohort-level ground truth: shared-vs-private states, and the need
for personalized medicine"); flip the BACKLOG cohort item to DONE. Run the full suite; commit on `dev`.

FEASIBILITY: much is plumbed — real-genome shared landscape, and `run_scrna_batches`' "confounded"
design (different tumours → different batches) is literally the multi-patient case. The genuinely new
work is the seed decoupling, the `Cohort` wrapper, the patient→batch multiplexing, and the ground-truth
bookkeeping. Integration/demux tools (scvi-tools, harmonypy) may need install — guard optional deps.
```
