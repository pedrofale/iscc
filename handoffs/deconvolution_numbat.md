# Handoff prompt — cashing in R13: spatial deconvolution (cell2location/RCTD) + Numbat

Saved 2026-07-15. Copy the block below into a fresh session. **BLOCKED ON R13** — needs the program layer
(deconvolution) and allele-specific expression / BAF (Numbat) from `handoffs/expression_programs.md`.
**Do not start until R13 has landed on `dev`.** These are the two benchmarks R13 was built for; without
them R13 is a realism upgrade with no payoff beyond scDEF. Follows the `clonealign`/`inferCNV` precedent
exactly (`validation/integration_common.py`, `validation/README_integration.md`). Branch from `dev`.

---

```
Build two integration benchmarks on iscc that R13 unblocked: (1) spatial deconvolution
(cell2location + RCTD) — the FLAGSHIP — and (2) Numbat (allele-aware CNA-from-expression). Each RUNS THE
REAL TOOL on iscc data and SCORES it against iscc's ground truth, following the clonealign/inferCNV
pattern already in the repo (read `validation/integration_common.py` and the "non-circular ground truth"
Results section in `manuscript/paper.tex` first — match that shape).

PRECONDITION: R13 must be merged (`DESIGN_expression.md`). Deconvolution needs the program layer (cell
types = program combinations → realistic co-expression); Numbat needs allele-specific expression / BAF.
If either is missing, STOP and report rather than faking it.

REPO & ENV
- Repo: /Users/pedroferreira/projects/iscc/repo (branch `dev`).
- Python/pytest: ~/miniconda3/envs/iscc/bin/python.
- Conventions: commit on `dev` WITH `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`; keep the
  FULL suite green; **every external tool in its OWN `iscc-<tool>` env** (never the core env — see
  `validation/README_integration.md`, "one dedicated env per external tool"); be honest about negatives.

WHY THESE TWO (state it in the paper): both invert a genotype/state→expression relationship that in iscc
EMERGES from evolution + microenvironment rather than being imposed — the non-circularity argument
already made for clonealign/inferCNV. R13 makes them testable at all.

=== DEMO 1: SPATIAL DECONVOLUTION (cell2location + RCTD) — FLAGSHIP ===
**Why iscc is unique here:** it emits BOTH an scRNA reference AND Visium spots from the SAME tumour, with
the TRUE per-spot cell-type/clone composition. Real benchmarks must borrow a reference from a different
sample and can never control the mismatch. iscc can.
- **Data gen — mirror how this is ACTUALLY done.** Grow ONE tumour (F8 niches + R13 programs ON). Take
  the **Visium section** from one region. Take a **SEPARATE biopsy (F1) from a DIFFERENT REGION of the
  SAME tumour**, push it through **dissociation (F2)**, and emit **scRNA** from that — *that* is the
  reference. The reference is always same-tumour but a **different sample**; it is NOT another patient's
  scRNA (nobody does that by choice). Keep the true per-spot composition (`visium.py` already records
  `n_cells`, member cell ids, dominant clone + clone fractions) AND the true composition of the reference
  biopsy — so the reference mismatch itself is measurable, not just its effect.
- **Tools:** `cell2location` (env `iscc-cell2location`), `RCTD`/`spacexr` (env `iscc-rctd`, R).
  destVI/Tangram optional — don't gold-plate.
- **Metrics:** per-spot composition accuracy (JSD / RMSE / correlation of true vs inferred proportions);
  per-cell-type recall; behaviour vs cells-per-spot / spot size.
- **THE HEADLINE EXPERIMENT — what does a realistic (same-tumour, different-sample) reference cost, and
  WHY?** The reference is imperfect for three REAL and SEPARABLE reasons, and iscc is the only thing that
  can dial them independently *and* know the truth for each:
    1. **Regional mismatch** — the scRNA biopsy came from a different part of the tumour, so it carries a
       different clonal / niche composition. Sweep biopsy distance / overlap with the section (under
       clonal territories — low `dispersal_rate` — a distant biopsy sees very different clones).
    2. **Dissociation bias (F2)** — the reference passes through dissociation, which distorts cell-type
       composition; the Visium section does not. Sweep the bias strength.
    3. **Assay / batch (F3)** — scRNA and Visium are different technologies and batches.
  **Ceiling control:** an oracle reference built from the EXACT cells in the section (no regional
  mismatch, no dissociation) = the upper bound; then degrade toward realism and ATTRIBUTE the error to
  each source. **That decomposition is the headline** — "what an imperfect reference actually costs, split
  into its three real causes" — not a raw accuracy number. Real benchmarks cannot do this: they never know
  the true per-spot composition, and cannot separate the three sources.
  *(Optional extreme endpoint, clearly labelled as such: a different patient's scRNA via the `Cohort`
  layer = the "no matched scRNA, fall back to an atlas" scenario. Not the main axis.)*
- **Narrative tie-in:** this makes deconvolution a DOWNSTREAM CONSEQUENCE of the paper's existing
  "Biopsy and dissociation shape the sampled data" Results section — the sampling biases already modelled
  there propagate into deconvolution error, with ground truth at both ends. Cross-reference it.
- **iscc-only extra axes (pick the interesting ones):** does accuracy degrade with **CNA burden** (clones
  become transcriptionally distinct)? Do the tools **confuse clones with cell types** (both are
  "populations" but only one is a program combination)? The latter is a nice tie-in to the R13 confound arc.

=== DEMO 2: NUMBAT (allele-aware CNA-from-expression) ===
**Why:** inferCNV (already benchmarked) uses expression only. Numbat adds **allele frequencies (BAF)** +
a phylogeny prior — it is the stronger, allele-aware successor, and R13's ASE exists precisely to make it
testable. The clean question iscc can answer: **does the allele layer actually help, and by how much?**
- **Data gen:** reuse `integration_common.py`'s multi-clone tumour (distinct segmental CNAs) with R13
  allele-specific expression ON → per-cell BAF available.
- **Score vs truth:** per-cell/segment CN correlation vs true `cell_cnv`; clone assignment (and Numbat's
  inferred clone tree) vs true clones; malignant-vs-normal AUC. **Head-to-head vs inferCNV** on the same
  tumour — that comparison is the deliverable.
- **Env:** `iscc-numbat` (R).
- **SCOPE THE INPUT INTERFACE FIRST — this is the real risk.** Numbat expects (a) a gene×cell count matrix
  and (b) **allele counts at phased SNP sites** (normally cellsnp-lite on a BAM + a population phasing
  reference). iscc's abstract genome has no phasing panel. Options, in order: (i) feed Numbat allele
  counts DIRECTLY (it accepts an allele dataframe) built from iscc's per-allele expression / F7b
  mutation-aware scRNA reads, bypassing pileup+phasing; (ii) use the real-genome mode if it maps more
  cleanly. **Spend the first pass scoping this and report the chosen route** — if neither fits, say so
  rather than contorting the simulator to suit the tool.

SHARED
- Extend `validation/integration_common.py` (shared data-gen + run/score helpers); thin per-tool runner
  scripts executed via `subprocess` into the dedicated envs, data crossing as files — the existing pattern.
- Document each new env's build recipe in `validation/README_integration.md`.
- Bib entries to ADD (flag "auto-added — verify"): cell2location (Kleshchevnikov et al. 2022), RCTD
  (Cable et al. 2022), Numbat (Gao et al. 2023).

DELIVERABLES
- `validation/validate_deconvolution.py` → `manuscript/figures/validation_deconvolution.png` (matched vs
  mismatched reference; accuracy vs cells-per-spot; the CNA-burden / clone-vs-celltype axis).
- `validation/validate_numbat.py` → `manuscript/figures/validation_numbat.png` (Numbat vs inferCNV
  head-to-head; CN correlation; clone recovery).
- Tests (`tests/test_deconvolution.py`; extend `tests/test_integration.py`): small deterministic tumour,
  accuracy > chance, optional deps guarded so they SKIP cleanly when an env is absent.
- Manuscript: fold into the "non-circular ground truth for multi-modal integration" Results section —
  deconvolution's matched-vs-mismatched reference as a headline; Numbat-vs-inferCNV as the "does allele
  information help" result. Wire figures in; keep cites/refs resolving.
- Flip the deconvolution + Numbat rows in `BACKLOG.md`. Full suite green. Commit on `dev`.

HONEST NOTES
- **Numbat's input interface is the main risk** — scope it before building; a documented "we fed allele
  counts directly because the phasing panel doesn't apply to an abstract genome" is a perfectly good
  outcome, and is itself worth a sentence in the paper.
- If deconvolution accuracy comes out high and boring in the matched case, that's EXPECTED — the interest
  is entirely in the mismatched-reference, CNA-burden and clone-vs-celltype axes. Don't tune to
  manufacture a failure; report what you find.
- Don't gold-plate: two tools per demo maximum, representative not exhaustive.
```
