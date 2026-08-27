# Handoff prompt — W0 + W3: iscc's own L–R database, and receptor-dependent CCI

Saved 2026-08-27, cut back the same day after a "don't over-engineer this" correction.
Design reference: `DESIGN_cci_spatial.md` sections **W0** and **W3**.

Target resolution is **Visium**, so **W2 is not a prerequisite**. W1 is dead.
**Parameter budget: ONE new knob** (`n_candidate_pairs`). Hold that line.

---

```
Implement W0 + W3 in iscc: make F8 emit its own ligand-receptor database, and make its CCI effect
RECEPTOR-DEPENDENT. Design: DESIGN_cci_spatial.md W0 and W3. Target resolution is VISIUM.

KEEP IT SMALL. The whole point of this change is ONE new parameter, `n_candidate_pairs`. Everything
else reuses F8's existing `emitter_type` / `lengthscale` / `strength` / `n_target_genes`, which are
already justified and validated. If you find yourself adding knobs, stop and re-read this line. Do
not add a channel-type system, class proportions, or expression-strata tuning.

REPO & ENV
- Repo: /Users/pedroferreira/projects/iscc/repo  (branch `dev`).
- Python/pytest: ~/miniconda3/envs/iscc/bin/python.  `conda run` is BLOCKED; use absolute paths.
- USE `git -C /Users/pedroferreira/projects/iscc/repo ...` FOR EVERY GIT COMMAND. The shell cwd resets
  between calls here and has already put a commit in the wrong repository once.
- Commit on `dev`; `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`; keep the full suite green
  (908 passed, 1 skipped). ASK BEFORE RUNNING ANY SIMULATION, one at a time (16 GB machine).
- No CCI env exists yet. Create `iscc-cellchat` (R + CellChat), one-env-per-tool as usual; write the
  kernelspec by hand (`IRkernel::installspec` shells out to `jupyter`, absent from those envs' PATH).

WHAT EXISTS (F8, `src/iscc/tumor/models/count.py`)
- `microenv_params = {"hypoxia": {...}, "cci": {...}}`; OFF => bit-identical output.
- `cci` keys: `emitter_type` ("immune"), `lengthscale` (2.0), `strength`, `n_target_genes`.
- Target genes are drawn at construction (~L408-419) from the LAYOUT stream
  (`layout_seed + LAYOUT_OFFSET_F8_PROGRAMS`), so a cohort shares one programme. Draw the ligand,
  receptor and candidate pairs from the SAME stream, for the same reason the epistasis network is:
  pooling patients is only justified against a shared network.
- `_emitter_density(type)` (~L2521): per-deme fraction of cells of that type.
- `_cci_field(...)` (~L2565): Gaussian-smoothed emitter density.
- `_microenv_deme_mod()` (~L2576): returns `(n_demes x n_genes)`, sets `self.microenv_truth`.
- INVARIANT: F8 is READOUT-ONLY. Growth byte-identical on/off; OFF bit-identical; ligand/receptor are
  read at materialisation and NEVER feed fitness.
- Tests `tests/test_microenvironment.py` (12); `validation/validate_microenvironment.py`.

W3 — RECEPTOR-DEPENDENCE (a formula change; ZERO new parameters)
Today the emitter contributes by being present, and the receiver's state is irrelevant, so every cell
in a deme gets the same multiplier. Change it to:

    ligand availability at deme d  = the existing smoothed field, but weighted by the emitters'
                                     LIGAND EXPRESSION rather than their bare density
                                     (generalise `_emitter_density`; per-genotype expression is cached)
    effect on receiver cell c      = strength x ligand_available[deme(c)] x receptor_expression[c]

`strength` and `lengthscale` are the ones F8 already has. The ligand and receptor are two designated
genes, drawn from the layout stream — designations, not parameters.

This is load-bearing: every tool in this space scores L-R pairs, so without receptor-dependence they
are tested on a signal that is not in the data. It is also why Visium needs no W2 — per-cell response
heterogeneity comes from the receptor term, not from cell positions.

PLAN FOR ONE STRUCTURAL CONSEQUENCE: the ligand term stays per-deme but the receptor term is per-cell,
so the effective modifier is per-cell x gene. Apply it in place against the existing cells x genes
matrix; do NOT build a second full matrix.

START WITH ONE WIRED CHANNEL. Multiple typed channels can wait until one is shown to work.

W0 — THE DATABASE (one new parameter)
Emit `n_candidate_pairs` candidate L-R pairs over iscc's OWN gene indices — abstract names are fine,
real gene symbols are NOT needed (that was W1; it is dead). Drawn from the layout stream. One pair is
the wired channel; every other pair is unwired.

DO NOT ENGINEER DECOY CLASSES. An earlier draft of this design proposed deliberately placing
"clone-confounded" pairs on clone-varying segments. That was rejected: it multiplies parameters, and
it writes the confound INTO the generative model, which is exactly what the paper's synthesis section
says iscc does not do. Pairs drawn at random already land on clone-varying segments by chance, so
their expression correlates between neighbouring cells purely because neighbours share a clone. The
confound is emergent and free.

The classes are therefore MEASURED, not specified:
  - active         = the wired pair (known by construction),
  - candidate      = every unwired pair (a neutral decoy needs no engineering),
  - clone-correlated = computed per candidate afterwards, as the correlation between its
                       ligand/receptor expression and clone identity. This is W4's analysis axis.
Report that distribution once the data exists; do not tune it.

EXPORTERS (verified; do not re-derive)
- CellChat: `interaction_input` / `complex_input` / `cofactor_input` / `geneInfo` CSVs via its
  documented `Update-CellChatDB` path.
- CellPhoneDB: `gene_input` / `protein_input` / `complex_input` / `interaction_input`, plus
  `--user-interactions-only` so no real pairs leak in.
- COMMOT takes an L-R dataframe directly — ASSUMED, not verified.
- Strict 1:1 pairs; empty complex/cofactor tables. No complexes in v1.

GROUND TRUTH (extend `microenv_truth`)
The candidate database, which pair is wired, per-cell received signal, and each cell's clone.

DONE
- Database generated and exported; a written database re-reads into CellChat without error.
- Receptor-dependence wired; F8 OFF still bit-identical, growth still byte-identical ON.
- The measured clone-correlation distribution over candidates, reported.
- A MINIMAL RECOVERABILITY CHECK, not the full benchmark: on a Visium dataset with one channel
  planted, does CellChat rank the wired pair above the unwired ones at all? If not, STOP and report —
  everything downstream depends on the signal being visible at spot resolution.
- Tests + a `validation/` script and figure in the existing style. Update DESIGN_cci_spatial.md and
  BACKLOG.md with what was actually built.

NOT IN SCOPE: W4's `dispersal_rate` confound sweep (next handoff, needs this landed). W2. Gene symbols.

CAVEAT TO RECORD: at deme resolution the planted signal is piecewise-constant over ~20-25 um blocks.
Visium spots are ~55 um, so it sits below the observation scale and is invisible here. It would show
at single-cell spatial resolution — the same condition that makes W2 necessary. Say so plainly.
```
