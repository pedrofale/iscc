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
- `iscc-cellchat` ALREADY EXISTS — CellChat 2.2.0.9001 on R 4.5.3. Do NOT rebuild it; the install was
  awkward and the route is recorded. READ `validation/README_cellchat.md` FIRST: it is the verified
  round-trip report and it will save you hours.

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

WHERE IT GOES (checked; this is a SMALL change, do not make it big)
Materialisation already loops over (deme_idx, gid) at ~L3188-3200 and does
`exp_row = exp_cache[gid] * deme_mod[deme_idx]`, caching the result in `mod_exp_cache[(deme_idx, gid)]`.
Expression is cached PER GENOTYPE, so the receiver's receptor level is just `exp_cache[gid][receptor]`
— already in scope in that loop. There is NO per-cell x gene matrix and none needs building.

  keep hypoxia as a genuine per-deme row;
  make the CCI factor a per-(deme, genotype) SCALAR:
      f = 1 + strength * ligand_avail[deme_idx] * receptor_level[gid]
      exp_row = exp_cache[gid] * hypoxia_mod[deme_idx]
      exp_row[cci_targets] *= f
`mod_exp_cache` already has exactly this granularity, so memory does not move.

THE ONE THING THAT NEEDS CARE — NORMALISATION.
`strength` is an ALREADY-CALIBRATED knob. Today `_cci_field` returns a smoothed density in [0,1] (a
fraction of carrying capacity), which is what gives `strength` a stable meaning. Weighting by raw
ligand expression and multiplying by raw receptor expression breaks that bound and SILENTLY RESCALES
`strength`, so its existing validation no longer applies. Normalise both terms back to [0,1]-ish —
dividing each by its mean across genotypes is the obvious choice — and state the choice in the
docstring. This is a definition, not a new parameter, but getting it wrong invalidates calibration
without any test failing.

SECOND, SMALLER: the allele-resolved path at ~L3200 (`exp_p_cache` / `exp_m_cache` * `mod_row`) needs
the same factor applied, or the allele layer and the total layer disagree whenever F8 is on.

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

EXPORTERS — PROVEN END TO END, constraints below are measured, not guessed
Full report: `validation/README_cellchat.md`. Headline: CellChat accepts a wholly invented database
over iscc-style identifiers and runs BOTH the RNA and the SPATIAL pipeline, returning our pairs. There
is no species validation anywhere once `object@DB` is replaced.

- `updateCellChatDB()` needs only `ligand` + `receptor`; it auto-fills `pathway_name`,
  `interaction_name`, `interaction_name_2` and the cofactor columns. `geneInfo` needs only `Symbol`.
- **THE HARD CONSTRAINT, AND IT FAILS SILENTLY.** `extractGeneSubset()` does
  `intersect(geneSet, geneInfo$Symbol)`. Any ligand or receptor NOT in `geneInfo$Symbol` is dropped
  with NO error — the interaction row survives, the pair simply never appears in the output. iscc MUST
  emit `geneInfo` listing every gene its database references, and the writer MUST assert
  `setequal(extractGene(db), expected_genes)` immediately after building. That one assertion catches
  the entire failure class, including the worse variant where passing `gene_info = NULL` substitutes
  the human table and silently yields zero genes.
- **NEVER write a single-column `complex` or `cofactor` table.** R drops a 1-column data.frame to a
  vector on `[i, ]` and CellChat dies with `no applicable method for 'select'`. Use ZERO columns or
  two or more. This bites precisely when writing a "minimal" cofactor CSV.
- **`annotation` must be exactly one of four strings**: `Secreted Signaling`, `ECM-Receptor`,
  `Non-protein Signaling`, `Cell-Cell Contact`. CellChat `factor()`s against those levels, so anything
  else becomes NA and corrupts the diffusive-vs-contact split. It is NOT auto-added.
- CellPhoneDB: `gene_input` / `protein_input` / `complex_input` / `interaction_input` plus
  `--user-interactions-only`. Documented but NOT round-trip tested — do CellChat first.
- COMMOT takes an L-R dataframe directly — ASSUMED, not verified.
- Strict 1:1 pairs; empty complex/cofactor tables. No complexes in v1.

SPATIAL MODE — WORKS, AND HAS ITS OWN DEMANDS
Verified on a 440-spot Visium-like hex grid in micrometres. The geometry genuinely bites: two groups
381 um apart scored exactly 0 against a 250 um contact range.
- Inputs: a 2-column `coordinates` (renamed to `x_cent`/`y_cent`), `spatial.factors` with BOTH `ratio`
  and `tol` (`ratio=1` if authoring in um, `tol=spot.size/2`), `meta$samples` as a FACTOR, and one of
  `contact.range` / `contact.knn.k` — `computeCommunProb` errors on its own defaults otherwise.
- **UNITS ARE A REAL DECISION FOR US.** `scale.distance` is coupled to the coordinate units and needs
  the minimum scaled distance >= 1. iscc's Visium coordinates are in DEME units, not um. Either emit
  micrometres (the project anchors a deme at ~20-25 um) or recompute `scale.distance` for deme units.
  Decide it explicitly and write it down; it fails loudly and suggests a value, so it will not slip
  through silently, but it will waste time.
- `netVisual(..., layout="spatial")` is BROKEN upstream in 2.2.0.9001 (passes an `idents.use` that is
  not one of its formals). Use `netVisual_aggregate(..., layout="spatial")`.

GROUND TRUTH (extend `microenv_truth`)
The candidate database, which pair is wired, per-cell received signal, and each cell's clone.

DONE
- Database generated and exported; a written database re-reads into CellChat without error.
- Receptor-dependence wired; F8 OFF still bit-identical, growth still byte-identical ON.
- The measured clone-correlation distribution over candidates, reported.
- A MINIMAL RECOVERABILITY CHECK, not the full benchmark: on a Visium dataset with one channel
  planted, does CellChat rank the wired pair above the unwired ones? SCORE BY `prob`, NOT by
  significance — communication probability is computed on GROUP-AVERAGED expression, so any non-zero
  expression floor gives every group pair a non-zero, "significant" edge. A dense, all-significant
  network is the expected background here, and p-values will not discriminate. If the wired pair does
  not outrank the unwired ones by probability, STOP and report: everything downstream depends on the
  planted signal being visible at spot resolution.
- Tests + a `validation/` script and figure in the existing style. Update DESIGN_cci_spatial.md and
  BACKLOG.md with what was actually built.

NOT IN SCOPE: W4's `dispersal_rate` confound sweep (next handoff, needs this landed). W2. Gene symbols.

CAVEAT TO RECORD: at deme resolution the planted signal is piecewise-constant over ~20-25 um blocks.
Visium spots are ~55 um, so it sits below the observation scale and is invisible here. It would show
at single-cell spatial resolution — the same condition that makes W2 necessary. Say so plainly.
```
