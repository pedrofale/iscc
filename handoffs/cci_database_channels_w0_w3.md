# Handoff prompt — W0 + W3: iscc's own L–R database, and receptor-dependent CCI channels

Saved 2026-08-27. Copy the block below as the opening message of a fresh session.
Design reference: `DESIGN_cci_spatial.md` sections **W0** and **W3**.

Target resolution is **Visium** (user decision 2026-08-27), so **W2 is NOT a prerequisite** — see the
corrected chain in the design doc. W1 is superseded and deleted.

---

```
Implement W0 + W3 in iscc: make F8 emit its OWN ligand-receptor database, and turn its single
smoothed density field into RECEPTOR-DEPENDENT, TYPED communication channels. Design:
DESIGN_cci_spatial.md sections W0 and W3. Target resolution is VISIUM.

REPO & ENV
- Repo: /Users/pedroferreira/projects/iscc/repo  (branch `dev`).
- Python/pytest: ~/miniconda3/envs/iscc/bin/python  (`~/miniconda3/envs/iscc/bin/python -m pytest`).
- `conda run` is BLOCKED; call env binaries by absolute path.
- USE `git -C /Users/pedroferreira/projects/iscc/repo ...` FOR EVERY GIT COMMAND. The shell cwd
  resets between calls here; a leading `cd` is not enough and has already put a commit in the wrong
  repository once.
- Conventions: commit on `dev`; `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>` trailer; keep
  the FULL suite green (currently 908 passed, 1 skipped); honest comments, no overselling.
- ASK BEFORE RUNNING ANY SIMULATION, one at a time (16 GB machine).
- There is NO CCI tool env yet. Create `iscc-cellchat` (R + CellChat) following the one-env-per-tool
  pattern; kernelspecs are written by hand because `IRkernel::installspec` shells out to `jupyter`,
  which is not on those envs' PATH (see `handoffs/` siblings and existing `ir-*` kernels).

WHAT EXISTS NOW (F8, in `src/iscc/tumor/models/count.py`)
- `microenv_params = {"hypoxia": {...}, "cci": {...}}`, optional; OFF => output bit-identical.
- `cci` keys today: `emitter_type` (default "immune"), `lengthscale` (2.0), `strength`,
  `n_target_genes`.
- Target genes are drawn at construction (~L408-419) from `np.random.default_rng(self.layout_seed +
  LAYOUT_OFFSET_F8_PROGRAMS)` — the LAYOUT stream, so a cohort shares one programme while each
  patient's evolution stays private. **The L-R database must be drawn the same way**, for the same
  reason the epistasis network is: pooling patients only makes sense against a shared network.
- `_emitter_density(type)` (~L2521) = per-deme fraction of cells of that type, normalised by `_cap`.
- `_cci_field(emitter_type, lengthscale)` (~L2565) = Gaussian-smoothed emitter density over deme
  coordinates.
- `_microenv_deme_mod()` (~L2576) returns an `(n_demes x n_genes)` multiplier and sets
  `self.microenv_truth`. Applied at materialisation.
- INVARIANT TO PRESERVE: F8 modifies the READOUT ONLY. Growth is byte-identical on/off, the modifier
  draws from a dedicated RNG, and OFF is bit-identical. Ligand and receptor levels are READ at
  materialisation and must NEVER feed back into fitness.
- Tests `tests/test_microenvironment.py` (12); validation `validation/validate_microenvironment.py`.

W0 — THE DATABASE
Emit a table of N candidate L-R pairs over iscc's OWN gene indices (abstract names are fine; real
gene symbols are NOT needed — that was W1 and it is dead). Drawn from the layout stream. Columns:
ligand gene, receptor gene, pathway/annotation, and a CLASS. Three classes, deliberately:

  1. ACTIVE          - wired into F8; genuinely spatially driven signalling.
  2. NEUTRAL DECOY   - ligand and receptor genes exist and are expressed COMPARABLY to the active
                       ones, but nothing is wired. Measures the false-positive rate against
                       expression alone.
  3. CLONE DECOY     - ligand/receptor expression CO-VARIES WITH CLONE, and therefore with space
                       (clones are territorial), but no signalling exists. These are the traps.

Concrete way to build class 3 with existing machinery: choose its ligand/receptor genes on SEGMENTS
whose copy number differs between clones, so CNA dosage makes their expression clone-dependent. Their
expression will then correlate between neighbouring cells purely because neighbours share a clone.

HARD CONSTRAINT: class-2 and class-3 ligand/receptor genes must be expressed comparably to class-1's.
If decoys are silent or flat they are rejected on expression grounds rather than on communication
evidence, and the benchmark measures nothing. Draw all three classes from similar expression strata
and CHECK it empirically, do not assume it.

EXPORTERS (verified these accept user databases; do not re-derive)
- CellChat: `interaction_input`, `complex_input`, `cofactor_input`, `geneInfo` CSVs, loaded via its
  documented `Update-CellChatDB` path.
- CellPhoneDB: `gene_input` / `protein_input` / `complex_input` / `interaction_input`, plus
  `--user-interactions-only` so ONLY our interactions are used and no real pairs leak in.
- COMMOT takes an L-R dataframe directly — ASSUMED, not verified. Check before relying on it.
- v1 scope: strict 1:1 pairs, empty complex/cofactor tables. Decide explicitly if you want complexes.

W3 — RECEPTOR-DEPENDENT TYPED CHANNELS
Replace the single `cci` block with a LIST of channels, each
`(emitter type, receiver type, ligand gene, receptor gene, target gene set, strength, lengthscale)`.

  ligand availability at deme d = kernel-weighted sum over nearby demes of their emitters'
                                  LIGAND EXPRESSION (not merely their density -- generalise
                                  `_emitter_density` to an expression-weighted version; per-genotype
                                  expression is already cached)
  effect on receiver cell c     = strength x ligand_available[deme(c)] x receptor_expression[c]

Receptor-dependence is the load-bearing change: every tool in this space scores L-R pairs, so without
it they are being tested on a signal that is not in the data.

STRUCTURAL CONSEQUENCE, PLAN FOR IT: the modifier stops being purely per-deme. The ligand term stays
per-deme, but the receptor term is PER-CELL, so the effective modifier is per-cell x gene. Apply it in
place during materialisation against the existing cells x genes expression matrix; do NOT materialise
a second full matrix. (This is also why W2 is not needed for Visium: per-cell response heterogeneity
comes from the receptor term, not from cell positions.)

GROUND TRUTH TO SURFACE (extend `microenv_truth`)
- the channel table and the full candidate database with class labels,
- per-cell received signal per channel,
- each cell's clone (W4 needs it, and class 3 is meaningless without it).

DONE FOR THIS HANDOFF
- Database generated, both exporters written, round-trip tested (a written database re-reads into
  CellChat without error).
- Channels wired; F8 OFF still bit-identical; growth still byte-identical ON.
- Empirical check that decoy ligand/receptor expression is comparable to active.
- A MINIMAL RECOVERABILITY CHECK, not the full benchmark: on a Visium dataset with channels planted,
  does CellChat recover the ACTIVE channels above the neutral decoys at all? If it cannot, stop and
  report -- everything downstream depends on the planted signal being visible at spot resolution.
- Tests + `validation/` script + figure in the existing style. Update DESIGN_cci_spatial.md W0/W3
  with what was built, and BACKLOG.md.

EXPLICITLY NOT IN SCOPE
- W4, the clone-vs-interaction confound sweep over `dispersal_rate`. That is the next handoff and
  needs this one landed first.
- W2 intra-deme layout. Not required at Visium resolution.
- Real gene symbols. W1 is dead.

CAVEAT TO RECORD IN THE WRITE-UP
At deme resolution the planted signal is piecewise-constant over ~20-25 um blocks. Visium spots are
~55 um, so this sits below the observation scale and is invisible to a Visium benchmark. It WOULD show
at single-cell spatial resolution, which is the condition that makes W2 necessary. Say so plainly
rather than leaving it to be found.
```
