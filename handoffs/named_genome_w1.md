# Handoff prompt — W1: a named genome (naming layer)

Saved 2026-08-27. Copy the block below as the opening message of a fresh session.
Design reference: `DESIGN_cci_spatial.md` (section **W1**). Motivation: the sCCIgen/scSpatialSIM
comparison — see the same doc's header and `BACKLOG.md`.

W1 is the FIRST link in a chain (`W1 symbols -> W2 intra-deme layout -> W3 L-R channels -> W4 the
confound benchmark`). It gates the CCI work because CellChat/CellPhoneDB look up ligand-receptor
pairs by gene symbol.

---

```
Implement W1 — a NAMED GENOME for iscc: real human gene symbols on real chromosome arms, replacing
the abstract `G_<segment>_<index>` gene names, as an opt-in layer. Design: DESIGN_cci_spatial.md
section W1. This is a naming/annotation layer, NOT a change to the evolutionary engine.

WHY IT MATTERS (don't skip): this gates the cell-cell-communication work. CellChat and CellPhoneDB
resolve ligand-receptor pairs BY GENE SYMBOL, so without real symbols they cannot be run on iscc
data at all, and a planted L-R channel would have nothing to match against.

REPO & ENV
- Repo: /Users/pedroferreira/projects/iscc/repo  (work on branch `dev`).
- Python/pytest: ~/miniconda3/envs/iscc/bin/python  (`~/miniconda3/envs/iscc/bin/python -m pytest`).
- `conda run` is BLOCKED; call env binaries by absolute path.
- Conventions: commit on `dev`; end commits with the `Co-Authored-By: Claude Opus 5
  <noreply@anthropic.com>` trailer; keep the FULL suite green (currently 908 passed, 1 skipped);
  be honest in comments and docs, no overselling. Match surrounding style.
- ASK BEFORE RUNNING ANY SIMULATION, and run them one at a time (16 GB machine).

READ FIRST
- `DESIGN_cci_spatial.md` — section W1 for the spec, the header for why.
- `src/iscc/inference/genome.py` — `GenomeSpec` ALREADY carries the ~39 human autosomal arms with
  real lengths (UCSC cytoBand), COSMIC/Davoli oncogene + TSG counts, and Charm scores; arm length
  already sets `segment_sizes`. It is wired only into the inference side today. REUSE IT.
- `validation/data/build_realgenome_data.py` — the existing "download + derive + record provenance"
  pattern that produced `validation/data/realgenome_arms.csv`. A gene-level annotation should follow
  the same pattern.

THE CHOKE POINT (this is the whole job, structurally)
- `src/iscc/tumor/components/selection.py::get_gene_names(gene_prefix='G')` builds every gene name as
  `f'{prefix}_{segment}_{i}'`. Everything downstream — `cell_exp`, `cell_cnv`, `cell_snv`, the assay
  outputs' `var_names`, the DNA loci — takes its columns from here (see `tumor/tumor.py:347` and
  `tumor/models/count.py:3090`).

BLAST RADIUS — code that PARSES the name format (all of it; I checked)
- `selection.py::gene_to_pos()` — splits the name to recover `(segment, position)`.
- `validation/make_analysis_data.py:440` — `int(g.split("_")[1])` to get a locus's segment.
- `validation/integration_common.py:266` — same.
=> A named genome MUST provide a name-INDEPENDENT gene->segment lookup, and those call sites must be
   migrated to it. Do not leave string parsing that silently breaks on real symbols.

REQUIREMENTS
1. Opt-in. Default behaviour (abstract `G_seg_i` names) must be UNCHANGED and bit-identical, so
   published numbers and the existing suite do not move. Naming is a config/flag, not a rewrite.
2. A reproducible, seeded bijection `abstract gene index -> real gene symbol`, stable within a run.
3. Symbols placed on arms consistently with the CNA structure: a gene's copy number is the copy
   number of the arm it sits on, and genes ordered by genomic position within their arm, so that
   contiguous abstract segments stay contiguous in real coordinates. (Arm length already drives
   `segment_sizes`, so the per-arm gene budget is already proportional to arm length.)
4. RESERVED SYMBOLS. The mapping must be able to GUARANTEE that a caller-supplied set of symbols is
   present and placed on their true arms — this is how W3's ligand/receptor genes get into the
   genome. A purely random assignment is not acceptable. Design the API for this from the start.
5. Symbols must surface in the assay outputs' `var_names` (scRNA, Visium, single-cell spatial) and in
   the DNA loci, not only in `cell_exp`.

EXPLICIT NON-GOALS (do not drift into these)
- Matching real per-gene marginals, dispersion, dropout or gene-gene correlation. That is a separate,
  heavier design ("anchor to a reference at the assay layer") and is DEFERRED by decision.
- Simulating 20k genes in the engine. Evolution acts on segments and a small driver set; genes are
  ride-alongs. The gene budget stays what the config says.
- Changing selection, fitness, or anything that alters growth.

DATA DEPENDENCY
A real gene annotation with genomic positions (symbol, chromosome, start) is needed. Decide the
source, derive a committed CSV under `validation/data/` following `build_realgenome_data.py`'s
provenance pattern, and keep the raw download git-ignored. State the genome build.

OPEN QUESTION TO RESOLVE FIRST — ASK THE USER
Which cell-cell-communication database will W3 target (CellChatDB human is the assumption)? It
determines which ligand/receptor symbols requirement 4 must reserve. Do not guess.

TESTS / DONE
- Unit tests: bijection stability under a fixed seed; reserved symbols always present and on the
  correct arm; gene->segment lookup agrees with the abstract layout; default-off is bit-identical.
- Migrate the two validation call sites off string parsing; full suite green.
- A short demo showing a scRNA output whose `var_names` are real symbols, with copy number varying
  by arm as expected.
- Update `DESIGN_cci_spatial.md` W1 with what was actually built, and BACKLOG.md.
```
