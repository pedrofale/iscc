# Handoff prompt — R13: gene-program expression backbone + genotype→expression realism

Saved 2026-07-15. Copy the block below into a fresh session. **Paper-1 work** (decision 2026-07-14).
This is a large, staged engine feature — the full spec (composition, parameter surface, validation) is in
`DESIGN_expression.md`; **read it first, don't re-derive**. Sibling handoff: `handoffs/epistasis.md` (R14).
Branch from current `dev`.

---

```
Build R13 for iscc: model expression as GENE PROGRAMS (the backbone) plus genotype-driven dosage/SNV
overlays, and validate that program-inference tools can recover the true programs — and how that degrades
with SNV/CNA burden. The full design (composition, parameter surface §4.1, validation §4.2) is in
DESIGN_expression.md — READ IT FIRST. Paper-1 work; do it staged and design-first.

REPO & ENV
- Repo: /Users/pedroferreira/projects/iscc/repo (branch `dev`).
- Python/pytest: ~/miniconda3/envs/iscc/bin/python.
- Conventions: commit on `dev` WITH `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`; keep the
  FULL suite green (451 now); each external tool in its OWN `iscc-<tool>` env (never the core env, see
  `validation/README_integration.md`); be honest — report weak/negative results as such.

WHY (the non-circularity argument): the DNA↔RNA integration tools (clonealign, inferCNV, Numbat,
cardelino, PhylEx) all INVERT a genotype→expression law. Today iscc's forward model IS roughly the law
they assume (linear dosage), the SNV effect reuses the FITNESS parameter, alleles are SUMMED (no ASE),
and genes are INDEPENDENT (no co-expression). R13 fixes all four.

=== THE ARCHITECTURE (DESIGN_expression.md §3-§4) ===
expr_{g,a} = base[type,g] · exp(Σ_k z_k·loading[k,g]) · dosage(CN_{g,a}; s_g) · cis_snv(class_{g,a}) · niche_g
  - PROGRAM layer (backbone): per-cell activity `z` over K programs; `loading` = gene×program matrix.
  - Genotype couples at THREE separable levels: (1) program activity, (2) gene dosage (CNA —
    CONTIGUOUS), (3) single-gene cis SNV (allele-specific).

=== HOW MUTATIONS DRIVE PROGRAM ACTIVITY — three routes (DESIGN_expression.md §3.1) ===
Do NOT tag programs onto a random subset of drivers and call it done — that would make fitness and
expression independent, which is wrong. Use:
- **Route 1 — phenotype-mediated (the DEFAULT; nearly free).** `Selection` ALREADY computes a per-clone
  phenotype vector from the genotype (`update_division_rate`, `update_dispersal_rate`,
  `update_immune_resistance`, `update_treatment_resistance`). Drive a program from each:
  `division_rate`→proliferation/cell-cycle, `dispersal_rate`→EMT/motility,
  `treatment_resistance`→drug-resistance, `immune_resistance`→immune-evasion. So the chain is
  **mutation → (existing CINner fitness) → phenotype → program → expression**: a mutation that raises
  division rate DOES raise the proliferation program, by construction, with no new gene tagging.
  Knobs: `phenotype_program_map`, `phenotype_program_strength` (0 ⇒ route 1 off).
- **Route 2 — direct program regulators (no fitness change):** a tagged driver subset shifts `z` without
  touching fitness — R12's plasticity cases (differentiation block, de-differentiation, aberrant program).
  Knobs: `prop_program_regulator`, `program_bias_strength`, `n_programs_per_regulator`.
- **Route 3 — niche→program:** F8's hypoxia/CCI, already built; generalise so any program can be tagged
  niche-responsive. Knobs: `niche_program_map`/strength.
- **Mechanics:** phenotype is PER-CLONE ⇒ it sets the program's MEAN across that clone's cells, with
  `activity_noise` giving within-clone spread (cycling is a per-cell state). **READOUT-ONLY — programs
  must NEVER feed back into fitness** (that loop is R8b/R12-v3; keep the F8 discipline). Keep the map
  **sparse/graded**: passengers and most drivers stay transcriptionally silent, else every clone becomes
  its own expression state and clone-vs-state clustering is trivially easy.
- **Consequence to expect and MEASURE:** route 1 deliberately couples fitness↔expression, so clone
  identity leaks into expression via a NON-DOSAGE route. Realistic (proliferation is readable), and it
  adds a genuine confounder for clonealign/inferCNV; for scDEF the proliferation program will correlate
  with clonal structure (tools may conflate "proliferation program" with "clone"). Report it, don't hide it.
  - **programs ⟂ CNAs** (functional/scattered vs positional/contiguous) is what keeps the benchmarks
    non-circular — preserve it.
- **SHARED WITH R12** (`DESIGN_celltrajectory.md`): the `z` + `loading` machinery IS R12's cell-state
  model. Build it ONCE. R12 owns how `z` moves (hierarchy/deformation/niche); R13 owns `z`→counts +
  overlays. Don't duplicate.

=== EXPOSE THESE PARAMETERS (user's explicit ask; full table in DESIGN_expression.md §4.1) ===
- `program_params`: `n_programs`, **`n_genes_per_program`**, **`program_overlap`** (gene sharing across
  programs), `loading_strength` (mean/sd), `loading_sparsity` (heavy-tailed: few strong markers + many
  weak), **`program_genomic_scatter`** (scattered genome-wide by default — the knob that operationalises
  programs ⟂ CNAs and lets you build a CNA-mimicking program as a control), `program_signs`,
  `seeded_programs` (anchor cell-cycle/EMT/the existing F8 hypoxia program).
- `activity_params`: `n_active_programs_per_cell`, `activity_dist`/`activity_mean`/`activity_sd`,
  `activity_noise` (within-clone spread), `celltype_program_bias`.
- genotype→program (the three routes above): `phenotype_program_map` + `phenotype_program_strength`
  (route 1, the default); `prop_program_regulator`, `program_bias_strength`, `n_programs_per_regulator`
  (route 2); `niche_program_map`/strength (route 3).
- `dosage_params`: `dosage_sensitivity_mean`/`_sd` (per-gene s_g), `dosage_saturation`, `allele_specific`.
- `snv_effect_params`: `p_lof`(→NMD)/`p_missense`/`p_splice`/`p_silent`, `nmd_strength`, and
  `snv_expression_effect` kept SEPARATE from the fitness `mut_effect`.
Document them all in `PARAMETERS.md` with defaults + valid ranges (that file's conventions), and surface
`program_truth` ground truth: `loading`, per-cell `z`, gene→program map, `s_g`, per-SNV class.

=== COMPARABILITY ACROSS TUMOURS/PATIENTS (MUST-HAVE — user requirement 2026-07-15) ===
The SAME gene programs must be re-used across simulations whenever the program parameters + the layout
seed match, so cohort-level program analysis is meaningful (same requirement as the shared driver
landscape). The mechanism ALREADY EXISTS — use it, don't invent one:
- `layout_seed` / `self.layout_rng` / `DEFAULT_LAYOUT_SEED` (`count.py:43-55`), decoupled from the per-run
  EVOLUTION seed. `Selection` already takes `rng=self.layout_rng` (count.py:123), so oncogene/TSG/driver
  identities are comparable by construction; baseline `celltype_exps` likewise (count.py:131).
- **Everything that is a property of the GENOME/landscape must come from the LAYOUT stream**, not
  `self.seed`: the program dictionary (gene→program map + `loading` matrix), the program-regulator
  assignment, and the per-gene dosage sensitivities `s_g`. **Event-level draws stay on the run seed**
  (which mutation happens when; a given SNV's functional class draw; the per-cell `z` noise).
- **Use INDEPENDENT sub-streams per landscape component** — e.g. `np.random.SeedSequence(layout_seed).spawn(n)`
  (or documented fixed offsets). Otherwise changing `n_programs` shifts one shared stream and silently
  reshuffles which genes are oncogenes — breaking comparability between configs that differ only in
  program parameters. This matters; get it right.
- **BUG TO FIX while you're here:** F8's program designation uses `prog_rng =
  np.random.default_rng(self.seed + 9973)` (`count.py:142`) — the RUN seed — so `_hypoxia_genes` /
  `_cci_target_genes` DIFFER per patient in a cohort. F8 predates `layout_seed` and was never migrated.
  Move it to the layout stream (the dedicated-stream intent was right, the seed source is wrong). This
  changes which arbitrary genes are hypoxia-responsive ⇒ regenerate the F8/PEtracer figures (results
  should be statistically unchanged) and re-check `tests/test_microenvironment.py`.
- **Tests (mirror the cohort's):** two `GenotypeTumor(same config, DIFFERENT evolution seed)` ⇒ IDENTICAL
  gene→program map + `loading` + `s_g` (and, already covered, identical oncogene/TSG sets); an explicit
  different `layout_seed` ⇒ different programs; changing `n_programs` does NOT change the oncogene/TSG
  layout (the independent-substream property).

=== HARD ENGINE PREREQUISITE ===
Stop **summing the `p`/`m` alleles** in `get_exp` (`components/cell.py:236`) — emit per-allele expression
so dosage and cis-SNV effects are allele-resolved and a **BAF** is available in RNA. This is what
Numbat/CalicoST/cardelino need and is the highest-leverage change in the suite. Shares the
allele-resolved-genome prerequisite with R10.

=== STAGING (DESIGN_expression.md §6) ===
Program backbone (with R12) is foundational; then v1 dosage realism → v2 allele/ASE → v3 SNV classes.
Keep each OFF-BY-DEFAULT / bit-identical when off (the F8 discipline) and compatible with the
genotype-count engine caching + tau-leaping. NB `z` is PER-CELL while genotype effects are PER-CLONE —
respect the (deme, genotype) caching; the per-cell `z` is drawn at materialisation.

=== VALIDATION — the scDEF benchmark (user's explicit ask; DESIGN_expression.md §4.2) ===
Show program-inference tools recover the true programs, and how that degrades with genotype burden.
- **Flagship: scDEF** (hierarchical Bayesian factor model → gene signatures + hierarchical cell states) in
  a dedicated `iscc-scdef` env. **Comparator: cNMF** (`iscc-cnmf`). Optionally Hotspot (bib:
  `detomaso_hotspot_2021`). Add bib entries for scDEF + cNMF (flag "auto-added — verify").
- **Metrics vs truth:** Hungarian-match each true program to an inferred factor; score gene-set
  Jaccard/AUPRC + loading cosine; activity recovery = corr(inferred activity, true `z_k`); count
  **spurious factors** (matching no true program). For scDEF also check the inferred HIERARCHY level vs
  program granularity.
- **THE SWEEP (the point):** low → high **SNV burden** (`mutation_rate`, `n_snvs_per_allele`) and low →
  high **CNA burden** (`cnv_prob`, `amp_prob` → fraction-genome-altered). Plot recovery vs burden.
- **Hypothesis to TEST (a likely headline, but report honestly if weak):** CNAs are CONTIGUOUS, so a high
  CNA burden induces *positional* co-expression (genes co-vary by shared segment, not function) which a
  factor model can absorb as **spurious programs** ⇒ true-program recovery degrades with FGA. The
  discriminating diagnostic is the orthogonality itself: are a factor's genes **positionally clustered**
  (CNA artefact) or **scattered** (real program)? If it reproduces, this is a direct sibling of the
  PEtracer lineage–space confound — *genotype structure confounds expression-program inference* — and goes
  in the same "structure misleads inference" arc as PEtracer + multi-region.
- **Controls:** burden≈0 ⇒ near-ceiling recovery (sanity); `program_genomic_scatter` low ⇒ a deliberately
  CNA-mimicking program — can the tools tell it from a real CNA?

DELIVERABLES: the program layer (+ `z` sampler, shared with R12) and the dosage/ASE/SNV overlays, all
off-by-default; allele-resolved expression + BAF; `program_truth` ground truth; parameters + `PARAMETERS.md`
docs + a `diagnose()` check if sensible; tests (off ⇒ bit-identical; programs recoverable in the easy
regime; alleles split correctly); `validation/validate_programs.py` → figure (recovery vs SNV/CNA burden,
scDEF vs cNMF, spurious-factor positional diagnostic); a manuscript Results subsection in the
"structure misleads inference"/integration arc; flip the BACKLOG R13 item. Full suite green; commit on `dev`.

HONEST NOTES: this is the biggest lever for the paper's non-circularity claim — without it a reviewer says
"your forward model is the tool's own assumption." Don't tune the sweep to manufacture the confound; if
CNA burden doesn't degrade program recovery, that is itself a reportable (and reassuring) result.
```
