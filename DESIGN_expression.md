# DESIGN — genotype→expression realism (CNA & SNV coupling) [design-first; NOT built]

Status: **PLANNED FOR PAPER 1 — build now** (decision 2026-07-14; R13 is no longer deferred). Design
written, not yet built. Companion to `RESEARCH_QUESTIONS.md` R13, `DESIGN_celltrajectory.md` (R12),
`DESIGN_features.md` §H (F8, niche layer), `DESIGN_epistasis.md` (R14, the selection-side sibling — also
paper 1 now). Motivated by the data-integration benchmark thesis: the DNA↔RNA integration benchmarks are
only credible if iscc's CNA/SNV→expression coupling is **realistic and non-circular**, not the simple
dosage law the tools themselves assume.

**KEY DESIGN DECISION (2026-07-14): expression is modelled as GENE PROGRAMS, and R12 and R13 share ONE
implementation.** The per-cell program-activity vector `z` is the same object as R12's "cell state":
R12 owns how `z` *moves* (differentiation hierarchy + genotype landscape deformation + niche), R13 owns
how `z` *becomes counts* plus the gene-level dosage/SNV overlays. Build the program layer once; both
features use it. See §3.

## 1. Why this matters (the non-circularity argument, sharpened)

The integration benchmarks (clonealign, inferCNV/CopyKAT, Numbat, cardelino, PhylEx) all *invert* a
genotype→expression relationship. A benchmark is only fair if iscc's forward model is **not** the
inverse model the tool assumes. Today it largely is, so the benchmarks risk being too easy / partly
circular, and the allele/SNV-based tools cannot be tested at all.

## 2. Current model (grounded in code)

`CancerCell.get_exp` (`components/cell.py:236`) + materialization in `models/count.py` (`make_cell_data`,
per-(deme,gid) modifier from F8):

- **Baseline:** `celltype_exps[type]` = independent `beta(0.1, 1.0)` per gene; oncogenes forced low
  (0.01), TSGs high (0.8). **Genes are independent** — no co-expression structure.
- **CNA → expression:** `exp = baseline + Σ_copies (baseline · mut_effect^bits)` — i.e. **additive,
  ~linear dosage** (each extra copy adds ≈ baseline). This is essentially the log-linear CN→expression
  law inferCNV/clonealign/Numbat assume ⇒ **circularity risk**.
- **SNV → expression:** a single factor `seg_mut_effects[seg] ** bits` — and `seg_mut_effects` is the
  **same parameter that drives fitness**. So (a) expression and selection are entangled through one
  knob, and (b) there is **no functional class** (LoF vs missense vs splice vs silent), no NMD, no
  two-hit interaction with CNA loss.
- **Alleles:** the genome tracks `p`/`m` haplotype bitsets, so **copy number is allele-resolved**, but
  `get_exp` **sums both alleles** ⇒ **no allele-specific expression / B-allele frequency** is emitted —
  exactly the signal Numbat and CalicoST rely on.
- **Niche (F8):** a per-deme×gene multiplier adds spatial program structure (done).

## 3. Expression = a program backbone + two gene-level genotype overlays

**Programs are the backbone.** A cell's expression state is a vector of activities `z = (z_1,…,z_K)`
over a small set of gene PROGRAMS — the recurrent meta-programs (cell cycle, hypoxia, EMT, stress,
stemness…; Gavish & Tirosh, Nature 2023). Expression is `base · exp(Σ_k z_k·loading[k,g])`; the
`loading` matrix says which genes each program moves. This is the co-expression / covariance structure
that deconvolution and integration methods actually depend on, and it **replaces the current
unrealistic independent-per-gene draws**. This `z` **is R12's cell state** (shared implementation).

**The genotype couples to expression at THREE separable levels** — keeping them separate is the whole game:

1. **Program activity** (program-level, per-CLONE): driver/regulator mutations shift *which* programs a
   cell expresses (an EMT-driver raises the EMT program; de-differentiation raises stemness). This is
   R12's landscape deformation, phrased here as "the genotype biases the `z` distribution."
2. **Gene dosage** (gene-level, **contiguous**): a CNA raises/lowers the genes *on that segment* — axis A.
3. **Single-gene cis** (gene-level): a LoF/NMD/splice SNV changes *that gene's* allele-specific
   expression — axis B.

**The load-bearing insight — programs ⟂ CNAs.** Programs are functional gene sets **scattered** across
the genome; CNAs are **contiguous** genomic segments. That orthogonality is exactly what makes the
integration benchmarks non-circular and non-trivial: inferCNV/Numbat must recover the *contiguous* dosage
signal against the program-structured background (running-window smoothing); clonealign must map clones
by dosage *despite* program variation (programs = the confounder); cell2location/RCTD must separate cell
types (= program combinations) from clones (= CNA). Variance splits naturally: **between-clone**
structure = genotype (dosage + program-bias, shared within a clone); **within-clone** heterogeneity =
the per-cell stochastic `z` — i.e. the shared-vs-private structure the cohort/integration benchmarks score.

The three pieces in detail (C = the backbone; A, B = the gene-level overlays):

### A. CNA → expression: dosage realism
- **Per-gene dosage sensitivity** `s_g ∈ [0,1]`: 0 = fully buffered/compensated, 1 = full linear
  dosage. Draw `s_g` from a distribution (most genes partially buffered; known dosage-sensitive genes /
  oncogenes near 1). Response `exp_g ∝ baseline · (1 + s_g·(CN_g/ploidy − 1))`, optionally **saturating**
  at high CN and floored at low CN (non-linear).
- **Allele-specific expression (ASE):** keep the `p`/`m` split through to expression so an amplified or
  lost homolog produces a **BAF in RNA**, not just total dosage. (Requires emitting the two allele
  layers instead of summing them.)
- **Payoff:** makes clonealign/inferCNV a **fair** test (their assumed law is only approximately true,
  and per-gene sensitivity is a real confounder), and **unlocks Numbat / CalicoST / STARCH** (allele).

### B. SNV → expression
- **Decouple the expression effect from the fitness effect** (today they share `seg_mut_effects`).
- **Functional classes** drawn per SNV: LoF/nonsense → **NMD** (expression loss on that allele; with a
  CNA loss of the other allele = biallelic **two-hit** inactivation of a TSG); missense in an oncogene →
  activity change, expression ≈ unchanged; splice → isoform/expression shift; **most passengers silent**.
- **ASE from cis SNVs:** an NMD or cis-regulatory SNV skews the allele ratio → mono-allelic/skewed
  expression, which is exactly what **cardelino** detects (SNV in expressed RNA) and Numbat's allele
  layer uses.
- **Payoff:** enables realistic **cardelino / PhylEx** benchmarks, ASE, and the TSG two-hit interaction.

### C. Multivariate structure (co-expression / programs)
- Replace independent-per-gene draws with a **low-dimensional program model** (this is
  `DESIGN_celltrajectory.md` R12): co-expression modules + cell-state programs (cell cycle, EMT,
  hypoxia, stress) give realistic gene–gene **covariance**; F8 already adds the spatial programs.
- **Payoff:** deconvolution (cell2location/RCTD) and integration (scVI/Harmony) depend on covariance and
  marginal realism; independent genes make those benchmarks artificially easy and make "shared vs
  private cell states" ill-defined.

## 4. Proposed composition (extends the multiplicative model)

Per allele `a ∈ {p, m}`, then summed (or emitted separately for ASE):

```
exp_{g,a} = base_{type,g} · dosage(CN_{g,a}; s_g) · snv_effect(class_{g,a}) · exp(Σ_k z_k·loading_{k,g}) · niche_g
exp_g     = exp_{g,p} + exp_{g,m}        # and BAF_g = exp_{g,p} / exp_g  (the ASE readout)
```

Composes cleanly with the existing CNA dosage, F8 niche modifier (done), and the R12 program term.

### 4.1 Parameter surface (exposed knobs)

All of these are user-facing config (`PARAMETERS.md` + `tumor.diagnose()` once built). Ground truth for
every one is surfaced (`program_truth`: `loading` matrix, per-cell `z`, gene→program map, `s_g`, SNV classes).

**COMPARABILITY (user requirement 2026-07-15).** The same program parameters + the same `layout_seed`
must yield the **same programs across simulations/patients**, exactly as the shared driver landscape
already does. Rule: anything that is a property of the **genome/landscape** — gene→program map,
`loading`, program-regulator assignment, per-gene `s_g`, and the R14 epistasis network — is drawn from
the **layout stream** (`layout_seed`/`layout_rng`, `count.py:43-55`; `Selection` already uses it, so
oncogene/TSG identities are comparable today). **Event-level** draws (which mutation occurs, a given
SNV's class, per-cell `z` noise) stay on the per-run evolution seed. Use **independent sub-streams per
component** (`SeedSequence(layout_seed).spawn(n)`) so changing `n_programs` does not reshuffle the
oncogene/TSG layout. NB **F8 currently violates this** (`prog_rng = default_rng(self.seed + 9973)`,
`count.py:142` — the run seed): migrate it to the layout stream (tracked in `BACKLOG.md`).

**Program dictionary — `program_params`** (the `loading` matrix, K × G):
| Knob | What it controls |
|---|---|
| `n_programs` (K) | how many gene programs exist |
| `n_genes_per_program` | genes each program loads on (int or a distribution) |
| `program_overlap` | how much programs SHARE genes (expected fraction of a program's genes also in another; 0 = disjoint modules, high = entangled) |
| `loading_strength` (mean/sd) | effect size — the log-fold a program imposes on its genes |
| `loading_sparsity` | within-program loading shape (uniform vs heavy-tailed: a few strong markers + many weak) — real programs are heavy-tailed |
| **`program_genomic_scatter`** | whether a program's genes are drawn **scattered genome-wide (default, realistic)** or positionally clustered. **This knob operationalises programs ⟂ CNAs** — set it low to build a program that *mimics* a CNA and test whether tools can tell them apart |
| `program_signs` | up-only vs bidirectional (up- and down-genes) |
| `seeded_programs` | optionally anchor some programs to known signatures (cell cycle, EMT, and the existing F8 hypoxia program) so they're interpretable |

**Per-cell activity — `activity_params`** (the `z` sampler; shared with R12):
| Knob | What it controls |
|---|---|
| `n_active_programs_per_cell` | activity sparsity (how many programs are on in a cell) |
| `activity_dist` + `activity_mean`/`activity_sd` | the distribution of `z` |
| `activity_noise` | **within-clone** spread of `z` → within-clone heterogeneity (vs between-clone genotype structure) |
| `celltype_program_bias` | baseline program activity per cell type (normal vs cancer) |

**Genotype→program coupling (level 1)** — the R12 landscape deformation:
| Knob | What it controls |
|---|---|
| `prop_program_regulator` | fraction of driver genes that are program regulators |
| `program_bias_strength` | how strongly a regulator mutation shifts `z` for its target program |
| `n_programs_per_regulator` | how many programs a regulator touches |

**Dosage (axis A) — `dosage_params`:** `dosage_sensitivity_mean`/`_sd` (the per-gene `s_g` distribution),
`dosage_saturation` (CN at which the response flattens), `allele_specific` (bool → emit per-allele
expression + BAF).

**SNV effects (axis B) — `snv_effect_params`:** class probabilities `p_lof` (→NMD) / `p_missense` /
`p_splice` / `p_silent`; `nmd_strength` (expression retained on the NMD allele); **`snv_expression_effect`
kept SEPARATE from the fitness `mut_effect`** (they're one knob today).

**Sweep knobs (already exist)** — used by the validation below: SNV burden (`mutation_rate`,
`n_snvs_per_allele`), CNA burden (`cnv_prob`, `amp_prob` → fraction-genome-altered).

## 4.2 Validation — program recovery vs genotype burden (the scDEF benchmark)

**The question:** can a gene-program / factor-inference tool recover the true programs from iscc's scRNA
counts, and **how does that degrade as SNV/CNA burden rises?** iscc is uniquely able to ask this because
it *knows* the true `loading` matrix and per-cell `z`.

- **Flagship tool: scDEF** (hierarchical Bayesian factor model → gene signatures + hierarchical cell
  states). **Comparator: cNMF** (the field-standard consensus-NMF GEP method); optionally Hotspot
  (`detomaso_hotspot_2021`, already in the bib) for modules. Each in its own env (`iscc-scdef`, `iscc-cnmf`).
- **Metrics vs ground truth:** per-true-program best-matching inferred factor (Hungarian matching) scored
  by gene-set **Jaccard/AUPRC** and loading **cosine similarity**; **activity recovery** = correlation of
  inferred factor activity vs true `z_k`; **#spurious factors** (matching no true program).
- **The sweep (the point):** low → high **SNV burden** and low → high **CNA burden**, measuring recovery.
- **The hypothesis worth testing (and a likely headline):** because CNAs are **contiguous**, a high CNA
  burden induces *positional* co-expression — genes co-vary because they share a copy-number segment, not
  a function. A factor model can absorb this as **spurious "programs"**, so true-program recovery should
  degrade with fraction-genome-altered. The **diagnostic that distinguishes artefact from biology** is
  exactly the orthogonality: are a factor's genes **positionally clustered** (CNA artefact) or **scattered**
  (real program)? If this reproduces, it is a direct sibling of the PEtracer lineage–space confound —
  *genotype structure confounds expression-program inference* — and belongs in the same "structure misleads
  inference" arc. Report honestly if the effect is weak.
- **Controls:** `program_genomic_scatter` low ⇒ a deliberately CNA-mimicking program (can tools separate
  it?); burden ≈ 0 ⇒ recovery should be near-ceiling (sanity).

## 5. What each benchmark needs (traceability)

| Tool | Needs from this doc |
|---|---|
| clonealign, inferCNV/CopyKAT | per-gene **dosage sensitivity** + saturation (so the law ≠ the assumption) |
| **Numbat, CalicoST, STARCH** | **allele-specific CN + ASE / BAF in RNA** (axis A + emit alleles) |
| **cardelino, PhylEx** | **SNV→expression** functional classes + ASE (axis B); SNVs callable in RNA (F7b, have) |
| cell2location/RCTD, scVI/Harmony | **co-expression / program covariance** (axis C = R12) |
| **scDEF** (flagship), cNMF, Hotspot | the **program layer itself** — true `loading` + per-cell `z` as ground truth, scored across the SNV/CNA-burden sweep (§4.2) |

## 6. Staged plan (when promoted)

1. **v1 — dosage realism:** per-gene `s_g` sensitivity + saturation (cheap; makes clonealign/inferCNV
   fair). Off-by-default so current outputs are unchanged; ground truth `s_g` surfaced.
2. **v2 — allele-resolved expression / ASE:** stop summing `p`/`m` in `get_exp`; emit per-allele
   expression → BAF in RNA (unlocks Numbat/CalicoST). Shares the allele-resolved-genome prerequisite
   with R10 (focal/allele CNA).
3. **v3 — SNV functional classes:** decouple expression effect from fitness `mut_effect`; add
   LoF/NMD/missense/splice/silent + the TSG two-hit (unlocks cardelino/PhylEx realism).
4. **program backbone (shared with R12 — foundational, not strictly last):** the program-loading layer
   (`z` + a `loading` dictionary) IS R12's cell-state model — build it jointly with R12. It provides the
   co-expression structure *and* the program-activity coupling point (level 1 above); the dosage (v1),
   ASE (v2) and SNV (v3) overlays sit on top of it. Order v1→v3 by cheapest-benchmark-win; the program
   backbone can proceed in parallel as the R12 build.

## 7. Open decisions
- Distribution/prior for per-gene dosage sensitivity `s_g` (calibrate from real CN–expression pairs?).
- How much allele resolution to emit (BAF only, or full allelic counts in reads?).
- Where SNV functional class comes from: abstract role-based (oncogene/TSG → class priors) vs real-genome
  annotation.
- Keep everything **off-by-default / bit-identical when off**, and compatible with the genotype-count
  engine caching + tau-leaping (the F8 discipline).

## 8. Relation to other threads
**R12 and R13 SHARE the program/`z` implementation** — one program dictionary + one per-cell `z` sampler.
R12 = the *dynamics* of `z` (hierarchy + genotype landscape deformation + niche); R13 = `z`→counts + the
dosage/SNV overlays. F8 (niche → program activity — done) · R10 (focal / allele-resolved CNA — shares the
"stop summing p/m alleles" prerequisite with R13's ASE) · `DESIGN_epistasis.md` (R14, selection-side
sibling, also paper 1) · R8b (microenvironment→fitness).

**The one hard engine prerequisite:** the genome must stop **summing the `p`/`m` alleles** at the
expression (and read) level, so dosage and cis-SNV effects can be per-allele and ASE/BAF is emitted —
this is what Numbat/CalicoST/cardelino need and is the single highest-leverage change across the suite.
Honest note: **this doc is the biggest lever for making the DNA↔RNA integration benchmarks credible** —
without it a reviewer can say "your forward model is the tool's own assumption."
