# DESIGN — genotype→expression realism (CNA & SNV coupling) [design-first; NOT built]

Status: **whiteboard / design-first** (started 2026-07-09). Companion to `RESEARCH_QUESTIONS.md` R13,
`DESIGN_celltrajectory.md` (R12, the co-expression/program layer), `DESIGN_features.md` §H (F8, niche
layer), `DESIGN_epistasis.md` (the selection-side sibling). Motivated by the data-integration
benchmark thesis: the credibility of the DNA↔RNA integration benchmarks rests on the CNA/SNV→expression
coupling being **realistic and non-circular**, not the simple dosage law the tools themselves assume.
Nothing here is built yet.

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

## 3. The three coupling axes to make realistic

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

## 5. What each benchmark needs (traceability)

| Tool | Needs from this doc |
|---|---|
| clonealign, inferCNV/CopyKAT | per-gene **dosage sensitivity** + saturation (so the law ≠ the assumption) |
| **Numbat, CalicoST, STARCH** | **allele-specific CN + ASE / BAF in RNA** (axis A + emit alleles) |
| **cardelino, PhylEx** | **SNV→expression** functional classes + ASE (axis B); SNVs callable in RNA (F7b, have) |
| cell2location/RCTD, scVI/Harmony | **co-expression / program covariance** (axis C = R12) |

## 6. Staged plan (when promoted)

1. **v1 — dosage realism:** per-gene `s_g` sensitivity + saturation (cheap; makes clonealign/inferCNV
   fair). Off-by-default so current outputs are unchanged; ground truth `s_g` surfaced.
2. **v2 — allele-resolved expression / ASE:** stop summing `p`/`m` in `get_exp`; emit per-allele
   expression → BAF in RNA (unlocks Numbat/CalicoST). Shares the allele-resolved-genome prerequisite
   with R10 (focal/allele CNA).
3. **v3 — SNV functional classes:** decouple expression effect from fitness `mut_effect`; add
   LoF/NMD/missense/splice/silent + the TSG two-hit (unlocks cardelino/PhylEx realism).
4. **v4 — program covariance:** the R12 program-loading layer for realistic multivariate structure.

## 7. Open decisions
- Distribution/prior for per-gene dosage sensitivity `s_g` (calibrate from real CN–expression pairs?).
- How much allele resolution to emit (BAF only, or full allelic counts in reads?).
- Where SNV functional class comes from: abstract role-based (oncogene/TSG → class priors) vs real-genome
  annotation.
- Keep everything **off-by-default / bit-identical when off**, and compatible with the genotype-count
  engine caching + tau-leaping (the F8 discipline).

## 8. Relation to other threads
F8 (niche programs — done) · R12/`DESIGN_celltrajectory.md` (program covariance = axis C) · R10 (focal /
allele-resolved CNA — shared prerequisite for ASE) · `DESIGN_epistasis.md` (selection-side sibling) ·
R8b (microenvironment→fitness). Honest note: **this is the single biggest lever for making the DNA↔RNA
integration benchmarks credible** — without it a reviewer can say "your forward model is the tool's
assumption."
