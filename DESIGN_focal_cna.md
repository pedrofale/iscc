# DESIGN — focal CNAs, WGD & whole-chromosome events (CNA-mechanism parity) [design-first]

Status: **design-first** (started 2026-07-16). Research framing: `RESEARCH_QUESTIONS.md` R10. Closes the
one honest capability gap vs CINner/SISTEM — iscc is arm-resolution, they have the full CNA mechanism
set. Companion handoff: `handoffs/wgd.md` (the cheap v1). Nothing here is built.

## 1. The gap
iscc's CNA model amplifies/deletes a **whole segment** (arm resolution in real-genome mode). CINner and
SISTEM both do *focal (sub-arm) amp/del, whole-chromosome missegregation, and whole-genome duplication
(WGD)*. So iscc can't represent a tight focal amplicon (*MYC*, *EGFR*), a focal TSG deletion, genome
doubling, or a whole-chromosome gain/loss as distinct events. The mechanisms are known (both competitors
implement them); the genuinely open part is doing focal cheaply.

## 2. Current representation (why this splits into easy + hard)
`self.genome` (`components/cell.py:54`) is per-segment: a segment is `{'p':[copy,…],'m':[copy,…]}`, each
`copy` a whole-segment SNV bitset. So **copy number is segment-granular** (a copy is always a whole
segment) while SNVs are position-resolved within a segment. Alleles are already resolved (`p`/`m`), and
`max_ploidy`/`max_cn`/`max_nullisomy` now actually bind (the 2026-07-14 viability fix). That representation
makes the three events very different in difficulty:

| Event | Granularity | Difficulty in the current rep |
|---|---|---|
| **WGD** | genome-wide | **cheap** — duplicate every copy in `p` and `m` once |
| **Whole-chromosome gain/loss** | segment-set | **moderate** — needs a chromosome→segment grouping, then act on all its segments on one homolog |
| **Focal (sub-arm) amp/del** | *sub*-segment | **the refactor** — the per-segment rep cannot express sub-segment CN |

## 3. The representation decision (the crux, for focal only)
Two ways to get sub-segment CN:
- **Fine-binning** (smaller segments): simple, but memory/per-event work scale **O(#bins)**, multiplied
  by the genotype-count caching. Expensive at gene resolution (~10⁴ bins).
- **Allele-specific interval / run-length CN (RECOMMENDED):** each homolog's CN is a piecewise-constant
  function — a sorted list of `(start, end, copy_state)` intervals, `copy_state` carrying the SNV bitset.
  A CNA of ANY span (focal / arm / chromosome) is an interval op: split at breakpoints, change CN over
  `[start,end]`. Memory is **O(#breakpoints)** (tens–hundreds, not thousands) — more general AND more
  scalable than fine-binning, and essentially how CINner represents allele-specific CN. It **subsumes
  segments/arms as special cases**, and shares the "per-homolog CN" plumbing with R13's ASE work
  (`DESIGN_expression.md`) — worth coordinating so the genome isn't refactored twice.

## 4. What you do NOT need to build
- **Fitness:** reuse it. A focal oncogene amplification already boosts `division_rate` through the
  existing dosage/driver pathway; the arm model's `s_arm` scores arm/chromosome events. Only the event
  *geometry* is new. WGD may want a small tolerance/fitness effect (it buffers deleterious loss) — decide.
- **Viability:** `max_ploidy`/`max_cn`/`max_nullisomy` already bind post-fix, so WGD and focal amps won't
  run away. WGD interacts directly with `max_ploidy` (a doubled genome near the limit is fragile).
- Each event type is a mechanism with its **own rate**, **ABC-estimable** exactly as CINner fits its
  mechanism probabilities (extends `mut_prob`/`cnv_prob`). Must stay genotype-count- and tau-leap-compatible.

## 5. Staged plan
- **v1 — WGD (cheap; ship alongside the Numbat benchmark).** A punctuated event doubling all copies on
  both homologs, gated by `max_ploidy`. WGD occurs in ~30–50% of real tumours and produces the
  doubling+loss allelic-imbalance signature **Numbat detects via BAF** — so it directly enriches the
  in-flight Numbat benchmark. Handoff: `handoffs/wgd.md`.
- **v2 — whole-chromosome missegregation.** Add a chromosome→segment grouping (real-genome/arm mode
  already groups segments into arms; abstract mode = consecutive-segment groups), then gain/lose the
  whole set on one homolog. Feeds `s_arm` directly.
- **v3 — focal amp/del (the interval refactor of §3).** The real work. Do it on top of / jointly with
  R13's allele-splitting.

## 6. Parameter surface
- `wgd_rate` (per-division probability), optional `wgd_tolerance` (fitness/viability buffer).
- `whole_chrom_rate`; chromosome structure (`n_chromosomes` or a segment→chromosome map).
- `focal_rate`, `focal_size_dist` (log-normal small span), `focal_amp_prob` (amp vs del).
All ABC-estimable (new mechanism-rate targets for M1). Surface ground truth: per-genotype `is_wgd`,
ploidy, and the true per-event (type, span, homolog) list.

## 7. Validation (mirrors CINner/SISTEM + the Davoli/Charm precedent)
- **WGD frequency** (~30–50% of tumours) + the doubling+loss **ploidy distribution** vs PCAWG.
- Recurrent **focal amplification of oncogenes / focal deletion of TSGs** — the focal analogue of the
  arm-level Davoli/Charm result already validated; compare to real **GISTIC peaks** (recurrent focal peaks).
- **Mechanism-rate recovery via ABC** — extend M1 to fit `wgd_rate` / `whole_chrom_rate` / `focal_rate`.

## 8. Scalability
Interval rep keeps per-genotype memory O(#breakpoints). The real cost is genotype *diversity* (more
distinct CN profiles → more cached genotypes), the same infinite-sites concern as SNVs (`DESIGN_scalability`
§3); focal events are rarer than SNVs, and copy-on-write genotype sharing already exists. Keep interval
ops O(log n) via sorted breakpoints.

## 9. Synergies & scope
- **R13 allele-splitting** (in) and **Numbat/CalicoST** (in flight) both hinge on per-homolog CN — WGD +
  focal make the Numbat benchmark a much stronger test than arm-only. **v1 WGD is a small paper-1 item**
  that closes a visible Table-1 gap and enriches Numbat now. **v3 focal** is a scoped follow-up milestone,
  gated on whether the CNA-caller/Numbat results need focal (GISTIC-peak) resolution to be compelling.

## 10. Open decisions
- Interval representation vs fine-binning for v3 (recommend interval) — and whether to fold it into R13's
  allele refactor.
- WGD fitness/tolerance effect: none, or a small buffer?
- Chromosome structure in abstract mode: how many, and does it need to be layout-seed-comparable across
  a cohort (the chromosome grouping is a landscape property → layout stream, like gene roles).
