# DESIGN — focal CNAs, WGD & whole-chromosome events (CNA-mechanism parity) [design-first]

Status: **v1 (WGD) BUILT** (2026-07-17); v2/v3 design-first. Research framing: `RESEARCH_QUESTIONS.md`
R10. Closes the one honest capability gap vs CINner/SISTEM — iscc is arm-resolution, they have the full
CNA mechanism set. Companion handoff: `handoffs/wgd.md` (the cheap v1, now shipped). v1 is the WGD event
+ `wgd_rate` (off by default, byte-identical when off); v2 (whole-chromosome) and v3 (focal) unbuilt.

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
- **v1 — WGD (cheap; ship alongside the Numbat benchmark). BUILT (2026-07-17).** A punctuated event
  doubling all copies on both homologs, gated by `max_ploidy`. WGD occurs in ~30–50% of real tumours and
  produces the doubling+loss allelic-imbalance signature **Numbat detects via BAF** — so it directly
  enriches the in-flight Numbat benchmark. Handoff: `handoffs/wgd.md`.
  - *As built:* `wgd_rate` (a separate per-division channel in `CancerCell.mutate`, off by default so
    growth is byte-identical when off — verified equal to the pre-WGD baseline). The WGD branch
    duplicates every copy on both homologs of every segment via the existing `update_genome_summary_cnv`
    seam, so ploidy/`highest_cn`/`nullisomy_count` update for free and the reject-at-birth viability gate
    (`max_ploidy`/`max_cn`) drops non-viable doublings unchanged. Ground truth: per-genotype `is_wgd`
    (monotone, inherited through `divide()`), surfaced as `cell_data["cell_wgd"]["is_wgd"]` in BOTH
    engines when WGD is on. Tests: `tests/test_wgd.py`. Validation: `validation/validate_wgd.py` (WGD
    frequency sweep + doubling+loss ploidy distribution).
  - *WGD allele-state axis (BUILT 2026-07-18, `validate_numbat.py --wgd-rate`).* The natural
    follow-up, and it corrected the naive expectation. Measurement (not theory) showed a **pure**
    doubling (the diploid 1+1 -> 2+2) is *unidentifiable* from relative expression + BAF — it is
    allelically balanced (BAF 0.5) and cancels under per-cell expression normalisation, so BOTH inferCNV
    and Numbat infer
    ~2n for WGD cells; iscc faithfully reproduces that identifiability limit. What WGD *does* create as
    the doubled genome erodes is high-copy **allelic imbalance** — even-total states (4+0, 3+1) whose
    total CN matches a balanced 2+2, so only the allele layer can see them. The axis scores exactly
    that: allelic-imbalance-STATE recovery *controlling for total CN* (the earlier benchmark scored
    only total CN and collapsed Numbat's `loh` state into `neu`=2, discarding this signal). Turning WGD
    on raises the allele-only-detectable segment fraction ~4-6x (≈1% → ≈4-6%), and Numbat recovers the
    imbalance at AUC ≈0.7-0.8 where inferCNV sits at chance. `numbat_runner.R` now emits per-(cell,seg)
    `cnv_state` / P(imbalance) / P(loh) (not just total CN) and degrades to neutral outputs rather than
    crashing when Numbat finds nothing (the mixed-ploidy pseudobulks are noisy). Ground truth:
    `integration_common.segment_allele_cn` (per-homolog CN from the genotype genome).
- **v2 — whole-chromosome missegregation.** Add a chromosome→segment grouping (real-genome/arm mode
  already groups segments into arms; abstract mode = consecutive-segment groups), then gain/lose the
  whole set on one homolog. Feeds `s_arm` directly.
- **v3 — focal amp/del (the interval refactor of §3).** The real work. Do it on top of / jointly with
  R13's allele-splitting.

## 6. Parameter surface
- `wgd_rate` (per-division probability). **No `wgd_tolerance`** — WGD robustness is emergent (see §10).
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
- WGD fitness/tolerance effect: **DECIDED — never add `wgd_tolerance`** (user 2026-07-17; closed, not
  open). WGD's robustness (masking of deleterious loss) is ALREADY EMERGENT: a 4n cell has a copy-number
  buffer (several deletions per segment before nullisomy) and the count-based CINner fitness dilutes the
  effect of any single loss, so post-WGD cells tolerate loss with no fitted term. An explicit
  `wgd_tolerance` would be a bolt-on, contrary to iscc's emerge-not-impose / non-circularity principle.
  Vindicated in v1: neutral WGD reaches the real 30–50% prevalence band (45% at `wgd_rate=0.05`) with no
  tolerance. The per-cell non-integer-ploidy *erosion* signature of real WGD tumours is therefore not a
  missing mechanism but a matter of DYNAMICS — it should emerge as subsequent copy-number loss
  accumulates over longer runs / larger tumours, buffered by the emergent robustness; no new knob. See the
  headline numbers in `validation/validate_wgd.py`.
- Chromosome structure in abstract mode: how many, and does it need to be layout-seed-comparable across
  a cohort (the chromosome grouping is a landscape property → layout stream, like gene roles).
