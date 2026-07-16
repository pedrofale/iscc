# Handoff prompt — v1 whole-genome duplication (WGD)

Saved 2026-07-16. Copy the block below into a fresh session. **Small, self-contained** — the cheap first
step of R10 (CNA-mechanism parity vs CINner/SISTEM). Full design + the staged plan in `DESIGN_focal_cna.md`;
research framing in `RESEARCH_QUESTIONS.md` R10. Deliberately independent of the running Numbat session but
timed to enrich it (WGD produces the BAF signature Numbat detects). Branch from current `dev`.

---

```
Add whole-genome duplication (WGD) as a mutation event to iscc — the cheap first step toward focal-CNA /
CINner-parity. Full design in DESIGN_focal_cna.md (READ §2, §4, §5 v1 first); this is v1 only, do NOT
attempt focal or whole-chromosome events here.

REPO & ENV
- Repo: /Users/pedroferreira/projects/iscc/repo (branch `dev`).
- Python/pytest: ~/miniconda3/envs/iscc/bin/python.
- Conventions: commit on `dev` WITH `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`; keep the
  FULL suite green (538 now); OFF-BY-DEFAULT / bit-identical when off (the F8 discipline); be honest.

WHY WGD IS CHEAP HERE: iscc's genome (`components/cell.py:54`) is per-segment
(`{'p':[copy,…],'m':[copy,…]}`, each copy a whole-segment SNV bitset), so **copy number is
segment-granular**. WGD = duplicate every copy in `p` and `m` once — it needs NO sub-segment structure
(that's the focal refactor, v3, not this). And the 2026-07-14 viability fix means `max_ploidy` now
actually binds, so a doubled genome is correctly gated instead of running away.

IMPLEMENTATION
1. **The event** — in `CancerCell.mutate` (`cell.py`), add a `wgd` branch alongside `snv`/`cnv`, fired
   with probability `wgd_rate` per division (its own channel; leave `snv_prob`/`cnv_prob` semantics
   intact). WGD: for every segment, for `hap in ('p','m')`, duplicate each existing copy (independent
   `.copy()` of the bitset — no aliasing, mirror the `cnv` amp branch); update `seg_cns` (all doubled),
   `ploidy`, `highest_cn`, `nullisomy_count` via the existing `update_genome_summary_*` seam. SNVs on a
   copy are carried into its duplicate (a WGD preserves and doubles existing mutations — biologically
   correct and matches the bitset copy).
2. **Viability gate:** WGD produces a non-viable daughter if it violates `max_ploidy`/`max_cn` — reuse the
   existing `Cell.mutate` viability check (the reject-at-birth seam the viability fix added). Do NOT add a
   parallel check; a WGD that busts `max_ploidy` is simply rejected like any other non-viable daughter.
3. **Both engines:** the count engine (`models/count.py`, mutate path in `update()` + the tau substep)
   and the cell engine both call `Cell.mutate`, so implementing it there covers both — verify with a test.
4. **Parameter:** `wgd_rate` in `cell_params.cancer` (default 0.0 ⇒ OFF ⇒ bit-identical). Optional
   `wgd_tolerance` (a small fitness/viability buffer, since WGD buffers deleterious loss) — implement only
   if trivial; otherwise leave as an open decision noted in DESIGN_focal_cna.md §10.
5. **NOT a landscape draw:** `wgd_rate` is an event rate, so no `layout_seed` comparability concern (unlike
   gene roles / programs). Nothing to wire into the layout stream.

GROUND TRUTH: per-genotype `is_wgd` (has this lineage undergone WGD) + the ploidy already in
`genome_summary`. Surface `is_wgd` in `cell_data` (like `cell_microenv`) so downstream benchmarks can use it.

PARAMETERS DOC: add `wgd_rate` to `PARAMETERS.md` (Copy-number section) with default 0.0 + valid range +
"drives genome doubling; interacts with max_ploidy". Consider a `diagnose()` note if a run's ploidy is
implausible.

VALIDATION (`validation/validate_wgd.py` → `manuscript/figures/validation_wgd.png`):
- **WGD frequency** — over replicate tumours, the fraction that acquire + retain WGD at a plausible
  `wgd_rate` should land in the real ~30–50% range (real WGD prevalence, PCAWG). Sweep `wgd_rate`.
- **Ploidy distribution** — the doubling+subsequent-loss signature: WGD tumours sit at higher, non-integer
  mean ploidy; compare the shape to PCAWG ploidy. (No need to fit exactly — show the mechanism produces
  the right qualitative distribution.)
- Print headline numbers.

TESTS (`tests/test_wgd.py`):
- `wgd_rate=0` ⇒ growth byte-identical to before (bit-identical-when-off).
- a forced WGD doubles every segment's copy count and carries SNVs into the duplicates.
- WGD that would exceed `max_ploidy` is rejected at birth (non-viable), not applied.
- both engines agree that WGD doubles copies (mirror `test_engines_agree_*`).

DELIVERABLES: the `wgd` event + `wgd_rate` param (off-by-default); `is_wgd` ground truth in `cell_data`;
`PARAMETERS.md` entry; `validation/validate_wgd.py` + figure; `tests/test_wgd.py`; a short manuscript
sentence (CNA-mechanism breadth — WGD frequency + ploidy vs PCAWG, closing the arm-only gap) + Table 1
update (iscc now has WGD); flip an R10-v1 BACKLOG item. Full suite green; commit on `dev`.

HONEST NOTES: keep it to WGD — resist scope-creeping into focal/whole-chromosome (that's the interval
refactor, a separate milestone). If WGD frequency can't reach ~30–50% at any sane `wgd_rate` without other
tumours going non-viable, that's a real finding about the interaction with the viability limits — report
it (it may motivate `wgd_tolerance`). Coordinate lightly with the Numbat session: once WGD lands, Numbat's
benchmark can add a WGD-detection axis (BAF), but that's the Numbat session's job, not this one.
```
