# Handoff prompt — implement the crowding fix (Option A) + re-baseline all affected results

Saved 2026-07-14. Copy the block below into a fresh session. This is a real ENGINE change that alters
**every spatially structured output**, so it must be paired with a full re-validation and a claims
sweep. Full diagnosis + fix + prototype validation are already written up in `DESIGN_crowding.md` — read
it first. Companion: `BACKLOG.md` ("Engine bug — carrying capacity not enforced", NOW), memory
`iscc-crowding-bug`, `PARAMETERS.md` (the caveat box to remove). Branch from current `dev`.

---

```
Implement the carrying-capacity / crowding fix for iscc (Option A) and update every claim, result, test
and figure influenced by it. The bug, root cause, fix, and a validated prototype are documented in
DESIGN_crowding.md — READ IT FIRST; do not re-derive.

REPO & ENV
- Repo: /Users/pedroferreira/projects/iscc/repo (branch `dev`).
- Python/pytest: ~/miniconda3/envs/iscc/bin/python.
- Conventions: commit on `dev` WITH `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`; be honest;
  match surrounding style. This change is NOT backward-compatible — expect to re-baseline fixtures.

THE BUG (verified, from DESIGN_crowding.md): crowding death is a FIXED absolute rate
`min(death_rate·K, max_death_rate)` (≤0.5), but selection raises each clone's `division_rate` up to
`max_birth_rate` (0.8). Once evolved div > the death cap, net growth is positive regardless of occupancy
→ demes over-fill (measured ~1,200–4,200 cells/deme at K=10) → the tumour is a dense PILE, not a spatial
spread. `carrying_capacity` is currently a threshold, not a capacity.

THE FIX — Option A (validated in a prototype: mean 8.9 cells/deme at K=10, 15,346 demes at 137k cells):
density-dependent death RELATIVE to each clone's OWN (evolved) division rate, with the clamp raised.
  death = base_death + max(0, division_rate(clone) − base_death) · (total / carrying_capacity)
  death = min(death, maximum_death_rate)     # with maximum_death_rate ≥ max_birth_rate
  (then add the existing immune + treatment terms unchanged)
This makes death == division at total==K and > division above K (a restoring force), for any evolved
division rate. Consider a small margin/steeper slope for a firmer cap (the plain form is marginally
stable exactly at K). Tune and document.

IMPLEMENTATION
1. `src/iscc/tumor/models/count.py` `_death_rate` (~line 304): replace the step-function crowd with the
   logistic form above.
2. **Mirror engine:** the count engine's `_death_rate` docstring says it mirrors the cell-level
   `Deme.get_cancer_death_rate`. Find and fix that too (cell-level/glandular engine) so both engines
   stay consistent; add/keep a test that they agree.
3. **Default config:** raise `maximum_death_rate` to ≥ `max_birth_rate` (e.g. 1.0) in the shipped configs
   (`notebooks/example_config.yaml`, `src/iscc/tumor/tumorconfigs/{glandular,mixed}.yaml`) and any
   validation/benchmark configs, else the clamp still defeats the fix.
4. **PRESERVE the well-mixed regime (critical).** Today `carrying_capacity=1` is a hack meaning "no
   crowding ceiling → unbounded growth" (used by the single-deme SISTEM benchmark). After the fix,
   K=1 would cap a deme at 1 cell — WRONG. Provide an explicit way to disable crowding (e.g.
   `carrying_capacity=None`/0 → well-mixed, or a `crowding` flag), and make
   `benchmark_scalability.py --tau-grid 1` still grow unbounded in one deme. The single-deme "5M cells
   in <3 min" SISTEM claim MUST still hold (re-measure it).
5. **Gating decision:** the user wants Option A as the new DEFAULT and all results updated (not gated).
   Re-baseline fixtures accordingly. (If some published result can't be cleanly re-baselined, flag it.)

RE-VALIDATE (the bulk of the work — every spatial output changes)
- `~/miniconda3/envs/iscc/bin/python -m pytest -q` GREEN. Growth-dependent fixtures WILL change; update
  them deliberately (don't just loosen assertions). Watch: tau-leaping tests, `test_microenvironment`
  (byte-identical/growth invariants), cohort, diagnostics.
- Re-run and confirm the headline results STILL HOLD (they should — they use small grids where
  territories come from low dispersal, not occupancy — but verify, and regenerate figures):
  * PEtracer: `validation/validate_petracer.py` — the lineage–space confound must still appear.
  * Multi-region: `validation/validate_multiregion_phylo.py` — spurious-parallelism story must still hold.
  * Operating envelope: `validation/validate_operating_envelope.py` — regenerate phase diagrams (the
    well-mixed / grid×K axes will shift), and ADD a `tumor.diagnose()` "demes over-filling" check
    (mean cells/deme ≫ carrying_capacity) — now demes SHOULD cap, so this becomes a passing check.
  * Cohort + microenvironment figures if spatial-density-dependent — re-run, confirm.
- Confirm the tumour now SPREADS: mean cells/deme ≈ K, occupied demes ∝ cells/K (a quick spatial run).

UPDATE CLAIMS / DOCS
- `PARAMETERS.md`: remove the "Known limitation" caveat box; fix `carrying_capacity` semantics ("cells
  per deme" now true) + its out-of-range description; document the new well-mixed option and that
  `maximum_death_rate` should be ≥ `max_birth_rate`.
- `DESIGN_crowding.md`: mark implemented/DONE; record final formula + any margin chosen + the
  re-validation outcomes.
- `BACKLOG.md`: flip "Engine bug — carrying capacity not enforced" to DONE.
- `DESIGN_operating_envelope.md` + `diagnostics.py`: add the over-fill QC check.
- Manuscript (`manuscript/paper.tex`): the well-mixed single-deme SISTEM claim STAYS (re-verify the
  number). Check any spatial-structure / tumour-size / carrying-capacity wording; regenerate spatial
  figures if their appearance changed materially. (Note: the abstract/intro two-scale-integration redraft
  and this scalability sentence may still be UNCOMMITTED in the working tree — coordinate.)
- Memory: update `iscc-crowding-bug` to DONE.

DELIVERABLES: the engine fix (both engines) + well-mixed option; raised default `maximum_death_rate`;
updated tests (re-baselined, + a "demes cap near K" test + engine-agreement test); re-run validations
with regenerated figures and confirmation the PEtracer/multi-region/SISTEM-well-mixed results hold; the
docs/claims sweep above; full suite green. Commit on `dev`.

HONEST NOTES: this changes all spatial output — re-baseline, don't paper over. The well-mixed regime is
the subtle part (don't let the fix cap the single-deme SISTEM benchmark). If the spatial figures shift
qualitatively (they may look more spread), that's the CORRECT behaviour — update the captions/claims to
match rather than tuning to reproduce the old pile.
```
