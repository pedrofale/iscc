# Handoff prompt — v1 compartment-dependent selection (+ context-dependent phenotype confound)

Saved 2026-07-18 (revised — now runs on the ductal-field substrate). Copy the block below into a fresh session.
**v1 of `DESIGN_phenotype_plasticity.md` — READ §0–§2 first, don't re-derive.** This is the GENETIC floor:
compartment-dependent selection via two new gene-based axes + the genetic-vs-environment expression confound
(free, via post-growth materialisation). It adds **no** carried epistate and **no** new dynamics parameters —
that's v2, explicitly out of scope here. Branch from current `dev`.

**PREREQUISITE: build `handoffs/ductal_field_substrate.md` FIRST** (multi-gland island field + gland_id labels
+ cross-gland dispersal). This handoff runs ON that substrate: glands = small epithelial rings at 2D positions
in sparse stroma, one founder, multi-focal DCIS→IDC. Count engine (`GenotypeTumor`) only for v1; the cell-engine
mirror is deferred (the substrate is count-only). If the substrate isn't in yet, stop and do it first.

The whole design principle is: **the immune-resistance axis is already exactly this pattern** — a heritable
gene-based trait that attenuates a *local* compartment hazard (`_death_rate`:
`immune_prob_kill·immune_fraction(deme)·(1−immune_resistance)`). v1 = do the same thing for the epithelial ring
(a **breach** trait) and the stroma (a **stromal_survival** trait). Most of the work is "replicate every place
the immune axis touches, for two new axes."

---

```
Implement v1 compartment-dependent selection for iscc, per DESIGN_phenotype_plasticity.md (READ §0–§2 first).
This is the GENETIC floor + the expression confound. Do NOT build any epistate / plasticity dynamics (that is
v2, §3 — out of scope). Keep it OFF-BY-DEFAULT and byte-identical when off.

REPO & ENV
- Repo: /Users/pedroferreira/projects/iscc/repo (branch `dev`). Python/pytest: ~/miniconda3/envs/iscc/bin/python.
- Conventions: commit on `dev` WITH `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`; keep the FULL
  suite green (569 now); OFF-BY-DEFAULT / byte-identical when off (verify with a test, like
  test_wgd_off_is_byte_identical); be honest.

THE PRINCIPLE (one mechanic, already in the engine for immune)
Each compartment contributes a LOCAL hazard to cancer death, attenuated by a MATCHING heritable resistance
trait. `_death_rate` (count.py:403) already does this for immune. v1 adds the same shape for the two ductal-
field compartments (gland epithelial rings + stroma, from the substrate handoff), but keyed DIFFERENTLY (see
DESIGN_phenotype_plasticity.md §2 / DESIGN_ductal_field.md §5):
    death += epithelial_barrier · epithelial_fraction(deme) · (1 − breach)   # LIVE wall-cell fraction
    death += stromal_hazard     · stromal_field(deme)       · (1 − stromal_survival)  # environmental region
THE EPITHELIAL BARRIER IS NEVER A FIXED LABEL: it reads the deme's LIVE epithelial-cell fraction (exactly like
immune_fraction), so it dilutes as cancer crosses the wall (the wall IS the epithelial cells). Cross-gland
(island) dispersal from the substrate BYPASSES the wall (lumen->lumen), so confined DCIS spread needs no breach
— breach gates only LOCAL escape into stroma. THE STROMAL HAZARD, by contrast, is an environmental FIELD
(stroma is seeded sparse, so a live-fraction term would be too weak): a per-deme stroma-region signal
(gland_id==-1, or an F8-style field), NOT the live stromal-cell fraction. Normals are NOT cleared in v1 —
cancer coexists with / passes through them — which is sufficient because the barrier selects on the PRESENCE of
normal cells, not
their removal (normals stay immortal, the crowding-fix invariant, DESIGN_crowding.md). Sequential invasion
emerges: lumen -> breach the ring -> survive the stroma -> (existing) resist immune where present. Each trait
is a mutation -> sequenceable -> recoverable.

PART A — the two heritable axes (mirror the IMMUNE-RESISTANCE axis EXACTLY, end to end)
The immune axis is the precise template. grep `immune_resistance`, `_ir`, `N_ir`, `n_ir`,
`immune_resistance_types` across selection.py / cell.py / count.py and REPLICATE every hit for TWO new axes:
`breach` and `stromal_survival`. Concretely:
- selection.py:
  * __init__ (line 9): add params `prop_breach=0.0, prop_stromal_survival=0.0, breach_effects=1.1,
    stromal_survival_effects=1.1`; store them (mirror prop_immune_resistance / immune_resistant_effects).
  * make_breach() / make_stromal_survival() (mirror make_immune_resistant, selection.py:143) — build
    `breach_types`/`breach` and `stromal_survival_types`/`stromal_survival` using `self.rng` (the LAYOUT rng —
    so gene roles are cohort-comparable, like every other make_*). CALL them in __init__ next to
    make_immune_resistant (selection.py:68).
  * get_breach() / get_stromal_survival() (mirror get_immune_resistant, selection.py:185).
  * update_breach(gs) / update_stromal_survival(gs) (mirror update_immune_resistance, selection.py:276) using
    gs['n_wt_breach']/gs['n_mut_breach']/self.N_breach and the *_effects.
  * self.N_breach / self.N_ss: set wherever self.N_ir is assigned (grep `self.N_ir =`) — same place, same way.
- cell.py:
  * genome_summary template (cell.py:63): add `n_wt_breach: n_breach*2, n_mut_breach: 0, n_wt_ss: n_ss*2,
    n_mut_ss: 0` (the template receives n_breach / n_ss — see count.py below).
  * update_genome_summary_mutation (cell.py:178, SNV branch — around lines 190/194-195) AND
    update_genome_summary_cnv (cell.py:214, the copy-number `sign` branch): count n_new_breach / n_new_ss and
    update n_mut_*/n_wt_* exactly as done for n_mut_disp / n_mut_ir, using selection.breach_types[seg] /
    stromal_survival_types[seg].
  * __init__ evolutionary_parameters (cell.py:118): add `['breach'] = 0.`, `['stromal_survival'] = 0.`.
  * update_evolutionary_parameters (cell.py:143): add — mirroring the immune line EXACTLY —
        self.evolutionary_parameters['breach'] = max(0.0, 1.0 - 1.0/selection.update_breach(gs))
        self.evolutionary_parameters['stromal_survival'] = max(0.0, 1.0 - 1.0/selection.update_stromal_survival(gs))
- count.py:
  * genome-summary template call (count.py:204, the `n_ir=...` seam): add
    `n_breach=len(self.selection.get_breach()), n_ss=len(self.selection.get_stromal_survival())`.

PART B — the two death terms (count engine)
- _epithelial_fraction (mirror _immune_fraction, count.py:391): fraction of the deme that is epithelial
  (self.genotypes[gid].type == "epithelial"). Reuse the total-passed-in optimisation. This is a LIVE fraction
  (the wall dilutes as cancer crosses).
- _stromal_field(deme_idx): the environmental stromal-region signal — NOT a live cell fraction. Simplest v1:
  1.0 if the deme is stroma (self.gland_id[deme_idx] == -1 from the substrate) else 0.0 (optionally smoothed
  by an F8-style field). Stroma is seeded sparse, so keying to the cell fraction would be too weak — use the
  region.
- _death_rate (count.py:455, right after the immune line): add the two terms, clamping the trait to [0,1] the
  same way `ir` is clamped (count.py:454):
        b  = min(max(rep.evolutionary_parameters["breach"], 0.0), 1.0)
        ss = min(max(rep.evolutionary_parameters["stromal_survival"], 0.0), 1.0)
        death += self._epithelial_barrier * self._epithelial_fraction(deme, total) * (1.0 - b)
        death += self._stromal_hazard     * self._stromal_field(deme_idx)          * (1.0 - ss)
- config: self._epithelial_barrier / self._stromal_hazard, DEFAULT 0.0 (off -> the terms vanish -> byte-
  identical). Put them next to self._immune_prob_kill (count.py:194) — read from spatial_params so the "payoff
  table" is edit-a-config, not edit-the-engine.

PART C — the cell engine: DEFERRED for v1
The ductal-field substrate is count-engine only, so implement the death terms in count.py's _death_rate ONLY.
Do NOT port to Deme.get_cancer_death_rate in this handoff (the cell engine has no ductal field yet). Note the
deferral; a later handoff mirrors both once the substrate is on the cell engine.

PART D — context-dependent phenotype = the confound (R13 route-3, ALREADY BUILT)
The confound is free: cell_exp is materialised post-growth as f(genotype, niche), and R13 route-3 (niche ->
program) is already wired (count.py:877, `P.niche_drive({"hypoxia":..., "cci":...})`). v1 only has to make the
COMPARTMENT a niche field:
- Add per-deme "epithelial" / "stromal" fractions (or a single compartment field) to `self.microenv_truth`
  (where hypoxia/cci live) and pass them into the P.niche_drive({...}) dict (count.py:882).
- In the program config (route-3 map, programs.py:62 "which niche field drives which program"), drive the
  seeded `emt`/invasive program from the epithelial-interface field. Now the SAME clone expresses the invasive
  program at the epithelial front and less in the stroma -> env-responsive phenotype + the genetic-vs-niche
  confound, with iscc knowing both contributions. (The GENETIC arm — a driver that also drives the invasive
  program — already exists via R13 route-1/2; optionally give breach genes an expression effect in
  make_expmap, selection.py:151, but that's not required for v1.)
- This is READOUT-ONLY: programs never feed back into fitness (programs.py:24) — keep it that way in v1.

OFF-BY-DEFAULT / BYTE-IDENTICAL (critical)
prop_breach=0 & prop_stromal_survival=0 & epithelial_barrier=0 & stromal_hazard=0 must reproduce current growth
EXACTLY (empty axes -> N_*=0 -> update_* returns 1 -> trait evo-param 0 -> zero hazard terms; new
genome_summary keys must not perturb any existing computation). Add a byte-identical test.

PARAMETERS.md: document prop_breach, prop_stromal_survival, breach_effects, stromal_survival_effects (Selection
section) and epithelial_barrier, stromal_hazard (spatial/microenv section), all default 0/off.

VALIDATION (validation/validate_compartment_selection.py -> manuscript/figures/validation_compartment.png):
Grow the DUCTAL FIELD (substrate handoff: n_glands>1, island dispersal on) with both axes + barriers ON. Show:
  (A) DCIS -> IDC: breach sweeps as cancer escapes glands into stroma; stromal_survival sweeps as it traverses
      stroma; multi-focal foci (from island dispersal) each a DCIS focus until a subclone breaches. plot_grid
      coloured by gland_id + compartment + trait. Barrier OFF -> confined DCIS only (control).
  (B) SELECTION-recovery benchmark: scDNA/bulkDNA -> can a selection-inference method recover WHICH genes are
      breach / stromal drivers and the COMPARTMENT each was selected in, vs iscc ground truth?
  (C) CONFOUND benchmark: for a FIXED clone, scRNA invasive-program activity by compartment (env-responsive
      phenotype); show a naive "invasive expression => invasive genotype" call is confounded by location, and
      quantify the genetic vs niche contributions iscc knows.
Print headline numbers.

TESTS (tests/test_compartment_selection.py):
- OFF-by-default byte-identical (both axes + barriers 0 -> identical to a ductal-field-substrate baseline).
- a breach-competent genotype has strictly LOWER death than a non-breacher in an epithelial(wall)-occupied
  deme, and EQUAL death in a pure-lumen/cancer deme (breach pays off only at the wall).
- stromal_survival: strictly LOWER death in a stroma-region deme (stromal_field==1), EQUAL elsewhere.
- ground truth: breach / stromal_survival gene counts + the per-genotype traits surface correctly.

DELIVERABLES: the two axes end-to-end (off-by-default); the two _death_rate terms in the COUNT engine (cell
engine deferred, Part C); the stromal field + epithelial live-fraction; compartment/gland as an R13 niche field
driving the invasive program; PARAMETERS.md; validate_compartment_selection.py + figure; tests; a short
manuscript paragraph (compartment-dependent selection on the ductal field + the genetic-vs-niche expression
confound); flip a BACKLOG item. Full suite green; commit on `dev`.

HONEST NOTES: keep it to v1 — resist adding a carried epistate, memory, noise, or selection-on-phenotype (that
is v2, and it's where the hard-to-tune knobs live; do NOT introduce them). If, off-by-default, growth is NOT
byte-identical, STOP and fix — a new genome_summary key or evo-param is perturbing an existing path. If the
epithelial ring (1 deme thick, count.py:347) turns out too thin to select breach meaningfully, prefer raising
epithelial_barrier over changing the geometry, and note it. The "payoff table" is the {epithelial_barrier,
stromal_hazard} config + the two prop_/effects — keep it small (the identifiability discipline, DESIGN §0); do
not add per-interaction coefficients or extra hazard fields, and do NOT add normal-cell clearance (not needed —
the barrier selects on the presence of normals, not their removal).
```
