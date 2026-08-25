# DESIGN — genotype→phenotype in the structured setting: compartment selection (v1) + plastic epistate (v2) [design-first]

Status: **v1 BUILT (header corrected 2026-08-24); v2 STILL DESIGN-FIRST.** Supersedes the ad-hoc
"compartment × payoff" plan from the other session. Extends R12 (cell-state trajectory) and R13 route-3
(niche→program); builds on F8 (microenvironment) and the existing gene-based `Selection`.

**Correction:** this header previously read "DESIGN-FIRST, NOT built — no engine code until sign-off". That
is stale: **every v1 component in §2 has since shipped** and the header was never updated. Verified
2026-08-24:

| §2 requirement | where |
|---|---|
| `prop_breach` / `prop_stromal_survival` axes + effects | `components/selection.py` `update_breach` (508), `update_stromal_survival` (513) |
| `_death_rate` epithelial + stromal hazard terms | `models/count.py:1458-1467`, exactly the specified form |
| `epithelial_barrier` / `stromal_hazard` knobs | `spatial_params`, e.g. `configs/landing.yaml:104-105` |
| `breach → emt` as the GENETIC arm of the confound | `tumor/programs.py:65` (and the doc's warning against `dispersal_rate → emt` is honoured) |
| compartment → niche→program (route 3) | `tumor/programs.py:418` |
| tests | `tests/test_compartment_selection.py`, `tests/test_ductal_field.py` |

**But "built" is not "validated".** §4 gates v2 on "v1 lands **AND** a benchmark motivates it", and the v1
benchmarks are §2(b) (can DNA-seq recover the breach / stromal-survival drivers and the compartment each was
selected in?) and §2(c) (the genotype-vs-compartment attribution confound). **Whether either was ever
executed is an open question — see §5.** Note also that the shipped `epithelial_barrier: 0.0` switches the
wall term OFF, leaving `stromal_hazard: 0.6` to carry the selection, so the barrier arm of v1 is present in
code but inert in the shipped config.

## 0. The problem, and the three papers that frame it
iscc's engine currently has **no phenotype layer**: a genotype's rates (`division/death/dispersal/immune_
resistance`) are a deterministic function of its genome, cached per genotype (`count.py` `_death_rate`,
`selection.update_*`). The microenvironment modulates only the *payoff* of a fixed phenotype (the immune term,
`_death_rate:453`). We want iscc to be the **ground-truth engine for genotype-vs-phenotype links** in the
structured (normal-cells + microenvironment) setting. Three papers set the frame:

- **Gatenbee, West, …, Graham, Anderson 2019** (bioRxiv 594598, spatial EGT of ductal-carcinoma
  immunoediting: pioneer–engineer, public goods, "space is the game changer"). The biology target — AND the
  cautionary tale: its **payoff matrix (`bG, bN, cC, …`) is unmeasurable**, so it validates only a qualitative
  prediction and stalls. The trap to avoid *and* the gap iscc fills.
- **Whiting, Househam, Baker, Sottoriva & Graham**, *Phenotypic noise and plasticity in cancer evolution*,
  **Trends in Cell Biology 34:451-464 (2024)** (corrected 2026-08-24: this doc previously cited it as
  *Trends in Cancer*): a phenotype has three sources —
  **genetic**, **plasticity** (non-genetic, *environment-responsive*, reversible), and **noise** (non-genetic,
  *environment-independent*, stochastic). Separating them from data is the field's open problem.
- **Yanai lab, Nat Genet 2022** (s41588-022-01141-9): recurrent cancer cell-state modules across 9 cancer
  types, **causally linked to TME cell types**, with invasive/pEMT states **at the leading edge** — i.e.
  microenvironment-driven, spatially patterned, reversible states. The validation target.

### The disciplining insight (do not skip — it defines the whole v1/v2 split)
Separate two layers, because the confound lives in only one of them.

- **Selection dynamics (rates).** Modelling rates already gives *process stochasticity*; compartment payoffs
  already give *context-dependent selection* (env reads a **fixed** phenotype differently). A plastic state
  that **equilibrates instantly to the current compartment is observationally identical to context-dependent
  selection** — it buys nothing for the dynamics.
- **The observable phenotype (expression).** iscc materialises `cell_exp` **after** growth, as a function of
  (genotype, niche) (R13 dosage/SNV effects + route-3 niche→program, F8 fields). So the **same clone in two
  compartments already expresses differently** — env-responsive phenotype in Graham's sense — and the
  **genetic-vs-environmental attribution confound is present in the data with ZERO epistate and zero new
  dynamics parameters.** It is memoryless (reflects the *final* location only), but the confound does not
  require memory. This is how the engine already works; v1 only has to feed the *compartment* into the
  niche→program input.

So the confound — the thing the framing papers are actually about — is **free in v1**. A carried, memoried
epistate is needed ONLY for the three things a post-growth, memoryless readout cannot represent:
  1. **memory / hysteresis** — phenotype reflecting the *trajectory*, not just the final location;
  2. **selection acting on the phenotype** — evolutionary feedback (the plastic state itself under selection),
     vs v1 where the phenotype is a downstream readout that does not affect fitness;
  3. **persistent env-independent noise** — heritable-with-decay stochastic states, i.e. **drug-tolerant
     persisters** that arise without a mutation and are transiently inherited.
None of the three is needed for the confound. So: **v1 = context-dependent selection + context-dependent
phenotype-at-materialisation** (shows the confound, no epistate, no new dynamics knobs). **v2 = the minimal
memoried + noisy epistate**, justified only by (1)–(3). v1 is a strict limit of v2 (τ→0, phenotype→readout),
so v2 always degenerates back to it.

## 1. What already exists (reuse, do not rebuild)
- **Gland geometry = lumen / epithelial ring / stroma.** `_seed_structure` (`count.py:345`) seeds an
  epithelial ring at `structure_radius`, the cancer founder just inside (lumen), stroma outside. The
  compartments are already spatial.
- **Four gene-based heritable axes** (division onc/TSG, dispersal, immune-resistance, treatment-resistance),
  each = mutations at designated genes (`selection.make_*`), fitness via `_rel_fitness`. Sequenceable →
  recoverable. This is *why* iscc can benchmark selection inference; do not replace it with an abstract vector.
- **Context-dependent selection, already implemented, for immune:** `_death_rate` adds
  `immune_prob_kill · immune_fraction(deme) · (1 − immune_resistance)` — a heritable trait that pays off only
  where immune cells are. Treatment resistance is likewise context-gated. This is the template v1 generalises.
- **F8 microenvironment fields** (hypoxia O2, CCI) + per-deme modifiers — the substrate that drives env effects.
- **R13 program activity `z`** (the per-cell phenotype vector) + **route-3 (niche→program)** design — the
  natural home for the v2 epistate. R12 is the plastic-landscape framing.

## 2. v1 — compartment-dependent selection (GENETIC; the identifiable floor)
**Runs on the ductal-field substrate** (`DESIGN_ductal_field.md`): many small epithelial-ring glands at
2D positions in moderate-density stroma, a single founder, local (cross-deme) + island (cross-gland) dispersal — so the
selection below plays out as multi-focal DCIS → IDC, and it's ST-usable. **Principle (one mechanic per
compartment):** a compartment contributes a local **hazard** to cancer death, attenuated by a **matching
heritable resistance trait** — exactly the existing immune term, generalised.

- **Two new gene-based axes** (mirror `make_immune_resistant`): `prop_breach`, `prop_stromal_survival`, with
  `breach_effects` / `stromal_survival_effects`, `N_breach`/`N_ss` counts, `update_breach`/`update_stromal_
  survival` returning `_rel_fitness` multipliers. They flow into `cell_data` + DNA-seq for free.
- **`_death_rate` gains two terms**, both keyed to LIVE cell fractions (like the immune term):
  ```
  death += epithelial_barrier · epithelial_fraction(deme) · (1 − breach)          # live wall cells
  death += stromal_hazard     · stromal_fraction(deme)    · (1 − stromal_survival) # live stromal cells
  ```
- **Neither is a fixed label — both read the deme's live cell fractions** (like the existing `immune_fraction`),
  so pressure changes as cancer accumulates and dilutes the resident normals. The epithelial wall *is* the
  epithelial cells (dilutes as cancer crosses); the stromal cells (fibroblasts etc., seeded at moderate
  density, `DESIGN_ductal_field.md` §2) *are* the hostile microenvironment. **Normals are not cleared** (cancer
  coexists with / passes through them); the hazard selects on *presence*, not removal, so no clearance is
  needed (normals stay immortal, `DESIGN_crowding.md`). Cross-gland (island) dispersal bypasses the wall
  entirely (lumen→lumen) → confined DCIS spread needs no breach; breach gates only the *local* escape into
  stroma.
- **Emergent behaviour (selection):** sequential invasion — lumen → breach the epithelial ring → survive the
  stroma → resist immune where present. Each barrier selects a *different* heritable trait, each recoverable by
  sequencing. The **selected traits are genetic** (permanent, additive, cannot switch off): who *survives* a
  compartment is genotype × context.
- **Context-dependent phenotype = the confound, for free.** `cell_exp` is materialised **after** growth as
  `f(genotype, niche)`, so v1 must feed the **compartment identity into the niche→program input** (R13 route-3
  — a few existing program parameters, NOT an epistate). Then the *same clone* expresses an invasive-ish
  program at the epithelial front and a different program in the stroma: **env-responsive phenotype, and the
  genetic-vs-environmental attribution confound, present in the data with zero carried state.** This phenotype
  is memoryless (final location only) and is a **readout** — it does not feed back into fitness in v1.
  - **The two arms of the `emt` confound, concretely.** The **niche arm** is route-3 (`epithelial → emt`). The
    **genetic arm** is route-1 **`breach → emt`** — the heritable invasion trait itself drives the invasive
    program (`DEFAULT_PHENOTYPE_PROGRAM_MAP`, added alongside the legacy `dispersal_rate → emt`). Do **not** use
    `dispersal_rate → emt` as the genetic arm: `prop_dispersal = 0` here, so dispersal is constant and that
    route contributes zero drive (inert). `breach` is both under selection at the front (§2) *and* the driver
    of the genetic component of the invasive signature — so the confound benchmark contrasts the *same* trait's
    genetic expression signal against the epithelial-niche signal.
- **Parameters (~4 selection knobs, all ground-truth):** `epithelial_barrier`, `stromal_hazard`, plus
  `prop_`/`_effects` for the two axes. The phenotype confound reuses existing R13/F8 niche→program knobs — **no
  new dynamics parameters.**
- **v1 validation:** (a) spatial invasion dynamics vs `structure_radius`/hazard coefficients; (b) selection
  benchmark — can DNA-seq + a selection-inference method **recover the breach / stromal-survival drivers** and
  the compartment each was selected in? (c) **the confound benchmark** — given scRNA state + spatial data, can
  a method attribute an invasive expression state to genotype vs compartment, when iscc knows both? (d) the
  "invasion requires a driver mutation" regime (the genetic baseline the v2 plastic regime is contrasted with).

**v1 is NOT:** memoried, phenotype-selected, or persister-bearing (the §0 (1)–(3) list). It *is* env-responsive
in its observable phenotype — enough for the confound — and it ships the spatial-invasion story on ~4 knobs.

## 3. v2 — the plastic epistate (heritable-with-decay + noise; the flagship)
The **single** new object, kept deliberately minimal to dodge the Gatenbee trap. **Justified only by the three
things v1's post-growth, memoryless phenotype cannot represent** (§0): (1) memory/hysteresis, (2) selection
acting on the phenotype, (3) persistent env-independent noise (persisters). The basic genetic-vs-environmental
confound is NOT a reason to build v2 — v1 already has it.

- **A low-dimensional epistate `s`.** Reuse R13's seeded programs — e.g. `s ∈ {invasive, proliferative,
  quiescent}` activities — NOT a free matrix. Discretised to a few levels for engine tractability.
- **Three dynamics terms, one per Graham category:**
  1. **genetic bias** `β_bias`: the genotype sets `s`'s attractor / lowers the barrier to a state (a driver
     makes the invasive state *more accessible*, not mandatory) — "the genetic determinant of a non-genetic
     phenotype".
  2. **plasticity** `τ_relax`: the local compartment/niche (F8) pulls `s` toward a target with relaxation
     timescale `τ_relax`. **This timescale is the entire difference from v1** (τ→0 ⇒ instantaneous ⇒ v1).
  3. **noise** `σ_noise`: env-independent stochastic switching of `s`.
- **Inheritance = the memory:** daughters inherit `s` with decay toward the genetic attractor + noise
  (epigenetic-like, not genetic). A cell that left the epithelium stays partly invasive for ~`τ_relax`.
- **Selection reads `s`, not the genotype:** `rates = f(baseline(genotype), s, compartment)`. So breach pays
  off when `s` is invasive **and** the cell is at the epithelium; move to stroma and `s` relaxes → the
  phenotype switches **reversibly** with the same genotype. The breacher→proliferator switch falls out.
- **Engine cost (be honest):** a memoried `s` travels with cells across demes, so it splits each genotype into
  **(genotype × epistate) sublineages**. Discretising `s` to a few levels keeps this a small multiplier on the
  genotype count, not a per-cell state — but it is a real change to the count engine's caching, not a free
  rate modifier. This cost is *why* v2 is only worth it for the attribution question.

### 3.1 Tunability discipline (the concern, addressed directly)
The worry — "a lot of stuff that's hard to tune" — is right, so the design is constrained to make it *not* so:
- `s` is **2–3 axes, few discrete levels** (not a continuum, not a matrix).
- **Exactly three new dynamics parameters**, each individually interpretable and each with a degenerate limit:
  `τ_relax` (memory; **τ→0 recovers v1**), `σ_noise` (noise; **σ→0 ⇒ deterministic plasticity**), `β_bias`
  (genetic pull; **β→∞ ⇒ genetic-only**). So **v2 CONTAINS v1 as a limit** and each knob is testable by
  degeneration — you are never tuning blind.
- These are **ground-truth knobs, never fit.** The deliverable is an **identifiability map over (τ, σ, β)**,
  not a calibration. We never claim the "right" values — we ask what data can recover.
- **The Gatenbee lesson is encoded as a hard scope rule:** no per-interaction payoff coefficients, no
  macrophage public-goods matrix in v2. Those are v3 (§4), gated on a specific benchmark, because each is a new
  unmeasurable parameter.

### 3.2 v2 validation = the flagship identifiability benchmark
The *basic* genetic-vs-environment confound is a v1 benchmark (§2). v2's benchmarks are the ones that need the
carried state — from a **known** `(τ, σ, β)`, ask what inference recovers:
- **Separate plasticity (env-responsive) from noise (env-independent)** — Graham's finer split, which *requires
  memory* to be identifiable (an env-independent state persists against the niche; a plastic one tracks it).
  v1 cannot pose this; v2 can, with ground truth.
- **Detect persisters** — a heritable-with-decay drug-tolerant state arising without a mutation; can a method
  distinguish it from a resistance driver? (needs (3).)
- **Migration-history / hysteresis inference** — does a cell's lagged state reveal where it *came from*? (needs
  (1); impossible for a memoryless readout.)
- **Phenotype-driven evolution** — when the plastic state is under selection ((2)), does the population evolve
  toward a state the genotype alone would not predict?
- **Reproduce Yanai** — recurrent, TME-driven, spatially-patterned states emerge from the niche→state pull
  (partly already in v1; v2 adds the leading-edge *persistence* behind the front).

### 3.3 THERAPY as a fourth driver of `s` — the drug-tolerant/resistant state (added 2026-08-24)
v2 as written above pulls `s` toward a target set by the **niche** (compartment / F8 fields). This subsection
adds the missing driver: **the drug itself**. It is a genuine extension, not a restatement — nothing above
lets therapy drive a state transition — and it is the concrete motivation that §4 asks for.

**Why it is needed (measured, not assumed).** iscc models treatment resistance as an **SNV at a genomic
locus**, and the genome undergoes segmental copy-number change. A deletion of the copy carrying that SNV
takes `n_mut_tr` 1 → 0 and the cell is sensitive again. The rate factorises as
`P(cnv) * (1/total_copy_number) * (1 - amp_prob)` and equals **the fraction of genome DELETED per division** —
so it is proportional to the tumour's deletion burden. Measured at the shipped rates: **4.3e-4 per division
(1 in 2,325)**, which is unremarkable per-locus LOH for a CIN+ tumour. What makes it dominate is supply: an
~80,000-cell resistant clone does ~30,000 divisions per generation, so ~12-17 revertants appear every
generation.

**It IS tunable — an earlier draft of this section wrongly said otherwise (corrected 2026-08-24).** Two of
the three factors are knobs, and lowering either lowers reversion proportionally:
- **`amp_prob` up** (0.5 → 0.8) halves reversion. Cost: measured ploidy inflation 2.34 → 3.06, and because
  `_trait_fitness` is `effect ** (2*n_mut/ploidy)` (`selection.py:434`) that dilutes EVERY graded trait.
- **`cnv_prob` down** scales reversion linearly with NO ploidy penalty. This was previously dismissed as
  "calibrated" (the shipped value targets ~1/3 of the genome altered for luminal breast, `count.py:78-83`),
  but that is a statement about a tumour TYPE, not a constraint: **a CIN-low tumour is a perfectly real
  regime** (MSI/hypermutant colorectal is typically near-diploid). Dropping `cnv_prob` 10x cuts reversion
  10x, to ~1.7 revertants/generation. UNTESTED — worth trying before building any of §3.3.

So the honest statement is not "reversion cannot be reduced" but **"reversion can only be reduced by
asserting a lower-CIN tumour"** — available and defensible, at the price of committing the escape-mode
figures to a low-CIN regime, which is a different tumour from the ductal-field/luminal one everything else
here is calibrated to.

Every lever tried against this failed or nearly so (2026-08-13..24, see memory `maley-escape-modes`):
- `amp_prob` 0.5→0.8 halves reversion but inflates ploidy, which dilutes EVERY graded trait (`2*n_mut/ploidy`).
- Raising `kill_rate` kills revertants faster but starves the de novo origin mode IV needs.
- `mutagenicity` **cannot shift the balance at all**: acquisition:reversion is 0.14 at 1.0 and 0.14 at 4.0,
  because `mutation_rate` is the mutate-vs-disperse FATE probability and scales both branches together.
  (`mutagenicity_target="snv"`, committed 22899ea, fixes that half — it scales `n_snvs_per_allele`, which
  feeds only the SNV branch, giving ratio 1.74 at x20 with reversion flat.)
- A trait LATCH (resistance can never decrease) reaches 100% but was REJECTED as unbiological, correctly:
  resistance alleles ARE lost. Do not re-propose it.

The residual floor is structural. Reversion needs EVERY mutated resistance position gone, and one deletion
removes one copy of one segment — so protection comes from carrying hits on SEVERAL segments:

    mode  resistance loci   mean n_mut_tr/cell   sensitive cells at relapse
    I         300                6.33                     0
    II         36                2.52                     2
    III        36                2.01                   276
    IV          2                1.86                   749

Mode I reaches exactly zero because 300 loci spread across all 12 segments give 6.33 hits no single deletion
can clear. **Mode IV is defined by resistance being rare enough to be ABSENT at the first dose**, which forces
few loci (2, and in the shipped layout both on the SAME segment), which forces low load. Zero and de novo are
close to mutually exclusive under a purely genomic resistance model. That is the wall this subsection exists
to go around.

**The mechanism.** Resistance becomes a carried epistate level rather than (only) a genomic trait:
- **Entry, genetic (`β_bias`):** acquiring `n_mut_tr > 0` sets the attractor for the tolerant level. The state
  is then carried INDEPENDENTLY, so a later CNA that deletes the triggering allele leaves the cell tolerant.
  This is the whole point: it breaks the coupling between resistance and copy-number accident.
- **Entry, drug-induced (`τ_relax` against a therapy target):** while dosed, `s` is pulled toward the tolerant
  level at a rate rising with **the dose the cell actually receives** — i.e. scaled by `(1 - protection)`,
  the same factor `_kill_amount` uses, so a cell the drug cannot reach is not reprogrammed by it.
- **Exit (`τ_relax` + `σ_noise`):** off drug the attractor returns to sensitive and `s` relaxes back over
  `τ_relax` divisions. Permanent resistance is the `τ_relax → ∞` limit; a pure persister is short `τ_relax`.
- **Cost:** the tolerant level charges proliferation like the other traits, so OFF drug a cell that exits is
  fitter — the state reverts by selection as well as by relaxation. That is the adaptive-therapy dynamic
  (Gatenby 2009; Strobl 2021) reproduced rather than asserted.

**Why this is NOT the rejected latch.** The latch claimed a genetic allele cannot be lost, which is false.
This claims resistance is a **phenotype that, once induced, is epigenetically maintained** — Sharma 2010's
chromatin-mediated reversible drug-tolerant state, and the "epigenetically reinforced stress response
regulation" of França/Yanai. Exit still happens; it is governed by `τ_relax`, a parameter we choose and can
degenerate (`τ→0` recovers no-state), rather than by a copy-number accident whose rate we do not control.

**Literature anchor.** França, ..., Yanai, *Drug-induced adaptation along a resistance continuum in cancer
cells* (bioRxiv 2022.06.21.496830; Nature 2024): **gradual dose escalation over ~a year**, explicitly modelled
on Baym's MEGA-plate, drives cells along a continuum — initial sensitivity → physiological adaptation →
dedifferentiation → stable resistance — through multiple **cell-state transitions** with distinct expression
programs (interferon response, lineage reprogramming, metabolic rewiring, oxidative stress). Changes are
"transcriptional, epigenetic AND genetic". Low doses **prime** cells for higher ones. This is de novo
resistance with no allele to lose, and it is the target behaviour.

**Dose escalation is the natural companion, and it is free.** `arc.py` builds a fresh `Chemotherapy` per
schedule entry, so a multi-phase escalating course is expressible in config TODAY, with no engine change. It
matters because a fixed dose is bounded by the resistant clone's own survival — under the graded map a
one-hit clone still absorbs 36% of the dose and dies above `kill_rate ≈ 1.22` — whereas under escalation the
clone tolerates a higher dose *because* it gained a level at the previous one. Escalation should therefore
select for INCREASING load, which is exactly the reversion-proofing that mode I gets for free and mode IV
cannot. Worth testing on the existing genomic model before building any of the above.

**Where the four escape modes sit in Whiting & Graham's taxonomy — and why that is the argument for this
section.** Their split of phenotypic variation is genetic / plasticity (environment-responsive) / noise
(cell-intrinsic stochastic), separated operationally by **lineage tracing**.

    mode  iscc mechanism                                          Whiting & Graham category
    I     SNVs at resistance loci, present in the naive tumour     GENETIC
    II    same, swept before therapy                               GENETIC
    III   same, rare pre-existing clone selected by the drug       GENETIC
    IV    same, arising under therapy-elevated mutation rate       GENETIC

**All four are genetic; iscc has ZERO coverage of plasticity or noise.** That is not a gap in the escape-mode
figures, it is the reason reversion has been intractable: a genomic allele is losable by copy-number accident,
and a genomic allele is the only kind of resistance the engine can currently represent. §3.3 adds the other
two categories — drug-induced transition is **plasticity**, `σ_noise` is **noise**, and `β_bias` (a mutation
setting a non-genetic state's attractor) is precisely the **genetic determinant of a non-genetic phenotype**
that their paper is about. Their operational criterion is lineage tracing, which iscc has as GROUND TRUTH —
so this is not only a fix for mode IV but a direct benchmark against their framework.

**v2 validation, therapy arm** (extends §3.2's "detect persisters"):
- **Mode IV to a true zero without forbidding reversion.** Current best is 99.10% resistant / 749 sensitive
  cells (rig, seed 5, `treatment_resistance_cost 0.35` below the measured 0.401 reversion break-even, plus
  `mutagenicity_target=snv` at x20). Target: 0 sensitive with reversion still firing at its calibrated rate.
- **Can inference tell a drug-induced tolerant STATE from a resistance DRIVER?** — the §3.2 question, now with
  a therapy-driven arm and a ground-truth `τ_relax`.
- **Reproduce the resistance continuum:** does escalating dose walk the population up the `s` ladder, and does
  a drug holiday walk it back down at `τ_relax`?

**Engine cost, restated for this arm.** §3 already flags that a carried `s` splits each genotype into
(genotype × epistate) sublineages. One trap specific to this arm: **`s` cannot be derived from the genome**
(that is the entire point), so a state transition must MINT A NEW GENOTYPE ID — two cells with identical
genomes and different states are different engine entities. Genotype ids already come from a counter rather
than genome content (`Cell.set_genotype_id`), so this is expressible, but it inflates the append-only
registry and that is the real scalability question to answer before building.

### 3.4 §3.3 BUILT AND MEASURED (2026-08-25) — results, and one correction to §3.3

The therapy arm shipped in `ad86f23` (`resistance_state_*` on `Selection`, all knobs default-inert, so
`resistance_state_on` is False and division is byte-identical when off). Verified independently: **908
passed / 1 skipped**, +21 over the 887 baseline, golden hashes unaffected.

**Headline.** On the mode-IV rig, seed 5, like-for-like against the genomic-only baseline (`PM5`, 749
sensitive / 99.095%): **749 → 0 sensitive, 100.0000% resistant, all four mode-IV criteria pass**, with the
full 90-generation drug-free tail and `cnv_prob`/`amp_prob` untouched. Reversion is NOT suppressed —
**12,545 of the 86,117 relapse cells (14.6%) have actually deleted the allele**; the state reclassifies them
rather than preventing the deletion. That is exactly the decoupling §3.3 was built for.

**CORRECTION to the "Exit" bullet above.** That bullet presents `τ_relax → ∞` as how one gets permanent
resistance, and the acceptance run duly used `relax = 0, cost = 0` — the most latch-like corner of the
feature, which is a poor advertisement for a mechanism whose whole defence is that exit remains possible.
A 2×2 over (relax, cost) at the original rig, seed 5, gives the actual structure:

    sensitive cells at relapse      cost=0      cost=0.35
                     relax=0             0              0      (100.0000% resistant)
                     relax=0.02     11,180         15,755

**Exit is the load-bearing knob; the cost is not.** Cost ALONE leaves the count at exactly 0 — with no exit
a revertant keeps the state and stays classified resistant — and only amplifies (~+40%) once exit is on.
Note exit is gated to cells with `n_mut_tr == 0` (`count.py:_apply_state_transitions`), i.e. precisely the
CNA revertants; an allele-anchored cell keeps its genetic attractor and never relaxes.

**But τ→∞ is NOT actually required, and the 15,755 was an artefact of MY parameterisation.** Setting both
costs to 0.35 does not price the two resistance routes equally: costs stack multiplicatively
(`selection.py:_division_cost_factor`), so an allele-anchored state cell divides at
`(1 - 0.35*0.6429) * (1 - 0.35) = 0.504` while an exited revertant pays neither and divides at 1.0 — a ~2×
per-generation edge over an 89-generation tail. Because `resistance_state_genetic` puts every allele-carrier
INTO the state, the state cost can simply REPLACE the genomic one:

    treatment_resistance_cost = 0.0    resistance_state_cost = 0.35    relax = 0.02

Then every in-state cell pays exactly 0.35 whether or not it still carries the allele, and only an EXITED
cell pays nothing. Measured: **2,029 sensitive, 97.658% resistant, 4/4 criteria, and 0 resistant at the
first dose** — the clean start survives, because the state cost is charged from the moment the allele is
acquired and so keeps resistance rare pre-treatment. That is **5.5× fewer sensitives than exit-only** and
7.8× fewer than the double-charged run.

So the two claims to keep separate when writing this up:
- **mode IV as a CLASSIFICATION is robust to a finite exit rate** — it passes 4/4 even at `relax=0.02` with
  no cost at all (11,180 sensitive, 87% resistant);
- **a relapse that READS as uniformly resistant needs only fair pricing, not permanence** — 97.66% at
  `relax=0.02`. The `τ→∞` corner buys the last 2.3 percentage points and nothing else.

**A SHARED MUTAGEN ACROSS ALL FOUR PANELS IS IMPOSSIBLE — and this is a result, not a tuning failure.**
Mode IV needs a mutagenic drug to manufacture its de novo clone. Putting that same drug on every panel
(`mutagenicity=20 mutagenicity_target=snv`) collapses II, III and IV into one story:

    mode   with shared mutagen                    without (the panels as published)
    I      315 (100.0%)   9.0x  PRE-EXISTING      2,297 (100.0%)   1.0x  PRE-EXISTING
    II      16 (  0.5%)   6.0x  DE NOVO           2,594 ( 97.5%)   1.0x  PRE-EXISTING
    III     21 (  0.8%)   5.1x  DE NOVO              33 (  1.5%)  11.7x  PRE-EXISTING
    IV       0 (  0.0%)  39.5x  DE NOVO               0 (  0.0%)  39.5x  DE NOVO

The mutagen seeds de novo resistance in II and III as well, and that lineage outruns the pre-existing clone
those panels are built around: II never completes its pre-treatment sweep (0.5% vs 97.5%) and III drops
below its own ≥10× responder bar (5.1× vs 11.7×). Since **III and IV differ only in where the relapse
lineage came from**, the drug that makes IV possible is exactly the drug that destroys III. Do not try to
retune this away — state the asymmetry in the caption.

**Registry cost, answered.** §3.3's closing paragraph flags genotype minting as "the real scalability
question to answer before building". Measured: at `induction = relax = 0` no twins are minted and the
genotype count equals the genomic baseline. With the state active the registry reaches ~556k genotypes on
the mode-IV rig (~616k for the fair-pricing run) against ~324k for a no-state panel — roughly a 1.7-1.9×
inflation, which the tau engine carries at rig scale without swapping. Not free, not prohibitive.

**Still not exercised at figure level:** the drug-induced ENTRY arm (`resistance_state_induction`). Every
result above uses genetic entry only; induction is unit-tested but has never driven a figure.

## 4. Staging / scope
- **v1 (build first):** ~4 selection params into `Selection` + `_death_rate`, plus feeding compartment into the
  existing R13/F8 niche→program input (no new dynamics knobs). Ships spatial invasion + the
  genetic-selection-recovery benchmark **and the genetic-vs-environment expression confound**; the identifiable
  baseline. No carried epistate.
- **v2 (the flagship; likely paper-2):** the epistate `s` + `(τ, σ, β)` + selection-reads-state; the
  genotype-vs-phenotype identifiability benchmark. Build only after v1 lands **and** a benchmark motivates it.
- **Not planned:** normal-cell clearance (mortal normals) and diffusible public-goods fields. Composition-based
  selection does not need clearance (the barrier selects on the *presence* of normals, §2), and we are not
  reproducing/benchmarking the Gatenbee model, so its clearance/public-goods mechanics carry no motivation here.
  Each would only add unmeasurable parameters (the §0 discipline).

## 5. Open decisions
- **v1 BENCHMARK STATUS — checked 2026-08-24, HALF-MET.**
  - §2(c) **the confound benchmark: RAN.** `notebooks/compartment_selection_confound.ipynb` is executed
    (4/5 code cells carry stored outputs): DCIS→IDC transition, escape traits at the invasion front, a
    matched no-gate control, and §4 "is invasive *expression* driven by genotype or by niche?".
  - §2(b) **driver recovery: NEVER RAN.** No dN/dS, no selection-inference tool, no recovery test anywhere —
    the notebook only mentions "infer" in passing, and `tests/test_compartment_selection.py`'s 12 tests are
    unit tests (byte-identical-when-off, trait surfaces, death-rate terms), not a benchmark.
  So §4's gate on v2 is **half satisfied**: the confound benchmark motivates it; the driver-recovery one is
  outstanding. Decide whether (b) is a prerequisite or can run in parallel.
- **Does §3.3's therapy arm need the full `s` ladder, or a two-level special case first?** A single axis with
  two levels (sensitive / tolerant), driven only by therapy, is a strict SUBSET of v2 and would settle mode IV
  now. The risk is building a parallel mechanism instead of a subset — if taken, it must be shaped so the full
  `s` generalises it rather than replacing it.
- **Test dose escalation on the existing GENOMIC model first** (§3.3): it is config-only today and predicts
  increasing `n_mut_tr` load per dose step. If that alone reversion-proofs mode IV, the therapy arm may not
  be needed for this particular problem.
- Reuse R13 `z` as the epistate vs a dedicated small state (recommend **reuse** — one phenotype object across
  the engine and the data modalities).
- Epistate discretisation level (tractability vs fidelity) — the main engine-cost dial.
- Does v1 `breach` gate **survival-in** or **dispersal-into** an epithelial-occupied deme? (mechanics detail;
  survival-in matches the immune template most directly.)
- Epithelial ring thickness (currently 1 deme) — may need thickness or just a strong `epithelial_barrier`.

## 6. What this deliberately avoids
- The abstract `(b, d, w)` trait vector (un-sequenceable → breaks iscc's inference-benchmark premise).
- A free-parameter payoff / game matrix (the Gatenbee unvalidatability trap).
- Deterministic genotype→phenotype (the current limitation we are fixing).
- A *carried* epistate without memory or noise (redundant — post-growth `expression = f(genotype, niche)`
  already gives the memoryless env-responsive phenotype and the confound; the §0 insight).
- Building v2 before v1, or v3 before a benchmark needs it.
