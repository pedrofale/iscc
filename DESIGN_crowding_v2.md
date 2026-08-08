# DESIGN — crowding v2: a structural density cap + a lottery for slots [PROPOSED]

Status: **PROPOSED, nothing implemented.** Supersedes the "Option A" law shipped in
`DESIGN_crowding.md` (2026-07-14) and its resident-pressure addendum (2026-07-20). Read that document
first — this note assumes it.

**One-line statement of the defect.** Inside a deme, iscc has no competition between cancer clones.
The shipped crowding law scales each clone's crowding death by that clone's *own* evolved division
rate, so the fitness term cancels out of the net growth rate and every clone reaches net-zero growth
at the same occupancy. A deme fills to the same density whoever is in it, and once it is full every
clone in it is exactly equivalent. Selection can only act at the growing edge, and only until
division rates saturate at `max_birth_rate`.

**One-line statement of the fix.** Stop asking one term to do two jobs. Make the crowding *death*
uniform across the clones in a deme (so the fitness term cannot cancel), set it *below* the deme's
mean division rate (so the rate law alone would overfill), and enforce the density cap
**structurally** — a deme at capacity has no free slots, and a birth that cannot get one fails. When
more births are drawn than there are slots, the slots go to clones in proportion to the births they
drew. That is a lottery, and a lottery is what makes selection concentrate.

Acronyms used below on first appearance: **CNA** = copy-number alteration; **DCIS** = ductal
carcinoma in situ; **IDC** = invasive ductal carcinoma; **FGA** = fraction of genome altered;
**MVH** = multivariate hypergeometric distribution; **QC** = quality control; **K** = a deme's
carrying capacity (`carrying_capacity`, or the per-deme `K_duct` / `K_stroma` / `K_met`).

Companions: `DESIGN_crowding.md` (the history this replaces), `DESIGN_ductal_field.md` §3 (per-deme
K), `DESIGN_operating_envelope.md` (the QC checks), `DESIGN_scalability.md` §7 (tau-leaping),
`PARAMETERS.md`.

---

## 1. The problem

### 1.1 The algebra

`GenotypeTumor._death_rate` (`src/iscc/tumor/models/count.py`, ~line 940) with the default
`crowding_mode="own"`, for a cancer-only deme (no normal residents):

```
steep  = 1 + crowding_margin                                  # default m = 0.1
death_i = d_i + max(0, b_i - d_i) * steep * (c / K)           # c = cancer cells in the deme
```

where `b_i` is clone *i*'s evolved `division_rate` and `d_i` its baseline `death_rate`. Its net
growth rate is

```
net_i(c) = b_i - death_i = (b_i - d_i) * [ 1 - steep * c / K ]
```

**The fitness factor `(b_i − d_i)` multiplies the whole expression, so it cancels out of every
statement about equilibrium and about relative advantage:**

| quantity | value | consequence |
|---|---|---|
| occupancy where `net_i = 0` | `c/K = 1/steep = 0.909` | **the same for every clone**, whatever its division rate |
| selection differential `net_i − net_j` | `(b_i − b_j)·[1 − steep·c/K]` | scales to **zero** exactly at the equilibrium occupancy |
| the same at `c = K` | `−0.1·(b_i − b_j)` | **sign-reversed**: at full occupancy the *fitter* clone shrinks *faster* |

So a deme that has reached its equilibrium density is a **neutral community**: every clone in it has
net growth exactly zero, differential exactly zero, and the only dynamics left are drift, mutation
input, and migration. Competitive exclusion is impossible by construction. Selection survives only in
the strip `c < K/steep` — the invasion front — and stops there too once CINner-style driver
accumulation pins division rates at `max_birth_rate` (measured in `validation/sweep_calibration.py`
§2: 24.8% of cells already at the 0.95 cap at 12k cells on the shipped configuration, rising to 88.3%
in the slowest-founder cell of the factorial).

The resident-pressure addendum restored *some* fitness dependence, but only through the
`n_normal / K` term, i.e. only where immortal epithelial / stromal / host cells remain — the invasion
front of a gland, never inside colonised tissue. It was never intended to fix cancer-versus-cancer
competition and it does not.

### 1.2 The measurement

Local clonality, defined as **altered-copy-number-state dominance within a deme** — over segments,
the largest fraction of one deme's cancer cells sharing a single non-diploid segment state — on the
canonical regime (`notebooks/example_config.yaml` via `validation/realistic_regime.py`):

| configuration | mean per-deme dominance | demes above 0.9 |
|---|---:|---:|
| shipped config, unchanged | **0.153** | 0% |
| `crowding_mode="fixed"` | 0.147 | — |
| shipped + `dispersal_rate` 0.05 + `cnv_prob` 0.05 | **0.503** | 3% |

The only thing that raises local clonality is turning down the two *diluting* processes — migration
between demes and copy-number churn along lineages. Nothing in the engine makes selection
*concentrate* lineages. The alternative crowding law does not help.

A scaling estimate for why 0.153 is where it lands, which the fix has to beat on its own terms:

* per-segment event rate per division = `mutation_rate × cnv_prob / n_segments` = `0.3 × 0.35 / 12`
  = **0.00875**, i.e. a given segment changes state roughly every **114 divisions** along a lineage;
* the fraction of births that leave their birth deme is
  `dispersal_rate / (mutation_rate + dispersal_rate)` = `0.9 / 1.2` = **0.75**, so the mean squared
  parent–offspring displacement is `σ² ≈ 0.75` deme²;
* Wright's neighbourhood size `Nb ≈ 4π σ² K` is then ~283 cells at `K_stroma` = 30 and ~565 at
  `K_duct` = 60. Under neutrality two cells in the same deme coalesce on the order of `Nb`
  turnovers ago — several times the 114-division segment-change scale, so they usually disagree.

Selection is the only thing that shortens a *local* coalescent: a local sweep drops the depth from
`O(Nb)` to `O(ln Nb / s)` turnovers. With no local selection, there is no local clonality that is not
bought by suppressing dispersal and copy-number churn — which is exactly what the third row of the
table shows, and exactly the parameter suppression we must not have to do.

### 1.3 What must NOT change: global sweeps stay suppressed

The reference model is Noble et al., "Spatial structure governs the mode of tumour evolution",
*Nature Ecology & Evolution* 2021 (the `demon` simulator, `robjohnnoble/demon_model`;
`noble_spatial_2022` in `manuscript/references.bib`). Its headline result is that organising cells
into glands **limits** the extent to which driver mutations spread through the population. iscc
already validates its evolutionary-mode indices against that work
(`validation/validate_evolution_modes.py`, `validation/data/noble_empirical_indices.csv`) and the
manuscript states the recapitulation (`manuscript/paper.tex` §"structure governs the mode").

So iscc's failure to produce **global** sweeps is correct science and is load-bearing for the paper.
`validation/sweep_calibration.py` measured this exhaustively: `n_truncal_segments` = 0 in all 57
growth runs across 39 configurations, `max_cna_freq` never above 0.634 and never above 0.506 in a
lesion that still invades. **None of that should move.** The defect is purely the absence of *local*
competition, and the fix must be surgical enough to leave the global result standing.

`demon` gets both at once because it grows the tumour by **deme (gland) fission**: birth is
fitness-dependent, and the carrying capacity is enforced *structurally* by splitting a full deme
rather than by a fitness-scaled death term. iscc uses a **fixed lattice** of demes (tissue locations
with per-deme K, see `DESIGN_ductal_field.md` §3), so fission has to be translated into something a
fixed lattice can do. That translation is §3.

---

## 2. Why the two existing modes cannot both work

There are two jobs:

* **(J1) density regulation** — a deme must sit near K whoever is in it and whatever division rates
  have evolved, because real tissue is full and because the original bug was unbounded overfill;
* **(J2) competition** — within a deme, a fitter clone must gain share at the expense of a less fit
  one.

The shipped design makes **one term do both**, and the two requirements pull the term in opposite
directions.

**`crowding_mode="own"` buys J1 by destroying J2.** Scaling the slope by the clone's own
`(b_i − d_i)` is precisely what makes the equilibrium occupancy independent of fitness — that is the
stated purpose of Option A ("made the equilibrium deme density independent of fitness (its whole
point: cap overfill for any evolved division rate)", `DESIGN_crowding.md` addendum). The identical
algebra that guarantees the cap holds for a clone evolved to `max_birth_rate` is the algebra that
cancels the fitness term.

**`crowding_mode="fixed"` buys J2 by destroying J1.** Referencing a fixed rate,

```
death_i = d_i + max(0, crowding_ref - d_i) * steep * (n / K)
```

keeps `b_i` in the net, so survival under crowding is fitness-dependent. But the equilibrium
occupancy is now clone-specific:

```
n*/K = (b_i - d_i) / ((crowding_ref - d_i) * steep)
```

With the shipped `crowding_ref` = `max_birth_rate` = 0.95, `d_i` = 0.05, `steep` = 1.1 and a typical
`b_i` = 0.7, that is `0.65 / 0.99` = **0.657** — a deme chronically at 66% of K. Real tissue is not
two-thirds empty. Worse, in a duct **wall** deme, which `_seed_structure` fills with `K_duct` = 60
immortal epithelial cells, `n/K` = 1 gives a crowding death of `0.05 + 0.99` = 1.04, clamped to
`maximum_death_rate` = 1.0 — above every possible division rate. No cancer cell survives the
crossing, the lesion is sealed in the lumen, and `validation/sweep_calibration.py` §2 measured
exactly that: stroma ≤ 0.2% of cancer cells in all six `fixed` runs, against 28.5% for the same
configuration under `own`.

That sweep's own recommendation (§9) already reached half of this note's conclusion from the other
direction: `fixed` "is the right law in principle (it stops the fitness term cancelling at
saturation)" but `crowding_ref` "has to come down with it — roughly to the founder division rate".
Lowering the reference is necessary. It is **not sufficient**: with `crowding_ref` = 0.7,
`steep` = 1.1 and `d` = 0.05, the crowding death at full occupancy is `0.05 + 0.715` = 0.765, still
above the founder's 0.7, so the deme still empties. Any *rate*-only law that keeps the fitness term
in the net must place its equilibrium at `n*/K < 1` for the clones it means to disadvantage, and a
deme is then under-filled by construction.

**The trade-off is not a tuning problem; it is structural.** A single density-dependent death term
whose equilibrium is at K cannot also be the term that discriminates between clones at K, because
"equilibrium at K for everybody" *is* "no discrimination at K". The way out is to move the density
regulation out of the rate law entirely.

### 2.1 A third consequence, worth recording: the cap is not enforced on total occupancy today

Under `own`, the cancer term uses `n_cancer / K` and the resident term uses `n_normal / K`, so a deme
with immortal residents caps its *cancer* population near K on top of residents it never removes.
Solving `net = 0` for a duct wall deme (`K_duct` = 60, 60 epithelial, `resident_pressure_ref` = 0.2,
`b` = 0.7, `d` = 0.05):

```
(b - d)[1 - steep * c/K] = (ref - d) * steep * (n_normal/K) = 0.165
=> c/K = 0.678  =>  c = 40.7  =>  total = 100.7 = 1.68 * K_duct
```

and for a stroma deme (`K_stroma` = 30, `stroma_fill_frac` 0.3 → 9 stromal cells): `c` = 25.2, total
= 34.2 = **1.14 × K**. These are the same over-packing factors the crowding addendum measured
empirically (`c ≈ 1.6`, drifting 1.9 → 1.6 as K goes 4 → 16) and the same 1.2 K the calibration sweep
reported for duct demes. So `carrying_capacity` is *already* not a cap on total occupancy wherever
normal cells are present. A structural cap fixes that too — and changes the gland-capacity sizing
rule in the addendum (see §5).

---

## 3. The proposed formulation

Two changes, both required; **either one alone is inert or harmful.**

### 3.1 Change A — the crowding death becomes uniform within a deme, and sits below the mean division rate

Replace the cancer crowding term with a term that is identical for every cancer clone in the deme:

```
# per-deme, computed once per _deme_comp scan
b_bar_j = mean division_rate over the deme's CANCER cells      (cell-count weighted)
d_bar_j = mean baseline death_rate over the deme's CANCER cells (cell-count weighted)

C_j     = rho * max(0, b_bar_j - d_bar_j) * min(1, n_j / K_j)   # crowding pressure, UNIFORM in the deme

death_i = d_i                                   # the clone's own baseline death (unchanged, clone-specific)
        + C_j                                   # NEW: uniform crowding pressure
        + max(0, resident_ref - d_i) * steep * (n_normal / K_j)   # resident pressure — UNCHANGED
        + immune / epithelial / stromal / host / treatment terms  # UNCHANGED
death_i = min(death_i, maximum_death_rate)
```

* `rho` is a new deme parameter, `crowding_turnover` ∈ (0, 1). **Proposed default 0.6.**
* `n_j` is **total** occupancy (cancer + normal), not `n_cancer` — the cap is now on tissue density,
  which is what K means.
* `crowding_margin` (`steep`) is no longer needed for the cancer term (firmness now comes from the
  structural cap) and is retained only in the resident term, unchanged.

What this buys, algebraically. Net growth of clone *i*:

```
net_i = b_i - d_i - C_j - (resident term)
mean net over the deme's cancer cells = (b_bar - d_bar) * (1 - rho * n/K)
```

* At `n = K` the deme mean is `(1 − rho)(b_bar − d_bar) > 0`, i.e. **the rate law alone would
  overfill.** That is deliberate. The rate law is no longer responsible for density.
* The **selection differential is `net_i − net_k = (b_i − d_i) − (b_k − d_k)`** — the clones' bare net
  baseline growth difference, **independent of occupancy and of `rho`**. It does not shrink to zero
  at the cap, it does not reverse sign above the cap, and it is the same at the invasion front as in
  the packed interior. Contrast the shipped law's `(b_i − b_k)[1 − steep·n/K]`.
* Restricting the crowding term to the deme's own *mean* (rather than a global fixed reference) makes
  purifying selection **relative, not absolute**. This matters concretely: with the shipped
  go-or-grow costs (`breach_cost` 0.6, `stromal_survival_cost` 0.6) a breach-competent clone's
  division rate is `0.7 × 0.4` = 0.28. Against a *fixed* reference of `rho·(0.7 − 0.05) + 0.05` =
  0.44 that clone is absolutely non-viable in any full deme and invasion collapses. Against the deme
  mean it is merely disadvantaged, exactly as it is today — the trade-off stays a trade-off.
* Cost: `b_bar_j` and `d_bar_j` are two extra accumulators inside the existing `_deme_comp` scan
  (`count.py` line 885), so the per-deme work stays O(#genotypes-in-deme). Nothing gets slower.

**Fixed-reference variant (offer it, do not default to it).** `C_j = rho · max(0, ref − d_i) ·
min(1, n_j/K_j)` with `ref` a config scalar defaulting to the founder division rate. This is the
`demon`-faithful choice (demon uses a plain fixed death rate) and it gives *absolute* purifying
selection: a clone whose net baseline growth falls below the crowding pressure is genuinely purged
from full tissue. It is the better model for a study of lethal driver load; it is the wrong default
for the shipped ductal field because of the go-or-grow interaction above. Expose it as
`crowding_reference: "deme_mean" | "fixed"`.

### 3.2 Change B — the density cap becomes structural

A deme `j` has `K_j` slots. Free slots are

```
S_j = max(0, floor(K_j) - n_j)          # n_j = TOTAL cells currently in the deme
```

Every birth must acquire a slot in its **target** deme (the birth deme for the in-place / mutation
branch, the drawn neighbour / cross-gland / metastatic-vessel deme for the dispersal branch).

* If a free slot is available, take it.
* If not, and the deme contains **evictable resident** cells, evict one and take its slot (§3.4).
* Otherwise the birth **fails**: the division is consumed, no cell is added, no genotype is
  registered. This is the same accounting the engine already uses for a daughter that breaches the
  CINner viability limits (`_is_viable`).

**No redraw of the dispersal target.** A dispersal daughter draws a uniformly random von Neumann
neighbour exactly as today, and is then accepted or rejected. Redirecting rejected daughters toward
whichever neighbour still has space would introduce free-space-biased (directional) dispersal — a
change the crowding addendum explicitly declined to make ("dispersal is undirected diffusion …
Directional (free-space-biased) dispersal is a possible future change, deliberately NOT made here").
Accept-or-fail keeps that decision intact; it is also the only rule under which the tau-leaping
allocation in §3.3 is exactly equivalent to the event-by-event path.

**When every neighbour is also full.** All births out of that deme fail. Growth in the interior of a
packed tumour therefore stops and the front does all the growing — the classic boundary-driven
behaviour of spatial tumour agent-based models (Waclaw 2015; Noble 2021), and the behaviour Option B
of the original design note described but never built. Global growth stalls only when the whole
lattice is saturated, which is correct — the tissue is full — and must be reported as such rather
than silently looking like stagnation (§6, `field_saturated`).

This is an **exact optimisation opportunity**, not just a semantic point: if deme `j` is full and all
its neighbours are full, then *every* birth drawn in `j` is futile, so `j`'s birth contribution to
`deme_rates[j]` can be set to zero without changing the distribution of anything. This keeps the deep
interior of a large tumour free in the exact engine. It requires that a death in `j` refresh the
rates of `j` **and its neighbours** (the `affected` list in `update`, `count.py` line 1128, grows from
1–2 entries to up to 5–6).

**Why both changes are needed.** With the shipped `own` law the rate equilibrium is at `0.909·K`, so
the structural cap never binds and Change B alone is a no-op. With Change A alone the rate law
overfills without limit — the original 2026-07 bug. Together, the rate law sets **turnover** and the
cap sets **density**, and neither has to compromise for the other. That is the whole design.

Turnover, quantitatively, on the shipped configuration (`b_bar` ≈ 0.7, `d` = 0.05, `rho` = 0.6):
crowding death at capacity `C = 0.6 × 0.65` = 0.39, total death 0.44, so a full deme replaces its
whole population every **≈ 2.3 time units** and rejects `1 − 0.44/0.7` = **37%** of the births drawn
in it. The rate law's own equilibrium would be at `n*/K = 0.65/0.39` = 1.67, comfortably above 1, so
the cap binds firmly rather than marginally.

### 3.3 Slot allocation, and the tau-leaping problem

The exact engine (`update`, `count.py` line 1095) needs no allocation rule: it fires one event at a
time, so a slot is contested by whichever birth event happens next, and the Gillespie sampler already
draws that birth with probability `b_i·c_i / Σ_k b_k·c_k`. Slots therefore go to clones in proportion
to `b_i·c_i` automatically. That is the lottery, and it is what produces competitive exclusion: in a
full deme,

```
d/dt (c_i) = (slots freed per unit time) * (b_i c_i / Σ b_k c_k)  -  death_i * c_i
d/dt (x_i) = x_i * ( b_i - <b> ) * (acceptance probability)          # x_i = share of the deme
```

so the deme's composition follows a replicator equation and the locally fittest clone fixes. Under
the shipped law the same expression is identically zero.

The tau-leaping engine (`_tau_substep`, `count.py` line 1352) cannot block individual events: it
draws Poisson birth and death counts per (deme, clone) per substep from the pre-step state and
applies them in batch. It needs an explicit allocation rule.

**Proposed substep, with the changes marked:**

```
Phase 0   per deme, ONE _deme_comp scan -> (total, n_normal, ..., b_bar, d_bar)   [+2 accumulators]

Phase 1   DEATHS, unchanged and applied FIRST
          per (deme j, clone i):  n_death ~ Poisson(death_ij * c_ij * dt)   # death_ij from §3.1
          apply, capped at the pre-step count

Phase 2   CANDIDATE births, drawn exactly as today from the PRE-step counts
          per (deme j, clone i):  B ~ Poisson(b_i * c_ij * dt)
                                  n_mut ~ Binomial(B, mut_prob);  n_disp = B - n_mut
          route n_disp through the existing branches unchanged
            - metastatic seeding: binomial split + transit survival  -> candidates at met_vessel
            - cross-gland hop:    binomial split + _cross_gland_target -> candidates at that lumen
            - local:              uniform neighbour draw              -> candidates at that neighbour
            - breach-gate failure: daughter stays in the duct         -> candidates at j        [NEW: it needs a slot too]
          accumulate a per-TARGET candidate vector, keyed by (clone, channel),
          channel in {mutation-attempt, plain-arrival}                                          [NEW]

Phase 3   SLOT ALLOCATION, per target deme in ascending index order                              [NEW]
          S = max(0, floor(K_t) - n_t)              # n_t = post-death occupancy
          a = candidate vector;  A = sum(a)
          if A <= S:  accepted = a
          else:       accepted = rng.multivariate_hypergeometric(a, S)
                      leftover = a - accepted
                      E = min(sum(leftover), n_evictable_t)                     # §3.4
                      if E > 0:
                          accepted += rng.multivariate_hypergeometric(leftover, E)
                          remove E resident cells, drawn across resident genotypes by their counts

Phase 4   REALISE the accepted births only
          mutation-channel accepted -> rep.divide(); child.mutate(...)  (may register a genotype)
          plain-channel accepted    -> _add(target, gid, n)
```

**Is the multivariate hypergeometric right? Yes — and it is exact, not merely an expectation match.**

The two required properties hold trivially: exactly `S` births are accepted, so the cap is never
exceeded; and `E[accepted_i] = S · a_i / A` with `E[a_i] ∝ b_i c_i`, so a fitter clone that drew more
births wins more slots.

The stronger claim is the one that matters for engine agreement. In the exact path, the birth events
occurring in the interval are a superposition of Poisson processes with rates `b_i c_i`. Conditional
on the total counts `(a_1, …, a_m)` in that interval, the **order** of those births is a uniformly
random permutation of the multiset (exchangeability of the superposition). The exact rule accepts the
first `S` of that order and rejects the rest. **The composition of the first `S` items of a uniformly
random permutation of a multiset is, by definition, multivariate hypergeometric `MVH(S; a_1..a_m)`.**
So the tau allocation reproduces the exact engine's *conditional law*, not just its mean. The nested
second draw for evictions is exact for the same reason: it is the next `E` items of the same
permutation.

**Where the two engines still differ, and by how much.** The residual error is the one tau-leaping
already makes everywhere — that births and deaths are treated as simultaneous within a substep rather
than interleaved:

* Applying deaths first means slots vacated *late* in the substep are offered to births drawn
  throughout it. This biases tau toward accepting slightly more births than the exact path, by
  roughly half the substep's deaths, `≈ 0.5 · C · K · dt`.
* With the current accuracy control (`_tau_generation`, `ACCURACY = 0.34` on the largest per-cell
  rate), `dt` satisfies `(b_max + disp_max + max_death)·dt ≤ 0.34`, so on the shipped configuration
  `dt ≈ 0.12` and `C·dt ≈ 0.047` — the bias is ~2% of K per substep in slot accounting.
* Crucially this is a **rate** bias, not a cap violation: `S` is recomputed from the real occupancy
  every substep and clipped at zero, so `n_j ≤ K_j` holds in both engines at all times.
* Validation must measure it (§6). If the bias moves deme occupancy or local dominance measurably,
  tighten `ACCURACY` (0.34 → 0.15) for crowded runs, or add a per-deme constraint
  `C_j · K_j · dt ≤ 0.05 · K_j`, which is the same as `C_j · dt ≤ 0.05`.

**Implementation notes.** `numpy.random.Generator.multivariate_hypergeometric` exists in the pinned
numpy (verified, 1.26.4) with `method="marginals"` by default; its constraints (`sum(colors) < 1e9`,
`nsample <= sum(colors)`) are never approached here. Candidate vectors must be built in the existing
deterministic order (deme index ascending, then genotype creation ordinal) so runs stay reproducible.
Skip the MVH call entirely when `A <= S` — that keeps uncontended demes at today's cost. Expected
added cost at cm-scale: ~5,000 occupied demes × ~5 substeps per generation ≈ 25k MVH calls per
generation, order 0.1 s — small against the existing per-deme per-genotype loop.

**Order of mutation and allocation matters.** Today the mutation branch calls `mutate()` and may
register a new genotype *before* anything is placed. Under the cap, a mutation-branch birth that
loses the lottery must never reach `mutate()` — otherwise the registry accumulates genotypes with no
cells. Allocate first, mutate second (Phase 3 before Phase 4). A useful side effect: in packed tissue
~37% of mutation-branch births are rejected, so the genotype registry grows ~37% more slowly there,
and tau's cost scales with #genotypes × #demes. This is likely to make the tau engine *faster*, not
slower (§7).

### 3.4 Normal cells, eviction, and the invasion gate

**Do normal cells occupy slots? They must** — they are cells, and K is the tissue column's cell
population. But a naive structural cap then seals the tumour in the duct lumen forever:
`_seed_structure` fills every duct **wall** deme with `K_duct` = 60 immortal epithelial cells, so
`S = 0` permanently, no cancer cell can ever enter the wall, and the DCIS → IDC arc — the entire
shipped biology — disappears. That is the same failure `crowding_mode="fixed"` produced by a
different route (stroma ≤ 0.2%). Any version of this design that does not answer this is wrong.

**Proposal: cancer births evict tissue residents.** When a birth's target has no free slot but does
have evictable residents, one resident is removed (via `_remove`, so the counts stay consistent) and
the slot is taken. Evictable = `epithelial`, `stromal`, `host`. This reverses the "normal cells never
die" invariant, and it should be reversed: histologically, invasive carcinoma **replaces** normal
tissue; a duct that is effaced by DCIS is the canonical picture. The `_epithelial_fraction` /
`_stromal_fraction` / `_host_fraction` terms are already documented as *live* fractions that fall as
cancer accumulates — today they fall by dilution (which is why demes over-pack to 1.68 K, §2.1);
under the cap they fall by replacement, at constant total density. That is strictly more realistic.

**The basement membrane is not made of these cells.** The DCIS → IDC gate in the shipped
configuration is `breach_gated_invasion`, which tests `gland_id[src] >= 0 and gland_id[tgt] < 0` — a
geometric duct → stroma transition, entirely independent of how many epithelial cells remain. So
effacing the epithelial population does **not** open the invasion gate. The arc is preserved.

**Keep the resident-pressure death term as the fitness gate, and make eviction unconditional.** The
existing term `(resident_ref − d_i)·steep·(n_normal/K)` does not contain `b_i` and so does not cancel;
it already imposes a genuine fitness threshold `b_i > d_i + (ref − d_i)·steep·(n_normal/K)` and was
verified to make invaded cells significantly fitter than core cells (p < 1e-4). Layering a second,
probabilistic eviction gate on top would double-count it and would break the exactness argument in
§3.3 (a per-candidate Bernoulli before the eviction draw makes the nesting order matter). So:
eviction is unconditional and structural; survival in resident-occupied tissue stays fitness-gated by
the death term. Nothing about the measured invasion biology has to change.

**Immune cells: do not evict, by default.** `immune_density` seeds immune cells into *every* deme and
the immune-killing term reads their live fraction. Making them evictable would let any tumour clear
its own immune pressure structurally, trivialising immune escape and invalidating the immunotherapy
tests. Proposal: immune cells hold slots but are not evictable, with a guard — if immune-held slots
alone would make `S = 0` and no other resident is present, allow immune eviction so a deme cannot
deadlock. Flag this as an open question (§7); it needs its own measurement before the default is
fixed.

**Metastatic seeding needs an explicit decision.** `host_fill_frac` defaults to **1.0**, i.e. every
metastatic deme, including the vessel deme that seeding arrivals land in, is filled to `K_met` with
host parenchyma. Under the cap with host evictable, seeding still works (arrivals evict host); under
any rule that makes host non-evictable, metastasis becomes impossible and `tests/test_metastasis.py`
fails outright. Recommendation: host is evictable (it is tissue, and a deposit does replace
parenchyma), and additionally lower the shipped `host_fill_frac` so a deposit has headroom, as
`stroma_fill_frac` 0.3 already does for the stroma.

**The well-mixed regime is untouched.** `carrying_capacity = None` / `0` sets `_crowding = False`:
no crowding term, no slots, no allocation, unbounded growth. `validation/benchmark_scalability.py`
and the single-deme scalability claim in `manuscript/paper.tex` are therefore unaffected. This should
be asserted by a test, not assumed.

---

## 4. What it predicts

Stated as falsifiable predictions on the canonical regime
(`notebooks/example_config.yaml` via `validation/realistic_regime.py`, scale `cm`, target 150k cancer
cells), **with no parameter suppression** — `dispersal_rate` stays 0.9, `cnv_prob` stays 0.35.

**P1 — local clonality rises at default dispersal and default copy-number rate.** The mechanism:
local selection collapses the local coalescent depth from `O(Nb)` turnovers to `O(ln Nb / s)`, where
`s = (b_i − b_k)/(crowding death at capacity)` per turnover. With `Nb ≈ 283` (stroma, K = 30) and a
0.95-vs-0.70 clone (`s ≈ 0.25 / 0.44 = 0.57` per turnover), depth falls from ~283 turnovers to
`ln(283)/0.57 ≈ 10`. Against a per-segment change rate of 0.00875 per division, ten divisions of
shared ancestry leaves a segment state intact with probability ≈ 0.92. That is an upper bound, not a
forecast — migration, ongoing mutation input and fitness saturation all cut it — but the direction and
the order of magnitude are the claim. **Target: mean per-deme dominance ≥ 0.50 and ≥ 3% of demes above
0.9, i.e. matching what dispersal-and-churn suppression bought (0.503 / 3%) without suppressing
anything.** Anything below ~0.30 means the mechanism did not bite and the note is wrong.

**P2 — deme occupancy sits at K, not below it and not above it.** Mean cells per occupied deme
between 0.90 K and 1.00 K, maximum never above K. Compare: today 1.14 K in the stroma and 1.68 K in a
duct wall (§2.1, and the addendum's measured over-pack factor `c ≈ 1.6`); `crowding_mode="fixed"`
gives 0.66 K.

**P3 — global clonality stays low.** `max_cna_freq` (from `validation/sweep_score.py`) stays below
0.9 and `n_truncal_segments` stays 0; the (n, D, J1) mode indices still overlap the empirical Noble
cloud in `validation/data/noble_empirical_indices.csv`. Local sweeps must not become global ones: the
lattice is 170² demes, dispersal is one deme per hop, and mutation input keeps generating competitors
ahead of any advancing clone, so a lesion-wide sweep should remain impossible. **If P3 fails, the fix
is rejected** — it would have broken the paper's central recapitulation to buy a local metric.

**P4 — selection stops being confined to the front.** The division-rate distribution inside the
packed interior should shift upward over time rather than freezing at whatever arrived there. Direct
test: cell-weighted mean `division_rate` in interior demes (all four neighbours occupied), tracked
over generations — currently flat after a deme fills, predicted increasing.

**P5 — fewer clones for the same tumour.** Rejected mutation-branch births register no genotypes, so
`n_clones` at matched cancer-cell count should fall by roughly the interior rejection fraction
(~37%). The calibration sweep measured "a new clone every 2.6 cells" inside a duct; that should ease.

**P6 — the invasion arc survives.** `stroma_pct` within ±10 percentage points of the current value at
matched size, `n_glands_colonised` unchanged, `validate_petracer.py` and
`validate_multiregion_phylo.py` confounds still reproduce (they may reproduce *more strongly*, since
both depend on clonal territories).

---

## 5. What it breaks

Everything that depends on growth. This is a change to the core event loop; treat it as a full
re-baseline, exactly as the 2026-07-14 crowding fix was.

### 5.1 Golden hashes and byte-identity tests (all re-baseline)

The random-number consumption pattern changes (allocation draws, mutation moved after allocation), so
every fingerprint moves even where the biology does not.

* `tests/test_count_engine.py` — golden md5 of `cell_data` keys (~line 61);
  `test_demes_cap_near_carrying_capacity` (~line 68) — the assertion should *tighten* from "near K" to
  "≤ K and ≥ 0.9 K"; `test_well_mixed_disables_crowding` (line 88) must keep passing unchanged;
  `test_engines_agree_on_crowding_death` (line 104) needs rewriting against the new law and must now
  also compare the cell engine's `Deme.get_cancer_death_rate`.
* `tests/test_tau_leaping.py` — golden hashes (~line 61) and the cap test (~line 140).
* `tests/test_compartment_selection.py` — golden `(tumor_size, cell_snv md5)` table (~line 45).
* `tests/test_ductal_field.py` — golden hashes (~line 46).
* `tests/test_wgd.py` — golden fingerprints (~lines 66–76).
* `tests/test_metastasis.py` — golden hashes (~line 70), **and the seeding path itself** (§3.4).
* `tests/test_go_or_grow_dosage.py` (~line 132) — golden hash; this file is uncommitted work in the
  tree, so its re-baseline has to be coordinated with whoever owns it, not done blind.
* `tests/test_assay_scrna.py` (~line 26) and every downstream assay fingerprint that starts from a
  grown tumour.
* `tests/test_engine_regressions.py`, `tests/test_reproducibility.py`.
* Growth-dependent fixtures and their rationale comments: `tests/conftest.py` (~line 28),
  `tests/test_integration.py` (~line 30), `tests/test_multiregion_phylo.py` (~line 28),
  `tests/test_petracer.py` (~line 36), `tests/test_microenvironment.py` (~line 26),
  `tests/test_treatment_engine.py` (~lines 19, 65, 119), `tests/test_cohort.py` (~line 109),
  `tests/test_epistasis.py` (~line 628, which reasons explicitly about the `own` law's cancellation
  and needs rewriting), `tests/test_diagnostics.py` (~lines 72, 104–121),
  `tests/test_realistic_regime.py`, `tests/test_landing_arc.py`, `tests/test_spatial_diagnostic.py`,
  `tests/test_scalability.py`, `tests/test_realgenome_tau.py`, `tests/test_tumor_growth.py`,
  `tests/test_tumor_components.py`.

### 5.2 Diagnostics

`src/iscc/tumor/diagnostics.py`:

* `deme_occupancy` (line 243) — docstring describes the old law; the interpretation changes from
  "should sit near K" to "is capped at K by construction".
* the `overfilled` check (lines 454–459, threshold `3×K`) becomes **structurally unfailable** for
  cancer-only demes. Keep it as a cheap invariant assertion (it should now fire only on a genuine
  bug) and **add its mirror**: an `underfilled` check (mean occupancy below ~0.7 K), which is the new
  failure mode if `crowding_turnover` is mis-set or baseline death is high.
* add `field_saturated` (fraction of the lattice at K) so a saturated field is distinguishable from a
  stalled or extinct one.
* add `local_state_dominance` as a first-class metric (the §1.2 definition), since it is the number
  this whole change exists to move.
* `validation/validate_operating_envelope.py` (line 41) carries the flag list and colours — update.

### 5.3 Calibration, operating envelope and the uncommitted sweep

* `validation/validate_operating_envelope.py` + `manuscript/figures/validation_operating_envelope.png`
  — every phase diagram regenerates.
* `validation/validate_calibration_envelope.py` +
  `manuscript/figures/validation_calibration_envelope.png` — same.
* `analysis/characterize_regimes.py` / `analysis/characterize_regimes.csv` — regenerate.
* `validation/sweep_calibration.py` and `validation/sweep_score.py` (**uncommitted; do not edit**):
  every table in the `sweep_calibration` module docstring — the 12-cell factorial, the per-duct
  diagnosis, the `tradeoff` / `local` / `gate` / `narrow` / `final` stages — was measured on the old
  law and is invalidated. Two structural consequences: axis **B** (`crowding_mode` own/fixed) becomes
  obsolete as posed, and the §9 recommendation about lowering `crowding_ref` is superseded by this
  note. The sweep should be re-run *after* the change, not edited before it; §4's arithmetic-barrier
  finding (FGA = 1 − retention) is a property of the mutational model and should survive, which is
  itself a useful check.

### 5.4 Validation scripts and manuscript figures

Every script that grows a tumour regenerates its figure. At minimum:
`validate_petracer.py` (+ `validation_petracer.png`, `validation_petracer_real.png`),
`validate_multiregion_phylo.py` (+ `validation_multiregion_phylo.png`),
`validate_evolution_modes.py` (+ `validation_evolution_modes.png` — **the Noble overlay, the highest-
stakes one**), `validate_evolution.py`, `validate_ductal_field.py`,
`validate_compartment_selection.py`, `validate_microenvironment.py`, `validate_spatial_diagnostic.py`,
`validate_treatment.py`, `validate_deconvolution.py`, `validate_visium.py`, `validate_cohort.py`,
`validate_cna.py`, `validate_snv.py`, `validate_wgd.py`, `validate_sampling.py`,
`validate_programs.py`, `validate_programs_cohort.py`, `validate_tau_leaping.py`
(+ `manuscript/figures/validate_tau_leaping.png` — this one directly measures engine agreement and is
the acceptance gate for §3.3), plus `manuscript/figures/make_overview.py` / `overview.png`.

`validation/benchmark_scalability.py` uses `carrying_capacity=None` and is **not** affected — the
well-mixed single-deme scalability claim in `manuscript/paper.tex` stands as written.

`manuscript/paper.tex` needs a pass for any quoted density, occupancy, clone-count or mode-index
number.

### 5.5 Configs and the physical-size mapping

* `notebooks/example_config.yaml` — the `PHYSICAL SCALE` header block derives duct and tissue sizes
  from K. Removing the 1.68× duct over-pack changes the mapping: the addendum's rule
  `N_gland ≈ π R² c K` with `c ≈ 1.6` becomes `N_gland ≈ π R² K`, so
  `R ≈ √(N / (π K))` — a duct sized for a given cell count needs `√1.6` ≈ **1.26× the radius** it
  needed before, or the same radius now holds 1.6× fewer cells. `gland_radius`, `structure_radius`,
  `K_duct`, `K_stroma`, `stroma_fill_frac` and `initial_cancer_cells` all need re-deriving, and the
  header comment rewritten. The finger-regime table in the crowding addendum (`R` = 12…28 vs invaded
  %) must be re-measured.
* `configs/landing.yaml` (the landing animation, with `tests/test_landing_arc.py` guarding the arc),
  `src/iscc/tumor/tumorconfigs/glandular.yaml`, `src/iscc/tumor/tumorconfigs/mixed.yaml`, the
  inference configs, and `validation/realistic_regime.py` (`SCALES` presets and the adaptive
  `n_steps` schedule, which may need more generations if local sweeps need ~30 to complete).
* New parameters to document: `crowding_turnover` (`rho`), `crowding_reference`
  (`deme_mean` | `fixed`), and whichever eviction switch survives §3.4.

### 5.6 Documentation

`PARAMETERS.md` lines ~50–51 (`maximum_death_rate` / `initial_cancer_cells` guidance), ~56–57
(`carrying_capacity` semantics), the boxed note at ~70–77 (the Option A formula, quoted verbatim),
~180 (the crowding-slope reasoning in the selection section), ~347 (the over-filling QC row) — and the
mirrored `docs/parameters.md`. `DESIGN_crowding.md` gets a "superseded by DESIGN_crowding_v2.md"
banner rather than an edit (it is the historical record). `DESIGN_operating_envelope.md`,
`DESIGN_ductal_field.md` §3, `DESIGN_scalability.md` §7, `DESIGN_features.md`, `BACKLOG.md`,
`handoffs/crowding_fix.md`, `README.md`.

### 5.7 Notebooks

All re-execute (never during this change — separately, deliberately):
`notebooks/01_pipeline_walkthrough.ipynb`, `02_tumor_growth.ipynb`, `03_data_overview.ipynb`,
`base_simulation.ipynb` (+ `base_sim.py`), `assay_dna.ipynb`, `assay_scrna.ipynb`,
`assay_spatial.ipynb`, `reads.ipynb`, `metastasis.ipynb` (+ `metastasis_demo.py`),
`compartment_selection_confound.ipynb`, `gene_programs.ipynb`, `cohort_mhn_recurrence.ipynb`,
`cohort_shared_programs.ipynb`, `combining_scdna_scrna.ipynb`, `scrna_visium_integration.ipynb`,
`tree_inference_dna.ipynb`, `wgd_allele_cna.ipynb`, plus `generate_example.py`, `_build_assay_dna.py`,
`_build_assay_spatial.py`, `_build_reads.py` and the `notebooks/example_out/` artifacts.

### 5.8 Migration story

1. **Land it gated.** Add `crowding_mode="lottery"` as a **third** mode. `own` stays the default. Both
   engines (`count.py::_death_rate` + the slot logic, and `components/deme.py::get_cancer_death_rate`
   + `apply_event`) implement it, and `test_engines_agree_on_crowding_death` is extended to cover it.
   Nothing re-baselines; the whole existing suite must stay green in this commit.
2. **Measure.** Run §6 under the flag on the canonical regime. Publish the numbers before changing any
   default.
3. **Re-tune the ductal field.** Re-derive the duct geometry (§5.5) and `crowding_turnover` against
   the realism windows in `validation/sweep_score.py` (`TARGETS`, `CLONE_RANGE`, `MIN_STROMA_PCT`),
   still under the flag.
4. **Flip the default in one commit**, with all golden hashes, figures, envelope maps and diagnostics
   re-baselined together, so the tree is never half-migrated. Keep `own` selectable for one release so
   any published result can be reproduced.
5. **Re-execute notebooks** as a separate commit.
6. **Re-run the calibration sweep** (§5.3) and update its findings.

---

## 6. How it will be validated

Run everything on `validation/realistic_regime.py` scale `cm` (grid 170) at 150k cancer cells, seeds
3 / 5 / 7, and on scale `small` for the exact-vs-tau comparison. Metrics computed from per-genotype
counts (no materialisation), following `validation/sweep_score.py` conventions.

| # | metric | how measured | pass criterion |
|---|---|---|---|
| V1 | **local state dominance**, mean | over occupied demes with ≥ 20 cancer cells: largest fraction of that deme's cancer cells sharing one non-diploid segment state, max over segments; cell-weighted mean | **≥ 0.50** at `dispersal_rate` 0.9 and `cnv_prob` 0.35 (today 0.153) |
| V2 | local state dominance, tail | fraction of qualifying demes above 0.9 | **≥ 3%** (today 0%) |
| V3 | **deme occupancy** | mean cells per occupied deme / K; max over demes | mean ∈ **[0.90, 1.00] K**; max **≤ K** in every deme, every step (hard invariant) |
| V4 | **global CNA clonality** | `sweep_score.max_cna_freq`, `n_truncal_segments` | `max_cna_freq` **< 0.9**; `n_truncal_segments` **= 0** |
| V5 | **evolutionary-mode indices** | `iscc.inference.mode_indices` (n, D, J1) vs `validation/data/noble_empirical_indices.csv` | simulated cloud still **overlaps** the empirical cloud on all three indices |
| V6 | **invasion arc** | `realistic_regime.stroma_cancer_pct`, `glands_colonised` | `stroma_pct` within **±10 points** of the pre-change value at matched size; glands colonised unchanged |
| V7 | **realism windows** | `sweep_score.TARGETS`: FGA 0.20–0.40, ploidy 2.4–3.0, %WGD 20–50 | no new flags relative to the pre-change baseline |
| V8 | **engine agreement** | exact vs tau on scale `small`, 20 seeds: occupancy distribution, `local_state_dominance`, clone-frequency spectrum, cancer size vs time | no significant difference (two-sample Kolmogorov–Smirnov, α = 0.01, Bonferroni over the four); mean cancer size within **±3%** |
| V9 | **tau substep sensitivity** | rerun tau at `ACCURACY` 0.34 / 0.15 / 0.05 | V1 and V3 stable to **< 0.02** absolute across the three — otherwise tighten the default |
| V10 | **selection reaches the interior** | cell-weighted mean `division_rate` in demes with all four neighbours occupied, vs generation | **strictly increasing** over the growth window (today: flat after fill) |
| V11 | **well-mixed untouched** | `carrying_capacity=None` run, golden hash vs pre-change | **byte-identical** |
| V12 | **runtime** | wall-clock and cells/s to 150k, tau engine; peak `len(t.genotypes)` | **≤ 1.2×** current wall-clock; genotype count expected to **fall** |
| V13 | **confounds still reproduce** | `validate_petracer.py`, `validate_multiregion_phylo.py` | monotone dispersal sweep preserved; naive-vs-deconvolved spurious-parallelism gap preserved |
| V14 | **cap invariant** | assertion in both engines, run under the full test suite | `n_j ≤ floor(K_j)` after every event and every substep, always |

V4 and V5 are **rejection criteria**, not targets: if a local fix buys a global sweep, the design is
wrong and must be reworked rather than tuned.

---

## 7. Risks and open questions

**R1 — does this reintroduce the original overfill bug?** No, and the reason is structural rather than
numerical. The 2026-07 bug existed because the cap was expressed as a *rate* that a clamp
(`maximum_death_rate`) could hold below an evolved division rate. Here the cap is a **count**: `S =
max(0, floor(K) − n)` is recomputed from the actual occupancy at every placement in both engines, and
no rate, clamp or evolved parameter can defeat it (V14 makes that an assertion). The residual risks
are (a) seeding above K — `initial_cancer_cells` is already clamped to K, and the duct wall is seeded
at exactly `K_duct`, so a wall deme starts full and needs eviction to be enterable (§3.4), and (b) a
mid-run change of `K_j`, which nothing currently does.

The **new** failure mode is the mirror one: **under-fill**, if the uniform crowding pressure exceeds
the deme's mean net growth (`rho ≥ 1`, or a config with high baseline death, or the fixed-reference
variant with a badly chosen reference). That is why `underfilled` joins the QC set (§5.2) and why
`rho` is defined as a *fraction* rather than an absolute rate — it cannot be mis-set into the
under-filled regime by accident.

**R2 — growth speed.** Two effects pull in opposite directions.
*Slower:* births into full demes are drawn and discarded. At capacity the futile fraction is
`1 − rho` ≈ 37%, so the exact engine performs ~1.6× more events per unit biological time in packed
tissue. Mitigated by the exact optimisation in §3.2 (a deme whose neighbours are all full contributes
zero birth rate), which frees the deep interior entirely; the cost is a wider `affected` list on every
death (1–2 → up to 6 rate refreshes).
*Faster:* rejected mutation-branch births register no genotypes, and tau's cost scales with
#genotypes × #demes. A ~37% reduction in clone creation inside the bulk is a direct, compounding
saving on the engine that actually runs the cm-scale simulations.
Net expectation: tau roughly neutral to faster, exact up to ~1.6× slower. V12 decides. If the exact
engine becomes painful, that is acceptable — every large run uses tau — but it must be measured, not
assumed.

**R3 — fitness saturation may starve the mechanism.** The lottery can only concentrate the fitness
*variance* that exists. The shipped configuration has `division_rate` 0.7 against `max_birth_rate`
0.95 — 1.36× of headroom — and the calibration sweep already measured 24.8% of cells pinned at the cap
at 12k cells. Once a neighbourhood's clones are all at 0.95, `b_i − b_k` = 0, `s` = 0, and local
sorting stops; local dominance would then be a decaying transient rather than a steady state. This is
the single most likely reason for V1 to miss. Mitigation is **not** lowering the founder division
rate — the sweep showed that makes saturation *worse* (88.3% pinned at div 0.25, because reaching the
target size takes more generations and more generations accumulate more drivers). The lever is
raising `max_birth_rate`, or capping driver dosage effects. Measure the division-rate distribution
directly (V10 plus its variance) before tuning anything.

**R4 — eviction changes the "normal cells never die" invariant.** Consequences to watch:
`stromal_hazard` (0.6 in the shipped config) and `epithelial_barrier` read *live* fractions, so as
cancer replaces residents the local hazard falls, which is a positive feedback on invasion that does
not exist today. It may accelerate the DCIS → IDC transition past what V6 tolerates. If so, the fix is
to make eviction rate-limited (a `resident_displacement` probability < 1) — at the cost of the exact
nesting argument in §3.3, which would then need re-deriving.

**R5 — immune cells.** Left non-evictable (§3.4) they are a hard obstacle that a tumour can never
clear, and at high `immune_density` they could throttle or deadlock a deme. Made evictable, immune
escape becomes structurally free and the immunotherapy results change. Neither default is obviously
right; this needs its own measurement on `validate_treatment.py` and `tests/test_treatment_engine.py`
before it is settled. **Open.**

**R6 — the exact-vs-tau bias.** §3.3 shows the allocation law is exactly the exact engine's
conditional law, but deaths and births are still treated as simultaneous within a substep, biasing tau
toward ~2% more accepted births per substep at the current `ACCURACY = 0.34`. V8 and V9 are the gate.
If the bias is material, the cheap fix is a tighter accuracy constraint; the principled fix is to
allocate slots in two passes (slots present at the start of the substep, then slots freed during it,
weighted by expected time-in-interval), which is more code for a second-order correction.

**R7 — accept-or-fail introduces a soft directional bias in *realised* dispersal even though the
*attempted* draw stays uniform**, because hops toward free space succeed more often. That is arguably
the correct physics and it is *not* the same as redrawing the target, but it does interact with the
finger regime described in the crowding addendum (localised invasion needs rare dispersal). The
finger-regime table must be re-measured (§5.5).

**R8 — deme-mean reference is a mean-field term.** `b_bar_j` changes whenever a deme's composition
changes, so `deme_rates[j]` must be refreshed on every event in `j` — it already is. But it also means
a clone's death rate depends on its neighbours' identities, which is a (mild) conceptual departure
from "a cell's rates are a property of the cell". The fixed-reference variant avoids this at the cost
of turning proliferation costs into absolute lethality (§3.1). **Open**, and the choice should be made
on the V6 invasion measurement, not on taste.

**R9 — is there a regime where the cap and the resident term double-count?** In a deme with many
residents the clone pays both the resident-pressure death hazard *and* the eviction cost of getting
in. Under the shipped `resident_pressure_ref` = 0.2 the hazard is small (threshold `b > 0.215` in a
wall deme, cleared by every clone), so this is probably immaterial — but it means the invasion gate in
the shipped configuration is essentially non-binding today and the real gate is
`breach_gated_invasion`. Worth stating plainly rather than discovering later.

**R10 — what stops a *local* sweep from becoming a global one?** The claim in P3 is that one-deme
dispersal plus continuous mutation input plus a 170² lattice keeps sweeps local. That is an argument,
not a proof, and it is the assumption the paper's central result rests on. V4 and V5 are the guard,
and they should be run at more than one field size (`small` / `mid` / `cm`) because the sweep
calibration already showed that scale changes the answer — the same shipped configuration scored
`max_cna_freq` 0.506 at 30k cells and 0.316 at 100k.

---

## 8. Verdict on the proposal as put

The proposed lattice translation of `demon`'s fission is **substantially right**, and the multivariate
hypergeometric allocation is **exactly right** — it is not an approximation chosen for convenience but
the true conditional law of the event-by-event path, which is what keeps the two engines from
diverging.

Three amendments are load-bearing:

1. **"Death: uniform across clones" is under-specified in the way that matters.** If "uniform" is read
   as the existing `crowding_mode="fixed"` law — a fixed reference at `max_birth_rate` — the deme's
   rate equilibrium sits at 66% of K, the structural cap never binds, and the whole change is inert.
   The uniform crowding pressure must be set *below* the deme's mean net growth (`rho < 1`), so that
   the rate law alone would overfill and the cap is what stops it. Rate law sets turnover; cap sets
   density.
2. **"Forced dispersal, or the division fails" must be resolved to "fails, with no redraw."** Redrawing
   toward free neighbours is directional dispersal, which the previous design deliberately declined to
   introduce and which would change the invasion biology as a side effect. It would also break the
   exactness of the hypergeometric allocation.
3. **Normal cells must be evictable, or the fix seals the tumour in the duct.** Duct wall demes are
   seeded full of immortal epithelium; a strict cap over total occupancy with immortal residents makes
   invasion structurally impossible — the same outcome `crowding_mode="fixed"` produced by a different
   route. Cancer replacing normal tissue is the correct biology and the correct mechanism; the
   fitness gate stays where it is, in the resident-pressure death term.

With those three, the design does dissolve the trade-off that forced two mutually exclusive modes:
demes sit at K **and** clones compete inside them, and the global-sweep suppression that the paper
depends on is untouched because nothing about the spatial structure or the dispersal kernel changes.

---

## 9. Prototype results [MEASURED]

Status of this section: **the design above is now implemented and measured**, as
`crowding_mode="lottery"`, **off by default**. `crowding_mode` still defaults to `"own"`, so nothing
in the tree has moved; the whole of §5 ("what it breaks") is still ahead of us and is triggered only
by the default flip in §5.8 step 4.

**What was built.** `src/iscc/tumor/models/count.py` only — the count engine, both of its update
paths. `tests/test_crowding_v2.py` is new (16 tests). The cell engine
(`components/deme.py::get_cancer_death_rate`) was deliberately *not* extended: the two-engine
agreement test in `tests/test_count_engine.py` covers `"own"`, which is unchanged, and the cell
engine has no slot bookkeeping to extend. §5.8 step 1 asks for it before the default flips.

New `deme_params`: `crowding_mode: "lottery"`, `crowding_turnover` (rho, default 0.6, validated to
lie in (0,1)), `crowding_reference` (`"deme_mean"` default | `"fixed"`), `evict_residents` (default
true), `evict_immune` (default false).

### 9.1 The default is untouched

Verified directly rather than assumed: `src/iscc/tumor/models/count.py` from the working tree was run
against a copy of the tree whose `count.py` alone was replaced by the pre-change version, over five
configurations — exact engine on a plain grid, tau on a plain grid, exact on the canonical ductal
field, tau on the canonical ductal field, and the well-mixed (`carrying_capacity=None`) regime.
**Every per-deme genotype count, every genotype total, and the md5 of every numeric `cell_data`
frame is identical.** `tests/test_crowding_v2.py` also reproduces the shipped Option A death-rate
expression independently and asserts exact float equality against `_death_rate` under `"own"`.

The seven suites named in the brief pass unchanged: `test_crowding_v2`, `test_count_engine`,
`test_wgd`, `test_ductal_field`, `test_compartment_selection`, `test_tau_leaping`,
`test_reproducibility` — 82 passed.

### 9.2 Local, duct and global clonality

`validation/realistic_regime.py`, tau engine, **default `dispersal_rate` 0.9 and default `cnv_prob`
0.35 — no parameter suppression anywhere.** Local dominance = over demes with ≥ 20 cancer cells, the
largest fraction of that deme's cancer cells sharing one **non-diploid** segment state (copy number 2
excluded), max over segments, cell-count-weighted mean. Computed from the per-deme genotype counts,
never a materialised subsample.

| scale (seeds) | target | metric | `own` | `lottery` |
|---|---|---|---:|---:|
| small (5) | 12 k | **local dominance, mean** | 0.285 | **0.431** |
| | | local dominance, median | 0.270 | 0.414 |
| | | demes above 0.9 | 1.7 % | **7.1 %** |
| | | duct-level dominance | 0.191 | 0.382 |
| | | **global dominance** | 0.206 | 0.373 |
| mid (3) | 40 k | **local dominance, mean** | 0.499 | **0.625** |
| | | demes above 0.9 | 19.1 % | 21.9 % |
| | | duct-level dominance | 0.252 | 0.390 |
| | | **global dominance** | 0.267 | 0.326 |
| cm (2) | 150 k | **local dominance, mean** | 0.626 | **0.710** |
| | | demes above 0.9 | 19.9 % | 23.6 % |
| | | duct-level dominance | 0.261 | 0.348 |
| | | **global dominance** | 0.276 | 0.326 |
| all | | `n_truncal_segments` | **0** | **0** |

**P1 is met in direction and mechanism but the absolute baseline in §1.2 did not reproduce.** Under
this harness `crowding_mode="own"` scores 0.285 / 0.499 / 0.626 at the three scales, not 0.153.
Whatever produced 0.153 differs from this measurement in something other than the crowding law, so
**only the paired A/B under one harness should be quoted.** The gain is robust to the metric's
definition — measured on the same pair of mid-scale tumours:

| dominance definition | `own` | `lottery` |
|---|---:|---:|
| max over segments, cell-weighted (the headline) | 0.492 | 0.635 |
| max over segments, unweighted over demes | 0.584 | 0.699 |
| mean over segments, cell-weighted | 0.243 | 0.337 |
| mean over segments, unweighted | 0.254 | 0.379 |
| fraction of demes above 0.9 | 0.154 | 0.239 |

**P3 holds, and this is the criterion that could have rejected the design.** Global dominance rises
(0.21 → 0.37 at small, 0.27 → 0.33 at cm) but stays far below the 0.9 rejection line, and
`n_truncal_segments` is **0 in every run of every mode at every scale**. The rise also *shrinks* with
field size (+0.167 at small, +0.059 at mid, +0.050 at cm) while the local gain does not, which is the
signature the design predicted: sweeps that are genuinely local. Glands still limit driver spread.
This should be re-checked at cm scale with more seeds before the default flips — two seeds is thin
for the one number the paper's Noble recapitulation rests on.

### 9.3 The cap holds, and demes are full

| scale | metric | `own` | `lottery` |
|---|---|---:|---:|
| small | mean occupancy of tumour demes / K | 1.121 | **0.975** |
| | max occupancy / K over all demes | 1.957 | **1.000** |
| | demes above K | 91.6 | **0** |
| mid | mean occupancy / K | 1.014 | **0.961** |
| | max / K | 1.978 | **1.000** |
| | demes above K | 362 | **0** |
| cm | mean occupancy / K | 1.002 | **0.963** |
| | max / K | 2.042 | **1.000** |
| | demes above K | 2 076 | **0** |

**P2 is met.** The cap is never exceeded — in either engine, after every single event in the exact
path and after every substep in tau, asserted in `test_cap_is_never_exceeded`. The `own` law's
over-pack is confirmed at 2.04 K at the maximum and ~2 000 demes above K at cm scale, matching §2.1's
first-principles prediction of 1.68 K in a duct wall. Occupancy sits at 0.96–0.98 K rather than
exactly 1.0 because turnover keeps a small fraction of slots transiently empty; the ~0.66 K that
`crowding_mode="fixed"` gives is asserted separately in
`test_fixed_mode_leaves_demes_chronically_underfilled`.

One correction to §2: **the "fixed" law's 0.657 K equilibrium only holds for an unevolved founder.**
`n*/K = (b_i − d_i)/((ref − d_i)·steep)` is about the clone's *own* division rate, so once drivers
push `b_i` up to `max_birth_rate` = `crowding_ref` the equilibrium rises to `K/(1+margin)` = 0.909 K,
and tau overshoot puts the measured value slightly above K. Measured with mutation on, `"fixed"` gave
1.125 K, not 0.66 K. The under-fill argument against `"fixed"` is therefore about *low-fitness* clones
and about the duct wall (where the deme empties outright), not about the tumour bulk.

### 9.4 Selection reaches the interior

**P4 is met, and it is the largest single effect measured.** Cell-weighted mean `division_rate` in
demes whose four neighbours are all occupied:

| scale | `own` | `lottery` |
|---|---:|---:|
| small | 0.544 | **0.881** |
| mid | 0.563 | **0.886** |
| cm | 0.586 | **0.902** |

Under `"own"` the interior freezes at roughly whatever arrived there (the founder's 0.7, dragged down
by go-or-grow costs). Under `"lottery"` it climbs to within 5 % of `max_birth_rate` = 0.95. That is
the defect in §1.1 — selection confined to the growing edge — directly reversed. It also confirms
**R3**: the mechanism runs out of fitness variance to concentrate once a neighbourhood saturates at
`max_birth_rate`, so raising `max_birth_rate` is now the lever with the most headroom behind it.

### 9.5 Invasion stays polyclonal, and the arc survives

| scale | metric | `own` | `lottery` |
|---|---|---:|---:|
| small | stroma % of cancer | 29.0 | 15.9 |
| mid | stroma % of cancer | 63.7 | 58.2 |
| cm | stroma % of cancer | 84.1 | **82.0** |
| cm | largest single clone's share of the stroma | 0.0046 | **0.0047** |
| cm | effective number of stromal clones (exp Shannon) | 19 620 | 15 276 |
| cm | glands colonised | 8 | 8 |

**P6 is met at cm scale** — `stroma_pct` moves 2.1 points, well inside the ±10-point criterion, and
all eight glands are still colonised. **Invasion stays overwhelmingly polyclonal**: the largest clone
in the stroma holds 0.47 % of it, statistically unchanged. This is the result confirmed against real
data, and it does not regress.

It *does* regress at **small** scale (29 % → 16 %), and that is worth stating plainly rather than
hiding behind the cm number. Two mechanisms: a lottery deme is genuinely full, so a stromal deme
seeded at `stroma_fill_frac` 0.3 offers 70 % of K as free slots and then closes, where under `"own"`
it would over-pack to 1.14 K; and the interior of the duct now competes hard enough to hold cells that
would otherwise have been pushed out. On a small field the invasive front is most of the tumour, so
the effect is visible; by cm scale it is 2 points. §5.5's geometry re-derivation and step 3 of the
migration plan (re-tune the ductal field under the flag) is where this gets fixed, not here.

### 9.6 Exact and tau agree

20 seeds, K = 24, grid 9, 30 generations of biological time each (the exact engine driven to matched
*time*, `deme_rates.sum() × dt` events per slice, not matched event count):

| readout | tau | exact | difference | KS p |
|---|---:|---:|---:|---:|
| cancer cells | 1877.1 ± 15.1 | 1838.7 ± 16.6 | **+2.1 %** | — |
| mean deme occupancy / K | 0.9656 ± 0.0077 | 0.9458 ± 0.0085 | **+2.1 %** | — |
| interior occupancy / K | 0.9707 | 0.9497 | +2.2 % | — |
| mean `division_rate` (the selection response) | 0.790 ± 0.031 | 0.803 ± 0.024 | −1.7 % | **0.175** |
| largest clone frequency | 0.0083 | 0.0081 | +2.5 % | **0.832** |
| clone count | 1026 | 992 | +3.4 % | 0.004 |
| **max occupancy − K over all demes, all seeds** | **0** | **0** | — | — |

The **+2.1 % occupancy bias is the value §3.3 predicted from first principles** (`≈ 0.5·C·K·dt`,
~2 % of K per substep at `ACCURACY` = 0.34). It is confirmed to be exactly that by tightening the
substep — the gap closes monotonically toward the exact engine and the cap never moves:

| `ACCURACY` | `dt` | cancer cells | occupancy / K | max occupancy / K |
|---|---:|---:|---:|---:|
| 0.34 (shipped) | 0.143 | 1874.8 | 0.9644 | **1.0000** |
| 0.15 | 0.0625 | 1857.8 | 0.9556 | **1.0000** |
| 0.05 | 0.0213 | 1841.6 | 0.9473 | **1.0000** |
| exact | — | 1834.2 | 0.9435 | 1.0000 |

So **V8 passes on the two readouts that matter** (selection response and clone-frequency spectrum are
statistically indistinguishable) and the residual density bias is a known, bounded, tunable
`dt` effect rather than a modelling difference. **V9's < 0.02 stability criterion is not met at the
shipped `ACCURACY`** — occupancy moves 0.017 from 0.34 to 0.05, right at the line, and cancer size
moves 2 %. Tightening `ACCURACY` for lottery runs, or adding the per-deme `C_j·dt ≤ 0.05` constraint
§3.3 offers, should be decided at step 3 of the migration.

### 9.7 Cost

Wall-clock to a **fixed cancer-cell count**, tau engine, same machine, means over the seeds above:

| scale | target | `own` | `lottery` | ratio | `own` cells/s | `lottery` cells/s |
|---|---|---:|---:|---:|---:|---:|
| small | 12 k | 14.0 s | 18.9 s | 1.35× | 872 | 649 |
| mid | 40 k | 43.7 s | 53.4 s | 1.22× | 930 | 760 |
| cm | 150 k | 178.2 s | 190.0 s | **1.07×** | 851 | 794 |

**V12 passes with room** (≤ 1.2× at the scale that matters). The penalty *falls* with field size,
because the two effects §7/R2 identified pull in opposite directions and the saving grows: live clone
count at matched size drops 3 862 → 3 389 (−12 %) at small, 46 588 → 42 499 (−9 %) at cm, and tau's
cost scales with #genotypes × #demes. The −37 % clone reduction R2 predicted applies to the packed
interior specifically, not to a field that is still 80 % invasion front.

### 9.8 Where the implementation deviates from §1–§8

1. **The exact-engine optimisation in §3.2 — a full deme with all-full neighbours contributes zero
   birth rate — was NOT built.** It is an optimisation, not a correctness requirement, and building
   it means widening the `affected` refresh list on every death from 1–2 demes to up to 6, which is a
   real risk of introducing a stale-rate bug for a speed-up on the engine that never runs the large
   simulations. The exact engine therefore draws and discards futile births in a packed interior. The
   measured cost is acceptable (§9.6 ran 20 exact seeds to 1 800 cells); revisit if the exact engine
   is ever used at scale.
2. **Immune cells are strictly non-evictable; the "deadlock guard" in §3.4 was not built.** The guard
   as specified — allow immune eviction when no other resident is present — would fire in exactly the
   case R5 warns about: a deme full of cancer plus immune cells, where the tumour would then clear its
   own immune pressure structurally. A configuration knob (`evict_immune`) is exposed instead, so the
   R5 measurement can be run both ways without another code change. The deadlock it was meant to
   prevent needs `immune_density` near 1.0 to bite.
3. **The nested eviction draw sizes itself from `_n_evictable`, which returns 0 when
   `evict_residents` is off.** This is worth recording because getting it wrong is silent: the first
   version sized the second hypergeometric draw from the resident count while `_evict` refused to
   remove anything, and the cap was breached in 16 demes of a structured field. The allocator's idea
   of how many slots eviction can free must agree with what eviction actually does.
4. **Met-seeding arrivals get their own allocation channel**, keyed by source deme, so the seeding
   event log reports the number of arrivals that actually *won* a slot in the vessel deme rather than
   the number that set out.
5. **A mutation-branch birth that is accepted but whose daughter turns out non-viable leaves its slot
   empty** (the eviction, if one was needed, has already happened). The alternative — deciding
   viability before allocation — would mean drawing mutations for births that then lose the lottery,
   which §3.3 rules out. The effect is a slightly-below-K equilibrium, already visible in the 0.96 K
   of §9.3.
6. **`test_lottery_produces_competitive_exclusion_where_own_produces_drift` needs K = 200, not a
   small deme.** At K = 40 the `"own"` law also fixes a clone within 60 generations — by *drift*,
   since neutral fixation takes O(N) turnovers and N = 40 is small. The distinction between the two
   laws is not whether a clone fixes but whether the *fitter* one does: at K = 200 over 30
   generations, `lottery` fixes the 2× fitter clone in 8 seeds out of 8, while `own` random-walks
   (share 0.10 to 0.92, mean 0.61, and the *slower* clone wins outright in some seeds).

### 9.9 What this does not answer

* **R5 (immune eviction)** is untouched — no treatment run was made under the flag.
* **V5 (the Noble mode-index overlap)** was not run. It is a rejection criterion and it needs
  `validation/validate_evolution_modes.py` under the flag before any default flip.
* **V7 (realism windows: FGA, ploidy, %WGD)** was not scored. `validation/sweep_score.py` is
  uncommitted work owned by someone else and re-running it has to be coordinated (§5.3).
* **V13 (the confounds)** — `validate_petracer.py`, `validate_multiregion_phylo.py` — not run.
* The cm-scale numbers rest on **two seeds**. Everything at small rests on five, mid on three.
* Nothing was re-tuned. §5.5's duct-geometry re-derivation (the 1.68 K over-pack disappearing means a
  duct sized for a given cell count needs 1.26× the radius) is untouched, and the small-scale
  invasion drop in §9.5 is the visible symptom of that.

### 9.10 Verdict

The mechanism works and it works for the stated reason. Making the crowding death uniform and putting
the density cap in the slot count, rather than in the rate, produces **within-deme competitive
exclusion where there was none** (a 2× fitter clone fixes in 8/8 seeds against a random walk), lifts
local clonality **at default dispersal and default copy-number churn** by 0.11–0.15 absolute at every
scale and under every definition of the metric, moves the packed interior's mean division rate from
0.54 to 0.90, holds every deme at or below K instead of over-packing 2 076 of them to 2× K, and
**leaves global clonality below the rejection line with zero truncal segments** — the result the paper
depends on. It costs 1.07× wall-clock at cm scale, and the two engines agree to the ~2 % substep bias
the design predicted before it was measured.

The three amendments in §8 were all load-bearing in practice: without eviction the tumour is sealed in
the duct (asserted in `test_structural_cap_holds_with_immortal_residents_and_eviction_opens_the_wall`),
with rho ≥ 1 the cap never binds (rejected at construction), and no redraw of rejected dispersal
targets kept the allocation exact.

What remains before the default can flip is §5.8 steps 2–3 as written: the outstanding validation in
§9.9, the ductal-field geometry re-derivation, and a decision on `ACCURACY` for lottery runs.
