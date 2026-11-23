# iscc parameters — defaults, valid ranges, and what to expect

`iscc` ships **sensible defaults** that sit inside its *operating envelope* — the region of parameter
space that produces realistic tumours. This document says what each knob does, its default, its
**valid range**, and **what happens outside it**. If you only change a few knobs and keep the rest at
their defaults, you will get a realistic multi-clone tumour.

Two safety nets back this up:

- The shipped configs (`notebooks/example_config.yaml`, `src/iscc/tumor/tumorconfigs/genotype.yaml`)
  are inside every valid range below.
- After growing, call **`tumor.diagnose()`** — a read-only quality-control check that flags degenerate
  ("crappy") tumours and tells you which knob to turn (see [§ Diagnosing a tumour](#diagnosing-a-tumour)).

The ranges below come from the operating-envelope characterization: `analysis/characterize_regimes.py`
(the sweep) → `manuscript/figures/validation_operating_envelope.png` (phase diagrams) →
and the manuscript "Operating regimes" section.

---

## The main knobs

Defaults are those in `notebooks/example_config.yaml`. Set them under the matching YAML block
(`cell_params.cancer`, `deme_params`, `spatial_params`, `selection_params`, `genome_params`).

!!! note "The defaults below are `example_config.yaml`'s DCIS→IDC tumour, at spatial-assay scale"
    A **deme is a ~50 µm 3-D column** of tissue (~4 cells in-plane), **not one cell**; its
    `carrying_capacity` is that column's whole cell population (`K_duct` 60 in a dense duct,
    `K_stroma` 30 in looser stroma). A `grid_size` of 170 is therefore ~**8.5 mm** of tissue (~10⁶
    cells) — larger than a Visium capture, so each assay samples a subset. The tumour is a **single
    clonal founder** growing a **breach-limited DCIS→IDC**: confined behind a high `epithelial_barrier`
    until a **rare, strong `breach`** mutation crosses the duct wall into a **permissive stroma** (low
    `stromal_hazard`), then invades. So `example_config.yaml` deliberately **turns ON** the
    compartment-selection axes (`epithelial_barrier` / `prop_breach`, `stromal_hazard` /
    `prop_stromal_survival`, `cross_gland_*`, `wgd_rate`) that are **off in the bare engine** — the
    tables below give `example_config`'s values, with the bare-engine off value noted where it differs.

### Mutations — `cell_params.cancer`
| Knob | Default | Valid range | Outside the range |
|---|---|---|---|
| `mutation_rate` | 0.3 | **[0, 1]**: the per-division probability of a mutation (an SNV event whose Poisson draws are all 0 leaves the genome unchanged) | **low** → monoclonal (no subclones); **high** → hypermutated mush (broken 1/f tail) |
| `n_snvs_per_allele` | 0.02 | ~0.1–3 | **low** → no SNV diversity; **high** → hypermutated mush |
| `snv_prob` / `cnv_prob` | 0.5 / 0.02 | relative weights (SNV vs CNA event) | all-SNV → no CNAs (inferCNV/clonealign demos degrade); all-CNA → no SNV phylogeny |

### Growth & survival — `cell_params.cancer` + `deme_params`
| Knob | Default | Valid range | Outside the range |
|---|---|---|---|
| `division_rate` | 0.7 | **> `death_rate`** | ≤ death → extinction |
| `death_rate` | 0.05 | **≪ `division_rate`** | ≥ division → extinction |
| `initial_cancer_cells` | 20 | ≥ 5 | 1 → founder extinction (a lone founder is prone to stochastic loss, and density-dependent crowding death raises that risk at small `carrying_capacity`) |
| `maximum_death_rate` | 1.0 | **> `max_birth_rate`** | clamps every death rate before the hazards, and is the default `crowding_d1`; at or below `max_birth_rate` construction fails (evolved clones would outrun the cap) |

### Spatial structure — `spatial_params` + `deme_params` + `cell_params.cancer`
| Knob | Default | Valid range | Outside the range |
|---|---|---|---|
| `carrying_capacity` | 30 | a real **per-deme cap** (cells/deme); `None` or `0` → **well-mixed** (no ceiling, unbounded growth) | too small (1–3) → a lone founder is prone to extinction (crowding ramps from occupancy 0) |
| `grid_size` × `carrying_capacity` | 170 × 30 | the tumour caps at ~`grid_size² × carrying_capacity`; size the grid **above** the target so it can spread (≳ 10⁴ demes for a 10⁵-cell tumour) | too small → no room to spread / no O₂ gradient / too few clones |
| `dispersal_rate` | 0.9 | **[0, 1]**: the per-division probability that a daughter disperses, independently of whether the division mutates | ≫ division → well-mixed, **no clonal territories** (silently breaks the PEtracer and multi-region benchmarks) |
| `structure_radius` / `n_structures` | 4 / 1 | glandular geometry (duct size / count); the cancer founds inside and spreads across the gland | — |
| `n_glands` | **8** | ductal-field substrate: number of small epithelial-ring glands scattered in stroma; `1` reproduces the single central ring | needs `structure_radius > 0`; too many for the grid → rejection sampling can't place them all |
| `gland_radius` | = `structure_radius` | ring radius of each gland (`structure_radius` 4 → 9 demes across, ~450 µm: lumen + wall) | too large × many glands → they don't fit / overlap |
| `min_gland_sep` | 14 (default `2·gland_radius + 2`) | minimum centre-to-centre spacing between glands | too small → glands touch (no stroma gap) |
| `K_duct` / `K_stroma` | 60 / 30 | per-compartment carrying capacity (a deme is a 3D column, so K is **moderate-to-large**, capturing depth — not a handful) | ST spot resolution is a **grid-spacing** constraint, not a K bound |
| `stroma_fill_frac` | 0.3 | baseline stromal occupancy as a fraction of `K_stroma` (stroma is less dense than epithelium but **not empty** — real fibroblast/immune/endothelial cells that the assays capture and that carry the stromal hazard; seed at a **moderate** 0.3–0.5, leaving headroom for an invasive mass) | `1.0` = the old dense fill (byte-identical); too sparse → the live stromal fraction (and its hazard) is negligible |
| `immune_density` | 0 (off) | static immune (T and NK) cells per stroma deme, as a fraction of `K_stroma`; in a field without glands, per deme of `carrying_capacity`. They hold their slots (cancer cannot displace them), count toward crowding and the resident pressure, and kill cancer cells at `prob_kill` × their share of the deme × (1 − immune resistance). The paper's immune variant uses 0.2 (6 per stroma deme beside 9 fibroblasts) | too high → the immune share of a full deme bounds cancer growth, and a small founder can die out |
| `immune_duct_density` | 0 | the same per duct-wall deme, as a fraction of `K_duct` (intraepithelial lymphocytes); empty lumens get none. The paper's immune variant keeps immune cells in the stroma only | — |
| `immune_composition` | `{immune: 1}` (all T and NK cells) | the shares of the immune subtypes among each deme's immune cells, drawn per deme: `immune` (T and NK cells, the only ones that kill), `myeloid` (macrophages, monocytes, dendritic and mast cells), `bplasma` (B and plasma cells). The paper's immune variant uses 0.575, 0.315 and 0.11, the shares of Wu et al. 2021's tumours corrected by the default recovery | shares outside these subtypes, negative or not summing to 1 → an error |
| `immune_cell_params.prob_kill` | 0.1 | the kill rate per unit share of T and NK cells in a deme. The paper's immune variant uses 14 (8 when every immune cell was a T or NK cell), with immune resistance at 0.2% of genes and effect 4.0 (one hit protects 75%): growth slows by a third and immune resistance is selected | high with weak resistance (effect 1.1, one hit protects 9%) → growth stalls before resistance is selected |
| `cross_gland_kappa` | 0.06 (0.0 = off) | island (intraductal) dispersal weight: a fraction ~`κ/(1+κ)` of a gland-resident clone's dispersing daughters seed **another gland's lumen** (bypassing the wall → confined DCIS, no breach), rate `κ·dispersal_rate` | stroma cells never hop; `0` → no cross-gland spread |
| `cross_gland_lambda` | 30 (`None` = uniform) | distance kernel for cross-gland targeting (prob ∝ `exp(−d/λ)` over gland centres); `None` = uniform (zero extra parameter) | 2D-section distance is only a proxy for ductal-tree proximity |
| `epithelial_barrier` | 10.0 (0.0 = off in the bare engine) | compartment-selection hazard: the epithelial ring adds `epithelial_barrier · epithelial_fraction(deme) · (1 − breach)` to a cancer cell's death | needs `structure_radius > 0`; too high → the ring is impassable and cancer stays in the lumen (raise `breach` fitness or lower the barrier) |
| `stromal_hazard` | 0.6 (0.0 = off in the bare engine) | compartment-selection hazard: the stroma adds `stromal_hazard · stromal_fraction(deme) · (1 − stromal_survival)` to death | needs `structure_radius > 0`; the stromal analogue of `epithelial_barrier` |

!!! note "`carrying_capacity` is a real per-deme cap (demon's two-valued death step under a slot cap)"
    A deme has `floor(crowding_overfill · K)` slots; a birth that finds none fails, and contested slots
    go to clones in proportion to the births they drew. A cancer cell dies at its baseline
    `death_rate` while the deme holds at most `K` cells and at `crowding_d1` (default
    `maximum_death_rate`) above it, the same for every clone in the deme, so clones in a full deme keep
    competing through their division rates. `crowding_d1`, after the `maximum_death_rate` clamp, must
    exceed `max_birth_rate` or construction fails. Immortal residents add
    `(resident_pressure_ref − death_rate)·(1 + crowding_margin)·n_normal/K`. Set
    `carrying_capacity: None` (or `0`) to disable crowding entirely — the **well-mixed** regime used
    for single-deme, unbounded-growth benchmarks.

| Knob | Default | Valid range | Outside the range |
|---|---|---|---|
| `crowding_overfill` | 1.5 | ≥ 1 | the slot cap as a multiple of `K`; at 1 a deme never passes `K` and never meets `crowding_d1` |
| `crowding_d1` | `maximum_death_rate` | > `max_birth_rate` (after the clamp) | the death rate above `K`; lower raises at construction |

The paper regime (`validation/paper/regime.py`) runs `crowding_overfill` 1.5 and `crowding_d1` =
`maximum_death_rate` = 5, with demon's division rule (a division mutates with probability
`mutation_rate` and, independently, sends a daughter out with probability `dispersal_rate`), random
streams keyed by (seed, purpose, step), and origin confinement counted in generations. These are the
engine's only laws; a config that names a removed one (`crowding_mode`, `crowding_law`,
`dispersal_law` other than the above, or a `"steps"` confinement clock) raises.

### Selection — `selection_params`
| Knob | Default | Valid range | Outside the range |
|---|---|---|---|
| `prop_driver` | 0.04 | 0.05–0.3 | high × strong effect → selective sweep (monoclonal) |
| `driver_effects` | 1.1 | 1.0–2 | ≫ 1 at low mutation rate → monoclonal sweep |
| `segment_drivers` | unset | one `[oncogenes, suppressors]` pair per segment, each summing to at most the segment's genes | unset → drivers are drawn gene by gene with `prop_driver`; set → exactly these counts at random loci of each segment, and `prop_driver` is unused. `arm_genome` sets it from each arm's Cancer Gene Census content, so an arm's driver load does not change with the resolution. A pair that does not fit raises. |
| `prop_dispersal` / `dispersal_effects` | 0.0 / 1.1 | as above | strong → invasion dominates, structure washes out |
| `prop_treatment_resistance` / `prop_immune_resistance` (+ `treatment_resistant_effects` / `immune_resistant_effects`) | 0.1 (treatment) / 0.02 (immune); effects 1.1 | as above | resistance is meant to **emerge**, not be pre-seeded |
| `prop_breach` / `breach_effects` | 0.001 / 2.8 (prop 0.0 = off in the bare engine) | compartment-selection axis: sites that if mutated let a clone cross the epithelial ring (attenuate `epithelial_barrier`); heritable, sequenceable | pairs with `spatial_params.epithelial_barrier`; both must be > 0 to have any effect |
| `prop_stromal_survival` / `stromal_survival_effects` | 0.02 / 2.2 (prop 0.0 = off in the bare engine) | compartment-selection axis: sites that if mutated let a clone survive the stroma (attenuate `stromal_hazard`) | pairs with `spatial_params.stromal_hazard` |
| `breach_cost` / `stromal_survival_cost` / `met_survival_cost` / `treatment_resistance_cost` | 0.0 (off) | 0.0–1.0 | **R15 go-or-grow trade-off** (compartment-context fitness): each dissemination/niche trait costs this fraction of `division_rate` per unit of the trait, applied **everywhere**, while its benefit (attenuating its compartment hazard) is compartment-gated. So a trait is net-favoured **only in its niche** and selected against elsewhere — this is what stops one super-clone from hitchhiking every trait into every compartment (proliferation stays in the DCIS, stromal in the IDC, met-survival in the deposit, resistance under chemo). | too high suppresses the trait even in its niche (e.g. a high `met_survival_cost` starves the metastasis; a high `treatment_resistance_cost` keeps resistance a rare pre-chemo subclone, which is usually what you want) |

#### Metastasis & drug-tolerance axes — `selection_params`

| Knob | Default | Valid range | Outside the range |
|---|---|---|---|
| `prop_met_survival` / `met_survival_effects` | 0.0 (off) / 1.1 | as the other trait axes | the dissemination axis for the metastatic compartment; attenuates the deposit's `met_hazard`. Pairs with `met_survival_cost` — a high cost starves the deposit |
| `prop_drug_tolerance` / `drug_tolerance_effects` / `drug_tolerance_cost` | 0.0 (off) / 1.1 / 0.0 | 0–1 / ≥1 / 0–1 | a **non-genetic persister** axis, distinct from `treatment_resistance`. **Currently NOT usable for a residual-disease floor** — the trait is a division discount, and a discount cannot hold a population at a floor under a proliferation-scaled kill (that needs dormancy, i.e. a division rate of ~0). Left in, default-off, until dormancy exists. |

!!! warning "`drug_tolerance_cost` is charged differently from every other cost"
    Every other `*_cost` is permanent, because the trait is permanent, and is applied in the
    division-cost factor. A persister state only exists while the drug does, so charging it
    permanently made the trait unusable — a cost high enough to hold the pool under drug also purged
    it to ~0.1% *before* treatment started. It is instead applied as a division multiplier **for the
    duration of the dose** (`GenotypeTumor._apply_treatment`).

#### Drug-induced resistance STATE — `selection_params` (optional; **all off by default**)

Resistance as a **carried cell state** rather than (only) a genomic allele, so that a CNA deleting
the triggering allele does not restore sensitivity. Every
knob defaults to its inert value; `Selection.resistance_state_on` is then `False` and division is
**byte-identical** to the no-state engine.

| Knob | Default | Valid range | Outside the range |
|---|---|---|---|
| `resistance_state_genetic` | `False` | bool | `True` → acquiring `n_mut_tr > 0` sets the state, which is then carried independently of the genome. This is the entry path every published result uses |
| `resistance_state_effect` | 0.0 | 0–1 | the protection a state cell gets, on the same scale as `treatment_resistance`. 0.99 ≈ takes essentially no drug |
| `resistance_state_cost` | 0.0 | 0–1 | division × `(1 − cost × state)`, charged **always**, on *and* off drug. It **stacks multiplicatively** with `treatment_resistance_cost`, so setting both to *c* charges an allele-anchored cell twice and an exited revertant not at all. To price the two routes equally, set `treatment_resistance_cost = 0` and charge the state only |
| `resistance_state_induction` | 0.0 | 0–1 | per-generation P(a sensitive cell is reprogrammed) = `induction × intensity`, where `intensity` is the dose the cell actually receives — a cell the drug cannot reach is not reprogrammed by it. **Never yet exercised at figure level** |
| `resistance_state_relax` | 0.0 | 0–1 | per-generation P(exit), i.e. τ = 1/`relax` generations. Applies **only** to cells with `n_mut_tr == 0` — a cell still carrying the allele keeps its genetic attractor and never relaxes. `0` is the permanent (τ→∞) limit |

!!! note "`induction` and `relax` are TAU-ENGINE ONLY"
    The reversible transitions live in `_tau_generation`. The *genetic* entry clause and the
    drug-protection term are in shared code and so are active under the exact engine too, but on an
    exact-engine run a non-zero `relax` is silently inert. Check `update_mode` before concluding that
    a result survived a finite exit rate.

!!! note "τ→∞ is a figure choice, not a requirement"
    Measured on the mode-IV rig: at `relax = 0.02` with fair single-charge pricing the relapse is
    97.7% resistant with the clean start intact, versus 100.0000% at `relax = 0`. The permanence buys
    the last ~2 percentage points.

#### Trait and selection model — `selection_params`

| Knob | Default | Valid range | Outside the range |
|---|---|---|---|
| `trait_source` | `"dosage"` | `"dosage"` \| `"mutation"` | what moves a dissemination/niche trait. `"dosage"` (historical, bit-for-bit) counts copy number *and* mutation; `"mutation"` divides out the wild-type dosage term so **only mutated copies move the trait**. The two agree on what a single heterozygous SNV is worth and differ only on what a copy-number change is worth |
| `selection_mode` | `"gene"` | `"gene"` \| `"arm"` | `"arm"` switches to the per-arm copy-number fitness of CINner and SISTEM, `∏_seg s_arm[seg] ** (cn[seg]/ploidy − 1)`, read from the per-segment copy numbers iscc already maintains. Dosage is relative to the genome-wide mean ploidy, so a whole-genome doubling leaves fitness unchanged; the `− 1` only rescales every genome by the same constant so the diploid genome has fitness 1 |
| `s_arm` | `None` (→ all 1.0) | length-`n_segments` array | per-arm selection coefficients. `s_arm[seg] > 1` → amplifying that arm is beneficial (oncogene-dominated); `< 1` → deleting it is (TSG-dominated). All 1.0 = arm selection off. Same scale as CINner's arm `s` and SISTEM's `1 + δ` |
| `rng` | `None` | — | **internal.** A `numpy` Generator for the layout draw; pass `layout_seed` instead if you want a reproducible gene layout |
| `segment_sizes` | `None` (→ uniform `segment_size`) | length-`n_segments` array | per-segment gene counts, for a genome with unequal arms. `None` gives every segment `segment_size` genes. Ploidy is the mean copy number weighted by these sizes, as CINner and SISTEM weight by length |

#### Viability limits — `selection_params`

CINner-style hard limits on what genome is compatible with life. A daughter breaching **any** of
them is not born: the division is consumed but yields no cell (`Selection.update_viability`;
enforced at birth in **both** engines — `GenotypeTumor._is_viable` and `Deme.apply_event`). The defaults
are loose, except `max_nullisomy`: the exact engine never reaches the other three at the shipped
defaults, so they only matter once you tighten them or push amplification/deletion hard.

| Knob | Default | Valid range | Outside the range |
|---|---|---|---|
| `max_ploidy` | 6 | 3–8 | tight (≲ 3) → caps CIN; near-diploid tumour, WGD impossible |
| `max_cn` | 12 | 6–20 | tight → caps focal amplification; interacts with `amp_prob` |
| `max_nullisomy` | 0 | 0–`n_segments` | the number of segments a viable cell may hold at zero copies. `0` (default, as CINner's `bound_homozygosity`): a segment is a chromosome arm or more, and a cell without any copy of one has lost essential genes. Above 0, dosage selection sweeps the loss of both copies of suppressor-rich segments (97% of the cancer cells of a paper-regime tumour at 2) |
| `max_mut_drivers` | 1000 | 1–(`prop_driver` × `n_genes`) + 4 | tight → caps the driver LOAD; `0` → any driver is fatal. **Config-set** (default 1000 = off) so it does not disturb e.g. hypermutation studies or the R14 MHN benchmarks; the cascade configs set it ~12 |

!!! note "`max_mut_drivers` = distinct driver *genes* PLUS activated dissemination programs (R15)"
    The load is `n_mutated_drivers` (distinct oncogene ∪ TSG genes carrying an SNV on **at least one
    copy** — a gene amplified to CN 4 counts once) **plus one per activated dissemination trait**
    (`breach` / `stromal_survival` / `met_survival` / `treatment_resistance`, counted by presence). So a
    "super-clone" that stacks every program pushes the load up and is culled under a realistic ceiling —
    the CINner mutational-load idea, extended to the compartment axes. Note this cannot *localize* traits
    (a metastatic-resistant cell legitimately carries ~8 drivers, so any ceiling that admits it also
    admits the super-clone) — that is the go-or-grow **cost**'s job; the ceiling only removes the
    biologically-absurd tail. Do not reach for `n_mut_onc` / `n_mut_tsg` as a substitute for the
    onc/TSG part — those are copy-weighted (CNVs scale them) and so are *not* a count of mutated drivers.

    The default of 1000 is far out of reach at the shipped defaults, so this limit moves no
    existing trajectory (`test_default_limits_are_a_noop_for_the_exact_engine`). It was inert in
    **both** engines until `Cell._recount_seg_drivers` began maintaining the count: the gate
    always read the limit correctly, but its input was never populated, so the check was
    `0 > 1000` forever. Now covered by `test_max_mut_drivers_limit_binds` and
    `test_n_mutated_drivers_counts_distinct_genes_not_mutated_copies`.

!!! note "`max_nullisomy` binds at the defaults"
    At the default 0, every daughter that loses the last copy of a segment is not born, under both
    engines. Real homozygous deletions are focal; `iscc` has no focal copy-number events, so a
    segment at zero copies would remove a whole chromosome arm from the DNA and the expression of
    the cell.

### Therapy — `Treatment` ops (`chemotherapy`, `targeted_therapy`, `immunotherapy`)

Scheduling and dosing knobs shared by every therapy op. **The two engines do not implement the drug
the same way** — see the warning below before comparing runs across engines.

| Knob | Default | Valid range | Outside the range |
|---|---|---|---|
| `start` / `duration` | 0 / `None` | steps | when dosing starts, and for how long. `None` = until the phase ends |
| `rounds` | 4 | ≥ 1 | number of dosing rounds |
| `dosage_decay` | 0.5 | 0–1 | how much the dose falls between rounds |
| `effectiveness` / `toxicity` | 0.9 / 0.1 | 0–1 | probability the drug acts on an **on-target** cell / on an off-target one. Off-target toxicity also kills the metastasis's host parenchyma, which progressively makes the deposit *safer* for the cancer |
| `sites` | `"both"` | `"both"` \| primary \| met | which compartment is dosed |
| `max_tumor_size` | 100000 | ≥ 1 | with `adaptive`, the burden threshold above which dosing resumes |
| `adaptive` | `False` | bool | `True` → dose only above `max_tumor_size`, the adaptive-therapy regimen |
| `kill_rate` | 1.5 | ≥ 0 | the drug's strength, read by the **count** engine |
| `kill_mode` | `"additive"` | `"additive"` \| `"proliferation"` | **the most consequential therapy knob.** `"additive"` gives every clone the same *absolute* extra death, so survival is decided by birth rate and a driver-saturated clone simply outruns the dose. `"proliferation"` scales the hazard with the clone's **own** division rate (`added_death = intensity × kill_rate × division_rate`), so every clone declines at its own rate whatever its fitness — and a cell at `division_rate = 0` takes exactly zero drug death. This is also the biology (chemotherapy kills replicating cells) |
| `rate_multiplier` | 2.0 | ≥ 1 | the **cell** engine's drug model: `death × rate_multiplier ** (1 − treatment_resistance)`. Unused by the count engine |
| `mutagenicity` | 1.0 (off) | ≥ 1 | multiplies `mutation_rate` while dosed. **This is not a mutation rate**: `mutation_rate` is a per-division probability capped at 1, so the realised boost saturates. It also moves SNVs and CNAs in lockstep, so it **cannot** shift the balance between acquiring resistance and reverting it (measured identical at 1.0 and 4.0) |
| `mutagenicity_target` | `"all"` | `"all"` \| `"snv"` | `"snv"` makes the drug a **point mutagen**: it scales `n_snvs_per_allele` instead, which feeds only the SNV branch, so acquiring a resistance SNV gets likelier while the CNA that deletes one stays flat. This is what decouples acquisition from reversion (acquisition:reversion 0.14 → 1.74 at ×20) |
| `mutagenicity_mode` | `"uniform"` | `"uniform"` \| `"dose"` | `"dose"` scales the mutator by `(1 − treatment_resistance)`, like every other drug effect. Note this is a **no-op for a de novo resistant clone**, which descends from a drug-exposed *sensitive* ancestor and inherits the raised rate through division — correct biology for a permanent repair-impairment mutator. It does protect a *pre-existing* resistant clone. Under `response="mechanistic"` it scales by the clone's exposure instead (its target-expression ratio, capped at 1) |
| `response` | `"label"` | `"label"` \| `"mechanistic"` | how the count engine decides whom the drug spares. `"label"` reads the resistance labels (`treatment_resistance`, `drug_tolerance`, the resistance state); the escape-mode studies are built on it. `"mechanistic"` ignores all three, and the persister cost with them: a clone is on target only if it **carries** the drug's `targets` (a copy with the mutation, or more copies than `amplification_threshold`), its dose scales with its expression of `target_genes`, and a broad drug with `kill_mode="proliferation"` kills in proportion to division rate. Count engine only (the per-cell engines raise), and Python API only: the schedule's `chemotherapy` op does not forward it |
| `target_genes` | `[]` | `(segment, position)` or flat gene index | mechanistic only. The dose is multiplied by the clone's mean expression of these genes over the founder's, so the founder takes the nominal dose. Expression is the dosage baseline × the program multiplier at the clone's *mean* activity — no per-cell noise, no niche |
| `expression_threshold` | 0 | ≥ 0 | mechanistic only, with `target_genes`: below this expression the drug does nothing to the clone — not even `toxicity` |
| `amplification_threshold` | `None` | copies | mechanistic only: more copies of the target's segment than this also count as carrying it, so the drug acts on an amplification until a deletion takes it back to the threshold |

!!! warning "Three different drug models live in this codebase"
    `kill_mode="additive"` and `kill_mode="proliferation"` are the count engine's two models; the
    **cell** engine instead uses `death × rate_multiplier ** (1 − tr)`
    (`Chemotherapy._apply`). They are not calibrated against one another, so a `kill_rate` tuned on
    one engine does not transfer, and a cross-engine comparison of therapy response is not
    like-for-like.

### Genome — `genome_params`
| Knob | Default | Valid range | Outside the range |
|---|---|---|---|
| `n_segments` × `segment_size` | `n_segments` required, `segment_size` 1000; the shipped configs use 12 × 500 (`notebooks/example_config.yaml`, the paper regime) and 10 × 1000 (the CLI's `genotype.yaml`), and `arm_genome` gives 39 arms over 6,000 genes | ≳ 100 genes | too few → trivial genome; assays degenerate, too few drivers |
| `segment_sizes` | unset (all `segment_size`) | one positive size per segment | sets unequal segments, e.g. arms of different lengths |
| `arms` | unset | one arm name per segment (`1p` … `22q`, the 39 modelled arms) | names the human arm each segment stands for; only the human genome layer reads it, growth never does |

#### Arm-homologous genome and the human genome layer — `iscc.data.human`

`arm_genome(total_genes=6000, weight="genes", driver_scale=1.0)` returns `genome_params` (one segment
per arm, with `arms`) and `selection_params` (`segment_drivers`). The number of growth genes is a
resolution traded between the detail of the dynamics and their cost; the layer works at any value.

| Knob | Default | Valid range | Outside the range |
|---|---|---|---|
| `total_genes` / `genes_per_segment` | 6000 / unset | ≳ 4 genes per arm | fewer → small arms cannot hold their drivers (capped, with the oncogene/suppressor split kept) |
| `weight` | `"genes"` | `"genes"` \| `"length"` \| `"uniform"` | how `total_genes` is shared between arms: by GRCh38-2020-A gene count, length, or equally |
| `driver_scale` | 1.0 | 0.1–1 | drivers per arm = this × the Census oncogenes and suppressors (297 + 293 over the 39 arms); 0.4 gives about the paper regime's driver load |
| `HumanGenomeLayer(seed=)` | the tumour's `layout_seed` | any int | fixes the germline: patients of one configuration share it unless given their own seed |
| `snp_window` | 1000 bp | 300–3000 | the 3' stretch of a gene a 3' library covers; only heterozygous SNPs there get allele counts |
| `snp_coverage` | 0.1 | 0.01–0.5 | probability a UMI of a gene covers a given SNP in its window (about read length / window). A preset, not fitted |
| `snp_concentration` | unset (binomial) | ≳ 10 | beta-binomial concentration of the haplotype split at a SNP (allelic-expression noise) |
| `Visium(kappa=)` on human genes | `HUMAN_VISIUM_KAPPA` = 300,000, the assay's default | ≥ 300,000 | by indirect inference: `estimate_visium` gives 82,165 on the in-tissue spots of 10x's V1_Breast_Cancer_Block_A_Section_1 (over the layer's genes), and a simulated paper-regime section reproduces that estimate within 1% from an input of 300,000 up (the direct value, used as input, over-disperses the section to 55,416, because the simulated tissue already carries the heterogeneity). Below 1,000 each spot's counts pile onto a few genes, so `write_visium` warns |

`layer.expression(tumor)` (`HumanExpression`) is the realistic path: each cell type takes its Human
Breast Cell Atlas baseline (epithelial = luminal epithelium, stromal = fibroblasts, immune = T and NK cells,
myeloid = macrophages, monocytes, dendritic and mast cells, bplasma = B and plasma cells, host = stroma), scaled by its relative RNA content; the cancer takes its `cell_of_origin`'s; named
programmes load on their MSigDB Hallmark sets (proliferation, EMT, hypoxia, interferon response; drug resistance on
xenobiotic metabolism as a proxy), the rest on random human genes, with the growth dictionary's loading
distribution scaled by `programme_scale` (0.5) under the bound `programme_max_log_fold` (5.0). The pair
is fitted on the cancer cells of 14 tumours of Wu et al. 2021 so that the fold changes of the Hallmark
genes between the top and bottom deciles of cells scored by each programme match
(`validation/data/fit_programme_strength.py`; 1.0 and 5.0 doubled them). The programme factor acts on
cancer cells only: a normal type's atlas baseline is already the mean of real cells of that type, their
usual programme activity included (`normal_programmes=True` applies it to normal cells too). On the
held-out 10x Block A section the two changes take the share of spot UMIs on the programme genes from
42% to 13%, against 11% in the real section. What the genotype and the niche do to a gene is read off
the growth genes. Pass it as `gene_lift` to `scRNA.run` (chunked, sparse counts) and `Visium.run`
(per-type pooling).
The scRNA presets add no dropout, which suits the human genes: a logistic dropout midpoint is in
counts per gene, so a midpoint of 1.5 UMIs deletes most of the human genes of a cell (410 genes per
cell at 1,913 UMIs, against 1,523 at 3,494 without it, on 3,019 cells of an arm-genome tumour at the
10x preset, measured when the layer emitted the 35,061 arm genes), and UMI counts are not
zero-inflated beyond the negative binomial (Svensson 2020). A run on human genes with
`dropout_mid > 0` warns. Over the human genes the 10x preset gives the zero fraction of PBMC3k within
0.02 at every mean.

The layer emits all 36,601 GRCh38-2020-A genes, as Cell Ranger's reference does. The 35,061 genes on
the 39 modelled arms each take the copy number per homolog of the growth gene covering its slice of
the arm and, with the plain `layer.lift`, that growth gene's expression. The other 1,540 (chrX 1,148,
chrY 111, chrM 13, and 268 on acrocentric short arms, centromeres and scaffolds) lie on no growth gene
(`layer.on_arm` False, `cover` and `segment` -1): under `layer.expression` they take their cell type's
atlas baseline and the cell's programmes at two copies, with no genotype or niche effect, and under the
plain `layer.lift` they are 0. They hold about a tenth of each type's atlas expression (chrM 5-6%).
Germline SNPs are the 1000 Genomes phase-3 common SNPs (allele frequency above 5%) inside the arm
genes, drawn heterozygous at 2·AF·(1 − AF) and phased exactly onto the homologs;
`fetch_snps()` downloads them (92 MB) and `ISCC_HUMAN_REF` points at the result.

Visium spots on the human genes take a per-gene platform factor (`HumanExpression(visium_platform=)`,
default the packaged `load_visium_platform_factor()` table): the sensitivity of Visium to each gene
relative to droplet scRNA-seq, fitted on the four tumours of Wu et al. 2021 measured by both assays
(7,101 genes, log-factor sd 0.62; lowest for the immediate-early genes that dissociation induces, such
as JUN, FOS and ATF3, and for nuclear lncRNAs such as MALAT1). It carries no part shared by the genes of
a cell type, so the composition of a spot is still the tissue's. On the held-out 10x Block A section it
raises the correlation of log gene shares from 0.846 to 0.851 and lowers the median absolute log2
ratio from 1.09 to 1.00 (`validation/data/check_visium_platform_factor.py`). `visium_platform=None`
draws spots from the scRNA-seq composition. scRNA-seq is unchanged.

### DNA assays — `bulkDNA`, `scDNA`

The coverage spread of a Dirichlet-multinomial depth draw is set by `kappa` and by the number of loci:
the excess squared coefficient of variation of a locus is about `n_loci / kappa`. Each default `kappa`
is therefore an `estimate_dna` fit stored with the number of loci it was fitted on (`kappa_loci`), and
the draw rescales it to the genome it runs on. A `kappa` passed on its own is used as given.

| Knob | Default | Valid range | Outside the range |
|---|---|---|---|
| `bulkDNA(kappa=, kappa_loci=)` | 121,241 on 20,000 loci (GIAB HG002 WGS) | `kappa_loci / kappa` ≲ 0.2 (the fits: 0.03 to 0.17) | coverage CV at two copies 0.48 at 30× on any genome size (HG002: 0.41). The old default, 2,000 used as given, gave 1.7 on 6,000 loci and left 39% of them below a tenth of the mean |
| `scDNA(kappa=, kappa_loci=)` | 202,799 on 5,787 loci (DLP+ OV2295) | `kappa_loci / kappa` ≲ 0.2 (the fits: 0.03 to 0.17) | at 9 reads per locus, a median within-cell CV of 0.40 and under 0.1% of loci without reads on 6,000 loci. The old default, 5 used as given, left 99% of loci without reads, so segment copy number could not be called |
| `nb_dispersion` (`depth_model="nb"`) | 0.165 bulk (HG002), 0.032 single-cell (DLP+) | ≲ 0.2 | the negative-binomial alternative, fitted on the same references |
| `scDNA(platform=)` | `None` | `"dlp"`, `"tapestri"` | the full fit on the reference of the platform, including its own depth per locus (DLP+ 830 reads per 500 kb bin, Tapestri 42.7 per amplicon). Tapestri reads the 584 amplicons of its fitted panel, and an explicit other `breadth` reads its own loci. DLP+ has no allele counts, so its allele layer and doublet rate stay at the single-cell priors (`ado_rate` 0.20, `doublet_rate` 0.02). An unknown name raises |

### Copy-number — `cell_params.cancer`
| Knob | Default | Valid range | Outside the range |
|---|---|---|---|
| `amp_prob` | 0.5 | 0.2–0.8 | high → copy-number-heavy genome (viability-capped) |
| `wgd_rate` | 0.0005 (0.0 = off in the bare engine) | 0.0–~0.05 | drives genome doubling; a separate per-division channel (leaves `snv_prob`/`cnv_prob` intact). Too high → most tumours WGD and pile against `max_ploidy` (doubled genomes are viability-capped); interacts directly with `max_ploidy`/`max_cn` |

`wgd_rate` is the per-(mutating-)division probability of a **whole-genome duplication**:
every copy on both homologs of every segment is doubled at once, so ploidy jumps 2→4 (then loss erodes it
back to a non-integer mean — the doubling+loss signature). A WGD daughter that breaches `max_ploidy`/`max_cn`
is **rejected at birth** like any other non-viable daughter, so `max_ploidy` (default 6) sets how much
post-WGD genome a lineage may carry. Off by default (0.0 → byte-identical to a run with no WGD). Sweeping it
lands tumour WGD prevalence in the real ~30–50 % range (`validation/validate_wgd.py`). Per-genotype ground
truth is surfaced as `cell_data["cell_wgd"]["is_wgd"]` (present only when `wgd_rate > 0`).

### Epistasis / dependency network (R14, optional; **off by default**) — `selection_params.epistasis_params`

Selection is **additive** by default (fitness reads the *count* of mutated drivers), so the true
event×event dependency network is empty. These knobs **plant a known network** — the ground truth the
cohort progression models (MHN, TreeMHN, CBN/H-CBN, REVOLVER) are scored against. With
`epistasis_params` absent, or `n_events: 0`, nothing is built and the engine is **bit-identical** to
the additive model. See `validation/validate_epistasis.py`.

An **event** is a disjoint module of `event_size` driver genes; it fires when ≥1 SNV lands anywhere in
the module, and is **monotone** (once acquired, never lost — the assumption MHN/CBN are defined under).
Modules rather than single genes because MHN/TreeMHN pool many tumours to fit **one** network, which
requires the same events to **recur across patients**; a single named gene out of ~10⁴ is hit in
almost no patient. `event_size: 1` recovers the single-gene case. `event_size` is one int for every
module, or a list of one int per event (`[2, 1, 1, 2, 2, 2, 4]`), so that events can be set to different
frequencies in a cohort; either way the modules come from one draw of the layout seed's epistasis
stream, cut in event order, so a single size and a list of that size draw the same modules.

Fitness gains `exp(Σᵢ βᵢxᵢ + Σᵢ<ⱼ Eᵢⱼxᵢxⱼ)` as a multiplier on the additive model.

| Knob | Default | Valid range | Outside the range |
|---|---|---|---|
| `n_events` | 0 (**off**) | 4–20 | 0 → additive engine, bit-identical; many → each event rarer, cohort needs to be larger |
| `event_size` | 20 | 1–~5% of the genome, per event | **the rate knob**, one int or a list of one per event: too small → events never fire (empty matrix); too large → every patient acquires every event (all-ones matrix, zero variance, nothing to correlate). Their sum must fit in the driver pool |
| `event_effect_mean` / `_sd` | 0.1 / 0.0 | 0–0.5 | the events' own marginal log-fitness effect (MHN's diagonal). 0 → events are marginally neutral and only `E` acts |
| `n_interactions` | `None` → use `network_sparsity` | 0–`n_events(n_events−1)/2` | 0 → empty `E` (the false-positive control) |
| `network_sparsity` | 0.2 | 0.05–0.5 | fraction of event pairs that interact, when `n_interactions` is unset |
| `network_topology` | `random` | `random` \| `hub` \| `chain` | `hub` = one master event wired to all; `chain` = a stepwise cascade |
| `interaction_strength` / `_sd` | 0.3 / 0.05 | 0.1–1.0 | **must fit under the fitness clamp — see the trap below** |
| `prop_synergy` | 0.5 | 0–1 | P(`E>0`); the rest are antagonistic |
| `mutual_exclusivity_strength` | 0.0 | 2–8 when used | magnitude of the strongly-negative (synthetic-lethal) edges |
| `n_exclusive_pairs` | 0 | ≤ pairs left after the interaction edges | how many such edges |
| `mutual_exclusivity_lethal` | `True` | `True` \| `False` | **`True` is required for exclusivity to mean anything** — see below. `False` leaves only the soft `E` effect |
| `lethal_death_rate` | 1.0 | ≥ `max_birth_rate`, ≤ `maximum_death_rate` | death rate of a synthetic-lethal genotype; above `maximum_death_rate` (which clamps every death rate) the tumour refuses to build |
| `planted` | `None` (drawn at random) | a dict, see below | states the whole network explicitly instead of drawing it |

**Planting an explicit network.** `planted` replaces every random draw above except the event modules
(still drawn from the layout seed, exactly as in the random mode), so every pair can be named before
any tumour is grown. Events are indexed from 0 (`E0`, `E1`, ...):

```yaml
epistasis_params:
  n_events: 6
  event_size: 10
  lethal_death_rate: 2.5            # needs deme_params.maximum_death_rate >= 2.5
  planted:
    effects: 0.05                   # each event's own log-fitness effect: a scalar or one per event (default 0)
    interactions: [[0, 1, 0.4]]     # pairwise log-additive terms E[i, j]
    exclusive: [[2, 3]]             # synthetic-lethal together (lethal_death_rate); no E term of their own
    constraints: [[4, 5]]           # 4 must precede 5 (a DAG), gated by dependency_params.gating_mode
dependency_params:
  gating_mode: accessibility
```

Nothing that is not planted acts: an event's own effect is 0 unless given, and a pair interacts only if
listed. The random-draw knobs (`event_effect_*`, `n_interactions`, `network_*`, `interaction_strength*`,
`prop_synergy`, `mutual_exclusivity_strength`, `n_exclusive_pairs`, and `n_constraints` / `dag_*` in
`dependency_params`) cannot be combined with `planted`, and a cycle, a self-pair, an out-of-range event,
a pair given twice, or an exclusive pair that is also an interaction is an error. The network rides in
`selection_params`, so the paper regime's `grow` takes it as `selection={"epistasis_params": ...,
"dependency_params": ...}`. `iscc.tumor.diagnostics.division_cap_share(tumor)` reports the share of
cancer cells whose division rate is at `max_birth_rate`, where any planted term stops acting (the clamp
trap below).

**Trap — a strongly negative `E` does NOT produce mutual exclusivity on its own.** It only suppresses
the *division* rate: a clone that has stopped dividing still dies only at its baseline rate (or at
`crowding_d1` in a deme above `K`), so it lingers and still reads as co-occurring — the opposite of the
signal you meant to plant. `mutual_exclusivity_lethal`
(default on) gives the combination `lethal_death_rate` so it is actually removed.

**And even then, exclusivity is not recoverable by MHN from a cross-sectional matrix in the regimes we
can simulate** — because that matrix is **patient-level** ("did this patient acquire event *i*
anywhere?") while `E` is **genotype-level** (it fires only in a cell carrying both). Measured: 59.5% of
patients read as co-occurring for MHN, but only 14.7% have any clone carrying both — **75% of MHN's
co-occurrences are two events sitting in different subclones, where `E` never fired.** That confound
(bulk co-occurrence ≠ same-cell co-occurrence) is real biology, and is exactly why TreeMHN exists.

**Trap — the fitness clamp silently eats `E`.** Division rate is `baseline × fitness`, capped at
`max_birth_rate`. With `driver_effects > 1` and many drivers the additive term already pins clones at
the cap, leaving **no room** for the interaction term: the planted network is then present in the
config and absent from the dynamics. Keep `log(max_birth_rate / division_rate)` comfortably larger
than the largest `Σ E` you plant (e.g. `division_rate: 0.2`, `max_birth_rate: 0.95` ⇒ ~1.56 of room),
and set `driver_effects: 1.0` if you want the network to be the *only* thing under selection.

#### Conjunctive / ordered constraints — `selection_params.dependency_params`

| Knob | Default | Valid range | Outside the range |
|---|---|---|---|
| `n_constraints` | 0 (**off**) | 1–`n_events` | 0 → no DAG, no ordering |
| `dag_depth` | 2 | 2–4 | the number of layers; edges only run from an earlier layer to a later one (acyclic by construction) |
| `dag_branching` | 2 | 1–3 | max parents per gated event; excess constraints are dropped, so the realised DAG may have fewer edges than `n_constraints` |
| `gating_mode` | `fitness` | `fitness` \| `accessibility` | **these are different generative stories — see below** |

**`gating_mode` is a modelling decision, not a tuning knob**, and it decides what a progression model
can recover:

- **`fitness`** — `B` arises freely but is *inert* until `A` is present; order emerges from selection.
  Softer and more biological, and closest to what MHN/TreeMHN assume (rates modulated, never zero).
- **`accessibility`** — `B` *cannot arise at all* until `A` is present; mutations in `B`'s module are
  vetoed. Order is imposed on the mutation process. This is the CBN/H-CBN story.

`validation/validate_epistasis.py` measures the consequence, and it is stark: under **accessibility**
gating the constraint is recovered perfectly (`B requires A` and the ordering, 1.00 in every network
draw, from true and reconstructed trees), while under **fitness** gating the same planted DAG leaves
essentially **no recoverable trace** (the conjunction holds in a small minority of lineages, and the
apparent "order" just reports which event is intrinsically faster). State which gating mode you used;
the two are not interchangeable.

#### The observable matters more than the cohort size

Pairwise `E` acts on **fitness** — how large the carrying clones grow — not on the **rate** at which
events arise (which is what MHN/CBN model). Two practical consequences when you design a benchmark
with `epistasis_params`:

- **Binary "event present" is saturated, and it is recurrent mutation that saturates it.** A favoured
  combination arises many times independently in one tumour, so it is already present at `E = 0`;
  raising `E` changes how much of the tumour its lineages occupy, which a presence call throws away.
  Sweep `min_freq` (a per-event **cancer-cell fraction** threshold, aggregated across clones — not a
  per-clone size filter) and check the column marginals: all-zero or all-one columns carry no
  information and no method can recover anything from them. `min_freq` has no default in
  `to_mhn_matrix` / `to_cbn_poset` / `patient_event_vector`; `iscc.integrations.DETECTION_LIMIT` (5%)
  is paper 1's. The tree exports take `detection`, the same limit on the collapsed clone tree (a node
  is kept when at least that share of the cancer cells descends through it), and
  `to_trait_matrix` / `to_trait_trees` read the dissemination traits (breach, stromal and metastatic
  survival) as events in the same shapes.
- **Tumour size is the binding constraint, not patient count.** A clone arising late has little time
  to expand, so in small tumours the frequency signal never develops. Adding patients does not fix
  this. If recovery looks flat, check the observable's marginals before concluding anything about the
  method.

### Gene programs / expression realism (optional; **off by default**) — `expression_params`

By default expression is the legacy model: genes are **independent** draws, a CNA adds ~1×baseline per
copy (**linear dosage**), the SNV effect **reuses the fitness knob**, and the `p`/`m` alleles are
**summed**. That is approximately the law the DNA↔RNA tools (clonealign, inferCNV, Numbat, cardelino)
*invert*, so benchmarking them against it is partly circular. `expression_params` replaces all four.
With it absent the engine is **unchanged**, and growth never reads it either way — programs are a
**readout**, never fitness. See `validation/validate_programs.py`.

```
exp_{g,a} = base_{type,g}/2 · exp(Σ_k z_k·loading_{k,g}) · dosage(CN_{g,a}; s_g) · snv(class_{g,a}) · niche_g
```

**Comparability:** the program dictionary, regulators and `s_g` are properties of the **genome**, so
they come from `layout_seed` (its own sub-stream) — two patients sharing a config get the **same
programs**, exactly as they already get the same oncogenes. Per-cell `z` and each SNV's class are
per-**run** draws off `seed`. Ground truth is surfaced on `tumor.program_truth` (+ `cell_program`,
`cell_exp_p`/`cell_exp_m`/`cell_rna_baf`).

!!! warning "The allele path uses a different expression SCALE"
    Legacy `get_exp` returns `base·(1+copies)` (wild-type diploid → **3×base**) while normal cells
    bypass it at **1×base**. With `expression_params` on, cancer and normal sit on **one** scale
    (wild-type diploid → `base` for both), which is what a malignant-vs-reference comparison needs.
    Only applies when the layer is on, so nothing existing re-baselines.

#### `cell_of_origin` — the cancer baseline
| Knob | Default | Valid range | Outside the range |
|---|---|---|---|
| `cell_of_origin` | unset | a normal type: `epithelial` \| `stromal` \| `immune` | unset → `base_{cancer,g}` is its own draw, unrelated to every normal type, so malignant cells separate from normal ones on baseline alone; set → cancer cells share that type's `base_{type,g}` and differ from it only by copy number, SNVs and programs. `cancer` or an unknown type raises. |

Growth and every layout draw are unchanged; only materialised cancer expression moves (and, under
targeted therapy, the target-expression check that reads it). For the ductal field the origin is
`epithelial`.

#### `program_params` — the dictionary (K × G `loading` matrix)
| Knob | Default | Valid range | Outside the range |
|---|---|---|---|
| `n_programs` (K) | 8 | 5–30 | too few → no co-expression structure to recover; too many → each is tiny/unidentifiable |
| `n_genes_per_program` | 30 | ~1–10% of the genome (int, or `[lo, hi]`) | too few → below a factor model's detection limit; too many → programs overlap into one blob |
| `program_overlap` | 0.1 | 0–0.5 | 0 → disjoint modules (unrealistically easy); high → entangled, tools cannot separate them |
| `loading_strength` | `{mean: 1.0, sd: 0.3}` | mean ~0.3–2 (log-fold) | ≪1 → programs vanish under Poisson noise; ≫2 → implausible fold-changes |
| `loading_sparsity` | 1.0 | 0–2 (σ of a mean-1 lognormal) | 0 → every gene loads equally (unreal); high → one marker gene carries the program |
| **`program_genomic_scatter`** | 1.0 | 0–1 | **the programs ⟂ CNAs knob.** 1 = scattered genome-wide (functional, realistic); **0 = a contiguous block that MIMICS a CNA** — the deliberate control for "can the tool tell a program from a copy-number segment?" |
| `program_signs` | `up` | `up` \| `bidirectional` | `bidirectional` gives each program up- AND down-genes (more realistic) |
| `seeded_programs` | `[proliferation, emt, hypoxia, drug_resistance, immune_evasion]` | any names | anchors programs so the phenotype/niche maps can address them **by name**; extras are unnamed background |
| `max_log_fold` | unset (unbounded) | ~3–7 (5 ≈ 150-fold) | unset → a gene's program log-fold `Σ_k z_k·loading_{k,g}` is unbounded, and with heavy-tailed loadings and strong drive one gene can take a cell's whole library (a warning is raised past a 10⁶-fold multiplier); set → it saturates as `c·tanh(x/c)`, unchanged for small log-folds and never beyond `c`. Not part of the dictionary: setting it leaves the gene–program map and loadings unchanged |

#### `activity_params` — the per-cell `z` sampler (shared with R12)
| Knob | Default | Valid range | Outside the range |
|---|---|---|---|
| `n_active_programs_per_cell` | all | 1–K | activity sparsity; ≥K → every program on in every cell |
| `activity_dist` | `lognormal` | `lognormal` \| `normal` \| `gamma` | `normal` allows negative activity (a program can be repressed) |
| `activity_mean` / `activity_sd` | 1.0 / 0.5 | mean ~0.5–2 | moment-matched to the chosen dist |
| `activity_noise` | 0.0 | 0–1 | **within-clone** spread. 0 → every cell of a clone is transcriptionally identical (clone == state, benchmark trivially easy); high → drowns the between-clone signal |
| `celltype_program_bias` | `{}` | per-type scalar or K-vector | baseline activity per cell type (normal vs cancer) |

#### Genotype → program coupling — `coupling_params` (the three routes of §3.1)
| Knob | Route | Default | Valid range | Outside the range |
|---|---|---|---|---|
| `phenotype_program_map` | **1** | `division_rate→proliferation`, `dispersal_rate→emt`, `treatment_resistance→drug_resistance`, `immune_resistance→immune_evasion` | any phenotype→program names | **the default coupling.** Reuses the existing role→phenotype map, so the chain is mutation → CINner fitness (incl. R14 epistasis) → phenotype → program → expression |
| `phenotype_program_strength` | **1** | 0.5 | 0–2 | **0 ⇒ route 1 off** (fitness and expression decoupled); high → fitter clones become trivially separable in RNA, a *non-dosage* clone leak that confounds clonealign/inferCNV and can make scDEF conflate "proliferation" with "clone" |
| `prop_program_regulator` | 2 | 0.0 | 0–0.1 | fraction of genes that shift `z` with **no** fitness change (R12 plasticity). Keep **sparse**: high → every clone is its own expression state |
| `program_bias_strength` | 2 | 0.5 | 0–2 | how hard a mutated regulator shifts its program (graded by VAF) |
| `n_programs_per_regulator` | 2 | 1 | 1–3 | how many programs one regulator touches |
| `niche_program_map` / `niche_program_strength` | 3 | none / 1.0 | field→program names | generalises the microenvironment layer: `{hypoxia: hypoxia}` makes the O₂ field drive the hypoxia **program** of cancer cells (a normal cell's programs are its type's). Composes with that layer's own gene-level modifier |

#### `dosage_params` — axis A (CNA → expression)
| Knob | Default | Valid range | Outside the range |
|---|---|---|---|
| `dosage_sensitivity_mean` / `_sd` | 0.7 / 0.25 | mean 0–1, clipped per gene | per-gene `s_g`: **1 = full linear dosage** (exactly the law the tools assume — circular); **0 = fully buffered** (copy number invisible in RNA, inferCNV/clonealign have nothing to find). Intermediate is the realistic case and the confounder that makes those benchmarks fair |
| `dosage_law` | `"haplotype"` | `"haplotype"` \| `"total"` | where `s_g` acts. `"haplotype"`: per haplotype, so a lost haplotype is silent whatever `s_g` (losses are never buffered) and a copy-neutral LOH lowers expression. `"total"`: on the total copy number, `1 + s_g·(min(C, sat)/2 − 1)`, so gains and losses are buffered alike, with the allelic split of the `"haplotype"` law. Fitted to TCGA breast tumours, `"total"` with mean 0.86 and sd 0.61 matches gains and losses together; `"haplotype"` cannot buffer losses (`validation/paper/supp_dosage.py`) |
| `dosage_saturation` | none | 4–12 (**total** CN) | CN beyond which expression stops rising (real amplicons saturate); absent → unbounded linear |
| `allele_specific` | `false` | bool | `true` → emit `cell_exp_p`/`cell_exp_m` + **`cell_rna_baf`** (the BAF in RNA that Numbat/CalicoST need). A deleted haplotype is silent regardless of `s_g` |

#### `snv_effect_params` — axis B (SNV → expression), **separate from the fitness `mut_effect`**
| Knob | Default | Valid range | Outside the range |
|---|---|---|---|
| `p_silent` / `p_missense` / `p_splice` / `p_lof` | 0.55 / 0.30 / 0.05 / 0.10 | normalised to 1 | class drawn per site (infinite-sites ⇒ one class per site, shared by every clone carrying it) |
| `nmd_strength` | 0.2 | 0–1 | expression **retained** on a LoF allele (nonsense-mediated decay). With a CNA loss of the other homolog this gives a biallelic **two-hit** TSG for free |
| `snv_expression_effect` | 0.5 | 0–2 | the splice-class expression shift. **Kept separate from the fitness effect**, so "advantageous" and "over-expressed" are not the same knob |

Silent and missense leave mRNA abundance alone (a missense changes protein *activity*, not
abundance — which is exactly why it is invisible to expression-based clone assignment).

### Microenvironment (optional; off by default) — `microenv_params`
| Knob | Default | Valid range | Outside the range |
|---|---|---|---|
| hypoxia `D` / `k` vs. tumour size | — | O₂ diffusion length **<** tumour radius | length ≫ tumour → uniform O₂, no gradient |
| `o2_source` | `uniform` | `uniform` \| `perfused` | `perfused` gives a hotter (more hypoxic) core than `uniform` |

### Spatiotemporal record (optional; off by default) — top-level config keys
Every run records when and where each cancer genotype arose (`tumor.genotype_births()`, written as
`genotype_births.csv`). These two keys additionally record the spatial state over time
(`tumor.spatial_snapshots()`, written as `spatial_history.csv.gz`); genotype engine only. Neither
draws a random number, so the tumour grows identically with the record on or off.

| Knob | Default | Valid range | Outside the range |
|---|---|---|---|
| `spatial_snapshot_every` | `0` (off) | `0`, or `k ≥ 1` trace snapshots between spatial snapshots | under tau-leaping a trace snapshot is one generation (× `snapshot_every`); under the exact engine it is one **event**, so a small `k` there stores a snapshot per birth or death — use a large `k` |
| `spatial_snapshot_budget` | `5000000` entries (~60 MB) | ≳ entries in one snapshot (≈ occupied demes × genotypes per deme) | when exceeded, every other snapshot is dropped and the interval doubles; a budget below one snapshot keeps only the first |

### Random streams and the confinement clock
Every per-run stream is keyed by `(seed, purpose, step)` through a `SeedSequence`, so different seeds
never share a stream. `spatial_params.origin_confinement.generations` counts generations at any `tau`
(the exact engine counts the Gillespie waiting time of each event).

### Materialisation memory
`make_cell_data` builds the per-cell tables, the program layer's expression multiplier included, chunk by chunk, so its working set does not grow with the number of cells beyond the tables it returns.
A further saving is chosen by the caller.

| Knob | Default | Valid range | Outside the range |
|---|---|---|---|
| `make_cell_data(layers=...)`, also on `Resection.slice` / `dissociate` | `None` (every table) | table names (`"cell_exp"`, `"cell_snv"`, `"cell_cnv"`, `"cell_rna_vaf"`, `"cell_snv_p"`, `"cell_snv_m"`, `"cell_exp_p"`, `"cell_exp_m"`, `"cell_rna_baf"`) and/or assay names standing for the tables that assay reads (`"visium"` and `"imaging"`: expression; `"scrna"`: expression, SNVs and RNA VAF; `"scdna"`: SNVs and copy number). The small per-cell tables are always built. Values are byte-identical to a full build | an unknown name raises |

### scRNA batches — `run_scrna_batches`, `run_cohort_batches`
At the 10x preset every batch takes the same settings and differs from the others only by its own draw of the per-gene factor (`sigma_batch`, 0.10) and of the depth shift (`depth_batch_sigma`, 0.05), which separates batches far less than real runs do.
`batch_kwargs` gives each batch its own settings, and `atlas_scrna_batches(n_batches, seed)` draws them from the fits to the 26 tumours of the breast cancer atlas of Wu et al. 2021, as the paper's batch model does.
The fits ship as `src/iscc/data/reference/wu2021_scrna_batches.tsv`, built from the paper's fits by `validation/data/build_atlas_batches.py`.
Each batch takes one tumour's `mu_lib`, `sigma_lib`, `dispersion`, `dropout_mid`, `dropout_shape` and `doublet_rate`, an `ambient_frac` drawn between 0.03 and 0.11, and the tumour's dissociation stress, which goes to `Resection.dissociate(stress_strength=...)`.

### Visium spots and depth — `iscc.data.Visium`
A deme is ~50 µm, so the default spot is Visium v1's: 55 µm across at a 100 µm pitch.
The capture field is normalised over the tissue spots, so a tissue spot holds `mu_counts` UMIs on average whatever share of the slide the section covers.

| Knob | Default | Valid range | Outside the range |
|---|---|---|---|
| `spot_pitch` / `spot_radius` | 2.0 / 0.55 demes | radius below half the pitch (spots apart, as on a slide) | a radius of 1 (the old default) made a spot as wide as the pitch: it pooled three times the cells of a 55 µm spot (~40 at 12 cells per deme) and tiled 91% of the tissue instead of 27% |
| `mu_counts` | 20,000 UMIs per tissue spot | ~10³–10⁵ | with spot swapping on it is the depth before swapping, and the tissue keeps a share of it: 0.96 on 10x's breast section (76% of the slide) at the fitted swap, 0.90 for a section on a quarter of the slide, whose distal bleed lands mostly off the tissue. `estimate_visium` given a raw matrix with all spots returns this depth |
| `kappa` (`count_model="dm"`) | 300,000 | ≳ 10⁵ | it inflates every gene's variance by (N + κ)/(1 + κ) at a spot depth N, whatever the gene set, so unlike the DNA concentration it is not rescaled by the number of genes; 50 (the old default) inflated it ~390-fold at 20,000 UMIs, and over 6,000 growth genes a spot detected 281 genes instead of ~4,700 and put 77% of its counts in 50 of them |

---

## Diagnosing a tumour

After growth, `tumor.diagnose()` checks each degenerate regime against a threshold and prints an
actionable hint for any failure. It is **read-only** — it never changes the simulation.

```python
from iscc.tumor.models import GenotypeTumor

tumor = GenotypeTumor(config=cfg, seed=0)
tumor.grow(n_steps=2000, seed=0)

dx = tumor.diagnose(verbose=True)   # prints [PASS]/[FAIL] per check, with hints
# or: from iscc.tumor.diagnostics import diagnose; diagnose(tumor, verbose=True)

# Override any threshold per call:
tumor.diagnose(thresholds={"shannon_min": 1.0})
```

### The regimes it catches (default thresholds, from `diagnostics.py`)
| Regime | Metric (threshold) | Culprit knob(s) — the hint |
|---|---|---|
| **extinct / too small** | cancer-cell count (< 25 fails; < 1000 advisory) | lower `death_rate` / raise `division_rate`, `initial_cancer_cells`, or grow longer |
| **monoclonal** | clonal Shannon diversity (< 0.5) | no-mutation → raise `mutation_rate` / `n_snvs_per_allele`; sweep → lower `driver_effects` / `prop_driver` |
| **hypermutated mush** | fraction of genome mutated / cell (> 0.5) | lower `mutation_rate` / `n_snvs_per_allele` |
| **well-mixed** | clone spatial confinement, ≥ 2 subclones (< 0.1) | lower `dispersal_rate` relative to `division_rate` |
| **demes over-filling** | mean cells/occupied-deme vs `carrying_capacity` (> 3×K) | raise `maximum_death_rate` to ≥ `max_birth_rate` (crowding cap not binding); skipped in the well-mixed regime |
| **no O₂ gradient** | hypoxia core–rim contrast (< 0.05) | grow a larger tumour, raise consumption `k`, or lower diffusion `D` |
| **CNA runaway** | fraction-genome-altered (> 0.95) | lower `amp_prob` / copy-number event rate |
| **clone == cell state** (skipped when programs are off) | within-clone share of program-activity variance (< 0.05) | raise `activity_noise` / `activity_sd`, or set `n_active_programs_per_cell` < `n_programs` — with all three off `z` collapses to the per-clone drive |
| **trivial genome** | gene count (< 100) | raise `n_segments` × `segment_size` |

Thresholds are deliberately lenient — they flag the clearly broken, not the merely unusual — and every
one is overridable via `thresholds={...}`.

---

## See also
- **Defaults to copy:** `notebooks/example_config.yaml`, `src/iscc/tumor/tumorconfigs/genotype.yaml`
- **The characterization:** `analysis/characterize_regimes.py`, `validation/validate_operating_envelope.py`
  → `manuscript/figures/validation_operating_envelope.png`
- **Output layout (not parameters):** `SCHEMA.md`
