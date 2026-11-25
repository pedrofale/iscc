# Parameter reference

The defaults of `iscc` are inside its *operating envelope*, the region of parameter space that produces
realistic tumors. For each parameter, this page gives what it does, its default, its **valid range** and
**what happens outside it**. Changing a few parameters and keeping the rest at their defaults gives a realistic
tumor with several clones.

Two checks support these defaults.

- The shipped configurations (`notebooks/example_config.yaml`, `src/iscc/tumor/tumorconfigs/genotype.yaml`)
  are inside every valid range below.
- After growth, [`tumor.diagnose()`][iscc.tumor.GenotypeTumor.diagnose] runs a read-only quality-control
  check that flags degenerate tumors and names the parameter to change (see [Diagnosing a tumor](#diagnosing-a-tumor)).

The ranges below come from a sweep of the operating envelope.

---

## Configuration parameters

Defaults are those of `notebooks/example_config.yaml`. Set them under the matching YAML block
(`cell_params.cancer`, `deme_params`, `spatial_params`, `selection_params`, `genome_params`).

!!! note "The defaults below describe the DCIS-to-IDC tumor of `example_config.yaml` at spatial-assay scale"
    A deme is a ~50 µm 3D column of tissue (~4 cells in-plane), not one cell, and its `carrying_capacity`
    is the whole cell population of that column (`K_duct` 60 in a dense duct, `K_stroma` 30 in looser
    stroma). A `grid_size` of 170 is therefore ~8.5 mm of tissue (~10⁶ cells), larger than a Visium
    capture, so each assay samples a subset. The tumor grows from a single clonal founder as a
    breach-limited ductal carcinoma in situ (DCIS), confined to the ducts until a clone with the `breach`
    trait crosses the duct wall (`breach_gated_invasion`) into a permissive stroma (low `stromal_hazard`),
    giving an invasive ductal carcinoma (IDC). `example_config.yaml` therefore turns on the
    compartment-selection parameters (`breach_gated_invasion` / `prop_breach`, `stromal_hazard` /
    `prop_stromal_survival`, `cross_gland_*`, `wgd_rate`), which are off in the bare engine, and the
    tables below give its values, with the bare-engine value where it differs.

### Mutations (`cell_params.cancer`)
| Parameter | Default | Valid range | Outside the range |
|---|---|---|---|
| `mutation_rate` | 0.3 (ER+ subtype preset; bare engine 0.1) | **[0, 1]**: the per-division probability of a mutation (a single-nucleotide variant (SNV) event whose Poisson draws are all 0 leaves the genome unchanged) | **low** → monoclonal (no subclones); **high** → hypermutated genome (broken 1/f tail) |
| `n_snvs_per_allele` | 0.1 (bare engine 0.5) | ~0.1–3 | **low** → no SNV diversity; **high** → hypermutated genome |
| `snv_prob` / `cnv_prob` | 0.5 / 0.045 (`cnv_prob` from the ER+ subtype preset; bare engine 0.5 / 0.5) | relative weights of SNV and copy number aberration (CNA) events | all SNV → no CNAs (inferCNV and clonealign examples degrade); all CNA → no SNV phylogeny |

### Genome-wide passengers (`genome_passengers`, top-level, on by default)

The SNVs above fall on the growth genes, a few thousand sites, so at tumor scale the same gene is
mutated in many lineages. [`PassengerCatalogue`][iscc.tumor.PassengerCatalogue] adds the neutral
passengers of the whole genome when cells are materialised, each at a site of its own, and bulk WGS
reports them in `passenger_data`. Growth never reads them.

| Parameter | Default | Valid range | Outside the range |
|---|---|---|---|
| `rate` | 2.0 | ≥ 0 passenger SNVs per cell division (healthy tissue about 1.1 to 1.4, tumors higher; Werner et al. 2020) | **0** → no passengers inside clones or on their edges, only the truncal ones; **high** → more subclonal passengers per clade, a denser 1/f tail and a larger `passenger_data` |
| `truncal` | 1,000 | ≥ 0 passengers carried by the founder | **0** → no clonal cluster at the passenger sites |

`genome_passengers=False` turns the layer off.

### Growth and survival (`cell_params.cancer`, `deme_params`)
| Parameter | Default | Valid range | Outside the range |
|---|---|---|---|
| `division_rate` | 0.7 (bare engine 1.0) | **> `death_rate`** | ≤ `death_rate` → extinction |
| `death_rate` | 0.05 (bare engine 0.1) | **≪ `division_rate`** | ≥ `division_rate` → extinction |
| `initial_cancer_cells` | 20 (bare engine 1) | ≥ 1; capped at `carrying_capacity` when crowding is on | 1 → a lone founder dies out with probability ≈ `death_rate` / `division_rate` (~7% at 0.05 / 0.7), so some runs end without cancer |
| `maximum_death_rate` | 2.2 (bare engine 1.0) | **> `max_birth_rate`** | clamps every death rate before the hazards, and is the default `crowding_d1`; with crowding on, at or below `max_birth_rate` construction fails (evolved clones would outgrow the cap) |

### Spatial structure (`spatial_params`, `deme_params`, `cell_params.cancer`)
| Parameter | Default | Valid range | Outside the range |
|---|---|---|---|
| `carrying_capacity` | 30 (bare engine 10) | a real **per-deme cap** (cells/deme); `None` or `0` → **well-mixed** (no ceiling, unbounded growth) | too small (1–3) → a deme holds at most `floor(crowding_overfill · K)` cells, and a deme above `K` cells takes the crowding death rate `crowding_d1`, above `max_birth_rate` |
| `grid_size` × `carrying_capacity` | 170 × 30 | the tumor caps at ~`grid_size² × carrying_capacity`; size the grid **above** the target so it can spread (≳ 10⁴ demes for a 10⁵-cell tumor) | too small → no room to spread, no O₂ gradient, too few clones |
| `dispersal_rate` | 0.9 (bare engine 0.1) | **[0, 1]**: the per-division probability that a daughter disperses, independently of whether the division mutates | high → most divisions send a daughter to a neighboring deme, so clones mix between demes: well-mixed, **no clonal territories** (silently breaks the PEtracer and multi-region comparisons) |
| `structure_radius` / `n_structures` | 4 / 1 | glandular geometry (duct size and count); the cancer is founded inside and spreads across the gland | — |
| `n_glands` | 8 | ductal-field substrate: number of small epithelial-ring glands scattered in stroma; `1` reproduces the single central ring | needs `structure_radius > 0`; too many for the grid → rejection sampling cannot place them all |
| `gland_radius` | = `structure_radius` | ring radius of each gland (`structure_radius` 4 → 9 demes across, ~450 µm, lumen and wall) | too large × many glands → glands do not fit or overlap |
| `min_gland_sep` | 14 (bare engine `2·gland_radius + 2`, 10 at `gland_radius` 4) | minimum center-to-center spacing between glands | too small → glands touch (no stroma gap) |
| `K_duct` / `K_stroma` | 60 / 30 | per-compartment carrying capacity (a deme is a 3D column, so K is **moderate to large**, capturing depth, not a few cells) | spot resolution constrains the **grid spacing**, not K |
| `stroma_fill_frac` | 0.3 | baseline stromal occupancy as a fraction of `K_stroma` (stroma is less dense than epithelium but **not empty**: its stromal cells, captured by the assays, carry the stromal hazard; seed at a **moderate** 0.3–0.5, leaving room for an invasive mass) | `1.0` = the former dense fill (byte-identical); too sparse → the live stromal fraction (and its hazard) is negligible |
| `stromal_composition` | `{stromal: 1}` (all fibroblasts) | the shares of the stromal subtypes among the stromal cells of each deme, drawn per deme: `stromal` (fibroblasts), `endothelial` (vascular and lymphatic endothelial cells), `pvl` (perivascular cells: pericytes and vascular smooth muscle). Every subtype counts as stroma for the stromal hazard and for eviction, so the composition changes the data and not the growth; `GenotypeTumor.split_stroma` gives it to a tumour grown without it. The paper's regime uses 0.335, 0.388 and 0.277, the shares of the stroma of Wu et al. 2021, recovered alike | shares outside these subtypes, negative or not summing to 1 → an error |
| `immune_density` | 0 (off) | static immune (T and NK) cells per stroma deme, as a fraction of `K_stroma`; in a field without glands, per deme of `carrying_capacity`. They neither divide nor die, count toward crowding and the resident pressure, leave a deme as cancer overfills it (`immune_displaceable`), and kill cancer cells at `prob_kill` × the immune share of the deme × (1 − immune resistance). The paper's immune variant uses 0.2 (6 per stroma deme beside 9 fibroblasts) | too high → the immune share of a full deme bounds cancer growth, and a small founder can die out |
| `immune_displaceable` | True | each cancer birth that leaves a deme above its carrying capacity removes one of its immune cells, while any remain, so tumor nests lose them and the stroma and invasive front keep them. False keeps every immune cell in place for the whole run | False → every deme a tumor takes over stays infiltrated at its seeded density |
| `immune_duct_density` | 0 | the same per duct-wall deme, as a fraction of `K_duct` (intraepithelial lymphocytes); empty lumens get none. The paper's immune variant keeps immune cells in the stroma only | — |
| `immune_composition` | `{immune: 1}` (all T and NK cells) | the shares of the immune subtypes among the immune cells of each deme, drawn per deme: `immune` (T and NK cells, the only ones that kill), `myeloid` (macrophages, monocytes, dendritic and mast cells), `bplasma` (B and plasma cells). The paper's immune variant uses 0.575, 0.315 and 0.11, the shares in the tumors of Wu et al. 2021 corrected by the default recovery | shares outside these subtypes, negative or not summing to 1 → an error |
| `immune_cell_params.prob_kill` | 0.1 | the kill rate per unit share of T and NK cells in a deme. The paper's immune variant uses 28 with displaceable immune cells (7 with immune cells kept in place), with immune resistance at 0.2% of genes and effect 4.0 (one hit protects 75%): growth takes 1.5 times as long and the mean immune resistance is 14.6 times the immune-free level | high with weak resistance (effect 1.1, one hit protects 9%) → growth stalls before resistance is selected |
| `cross_gland_kappa` | 0.06 (0.0 = off) | island (intraductal) dispersal weight: a fraction ~`κ/(1+κ)` of the dispersing daughters of a gland-resident clone seed **another gland's lumen** (bypassing the wall → confined DCIS, no breach), rate `κ·dispersal_rate` | stroma cells never jump between glands; `0` → no cross-gland spread |
| `cross_gland_lambda` | 30 (`None` = uniform) | distance kernel for cross-gland targeting (probability ∝ `exp(−d/λ)` over gland centers); `None` = uniform (no extra parameter) | 2D-section distance is only a proxy for ductal-tree proximity |
| `epithelial_barrier` | 0.0 (off, as in the bare engine) | compartment-selection hazard: the epithelial ring adds `epithelial_barrier · epithelial_fraction(deme) · (1 − breach)` to cancer-cell death. `example_config.yaml` gates invasion with `breach_gated_invasion: true` instead (bare engine `false`): a dispersal hop from a duct deme into a stroma deme succeeds with probability equal to the `breach` trait of the clone, and otherwise the cell stays in its duct | needs `structure_radius > 0`; too high → the ring is impassable and cancer stays in the lumen (raise `breach` fitness or lower the barrier) |
| `stromal_hazard` | 0.6 (0.0 = off in the bare engine) | compartment-selection hazard: the stroma adds `stromal_hazard · stromal_fraction(deme) · (1 − stromal_survival)` to death | needs `structure_radius > 0`; the stromal analog of `epithelial_barrier` |

!!! note "`carrying_capacity` is a real per-deme cap (the two-valued death step of demon under a slot cap)"
    A deme has `floor(crowding_overfill · K)` slots. A birth that finds none fails, and contested slots
    go to clones in proportion to the births they drew. A cancer cell dies at the baseline `death_rate`
    while the deme contains at most `K` cells and at `crowding_d1` (default `maximum_death_rate`) above
    it, the same for every clone in the deme, so clones in a full deme keep competing through their
    division rates. After the `maximum_death_rate` clamp, `crowding_d1` must exceed `max_birth_rate` or
    construction fails. Immortal residents add
    `(resident_pressure_ref − death_rate)·(1 + crowding_margin)·n_normal/K`. Set `carrying_capacity: None`
    (or `0`) to disable crowding entirely, which gives the **well-mixed** regime used for single-deme,
    unbounded-growth benchmarks.

| Parameter | Default | Valid range | Outside the range |
|---|---|---|---|
| `crowding_overfill` | 1.5 | ≥ 1 | the slot cap as a multiple of `K`; at 1 a deme never exceeds `K` and never reaches `crowding_d1` |
| `crowding_d1` | `maximum_death_rate` | > `max_birth_rate` (after the clamp) | the death rate above `K`; a lower value raises at construction |

The paper regime (`validation/paper/regime.py`) runs `crowding_overfill` 1.5 and `crowding_d1` =
`maximum_death_rate` = 5, with the division rule of demon (a division mutates with probability
`mutation_rate` and, independently, sends a daughter out with probability `dispersal_rate`), random
streams keyed by (seed, purpose, step), and origin confinement counted in generations. These are the only
rules of the engine, and a configuration that names a removed one (`crowding_mode`, `crowding_law`,
`dispersal_law` other than the above, or a `"steps"` confinement clock) raises.

### Selection (`selection_params`)
| Parameter | Default | Valid range | Outside the range |
|---|---|---|---|
| `prop_driver` | 0.04 (bare engine 0.1) | ~0.04–0.3 | high × strong effect → selective sweep (monoclonal) |
| `driver_effects` | 1.1 | 1.0–2 | ≫ 1 at low mutation rate → monoclonal sweep |
| `segment_drivers` | unset | one `[oncogenes, suppressors]` pair per segment, each summing to at most the segment's genes | unset → drivers are drawn gene by gene with `prop_driver`; set → exactly these counts at random loci of each segment, and `prop_driver` is unused. [`arm_genome`][iscc.data.human.arm_genome] sets it from the Cancer Gene Census content of each arm, so the driver load of an arm does not change with the resolution. A pair that does not fit raises. |
| `prop_dispersal` / `dispersal_effects` | 0.0 / 1.1 (bare engine 0.1 / 1.1) | as above | strong → invasion dominates, spatial structure is lost |
| `prop_treatment_resistance` / `prop_immune_resistance` (+ `treatment_resistant_effects` / `immune_resistant_effects`) | 0.0 (treatment) / 0.02 (immune), bare engine 0.1 / 0.1; effects 1.1 | as above | resistance is meant to **emerge**, not be seeded in advance |
| `prop_breach` / `breach_effects` | 0.02 / 2.8 (bare engine 0.0, off / 1.1) | compartment-selection trait: sites that if mutated let a clone cross the epithelial ring (attenuate `epithelial_barrier`, or pass the `breach_gated_invasion` gate); heritable, sequenceable | its benefit acts only through `spatial_params.epithelial_barrier` (> 0) or `spatial_params.breach_gated_invasion` |
| `prop_stromal_survival` / `stromal_survival_effects` | 0.02 / 2.2 (bare engine 0.0, off / 1.1) | compartment-selection trait: sites that if mutated let a clone survive the stroma (attenuate `stromal_hazard`) | pairs with `spatial_params.stromal_hazard` |
| `breach_cost` / `stromal_survival_cost` / `met_survival_cost` / `treatment_resistance_cost` | 0.6 / 0.6 / 0.0 / 0.0 (bare engine 0.0, off) | 0.0–1.0 | **go-or-grow trade-off** (compartment-dependent fitness): each dissemination or niche trait costs this fraction of `division_rate` per unit of the trait, applied **everywhere**, while its benefit (attenuating its compartment hazard) applies only there. A trait is therefore favored **only in its niche** and selected against elsewhere, which stops one clone from carrying every trait into every compartment (proliferation stays in the DCIS, stromal survival in the IDC, metastatic survival in the deposit, resistance under chemotherapy). | too high suppresses the trait even in its niche (a high `met_survival_cost` starves the metastasis; a high `treatment_resistance_cost` keeps resistance a rare subclone before chemotherapy, which is usually the intent) |

#### Metastasis and drug-tolerance traits (`selection_params`)

| Parameter | Default | Valid range | Outside the range |
|---|---|---|---|
| `prop_met_survival` / `met_survival_effects` | 0.0 (off) / 1.1 | as for the other traits | the dissemination trait of the metastatic compartment; attenuates the `met_hazard` of the deposit. Pairs with `met_survival_cost`, and a high cost starves the deposit |
| `prop_drug_tolerance` / `drug_tolerance_effects` / `drug_tolerance_cost` | 0.0 (off) / 1.1 / 0.0 | 0–1 / ≥1 / 0–1 | a **non-genetic persister** trait, distinct from `treatment_resistance`. **Currently not usable for a residual-disease floor**: the trait is a division discount, and a discount cannot hold a population at a floor under a proliferation-scaled kill (that needs dormancy, a division rate of ~0). Kept, off by default, until dormancy exists. |

!!! warning "`drug_tolerance_cost` is charged differently from every other cost"
    Every other `*_cost` is permanent, because the trait is permanent, and is applied in the
    division-cost factor. A persister state exists only while the drug does, and a permanent charge makes
    the trait unusable, because a cost high enough to hold the pool under drug also purges it to ~0.1%
    *before* treatment starts. It is instead applied as a division multiplier **for the duration of the
    dose**.

#### Drug-induced resistance state (`selection_params`, optional, all off by default)

These parameters model resistance as a **carried cell state** rather than only a genomic allele, so that
a CNA deleting the triggering allele does not restore sensitivity. Every parameter defaults to its inert
value, so `Selection.resistance_state_on` is `False` and division is **byte-identical** to the engine
without the state.

| Parameter | Default | Valid range | Outside the range |
|---|---|---|---|
| `resistance_state_genetic` | `False` | bool | `True` → acquiring `n_mut_tr > 0` sets the state, which is then carried independently of the genome. `validation/validate_escape_modes.py` uses this path |
| `resistance_state_effect` | 0.0 | 0–1 | the protection the state confers, on the same scale as `treatment_resistance`. 0.99 ≈ almost no drug effect |
| `resistance_state_cost` | 0.0 | 0–1 | division × `(1 − cost × state)`, charged **always**, on *and* off drug. It **stacks multiplicatively** with `treatment_resistance_cost`, so setting both to *c* charges an allele-anchored cell twice and an exited revertant not at all. To price the two routes equally, set `treatment_resistance_cost = 0` and charge the state only |
| `resistance_state_induction` | 0.0 | 0–1 | per-generation P(a sensitive cell is reprogrammed) = `induction × intensity`, where `intensity` is the dose that actually reaches the cell, so a cell the drug cannot reach is not reprogrammed by it |
| `resistance_state_relax` | 0.0 | 0–1 | per-generation P(exit), so τ = 1/`relax` generations. Applies **only** to cells with `n_mut_tr == 0`: a cell still carrying the allele keeps the genetic attractor and never relaxes. `0` is the permanent (τ→∞) limit |

!!! note "`induction` and `relax` act only in the tau-leaping engine"
    The reversible transitions are in the tau-leaping step. The *genetic* entry clause and the
    drug-protection term are in shared code and so are active under the exact engine too, but in an
    exact-engine run a non-zero `relax` is silently inert. Check `update_mode` before concluding that a
    result holds under a finite exit rate.

!!! note "A permanent state (τ→∞) is not required"
    We measured this on escape mode IV of `validation/validate_escape_modes.py`: at `relax = 0.02` with
    fair single-charge pricing the relapse is 97.7% resistant with the clean start intact, against
    100.0000% at `relax = 0`. Permanence adds only the last ~2 percentage points.

#### Trait and selection model (`selection_params`)

| Parameter | Default | Valid range | Outside the range |
|---|---|---|---|
| `trait_source` | `"mutation"` (bare engine `"dosage"`) | `"dosage"` \| `"mutation"` | what moves a dissemination or niche trait. `"dosage"` (the original model, bit-for-bit) counts copy number *and* mutation; `"mutation"` divides out the wild-type dosage term so **only mutated copies move the trait**. The two agree on the value of a single heterozygous SNV and differ only on the value of a copy-number change |
| `selection_mode` | `"gene"` | `"gene"` \| `"arm"` | `"arm"` switches to the per-arm copy-number fitness of CINner and SISTEM, `∏_seg s_arm[seg] ** (cn[seg]/ploidy − 1)`, computed from the per-segment copy numbers `iscc` already maintains. Dosage is relative to the genome-wide mean ploidy, so a whole-genome doubling leaves fitness unchanged; the `− 1` only rescales every genome by the same constant so the diploid genome has fitness 1 |
| `s_arm` | `None` (→ all 1.0) | length-`n_segments` array | per-arm selection coefficients. `s_arm[seg] > 1` → amplifying that arm is beneficial (oncogene-dominated); `< 1` → deleting it is (suppressor-dominated). All 1.0 = arm selection off. Same scale as CINner's arm `s` and SISTEM's `1 + δ` |
| `rng` | `None` | — | **internal.** A `numpy` Generator for the layout draw; pass `layout_seed` instead for a reproducible gene layout |
| `segment_sizes` | `None` (→ uniform `segment_size`) | length-`n_segments` array | per-segment gene counts, for a genome with unequal arms. `None` gives every segment `segment_size` genes. Ploidy is the mean copy number weighted by these sizes, as CINner and SISTEM weight by length |

#### Viability limits (`selection_params`)

These are hard limits, as in CINner, on which genomes are compatible with life. A daughter that breaches
**any** of them is not born: the division is consumed but yields no cell, in both engines. The defaults
are loose, except `max_nullisomy`, and the exact engine never reaches the other three at the shipped
defaults, so they matter only once tightened or under strong amplification or deletion.

| Parameter | Default | Valid range | Outside the range |
|---|---|---|---|
| `max_ploidy` | 6 | 3–8 | tight (≲ 3) → caps chromosomal instability (CIN); near-diploid tumor, whole-genome duplication (WGD) impossible |
| `max_cn` | 12 | 6–20 | tight → caps the total copy number of any segment; interacts with `amp_prob` |
| `max_nullisomy` | 0 | 0–`n_segments` | the number of segments a viable cell may have at zero copies. `0` (default, as `bound_homozygosity` in CINner): a segment is a chromosome arm or more, and a cell without any copy of one has lost essential genes. Above 0, dosage selection sweeps the loss of both copies of suppressor-rich segments (97% of the cancer cells of a paper-regime tumor at 2) |
| `max_mut_drivers` | 12 (bare engine 1000, off) | 1–(`prop_driver` × `n_genes`) + 4 | tight → caps the driver load; `0` → any driver is fatal. The engine default of 1000 is off, so the limit does not disturb, for example, hypermutation studies or the MHN benchmarks unless a configuration sets it; `example_config.yaml` and the landing configurations in `configs/` set 12 |

!!! note "`max_mut_drivers` counts distinct driver genes plus activated dissemination traits"
    The load is `n_mutated_drivers` (distinct oncogenes and tumor suppressors with an SNV on **at least
    one copy**, so a gene amplified to copy number 4 counts once) **plus one per activated dissemination
    trait** (`breach` / `stromal_survival` / `met_survival` / `treatment_resistance` / `drug_tolerance`,
    counted by presence). A clone that stacks every trait therefore has a high load and is removed under a realistic
    ceiling, which extends the mutational-load limit of CINner to the compartment traits. The ceiling
    cannot *localize* traits (a resistant metastatic cell carries ~8 drivers, so any ceiling that admits
    it also admits a clone with every trait), which is the role of the go-or-grow **cost**, and it only
    removes the biologically implausible tail. `n_mut_onc` and `n_mut_tsg` are no substitute for the
    oncogene and suppressor part, because they are weighted by copy number and so do *not* count mutated
    drivers.

    The engine default of 1000 is far out of reach, so the limit is off unless a configuration sets it;
    `example_config.yaml` and the landing configurations set 12.

!!! note "`max_nullisomy` is active at the defaults"
    At the default 0, a daughter that loses the last copy of a segment is not born, under both engines.
    Real homozygous deletions are focal, while `iscc` has no focal copy-number events, so a segment at
    zero copies would remove a whole chromosome arm from the DNA and the expression of the cell.

### Therapy (`Treatment` operations `chemotherapy`, `targeted_therapy`, `immunotherapy`)

Every therapy operation ([`Treatment`][iscc.treatment.Treatment]) shares these scheduling and dosing
parameters. **The two engines do not implement the drug the same way**, so read the warning below before
comparing runs across engines.

| Parameter | Default | Valid range | Outside the range |
|---|---|---|---|
| `start` / `duration` | 0 / `None` | steps | when dosing starts, and for how long. `None` = until the phase ends |
| `rounds` | 4 | ≥ 1 | number of dosing rounds |
| `dosage_decay` | 0.5 | 0–1 | how much the dose falls between rounds |
| `effectiveness` / `toxicity` | 0.9 / 0.1 | 0–1 | probability the drug acts on an **on-target** cell / on an off-target one. Off-target toxicity also kills the host parenchyma of the metastasis, which progressively makes the deposit *safer* for the cancer |
| `sites` | `"both"` | `"both"` \| primary \| met | which compartment is dosed |
| `max_tumor_size` | 100000 | ≥ 1 | with `adaptive`, the burden threshold above which dosing resumes |
| `adaptive` | `False` | bool | `True` → dose only above `max_tumor_size`, the adaptive-therapy regimen |
| `kill_rate` | 1.5 | ≥ 0 | the drug strength, read by the **count** engine |
| `kill_mode` | `"additive"` | `"additive"` \| `"proliferation"` | **the most consequential therapy parameter.** `"additive"` gives every clone the same *absolute* extra death, so birth rate decides survival and a driver-saturated clone outgrows the dose. `"proliferation"` scales the hazard with the division rate of the clone itself (`added_death = intensity × kill_rate × division_rate`), so every clone declines, at a clone-specific rate, whatever the fitness, and a cell at `division_rate = 0` has exactly zero drug death. This is also the biology (chemotherapy kills replicating cells) |
| `rate_multiplier` | 2.0 | ≥ 1 | the drug model of the **cell** engine: `death × rate_multiplier ** (1 − treatment_resistance)`. Unused by the count engine |
| `mutagenicity` | 1.0 (off) | ≥ 1 | multiplies `mutation_rate` while dosed. **This is not a mutation rate**: `mutation_rate` is a per-division probability capped at 1, so the realized boost saturates. It also moves SNVs and CNAs in lockstep, so it **cannot** shift the balance between acquiring resistance and reverting it (measured identical at 1.0 and 4.0) |
| `mutagenicity_target` | `"all"` | `"all"` \| `"snv"` | `"snv"` makes the drug a **point mutagen**: it scales `n_snvs_per_allele` instead, which feeds only the SNV branch, so acquiring a resistance SNV becomes likelier while the CNA that deletes one stays flat. This decouples acquisition from reversion (acquisition:reversion 0.14 → 1.74 at ×20) |
| `mutagenicity_mode` | `"uniform"` | `"uniform"` \| `"dose"` | `"dose"` scales the mutator by `(1 − treatment_resistance)`, like every other drug effect. This has **no effect on a de novo resistant clone**, which descends from a drug-exposed *sensitive* ancestor and inherits the raised rate through division, as expected for a permanent mutator that impairs repair. It does protect a *pre-existing* resistant clone. Under `response="mechanistic"` it scales by the exposure of the clone instead (the target-expression ratio, capped at 1) |
| `response` | `"label"` | `"label"` \| `"mechanistic"` | how the count engine decides whom the drug spares. `"label"` reads the resistance labels (`treatment_resistance`, `drug_tolerance`, the resistance state); the escape-mode validation is built on it. `"mechanistic"` ignores all three, and the persister cost with them: a clone is on target only if it **carries** the drug's `targets` (a copy with the mutation, or more copies than `amplification_threshold`), the dose scales with the expression of `target_genes` in the clone, and a broad drug with `kill_mode="proliferation"` kills in proportion to division rate. Count engine only (the per-cell engines raise), and Python API only: the schedule's `chemotherapy` operation does not forward it |
| `target_genes` | `[]` | `(segment, position)` or flat gene index | mechanistic only. The dose is multiplied by the mean expression of these genes in the clone over that in the founder, so the dose of the founder is nominal. Expression is the dosage baseline × the program multiplier at the *mean* activity of the clone, without per-cell noise or niche |
| `expression_threshold` | 0 | ≥ 0 | mechanistic only, with `target_genes`: below this expression the drug does nothing to the clone, not even `toxicity` |
| `amplification_threshold` | `None` | copies | mechanistic only: more copies of the segment of the target than this also count as carrying it, so the drug acts on an amplification until a deletion takes it back to the threshold |

!!! warning "The engines use three different drug models"
    `kill_mode="additive"` and `kill_mode="proliferation"` are the two models of the count engine, while
    the **cell** engine uses `death × rate_multiplier ** (1 − tr)`
    ([`Chemotherapy`][iscc.treatment.Chemotherapy]). They are not calibrated against one another, so a
    `kill_rate` tuned on one engine does not transfer, and a cross-engine comparison of therapy response
    is not like for like.

### Genome (`genome_params`)
| Parameter | Default | Valid range | Outside the range |
|---|---|---|---|
| `n_segments` × `segment_size` | `n_segments` required, `segment_size` 1000; the shipped configurations use 12 × 500 (`notebooks/example_config.yaml`, the paper regime) and 10 × 1000 (`genotype.yaml` of the command-line tool), and [`arm_genome`][iscc.data.human.arm_genome] gives 39 arms over 6,000 genes | ≳ 100 genes | too few → trivial genome; assays degenerate, too few drivers |
| `segment_sizes` | unset (all `segment_size`) | one positive size per segment | sets unequal segments, for example arms of different lengths |
| `arms` | unset | one arm name per segment (`1p` … `22q`, the 39 modeled arms) | names the human chromosome arm each segment stands for; only the human genome layer uses it, growth never does |

#### Arm-homologous genome and the human genome layer (`iscc.data.human`)

`arm_genome(total_genes=6000, weight="genes", driver_scale=1.0)` returns `genome_params` (one segment per
arm, with `arms`) and `selection_params` (`segment_drivers`). The number of growth genes is a resolution
that trades the detail of the dynamics against their cost; the layer works at any value.

| Parameter | Default | Valid range | Outside the range |
|---|---|---|---|
| `total_genes` / `genes_per_segment` | 6000 / unset | ≳ 4 genes per arm | fewer → small arms cannot contain their drivers (capped, with the oncogene/suppressor split kept) |
| `weight` | `"genes"` | `"genes"` \| `"length"` \| `"uniform"` | how `total_genes` is shared between arms: by GRCh38-2020-A gene count, length, or equally |
| `driver_scale` | 1.0 | 0.1–1 | drivers per arm = this × the Census oncogenes and suppressors (297 + 293 over the 39 arms); 0.4 gives about the driver load of the paper regime |
| `HumanGenomeLayer(seed=)` | the `layout_seed` of the tumor | any int | fixes the germline: patients of one configuration share it unless given their own seed |
| `snp_window` | 1000 bp | 300–3000 | the 3' stretch of a gene that a 3' library covers; only heterozygous SNPs there receive allele counts |
| `snp_coverage` | 0.1 | 0.01–0.5 | probability that a UMI of a gene covers a given SNP in the window (about read length / window). A preset, not fitted |
| `snp_concentration` | unset (binomial) | ≳ 10 | beta-binomial concentration of the haplotype split at a SNP (allelic-expression noise) |
| `Visium(kappa=)` on human genes | `HUMAN_VISIUM_KAPPA` = 1,400,000, the assay's default | 10⁶ to 2×10⁶ | by indirect inference: [`estimate_visium`][iscc.data.estimate_visium.estimate_visium] gives 89,197 on the 3,798 in-tissue spots of the 10x section V1_Breast_Cancer_Block_A_Section_1 over all 36,601 features, and on a section of the paper's regrown immune tumor it gives 87,996 at an input of 10⁶ and 90,274 at 2×10⁶ (means of three Visium seeds, seed sd about 2,000), so the estimate equals the real value at about 1.4×10⁶ (interpolated, not run at that value). The direct value, used as input, over-disperses a simulated section, because the simulated tissue already contains the heterogeneity. Below 1,000 each spot's counts concentrate on a few genes, so [`write_visium`][iscc.data.human.write_visium] warns |

[`layer.expression(tumor)`][iscc.data.human.HumanGenomeLayer.expression]
([`HumanExpression`][iscc.data.human.HumanExpression]) is the realistic path. Each cell type has a
baseline from the Human Breast Cell Atlas (epithelial = luminal epithelium, stromal = fibroblasts, immune
= T and NK cells, myeloid = macrophages, monocytes, dendritic and mast cells, bplasma = B and plasma
cells, host = stroma), scaled by relative RNA content, and cancer cells have that of the
`cell_of_origin`. Named programs load on their MSigDB Hallmark sets (proliferation, EMT, hypoxia,
interferon response, and drug resistance on xenobiotic metabolism as a proxy), the rest on random human
genes, with the growth dictionary's loading distribution scaled by `programme_scale` (0.5) under the
bound `programme_max_log_fold` (5.0). We fitted the pair on the cancer cells of 14 tumors of Wu et al.
2021 so that the fold changes of the Hallmark genes between the top and bottom deciles of cells scored by
each program match (`validation/data/fit_programme_strength.py`; 1.0 and 5.0 doubled them). The program
factor acts on cancer cells only, because the atlas baseline of a normal type is already the mean of real
cells of that type, their usual program activity included (`normal_programmes=True` applies it to normal
cells too). On the held-out 10x Block A section the two changes take the share of spot UMIs on the
program genes from 42% to 13%, against 11% in the real section. What the genotype and the niche do to a
gene is read from the growth genes. Pass it as `gene_lift` to [`scRNA.run`][iscc.data.scRNA.run]
(chunked, sparse counts) and [`Visium.run`][iscc.data.Visium.run] (per-type pooling). The scRNA presets
add no dropout, which suits the human genes: a logistic dropout midpoint is in counts per gene, so a
midpoint of 1.5 UMIs deletes most of the human genes of a cell (410 genes per cell at 1,913 UMIs, against
1,523 at 3,494 without it, on 3,019 cells of an arm-genome tumor at the 10x preset, measured when the
layer emitted the 35,061 arm genes), and UMI counts are not zero-inflated beyond the negative binomial
(Svensson 2020). A run on human genes with `dropout_mid > 0` warns. Over the human genes the 10x preset
gives the zero fraction of PBMC3k within 0.02 at every mean.

The layer emits all 36,601 GRCh38-2020-A genes, as the Cell Ranger reference does. Each of the 35,061
genes on the 39 modeled arms is assigned the copy number per homolog of the growth gene covering the
corresponding slice of the arm and, with the plain `layer.lift`, the expression of that growth gene. The other 1,540
(chrX 1,148, chrY 111, chrM 13, and 268 on acrocentric short arms, centromeres and scaffolds) are covered by no
growth gene (`layer.on_arm` False, `cover` and `segment` -1): under `layer.expression` they are given the
atlas baseline of the cell type and the programs of the cell at two copies, with no genotype or niche
effect, and under the plain `layer.lift` they are 0. They account for about a tenth of the atlas
expression of each type (chrM 5-6%). Germline SNPs are the 1000 Genomes phase-3 common SNPs (allele
frequency above 5%) inside the arm genes, drawn heterozygous at 2·AF·(1 − AF) and phased exactly onto the
homologs; [`fetch_snps()`][iscc.data.human.fetch_snps] downloads them (92 MB) and `ISCC_HUMAN_REF` points
at the result.

Visium spots on the human genes take a per-gene platform factor (`HumanExpression(visium_platform=)`,
default the packaged [`load_visium_platform_factor()`][iscc.data.human.load_visium_platform_factor]
table): the sensitivity of Visium to each gene relative to droplet scRNA-seq, fitted on the four tumors
of Wu et al. 2021 measured by both assays (7,101 genes, log-factor sd 0.62; lowest for the
immediate-early genes that dissociation induces, such as JUN, FOS and ATF3, and for nuclear lncRNAs such
as MALAT1). It has no component shared by the genes of a cell type, so the composition of a spot remains
that of the tissue. On the held-out 10x Block A section it raises the correlation of log gene shares from
0.846 to 0.851 and lowers the median absolute log2 ratio from 1.09 to 1.00
(`validation/data/check_visium_platform_factor.py`). `visium_platform=None` draws spots from the
scRNA-seq composition. scRNA-seq is unchanged.

### DNA assays (`bulkDNA`, `scDNA`)

In [`bulkDNA`][iscc.data.bulkDNA] and [`scDNA`][iscc.data.scDNA], the coverage spread of a
Dirichlet-multinomial depth draw is set by `kappa` and by the number of loci: the excess squared
coefficient of variation of a locus is about `n_loci / kappa`. Each default `kappa` is therefore an
[`estimate_dna`][iscc.data.estimate_dna.estimate_dna] estimate stored with the number of loci it was
estimated on (`kappa_loci`), and the draw rescales it to the genome of the run. A `kappa` passed on its
own is used as given.

| Parameter | Default | Valid range | Outside the range |
|---|---|---|---|
| `bulkDNA(kappa=, kappa_loci=)` | 230,823 on 20,000 loci (GIAB HG002 WGS, allele depths over all datasets) | `kappa_loci / kappa` ≲ 0.2 (estimates: 0.03 to 0.17) | coverage CV at two copies 0.38 at 30× on any genome size (HG002: 0.30 at 298×). A former default of 2,000, used as given, gave 1.7 on 6,000 loci and left 39% of them below a tenth of the mean. The fit of 121,241 before 2026-11-25 read the benchmark's `AD`, which drops the 300× Illumina dataset at 13% of sites, and gave 0.47 |
| `scDNA(kappa=, kappa_loci=)` | 204,222 on 5,787 loci (DLP+ OV2295) | `kappa_loci / kappa` ≲ 0.2 (estimates: 0.03 to 0.17) | at 9 reads per locus, a median within-cell CV of 0.40 and under 0.1% of loci without reads on 6,000 loci. A former default of 5, used as given, left 99% of loci without reads, so segment copy number could not be called |
| `nb_dispersion` (`depth_model="nb"`) | 0.087 bulk (HG002), 0.026 single-cell (DLP+) | ≲ 0.2 | the negative-binomial alternative, fitted on the same references |
| `scDNA(platform=)` | `None` | `"dlp"`, `"tapestri"` | all the estimates on the reference of the platform, including its own depth per locus (DLP+ 830 reads per 500 kb bin, Tapestri 42.7 per amplicon). Tapestri reads the 584 amplicons of its fitted panel, and an explicit other `breadth` reads its own loci. DLP+ has no allele counts, so its allele layer and doublet rate stay at the single-cell priors (`ado_rate` 0.20, `doublet_rate` 0.02). An unknown name raises |

### Copy number (`cell_params.cancer`)
| Parameter | Default | Valid range | Outside the range |
|---|---|---|---|
| `amp_prob` | 0.5 | 0.2–0.8 | high → copy-number-heavy genome (viability-capped) |
| `wgd_rate` | 0.001 (ER+ subtype preset; 0.0 = off in the bare engine) | 0.0–~0.05 | drives genome doubling; a separate per-division channel (leaves `snv_prob` / `cnv_prob` intact). Too high → most tumors undergo WGD and accumulate at `max_ploidy` (doubled genomes are viability-capped); interacts directly with `max_ploidy` / `max_cn` |

`wgd_rate` is the per-(mutating-)division probability of a **whole-genome duplication**: every copy on
both homologs of every segment is doubled at once, so ploidy jumps 2→4 (then loss erodes it back to a
non-integer mean, the doubling-and-loss signature). A WGD daughter that breaches `max_ploidy` or `max_cn`
is **rejected at birth** like any other non-viable daughter, so `max_ploidy` (default 6) sets how much
post-WGD genome a lineage may carry. Off in the bare engine (0.0 → byte-identical to a run with no WGD). Sweeping
it brings tumor WGD prevalence into the real ~30–50 % range (`validation/validate_wgd.py`). The ground
truth per genotype is `cell_data["cell_wgd"]["is_wgd"]` (present only when `wgd_rate > 0`).

### Epistasis and dependency network (`selection_params.epistasis_params`, optional, off by default)

Selection is **additive** by default (fitness depends on the *count* of mutated drivers), so the true
dependency network between events is empty. These parameters **plant a known network**, the ground truth
for cohort progression models (MHN, TreeMHN, CBN/H-CBN, REVOLVER). With `epistasis_params` absent or
`n_events: 0`, nothing is built and the engine is **bit-identical** to the additive model (see
`validation/validate_epistasis.py`).

An **event** is a disjoint module of `event_size` driver genes. It occurs when ≥1 SNV hits the module,
and it is **monotone** (once acquired, never lost, the assumption under which MHN and CBN are defined).
Events are modules rather than single genes because MHN and TreeMHN pool many tumors to learn **one**
network, which requires the same events to **recur across patients**, and a single named gene out of ~10⁴
is hit in almost no patient. `event_size: 1` recovers the single-gene case. `event_size` is one int for
every module, or a list of one int per event (`[2, 1, 1, 2, 2, 2, 4]`), so that events can be set to
different frequencies in a cohort. Either way the modules come from one draw of the epistasis stream of
the layout seed, cut in event order, so a single size and a list of that size draw the same modules.

Fitness is multiplied by `exp(Σᵢ βᵢxᵢ + Σᵢ<ⱼ Eᵢⱼxᵢxⱼ)` on top of the additive model.

| Parameter | Default | Valid range | Outside the range |
|---|---|---|---|
| `n_events` | 0 (**off**) | 4–20 | 0 → additive engine, bit-identical; many → each event is rarer and the cohort must be larger |
| `event_size` | 20 | 1–~5% of the genome, per event | **sets the rate of each event**, as one int or a list of one per event: too small → events never occur (empty matrix); too large → every patient acquires every event (all-ones matrix, zero variance, nothing to correlate). The sum of the sizes must fit in the driver pool |
| `event_effect_mean` / `_sd` | 0.1 / 0.0 | 0–0.5 | the marginal log-fitness effect of each event (the diagonal of MHN). 0 → events are marginally neutral and only `E` acts |
| `n_interactions` | `None` → use `network_sparsity` | 0–`n_events(n_events−1)/2` | 0 → empty `E` (the false-positive control) |
| `network_sparsity` | 0.2 | 0.05–0.5 | fraction of event pairs that interact, when `n_interactions` is unset |
| `network_topology` | `random` | `random` \| `hub` \| `chain` | `hub` = one master event connected to all others; `chain` = a stepwise cascade |
| `interaction_strength` / `_sd` | 0.3 / 0.05 | 0.1–1.0 | **must fit under the fitness clamp** (see below) |
| `prop_synergy` | 0.5 | 0–1 | P(`E>0`); the rest are antagonistic |
| `mutual_exclusivity_strength` | 0.0 | 2–8 when used | magnitude of the strongly negative (synthetic-lethal) edges |
| `n_exclusive_pairs` | 0 | ≤ pairs left after the interaction edges | how many such edges |
| `mutual_exclusivity_lethal` | `True` | `True` \| `False` | **`True` is required for exclusivity to have any effect** (see below). `False` leaves only the soft `E` effect |
| `lethal_death_rate` | 1.0 | ≥ `max_birth_rate`, ≤ `maximum_death_rate` | death rate of a synthetic-lethal genotype; above `maximum_death_rate` (which clamps every death rate) the tumor cannot be constructed |
| `planted` | `None` (drawn at random) | a dict, see below | states the whole network explicitly instead of drawing it |

**Planting an explicit network.** `planted` replaces every random draw above except the event modules
(still drawn from the layout seed, exactly as in the random mode), so every pair can be named before any
tumor is grown. Events are indexed from 0 (`E0`, `E1`, ...).

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

Nothing that is not planted acts: the own effect of an event is 0 unless given, and a pair interacts only
if listed. The random-draw parameters (`event_effect_*`, `n_interactions`, `network_*`,
`interaction_strength*`, `prop_synergy`, `mutual_exclusivity_strength`, `n_exclusive_pairs`, and
`n_constraints` / `dag_*` in `dependency_params`) cannot be combined with `planted`, and a cycle, a
self-pair, an out-of-range event, a pair given twice, or an exclusive pair that is also an interaction is
an error. The network is part of `selection_params`, so `grow` in the paper regime takes it as
`selection={"epistasis_params": ..., "dependency_params": ...}`.
`iscc.tumor.diagnostics.division_cap_share(tumor)` reports the share of cancer cells whose division rate
is at `max_birth_rate`, where no planted term acts (see the fitness clamp below).

**A strongly negative `E` does not produce mutual exclusivity on its own.** It only suppresses the
*division* rate: a clone that has stopped dividing still dies only at the baseline rate (or at
`crowding_d1` in a deme above `K`), so it persists and still appears as co-occurring, the opposite of the
planted signal. `mutual_exclusivity_lethal` (default on) gives the combination `lethal_death_rate` so
that it is removed.

**Even then, MHN cannot recover exclusivity from a cross-sectional matrix in the regimes we can
simulate**, because that matrix is **patient-level** (did the patient acquire event *i* anywhere?) while
`E` is **genotype-level** (it acts only in a cell carrying both). We measured that 59.5% of patients
appear as co-occurring to MHN, but only 14.7% have any clone carrying both: **75% of the co-occurrences
seen by MHN are two events in different subclones, where `E` never acted.** This confounding of bulk
co-occurrence with same-cell co-occurrence is real biology, and is the reason TreeMHN exists.

**The fitness clamp can silently remove `E`.** The division rate is `baseline × fitness`, capped at
`max_birth_rate`. With `driver_effects > 1` and many drivers the additive term already holds clones at
the cap, leaving **no room** for the interaction term, so the planted network is present in the
configuration and absent from the dynamics. Keep `log(max_birth_rate / division_rate)` well above the
largest planted `Σ E` (for example `division_rate: 0.2`, `max_birth_rate: 0.95` ⇒ ~1.56 of headroom), and
set `driver_effects: 1.0` to make the network the *only* source of selection.

#### Conjunctive and ordered constraints (`selection_params.dependency_params`)

| Parameter | Default | Valid range | Outside the range |
|---|---|---|---|
| `n_constraints` | 0 (**off**) | 1–`n_events` | 0 → no DAG, no ordering |
| `dag_depth` | 2 | 2–4 | the number of layers; edges only run from an earlier layer to a later one (acyclic by construction) |
| `dag_branching` | 2 | 1–3 | maximum number of parents per gated event; excess constraints are dropped, so the realized DAG may have fewer edges than `n_constraints` |
| `gating_mode` | `fitness` | `fitness` \| `accessibility` | **these are different generative models** (see below) |

**`gating_mode` is a modeling decision, not a tuning parameter**, and it decides what a progression model
can recover.

- **`fitness`**: `B` arises freely but is *inert* until `A` is present, and order emerges from
  selection. It is softer, more biological, and closest to what MHN and TreeMHN assume (rates
  modulated, never zero).
- **`accessibility`**: `B` *cannot arise at all* until `A` is present, because mutations in the
  module of `B` are blocked. Order is imposed on the mutation process, as in CBN and H-CBN.

We measured the consequence in `validation/validate_epistasis.py`. Under **accessibility** gating the
constraint is recovered perfectly (`B requires A` and the ordering, 1.00 in every network draw, from true
and reconstructed trees), while under **fitness** gating the same planted DAG leaves essentially **no
recoverable trace** (the conjunction holds in a small minority of lineages, and the apparent order only
reflects which event is intrinsically faster). State which gating mode was used, since the two are not
interchangeable.

#### The observable matters more than the cohort size

Pairwise `E` acts on **fitness**, how large the carrying clones grow, not on the **rate** at which events
arise (which is what MHN and CBN model). Two practical consequences follow for a benchmark with
`epistasis_params`.

- **Recurrent mutation saturates the binary presence of an event.** A favored combination arises
  many times independently in one tumor, so it is already present at `E = 0`, and raising `E`
  changes how much of the tumor its lineages occupy, which a presence call discards. Sweep
  `min_freq` (a per-event **cancer-cell fraction** threshold, aggregated across clones, not a
  per-clone size filter) and check the column marginals: all-zero or all-one columns contain no
  information and no method can recover anything from them. `min_freq` has no default in
  `to_mhn_matrix` / `to_cbn_poset` / `patient_event_vector`; `iscc.integrations.DETECTION_LIMIT` (5%)
  is the limit of the paper. The tree exports take `detection`, the same limit on the collapsed clone
  tree (a node is kept when at least that share of the cancer cells descends through it), and
  `to_trait_matrix` / `to_trait_trees` export the dissemination traits (breach, stromal and
  metastatic survival) as events in the same formats.
- **Tumor size, not patient count, is the limiting factor.** A clone arising late has
  limited time to expand, so in small tumors the frequency signal never develops, and adding
  patients does not change this. If recovery appears flat, check the marginals of the observable
  before drawing conclusions about the method.

### Gene programs and expression realism (`expression_params`, optional, off by default)

By default expression is a simple model: genes are **independent** draws, a CNA adds ~1×baseline per copy
(**linear dosage**), the SNV effect **reuses the fitness parameter**, and the `p` and `m` alleles are
**summed**. This is approximately the relation that DNA–RNA tools (clonealign, inferCNV, Numbat,
cardelino) *invert*, so benchmarking them against it is partly circular. `expression_params` replaces all
four. Without it the engine is **unchanged**, and growth never uses it either way, because programs are a
**readout**, never fitness (see `validation/validate_programs.py`).

```
exp_{g,a} = base_{type,g}/2 · exp(Σ_k z_k·loading_{k,g}) · dosage(CN_{g,a}; s_g) · snv(class_{g,a}) · niche_g
```

**Comparability.** The program dictionary, the regulators and `s_g` are properties of the **genome**, so
they come from `layout_seed` (on its own sub-stream), and two patients sharing a configuration have the
**same programs**, exactly as they have the same oncogenes. Per-cell `z` and the class of each SNV are
per-**run** draws from `seed`. The ground truth is in `tumor.program_truth` (and `cell_program`,
`cell_exp_p` / `cell_exp_m` / `cell_rna_baf`).

!!! warning "The allele path uses a different expression scale"
    Without `expression_params`, `get_exp` returns `base·(1+copies)` (wild-type diploid → **3×base**)
    while normal cells bypass it at **1×base**. With `expression_params` on, cancer and normal cells
    share **one** scale (wild-type diploid → `base` for both), as a comparison of malignant cells with a
    reference needs. It applies only with the layer on, so existing outputs keep their baseline.

#### Cancer baseline (`cell_of_origin`)
| Parameter | Default | Valid range | Outside the range |
|---|---|---|---|
| `cell_of_origin` | unset | a normal type: `epithelial` \| `stromal` \| `immune` | unset → `base_{cancer,g}` is a separate draw, unrelated to every normal type, so malignant cells separate from normal ones on baseline alone; set → cancer cells share the `base_{type,g}` of that type and differ from it only by copy number, SNVs and programs. `cancer` or an unknown type raises. |

Growth and every layout draw are unchanged, and only the materialized cancer expression changes (and,
under targeted therapy, the target-expression check that reads it). The ductal field uses the
`epithelial` origin.

#### Program dictionary (`program_params`, the K × G `loading` matrix)
| Parameter | Default | Valid range | Outside the range |
|---|---|---|---|
| `n_programs` (K) | 8 | 5–30 | too few → no co-expression structure to recover; too many → each is tiny/unidentifiable |
| `n_genes_per_program` | 30 | ~1–10% of the genome (int, or `[lo, hi]`) | too few → below a factor model's detection limit; too many → programs merge into one |
| `program_overlap` | 0.1 | 0–0.5 | 0 → disjoint modules (unrealistically easy); high → entangled, tools cannot separate them |
| `loading_strength` | `{mean: 1.0, sd: 0.3}` | mean ~0.3–2 (log-fold) | ≪1 → programs vanish under Poisson noise; ≫2 → implausible fold changes |
| `loading_sparsity` | 1.0 | 0–2 (σ of a mean-1 lognormal) | 0 → every gene loads equally (unrealistic); high → the program reduces to one marker gene |
| `program_genomic_scatter` | 1.0 | 0–1 | **separates programs from CNAs.** 1 = scattered genome-wide (functional, realistic); **0 = a contiguous block that mimics a CNA**, the deliberate control for "can the tool tell a program from a copy-number segment?" |
| `program_signs` | `up` | `up` \| `bidirectional` | `bidirectional` gives each program up- and down-regulated genes (more realistic) |
| `seeded_programs` | `[proliferation, emt, hypoxia, drug_resistance, immune_evasion]` | any names | anchors programs so the phenotype and niche maps can address them **by name**; extras are unnamed background |
| `max_log_fold` | unset (unbounded) | ~3–7 (5 ≈ 150-fold) | unset → the program log-fold `Σ_k z_k·loading_{k,g}` of a gene is unbounded, and with heavy-tailed loadings and strong drive the whole library of a cell can go to one gene (a warning is raised past a 10⁶-fold multiplier); set → it saturates as `c·tanh(x/c)`, unchanged for small log-folds and never beyond `c`. Not part of the dictionary: setting it leaves the gene–program map and loadings unchanged |

#### Program activity (`activity_params`, the per-cell `z` sampler)
| Parameter | Default | Valid range | Outside the range |
|---|---|---|---|
| `n_active_programs_per_cell` | all | 1–K | activity sparsity; ≥K → every program on in every cell |
| `activity_dist` | `lognormal` | `lognormal` \| `normal` \| `gamma` | `normal` allows negative activity (a program can be repressed) |
| `activity_mean` / `activity_sd` | 1.0 / 0.5 | mean ~0.5–2 | moment-matched to the chosen distribution |
| `activity_noise` | 0.0 | 0–1 | **within-clone** spread. 0 → every cell of a clone is transcriptionally identical (clone equals state, benchmarks trivially easy); high → masks the between-clone signal |
| `celltype_program_bias` | `{}` | per-type scalar or K-vector | baseline activity per cell type (normal or cancer) |

#### Genotype-to-program coupling (`coupling_params`)
| Parameter | Route | Default | Valid range | Outside the range |
|---|---|---|---|---|
| `phenotype_program_map` | phenotype | `division_rate→proliferation`, `dispersal_rate→emt`, `treatment_resistance→drug_resistance`, `immune_resistance→immune_evasion` | any phenotype→program names | **the default coupling.** Reuses the existing role→phenotype map, so the chain is mutation → CINner fitness (including epistasis) → phenotype → program → expression |
| `phenotype_program_strength` | phenotype | 0.5 | 0–2 | **0 ⇒ phenotype route off** (fitness and expression decoupled); high → fitter clones become trivially separable in RNA, a *non-dosage* clone signal that confounds clonealign and inferCNV and can make scDEF conflate proliferation with clone |
| `prop_program_regulator` | regulator | 0.0 | 0–0.1 | fraction of genes whose mutation shifts `z` with **no** fitness change (plasticity). Keep it **sparse**: high → every clone is its own expression state |
| `program_bias_strength` | regulator | 0.5 | 0–2 | how strongly a mutated regulator shifts its program (graded by VAF) |
| `n_programs_per_regulator` | regulator | 1 | 1–3 | how many programs one regulator affects |
| `niche_program_map` / `niche_program_strength` | niche | none / 1.0 | field→program names | generalizes the microenvironment layer: `{hypoxia: hypoxia}` makes the O₂ field drive the hypoxia **program** of cancer cells (the programs of a normal cell are those of the cell type). Combines with the gene-level modifier of that layer |

#### Copy-number dosage (`dosage_params`, CNA → expression)
| Parameter | Default | Valid range | Outside the range |
|---|---|---|---|
| `dosage_sensitivity_mean` / `_sd` | 0.7 / 0.25 | mean 0–1, clipped per gene | per-gene `s_g`: **1 = full linear dosage** (exactly the relation the tools assume, so circular); **0 = fully buffered** (copy number invisible in RNA, nothing for inferCNV and clonealign to detect). Intermediate is the realistic case and the confounder that makes those benchmarks fair |
| `dosage_law` | `"haplotype"` | `"haplotype"` \| `"total"` | where `s_g` acts. `"haplotype"`: per haplotype, so a lost haplotype is silent whatever `s_g` (losses are never buffered) and a copy-neutral LOH lowers expression. `"total"`: on the total copy number, `1 + s_g·(min(C, sat)/2 − 1)`, so gains and losses are buffered alike, with the allelic split of the `"haplotype"` law. Fitted to TCGA breast tumors, `"total"` with mean 0.86 and sd 0.61 matches gains and losses together; `"haplotype"` cannot buffer losses (`validation/paper/supp_dosage.py`) |
| `dosage_saturation` | none | 4–12 (**total** copy number) | copy number beyond which expression stops rising (real amplicons saturate); absent → unbounded linear |
| `allele_specific` | `false` | bool | `true` → emit `cell_exp_p` / `cell_exp_m` and **`cell_rna_baf`** (the RNA B-allele frequency that Numbat and CalicoST need). A deleted haplotype is silent regardless of `s_g` |

#### SNV effects on expression (`snv_effect_params`, separate from the fitness `mut_effect`)
| Parameter | Default | Valid range | Outside the range |
|---|---|---|---|
| `p_silent` / `p_missense` / `p_splice` / `p_lof` | 0.55 / 0.30 / 0.05 / 0.10 | normalized to 1 | class drawn per site (infinite sites ⇒ one class per site, shared by every clone that carries it) |
| `nmd_strength` | 0.2 | 0–1 | expression **retained** on a loss-of-function (LoF) allele (nonsense-mediated decay). With a CNA loss of the other homolog this gives a biallelic **two-hit** tumor suppressor without extra parameters |
| `snv_expression_effect` | 0.5 | 0–2 | the splice-class expression shift, **kept separate from the fitness effect** so that advantageous and over-expressed are different parameters |

Silent and missense variants do not change mRNA abundance (a missense variant changes protein *activity*,
not abundance, which is why it is invisible to expression-based clone assignment).

### Microenvironment (`microenv_params`, optional, off by default)
| Parameter | Default | Valid range | Outside the range |
|---|---|---|---|
| hypoxia `D` / `k` against tumor size | — | O₂ diffusion length **<** tumor radius | length ≫ tumor → uniform O₂, no gradient |
| `o2_source` | `uniform` | `uniform` \| `perfused` | `perfused` gives a hotter (more hypoxic) core than `uniform` |

### Spatiotemporal record (top-level configuration keys, optional, off by default)
Every run records when and where each cancer genotype arose
([`tumor.genotype_births()`][iscc.tumor.GenotypeTumor.genotype_births], written as
`genotype_births.csv`). These two keys also record the spatial state over time
([`tumor.spatial_snapshots()`][iscc.tumor.GenotypeTumor.spatial_snapshots], written as
`spatial_history.csv.gz`), in the genotype engine only. Neither key draws a random number, so the tumor
grows identically with the record on or off.

| Parameter | Default | Valid range | Outside the range |
|---|---|---|---|
| `spatial_snapshot_every` | `0` (off) | `0`, or `k ≥ 1` trace snapshots between spatial snapshots | under tau-leaping a trace snapshot is one generation (× `snapshot_every`); under the exact engine it is one **event**, so a small `k` there stores a snapshot per birth or death; use a large `k` |
| `spatial_snapshot_budget` | `5000000` entries (~60 MB) | ≳ entries in one snapshot (≈ occupied demes × genotypes per deme) | when exceeded, every other snapshot is dropped and the interval doubles; a budget below one snapshot keeps only the first |

### Random streams and the confinement clock
Every per-run stream is keyed by `(seed, purpose, step)` through a `SeedSequence`, so different seeds
never share a stream. `spatial_params.origin_confinement.generations` counts generations at any `tau`
(the exact engine counts the Gillespie waiting time of each event).

### Materialization memory
[`make_cell_data`][iscc.tumor.GenotypeTumor.make_cell_data] builds the per-cell tables, the expression multiplier of the program layer included, chunk by chunk, so its working set does not grow with the number of cells beyond the tables it returns.
The caller can choose a further saving.

| Parameter | Default | Valid range | Outside the range |
|---|---|---|---|
| `make_cell_data(layers=...)`, also on [`Resection.slice`][iscc.sample.Resection.slice] / [`dissociate`][iscc.sample.Resection.dissociate] | `None` (every table) | table names (`"cell_exp"`, `"cell_snv"`, `"cell_cnv"`, `"cell_rna_vaf"`, `"cell_snv_p"`, `"cell_snv_m"`, `"cell_exp_p"`, `"cell_exp_m"`, `"cell_rna_baf"`) and/or assay names standing for the tables that assay reads (`"visium"` and `"imaging"`: expression; `"scrna"`: expression, SNVs and RNA VAF; `"scdna"`: SNVs and copy number). The small per-cell tables are always built. Values are byte-identical to a full build | an unknown name raises |

### scRNA batches (`run_scrna_batches`, `run_cohort_batches`)
Every batch of [`run_scrna_batches`][iscc.data.run_scrna_batches] and [`run_cohort_batches`][iscc.cohort.run_cohort_batches] takes the same settings at the 10x preset and differs from the others only by its own draw of the per-gene factor (`sigma_batch`, 0.29, fitted on two 10x runs of one blood sample so that the simulated difference between them has the real size) and of the depth shift (`depth_batch_sigma`, 0.05).
`batch_kwargs` gives each batch its own settings, and [`atlas_scrna_batches`][iscc.data.atlas_scrna_batches]`(n_batches, seed)` draws them from the estimates for the 26 tumors of the breast cancer atlas of Wu et al. 2021, as the paper's batch model does.
The estimates are packaged as `src/iscc/data/reference/wu2021_scrna_batches.tsv`, built from the paper's estimates by `validation/data/build_atlas_batches.py`.
Each batch takes one tumor's `mu_lib`, `sigma_lib`, `dispersion`, `dropout_mid`, `dropout_shape` and `doublet_rate`, an `ambient_frac` drawn between 0.03 and 0.11, and the tumor's dissociation stress, which goes to `Resection.dissociate(stress_strength=...)`.
In both presets the library of a cell is proportional to the RNA content of the cell (`content_exponent` 1; [`cell_rna_content`][iscc.data.cell_rna_content]), so T and NK cells take about 0.28 of the UMIs of an epithelial cell, as in the Human Breast Cell Atlas, and `sigma_lib` is the spread within a cell type.

### Visium spots and depth (`iscc.data.Visium`)
A deme is ~50 µm, so the default [`Visium`][iscc.data.Visium] spot is that of Visium v1, 55 µm across at a 100 µm pitch.
The capture field is normalized over the tissue spots, so a tissue spot contains `mu_counts` UMIs on average, whatever share of the slide the section covers.
A section cut with a small `max_cells` leaves one or two cells per spot, where real spots hold several to tens; `Visium` warns (`SparseSpotWarning`) when the median of the tissue spots is below 3.

| Parameter | Default | Valid range | Outside the range |
|---|---|---|---|
| `spot_pitch` / `spot_radius` | 2.0 / 0.55 demes | radius below half the pitch (spots apart, as on a slide) | a radius of 1 (a former default) made a spot as wide as the pitch: it pooled three times the cells of a 55 µm spot (~40 at 12 cells per deme) and tiled 91% of the tissue instead of 27% |
| `mu_counts` | 20,000 UMIs per tissue spot | ~10³–10⁵ | with spot swapping on, it is the depth before swapping, and the tissue keeps a share of it: 0.96 on the 10x breast section (76% of the slide) at the fitted swap, 0.90 for a section on a quarter of the slide, whose distal bleed falls mostly off the tissue. Given a raw matrix with all spots, [`estimate_visium`][iscc.data.estimate_visium.estimate_visium] returns this depth |
| `content_exponent` | 1 (depth proportional to the summed RNA content of the cells of a spot) | 0–1.5 | at 0 the depth follows the capture field alone and, on the showcase at full density, correlates with the cancer share of a spot the wrong way (Spearman −0.33, against +0.44 in the 10x breast section); 1.5 overshoots (+0.70) |
| `kappa` (`count_model="dm"`) | 1,400,000 (`HUMAN_VISIUM_KAPPA`) | ≳ 10⁵ | it inflates the variance of every gene by (N + κ)/(1 + κ) at a spot depth N, whatever the gene set, so unlike the DNA concentration it is not rescaled by the number of genes. A former default of 50 inflated it ~390-fold at 20,000 UMIs, and over 6,000 growth genes a spot detected 281 genes instead of ~4,700 and had 77% of its counts in 50 of them |

---

## Diagnosing a tumor

After growth, [`tumor.diagnose()`][iscc.tumor.GenotypeTumor.diagnose] checks each degenerate regime against a
threshold and prints a hint for any failure. It is **read-only** and never changes the simulation.

```python
from iscc.tumor.models import GenotypeTumor

tumor = GenotypeTumor(config=cfg, seed=0)
tumor.grow(n_steps=2000, seed=0)

dx = tumor.diagnose(verbose=True)   # prints [PASS]/[FAIL] per check, with hints
# or: from iscc.tumor.diagnostics import diagnose; diagnose(tumor, verbose=True)

# Override any threshold per call:
tumor.diagnose(thresholds={"shannon_min": 1.0})
```

### Degenerate regimes and default thresholds
| Regime | Metric (threshold) | Parameters to change (the hint) |
|---|---|---|
| **extinct / too small** | cancer-cell count (< 25 fails; < 1000 advisory) | lower `death_rate` / raise `division_rate`, `initial_cancer_cells`, or grow longer |
| **monoclonal** | clonal Shannon diversity (< 0.5) | no mutations → raise `mutation_rate` / `n_snvs_per_allele`; selective sweep → lower `driver_effects` / `prop_driver` |
| **low mutation** | mean mutated sites per cell (< 1.0) | raise `mutation_rate` / `n_snvs_per_allele` |
| **hypermutated** | fraction of genome mutated per cell (> 0.5) | lower `mutation_rate` / `n_snvs_per_allele` |
| **no resolvable subclones** (checked only at Shannon diversity ≥ 0.5) | genotypes above 1% of the cancer cells (< 2) | lower `mutation_rate` / `n_snvs_per_allele`, or grow larger so clones expand |
| **well-mixed** | clone spatial confinement, ≥ 2 subclones (< 0.1) | lower `dispersal_rate` |
| **demes over their slot cap** | demes holding more than `floor(crowding_overfill · K)` cells, with the K of each deme (> 0) | none: the engine keeps every deme within its slot cap, so a failure is an engine bug to report; skipped in the well-mixed regime |
| **no O₂ gradient** | hypoxia core–rim contrast (< 0.05) | grow a larger tumor, raise consumption `k`, or lower diffusion `D` |
| **CNA runaway** | fraction of genome altered (> 0.95) | lower `amp_prob` / `max_cn` |
| **clone equals cell state** (skipped when programs are off) | within-clone share of program-activity variance (< 0.05) | raise `activity_noise` / `activity_sd`, or set `n_active_programs_per_cell` < `n_programs`; with all three off, `z` collapses to the per-clone drive |
| **trivial genome** | gene count (< 100) | raise `n_segments` × `segment_size` |

The thresholds are deliberately lenient, flagging tumors that are clearly broken rather than merely
unusual, and each one can be overridden with `thresholds={...}`.

---

## See also
- **Default configurations:** `notebooks/example_config.yaml`, `src/iscc/tumor/tumorconfigs/genotype.yaml`
