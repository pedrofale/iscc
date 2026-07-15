# iscc parameters — defaults, valid ranges, and what to expect

`iscc` ships **sensible defaults** that sit inside its *operating envelope* — the region of parameter
space that produces realistic tumours. This document says what each knob does, its default, its
**valid range**, and **what happens outside it**. If you only change a few knobs and keep the rest at
their defaults, you will get a realistic multi-clone tumour.

Two safety nets back this up:

- The shipped configs (`notebooks/example_config.yaml`, `src/iscc/tumor/tumorconfigs/{glandular,mixed}.yaml`)
  are inside every valid range below.
- After growing, call **`tumor.diagnose()`** — a read-only quality-control check that flags degenerate
  ("crappy") tumours and tells you which knob to turn (see [§ Diagnosing a tumour](#diagnosing-a-tumour)).

The ranges below come from the operating-envelope characterization: `analysis/characterize_regimes.py`
(the sweep) → `manuscript/figures/validation_operating_envelope.png` (phase diagrams) →
`DESIGN_operating_envelope.md` (design) and the manuscript "Operating regimes" section.

---

## The main knobs

Defaults are those in `notebooks/example_config.yaml`. Set them under the matching YAML block
(`cell_params.cancer`, `deme_params`, `spatial_params`, `selection_params`, `genome_params`).

### Mutations — `cell_params.cancer`
| Knob | Default | Valid range | Outside the range |
|---|---|---|---|
| `mutation_rate` | 0.2 | ~0.05–2 | **low** → monoclonal (no subclones); **high** → hypermutated mush (broken 1/f tail) |
| `n_snvs_per_allele` | 0.3 | ~0.1–3 | **low** → no SNV diversity; **high** → hypermutated mush |
| `snv_prob` / `cnv_prob` | 0.5 / 0.5 | relative weights (SNV vs CNA event) | all-SNV → no CNAs (inferCNV/clonealign demos degrade); all-CNA → no SNV phylogeny |

### Growth & survival — `cell_params.cancer` + `deme_params`
| Knob | Default | Valid range | Outside the range |
|---|---|---|---|
| `division_rate` | 0.3 | **> `death_rate`** | ≤ death → extinction |
| `death_rate` | 0.02 | **≪ `division_rate`** | ≥ division → extinction |
| `initial_cancer_cells` | 5 | ≥ 5 | 1 → founder extinction (a lone founder is prone to stochastic loss, and density-dependent crowding death raises that risk at small `carrying_capacity`) |
| `maximum_death_rate` | 1.0 | **≥ `max_birth_rate`** (0.8) | caps crowding death; **below `max_birth_rate` re-opens the over-fill bug** (evolved clones outrun the cap) |

### Spatial structure — `spatial_params` + `deme_params` + `cell_params.cancer`
| Knob | Default | Valid range | Outside the range |
|---|---|---|---|
| `carrying_capacity` | 10 | a real **per-deme cap** (cells/deme); `None` or `0` → **well-mixed** (no ceiling, unbounded growth) | too small (1–3) → a lone founder is prone to extinction (crowding ramps from occupancy 0) |
| `grid_size` × `carrying_capacity` | 50 × 10 | the tumour caps at ~`grid_size² × carrying_capacity`; size the grid **above** the target so it can spread (≳ 10⁴ demes for a 10⁵-cell tumour) | too small → no room to spread / no O₂ gradient / too few clones |
| `dispersal_rate` | 0.1 | **≲ `division_rate`** | ≫ division → well-mixed, **no clonal territories** (silently breaks the PEtracer and multi-region benchmarks) |
| `structure_radius` / `n_structures` | 20 / 1 | glandular geometry (duct size / count); the cancer founds inside and spreads across the gland | — |

!!! note "`carrying_capacity` is a real per-deme cap (density-dependent crowding, 2026-07-14)"
    Crowding death rises **relative to each clone's own (evolved) division rate**
    (`death = death_rate + (division_rate − death_rate)·(1 + crowding_margin)·occupancy/K`, clamped at
    `maximum_death_rate`), so a deme's occupancy caps near `carrying_capacity` even for clones whose
    division has evolved up to `max_birth_rate`, and the tumour **spreads** (occupied demes ∝ cells/K)
    instead of piling into a few demes. This needs `maximum_death_rate ≥ max_birth_rate` (default 1.0).
    Set `carrying_capacity: None` (or `0`) to disable crowding entirely — the **well-mixed** regime used
    for single-deme, unbounded-growth benchmarks. See `DESIGN_crowding.md`.

### Selection — `selection_params`
| Knob | Default | Valid range | Outside the range |
|---|---|---|---|
| `prop_driver` | 0.1 | 0.05–0.3 | high × strong effect → selective sweep (monoclonal) |
| `driver_effects` | 1.1 | 1.0–2 | ≫ 1 at low mutation rate → monoclonal sweep |
| `prop_dispersal` / `dispersal_effects` | 0.1 / 1.1 | as above | strong → invasion dominates, structure washes out |
| `prop_treatment_resistance` / `prop_immune_resistance` (+ effects) | 0.1 / 1.1 | as above | resistance is meant to **emerge**, not be pre-seeded |

### Genome — `genome_params`
| Knob | Default | Valid range | Outside the range |
|---|---|---|---|
| `n_segments` × `segment_size` | 5 × 200 (= 1000 genes) | ≳ 100 genes | too few → trivial genome; assays degenerate, too few drivers |

### Copy-number — `cell_params.cancer`
| Knob | Default | Valid range | Outside the range |
|---|---|---|---|
| `amp_prob` | 0.5 | 0.2–0.8 | high → copy-number-heavy genome (viability-capped) |

### Epistasis / dependency network (R14, optional; **off by default**) — `selection_params.epistasis_params`

Selection is **additive** by default (fitness reads the *count* of mutated drivers), so the true
event×event dependency network is empty. These knobs **plant a known network** — the answer key the
cohort progression models (MHN, TreeMHN, CBN/H-CBN, REVOLVER) are scored against. With
`epistasis_params` absent, or `n_events: 0`, nothing is built and the engine is **bit-identical** to
the additive model. See `DESIGN_epistasis.md` and `validation/validate_epistasis.py`.

An **event** is a disjoint module of `event_size` driver genes; it fires when ≥1 SNV lands anywhere in
the module, and is **monotone** (once acquired, never lost — the assumption MHN/CBN are defined under).
Modules rather than single genes because MHN/TreeMHN pool many tumours to fit **one** network, which
requires the same events to **recur across patients**; a single named gene out of ~10⁴ is hit in
almost no patient. `event_size: 1` recovers the single-gene case.

Fitness gains `exp(Σᵢ βᵢxᵢ + Σᵢ<ⱼ Eᵢⱼxᵢxⱼ)` as a multiplier on the additive model.

| Knob | Default | Valid range | Outside the range |
|---|---|---|---|
| `n_events` | 0 (**off**) | 4–20 | 0 → additive engine, bit-identical; many → each event rarer, cohort needs to be larger |
| `event_size` | 20 | 1–~5% of the genome | **the rate knob**: too small → events never fire (empty matrix); too large → every patient acquires every event (all-ones matrix, zero variance, nothing to correlate) |
| `event_effect_mean` / `_sd` | 0.1 / 0.0 | 0–0.5 | the events' own marginal log-fitness effect (MHN's diagonal). 0 → events are marginally neutral and only `E` acts |
| `n_interactions` | `None` → use `network_sparsity` | 0–`n_events(n_events−1)/2` | 0 → empty `E` (the false-positive control) |
| `network_sparsity` | 0.2 | 0.05–0.5 | fraction of event pairs that interact, when `n_interactions` is unset |
| `network_topology` | `random` | `random` \| `hub` \| `chain` | `hub` = one master event wired to all; `chain` = a stepwise cascade |
| `interaction_strength` / `_sd` | 0.3 / 0.05 | 0.1–1.0 | **must fit under the fitness clamp — see the trap below** |
| `prop_synergy` | 0.5 | 0–1 | P(`E>0`); the rest are antagonistic |
| `mutual_exclusivity_strength` | 0.0 | 2–8 when used | magnitude of the strongly-negative (synthetic-lethal) edges |
| `n_exclusive_pairs` | 0 | ≤ pairs left after the interaction edges | how many such edges |

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
draw), while under **fitness** gating the same planted DAG leaves essentially **no recoverable trace**
(the conjunction holds in ~23% of lineages, and the apparent "order" just reports which event is
intrinsically faster). The same script shows that pairwise `E` — which acts on fitness — is recovered
at **chance** by a cross-sectional method regardless of cohort size or how hard it is planted, because
"which events did this patient ever acquire" is blind to how large a clone grew. State which gating
mode you used; the two are not interchangeable.

### Gene programs / expression realism (R13, optional; **off by default**) — `expression_params`

By default expression is the legacy model: genes are **independent** draws, a CNA adds ~1×baseline per
copy (**linear dosage**), the SNV effect **reuses the fitness knob**, and the `p`/`m` alleles are
**summed**. That is approximately the law the DNA↔RNA tools (clonealign, inferCNV, Numbat, cardelino)
*invert*, so benchmarking them against it is partly circular. `expression_params` replaces all four.
With it absent the engine is **unchanged**, and growth never reads it either way — programs are a
**readout**, never fitness (that loop is R8b/R12-v3). See `DESIGN_expression.md`,
`validation/validate_programs.py`.

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
| `niche_program_map` / `niche_program_strength` | 3 | none / 1.0 | field→program names | generalises F8: `{hypoxia: hypoxia}` makes the O₂ field drive the hypoxia **program**. Composes with F8's own gene-level modifier |

#### `dosage_params` — axis A (CNA → expression)
| Knob | Default | Valid range | Outside the range |
|---|---|---|---|
| `dosage_sensitivity_mean` / `_sd` | 0.7 / 0.25 | mean 0–1, clipped per gene | per-gene `s_g`: **1 = full linear dosage** (exactly the law the tools assume — circular); **0 = fully buffered** (copy number invisible in RNA, inferCNV/clonealign have nothing to find). Intermediate is the realistic case and the confounder that makes those benchmarks fair |
| `dosage_saturation` | none | 4–12 (**total** CN) | CN beyond which expression stops rising (real amplicons saturate); absent → unbounded linear |
| `allele_specific` | `false` | bool | `true` → emit `cell_exp_p`/`cell_exp_m` + **`cell_rna_baf`** (the BAF in RNA that Numbat/CalicoST need). A deleted haplotype is silent regardless of `s_g` |

#### `snv_effect_params` — axis B (SNV → expression), **separate from the fitness `mut_effect`**
| Knob | Default | Valid range | Outside the range |
|---|---|---|---|
| `p_silent` / `p_missense` / `p_splice` / `p_lof` | 0.55 / 0.30 / 0.05 / 0.10 | normalised to 1 | class drawn per site (infinite-sites ⇒ one class per site, shared by every clone carrying it) |
| `nmd_strength` | 0.2 | 0–1 | expression **retained** on a LoF allele (nonsense-mediated decay). With a CNA loss of the other homolog this gives a biallelic **two-hit** TSG for free |
| `snv_expression_effect` | 0.5 | 0–2 | the splice-class expression shift. **Kept separate from the fitness effect** — they were one knob before R13, which entangled "advantageous" with "over-expressed" |

Silent and missense leave mRNA abundance alone (a missense changes protein *activity*, not
abundance — which is exactly why it is invisible to expression-based clone assignment).

### Microenvironment (F8, optional; off by default) — `microenv_params`
| Knob | Default | Valid range | Outside the range |
|---|---|---|---|
| hypoxia `D` / `k` vs. tumour size | — | O₂ diffusion length **<** tumour radius | length ≫ tumour → uniform O₂, no gradient |
| `o2_source` | `uniform` | `uniform` \| `perfused` | `perfused` gives a hotter (more hypoxic) core than `uniform` |

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
| **clone == cell state** (R13; skipped when programs are off) | within-clone share of program-activity variance (< 0.05) | raise `activity_noise` / `activity_sd`, or set `n_active_programs_per_cell` < `n_programs` — with all three off `z` collapses to the per-clone drive |
| **trivial genome** | gene count (< 100) | raise `n_segments` × `segment_size` |

Thresholds are deliberately lenient — they flag the clearly broken, not the merely unusual — and every
one is overridable via `thresholds={...}`.

---

## See also
- **Defaults to copy:** `notebooks/example_config.yaml`, `src/iscc/tumor/tumorconfigs/{glandular,mixed}.yaml`
- **The characterization:** `analysis/characterize_regimes.py`, `validation/validate_operating_envelope.py`
  → `manuscript/figures/validation_operating_envelope.png`
- **Design rationale:** `DESIGN_operating_envelope.md`
- **Output layout (not parameters):** `SCHEMA.md`
