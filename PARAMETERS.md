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
