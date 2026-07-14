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
| `initial_cancer_cells` | 5 | ≥ 5 | 1 → ~7% founder extinction (stochastic loss of a single seeded cell) |
| `maximum_death_rate` | 0.5 | 0.3–0.8 | caps crowding death; very low → no density regulation |

### Spatial structure — `spatial_params` + `deme_params` + `cell_params.cancer`
| Knob | Default | Valid range | Outside the range |
|---|---|---|---|
| `grid_size` × `carrying_capacity` | 25 × 5 | enough for ≳ 10³ cancer cells | too small → no O₂ gradient, too few clones |
| `dispersal_rate` | 0.1 | **≲ `division_rate`** | ≫ division → well-mixed, **no clonal territories** (silently breaks the PEtracer and multi-region benchmarks) |
| `structure_radius` / `n_structures` | 5 / 1 | glandular geometry (duct size / count) | — |

!!! warning "Known limitation — `carrying_capacity` does not hard-cap deme occupancy (2026-07-09)"
    Crowding is a fixed density-dependent *death* rate capped by `maximum_death_rate`, but selection
    raises clones' division rate up to `max_birth_rate`, so evolved clones outgrow the death cap and
    demes over-fill (a dense pile rather than a spread tumour) at large scale. `carrying_capacity` is
    therefore effectively a threshold, not a capacity, in the current engine. Small-grid runs (PEtracer,
    multi-region) are unaffected; large-scale spatial fidelity and any tumour-size/density realism are.
    Fix tracked in `DESIGN_crowding.md` / `BACKLOG.md`.

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
