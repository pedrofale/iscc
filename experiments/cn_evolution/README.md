# CN evolution analysis

Grid-searched copy-number evolution analysis over `iscc` simulations, built on
[`iscc.cnevo`](../../src/iscc/cnevo).

## Seven co-equal questions

Each has its own metric set, output file, summary table and `*_only` target. **None is an input to
another** — cross-question relationships are a separate analysis, not a construction.

| | Question | Output | Summary |
|---|---|---|---|
| Q1 | Clonal dynamics and sweeps | `pop/sweep_metrics.json` | `sweep_summary.tsv` |
| Q2 | Clonal diversity over time | `pop/diversity_trajectory.tsv` | `diversity_summary.tsv` |
| Q3 | Demography: r- vs K-phase | `pop/growth_phase.json` | `growth_phase_summary.tsv` |
| Q4 | Copy-number landscape | `pop/cn_landscape.{tsv,json}` | `landscape_summary.tsv` |
| Q5 | CN data quality | `K{n}/sim/data_quality.json` | `quality_summary.tsv` |
| Q6 | Tree-reconstruction potential | `K{n}/sim/reconstruction_potential.json` | `reconstruction_summary.tsv` |
| Q7 | Spatial structure and multi-focality | `pop/spatial_structure.json` | `structure_summary.tsv` |

Q1–Q4 and Q7 describe the whole run and are computed once per tumour. Q5–Q6 depend on the sampled
clone set, so they live under `K{n}/`.

## Three tissue scenarios

All run side by side, as the outermost path segment:

* **unstructured** (`structure_radius: 0`) — cancer expands freely on the lattice, capped only by a
  scalar carrying capacity. The closest analogue to a non-spatial simulation.
* **structured** (`structure_radius > 0`) — a ductal field of healthy epithelial-ring glands in
  stroma, with the lesion founding *inside* a gland. Multi-focal spread and invasion; Q7 is defined
  only here (unstructured runs write a null payload, so the summary tables keep one schema).
* **structured_confined** — the same field, but the founding acinus stays closed until it is
  distended (`origin_confinement`).

**The confinement variant is not cosmetic.** Measured on this engine (grid 14, 80 generations):

| | escape generation | `trunk_fraction` | trunk events |
|---|---|---|---|
| structured | 6 | **0.00** | 0 |
| structured_confined | 44 | **0.45** | 6 |

Without confinement the lesion breaches almost immediately and *no* copy-number alteration is shared
by every sampled clone — there is no truncal layer at all. Since trunk burden is the mechanism
behind non-reconstructable data, a structured scenario without confinement does not actually differ
from an unstructured one on the axis that matters. Run all three.

## Running

```bash
# smoke test: both scenarios, one seed, ~2 minutes
snakemake -s Snakefile --configfile config_test.yaml --cores 4

# full grid
snakemake -s Snakefile --configfile config.yaml --cores all

# one question, over checkpoints that already exist
snakemake -s Snakefile --configfile config.yaml --cores 4 dynamics_only
snakemake -s Snakefile --configfile config.yaml --cores 4 reconstruction_only
```

Targets: `all`, `simulate_only`, `dynamics_only` (Q1–Q3), `landscape_only` (Q4), `quality_only`
(Q5), `reconstruction_only` (Q6), `structure_only` (Q7). Each rebuilds only its own question —
`reconstruction_only` re-runs `compute_phylo` and its aggregator and nothing else, no re-simulation.

## Cross-cutting analysis

Run **after** the grid, never as part of it. Neither script is a dependency of any question, so no
metric is ever computed in order to explain another — the relationships are findings, not
construction.

```bash
python scripts/analyze_cross_question.py --results-dir results --output cross_question.tsv
python scripts/analyze_scenarios.py      --results-dir results --output scenarios.tsv
```

Both refuse to over-claim on a small grid: they print the number of runs behind each number and warn
when there are too few seeds to mean anything.

The scripts run under the interpreter that launched Snakemake; override with
`--config python_bin=/path/to/python`.

## Path grammar

```
results/_pop/{scenario}/{evo}/{selection}/{genome}/{pop}/seed{N}/   tumour checkpoint
results/{scenario}/{evo}/{selection}/{genome}/{pop}/seed{N}/pop/    Q1-Q4, Q7 + figures
results/{scenario}/{evo}/{selection}/{genome}/{pop}/seed{N}/K{n}/sim/  Q5-Q6 + figures
```

Every summary table carries `scenario` as its first grouping key, and the axes are recovered from
the path, so a dataset's parameters are readable from where it lives.

## Three traps the config layer exists to avoid

Each of these silently disables a grid axis rather than failing, so they are handled in
`scripts/_config.py` — check there before "fixing" a config that looks odd.

1. **`max_birth_rate` must exceed `division_rate`.** The engine default is `0.3`; raising
   `division_rate` without raising the cap clamps *every* clone to the same rate and the selection
   axis does nothing at all. The cap is raised automatically.
2. **In a glandular scenario the per-deme capacity is `K_duct` / `K_stroma`**, and a bare
   `carrying_capacity` is ignored. The demographic axis is therefore a `capacity_scale` applied to
   whichever capacity keys the scenario actually uses.
3. **Structured runs need `prop_breach`** (and `prop_stromal_survival`) to be non-zero or invasion
   is impossible and the lesion stays confined forever. These live in the scenario block, not
   `selection_grid`, so the selection axis means the same thing in both scenarios.

## Cost

Structured runs carry immortal resident cells *and* mint a genotype per CNA: roughly 16 s and
~56k registry entries at grid 12 / 100 steps, against well under a second for an unstructured run
of the same length. The genotype **registry** (every genotype ever created) is the quantity that
matters for cost, not the live clone count — a run with 415 live clones can hold 11k registry
entries, which is why `max_clones_for_engine_plots` gates on the registry. The configs give them a reduced grid via `_restrict`; widen it only if the
scenario contrast turns out to matter. `coarsen_passengers: true` and `max_cells` are on by default
for the same reason, and `diversity_stride` / `landscape_stride` subsample the per-generation
metrics (recomputing `J1` at every snapshot is the expensive part of Q2).
