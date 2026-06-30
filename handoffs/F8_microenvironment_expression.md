# Handoff prompt — F8: microenvironment-driven gene expression

Saved 2026-06-30. Copy the block below as the opening message of a fresh session dedicated to F8.
Design reference: `DESIGN_features.md` section H + the F8 milestone bullet. Positioning/why-it-matters:
the memory note `iscc-paper-positioning.md` and `BACKLOG.md`.

---

```
Implement F8 — microenvironment-driven gene expression — in iscc. This is a new ENGINE feature
(not a write-up): a cell's expression must depend on its spatial microenvironment, not only its
genotype/CNA/cell-type. It is the keystone that lets iscc serve as the benchmark substrate for
DNA+RNA+spatial integration methods (paper 1's SISTEM-analog claim).

REPO & ENV
- Repo: /Users/pedroferreira/projects/iscc/repo  (work on branch `dev`).
- Python/pytest: ~/miniconda3/envs/iscc/bin/python  (e.g. `~/miniconda3/envs/iscc/bin/python -m pytest`).
- Conventions: commit on `dev`; end commits with the `Co-Authored-By: Claude Opus 4.8
  <noreply@anthropic.com>` trailer; keep the FULL test suite green; be honest in comments/docs
  (no overselling). Match surrounding code style.

READ FIRST
- `DESIGN_features.md` section **H** (full F8 design) and the **F8** milestone bullet in section F.
- Current expression code (the thing to extend), in order:
    - `src/iscc/tumor/components/cell.py` -> `get_exp(self, seg_mut_effects)` (~L236): per-cell
      expression = baseline(cell_type) * CNA dosage * mutation effects. NO spatial term today.
    - `src/iscc/tumor/models/count.py` (the DEFAULT genotype-count engine): per-cell-type baseline
      `celltype_exps` (~L105) and where per-cell `cell_exp` is materialized in `make_cell_data`
      (~L575). THIS is the primary path to modify (the count engine is the default; `glandular.py`
      is the cell-level reference path — keep it consistent but the count engine is what matters).
    - `src/iscc/tumor/tumor.py` -> `set_cell_exps` (~L205) for the cell-level path.
- Verify how the count engine exposes deme positions and per-deme composition (genotype counts /
  cell-type counts per deme on the 2D grid) — the field solver and CCI aggregation operate at DEME
  resolution, so you need per-deme coordinates + per-deme cell density and cell-type/ligand emitters.

WHAT TO BUILD (two coupled, deme-resolution mechanisms; both OPTIONAL and OFF by default so existing
behaviour and tests are unchanged when disabled)
1. Diffusible field(s) — translate PhysiCell/BioFVM. Solve a STEADY-STATE reaction-diffusion on the
   deme lattice for O2: supplied at the tumour boundary (or seeded "vessels"), consumed ∝ local cell
   density, with a tunable diffusion lengthscale → a per-deme hypoxia field in [0,1]. Keep it cheap
   (sparse linear solve / iterative relaxation on the small deme grid; recompute per snapshot, NOT
   per event, so it stays tau-leaping-compatible). Generic secreted-factor fields can reuse the same
   solver. Recapitulate the viable-rim / hypoxic-core pattern.
2. Cell–cell communication (ligand–receptor) — translate scMultiSim/sCCIgen. For each L–R pair, a
   cell's receptor-target genes are modulated by the ligand-emitting cell density in its DEME
   NEIGHBOURHOOD (aggregate neighbours' emitter density). Emitters are cell-types or genotypes.

CHANGE TO get_exp (new multiplicative terms on designated gene programs; baseline behaviour preserved):
    cell_exp = baseline(cell_type) * dosage(CNA) * mut_effects(SNV)
                                   * g_hypoxia(O2_field@deme, hypoxia_program)
                                   * g_CCI(neighbour_ligands@deme, receptor_program)
- New config/params: a hypoxia-responsive gene program + field params (supply/consumption/lengthscale);
  L–R pairs + receptor-target program + a CCI-strength knob. All default to no-op.
- The per-deme field/CCI values must be available where per-cell exp is materialized (count engine
  `make_cell_data`); precompute per-deme vectors once per snapshot and index by the cell's deme.

VALIDATION (add `validation/validate_microenvironment.py` → `manuscript/figures/validation_microenvironment.png`)
- Hypoxia program elevated in low-O2 core demes, depleted at the rim (the rim/core signature).
- CCI-target genes elevated at clone / cell-type boundaries.
- A spatial-niche or autocorrelation readout showing niche structure appears only with F8 on.
- Sanity: with F8 OFF, expression is bit-identical to current output.

DELIVERABLES
- Engine code (count engine primary; cell-level path consistent), config plumbing, OFF-by-default.
- Tests under `tests/` (field solver correctness; OFF == current; hypoxia/CCI gradients present when ON).
- `notebooks/_build_microenvironment_expression.py` + executed `notebooks/microenvironment_expression.ipynb`
  (sweep O2 supply/consumption and CCI strength → hypoxia gradient, boundary crosstalk, niches vs
  ground truth). Add a row to the deliverables table in `DESIGN_features.md` §G (already noted).
- Update the F8 milestone bullet in `DESIGN_features.md` from PLANNED → DONE with what shipped.
- Run the full pytest suite; report pass/fail honestly. Commit on `dev` (with the Co-Authored-By trailer).

FOLLOW-ON (do NOT do here, just leave the hook): once F8 is in, the payoff demo is running clonealign
(DNA↔RNA clone assignment), a CCI/niche method (CellPhoneDB/COMMOT/NICHES), and a deconvolution tool
(cell2location/Tangram) on iscc data and scoring against ground truth — the paper-1 integration figure.
```
