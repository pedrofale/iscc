# DESIGN_cohort.md — the multi-patient COHORT layer

*(design-first, like DESIGN_features §H / F8. Companion to `BACKLOG.md` "Multi-patient cohort".)*

## 0. Goal (why this exists)

Run **many patients (tumours) separately over a SHARED driver landscape**, pool them into batches
flexibly, and surface **cohort-level ground truth** that no simulator and no real cohort can give:
recurrent-vs-private drivers, per-cell patient-of-origin, true shared-vs-private cell-state labels,
and — coupled to the treatment module — per-patient **subgroup + true therapy response**. This
unlocks benchmarks for **multi-patient integration, demultiplexing, recurrence/driver detection, and
personalized medicine / patient stratification**, each with a known answer key.

The engine already does most of the hard part. This milestone is: (1) a **prerequisite engine fix**
so same-config runs are comparable by construction; (2) a thin `Cohort` **wrapper** that loops the
existing `GenotypeTumor` over per-patient evolution seeds with optional per-subgroup config deltas;
(3) a **patient→batch multiplexing** step that reuses the scRNA "confounded"/multi-batch machinery;
and (4) the **ground-truth bookkeeping**.

---

## 1. PREREQUISITE ENGINE FIX — comparability by default (foundational, not cohort-specific)

### The bug
`Selection.make_drivers()` (`components/selection.py:83`) draws driver/oncogene/TSG **positions**
from `self.rng`. Both engines seed that `rng` from the **run seed**:
* base `Tumor.__init__` (`tumor/tumor.py:26`): `self.rng = np.random.default_rng(seed)`, then
  `Selection(rng=self.rng)` (`:42`) and `make_celltype_exps` (`:88`, `self.rng.beta(...)`);
* `GenotypeTumor.__init__` (`tumor/models/count.py:44`): identical pattern (`:108`, `:116`).

So a different `seed` → a different gene-role layout → a different set of oncogenes/TSGs. Two "runs
of the same config" therefore **share no driver identities**, which makes recurrence/cohort analysis
meaningless. (Real-genome mode already has a fixed shared `genome_spec`, so it is comparable; this
fixes ABSTRACT mode to match that guarantee.)

### What actually depends on `self.rng` at construction (verified in code)
1. `Selection(rng=self.rng)` — the **gene-role layout** (driver/onc/TSG, dispersal, TR, IR positions).
2. `make_celltype_exps` — the **per-cell-type baseline expression** (`beta` draws + driver overrides).
3. Spatial **structure seeding** (`_seed_structure` / `make_structure`) uses `self.rng.choice` **only
   when `structure_radius > 0`**; with the default `structure_radius = 0` it is not consumed.

Crucially, **evolution does NOT use `self.rng`**: `grow()`/`update()` draw from a *fresh*
`np.random.default_rng(seed + step + local_step)` (count.py:424; tumor.py:186), and tau-leaping from
`default_rng(seed + step)`. So the stochastic dynamics are already keyed off the run seed via an
independent stream — `self.rng`'s post-construction state never touches the trajectory.

### The fix — a config-determined **layout seed**, decoupled from the per-run **evolution seed**
Add `layout_seed` (a config-determined constant, shared across patients) and route the **layout +
baseline expression** through a dedicated `layout_rng = np.random.default_rng(layout_seed)`; keep
`self.rng = np.random.default_rng(seed)` for the per-run spatial seeding. Then:

* `Selection` positions and `celltype_exps` depend **only** on `layout_seed` → any two same-config
  runs (any evolution seeds) share **identical** oncogene/TSG/driver/dispersal/TR/IR gene sets **and**
  the same shared cell-type baseline expression (so a "shared cell state" is genuinely shared across
  patients — the substrate for the shared-vs-private benchmark).
* Evolution (which clones arise, private passenger SNVs, patient-specific CNAs, spatial structure)
  stays private, driven by the per-run `seed`.

**Default & back-compat.** `layout_seed` defaults to the module constant `DEFAULT_LAYOUT_SEED = 42`
→ **comparability by default** (all runs share a layout unless overridden). Back-compat holds because:
* the common `structure_radius = 0` path never consumes `self.rng` at construction, so splitting the
  layout into `layout_rng` is a no-op for the trajectory at *every* seed;
* for the ubiquitous test seed `42`, `layout_rng` and `self.rng` are both `default_rng(42)` and the
  layout+celltype draw sequence is byte-identical to today;
* evolution is unchanged everywhere (fresh per-step rng).
The only observable change is the intended one: at seeds ≠ 42 the **driver layout is now the shared
`42` layout** instead of a per-seed one. Existing suites assert *statistics* (survival fraction, size
ratios) or *same-seed determinism* / *diff-seed-differs* (guarded by an `or` on `cell_snv`), all of
which are preserved. `make_selection`/`Selection(...)` unit fixtures are unaffected (they pass their
own rng). Verified by running the full suite before and after.

`Selection` itself needs no signature change — its `rng` is *already* used only for the layout; the
bug was purely which rng the engines handed it. We add a one-line docstring note to that effect.

**Comparability test** (`tests/test_cohort.py`): two `GenotypeTumor(same config, different seed)` have
**identical** `get_oncogenes()/get_tsgs()/get_dispersal_genes()/get_treatment_resistant()/
get_immune_resistant()` index sets **and** identical `celltype_exps`, while their grown `cell_snv`
differs. Also: passing distinct explicit `layout_seed`s yields *different* layouts (the knob works).

---

## 2. THE COHORT LAYER (`iscc/cohort/`)

New subpackage `iscc/cohort/` (mirrors `iscc/sample`, `iscc/treatment`):
`cohort.py` (`Cohort`, `PatientResult`, `Subgroup`), `batch.py` (patient→batch mapping + pooled
emission), `groundtruth.py` (cohort ground-truth tables), `hashing.py` (cell-hashing HTO readout +
demux, the RNA-modality demultiplexer), `__init__.py`.

### 2.1 `Cohort` — N tumours, one shared landscape
```python
Cohort(base_config, patient_seeds, subgroups=None, layout_seed=DEFAULT_LAYOUT_SEED,
       genome_params=..., selection_params=..., cancer_cell_params=..., deme_params=...,
       spatial_params=..., grow_steps=..., subgroup_assignment=None)
```
* `base_config` = ONE shared config (the same params every `GenotypeTumor` uses). All patients pass
  the **same `layout_seed`** → shared driver landscape by construction.
* `patient_seeds` = list of per-patient **evolution** seeds (private dynamics). `len == n_patients`.
* `subgroups` (optional, personalized medicine): list of `Subgroup(name, selection_delta,
  therapy_response)` where `selection_delta` is merged onto the base `selection_params` for patients
  in that subgroup. `subgroup_assignment` maps patient index → subgroup name (default: round-robin /
  single default subgroup "all").
* `.run()` grows each patient's `GenotypeTumor` (tau or exact per `base_config`), materialises
  `cell_data`, and stores a `PatientResult(seed, subgroup, tumor, cell_data)`.

**Subgroup invariant (documented + guarded).** Subgroup deltas may change **effect scalars**
(`driver_effects`, `treatment_resistant_effects`, `immune_resistant_effects`, `dispersal_effects`)
but SHOULD keep the `prop_*` (which set gene *positions*) shared, so the driver landscape stays
identical across subgroups. Because `make_drivers` (using only `prop_driver`) runs FIRST in the
layout rng stream, oncogene/TSG identities are shared across subgroups even if a later `prop_*`
differs; still, we warn if a subgroup delta touches a `prop_*`, since that desyncs the resistance/IR
layout. This is exactly how molecular subtypes work: **same recurrent driver genes, different fitness
/ therapy consequences**.

**Resistance mechanism — it must EMERGE, never seeded.** The whole point of a mechanistic simulator is
that resistance ARISES from mutation + selection; injecting a resistant subclone (or a truncal
resistance mutation) would hand-impose the very answer the benchmark is meant to test — the "bolt-on
simulator" anti-pattern iscc avoids elsewhere (cf. the clonealign non-circularity argument). So a
subtype is defined ONLY by an **effect scalar** (`treatment_resistant_effects`). During an untreated
**burn-in**, treatment-resistance mutations arise spontaneously at the shared resistance loci and drift
as **neutral standing variation** (resistance is inert without drug). **Adjuvant** therapy then
**selects** them: in the high-effect (resistant) subtype the standing resistant cells survive and
**relapse**; in the low-effect (sensitive) subtype the same mutations are inert and the tumour is
eradicated. The differential response is thus a genuine evolutionary outcome. (Inherited **germline**
variants — `Subgroup.germline_mutations` and the per-patient private demux markers — are the only
pre-seeded alterations; they are applied to EVERY cell of the patient, tumour AND normal, as real
germline variants are, and never model acquired resistance.) **Honest consequence for the biomarker:**
at BASELINE the two
subtypes are molecularly indistinguishable (the same standing resistance mutations are present in both;
only their functional effect differs), so a bulk baseline call is **non-predictive** — a realistic
precision-oncology point iscc surfaces. Recovery of the responsive subtype comes from the therapy-
selected **emergent** signature (the relapsed tumour is clonally enriched for the selected resistance
mutations) and the response readout itself (who benefits, the known answer).

### 2.2 Patient→batch multiplexing (`cohort/batch.py`)
A **user-specified assignment** of patients to sequencing batches:
* **1:1** (`mapping="one_to_one"`): each patient is its own batch — pure biological-per-batch
  ("confounded") design; the multi-patient integration case.
* **N:1** (`mapping="multiplex"`, `capacity=k`): pool up to `k` patients into one batch (cell-hashing
  / genetic-multiplexing style). Each batch then carries **both** biological variation (different
  patients) **and** one shared technical signature (one `Batch` realization) — and, because several
  patients share a lane, a **demultiplexing** ground truth (patient-of-origin per pooled cell).
* Arbitrary explicit `dict{patient_idx: batch_id}` also accepted.

Emission reuses the scRNA machinery: for each batch we take the **pooled `cell_data`** of its
patients (cell names namespaced `P{pid}::C{i}` to avoid collisions) and run one `scRNA` batch
(one `Batch` seed = one technical signature) over those cells. This is literally the "confounded"
design the `run_scrna_batches` docstring points at, extended to N:1. Output: per-batch `AnnData`
with `.obs` carrying `patient`, `subgroup`, `clone`, `cell_type`, `batch`, plus a concatenated
cohort `AnnData` (`iscc.integrations.to_anndata`-compatible) for integration/demux tools.

### 2.3 Cohort ground truth (`cohort/groundtruth.py`)
Surfaced as tidy tables / arrays (what real cohorts can never give):
* **Recurrence per gene** — for each driver gene, the number of patients carrying ≥1 mutation in it
  (`recurrence_table`): the true recurrent-vs-private split (recurrent driver vs private passenger).
* **Per-patient private mutations** — mutated loci unique to one patient.
* **Per-cell patient-of-origin** — the `patient` label on every pooled cell (demux answer key).
* **Shared-vs-private cell state** — coarse cell **type** (cancer/epithelial/stromal/immune) is the
  SHARED axis (shared baseline expression across patients); **clone / patient identity** is the
  PRIVATE axis. A per-cell `(shared_state, private_state)` label pair scores integration
  over/under-correction.
* **Per-patient subgroup + true therapy response** — the subgroup label and its `therapy_response`
  (the stratification / biomarker answer key).

---

## 3. BENCHMARKS (first pass: the 3 strongest, all demonstrable locally; external seams wired)

`validation/validate_cohort.py` → `manuscript/figures/validation_cohort.png`. Self-contained
(numpy/scipy/sklearn) so it runs in the core `iscc` env and in CI; heavy external tools are optional
and skipped gracefully (own dedicated envs, clonealign/inferCNV pattern).

1. **Recurrence / driver detection** *(self-contained, headline for "shared landscape")*. Across N
   patients over the shared landscape, count per-gene recurrence; a MutSigCV/dNdScv-style recurrence
   score (mutation recurrence vs a passenger background) recovers the true recurrent driver set.
   Report **precision/recall / PR-AUC** vs the known recurrent drivers, and show private passengers
   are correctly *not* called. Contrast with a single-patient analysis (recurrence undefined) to make
   the cohort the unit of inference.
2. **Personalized medicine / stratification** *(self-contained, uses the treatment module — the
   headline personalized-medicine story)*. Two subgroups (sensitive S / resistant R, same drivers,
   different `treatment_resistant_effects`). Run the SAME therapy across the cohort: S tumours
   regress, R tumours escape (a burden/size readout = ground-truth differential response). Then
   **recover the responsive subgroup from molecular profiles** (a classifier on expression/mutation
   → subgroup) and show the biomarker maps to the resistance axis — "which patients benefit from
   which therapy," with a known answer.
3. **Multi-patient batch integration** *(self-contained scoring; optional Harmony/scVI via dedicated
   env)*. 1:1 pooling → one batch per patient. Score whether integration aligns the SHARED cell
   states across patients while preserving PRIVATE ones: **iLISI** (batch mixing) and **ARI** vs the
   shared-vs-private labels, comparing naive concatenation (over-separated) to a corrected embedding.
   The central over/under-correction failure mode, with no real truth. `iscc-harmony` (harmonypy) /
   `iscc-scvi` (scvi-tools/scANVI) wired behind an `_available()` skip guard.

**Demultiplexing — PER MODALITY (the two methods used in practice).** DNA assays (WGS/WES/scDNA) cover
the genome/exome broadly, so germline SNPs are genotyped reliably per cell → pooled DNA is demuxed
**genetically** (souporcell/vireo/demuxlet): cluster pooled cells by their germline-variant genotype
and assign to patient. Because germline is carried by EVERY cell (tumour and normal — see the germline
mutation model above), all cells demux, cancer and normal alike; `iscc-demux` (vireo) seam wired.
Droplet **scRNA**, by contrast, only covers sparse expressed loci and cannot genotype germline SNPs
per cell reliably — so pooled scRNA is demuxed by **cell hashing** (a per-sample HTO/MULTI-seq oligo
barcode, `iscc.cohort.emit_cell_hashtags` / `demux_hashtags`): each cell gets its own hashtag + an
ambient soup of all hashtags + a doublet fraction with a second hashtag; singlet assignment is
near-perfect and the real challenge is **doublet detection** (hashing is used with cell super-loading).

### External-env convention (follows `validation/README_integration.md`)
Each external integration/demux tool gets its **own** dedicated conda env — never the core `iscc`
env: `iscc-harmony` (harmonypy), `iscc-scvi` (scvi-tools/scANVI), `iscc-demux` (vireo). Pattern:
env-var-overridable interpreter path + `<tool>_available()` skip guard + a `subprocess` runner
script; data crosses as files (AnnData/CSV). Shared helpers in `validation/cohort_common.py`.

---

## 4. Deliverables & checklist
- [x] `DESIGN_cohort.md` (this doc).
- [ ] Engine seed-decoupling fix (`selection.py` note; `tumor/tumor.py`; `tumor/models/count.py`).
- [ ] `tests/test_cohort.py` — comparability test + `Cohort` / batch-mapping / ground-truth tests.
- [ ] `iscc/cohort/` package (`cohort.py`, `batch.py`, `groundtruth.py`, `__init__.py`).
- [ ] `validation/validate_cohort.py` (+ `validation/cohort_common.py`) → figure.
- [ ] Manuscript Results subsection ("iscc provides cohort-level ground truth …").
- [ ] Flip the BACKLOG cohort item to DONE.
- [ ] Full pytest suite green; commit on `dev`.
</content>
</invoke>
