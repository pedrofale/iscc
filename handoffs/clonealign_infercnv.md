# Handoff prompt — clonealign + inferCNV integration benchmarks

Saved 2026-07-03. Copy the block below into a fresh session. These are the two near-ready
ground-truth-utility demos (scMultiSim-style: run a real integration method on iscc data, score vs
known truth). Companion context: `BACKLOG.md` (benchmark suite), memory `iscc-paper-positioning.md`
(the non-circularity argument), the PEtracer sections in `manuscript/paper.tex` (the pattern to
match). Branch from current `dev`. **Do the multi-patient cohort work AFTER these** (per user).

---

```
Build two integration-benchmark demonstrations for iscc: clonealign (DNA↔RNA clone assignment) and
inferCNV (CNA-from-expression). Each RUNS A REAL DOWNSTREAM METHOD on iscc-simulated data and SCORES
it against iscc's known ground truth — the scMultiSim/SISTEM "we provide ground truth" convention,
pairing with the PEtracer section already in the paper.

REPO & ENV
- Repo: /Users/pedroferreira/projects/iscc/repo (branch `dev`).
- Python/pytest: ~/miniconda3/envs/iscc/bin/python.
- Conventions: commit on `dev` WITH `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`;
  keep the full pytest suite green; be honest (no overselling). Match surrounding style.

WHY THIS MATTERS — the non-circularity argument (state it in the paper):
A benchmark for integration methods must be non-circular. A bolt-on (CNAs from one sim + expression
imposed on top) has to hand-impose the CNA→expression model; if that's the dosage model clonealign/
inferCNV assume, the benchmark tests the method on its own assumption. In iscc the CNA→expression
coupling EMERGES from evolution (dosage + selection) and the microenvironment (F8), so it is a FAIR
test, and iscc additionally supplies confounds (F8 spatial programs) a bolt-on lacks.

ENABLERS (verified):
- `iscc.integrations`: `to_anndata(cell_data)` (cells×genes AnnData, `obsm["spatial"]`, clone/type
  `obs`), plus lineage export. Per-cell ground truth in `tumor.cell_data`: `cell_type` (clone/gid),
  `cell_cnv` (per-gene copy number), `cell_snv`, `cell_exp`, `cell_crd`.
- Assays: `from iscc.data import scDNA, scRNA` (+ `scSpatial`). scDNA gives per-cell CN + het/VAF;
  scRNA gives counts; both from the SAME cells (shared clones) → the DNA↔RNA link with ground truth.
- squidpy/scanpy/anndata installed. Check `infercnvpy` (pip, scverse, Python-native inferCNV) — likely
  needs install. Hotspot/cassiopeia/copyKAT/clonealign(R) NOT installed.

DATA GENERATION (shared): grow a multi-clone tumour (several clones with DISTINCT segmental CNAs; F8
optional to add a spatial confound), take a sample, and run scDNA + scRNA on it. Keep the per-cell
true clone label and true per-cell/segment copy number.

=== DEMO 1: clonealign (DNA↔RNA integration) ===
Assign each scRNA cell to a clone whose scDNA-derived copy-number profile best explains its
expression (the clonealign model: expression ∝ copy number). Score assignment ACCURACY / AUC vs the
true clone-of-origin. Show it works BECAUSE of the dosage coupling, and degrades where the dosage
effect is weak (low-CN-variance genes) — a real, non-circular benchmark.
- Method: the R `clonealign` (kieranrcampbell/clonealign) is the gold standard but R/TensorFlow-heavy.
  PREFER a compact PYTHON implementation of the same model (per-cell categorical assignment maximising
  a CN-dosage likelihood; the model is simple) so the demo is self-contained and reproducible; cite
  clonealign (campbell_clonealign_2019, already in the bib) as the method. Optionally add R clonealign
  via rpy2 as a gold-standard cross-check.
- Figure: (A) UMAP/embedding coloured by assigned vs true clone; (B) accuracy/AUC vs clone; (C)
  accuracy vs the fraction of genes with a clone-specific CN effect (the dosage dependence).

=== DEMO 2: inferCNV / copyKAT (CNA-from-expression) ===
Infer per-cell copy number FROM expression and score correlation/accuracy vs the true `cell_cnv`.
- Method: use `infercnvpy` (Python, pip) — running-window smoothing of expression along the genome.
  Show it recovers the clonal CNA structure (segment gains/losses) and the per-cell CN correlation.
- Figure: (A) inferred vs true CN heatmap (cells×segments); (B) per-segment correlation; (C)
  malignant-vs-normal separation (if normals present).

DELIVERABLES
- `validation/validate_integration.py` (or two scripts) → `manuscript/figures/validation_clonealign.png`,
  `validation_infercnv.png`. Print the headline scores (AUC, correlation).
- Tests `tests/test_integration.py` (small deterministic tumour: clonealign accuracy > chance;
  inferCNV recovers the amplified/deleted segments). Guard optional deps (infercnvpy) gracefully.
- Manuscript: a Results subsection "iscc provides non-circular ground truth for multi-modal
  integration" pairing clonealign + inferCNV with the PEtracer section (same "ground-truth utility"
  frame; cite clonealign + inferCNV/copyKAT). Wire the figures in; keep cites/refs resolving.
- Docs: flip the clonealign/inferCNV items in `BACKLOG.md` to DONE. Run the full suite; commit on `dev`.

Add bib entries as needed: inferCNV (Patel/Tirosh 2014, Science) and/or copyKAT (Gao et al. 2021,
Nat Biotechnol); infercnvpy is the scverse reimplementation. clonealign is already cited.
```
