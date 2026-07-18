# Handoff prompt — showcase notebooks (integration, WGD/allele-CNA, gene programs)

Saved 2026-07-18. Copy the block below into a fresh session. **Docs/pedagogy work, not an engine change.**
Goal: three EXECUTED Jupyter notebooks in `notebooks/` that show iscc's *science* clearly (the existing
tutorials already cover the plumbing). The user explicitly decided: (a) build these three, (b) executed but
`notebooks/`-only for now — do NOT wire them into the MkDocs nav (`mkdocs.yml`) or copy to `docs/tutorials/`,
(c) **every notebook must grow a SPATIALLY STRUCTURED tumour and visualise that structure.** Branch from `dev`.

Context the fresh session lacks: the current `notebooks/` has 7 solid executed tutorials (pipeline, growth,
data-overview, assay_dna/scrna/spatial, reads) plus 5 near-empty STUBS. These three deliverables are two new
notebooks + one stub rewrite. The science they show already exists and is validated under `validation/`
(`validate_numbat.py`, `validate_programs.py`, `validate_clonealign.py`, `integration_common.py`,
`programs_common.py`) — the notebooks are a *self-contained, teachable* view of it, NOT a re-run of the heavy
external-tool benchmarks.

---

```
Create three EXECUTED showcase notebooks in notebooks/ for iscc. These demonstrate the SCIENCE (the existing
tutorials cover the mechanics). Do NOT add them to mkdocs.yml or docs/tutorials/ — notebooks/ only, for now.

REPO & ENV
- Repo: /Users/pedroferreira/projects/iscc/repo (branch `dev`).
- Python/Jupyter: ~/miniconda3/envs/iscc/bin/python (the core `iscc` env). Build with nbformat, execute with
  nbclient/ExecutePreprocessor (or `jupyter nbconvert --to notebook --execute --inplace`) using THIS env's
  kernel, so outputs are saved. Every notebook must run end-to-end with NO errors in the core env.
- Conventions: commit on `dev` WITH `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`; be honest; do
  not break the test suite (you are only adding notebooks, so it should be untouched).
- Match the existing notebook idiom EXACTLY — read notebooks/01_pipeline_walkthrough.ipynb and
  notebooks/assay_scrna.ipynb first: narrative markdown + focused matplotlib code cells, the public `iscc`
  API, `%matplotlib inline`, small figures, a "Next / see also" pointer at the end.

THREE HARD REQUIREMENTS (the user asked for these specifically)
1. SPATIALLY STRUCTURED simulations. Every notebook grows a tumour with `spatial_params` containing
   `structure_radius > 0` (a real gland with a normal epithelial/stromal compartment), NOT the well-mixed
   `structure_radius = 0` regime that programs_common uses. And each notebook must VISUALISE the spatial
   structure — a `cell_crd` (row/col) scatter coloured by cell type / clone / program / is_wgd, and/or
   `tumor.plot_grid(ax=...)` — because showing iscc's spatial modelling is part of "showing the aspects
   clearly". Reuse `tumor.plot_muller(ax=...)` where a clonal-dynamics view helps.
2. SELF-CONTAINED in the core env. Do the analyses with numpy/scipy/sklearn INLINE (a mini-clonealign by
   CN-profile matching; sklearn NMF for program recovery). Do NOT shell out to the heavy dedicated tool envs
   (iscc-clonealign / iscc-numbat / iscc-scdef) from a notebook — instead end each notebook with a short
   pointer to the full external-tool benchmark under validation/. Rationale: notebooks must always execute.
3. EXECUTED with outputs, in notebooks/ only.

THE MICROENVIRONMENT IS PART OF THE DATA — DO NOT FILTER TO CANCER CELLS. Real tumour datasets are MIXTURES:
malignant cells PLUS microenvironment (epithelial / stromal / immune). Practitioners analyse that mixture, and
so must these notebooks. Assay and analyse the MIXED population exactly as a real study would; the normal cells
ARE the reference inferCNV/Numbat need, the background deconvolution must resolve, and the diploid / balanced-
BAF anchor. iscc's ground truth (cell type, clone, CN, is_wgd, program z) is used ONLY to SCORE and COLOUR
results — never to pre-filter the input. Where a metric is intrinsically about the malignant cells (clone-
recovery accuracy, per-clone CN, program activity), compute it on the malignant subset IDENTIFIED within the
mixed input via ground truth — but the analysis INPUT stays the mixture, and a natural first step in each
notebook is separating malignant from normal (as a real pipeline does).

TUMOUR PURITY (the real numeric gotcha, verified): with structure_radius>0 the normal compartment dominates
(grid_size=20, structure_radius=5 → ~2500 stromal vs ~110 cancer ≈ 4% purity, unrealistically low). So build a
REALISTIC mixture for the assays: keep all/most malignant cells and SUBSAMPLE the microenvironment to a
plausible tumour purity (aim ~30–60% malignant — e.g. all malignant + n_normal≈150 sampled normal cells, the
way validation/integration_common.build_cna_inputs does it), and/or raise initial_cancer_cells/steps. NEVER
drop structure_radius to 0 to dodge this — spatial structure AND a present microenvironment are both hard
requirements. See integration_common.build_infercnv_inputs / build_cna_inputs for the exact malignant+normal
assembly to copy.

SHARED SPATIAL CONFIG (proven; the integration benchmark uses it)
    from iscc.tumor.models import GenotypeTumor
    GENOME    = {"n_segments": 12, "segment_size": 50}
    SELECTION = {"prop_driver": 0.2, "prop_dispersal": 0.0, "prop_immune_resistance": 0.0,
                 "prop_treatment_resistance": 0.0}
    DEME      = {"carrying_capacity": 8, "initial_cancer_cells": 4}
    SPATIAL   = {"grid_size": 20, "structure_radius": 5}      # << spatial structure + normal compartment
    CANCER    = {"division_rate": 0.6, "death_rate": 0.03, "max_birth_rate": 0.98,
                 "mutation_rate": 1.1, "dispersal_rate": 0.5}
    t = GenotypeTumor(seed=SEED, genome_params=GENOME, selection_params=SELECTION,
                      cancer_cell_params=CANCER, deme_params=DEME, spatial_params=SPATIAL,
                      expression_params=EXPR)      # EXPR per-notebook (see below); omit for NB1's DNA/CN view
    t.grow(n_steps=750, seed=SEED); t.make_cell_data()

API CHEAT-SHEET (all verified this session)
- cell_data = t.cell_data (dict of DataFrames). Base keys: cell_snv, cell_cnv (TOTAL CN per gene),
  cell_exp, cell_type (column "cell_id" = the per-cell genotype id), cell_crd (columns row, col — for
  spatial plots). Conditionals: cell_program (programs on), cell_wgd["is_wgd"] (wgd_rate>0),
  cell_exp_p / cell_exp_m / cell_rna_baf (dosage_params allele_specific=True).
- Cell type per cell: ty = np.array([t.genotypes[g].type for g in cd["cell_type"]["cell_id"].values])
  -> "cancer" / "epithelial" / "stromal". cancer_mask = ty == "cancer".
- Assays: from iscc.data import scRNA, scDNA, bulkDNA
    rna = scRNA(n_cells=len(cells), protocol="10x", seed=S).run(cd, cell_subset=cells); rna.observed_counts
    dna = scDNA(n_cells=len(cells), breadth="wgs", seed=S).run(cd, cell_subset=cells); dna.coverage/.genes/.cells
  (see notebooks/assay_scrna.ipynb and assay_dna.ipynb for the exact call shape.)
- Per-segment total CN per cell: cnv = cd["cell_cnv"].values; segment CN = take the first gene of each
  segment via segment offsets np.concatenate([[0], np.cumsum(t.selection.segment_sizes)]).
- Per-HOMOLOG CN ground truth (cell_cnv is total only — needed for the allele/WGD notebook). Either
  `from` the validation helper: sys.path.insert(0,"validation"); from integration_common import
  segment_allele_cn  (returns cell_id -> (n_seg,2) array of (p_cn, m_cn)); OR inline the 6 lines:
      def segment_allele_cn(t):
          gid = t.cell_data["cell_type"]["cell_id"].astype(str).values
          idx = np.asarray(t.cell_data["cell_type"].index); g = t.genotypes; cache={}; out={}
          for cell, gg in zip(idx, gid):
              if gg not in cache:
                  rep = g.get(gg)
                  cache[gg] = (np.array([(len(s["p"]), len(s["m"])) for s in rep.genome])
                               if rep is not None and hasattr(rep, "genome") else None)
              out[cell] = cache[gg]
          return out
- Programs (R13): EXPR from validation/programs_common.py `expression_params()` (import it, or inline its
  dict). program_truth = t.program_truth after make_cell_data(): keys loading (K x genes), program_names,
  program_genes (list of gene-index arrays), gene_program_map. cd["cell_program"] is cells x K with columns =
  program names (proliferation, emt, hypoxia, drug_resistance, immune_evasion, program_5).
- WGD: add "wgd_rate": 0.05 to CANCER and set EXPR dosage_params allele_specific=True. is_wgd per cell =
  cd["cell_wgd"]["is_wgd"]. Per-genotype ploidy = t.genotypes[g].genome_summary["ploidy"].

===================================================================================================
NOTEBOOK 1 — notebooks/combining_scdna_scrna.ipynb  (REWRITE the stub; the flagship integration story)
===================================================================================================
The stub currently names SCICoNE / SCATrEx — DELETE that, those are not the tools iscc built on. New story:
one spatially-structured tumour → scDNA + scRNA from the SAME cells → the DNA<->RNA link is EMERGENT (per-
allele dosage), not imposed → reconstruct subclones from RNA using the DNA-defined CN profiles (clonealign's
idea) and score against iscc's true clone labels. Grow with the shared spatial config (no EXPR needed unless
you want; dosage is on by default). Cells:
  1. MD: title + intro (non-circularity: CN->expression dosage EMERGES; iscc alone supplies true clone label
     + true CN). State that the dataset is a MIXTURE — malignant + microenvironment — as in a real study.
     One line: "the production tools clonealign / inferCNV / Numbat do this properly — see
     validation/validate_clonealign.py, validate_infercnv.py, validate_numbat.py."
  2. CO: imports + grow the spatial tumour; print total / malignant / normal counts and the malignant
     fraction. Assemble the assayed dataset = all malignant cells + a subsample of microenvironment cells at a
     realistic purity (all cancer + n_normal≈150 normal); this MIXTURE is the input to everything below.
  3. CO: SPATIAL viz — cell_crd scatter coloured by cell type (cancer vs epithelial vs stromal), and/or
     t.plot_grid. Show the gland + where the microenvironment sits.
  4. CO: scRNA on the MIXTURE (malignant + normal). Embed/cluster the cells and show malignant vs normal
     SEPARATE (colour by the true cell-type label to score it) — the real first step, and why the normal
     cells are the reference, not noise to be discarded.
  5. CO: define clones on the malignant cells from per-segment CN (sklearn AgglomerativeClustering, k≈3–4);
     heatmap of the clone x segment consensus CN. (Malignant identified within the mixture via ground truth.)
  6. CO: scDNA on the mixture -> per-clone recovered CN for the malignant clones; concordance with truth.
  7. MD: the emergent CN->expression link.
  8. CO: per-segment mean expression vs true CN across clones (scatter or line) — the dosage relationship,
     emergent not imposed; normal cells sit at CN≈2 / baseline as the anchor.
  9. CO: reconstruct clones from RNA — assign each MALIGNANT RNA cell to the clone whose CN profile best
     matches its per-segment mean expression (correlation / nearest-profile); report accuracy vs the true
     clone label. (Input is the mixture; the metric is naturally on the malignant subset.)
 10. MD: "see validation/ for the real clonealign/Numbat benchmark" + Next pointer.

===================================================================================================
NOTEBOOK 2 — notebooks/wgd_allele_cna.ipynb  (NEW; WGD + allele-specific CNA — showcases the Numbat result)
===================================================================================================
EXPR = expression_params with dosage_params allele_specific=True (so cell_rna_baf exists); CANCER += wgd_rate.
Cells:
  1. MD: intro — WGD as a punctuated genome doubling (the diploid 1+1 -> 2+2); iscc surfaces is_wgd; the
     allele layer (BAF) exposes imbalance total copy number cannot. (Get the notation RIGHT: a normal genome
     is 1+1; WGD -> 2+2; the copy-neutral-detectable states are e.g. 4+0 / 3+1 at a total matching 2+2.)
  2. CO: grow the spatial tumour with wgd_rate≈0.05 + allele_specific; print malignant/normal counts + purity
     + WGD-cell fraction. The dataset stays a mixture (malignant + microenvironment); the normal cells are the
     diploid / balanced-BAF anchor a real CNA caller relies on.
  3. CO: SPATIAL viz — cell_crd scatter coloured by is_wgd for malignant cells, with the microenvironment
     shown (e.g. grey) — WGD subclones sitting in the gland alongside normal cells.
  4. CO: ploidy signature — histogram of per-cell mean ploidy (cd["cell_cnv"].mean(1)) for the WHOLE mixture:
     normal cells at ~2 (the diploid anchor), malignant split by is_wgd into a near-diploid non-WGD mode and an
     elevated WGD mode (the doubling-then-loss signature). Showing the normal anchor is the point.
  5. MD: a PURE doubling is allelically balanced (BAF 0.5) and cancels under per-cell normalisation, so it is
     unidentifiable from total CN + BAF; only the doubling+LOSS erosion creates allelic imbalance.
  6. CO: grow a WGD-OFF twin (same seed, wgd_rate=0), compute with segment_allele_cn the fraction of MALIGNANT
     segments that are "allele-only detectable" = imbalanced (|p-m|>=1) at EVEN total CN (total-CN-blind); show
     it rises WGD-off -> WGD-on (≈1% -> several %). Ground-truth version of the Numbat figure's panel A. NB:
     off/on are different RNG trajectories (the WGD draw shifts the stream) — frame as a cohort contrast,
     average over a few seeds if noisy. (This metric is intrinsic to the malignant cells; the normal cells,
     being balanced diploid, are the ~0 baseline.)
  7. CO: the emergent allele signal — |cell_rna_baf - 0.5| at CN-altered vs balanced segments, MALIGNANT vs
     NORMAL: imbalance localises to the CNAs in malignant cells while normal cells stay ~0.5 (the anchor that
     makes the signal callable at all).
  8. MD: "the full head-to-head — Numbat recovers this allelic state where expression-only inferCNV is at
     chance — is validation/validate_numbat.py --wgd-rate 0.04" + Next pointer.

===================================================================================================
NOTEBOOK 3 — notebooks/gene_programs.ipynb  (NEW; R13 expression realism)
===================================================================================================
EXPR = validation/programs_common.py expression_params() (import it). Grow the SPATIAL config with EXPR; analyse
the MIXTURE (malignant + microenvironment) — the programs live in the malignant cells, but a real scRNA study
sees all cell types, and a factor model must separate cell-type identity (immune/stromal) from within-malignant
programs. Use ground truth only to colour/score. Cells:
  1. MD: intro — expression = gene PROGRAMS (co-expressed modules) x per-cell activity, plus genotype dosage;
     iscc knows the true loading + per-cell z (program_truth), ground truth no real dataset has. Note the input
     is a mixed tumour: normal cells carry baseline expression (no programs), malignant cells carry the programs.
  2. CO: grow; assay scRNA on the MIXTURE (malignant + subsampled microenvironment at realistic purity); show
     cd["cell_program"] for the malignant cells (per-cell activity z, columns = program names).
  3. CO: SPATIAL viz — cell_crd scatter coloured by dominant program (argmax z) for malignant cells, with the
     microenvironment shown; programs varying across the gland.
  4. CO: the true loading matrix (program_truth["loading"], K x genes) heatmap; and show program_genes are
     SCATTERED across the genome (not one contiguous block).
  5. CO: embed the MIXTURE (e.g. PCA/UMAP or a program-activity heatmap) coloured by cell type AND by dominant
     program — normal cells form their own cell-type structure; programs are the within-malignant axis. Showing
     the tool must disentangle the two is itself a realism point.
  6. MD: programs are functional (scattered genome-wide) whereas CNAs are positional (contiguous) — a factor
     model can mistake a copy-number segment for a "program"; that is the confound.
  7. CO: the positional-clustering diagnostic (import from programs_common: positional_clustering,
     scattered_null) — true programs sit near the scattered null; a CNA segment approaches 1.0.
  8. CO: recovery in the core env — sklearn.decomposition.NMF on Poisson-sampled counts of the MIXTURE (see
     programs_common.counts_anndata for making counts from cd["cell_exp"]), Hungarian-match factors to the true
     loading, report mean loading cosine for the malignant programs; note normal cells load onto a distinct
     cell-type factor, exactly as they would on real data.
  9. MD: "the full scDEF/cNMF recovery-vs-burden benchmark is validation/validate_programs.py" + Next pointer.

ACCEPTANCE / DELIVERABLES
- Three notebooks in notebooks/ (combining_scdna_scrna rewritten; wgd_allele_cna + gene_programs new), each
  executed with outputs, running clean in the core env, EACH growing a structure_radius>0 tumour and showing
  its spatial structure. Keep each notebook's total runtime to a couple of minutes (grows are ~0.2–2s; the
  cost is the assays/NMF — keep cell counts modest).
- Do NOT touch mkdocs.yml or docs/tutorials/ (notebooks/ only, per the user).
- Commit on `dev` with the Co-Authored-By trailer. A one-line note in BACKLOG.md under the notebook/pedagogy
  track is welcome but optional.

HONEST NOTES: if a self-contained reconstruction (NB1 clone recovery, NB3 NMF) lands weak, report the number
honestly rather than tuning it — the point is to SHOW the ground truth and the emergent signal, not to claim a
tool result (the validation/ scripts own the tool claims). If the malignant-cell count is too low for a clean
demo, raise initial_cancer_cells / steps / grid_size (or subsample less microenvironment) rather than dropping
structure_radius to 0 or filtering out the normal cells — spatial structure AND an analysed microenvironment
are both hard requirements.
```
