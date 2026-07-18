# Handoff prompt — showcase notebook SUITE (one shared spatial sim → single-tumour + cohort notebooks)

Saved 2026-07-18 (expanded from the 3-notebook version). Copy the block below into a fresh session.
**Docs/pedagogy work, not an engine change.** Deliverable: a coherent set of EXECUTED notebooks in
`notebooks/` that all draw from **one shared, spatially-structured tumour-evolution simulation** (shown in its
own notebook), plus a small **5-tumour cohort** for cross-patient notebooks. Branch from `dev`.

User decisions this session (authoritative):
- ONE spatially-structured simulation underlies all single-tumour science notebooks; show it in a SEPARATE
  base-simulation notebook. **The base sim must reach ≥10,000 cancer cells.**
- Analyse the MIXED tumour (malignant + microenvironment) as a real study does — ground truth only scores/
  colours, never pre-filters to cancer cells.
- Then a **cohort of 5 separate simulations** for cohort notebooks: (a) batch integration of scRNA + scDEF for
  shared programs, (b) MHN + TreeMHN on the DNA for recurrent mutational patterns.
- Executed, `notebooks/`-only for now — do NOT touch `mkdocs.yml` or `docs/tutorials/`.

VERIFIED THIS SESSION (so you don't re-derive): a spatially-structured tumour (grid_size=60,
structure_radius=8, carrying_capacity=8, tau-leaping) reaches **10.8k cancer cells (~38k total, ~28% purity)
in ~41 s**, make_cell_data ~1.2 s, ~1–2 GB RAM with all layers on. So the base sim is a ~1-minute deterministic
grow — RE-GROW it in each notebook via a shared helper rather than saving giant CSVs (38k×genes matrices are
too big for disk/repo).

---

```
Build a suite of EXECUTED showcase notebooks in notebooks/ for iscc, all based on ONE shared spatially-
structured simulation, plus a 5-tumour cohort. These show the SCIENCE (the existing 7 tutorials cover the
mechanics). Do NOT add anything to mkdocs.yml or docs/tutorials/ — notebooks/ only, for now.

REPO & ENV
- Repo: /Users/pedroferreira/projects/iscc/repo (branch `dev`).
- Python/Jupyter: ~/miniconda3/envs/iscc/bin/python (the core `iscc` env). Build with nbformat, execute with
  nbclient/ExecutePreprocessor (or `jupyter nbconvert --to notebook --execute --inplace`) in THIS env so
  outputs are saved. Every notebook must run end-to-end with NO errors in the core env.
- Conventions: commit on `dev` WITH `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`; be honest; do
  not break the test suite (you only add notebooks).
- Match the existing idiom EXACTLY — read notebooks/01_pipeline_walkthrough.ipynb and notebooks/assay_scrna.ipynb
  first: narrative markdown + focused matplotlib code cells, the public `iscc` API, `%matplotlib inline`, small
  figures, a "Next / see also" pointer at the end.

============================================================================================================
THE SHARED SUBSTRATE — do this FIRST
============================================================================================================
Create notebooks/base_sim.py (a small importable helper, NOT a notebook) that every science notebook imports,
so they all use the SAME simulation deterministically:

    # notebooks/base_sim.py
    import sys, os, numpy as np
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "validation"))
    import programs_common as PC                     # reuse the vetted expression_params()
    from iscc.tumor.models import GenotypeTumor

    BASE_SEED    = 3
    COHORT_SEEDS = [3, 4, 5, 6, 7]                    # 5 tumours
    GENOME    = {"n_segments": 12, "segment_size": 50}
    SELECTION = {"prop_driver": 0.2, "prop_dispersal": 0.0, "prop_immune_resistance": 0.0,
                 "prop_treatment_resistance": 0.0}
    CANCER    = {"division_rate": 0.6, "death_rate": 0.03, "max_birth_rate": 0.98,
                 "mutation_rate": 1.1, "dispersal_rate": 0.5, "wgd_rate": 0.05}   # WGD on (realistic)
    DEME      = {"carrying_capacity": 8, "initial_cancer_cells": 4}
    SPATIAL   = {"grid_size": 60, "structure_radius": 8}   # SPATIAL structure + microenvironment
    def EXPR():                                       # programs ON + allele-specific ON
        e = PC.expression_params(scatter=1.0)
        e["dosage_params"]["allele_specific"] = True
        return e

    def grow_base_tumor(seed=BASE_SEED, target_cancer=10000):
        # DO NOT pass layout_seed -> defaults to DEFAULT_LAYOUT_SEED so every seed shares the SAME gene
        # roles / program dictionary / epistasis landscape (the cohort-comparability guarantee).
        t = GenotypeTumor(seed=seed, genome_params=GENOME, selection_params=SELECTION,
                          cancer_cell_params=CANCER, deme_params=DEME, spatial_params=SPATIAL,
                          expression_params=EXPR(), update_mode="tau", tau=1.0)
        while True:
            t.grow(n_steps=10, seed=seed)
            ncan = sum(c for g, c in t.genotypes_counts.items() if c > 0 and t._is_cancer(g))
            if ncan >= target_cancer:
                break
        t.make_cell_data()
        return t

    def grow_cohort(seeds=COHORT_SEEDS, target_cancer=10000):
        # generator — grow one at a time so you never hold 5 full tumours in RAM (see MEMORY below)
        for s in seeds:
            yield s, grow_base_tumor(seed=s, target_cancer=target_cancer)

MEMORY: a materialised base tumour is ~1–2 GB (≈38k cells × the per-cell matrices, extra with programs +
allele layers). Fine for one tumour. For the cohort NEVER hold 5 at once — in grow_cohort, grow → run the
assay to a COMPACT matrix (a few hundred cells × genes) → keep only that → let the tumour be GC'd → next seed.
If RAM is tight, shrink GENOME (e.g. n_segments=10) or structure_radius (fewer normal cells), keeping ≥10k
cancer + a present microenvironment.

THREE HARD REQUIREMENTS (all notebooks)
1. SPATIALLY STRUCTURED sims (structure_radius>0), and each notebook VISUALISES the spatial structure —
   cell_crd (row/col) scatter and/or tumor.plot_grid(ax=...), plus tumor.plot_muller(ax=...) where clonal
   dynamics help.
2. THE MICROENVIRONMENT IS PART OF THE DATA — never filter to cancer cells. Assay/analyse the MIXTURE
   (malignant + epithelial/stromal); the normal cells are the CNA-caller reference, the deconvolution
   background, and the diploid/balanced-BAF anchor. Ground truth (cell type, clone, CN, is_wgd, program z,
   spot composition, true tree) scores/colours only. Where a metric is intrinsically about malignant cells,
   compute it on the malignant subset IDENTIFIED within the mixed input. A realistic first step in most
   notebooks is separating malignant from normal, as a real pipeline does. For assays, subsample the
   microenvironment to a realistic purity (~30–60% malignant) rather than assaying all ~28k normal cells.
3. SELF-CONTAINED in the core env for the SINGLE-TUMOUR notebooks: do analyses inline (mini-clonealign by CN-
   profile matching; NJ/UPGMA trees; NNLS spot deconvolution; sklearn NMF), and POINT to the full external-
   tool benchmark under validation/ rather than shelling out. EXCEPTION: the two COHORT notebooks are named
   around specific tools (scDEF, MHN, TreeMHN) — for those, REUSE the validation runners in their dedicated
   envs (guarded to skip / fall back to a self-contained approximation if the env is absent), so they show the
   real tool when available and still execute otherwise.

API CHEAT-SHEET (all verified this session)
- cell_data = t.cell_data. Base keys: cell_snv, cell_cnv (TOTAL CN/gene), cell_exp, cell_type (col "cell_id" =
  genotype id), cell_crd (row, col). With the base config also: cell_program, cell_wgd["is_wgd"],
  cell_exp_p/cell_exp_m/cell_rna_baf (allele layer). Cell type: ty = np.array([t.genotypes[g].type for g in
  cd["cell_type"]["cell_id"].values]) -> "cancer"/"epithelial"/"stromal"; cancer_mask = ty=="cancer".
- Assays: from iscc.data import scRNA, scDNA, bulkDNA, Visium  (registry: iscc.data.ASSAYS =
  {bdna, scdna, scrna, visium, scspatial}).
    scRNA(n_cells=, protocol="10x", seed=).run(cd, cell_subset=cells).observed_counts        # cells x genes
    scDNA(n_cells=, breadth="wgs", seed=).run(cd, cell_subset=cells) -> .coverage/.genes/.cells
    bulkDNA(n_reads=, data_mode="counts", fpr=, fnr=).run(cd) -> .observed_data{coverage,alt_counts}  # bulk VAF
    Visium(n_reads=, n_spots_x=, n_spots_y=, spot_radius=).run(cd) -> AnnData: X spots x genes,
        obsm["spatial"], .obs = per-spot ground truth (n_cells, member ids, dominant clone, clone fractions)
  (See notebooks/assay_*.ipynb and validation/validate_{visium,deconvolution,multiregion_phylo}.py for exact
  call shapes.)
- Per-segment total CN: segment offsets = np.concatenate([[0], np.cumsum(t.selection.segment_sizes)]).
- Per-HOMOLOG CN ground truth (cell_cnv is total only): from validation/integration_common.py import
  segment_allele_cn (cell_id -> (n_seg,2) (p_cn,m_cn)); imbalanced = |p-m|>=1; allele-only = imbalanced & even
  total.
- Programs: PC.expression_params(); program_truth = t.program_truth (keys loading K×genes, program_names,
  program_genes, gene_program_map). cd["cell_program"] cells×K, columns=program names. Recurrence/shared: two
  tumours grown WITHOUT overriding layout_seed share the SAME loading + gene roles (comparability).
  positional_clustering / scattered_null diagnostics live in validation/programs_common.py.
- True clonal tree: t.genotypes_parents (child_gid -> parent_gid); Newick via iscc.integrations.to_newick.
  Per-lineage event order for epistasis: t.epistasis_ground_truth() (see count.py). validate_multiregion_phylo.py
  is the phylo reference.
- WGD ground truth: cd["cell_wgd"]["is_wgd"]; per-genotype ploidy t.genotypes[g].genome_summary["ploidy"].

============================================================================================================
NOTEBOOK 0 — notebooks/base_simulation.ipynb  (SHOW the shared simulation)
============================================================================================================
Grow the base tumour via base_sim.grow_base_tumor() and SHOW it (this is the "separate notebook" the user
asked for). Cells:
  1. MD: intro — "every science notebook in this folder is built on THIS one spatially-structured tumour"; list
     the features on (spatial gland + microenvironment, CINner selection, WGD, allele-specific expression, gene
     programs), and that it grows to ≥10k cancer cells.
  2. CO: from base_sim import grow_base_tumor; t = grow_base_tumor(); print malignant / normal counts, purity,
     WGD fraction, #genotypes; assert cancer ≥ 10000.
  3. CO: SPATIAL structure — tumor.plot_grid + a cell_crd scatter coloured by cell type (cancer vs epithelial
     vs stromal); show the gland with its microenvironment.
  4. CO: clonal dynamics — tumor.plot_muller; note the clone/genotype richness.
  5. CO: the ground-truth matrices (cell_snv / cell_cnv / cell_exp heatmaps, as in 01_pipeline_walkthrough),
     one line each on what later notebooks read.
  6. MD: a table/list mapping each downstream notebook to the aspect it uses. Next pointers.

============================================================================================================
SINGLE-TUMOUR SCIENCE NOTEBOOKS (each: `from base_sim import grow_base_tumor`; mixed microenvironment; self-contained)
============================================================================================================
NB1 notebooks/combining_scdna_scrna.ipynb  (REWRITE the stub — flagship integration; delete the SCICoNE/SCATrEx text)
  Story: scDNA + scRNA from the SAME mixed tumour → the DNA↔RNA link EMERGES (per-allele dosage) → separate
  malignant/normal → reconstruct subclones from RNA using DNA-defined CN profiles (clonealign's idea), scored
  vs truth. Cells: intro (non-circularity; mixture); grow + assemble a realistic-purity mixture; spatial viz by
  cell type; scRNA on the mixture + embed/cluster showing malignant vs normal separate (colour by true type);
  define clones on malignant CN + consensus heatmap; scDNA concordance; per-segment expression-vs-CN dosage
  (normal cells anchor CN≈2); reconstruct malignant clones from RNA + accuracy vs truth; pointer to
  validation/validate_clonealign.py / validate_infercnv.py / validate_numbat.py.

NB2 notebooks/wgd_allele_cna.ipynb  (WGD + allele-specific CNA — the ground-truth view of the Numbat result)
  Notation: a normal genome is 1+1; WGD → 2+2; the copy-neutral-detectable states are e.g. 4+0 / 3+1 at a total
  matching the balanced 2+2. Cells: intro; grow (WGD already on in base) + print WGD fraction/purity; spatial
  viz coloured by is_wgd with microenvironment shown; per-cell ploidy histogram over the WHOLE mixture (normal
  ~2 anchor; malignant split by is_wgd); MD (a pure doubling is balanced → unidentifiable from total CN+BAF;
  only doubling+LOSS creates imbalance); allele-only fraction WGD-off (grow_base_tumor with a wgd_rate=0 variant
  — pass a modified config) vs WGD-on via segment_allele_cn (≈1% → several %); the emergent |cell_rna_baf-0.5|
  at CN-altered vs balanced, MALIGNANT vs NORMAL (normal ~0.5 anchor); pointer to
  validation/validate_numbat.py --wgd-rate.

NB3 notebooks/gene_programs.ipynb  (R13 expression realism; analyse the mixture)
  Cells: intro (programs live in malignant cells; a real scRNA study sees all cell types, so a factor model
  must separate cell-type identity from within-malignant programs); grow + scRNA on the mixture; show
  cd["cell_program"] for malignant; spatial viz coloured by dominant program; true loading heatmap +
  program_genes SCATTERED across the genome; embed the MIXTURE coloured by cell type AND by dominant program;
  MD (programs scattered vs CNAs contiguous — the confound); positional_clustering / scattered_null diagnostic;
  recovery via sklearn NMF on the MIXTURE counts (Hungarian-match to true loading, report cosine for malignant
  programs; note normal cells load a distinct cell-type factor); pointer to validation/validate_programs.py.

NB4 notebooks/tree_inference_dna.ipynb  (NEW — clonal tree inference from bulk DNA-seq AND scDNA-seq)
  iscc supplies the TRUE clonal tree (t.genotypes_parents; iscc.integrations.to_newick) — ground truth no real
  dataset has. Cells:
   1. MD: intro — reconstructing the clonal phylogeny from DNA, two ways; iscc gives the true tree to score
      against. (One line: the multi-region caveat — a bulk "sample tree" ≠ the clone tree — is quantified in
      validation/validate_multiregion_phylo.py.)
   2. CO: grow the base tumour; render the TRUE clone tree (from genotypes_parents; a simple layout or to_newick
      + a light tree plot). Spatial viz of where the clones sit (cell_crd coloured by clone).
   3. CO: BULK DNA-seq route — bulkDNA over the mixed tumour (or over a few spatial regions via Biopsy multi-
      region); cluster mutations by VAF into subclones and build a subclonal tree (pigeonhole/ordering, or a
      simple hierarchical tree of VAF-clusters). Compare topology to the true clone tree.
   4. CO: scDNA-seq route — scDNA on a few hundred cells of the MIXTURE; build a distance tree
      (NJ/UPGMA on per-cell SNV±CNV profiles); normal cells root/outgroup near diploid. Compare to the true
      tree (e.g. Robinson–Foulds or a clade-recovery score) — single-cell resolves subclones bulk VAF cannot.
   5. MD: bulk-vs-single-cell trade-off; pointer to validate_multiregion_phylo.py (+ RevBayes head-to-head if
      relevant). Keep the tree methods lightweight (scipy/sklearn/Bio.Phylo) — do NOT require a heavy phylo env.

NB5 notebooks/scrna_visium_integration.ipynb  (NEW — integrate scRNA-seq reference with Visium spatial data)
  Real spatial workflow: a scRNA reference deconvolves Visium spots (each spot = a mixture of cells). iscc knows
  the TRUE per-spot composition (Visium .obs). Cells:
   1. MD: intro — Visium spots mix several cells; use a scRNA reference to deconvolve; iscc gives true spot
      composition. The mixture (malignant + microenvironment) is the whole point of deconvolution.
   2. CO: grow the base tumour; Visium(...) over the section → spots × genes AnnData + per-spot truth; scRNA(...)
      on a cell subset as the reference (with cell-type / clone labels).
   3. CO: spatial viz — the Visium spots over the section (obsm["spatial"]) coloured by dominant clone/type
      (from .obs); alongside the single-cell cell_crd map. Show spots straddle the tumour/microenvironment
      boundary (the spatial-mixing artifact).
   4. CO: deconvolve — build per-cell-type (and/or per-clone) mean expression signatures from the scRNA
      reference; solve each spot with NNLS (scipy.optimize.nnls) for the cell-type/clone fractions.
   5. CO: score — inferred vs true spot fractions (correlation / per-type scatter); show recovery + where it
      blurs at boundaries. Pointer to validation/validate_deconvolution.py + validate_visium.py (cell2location /
      RCTD do this properly).

============================================================================================================
COHORT NOTEBOOKS (5 simulations; same layout_seed → SHARED programs / gene roles / epistasis = ground truth)
============================================================================================================
Both grow the cohort via base_sim.grow_cohort() (5 seeds, shared landscape). Grow ONE AT A TIME, assay to a
compact matrix, release (MEMORY note above). Because layout_seed is shared, the true gene programs, the gene
roles (oncogenes/TSGs/drivers), and the epistasis landscape are IDENTICAL across the 5 tumours — that is the
cross-patient ground truth these notebooks recover.

NB6 notebooks/cohort_shared_programs.ipynb  (batch integration of scRNA + scDEF → shared programs)
  1. MD: intro — 5 "patients" share the same underlying programs (comparability); can we recover the SHARED
     programs across batches despite per-tumour batch effects? (This is the cohort-scale integration thesis.)
  2. CO: grow_cohort(); for each tumour, scRNA on a mixed subsample; tag each cell with a batch id; concatenate
     into one AnnData; show the raw batch effect (cluster by batch, not biology).
  3. CO: batch-correct — reuse validation/harmony_runner.py (iscc-harmony env) or scDEF's own batch path via
     validation/scdef_runner.py (iscc-scdef env), guarded; if absent, fall back to a simple in-core correction
     (e.g. per-batch centring / scanpy Harmony if importable) and SAY which path ran.
  4. CO: run scDEF (validation/programs_common.run_tool("scdef", ..., batch_key=...)) OR the fallback (NMF on
     the integrated matrix); recover programs shared across the cohort.
  5. CO: score vs the SHARED true loading (program_truth is identical across tumours): recovered-vs-true cosine;
     show the shared programs are recovered across patients. Pointer to validation/validate_programs_cohort.py.

NB7 notebooks/cohort_mhn_recurrence.ipynb  (MHN + TreeMHN on DNA → recurrent mutational patterns)
  1. MD: intro — across 5 patients sharing gene roles (+ optionally an epistasis landscape), which mutations
     RECUR and in what ORDER? MHN learns recurrent co-occurrence/exclusivity; TreeMHN adds ordering.
  2. CO: grow_cohort(); for each tumour, call scDNA/bulkDNA (or read the ground-truth driver events directly)
     to build a patients × genes binary alteration matrix over the recurrent driver genes; show the recurrence
     spectrum (which genes recur across patients).
  3. CO: MHN — run validation/mhn_runner.py (iscc-mhn env), guarded; else a self-contained co-occurrence /
     pairwise odds-ratio approximation. Show the inferred interaction network.
  4. CO: TreeMHN — use the per-lineage event ORDER ground truth (t.epistasis_ground_truth()) as the mutation
     trees input; run TreeMHN (its env) guarded, else summarise the recurrent ORDERINGS directly. 
  5. CO: score vs ground truth — recovered recurrent drivers / interactions / orderings vs the shared gene
     roles + epistasis landscape. Pointer to validation/validate_epistasis.py. (If enabling an explicit
     epistasis network gives a cleaner ground truth, add its params to SELECTION per validate_epistasis.py and
     note it.)

ACCEPTANCE / DELIVERABLES
- notebooks/base_sim.py + these notebooks: base_simulation, combining_scdna_scrna (rewrite), wgd_allele_cna,
  gene_programs, tree_inference_dna, scrna_visium_integration, cohort_shared_programs, cohort_mhn_recurrence —
  each EXECUTED with outputs, running clean in the core env, each growing a structure_radius>0 tumour to ≥10k
  cancer cells (single-tumour) / ≥10k per cohort member, and showing its spatial structure + microenvironment.
- Keep per-notebook runtime sane: the base grow is ~40–60 s; assays/NMF/trees on subsampled cells are the rest.
  Cohort notebooks grow 5 tumours sequentially (~4–6 min) — assay-then-release to bound RAM.
- Do NOT touch mkdocs.yml or docs/tutorials/ (notebooks/ only). Commit on `dev` with the Co-Authored-By trailer.
  A one-line BACKLOG.md note under the notebook/pedagogy track is welcome.

HONEST NOTES: if a self-contained reconstruction (clone recovery, NNLS deconvolution, NMF, a distance tree)
lands weak, report the number honestly — the notebooks SHOW the ground truth and the emergent signal; the
validation/ scripts own the tool claims. If reaching ≥10k cancer cells strains RAM, shrink GENOME or
structure_radius (keeping the microenvironment present) rather than lowering the target or dropping spatial
structure — ≥10k cancer cells, spatial structure, and an analysed microenvironment are all hard requirements.
For the cohort, if an external tool env (iscc-scdef / iscc-harmony / iscc-mhn / iscc-treemhn) is missing, the
notebook must still execute via the self-contained fallback and clearly state which path ran.
```
