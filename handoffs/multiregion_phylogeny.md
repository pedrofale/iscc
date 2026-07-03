# Handoff prompt — "Multi-region trees are not phylogenies" benchmark

Saved 2026-07-03. Copy the block below into a fresh session. A self-contained ground-truth benchmark
(scMultiSim/SISTEM convention) and a thematic sibling of the PEtracer confound ("spatial structure
misleads inference"). Companion: `BACKLOG.md` (Paper-1 benchmark suite), the PEtracer sections in
`manuscript/paper.tex` (the pattern to match). Branch from current `dev`.

---

```
Build the "multi-region trees are not phylogenies" benchmark for iscc: reproduce AND quantitatively
extend Alves, Prieto & Posada, "Multiregional Tumor Trees Are Not Phylogenies" (PMC5549612; Posada is
the CellCoal author, already cited as posada_cellcoal_2020). iscc uniquely has the TRUE clone
phylogeny AND real spatial admixture, so it can show — with a ground-truth answer key — that trees
built from multi-region bulk samples are misleading, and by how much.

REPO & ENV
- Repo: /Users/pedroferreira/projects/iscc/repo (branch `dev`).
- Python/pytest: ~/miniconda3/envs/iscc/bin/python.
- Conventions: commit on `dev` WITH `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`; keep
  the full pytest suite green; be honest. Match surrounding style.

THE CLAIM TO REPRODUCE: bulk multi-region samples are ADMIXED — each region is a MIXTURE of clones at
different proportions (not an isolated lineage), so a "sample tree" built from regional bulk
mutational/VAF profiles reflects SIMILARITY, not evolutionary history → spurious parallel mutations,
biased divergence times, reversed ordering. The paper's fix: deconvolve clones per region first, then
build the tree. The paper used illustrative simulations only; iscc adds a QUANTITATIVE sweep vs
ground truth.

ENABLERS (verified this session):
- TRUE phylogeny: the engine tracks `genotypes_parents` (child gid → parent gid); cells map to gids
  via the `cell_type` column; `iscc.integrations.to_newick(tumor)` / `to_lineage_tree(tumor)` export it.
- REAL spatial admixture: grow with LOW `dispersal_rate` → clonal territories that still intermix, so
  each region contains a clone mixture. Sweep `dispersal_rate` to vary admixture.
- Multi-region biopsy: `from iscc.sample.biopsy.biopsy import Biopsy`;
  `Biopsy(cell_data, rng).sample(biopsy_type="multiregion", n_regions=k)` → (chosen, region_series,
  geom); `region_series` gives each sampled cell its region label.
- Bulk regional profile: `from iscc.data import bulkDNA`; run on each region's cells → per-gene
  coverage + alt counts → a regional VAF / mutation-presence profile (the admixed bulk).
- Per-cell ground truth in `cell_data`: `cell_snv` (which cell carries which mutation), `cell_type`
  (true clone), `cell_crd`. So the TRUE per-mutation lineage history is known.

THE DEMONSTRATION
1. Grow a spatial multi-clone tumour; take K multi-region biopsies; bulk-DNA each region → regional
   mutation/VAF profiles (admixed).
2. Build the NAIVE "sample tree" from the regional profiles (e.g. neighbour-joining on a region×region
   VAF/Hamming distance, or parsimony on mutation presence) — the tree a multi-region study would draw.
3. Score it against the TRUTH using metrics that don't require matched leaf sets (region-leaves vs
   clone-leaves differ), the key one being **spurious parallel mutations**: count mutations the
   region-based reconstruction infers as INDEPENDENTLY arising (homoplasy / convergent) but which the
   lineage shows are a SINGLE event — the direct signature of admixture. Also report divergence-time /
   ordering distortions where computable.
4. **More regions doesn't fix it:** sweep K and show the spurious-parallelism rate does NOT vanish
   (admixture, not sampling density, is the problem) — the counterintuitive headline.
5. **Admixture drives it:** sweep `dispersal_rate` (territories ↔ intermixed) and show the error scales
   with the per-region clone-admixture.
6. **The fix:** deconvolve clones per region first (a Clomial-style per-region clone-fraction recovery,
   or use iscc's per-cell truth as the oracle deconvolution to bound the achievable), then build the
   clone tree and compare to the true clone tree (RF distance on matched clone leaves) → recovers the
   truth. Mirrors the paper's Clomial comparison.

DELIVERABLES
- `validation/validate_multiregion_phylo.py` → `manuscript/figures/validation_multiregion_phylo.png`
  (naive sample tree vs true clone tree; spurious-parallelism rate vs #regions and vs dispersal; the
  deconvolution fix). Print headline numbers.
- Tests `tests/test_multiregion_phylo.py` (small deterministic tumour: the naive tree has > 0 spurious
  parallelisms while the deconvolved/true tree does not; error does not fall to 0 with more regions).
  Guard any optional tree dep gracefully.
- Manuscript: fold into the "iscc as a benchmarking substrate" arc — a short Results paragraph beside
  the PEtracer section ("spatial admixture makes multi-region sample trees misleading"), cite the
  Alves/Prieto/Posada paper (ADD the bib entry, flagged auto-added/verify) and posada_cellcoal_2020.
- Flip the BACKLOG item to DONE. Run the full suite; commit on `dev`.

METHOD/DEPS: neighbour-joining + distances via scipy/numpy (self-contained); a small Robinson–Foulds
for the deconvolved-vs-true clone-tree comparison (or `dendropy` if you install it — guard it). No
ete3/cassiopeia needed. Keep the tumour small enough that the analysis is fast and deterministic.
```
