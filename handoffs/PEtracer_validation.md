# Handoff prompt — PEtracer validation (lineage + spatial expression decomposition)

Saved 2026-07-03. Copy the block below into a fresh session. Companion context: `BACKLOG.md`
(PEtracer entries), memory `iscc-paper-positioning.md`, and the F8/F9 features this depends on
(`DESIGN_features.md` §H + F8/F9 milestones). Branch from current `dev` (after the F8 perfused
commit). Enablers below were verified in the planning session.

---

```
Implement the PEtracer validation for iscc — two tiers. This validates iscc's core differentiator
(the COUPLING of lineage + space + expression) against the analysis the Weissman lab pioneered.

REPO & ENV
- Repo: /Users/pedroferreira/projects/iscc/repo (branch `dev`).
- Python/pytest: ~/miniconda3/envs/iscc/bin/python.
- Conventions: commit on `dev` WITH the `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`
  trailer; keep the full pytest suite green; be honest (no overselling). Match surrounding style.

BACKGROUND
PEtracer (Weissman lab, Science 2025; https://www.science.org/doi/10.1126/science.adx3800) reads
lineage (prime-editing barcodes) + MERFISH spatial expression jointly, and decomposes expression
into CELL-INTRINSIC (heritable, lineage-autocorrelated) vs CELL-EXTRINSIC (spatial/microenvironment)
modules — via Hotspot / PhyloVision run on the lineage tree vs the spatial graph. iscc now has BOTH
components WITH GROUND TRUTH:
  * intrinsic/heritable = clone/genotype/CNA-driven expression (cells sharing a lineage share it);
  * extrinsic/spatial   = the F8 hypoxia + CCI programs (spatial, NOT heritable);
  * readout             = the F9 single-cell spatial assay (`scSpatial`: panel counts + coords);
  * lineage             = the engine's `genotypes_parents` (child gid -> parent gid) → a clone tree.

THE FLAGSHIP FINDING — the lineage-space CONFOUND (design Tier 1 to expose it).
In a growing tumour, clonal territories mean closely-related cells sit in the SAME spatial region.
So a purely ENVIRONMENTAL signal (hypoxia) acquires LINEAGE autocorrelation, and a tree-based method
(Hotspot-on-tree) MIS-CLASSIFIES it as heritable. Real PEtracer data cannot catch this — there is no
ground truth. iscc KNOWS the true category (F8 genes are extrinsic by construction), so it can reveal
the mis-attribution. This is the point of the whole exercise; make it the headline. The confound is a
FUNCTION of the lineage-space correlation, which iscc can TUNE via the cancer `dispersal_rate`:
  * LOW dispersal  -> strong clonal territories -> lineage≈space -> hypoxia genes look heritable (confound);
  * HIGH dispersal -> intermixed clones -> lineage⊥space -> confound resolved.
Showing both is the demonstration.

ENABLERS (verified in planning):
- `on.genotypes_parents` maps child genotype_id -> parent genotype_id (str ids); cell_type column
  holds each cell's genotype_id. Cancer cells: `t.genotypes[gid].type == "cancer"`.
- F8: `t.microenv_truth["hypoxia_genes"]` / `["cci_target_genes"]` are the EXTRINSIC ground-truth sets;
  per-cell `cell_microenv` has hypoxia_level/cci_level. Use `o2_source="perfused"`.
- F9: `from iscc.data import scSpatial`; `scSpatial(platform="merfish", n_panel_genes=..).run(cell_data)`.
- squidpy / scanpy / anndata installed; Hotspot / phylovision / cassiopeia / ete3 NOT (hotspotsc is a
  pip install if you want the exact tool). Method: implement autocorrelation DIRECTLY (self-contained)
  — spatial via squidpy or a Gaussian-weight Moran's I on (jittered) coords; lineage via a tree-graph
  Moran's I. Optionally cross-check with real Hotspot.

=== TIER 1 — self-contained ground-truth benchmark (do first) ===
1. First-class lineage export: a small helper (start the `iscc.integrations` seam) building the clone
   tree from `genotypes_parents` — `to_lineage_tree(tumor)` (adjacency + per-genotype LCA distances)
   and/or `to_newick(tumor)`. Also `to_anndata(cell_data)` for the F9 output.
2. Grow tumours with F8 ON at LOW and HIGH dispersal; emit F9; restrict to cancer cells.
3. Per gene compute LINEAGE autocorrelation (tree-distance-weighted Moran's I) and SPATIAL
   autocorrelation (coord Moran's I). Ground-truth categories: extrinsic (F8 hypoxia/CCI), intrinsic
   (top between-genotype-variance genes from the F8-OFF run), neutral (rest).
4. `validation/validate_petracer.py` -> `manuscript/figures/validation_petracer.png`:
   (A) clone spatial map (shows lineage≈space at low dispersal); (B) per-gene scatter spatial-vs-
   lineage autocorr coloured by TRUE category — hypoxia genes landing high on the LINEAGE axis = the
   confound; (C) confound vs dispersal: hypoxia-gene apparent-heritability rises with the lineage-space
   correlation; (D) the resolved case (high dispersal). Print the recovery/confusion numbers.
5. Tests (`tests/test_petracer.py`): lineage tree well-formed (founder root, distances ≥0); the
   decomposition recovers the split under intermixing; the confound (hypoxia lineage-autocorr high)
   appears under clonal territories. Keep it a deterministic, small-tumour test.

=== TIER 2 — real PEtracer data comparison (then) ===
6. Reduce the real data to a compact cached form (the DNA-reference pattern in
   `validation/data/build_dna_reference.py`: reduce raw -> small .npz, DON'T commit raw, fall back to
   Tier-1/synthetic when absent). Sources: Figshare 10.6084/m9.figshare.28473866 (processed MERFISH
   h5ad + lineage trees, custom `h5td`), GEO GSE290975 (scRNA); code/format reference
   https://github.com/jweissmanlab/PEtracer-2025 (see `tumor_tracing/`). NOTE: Figshare blocks bots —
   the user may need to download manually.
7. Compute the SAME statistics on real vs iscc: per-gene lineage/spatial autocorrelation
   distributions, clone-territory sizes, tree balance/clone-size distribution. Show iscc reproduces
   the regime. CAVEATS to state honestly: MERFISH↔F9 platform match is fine; the model is a mouse
   syngeneic METASTASIS (multi-site — iscc grows single tumours, multi-site = RESEARCH_QUESTIONS R9,
   OUT OF SCOPE) so validate PER-TUMOUR.

DELIVERABLES: the `iscc.integrations` lineage/anndata export helpers (+ tests); validate_petracer.py
+ figure; Tier-2 reduce script + comparison; docs (flip BACKLOG PEtracer items to DONE, note the
confound finding). Run the full suite; commit on `dev` with the Co-Authored-By trailer.

STARTER SCAFFOLD (the planning-session prototype — the core computation; adapt/expand):
--------------------------------------------------------------------------------
import numpy as np
from iscc.tumor.models import GenotypeTumor
GENOME={"n_segments":10,"segment_size":100}
SEL={"prop_driver":0.1,"prop_dispersal":0.1,"prop_immune_resistance":0.1,"prop_treatment_resistance":0.1}
CANCER={"division_rate":0.6,"death_rate":0.03,"max_birth_rate":0.9,"mutation_rate":0.6,"dispersal_rate":0.10}  # LOW disp -> territories
DEME={"carrying_capacity":6}; SPATIAL={"grid_size":26,"structure_radius":0}
MP={"hypoxia":{"strength":1.0,"n_genes":60,"o2_consumption":1.5,"o2_supply":0.5,"o2_source":"perfused"},
    "cci":{"strength":0.8,"n_target_genes":60,"emitter_type":"cancer","lengthscale":3.0}}
t=GenotypeTumor(seed=3,genome_params=GENOME,selection_params=SEL,cancer_cell_params=CANCER,
                deme_params=DEME,spatial_params=SPATIAL,microenv_params=MP); t.grow(n_steps=320,seed=3)
cd=t.cell_data; gid=cd["cell_type"]["cell_id"].astype(str).values
is_cancer=np.array([g in t.genotypes and t.genotypes[g].type=="cancer" for g in gid])
# genotype tree distances (LCA) from t.genotypes_parents:
parents=t.genotypes_parents
def root_path(g):
    p=[g]
    while g in parents: g=parents[g]; p.append(g)
    return p
present=sorted(set(gid[is_cancer])); paths={g:root_path(g) for g in present}
depth={g:len(paths[g])-1 for g in present}
def dist(a,b):
    sa=set(paths[a])
    for x in paths[b]:
        if x in sa: return depth[a]+depth[b]-2*depth[x]
    return depth[a]+depth[b]
gl=list(present); gi={g:i for i,g in enumerate(gl)}; G=len(gl); Dg=np.zeros((G,G))
for i in range(G):
    for j in range(i+1,G): Dg[i,j]=Dg[j,i]=dist(gl[i],gl[j])
# cell-level lineage + spatial weights, vectorised Moran's I over all genes:
cells=np.where(is_cancer)[0]
cg=np.array([gi[gid[c]] for c in cells]); exp=cd["cell_exp"].values[cells]
crd=cd["cell_crd"].values[cells].astype(float)+np.random.default_rng(1).normal(0,0.15,(len(cells),2))
Wlin=np.exp(-Dg[np.ix_(cg,cg)]/2.0); np.fill_diagonal(Wlin,0)
d2=((crd[:,None,:]-crd[None,:,:])**2).sum(-1); Wsp=np.exp(-d2/(2*2.0**2)); np.fill_diagonal(Wsp,0)
def moran_all(W,Z):
    Z=Z-Z.mean(0); num=(Z*(W@Z)).sum(0); den=(Z**2).sum(0)
    return (len(Z)/W.sum())*np.divide(num,den,out=np.zeros_like(num),where=den>0)
I_lin=moran_all(Wlin,exp); I_sp=moran_all(Wsp,exp)
hyp=t.microenv_truth["hypoxia_genes"]     # extrinsic ground truth; expect I_lin[hyp] HIGH under low dispersal = the confound
--------------------------------------------------------------------------------
Watch for: cells share a deme coordinate (jitter for spatial KNN, as above); subsample cancer cells
(~600) if the N×N weights get large; use the F8-OFF twin run to define the intrinsic (clone-driven)
gene set; consider running on F9 OBSERVED counts (adds realistic noise) as well as true cell_exp.
```
