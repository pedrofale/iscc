"""Builder for notebooks/assay_spatial.ipynb — the Visium (spatial transcriptomics) tutorial.

Run with the iscc env to (re)generate the notebook, then execute it with nbconvert.
Kept in-repo so the tutorial is reproducible.
"""
import os
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))


md(r"""# Spatial transcriptomics: a Visium slide over a simulated tumour

We grow one cm-scale breast lesion, cut a dense window out of it, lay a 10x Visium slide over that
window and assay it — then check every spot against the ground truth we still hold: which cells it
pooled, which clone they belong to, and what they were really expressing.

What you get:

1. the ground truth — cell types, clones, gene programs, traits;
2. an H&E image of the tissue;
3. the Visium spots on that tissue, coloured by the true clone and cell type under each spot;
4. what each assay parameter does;
5. spot counts vs the true expression of the cells underneath;
6. normalise + UMAP the spots;
7. fit the assay parameters back from real 10x data.""")

code(r"""%matplotlib inline
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scanpy as sc
import squidpy as sq
from scipy.ndimage import uniform_filter
from scipy.spatial import cKDTree

sys.path.insert(0, os.getcwd())          # notebooks/ — the shared tumour substrate
import base_sim as B

from iscc.data import (Visium, VisiumBatch, VisiumBatchHyperParams, morans_i,
                       estimate_visium, estimate_visium_from_assay)
from iscc.integrations import to_anndata, clones_from_clades, clone_summary, biological_types

sc.settings.set_figure_params(dpi=80, facecolor="white")
sc.settings.verbosity = 0
np.random.seed(0)""")

md(r"""## The tissue

`grow_base_tumor` grows the shared substrate: a multi-focal duct lesion on a centimetre-scale
field, with the gene-program layer switched on. Takes a couple of minutes.""")

code(r"""tumor = B.grow_base_tumor()
G = tumor.grid_size
print(f"grid {G} x {G} demes, {tumor.get_tumor_size():,} cells alive, "
      f"{len(tumor.cell_data['cell_crd']):,} materialised as a whole-tumour subsample")""")

md(r"""**The density trap.** A whole-tumour subsample spread over the full field leaves about *one*
cell per Visium spot — nothing to deconvolve. So find the busiest window, then materialise **that
window only**, at local density: `primary_window` picks the demes, `make_cell_data(region=...)`
expands them. `depth_frac` thins each deme's cell column, which is what a thin physical section
does.""")

code(r"""SIDE = 90                       # window side, in demes
crd = tumor.cell_data["cell_crd"]
cancer = biological_types(tumor.cell_data["cell_type"].iloc[:, 0]) == "cancer"
dens, _, _ = np.histogram2d(crd["row"].values[cancer], crd["col"].values[cancer],
                            bins=G, range=[[0, G], [0, G]])
row_c, col_c = np.unravel_index(uniform_filter(dens, SIDE, mode="constant").argmax(), dens.shape)

section = tumor.make_cell_data(region=tumor.primary_window(side=SIDE, center=(row_c, col_c)),
                               depth_frac=0.10, max_cells=60000)
types = pd.Series(biological_types(section["cell_type"].iloc[:, 0]), index=section["cell_crd"].index)
print(f"window centred on deme ({row_c}, {col_c}), {SIDE} x {SIDE} demes")
print(f"{len(section['cell_crd']):,} cells in the section  ->  "
      + ", ".join(f"{n:,} {t}" for t, n in types.value_counts().items()))""")

md(r"""## 1. Ground truth

Everything below is checked against this. `clones_from_clades` gives the shared clone definition —
a clade of the true lineage tree holding at least `min_cells` sampled cells. Normal cells get no
clone; cancer cells above every clade land in `other`.""")

code(r"""clones = clones_from_clades(tumor, min_cells=150)
summary = clone_summary(tumor, clones)
print(f"{len(summary) - 1} clones + the ancestral backbone ('other')")
summary[["n_cells", "n_genotypes", "division_rate", "breach", "stromal_survival",
         "immune_resistance", "traits"]].round(3)""")

md(r"""One `AnnData` carries the whole section: expression in `X`, the mutation and copy-number
matrices as layers, coordinates in `obsm["spatial"]`, program activity in `obsm["program"]`, and
the true program-by-gene loadings in `varm`. Copy the program activities into `.obs` so scanpy can
plot them.""")

code(r"""ad = to_anndata(tumor)
ad.obs["clone"] = clones
programs = list(ad.uns["program_names"])
ad.obs[programs] = ad.obsm["program"]
ad""")

md(r"""### Program activity by cell type and by clone

The invasive `emt` program is driven two ways here: genetically, by the `breach` trait, and by the
niche — the epithelial compartment itself pushes it up. Both show.""")

code(r"""by_type = ad.obs.groupby("cell_type", observed=True)[programs].mean()
by_clone = ad.obs[ad.obs.cell_type == "cancer"].groupby("clone", observed=True)[programs].mean()

fig, axes = plt.subplots(1, 2, figsize=(13, 3.6),
                         gridspec_kw={"width_ratios": [len(by_type), len(by_clone)]})
for ax, tab, title in zip(axes, [by_type.T, by_clone.T], ["by cell type", "by clone (cancer only)"]):
    im = ax.imshow(tab.values, cmap="magma", aspect="auto", vmin=0.2, vmax=1.2)
    ax.set_xticks(range(tab.shape[1])); ax.set_xticklabels(tab.columns, rotation=45, ha="right")
    ax.set_yticks(range(tab.shape[0])); ax.set_yticklabels(tab.index)
    ax.set_title(f"mean program activity {title}")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
plt.tight_layout(); plt.show()
by_clone.round(2)""")

md(r"""Epithelial cells run `emt` several times above cancer and stroma — that is the niche arm.
The clones then differ among themselves, which is the genetic arm. Every other program sits near
0.5 in every group: deliberately flat, so anything a method finds there is a false positive.""")

md(r"""### Where the cells and the programs are""")

code(r"""axs = sc.pl.embedding(ad, basis="spatial", color=["cell_type", "clone"], size=10,
                      na_color="#e9e9e9", ncols=2, show=False)
for a in np.atleast_1d(axs):
    a.invert_yaxis()   # match the image convention used by the H&E and squidpy plots below
plt.show()""")

code(r"""axs = sc.pl.embedding(ad, basis="spatial", color=programs, size=10, ncols=3,
                      vmax="p99", show=False)
for a in np.atleast_1d(axs):
    a.invert_yaxis()
plt.show()""")

md(r"""Several cancer foci in a stromal field, and each focus is a coherent patch of clones —
clonal structure here is spatial, not scattered. `emt` lights up on the foci and their epithelial
rings; the other programs are near flat, which is the null this notebook needs.""")

md(r"""### Traits, and the true program-by-gene loadings

The traits are the evolutionary state each clone carries. The loading matrix is what a
program-inference method would have to recover: 25 genes per program, out of 6,000.""")

code(r"""traits = ["division_rate", "breach", "stromal_survival", "immune_resistance",
          "n_mut_onc", "n_mut_tsg"]
display(ad.obs[ad.obs.cell_type == "cancer"].groupby("clone", observed=True)[traits].mean().round(3))

loading = pd.DataFrame(ad.varm["program_loading"], index=ad.var_names, columns=programs)
member = loading.loc[loading.abs().sum(1) > 0]
shared = int(((member != 0).sum(1) > 1).sum())
fig, ax = plt.subplots(figsize=(12, 2.6))
im = ax.imshow(member.T.values, cmap="magma", aspect="auto",
               vmin=0, vmax=np.quantile(member.values[member.values != 0], 0.95))
ax.set_yticks(range(len(programs))); ax.set_yticklabels(programs)
ax.set_xlabel(f"{len(member)} genes with a non-zero loading (of {ad.n_vars}); "
              f"{shared} belong to more than one program")
ax.set_title("true program x gene loadings")
fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
plt.show()""")

md(r"""The programs are near-disjoint gene sets, so almost nothing here is confounded by shared
membership — a method that mixes two programs did it on its own.""")

md(r"""## 2. An H&E image of the tumour

`he_image` paints every occupied deme from the full cell counts, so it shows the whole lesion, not
just the cells we materialised. Dense duct cores read dark purple, loose stroma pale pink.""")

code(r"""he, px = tumor.he_image(px=4, darkness=0.9, sigma_frac=0.5)
lo_r, lo_c = int((row_c - SIDE // 2) * px), int((col_c - SIDE // 2) * px)
span = int(SIDE * px)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(he)
axes[0].add_patch(patches.Rectangle((lo_c, lo_r), span, span, fill=False, ec="k", lw=1.6))
axes[0].set_title(f"whole lesion, {G} x {G} demes  (box = the assayed window)")
axes[1].imshow(he[lo_r:lo_r + span, lo_c:lo_c + span])
axes[1].set_title(f"the window, {SIDE} x {SIDE} demes")
for a in axes:
    a.axis("off")
plt.tight_layout(); plt.show()""")

md(r"""Multi-focal disease: several separate duct foci, three of them inside the window.""")

md(r"""## 3. The Visium slide on the tissue

`section_frac=1.0` places the section on the fixed 10x v1 slide (78 x 64 = 4,992 spots) and renders
the slide's own tissue image. `run` pools the cells under each spot and emits UMI counts.""")

code(r"""vz = Visium(seed=10, section_frac=1.0, spot_pitch=2.0, spot_radius=1.2, count_model="dm",
            mu_counts=8000.0, sigma_counts=0.45, field_lengthscale=14.0, field_sigma=0.6,
            edge_sigma=0.3, ambient_frac=0.05).run(section)
spots = vz.to_anndata()
on_tissue = spots.obs["n_cells"].values > 0
print(f"{spots.n_obs} spots on the slide, {on_tissue.sum()} of them over tissue")
print(f"cells per on-tissue spot: median {np.median(spots.obs['n_cells'][on_tissue]):.0f}, "
      f"mean {spots.obs['n_cells'][on_tissue].mean():.1f}, max {spots.obs['n_cells'].max()}")""")

md(r"""**Median 5 cells per spot** — the real Visium range. `spot_members` lists the cell ids under
each spot, so the ground-truth label of a spot is a majority vote over its cells. Vote over the
clone labels, never over the raw genotype id: there are more than a thousand distinct genotypes in
this section.""")

code(r"""def majority(labels, empty):
    # per-spot majority vote of a per-cell label Series; NaNs (e.g. normal cells) ignored
    out = []
    for ids in vz.spot_members:
        v = labels.reindex(pd.Index(ids)).dropna() if len(ids) else pd.Series(dtype=object)
        out.append(str(v.value_counts().idxmax()) if len(v) else empty)
    return pd.Categorical(out)

print(f"{section['cell_type'].iloc[:, 0].nunique()} distinct genotypes vs "
      f"{clones.dropna().nunique()} clone labels in this section")
spots.obs["type_major"] = majority(types, empty="off tissue")
spots.obs["clone_major"] = majority(clones.reindex(types.index), empty="no cancer cells")
spots.obs[["n_cells", "n_counts", "type_major", "clone_major"]].head()""")

md(r"""The placement affine is not stored anywhere, so the H&E of section 2 cannot be aligned onto
the slide pixel-for-pixel. Rather than fake it, the squidpy view below uses the **slide's own**
tissue image, the one the assay rendered and stored in `uns["spatial"]`.""")

code(r"""library = list(spots.uns["spatial"])[0]
on = spots[on_tissue].copy()

fig, axes = plt.subplots(1, 3, figsize=(19, 6))
axes[0].imshow(spots.uns["spatial"][library]["images"]["hires"])
axes[0].set_title("the slide's own tissue image"); axes[0].axis("off")
sq.pl.spatial_scatter(on, color="type_major", img=True, shape="circle", ax=axes[1],
                      frameon=False, title="majority cell type")
sq.pl.spatial_scatter(on, color="clone_major", img=True, shape="circle", ax=axes[2],
                      frameon=False, title="majority clone")
plt.tight_layout(); plt.show()""")

md(r"""Cancer-majority spots pick out the foci, and inside them the clone vote reproduces the
clonal patches of section 1. Most on-tissue spots hold no cancer cell at all — this section is
mostly stroma, which is exactly what makes spot-level deconvolution hard.""")

md(r"""## 4. What the parameters do

Four knobs shape the spatial technical signal. The first three are cheap to sweep with
`VisiumBatch` directly — it draws the capture field for a set of spot coordinates without
re-running the assay. Sweep them on the **on-tissue** coordinates, which is the array a real
section covers.""")

code(r"""coords = vz.spot_coords[on_tissue]
genes, base_profile = list(vz.spot_counts.columns), vz.raw_expr.values.sum(0)

def field(lengthscale=14.0, sigma=0.6, edge=0.0, seed=1):
    h = VisiumBatchHyperParams(field_lengthscale=lengthscale, field_sigma=sigma, edge_sigma=edge)
    return VisiumBatch(h, seed=seed).realize(genes, base_profile, coords).capture_field

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for ax, L in zip(axes[0], [2.0, 10.0, 40.0]):
    f = field(lengthscale=L)
    ax.scatter(coords[:, 1], coords[:, 0], c=f, cmap="viridis", s=6, vmin=0.2, vmax=2.2)
    ax.set_title(f"field_lengthscale={L:g}   Moran's I={morans_i(f, coords):.2f}")
for ax, S in zip(axes[1], [0.2, 0.6, 1.2]):
    f = field(sigma=S)
    ax.scatter(coords[:, 1], coords[:, 0], c=f, cmap="viridis", s=6, vmin=0.2, vmax=2.2)
    ax.set_title(f"field_sigma={S:g}   spread={f.std() / f.mean():.2f}")
for ax in axes.ravel():
    ax.invert_yaxis(); ax.set_aspect("equal"); ax.axis("off")
plt.tight_layout(); plt.show()""")

md(r"""`field_lengthscale` sets the *size* of the capture patches, `field_sigma` their *depth*.
Moran's I — the spatial autocorrelation of the field — rises with the length scale and saturates
once the patches are much bigger than the spot pitch.""")

code(r"""lengthscales = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
mi = [morans_i(field(lengthscale=L), coords) for L in lengthscales]
fig, ax = plt.subplots(figsize=(5.5, 3.6))
ax.plot(lengthscales, mi, "o-")
ax.set(xscale="log", xlabel="field_lengthscale (deme units; spot pitch = 2)",
       ylabel="Moran's I of the capture field", ylim=(0, 1.05))
plt.show()""")

md(r"""### The tissue edge

`edge_sigma` lowers capture near the boundary. Be careful what boundary: it measures distance to
the **extent of the spot array it is given**, not to the tissue outline. On the fixed slide those
are different things — the slide's edge spots are off-tissue already, so the falloff lands on empty
glass. Here it is drawn on the on-tissue array, where it does what it says.""")

code(r"""fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
f = field(sigma=0.0, edge=0.6)
sc0 = axes[0].scatter(coords[:, 1], coords[:, 0], c=f, cmap="viridis", s=7)
axes[0].set_title("edge_sigma=0.6 on the on-tissue array"); axes[0].invert_yaxis()
axes[0].set_aspect("equal"); axes[0].axis("off")
fig.colorbar(sc0, ax=axes[0], fraction=0.04)

border = np.minimum(np.minimum(coords[:, 0] - coords[:, 0].min(), coords[:, 0].max() - coords[:, 0]),
                    np.minimum(coords[:, 1] - coords[:, 1].min(), coords[:, 1].max() - coords[:, 1]))
for e in [0.0, 0.3, 0.6]:
    axes[1].scatter(border, field(sigma=0.0, edge=e), s=4, alpha=0.4, label=f"edge_sigma={e:g}")
axes[1].set(xlabel="distance to the edge of the spot array", ylabel="capture efficiency")
axes[1].legend()
plt.tight_layout(); plt.show()""")

md(r"""### Lateral mRNA diffusion

`diffusion_sigma` bleeds each spot's expression into its neighbours before counting — the spatial
blur deconvolution has to undo. This one needs a re-run, so use a square array over the window
instead of the full slide. Track a cancer signature: the 25 genes most up in cancer cells over
stromal cells in the ground truth.""")

code(r"""by_type_expr = section["cell_exp"].groupby(types).mean()
markers = list(np.log2((by_type_expr.loc["cancer"] + 1) / (by_type_expr.loc["stromal"] + 1))
               .sort_values().index[-25:])

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
for ax, s in zip(axes, [0.0, 2.0, 6.0]):
    a = Visium(seed=10, section_frac=1.0, spot_pitch=2.0, spot_radius=1.2, diffusion_sigma=s,
               mu_counts=8000.0, field_sigma=0.0, edge_sigma=0.0,
               ambient_frac=0.0).run(section, grid_side=SIDE)
    keep = a.obs["n_cells"].values > 0
    frac = (a.spot_counts[markers].sum(1) / a.obs["n_counts"].replace(0, np.nan)).values[keep]
    ax.scatter(a.spot_coords[keep, 1], a.spot_coords[keep, 0], c=frac, cmap="magma", s=9,
               vmin=0, vmax=np.nanquantile(frac, 0.98))
    ax.set_title(f"diffusion_sigma={s:g}"); ax.invert_yaxis(); ax.set_aspect("equal"); ax.axis("off")
    del a
fig.suptitle("cancer signature: fraction of the spot's counts", y=1.02)
plt.tight_layout(); plt.show()""")

md(r"""At `diffusion_sigma=0` the signature stops at the edge of each focus. At 6 it has smeared
right across the stroma between them, and a deconvolution method would report tumour where there is
none.""")

md(r"""## 5. What the spots measure vs what was under them

`raw_expr` is the summed true expression of a spot's cells and `n_cells` says how many there were,
so their ratio is the mean cell underneath. Compare it to the counts, gene by gene. Take the three
genes detected in the most spots — the ones a spot has any chance of measuring.""")

code(r"""from scipy.stats import rankdata, spearmanr

truth = vz.raw_expr.div(vz.obs["n_cells"].replace(0, np.nan), axis=0)
observed = vz.spot_counts.div(vz.obs["n_counts"].replace(0, np.nan), axis=0) * 1e4
detection = (vz.spot_counts.loc[on_tissue] > 0).mean()
share = vz.raw_expr.loc[on_tissue].sum() / vz.raw_expr.loc[on_tissue].sum().sum()
check = list(detection.sort_values(ascending=False).index[:3])

fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
for ax, g in zip(axes, check):
    x, y = truth[g].values[on_tissue], observed[g].values[on_tissue]
    ax.scatter(np.log1p(x), np.log1p(y), s=6, alpha=0.25)
    ax.axhline(np.log1p(1e4 * vz.hypers.ambient_frac * share[g]), color="crimson", ls="--", lw=1,
               label="ambient floor")
    ax.set(xlabel="log1p mean true expression under the spot", ylabel="log1p counts per 10k",
           title=f"{g}   {100 * share[g]:.1f}% of the library\nSpearman r = "
                 f"{spearmanr(x, y, nan_policy='omit').statistic:.2f}")
    ax.legend(fontsize=8)
plt.tight_layout(); plt.show()""")

md(r"""Two of the three track the truth closely. The third does not, and the dashed line says why:
it carries most of this tissue's transcripts, so the 5% ambient soup — drawn from the pooled
profile — puts a floor of a few hundred counts per 10k of it in *every* spot, on tissue or not.
Ambient contamination hurts the most abundant transcript the most.

Across all genes, recovery is a straight function of how often a gene is detected at all.""")

code(r"""tr = rankdata(truth.values[on_tissue], axis=0)
ob = rankdata(observed.values[on_tissue], axis=0)
tr, ob = tr - tr.mean(0), ob - ob.mean(0)
with np.errstate(invalid="ignore", divide="ignore"):
    r_gene = pd.Series((tr * ob).sum(0) / np.sqrt((tr ** 2).sum(0) * (ob ** 2).sum(0)),
                       index=truth.columns)
binned = r_gene.groupby(pd.cut(detection, [0, .02, .05, .1, .2, .4, 1.01], right=False),
                        observed=True).median()

fig, ax = plt.subplots(figsize=(7, 3.6))
ax.plot([str(i) for i in binned.index], binned.values, "o-")
ax.set(xlabel="fraction of on-tissue spots where the gene is detected",
       ylabel="median Spearman r")
ax.tick_params(axis="x", rotation=30)
plt.tight_layout(); plt.show()""")

md(r"""Individual genes are noisy; a **signature** is less so. The same cancer-marker set as above,
true against observed, on the same colour scale.""")

code(r"""score_true = (vz.raw_expr[markers].sum(1) / vz.raw_expr.sum(1).replace(0, np.nan)).values
score_obs = (vz.spot_counts[markers].sum(1) / vz.obs["n_counts"].replace(0, np.nan)).values
top = np.nanquantile(score_true[on_tissue], 0.95)
print("cancer-signature Spearman r = "
      f"{spearmanr(score_true[on_tissue], score_obs[on_tissue], nan_policy='omit').statistic:.2f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, vals, title in [(axes[0], score_true, "true cancer-signature fraction"),
                        (axes[1], score_obs, "observed cancer-signature fraction")]:
    sc_ = ax.scatter(vz.spot_coords[on_tissue, 1], vz.spot_coords[on_tissue, 0],
                     c=vals[on_tissue], cmap="magma", s=8, vmin=0, vmax=top)
    ax.set_title(title); ax.invert_yaxis(); ax.set_aspect("equal"); ax.axis("off")
    fig.colorbar(sc_, ax=ax, fraction=0.04)
plt.tight_layout(); plt.show()""")

md(r"""Pooled over 25 genes the assay puts the tumour back where it was — but on a raised
baseline. The truth is black across the stroma; the observation never is, because the ambient soup
gives every spot a little cancer signal.""")

md(r"""## 6. Normalise and UMAP the spots

Standard scanpy from here. Add the mean program activity of each spot's cells as ground truth to
colour by.""")

code(r"""prog = section["cell_program"]
spots.obs[programs] = np.vstack([prog.reindex(pd.Index(ids)).mean().values if len(ids)
                                 else np.full(len(programs), np.nan) for ids in vz.spot_members])

on = spots[on_tissue].copy()
sc.pp.normalize_total(on, target_sum=1e4)
sc.pp.log1p(on)
sc.pp.pca(on, n_comps=30)
sc.pp.neighbors(on, n_neighbors=15)
sc.tl.umap(on)
sc.pl.umap(on, color=["type_major", "clone_major", "emt", "proliferation", "n_cells"],
           ncols=3, size=25, frameon=False)""")

md(r"""Cancer-majority spots separate from stromal ones and the clone patches sit inside that arm —
but the boundary is soft, because a spot is a mixture and most of them are mostly stroma. `emt`
runs along the cancer arm; `n_cells` shows how much of the embedding is really driven by how much
tissue a spot caught.""")

md(r"""## 7. Fitting the parameters from data

`estimate_visium` reads the technical parameters back off a spot-by-gene count matrix and its
coordinates. First the round trip on our own assay, where the answer is known.""")

code(r"""est = estimate_visium_from_assay(vz)
truth_hypers = {"mu_counts": 8000.0, "sigma_counts": 0.45, "field_sigma": 0.6,
                "field_lengthscale": 14.0}
pd.DataFrame({"true": truth_hypers,
              "fitted": {k: getattr(est.hypers, k) for k in truth_hypers}}).round(3)""")

md(r"""Close, not exact — the fit sees the capture field only through the counts, where it is
mixed up with the biological signal, so it under-reads both the field's strength and its length
scale and pushes the surplus into the mean library size.

Only `mu_counts`, `sigma_counts`, `field_sigma`, `field_lengthscale` and the count overdispersion
come out of the fit at all — `est.fitted` says so. Diffusion, ambient fraction and the edge falloff
are carried from the preset: a single section's counts cannot separate them.""")

code(r"""print("fitted from the data:", est.fitted)
print("carried from the preset:",
      [k for k in VisiumBatchHyperParams().to_dict() if k not in est.fitted])""")

md(r"""### On a real 10x section

Now a real Visium breast section. **Its coordinates are in pixels** — a nearest-neighbour spacing
of a few hundred, not of ~2 — so the fitted length scale is meaningless until the coordinates are
rescaled to the same units the simulator uses. Rescale by the spot pitch and it becomes
interpretable.""")

code(r"""try:
    sc.settings.datasetdir = os.path.join(os.getcwd(), "..", "data")
    real = sc.datasets.visium_sge(sample_id="V1_Breast_Cancer_Block_A_Section_1")
    real.var_names_make_unique()
    xy = np.asarray(real.obsm["spatial"], dtype=float)
    pitch_px = float(np.median(cKDTree(xy).query(xy, k=2)[0][:, 1]))
    print(f"{real.n_obs} spots x {real.n_vars} genes; spot pitch = {pitch_px:.0f} pixels")

    fits = {"raw pixels": estimate_visium(real, coords=xy),
            "rescaled to spot pitch": estimate_visium(real, coords=xy / pitch_px * 2.0)}
except Exception as exc:                       # no cached dataset / no network -> stay offline
    print(f"real 10x section unavailable ({type(exc).__name__}: {exc});"
          " falling back to the simulated assay")
    fits = {"simulated assay": est}

pd.DataFrame({name: {k: getattr(f.hypers, k) for k in
                     ["mu_counts", "sigma_counts", "field_sigma", "field_lengthscale", "kappa"]}
              for name, f in fits.items()}).round(3)""")

md(r"""Depth and field strength are scale-free and identical either way; the length scale is not.
In pixels it comes out in the thousands, which says nothing. Rescaled, it lands near 18 — about
nine spot pitches, and exactly the shipped `field_lengthscale` default, because that default was
calibrated on this section.

`kappa` is a different story: fitted at ~89,000 on 36,601 mostly-zero real genes, it is a
sparse-transcriptome artefact and does not transfer to the dense gene model here. That is why the
Visium preset keeps its own value instead of taking this one.""")

md(r"""## Recap

* A cm-scale field has to be **windowed** before it is assayable — `primary_window` +
  `make_cell_data(region=..., depth_frac=...)` got us to a median of 5 cells per spot.
* Every spot keeps its cell list, so the clone and cell-type label of a spot is a majority vote
  over ground truth, and every count can be checked against the cells that produced it.
* The spatial signature lives in the capture field (`field_lengthscale`, `field_sigma`,
  `edge_sigma`) and in lateral diffusion (`diffusion_sigma`).
* Per-gene recovery follows detection, and the ambient soup puts a floor under the most abundant
  transcripts — pooling genes into a signature is what survives.
* `estimate_visium` recovers the identifiable half of that from real data — after you put the
  coordinates in sensible units.""")


nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assay_spatial.ipynb")
with open(out_path, "w") as f:
    nbf.write(nb, f)
print(f"wrote {out_path}")
