"""Lineage-vs-spatial expression decomposition (the PEtracer analysis).

PEtracer (Weissman lab, *Science* 2025) reads clonal lineage (prime-editing barcodes) and spatial
expression (MERFISH) jointly, then decomposes each gene's variation into a **cell-intrinsic**
(heritable, lineage-autocorrelated) component and a **cell-extrinsic** (spatial / microenvironment)
component — via a phylogenetic-autocorrelation method (Hotspot / PhyloVision) run on the lineage
tree versus the spatial graph.

This module reproduces that decomposition self-contained on an iscc tumour: for each gene it computes

  * ``I_lineage`` — Moran's I with a **lineage** weight ``w_ij = exp(-d_tree(i,j)/L)`` where
    ``d_tree`` is the LCA tree distance between the two cells' clones (a phylogenetic autocorrelation);
  * ``I_spatial`` — Moran's I with a **spatial** weight ``w_ij = exp(-d_ij^2 / 2L^2)`` on the cells'
    (jittered) coordinates.

Unlike real PEtracer data, iscc *knows the true category* of every gene (F8 hypoxia/CCI genes are
extrinsic by construction; clone-driven genes are intrinsic), which is what makes the **lineage–space
confound** measurable: under clonal territories a purely extrinsic (hypoxia) signal acquires lineage
autocorrelation and a tree-based method mis-labels it as heritable. See
``validation/validate_petracer.py``.
"""
import numpy as np
import pandas as pd

from .lineage import to_lineage_tree


def _moran_all(W, Z):
    """Vectorised Moran's I for every column of ``Z`` (cells x genes) under weight matrix ``W``.

    ``I_g = (n / sum W) * (z_g^T W z_g) / (z_g^T z_g)`` with ``z_g`` the mean-centred gene column.
    Genes with zero variance return 0."""
    Z = Z - Z.mean(axis=0, keepdims=True)
    num = (Z * (W @ Z)).sum(axis=0)
    den = (Z * Z).sum(axis=0)
    Wsum = W.sum()
    n = Z.shape[0]
    out = np.zeros_like(num)
    if Wsum <= 0:
        return out
    return (n / Wsum) * np.divide(num, den, out=out, where=den > 0)


class LineageSpatialDecomposition:
    """Per-gene lineage vs spatial autocorrelation + the ground-truth confound summary.

    Attributes
    ----------
    genes : list[str]
    table : pandas.DataFrame
        Indexed by gene, columns ``I_lineage`` / ``I_spatial``.
    cells : numpy.ndarray
        The cancer-cell ids the decomposition was computed on.
    tree : LineageTree
    lineage_space_corr : float
        Pearson correlation between pairwise lineage (tree) distance and spatial distance across the
        analysed cells — the driver of the confound (high ⇒ clonal territories ⇒ confound).
    """

    def __init__(self, genes, I_lineage, I_spatial, cells, tree, lineage_space_corr,
                 field_autocorr=None):
        self.genes = list(genes)
        self.I_lineage = np.asarray(I_lineage)
        self.I_spatial = np.asarray(I_spatial)
        self.cells = np.asarray(cells)
        self.tree = tree
        self.lineage_space_corr = float(lineage_space_corr)
        # lineage/spatial autocorrelation of the GROUND-TRUTH extrinsic fields (hypoxia/cci level),
        # measured directly (no expression noise): the confound driver — how much a purely spatial
        # field leaks onto the lineage axis under clonal territories. {field: {"lineage","spatial"}}.
        self.field_autocorr = dict(field_autocorr or {})
        self.table = pd.DataFrame(
            {"I_lineage": self.I_lineage, "I_spatial": self.I_spatial},
            index=pd.Index(self.genes, name="gene"),
        )

    def called_heritable(self):
        """Boolean per gene: a tree-based caller labels it heritable when lineage autocorrelation
        dominates spatial autocorrelation (``I_lineage > I_spatial``)."""
        return self.I_lineage > self.I_spatial

    def confusion(self, categories):
        """Fraction of genes CALLED heritable (``I_lineage > I_spatial``) within each true category.

        ``categories`` maps ``name -> gene-index array`` (indices into ``self.genes``). For a truly
        *extrinsic* category a high value is the mis-classification rate (the confound)."""
        herit = self.called_heritable()
        out = {}
        for name, idx in categories.items():
            idx = np.asarray(idx, dtype=int)
            out[name] = float(herit[idx].mean()) if idx.size else float("nan")
        return out


def _cancer_mask(tumor, cell_data):
    gid = cell_data["cell_type"].iloc[:, 0].astype(str).values
    genos = tumor.genotypes
    is_cancer = np.array([g in genos and genos[g].type == "cancer" for g in gid])
    return gid, is_cancer


def decompose_lineage_spatial(tumor, expression=None, lineage_lengthscale=2.0,
                              spatial_lengthscale=2.0, coord_jitter=0.15,
                              max_cells=600, seed=0):
    """Run the lineage-vs-spatial decomposition on a grown ``tumor`` (cancer cells only).

    Parameters
    ----------
    tumor : GenotypeTumor
        A grown tumour; uses ``tumor.cell_data`` (materialised if needed).
    expression : pandas.DataFrame, optional
        Override the expression matrix (cells x genes), e.g. F9 *observed* counts. Defaults to the
        true ``cell_exp``. Rows are aligned to the analysed cancer cells by id.
    lineage_lengthscale, spatial_lengthscale : float
        Kernel lengthscales for the lineage (tree-distance) and spatial (Euclidean) weights.
    coord_jitter : float
        Std of Gaussian jitter added to (row, col) so cells sharing a deme coordinate are separable.
    max_cells : int
        Subsample cancer cells to at most this many (the weight matrices are N x N).
    seed : int
        RNG seed for subsampling + jitter.

    Returns
    -------
    LineageSpatialDecomposition
    """
    if getattr(tumor, "cell_data", None) is None:
        tumor.make_cell_data()
    cell_data = tumor.cell_data
    gid, is_cancer = _cancer_mask(tumor, cell_data)

    rng = np.random.default_rng(seed)
    cells_idx = np.where(is_cancer)[0]
    if max_cells is not None and cells_idx.size > max_cells:
        cells_idx = np.sort(rng.choice(cells_idx, max_cells, replace=False))
    cell_ids = cell_data["cell_exp"].index.values[cells_idx]
    clones = gid[cells_idx]

    # expression matrix (true or observed) restricted to the analysed cells
    exp_df = cell_data["cell_exp"] if expression is None else expression
    exp = np.asarray(exp_df.reindex(cell_ids).values, dtype=float)
    genes = list(exp_df.columns)

    # lineage weights: tree distance between each pair of cells (via their clones)
    tree = to_lineage_tree(tumor)
    present = list(dict.fromkeys(clones.tolist()))          # unique clones, stable order
    gi = {g: i for i, g in enumerate(present)}
    Dg = tree.distance_matrix(present)                      # (n_clones, n_clones)
    cg = np.array([gi[g] for g in clones])
    Dlin = Dg[np.ix_(cg, cg)]
    Wlin = np.exp(-Dlin / max(float(lineage_lengthscale), 1e-9))
    np.fill_diagonal(Wlin, 0.0)

    # spatial weights: jittered coordinates
    crd = cell_data["cell_crd"].reindex(cell_ids).values.astype(float)
    crd = crd + rng.normal(0.0, coord_jitter, crd.shape)
    d2 = ((crd[:, None, :] - crd[None, :, :]) ** 2).sum(-1)
    L = max(float(spatial_lengthscale), 1e-9)
    Wsp = np.exp(-d2 / (2.0 * L * L))
    np.fill_diagonal(Wsp, 0.0)

    I_lin = _moran_all(Wlin, exp)
    I_sp = _moran_all(Wsp, exp)

    # lineage-space correlation (a coarse confound driver): pairwise tree vs spatial distance
    iu = np.triu_indices(len(cell_ids), 1)
    dsp = np.sqrt(d2[iu])
    ls = Dlin[iu]
    if ls.std() > 0 and dsp.std() > 0:
        ls_corr = float(np.corrcoef(ls, dsp)[0, 1])
    else:
        ls_corr = float("nan")

    # autocorrelation of the ground-truth extrinsic FIELDS themselves (hypoxia/cci level) — the
    # clean confound driver: a purely spatial field acquiring lineage autocorrelation under
    # territories, measured without any expression/CNA noise.
    field_autocorr = {}
    me = cell_data.get("cell_microenv")
    if me is not None:
        mv = me.reindex(cell_ids)
        F = np.asarray(mv.values, dtype=float)
        FL = _moran_all(Wlin, F)
        FS = _moran_all(Wsp, F)
        for j, name in enumerate(mv.columns):
            field_autocorr[name] = {"lineage": float(FL[j]), "spatial": float(FS[j])}

    return LineageSpatialDecomposition(genes, I_lin, I_sp, cell_ids, tree, ls_corr,
                                       field_autocorr=field_autocorr)
