"""AnnData export of iscc cell data (DESIGN_features §I).

Packs the engine's per-cell matrices into a standard ``AnnData`` so external tools (scanpy, squidpy,
Hotspot, …) can consume iscc's tumour directly:

  * ``X``            = per-cell expression (``cell_exp``);
  * ``obsm["spatial"]`` = per-cell (row, col) coordinates (``cell_crd``);
  * ``obs``          = clone (genotype id) / cell type / coords / microenvironment ground truth;
  * ``layers``       = the other per-cell matrices (``cell_snv``, ``cell_cnv``) when present.

``anndata`` is imported lazily so importing :mod:`iscc.integrations` never requires it.
"""
import numpy as np
import pandas as pd


def to_anndata(cell_data, X="cell_exp", layers=("cell_snv", "cell_cnv"), obs_extra=True):
    """Build an ``AnnData`` from an iscc ``cell_data`` dict.

    Parameters
    ----------
    cell_data : dict
        The engine's materialised cell data (``tumor.cell_data``): expects ``cell_exp`` and, when
        available, ``cell_crd`` / ``cell_type`` / ``cell_deme`` / ``cell_microenv``.
    X : str
        Which per-cell matrix becomes ``.X`` (default ``"cell_exp"``).
    layers : tuple[str]
        Additional per-cell gene matrices to attach as ``.layers`` (skipped if absent).
    obs_extra : bool
        Attach the microenvironment ground truth (``cell_microenv``: hypoxia/cci levels) to ``.obs``
        when present.
    """
    import anndata as ad

    exp = cell_data[X]
    genes = list(exp.columns)
    obs = pd.DataFrame(index=exp.index.astype(str))

    ctype = cell_data.get("cell_type")
    if ctype is not None:
        obs["clone"] = ctype.reindex(exp.index).iloc[:, 0].astype(str).values
    deme = cell_data.get("cell_deme")
    if deme is not None:
        obs["deme"] = deme.reindex(exp.index).iloc[:, 0].astype(int).values
    crd = cell_data.get("cell_crd")
    if crd is not None:
        crd = crd.reindex(exp.index)
        obs["row"] = crd["row"].values
        obs["col"] = crd["col"].values
    if obs_extra and cell_data.get("cell_microenv") is not None:
        me = cell_data["cell_microenv"].reindex(exp.index)
        for c in me.columns:
            obs[c] = me[c].values

    adata = ad.AnnData(
        X=np.asarray(exp.values, dtype=np.float32),
        obs=obs,
        var=pd.DataFrame(index=pd.Index(genes, name="gene")),
    )
    if crd is not None:
        adata.obsm["spatial"] = crd[["row", "col"]].values.astype(float)
    for name in layers:
        mat = cell_data.get(name)
        if mat is not None and list(mat.columns) == genes:
            adata.layers[name] = np.asarray(mat.reindex(exp.index).values, dtype=np.float32)
    adata.uns["source"] = "iscc"
    return adata
