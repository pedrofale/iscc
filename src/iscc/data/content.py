"""The relative RNA content of each cell, for library sizes that follow what a cell or a spot holds.

The RNA content of a normal cell type is its mean counts in the Human Breast Cell Atlas relative to the
epithelium (``human/reference/hbca_types.tsv``: fibroblasts 0.62, T and NK cells 0.28, myeloid cells
0.74, B and plasma cells 0.72, endothelial cells 0.62, perivascular cells 0.42). The content of a cancer
cell is that of the epithelium, the cell of origin, doubled when the genome of the cell is doubled; the
host tissue of the metastatic site has the content of the stroma. ``scRNA`` and ``Visium`` scale the
library of a cell or a spot by
this content raised to ``content_exponent`` (1 by default, proportional capture; 0 leaves it out).
"""
import os

import pandas as pd

_TABLE = None


def rna_content_table():
    """``{genotype id of a normal cell type: relative RNA content}``, from the atlas table.

    The genotype id of a normal cell is the name of the cell type (``epithelial``, ``stromal`` for
    fibroblasts, ``endothelial``, ``pvl``, ``immune`` for T and NK cells, ``myeloid``, ``bplasma``,
    ``host``); any other id is a cancer clone.

    Returns
    -------
    dict
        The RNA content of each normal type relative to the epithelium (1).
    """
    global _TABLE
    if _TABLE is None:
        path = os.path.join(os.path.dirname(__file__), "human", "reference", "hbca_types.tsv")
        df = pd.read_csv(path, sep="\t")
        table = dict(zip(df["type"].astype(str), df["rna_content"].astype(float)))
        table.setdefault("host", table["stromal"])
        _TABLE = table
    return dict(_TABLE)


def cell_rna_content(cell_data, cells=None):
    """The relative RNA content of each cell of ``cell_data``.

    Parameters
    ----------
    cell_data : dict
        A tumour's or a sample's per-cell tables; reads ``cell_type`` (the genotype id of each cell)
        and, if present, ``cell_wgd`` (whether the genome of the cell is doubled).
    cells : array-like, optional
        The cells to return, in this order. Default None returns every cell of ``cell_type``.

    Returns
    -------
    pandas.Series
        The RNA content of each cell relative to the normal epithelium (1); 1 for every cell when
        ``cell_data`` has no ``cell_type`` table (expression without types: nothing to scale by).
    """
    if "cell_type" not in cell_data:
        index = pd.Index(cells) if cells is not None else cell_data["cell_exp"].index
        return pd.Series(1.0, index=index)
    ids = cell_data["cell_type"]["cell_id"].astype(str)
    if cells is not None:
        ids = ids.reindex(cells)
    table = rna_content_table()
    content = ids.map(table)
    cancer = content.isna() & ids.notna()
    content = content.fillna(table["epithelial"]).astype(float)
    wgd = cell_data.get("cell_wgd")
    if wgd is not None and cancer.any():
        doubled = wgd.iloc[:, 0].reindex(content.index).fillna(False).astype(bool)
        content[cancer & doubled] *= 2.0
    return content
