"""iscc.integrations — interchange/export seam (DESIGN_features §I).

Exports iscc's per-cell substrate (lineage tree, spatial coordinates, clone identity, expression)
in the standard formats external tools consume, and provides the lineage-vs-spatial expression
decomposition the PEtracer validation is built on:

  * ``to_lineage_tree(tumor)`` / ``to_newick(tumor)`` — the clone tree from ``genotypes_parents``;
  * ``to_anndata(cell_data)`` — cells x genes AnnData with ``obsm["spatial"]`` + clone/type ``obs``;
  * ``decompose_lineage_spatial(tumor)`` — per-gene lineage (tree) vs spatial autocorrelation, the
    intrinsic-vs-extrinsic decomposition (Hotspot/PhyloVision-style) with iscc ground truth.

Additive and low-risk: nothing here changes the engine; each optional external dependency
(``anndata``) is imported lazily so the base import stays dependency-free.
"""
from .lineage import LineageTree, to_lineage_tree, to_newick
from .anndata import to_anndata
from .petracer import decompose_lineage_spatial, LineageSpatialDecomposition

__all__ = [
    "LineageTree", "to_lineage_tree", "to_newick",
    "to_anndata",
    "decompose_lineage_spatial", "LineageSpatialDecomposition",
]
