"""iscc.integrations — the interchange / export seam.

Exports iscc's per-cell substrate (lineage tree, spatial coordinates, clone identity, expression)
in the standard formats external tools consume, and provides the lineage-vs-spatial expression
decomposition the PEtracer validation is built on:

  * ``to_lineage_tree(tumor)`` / ``to_newick(tumor)`` — the clone tree from ``genotypes_parents``;
  * ``to_anndata(cell_data)`` — cells x genes AnnData with ``obsm["spatial"]`` + clone/type ``obs``,
    and ``from_anndata`` to read one back as per-cell tables;
  * ``decompose_lineage_spatial(tumor)`` — per-gene lineage (tree) vs spatial autocorrelation, the
    intrinsic-vs-extrinsic decomposition (Hotspot/PhyloVision-style) with iscc ground truth;
  * ``to_mhn_matrix`` / ``to_treemhn_trees`` + ``score_edges``/``score_order`` — the cohort
    progression-model seam (MHN/TreeMHN/CBN/REVOLVER), scored against the epistasis network iscc
    planted.

Additive and low-risk: nothing here changes the engine; each optional external dependency
(``anndata``) is imported lazily so the base import stays dependency-free.
"""
from .lineage import LineageTree, to_lineage_tree, to_newick
from .clones import clones_from_clades, clone_summary
from .anndata import to_anndata, from_anndata, biological_types
from .petracer import decompose_lineage_spatial, LineageSpatialDecomposition
from .progression import (
    to_mhn_matrix, to_cell_fraction_matrix, to_mutation_tree, to_treemhn_trees, to_cbn_poset,
    clone_events, event_cell_fractions, patient_event_vector, cooccurrence_scores, top_edges,
    score_edges, score_order, score_exclusivity,
)
from .multiregion import (
    true_origin_counts, region_bulk_profiles, oracle_clone_profiles,
    count_spurious_parallel, multiregion_phylogeny,
    neighbor_joining, hamming_nj_tree, fitch_length, robinson_foulds,
    clone_lineage_tree, ordering_reversal_rate,
)
from .cci import cci_database, write_cci_database, referenced_genes, clone_correlation

__all__ = [
    "LineageTree", "to_lineage_tree", "to_newick",
    # the shared clone definition: lineage clades with enough sampled cells
    "clones_from_clades", "clone_summary",
    # the whole tumour as one AnnData, and the way back
    "to_anndata", "from_anndata", "biological_types",
    "decompose_lineage_spatial", "LineageSpatialDecomposition",
    # cohort progression models (MHN/TreeMHN/CBN/REVOLVER) vs the planted epistasis network
    "to_mhn_matrix", "to_cell_fraction_matrix", "to_mutation_tree", "to_treemhn_trees",
    "to_cbn_poset", "clone_events", "event_cell_fractions", "patient_event_vector",
    "cooccurrence_scores", "top_edges",
    "score_edges", "score_order", "score_exclusivity",
    # multi-region "sample trees are not phylogenies" benchmark
    "true_origin_counts", "region_bulk_profiles", "oracle_clone_profiles",
    "count_spurious_parallel", "multiregion_phylogeny",
    "neighbor_joining", "hamming_nj_tree", "fitch_length", "robinson_foulds",
    "clone_lineage_tree", "ordering_reversal_rate",
    # W0: iscc's own ligand-receptor database (CellChat/CellPhoneDB), and the W4 clone-correlation axis
    "cci_database", "write_cci_database", "referenced_genes", "clone_correlation",
]
