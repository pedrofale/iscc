"""The human genome layer: real genes, germline SNPs and tool-ready outputs from a coarse growth genome.

Growth keeps a coarse genome; this layer lifts it to the human genome when cells are measured::

    from iscc.data import scRNA, Visium
    from iscc.data.human import arm_genome, HumanGenomeLayer, HUMAN_VISIUM_KAPPA, write_visium

    cfg = arm_genome(total_genes=6000)                  # one growth segment per chromosome arm
    tumor = GenotypeTumor(genome_params=cfg["genome_params"],
                          selection_params={**selection_params, **cfg["selection_params"]}, ...)
    tumor.grow(...)
    layer = HumanGenomeLayer(tumor)                     # gene map + germline SNPs
    expr = layer.expression(tumor)                      # atlas baselines + Hallmark programmes
    vz = Visium(seed=1, kappa=HUMAN_VISIUM_KAPPA).run(section, gene_lift=expr)
    write_visium("out/visium", vz, layer, allele=layer.visium_allele_counts(vz, tumor), tumor=tumor)
"""
from .genome import arm_genome
from .expression import HumanExpression
from .layer import AlleleCounts, GeneLift, HumanGenomeLayer
from .ready import check_tool_ready
from .reference import (HumanReference, fetch_snps, load_atlas_baselines, load_hallmark_programs,
                        load_reference, prepare_snps, read_snps)
from .writers import HUMAN_VISIUM_KAPPA, barcodes, write_scrna, write_visium

__all__ = [
    "arm_genome", "HumanGenomeLayer", "GeneLift", "AlleleCounts", "HumanReference", "load_reference",
    "fetch_snps", "prepare_snps", "read_snps", "write_visium", "write_scrna", "check_tool_ready",
    "barcodes", "HUMAN_VISIUM_KAPPA", "HumanExpression", "load_atlas_baselines",
    "load_hallmark_programs",
]
