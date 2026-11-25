from .dna import (
    DNA, bulkDNA, scDNA, run_dna_batches, genome_features, genome_bases, DNA_BREADTH_PRESETS,
    SCDNA_PLATFORM_PRESETS,
)
from .rna import scRNA, run_scrna_batches, concat_batches, atlas_scrna_batches
from .visium import Visium, morans_i
from .content import cell_rna_content, rna_content_table
from .imaging import scSpatial, IMAGING_PRESETS
from .batch import (
    Batch, RNABatchHyperParams, COUNT_MODELS,
    DNABatch, DNABatchHyperParams, DNA_DEPTH_MODELS,
    VisiumBatch, VisiumBatchHyperParams,
)
from .estimate import estimate_rna, RNAEstimate
from .estimate_dna import estimate_dna, estimate_dna_from_assay, DNAEstimate
from .estimate_visium import estimate_visium, estimate_visium_from_assay, VisiumEstimate

ASSAY_NAMES = {
    'bdna': 'Bulk DNA',
    'scdna': 'scDNA',
    'scrna': 'scRNA',
    'visium': 'Visium',
    'scspatial': 'Single-cell spatial',
}

ASSAYS = {
    'bdna': bulkDNA,
    'scdna': scDNA,
    'scrna': scRNA,
    'visium': Visium,
    'scspatial': scSpatial,
}
