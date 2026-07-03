from .dna import (
    bulkDNA, scDNA, cfDNA, run_dna_batches, genome_features, genome_bases, DNA_BREADTH_PRESETS,
)
from .rna import scRNA, run_scrna_batches, concat_batches
from .visium import Visium, morans_i
from .imaging import scSpatial, IMAGING_PRESETS
from .batch import (
    Batch, BatchHyperParams, COUNT_MODELS,
    DNABatch, DNABatchHyperParams, DNA_DEPTH_MODELS,
    VisiumBatch, VisiumBatchHyperParams,
)
from .estimate import estimate, RNAEstimate
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
