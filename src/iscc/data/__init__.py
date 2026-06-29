from .dna import bulkDNA, scDNA, cfDNA, run_dna_batches, genome_features, DNA_BREADTH_PRESETS
from .rna import scRNA, run_scrna_batches, concat_batches
from .visium import Visium
from .batch import (
    Batch, BatchHyperParams, COUNT_MODELS,
    DNABatch, DNABatchHyperParams, DNA_DEPTH_MODELS,
)
from .estimate import estimate, RNAEstimate

ASSAY_NAMES = {
    'bdna': 'Bulk DNA',
    'scdna': 'scDNA',
    'scrna': 'scRNA',
    'visium': 'Visium',
}

ASSAYS = {
    'bdna': bulkDNA,
    'scdna': scDNA,
    'scrna': scRNA,
    'visium': Visium,
}
