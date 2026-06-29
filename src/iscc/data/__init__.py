from .dna import bulkDNA, scDNA
from .rna import scRNA, run_scrna_batches, concat_batches
from .visium import Visium
from .batch import Batch, BatchHyperParams, COUNT_MODELS

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
