from .dna import bulkDNA, scDNA
from .rna import scRNA
from .visium import Visium

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
