import numpy as np

class Biopsy(object):
    def __init__(self, storage, ):
        self.storage = 'fresh frozen'
        # https://www.biochain.com/blog/ffpe-vs-frozen-tissue-samples/

    def run(self, tumor):
        if tumor.type != 'mixed':
            cell_coords = self.take_spatial_sample(tumor, )

    def get_regions(self, n_regions):
        # According to some schedule...
        pass
    
    def take_spatial_sample(self, tumor):
        pass

    def take_mixed_sample(self, tumor):
        pass

    def store_freshfrozen(self):
        self.dna.degrade()
        self.rna.degrade()

    def store_ffpe(self):
        self.dna.degrade()
        self.rna.degrade()
