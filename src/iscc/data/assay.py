"""Base assay (DESIGN_features.md §B / §E).

An `Assay` is `protocol + batch hyper-parameters + seed`. Its job is to turn the
biological ground truth (a sampled tumor's `cell_data`) into observed molecular data by
passing it through a `Batch` realization (see `batch.py`). The base holds the protocol
label and seed; modality subclasses (`scRNA`, `bulkDNA`, `scDNA`, `Visium`) own the
modality-specific hyper-parameters and assemble their `Batch`.
"""


class Assay(object):
    def __init__(self, seed=42, protocol=None):
        self.seed = seed
        self.protocol = protocol

    def run(self, cell_data, **kwargs):
        pass

    def write(self, output_path):
        pass
