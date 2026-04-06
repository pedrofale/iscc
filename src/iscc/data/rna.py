from .assay import Assay

import numpy as np
import pandas as pd
import os

def sample_umis(activities, coverage):
    return np.random.multinomial(coverage, activities/np.sum(activities))

class scRNA(Assay):
    def __init__(self, n_reads=1000, n_cells=100, **assay_kwargs):
        super(scRNA, self).__init__(**assay_kwargs)
        self.n_reads = int(n_reads) # total reads per cell
        self.n_cells = int(n_cells) # target number of cells

    def run(self, cell_data, **kwargs):
        cell_states = cell_data['cell_exp']
        if len(cell_states.index) < self.n_cells:
            self.n_cells = len(cell_states.index)
        target_cells = np.random.choice(cell_states.index, size=self.n_cells, replace=False)
        target_genes = cell_states.columns

        target_cell_states = cell_states.loc[target_cells, target_genes]
        umi_counts = np.vstack([sample_umis(target_cell_states.loc[c].values, self.n_reads) for c in target_cells])
        self.observed_counts = pd.DataFrame(umi_counts, index=target_cells, columns=target_genes)

    def write(self, out_path):
        self.observed_counts.to_csv(os.path.join(out_path, 'umis.csv'))