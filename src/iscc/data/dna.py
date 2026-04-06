from .assay import Assay
import pandas as pd
import numpy as np
import os

class DNA(Assay):
    def __init__(self, data_mode='counts', fpr=0.1, fnr=0.1, n_reads=1000, target_genes='all', **assay_kwargs):
        super(DNA, self).__init__(**assay_kwargs)
        self.data_mode = data_mode # counts or reads
        self.fpr = fpr
        self.fnr = fnr
        self.n_reads = n_reads # total reads 
        self.target_genes = target_genes # 'all', fraction or list

class bulkDNA(DNA):
    def __init__(self, **dna_kwargs):
        super(bulkDNA, self).__init__(**dna_kwargs)

    def library_preparation(self, sample):
        pass

    def sequencing(self):
        pass

    def run(self, cell_data, **kwargs):
        cell_snvs = cell_data['cell_snv']
        
        target_cells = cell_snvs.index
        n_cells = len(target_cells)

        all_genes = cell_snvs.columns
        if self.target_genes == 'all':
            target_genes = all_genes
        elif isinstance(self.target_genes, float):
            target_genes = np.random.choice(all_genes, size=int(self.target_genes * len((all_genes))), replace=False)
        elif isinstance(self.target_genes, list):
            target_genes = self.target_genes
        n_genes = len(target_genes)

        target_cell_snvs = cell_snvs.loc[target_cells,target_genes]
        if self.data_mode == 'counts':
            self.coverage = np.random.multinomial(self.n_reads, pvals=[1/len(target_genes)]*len(target_genes)) # distribute across genes
            self.cell_coverages = np.vstack([np.random.multinomial(self.coverage[g], pvals=[1/n_cells]*n_cells) for g in range(len(self.coverage))]).T # distribute across cells
            self.cell_alt_counts = np.zeros((n_cells, n_genes))
            self.cell_alt_counts[target_cell_snvs == 1] = self.cell_coverages[target_cell_snvs == 1] - np.random.binomial(self.cell_coverages[target_cell_snvs == 1], self.fnr)
            self.cell_alt_counts[target_cell_snvs == 0] = np.random.binomial(self.cell_coverages[target_cell_snvs == 0], self.fpr)
            self.observed_data = pd.DataFrame(dict(coverage=self.coverage.astype(int), alt_counts=np.sum(self.cell_alt_counts, axis=0).astype(int)), index=target_genes)
        # Otherwise, generate reads
        elif self.data_mode == 'reads':
            raise NotImplementedError("DNA data generation currently only supports binary or counts matrices. Please set data_mode to 'binary' or 'counts'.")

    def write(self, out_path, **kwargs):
        if self.data_mode == 'counts':
            # CSV or VCF? Or both...
            self.observed_data.to_csv(os.path.join(out_path, 'counts.csv'))
        elif self.data_mode == 'reads':
            # Output FASTQ file
            raise NotImplementedError("DNA data generation currently only supports binary or counts matrices. Please set data_mode to 'binary' or 'counts'.")

class cfDNA(DNA):
    def __init__(self):
        # Take a blood sample and capture cell-free DNA fragments 
        pass

class scDNA(DNA):
    def __init__(self, n_cells=100, **dna_kwargs):
        super(scDNA, self).__init__(**dna_kwargs)
        self.n_cells = n_cells # target number of cells


    def run(self, cell_data, **kwargs):
        cell_snvs = cell_data['cell_snv']
        if len(cell_snvs.index) < self.n_cells:
            self.n_cells = len(cell_snvs.index)
        target_cells = np.random.choice(cell_snvs.index, size=self.n_cells, replace=False)

        all_genes = cell_snvs.columns
        if self.target_genes == 'all':
            target_genes = all_genes
        elif isinstance(self.target_genes, float):
            target_genes = np.random.choice(all_genes, size=int(self.target_genes * len((all_genes))), replace=False)
        elif isinstance(self.target_genes, list):
            target_genes = self.target_genes

        target_cell_snvs = cell_snvs.loc[target_cells,target_genes].astype(int)
        if self.data_mode == 'binary':
            self.observed_snvs = np.zeros(target_cell_snvs.shape)
            self.observed_snvs[target_cell_snvs == 1] = 1-np.random.binomial(target_cell_snvs.values[target_cell_snvs == 1], self.fnr)
            self.observed_snvs[target_cell_snvs == 0] = np.random.binomial(1-target_cell_snvs.values[target_cell_snvs == 0], self.fpr)
            self.observed_snvs = pd.DataFrame(self.observed_snvs.astype(int), index=target_cells, columns=target_genes)
        elif self.data_mode == 'counts':
            self.coverage = np.random.multinomial(self.n_reads, [1/len(target_genes)]*len(target_genes), size=len(target_cells))
            self.alt_counts = np.zeros(target_cell_snvs.shape)
            self.alt_counts[target_cell_snvs == 1] = self.coverage[target_cell_snvs == 1] - np.random.binomial(self.coverage[target_cell_snvs == 1], self.fnr)
            self.alt_counts[target_cell_snvs == 0] = np.random.binomial(self.coverage[target_cell_snvs == 0], self.fpr)
            self.coverage = pd.DataFrame(self.coverage.astype(int), index=target_cells, columns=target_genes)
            self.alt_counts = pd.DataFrame(self.alt_counts.astype(int), index=target_cells, columns=target_genes)
        # Otherwise, generate reads
        elif self.data_mode == 'reads':
            raise NotImplementedError("DNA data generation currently only supports binary or counts matrices. Please set data_mode to 'binary' or 'counts'.")

    def write(self, out_path, **kwargs):
        if self.data_mode == 'binary':
            self.observed_snvs.to_csv(os.path.join(out_path, 'observed_snvs.csv'))
        elif self.data_mode == 'counts':
            self.coverage.to_csv(os.path.join(out_path, 'coverage.csv'))
            self.alt_counts.to_csv(os.path.join(out_path, 'alt_counts.csv'))
        elif self.data_mode == 'reads':
            # Output FASTQ file
            raise NotImplementedError("DNA data generation currently only supports binary or counts matrices. Please set data_mode to 'binary' or 'counts'.")