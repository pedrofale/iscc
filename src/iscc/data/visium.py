from .assay import Assay

import numpy as np
import pandas as pd
import os

class Visium(Assay):
    def __init__(self, n_reads=1000, spot_radius=2.5, n_spots_row=25, n_spots_col=25, **assay_kwargs):
        super(Visium, self).__init__(**assay_kwargs)
        self.n_reads = n_reads # total reads per spot
        self.spot_radius = spot_radius
        self.n_spots_row = n_spots_row
        self.n_spots_col = n_spots_col
        
    def run(self, cell_data, grid_side: int):
        cell_states = cell_data['cell_exp']
        cell_crd = cell_data['cell_crd'].astype(int)
        n_genes = cell_states.shape[1]
        radius_squared = self.spot_radius ** 2
    
        rr = np.linspace(self.spot_radius, grid_side - self.spot_radius, self.n_spots_row)
    
        cc = np.linspace(self.spot_radius, grid_side - self.spot_radius, self.n_spots_col)
    
        rr, cc = np.meshgrid(rr, cc)
        rr = rr.flatten()
        cc = cc.flatten()
    
        self.spot_umi = np.zeros((rr.shape[0], n_genes)) # for each spot
        self.spot_cell_counts = np.zeros(rr.shape[0])
        self.spot_crd = np.hstack((rr[:, np.newaxis], cc[:, np.newaxis]))
        cell_id_list = []
        cell_id_str = []
        for spot, (r, c) in enumerate(zip(rr, cc)):
            deltas = (cell_crd['row'] - r)**2 + (cell_crd['col'] - c)**2
            in_spot = cell_crd.iloc[np.where(deltas < radius_squared)].index
    
            in_spot = np.array(in_spot)
    
            cell_id_list.append(in_spot.tolist())
            if len(in_spot) > 0:
                cell_id_str.append(','.join(in_spot.tolist()))
            else:
                cell_id_str.append('')
    
            if len(in_spot) > 0:
                joint_expr = cell_states.loc[in_spot].values.mean(axis=0) # average transcriptional activities of all cells assigned to this spot
                gene_prob = joint_expr / joint_expr.sum()
                self.spot_umi[spot, :] = np.random.multinomial(self.n_reads, gene_prob)
            self.spot_cell_counts[spot] = len(in_spot)
            
    
        spot_names = [f'S{i}' for i in range(rr.shape[0])]
        self.spot_umi = pd.DataFrame(self.spot_umi, index=spot_names, columns=cell_data['cell_exp'].columns).astype(int)
        self.spot_crd = pd.DataFrame(self.spot_crd, index=spot_names, columns=['row', 'col']).astype(float)
        self.spot_cell_counts = pd.DataFrame(self.spot_cell_counts, index=spot_names, columns=['n_cells']).astype(int)
        self.spot_cell_ids = pd.DataFrame(cell_id_str, index=spot_names, columns=['cell_ids'])

    def write(self, out_path):
        self.spot_umi.to_csv(os.path.join(out_path, 'spot_umi.csv'))
        self.spot_crd.to_csv(os.path.join(out_path, 'spot_crd.csv'))
        self.spot_cell_counts.to_csv(os.path.join(out_path, 'spot_cell_counts.csv'))
        self.spot_cell_ids.to_csv(os.path.join(out_path, 'spot_cell_ids.csv'))
