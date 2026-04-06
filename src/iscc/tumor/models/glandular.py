from ..tumor import Tumor
from ..components.deme import Deme
from ..components.cell import EpithelialCell, StromalCell

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
from collections import Counter

def bresenham_circumference(x0, y0, radius):
    x = radius
    y = 0
    err = 0

    points = []

    while x >= y:
        points.append((x0 + x, y0 + y))
        points.append((x0 + y, y0 + x))
        points.append((x0 - y, y0 + x))
        points.append((x0 - x, y0 + y))
        points.append((x0 - x, y0 - y))
        points.append((x0 - y, y0 - x))
        points.append((x0 + y, y0 - x))
        points.append((x0 + x, y0 - y))

        y += 1
        err += 1 + 2*y
        if 2*(err-x) + 1 > 0:
            x -= 1
            err += 1 - 2*x

    points = pd.DataFrame(points).drop_duplicates().values
    points = [(p[0], p[1]) for p in points]
    return points

def get_inside(border):
    points = []
    for (x,y) in border:
        # Add points inside -- general for any shape
        # find another x with this y
        for b in border:
            if b[1] == y:
                if b[0]-x > 1: # if to the right
                    points.extend([(x_,y) for x_ in range(x+1,b[0])])
                if b[0]-x > 1: # if to the left
                    points.extend([(x_,y) for x_ in range(b[0],x-1)])
    points = pd.DataFrame(points).drop_duplicates().values
    points = [(p[0], p[1]) for p in points]
    return points

class GlandularTumor(Tumor):
    def __init__(self, n_structures=1, structure_radius=0, grid_size=10, **tumor_kwargs):
        super(GlandularTumor, self).__init__(**tumor_kwargs)
        self.type = 'glandular'

        if self.config is not None:
            self.spatial_params = self.config['spatial_params']
            grid_size = self.spatial_params['grid_size']
            n_structures = self.spatial_params['n_structures']
            structure_radius = self.spatial_params['structure_radius']
        
        self.grid_size = grid_size
        self.n_structures = n_structures
        self.structure_radius = structure_radius
        self.make_grid()

    def make_grid(self):
        # Initialize grid of empty demes
        self.positions = []
        self.grid = []
        self.deme_list = []
        for grid_row in range(self.grid_size):
            row = []
            for grid_col in range(self.grid_size):
                deme = Deme(
                    tumor=self,
                    row=grid_row,
                    col=grid_col,
                    **self.deme_params
                )
                self.deme_list.append(deme)
                row.append(deme)
            self.grid.append(row)

        # Initialize cell positions
        center = int(self.grid_size / 2)
        if self.structure_radius <= 0:
            # Put cancer cell in center deme    
            self.grid[center][center].add_cell(self.cancer_cell) 
            self.grid[center][center].deme_rate = self.cancer_cell.evolutionary_parameters['death_rate'] + self.cancer_cell.evolutionary_parameters['division_rate']
        else:
            self.make_structure(center)                

        self.deme_rates = []
        for i, deme in enumerate(self.deme_list):
            self.deme_rates.append(deme.deme_rate)
            deme.id = i
        self.deme_rates = np.array(self.deme_rates)            

    def make_structure(self, center):
        structure_borders = []
        structure_in_borders = []
        structure_circles = []
        for s_idx in range(self.n_structures):
            # Get border
            border = bresenham_circumference(center, center, self.structure_radius)
            structure_borders.append(border)

            # add healthy epithelial cells in border of shape
            for (row,col) in border:
                # Fill them up to carrying capacity
                i = 0
                while i < self.grid[row][col].carrying_capacity:
                    epithelial_cell = EpithelialCell(n_segments=self.genome_params['n_segments'],
                                                     **self.epithelial_cell_params)
                    self.grid[row][col].add_cell(epithelial_cell)
                    self.grid[row][col].deme_rate = epithelial_cell.evolutionary_parameters['death_rate']
                    i += 1

            # Get inside
            circle = get_inside(border)
            structure_circles.append(circle)

            in_border = bresenham_circumference(center, center, self.structure_radius-1)                        
            structure_in_borders.append(in_border)
            if s_idx == 0:
                # add a cancer cell inside the border 
                pos = np.random.choice(len(in_border))
                self.grid[in_border[pos][0]][in_border[pos][1]].add_cell(self.cancer_cell)
                self.grid[in_border[pos][0]][in_border[pos][1]].deme_rate = self.cancer_cell.evolutionary_parameters['death_rate'] + self.cancer_cell.evolutionary_parameters['division_rate']

        # add healthy stromal cells outside of border
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if (row,col) not in structure_circles and (row,col) not in structure_borders:
                    # Fill them up to carrying capacity
                    i = 0
                    if len(self.grid[row][col].cells) > 0:
                        continue
                    while i < self.grid[row][col].carrying_capacity:
                        stromal_cell = StromalCell(n_segments=self.genome_params['n_segments'],
                                                   **self.stromal_cell_params)
                        self.grid[row][col].add_cell(stromal_cell)
                        self.grid[row][col].deme_rate = stromal_cell.evolutionary_parameters['death_rate']
                        i += 1    

    def get_neighboring_demes(self, deme):
        grid_row = deme.row
        grid_col = deme.col

        possible_demes = []
        pos = []
        # Von Neumann neighborhood
        for tup in [(grid_row - 1, grid_col), (grid_row, grid_col + 1), (grid_row + 1, grid_col), (grid_row, grid_col - 1)]:
            if tup[0] > 0 and tup[0] < self.grid_size:
                if tup[1] > 0 and tup[1] < self.grid_size:
                    pos.append(tup)
                    possible_demes.append(self.grid[tup[0]][tup[1]])

        # Other structure
        # for in_border in self.structure_in_borders:
        #     for tup in in_border:
        #         possible_demes.append(self.grid[tup[0]][tup[1]])

        return possible_demes


    def write(self, output_path):
        super().write(output_path)

        # if tumor is spatial, write spatial info
        genotype_matrix = self.get_genotype_matrix()
        pd.DataFrame(genotype_matrix).to_csv(os.path.join(output_path, "grid.csv"))

        # Save genotype counts per deme in this step
        coords = []
        gcounts = []
        for deme in self.deme_list:
            coords.append(f'{deme.row},{deme.col}')
            gcounts.append(deme.genotypes_counts)
        df = pd.DataFrame(gcounts).fillna(0)
        df.index = coords
        df.to_csv(os.path.join(output_path, f"genotype_counts_demes.csv"))    

    def plot_grid(
        self,
        color=None,
        cmap="viridis",
        expand_demes=False,
        ax=None,
        figsize=(10, 10),
        dpi=100,
    ):
        if self.cell_data is None:
            self.make_cell_data()
        else:
            n_cells = sum(self.genotypes_counts.values())
            if self.cell_data['cell_type'].shape[0] != n_cells:
                self.make_cell_data()
        
        if color is None:
            color = ["cell_type"]
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        else:
            plt.sca(ax)

        for i, color_key in enumerate(color):
            # Create grid for this color_key
            if color_key == 'cell_type':
                # Make colormap according to muller plot
                type_cmap = self.get_cell_type_colors()
                cell_data = self.cell_data["cell_deme"].join(self.cell_data["cell_crd"])
                cell_data['val'] = self.cell_data[color_key]
                deme_data = cell_data.groupby(["deme_id"]).agg(pd.Series.mode)
                grid = np.zeros((self.grid_size, self.grid_size, 4)) + 1.
                grid[:,:,-1] = 1.
                for genotype in type_cmap.keys():
                    idx = np.where(deme_data["val"] == genotype)
                    row, col = deme_data.iloc[idx][["row", "col"]]
                    grid[row,col] = type_cmap[genotype]
                ax.imshow(grid)
            else:
                if "snv_" in color_key:
                    gene = color_key.split("_")[1]
                    cell_data = self.cell_data["cell_deme"].join(self.cell_data["cell_crd"])
                    cell_data['val'] = self.cell_data["cell_snv"][gene]
                    deme_data = cell_data.groupby(["deme_id"]).mean()
                elif "cnv_" in color_key:
                    gene = color_key.split("_")[1]
                    cell_data = self.cell_data["cell_deme"].join(self.cell_data["cell_crd"])
                    cell_data['val'] = self.cell_data["cell_cnv"][gene]
                    deme_data = cell_data.groupby(["deme_id"]).mean()
                elif "exp_" in color_key:
                    gene = color_key.split("_")[1]
                    cell_data = self.cell_data["cell_deme"].join(self.cell_data["cell_crd"])
                    cell_data['val'] = self.cell_data["cell_exp"][gene]
                    deme_data = cell_data.groupby(["deme_id"]).mean()
                for name in self.cell_data.keys():
                    if color_key in self.cell_data.columns:
                        cell_data = self.cell_data[name][color_key]
                        deme_data = cell_data.groupby(["deme_id"]).mean()
                        break
                grid = np.zeros((self.grid_size, self.grid_size), dtype=float)
                grid[deme_data["row"], deme_data["col"]] = deme_data["val"]
                ax.imshow(grid, cmap=cmap)

            ax.set_title(color_key)


        plt.axis("off")
        return ax
