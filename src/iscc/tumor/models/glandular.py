from ..tumor import Tumor
from ..components.deme import Deme
from ..components.cell import EpithelialCell, StromalCell
from ...constants import normal_names

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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
    """Cell-level (one-object-per-cell) glandular tumour engine on a deme grid.

    The reference implementation of the iscc growth process: every cell is an explicit
    ``Cell`` object, so it is the most faithful model (and the ground truth the count
    engine is validated against), but it does not scale to large tumours — prefer
    [`GenotypeTumor`][iscc.tumor.GenotypeTumor] for anything big. Cells live in demes on a square grid; with
    ``structure_radius > 0`` the founder is seeded inside a ring-shaped gland and
    spreads outward, otherwise it starts as a micro-lesion in the centre deme.

    Accepts a single ``config`` YAML path or the individual parameter blocks (forwarded
    to the ``Tumor`` base). Every knob, with its default and valid range, is covered in
    the parameter documentation.

    Parameters
    ----------
    n_structures : int, optional
        Number of glandular structures to seed (default 1). Read from
        ``spatial_params.n_structures`` when a ``config`` is given.
    structure_radius : int, optional
        Radius of the glandular ring in demes (default 0 = no gland; the founder is
        seeded in the centre deme). Read from ``spatial_params.structure_radius`` under
        a ``config``.
    grid_size : int, optional
        Side length of the square deme grid (default 10). Read from
        ``spatial_params.grid_size`` under a ``config``.
    **tumor_kwargs
        Forwarded to the [`Tumor`][iscc.tumor.Tumor] base. See [`Tumor`][iscc.tumor.Tumor]
        for the shared ``genome_params`` / ``selection_params`` / ``deme_params`` /
        cell-type blocks (as well as ``config``, ``seed`` and ``layout_seed``).
    """

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
            # Seed an established micro-lesion in the centre deme (see _seed_founders).
            self._seed_founders(self.grid[center][center])
        else:
            self.make_structure(center)

    def _seed_founders(self, deme):
        """Seed the founder cancer clone into a deme as ``initial_cancer_cells`` identical cells.

        Density-dependent crowding (DESIGN_crowding.md) makes a single founder prone to stochastic
        extinction (crowding death ramps up from occupancy 0), so — mirroring the count engine — we
        seed a small cluster, capped by the deme's carrying capacity. The extra founders are clones
        of ``self.cancer_cell`` (same genotype, no mutation)."""
        n = int(self.deme_params.get("initial_cancer_cells", 1))
        cap = deme.carrying_capacity
        if getattr(deme, "_crowding", cap is not None) and cap:
            n = min(n, int(cap))
        n = max(1, n)
        dr = (self.cancer_cell.evolutionary_parameters['death_rate']
              + self.cancer_cell.evolutionary_parameters['division_rate'])
        deme.add_cell(self.cancer_cell)
        deme.deme_rate = dr
        for _ in range(n - 1):
            clone = self.cancer_cell.divide()          # same genotype (no mutation on plain divide)
            deme.add_cell(clone, genotype_id=self.cancer_cell.genotype_id)
            deme.deme_rate += dr

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
                                                     segment_size=self.genome_params.get('segment_size', 1000),
                                                     **self.epithelial_cell_params)
                    self.grid[row][col].add_cell(epithelial_cell)
                    self.grid[row][col].deme_rate += epithelial_cell.evolutionary_parameters['death_rate']
                    i += 1

            # Get inside
            circle = get_inside(border)
            structure_circles.append(circle)

            in_border = bresenham_circumference(center, center, self.structure_radius-1)                        
            structure_in_borders.append(in_border)
            if s_idx == 0:
                # seed the founder micro-lesion inside the border
                pos = self.rng.choice(len(in_border))
                self._seed_founders(self.grid[in_border[pos][0]][in_border[pos][1]])

        # add healthy stromal cells outside of border.
        # structure_borders / structure_circles are lists of per-structure point lists;
        # flatten them into a single set of occupied (row, col) coordinates.
        occupied = set()
        for border in structure_borders:
            occupied.update(border)
        for circle in structure_circles:
            occupied.update(circle)
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if (row, col) not in occupied:
                    # Fill them up to carrying capacity
                    i = 0
                    if len(self.grid[row][col].cells) > 0:
                        continue
                    while i < self.grid[row][col].carrying_capacity:
                        stromal_cell = StromalCell(n_segments=self.genome_params['n_segments'],
                                                   segment_size=self.genome_params.get('segment_size', 1000),
                                                   **self.stromal_cell_params)
                        self.grid[row][col].add_cell(stromal_cell)
                        self.grid[row][col].deme_rate += stromal_cell.evolutionary_parameters['death_rate']
                        i += 1

    def get_neighboring_demes(self, deme):
        grid_row = deme.row
        grid_col = deme.col

        possible_demes = []
        pos = []
        # Von Neumann neighborhood
        for tup in [(grid_row - 1, grid_col), (grid_row, grid_col + 1), (grid_row + 1, grid_col), (grid_row, grid_col - 1)]:
            if 0 <= tup[0] < self.grid_size:
                if 0 <= tup[1] < self.grid_size:
                    pos.append(tup)
                    possible_demes.append(self.grid[tup[0]][tup[1]])

        return possible_demes


    def write(self, output_path):
        """Write the tumour to ``output_path``, adding the spatial layout to the base output.

        Extends the base ``Tumor.write`` with the per-deme spatial files: the
        most-frequent-genotype grid (``grid.csv``) and the per-deme genotype counts
        (``genotype_counts_demes.csv``).

        Parameters
        ----------
        output_path : str or pathlib.Path
            Directory to create and write into.
        """
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
        """Plot the deme grid, colouring each deme by one or more per-cell attributes.

        Rebuilds the per-cell data if stale, then draws one panel per entry in ``color``.
        ``"cell_type"`` shades each deme by its dominant genotype / cell type with a legend;
        an evolutionary-parameter name, or a ``snv_`` / ``cnv_`` / ``exp_`` + gene key, shades
        by the per-deme mean value with a colorbar.

        Parameters
        ----------
        color : list of str, optional
            Attribute keys to plot (default ``["cell_type"]``).
        cmap : str, optional
            Matplotlib colormap for the scalar (non ``cell_type``) panels (default ``"viridis"``).
        expand_demes : bool, optional
            Unused placeholder (default ``False``).
        ax : matplotlib.axes.Axes, optional
            Axes to draw into; a new figure is created when ``None``.
        figsize : tuple, optional
            Figure size when a new figure is created (default ``(10, 10)``).
        dpi : int, optional
            Figure resolution when a new figure is created (default 100).

        Returns
        -------
        matplotlib.axes.Axes
            The axes drawn into.
        """
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
                cell_data['val'] = self.cell_data['cell_type']['cell_id'].values
                # Take first mode to handle ties
                deme_data = cell_data.groupby(["deme_id"]).agg(lambda x: x.mode().iloc[0])
                grid = np.zeros((self.grid_size, self.grid_size, 4)) + 1.
                grid[:,:,-1] = 1.
                legend_patches = []
                for genotype, color in type_cmap.items():
                    mask = deme_data["val"] == genotype
                    rows = deme_data.loc[mask, "row"].astype(int).values
                    cols = deme_data.loc[mask, "col"].astype(int).values
                    if len(rows) > 0:
                        grid[rows, cols] = color
                        label = genotype if genotype in normal_names else f'cancer ({genotype[:6]}…)'
                        legend_patches.append(mpatches.Patch(color=color, label=label))
                ax.imshow(grid)
                ax.legend(handles=legend_patches, loc='upper right', fontsize=7, framealpha=0.8)
            else:
                base = self.cell_data["cell_deme"].join(self.cell_data["cell_crd"])
                if "snv_" in color_key:
                    gene = color_key.split("_", 1)[1]
                    base['val'] = self.cell_data["cell_snv"][gene]
                elif "cnv_" in color_key:
                    gene = color_key.split("_", 1)[1]
                    base['val'] = self.cell_data["cell_cnv"][gene]
                elif "exp_" in color_key:
                    gene = color_key.split("_", 1)[1]
                    base['val'] = self.cell_data["cell_exp"][gene]
                elif color_key in self.cell_data["cell_evo"].columns:
                    base['val'] = self.cell_data["cell_evo"][color_key].values
                elif color_key in self.cell_data["cell_exp"].columns:
                    base['val'] = self.cell_data["cell_exp"][color_key].values
                elif color_key in self.cell_data["cell_snv"].columns:
                    base['val'] = self.cell_data["cell_snv"][color_key].values
                else:
                    continue
                deme_data = base.groupby(["deme_id"]).mean(numeric_only=True)
                grid = np.zeros((self.grid_size, self.grid_size), dtype=float)
                grid[deme_data["row"].astype(int), deme_data["col"].astype(int)] = deme_data["val"]
                im = ax.imshow(grid, cmap=cmap)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            ax.set_title(color_key)


        plt.axis("off")
        return ax
