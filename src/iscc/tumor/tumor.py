from .components.selection import Selection
from .components.cell import CancerCell
from ..constants import *

import os
from pathlib import Path
import yaml

import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy

import matplotlib
import matplotlib.pyplot as plt
import pymuller

from collections import Counter

class Tumor(object):
    def __init__(self, config=None, genome_params=dict(), cancer_cell_params=dict(), epithelial_cell_params=dict(), stromal_cell_params=dict(), immune_cell_params=dict(), deme_params=dict(), selection_params=dict(), seed=42):
        self.config = None
        # Seeded generator for everything set up at construction time (driver layout,
        # baseline expression, spatial seeding) so that config + seed -> identical tumor.
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.genome_params = genome_params
        if config is not None:
            with open(config) as f:
                self.config = yaml.safe_load(f)

            selection_params = self.config['selection_params']
            deme_params = self.config['deme_params']
            self.genome_params = self.config['genome_params']
            self.cell_params = self.config['cell_params']

            cancer_cell_params = self.cell_params['cancer']
            epithelial_cell_params = self.cell_params['epithelial']
            stromal_cell_params = self.cell_params['stromal']
            immune_cell_params = self.cell_params['immune']

        self.selection = Selection(
            n_segments=self.genome_params['n_segments'],
            segment_size=self.genome_params.get('segment_size', 1000),
            rng=self.rng,
            **selection_params,
        )
        self.n_genes = self.selection.n_genes

        self.cancer_cell = CancerCell(
            n_segments = self.genome_params['n_segments'],
            segment_size = self.genome_params['segment_size'],
            seed=seed,
            n_onc=len(self.selection.get_oncogenes()),
            n_tsg=len(self.selection.get_tsgs()),
            n_disp=len(self.selection.get_dispersal_genes()),
            n_ir=len(self.selection.get_immune_resistant()),
            n_tr=len(self.selection.get_treatment_resistant()),
            **cancer_cell_params,
        )
        
        self.epithelial_cell_params = epithelial_cell_params
        self.stromal_cell_params = stromal_cell_params
        self.immune_cell_params = immune_cell_params
        self.deme_params = deme_params
        self.celltypes = ['cancer', 'epithelial', 'stromal', 'immune']
        self.make_celltype_exps()

        self.deme_list = []
        self.deme_rates = np.array([])

        self.genotypes_parents = dict()
        self.genotypes_counts = Counter()
        self.cell_data = None
        self.traces = []
        self.step = 0


    @classmethod
    def load(cls, yaml_):
        obj = cls()

        return obj

    def make_celltype_exps(self):
        self.celltype_exps = dict()
        for celltype in self.celltypes:
            exp = self.rng.beta(.1, 1., size=self.n_genes)
            # get_tsgs()/get_oncogenes() already return the gene indices; index with
            # them directly (np.where(idx_array) would return 0..k-1, the wrong genes).
            exp[self.selection.get_tsgs()] = 0.8
            exp[self.selection.get_oncogenes()] = 0.01
            self.celltype_exps[celltype] = exp

    def get_genotype_frequencies(self, normalize=True):
        # Get unique genotypes and their frequencies
        genotypes = list(self.genotypes_counts.keys())
        snvs = []
        counts = []
        cancer_genotypes = []
        for genotype in genotypes:
            if genotype not in normal_names:
                cancer_genotypes.append(genotype)
                counts.append(self.genotypes_counts[genotype])
        freqs = np.array(counts)
        if normalize:
            freqs = freqs / np.sum(freqs)
        return snvs, freqs

    def get_deme_genotype_frequencies(self, normalize=True):
        # Get unique genotypes and their frequencies per deme
        # Create grid containing genotype freqs at each deme
        grid = []
        for grid_row in range(self.grid_size):
            row = []
            for grid_col in range(self.grid_size):
                if len(self.grid[grid_row][grid_col].cells) > 0:
                    row.append(self.grid[grid_row][grid_col].get_genotype_frequencies(normalize=normalize))
                else:
                    row.append("")
            grid.append(row)
        return grid

    def get_genotype_matrix(self):
        # Create grid containing most frequent genotype at each deme
        grid = []
        for grid_row in range(self.grid_size):
            row = []
            for grid_col in range(self.grid_size):
                if len(self.grid[grid_row][grid_col].cells) > 0:
                    row.append(self.grid[grid_row][grid_col].get_most_frequent_genotype())
                else:
                    row.append("")
            grid.append(row)
        return grid

    def update_genotype_parents(self):
        # Full resync from demes. Not needed during normal growth (kept incrementally
        # via register_parent); retained as a utility / consistency check.
        self.genotypes_parents = dict()
        for deme in self.deme_list:
            self.genotypes_parents.update(deme.genotypes_parents)

    def update_genotype_counts(self):
        # Full O(demes x genotypes) resync. Not called per step anymore; the tumor-level
        # counts are maintained incrementally by register_birth/register_death.
        self.genotypes_counts = Counter()
        for deme in self.deme_list:
            self.genotypes_counts = self.genotypes_counts + deme.genotypes_counts

    # --- incremental tumor-level bookkeeping (called from Deme on each event) -------
    def register_birth(self, genotype_id):
        self.genotypes_counts[genotype_id] += 1

    def register_death(self, genotype_id):
        self.genotypes_counts[genotype_id] -= 1
        if self.genotypes_counts[genotype_id] <= 0:
            del self.genotypes_counts[genotype_id]

    def register_parent(self, child_id, parent_id):
        self.genotypes_parents[child_id] = parent_id

    def is_extinct(self):
        return self.deme_rates.sum() == 0

    def update(self, treat=False, treatment=None, rng=None, batch_size=1):
        if self.is_extinct():
            return
        # Choose a few demes to update, proportionally to their rates. Sample indices
        # rather than the Deme objects directly: rng.choice over a list of objects
        # rebuilds an object ndarray every call (the dominant cost at scale); sampling
        # an int index draws the identical random numbers ~60x faster.
        p = self.deme_rates / self.deme_rates.sum()
        idx = rng.choice(len(self.deme_list), p=p, size=min(batch_size, len(self.deme_list)), replace=False)
        demes = [self.deme_list[i] for i in idx]
        for deme in demes:
            rng = rng.spawn(1)[0]
            deme.update(treat=treat, treatment=treatment, rng=rng, batch_size=batch_size)
        # Tumor-level counts/parents are kept up to date incrementally by the demes
        # (register_birth/death/parent), so no full rebuild is needed here.

    def grow(self, n_steps=10, seed=42, treatment=None, batch_size=1, **kwargs):
        new_traces = [dict(genotypes_counts=deepcopy(self.genotypes_counts))]

        for local_step in tqdm(range(n_steps - 1)):
            rng = np.random.default_rng(seed + self.step + local_step)
            run_treatment = False
            if treatment is not None:
                if treatment.get_dosage(self.step + local_step, self.get_tumor_size()) > 0:
                    run_treatment = True
            self.update(treat=run_treatment, treatment=treatment, rng=rng, batch_size=batch_size)
            new_traces.append(dict(genotypes_counts=deepcopy(self.genotypes_counts)))

        self.step += n_steps
        self.traces.extend(new_traces)
        self.make_cell_data()
        return new_traces

    def copy(self):
        return deepcopy(self)

    def get_tumor_size(self):
        return sum(self.genotypes_counts.values())

    def set_cell_exps(self):
        for deme in self.deme_list:
            for cell in deme.cells:        
                if cell.type == 'cancer':
                    cell.baseline_exp = self.celltype_exps['cancer']
                    cell.exp = cell.get_exp(self.selection.mut_effects)
                else:
                    cell.exp = self.celltype_exps[cell.type]

    def make_cell_data(self, cell_prefix='C', **kwargs):
        """
        Make a dataframe containing, for each cell:
        type, evolutionary_parameters, deme_id, gene_exp, gene_mut, gene_cnv
        """
        n_cells = sum(self.genotypes_counts.values())
        cell_evo = []
        cell_snv = np.zeros((n_cells, self.n_genes))
        cell_cnv = np.zeros((n_cells, self.n_genes))
        cell_exp = np.zeros((n_cells, self.n_genes))
        cell_crd = np.zeros((n_cells, 2))
        cell_type = []        
        cell_deme = []
        cell_names = []
        i = 0
        for deme in self.deme_list:
            for cell in deme.cells:
                cell_evo.append(cell.evolutionary_parameters)
                if cell.type == 'cancer':
                    cell.baseline_exp = self.celltype_exps['cancer']
                    cell_exp[i] = cell.get_exp(self.selection.mut_effects)
                else:
                    cell_exp[i] = self.celltype_exps[cell.type]
                cell_snv[i] = cell.get_snvs()
                cell_cnv[i] = cell.get_cnvs()
                cell_crd[i][0], cell_crd[i][1] = deme.row, deme.col # should be after expanding demes
                cell_type.append(cell.genotype_id)
                cell_deme.append(deme.id)
                cell_names.append(f'{cell_prefix}{i}')
                i += 1

        gene_names = self.selection.get_gene_names(**kwargs)
        snv_df = pd.DataFrame(cell_snv.astype(float), index=cell_names, columns=gene_names)

        cell_evo_df = pd.DataFrame(cell_evo, index=cell_names)
        for col, idx in [
            ('n_mut_onc',  self.selection.get_oncogenes()),
            ('n_mut_tsg',  self.selection.get_tsgs()),
            ('n_mut_disp', self.selection.get_dispersal_genes()),
            ('n_mut_ir',   self.selection.get_immune_resistant()),
            ('n_mut_tr',   self.selection.get_treatment_resistant()),
        ]:
            cat_genes = [gene_names[i] for i in idx]
            cell_evo_df[col] = (snv_df[cat_genes] > 0).sum(axis=1).values

        self.cell_data = dict(
            cell_evo=cell_evo_df,
            cell_exp=pd.DataFrame(cell_exp.astype(float), index=cell_names, columns=gene_names),
            cell_snv=snv_df,
            cell_cnv=pd.DataFrame(cell_cnv.astype(float), index=cell_names, columns=gene_names),
            cell_crd=pd.DataFrame(cell_crd.astype(int), index=cell_names, columns=['row', 'col']),
            cell_type=pd.DataFrame(cell_type, index=cell_names, columns=['cell_id']),
            cell_deme=pd.DataFrame(cell_deme, index=cell_names, columns=['deme_id'])
        )
        # cell_rna_vaf (F7b): EXPECTED allele FRACTION in RNA = (v·e)/(v·e + (1-v)), v = DNA-VAF
        # (cell_snv), e = per-locus expression effect (selection.mut_effects); see count.py for the
        # full derivation. Per-gene baseline CANCELS in the fraction, so at a neutral locus (e=1)
        # the *expected fraction* equals DNA-VAF; e>1 inflates / e<1 deflates. The OBSERVED scRNA-VAF
        # is this sampled at the gene's expression depth (per-gene, highly variable) and does NOT
        # match DNA-VAF at neutral loci — depth sampling + obs_fidelity live in reads/rna.py.
        flat_eff = (np.concatenate(self.selection.mut_effects)
                    if self.selection.mut_effects else np.ones(self.n_genes))
        num = cell_snv * flat_eff
        denom = num + (1.0 - cell_snv)
        rna_vaf = np.divide(num, denom, out=np.zeros_like(cell_snv, dtype=float), where=denom > 0)
        self.cell_data["cell_rna_vaf"] = pd.DataFrame(rna_vaf, index=cell_names, columns=gene_names)



    def get_gene_data(self, **kwargs):
        return self.selection.get_gene_data(**kwargs)

    def write(self, output_path):
        # Make output directory
        Path(output_path).mkdir(parents=True, exist_ok=True)

        # if simulated something, write traces
        if len(self.traces) > 0:
            genotypes, _ = self.get_genotype_frequencies()
            parents = self.genotypes_parents

            pd.DataFrame([t["genotypes_counts"] for t in self.traces]).fillna(0).to_csv(
                os.path.join(output_path, "trace_counts.csv")
            )
            pd.DataFrame([parents]).to_csv(os.path.join(output_path, "parents.csv"))
            pd.DataFrame(genotypes).to_csv(os.path.join(output_path, "genotypes.csv"))

        # Make gene data
        gene_data = self.get_gene_data()
        Path(os.path.join(output_path, 'gene_data')).mkdir(parents=True, exist_ok=True)
        for mat in gene_data:
            pd.DataFrame(gene_data[mat]).to_csv(os.path.join(output_path, 'gene_data', f'{mat}.csv'))

        # Make cells by genes matrices
        if self.cell_data is None:
            self.make_cell_data()
        cell_data = self.cell_data
        Path(os.path.join(output_path, f'cell_data')).mkdir(parents=True, exist_ok=True)
        for mat in cell_data:
            pd.DataFrame(cell_data[mat]).to_csv(os.path.join(output_path, f'cell_data', f'{mat}.csv'))

    @staticmethod
    def _prepare_plots(genotype_counts, genotype_parents):
        pop_df = genotype_counts
        pop_df["Generation"] = np.arange(pop_df.shape[0])
        pop_df = (
            pop_df.melt(
                id_vars=["Generation"], value_name="Population", var_name="Identity"
            )
            .sort_values("Generation")
            .reset_index(drop=True)
        )

        anc_df = genotype_parents.melt(var_name="Identity", value_name="Parent").astype(str)

        color_by = pd.Series(
            np.arange(anc_df.shape[0] + 1), index=pop_df["Identity"].unique()
        )

        return pop_df, anc_df, color_by

    @staticmethod
    def get_colormap(populations_df, adjacency_df, color_by, colormap):
        x = populations_df["Generation"].unique()
        y_table = pymuller.logic._get_y_values(populations_df, adjacency_df, 10)
        final_order = y_table.columns.values
        Y = y_table.to_numpy().T
        cmap = plt.get_cmap(colormap)
        color_by = color_by.copy()
        ordered_colors = color_by.loc[final_order]
        norm = matplotlib.colors.Normalize(
            vmin=np.min(ordered_colors), vmax=np.max(ordered_colors)
        )
        color_list = cmap(norm(ordered_colors.values))
        # color_map[0] = color_map[-1] = [1, 1, 1, 1]
        return color_list, final_order

    def plot_muller(self, ax=None, colormap='gnuplot', normalize=True, smoothing_std=0.1, show_axes=True):
        # Prepare plot data
        # Keep only cancer cell statistics
        genotype_counts = pd.DataFrame([t["genotypes_counts"] for t in self.traces]).fillna(0)
        genotype_counts.columns = genotype_counts.columns.astype(str)
        genotype_parents = pd.DataFrame(self.genotypes_parents, index=[0])
        genotype_counts = genotype_counts.drop(columns=list(set(normal_names).intersection(set(genotype_counts.columns))))
        genotype_parents = genotype_parents.drop(columns=list(set(normal_names).intersection(set(genotype_counts.columns))))    

        pop_df, anc_df, color_by = self._prepare_plots(genotype_counts, genotype_parents)

        # Plot Muller diagram
        if ax is None:
            fig, ax = plt.subplots()
        
        pymuller.muller(
            pop_df,
            anc_df,
            color_by,
            ax=ax,
            colorbar=False,
            colormap=colormap,
            normalize=normalize,
            background_strain=False,
            smoothing_std=smoothing_std,
        )
        if show_axes:
            ax.set_xlabel('Step')
            ax.set_ylabel('Cells' if not normalize else 'Frequency')
        else:
            ax.axis('off')

    def get_cell_type_colors(self, colormap="gnuplot"):
        # Keep only cancer cell statistics
        genotype_counts = pd.DataFrame([t["genotypes_counts"] for t in self.traces]).fillna(0)
        genotype_counts.columns = genotype_counts.columns.astype(str)
        genotype_parents = pd.DataFrame(self.genotypes_parents, index=[0])
        genotype_counts = genotype_counts.drop(columns=list(set(normal_names).intersection(set(genotype_counts.columns))))
        genotype_parents = genotype_parents.drop(columns=list(set(normal_names).intersection(set(genotype_counts.columns))))    
        pop_df, anc_df, color_by = self._prepare_plots(genotype_counts, genotype_parents)
        cmap, deme_ids = self.get_colormap(pop_df, anc_df, color_by, colormap) # only make genotype colormap for cancer cells, for the others use fixed
        # Add normal cells in grid plot
        cmap = dict(zip(deme_ids, cmap))
        for normal_name in normal_names:
            cmap[normal_name] = normal_cmap_rgba[normal_name]
        return cmap
