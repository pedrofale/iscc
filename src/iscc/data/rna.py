from .assay import Assay

import numpy as np
import pandas as pd
import os


class scRNA(Assay):
    """Single-cell RNA-seq with realistic count statistics.

    Each cell's expected per-gene counts are its (ground-truth) expression activities scaled
    by a **variable library size** (lognormal, so library size varies cell-to-cell like real
    10x data). Observed counts are drawn from a **negative binomial** (gamma-Poisson), giving
    the overdispersion and sparsity of real scRNA-seq rather than the underdispersed,
    fixed-depth counts of a plain multinomial.
    """

    def __init__(self, n_reads=1000, n_cells=100, lib_size_sigma=0.35, dispersion=0.3, **assay_kwargs):
        super(scRNA, self).__init__(**assay_kwargs)
        self.n_reads = int(n_reads)          # mean library size (counts per cell)
        self.n_cells = int(n_cells)          # target number of cells
        self.lib_size_sigma = lib_size_sigma  # lognormal sigma -> library-size variation
        self.dispersion = dispersion          # NB overdispersion (var = mu + dispersion*mu^2)

    def run(self, cell_data, **kwargs):
        rng = np.random.default_rng(self.seed)
        cell_states = cell_data["cell_exp"]
        if len(cell_states.index) < self.n_cells:
            self.n_cells = len(cell_states.index)
        target_cells = rng.choice(cell_states.index, size=self.n_cells, replace=False)
        genes = cell_states.columns

        activities = cell_states.loc[target_cells].values.astype(float)
        totals = activities.sum(axis=1, keepdims=True)
        probs = np.divide(activities, totals, out=np.zeros_like(activities), where=totals > 0)

        # variable library size (lognormal with mean ~ n_reads)
        mu_ln = np.log(self.n_reads) - 0.5 * self.lib_size_sigma ** 2
        lib = rng.lognormal(mean=mu_ln, sigma=self.lib_size_sigma, size=self.n_cells)
        mu = probs * lib[:, None]  # expected counts per cell x gene

        # negative binomial via gamma-Poisson mixture: gamma mean = mu, => Poisson overdispersed
        shape = 1.0 / self.dispersion
        rate = rng.gamma(shape=shape, scale=np.maximum(mu, 0) * self.dispersion)
        counts = rng.poisson(rate)

        self.observed_counts = pd.DataFrame(counts.astype(int), index=target_cells, columns=genes)

    def write(self, out_path):
        self.observed_counts.to_csv(os.path.join(out_path, "umis.csv"))
