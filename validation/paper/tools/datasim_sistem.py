"""SISTEM's single-cell read counts at DLP+'s depth, for the data-simulator supplement (supp_datasim.py).

    python datasim_sistem.py <out_dir> <coverage> [seed]

Run with the Python of an environment holding SISTEM 1.0.5 (samsonweiner/sistem). One primary site grows
under the quickstart's driver-region selection (as comparisons/sistem_runtime.py) to 5,000 detectable
cells; 100 cells are sampled, their single-cell lineage is built, and read counts are drawn per 500 kb
bin (``region_len`` = ``bin_size`` = 500 kb) at ``coverage``, which datasim_run.sh sets so that the mean
reads per bin at DLP+'s mean copy number match DLP+'s.

Coverage is uniform (``lorenz_y`` 0.5, the reads of a bin Poisson): in SISTEM 1.0.5 the non-uniform path
of ``gen_readcounts_singlecell`` raises ``TypeError: object of type 'int' has no len()``
(``draw_readcounts_nonuniform(len(len(chromosome.seq)), ...)`` in ``sistem/data/reads.py``), checked on
2026-11-25 at DLP+'s Lorenz point. SISTEM writes ``readcounts.tsv`` and ``observed_CNPs.tsv`` to <out_dir>.
"""
import os
import sys

import numpy as np
from sistem import GrowthSimulator, Parameters
from sistem.anatomy import StaticAnatomy
from sistem.data import gen_readcounts_singlecell
from sistem.selection import RandomRegionLibrary

BIN = 500_000


def main():
    out, coverage = sys.argv[1], float(sys.argv[2])
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    os.makedirs(out, exist_ok=True)
    np.random.seed(seed)
    params = Parameters(out_dir=out, nsites=1, epsilon=0.0, min_detectable=5000, capacities=50000,
                        focal_driver_rate=5e-4, SNV_pass_rate=0.01, ncells_prim=100, ncells_normal=0,
                        min_mut_fraction=0.05, region_len=BIN, bin_size=BIN, coverage=coverage, lorenz_y=0.5)
    library = RandomRegionLibrary(params=params)
    library.initialize(params=params)
    anatomy = StaticAnatomy(libraries=library, params=params)
    anatomy.initialize_distances(method="random")   # required even with a single site
    gs = GrowthSimulator(anatomy)
    gs.simulate_agents(params=params)
    gs.sample_cells(params=params)
    tree = gs.simulate_singlecell_lineage(params=params)
    gen_readcounts_singlecell(tree, params=params, lorenz_y=0.5)


if __name__ == "__main__":
    main()
