"""SISTEM end to end at a given population size, for the runtime comparison.

    python sistem_runtime.py <n_cells> <n_sampled> <seed> <outdir>

SISTEM (samsonweiner/sistem) grows well-mixed anatomical sites in discrete generations. Here one site
(the primary, no metastasis) grows under its quickstart's driver-region selection until it holds
<n_cells> cells; <n_sampled> cells are then sampled, their clonal lineage built with passenger
mutations, and bulk read counts generated. Synthetic reads are not generated (they need DWGSIM).
"""
import json
import os
import sys
import time

import numpy as np

from sistem import GrowthSimulator, Parameters
from sistem.anatomy import StaticAnatomy
from sistem.data import gen_readcounts_bulk
from sistem.selection import RandomRegionLibrary


def main():
    n_cells, n_sampled, seed, outdir = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    os.makedirs(outdir, exist_ok=True)
    np.random.seed(seed)
    t0 = time.time()
    params = Parameters(out_dir=outdir, nsites=1, epsilon=0.0, min_detectable=n_cells,
                        capacities=int(10 * n_cells), focal_driver_rate=5e-4, SNV_pass_rate=0.01,
                        ncells_prim=n_sampled, ncells_normal=0, min_mut_fraction=0.05, coverage=100)
    library = RandomRegionLibrary(params=params)
    library.initialize(params=params)
    anatomy = StaticAnatomy(libraries=library, params=params)
    anatomy.initialize_distances(method="random")   # required even with a single site
    gs = GrowthSimulator(anatomy)
    gs.simulate_agents(params=params)
    t_grow = time.time() - t0
    gs.sample_cells(params=params)
    tree = gs.simulate_clonal_lineage(params=params)
    gen_readcounts_bulk(tree, params=params)
    rec = dict(tool="SISTEM", n_cells=n_cells, n_sampled=n_sampled, seed=seed, grow_s=t_grow,
               total_s=time.time() - t0, cells_reached=int(sum(gs.site_counts)))
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
