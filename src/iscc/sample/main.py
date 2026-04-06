"""
Simulate molecular data from a tumor.
"""
from .biopsy import *
from .dissociation import *
from .slice import *

import numpy as np
import pandas as pd

import click
import os
from pathlib import Path
import logging
import yaml

@click.command(help="Simulate molecular data from a tumor.")
@click.argument(
    "cell-data-path",
    type=click.Path(exists=True, dir_okay=True),
)
@click.option("-a", "--assay", default="bdna", help="Type of molecular data to obtain.")
@click.option("--assay-config", default='assayconfig/bdna.yaml', type=click.Path(exists=True, dir_okay=False), help="Config file for assay type")
@click.option("-s", "--spatial", default=True, help="Wether to consider the spatial structure in the assay.")
@click.option("--biopsy-config", default='biopsyconfigs/biopsy.yaml', type=click.Path(exists=True, dir_okay=False), help="Config file for spatial-aware sample")
@click.option("--grid-file", default="", help="Path to grid file (optional). Assumes each pixel is a cell.")
@click.option(
    "--log", default=0, help="Logging level. 0 for no logging, 1 for info, 2 for debug."
)
@click.option("-o", "--output-path", default="./sample_out", help="Output directory")
def main(
    cell_data_path,
    assay,
    assay_config,
    spatial,
    biopsy_config,
    grid_file,
    log,
    output_path,
):
    if log == 0:
        log = logging.CRITICAL
    elif log == 1:
        log = logging.INFO
    elif log == 2:
        log == logging.DEBUG
    logging.basicConfig(level=log)

    # Make output directory
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # Read configs
    with open(biopsy_config) as f:
        biopsy_config = yaml.safe_load(f)
    with open(assay_config) as f:
        assay_config = yaml.safe_load(f)        

    # Read cell data
    cell_data = dict(cell_snv=pd.read_csv(os.path.join(cell_data_path, 'cell_snv.csv'), index_col=0),
                     cell_exp=pd.read_csv(os.path.join(cell_data_path, 'cell_exp.csv'), index_col=0),
                     cell_crd=pd.read_csv(os.path.join(cell_data_path, 'cell_crd.csv'), index_col=0),
    )
    cell_ids = cell_data['cell_snv'].index
    grid_side = None

    if not spatial:
        sampled_cell_ids = sample(cell_ids, **biopsy_config) # subset of cells
        # Subsample cells
        for key in cell_data:
            cell_data[key] = cell_data[key].loc[sampled_cell_ids]
    else:
        grid = pd.read_csv(grid_file, index_col=0, dtype=str)
        grid_side = grid.shape[0]
        # Select regions in space
        # cell_data = sample_regions(cell_data, grid, **biopsy_config) # subsections of the grid

    logging.info(f"Performing {ASSAY_NAMES[assay]}.")
    assay = ASSAYS[assay](**assay_config)
    assay.run(cell_data, grid_side=grid_side)
    assay.write(output_path)
    logging.info(f"Saved results to {output_path}.")


if __name__ == "__main__":
    main()
