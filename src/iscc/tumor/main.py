"""
Simulate tumor growth under different spatial models. Inspired by Noble et al, 2019.
"""
from .components.cell import CancerCell
from .components.selection import Selection
from .tumor import Tumor

import numpy as np
import pandas as pd

import click
import os
from pathlib import Path
import logging
import yaml

MODE_LIST = [
    simulate_nonspatial,
    simulate_invasion,
    simulate_fission,
    simulate_boundary,
]


@click.command(help="Simulate tumor evolution under different spatial constraints.")
@click.option("--sim-config", type=click.Path(exists=True, dir_okay=False), help="Config file with simulation parameters")
@click.option(
    "-s",
    "--steps",
    default=1000,
    help="Number of steps in simulation.",
)
@click.option(
    "-r",
    "--random_seed",
    default=42,
    help="Random seed for the pseudo random number generator.",
)
@click.option(
    "--record-after-steps", default=0, help="Record the simulation state every S steps."
)
@click.option(
    "--log", default=0, help="Logging level. 0 for no logging, 1 for info, 2 for debug."
)
@click.option("-o", "--output-path", default="./sim_out", help="Output directory")
def main(
    sim_config,
    steps,
    random_seed,
    log,
    record_after_steps,
    output_path,
):
    if log == 0:
        log = logging.CRITICAL
    elif log == 1:
        log = logging.INFO
    elif log == 2:
        log == logging.DEBUG
    logging.basicConfig(level=log)

    with open(sim_config) as f:
        config = yaml.safe_load(f)  

    cancer_cell = CancerCell(
        n_segments=config['cell_params']['n_segments'],
        seed=random_seed,
        **config['cell_params']['cancer_params'],
    )

    selection = Selection(
        n_segments=cancer_cell.n_segments,
        **config['selection_params'],
    )

    tumor = Tumor(cancer_cell, selection,
                  epithelial_cell_params=config['cell_params']['epithelial_params'],
                    stromal_cell_params=config['cell_params']['stromal_params'],
                    immune_cell_params=config['cell_params']['immune_params'],
                    deme_params=config['deme_params'],
                    **config['spatial_params'])

    if record_after_steps > 0:
        records = int(steps/record_after_steps)
    else:
        records = 1
        record_after_steps = steps
    envs = []
    env, traces, treatment_target, cells_killed = MODE_LIST[config['mode']](
        record_after_steps,
        tumor,
        seed=random_seed,
        **config['treatment_params'],
    )
    genotypes, _ = env.get_genotype_frequencies()
    parents = env.genotypes_parents

    # Make output directory
    Path(output_path).mkdir(parents=True, exist_ok=True)

    pd.DataFrame([t["genotypes_counts"] for t in traces]).fillna(0).to_csv(
        os.path.join(output_path, "trace_counts_0.csv")
    )
    pd.DataFrame([parents]).to_csv(os.path.join(output_path, "parents_0.csv"))
    pd.DataFrame(genotypes).to_csv(os.path.join(output_path, "genotypes_0.csv"))

    # Make gene data
    gene_data = env.get_gene_data()
    Path(os.path.join(output_path, 'gene_data')).mkdir(parents=True, exist_ok=True)
    for mat in gene_data:
        pd.DataFrame(gene_data[mat]).to_csv(os.path.join(output_path, 'gene_data', f'{mat}.csv'))

    if config['mode'] > 0:
        genotype_matrix = env.get_genotype_matrix()
        pd.DataFrame(genotype_matrix).to_csv(os.path.join(output_path, "grid_0.csv"))

        # Save genotype counts per deme in this step
        coords = []
        gcounts = []
        for deme in env.deme_list:
            coords.append(f'{deme.row},{deme.col}')
            gcounts.append(deme.genotypes_counts)
        df = pd.DataFrame(gcounts).fillna(0)
        df.index = coords
        df.to_csv(os.path.join(output_path, f"genotype_counts_demes_0.csv"))

    for i in range(1, records):
        env, traces, treatment_target, cells_killed = MODE_LIST[config['mode']](
            record_after_steps,
            env,
            traces=traces,
            treatment_target=treatment_target,
            cells_killed=cells_killed,
            seed=random_seed + i,
            **config['treatment_params'],         
        )
        genotypes, _ = env.get_genotype_frequencies()
        parents = env.genotypes_parents


        Path(output_path).mkdir(parents=True, exist_ok=True)
        pd.DataFrame([t["genotypes_counts"] for t in traces]).fillna(0).to_csv(
            os.path.join(output_path, f"trace_counts_{i}.csv")
        )
        pd.DataFrame([parents]).to_csv(os.path.join(output_path, f"parents_{i}.csv"))
        pd.DataFrame(genotypes).to_csv(os.path.join(output_path, f"genotypes_{i}.csv"))

        # Make cells by genotypes matrix
        cell_data = env.get_cell_data()
        Path(os.path.join(output_path, f'cell_data_{i}')).mkdir(parents=True, exist_ok=True)
        for mat in cell_data:
            os.mkdir(exists_ok=True)
            pd.DataFrame(cell_data[mat]).to_csv(os.path.join(output_path, f'cell_data_{i}', f'{mat}.csv'))

        if config['mode'] > 0:
            genotype_matrix = env.get_genotype_matrix()
            pd.DataFrame(genotype_matrix).to_csv(os.path.join(output_path, f"grid_{i}.csv"))

            # Save genotype counts per deme in this step
            coords = []
            gcounts = []
            for deme in env.deme_list:
                coords.append(f'{deme.row},{deme.col}')
                gcounts.append(deme.genotypes_counts)
            df = pd.DataFrame(gcounts).fillna(0)
            df.index = coords
            df.to_csv(os.path.join(output_path, f"genotype_counts_demes_{i}.csv"))
                
    # Make cells by genes matrices
    env.set_cell_exps()
    cell_data = env.get_cell_data()
    Path(os.path.join(output_path, f'cell_data_{records-1}')).mkdir(parents=True, exist_ok=True)
    for mat in cell_data:
        pd.DataFrame(cell_data[mat]).to_csv(os.path.join(output_path, f'cell_data_{records-1}', f'{mat}.csv'))


    print(f"Simulation in mode {config['mode']} finished.")

    print(f"Saved results to {output_path}.")


if __name__ == "__main__":
    main()
