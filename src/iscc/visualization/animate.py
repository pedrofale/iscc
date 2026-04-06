"""
Create an animation of a tumor growth given a list of outputs from a simulation on a grid
"""
from .util import *

import pandas as pd
import matplotlib.pyplot as plt
from celluloid import Camera

import click
import os
from pathlib import Path

from pymuller import muller


@click.command(help="Plot the evolution of a tumor.")
@click.argument(
    "grids-dir",
    type=click.Path(exists=True, dir_okay=True),
)
@click.argument(
    "genotype-counts",
    type=click.Path(exists=True, dir_okay=False),
)
@click.argument(
    "genotype-parents",
    type=click.Path(exists=True, dir_okay=False),
)
@click.option("--colormap", default="gnuplot", help="Colormap for genotypes.")
@click.option("--figw", default=8, help="Figure width.")
@click.option("--figh", default=8, help="Figure height.")
@click.option("--dpi", default=100, help="DPI for figures.")
@click.option("--fps", default=15, help="Frames per second for animation.")
@click.option("--bitrate", default=1800, help="Bitrate for animation.")
@click.option("--interval", default=50, help="Interval (ms) for animation.")
@click.option(
        "--suffix", default="")
@click.option(
        "--file-format", default="png")
@click.option(
    "-o", "--output-path", default="./", help="Directory to write figures into."
)
def main(
    grids_dir,
    genotype_counts,
    genotype_parents,
    colormap,
    figw,
    figh,
    dpi,
    fps,
    bitrate,
    interval,
    suffix,
    file_format,
    output_path,
):
    # Make colormap
    genotype_counts = pd.read_csv(genotype_counts, index_col=0)
    genotype_parents = pd.read_csv(genotype_parents, index_col=0, dtype=str)
    pop_df, anc_df, color_by = prepare_plots(genotype_counts, genotype_parents)
    cmap, genotypes = get_colormap(pop_df, anc_df, color_by, colormap)

    # Load grids
    grid_file_path_list = [f for f in os.listdir(grids_dir) if ("grid_" in f and ".csv" in f)]
    grid_file_path_nums = [int(f.split("_")[-1].split(".")[0]) for f in grid_file_path_list]
    def argsort(seq):
        # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
        return sorted(range(len(seq)), key=seq.__getitem__)
    grid_file_path_list = [grid_file_path_list[i] for i in argsort(grid_file_path_nums)]
    grids = []
    for grid_file_path in grid_file_path_list:
        grid = pd.read_csv(os.path.join(grids_dir, grid_file_path), index_col=0, dtype=str)
        grids.append(grid)

    # Make animation
    fig, ax = plt.subplots(figsize=(figw, figh), dpi=dpi)
    camera = Camera(fig)
    for i in range(len(grids)):
        plot_grid(grids[i], cmap, genotypes, ax=ax)
        camera.snap()
    animation = camera.animate()
    animation.save(os.path.join(output_path, f'slice{suffix}.gif'))
    #
    #
    # def animate(i):
    #     return plot_grid(grids[i], cmap, genotypes, ax=ax)
    # writer = animation.PillowWriter(fps=fps,
    #                                 metadata=dict(artist='Me'),
    #                                 bitrate=bitrate)
    # ani = animation.FuncAnimation(fig, animate, repeat=True,
    #                                 frames=len(grids), interval=interval)
    # ani.save(os.path.join(output_path, f'slice{suffix}.gif'), writer=writer)

if __name__ == "__main__":
    main()
