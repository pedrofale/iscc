"""
Create a cartoon of a tumor given the frequencies of different genotypes.
"""
from .util import *
from ..constants import *

import pandas as pd
import matplotlib.pyplot as plt

import click
import os
from pathlib import Path

from pymuller import muller


@click.command(help="Plot the evolution of a tumor.")
@click.argument(
    "genotype-counts",
    type=click.Path(exists=True, dir_okay=False),
)
@click.argument(
    "genotype-parents",
    type=click.Path(exists=True, dir_okay=False),
)
@click.option("-c", "--cells", default=100, help="Number of cells in slice plot.")
@click.option(
    "-r",
    "--average-radius",
    default=10,
    help="Average radius of circles in slice plot.",
)
@click.option("--grid-file", default="", help="Path to grid file.")
@click.option("--deme-counts-file", default="", help="Path to genotype counts per deme file.")
@click.option("--colormap", default="gnuplot", help="Colormap for genotypes.")
@click.option("--dpi", default=100, help="DPI for figures.")
@click.option("--plot", is_flag=True, help="Plot all the figures.")
@click.option("--do-muller", is_flag=True, help="Make a Muller plot.")
@click.option("--do-slice", is_flag=True, help="Make a slice plot.")
@click.option("--do-tree", is_flag=True, help="Make a clone tree plot.")
@click.option(
    "--normalize", is_flag=True, help="Normalize the abundances in the Muller plot."
)
@click.option("--smoothing-std", default=10)
@click.option("--labels", is_flag=True, help="Annotate the clone tree plot.")
@click.option(
    "--remove", is_flag=True, help="Remove empty clones in the clone tree plot."
)
@click.option(
    "--expand", default=1, help="Expand each grid into a grid of this side"
)
@click.option(
        "--file-format", default="png")
@click.option(
        "--suffix", default="")
@click.option(
    "-o", "--output-path", default="./fig_out", help="Directory to write figures into."
)
def main(
    genotype_counts,
    genotype_parents,
    cells,
    average_radius,
    grid_file,
    deme_counts_file,
    colormap,
    dpi,
    plot,
    do_muller,
    do_slice,
    do_tree,
    normalize,
    smoothing_std,
    labels,
    remove,
    expand,
    file_format,
    suffix,
    output_path,
):
    genotype_counts = pd.read_csv(genotype_counts, index_col=0)
    genotype_parents = pd.read_csv(genotype_parents, index_col=0, dtype=str)
    if grid_file != "":
        grid = pd.read_csv(grid_file, index_col=0, dtype=str)

    if deme_counts_file != "" and expand > 1:   
        genotype_counts_grid = pd.read_csv(deme_counts_file, index_col=0)
        grid = expand_grid(
                genotype_counts_grid, # dataframe of index x,y and genotypes in columns
                grid.shape[0],
                expand)

    # Keep only cancer cell statistics
    genotype_counts = genotype_counts.drop(columns=list(set(normal_names).intersection(set(genotype_counts.columns))))
    genotype_parents = genotype_parents.drop(columns=list(set(normal_names).intersection(set(genotype_counts.columns))))

    pop_df, anc_df, color_by = prepare_plots(genotype_counts, genotype_parents)
    if anc_df.empty:
        c = np.zeros((4,))
        c[-1] = 1.
        cmap = [c]
        deme_ids = [pop_df.iloc[0]['Identity']]
    else:
        cmap, deme_ids = get_colormap(pop_df, anc_df, color_by, colormap) # only make genotype colormap for cancer cells, for the others use fixed
    # Add normal cells in grid plot
    cmap = np.vstack([cmap, normal_colors_rgba])
    deme_ids = np.concatenate([deme_ids, normal_names])

    if plot:
        fig, ax_list = plt.subplots(ncols=3, sharex=False, dpi=dpi, figsize=(8, 2))
        muller(
            pop_df,
            anc_df,
            color_by,
            ax=ax_list[0],
            colorbar=False,
            colormap=colormap,
            normalize=normalize,
            background_strain=False,
            smoothing_std=smoothing_std,
        )
        plt.axis("off")

        if grid_file == "":
            plot_deme(
                cells,
                genotype_counts.iloc[-1],
                pop_df,
                anc_df,
                color_by,
                average_radius=average_radius,
                colormap=colormap,
                ax=ax_list[1],
            )
        else:
    


            plot_grid(grid, cmap, deme_ids, ax=ax_list[1])

        plot_tree(
            genotype_parents,
            pop_df,
            anc_df,
            color_by,
            genotype_counts=genotype_counts.iloc[-1],
            filter_clones=remove,
            labels=labels,
            colormap=colormap,
            ax=ax_list[2],
        )
        plt.show()
    else:
        Path(output_path).mkdir(parents=True, exist_ok=True)
        if do_muller:
            print("Plotting Muller diagram...")
            ax = muller(
                pop_df,
                anc_df,
                color_by,
                colorbar=False,
                colormap=colormap,
                normalize=normalize,
                background_strain=False,
                smoothing_std=smoothing_std,
            )
            plt.axis("off")
            plt.savefig(
                os.path.join(output_path, f"muller{suffix}.{file_format}"), dpi=dpi, bbox_inches="tight",
                transparent=True
            )

        if do_slice:
            if grid_file == "":
                ax = plot_deme(
                    cells,
                    genotype_counts.iloc[-1],
                    pop_df,
                    anc_df,
                    color_by,
                    average_radius=average_radius,
                    colormap=colormap,
                )
            else:
                ax = plot_grid(grid, cmap, deme_ids)
            plt.savefig(
                os.path.join(output_path, f"slice{suffix}.{file_format}"), dpi=dpi, bbox_inches="tight",
                transparent=True
            )

        if do_tree:
            ax = plot_tree(
                genotype_parents,
                pop_df,
                anc_df,
                color_by,
                genotype_counts=genotype_counts.iloc[-1],
                filter_clones=remove,
                labels=labels,
                colormap=colormap,
            )
            plt.savefig(
                os.path.join(output_path, f"tree{suffix}.{file_format}"), dpi=dpi, bbox_inches="tight",
                transparent=True
            )


if __name__ == "__main__":
    main()
