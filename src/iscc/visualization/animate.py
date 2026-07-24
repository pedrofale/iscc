"""
Create an animation of a tumor growth given a list of outputs from a simulation on a grid.

Two modes:
  * DEFAULT  — the historical single-grid growth animation (standard Muller), from grid_*.csv +
               genotype-count/parent CSVs.
  * --compartment — the primary + metastasis COMPOSITE (grids left, centered symmetric Mullers right,
               shared time axis, shared clone colormap), rendered from the compartment trajectory an
               `isccsim` schedule run writes. `--splash` selects the minimal HERO variant.
"""
from .util import *

import pandas as pd
import matplotlib.pyplot as plt
from celluloid import Camera

import click
import os
from pathlib import Path

from pymuller import muller


@click.command(help="Plot the evolution of a tumor (single-grid, or --compartment primary+metastasis).")
@click.argument(
    "grids-dir",
    type=click.Path(exists=True, dir_okay=True),
)
@click.argument(
    "genotype-counts",
    type=click.Path(exists=True, dir_okay=False),
    required=False,
)
@click.argument(
    "genotype-parents",
    type=click.Path(exists=True, dir_okay=False),
    required=False,
)
@click.option("--compartment", is_flag=True,
              help="Render the primary+metastasis composite from an isccsim compartment trajectory "
                   "(GRIDS-DIR is the isccsim output dir). Ignores the two count/parent arguments.")
@click.option("--splash", is_flag=True,
              help="With --compartment: the minimal HERO variant (no legend/axis numbers, "
                   "'Primary'/'Metastasis' titles, per-panel event labels). Default: fully labelled.")
@click.option("--poster", is_flag=True,
              help="With --compartment: also write a poster PNG (final frame) next to the GIF.")
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
    "-o", "--output-path", default="./", help="Output directory (default) OR the output GIF path "
    "(--compartment). A --compartment path without a .gif suffix is treated as a directory.")
def main(
    grids_dir,
    genotype_counts,
    genotype_parents,
    compartment,
    splash,
    poster,
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
    if compartment:
        return _render_compartment(grids_dir, splash, output_path, poster)

    if genotype_counts is None or genotype_parents is None:
        raise click.UsageError(
            "the default single-grid animation needs GENOTYPE_COUNTS and GENOTYPE_PARENTS "
            "(or pass --compartment to render an isccsim trajectory).")

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


def _render_compartment(grids_dir, splash, output_path, poster=False):
    """Render the primary+metastasis composite GIF from the isccsim compartment trajectory in
    GRIDS-DIR (the isccsim output directory). Promotes the landing-animation build_figure/centered-
    Muller/GIF logic into the CLI (see iscc.visualization.compartment)."""
    from ..tumor import arc
    from . import compartment as comp

    traj = arc.read_trajectory(grids_dir)

    # -o is the output GIF path in this mode; a path with no .gif suffix is a directory to drop the
    # default-named GIF into.
    stem = "landing_hero" if splash else "landing_full"
    if output_path.lower().endswith(".gif"):
        gif_path = output_path
    else:
        gif_path = os.path.join(output_path, f"{stem}.gif")

    size_mb = comp.render_animation(traj, gif_path, splash=splash, poster=poster)
    click.echo(f"wrote {gif_path}  ({size_mb:.2f} MB, {len(traj['frames'])} frames, "
               f"{'splash/minimal' if splash else 'full-labelled'})")


if __name__ == "__main__":
    main()
