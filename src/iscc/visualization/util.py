import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from ast import literal_eval

import pymuller
import packcircles as pc
import networkx as nx
from networkx.drawing.nx_agraph import write_dot, graphviz_layout


def prepare_plots(genotype_counts, genotype_parents):
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


def plot_deme(
    n_cells,
    genotype_counts,
    populations_df,
    adjacency_df,
    color_by,
    average_radius=10,
    colormap="gnuplot",
    ax=None,
    dpi=200,
    figsize=(10, 10),
):
    """Create a circle for each cell and color it by genotype.
    The colors are consistent with the Muller plots.
    """
    # Re-use PyMuller to get the same genotype colors
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
    color_map = cmap(norm(ordered_colors.values))

    radii = []
    colors = []
    n_total_cells = sum(genotype_counts.values)
    min_r = max(1, int(average_radius / 2))
    max_r = int(average_radius * 2)
    for i, genotype in enumerate(final_order):
        n = max(1, int(n_cells * genotype_counts[genotype] / n_total_cells))
        g_radii = np.random.randint(min_r, max_r, size=n).tolist()
        radii = radii + g_radii
        colors = colors + [color_map[i]] * n

    # Randomly permute cells to model well mixed population in deme
    perm = np.random.choice(len(radii), size=len(radii), replace=False)
    radii = np.array(radii)[perm]
    colors = np.array(colors)[perm]

    # Pack circles
    circles = pc.pack(radii)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        plt.sca(ax)
    for i, (x, y, radius) in enumerate(circles):
        patch = plt.Circle((x, y), radius, color=colors[i], alpha=0.65)
        ax.add_patch(patch)
    ax.set(xlim=(-150, 140), ylim=(-180, 170))
    plt.axis("off")


def plot_grid(
    genotype_grid,
    colormap,
    genotypes,
    ax=None,
    figsize=(10, 10),
    dpi=100,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        plt.sca(ax)
    # Turn the genotype grid into grid of colors
    color_grid = np.zeros((genotype_grid.shape[0], genotype_grid.shape[1], 4)) + 1.
    color_grid[:,:,-1] = 1.
    for i, genotype in enumerate(genotypes):
        idx = np.where(genotype_grid == genotype)
        color_grid[idx] = colormap[i]
    ax.imshow(color_grid)
    plt.axis("off")
    return ax


def expand_grid(
    genotype_counts_grid, # dataframe of index x,y and genotypes in columns
    original_grid_side,
    minigrid_side, 
):
    """If each entry is a deme with many cells, expand each entry to a grid of specified size
    and color the entries in there by the proportion of the genotype_ids present in it
    Returns a genotype grid
    """
    new_grid_side = original_grid_side * minigrid_side
    new_grid = np.zeros((new_grid_side, new_grid_side), dtype=object)

    cnt = 0
    for ij in genotype_counts_grid.index:
        deme_i, deme_j = np.array(ij.split(',')).astype(int).tolist()
        deme_i_start, deme_j_start = deme_i * minigrid_side, deme_j * minigrid_side
        deme_i_end, deme_j_end = deme_i_start + minigrid_side, deme_j_start + minigrid_side
        for pos_x in range(deme_i_start, deme_i_end):
            for pos_y in range(deme_j_start, deme_j_end):
                # Sample genotype positions in each grid point
                rng = np.random.default_rng(cnt)
                counts = genotype_counts_grid.loc[ij]
                if np.sum(counts) == 0:
                    new_grid[pos_x,pos_y] = ""
                else:
                    new_grid[pos_x,pos_y] = rng.choice(genotype_counts_grid.columns, p=counts/np.sum(counts))
                cnt += 1
    return new_grid


def plot_tree(
    genotype_parents,
    populations_df,
    adjacency_df,
    color_by,
    genotype_counts=None,
    labels=False,
    filter_clones=True,
    colormap="gnuplot",
    ax=None,
):
    # Re-use PyMuller to get the same genotype colors
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
    color_map = cmap(norm(ordered_colors.values))

    G = nx.DiGraph()

    root = ["0"] * len(list(genotype_parents.columns[0]))
    root = "".join(root)
    G.add_node(root)

    new_parent_dict = genotype_parents.to_dict("records")[0]
    if filter_clones:
        for clone in genotype_parents.columns:
            if clone != root:
                if genotype_counts[clone] < 2:
                    if clone in new_parent_dict.values():
                        # Get children and attach to parent of this node
                        children_pos = [
                            i
                            for i, x in enumerate(list(new_parent_dict.values()))
                            if x == clone
                        ]
                        for child_pos in children_pos:
                            child = list(new_parent_dict.keys())[child_pos]
                            new_parent_dict[child] = new_parent_dict[clone]
                    del new_parent_dict[clone]

    for clone in list(new_parent_dict.keys()):
        G.add_node(clone)
        G.add_edge(new_parent_dict[clone], clone)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure if hasattr(ax, "figure") else ax.fig

    color_values = []
    label_values = dict()
    for node in G.nodes():
        if labels and genotype_counts is not None:
            label_values[node] = int(genotype_counts[node])
        color_values.append(
            matplotlib.colors.to_hex(color_map[np.where(final_order == node)[0]][0])
        )

    pos = graphviz_layout(G, prog="dot")
    nx.draw(
        G,
        pos,
        with_labels=labels,
        arrows=False,
        node_color=color_values,
        labels=label_values,
        ax=ax,
    )

    # Fix the margins
    x_values, y_values = zip(*pos.values())
    x_min, x_max = (min(x_values), max(x_values))
    y_min, y_max = (min(y_values), max(y_values))
    x_margin = (x_max - x_min) * 0.25
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    y_margin = (y_max - y_min) * 0.25
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    return ax
