"""Engine-agnostic plotting helpers shared by the tumor engines.

These operate purely on the data both engines expose — `traces`
(list of ``{"genotypes_counts": {gid: count}}``), `genotypes_parents`, the materialised
`cell_data` dict, and the grid size — so the same Muller / spatial-grid plots work for the
cell-level and genotype-level engines.
"""
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import pymuller

from ..constants import normal_names, normal_cmap_rgba


def _prepare(genotype_counts, genotype_parents):
    pop_df = genotype_counts
    pop_df["Generation"] = np.arange(pop_df.shape[0])
    pop_df = (
        pop_df.melt(id_vars=["Generation"], value_name="Population", var_name="Identity")
        .sort_values("Generation")
        .reset_index(drop=True)
    )
    anc_df = genotype_parents.melt(var_name="Identity", value_name="Parent").astype(str)
    color_by = pd.Series(np.arange(anc_df.shape[0] + 1), index=pop_df["Identity"].unique())
    return pop_df, anc_df, color_by


def _get_colormap(pop_df, anc_df, color_by, colormap):
    y_table = pymuller.logic._get_y_values(pop_df, anc_df, 10)
    final_order = y_table.columns.values
    cmap = plt.get_cmap(colormap)
    ordered_colors = color_by.copy().loc[final_order]
    norm = matplotlib.colors.Normalize(vmin=np.min(ordered_colors), vmax=np.max(ordered_colors))
    return cmap(norm(ordered_colors.values)), final_order


def _cancer_only(traces, genotypes_parents):
    genotype_counts = pd.DataFrame([t["genotypes_counts"] for t in traces]).fillna(0)
    genotype_counts.columns = genotype_counts.columns.astype(str)
    genotype_parents = pd.DataFrame(genotypes_parents, index=[0])
    drop = list(set(normal_names).intersection(set(genotype_counts.columns)))
    genotype_counts = genotype_counts.drop(columns=drop)
    genotype_parents = genotype_parents.drop(columns=[c for c in drop if c in genotype_parents.columns])
    return genotype_counts, genotype_parents


def plot_muller(traces, genotypes_parents, ax=None, colormap="gnuplot", normalize=True,
                smoothing_std=0.1, show_axes=True):
    genotype_counts, genotype_parents = _cancer_only(traces, genotypes_parents)
    if ax is None:
        _, ax = plt.subplots()
    # pymuller needs at least a couple of clones with an ancestry edge; a tumor that went
    # extinct or never diversified has none, so draw an informative empty panel instead.
    if genotype_counts.shape[1] == 0 or genotype_parents.shape[1] == 0 or genotype_counts.values.sum() == 0:
        ax.axis("off")
        ax.text(0.5, 0.5, "no surviving clones", ha="center", va="center", transform=ax.transAxes)
        return ax
    pop_df, anc_df, color_by = _prepare(genotype_counts, genotype_parents)
    pymuller.muller(pop_df, anc_df, color_by, ax=ax, colorbar=False, colormap=colormap,
                    normalize=normalize, background_strain=False, smoothing_std=smoothing_std)
    if show_axes:
        ax.set_xlabel("Step")
        ax.set_ylabel("Frequency" if normalize else "Cells")
    else:
        ax.axis("off")
    return ax


def cell_type_colors(traces, genotypes_parents, colormap="gnuplot"):
    genotype_counts, genotype_parents = _cancer_only(traces, genotypes_parents)
    pop_df, anc_df, color_by = _prepare(genotype_counts, genotype_parents)
    cmap, ids = _get_colormap(pop_df, anc_df, color_by, colormap)
    cmap = dict(zip(ids, cmap))
    for name in normal_names:
        cmap[name] = normal_cmap_rgba[name]
    return cmap


def plot_grid(cell_data, grid_size, traces, genotypes_parents, color=None, cmap="viridis",
              ax=None, figsize=(10, 10), dpi=100):
    if color is None:
        color = ["cell_type"]
    if ax is None:
        _, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        plt.sca(ax)

    for color_key in color:
        if color_key == "cell_type":
            type_cmap = cell_type_colors(traces, genotypes_parents)
            base = cell_data["cell_deme"].join(cell_data["cell_crd"])
            base["val"] = cell_data["cell_type"]["cell_id"].values
            deme_data = base.groupby(["deme_id"]).agg(lambda x: x.mode().iloc[0])
            grid = np.ones((grid_size, grid_size, 4))
            legend_patches = []
            for genotype, col in type_cmap.items():
                mask = deme_data["val"] == genotype
                rows = deme_data.loc[mask, "row"].astype(int).values
                cols = deme_data.loc[mask, "col"].astype(int).values
                if len(rows) > 0:
                    grid[rows, cols] = col
                    label = genotype if genotype in normal_names else f"cancer ({str(genotype)[:6]}…)"
                    legend_patches.append(mpatches.Patch(color=col, label=label))
            ax.imshow(grid)
            ax.legend(handles=legend_patches, loc="upper right", fontsize=7, framealpha=0.8)
        else:
            base = cell_data["cell_deme"].join(cell_data["cell_crd"])
            val = None
            for prefix, key in [("snv_", "cell_snv"), ("cnv_", "cell_cnv"), ("exp_", "cell_exp")]:
                if color_key.startswith(prefix):
                    val = cell_data[key][color_key.split("_", 1)[1]]
            if val is None:
                for key in ["cell_evo", "cell_exp", "cell_snv"]:
                    if key in cell_data and color_key in cell_data[key].columns:
                        val = cell_data[key][color_key].values
                        break
            if val is None:
                continue
            base["val"] = val
            deme_data = base.groupby(["deme_id"]).mean(numeric_only=True)
            grid = np.zeros((grid_size, grid_size), dtype=float)
            grid[deme_data["row"].astype(int), deme_data["col"].astype(int)] = deme_data["val"]
            im = ax.imshow(grid, cmap=cmap)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(color_key)
    plt.axis("off")
    return ax
