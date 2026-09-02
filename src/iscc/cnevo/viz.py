"""Figures for the CN evolution analysis.

Matplotlib only, Agg-safe (every function takes or creates an ``ax`` and never calls ``show``), so
these work unchanged inside a Snakemake rule. Copy-number heatmaps reuse the repo's existing
convention — the diverging ``CNV_CMAP`` centred on CN 2 via
:func:`~iscc.tumor.viz.cnv_norm` — so a CN panel here reads the same as one from ``plot_grid``.

The population-level figures (Muller plot, clone tree, tissue) are NOT reimplemented here: use
``tumor.plot_muller()``, ``tumor.plot_clone_tree()``, ``tumor.plot_phylogeny()`` and
``tumor.plot_tissue()`` directly.
"""
import numpy as np

from ..tumor.viz import CNV_CMAP, cnv_norm

__all__ = ["tree_leaf_order", "plot_cn_heatmap", "plot_pairwise_cn_distance",
           "plot_pairwise_shared_events", "plot_diversity_over_time",
           "plot_mode_space_trajectory", "plot_growth_phase", "plot_cn_landscape_over_time",
           "plot_segment_recurrence", "plot_colonisation_curve", "plot_focus_divergence"]


def _resample(y, n):
    """Put a per-snapshot series onto ``n`` points, so a strided trajectory still lines up."""
    y = np.asarray([np.nan if v is None else float(v) for v in y], dtype=float)
    if y.size == n:
        return y
    if y.size == 0:
        return np.full(n, np.nan)
    return np.interp(np.linspace(0, y.size - 1, n), np.arange(y.size), y)


def _ax(ax, figsize=(8, 5)):
    if ax is None:
        import matplotlib.pyplot as plt
        _, ax = plt.subplots(figsize=figsize)
    return ax


def tree_leaf_order(children, root, leaf_of):
    """Clone ids in depth-first leaf order — the row order a CN heatmap should use.

    Sorting rows by the tree makes shared copy-number blocks line up into contiguous bands, which
    is what makes a truncal alteration visible as a band rather than scattered rows.
    """
    leaf_to_clone = {v: k for k, v in leaf_of.items()}
    order, stack = [], [root]
    while stack:
        n = stack.pop()
        kids = children.get(n, [])
        if not kids:
            c = leaf_to_clone.get(n)
            if c is not None and c not in order:
                order.append(c)
        else:
            stack.extend(reversed(kids))
    for c in leaf_of:                        # anything the walk missed
        if c not in order:
            order.append(c)
    return order


def plot_cn_heatmap(total_cn, clone_ids, coords, ax=None, title="Ground-truth copy number"):
    """Clones x segments copy-number heatmap with chromosome boundaries marked."""
    ax = _ax(ax, (10, 5))
    m = np.asarray(total_cn, dtype=float)
    im = ax.imshow(m, aspect="auto", interpolation="nearest",
                   cmap=CNV_CMAP, norm=cnv_norm(m))
    chrom = coords.sort_values("segment")["chrom"].to_numpy()
    edges = [i for i in range(1, len(chrom)) if chrom[i] != chrom[i - 1]]
    for e in edges:
        ax.axvline(e - 0.5, color="0.25", lw=0.6)
    ax.set_yticks(range(len(clone_ids)))
    ax.set_yticklabels(clone_ids, fontsize=6)
    ax.set_xlabel("segment"); ax.set_ylabel("clone (tree order)")
    ax.set_title(title)
    ax.figure.colorbar(im, ax=ax, label="total CN", fraction=0.025)
    return ax


def plot_pairwise_cn_distance(total_cn, clone_ids, ax_heat=None, ax_hist=None):
    """Pairwise L1 copy-number distance: heatmap plus the off-diagonal histogram.

    Zero-distance pairs are the failure mode Q5's ``all_unique_cnps`` reports — two clones no
    CN-based method can tell apart — so they are called out explicitly on the histogram.
    """
    m = np.asarray(total_cn, dtype=float)
    n = m.shape[0]
    D = np.abs(m[:, None, :] - m[None, :, :]).sum(axis=2)
    ax_heat = _ax(ax_heat, (6, 5))
    im = ax_heat.imshow(D, cmap="viridis", interpolation="nearest")
    ax_heat.set_xticks(range(n)); ax_heat.set_xticklabels(clone_ids, rotation=90, fontsize=6)
    ax_heat.set_yticks(range(n)); ax_heat.set_yticklabels(clone_ids, fontsize=6)
    ax_heat.set_title(f"Pairwise CN L1 distance (max CN {m.max():.0f})")
    ax_heat.figure.colorbar(im, ax=ax_heat, fraction=0.045)

    iu = np.triu_indices(n, 1)
    off = D[iu]
    ax_hist = _ax(ax_hist, (6, 4))
    if off.size:
        ax_hist.hist(off, bins=min(40, max(5, off.size // 2)), color="#4c78a8")
        n_zero = int((off == 0).sum())
        ax_hist.axvline(0, color="crimson", lw=1.5)
        ax_hist.set_title(f"{n_zero} identical pair(s) of {off.size}"
                          + ("  <-- indistinguishable" if n_zero else ""))
    ax_hist.set_xlabel("L1 CN distance"); ax_hist.set_ylabel("# pairs")
    return ax_heat, ax_hist


def plot_pairwise_shared_events(shared, clone_ids, ax=None):
    """Upper-triangle heatmap of shared inherited CN events between clone pairs."""
    ax = _ax(ax, (6, 5))
    m = np.array(shared, dtype=float)
    m[np.tril_indices(m.shape[0])] = np.nan
    im = ax.imshow(m, cmap="magma", interpolation="nearest")
    ax.set_xticks(range(len(clone_ids))); ax.set_xticklabels(clone_ids, rotation=90, fontsize=6)
    ax.set_yticks(range(len(clone_ids))); ax.set_yticklabels(clone_ids, fontsize=6)
    ax.set_title("Shared inherited CN events")
    ax.figure.colorbar(im, ax=ax, fraction=0.045)
    return ax


def plot_diversity_over_time(traj, growth=None, ax=None):
    """Diversity indices vs generation, with the r/K split shaded when it is known."""
    ax = _ax(ax, (9, 4.5))
    ax.plot(traj["gen"], traj["eff_genotypes"], color="#4c78a8", label="effective genotypes")
    ax.plot(traj["gen"], traj["n_genotypes"], color="#9ecae9", ls="--", label="genotypes")
    ax.set_xlabel("generation"); ax.set_ylabel("count")
    ax2 = ax.twinx()
    ax2.plot(traj["gen"], traj["D"], color="#54a24b", label="Noble D (driver combos)")
    ax2.set_ylabel("D", color="#54a24b")
    if growth and growth.get("t_rK") is not None:
        ax.axvspan(growth["t_rK"], traj["gen"].max(), color="0.85", zorder=0)
        ax.axvline(growth["t_rK"], color="0.4", lw=1)
        ax.text(growth["t_rK"], ax.get_ylim()[1], " K phase", va="top", fontsize=8, color="0.3")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    ax.set_title("Clonal diversity over time")
    return ax


def plot_mode_space_trajectory(traj, empirical=None, ax=None):
    """Path through Noble's ``(n, D)`` evolutionary-mode space, ``J1`` as marker size.

    ``empirical`` is an optional DataFrame of real tumours (the repo ships
    ``validation/data/noble_empirical_indices.csv`` with columns ``n``, ``D``, ``J1``) to plot
    behind the trajectory for context.
    """
    ax = _ax(ax, (6.5, 5.5))
    if empirical is not None and len(empirical):
        ax.scatter(empirical["n"], empirical["D"], s=18, c="0.8", edgecolor="none",
                   label="real tumours", zorder=1)
    ok = traj.dropna(subset=["n_drivers", "D"])
    j1 = ok["J1"].fillna(0.0).to_numpy()
    ax.plot(ok["n_drivers"], ok["D"], color="0.5", lw=0.8, zorder=2)
    sc = ax.scatter(ok["n_drivers"], ok["D"], c=ok["gen"], cmap="viridis",
                    s=10 + 60 * j1, zorder=3)
    if len(ok):
        ax.scatter(ok["n_drivers"].iloc[-1], ok["D"].iloc[-1], marker="*", s=220,
                   color="crimson", zorder=4, label="endpoint")
    ax.set_xlabel("n  (mean drivers per cell)"); ax.set_ylabel("D  (clonal diversity)")
    ax.set_title("Trajectory through mode space (size = J1)")
    ax.legend(fontsize=8)
    ax.figure.colorbar(sc, ax=ax, label="generation", fraction=0.04)
    return ax


def plot_growth_phase(traj, growth, occupancy=None, axes=None):
    """``N(t)``, per-capita growth and crowding, stacked on a shared generation axis.

    The bottom panel carries the signal ``t_rK`` is actually derived from — the cell-weighted
    crowding index, with its threshold marked — not the per-capita growth in the middle panel,
    which decays under ordinary front-limited expansion and would put the transition far too early.
    """
    import matplotlib.pyplot as plt
    if axes is None:
        _, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    gen = traj["gen"].to_numpy()
    n = traj["n_cells"].to_numpy()
    axes[0].plot(gen, n, color="#4c78a8"); axes[0].set_ylabel("cancer cells")
    axes[0].set_yscale("log" if np.nanmax(n) > 100 else "linear")
    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.gradient(np.log(np.maximum(n, 1e-12))) / np.maximum(np.gradient(gen), 1e-12)
    axes[1].plot(gen, g, color="#e45756"); axes[1].axhline(0, color="0.7", lw=0.8)
    axes[1].set_ylabel("per-capita growth")
    if occupancy is not None:
        for key, col in (("crowding_index", "#54a24b"),
                         ("crowding_index_duct", "#b279a2"),
                         ("crowding_index_stroma", "#eeca3b")):
            if key in occupancy:
                axes[2].plot(gen, _resample(occupancy[key], gen.size), label=key, color=col)
        axes[2].axhline(0.75, color="0.6", ls=":", lw=1)
        axes[2].legend(fontsize=7)
    axes[2].set_ylabel("crowding index"); axes[2].set_xlabel("generation")
    if growth.get("t_rK") is not None:
        for a in axes:
            a.axvline(growth["t_rK"], color="0.3", ls="--", lw=1)
        axes[0].set_title(f"r -> K transition at generation {growth['t_rK']:.0f} "
                          f"({growth['frac_gens_in_K']:.0%} of the run in K)")
    else:
        axes[0].set_title("No r -> K transition detected (still expanding)")
    return axes


def plot_cn_landscape_over_time(traj, axes=None):
    """FGA, ploidy, LOH/nullisomy and WGD fraction vs generation."""
    import matplotlib.pyplot as plt
    if axes is None:
        _, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    a = np.ravel(axes)
    gen = traj["gen"]
    a[0].plot(gen, traj["fga"], color="#4c78a8"); a[0].set_ylabel("FGA")
    a[1].plot(gen, traj["mean_ploidy"], color="#e45756")
    a[1].fill_between(gen, traj["mean_ploidy"] - traj["ploidy_sd"],
                      traj["mean_ploidy"] + traj["ploidy_sd"], color="#e45756", alpha=0.2)
    a[1].set_ylabel("ploidy")
    a[2].plot(gen, traj["frac_segments_loh"], color="#54a24b", label="LOH")
    a[2].plot(gen, traj["frac_segments_nullisomy"], color="0.4", label="nullisomy")
    a[2].set_ylabel("segment fraction"); a[2].legend(fontsize=8); a[2].set_xlabel("generation")
    a[3].plot(gen, traj["wgd_frac"], color="#b279a2"); a[3].set_ylabel("WGD fraction")
    a[3].set_xlabel("generation")
    a[0].set_title("Copy-number landscape over time")
    return axes


def plot_segment_recurrence(summary, ax=None):
    """Per-segment gain / loss frequency at the end of the run."""
    ax = _ax(ax, (9, 4))
    gain = np.asarray(summary["gain_freq"], dtype=float)
    loss = np.asarray(summary["loss_freq"], dtype=float)
    x = np.arange(gain.size)
    ax.bar(x, gain, color="#e45756", label="gain")
    ax.bar(x, -loss, color="#4c78a8", label="loss")
    ax.axhline(0, color="0.3", lw=0.8)
    ax.set_xlabel("segment"); ax.set_ylabel("cell fraction   (loss below zero)")
    ax.set_title("Copy-number recurrence spectrum")
    ax.legend(fontsize=8)
    return ax


def plot_colonisation_curve(structure, ax=None):
    """Glands colonised over time, with the stromal-escape generation marked."""
    ax = _ax(ax, (7, 4))
    curve = (structure or {}).get("colonisation_curve")
    if not curve:
        ax.text(0.5, 0.5, "no colonisation history\n(needs trace_occupancy)",
                ha="center", va="center", transform=ax.transAxes)
        return ax
    g, k = zip(*curve)
    ax.step(g, k, where="post", color="#4c78a8")
    ax.set_ylim(0, max(k) + 1)
    if structure.get("t_escape") is not None:
        ax.axvline(structure["t_escape"], color="crimson", ls="--", lw=1)
        ax.text(structure["t_escape"], ax.get_ylim()[1], " stromal escape",
                va="top", fontsize=8, color="crimson")
    ax.set_xlabel("generation"); ax.set_ylabel("glands colonised")
    ax.set_title("Multi-focal colonisation")
    return ax


def plot_focus_divergence(structure, ax=None):
    """Within- vs between-focus copy-number divergence — the island-bottleneck signature."""
    ax = _ax(ax, (4.5, 4))
    w = (structure or {}).get("within_focus_cn_divergence", float("nan"))
    b = (structure or {}).get("between_focus_cn_divergence", float("nan"))
    ax.bar(["within focus", "between foci"], [w, b], color=["#4c78a8", "#e45756"])
    ax.set_ylabel("mean |dCN| per segment")
    ax.set_title("Focus divergence")
    return ax
