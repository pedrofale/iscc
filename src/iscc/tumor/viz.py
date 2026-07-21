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
    # One colour per identity present in the population. (When every non-root identity has a
    # single ancestry edge this equals anc_df.shape[0] + 1, the former hard-coded length; using
    # the identity count directly keeps it correct when roots/dangling edges differ -- e.g. under
    # tau-leaping, where a generation elapses between snapshots.)
    ids = pop_df["Identity"].unique()
    color_by = pd.Series(np.arange(len(ids)), index=ids)
    return pop_df, anc_df, color_by


def _get_colormap(pop_df, anc_df, color_by, colormap):
    y_table = pymuller.logic._get_y_values(pop_df, anc_df, 10)
    final_order = y_table.columns.values
    cmap = plt.get_cmap(colormap)
    ordered_colors = color_by.copy().loc[final_order]
    norm = matplotlib.colors.Normalize(vmin=np.min(ordered_colors), vmax=np.max(ordered_colors))
    return cmap(norm(ordered_colors.values)), final_order


def _merge_small_clones(genotype_counts, full_parents, min_freq):
    """Collapse clones that never reach ``min_freq`` of the GROWN cancer population into their nearest
    surviving ancestor — the legibility rule Noble et al. (demon) use for Muller plots: a subclone
    below a sensitivity threshold is shown as part of its parent. Without it an infinite-sites tumour
    spawns tens of thousands of genotype "clones" (one per mutation event) and the layout is both
    unreadably fine and very slow.

    ``genotype_counts`` is timepoints x genotype cell counts (cancer only). ``full_parents`` is the
    COMPLETE ``genotypes_parents`` map (child->parent over ALL genotypes ever, including the transient
    intermediates that tau-leaping never snapshots) — needed because the snapshotted clones are a
    fragmented forest; we reconnect each present clone to its nearest PRESENT ancestor through the full
    chain to recover the true induced genealogy, then threshold. Roots (no present ancestor) are the
    true founders (~1). Returns merged (counts, one-row parents)."""
    from collections import defaultdict
    cols = [c for c in genotype_counts.columns if c != "Generation"]
    if len(cols) <= 1:
        return genotype_counts, pd.DataFrame(index=[0])
    present = set(cols)
    fp = {str(k): str(v) for k, v in dict(full_parents).items()}

    # effective parent = nearest present ancestor via the full chain (memoised, ~linear over the map)
    eff = {}
    for c in cols:
        path, x = [], fp.get(c)
        while x is not None and x not in present and x not in eff:
            path.append(x); x = fp.get(x)
        res = eff.get(x, x) if (x is not None and x not in present) else (x if x in present else None)
        for p in path:
            eff[p] = res
        eff[c] = res
    epar = {c: eff.get(c) for c in cols}
    children = defaultdict(list)
    for c in cols:
        if epar[c] in present:
            children[epar[c]].append(c)
    roots = [c for c in cols if epar[c] not in present]

    order, seen = [], set()
    for r in roots:
        stack = [(r, False)]
        while stack:
            x, done = stack.pop()
            if done:
                order.append(x); continue
            if x in seen:
                continue
            seen.add(x); stack.append((x, True))
            for ch in children.get(x, []):
                stack.append((ch, False))
    order += [c for c in cols if c not in seen]

    subtree = {c: genotype_counts[c].to_numpy(float).copy() for c in cols}
    for c in order:
        for ch in children.get(c, []):
            subtree[c] += subtree[ch]
    # threshold on the fraction of the GROWN population (~= final): early on the whole tumour is a
    # handful of cells, so a per-timestep frequency would wave through every transient founder.
    total = genotype_counts[cols].to_numpy(float).sum(axis=1)
    ok = total > 0
    ref = float(total[ok].max()) if ok.any() else 1.0
    def frac(arr):
        return float(np.max(arr[ok])) / ref if ok.any() else 0.0
    shown = {c for c in cols if epar[c] not in present or frac(subtree[c]) >= min_freq}

    def nearest(c):
        x = c
        while x not in shown and x is not None:
            x = epar.get(x)
        return x

    merged = {s: genotype_counts[s].to_numpy(float).copy() for s in shown}
    for c in cols:
        if c not in shown:
            a = nearest(c)
            if a in merged:
                merged[a] += genotype_counts[c].to_numpy(float)
    new_counts = pd.DataFrame(merged, index=genotype_counts.index)
    kept_edges = {c: epar[c] for c in shown if epar.get(c) in shown}
    new_parents = pd.DataFrame(kept_edges, index=[0]) if kept_edges else pd.DataFrame(index=[0])
    return new_counts, new_parents


def _collapse_by_drivers(genotype_counts, full_parents, driver_map):
    """Collapse genotype clones into DRIVER clones — Noble's Muller convention, "colours represent
    clones with distinct combinations of driver mutations". A driver clone is a maximal chain of
    genotypes on the lineage tree sharing one driver signature (``driver_map[gid]``, the set of mutated
    driver genes); a new clone starts wherever the signature changes from parent to child. This is a
    CONTRACTION of the (acyclic) genotype tree, so the driver clones are guaranteed to form a tree even
    though signatures are NOT monotone (a CNA can delete a driver-SNV copy, shrinking the signature) —
    contracting by global signature equality instead would create cycles. Counts are summed so
    frequencies are preserved. ``driver_map`` must cover ancestors too. Returns (counts, one-row
    parents); the clone identities are the band-founding genotype ids (unique, so homoplasic signatures
    stay distinct)."""
    import collections
    cols = [c for c in genotype_counts.columns
            if c != "Generation" and driver_map.get(c) is not None]
    if not cols:
        return genotype_counts, pd.DataFrame(index=[0])
    fp = {str(k): str(v) for k, v in dict(full_parents).items()}
    sig = driver_map

    # band(g): the genotype at which g's maximal same-signature ancestor chain begins (its "driver
    # clone" id). Memoised, iterative (the tree is deep). A guard breaks any accidental parent cycle.
    band = {}
    def band_of(g):
        path, x = [], g
        while x not in band:
            p = fp.get(x)
            if p is None or sig.get(p) != sig.get(x) or x in path:
                band[x] = x; break
            path.append(x); x = p
        b = band[x]
        for y in path:
            band[y] = b
        return b

    by_band = collections.defaultdict(list)
    for c in cols:
        by_band[band_of(c)].append(c)
    dcounts = {b: genotype_counts[members].to_numpy(float).sum(axis=1)
               for b, members in by_band.items()}
    driver_counts = pd.DataFrame(dcounts, index=genotype_counts.index)

    edges = {}
    for b in by_band:                       # parent = nearest present band above the band founder
        x = fp.get(b)
        while x is not None:
            xb = band_of(x)
            if xb != b and xb in by_band:
                edges[b] = xb; break
            x = fp.get(x)
    driver_parents = pd.DataFrame(edges, index=[0]) if edges else pd.DataFrame(index=[0])
    return driver_counts, driver_parents


def _cancer_only(traces, genotypes_parents):
    genotype_counts = pd.DataFrame([t["genotypes_counts"] for t in traces]).fillna(0)
    genotype_counts.columns = genotype_counts.columns.astype(str)
    genotype_parents = pd.DataFrame(genotypes_parents, index=[0])
    drop = list(set(normal_names).intersection(set(genotype_counts.columns)))
    genotype_counts = genotype_counts.drop(columns=drop)
    genotype_parents = genotype_parents.drop(columns=[c for c in drop if c in genotype_parents.columns])
    # Keep only ancestry edges whose child AND parent both appear in the snapshotted counts.
    # Clones created and lost within a single snapshot interval (common under tau-leaping, where a
    # whole generation elapses between snapshots) get a parent edge but never enter `traces`;
    # dropping those dangling edges keeps the population and ancestry frames consistent so the
    # Muller layout stays well-defined. For the exact engine (snapshot every event) there are none,
    # so this is a no-op there.
    present = set(genotype_counts.columns)
    keep = [c for c in genotype_parents.columns
            if c in present and str(genotype_parents[c].iloc[0]) in present]
    genotype_parents = genotype_parents[keep]
    return genotype_counts, genotype_parents


def plot_muller(traces, genotypes_parents, ax=None, colormap="gnuplot", normalize=True,
                smoothing_std=0.1, show_axes=True, min_freq=None, driver_map=None):
    """Muller plot of clonal dynamics.

    ``driver_map`` ({genotype_id -> hashable driver signature}, supplied by the engine) colours by
    DISTINCT DRIVER-MUTATION COMBINATIONS rather than by genotype — Noble's demon convention — so
    passenger-only diversity collapses away (tens of thousands of genotype clones become a handful of
    driver clones). ``min_freq`` (a fraction) additionally merges clones whose subtree never reaches
    that share of the grown cancer population into their nearest ancestor (Noble-style sensitivity
    threshold). The two compose: driver-collapse first, then the size threshold. Both default off, in
    which case every genotype is a clone (the historical behaviour)."""
    genotype_counts, genotype_parents = _cancer_only(traces, genotypes_parents)
    # FULL parent map used to reconnect ancestry; replaced by the driver-tree edges after a collapse.
    merge_parents = genotypes_parents
    if driver_map:
        genotype_counts, genotype_parents = _collapse_by_drivers(
            genotype_counts, genotypes_parents, driver_map)
        merge_parents = {c: genotype_parents[c].iloc[0] for c in genotype_parents.columns}
    if min_freq:
        genotype_counts, genotype_parents = _merge_small_clones(
            genotype_counts, merge_parents, min_freq)
    if ax is None:
        _, ax = plt.subplots()
    # pymuller needs at least a couple of clones with an ancestry edge; a tumor that went
    # extinct or never diversified has none, so draw an informative empty panel instead.
    if genotype_counts.shape[1] == 0 or genotype_parents.shape[1] == 0 or genotype_counts.values.sum() == 0:
        ax.axis("off")
        ax.text(0.5, 0.5, "no surviving clones", ha="center", va="center", transform=ax.transAxes)
        return ax
    pop_df, anc_df, color_by = _prepare(genotype_counts, genotype_parents)
    # pymuller orders strains by recursing to the depth of the clone tree; a deep tree (many nested
    # clones) can exceed the default limit, so bump it for the call and restore afterwards.
    import sys as _sys
    _old_limit = _sys.getrecursionlimit()
    try:
        _sys.setrecursionlimit(max(_old_limit, 20000))
        pymuller.muller(pop_df, anc_df, color_by, ax=ax, colorbar=False, colormap=colormap,
                        normalize=normalize, background_strain=False, smoothing_std=smoothing_std)
    finally:
        _sys.setrecursionlimit(_old_limit)
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


def _expanded_cell_grid(cell_data, grid_size, traces, genotypes_parents, section_frac, seed,
                        cancer_color):
    """Cell-resolution image of the grid: each deme becomes an ``s×s`` block of its INDIVIDUAL cells.

    Normal cells take their type colour; cancer cells take ``cancer_color`` (a single colour — the
    clean tissue view, cancer-vs-normal) or, when ``cancer_color`` is None, their per-clone Muller
    colour. Under infinite-sites most divisions spawn a new (passenger-differentiated) genotype, so
    per-clone is usually a noisy rainbow; the single colour reads as tissue structure.

    A 2D deme stands for a 3D column of up to K cells (DESIGN_ductal_field.md §3), so a flat section
    should show only a SLICE of that depth, not the whole column: ``section_frac`` (≈
    section_thickness / column_depth) is the fraction of each deme's cells sampled UNIFORMLY at random
    into the section (1.0 = the whole column). Cells are scattered within the deme's block — the count
    engine tracks per-deme COUNTS, not sub-deme positions, so the intra-deme layout is cosmetic while
    the composition (and the sampled section depth) is exact. Returns ``(rgb, s, type_cmap)``.
    """
    from collections import defaultdict
    rng = np.random.default_rng(seed)
    demes = cell_data["cell_deme"]["deme_id"].values
    crd = cell_data["cell_crd"].values
    ids = cell_data["cell_type"]["cell_id"].values
    deme_cells, deme_rc = defaultdict(list), {}
    for i in range(len(ids)):
        d = int(demes[i])
        deme_cells[d].append(ids[i])
        deme_rc[d] = (int(crd[i, 0]), int(crd[i, 1]))
    max_sec = max((max(1, int(round(section_frac * len(v)))) for v in deme_cells.values()), default=1)
    s = max(1, int(np.ceil(np.sqrt(max_sec))))
    type_cmap = cell_type_colors(traces, genotypes_parents)
    fixed_cancer = None if cancer_color is None else np.asarray(matplotlib.colors.to_rgb(cancer_color))
    fallback = np.array([0.84, 0.15, 0.16])                     # cancer red if a clone is uncoloured

    def cell_col(g):
        if g in normal_names:
            return np.asarray(type_cmap[g])[:3] if g in type_cmap else fallback
        if fixed_cancer is not None:
            return fixed_cancer
        col = type_cmap.get(g)
        return fallback if col is None else np.asarray(col)[:3]

    img = np.ones((grid_size * s, grid_size * s, 3))
    for d, cells in deme_cells.items():
        n_sec = min(s * s, int(round(section_frac * len(cells))))
        sample = ([cells[i] for i in rng.choice(len(cells), size=n_sec, replace=False)]
                  if cells and n_sec else [])
        r, c = deme_rc[d]
        block = np.ones((s * s, 3))
        if sample:
            for p, g in zip(rng.choice(s * s, size=len(sample), replace=False), sample):
                block[p] = cell_col(g)
        img[r * s:(r + 1) * s, c * s:(c + 1) * s] = block.reshape(s, s, 3)
    return img, s, type_cmap


def plot_grid(cell_data, grid_size, traces, genotypes_parents, color=None, cmap="viridis",
              ax=None, figsize=(10, 10), dpi=100, expand_demes=False, section_frac=1.0, expand_seed=0,
              cancer_color="#d62728"):
    if color is None:
        color = ["cell_type"]
    if ax is None:
        _, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        plt.sca(ax)

    for color_key in color:
        if expand_demes and color_key == "cell_type":
            # cell-resolution (deme-expanded) view — each deme is a block of its individual cells,
            # section-sampled to `section_frac` of the deme's depth. See _expanded_cell_grid.
            img, s, type_cmap = _expanded_cell_grid(cell_data, grid_size, traces, genotypes_parents,
                                                    section_frac, expand_seed, cancer_color)
            ax.imshow(img, interpolation="nearest")
            present = set(cell_data["cell_type"]["cell_id"].values)
            legend_patches = [mpatches.Patch(color=type_cmap[n], label=n)
                              for n in normal_names if n in present and n in type_cmap]
            n_clones = sum(1 for g in present if g not in normal_names)
            if n_clones:
                lbl = "cancer" if cancer_color is not None else f"cancer ({n_clones} clones, by colour)"
                legend_patches.append(mpatches.Patch(
                    color=cancer_color if cancer_color is not None else (0.84, 0.15, 0.16), label=lbl))
            ax.legend(handles=legend_patches, loc="upper right", fontsize=7, framealpha=0.8)
            ax.set_title("cell_type (cells)")
            ax.set_xticks([]); ax.set_yticks([])
            continue
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
        elif color_key == "cancer_frac":
            # per-deme fraction of cells that are cancer (minority cancer is invisible under the
            # mode-based cell_type view, so this exposes early multi-focal / intraductal spread).
            base = cell_data["cell_deme"].join(cell_data["cell_crd"])
            ids = cell_data["cell_type"]["cell_id"].values
            base["val"] = np.array([g not in normal_names for g in ids], dtype=float)
            deme_data = base.groupby(["deme_id"]).mean(numeric_only=True)
            grid = np.zeros((grid_size, grid_size), dtype=float)
            grid[deme_data["row"].astype(int), deme_data["col"].astype(int)] = deme_data["val"]
            im = ax.imshow(grid, cmap=cmap, vmin=0.0, vmax=1.0)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(color_key)
        else:
            base = cell_data["cell_deme"].join(cell_data["cell_crd"])
            val = None
            for prefix, key in [("snv_", "cell_snv"), ("cnv_", "cell_cnv"), ("exp_", "cell_exp")]:
                if color_key.startswith(prefix):
                    val = cell_data[key][color_key.split("_", 1)[1]]
            if val is None:
                # search every per-cell frame, including the ductal-field (cell_gland) and F8
                # (cell_microenv) labels, so `color=["gland_id"]` / `["hypoxia_level"]` just work.
                for key in ["cell_evo", "cell_exp", "cell_snv", "cell_gland", "cell_microenv",
                            "cell_program"]:
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
