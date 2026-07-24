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
                        cancer_color, type_cmap=None, empty_color=None):
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
    the composition (and the sampled section depth) is exact. ``empty_color`` recolours the empty
    background (empty demes + unfilled positions); ``None`` keeps the historical white. Returns
    ``(rgb, s, type_cmap)``.
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
    type_cmap = type_cmap if type_cmap is not None else cell_type_colors(traces, genotypes_parents)
    fixed_cancer = None if cancer_color is None else np.asarray(matplotlib.colors.to_rgb(cancer_color))
    fallback = np.array([0.84, 0.15, 0.16])                     # cancer red if a clone is uncoloured

    def cell_col(g):
        if g in normal_names:
            return np.asarray(type_cmap[g])[:3] if g in type_cmap else fallback
        if fixed_cancer is not None:
            return fixed_cancer
        col = type_cmap.get(g)
        return fallback if col is None else np.asarray(col)[:3]

    bg = None if empty_color is None else np.asarray(matplotlib.colors.to_rgb(empty_color))
    img = np.ones((grid_size * s, grid_size * s, 3)) if bg is None else np.tile(bg, (grid_size * s, grid_size * s, 1))
    for d, cells in deme_cells.items():
        n_sec = min(s * s, int(round(section_frac * len(cells))))
        sample = ([cells[i] for i in rng.choice(len(cells), size=n_sec, replace=False)]
                  if cells and n_sec else [])
        r, c = deme_rc[d]
        block = np.ones((s * s, 3)) if bg is None else np.tile(bg, (s * s, 1))
        if sample:
            for p, g in zip(rng.choice(s * s, size=len(sample), replace=False), sample):
                block[p] = cell_col(g)
        img[r * s:(r + 1) * s, c * s:(c + 1) * s] = block.reshape(s, s, 3)
    return img, s, type_cmap


def plot_grid(cell_data, grid_size, traces, genotypes_parents, color=None, cmap="viridis",
              ax=None, figsize=(10, 10), dpi=100, expand_demes=False, section_frac=1.0, expand_seed=0,
              cancer_color="#d62728", type_cmap=None, empty_color=None):
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
                                                    section_frac, expand_seed, cancer_color, type_cmap,
                                                    empty_color=empty_color)
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
            type_cmap = type_cmap if type_cmap is not None else cell_type_colors(traces, genotypes_parents)
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


# --- metastasis: two-compartment views sharing one clone colormap (R9) -------------------------

def _compartment_cancer_counts(traces, key, all_cols):
    """Per-generation cancer counts for ONE compartment as a (generations x all_cols) frame, zero-
    filled to the GLOBAL cancer clone set (`all_cols`) so it shares the global ancestry/colour basis.
    Normal-cell columns in the per-compartment snapshot are dropped by the reindex."""
    df = pd.DataFrame([t.get(key, {}) for t in traces]).fillna(0)
    df.columns = df.columns.astype(str)
    return df.reindex(columns=all_cols, fill_value=0.0)


def _driver_band_map(cols, full_parents, driver_map):
    """{gid -> band id}: each genotype maps to the founder of its maximal same-signature ancestor chain
    (its 'driver/functional clone'). The SAME contraction ``_collapse_by_drivers`` uses, exposed so the
    two-compartment Muller can aggregate each compartment's per-genotype counts into the SAME bands."""
    fp = {str(k): str(v) for k, v in dict(full_parents).items()}
    sig, band = driver_map, {}

    def band_of(g):
        path, x = [], g
        while x not in band:
            p = fp.get(x)
            if p is None or sig.get(p) != sig.get(x) or x in path:
                band[x] = x
                break
            path.append(x)
            x = p
        b = band[x]
        for y in path:
            band[y] = b
        return b

    return {c: band_of(c) for c in cols if driver_map.get(c) is not None}


def _aggregate_by_map(comp_df, gmap, basis_cols):
    """Sum a per-genotype count frame's columns into basis columns (bands) via ``gmap`` (gid -> band)."""
    out = pd.DataFrame(0.0, index=comp_df.index, columns=[str(c) for c in basis_cols])
    for g in comp_df.columns:
        b = str(gmap.get(g, g))
        if b in out.columns:
            out[b] = out[b].values + comp_df[g].values
    return out


def _merge_shown_map(genotype_counts, full_parents, min_freq):
    """{col -> the SHOWN clone it merges into} under the min_freq size threshold (Noble sensitivity),
    mirroring ``_merge_small_clones`` so per-compartment counts and grid colours aggregate into the
    same shown clones the Muller draws. Cols never reaching min_freq of the grown population fold into
    their nearest surviving ancestor."""
    from collections import defaultdict
    cols = [c for c in genotype_counts.columns if c != "Generation"]
    if len(cols) <= 1:
        return {c: c for c in cols}
    present = set(cols)
    fp = {str(k): str(v) for k, v in dict(full_parents).items()}
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
    order, seen = [], set()
    for r in [c for c in cols if epar[c] not in present]:
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
    total = genotype_counts[cols].to_numpy(float).sum(axis=1)
    ok = total > 0
    ref = float(total[ok].max()) if ok.any() else 1.0
    shown = {c for c in cols if epar[c] not in present
             or (ok.any() and float(np.max(subtree[c][ok])) / ref >= min_freq)}

    def nearest(c):
        x = c
        while x is not None and x not in shown:
            x = epar.get(x)
        return x

    return {c: nearest(c) for c in cols}


def _display_basis(traces, genotypes_parents, driver_map=None, min_freq=None):
    """The clone basis a Muller/grid is drawn on, plus the map from each raw genotype to its basis
    clone. Optionally collapses genotypes to FUNCTIONAL clones (``driver_map``) and/or merges clones
    below a size threshold (``min_freq``). Returns (basis_counts, basis_parents, gid->clone map,
    basis_cols) — the map lets each compartment's counts and the grid colours aggregate into the SAME
    clones, so colours stay shared across both Muller bands and both grids."""
    gcounts, gparents = _cancer_only(traces, genotypes_parents)
    all_gids = [c for c in gcounts.columns if c != "Generation"]
    counts = gcounts.copy()
    counts["Generation"] = np.arange(gcounts.shape[0])
    if driver_map:
        gmap = {str(g): str(b) for g, b in _driver_band_map(all_gids, genotypes_parents, driver_map).items()}
        counts, parents = _collapse_by_drivers(counts, genotypes_parents, driver_map)
        merge_parents = {c: parents[c].iloc[0] for c in parents.columns}
    else:
        gmap = {str(g): str(g) for g in all_gids}
        parents, merge_parents = gparents, genotypes_parents
    if min_freq:
        smap = _merge_shown_map(counts, merge_parents, min_freq)
        counts, parents = _merge_small_clones(counts, merge_parents, min_freq)
        gmap = {g: str(smap.get(b, b)) for g, b in gmap.items()}
    basis_cols = [c for c in counts.columns if c != "Generation"]
    return counts, parents, gmap, basis_cols


def functional_clone_colors(traces, genotypes_parents, driver_map=None, colormap="gnuplot", min_freq=None):
    """{gid -> rgba} where each cancer genotype takes the colour of its display clone (functional clone
    and/or size-merged), plus the normal-type colours. Matches the clone colours
    ``plot_muller_compartments`` draws for the same (driver_map, min_freq), so the two grids and both
    Muller bands share one palette."""
    basis_counts, basis_parents, gmap, _ = _display_basis(traces, genotypes_parents, driver_map, min_freq)
    pop_df, anc, color_by = _prepare(basis_counts.copy(), basis_parents)
    band_colors, band_order = _get_colormap(pop_df, anc, color_by, colormap)
    band_cmap = dict(zip([str(b) for b in band_order], band_colors))
    out = {g: band_cmap[str(gmap.get(g, g))] for g in gmap if str(gmap.get(g, g)) in band_cmap}
    for name in normal_names:
        out[name] = normal_cmap_rgba[name]
    return out


def _band_y_at(pop_df, anc, band_id, gen, smoothing_std):
    """Stacked y-centre of ``band_id``'s band at generation ``gen`` in the pymuller stackplot built
    from (pop_df, anc) — for placing a marker exactly on a clone's band. Returns (x, y) or None."""
    yt = pymuller.logic._get_y_values(pop_df, anc, smoothing_std)   # index=generation, cols=strain slots
    order = [str(c) for c in yt.columns]
    if str(band_id) not in order:
        return None
    gens = list(yt.index)
    gi = min(range(len(gens)), key=lambda i: abs(gens[i] - gen))
    heights = yt.to_numpy()[gi].astype(float)
    bottoms = np.concatenate([[0.0], np.cumsum(heights)])
    idxs = [i for i, c in enumerate(order) if c == str(band_id)]
    return gens[gi], float((bottoms[idxs[0]] + bottoms[idxs[-1] + 1]) / 2.0)


# Stage-dominant driver colouring (todo #14): a clone is coloured by the HIGHEST arc-stage driver it
# carries, so the selection cascade reads as a few categories rather than a per-clone rainbow. Order:
# proliferation (onc/TSG, in the ducts) -> invasion (breach/stromal_survival, in the stroma) ->
# met survival (in the deposit) -> chemo resistance (under treatment). GenotypeTumor._stage_colors maps
# each genotype to STAGE_PALETTE[stage]; here we only own the palette + drawing.
STAGE_NAMES = ["no driver", "proliferation (onc/TSG)", "invasion (breach/stromal)",
               "met survival", "chemo resistance"]
STAGE_PALETTE = [(0.80, 0.80, 0.80, 1.0),   # none — grey
                 (0.27, 0.45, 0.71, 1.0),   # proliferation — blue
                 (0.33, 0.66, 0.41, 1.0),   # invasion — green
                 (0.55, 0.42, 0.72, 1.0),   # met survival — purple
                 (0.79, 0.28, 0.28, 1.0)]   # chemo resistance — red


def stage_legend(present=None):
    """(label, colour) pairs for the stage palette (optionally only the stage indices in ``present``)."""
    idxs = range(len(STAGE_NAMES)) if present is None else sorted(set(present))
    return [(STAGE_NAMES[i], STAGE_PALETTE[i]) for i in idxs]


def _draw_muller_panel(ax, pop_df, anc, band_colors, normalize, smoothing_std,
                       default=(0.82, 0.82, 0.82, 1.0)):
    """Draw ONE Muller panel with EXPLICIT per-band colours, bypassing pymuller's colormap
    normalisation so a categorical palette maps exactly. Mirrors pymuller._muller_plot's layout."""
    import sys as _sys
    pop = pop_df.copy()
    if normalize:
        pop["Population"] = pop.groupby("Generation")["Population"].transform(
            lambda v: v / v.sum() if v.sum() else v)
    x = pop["Generation"].unique()
    _old = _sys.getrecursionlimit()
    try:
        _sys.setrecursionlimit(max(_old, 20000))
        yt = pymuller.logic._get_y_values(pop, anc, smoothing_std)
    finally:
        _sys.setrecursionlimit(_old)
    colors = [band_colors.get(str(c), default) for c in yt.columns.values]
    ax.stackplot(x, yt.to_numpy().T, colors=colors)
    return ax


def plot_muller_compartments(traces, genotypes_parents, axes=None, colormap="gnuplot",
                             normalize=False, smoothing_std=0.1, labels=("primary", "metastasis"),
                             mark_generations=None, driver_map=None, min_freq=None,
                             star_genotype=None, star_gen=None, clone_colors=None, color_legend=None):
    """Two stacked Muller panels — primary (top) and metastasis (bottom) — that SHARE ONE colormap, so
    a clone keeps its colour across both bands (and matches plot_grid_compartments / plot_muller). The
    met founder therefore appears in the met band with the SAME colour as its — usually minor — parent
    clone in the primary band.

    ``driver_map`` ({gid -> hashable signature}) colours by DISTINCT functional clones (Noble's demon
    convention) rather than by genotype, and ``min_freq`` (a fraction) additionally merges clones whose
    subtree never reaches that share of the grown population into their nearest ancestor. Without them
    an infinite-sites tumour spawns thousands of near-identically-coloured genotype bands and no
    selective sweep is visible; WITH the functional signature + a size threshold the DCIS→IDC breach
    sweep, the met founder, and the post-chemo resistant escape each show as a band. The collapse is
    applied to the global basis and the SAME clone map aggregates each compartment's counts, so colours
    stay shared.

    `mark_generations` is an optional list of (generation, label) drawn as vlines (seeding / resection /
    chemo start+end)."""
    gcounts, _gp = _cancer_only(traces, genotypes_parents)
    all_gids = [c for c in gcounts.columns if c != "Generation"]
    basis_counts, basis_parents, gmap, basis_cols = _display_basis(
        traces, genotypes_parents, driver_map, min_freq)
    # shared colour basis: color_by + the shared ancestry over the (collapsed) clone set
    pop_full, anc_full, color_by = _prepare(basis_counts.copy(), basis_parents)
    # the clone that seeded the met -> its display band, starred in BOTH panels at the seeding moment,
    # in the band's OWN colour (matching its Muller band + the grids) with a black edge for visibility.
    star_band = str(gmap.get(str(star_genotype))) if star_genotype is not None else None
    star_color = "gold"
    # explicit per-clone colours (e.g. stage-dominant driver): a band takes its founder gid's colour,
    # drawn via the manual panel drawer so a categorical palette maps exactly.
    band_colors = None
    if clone_colors is not None:
        band_colors = {str(b): clone_colors.get(str(b), (0.82, 0.82, 0.82, 1.0)) for b in basis_cols}
        if star_band is not None:
            star_color = band_colors.get(str(star_band), star_color)
    elif star_band is not None:
        _bcolors, _border = _get_colormap(pop_full, anc_full, color_by, colormap)
        star_color = dict(zip([str(b) for b in _border], _bcolors)).get(str(star_band), "gold")
    if axes is None:
        _, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    import sys as _sys
    for ax, key, label in zip(axes, ("primary_counts", "met_counts"), labels):
        comp = _aggregate_by_map(_compartment_cancer_counts(traces, key, all_gids), gmap, basis_cols)
        if comp.values.sum() == 0 or anc_full.shape[0] == 0 or not basis_cols:
            ax.axis("off")
            ax.text(0.5, 0.5, f"no {label} clones", ha="center", va="center", transform=ax.transAxes)
            continue
        comp = comp.copy()
        comp["Generation"] = np.arange(comp.shape[0])
        pop_df = (comp.melt(id_vars=["Generation"], value_name="Population", var_name="Identity")
                  .sort_values("Generation").reset_index(drop=True))
        if band_colors is not None:
            _draw_muller_panel(ax, pop_df, anc_full, band_colors, normalize, smoothing_std)
        else:
            _old = _sys.getrecursionlimit()
            try:
                _sys.setrecursionlimit(max(_old, 20000))
                pymuller.muller(pop_df, anc_full, color_by, ax=ax, colorbar=False, colormap=colormap,
                                normalize=normalize, background_strain=False, smoothing_std=smoothing_std)
            finally:
                _sys.setrecursionlimit(_old)
        ax.set_ylabel(f"{label}\n" + ("frequency" if normalize else "cells"))
        ax.set_xlabel("")  # clear pymuller's per-panel "Time" label; only the bottom panel is labelled
        for spec in (mark_generations or []):
            gen, lbl = spec[0], spec[1]
            ax.axvline(gen, color="k", ls="--", lw=1, alpha=0.7)
            ax.text(gen, ax.get_ylim()[1] * 0.98, str(lbl), rotation=90, va="top", ha="right", fontsize=7)
        if star_band is not None and star_gen is not None:
            xy = _band_y_at(pop_df, anc_full, star_band, star_gen, smoothing_std)
            if xy is not None:
                ax.plot(xy[0], xy[1], marker="*", markersize=20, markerfacecolor=star_color,
                        markeredgecolor="black", markeredgewidth=1.2, zorder=6, linestyle="none")
    axes[-1].set_xlabel("snapshot")
    # one legend on the top panel: the colour categories (if any) + the seeding-star marker
    from matplotlib.lines import Line2D
    handles = [mpatches.Patch(color=c, label=l) for l, c in (color_legend or [])]
    if star_band is not None and star_gen is not None:
        handles.append(Line2D([], [], marker="*", markerfacecolor=star_color, markeredgecolor="black",
                              linestyle="none", markersize=12, label="seeding clone"))
    if handles:
        axes[0].legend(handles=handles, fontsize=7, loc="upper left")
    return axes


def plot_grid_compartments(cell_data, primary_grid_size, met_grid_size, traces, genotypes_parents,
                           color=None, axes=None, figsize=(16, 7), dpi=100,
                           labels=("primary", "metastasis"), driver_map=None, min_freq=None,
                           clone_colors=None, color_legend=None, **kwargs):
    """Two spatial grids side by side — primary and metastasis — sharing ONE clone colormap, so the
    same clone is the same colour in both grids (and matches plot_muller_compartments). Splits
    cell_data by cell_compartment (0 = primary, 1 = met); the met panel uses the met-local coords and
    met_grid_size, so the two grids are never overlaid. ``clone_colors`` ({gid -> rgba}, e.g. stage-
    dominant driver) overrides the palette; else ``driver_map`` / ``min_freq`` colour by functional
    clone, else by genotype."""
    if "cell_compartment" not in cell_data:
        raise ValueError("plot_grid_compartments needs cell_compartment (grow with met_grid_size > 0)")
    shared = clone_colors if clone_colors is not None else (
        functional_clone_colors(traces, genotypes_parents, driver_map, min_freq=min_freq)
        if (driver_map or min_freq) else cell_type_colors(traces, genotypes_parents))
    comp = cell_data["cell_compartment"]["compartment"].to_numpy()
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    for ax, val, gs, label in zip(axes, (0, 1), (primary_grid_size, met_grid_size), labels):
        mask = comp == val
        sub = {k: df[mask] for k, df in cell_data.items()}
        plot_grid(sub, gs, traces, genotypes_parents, color=color, ax=ax, type_cmap=shared, **kwargs)
        ax.set_title(f"{label} ({int(mask.sum())} cells)")
    if color_legend:
        axes[0].legend(handles=[mpatches.Patch(color=c, label=l) for l, c in color_legend],
                       fontsize=7, loc="upper left")
    return axes


def plot_muller_founders(traces, genotypes_parents, founder_gids, functional_map, ax=None,
                         colormap=None, smoothing_std=0.1, mark_generations=None,
                         highlight="crimson", other="0.82"):
    """A SINGLE Muller of the PRIMARY tumour with the clone(s) that seeded the metastasis HIGHLIGHTED
    (in ``highlight``) against a grey background of the rest, so you can read off which — usually
    minor — primary population founds the met. A primary clone is a 'founder' iff it shares a
    FUNCTIONAL signature with a met seeding genotype (``founder_gids``). No size-merge, so a minor
    founder shows as a thin band (the confound's point). ``mark_generations`` draws event vlines."""
    import matplotlib
    from matplotlib.patches import Patch
    gcounts, gparents = _cancer_only(traces, genotypes_parents)
    all_gids = [c for c in gcounts.columns if c != "Generation"]
    founder_sigs = {functional_map.get(str(g)) for g in founder_gids if str(g) in functional_map}
    comp = _compartment_cancer_counts(traces, "primary_counts", all_gids).copy()
    comp["Generation"] = np.arange(comp.shape[0])
    pop_df = (comp.melt(id_vars=["Generation"], value_name="Population", var_name="Identity")
              .sort_values("Generation").reset_index(drop=True))
    _, anc, _ = _prepare(gcounts.copy(), gparents)
    ids = pop_df["Identity"].unique()
    color_by = pd.Series([1.0 if functional_map.get(str(i)) in founder_sigs else 0.0 for i in ids],
                         index=ids)
    cmap = colormap or matplotlib.colors.ListedColormap([other, highlight])
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 4.2))
    import sys as _sys
    _old = _sys.getrecursionlimit()
    try:
        _sys.setrecursionlimit(max(_old, 20000))
        pymuller.muller(pop_df, anc, color_by, ax=ax, colorbar=False, colormap=cmap,
                        normalize=False, background_strain=False, smoothing_std=smoothing_std)
    finally:
        _sys.setrecursionlimit(_old)
    ax.set_xlabel("snapshot")
    ax.set_ylabel("primary cells")
    for spec in (mark_generations or []):
        gen, lbl = spec[0], spec[1]
        ax.axvline(gen, color="k", ls="--", lw=1, alpha=0.7)
        ax.text(gen, ax.get_ylim()[1] * 0.98, str(lbl), rotation=90, va="top", ha="right", fontsize=7)
    ax.legend(handles=[Patch(color=highlight, label="seeds the metastasis"),
                       Patch(color=other, label="other primary clones")], fontsize=8, loc="upper left")
    return ax


def plot_clone_grid_series(tumor, panels, driver_map=None, min_freq=None, ncols=4, figsize=None,
                           clone_colors=None, color_legend=None):
    """A SERIES of spatial grids from per-deme snapshots captured during a run, each coloured by its
    DOMINANT CLONE using the SAME clone colormap as the Muller plots — so a clone keeps one colour
    across every panel and both compartments.

    ``tumor.arc_snapshots`` must hold ``{label: [per-deme {gid: count}]}`` (see
    ``base_sim.grow_metastasis_tumor``). ``panels`` is a list of ``(label, compartment, title)`` with
    compartment in {"primary", "met"}. A deme is coloured by its dominant CANCER clone when any cancer
    is present (so a minority deposit stays visible), else by its dominant resident type.
    ``clone_colors`` ({gid -> rgba}, e.g. stage-dominant driver) overrides the palette."""
    cmap = clone_colors if clone_colors is not None else (
        functional_clone_colors(tumor.traces, tumor.genotypes_parents, driver_map, min_freq=min_freq)
        if (driver_map or min_freq) else cell_type_colors(tumor.traces, tumor.genotypes_parents))
    snaps = getattr(tumor, "arc_snapshots", {}) or {}
    n = len(panels)
    ncols = min(ncols, n) or 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize or (3.1 * ncols, 3.7 * nrows))
    axes = np.atleast_1d(axes).ravel()
    fig.subplots_adjust(hspace=0.35)   # room for the 2-line per-panel titles
    if color_legend:
        fig.legend(handles=[mpatches.Patch(color=c, label=l) for l, c in color_legend],
                   fontsize=7, loc="lower center", ncol=len(color_legend))
    for ax, (label, comp, title) in zip(axes, panels):
        deme_counts = snaps.get(label)
        if deme_counts is None:
            ax.axis("off")
            ax.set_title(f"{title}\n(no snapshot)", fontsize=8)
            continue
        if comp == "primary":
            lo, hi, G = 0, tumor.n_primary_demes, tumor.grid_size
        else:
            lo, hi, G = tumor.n_primary_demes, len(deme_counts), tumor.met_grid_size
        img = np.ones((G, G, 4))
        n_cancer = 0
        for i in range(lo, min(hi, len(deme_counts))):
            d = deme_counts[i]
            if not d:
                continue
            r, c = tumor.deme_coords[i]
            cancer = {g: v for g, v in d.items() if g not in normal_names}
            pick = max(cancer, key=cancer.get) if cancer else max(d, key=d.get)
            n_cancer += sum(cancer.values())
            col = cmap.get(pick)
            if col is not None:
                img[r, c] = col
        ax.imshow(img, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{title}\n{n_cancer} cancer cells", fontsize=8)
    for ax in axes[n:]:
        ax.axis("off")
    return axes


def plot_clone_tree(traces, genotypes_parents, clone_colors=None, driver_map=None, min_freq=0.02,
                    ax=None, color_legend=None, title=None, size_scale=1.0):
    """Ground-truth clone TREE: the true ``genotypes_parents`` genealogy collapsed to the display clones
    (functional ``driver_map`` and/or ``min_freq`` size threshold). Each node is a clone placed at its
    TRUE mutational depth from the founder (x), sized by its PEAK population, and coloured by
    ``clone_colors`` (e.g. the stage-dominant driver) — so the lineage relationships are read directly
    from the truth rather than inferred from the Muller. Elbow edges join a clone to its parent clone."""
    from collections import defaultdict
    basis_counts, basis_parents, _gmap, basis_cols = _display_basis(
        traces, genotypes_parents, driver_map, min_freq)
    present = [str(c) for c in basis_cols]
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))
    if not present:
        ax.axis("off"); ax.text(0.5, 0.5, "no clones", ha="center", va="center", transform=ax.transAxes)
        return ax
    pmap = ({str(c): str(basis_parents.iloc[0][c]) for c in basis_parents.columns}
            if basis_parents.shape[1] else {})
    presset = set(present)
    children, roots = defaultdict(list), []
    for c in present:
        p = pmap.get(c)
        (children[p].append(c) if (p in presset and p != c) else roots.append(c))
    # TRUE mutational depth of each clone from the founder, over the FULL genealogy
    fp = {str(k): str(v) for k, v in dict(genotypes_parents).items()}

    def true_depth(g):
        d, cur, seen = 0, str(g), set()
        while cur in fp and fp[cur] != cur and cur not in seen:
            seen.add(cur); cur = fp[cur]; d += 1
        return d

    xpos = {c: true_depth(c) for c in present}
    peak = {c: float(basis_counts[c].max()) if c in basis_counts else 0.0 for c in present}
    pmax = max(peak.values()) or 1.0
    ypos, leaf = {}, [0.0]

    def place(node):
        ch = sorted(children.get(node, []), key=lambda z: xpos.get(z, 0))
        if not ch:
            y = leaf[0]; leaf[0] += 1.0
        else:
            y = float(np.mean([place(c) for c in ch]))
        ypos[node] = y
        return y

    for r in sorted(roots, key=lambda z: xpos.get(z, 0)):
        place(r)
    ax.figure.set_size_inches(9, max(3.0, 0.32 * max(leaf[0], 1)))
    cc, default = (clone_colors or {}), (0.6, 0.6, 0.6, 1.0)
    for c in present:                                   # elbow edges: parent clone -> child clone
        p = pmap.get(c)
        if p in presset and p in ypos:
            ax.plot([xpos[p], xpos[p], xpos[c]], [ypos[p], ypos[c], ypos[c]], color="0.7", lw=0.8, zorder=1)
    for c in present:
        s = (30 + 320 * np.sqrt(peak.get(c, 0) / pmax)) * size_scale
        ax.scatter([xpos[c]], [ypos[c]], s=s, color=cc.get(c, default), edgecolor="black",
                   linewidth=0.5, zorder=2)
    ax.set_yticks([]); ax.set_xlabel("mutational distance from founder")
    ax.set_ylim(leaf[0], -1)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    if title:
        ax.set_title(title)
    if color_legend:
        ax.legend(handles=[mpatches.Patch(color=c, label=l) for l, c in color_legend],
                  fontsize=7, loc="lower right")
    return ax
