"""AnnData export: a whole simulated tumour as ONE standard object.

The engine hands its ground truth over as a dict of dataframes (``tumor.cell_data``) plus a handful
of per-gene / per-clone tables hanging off the tumour. That is a bespoke structure: every analysis
has to learn it. This module packs the lot into a single ``AnnData``, the format scanpy, squidpy,
Hotspot, Numbat and friends already read, so a simulated tumour can be written to one ``.h5ad`` and
picked up anywhere::

    adata = to_anndata(tumor)          # or to_anndata(tumor.cell_data)
    tumor.write_h5ad("tumour.h5ad")    # same object, straight to disk

Where everything lands
----------------------
``X``      per-cell expression (``cell_exp``).

``layers`` every other cells x genes matrix that the run produced — SNV VAFs (total and per
           homolog), copy number, the RNA allele fraction / BAF, the allele-resolved expression.

``obs``    one column per field of every per-cell annotation table: the genotype id, a coarse
           biological cell type (cancer / epithelial / stromal / immune / host), the clone label,
           deme, gland, compartment, whole-genome-duplication flag, microenvironment levels, and the
           clone's evolutionary rates and mutated-driver tallies.

``obsm``   ``"spatial"`` — the cell's tissue coordinates as ``(x, y)``, the convention 10x and
           squidpy use, so a plot of these cells and a plot of a Visium slide cut from the same
           tissue come out the same way up. The raw grid ``row`` / ``col`` are kept in ``obs``.
           ``"program"`` — per-cell gene-program activity, when the program layer is on.

``var``    per-gene roles (which genes are oncogenes, tumour suppressors, dispersal / resistance /
           breach / survival drivers, inherited variants), the genome segment each gene sits in, and
           the dosage / mutation-effect annotations when the program layer is on.

``varm``   ``"program_loading"`` — genes x programs loading matrix; ``"program_member"`` — the
           boolean membership map.

``uns``    program names, the clone tree (Newick plus a parent/child edge table), the clone-count
           trace, what the ``obs["clone"]`` labels mean (the ``clone_*`` entries), the grid size,
           the seed and the parameters the tumour was built with.

Which frame goes where is decided by SHAPE, not by a fixed list of names: any per-cell table whose
columns are the gene index becomes a layer, anything else becomes ``obs`` columns. Optional layers
(microenvironment, glands, metastasis, programs, allele resolution) appear only in the runs that
enable them, and they are picked up automatically.

``anndata`` is imported lazily so importing :mod:`iscc.integrations` never requires it.
"""
import numpy as np
import pandas as pd

#: The per-cell tables that are NOT plain annotation columns and get their own destination.
_SPECIAL = {"cell_crd", "cell_program"}

#: The layer set the original dict-only export attached. Kept as the default for that path so
#: existing callers get exactly what they always got.
_LEGACY_LAYERS = ("cell_snv", "cell_cnv")

#: Cache of the genotype ids that name a normal cell type rather than a cancer clone (filled on
#: first use so this module keeps importing without pulling the engine in).
_NORMAL_IDS = None


def _normal_ids():
    global _NORMAL_IDS
    if _NORMAL_IDS is None:
        from ..constants import normal_names
        _NORMAL_IDS = set(normal_names)
    return _NORMAL_IDS


def biological_types(genotype_ids):
    """Coarse cell type per cell: cancer, epithelial, stromal, immune or host.

    A normal cell's genotype id IS its type name; anything else is a cancer clone.
    """
    normals = _normal_ids()
    return np.array([g if g in normals else "cancer" for g in map(str, genotype_ids)], dtype=object)


# ------------------------------------------------------------------ uns / value sanitising
def _scalar(v):
    return isinstance(v, (str, bool, int, float, np.bool_, np.integer, np.floating))


def _clean(value):
    """Coerce a value into something h5ad can store, or return ``None`` to drop it.

    Handles the two shapes h5ad refuses outright: ``None`` (no HDF5 representation) and dicts with
    non-string keys (the program membership map is keyed by gene index).
    """
    if value is None:
        return None
    if _scalar(value):
        return value
    if isinstance(value, np.ndarray):
        return value if value.dtype != object else np.asarray(value, dtype=str)
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if not isinstance(k, str):
                return None                      # int-keyed maps are exported in another form
            cv = _clean(v)
            if cv is not None:
                out[str(k)] = cv
        return out or None
    if isinstance(value, (list, tuple)):
        if not len(value):
            return None
        if all(_scalar(v) for v in value):
            arr = np.asarray(value)
            return np.asarray(arr, dtype=str) if arr.dtype == object else arr
        return None
    return None


# ------------------------------------------------------------------ source resolution
_MISSING = object()


def _resolve(source):
    """Split the argument into ``(tumor_or_None, cell_data_dict)``."""
    if isinstance(source, dict):
        return None, source
    cell_data = getattr(source, "cell_data", _MISSING)
    if cell_data is _MISSING:
        raise TypeError(
            "to_anndata expects a tumour (GenotypeTumor) or a cell_data dict, got "
            f"{type(source).__name__}."
        )
    if not cell_data:
        maker = getattr(source, "make_cell_data", None)
        if maker is None:
            raise ValueError("this tumour has no per-cell data and cannot materialise any.")
        # Bounded by the tumour's own max_cells, so this stays memory-safe at large scale.
        cell_data = maker()
    return source, cell_data


def _frames(cell_data):
    """The per-cell tables, as ``{name: DataFrame}`` (anything else in the dict is ignored)."""
    return {k: v for k, v in cell_data.items() if isinstance(v, pd.DataFrame)}


# ------------------------------------------------------------------ obs / layers assembly
def _add_obs(obs, name, values, taken):
    """Write a column, suffixing on a name clash so nothing is silently overwritten."""
    col = name
    if col in taken:
        col = f"{name}_{taken[name]}"
    taken[name] = taken.get(name, 0) + 1
    obs[col] = values
    return col


#: Candidate ``min_cells`` values the automatic clone resolution tries, coarsest last.
_MIN_CELLS_LADDER = (5, 8, 12, 20, 32, 50, 80, 125, 200, 320, 500, 800, 1250, 2000)

#: Most clones the automatic choice hands back. Past ~a dozen the annotation stops being a readable
#: colouring (matplotlib's categorical palettes run out at 10-20) and turns into confetti.
_AUTO_MAX_CLONES = 12

#: What ``clone_min_cells`` means when it is left at its default.
AUTO = "auto"


def _clone_stats(labels, unassigned):
    """``(n_clones, n_labelled, unassigned_fraction)`` of a clone labelling."""
    lab = labels.dropna().astype(str)
    n = int(len(lab))
    if not n:
        return 0, 0, 0.0
    n_other = int((lab == unassigned).sum())
    return int(lab.nunique()) - (1 if n_other else 0), n, n_other / n


def _resolve_clones(tumor, min_cells):
    """Clone labels for the tumour's sampled cells: ``(labels, min_cells, n_clones, other_frac)``.

    An explicit ``min_cells`` is used as given. ``"auto"`` searches :data:`_MIN_CELLS_LADDER`
    instead, because no fixed count works at every scale: the fraction of cells left in the
    ``"other"`` bucket (the ancestral backbone, which carries a large share of the cells in a
    tumour where ancestral genotypes are still alive) moves *non-monotonically* with ``min_cells``,
    so neighbouring values can differ by 20 percentage points in either direction. The search keeps
    the candidate that strands the fewest cells in ``"other"`` while still giving between two and
    :data:`_AUTO_MAX_CLONES` clones — ties go to the coarser (larger) value. Clone counts fall as
    ``min_cells`` rises, so the walk stops as soon as everything has collapsed into one clone.
    """
    from .clones import clones_from_clades, UNASSIGNED

    if min_cells != AUTO:
        labels = clones_from_clades(tumor, min_cells=int(min_cells))
        n_clones, _, other = _clone_stats(labels, UNASSIGNED)
        return labels, int(min_cells), n_clones, other

    best = nearest = None
    for m in _MIN_CELLS_LADDER:
        labels = clones_from_clades(tumor, min_cells=m)
        n_clones, n_labelled, other = _clone_stats(labels, UNASSIGNED)
        candidate = (labels, m, n_clones, other)
        if n_labelled == 0:                                  # no cancer cells to label at all
            return candidate
        if 2 <= n_clones <= _AUTO_MAX_CLONES:
            if best is None or other <= best[3]:             # <= : keep the coarsest of a tie
                best = candidate
        if nearest is None or abs(n_clones - _AUTO_MAX_CLONES) < abs(nearest[2] - _AUTO_MAX_CLONES):
            nearest = candidate
        if n_clones <= 1:                                    # coarser values cannot split it again
            break
    return best if best is not None else nearest


def _clone_labels(tumor, index, min_cells):
    """``(categorical clone labels or None, uns bookkeeping)`` for the sampled cells.

    Normal (non-cancer) cells have no clone and stay missing, which is what the shared clone
    definition returns; the categorical survives an h5ad round-trip with those cells unlabelled.
    The bookkeeping records how the labels were defined, so the annotation is self-describing.
    """
    if tumor is None or min_cells is None:
        return None, {"clone_definition": "genotype",
                      "clone_definition_doc": "clone = the raw genotype id (clade labelling off)"}
    try:
        from .clones import UNASSIGNED
        labels, resolved, n_clones, other = _resolve_clones(tumor, min_cells)
    except Exception:
        return None, {"clone_definition": "genotype",
                      "clone_definition_doc": "clone = the raw genotype id (no clone tree available)"}
    labels = labels.reindex(pd.Index([str(i) for i in index]))
    if labels.isna().all():
        return None, {"clone_definition": "genotype",
                      "clone_definition_doc": "clone = the raw genotype id (no cancer cells sampled)"}
    uns = {
        "clone_definition": "clade",
        "clone_definition_doc": (
            f"clone = a clade of the true lineage tree holding at least {resolved} sampled cells; "
            f"cancer cells above every clade are labelled {UNASSIGNED!r}; normal cells are "
            "unlabelled"),
        "clone_min_cells": int(resolved),
        "clone_min_cells_mode": "auto" if min_cells == AUTO else "fixed",
        "clone_n_clones": int(n_clones),
        "clone_other_frac": float(other),
    }
    return pd.Categorical(labels.values), uns


# ------------------------------------------------------------------ gene annotation
def _var_from_tumor(tumor, genes):
    """Per-gene role / segment / dosage columns, as a dict of arrays aligned to ``genes``."""
    out = {}
    getter = getattr(tumor, "get_gene_data", None)
    if getter is not None:
        try:
            gene_data = getter()
        except Exception:
            gene_data = {}
        for key, frame in (gene_data or {}).items():
            try:
                col = pd.DataFrame(frame).reindex(genes).iloc[:, 0]
            except Exception:
                continue
            out[key] = col.values

    selection = getattr(tumor, "selection", None)
    sizes = getattr(selection, "segment_sizes", None)
    if sizes is not None and int(np.sum(sizes)) == len(genes):
        out["segment"] = np.repeat(np.arange(len(sizes)), sizes).astype(int)
        out["position_in_segment"] = np.concatenate([np.arange(s) for s in sizes]).astype(int)

    truth = getattr(tumor, "program_truth", None) or {}
    for key in ("dosage_sensitivity", "snv_class", "snv_exp_effect"):
        vals = truth.get(key)
        if vals is not None and np.ndim(vals) == 1 and len(vals) == len(genes):
            out[key] = np.asarray(vals)
    names = truth.get("snv_class_names")
    if names is not None and "snv_class" in out:
        idx = np.asarray(out["snv_class"], dtype=int)
        out["snv_class_name"] = np.asarray(names, dtype=str)[idx]
    return out


def _program_varm(tumor, n_genes):
    """``(varm, uns)`` fragments for the gene-program layer (empty when it is off)."""
    varm, uns = {}, {}
    truth = getattr(tumor, "program_truth", None) or {}
    loading = truth.get("loading")
    if loading is not None and np.asarray(loading).ndim == 2:
        loading = np.asarray(loading, dtype=float)
        if loading.shape[1] == n_genes:
            varm["program_loading"] = loading.T.copy()
    names = truth.get("program_names")
    if names is not None and len(names):
        uns["program_names"] = np.asarray(names, dtype=str)
    # anndata refuses a uns dict with integer keys, and `gene_program_map` is keyed by gene index.
    # The same information as a boolean genes x programs matrix is both storable and easier to use.
    gpm = truth.get("gene_program_map")
    n_programs = len(names) if names is not None else 0
    if gpm and n_programs:
        member = np.zeros((n_genes, n_programs), dtype=bool)
        for gene, programs in gpm.items():
            gi = int(gene)
            if 0 <= gi < n_genes:
                for p in programs:
                    if 0 <= int(p) < n_programs:
                        member[gi, int(p)] = True
        varm["program_member"] = member
    # Same story for the regulator map ({gene index: [(target, sign), ...]}): export it long-form.
    regs = truth.get("regulator_genes")
    if regs:
        edges, signs = [], []
        for reg, targets in regs.items():
            for target, sign in targets:
                edges.append((int(reg), int(target)))
                signs.append(float(sign))
        if edges:
            uns["regulator_edges"] = np.asarray(edges, dtype=int)
            uns["regulator_signs"] = np.asarray(signs, dtype=float)
    return varm, uns


# ------------------------------------------------------------------ tumour-level uns
def _tree_uns(tumor):
    out = {}
    parents = dict(getattr(tumor, "genotypes_parents", {}) or {})
    if parents:
        child = np.asarray([str(c) for c in parents], dtype=str)
        parent = np.asarray([str(parents[c]) for c in parents], dtype=str)
        out["clone_tree"] = pd.DataFrame({"child": child, "parent": parent})
    try:
        from .lineage import to_newick
        newick = to_newick(tumor)
    except Exception:
        newick = None
    if newick:
        out["clone_tree_newick"] = newick
    return out


def _trace_uns(tumor):
    """The clonal-abundance trace as ONE dataframe (``tumor.traces`` is a list of nested dicts)."""
    traces = getattr(tumor, "traces", None)
    if not traces:
        return {}
    try:
        counts = pd.DataFrame([t["genotypes_counts"] for t in traces]).fillna(0.0)
    except Exception:
        return {}
    counts.columns = [str(c) for c in counts.columns]
    counts.index = pd.Index([str(i) for i in range(len(counts))], name="step")
    return {"trace_counts": counts.astype(float)}


_CONFIG_KEYS = (
    "seed", "layout_seed", "grid_size", "met_grid_size", "n_primary_demes", "n_genes",
    "n_segments", "segment_size", "carrying_capacity", "structure_radius", "n_glands",
    "gland_radius", "update_mode", "tau", "coarsen_passengers", "max_cells", "genome_mode",
    "initial_cancer_cells", "immune_density",
)


def _config_uns(tumor):
    cfg = {}
    for key in _CONFIG_KEYS:
        val = _clean(getattr(tumor, key, None))
        if val is not None:
            cfg[key] = val
    genome = _clean(getattr(tumor, "genome_params", None))
    if genome:
        cfg["genome_params"] = genome
    out = {"grid_size": int(getattr(tumor, "grid_size", 0)), "seed": int(getattr(tumor, "seed", 0))}
    if cfg:
        out["config"] = cfg
    return out


# ------------------------------------------------------------------ the export
def to_anndata(source, X="cell_exp", layers=None, obs_extra=True, spatial="xy",
               clone_min_cells=AUTO):
    """Pack a simulated tumour (or one materialised ``cell_data`` dict) into an ``AnnData``.

    Parameters
    ----------
    source : GenotypeTumor or dict
        A grown tumour, or the per-cell tables it materialised (``tumor.cell_data``, or the dict a
        :class:`iscc.sample.Resection` returns). Given a tumour, the export also carries the
        tumour-level ground truth: gene roles, the clone tree, the program dictionary, the clonal
        trace and the run parameters. Given a tumour with nothing materialised yet, the cells are
        materialised first (bounded by the tumour's own cell cap).
    X : str
        Which per-cell matrix becomes ``.X`` (default ``"cell_exp"``).
    layers : {"all", None} or sequence of str
        Which other per-cell gene matrices become ``.layers``. ``"all"`` takes every one the run
        produced (the default for a tumour); a sequence names them explicitly. For a ``cell_data``
        dict the default stays the historical ``("cell_snv", "cell_cnv")``.
    obs_extra : bool
        Attach the per-cell annotation tables (deme, gland, compartment, microenvironment,
        evolutionary rates, ...) to ``.obs``. ``False`` keeps only the identity and coordinate
        columns.
    spatial : {"xy", "rowcol"}
        Axis order of ``obsm["spatial"]``. ``"xy"`` (default) writes ``(x, y) = (col, row)``, the
        10x / squidpy convention that Visium output already uses, so cell-level and spot-level
        objects from the same tissue plot in the same orientation. ``"rowcol"`` writes the raw grid
        order instead. ``obs["row"]`` and ``obs["col"]`` carry the raw values either way.
    clone_min_cells : {"auto"} or int or None
        Smallest clade that counts as its own clone for the ``obs["clone"]`` label (tumour path
        only). ``"auto"`` (default) picks the value that leaves the fewest cells outside a named
        clone while still giving a readable number of them — a fixed count cannot do that, because
        the share of cells stranded in the ``"other"`` bucket rises and falls with ``min_cells``
        differently in every tumour. An integer fixes it by hand; ``None`` skips the clade
        labelling and falls back to the genotype id. Either way ``uns["clone_definition"]`` and the
        ``clone_*`` entries next to it record what the labels mean and how many cells fell outside.
    """
    import anndata as ad

    if spatial not in ("xy", "rowcol"):
        raise ValueError(f"spatial must be 'xy' or 'rowcol', got {spatial!r}")

    tumor, cell_data = _resolve(source)
    frames = _frames(cell_data)
    if X not in frames:
        raise KeyError(f"{X!r} is not among the materialised per-cell tables ({sorted(frames)}).")

    exp = frames[X]
    genes = list(exp.columns)
    index = exp.index
    full = tumor is not None or layers == "all"
    if layers is None:
        layers = "all" if tumor is not None else _LEGACY_LAYERS

    # --- split the per-cell tables by SHAPE: gene-wide -> layer, otherwise -> obs -----------
    gene_frames, annot_frames = {}, {}
    for name, frame in frames.items():
        if name == X or name in _SPECIAL:
            continue
        (gene_frames if list(frame.columns) == genes else annot_frames)[name] = frame

    obs = pd.DataFrame(index=index.astype(str))
    taken = {}
    genotype = None
    ctype = frames.get("cell_type")
    if ctype is not None:
        genotype = ctype.reindex(index).iloc[:, 0].astype(str).values

    clone_uns = {}
    if full:
        if genotype is not None:
            # `cell_type` stores the GENOTYPE id, so it is surfaced under that name and the coarse
            # biological type is derived alongside it.
            _add_obs(obs, "genotype", genotype, taken)
            _add_obs(obs, "cell_type", biological_types(genotype), taken)
            clone, clone_uns = _clone_labels(tumor, index, clone_min_cells)
            _add_obs(obs, "clone", genotype if clone is None else clone, taken)
        if obs_extra:
            for name, frame in annot_frames.items():
                if name == "cell_type":
                    continue
                frame = frame.reindex(index)
                for col in frame.columns:
                    _add_obs(obs, str(col), frame[col].values, taken)
    else:
        # Historical dict export: clone / deme / coordinates (+ microenvironment) only.
        if genotype is not None:
            obs["clone"] = genotype
        deme = frames.get("cell_deme")
        if deme is not None:
            obs["deme"] = deme.reindex(index).iloc[:, 0].astype(int).values

    crd = frames.get("cell_crd")
    if crd is not None:
        crd = crd.reindex(index)
        obs["row"] = crd["row"].values
        obs["col"] = crd["col"].values
    if not full and obs_extra and "cell_microenv" in annot_frames:
        me = annot_frames["cell_microenv"].reindex(index)
        for col in me.columns:
            obs[col] = me[col].values

    adata = ad.AnnData(
        X=np.asarray(exp.values, dtype=np.float32),
        obs=obs,
        var=pd.DataFrame(index=pd.Index(genes, name="gene")),
    )

    # --- obsm ------------------------------------------------------------------------------
    if crd is not None:
        # (x, y) = (col, row): the 10x/squidpy order, matching what the Visium export writes, so a
        # cell-level plot and a spot-level plot of the same tissue are not transposed relative to
        # each other. The raw grid indices stay in obs["row"] / obs["col"].
        order = ["col", "row"] if spatial == "xy" else ["row", "col"]
        adata.obsm["spatial"] = crd[order].values.astype(float)
        adata.uns["spatial_axes"] = "xy" if spatial == "xy" else "row_col"
    program = frames.get("cell_program")
    if program is not None and full:
        program = program.reindex(index)
        adata.obsm["program"] = program.values.astype(float)
        adata.uns["program_names"] = np.asarray([str(c) for c in program.columns], dtype=str)

    # --- layers ----------------------------------------------------------------------------
    names = sorted(gene_frames) if layers == "all" else [n for n in layers if n in gene_frames]
    for name in names:
        adata.layers[name] = np.asarray(gene_frames[name].reindex(index).values, dtype=np.float32)

    # --- var / varm / uns ------------------------------------------------------------------
    adata.uns["source"] = "iscc"
    if tumor is not None:
        for key, vals in _var_from_tumor(tumor, genes).items():
            adata.var[key] = vals
        varm, uns = _program_varm(tumor, len(genes))
        for key, mat in varm.items():
            adata.varm[key] = mat
        for block in (uns, _tree_uns(tumor), _trace_uns(tumor), _config_uns(tumor)):
            for key, val in block.items():
                if key not in adata.uns or key == "program_names":
                    adata.uns[key] = val
        for key, val in clone_uns.items():
            adata.uns[key] = val
    return adata


# ------------------------------------------------------------------ the way back
#: obs columns that are export bookkeeping rather than an engine annotation table.
_DERIVED_OBS = ("genotype", "cell_type", "clone", "row", "col")


def from_anndata(adata, X="cell_exp"):
    """Rebuild a ``cell_data``-style dict from an :func:`to_anndata` export.

    Returns the per-cell tables (``cell_exp`` and every layer as cells x genes frames, plus
    ``cell_crd`` / ``cell_type`` / ``cell_program`` when the export carried them) so a written
    ``.h5ad`` can be fed straight back to the assays. Annotation columns that came from optional
    tables are returned as one ``cell_obs`` frame rather than being split back up.
    """
    genes = pd.Index([str(g) for g in adata.var_names], name="gene")
    index = pd.Index([str(c) for c in adata.obs_names])
    out = {X: pd.DataFrame(np.asarray(adata.X), index=index, columns=genes)}
    for name in adata.layers:
        out[name] = pd.DataFrame(np.asarray(adata.layers[name]), index=index, columns=genes)

    obs = adata.obs
    if {"row", "col"} <= set(obs.columns):
        out["cell_crd"] = pd.DataFrame({"row": obs["row"].values, "col": obs["col"].values},
                                       index=index).astype(int)
    elif "spatial" in adata.obsm:
        crd = np.asarray(adata.obsm["spatial"])
        row, col = (crd[:, 1], crd[:, 0]) if adata.uns.get("spatial_axes") != "row_col" \
            else (crd[:, 0], crd[:, 1])
        out["cell_crd"] = pd.DataFrame({"row": row, "col": col}, index=index).astype(int)
    for key in ("genotype", "clone"):
        if key in obs.columns:
            out["cell_type"] = pd.DataFrame({"cell_id": obs[key].astype(str).values}, index=index)
            break
    if "program" in adata.obsm:
        names = adata.uns.get("program_names")
        cols = [str(n) for n in names] if names is not None else \
            [f"program_{i}" for i in range(np.asarray(adata.obsm["program"]).shape[1])]
        out["cell_program"] = pd.DataFrame(np.asarray(adata.obsm["program"]), index=index,
                                           columns=cols)
    extra = [c for c in obs.columns if c not in _DERIVED_OBS]
    if extra:
        out["cell_obs"] = obs[extra].copy()
        out["cell_obs"].index = index
    return out
