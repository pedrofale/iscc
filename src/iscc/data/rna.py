"""Single-cell RNA-seq assay with a protocol-aware batch-effect model (DESIGN_features §B).

Generative model for cell *c*, gene *g*, batch *b* (DESIGN_features §B.2), starting from
the tumour's ground-truth expression `lambda_cg` (= `cell_data['cell_exp']`):

    beta_gb ~ LogNormal(0, sigma_batch^2)        per-gene batch factor, shared across batch b
    ell_c   ~ LogNormal(mu_lib,b, sigma_lib^2)   per-cell library size (depth)
    comp_cg = normalize_g( beta_gb * lambda_cg ) per-cell composition (batch reshapes it)
    y_cg    ~ NB(ell_c * comp_cg, phi_b)         counts  (pluggable; see batch.COUNT_MODELS)
    + ambient soup counts, + doublets, + optional dropout

Everything above (biology -> library -> batch) is shared; only the final count draw is
swapped via `count_model` (default "nb"; "dm" Dirichlet-Multinomial is the implemented
compositional alternative, the Visium assay's default — see `batch.COUNT_MODELS`).

Protocol presets set which technical components dominate:
  * 10x        — droplet / UMI / 3': prominent ambient RNA + doublets, moderate depth.
  * Smart-seq3 — plate / 5' UMI + full-length: higher sensitivity / lower dropout than 10x,
                 low amplification noise (low dispersion), and real plate structure — cells
                 are index-sorted one per well onto `n_wells`-well plates, and depth is
                 plate-nested (a per-plate offset x a per-well term).

Two `scRNA` instances with the same hyper-parameters and different seeds are two batches:
same biology, different technical signature (§B.3). `run_scrna_batches` builds a labelled
multi-batch benchmark (shared / balanced-split / confounded designs).
"""
from .assay import Assay
from .batch import Batch, RNABatchHyperParams, COUNT_MODELS, assign_plates, assign_plates_from_map

import os

import numpy as np
import pandas as pd


# Protocol-typical hyper-parameter presets (DESIGN_features §B.2, §D).
PROTOCOL_PRESETS = {
    # Droplet protocol: no plates, so n_wells = 0 switches the whole plate layer off.
    "10x": dict(
        sigma_batch=0.10, mu_lib=4000.0, sigma_lib=0.35, dispersion=0.30,
        ambient_frac=0.05, doublet_rate=0.05, dropout_mid=1.5, dropout_shape=1.0,
        well_sigma=0.0, n_wells=0, n_plates=0, plate_sigma=0.0,
        depth_batch_sigma=0.05, kappa=50.0,
    ),
    # 5' UMIs suppress amplification noise (low dispersion); plate prep -> low ambient /
    # few doublets, higher sensitivity (deeper) / lower dropout. Cells are index-sorted one
    # per well onto 384-well plates (n_plates=0 -> as many plates as the cells need).
    # plate_sigma defaults to 0 (plates are bookkeeping only) so the preset reproduces the
    # depths this protocol produced before plates existed; set it to ~0.1-0.2 to switch on
    # the shared per-plate depth offset.
    "smartseq3": dict(
        sigma_batch=0.10, mu_lib=8000.0, sigma_lib=0.45, dispersion=0.15,
        ambient_frac=0.005, doublet_rate=0.01, dropout_mid=0.0, dropout_shape=1.0,
        well_sigma=0.15, n_wells=384, n_plates=0, plate_sigma=0.0,
        depth_batch_sigma=0.05, kappa=50.0,
    ),
}
# Hyper-parameters that count things (plates / wells) rather than measuring a magnitude.
_INT_PARAMS = ("n_wells", "n_plates")
# Accept common spellings.
_PROTOCOL_ALIASES = {
    "10x": "10x", "10X": "10x", "tenx": "10x", "droplet": "10x",
    "smartseq3": "smartseq3", "smart-seq3": "smartseq3", "smart_seq3": "smartseq3",
    "ss3": "smartseq3", "plate": "smartseq3",
}


def resolve_protocol(protocol):
    key = _PROTOCOL_ALIASES.get(str(protocol).lower().replace(" ", ""))
    if key is None:
        raise ValueError(
            f"unknown protocol {protocol!r}; choose from {sorted(set(_PROTOCOL_ALIASES))}"
        )
    return key


class scRNA(Assay):
    """scRNA-seq assay: applies a protocol-keyed `Batch` to a sample's `cell_exp`.

    The ``protocol`` preset sets typical magnitudes for every technical knob below; any of
    them can be overridden explicitly (``None`` keeps the preset value). Legacy aliases are
    accepted: ``n_reads`` maps onto ``mu_lib`` and ``lib_size_sigma`` onto ``sigma_lib``.

    Parameters
    ----------
    protocol : str, default "10x"
        Hyper-parameter preset selecting which technical components dominate: ``"10x"``
        (droplet / UMI / 3': prominent ambient RNA + doublets, moderate depth) or
        ``"smartseq3"`` (plate / 5' UMI + full-length: higher sensitivity, lower dropout and
        amplification noise, and real plate/well structure). Common spellings are accepted
        (``"10X"``, ``"tenx"``, ``"droplet"``; ``"smart-seq3"``, ``"ss3"``, ``"plate"``).
        Sets the defaults of every hyper-parameter below.
    n_cells : int, default 100
        Number of cells to sample and assay, capped at the number available. Ignored when
        ``cell_subset`` is passed to ``run``.
    count_model : str, default "nb"
        Final count-emission model: ``"nb"`` negative-binomial (independent per-gene
        overdispersion, what ``estimate_rna`` fits) or ``"dm"`` Dirichlet-multinomial (a fixed
        library total with gene-gene competition for reads).
    batch_label : str, optional
        Label recorded for this batch/realization; defaults to ``f"batch{seed}"``.
    n_reads : float, optional
        Legacy alias for ``mu_lib`` (mean library size).
    lib_size_sigma : float, optional
        Legacy alias for ``sigma_lib`` (per-cell library-size LogNormal sd).
    dispersion : float, optional
        Negative-binomial overdispersion phi (``var = mu + phi*mu^2``); used only when
        ``count_model="nb"`` (phi <= 0 collapses to Poisson). Default None -> preset
        (10x: 0.30, smartseq3: 0.15).
    sigma_batch : float, optional
        Per-gene batch-factor LogNormal sd; ``beta_gb ~ LogNormal(0, sigma_batch^2)`` is
        shared across cells and reshapes the per-gene composition (Splatter ``batch.facScale``).
        Larger -> stronger batch effect. Default None -> preset (0.10).
    mu_lib : float, optional
        Mean library size (UMIs / cell) = sequencing depth; per-cell library
        ``ell_c ~ LogNormal(log mu_lib_b, sigma_lib^2)``. Default None -> preset
        (10x: 4000, smartseq3: 8000).
    sigma_lib : float, optional
        Per-cell library-size LogNormal sd (cell-to-cell depth variation). Default None ->
        preset (10x: 0.35, smartseq3: 0.45).
    ambient_frac : float, optional
        Fraction of each cell's library drawn as ambient "soup" — Poisson counts from the
        pooled expression profile added to every cell. Default None -> preset
        (10x: 0.05, smartseq3: 0.005).
    doublet_rate : float, optional
        Fraction of barcodes merged with a random partner cell (two cells combined into one).
        Default None -> preset (10x: 0.05, smartseq3: 0.01).
    dropout_mid : float, optional
        Logistic dropout midpoint in expected-count space (Splatter-style zero-inflation keyed
        to mean count); 0 disables dropout. Default None -> preset (10x: 1.5, smartseq3: 0.0,
        i.e. disabled).
    dropout_shape : float, optional
        Logistic dropout steepness (active only when ``dropout_mid > 0``). Default None ->
        preset (1.0).
    well_sigma : float, optional
        Per-well library-size LogNormal sd. A plate protocol sorts one cell per well, so this
        is both the per-well and the per-cell depth term; 0 disables it. Default None ->
        preset (10x: 0.0, smartseq3: 0.15).
    n_wells : int, optional
        Wells per plate — 384 for a standard Smart-seq3 plate (96 / 1536 also work; the
        row-by-column grid follows the standard 2:3 plate shape). 0 means the protocol has no
        plates, which switches the plate layer off entirely. Default None -> preset
        (10x: 0, smartseq3: 384).
    n_plates : int, optional
        How many plates the assayed cells are spread over; the cells are split as evenly as
        possible across them. 0 (the preset) means "as many plates as the cells need". Since
        one cell occupies one well, at most ``n_plates * n_wells`` cells fit: asking for more
        GROWS the plate count to what is needed and warns. Default None -> preset (0).
    plate_sigma : float, optional
        Per-plate library-size LogNormal sd: one offset drawn per plate and shared by every
        cell on it, so cells prepared on the same plate have correlated depth (plate-nested
        depth = plate effect x well effect x per-cell library). 0 (the preset) leaves plates
        as pure bookkeeping — ids in ``obs``, no effect on the counts; ~0.1-0.2 is a
        realistic plate-to-plate spread. Default None -> preset (0.0).
    depth_batch_sigma : float, optional
        Per-batch depth-shift LogNormal sd, so different seeds (batches) differ in realized
        depth. Default None -> preset (0.05).
    kappa : float, optional
        Dirichlet concentration for the ``"dm"`` count model only: large kappa -> proportions
        ~ composition (multinomial/Poisson-like), small kappa -> lumpy, over-dispersed
        proportions. Default None -> preset (50.0).
    seed : int, default 42
        RNG seed. Fixes both the technical signature of the batch (per-gene factor, depth,
        ambient soup) and the sampled-cell selection; two instances with the same
        hyper-parameters and different seeds are two batches (same biology, different
        technical noise).
    plate_layout : dict or pandas.Series, optional
        Which plate (and well) each assayed cell is sorted into, replacing the default even,
        sequential split over ``n_plates``. Keyed by cell id, covering every assayed cell (it
        may hold more). Each value is either a ``(plate, well)`` pair, the well a 0-based
        index in reading order (A01 = 0, A02 = 1, ...) or a name such as ``"A01"``, or a plate
        label alone, in which case each plate's wells are filled in reading order in the order
        the cells are assayed. A plate takes at most ``n_wells`` cells, one per well; a layout
        that does not fit raises rather than growing the plate count. ``obs["plate"]`` names
        the plates by their labels (an integer label ``n`` becomes ``"P<n>"``). Use it to
        model how the cells were actually sorted, e.g. a design that confounds plate with
        region. Needs a plate protocol (``n_wells > 0``), and ``n_plates`` must then be 0 or
        the number of plates the map names. Default None -> the sequential layout.
    """

    def __init__(self, protocol="10x", n_cells=100, count_model="nb", batch_label=None,
                 n_reads=None, lib_size_sigma=None, dispersion=None,
                 sigma_batch=None, mu_lib=None, sigma_lib=None,
                 ambient_frac=None, doublet_rate=None, dropout_mid=None, dropout_shape=None,
                 well_sigma=None, n_wells=None, n_plates=None, plate_sigma=None,
                 depth_batch_sigma=None, kappa=None,
                 seed=42, plate_layout=None):
        super(scRNA, self).__init__(seed=seed, protocol=resolve_protocol(protocol))
        self.n_cells = int(n_cells)
        if count_model not in COUNT_MODELS:
            raise ValueError(f"unknown count_model {count_model!r}; choose {sorted(COUNT_MODELS)}")
        self.count_model = count_model
        self.batch_label = batch_label

        params = dict(PROTOCOL_PRESETS[self.protocol])
        params["protocol"] = self.protocol
        # legacy aliases
        if n_reads is not None:
            params["mu_lib"] = float(n_reads)
        if lib_size_sigma is not None:
            params["sigma_lib"] = float(lib_size_sigma)
        # explicit hyper-parameter overrides
        for name, val in dict(
            sigma_batch=sigma_batch, mu_lib=mu_lib, sigma_lib=sigma_lib, dispersion=dispersion,
            ambient_frac=ambient_frac, doublet_rate=doublet_rate, dropout_mid=dropout_mid,
            dropout_shape=dropout_shape, well_sigma=well_sigma, n_wells=n_wells,
            n_plates=n_plates, plate_sigma=plate_sigma, depth_batch_sigma=depth_batch_sigma,
            kappa=kappa,
        ).items():
            if val is not None:
                params[name] = int(val) if name in _INT_PARAMS else float(val)
        for name in _INT_PARAMS:            # presets / estimates may arrive as floats
            params[name] = int(params.get(name, 0))
        if params["n_wells"] < 0 or params["n_plates"] < 0:
            raise ValueError("n_wells and n_plates must be non-negative "
                             f"(got n_wells={params['n_wells']}, n_plates={params['n_plates']})")

        self.hypers = RNABatchHyperParams(**params)
        if plate_layout is not None and self.hypers.n_wells == 0:
            raise ValueError("plate_layout needs a plate protocol (n_wells > 0), "
                             f"but protocol {self.protocol!r} has n_wells=0")
        self.plate_layout = plate_layout
        # convenience mirror for callers that still read `.n_reads`
        self.n_reads = self.hypers.mu_lib
        # plate/well placement of the assayed cells; set by `run` (None without plates)
        self.plates = None

    # -- selection ---------------------------------------------------------------------
    def _select_cells(self, cell_states, cell_subset):
        if cell_subset is not None:
            return np.asarray(list(cell_subset))
        n = min(self.n_cells, len(cell_states.index))
        rng_sel = np.random.default_rng(self.seed)
        return rng_sel.choice(np.asarray(cell_states.index), size=n, replace=False)

    # -- run ---------------------------------------------------------------------------
    def run(self, cell_data, cell_subset=None, gene_lift=None, **kwargs):
        """Assay single-cell RNA expression for the sampled cells.

        Draws per-cell UMI counts from each cell's expression state through one
        sequencing batch: per-gene capture factor, library size, negative-binomial
        overdispersion, ambient soup, dropout, and doublets. On a plate protocol the cells
        are first placed one per well on the plates, and their depth picks up the offset
        shared by everything on the same plate.

        Parameters
        ----------
        cell_data : dict
            Per-cell ground-truth tables from the sampling stage; uses the expression
            table ``cell_exp``.
        cell_subset : array-like of cell IDs, optional
            Fix the assayed cells so that several batches share the same biology. By
            default every sampled cell is used.
        gene_lift : GeneLift or HumanExpression, optional
            Count human genes instead of the growth genes. A ``GeneLift``
            (:attr:`HumanGenomeLayer.lift <iscc.data.human.HumanGenomeLayer>`) copies each growth
            gene's expression to the human genes it covers. A ``HumanExpression``
            (:meth:`HumanGenomeLayer.expression <iscc.data.human.HumanGenomeLayer.expression>`) uses
            atlas baselines per cell type and Hallmark programmes; its counts are drawn in chunks of
            cells and kept sparse (``counts_csr``, and ``observed_counts`` as a sparse frame), so no
            cells x human-genes dense matrix is formed. Default None counts the growth genes.

        Returns
        -------
        scRNA
            ``self``, with the cell-by-gene UMI matrix in ``observed_counts`` and the
            QC/ground-truth table in ``obs``. Call ``to_anndata`` or ``write`` to
            export.
        """
        cell_states = cell_data["cell_exp"]
        target_cells = self._select_cells(cell_states, cell_subset)
        self.n_cells = len(target_cells)
        genes = cell_states.columns
        if gene_lift is not None and hasattr(gene_lift, "cell_expression"):
            return self._run_human(cell_data, np.asarray(target_cells), gene_lift)

        activities = cell_states.loc[target_cells].values.astype(float)
        if gene_lift is not None:
            activities = gene_lift(activities)
            genes = gene_lift.genes
        totals = activities.sum(axis=1, keepdims=True)
        probs = np.divide(activities, totals, out=np.zeros_like(activities), where=totals > 0)

        # batch realization (per-gene factor, depth, dispersion, ambient soup)
        label = self.batch_label if self.batch_label is not None else f"batch{self.seed}"
        base_profile = activities.sum(axis=0)  # pooled expression -> ambient soup
        self.batch = Batch(self.hypers, seed=self.seed, label=label).realize(genes, base_profile)

        # plate protocols: place the cells one per well before drawing depth, so cells that
        # share a plate share its depth offset (droplet protocols have n_wells = 0 -> None).
        # A caller-given plate map replaces the default sequential split.
        if self.hypers.n_wells <= 0:
            self.plates = None
        elif self.plate_layout is not None:
            self.plates = assign_plates_from_map(target_cells, self.plate_layout,
                                                 self.hypers.n_wells, self.hypers.n_plates)
        else:
            self.plates = assign_plates(self.n_cells, self.hypers.n_wells, self.hypers.n_plates)

        # biology -> library -> batch (shared), then the pluggable count draw
        lib = self.batch.library_factors(self.n_cells, plates=self.plates)
        comp = self.batch.composition(probs)
        counts = self.batch.emit(comp, lib, count_model=self.count_model)
        counts = self.batch.add_ambient(counts, lib)
        counts, self.p_drop = self.batch.apply_dropout(counts, comp, lib)
        counts, is_doublet = self.batch.apply_doublets(counts)
        counts = counts.astype(int)

        self.observed_counts = pd.DataFrame(counts, index=target_cells, columns=genes)
        self._build_obs(cell_data, target_cells, counts, is_doublet, label)
        return self

    def _run_human(self, cell_data, target_cells, expr):
        """The ``HumanExpression`` path of :meth:`run`: the same batch, library, count, ambient,
        dropout and doublet steps, drawn in chunks of ``expr.chunk`` cells into a sparse matrix."""
        import scipy.sparse as sp
        genes = expr.genes
        n = len(target_cells)
        self.n_cells = n
        cuts = list(range(0, n, expr.chunk))
        base_profile = np.zeros(len(genes))
        for lo in cuts:                                   # the ambient soup needs every cell first
            base_profile += expr.cell_expression(cell_data, target_cells[lo:lo + expr.chunk]).sum(axis=0)
        label = self.batch_label if self.batch_label is not None else f"batch{self.seed}"
        self.batch = Batch(self.hypers, seed=self.seed, label=label).realize(genes, base_profile)
        if self.hypers.n_wells <= 0:
            self.plates = None
        elif self.plate_layout is not None:
            self.plates = assign_plates_from_map(target_cells, self.plate_layout,
                                                 self.hypers.n_wells, self.hypers.n_plates)
        else:
            self.plates = assign_plates(n, self.hypers.n_wells, self.hypers.n_plates)
        lib = self.batch.library_factors(n, plates=self.plates)
        blocks = []
        for lo in cuts:
            act = expr.cell_expression(cell_data, target_cells[lo:lo + expr.chunk])
            tot = act.sum(axis=1, keepdims=True)
            probs = np.divide(act, tot, out=np.zeros_like(act), where=tot > 0)
            comp = self.batch.composition(probs)
            l = lib[lo:lo + expr.chunk]
            c = self.batch.emit(comp, l, count_model=self.count_model)
            c = self.batch.add_ambient(c, l)
            c, _ = self.batch.apply_dropout(c, comp, l)
            blocks.append(sp.csr_matrix(c.astype(np.int32)))
        counts = sp.vstack(blocks).tocsr() if blocks else sp.csr_matrix((0, len(genes)), dtype=np.int32)
        counts, is_doublet = _sparse_doublets(self.batch, counts)
        self.p_drop = None
        self.counts_csr = counts
        self.observed_counts = pd.DataFrame.sparse.from_spmatrix(counts, index=target_cells, columns=genes)
        self._build_obs(cell_data, target_cells, counts, is_doublet, label)
        return self

    def _build_obs(self, cell_data, target_cells, counts, is_doublet, label):
        """Assemble the per-cell ground-truth + QC table (-> AnnData `.obs`)."""
        obs = pd.DataFrame(index=pd.Index(target_cells, name="cell"))
        obs["batch"] = label
        obs["protocol"] = self.protocol
        # where the cell physically sat (plate protocols only); plate ids carry the batch
        # label, so concatenated batches stay distinct physical plates
        if self.plates is not None:
            obs["plate"] = self.plates.plate_ids(label)
            obs["plate_index"] = self.plates.plate
            obs["well"] = self.plates.well_ids()
            obs["well_row"] = self.plates.row
            obs["well_col"] = self.plates.col
        obs["n_counts"] = np.asarray(counts.sum(axis=1)).ravel()
        obs["n_genes"] = np.asarray((counts > 0).sum(axis=1)).ravel()
        obs["is_doublet"] = is_doublet
        # surface ground truth (clone / type / coords) when present
        ctype = cell_data.get("cell_type")
        if ctype is not None:
            obs["clone"] = ctype.reindex(target_cells).iloc[:, 0].astype(str).values
        crd = cell_data.get("cell_crd")
        if crd is not None:
            crd = crd.reindex(target_cells)
            obs["row"] = crd["row"].values
            obs["col"] = crd["col"].values
        self.obs = obs

    # -- outputs -----------------------------------------------------------------------
    def to_anndata(self):
        import anndata as ad

        csr = getattr(self, "counts_csr", None)
        adata = ad.AnnData(
            X=self.observed_counts.values.astype(np.float32) if csr is None else csr.astype(np.float32),
            obs=self.obs.copy(),
            var=pd.DataFrame(index=self.observed_counts.columns),
        )
        if {"row", "col"} <= set(self.obs.columns):
            # 10x/squidpy convention: obsm['spatial'] is (x, y) = (col, row), the same order the
            # Visium export uses, so a cell-level plot and a spot-level plot of the same tissue come
            # out the same way up. The raw grid indices stay in obs['row'] / obs['col'].
            adata.obsm["spatial"] = self.obs[["col", "row"]].values.astype(float)
            adata.uns["spatial_axes"] = "xy"
        adata.uns["assay"] = "scRNA"
        adata.uns["protocol"] = self.protocol
        adata.uns["count_model"] = self.count_model
        adata.uns["batch_seed"] = self.seed
        adata.uns["hyperparams"] = self.hypers.to_dict()
        if self.plates is not None:
            # realized layout (the plate count may have grown to fit the cells)
            adata.uns["plate_layout"] = self.plates.to_dict()
        return adata

    def write(self, out_path, write_h5ad=True):
        os.makedirs(out_path, exist_ok=True)
        # CSV kept for backward compatibility with the existing pipeline / tests; not for human-gene
        # counts, which are sparse and far wider (write them with iscc.data.human.write_scrna)
        if getattr(self, "counts_csr", None) is None:
            self.observed_counts.to_csv(os.path.join(out_path, "umis.csv"))
        if write_h5ad:
            label = self.batch.label
            self.to_anndata().write_h5ad(os.path.join(out_path, f"{label}.h5ad"))


def _sparse_doublets(batch, counts):
    """`Batch.apply_doublets` on a sparse barcodes x genes matrix: the same draws, merging each
    chosen barcode with its partner's counts."""
    import scipy.sparse as sp
    n = counts.shape[0]
    is_doublet = np.zeros(n, dtype=bool)
    if batch.h.doublet_rate <= 0 or n < 2:
        return counts, is_doublet
    n_dbl = int(round(batch.h.doublet_rate * n))
    if n_dbl == 0:
        return counts, is_doublet
    idx = batch.rng.choice(n, size=n_dbl, replace=False)
    partners = batch.rng.choice(n, size=n_dbl, replace=True)
    partners = np.where(partners == idx, (partners + 1) % n, partners)
    add = sp.csr_matrix((np.ones(n_dbl), (idx, partners)), shape=(n, n))
    is_doublet[idx] = True
    return (counts + (add @ counts)).tocsr().astype(np.int32), is_doublet


def run_scrna_batches(cell_data, n_batches=2, base_seed=42, design="shared",
                      protocol="10x", n_cells=100, count_model="nb", plate_layout=None,
                      **scrna_kwargs):
    """Generate `n_batches` scRNA batches that share biology but differ technically (§B.3).

    designs:
      * ``"shared"``  — every batch measures the SAME cells (exact cell<->cell correspondence
                        across batches; a replicate-concordance / batch-correction benchmark).
      * ``"split"``   — the sampled cells are partitioned across batches (balanced one-tumour
                        design: each cell appears in exactly one batch).
    (The ``"confounded"`` design — different tumours -> different batches — is produced by
    calling this once per tumour with a distinct `cell_data` and concatenating the results.)

    Returns a list of run `scRNA` instances (one per batch), each with `.observed_counts`,
    `.obs`, `.batch`, and `.to_anndata()`. On a plate protocol each batch is its own plate
    run: its plate ids carry the batch label, so they stay distinct after `concat_batches`.

    `plate_layout` (see `scRNA`) is one plate map for the whole run, passed to every batch:
    each batch looks up its own cells in it, so under ``"split"`` a batch sorts only its share
    of the cells and under ``"shared"`` every batch repeats the same sort as its own plate run.
    """
    cell_states = cell_data["cell_exp"]
    n_avail = len(cell_states.index)
    sel_rng = np.random.default_rng(base_seed)  # biology selection (NOT technical)

    if design == "shared":
        n = min(n_cells, n_avail)
        chosen = sel_rng.choice(np.asarray(cell_states.index), size=n, replace=False)
        subsets = [chosen for _ in range(n_batches)]
    elif design == "split":
        n = min(n_cells * n_batches, n_avail)
        chosen = sel_rng.choice(np.asarray(cell_states.index), size=n, replace=False)
        subsets = [np.asarray(s) for s in np.array_split(chosen, n_batches)]
    else:
        raise ValueError(
            f"unknown design {design!r}; use 'shared', 'split', or build 'confounded' "
            "by calling run_scrna_batches per tumour"
        )

    assays = []
    for i, subset in enumerate(subsets):
        assay = scRNA(
            protocol=protocol, n_cells=len(subset), count_model=count_model,
            batch_label=f"batch{i}", seed=base_seed + 1 + i, plate_layout=plate_layout,
            **scrna_kwargs,
        )
        assay.run(cell_data, cell_subset=subset)
        assays.append(assay)
    return assays


def concat_batches(assays):
    """Concatenate per-batch AnnData into one labelled object (for integration benchmarks)."""
    import anndata as ad

    adatas = [a.to_anndata() for a in assays]
    combined = ad.concat(adatas, join="outer", label="batch_index", index_unique="-")
    combined.uns["protocol"] = adatas[0].uns["protocol"]
    combined.uns["count_model"] = adatas[0].uns["count_model"]
    combined.uns["n_batches"] = len(adatas)
    return combined
