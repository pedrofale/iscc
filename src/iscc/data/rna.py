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
                 low amplification noise (low dispersion), extra well-in-plate nesting.

Two `scRNA` instances with the same hyper-parameters and different seeds are two batches:
same biology, different technical signature (§B.3). `run_scrna_batches` builds a labelled
multi-batch benchmark (shared / balanced-split / confounded designs).
"""
from .assay import Assay
from .batch import Batch, BatchHyperParams, COUNT_MODELS

import os

import numpy as np
import pandas as pd


# Protocol-typical hyper-parameter presets (DESIGN_features §B.2, §D).
PROTOCOL_PRESETS = {
    "10x": dict(
        sigma_batch=0.10, mu_lib=4000.0, sigma_lib=0.35, dispersion=0.30,
        ambient_frac=0.05, doublet_rate=0.05, dropout_mid=1.5, dropout_shape=1.0,
        well_sigma=0.0, depth_batch_sigma=0.05, kappa=50.0,
    ),
    # 5' UMIs suppress amplification noise (low dispersion); plate prep -> low ambient /
    # few doublets, higher sensitivity (deeper) / lower dropout; well-in-plate nesting.
    "smartseq3": dict(
        sigma_batch=0.10, mu_lib=8000.0, sigma_lib=0.45, dispersion=0.15,
        ambient_frac=0.005, doublet_rate=0.01, dropout_mid=0.0, dropout_shape=1.0,
        well_sigma=0.15, depth_batch_sigma=0.05, kappa=50.0,
    ),
}
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
        amplification noise, extra well-in-plate nesting). Common spellings are accepted
        (``"10X"``, ``"tenx"``, ``"droplet"``; ``"smart-seq3"``, ``"ss3"``, ``"plate"``).
        Sets the defaults of every hyper-parameter below.
    n_cells : int, default 100
        Number of cells to sample and assay, capped at the number available. Ignored when
        ``cell_subset`` is passed to ``run``.
    count_model : str, default "nb"
        Final count-emission model: ``"nb"`` negative-binomial (independent per-gene
        overdispersion, what ``estimate`` fits) or ``"dm"`` Dirichlet-multinomial (a fixed
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
        Per-cell "well" / plate-position LogNormal sd — an extra per-cell multiplier nesting a
        plate-position effect inside the batch (Smart-seq3); 0 disables. Default None ->
        preset (10x: 0.0, smartseq3: 0.15).
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
    """

    def __init__(self, protocol="10x", n_cells=100, count_model="nb", batch_label=None,
                 n_reads=None, lib_size_sigma=None, dispersion=None,
                 sigma_batch=None, mu_lib=None, sigma_lib=None,
                 ambient_frac=None, doublet_rate=None, dropout_mid=None, dropout_shape=None,
                 well_sigma=None, depth_batch_sigma=None, kappa=None,
                 seed=42):
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
            dropout_shape=dropout_shape, well_sigma=well_sigma, depth_batch_sigma=depth_batch_sigma,
            kappa=kappa,
        ).items():
            if val is not None:
                params[name] = float(val)

        self.hypers = BatchHyperParams(**params)
        # convenience mirror for callers that still read `.n_reads`
        self.n_reads = self.hypers.mu_lib

    # -- selection ---------------------------------------------------------------------
    def _select_cells(self, cell_states, cell_subset):
        if cell_subset is not None:
            return np.asarray(list(cell_subset))
        n = min(self.n_cells, len(cell_states.index))
        rng_sel = np.random.default_rng(self.seed)
        return rng_sel.choice(np.asarray(cell_states.index), size=n, replace=False)

    # -- run ---------------------------------------------------------------------------
    def run(self, cell_data, cell_subset=None, **kwargs):
        """Assay single-cell RNA expression for the sampled cells.

        Draws per-cell UMI counts from each cell's expression state through one
        sequencing batch: per-gene capture factor, library size, negative-binomial
        overdispersion, ambient soup, dropout, and doublets.

        Parameters
        ----------
        cell_data : dict
            Per-cell ground-truth tables from the sampling stage; uses the expression
            table ``cell_exp``.
        cell_subset : array-like of cell IDs, optional
            Fix the assayed cells so that several batches share the same biology. By
            default every sampled cell is used.

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

        activities = cell_states.loc[target_cells].values.astype(float)
        totals = activities.sum(axis=1, keepdims=True)
        probs = np.divide(activities, totals, out=np.zeros_like(activities), where=totals > 0)

        # batch realization (per-gene factor, depth, dispersion, ambient soup)
        label = self.batch_label if self.batch_label is not None else f"batch{self.seed}"
        base_profile = activities.sum(axis=0)  # pooled expression -> ambient soup
        self.batch = Batch(self.hypers, seed=self.seed, label=label).realize(genes, base_profile)

        # biology -> library -> batch (shared), then the pluggable count draw
        lib = self.batch.library_factors(self.n_cells)
        comp = self.batch.composition(probs)
        counts = self.batch.emit(comp, lib, count_model=self.count_model)
        counts = self.batch.add_ambient(counts, lib)
        counts, self.p_drop = self.batch.apply_dropout(counts, comp, lib)
        counts, is_doublet = self.batch.apply_doublets(counts)
        counts = counts.astype(int)

        self.observed_counts = pd.DataFrame(counts, index=target_cells, columns=genes)
        self._build_obs(cell_data, target_cells, counts, is_doublet, label)
        return self

    def _build_obs(self, cell_data, target_cells, counts, is_doublet, label):
        """Assemble the per-cell ground-truth + QC table (-> AnnData `.obs`)."""
        obs = pd.DataFrame(index=pd.Index(target_cells, name="cell"))
        obs["batch"] = label
        obs["protocol"] = self.protocol
        obs["n_counts"] = counts.sum(axis=1)
        obs["n_genes"] = (counts > 0).sum(axis=1)
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

        adata = ad.AnnData(
            X=self.observed_counts.values.astype(np.float32),
            obs=self.obs.copy(),
            var=pd.DataFrame(index=self.observed_counts.columns),
        )
        if {"row", "col"} <= set(self.obs.columns):
            adata.obsm["spatial"] = self.obs[["row", "col"]].values.astype(float)
        adata.uns["assay"] = "scRNA"
        adata.uns["protocol"] = self.protocol
        adata.uns["count_model"] = self.count_model
        adata.uns["batch_seed"] = self.seed
        adata.uns["hyperparams"] = self.hypers.to_dict()
        return adata

    def write(self, out_path, write_h5ad=True):
        os.makedirs(out_path, exist_ok=True)
        # CSV kept for backward compatibility with the existing pipeline / tests
        self.observed_counts.to_csv(os.path.join(out_path, "umis.csv"))
        if write_h5ad:
            label = self.batch.label
            self.to_anndata().write_h5ad(os.path.join(out_path, f"{label}.h5ad"))


def run_scrna_batches(cell_data, n_batches=2, base_seed=42, design="shared",
                      protocol="10x", n_cells=100, count_model="nb", **scrna_kwargs):
    """Generate `n_batches` scRNA batches that share biology but differ technically (§B.3).

    designs:
      * ``"shared"``  — every batch measures the SAME cells (exact cell<->cell correspondence
                        across batches; a replicate-concordance / batch-correction benchmark).
      * ``"split"``   — the sampled cells are partitioned across batches (balanced one-tumour
                        design: each cell appears in exactly one batch).
    (The ``"confounded"`` design — different tumours -> different batches — is produced by
    calling this once per tumour with a distinct `cell_data` and concatenating the results.)

    Returns a list of run `scRNA` instances (one per batch), each with `.observed_counts`,
    `.obs`, `.batch`, and `.to_anndata()`.
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
            batch_label=f"batch{i}", seed=base_seed + 1 + i, **scrna_kwargs,
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
