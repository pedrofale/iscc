"""Single-cell spatial (imaging-based) transcriptomics assay — F9 (DESIGN_features milestone).

Imaging platforms (MERFISH / Xenium / seqFISH) measure a **targeted gene panel** at **single-cell
resolution** in situ. This assay sits between the two existing spatial/expression modalities:

  * unlike **Visium** (F6, `visium.py`) there is NO spot aggregation — each cell is its own
    observation, with its own `cell_crd` coordinates;
  * unlike **scRNA** (F3, `rna.py`) only a `panel` subset of the transcriptome is measured
    (*transcriptome coverage*), at higher per-molecule sensitivity (near-zero dropout) but lower
    total depth.

It reuses the F3 scRNA count machinery unchanged (`batch.Batch`: per-gene factor, per-cell library,
NB/DM count draw, ambient soup), restricted to the panel genes, keeping each cell's coordinates.

Two knobs the caller sets (per the design):
  * **transcriptome coverage** — `panel` (explicit gene list) or `n_panel_genes` (top-N by mean
    expression); default = whole transcriptome (high-plex imaging).
  * **data distribution** — `mu_counts` (per-cell total molecules over the panel), `dispersion`,
    `dropout_mid`, `ambient_frac`, ... (imaging presets: low depth, near-zero dropout, no droplets).

Output is AnnData: `X` = cells x panel-genes counts, `obsm["spatial"]` = per-cell (row, col),
`.obs` = per-cell ground truth (clone / type / coords) + QC, `.uns` = hyperparams.
"""
from .assay import Assay
from .batch import Batch, RNABatchHyperParams, COUNT_MODELS

import os

import numpy as np
import pandas as pd


# Platform-typical magnitudes: targeted, high-sensitivity, low-depth, near-zero dropout, and no
# droplet artifacts (segmentation, not barcoding) — the imaging counterpart of PROTOCOL_PRESETS.
IMAGING_PRESETS = {
    "merfish": dict(
        sigma_batch=0.08, mu_lib=500.0, sigma_lib=0.35, dispersion=0.20,
        ambient_frac=0.02, doublet_rate=0.0, dropout_mid=0.0, dropout_shape=1.0,
        well_sigma=0.0, depth_batch_sigma=0.05, kappa=50.0,
    ),
    "xenium": dict(
        sigma_batch=0.08, mu_lib=300.0, sigma_lib=0.35, dispersion=0.20,
        ambient_frac=0.02, doublet_rate=0.0, dropout_mid=0.0, dropout_shape=1.0,
        well_sigma=0.0, depth_batch_sigma=0.05, kappa=50.0,
    ),
}
_PLATFORM_ALIASES = {
    "merfish": "merfish", "vizgen": "merfish", "seqfish": "merfish",
    "xenium": "xenium", "cosmx": "xenium", "10ximaging": "xenium",
}


def resolve_platform(platform):
    key = _PLATFORM_ALIASES.get(str(platform).lower().replace(" ", "").replace("-", ""))
    if key is None:
        raise ValueError(
            f"unknown imaging platform {platform!r}; choose from {sorted(set(_PLATFORM_ALIASES))}"
        )
    return key


class scSpatial(Assay):
    """Single-cell spatial (imaging) assay: per-cell panel counts at `cell_crd`, reusing `Batch`.

    Parameters
    ----------
    platform : str
        Imaging preset ("merfish" / "xenium" and aliases); sets the default technical magnitudes.
    panel : list[str], optional
        Explicit gene panel (transcriptome coverage). Overrides `n_panel_genes`.
    n_panel_genes : int, optional
        Size of a panel selected as the top-N genes by mean expression. `None` (with `panel=None`)
        measures the whole transcriptome.
    mu_counts, dispersion, ambient_frac, dropout_mid, sigma_batch, sigma_lib : float, optional
        Explicit overrides of the data-distribution hyper-parameters (`mu_counts` -> `mu_lib`).
    """

    def __init__(self, platform="merfish", panel=None, n_panel_genes=None, count_model="nb",
                 batch_label=None, mu_counts=None, dispersion=None, ambient_frac=None,
                 dropout_mid=None, sigma_batch=None, sigma_lib=None, seed=42, **assay_kwargs):
        key = resolve_platform(platform)
        super(scSpatial, self).__init__(seed=seed, protocol=key)
        if count_model not in COUNT_MODELS:
            raise ValueError(f"unknown count_model {count_model!r}; choose {sorted(COUNT_MODELS)}")
        self.count_model = count_model
        self.batch_label = batch_label
        self.panel = list(panel) if panel is not None else None
        self.n_panel_genes = int(n_panel_genes) if n_panel_genes is not None else None

        params = dict(IMAGING_PRESETS[key])
        params["protocol"] = key
        if mu_counts is not None:                              # alias -> per-cell library
            params["mu_lib"] = float(mu_counts)
        for name, val in dict(
            dispersion=dispersion, ambient_frac=ambient_frac, dropout_mid=dropout_mid,
            sigma_batch=sigma_batch, sigma_lib=sigma_lib,
        ).items():
            if val is not None:
                params[name] = float(val)
        self.hypers = RNABatchHyperParams(**params)
        self.n_reads = self.hypers.mu_lib                      # convenience mirror

    # -- panel selection (transcriptome coverage) --------------------------------------
    def _select_panel(self, cell_exp):
        genes = list(cell_exp.columns)
        if self.panel is not None:
            keep = [g for g in self.panel if g in cell_exp.columns]
            if not keep:
                raise ValueError("none of the requested `panel` genes are in cell_exp")
            return keep
        if self.n_panel_genes is not None and self.n_panel_genes < len(genes):
            mean_exp = np.asarray(cell_exp.mean(axis=0).values, dtype=float)
            top = np.argsort(mean_exp)[::-1][: self.n_panel_genes]
            return [genes[i] for i in sorted(top.tolist())]    # keep genome order
        return genes

    # -- run ---------------------------------------------------------------------------
    def run(self, cell_data, cell_subset=None, **kwargs):
        """Realize the imaging assay on `cell_data`. Measures all cells in situ (or `cell_subset`)."""
        cell_exp = cell_data["cell_exp"]
        panel_genes = self._select_panel(cell_exp)
        target = (np.asarray(list(cell_subset)) if cell_subset is not None
                  else np.asarray(cell_exp.index))
        self.n_cells = len(target)

        activities = cell_exp.loc[target, panel_genes].values.astype(float)
        totals = activities.sum(axis=1, keepdims=True)
        probs = np.divide(activities, totals, out=np.zeros_like(activities), where=totals > 0)

        label = self.batch_label if self.batch_label is not None else f"img{self.seed}"
        base_profile = activities.sum(axis=0)                  # pooled panel expression -> ambient
        self.batch = Batch(self.hypers, seed=self.seed, label=label).realize(panel_genes, base_profile)

        lib = self.batch.library_factors(self.n_cells)
        comp = self.batch.composition(probs)
        counts = self.batch.emit(comp, lib, count_model=self.count_model)
        counts = self.batch.add_ambient(counts, lib)
        counts, self.p_drop = self.batch.apply_dropout(counts, comp, lib)
        counts = counts.astype(int)

        self.genes = panel_genes
        self.observed_counts = pd.DataFrame(counts, index=target, columns=panel_genes)
        self._build_obs(cell_data, target, counts, label)
        return self

    def _build_obs(self, cell_data, target_cells, counts, label):
        """Per-cell ground truth + QC (-> AnnData `.obs`); coords are the imaging headline."""
        obs = pd.DataFrame(index=pd.Index(target_cells, name="cell"))
        obs["batch"] = label
        obs["platform"] = self.protocol
        obs["n_counts"] = counts.sum(axis=1)
        obs["n_genes"] = (counts > 0).sum(axis=1)
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
        adata.uns["assay"] = "scspatial"
        adata.uns["platform"] = self.protocol
        adata.uns["count_model"] = self.count_model
        adata.uns["batch_seed"] = self.seed
        adata.uns["n_panel_genes"] = len(self.genes)
        adata.uns["hyperparams"] = self.hypers.to_dict()
        return adata

    def write(self, out_path, write_h5ad=True):
        os.makedirs(out_path, exist_ok=True)
        self.observed_counts.to_csv(os.path.join(out_path, "counts.csv"))
        if write_h5ad:
            self.to_anndata().write_h5ad(os.path.join(out_path, f"{self.batch.label}.h5ad"))
