"""Spatial transcriptomics (Visium) assay with a spatially-correlated batch model (F6).

Upgrades the original grid-spots -> averaged-expression -> fixed-multinomial stub to a proper
assay matching the scRNA (F3) / DNA (F4/F5) modalities: an `Assay` + a `VisiumBatch` realization
(`batch.py`) + hyper-parameters (`VisiumBatchHyperParams`) + AnnData output, with an `estimate`
(`estimate_visium`, M4 §C.2) and validation (`validate_visium.py`).

Generative model for spot *s*, gene *g* (DESIGN_features §D "spatial" row):

  1. **Spot layout** — Visium-like hexagonal spots over the 2D section (`spot_pitch` spacing,
     `spot_radius` capture radius, ~1-10 cells/spot). iscc is 2D (single section, no z).
  2. **Spot->cell aggregation** — pool the `cell_exp` of cells whose `cell_crd` falls in the spot;
     record per-spot ground truth (n_cells, member cell ids, dominant clone + clone fractions).
  3. **Lateral mRNA diffusion** — a Gaussian kernel (`diffusion_sigma`) bleeds expression to
     neighbouring spots (the spatial-mixing artifact spatial-deconvolution methods fight).
  4. **Spatially-correlated capture-efficiency field** — the HEADLINE piece: a smooth positive
     random field over the spot coords (squared-exponential GP, `field_lengthscale` / `field_sigma`)
     times an `edge_sigma` tissue-boundary falloff, scaling the per-spot library so the realized
     depth has positive spatial autocorrelation (Moran's I > 0). Lives in `VisiumBatch`.
  5. **Shared §B.2 layer** — per-gene batch factor (`sigma_batch`), per-spot library
     (LogNormal `mu_counts`/`sigma_counts`), the pluggable NB/DM count draw (`COUNT_MODELS`),
     and ambient soup (`ambient_frac`).

Output is AnnData: `X` = spots x genes counts, `obsm["spatial"]` = spot (row, col), `.obs` = the
per-spot ground truth + QC, `.uns` = {assay, count_model, hyperparams, capture field}.
"""
from .assay import Assay
from .batch import VisiumBatch, VisiumBatchHyperParams, COUNT_MODELS

import os

import numpy as np
import pandas as pd


def morans_i(values, coords, lengthscale=None):
    """Moran's I — global spatial autocorrelation of `values` at `coords` (n, 2).

    ``I = (n / W) * sum_ij w_ij z_i z_j / sum_i z_i^2`` with z the mean-centred values and a
    Gaussian distance weight ``w_ij = exp(-d_ij^2 / (2*L^2))`` (self-weight 0). I > 0 means
    nearby spots have similar values (positive autocorrelation); ~0 means no spatial structure.
    `lengthscale` L defaults to the median nearest-neighbour distance (a local-neighbourhood scale).
    """
    v = np.asarray(values, dtype=float).ravel()
    coords = np.asarray(coords, dtype=float)
    n = v.shape[0]
    if n < 3:
        return float("nan")
    diff = coords[:, None, :] - coords[None, :, :]
    d2 = np.sum(diff * diff, axis=-1)
    if lengthscale is None:
        d = np.sqrt(d2)
        np.fill_diagonal(d, np.inf)
        lengthscale = float(np.median(d.min(axis=1)))
    L = max(float(lengthscale), 1e-9)
    W = np.exp(-d2 / (2.0 * L * L))
    np.fill_diagonal(W, 0.0)
    z = v - v.mean()
    denom = np.sum(z * z)
    Wsum = W.sum()
    if denom <= 0 or Wsum <= 0:
        return float("nan")
    num = z[None, :] @ W @ z[:, None]
    return float((n / Wsum) * (num.item() / denom))


class Visium(Assay):
    """Visium spatial assay: lays spots over the section, aggregates cells, applies a `VisiumBatch`.

    Any hyper-parameter below can be overridden explicitly (``None`` keeps the default, which
    is calibrated to a real 10x Visium section). Legacy aliases are accepted: ``n_reads`` maps
    onto ``mu_counts``, and a spot grid requested via ``n_spots_x`` / ``n_spots_y`` derives
    ``spot_pitch`` from ``grid_side`` at run time. The number of spots (and cells per spot)
    follows from ``grid_side`` (passed to ``run``) and the sampled ``cell_data``, so there is
    no ``n_cells``.

    Parameters
    ----------
    count_model : str, default "dm"
        Final count-emission model: ``"dm"`` Dirichlet-multinomial (compositional per-spot
        capture, the Visium default) or ``"nb"`` negative-binomial (independent per-gene).
    batch_label : str, optional
        Section/batch label; defaults to ``f"visium{seed}"``.
    n_reads : float, optional
        Legacy alias for ``mu_counts`` (mean per-spot library size).
    spot_pitch : float, default 2.0
        Spot centre-to-centre spacing (coordinate units); sets spot density / number of spots.
        Overridden at run time if ``n_spots_x`` / ``n_spots_y`` are given.
    spot_radius : float, default 1.0
        Spot capture radius: cells within this distance of a spot centre are pooled into it
        (~1-10 cells / spot).
    mu_counts : float, default 20000.0
        Mean per-spot library size (total UMIs / spot); ``ell_s ~ LogNormal(log mu_counts,
        sigma_counts^2) × capture_field``.
    sigma_counts : float, default 0.45
        Per-spot library-size LogNormal sd (spot-to-spot depth variation).
    field_lengthscale : float, default 18.0
        Spatial autocorrelation length of the capture-efficiency field (the squared-exponential
        GP length-scale, in coordinate units); larger -> smoother field -> higher Moran's I.
    field_sigma : float, default 0.70
        Capture-field strength (log-space sd of the smooth positive field); 0 -> flat field
        (no spatial capture bias).
    edge_sigma : float, default 0.30
        Tissue-boundary capture falloff: the very edge of the section is reduced by this
        fraction, the interior stays ~1. 0 disables; a value < 1 keeps efficiency positive.
    diffusion_sigma : float, default 0.0
        Lateral mRNA-bleed Gaussian kernel sd, spreading each spot's expression into its
        neighbours before counting; 0 disables (no bleed).
    sigma_batch : float, default 0.10
        Per-gene batch-factor LogNormal sd, shared across spots (Splatter ``batch.facScale``).
    ambient_frac : float, default 0.05
        Fraction of each spot's library drawn as ambient "soup" contamination.
    kappa : float, default 50.0
        Dirichlet-multinomial concentration (only for ``count_model="dm"``); large -> ~
        multinomial, small -> lumpy proportions.
    nb_dispersion : float, default 0.30
        Negative-binomial overdispersion phi (``var = mu + phi*mu^2``; only for
        ``count_model="nb"``).
    n_spots_x : int, optional
        Legacy alias: request a fixed spot-grid width. When both ``n_spots_x`` and
        ``n_spots_y`` are set, the pitch is derived as ``grid_side / max(n_spots_x, n_spots_y)``
        at run time instead of using ``spot_pitch``.
    n_spots_y : int, optional
        Legacy alias: request a fixed spot-grid height (see ``n_spots_x``).
    seed : int, default 42
        RNG seed. Fixes the technical signature (per-gene batch factor, spatial capture field,
        per-spot depth) and is reproducible.
    """

    def __init__(self, count_model="dm", batch_label=None, n_reads=None,
                 spot_pitch=None, spot_radius=None, mu_counts=None, sigma_counts=None,
                 field_lengthscale=None, field_sigma=None, edge_sigma=None,
                 diffusion_sigma=None, sigma_batch=None, ambient_frac=None,
                 kappa=None, nb_dispersion=None,
                 n_spots_x=None, n_spots_y=None, seed=42, **assay_kwargs):
        super(Visium, self).__init__(seed=seed, protocol="visium")
        if count_model not in COUNT_MODELS:
            raise ValueError(f"unknown count_model {count_model!r}; choose {sorted(COUNT_MODELS)}")
        self.count_model = count_model
        self.batch_label = batch_label
        # a spot grid requested as n_spots_x/n_spots_y is resolved to a pitch at run time
        self._n_spots_x = n_spots_x
        self._n_spots_y = n_spots_y

        params = VisiumBatchHyperParams().to_dict()
        params["count_model"] = count_model
        if n_reads is not None:                            # legacy alias -> per-spot library
            params["mu_counts"] = float(n_reads)
        for name, val in dict(
            spot_pitch=spot_pitch, spot_radius=spot_radius, mu_counts=mu_counts,
            sigma_counts=sigma_counts, field_lengthscale=field_lengthscale,
            field_sigma=field_sigma, edge_sigma=edge_sigma, diffusion_sigma=diffusion_sigma,
            sigma_batch=sigma_batch, ambient_frac=ambient_frac, kappa=kappa,
            nb_dispersion=nb_dispersion,
        ).items():
            if val is not None:
                params[name] = float(val)
        self.hypers = VisiumBatchHyperParams(**params)
        self.n_reads = self.hypers.mu_counts               # convenience mirror

    # -- spot layout -------------------------------------------------------------------
    def _spot_layout(self, grid_side):
        """Hexagonally-packed Visium-like spots over [0, grid_side]^2 (single 2D section).

        Rows are offset by half a pitch (hex packing) and spaced by ``pitch*sqrt(3)/2``. If the
        caller fixed `n_spots_x`/`n_spots_y`, the pitch is derived from `grid_side` instead.
        """
        pitch = float(self.hypers.spot_pitch)
        if self._n_spots_x and self._n_spots_y:
            pitch = grid_side / max(self._n_spots_x, self._n_spots_y)
            self.hypers.spot_pitch = pitch
        coords = []
        row_dy = pitch * np.sqrt(3.0) / 2.0
        r = pitch / 2.0
        ri = 0
        while r <= grid_side - pitch / 2.0 + 1e-9:
            offset = (pitch / 2.0) if (ri % 2) else 0.0
            c = pitch / 2.0 + offset
            while c <= grid_side - pitch / 2.0 + 1e-9:
                coords.append((r, c))
                c += pitch
            r += row_dy
            ri += 1
        if not coords:                                     # tiny grid -> a single centre spot
            coords = [(grid_side / 2.0, grid_side / 2.0)]
        return np.asarray(coords, dtype=float)

    # -- diffusion ---------------------------------------------------------------------
    def _diffuse(self, raw, coords):
        """Lateral mRNA bleed: replace each spot's expression by a Gaussian-weighted neighbourhood
        average (`diffusion_sigma`). ``diffusion_sigma<=0`` is the identity (no bleed)."""
        sig = float(self.hypers.diffusion_sigma)
        if sig <= 0:
            return raw
        diff = coords[:, None, :] - coords[None, :, :]
        d2 = np.sum(diff * diff, axis=-1)
        W = np.exp(-d2 / (2.0 * sig * sig))
        W = W / W.sum(axis=1, keepdims=True)
        return W @ raw

    # -- run ---------------------------------------------------------------------------
    def run(self, cell_data, grid_side):
        """Assay a 10x Visium spatial-transcriptomics section over the sampled cells.

        Lays out a ``grid_side`` x ``grid_side`` grid of spots, pools each spot's
        cells, applies lateral mRNA diffusion and a smooth spatial capture field, and
        draws per-spot UMI counts.

        Parameters
        ----------
        cell_data : dict
            Per-cell ground-truth tables from the sampling stage; uses the expression
            table ``cell_exp`` and the spatial coordinates ``cell_crd``.
        grid_side : int
            Number of spots along each side of the (square) section.

        Returns
        -------
        Visium
            ``self``, with the spot-by-gene UMI matrix in ``spot_counts``, spot
            coordinates in ``spot_coords``, and the per-spot ground-truth/QC table in
            ``obs``. Call ``to_anndata`` or ``write`` to export.
        """
        cell_exp = cell_data["cell_exp"]
        cell_crd = cell_data["cell_crd"]
        genes = cell_exp.columns
        n_genes = len(genes)
        exp_vals = cell_exp.values.astype(float)
        crd = cell_crd[["row", "col"]].values.astype(float)
        cell_ids = np.asarray(cell_exp.index)
        ctype = cell_data.get("cell_type")
        clone_labels = ctype.reindex(cell_exp.index).iloc[:, 0].astype(str).values if ctype is not None else None

        spot_coords = self._spot_layout(grid_side)
        n_spots = spot_coords.shape[0]
        r2 = float(self.hypers.spot_radius) ** 2

        # 1-2. spot -> cell aggregation (pool cell_exp; record ground truth)
        raw = np.zeros((n_spots, n_genes))
        n_cells_arr = np.zeros(n_spots, dtype=int)
        members = []
        dominant = []
        clone_fracs = []
        for s, (sr, sc) in enumerate(spot_coords):
            d2 = (crd[:, 0] - sr) ** 2 + (crd[:, 1] - sc) ** 2
            inb = np.where(d2 < r2)[0]
            members.append(cell_ids[inb])
            n_cells_arr[s] = inb.size
            if inb.size:
                raw[s] = exp_vals[inb].sum(axis=0)
            if inb.size and clone_labels is not None:
                labs, cnts = np.unique(clone_labels[inb], return_counts=True)
                dominant.append(str(labs[cnts.argmax()]))
                clone_fracs.append(dict(zip(labs.tolist(), (cnts / cnts.sum()).tolist())))
            else:
                dominant.append("")
                clone_fracs.append({})

        # 3. lateral mRNA diffusion, then per-spot composition (lambda)
        diffused = self._diffuse(raw, spot_coords)
        totals = diffused.sum(axis=1, keepdims=True)
        probs = np.divide(diffused, totals, out=np.zeros_like(diffused), where=totals > 0)

        # 4-5. batch realization (per-gene factor + spatial capture field), then biology -> library
        # -> batch -> the pluggable count draw + ambient
        label = self.batch_label if self.batch_label is not None else f"visium{self.seed}"
        base_profile = raw.sum(axis=0)                     # pooled expression -> ambient soup
        self.batch = VisiumBatch(self.hypers, seed=self.seed, label=label).realize(
            genes, base_profile, spot_coords)
        occupied = (n_cells_arr > 0).astype(float)
        lib = self.batch.spot_library(occupied=occupied)
        comp = self.batch.composition(probs)
        counts = self.batch.emit(comp, lib, count_model=self.count_model)
        counts = self.batch.add_ambient(counts, lib)
        counts = counts.astype(int)

        # -- store outputs (DataFrames + ground truth) ---------------------------------
        spot_names = [f"S{i}" for i in range(n_spots)]
        self.spot_names = spot_names
        self.genes = list(genes)
        self.spot_coords = spot_coords
        self.spot_members = members
        self.spot_counts = pd.DataFrame(counts, index=spot_names, columns=genes)
        self.capture_field = self.batch.capture_field
        self.raw_expr = pd.DataFrame(raw, index=spot_names, columns=genes)
        obs = pd.DataFrame(index=pd.Index(spot_names, name="spot"))
        obs["batch"] = label
        obs["n_cells"] = n_cells_arr
        obs["n_counts"] = counts.sum(axis=1)
        obs["library"] = lib
        obs["capture_field"] = self.capture_field
        obs["clone"] = dominant
        obs["clone_frac"] = [max(f.values()) if f else 0.0 for f in clone_fracs]
        obs["row"] = spot_coords[:, 0]
        obs["col"] = spot_coords[:, 1]
        self.obs = obs
        self.clone_fracs = clone_fracs
        # legacy attributes kept for any caller of the old stub
        self.spot_umi = self.spot_counts
        self.spot_crd = pd.DataFrame(spot_coords, index=spot_names, columns=["row", "col"])
        self.spot_cell_counts = pd.DataFrame({"n_cells": n_cells_arr}, index=spot_names)
        self.spot_cell_ids = pd.DataFrame(
            [",".join(m.tolist()) for m in members], index=spot_names, columns=["cell_ids"])
        return self

    # -- outputs -----------------------------------------------------------------------
    def to_anndata(self):
        import anndata as ad

        adata = ad.AnnData(
            X=self.spot_counts.values.astype(np.float32),
            obs=self.obs.copy(),
            var=pd.DataFrame(index=self.spot_counts.columns),
        )
        adata.obsm["spatial"] = self.spot_coords.astype(float)
        adata.uns["assay"] = "visium"
        adata.uns["protocol"] = self.protocol
        adata.uns["count_model"] = self.count_model
        adata.uns["batch_seed"] = self.seed
        adata.uns["hyperparams"] = self.hypers.to_dict()
        adata.uns["capture_field"] = np.asarray(self.capture_field, dtype=float)
        return adata

    def write(self, out_path, write_h5ad=True):
        os.makedirs(out_path, exist_ok=True)
        # CSVs kept for backward compatibility with the previous pipeline / tests
        self.spot_counts.to_csv(os.path.join(out_path, "spot_umi.csv"))
        self.spot_crd.to_csv(os.path.join(out_path, "spot_crd.csv"))
        self.spot_cell_counts.to_csv(os.path.join(out_path, "spot_cell_counts.csv"))
        self.spot_cell_ids.to_csv(os.path.join(out_path, "spot_cell_ids.csv"))
        if write_h5ad:
            self.to_anndata().write_h5ad(os.path.join(out_path, f"{self.batch.label}.h5ad"))
