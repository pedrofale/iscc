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

Output is a standard 10x/squidpy AnnData: `X` = spots x genes counts, `obsm["spatial"]` = spot
(x, y) = (col, row), `.obs` = per-spot ground truth + QC (incl. `in_tissue`, `total_counts`),
`.uns["spatial"]` = the section's scalefactors (spot diameter), and `.uns` also carries
{assay, count_model, hyperparams, capture field}. It plots directly with `sq.pl.spatial_scatter`
/ `sc.pl.spatial` — no custom plotting helpers.
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
    section_frac : float, optional
        Take a thin physical **section** of a count-based (deme) tumor before assaying: place
        each deme's cells within its unit cell and keep this fraction of each deme's 3-D column
        (see :func:`iscc.sample.spatialize`). The tissue is then centred on the fixed slide and a
        morphology *image* is attached to the output. ``None`` (default) uses ``cell_crd`` as
        given (no placement) — appropriate when the coordinates are already at cell resolution.
    placement : tuple of float, optional
        Where the fixed slide sits on the section, as a ``(row, col)`` in the section's coordinate
        frame that the slide's capture area is centred on. ``None`` (default) centres the section's
        centroid on the slide. When the section is **larger than the slide** the slide images only
        the part it covers, so ``placement`` chooses *which* region (e.g. a particular lesion) is
        captured rather than always the middle. The slide's footprint is
        ``placement ± (slide_height/2, slide_width/2)`` in section coordinates.
    rotation : float, default 0.0
        Degrees to rotate the section on the slide before placing it (about ``placement`` / the
        centroid). The v1 slide is wider than it is tall, so a tall section captures more of itself
        rotated 90° to lie along the slide's long axis.
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
                 kappa=None, nb_dispersion=None, section_frac=None, placement=None, rotation=0.0,
                 n_spots_x=None, n_spots_y=None, seed=42):
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
        self.section_frac = section_frac                   # None -> use cell_crd as given
        # where the fixed slide sits on the section: None centres the section on the slide; a
        # (row, col) in the section's coordinate frame centres the slide's capture area on that point
        # (so a section larger than the slide can be imaged at a chosen location, not just the middle).
        self.placement = placement
        self.rotation = float(rotation)                    # degrees to rotate the section on the slide
        self._grid = None                                  # placement state set by place_grid()/run()

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

    # -- fixed v1 slide + section placement --------------------------------------------
    def _visium_v1_layout(self):
        """The fixed 10x Visium v1 capture array: 78 x 64 hexagonally-packed spots (**4,992**).

        A real slide is a fixed grid of spots regardless of the tissue placed on it; ``spot_pitch``
        sets the physical spacing (100 um in v1 = one coordinate unit at the default pitch)."""
        pitch = float(self.hypers.spot_pitch)
        n_cols, n_rows = 78, 64
        row_dy = pitch * np.sqrt(3.0) / 2.0
        coords = [(pitch / 2.0 + ri * row_dy, pitch / 2.0 + (pitch / 2.0 if ri % 2 else 0.0) + ci * pitch)
                  for ri in range(n_rows) for ci in range(n_cols)]
        return np.asarray(coords, dtype=float)

    def _layout(self, grid_side):
        """Spot coordinates for the capture grid: the fixed v1 slide when ``grid_side`` is None,
        else a square hex array over ``[0, grid_side]^2``."""
        return self._visium_v1_layout() if grid_side is None else self._spot_layout(grid_side)

    def _place_section(self, cell_data, spot_coords):
        """Place a thin section of a deme-based tumor onto the fixed spot array.

        :func:`iscc.sample.spatialize` (seeded by ``self.seed``) gives sub-deme cell positions; the
        section is then rotated by ``self.rotation`` degrees about the placement point and that point
        is shifted onto the centre of ``spot_coords``. The placement point is the section centroid, or
        ``self.placement`` — so a section larger than the slide is imaged at a chosen location and
        orientation, not always centred axis-aligned."""
        from ..sample import spatialize
        placed = spatialize(cell_data, section_frac=self.section_frac, seed=self.seed)
        crd = placed["cell_crd"][["row", "col"]].to_numpy(float)
        target = crd.mean(axis=0) if self.placement is None else np.asarray(self.placement, dtype=float)
        if self.rotation:
            th = np.deg2rad(self.rotation)
            rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
            crd = (crd - target) @ rot.T + target
        crd = crd + (spot_coords.mean(axis=0) - target)          # placement point -> slide centre
        placed["cell_crd"] = placed["cell_crd"].assign(row=crd[:, 0], col=crd[:, 1])
        return placed

    def place_grid(self, cell_data, grid_side=None, px=6):
        """Place the fixed spot grid on a section WITHOUT assaying it, and store the placement.

        Lays the slide over the (placed, optionally rotated) section, records which spots cover tissue,
        and renders the section's H&E over its whole extent. Afterwards :meth:`to_anndata` returns a
        standard 10x AnnData of the spot grid on the full H&E — with an ``in_tissue`` flag but ZERO
        expression — ready to plot with squidpy/scanpy to judge the placement before assaying::

            vz.place_grid(section)
            sq.pl.spatial_scatter(vz.to_anndata(), color="in_tissue", img=True)   # check the placement
            vz.run()                                                              # then assay it

        :meth:`run` calls this itself if you did not, and reuses it if you did.

        Parameters
        ----------
        cell_data : dict
            The section to image (e.g. from :meth:`Resection.slice`).
        grid_side : int, optional
            Capture-area side; ``None`` (default) uses the fixed v1 slide.
        px : int, default 6
            Pixels per coordinate unit for the H&E image (only rendered when ``section_frac`` is set).
        """
        from scipy.spatial import cKDTree
        spot_coords = self._layout(grid_side)
        he, scalef = None, 1.0
        if self.section_frac is not None:
            placed = self._place_section(cell_data, spot_coords)
            crd = placed["cell_crd"][["row", "col"]].to_numpy(float)
            # a frame covering BOTH tissue and spot grid, shifted non-negative, for a full-section H&E
            off = np.vstack([crd, spot_coords]).min(axis=0)
            crd, spot_coords = crd - off, spot_coords - off
            placed["cell_crd"] = placed["cell_crd"].assign(row=crd[:, 0], col=crd[:, 1])
            from ..sample import tissue_image
            H = int(np.ceil(max(crd[:, 0].max(), spot_coords[:, 0].max()))) + 1
            W = int(np.ceil(max(crd[:, 1].max(), spot_coords[:, 1].max()))) + 1
            he, scalef = tissue_image(crd, (H, W), px=px)
        else:
            placed = cell_data
            crd = placed["cell_crd"][["row", "col"]].to_numpy(float)
        members = (cKDTree(crd).query_ball_point(spot_coords, float(self.hypers.spot_radius))
                   if len(crd) else [[] for _ in spot_coords])
        self._grid = {
            "placed": placed, "spot_coords": spot_coords,
            "members": [np.asarray(m, dtype=int) for m in members],
            "n_cells": np.array([len(m) for m in members]),
            "he": he, "scalef": float(scalef), "ran": False,
        }
        return self

    def section_image(self, cell_data, grid_side=None, px=14):
        """Render the placed tissue section as an H&E-like image, **without** running the assay.

        Preview the slide's tissue morphology before assaying it. Requires ``section_frac`` to be
        set (otherwise there is no placement to render). The same placement (seed) is used by
        :meth:`run`, so this preview matches the ``img=True`` background of the assayed AnnData.

        Parameters
        ----------
        cell_data : dict
            Per-cell tables (uses ``cell_deme`` + ``cell_crd``).
        grid_side : int, optional
            Capture-area side; ``None`` (default) uses the fixed v1 slide (4,992 spots).
        px : int, default 14
            Pixels per coordinate unit.

        Returns
        -------
        numpy.ndarray, shape (H, W, 3), float32
            The H&E-like tissue-morphology image (see :func:`iscc.sample.tissue_image`).
        """
        from ..sample import tissue_image
        if self.section_frac is None:
            raise ValueError("section_image requires section_frac to be set on the Visium assay")
        spot_coords = self._layout(grid_side)
        extent = (int(np.ceil(spot_coords[:, 0].max())), int(np.ceil(spot_coords[:, 1].max())))
        placed = self._place_section(cell_data, spot_coords)
        img, _ = tissue_image(placed["cell_crd"][["row", "col"]].to_numpy(float), extent, px=px)
        return img

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
    def run(self, cell_data=None, grid_side=None):
        """Assay a 10x Visium spatial-transcriptomics section over the placed spot grid.

        The slide is a **fixed** capture grid laid over the section (see :meth:`place_grid`): it pools
        each spot's cells, applies lateral mRNA diffusion and a smooth spatial capture field, and draws
        per-spot UMI counts. If you called :meth:`place_grid` first, ``run`` reuses that placement;
        otherwise pass the section here and ``run`` places it itself (per ``placement`` / ``rotation``).
        A morphology tissue image is attached to :meth:`to_anndata` for ``img=True`` overlays.

        Parameters
        ----------
        cell_data : dict, optional
            The section to assay; omit only if :meth:`place_grid` was already called. Passing it
            (re-)places the grid.
        grid_side : int, optional
            Capture-area side; ``None`` (default) uses the fixed 10x v1 slide (4,992 spots).

        Returns
        -------
        Visium
            ``self``, with the spot-by-gene UMI matrix in ``spot_counts`` etc. Export with
            :meth:`to_anndata` / ``write``.
        """
        if cell_data is not None or self._grid is None:
            if cell_data is None:
                raise ValueError("run() needs a section (cell_data) unless place_grid() was called first")
            self.place_grid(cell_data, grid_side=grid_side)
        g = self._grid
        placed, spot_coords, members = g["placed"], g["spot_coords"], g["members"]
        cell_exp = placed["cell_exp"]
        genes = cell_exp.columns
        n_genes = len(genes)
        exp_vals = cell_exp.values.astype(float)
        cell_ids = np.asarray(cell_exp.index)
        ctype = placed.get("cell_type")
        clone_labels = ctype.reindex(cell_exp.index).iloc[:, 0].astype(str).values if ctype is not None else None
        n_spots = spot_coords.shape[0]
        n_cells_arr = g["n_cells"]

        # 1-2. spot -> cell aggregation over the pre-placed grid (pool cell_exp; record ground truth)
        raw = np.zeros((n_spots, n_genes))
        member_ids, dominant, clone_fracs = [], [], []
        for s, inb in enumerate(members):
            member_ids.append(cell_ids[inb])
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

        # 4-5. batch realization + pluggable count draw + ambient
        label = self.batch_label if self.batch_label is not None else f"visium{self.seed}"
        base_profile = raw.sum(axis=0)                     # pooled expression -> ambient soup
        self.batch = VisiumBatch(self.hypers, seed=self.seed, label=label).realize(
            genes, base_profile, spot_coords)
        occupied = (n_cells_arr > 0).astype(float)
        lib = self.batch.spot_library(occupied=occupied)
        comp = self.batch.composition(probs)
        counts = self.batch.emit(comp, lib, count_model=self.count_model)
        counts = self.batch.add_ambient(counts, lib).astype(int)

        # -- store outputs (DataFrames + ground truth) ---------------------------------
        spot_names = [f"S{i}" for i in range(n_spots)]
        self.spot_names = spot_names
        self.genes = list(genes)
        self.spot_coords = spot_coords
        self.spot_members = member_ids
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
            [",".join(np.asarray(m).astype(str).tolist()) for m in member_ids],
            index=spot_names, columns=["cell_ids"])
        self._tissue_image, self._tissue_scalef = g["he"], g["scalef"]
        g["ran"] = True
        return self

    # -- outputs -----------------------------------------------------------------------
    def to_anndata(self):
        """Standard 10x AnnData of the spot grid. Works after :meth:`place_grid` (the placed grid on
        the H&E, ``X`` all zero — just the placement) or after :meth:`run` (with UMI counts)."""
        import anndata as ad
        if self._grid is None:
            raise ValueError("call place_grid(section) or run(section) before to_anndata()")
        g = self._grid
        spot_coords, n_cells = g["spot_coords"], g["n_cells"]
        n_spots = spot_coords.shape[0]
        spot_names = [f"S{i}" for i in range(n_spots)]
        label = self.batch_label if self.batch_label is not None else f"visium{self.seed}"
        if g["ran"]:
            genes = list(self.spot_counts.columns)
            X = self.spot_counts.values.astype(np.float32)
            obs = self.obs.copy()
        else:                                               # placed but not assayed -> zero expression
            genes = list(g["placed"]["cell_exp"].columns) if "cell_exp" in g["placed"] else ["_placeholder"]
            X = np.zeros((n_spots, len(genes)), dtype=np.float32)
            obs = pd.DataFrame({"n_cells": n_cells, "row": spot_coords[:, 0], "col": spot_coords[:, 1]},
                               index=pd.Index(spot_names, name="spot"))
        adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=genes))
        # 10x/squidpy convention: obsm['spatial'] is (x, y) = (col, row) in full-res coords, so
        # `sq.pl.spatial_scatter(..., img=True)` overlays the tissue image the right way up. Includes an
        # `in_tissue` flag (spots covering >=1 cell), `total_counts`, and a `uns['spatial']` entry with
        # the spot diameter, so the fixed grid plots directly with squidpy/scanpy either way.
        adata.obsm["spatial"] = spot_coords[:, [1, 0]].astype(float)
        adata.obs["in_tissue"] = (n_cells > 0).astype(int)
        adata.obs["total_counts"] = X.sum(axis=1).astype(float)
        diam = float(2.0 * self.hypers.spot_radius)
        img = g["he"]
        adata.uns["spatial"] = {label: {
            "images": {"hires": img} if img is not None else {},
            "scalefactors": {
                "tissue_hires_scalef": float(g["scalef"]),
                "spot_diameter_fullres": diam,
                "fiducial_diameter_fullres": diam,
            },
        }}
        adata.uns["assay"] = "visium"
        adata.uns["protocol"] = self.protocol
        adata.uns["count_model"] = self.count_model
        adata.uns["batch_seed"] = self.seed
        adata.uns["hyperparams"] = self.hypers.to_dict()
        if g["ran"]:
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
