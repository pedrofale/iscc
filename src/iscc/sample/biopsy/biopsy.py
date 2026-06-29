"""F1 — solid + liquid biopsy: spatial region selection over the cell grid.

A `Biopsy` operates on a `cell_data` dict (the per-cell ground truth written by
`isccsim`) and returns the *indices* of the sampled cells plus a per-cell
``region`` label. The per-cell ground truth is never modified — only subset.

Geometries (all defined on the ``cell_crd`` row/col grid):

* **punch**     — a disk (centre + radius); one region ``"punch"``.
* **needle**    — a narrow strip (rectangle of given width along an axis/angle
                  through the tumour); one region ``"needle"``.
* **multiregion** — ``k`` disjoint disks, EACH tagged with its own ``region_i``
                  label. This is the multi-region-heterogeneity substrate for
                  phylogeny: you only see the clones present in each region.
* **liquid**    — a small number of circulating cells (CTCs), biased toward
                  high-dispersal / invasive clones (a per-cell dispersal signal
                  from ``cell_evo``); very low counts; region ``"blood"``.

The geometry (method, centres, radii/width, k) is returned so the caller can
record it in ``sample_meta.yaml``.
"""
import numpy as np
import pandas as pd

# Non-cancer genotype ids are the literal biological types; everything else is a
# cancer clone id (see Tumor.make_cell_data / Cell.set_genotype_id).
NORMAL_TYPES = ("immune", "stromal", "epithelial")

# Per-cell dispersal / invasiveness signals in cell_evo, in order of preference.
DISPERSAL_SIGNALS = ("n_mut_disp", "dispersal_rate")


class Biopsy(object):
    """Spatial region selection over a tumour's ``cell_data``.

    Parameters
    ----------
    cell_data : dict[str, pandas.DataFrame]
        The isccsim ground truth (must include ``cell_crd``; ``cell_type`` and
        ``cell_evo`` enable liquid biopsy).
    rng : numpy.random.Generator, optional
        Seeded generator for reproducibility.
    grid_size : int, optional
        Grid extent. If omitted, bounds are inferred from ``cell_crd``.
    """

    def __init__(self, cell_data, rng=None, grid_size=None):
        self.cell_data = cell_data
        self.rng = rng if rng is not None else np.random.default_rng()
        crd = cell_data["cell_crd"]
        self.index = crd.index
        self.coords = crd[["row", "col"]].to_numpy(dtype=float)
        if grid_size is not None:
            hi = float(grid_size - 1)
            self.bounds = ((0.0, hi), (0.0, hi))
        else:
            self.bounds = (
                (self.coords[:, 0].min(), self.coords[:, 0].max()),
                (self.coords[:, 1].min(), self.coords[:, 1].max()),
            )

    # ------------------------------------------------------------------ helpers
    def _center(self, center):
        """Resolve a centre: explicit (row, col) or the occupied-cell centroid."""
        if center is not None:
            return np.asarray(center, dtype=float)
        return self.coords.mean(axis=0)

    def _extent(self):
        (r0, r1), (c0, c1) = self.bounds
        return max(r1 - r0, c1 - c0)

    # --------------------------------------------------------------- geometries
    def punch_mask(self, center=None, radius=None):
        """Disk of ``radius`` around ``center``. Returns (mask, geometry)."""
        c = self._center(center)
        if radius is None:
            radius = max(1.0, 0.25 * self._extent())
        d2 = ((self.coords - c) ** 2).sum(axis=1)
        mask = d2 <= radius ** 2
        return mask, dict(method="punch", center=c.tolist(), radius=float(radius))

    def needle_mask(self, center=None, width=None, angle=0.0):
        """Narrow strip of ``width`` along ``angle`` (deg) through ``center``.

        The core is the set of cells whose perpendicular distance to the line
        through ``center`` in direction ``angle`` is <= width/2 — an infinite
        strip clipped by the tumour, i.e. a needle core.
        """
        c = self._center(center)
        if width is None:
            width = max(1.0, 0.1 * self._extent())
        theta = np.deg2rad(angle)
        direction = np.array([np.cos(theta), np.sin(theta)])
        rel = self.coords - c
        # |rel x direction| = perpendicular distance to the line.
        perp = np.abs(rel[:, 0] * direction[1] - rel[:, 1] * direction[0])
        mask = perp <= width / 2.0
        return mask, dict(method="needle", center=c.tolist(),
                          width=float(width), angle=float(angle))

    def multiregion(self, k=3, radius=None, max_tries=2000):
        """``k`` disjoint disks, each tagged with its own ``region_i`` label.

        Centres are drawn from occupied cells and kept pairwise >= 2*radius
        apart, so the disks never overlap. Returns (mask, region_array,
        geometry); ``region_array`` is full-length over ``self.index`` with
        ``""`` for unsampled cells.
        """
        if radius is None:
            radius = max(1.0, 0.12 * self._extent())
        occ = self.coords
        centers = []
        tries = 0
        while len(centers) < k and tries < max_tries:
            tries += 1
            cand = occ[self.rng.integers(0, len(occ))]
            if all(np.hypot(*(cand - cc)) >= 2 * radius for cc in centers):
                centers.append(cand)
        region = np.full(len(occ), "", dtype=object)
        for ri, cc in enumerate(centers):
            d2 = ((occ - cc) ** 2).sum(axis=1)
            inside = (d2 <= radius ** 2) & (region == "")  # disjoint by construction
            region[inside] = f"region_{ri}"
        mask = region != ""
        geom = dict(method="multiregion", k=len(centers),
                    centers=[cc.tolist() for cc in centers], radius=float(radius))
        return mask, region, geom

    def liquid_mask(self, n=20, signal=None):
        """Small circulating-cell sample biased toward high-dispersal clones.

        CTCs are tumour cells, so the pool is restricted to cancer clones (falls
        back to all cells if none). Sampling weight comes from a per-cell
        dispersal signal in ``cell_evo`` (``n_mut_disp`` then ``dispersal_rate``);
        if absent, degrades gracefully to a uniform draw.
        """
        ct = self.cell_data.get("cell_type")
        if ct is not None:
            types = ct["cell_id"].astype(str).to_numpy()
            pool = np.where(~np.isin(types, NORMAL_TYPES))[0]
        else:
            pool = np.array([], dtype=int)
        if pool.size == 0:
            pool = np.arange(len(self.index))
        weights, used = self._dispersal_weights(pool, signal)
        n = int(min(n, pool.size))
        chosen_local = self.rng.choice(pool, size=n, replace=False, p=weights)
        mask = np.zeros(len(self.index), dtype=bool)
        mask[chosen_local] = True
        return mask, dict(method="liquid", n=n, dispersal_signal=used)

    def _dispersal_weights(self, pool, signal=None):
        """Normalized sampling weights over ``pool`` from a dispersal signal.

        Returns (weights, signal_name); weights is None (=> uniform) when no
        usable signal is present.
        """
        evo = self.cell_data.get("cell_evo")
        if evo is None:
            return None, None
        cols = [signal] if signal else DISPERSAL_SIGNALS
        for col in cols:
            if col and col in evo.columns:
                vals = np.clip(evo.iloc[pool][col].to_numpy(dtype=float), 0.0, None)
                if vals.sum() <= 0:
                    return None, col
                # additive smoothing: zero-dispersal cells stay possible but rare.
                alpha = 0.05 * vals[vals > 0].mean()
                w = vals + alpha
                return w / w.sum(), col
        return None, None

    # ---------------------------------------------------------------- dispatch
    def sample(self, biopsy_type="punch", center=None, radius=None, width=None,
               angle=0.0, n_regions=3, n_liquid=20, signal=None):
        """Run a biopsy. Returns (chosen_index, region_series, geometry).

        ``region_series`` is a pandas Series of the ``region`` label indexed by
        the chosen cell names (the contents of ``cell_region.csv``).
        """
        if biopsy_type == "punch":
            mask, geom = self.punch_mask(center=center, radius=radius)
            region = np.where(mask, "punch", "")
        elif biopsy_type == "needle":
            mask, geom = self.needle_mask(center=center, width=width, angle=angle)
            region = np.where(mask, "needle", "")
        elif biopsy_type == "multiregion":
            mask, region, geom = self.multiregion(k=n_regions, radius=radius)
        elif biopsy_type == "liquid":
            mask, geom = self.liquid_mask(n=n_liquid, signal=signal)
            region = np.where(mask, "blood", "")
        else:
            raise ValueError(f"Unknown biopsy_type {biopsy_type!r}")
        chosen = self.index[mask]
        region_series = pd.Series(np.asarray(region)[mask], index=chosen, name="region")
        return chosen, region_series, geom
