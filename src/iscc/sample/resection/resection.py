"""Resection — cut a grown tumour (the resected specimen) into the samples the assays take.

A specimen is physically divided by two orthogonal cuts before any assay runs:

* an **in-plane cut** (:meth:`Resection.bisect`) splits the 2-D tissue into disjoint parts — one is
  dissociated for the sequencing assays (bulk DNA / scDNA / scRNA), the rest is kept for imaging;
* a **depth cut** (:meth:`Resection.slice`) thins a part to ~one layer — a thin histology / Visium
  section that keeps the 2-D structure but lowers each deme-column's occupancy (a deme's ``K`` stands
  for the cells in its 3-D column).

The cuts operate on the tumour's per-deme genotype counts and materialise only what each sample needs
(via :meth:`GenotypeTumor.make_cell_data`), so ``Resection`` stays memory-safe at cm-scale — you never
materialise the whole tumour just to sample a piece of it.
"""


class Resection:
    """A resected tumour specimen, cut into the samples the assays take.

    Parameters
    ----------
    tumor : GenotypeTumor
        The grown (count-based) tumour standing in for the resected specimen.

    Examples
    --------
    >>> spec = Resection(tumor)
    >>> cut, remainder = spec.bisect(frac=0.42)          # in-plane cut
    >>> cd = spec.dissociate(cut, max_cells=50000)       # sequencing sample (full depth)
    >>> section = spec.slice(remainder, depth_frac=0.5)  # imaging section (thin slice)
    """

    def __init__(self, tumor):
        self.tumor = tumor

    def bisect(self, frac=0.5, axis="x"):
        """In-plane cut: split the specimen into two disjoint 2-D parts.

        Cuts the primary grid at ``frac`` of its side along ``axis`` and returns the deme indices on
        each side, ready to pass to :meth:`dissociate` / :meth:`slice` (or ``make_cell_data(region=)``).

        Parameters
        ----------
        frac : float, default 0.5
            Fraction of the grid (along ``axis``) that falls in the first part.
        axis : {"x", "y"}, default "x"
            Cut axis: ``"x"`` splits by column, ``"y"`` by row.

        Returns
        -------
        (list of int, list of int)
            ``(part, remainder)`` deme-index lists partitioning the occupied grid.
        """
        if axis not in ("x", "y"):
            raise ValueError("axis must be 'x' or 'y'")
        G = self.tumor.grid_size
        s = int(frac * G)
        part, remainder = [], []
        for r in range(G):
            for c in range(G):
                key = c if axis == "x" else r
                (part if key < s else remainder).append(r * G + c)
        return part, remainder

    def dissociate(self, region=None, max_cells=None):
        """Dissociate a 2-D part into a per-cell table for the sequencing assays.

        Materialises the part at FULL column depth (all ~K cells/deme, subsampled to ``max_cells``);
        the cells lose their spatial position, as in a real dissociation.

        Parameters
        ----------
        region : iterable of int, optional
            Deme indices of the part to dissociate (e.g. the first element of :meth:`bisect`). ``None``
            dissociates the whole specimen.
        max_cells : int, optional
            Cap on materialised cells (a representative subsample above it).

        Returns
        -------
        dict
            The per-cell ``cell_data`` tables (pass to ``bulkDNA`` / ``scDNA`` / ``scRNA``).
        """
        return self.tumor.make_cell_data(region=region, max_cells=max_cells)

    def slice(self, region=None, depth_frac=0.5, max_cells=None):
        """Cut a thin histology / Visium **section**: a depth cut of a 2-D part.

        Keeps ``depth_frac`` of each deme's 3-D column (default half — "take away half"), leaving the
        2-D field intact, so a spatial assay sees the whole structure at a realistic ~one-layer density.

        Parameters
        ----------
        region : iterable of int, optional
            Deme indices of the part to section (e.g. the remainder from :meth:`bisect`). ``None``
            sections the whole specimen.
        depth_frac : float, default 0.5
            Fraction of each deme's column kept in the slice.
        max_cells : int, optional
            Cap on materialised cells (bounds the section at cm-scale).

        Returns
        -------
        dict
            The per-cell ``cell_data`` tables of the thin section (pass to ``Visium``).
        """
        return self.tumor.make_cell_data(region=region, depth_frac=depth_frac, max_cells=max_cells)
