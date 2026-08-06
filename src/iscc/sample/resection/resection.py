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

Taking a sample does not change the tumour: :meth:`dissociate` and :meth:`slice` RETURN their
per-cell tables and leave ``tumor.cell_data`` exactly as they found it, so cutting a section never
silently redefines what ``to_anndata(tumor)`` / ``tumor.write_h5ad(...)`` export. Pass
``install=True`` when you *want* the sample to become the tumour's own table — that is what the
tumour-level helpers that read ``tumor.cell_data`` (``clones_from_clades``, ``to_anndata(tumor)``)
then see.
"""


class _CellDataGuard:
    """Context manager that leaves ``tumor.cell_data`` as it found it.

    ``make_cell_data`` both returns the sample AND installs it on the tumour. A sample is not the
    tumour, so :class:`Resection` runs it inside this guard: the returned tables are unaffected
    (``make_cell_data`` rebinds ``cell_data`` to a fresh dict rather than mutating the old one), and
    the tumour keeps whatever it had before the cut.
    """

    __slots__ = ("tumor", "_had", "_saved")

    def __init__(self, tumor):
        self.tumor = tumor

    def __enter__(self):
        self._had = hasattr(self.tumor, "cell_data")
        self._saved = getattr(self.tumor, "cell_data", None)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._had:
            self.tumor.cell_data = self._saved
        else:                                    # nothing was installed before: leave it that way
            try:
                del self.tumor.cell_data
            except AttributeError:
                pass
        return False


class Resection:
    """A resected tumour specimen, cut into the samples the assays take.

    Parameters
    ----------
    tumor : GenotypeTumor
        The grown (count-based) tumour standing in for the resected specimen.
    compartment : {"primary", "met", "both"}, default "primary"
        Which grid the specimen is taken from. ``"primary"`` (default) resects only the primary
        tumour; ``"met"`` resects only the metastatic deposit (errors if the tumour has no
        metastasis); ``"both"`` resects both grids (falls back to the primary alone when there is
        no metastasis). This sets the deme set that :meth:`bisect` partitions and that
        :meth:`dissociate` / :meth:`slice` default to when ``region`` is ``None``.

    Examples
    --------
    >>> spec = Resection(tumor)                          # primary by default
    >>> cut, remainder = spec.bisect(frac=0.42)          # in-plane cut
    >>> cd = spec.dissociate(cut, max_cells=50000)       # sequencing sample (full depth)
    >>> section = spec.slice(remainder, depth_frac=0.5)  # imaging section (thin slice)
    >>> met = Resection(tumor, compartment="met").dissociate()   # the metastasis instead

    Notes
    -----
    Both samplers leave ``tumor.cell_data`` untouched — take as many cuts as you like without
    redefining what the tumour itself exports.
    """

    def __init__(self, tumor, compartment="primary"):
        if compartment not in ("primary", "met", "both"):
            raise ValueError("compartment must be 'primary', 'met', or 'both'")
        self.tumor = tumor
        self.compartment = compartment
        self._grids = self._compartment_grids()

    def _compartment_grids(self):
        """The ``(base_deme_index, grid_side)`` of each grid in the resected compartment.

        Primary demes occupy indices ``0 .. grid_size**2 - 1``; the metastasis (when present) is a
        ``met_grid_size`` lattice appended at ``n_primary_demes``. Returns one entry per included
        grid so :meth:`bisect` can cut each in-plane and :meth:`dissociate` / :meth:`slice` can
        enumerate their demes.
        """
        t = self.tumor
        has_met = bool(getattr(t, "_met_enabled", False)) and int(getattr(t, "met_grid_size", 0)) > 0
        grids = []
        if self.compartment in ("primary", "both"):
            grids.append((0, int(t.grid_size)))
        if self.compartment in ("met", "both"):
            if has_met:
                grids.append((int(t.n_primary_demes), int(t.met_grid_size)))
            elif self.compartment == "met":
                raise ValueError("compartment='met' but this tumour has no metastasis "
                                 "(met_grid_size=0)")
        return grids

    def _all_demes(self):
        """Every deme index in the resected compartment (both grids for ``compartment='both'``)."""
        out = []
        for base, side in self._grids:
            out.extend(range(base, base + side * side))
        return out

    def bisect(self, frac=0.5, axis="x"):
        """In-plane cut: split the specimen into two disjoint 2-D parts.

        Cuts each grid of the resected :attr:`compartment` at ``frac`` of its side along ``axis`` and
        returns the deme indices on each side, ready to pass to :meth:`dissociate` / :meth:`slice`
        (or ``make_cell_data(region=)``). For ``compartment="both"`` each grid is cut at the same
        fraction and the parts are unioned.

        Parameters
        ----------
        frac : float, default 0.5
            Fraction of the grid (along ``axis``) that falls in the first part.
        axis : {"x", "y"}, default "x"
            Cut axis: ``"x"`` splits by column, ``"y"`` by row.

        Returns
        -------
        (list of int, list of int)
            ``(part, remainder)`` deme-index lists partitioning the compartment.
        """
        if axis not in ("x", "y"):
            raise ValueError("axis must be 'x' or 'y'")
        part, remainder = [], []
        for base, G in self._grids:
            s = int(frac * G)
            for r in range(G):
                for c in range(G):
                    key = c if axis == "x" else r
                    (part if key < s else remainder).append(base + r * G + c)
        return part, remainder

    def _materialise(self, install, **kwargs):
        """``make_cell_data`` for a sample: return its tables, leave the tumour's own alone."""
        if install:
            return self.tumor.make_cell_data(**kwargs)
        with _CellDataGuard(self.tumor):
            return self.tumor.make_cell_data(**kwargs)

    def dissociate(self, region=None, max_cells=None, install=False):
        """Dissociate a 2-D part into a per-cell table for the sequencing assays.

        Materialises the part at FULL column depth (all ~K cells/deme, subsampled to ``max_cells``);
        the cells lose their spatial position, as in a real dissociation.

        Parameters
        ----------
        region : iterable of int, optional
            Deme indices of the part to dissociate (e.g. the first element of :meth:`bisect`). ``None``
            dissociates the whole resected :attr:`compartment` (the primary by default).
        max_cells : int, optional
            Cap on materialised cells (a representative subsample above it).
        install : bool, default False
            Also make this sample the tumour's own ``cell_data``. Off by default: a sample is not
            the tumour, and installing it would silently redefine what ``to_anndata(tumor)`` and
            ``tumor.write_h5ad(...)`` export. Turn it on when you want the tumour-level helpers
            that read ``tumor.cell_data`` (``clones_from_clades``, ``to_anndata(tumor)``) to work
            on exactly these cells.

        Returns
        -------
        dict
            The per-cell ``cell_data`` tables (pass to ``bulkDNA`` / ``scDNA`` / ``scRNA``).
        """
        if region is None:
            region = self._all_demes()
        return self._materialise(install, region=region, max_cells=max_cells)

    def slice(self, region=None, depth_frac=0.5, max_cells=None, install=False):
        """Cut a thin histology / Visium **section**: a depth cut of a 2-D part.

        Keeps ``depth_frac`` of each deme's 3-D column (default half — "take away half"), leaving the
        2-D field intact, so a spatial assay sees the whole structure at a realistic ~one-layer density.

        Parameters
        ----------
        region : iterable of int, optional
            Deme indices of the part to section (e.g. the remainder from :meth:`bisect`). ``None``
            sections the whole resected :attr:`compartment` (the primary by default).
        depth_frac : float, default 0.5
            Fraction of each deme's column kept in the slice. Note this thins the same 3-D column
            that ``Visium(section_frac=...)`` thins, so the two **compose multiplicatively**: keep
            one of them at ``1.0`` (the tutorials slice here and leave ``section_frac=1.0``).
        max_cells : int, optional
            Cap on materialised cells (bounds the section at cm-scale).
        install : bool, default False
            Also make this section the tumour's own ``cell_data`` — see :meth:`dissociate`. Off by
            default, so taking a section never changes what the tumour itself exports.

        Returns
        -------
        dict
            The per-cell ``cell_data`` tables of the thin section (pass to ``Visium``).
        """
        if region is None:
            region = self._all_demes()
        return self._materialise(install, region=region, depth_frac=depth_frac,
                                 max_cells=max_cells)
