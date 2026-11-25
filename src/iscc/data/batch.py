"""Assay batch-effect model.

A **batch** is one realization of an assay instance — a single reaction / lane / plate
keyed by a seed. The *hyper-parameters* (`RNABatchHyperParams`) are protocol-typical
magnitudes; one seed draws a concrete *realization* (`Batch`) whose technical signature
is applied consistently across every cell of that instance and is reproducible from the
seed. Two `Batch`es with identical hyper-parameters but different seeds = same biology,
two technical signatures.

The hyper-parameter names here are exactly what `estimate()` fits from real data, so they
are named for that purpose:

    sigma_batch       per-gene batch-factor LogNormal sd  (Splatter `batch.facScale`)
    mu_lib            mean library size (counts / cell)    -> sequencing depth
    sigma_lib         per-cell library-size LogNormal sd
    dispersion        NB overdispersion phi  (var = mu + dispersion * mu^2)
    ambient_frac      fraction of the library that is ambient ("soup") contamination
    doublet_rate      fraction of barcodes that are doublets (two cells merged)
    dropout_mid       logistic dropout midpoint in expected-count space (0 disables)
    dropout_shape     logistic dropout steepness
    well_sigma        per-well LogNormal sd  (plate protocols sort one cell per well)
    n_wells           wells per plate (0 = not a plate protocol, e.g. droplet)
    n_plates          plates the cells are spread over (0 = as many as they need)
    plate_sigma       per-PLATE library LogNormal sd (cells on a plate share a depth offset)
    depth_batch_sigma per-batch depth-shift LogNormal sd (so batches differ in depth)
    kappa             Dirichlet concentration (only for the `dm` count model)

The **count emission is a pluggable step**: everything above (biology -> library ->
batch) is shared and only the final draw is swapped via ``COUNT_MODELS``.
"""
from dataclasses import dataclass, asdict
from typing import Optional

import math
import re
import warnings

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------------------
# Plate / well layout — the physical container of a plate-based protocol (Smart-seq3).
#
# Plate protocols index-sort ONE cell per well, so the well is the cell's address and the
# plate is the batch-within-a-batch: every well on a plate goes through the same lysis /
# RT / cDNA mix and the same library pooling, so its cells share a depth offset. That makes
# depth PLATE-NESTED (plate effect x well effect), which is what `Batch.library_factors`
# below draws. Layout here is purely deterministic bookkeeping (no RNG): where each cell
# sits. Droplet protocols have no wells (`n_wells = 0`) and skip all of it.
# --------------------------------------------------------------------------------------
_PLATE_ASPECT = 1.5   # standard microplates are 2:3 — 96 = 8x12, 384 = 16x24, 1536 = 32x48


def well_grid(n_wells):
    """(n_rows, n_cols) of a microplate with `n_wells` wells, using the standard 2:3 aspect.

    Reproduces every standard format (6, 12, 24, 48, 96, 384, 1536 wells). A non-standard
    well count is rounded up to the smallest grid of that aspect that holds it, so the
    trailing positions of the last row are simply unused.
    """
    n_wells = int(n_wells)
    if n_wells <= 0:
        return 0, 0
    n_rows = max(1, int(round(math.sqrt(n_wells / _PLATE_ASPECT))))
    n_cols = n_wells // n_rows
    if n_rows * n_cols != n_wells:
        n_cols = int(math.ceil(n_wells / n_rows))
    return n_rows, n_cols


def row_label(i):
    """Plate row letter for 0-based row `i`: 0 -> "A", 25 -> "Z", 26 -> "AA" (1536-well)."""
    i = int(i)
    out = ""
    while True:
        out = chr(ord("A") + i % 26) + out
        i = i // 26 - 1
        if i < 0:
            return out


@dataclass
class PlateLayout:
    """Where each cell physically sits: its plate, its well, and the well's row/column.

    Assignment is deterministic (no RNG). By default the cells are split as evenly as possible
    across the plates, and within a plate the wells are filled in reading order (A01, A02, ...
    then B01). Because one cell occupies one well, `n_plates * n_wells` is a hard capacity —
    see [`assign_plates`][iscc.data.batch.assign_plates]. A caller-given plate map replaces
    that split (`assign_plates_from_map`), and then the plates carry the map's own names.

    Attributes
    ----------
    n_cells : int
        Number of cells laid out.
    n_wells : int
        Wells per plate.
    n_plates : int
        Number of plates actually used (may exceed the requested count — see `assign_plates`).
    n_rows, n_cols : int
        Well-grid shape of one plate (e.g. 16 x 24 for a 384-well plate).
    plate : ndarray of int
        0-based plate index per cell.
    well : ndarray of int
        0-based well index within that plate.
    row, col : ndarray of int
        0-based row / column of that well on the plate.
    labels : list of str, optional
        Name of each plate, by plate index, when the layout came from a plate map; None for
        the default split, whose plates are named by index ("P0", "P1", ...).
    """
    n_cells: int
    n_wells: int
    n_plates: int
    n_rows: int
    n_cols: int
    plate: np.ndarray
    well: np.ndarray
    row: np.ndarray
    col: np.ndarray
    labels: list = None

    def well_ids(self):
        """Human-readable well names ("A01" ... "P24")."""
        return np.array([f"{row_label(r)}{c + 1:02d}" for r, c in zip(self.row, self.col)])

    def plate_ids(self, label):
        """Plate names qualified by the batch `label` ("batch0_P1").

        Two batches are two physical plate runs, so qualifying keeps their plates distinct
        when several batches are concatenated.
        """
        if self.labels is not None:
            return np.array([f"{label}_{self.labels[p]}" for p in self.plate])
        return np.array([f"{label}_P{p}" for p in self.plate])

    def to_dict(self):
        out = dict(n_plates=int(self.n_plates), n_wells=int(self.n_wells),
                   n_rows=int(self.n_rows), n_cols=int(self.n_cols))
        if self.labels is not None:
            out["plate_labels"] = list(self.labels)
        return out


def assign_plates(n_cells, n_wells, n_plates=0):
    """Lay `n_cells` cells out on plates of `n_wells` wells, one cell per well.

    Parameters
    ----------
    n_cells : int
        Cells to place.
    n_wells : int
        Wells per plate (e.g. 384). Must be positive.
    n_plates : int, default 0
        How many plates to spread the cells over; 0 means "as many as they need".

    Returns
    -------
    PlateLayout
        Deterministic per-cell plate / well placement.

    Notes
    -----
    **Capacity.** One cell occupies one well, so at most `n_plates * n_wells` cells fit. If
    more cells are requested the plate count is GROWN to the number actually needed (rather
    than raising), with a warning naming the old and new counts — running more plates is what
    a wet lab would do. The realized count is `PlateLayout.n_plates`.
    """
    n_cells = int(n_cells)
    n_wells = int(n_wells)
    requested = int(n_plates)
    if n_wells <= 0:
        raise ValueError(f"n_wells must be positive to lay cells out on plates, got {n_wells}")
    if n_cells < 0:
        raise ValueError(f"n_cells must be non-negative, got {n_cells}")
    needed = int(math.ceil(n_cells / n_wells))
    realized = max(requested, needed, 1)
    if 0 < requested < needed:
        warnings.warn(
            f"{n_cells} cells do not fit on {requested} plate(s) of {n_wells} wells "
            f"({requested * n_wells} wells, one cell per well); growing to {realized} plates.",
            stacklevel=2,
        )
    # split the cells as evenly as possible over the plates, then fill wells in reading order
    per_plate = np.full(realized, n_cells // realized, dtype=int)
    per_plate[: n_cells % realized] += 1
    plate = np.repeat(np.arange(realized), per_plate)
    well = np.concatenate([np.arange(c, dtype=int) for c in per_plate])
    n_rows, n_cols = well_grid(n_wells)
    return PlateLayout(
        n_cells=n_cells, n_wells=n_wells, n_plates=realized, n_rows=n_rows, n_cols=n_cols,
        plate=plate.astype(int), well=well.astype(int),
        row=(well // n_cols).astype(int), col=(well % n_cols).astype(int),
    )


_WELL_NAME = re.compile(r"^([A-Za-z]+)(\d+)$")


def _plate_name(label):
    """A plate label as a name: integers become "P<n>" (like the default plates), the rest str."""
    if isinstance(label, (int, np.integer)) and not isinstance(label, bool):
        return f"P{label}"
    return str(label)


def _well_index(well, n_cols):
    """0-based well index from an int (already an index) or a well name ("A01", "P24")."""
    if isinstance(well, (int, np.integer)) and not isinstance(well, bool):
        return int(well)
    m = _WELL_NAME.match(str(well))
    if m is None:
        raise ValueError(f"well {well!r} is neither a 0-based index nor a well name like 'A01'")
    row = 0
    for ch in m.group(1).upper():                # inverse of `row_label`: A -> 0, AA -> 26
        row = row * 26 + ord(ch) - ord("A") + 1
    col = int(m.group(2)) - 1
    if not 0 <= col < n_cols:
        raise ValueError(f"well {well!r}: column {col + 1} is not on a plate with {n_cols} columns")
    return (row - 1) * n_cols + col


def assign_plates_from_map(cells, plate_map, n_wells, n_plates=0):
    """Lay `cells` out on plates by a caller-given plate map, one cell per well.

    Parameters
    ----------
    cells : array-like
        Ids of the cells to place, in assay order.
    plate_map : dict or pandas.Series
        Keyed by cell id; must cover every cell in `cells` and may hold more. Each value is
        either a ``(plate, well)`` pair — the well an int (0-based, reading order: A01 = 0,
        A02 = 1, ...) or a name such as ``"A01"`` — or a plate label alone, in which case the
        wells of each plate are filled in reading order, in the order the cells come in
        `cells`. All values take one of the two forms.
    n_wells : int
        Wells per plate. Must be positive.
    n_plates : int, default 0
        The map decides the plates, so this must be 0 or exactly the number of plates the map
        names; any other count is an error.

    Returns
    -------
    PlateLayout
        The placement. Plates are indexed in sorted label order over the WHOLE map, so a plate
        keeps its index in every batch that draws cells from the same map, and they are named
        by their labels (an integer label ``n`` is named ``"P<n>"``).

    Notes
    -----
    **Capacity.** One cell occupies one well, so a plate takes at most `n_wells` of the cells
    being placed and no two of them may share a well. Unlike `assign_plates` this raises
    rather than growing the plate count: the map names the plates, so there is none to add.
    """
    n_wells = int(n_wells)
    if n_wells <= 0:
        raise ValueError(f"n_wells must be positive to lay cells out on plates, got {n_wells}")
    if isinstance(plate_map, pd.DataFrame) or not isinstance(plate_map, (dict, pd.Series)):
        raise TypeError("plate_layout must be a dict or pandas Series keyed by cell id, got "
                        f"{type(plate_map).__name__}")
    pmap = pd.Series(plate_map, dtype=object)
    if not pmap.index.is_unique:
        raise ValueError("plate_layout lists a cell more than once")
    paired = np.array([isinstance(v, (tuple, list)) for v in pmap.values], dtype=bool)
    if paired.any() and not paired.all():
        raise ValueError("plate_layout mixes (plate, well) pairs with bare plate labels; use one")
    paired = bool(paired.size) and bool(paired.all())
    if paired and any(len(v) != 2 for v in pmap.values):
        raise ValueError("plate_layout pairs must be (plate, well)")
    plate_labels = [v[0] for v in pmap.values] if paired else list(pmap.values)
    codes, uniques = pd.factorize(pd.Series(plate_labels, dtype=object), sort=True)
    if (codes < 0).any():
        raise ValueError("plate_layout has a missing plate label")
    labels = [_plate_name(u) for u in uniques]
    if len(set(labels)) != len(labels):
        raise ValueError(f"plate labels {list(uniques)} collide once named as {labels}")
    if int(n_plates) not in (0, len(labels)):
        raise ValueError(f"n_plates={int(n_plates)} contradicts plate_layout, which names "
                         f"{len(labels)} plate(s)")

    cells = np.asarray(cells)
    missing = ~pd.Index(cells).isin(pmap.index)
    if missing.any():
        raise ValueError(f"plate_layout has no plate for {int(missing.sum())} assayed cell(s), "
                         f"e.g. {list(cells[missing][:3])}")
    pos = pmap.index.get_indexer(cells)
    plate = codes[pos].astype(int)
    n_rows, n_cols = well_grid(n_wells)
    if paired:
        well = np.array([_well_index(pmap.values[i][1], n_cols) for i in pos], dtype=int)
        bad = (well < 0) | (well >= n_wells)
        if bad.any():
            i = int(np.flatnonzero(bad)[0])
            raise ValueError(f"cell {cells[i]!r}: well {pmap.values[pos[i]][1]!r} is not on a "
                             f"{n_wells}-well plate")
        taken = pd.Series(list(zip(plate, well))).duplicated(keep=False).to_numpy()
        if taken.any():
            i = np.flatnonzero(taken)[:2]
            w = int(well[i[0]])
            raise ValueError(f"cells {list(cells[i])} share plate {labels[plate[i[0]]]} well "
                             f"{row_label(w // n_cols)}{w % n_cols + 1:02d} (one cell per well)")
    else:
        sizes = np.bincount(plate, minlength=len(labels))
        if sizes.max(initial=0) > n_wells:
            p = int(sizes.argmax())
            raise ValueError(f"plate {labels[p]} holds {int(sizes[p])} of the cells but has "
                             f"{n_wells} wells (one cell per well)")
        well = pd.Series(plate).groupby(plate).cumcount().to_numpy(dtype=int)
    return PlateLayout(
        n_cells=len(cells), n_wells=n_wells, n_plates=len(labels), n_rows=n_rows, n_cols=n_cols,
        plate=plate, well=well, row=(well // n_cols).astype(int), col=(well % n_cols).astype(int),
        labels=labels,
    )


# --------------------------------------------------------------------------------------
# Pluggable count-emission step.
#
# Each emitter has the signature  emit(rng, comp, lib, batch) -> counts (n_cells, n_genes)
# where `comp` is the per-cell normalized composition (rows sum to 1, already carrying the
# per-gene batch factor beta and the biological signal lambda) and `lib` is the per-cell
# library size. Keeping this signature shared means a new count model is a drop-in, not a
# rewrite: the biology->library->batch pipeline never changes.
# --------------------------------------------------------------------------------------
def _emit_nb(rng, comp, lib, batch):
    """Default `count_model="nb"`: y_cg ~ NB(mu = lib_c * comp_cg, phi_b).

    Gamma-Poisson mixture so var = mu + phi*mu^2 (the field-standard Splatter/DESeq2/edgeR
    parameterization, the one `estimate()` fits).
    """
    mu = np.maximum(lib[:, None] * comp, 0.0)
    phi = batch.dispersion
    if phi <= 0:
        return rng.poisson(mu)
    shape = 1.0 / phi
    gamma = rng.gamma(shape=shape, scale=mu * phi)
    return rng.poisson(gamma)


def _emit_dm(rng, comp, lib, batch):
    """`count_model="dm"` (Dirichlet-Multinomial): a fixed library total with gene-gene
    competition for reads (the compositional alternative to the independent-per-gene NB).

    The shared pipeline already produces exactly the two pieces this needs — the per-cell
    composition `comp` (= normalized beta_gb * lambda_cg) and the per-cell library `lib`:

        for each cell c:
            N_c = round(lib_c)                              # fixed total
            p_c ~ Dirichlet(kappa_b * comp_c)              # proportions
            y_c ~ Multinomial(N_c, p_c)

    `kappa` (= `RNABatchHyperParams.kappa` / `VisiumBatchHyperParams.kappa`) is the Dirichlet
    concentration: large kappa -> p≈comp (multinomial/Poisson-like); small kappa -> lumpy,
    over-dispersed proportions. NB is the default for scRNA (the model `estimate()` fits); DM is the
    default-eligible model for the Visium assay (compositional per-spot capture).
    """
    comp = np.asarray(comp, dtype=float)
    N = np.round(np.asarray(lib, dtype=float)).astype(np.int64)
    kappa = float(batch.h.kappa)
    out = np.zeros(comp.shape, dtype=np.int64)
    for c in range(comp.shape[0]):
        s = comp[c].sum()
        if N[c] <= 0 or s <= 0:
            continue
        alpha = np.maximum(kappa * comp[c], 1e-12)
        p = rng.dirichlet(alpha)
        out[c] = rng.multinomial(N[c], p)
    return out


COUNT_MODELS = {"nb": _emit_nb, "dm": _emit_dm}


@dataclass
class RNABatchHyperParams:
    """The scRNA assay's technical parameters — a defaults container.

    Holds the protocol-typical magnitudes of the scRNA count model: the batch and
    library-size scales, the NB overdispersion and logistic dropout curve, and the
    ambient / doublet contamination rates. The [`scRNA`][iscc.data.scRNA] assay builds
    one from its protocol preset and lets you override any field.
    [`estimate_rna`][iscc.data.estimate_rna] fits the library size and the dispersion from real
    data, the batch spreads when the data hold several batches, and the dropout curve only with
    ``fit_dropout=True``.

    Attributes
    ----------
    protocol : str
        Protocol preset the magnitudes come from (e.g. "10x", "smartseq3").
    sigma_batch : float
        Per-gene batch-factor LogNormal sd; beta_gb ~ LogNormal(0, sigma_batch^2), shared
        across every cell of the batch.
    mu_lib : float
        Mean library size (counts / cell) — the sequencing depth.
    sigma_lib : float
        Per-cell library-size LogNormal sd.
    dispersion : float
        NB overdispersion phi (var = mu + phi*mu^2); used only by the "nb" count model
        (phi <= 0 collapses to Poisson).
    ambient_frac : float
        Fraction of the library that is ambient ("soup") contamination. The soup is a share of
        the library, not counts on top of it, so the mean library stays ``mu_lib``.
    doublet_rate : float
        Fraction of barcodes that are doublets (two cells merged).
    dropout_mid : float
        Logistic dropout midpoint in expected-count space: the expected count of a gene in the
        droplet, ambient molecules included (0, the default, disables dropout).
    dropout_shape : float
        Logistic dropout steepness (active only when dropout_mid > 0).
    well_sigma : float
        Per-well library-size LogNormal sd. Plate protocols (Smart-seq3) sort one cell per
        well, so this is both the per-well and the per-cell term; 0 disables it.
    n_wells : int
        Wells per plate (384 for Smart-seq3). 0 marks a protocol with no plates (droplet),
        which turns the whole plate layer off.
    n_plates : int
        How many plates the assayed cells are spread over; 0 means "as many as they need".
        Grown automatically if the cells do not fit (one cell per well).
    plate_sigma : float
        Per-PLATE library-size LogNormal sd: every cell on a plate shares one depth offset,
        so depth is plate-nested (plate effect x well effect). 0 disables the plate effect
        and leaves the layout as pure bookkeeping.
    depth_batch_sigma : float
        Per-batch depth-shift LogNormal sd (so batches differ in depth).
    kappa : float
        Dirichlet concentration for the "dm" count model only (large kappa ->
        multinomial/Poisson-like, small kappa -> lumpy over-dispersed proportions).
    content_exponent : float
        How the library of a cell or a spot follows the RNA content of the cell or the spot
        ([`cell_rna_content`][iscc.data.cell_rna_content]): the library is multiplied by the
        content raised to this power, rescaled so that the mean depth of the batch is kept. 1, set by
        both scRNA presets, makes the depth proportional to it; 0 (the field's default, which the
        imaging assay keeps) leaves the content out.
    """
    # mu_lib / sigma_lib CALIBRATED to real 10x PBMC3k (`estimate_rna(sc.datasets.pbmc3k())`) — the
    # scale-free library-size magnitudes transfer. `dispersion` is the NB dispersion WITHIN cell types
    # of PBMC3k (0.345 by `estimate_rna`; pooling all cells gives 0.61, which also holds the variance
    # between types, and a tumour draws its own cell types on top). Dropout is off: within types the
    # NB alone reproduces the PBMC3k zero fraction at every mean, so UMI counts leave no zeros for a
    # dropout curve to explain (Svensson 2020). `kappa` is the PBMC3k fit of the 10x preset (see rna.py).
    protocol: str = "10x"
    sigma_batch: float = 0.10
    mu_lib: float = 2400.0      # real PBMC3k ~2,367 UMIs/cell
    sigma_lib: float = 0.43     # real PBMC3k 0.431
    dispersion: float = 0.35
    ambient_frac: float = 0.05
    doublet_rate: float = 0.05
    dropout_mid: float = 0.0
    dropout_shape: float = 1.0
    well_sigma: float = 0.0
    n_wells: int = 0            # 0 = no plates (droplet); Smart-seq3 uses 384
    n_plates: int = 0           # 0 = as many plates as the cells need
    plate_sigma: float = 0.0    # 0 = no shared plate depth offset (default: off)
    depth_batch_sigma: float = 0.05
    kappa: float = 1500.0
    content_exponent: float = 0.0

    def to_dict(self):
        return asdict(self)


class Batch:
    """One seeded realization of an assay instance (a reaction / lane / plate).

    Draws and stores the technical state shared across all cells of the batch (per-gene
    factor, realized depth, dispersion, ambient soup profile) and exposes the per-cell
    operations used by the modality assay (library sizes, composition, count emission,
    ambient, dropout, doublets). All draws come from a single seeded RNG, so the whole
    technical signature is reproducible from `seed` and differs between seeds.
    """

    def __init__(self, hypers: RNABatchHyperParams, seed=0, label=None):
        self.h = hypers
        self.seed = int(seed)
        self.label = str(label) if label is not None else f"batch{self.seed}"
        self.rng = np.random.default_rng(self.seed)
        self._realized = False
        self.plate_factor = None      # per-plate depth offsets, set by `library_factors`

    @property
    def dispersion(self):
        return self._phi

    def realize(self, genes, base_profile):
        """Draw the batch-level technical state. `base_profile` = pooled expression (soup)."""
        n = len(genes)
        self.genes = list(genes)
        # per-gene batch factor beta_gb ~ LogNormal(0, sigma_batch^2), shared across cells
        self.beta = self.rng.lognormal(mean=0.0, sigma=self.h.sigma_batch, size=n)
        # per-batch realized depth mu_lib,b (so two batches differ in depth)
        self.mu_lib_b = float(self.h.mu_lib * self.rng.lognormal(0.0, self.h.depth_batch_sigma))
        # per-batch dispersion phi_b
        self._phi = float(self.h.dispersion)
        # ambient ("soup") profile for this batch: normalized pooled expression
        amb = np.asarray(base_profile, dtype=float)
        tot = amb.sum()
        self.ambient_profile = amb / tot if tot > 0 else np.full(n, 1.0 / n)
        self._realized = True
        return self

    def scale_by_content(self, lib, content, mask=None):
        """Libraries times (RNA content / its mean) ** ``content_exponent``, rescaled so that the mean
        over ``mask`` (default every entry) is kept; an entry without content gets none. Returns
        ``lib`` unchanged when the exponent is 0 or ``content`` is None."""
        a = float(getattr(self.h, "content_exponent", 0.0) or 0.0)
        if a == 0.0 or content is None:
            return lib
        c = np.asarray(content, dtype=float)
        m = (np.ones(len(c), bool) if mask is None else np.asarray(mask, bool)) & (c > 0)
        if not m.any():
            return lib
        f = np.zeros(len(c))
        f[c > 0] = (c[c > 0] / c[m].mean()) ** a
        f[m] /= f[m].mean()
        return np.asarray(lib, dtype=float) * f

    def library_factors(self, n_cells, plates=None):
        """Per-cell library size ell_c ~ LogNormal(log mu_lib,b, sigma_lib^2).

        Plate protocols nest two further multipliers inside the batch:

        * a per-WELL term (`well_sigma`) — one cell per well, so it is also per-cell;
        * a per-PLATE term (`plate_sigma`), drawn ONCE per plate and shared by every cell on
          it, so cells prepared together carry the same depth offset. It is mean-1, so
          switching it on adds depth variance without moving the mean depth.

        `plates` is the `PlateLayout` (or a per-cell plate-index array) placing the cells;
        pass None for protocols without plates. With `plate_sigma = 0` nothing plate-related
        is drawn, so the library stream is exactly the one a plate-free model produces.
        """
        mu_ln = np.log(self.mu_lib_b) - 0.5 * self.h.sigma_lib ** 2
        lib = self.rng.lognormal(mean=mu_ln, sigma=self.h.sigma_lib, size=n_cells)
        if self.h.well_sigma > 0:
            lib = lib * self.rng.lognormal(0.0, self.h.well_sigma, size=n_cells)
        if self.h.plate_sigma > 0 and plates is not None and n_cells > 0:
            if isinstance(plates, PlateLayout):
                idx, n_plates = plates.plate, plates.n_plates
            else:
                idx = np.asarray(plates, dtype=int)
                n_plates = int(idx.max()) + 1 if idx.size else 0
            s = float(self.h.plate_sigma)
            # mean-1 LogNormal: one draw per plate, indexed by the cell's plate
            self.plate_factor = self.rng.lognormal(mean=-0.5 * s * s, sigma=s, size=n_plates)
            lib = lib * self.plate_factor[np.asarray(idx, dtype=int)]
        return lib

    def composition(self, probs):
        """Apply the per-gene batch factor to normalized expression, renormalize per cell.

        `probs` (n_cells, n_genes) is normalized biological expression (lambda). The batch
        factor reshapes the per-gene composition (Splatter-style); library size sets the
        total separately, so the realized library stays centered on the requested depth.
        """
        base = probs * self.beta[None, :]
        tot = base.sum(axis=1, keepdims=True)
        return np.divide(base, tot, out=np.zeros_like(base), where=tot > 0)

    def emit(self, comp, lib, count_model="nb"):
        if count_model not in COUNT_MODELS:
            raise ValueError(
                f"unknown count_model {count_model!r}; choose from {sorted(COUNT_MODELS)}"
            )
        return COUNT_MODELS[count_model](self.rng, comp, lib, self)

    def _soup_share(self):
        return min(max(float(self.h.ambient_frac), 0.0), 1.0)

    def own_library(self, lib):
        """The part of each library that is the molecules of the cell itself: ``(1 - ambient_frac) * lib``.

        The ambient soup is a share of the library (``add_ambient`` draws the rest), so the expected
        total of a cell stays ``lib`` whatever ``ambient_frac`` is, as in a real droplet, whose
        observed library holds both.
        """
        return np.asarray(lib, dtype=float) * (1.0 - self._soup_share())

    def _ambient_mu(self, lib):
        """Expected ambient counts per cell and gene: ``ambient_frac * lib_c`` spread over the soup."""
        return (self._soup_share() * np.asarray(lib, dtype=float))[:, None] * self.ambient_profile[None, :]

    def add_ambient(self, counts, lib):
        """Add ambient contamination: Poisson counts from the batch soup profile, ``ambient_frac`` of
        each library ``lib`` (the counts of the cell itself are drawn on ``own_library(lib)``)."""
        if self.h.ambient_frac <= 0:
            return counts
        return counts + self.rng.poisson(self._ambient_mu(lib))

    def apply_dropout(self, counts, comp, lib):
        """Optional Splatter-style logistic dropout (zero-inflation), keyed to mean count.

        The count of a gene in a droplet is zeroed with probability
        ``1 / (1 + (mu / dropout_mid) ** dropout_shape)``, where ``mu`` is the expected count of the
        gene in the droplet: the molecules of the cell plus the ambient ones. Dropout acts on every
        molecule in the droplet, so ambient counts survive it as often as the counts of the cell at
        the same expected count. Keying it to the expression of the cell alone would delete the
        ambient counts of every gene the cell does not express.

        Returns (counts, p_drop). `p_drop` is None when dropout is disabled.
        """
        if self.h.dropout_mid <= 0:
            return counts, None
        mu = self.own_library(lib)[:, None] * comp
        if self.h.ambient_frac > 0:
            mu = mu + self._ambient_mu(lib)
        mu = np.maximum(mu, 1e-12)
        p_drop = 1.0 / (1.0 + np.exp(self.h.dropout_shape * (np.log(mu) - np.log(self.h.dropout_mid))))
        keep = self.rng.random(counts.shape) >= p_drop
        return counts * keep, p_drop

    def apply_doublets(self, counts):
        """Merge a `doublet_rate` fraction of barcodes with a random partner cell.

        Returns (counts, is_doublet) where `is_doublet` marks the merged barcodes.
        """
        n = counts.shape[0]
        is_doublet = np.zeros(n, dtype=bool)
        if self.h.doublet_rate <= 0 or n < 2:
            return counts, is_doublet
        n_dbl = int(round(self.h.doublet_rate * n))
        if n_dbl == 0:
            return counts, is_doublet
        idx = self.rng.choice(n, size=n_dbl, replace=False)
        partners = self.rng.choice(n, size=n_dbl, replace=True)
        counts = counts.copy()
        for i, p in zip(idx, partners):
            if p == i:
                p = (p + 1) % n
            counts[i] = counts[i] + counts[p]
            is_doublet[i] = True
        return counts, is_doublet


# ======================================================================================
# DNA assay batch model — shared by bulk and single-cell.
#
# The physics of DNA coverage differ from scRNA expression, so the pluggable choices flip:
# the *depth* default is **Dirichlet-Multinomial** (compositional — a fixed read
# budget partitioned across loci, proportions driven by COPY NUMBER) rather than NB, and
# the *allele* layer is a separate binomial/beta-binomial draw with explicit ADO for
# single-cell. The unifying knob across bulk vs single-cell is **kappa = the amplification
# regime**: the Dirichlet concentration over all loci, so the excess per-locus coverage
# variance is about n_loci / kappa (large -> multinomial/Poisson, small -> lumpy). Because
# that depends on the number of loci, a kappa travels with the locus count it was fitted
# on (`kappa_loci`) and is rescaled to the genome it is drawn on.
#
# All hyper-parameter names here are `estimate_dna()` targets: `kappa`
# (amplification regime), `capture_sigma` (per-target/amplicon efficiency), `ado_rate`,
# `beta_binom_conc`, the GC-curve params, and `error_rate`.
# ======================================================================================

# ---- Pluggable DNA depth-emission step ------------------------------------------------
# Each emitter has the signature  emit(rng, weights, N, h) -> counts (n_loci,)  where
# `weights` are non-negative per-locus relative coverage drivers (∝ CN · length · GC /
# mappability efficiency), `N` is the total read budget, and `h` is the hyper-parameters.
# Keeping the (weights, N) interface shared means DM and NB are true drop-ins.
def _dna_depth_dm(rng, weights, N, h):
    """Default DNA depth model: Dirichlet-Multinomial (compositional, fixed budget N).

        p_seg ~ Dirichlet(kappa * p_bar),  p_bar ∝ weights;   y ~ Multinomial(N, p)

    The coverage CV^2 at a locus is about 1/mean + n_loci/kappa: large `kappa` → p≈p_bar →
    multinomial/Poisson; small `kappa` → lumpy over-Poisson coverage (whole-genome
    amplification). With `kappa_loci` set, `kappa` is first rescaled to
    ``kappa * n_loci / kappa_loci``, which keeps the excess at ``kappa_loci / kappa`` on any
    number of loci. Because the budget is fixed, an
    amplified (high-CN) segment compositionally *steals* reads from the rest — exactly the
    coupling independent-per-bin models miss, and the reason depth is RELATIVE not absolute.
    """
    w = np.asarray(weights, dtype=float)
    s = w.sum()
    N = int(round(N))
    if s <= 0 or N <= 0:
        return np.zeros(len(w), dtype=int)
    p_bar = w / s
    kappa = h.kappa if not h.kappa_loci else h.kappa * len(w) / h.kappa_loci
    alpha = np.maximum(kappa * p_bar, 1e-12)
    p = rng.dirichlet(alpha)
    return rng.multinomial(N, p)


def _dna_depth_nb(rng, weights, N, h):
    """Alternative DNA depth model: independent Negative-Binomial per locus/bin.

    The field convention for BULK CNA tools (HMMcopy / CNVkit / Control-FREEC): each bin
    drawn independently with mean ∝ its weight (no fixed-total coupling, so the total
    fluctuates and high-CN bins do not steal from others). `nb_dispersion` sets the
    over-Poisson lumpiness (var = mu + phi*mu^2); phi<=0 collapses to Poisson.
    """
    w = np.asarray(weights, dtype=float)
    s = w.sum()
    if s <= 0:
        return np.zeros(len(w), dtype=int)
    mu = np.maximum(N * (w / s), 0.0)  # expected per-locus coverage
    phi = h.nb_dispersion
    if phi <= 0:
        return rng.poisson(mu)
    shape = 1.0 / phi
    gamma = rng.gamma(shape=shape, scale=mu * phi)
    return rng.poisson(gamma)


DNA_DEPTH_MODELS = {"dm": _dna_depth_dm, "nb": _dna_depth_nb}


@dataclass
class DNABatchHyperParams:
    """The DNA assay's technical parameters — a defaults container.

    Holds the protocol-typical magnitudes of the DNA coverage model: the depth emission
    (capture breadth, mean coverage, and the DM/NB amplification regime), the
    coverage-bias terms (GC curve, mappability, capture efficiency, depth shift), and the
    allele layer (sequencing error, single-cell ADO, beta-binomial overdispersion,
    doublets, FFPE C>T). The [`scDNA`][iscc.data.scDNA] / [`bulkDNA`][iscc.data.bulkDNA]
    assays build one and let you override any field, and
    [`estimate_dna`][iscc.data.estimate_dna.estimate_dna] fits these same parameters from real data.

    Attributes
    ----------
    breadth : str
        Capture breadth {wgs, wes, panel} (sets the locus set + depth regime).
    depth_model : str
        Depth emission {"dm" (default), "nb"}.
    mu_depth : float
        Mean per-locus coverage (the depth regime; breadth sets it).
    kappa : float
        DM concentration = amplification regime: the excess per-locus coverage CV^2 is about
        ``n_loci / kappa`` (large = near-Poisson, small = lumpy). The default 230,823 is the
        GIAB HG002 WGS fit, as in [`bulkDNA`][iscc.data.bulkDNA].
    kappa_loci : int, optional
        Number of loci ``kappa`` was fitted on. The DM's excess per-locus coverage variance is
        about ``n_loci / kappa``, so a ``kappa`` fitted on one locus count gives a different
        coverage spread on another. When set, the depth draw rescales ``kappa`` to its own locus
        count (``kappa * n_loci / kappa_loci``), so a fit transfers. ``None`` uses ``kappa`` as
        given. The default 20,000 is the number of HG002 sites the default ``kappa`` was fitted
        on, so set the two together. [`estimate_dna`][iscc.data.estimate_dna.estimate_dna] sets
        both, and the assays drop the default ``kappa_loci`` when given a ``kappa`` alone.
    nb_dispersion : float
        NB per-bin overdispersion phi (only for depth_model="nb"). The default 0.0866 is the
        GIAB HG002 WGS fit.
    gc_curve_sigma : float
        Per-batch GC->coverage curve strength (the GC bias a panel-of-normals fits).
    mappability_sigma : float
        Not used by the DNA assay.
    capture_sigma : float
        Per-target (WES) / per-amplicon (panel) capture-efficiency LogNormal sd.
    error_rate : float
        Per-base sequencing error (false alt/ref on the allele layer).
    depth_batch_sigma : float
        Per-batch depth-shift LogNormal sd (batches differ in depth).
    ado_rate : float
        Single-cell allelic dropout Bernoulli prob (one allele lost at a locus).
    beta_binom_conc : float
        Single-cell allele-fraction overdispersion (Beta-Binomial concentration).
    doublet_rate : float
        Single-cell doublet fraction.
    ffpe_ct_rate : float
        Optional FFPE C>T deamination extra-error at C-sites.
    """
    breadth: str = "wgs"
    depth_model: str = "dm"
    mu_depth: float = 30.0
    kappa: float = 230_823.0             # GIAB HG002 WGS (estimate_dna), on the loci below
    kappa_loci: Optional[int] = 20_000
    nb_dispersion: float = 0.0866        # GIAB HG002 WGS (estimate_dna, depth_model="nb")
    gc_curve_sigma: float = 0.20
    mappability_sigma: float = 0.10
    capture_sigma: float = 0.0
    error_rate: float = 0.001
    depth_batch_sigma: float = 0.05
    ado_rate: float = 0.0
    beta_binom_conc: float = 30.0
    doublet_rate: float = 0.0
    ffpe_ct_rate: float = 0.0

    def to_dict(self):
        return asdict(self)


class DNABatch:
    """One seeded realization of a DNA assay instance (a library / run / chip).

    Mirrors `Batch` (scRNA) for DNA: draws the run-level technical
    state shared across the assay — a **per-batch GC->coverage curve**, a per-locus
    mappability multiplier, a systematic **per-target/amplicon capture efficiency**, a
    per-batch **depth shift**, and a per-locus **error rate** (+ optional FFPE C>T) — then
    exposes the per-locus operations the bulk/single-cell DNA assays compose: the pluggable
    depth draw and the allele layer (binomial for bulk; beta-binomial + ADO for single-cell).
    """

    def __init__(self, hypers: DNABatchHyperParams, seed=0, label=None):
        self.h = hypers
        self.seed = int(seed)
        self.label = str(label) if label is not None else f"dnabatch{self.seed}"
        self.rng = np.random.default_rng(self.seed)
        self._realized = False

    def realize(self, gc, mappability, ct_sites=None):
        """Draw the batch-level technical state for the observed loci.

        `gc` and `mappability` are *genome* properties (stable across batches); the GC
        *curve* and capture draws are the per-batch (technical) signature. The combined
        per-locus efficiency is normalized to mean 1 so it reshapes coverage (bias) without
        moving the absolute depth scale, which `mu_depth` / the budget set separately.
        """
        gc = np.asarray(gc, dtype=float)
        mappability = np.asarray(mappability, dtype=float)
        n = len(gc)
        self.n = n
        # per-batch realized depth (so two batches differ in depth)
        self.depth = float(self.h.mu_depth * self.rng.lognormal(0.0, self.h.depth_batch_sigma))
        # per-batch GC->coverage curve: a smooth unimodal bias peaking at gc_opt.
        # gc_curve_sigma scales both the peak jitter and the curvature (bias strength).
        self.gc_opt = float(0.42 + self.rng.normal(0.0, 0.05) * self.h.gc_curve_sigma / 0.20)
        self.gc_strength = float(self.rng.uniform(4.0, 10.0) * self.h.gc_curve_sigma / 0.20)
        gc_mult = np.exp(-self.gc_strength * (gc - self.gc_opt) ** 2)
        # systematic per-target/amplicon capture efficiency MEAN (WES/panel); 1 for WGS.
        if self.h.capture_sigma > 0:
            self.capture_eff = self.rng.lognormal(0.0, self.h.capture_sigma, size=n)
        else:
            self.capture_eff = np.ones(n)
        eff = gc_mult * mappability * self.capture_eff
        m = eff.mean()
        self._eff_norm = float(m)
        self.efficiency = eff / m if m > 0 else np.ones(n)
        # per-locus effective error (+ FFPE C>T deamination at C-sites)
        err = np.full(n, self.h.error_rate, dtype=float)
        if self.h.ffpe_ct_rate > 0 and ct_sites is not None:
            err = err + self.h.ffpe_ct_rate * np.asarray(ct_sites, dtype=float)
        self.error = np.clip(err, 0.0, 1.0)
        self._realized = True
        return self

    # -- depth -------------------------------------------------------------------------
    def emit_depth(self, weights, N):
        """Draw per-locus coverage via the pluggable depth model (DM default / NB)."""
        return DNA_DEPTH_MODELS[self.h.depth_model](self.rng, weights, N, self.h)

    def locus_efficiency(self, gc, mappability):
        """The efficiency of further loci of a WGS library under this batch's GC curve, on the
        scale of the realised loci (mean 1 there). There is no capture term: the further loci are
        the genome-wide passenger sites, which a WGS library reads with the rest of the genome."""
        eff = np.exp(-self.gc_strength * (np.asarray(gc, float) - self.gc_opt) ** 2) * np.asarray(mappability, float)
        return eff / self._eff_norm if self._eff_norm > 0 else np.ones(len(eff))

    # -- allele layer ------------------------------------------------------------------
    def alleles_binomial(self, coverage, true_af, error=None):
        """BULK allele draw: alt ~ Binomial(coverage, p_eff) with sequencing error.

        p_eff = true_af*(1-e) + (1-true_af)*e folds the per-base error in both directions
        (false alt on ref bases, false ref on alt bases). ``error`` replaces the per-locus error
        of the realised loci, for loci beyond them.
        """
        e = self.error if error is None else error
        p = np.clip(np.asarray(true_af) * (1 - e) + (1 - np.asarray(true_af)) * e, 0.0, 1.0)
        return self.rng.binomial(coverage, p)

    def apply_ado(self, true_af):
        """SINGLE-CELL allelic dropout: with prob `ado_rate`, one allele is lost at a het
        locus, flipping the observed fraction to 0 or 1 (the dominant single-cell allele
        artifact — modelled as a separate Bernoulli layer, NOT via the depth distribution).

        Returns (af_observed, ado_mask). Only heterozygous loci (0<af<1) can drop.
        """
        af = np.asarray(true_af, dtype=float)
        mask = np.zeros(af.shape, dtype=bool)
        if self.h.ado_rate <= 0:
            return af.copy(), mask
        het = (af > 0) & (af < 1)
        ado = (self.rng.random(af.shape) < self.h.ado_rate) & het
        drop_alt = self.rng.random(af.shape) < 0.5
        af_obs = af.copy()
        af_obs[ado & drop_alt] = 0.0       # alt allele lost -> looks homozygous reference
        af_obs[ado & ~drop_alt] = 1.0      # ref allele lost -> looks homozygous alt
        return af_obs, ado

    def allele_balance(self, af, apply_error=True):
        """SINGLE-CELL realized per-locus allele fraction theta (Beta-Binomial mean), WITHOUT
        sampling reads.

        theta ~ Beta(c*p, c*(1-p)) with c = `beta_binom_conc` (small c = lumpier allele balance).
        ``apply_error`` controls whether the per-base sequencing-error floor is folded in:
          * ``True`` (count layer): p = af*(1-e)+(1-af)*e — the OBSERVED allele fraction a count
            caller sees (`alleles_betabinom` samples reads from it).
          * ``False`` (read layer): p = af — the TRUE molecular allele balance (amplification
            overdispersion only), so the read emitter sets per-copy alt multiplicity WITHOUT the
            error floor and the simulator's per-base error (DWGSIM `-e`) is the single read-error
            source (no double-counting; mirrors the scRNA path).
        Degenerate p in {0,1} (e.g. after ADO) collapses to a fixed fraction.
        """
        af = np.asarray(af, dtype=float)
        if apply_error:
            p = np.clip(af * (1 - self.error) + (1 - af) * self.error, 0.0, 1.0)
        else:
            p = np.clip(af, 0.0, 1.0)
        c = self.h.beta_binom_conc
        a = np.maximum(c * p, 1e-9)
        b = np.maximum(c * (1 - p), 1e-9)
        theta = self.rng.beta(a, b)
        return np.where(p <= 0, 0.0, np.where(p >= 1, 1.0, theta))

    def alleles_betabinom(self, coverage, af):
        """SINGLE-CELL allele draw: alt ~ Binomial(coverage, allele_balance(af))."""
        theta = self.allele_balance(af)
        return self.rng.binomial(coverage, np.clip(theta, 0.0, 1.0))


# ======================================================================================
# Visium (spatial transcriptomics) batch model.
#
# The distinctive requirement of the spatial assay is that the technical noise is SPATIALLY
# CORRELATED: neighbouring spots share capture efficiency. So the headline component is a
# smooth POSITIVE random field over the spot coordinates (a squared-exponential GP), whose
# autocorrelation length is `field_lengthscale` and strength `field_sigma`, multiplied by a
# tissue-boundary efficiency falloff (`edge_sigma`). It scales the per-spot library size, so
# the realized per-spot depth carries positive spatial autocorrelation (Moran's I > 0) — the
# signal `estimate_visium()` inverts and `validate_visium.py` checks.
#
# Everything else is shared with the scRNA batch model: the per-gene LogNormal batch
# factor (`sigma_batch`), ambient soup (`ambient_frac`), and the pluggable NB/DM count draw
# (`COUNT_MODELS`). `VisiumBatch` therefore subclasses `Batch` and reuses `composition` /
# `emit` / `add_ambient` as is; only the realization (per-gene factor + spatial field) and
# the per-spot library draw are spatial-specific. All hyper-parameter names here are the
# `estimate_visium()` targets.
# ======================================================================================
@dataclass
class VisiumBatchHyperParams:
    """The Visium assay's technical parameters — a defaults container.

    Holds the protocol-typical magnitudes of the Visium spatial count model: the spot
    geometry (pitch, radius), the per-spot library size, the spatially-correlated
    capture-efficiency field (lengthscale, strength, tissue-edge falloff), lateral mRNA
    diffusion, and the shared per-gene batch factor / ambient soup / pluggable count
    model. The [`Visium`][iscc.data.Visium] assay builds one and lets you override any
    field, and [`estimate_visium`][iscc.data.estimate_visium.estimate_visium] fits these same parameters
    from real data.

    Attributes
    ----------
    protocol : str
        Protocol preset the magnitudes come from (e.g. "visium").
    spot_pitch : float
        Spot centre-to-centre spacing (sets spot density / spots-per-tissue). The default of 2.0 demes
        is the 100 um pitch of Visium v1, with a deme of ~50 um.
    spot_radius : float
        Spot capture radius (sets spot->cell aggregation). The default of 0.55 demes is the 55 um
        spot of Visium v1, so a spot pools about one deme's share of the section.
    mu_counts : float
        Mean library size of a tissue spot (total UMIs / spot) before spot swapping; without
        swapping it is the mean depth of the tissue spots.
    sigma_counts : float
        Per-spot library-size LogNormal sd.
    field_lengthscale : float
        Capture-field spatial autocorrelation scale (the SE-GP length-scale); larger ->
        smoother field -> higher Moran's I.
    field_sigma : float
        Capture-field strength (log-space sd of the smooth field).
    edge_sigma : float
        Tissue-boundary capture-efficiency falloff (0 = none; <1 keeps it positive).
    edge_reference : str
        What the falloff is measured from: ``"grid"`` (the spot grid's bounding box, the
        default) or ``"tissue"`` (the nearest spot without tissue, so the tissue's own edge is dimmed
        whatever the slide's size).
    diffusion_sigma : float
        Lateral smoothing of composition between spots: Gaussian kernel sd (moves no counts).
    bleed_rate : float
        Spot swapping (SpotClean's bleeding rate): the share of each spot's counts that leaves it
        for other spots of the slide, tissue or not; 0 = off. See
        [`bleed`][iscc.data.visium_background.bleed].
    bleed_distal : float
        Share of the bled counts spread uniformly over the whole slide (the slide-wide ambient floor).
    bleed_sigma : float
        Sd of the Gaussian kernel the rest of the bled counts spread through (coordinate units).
    sigma_batch : float
        Per-gene batch-factor LogNormal sd.
    ambient_frac : float
        Fraction of the spot library that is ambient ("soup") contamination.
    count_model : str
        Count emission {"dm" (default, compositional capture), "nb"}.
    kappa : float
        Dirichlet-Multinomial concentration (only for count_model="dm"). A gene's variance is
        ``(N + kappa) / (1 + kappa)`` times its multinomial variance at a spot depth ``N``, whatever
        the number of genes, so a value fitted on one gene set applies to any other.
    nb_dispersion : float
        NB overdispersion phi (var = mu + phi*mu^2; only for count_model="nb").
    content_exponent : float
        How the library of a spot follows its RNA content
        (summed over its cells, [`cell_rna_content`][iscc.data.cell_rna_content]): the library is multiplied by the
        content raised to this power, rescaled so that the mean depth of the batch is kept. 1 (the
        default) makes the depth proportional to it; 0 leaves the content out.
    """
    # Defaults CALIBRATED to a real 10x Visium breast-cancer section (estimate_visium on
    # `scanpy.datasets.visium_sge('V1_Breast_Cancer_Block_A_Section_1')`), so out-of-the-box Visium
    # is realistic with no user data. The spot geometry is Visium v1 at ~50 um per deme.
    # `kappa` transfers between gene sets: the Dirichlet-multinomial inflates every gene's variance by
    # (N + kappa) / (1 + kappa), which depends on the spot depth N and not on the number of genes or
    # their composition, so unlike the DNA concentration (kappa_loci) it is not rescaled by the number
    # of genes. The direct fit (89,197 over the section's 3,798 in-tissue spots) reads the tissue's
    # heterogeneity as count noise, so it bounds the technical kappa from below; a simulated section
    # reproduces the real estimate at an input of about 1,400,000 (HUMAN_VISIUM_KAPPA, by indirect
    # inference). The old default of 50 inflated each gene's variance about 390-fold at 20,000 UMIs
    # per spot: over 6,000 growth genes a spot detected 281 genes instead of ~4,700 and put 77% of
    # its counts in 50 of them.
    protocol: str = "visium"
    spot_pitch: float = 2.0
    spot_radius: float = 0.55        # 55 um spot at the 100 um pitch of Visium v1
    mu_counts: float = 20000.0       # real ~21,815 UMIs/spot (deep Visium)
    sigma_counts: float = 0.45       # real 0.455
    field_lengthscale: float = 18.0  # real ~9 spots x spot_pitch 2.0 (coord units)
    field_sigma: float = 0.70        # real 0.697 (capture-field strength)
    edge_sigma: float = 0.30
    edge_reference: str = "grid"
    diffusion_sigma: float = 0.0
    bleed_rate: float = 0.0
    bleed_distal: float = 0.0
    bleed_sigma: float = 0.0
    sigma_batch: float = 0.10
    ambient_frac: float = 0.05
    count_model: str = "dm"
    kappa: float = 1_400_000.0       # technical noise near multinomial; see the note above
    nb_dispersion: float = 0.30
    # 1: a spot's depth is proportional to the RNA content of its cells, which gives the real breast
    # section's relation of depth to composition (vault: decision-library-size-from-rna-content)
    content_exponent: float = 1.0

    def to_dict(self):
        return asdict(self)


class VisiumBatch(Batch):
    """One seeded realization of a Visium section (a slide / capture area).

    Mirrors `Batch` (scRNA) for spatial transcriptomics: draws the per-gene
    batch factor + ambient soup (shared with `Batch`) AND the spatial-specific piece — a smooth
    positive **capture-efficiency field** over the spot coordinates (a squared-exponential GP times
    a tissue-boundary falloff). It reuses `composition` / `emit` / `add_ambient` from `Batch`
    as is; only `realize` (which also samples the spatial field) and `spot_library` (the
    per-spot depth, scaled by the field) are overridden. Reproducible from `seed`.
    """

    def __init__(self, hypers: VisiumBatchHyperParams, seed=0, label=None):
        self.h = hypers
        self.seed = int(seed)
        self.label = str(label) if label is not None else f"visium{self.seed}"
        self.rng = np.random.default_rng(self.seed)
        self._realized = False

    def realize(self, genes, base_profile, spot_coords, in_tissue=None):
        """Draw the section-level technical state for `spot_coords` (n_spots, 2).

        `base_profile` (pooled spot expression) sets the ambient soup; the per-gene batch factor
        and the spatial capture field are the technical signature (reproducible from the seed).
        `in_tissue` (a per-spot boolean) marks the spots the capture field is normalised over, and
        is the edge of ``edge_reference="tissue"``; without it every spot counts as tissue.
        """
        n = len(genes)
        self.genes = list(genes)
        # per-gene batch factor beta_g ~ LogNormal(0, sigma_batch^2), shared across spots
        self.beta = self.rng.lognormal(mean=0.0, sigma=self.h.sigma_batch, size=n)
        # NB dispersion (used only when count_model="nb"; exposed via the `dispersion` property)
        self._phi = float(self.h.nb_dispersion)
        # ambient ("soup") profile for this section: normalized pooled expression
        amb = np.asarray(base_profile, dtype=float)
        tot = amb.sum()
        self.ambient_profile = amb / tot if tot > 0 else np.full(n, 1.0 / n)
        # the headline piece: a spatially-autocorrelated positive capture-efficiency field
        self.spot_coords = np.asarray(spot_coords, dtype=float)
        self.capture_field = self._draw_capture_field(self.spot_coords, in_tissue=in_tissue)
        self._realized = True
        return self

    # -- the spatially-correlated capture-efficiency field -----------------------------
    def _draw_capture_field(self, coords, in_tissue=None):
        """Smooth POSITIVE random field over spot coords: squared-exponential GP x edge falloff.

        A zero-mean GP with an RBF (squared-exponential) covariance ``K_ij = exp(-||x_i - x_j||^2
        / (2*ell^2))`` (ell = `field_lengthscale`) is sampled on the spot grid and pushed through a
        LogNormal link (strength `field_sigma`) so the field is positive. Larger `field_lengthscale`
        -> smoother field -> stronger positive spatial autocorrelation (higher Moran's I). The field
        is then multiplied by a tissue-boundary falloff (`edge_sigma`) and normalized to mean 1 over
        the tissue spots (`in_tissue`; every spot when it is None), so it RESHAPES capture (a spatial
        bias) without moving the mean depth of a tissue spot, which `mu_counts` sets.
        Normalising over the whole slide instead scales a section that covers part of it by the field's
        mean over the tissue: 0.55 to 1.49 over five seeds for a section on 28% of the v1 slide, since
        the field is correlated over ~9 pitches.
        """
        coords = np.asarray(coords, dtype=float)
        n = coords.shape[0]
        if n == 0:
            return np.ones(0)
        ell = max(float(self.h.field_lengthscale), 1e-6)
        diff = coords[:, None, :] - coords[None, :, :]
        d2 = np.sum(diff * diff, axis=-1)
        K = np.exp(-d2 / (2.0 * ell * ell))
        K[np.diag_indices(n)] += 1e-6                      # jitter so K is positive-definite
        L = np.linalg.cholesky(K)
        g = L @ self.rng.standard_normal(n)                # zero-mean unit-variance SE-GP sample
        # LogNormal link -> positive field with mean ~1 (before edge / renormalization)
        field = np.exp(self.h.field_sigma * g - 0.5 * self.h.field_sigma ** 2)
        field = field * self._edge_falloff(coords, in_tissue=in_tissue)
        tis = None if in_tissue is None else np.asarray(in_tissue, dtype=bool)
        m = field[tis].mean() if tis is not None and tis.any() else field.mean()
        return field / m if m > 0 else np.ones(n)

    def _edge_falloff(self, coords, in_tissue=None):
        """Capture-efficiency falloff: lower efficiency near an edge.

        With ``edge_reference="grid"`` (the default) the edge is the spot grid's bounding box: the
        distance of each spot to its nearest side, normalized to [0, 1] (0 at the side, 1 at the
        centre). With ``edge_reference="tissue"`` it is the tissue's own boundary: the distance of each
        tissue spot beyond the tissue's outermost ring, in spot pitches (0 on that ring; spots outside
        the tissue also sit at 0), so the falloff is confined to about one pitch of the edge whatever
        the size of the tissue, as on a real section. Either way efficiency =
        ``1 - edge_sigma * exp(-3*dnorm)``, so the very edge is reduced by `edge_sigma` and the
        interior is ~1. `edge_sigma<1` keeps it positive.
        """
        n = coords.shape[0]
        if self.h.edge_sigma <= 0 or n == 0:
            return np.ones(n)
        reference = getattr(self.h, "edge_reference", "grid")
        if reference not in ("grid", "tissue"):
            raise ValueError(f"edge_reference must be 'grid' or 'tissue', got {reference!r}")
        if reference == "tissue" and in_tissue is not None:
            from scipy.spatial import cKDTree
            tis = np.asarray(in_tissue, dtype=bool)
            if tis.all() or not tis.any():
                return np.ones(n)
            dist = np.zeros(n)
            d = cKDTree(coords[~tis]).query(coords[tis])[0]
            dist[tis] = d - d.min()                        # the tissue's outermost spots sit at 0
            return 1.0 - float(self.h.edge_sigma) * np.exp(-3.0 * dist / float(self.h.spot_pitch))
        r, c = coords[:, 0], coords[:, 1]
        dr = np.minimum(r - r.min(), r.max() - r)
        dc = np.minimum(c - c.min(), c.max() - c)
        dist = np.minimum(dr, dc)
        span = 0.5 * min(np.ptp(r), np.ptp(c))
        if span <= 0:
            return np.ones(n)
        dnorm = dist / span
        return 1.0 - float(self.h.edge_sigma) * np.exp(-3.0 * dnorm)

    # -- per-spot library size (scaled by the spatial capture field) -------------------
    def spot_library(self, occupied=None):
        """Per-spot library size ell_s ~ LogNormal(log mu_counts, sigma_counts^2) x capture_field_s.

        The capture field injects the spatial autocorrelation into the realized depth. `occupied`
        (a 0/1 per-spot mask) zeroes the library of off-tissue (empty) spots so they emit nothing.
        """
        n = self.capture_field.shape[0]
        mu_ln = np.log(self.h.mu_counts) - 0.5 * self.h.sigma_counts ** 2
        lib = self.rng.lognormal(mean=mu_ln, sigma=self.h.sigma_counts, size=n)
        lib = lib * self.capture_field
        if occupied is not None:
            lib = lib * np.asarray(occupied, dtype=float)
        return lib
