"""Genome-wide passenger mutations at infinite sites, re-created when cells are materialised.

The model is described on [`PassengerCatalogue`][iscc.tumor.PassengerCatalogue]; ``build`` makes one
for the cells of a materialisation, and ``GenotypeTumor.make_cell_data`` calls it.
"""
import numpy as np
import pandas as pd

#: Genome-wide passenger SNVs per cell division. Werner et al. 2020 (Nat Commun 11:1035) measured
#: about 1.1 to 1.4 per division in healthy tissue and higher rates in tumours.
DEFAULT_RATE = 2.0

#: Passengers the founder carries when it transforms (the clonal burden of the tumour).
DEFAULT_TRUNCAL = 1000

#: Offset of the passenger stream (``GenotypeTumor._run_rng``), apart from every other stream.
RNG_OFFSET = 919191

#: Seed of the per-site GC content and mappability of the passenger sites (a genome property).
FEATURE_SEED = 20251124

_KIND_TRUNCAL, _KIND_STEM, _KIND_WITHIN = 0, 1, 2
KINDS = ("truncal", "stem", "within")


def resolve(value):
    """The passenger settings of a tumour: ``None`` (the defaults), ``False`` or a ``rate`` of 0
    (off, returns ``None``), or a dict with ``rate`` and ``truncal``."""
    if value is False:
        return None
    out = dict(rate=DEFAULT_RATE, truncal=DEFAULT_TRUNCAL)
    if isinstance(value, dict):
        unknown = set(value) - set(out)
        if unknown:
            raise TypeError(f"unknown genome_passengers keys {sorted(unknown)}; use 'rate' and 'truncal'")
        out.update(value)
    elif value not in (None, True):
        raise TypeError("genome_passengers takes None, True, False or a dict with 'rate' and 'truncal'")
    out = dict(rate=float(out["rate"]), truncal=int(out["truncal"]))
    if out["rate"] < 0 or out["truncal"] < 0:
        raise ValueError("genome_passengers 'rate' and 'truncal' must be non-negative")
    return out if (out["rate"] > 0 or out["truncal"] > 0) else None


class PassengerCatalogue:
    """The genome-wide passengers of materialised cells, each at a site of its own, and the clades
    that carry them.

    The engine places point mutations on its growth genes, a few thousand sites, so at the scale of
    a tumor the same gene is mutated in many lineages and its allele fraction sums them. Most
    somatic mutations of a real tumor are neutral passengers outside genes, each at a site of its
    own, and the methods that reconstruct subclones from allele fractions (PyClone-VI, MOBSTER,
    DPClust) assume exactly that: one mutation, one lineage. When cells are materialised,
    ``iscc`` adds such passengers as a DNA-only layer beside the gene-level SNVs, which it leaves as
    they are, and [`bulkDNA`][iscc.data.bulkDNA] reads them with the WGS breadth.

    Every passenger has its own site, so its carriers form one clade of the sampled genealogy, until
    a copy-number loss removes it from part of the clade. The genealogy has two levels:

    * the clone tree of the engine: the founder carries the truncal passengers, and the edge into
      each clone carries the passengers its founding cell gathered inside its parent clone before
      it mutated, propagated down the subtree through the copy-number event of each edge (a
      doubling doubles the copies of a passenger; a gain or a loss of a homolog copies or removes a
      carrying copy with probability copies carrying over copies);
    * inside each materialised clone, a random binary genealogy of its sampled cells, with Yule
      (uniform) splits and the same expected number of passengers on every branch. A Yule tree of
      ``n`` leaves has on average ``2n / (k (k + 1))`` clades of ``k`` leaves, so the number of
      passengers carried by at least a fraction ``f`` of a clone falls as ``1/f``, the spectrum of
      neutral growth (Williams et al. 2016).

    The ``rate`` of the ``genome_passengers`` setting of
    [`GenotypeTumor`][iscc.tumor.GenotypeTumor] is passengers per cell division. A branch of the
    genealogy inside a clone carries on average ``rate / (2 beta)`` passengers, where
    ``beta = 1 - death rate / division rate`` of the clone is the share of divisions whose lineage
    survives, so that ``rate / (beta f)`` passengers are carried by at least a fraction ``f`` of the
    clone, as in neutral growth (Williams et al. 2016). Cells in full demes keep dividing and dying
    without net growth; the passengers of that turnover stay in a few cells, below what bulk
    sequencing resolves, and are not drawn. The edge into a clone carries the passengers of its
    founding lineage since its parent clone arose: ``rate`` times the division rate of the parent
    times the generations between the two births. All of them mark the new clone, although in a
    real tumor the oldest are shared with the cells of the parent that descend from the same
    ancestor, so the clusters of clone-specific passengers are an upper bound. Growth never reads the layer, and it draws from
    a random stream of its own, so every other output of a materialisation is the same with it on
    or off.

    The catalogue travels in ``cell_data["cell_lineage"].attrs["passengers"]``, where every
    resection, section or dissociation of the cells keeps it, and is never copied (it does not
    change after it is built). A cell is addressed by its row of ``cell_lineage``: its clone
    (``lineage_clone``) and its leaf in the genealogy of its clone (``lineage_leaf``, -1 for a
    normal cell).

    Parameters
    ----------
    segment, homolog, position, kind : array-like
        Per passenger: the segment index of its site, its homolog (0 for ``p``, 1 for ``m``), its
        position along the segment as a fraction in [0, 1), and its kind (0 truncal, 1 on the
        edge into a clone, 2 inside a clone).
    clones : list of str
        The genotypes of the materialised cells, cancer and normal.
    cn : array-like
        Total copy number of each clone (rows) on each segment (columns).
    carriers : tuple of array-like
        ``(passenger, clone, copies)``: the copies carrying each truncal or edge passenger in each
        clone that carries it.
    within : tuple of array-like
        ``(passenger, clone, first leaf, end leaf)``: the leaf range of the clade inside its clone
        that carries each passenger of kind 2, on one copy.
    cancer_clones : iterable of str
        The cancer genotypes among ``clones``.

    Attributes
    ----------
    names : pandas.Index
        The passenger site ids (``P0``, ``P1``, ...).
    """

    def __init__(self, segment, homolog, position, kind, clones, cn, carriers, within, cancer_clones):
        self.segment = np.asarray(segment, dtype=np.int32)
        self.homolog = np.asarray(homolog, dtype=np.int8)
        self.position = np.asarray(position, dtype=np.float32)
        self.kind = np.asarray(kind, dtype=np.int8)
        self.names = pd.Index([f"P{i}" for i in range(len(self.segment))], name="passenger")
        self.clones = list(clones)
        self._clone_ix = {c: i for i, c in enumerate(self.clones)}
        self.cn = np.asarray(cn, dtype=np.int32)
        # inherited passengers (truncal and edge): rows (passenger, clone, copies carrying it)
        self.car_p, self.car_c, self.car_m = (np.asarray(a, dtype=np.int32) for a in carriers)
        # passengers inside a clone: (passenger, clone, first leaf, end leaf), on one copy
        self.w_p, self.w_c, self.w_a, self.w_b = (np.asarray(a, dtype=np.int32) for a in within)
        self._cancer = np.array([c in set(cancer_clones) for c in self.clones], dtype=bool)

    def __deepcopy__(self, memo):            # immutable: pandas copies attrs deeply, share instead
        return self

    def __len__(self):
        return len(self.segment)

    def _index(self, lineage):
        ci = np.array([self._clone_ix.get(str(c), -1) for c in lineage["lineage_clone"].to_numpy()],
                      dtype=np.int64)
        return ci, lineage["lineage_leaf"].to_numpy(dtype=np.int64)

    def pooled(self, lineage):
        """The pooled counts of every passenger over a set of cells, as bulk sequencing sees them.

        Parameters
        ----------
        lineage : pandas.DataFrame
            Rows of ``cell_data["cell_lineage"]``: the pooled cells.

        Returns
        -------
        alt_copies, total_copies, carriers : numpy.ndarray
            Per passenger: the copies carrying it, the copies of its segment, and the cells
            carrying it, over the pooled cells.
        n_cancer : int
            The number of pooled cancer cells, so that ``carriers / n_cancer`` is the true
            cancer-cell fraction.
        """
        n = len(self)
        ci, leaf = self._index(lineage)
        ok = ci >= 0
        n_c = np.bincount(ci[ok], minlength=len(self.clones)).astype(float)
        total = (n_c @ self.cn)[self.segment] if n else np.zeros(0)
        alt = np.bincount(self.car_p, weights=n_c[self.car_c] * self.car_m, minlength=n).astype(float)
        car = np.bincount(self.car_p, weights=n_c[self.car_c] * (self.car_m > 0), minlength=n).astype(float)
        for c in np.unique(self.w_c):
            rows = np.flatnonzero(self.w_c == c)
            leaves = np.sort(leaf[ci == c])
            k = np.searchsorted(leaves, self.w_b[rows]) - np.searchsorted(leaves, self.w_a[rows])
            alt[self.w_p[rows]] += k
            car[self.w_p[rows]] += k
        return alt, total, car, int(n_c[self._cancer].sum())

    def cell_vaf(self, lineage):
        """The true allele fraction of every passenger in every cell.

        Parameters
        ----------
        lineage : pandas.DataFrame
            Rows of ``cell_data["cell_lineage"]``.

        Returns
        -------
        scipy.sparse.csr_matrix
            Cells (in the order of ``lineage``) × passengers (in the order of ``names``): the
            copies carrying the passenger over the copies of its segment in the clone of the cell.
        """
        import scipy.sparse as sp
        ci, leaf = self._index(lineage)
        rows, cols, vals = [], [], []
        for c in np.unique(ci[ci >= 0]):
            cells = np.flatnonzero(ci == c)
            sel = np.flatnonzero(self.car_c == c)
            if len(sel):
                p = self.car_p[sel]
                v = self.car_m[sel] / np.maximum(self.cn[c, self.segment[p]], 1)
                rows.append(np.repeat(cells, len(p)))
                cols.append(np.tile(p, len(cells)))
                vals.append(np.tile(v, len(cells)))
            sel = np.flatnonzero(self.w_c == c)
            if len(sel):
                order = cells[np.argsort(leaf[cells], kind="stable")]
                leaves = leaf[order]
                lo = np.searchsorted(leaves, self.w_a[sel])
                hi = np.searchsorted(leaves, self.w_b[sel])
                k = hi - lo
                # the cells of each passenger are a contiguous run of the leaf-sorted cells
                starts = np.repeat(lo - np.cumsum(k) + k, k) + np.arange(k.sum())
                rows.append(order[starts])
                p = self.w_p[sel]
                cols.append(np.repeat(p, k))
                vals.append(np.repeat(1.0 / np.maximum(self.cn[c, self.segment[p]], 1), k))
        n_cells = len(ci)
        if not rows:
            return sp.csr_matrix((n_cells, len(self)))
        return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                             shape=(n_cells, len(self)))

    def sites(self):
        """The passenger sites as a table indexed by ``names``: ``segment``, ``homolog`` (``p`` or
        ``m``), ``position`` along the segment as a fraction, and ``kind`` (``truncal``, ``stem``
        on the edge into a clone, or ``within`` a clone). ``HumanGenomeLayer.passenger_positions``
        places them on the human genome."""
        return pd.DataFrame({"segment": self.segment,
                             "homolog": np.where(self.homolog == 0, "p", "m"),
                             "position": self.position,
                             "kind": [KINDS[k] for k in self.kind]}, index=self.names)


# ------------------------------------------------------------------------------------------------
def _yule_edges(n, u):
    """The branches of a random binary genealogy of ``n`` leaves with Yule (uniform) splits, as
    leaf ranges ``[a, b)``: every node but the root. ``u`` holds ``n - 1`` uniform numbers, one per
    split. A single leaf has one branch, its own."""
    if n <= 1:
        return np.zeros(1, np.int64), np.ones(1, np.int64)
    a_out, b_out = [], []
    stack = [(0, n)]
    j = 0
    while stack:
        a, b = stack.pop()
        k = 1 + int(u[j] * (b - a - 1))
        j += 1
        for lo, hi in ((a, a + k), (a + k, b)):
            a_out.append(lo)
            b_out.append(hi)
            if hi - lo >= 2:
                stack.append((lo, hi))
    return np.asarray(a_out, np.int64), np.asarray(b_out, np.int64)


def _edge_event(hap_parent, hap_child):
    """The copy-number event on a clone edge: ``("wgd",)`` when more than one (segment, homolog)
    changed, else ``("cn", [(segment, homolog, delta, copies before), ...])``."""
    changes = []
    for s, ((pp, pm), (cp, cm)) in enumerate(zip(hap_parent, hap_child)):
        if cp != pp:
            changes.append((s, 0, cp - pp, pp))
        if cm != pm:
            changes.append((s, 1, cm - pm, pm))
    return ("wgd",) if len(changes) > 1 else ("cn", changes)


def _carry(active, event, rng):
    """The copies of each passenger after a clone edge: a doubling doubles them; a gain copies a
    carrying copy with probability ``copies carrying / copies``, a loss removes one with the same
    probability, one copy at a time. Passengers left on no copy are dropped."""
    p, seg, hom, mult = active
    if event[0] == "wgd":
        return p, seg, hom, mult * 2
    mult = mult.copy()
    for s, h, delta, before in event[1]:
        sel = np.flatnonzero((seg == s) & (hom == h))
        if not len(sel):
            continue
        m = mult[sel].astype(float)
        cn = float(before)
        step = 1 if delta > 0 else -1
        for _ in range(abs(int(delta))):
            if cn <= 0:
                break
            m += step * (rng.random(len(sel)) < m / cn)
            cn += step
        mult[sel] = m.astype(mult.dtype)
    keep = mult > 0
    return p[keep], seg[keep], hom[keep], mult[keep]


def build(tumor, types, names, settings, rng):
    """The passengers of the materialised cells ``names`` of genotypes ``types``.

    Returns ``(catalogue, lineage)``: the ``PassengerCatalogue`` and the per-cell frame
    (``lineage_clone``, ``lineage_leaf``) that carries it in its ``attrs``."""
    types = np.asarray([str(t) for t in types], dtype=object)
    n_seg = int(tumor.n_segments)
    seg_w = np.asarray(tumor.selection.segment_sizes[:n_seg], dtype=float)
    seg_w = seg_w / seg_w.sum()
    G = tumor.genotypes
    parents = tumor.genotypes_parents
    counts = tumor.genotypes_counts

    clones = sorted(set(types), key=lambda g: (not tumor._is_cancer(g), G[g].ord if g in G else 0, g))
    clone_ix = {g: i for i, g in enumerate(clones)}
    hap = {}

    def hap_of(g):
        if g not in hap:
            hap[g] = tumor._hap_cns(g)
        return hap[g]

    cn = np.array([[cp + cm for cp, cm in hap_of(g)] for g in clones], dtype=np.int32).reshape(len(clones), n_seg)

    # leaves: the sampled cells of each cancer clone in a random order
    leaf = np.full(len(types), -1, dtype=np.int64)
    mat_cancer = [g for g in clones if tumor._is_cancer(g)]
    code = np.array([clone_ix[t] for t in types], dtype=np.int64)
    by_clone = np.argsort(code, kind="stable")
    bounds = np.searchsorted(code[by_clone], np.arange(len(clones) + 1))
    per_clone = {}
    for g in mat_cancer:
        i = clone_ix[g]
        idx = by_clone[bounds[i]:bounds[i + 1]]
        leaf[idx[rng.permutation(len(idx))]] = np.arange(len(idx))
        per_clone[g] = len(idx)

    # the clone tree of the sampled cancer cells: their clones and every ancestor
    closure = set()
    for g in mat_cancer:
        cur = g
        while cur is not None and cur not in closure:
            closure.add(cur)
            cur = parents.get(cur)
    children = {g: [] for g in closure}
    roots = []
    for g in sorted(closure, key=lambda x: G[x].ord):
        p = parents.get(g)
        (children[p] if p in closure else roots).append(g)
    order, stack = [], list(reversed(roots))
    while stack:
        g = stack.pop()
        order.append(g)
        stack.extend(reversed(children[g]))
    subtree = {g: float(counts.get(g, 0)) for g in closure}
    for g in reversed(order):
        p = parents.get(g)
        if p in subtree:
            subtree[p] += subtree[g]

    # passengers per branch: rate / (2 beta), beta the share of divisions whose lineage survives
    lam = {}
    for g in closure:
        ep = G[g].evolutionary_parameters
        div = float(ep.get("division_rate", 0.0))
        beta = 1.0 - float(ep.get("death_rate", 0.0)) / div if div > 0 else 1.0
        lam[g] = 0.5 * settings["rate"] / float(np.clip(beta, 0.05, 1.0))

    seg_l, hom_l, kind_l = [], [], []                   # passenger attributes, appended in blocks
    n_made = [0]

    def place(g, n, kind):
        """``n`` new passengers on clone ``g``'s genome: a segment by length, a homolog by its
        copies there. One drawn onto a segment the clone has lost is not made. Returns the ids,
        segments and homologs of the passengers made, and which of the ``n`` they are."""
        if n <= 0:
            return np.zeros(0, np.int64), np.zeros(0, np.int32), np.zeros(0, np.int8), np.zeros(0, bool)
        segs = rng.choice(n_seg, size=n, p=seg_w)
        cp = np.array([h[0] for h in hap_of(g)], float)[segs]
        cm = np.array([h[1] for h in hap_of(g)], float)[segs]
        hom = np.where(rng.random(n) < cp / np.maximum(cp + cm, 1e-12), 0, 1).astype(np.int8)
        ok = (cp + cm) > 0
        segs, hom = segs[ok], hom[ok]
        ids = n_made[0] + np.arange(len(segs))
        n_made[0] += len(segs)
        seg_l.append(segs)
        hom_l.append(hom)
        kind_l.append(np.full(len(segs), kind, np.int8))
        return ids, segs.astype(np.int32), hom, ok

    # inherited passengers: truncal on the founder, edge passengers on every other clone, carried
    # down the tree through the copy-number event of each edge
    car_p, car_c, car_m = [], [], []
    pending = []
    for r in roots:
        n_born = settings["truncal"] if parents.get(r) is None else 0
        pending.append((r, (np.zeros(0, np.int64), np.zeros(0, np.int32), np.zeros(0, np.int8),
                            np.zeros(0, np.int32)), n_born))
    while pending:
        g, active, extra = pending.pop()
        p = parents.get(g)
        if p in closure and extra == 0:
            # the divisions of the founding lineage since the parent clone arose: a lineage divides at
            # the division rate whatever the deaths around it. Without birth times (the exact engine
            # keeps none), the depth of a random leaf in a Yule genealogy of the parent's cells.
            t_g, t_p = getattr(G[g], "birth_time", np.nan), getattr(G[p], "birth_time", np.nan)
            if np.isfinite([t_g, t_p]).all():
                n_div = float(G[p].evolutionary_parameters.get("division_rate", 1.0)) * max(t_g - t_p, 0.0)
                extra = int(rng.poisson(settings["rate"] * n_div))
            else:
                depth = max(1.0, np.log(max(subtree[p], 1.0)))
                extra = int(rng.poisson(lam[p] * depth))
        ids, segs, hom, _ = place(g, extra, _KIND_TRUNCAL if parents.get(g) is None else _KIND_STEM)
        act = tuple(np.concatenate([a, b]) for a, b in
                    zip(active, (ids, segs, hom, np.ones(len(ids), np.int32))))
        if g in clone_ix:
            car_p.append(act[0])
            car_c.append(np.full(len(act[0]), clone_ix[g], np.int32))
            car_m.append(act[3])
        for c in reversed(children[g]):
            pending.append((c, _carry(act, _edge_event(hap_of(g), hap_of(c)), rng), 0))

    # passengers inside each sampled clone, on a Yule genealogy of its sampled cells
    w_p, w_c, w_a, w_b = [], [], [], []
    for g in mat_cancer:
        n = per_clone[g]
        a, b = _yule_edges(n, rng.random(max(n - 1, 0)))
        k = rng.poisson(lam[g], size=len(a))
        ids, _, _, keep = place(g, int(k.sum()), _KIND_WITHIN)
        w_p.append(ids)
        w_c.append(np.full(len(ids), clone_ix[g], np.int64))
        w_a.append(np.repeat(a, k)[keep])
        w_b.append(np.repeat(b, k)[keep])

    def cat(parts, dtype):
        return np.concatenate(parts).astype(dtype) if parts else np.zeros(0, dtype)

    n_pass = n_made[0]
    catalogue = PassengerCatalogue(
        segment=cat(seg_l, np.int32), homolog=cat(hom_l, np.int8),
        position=rng.random(n_pass), kind=cat(kind_l, np.int8), clones=clones, cn=cn,
        carriers=(cat(car_p, np.int64), cat(car_c, np.int32), cat(car_m, np.int32)),
        within=(cat(w_p, np.int64), cat(w_c, np.int64), cat(w_a, np.int64), cat(w_b, np.int64)),
        cancer_clones=mat_cancer)
    lineage = pd.DataFrame({"lineage_clone": types.astype(str), "lineage_leaf": leaf},
                           index=pd.Index(names))
    lineage.attrs["passengers"] = catalogue
    return catalogue, lineage
