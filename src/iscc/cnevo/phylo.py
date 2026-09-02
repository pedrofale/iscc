"""CN data quality (Q5) and tree-reconstruction potential (Q6).

Q5 asks whether a sampled clone set is a usable copy-number benchmark at all: are the clones
CN-distinguishable, and does every edge of the true tree carry copy-number signal? Q6 asks how
much of the true tree a CN-based method could recover in principle — Neighbour-Joining trees built
from several CN-derived distances, scored against the truth by Robinson–Foulds.

Both are computed on the **clone** tree. iscc's genealogy is over genotypes, so
:func:`~iscc.integrations.multiregion.clone_lineage_tree` pruned to the sampled clones is the true
tree, with one representative cell per clone.

The tree machinery is iscc's own (:mod:`iscc.integrations.multiregion`); this module adds the
normalisation and the CN-derived distances. :func:`normalized_rf` reports the ``[0, 1]`` quantity
CN-reconstructability results are usually stated in, where 0 is a perfect recovery, and ``None``
below 4 leaves where it is undefined.
"""
import numpy as np
from scipy.stats import spearmanr

from ..integrations.multiregion import (
    bipartitions, clone_lineage_tree, neighbor_joining, robinson_foulds, root_children,
)
from .events import cna_event_table, inherited_event_counters, pairwise_shared_matrix
from .profile import breakpoint_sets, clone_segment_cn, segment_coordinates

__all__ = ["normalized_rf", "nj_rf", "true_clone_tree", "data_quality",
           "reconstruction_potential"]


def true_clone_tree(tumor, clones):
    """``(children, root, leaf_of_clone)`` — the true tree with the sampled clones as its leaves."""
    clones = [str(c) for c in clones]
    children, root, leafmap = clone_lineage_tree(tumor, present_clones=clones)
    want = set(clones)
    leaf_of = {}
    for leaf, clone in leafmap.items():
        if clone in want and clone not in leaf_of:
            leaf_of[clone] = leaf
    return children, root, leaf_of


def normalized_rf(children_a, root_a, leafmap_a, children_b, root_b, leafmap_b):
    """Robinson–Foulds distance normalised to ``[0, 1]``; ``None`` when fewer than 4 shared leaves.

    The normaliser is ``|splits(A)| + |splits(B)|``, the largest symmetric difference the two split
    sets could have, so the score is in ``[0, 1]`` by construction.

    The textbook ``2*(n-3)`` normaliser is WRONG here and silently returns values above 1: it
    assumes both trees are fully resolved unrooted binary trees, whereas the true clone tree is
    rooted and carries polytomies, so it can hold up to ``n-2`` splits per tree. Observed in
    practice — an 8-clone tree scoring 1.1 — which is what this normaliser fixes.

    ``recall``, the fraction of the truth's splits that were recovered, is also returned. NJ always
    returns a fully-resolved tree while the truth has polytomies, so recall is the fairer "did we
    recover history?" read; the normalised RF additionally penalises NJ's invented resolution.
    ``rf_2n6`` carries the ``2*(n-3)`` value for comparison with studies that report it.
    """
    res = robinson_foulds(children_a, root_a, leafmap_a, children_b, root_b, leafmap_b)
    n = int(res["n_shared"])
    if n < 4:
        return None
    denom = float(res["n_a"] + res["n_b"])
    legacy = 2.0 * (n - 3)
    return dict(rf=float(res["rf"]), n_shared=n, recall=float(res["recall"]),
                n_a=int(res["n_a"]), n_b=int(res["n_b"]),
                normalized_rf=float(res["rf"] / denom) if denom > 0 else None,
                rf_2n6=float(res["rf"] / legacy) if legacy > 0 else None)


def nj_rf(D, labels, true_children, true_root, true_leafmap):
    """Build an NJ tree from distance matrix ``D`` and score it against the true tree.

    Returns the ``normalized_rf`` float, or ``None`` when there are too few leaves or the distance
    matrix is degenerate (all-equal distances give NJ nothing to work with).
    """
    D = np.asarray(D, dtype=float)
    n = D.shape[0]
    if n < 4 or not np.isfinite(D).all():
        return None
    off = D[~np.eye(n, dtype=bool)]
    if off.size == 0 or np.allclose(off, off[0]):
        return None                      # no signal: every pair equidistant
    try:
        adj = neighbor_joining(D)
        children = root_children(adj, n)          # first internal node id == n
        leafmap = {i: labels[i] for i in range(n)}
        res = normalized_rf(children, n, leafmap, true_children, true_root, true_leafmap)
    except Exception:
        return None
    return None if res is None else res["normalized_rf"]


def _tree_distance_matrix(children, root, leaves):
    """Topological (edge-count) distance between leaves on a rooted tree."""
    adj = {}
    for p, ks in children.items():
        for k in ks:
            adj.setdefault(p, []).append(k)
            adj.setdefault(k, []).append(p)
    idx = {l: i for i, l in enumerate(leaves)}
    n = len(leaves)
    D = np.zeros((n, n), dtype=float)
    for l in leaves:                                   # BFS from each leaf
        dist, stack = {l: 0}, [l]
        while stack:
            u = stack.pop(0)
            for v in adj.get(u, []):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    stack.append(v)
        for m in leaves:
            D[idx[l], idx[m]] = dist.get(m, np.nan)
    return D


def _cn_context(tumor, clones, coords=None):
    """Shared setup: true tree, event log, inherited-event counters, CN profiles, breakpoints."""
    clones = [str(c) for c in clones]
    children, root, leaf_of = true_clone_tree(tumor, clones)
    clones = [c for c in clones if c in leaf_of]          # keep only clones that made it onto the tree
    leaves = [leaf_of[c] for c in clones]
    events = cna_event_table(tumor, clones)
    counters = inherited_event_counters(events, leaves, children, root)
    coords = segment_coordinates(tumor) if coords is None else coords
    _, total_cn, _ = clone_segment_cn(tumor, clones)
    bps = breakpoint_sets(total_cn, coords)
    return dict(clones=clones, leaves=leaves, children=children, root=root, leaf_of=leaf_of,
                events=events, counters=counters, total_cn=total_cn, bps=bps)


# --------------------------------------------------------------------------------------
# Q5 — CN data quality
# --------------------------------------------------------------------------------------
def data_quality(tumor, clones, n_requested=None):
    """Is this clone set a usable copy-number benchmark?

    Two criteria, following the SISTEM study this mirrors:

    1. **Unique CN profiles** — no two sampled clones share a total-CN profile. Identical profiles
       are indistinguishable to any CN-based method, so they put a hard floor on reconstruction
       error no method can beat.
    2. **CNA on every leaf edge** — every parent→sampled-clone edge carries at least one CN event.
       An edge with none is invisible in copy number: the clone is CN-identical to its parent.

    ``trunk_fraction`` is the share of a clone's inherited events that *every* sampled clone also
    carries — the truncal burden. High trunk fraction means most of the CN signal is shared and
    little of it distinguishes clones, which is the mechanism behind non-reconstructable data.
    """
    ctx = _cn_context(tumor, clones)
    cl, total_cn, counters = ctx["clones"], ctx["total_cn"], ctx["counters"]
    n = len(cl)
    out = dict(n_clones=n,
               n_requested=int(n_requested) if n_requested is not None else n,
               under_sampled=bool(n_requested is not None and n < int(n_requested)))
    if n == 0:
        return out

    uniq = np.unique(total_cn, axis=0)
    dup = sum(1 for i in range(n) for j in range(i + 1, n)
              if np.array_equal(total_cn[i], total_cn[j]))
    out.update(all_unique_cnps=bool(uniq.shape[0] == n),
               n_unique_cnps=int(uniq.shape[0]), n_duplicate_pairs=int(dup))

    # per-edge event counts over the true tree (an edge is identified by its child node)
    per_clone = ctx["events"].groupby(ctx["events"]["clone"].astype(str)).size().to_dict()
    parent = {c: p for p, ks in ctx["children"].items() for c in ks}
    leaf_edges, internal_edges = [], []
    for node in set(parent):
        if str(node).endswith("#s"):
            continue                                   # zero-length pseudo-leaf, not a real edge
        k = per_clone.get(str(node), 0)
        (leaf_edges if not [x for x in ctx["children"].get(node, [])
                            if not str(x).endswith("#s")] else internal_edges).append(k)
    allc = leaf_edges + internal_edges
    out.update(
        leaf_edges_total=len(leaf_edges),
        leaf_edges_with_cna=int(sum(1 for k in leaf_edges if k > 0)),
        leaf_edges_empty=int(sum(1 for k in leaf_edges if k == 0)),
        all_leaf_edges_covered=bool(leaf_edges and all(k > 0 for k in leaf_edges)),
        n_edges_total=len(allc),
        n_edges_with_cna=int(sum(1 for k in allc if k > 0)),
        min_events_per_edge=int(min(allc)) if allc else 0,
        mean_events_per_edge=float(np.mean(allc)) if allc else float("nan"),
        median_events_per_edge=float(np.median(allc)) if allc else float("nan"),
        max_events_per_edge=int(max(allc)) if allc else 0,
        frac_edges_ge2_events=float(np.mean([k >= 2 for k in allc])) if allc else float("nan"),
    )

    if counters:
        shared = set(counters[0])
        for c in counters[1:]:
            shared &= set(c)
        mean_inherited = float(np.mean([len(c) for c in counters]))
        out.update(trunk_event_count=int(len(shared)),
                   mean_inherited_events=mean_inherited,
                   trunk_fraction=float(min(len(shared) / mean_inherited, 1.0))
                   if mean_inherited > 0 else float("nan"))

    # allele-level pathology: LOH and nullisomy over the sampled clones
    _, _, allele = clone_segment_cn(tumor, cl)
    if np.isfinite(allele).all():
        null = (total_cn <= 0)
        loh = (allele.min(axis=2) <= 0) & ~null
        out.update(frac_segments_nullisomy=float(null.mean()),
                   frac_segments_loh=float(loh.mean()))
    out["both_criteria_met"] = bool(out.get("all_unique_cnps") and
                                    out.get("all_leaf_edges_covered"))
    return out


# --------------------------------------------------------------------------------------
# Q6 — tree-reconstruction potential
# --------------------------------------------------------------------------------------
def reconstruction_potential(tumor, clones):
    """How recoverable is the true clone tree from copy number alone?

    Two families of read-out:

    *Phylogenetic signal* — ``phylo_signal_spearman`` is the correlation between how many CN events
    a pair of clones shares and how close they are on the true tree (negated, so higher is better);
    ``nearest_sister_recovered`` is the fraction of clones whose most-event-sharing partner really
    is their nearest tree neighbour. These need no tree building and say whether the signal exists.

    *Reconstruction* — normalised RF of NJ trees built from four CN distances: symmetric difference
    of inherited event sets (``nj_event_rf``), of CN breakpoint sets (``nj_breakpoint_rf``), and the
    two reciprocal-shared variants driven by how much pairs have in common rather than how much
    they differ. Lower is better; ``None`` means the metric was undefined (fewer than 4 clones, or
    a degenerate distance matrix).

    ``nj_rf_floor`` is the score NJ gets from the **true** tree distances, and is the reference the
    CN scores should be read against rather than against 0. It is generally NOT zero: the true clone
    tree carries polytomies while NJ always returns a fully-resolved binary tree, so NJ invents
    splits the truth does not contain and RF charges for them. A ``nj_event_rf`` equal to the floor
    is therefore a perfect reconstruction, not a mediocre one.

    It is a reference, NOT a lower bound: a CN distance can beat it, and does in practice. The
    topological distance gives every edge length 1, whereas an event-count distance also carries how
    much divergence each edge represents, and that extra information can resolve a polytomy the
    unit-length distances cannot. Do not treat ``score < floor`` as a bug.
    """
    ctx = _cn_context(tumor, clones)
    cl, counters, bps = ctx["clones"], ctx["counters"], ctx["bps"]
    n = len(cl)
    out = dict(n_clones=n)
    if n < 2:
        return {**out, "phylo_signal_spearman": None, "nearest_sister_recovered": None,
                "nj_event_rf": None, "nj_shared_events_rf": None,
                "nj_breakpoint_rf": None, "nj_shared_breakpoint_rf": None}

    shared = pairwise_shared_matrix(counters)
    tree_d = _tree_distance_matrix(ctx["children"], ctx["root"], ctx["leaves"])

    iu = np.triu_indices(n, 1)
    sig = None
    if n >= 3:
        a, b = shared[iu], tree_d[iu]
        if np.isfinite(a).all() and np.isfinite(b).all() and a.std() > 0 and b.std() > 0:
            rho = spearmanr(a, b).correlation
            sig = None if not np.isfinite(rho) else float(-rho)
    out["phylo_signal_spearman"] = sig

    rec, denom = 0, 0
    for i in range(n):
        denom += 1
        row = shared[i].copy()
        row[i] = -np.inf
        if not np.isfinite(row).any() or row.max() <= 0:
            continue
        best = set(np.flatnonzero(row == row.max()))
        td = tree_d[i].copy()
        td[i] = np.inf
        nearest = set(np.flatnonzero(td == td.min()))
        rec += bool(best & nearest)
    out["nearest_sister_recovered"] = float(rec / denom) if denom else None

    # event-set distances
    keys = [set(c) for c in counters]
    d_ev = np.zeros((n, n))
    d_sh = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d_ev[i, j] = d_ev[j, i] = float(len(keys[i] ^ keys[j]))
            d_sh[i, j] = d_sh[j, i] = 1.0 / (shared[i, j] + 1.0)
    # breakpoint-set distances
    d_bp = np.zeros((n, n))
    d_bsh = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d_bp[i, j] = d_bp[j, i] = float(len(bps[i] ^ bps[j]))
            d_bsh[i, j] = d_bsh[j, i] = 1.0 / (len(bps[i] & bps[j]) + 1.0)

    args = (ctx["leaves"], ctx["children"], ctx["root"],
            {ctx["leaf_of"][c]: ctx["leaf_of"][c] for c in cl})
    out.update(
        nj_event_rf=nj_rf(d_ev, *args),
        nj_shared_events_rf=nj_rf(d_sh, *args),
        nj_breakpoint_rf=nj_rf(d_bp, *args),
        nj_shared_breakpoint_rf=nj_rf(d_bsh, *args),
        # NJ handed the TRUE distances: the polytomy-induced floor these scores sit on.
        nj_rf_floor=nj_rf(tree_d, *args),
    )
    return out
