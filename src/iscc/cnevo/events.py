"""Derived copy-number event log.

The engine keeps no CNA event history — ``event_bits`` / ``event_groups`` track *epistasis*
events, not copy number. But the genealogy plus the full genotype registry make the log exactly
recoverable: every genotype differs from its parent by the one mutating division that created it,
so diffing per-homolog copy counts along ``genotypes_parents`` recovers each CNA with its segment,
homolog and direction. Nothing is inferred — this is the true event list, not a parsimony
reconstruction.

Every engine CNA changes **one segment by one copy on one homolog**
(:meth:`iscc.tumor.components.cell.CancerCell.mutate`), so an amplification and a deletion of the
same segment never cancel within one edge. A whole-genome duplication doubles every copy at once
and is emitted as a single ``wgd`` row rather than one row per segment.

The **row index is the event id** — the convention every shared-event metric downstream relies on.
Re-sorting or rewriting the table silently changes those metrics, so treat the order as part of
the contract.
"""
from collections import Counter

import numpy as np
import pandas as pd

__all__ = ["cna_event_table", "inherited_event_counters", "pairwise_shared_matrix"]

_COLUMNS = ["clone", "parent", "type", "segment", "allele", "copies", "driver"]


def _allele_counts(rep, n_seg):
    """``(n_seg, 2)`` per-homolog copy counts for a genotype, or ``None`` if it has no genome."""
    if rep is None or not hasattr(rep, "genome") or not rep.genome:
        return None
    return np.array([(len(s["p"]), len(s["m"])) for s in rep.genome], dtype=int)


def _driver_segments(tumor):
    """Boolean mask over segments: does the segment contain any driver position?"""
    n_seg = int(tumor.n_segments)
    sizes = np.asarray(tumor.selection.segment_sizes, dtype=int)
    offs = np.concatenate([[0], np.cumsum(sizes)]).astype(int)
    idx = np.concatenate([np.asarray(tumor.selection.get_oncogenes(), dtype=int),
                          np.asarray(tumor.selection.get_tsgs(), dtype=int)])
    mask = np.zeros(n_seg, dtype=bool)
    for s in range(n_seg):
        mask[s] = bool(np.any((idx >= offs[s]) & (idx < offs[s + 1])))
    return mask


def cna_event_table(tumor, clones=None):
    """One row per copy-number event, attributed to the genotype on whose edge it arose.

    Parameters
    ----------
    tumor : GenotypeTumor
    clones : sequence of str, optional
        Restrict to events on the ancestry of these genotypes. Defaults to every cancer genotype
        in the registry, so the table covers the whole genealogy including extinct lineages.

    Returns
    -------
    DataFrame with columns ``clone, parent, type, segment, allele, copies, driver``.
    ``type`` is ``amplification`` / ``deletion`` / ``wgd``; ``allele`` is ``p`` / ``m`` (``None``
    for WGD); ``copies`` is the signed change in that homolog's copy count; ``driver`` is whether
    the segment carries any driver position.
    """
    parents = tumor.genotypes_parents
    genos = tumor.genotypes
    n_seg = int(tumor.n_segments)

    if clones is None:
        nodes = [g for g in genos if tumor._is_cancer(g)]
    else:                                  # every ancestor of the requested clones
        nodes, seen = [], set()
        for c in map(str, clones):
            cur = c
            while cur is not None and cur not in seen:
                seen.add(cur)
                if cur in genos:
                    nodes.append(cur)
                cur = parents.get(cur)
    # Creation order keeps the event ids stable and ancestors ahead of descendants.
    nodes = sorted(set(nodes), key=lambda g: getattr(genos[g], "ord", 0))

    drv = _driver_segments(tumor)
    rows = []
    for gid in nodes:
        pid = parents.get(gid)
        if pid is None or pid not in genos:
            continue                       # founder: nothing to diff against
        child = _allele_counts(genos[gid], n_seg)
        parent = _allele_counts(genos[pid], n_seg)
        if child is None or parent is None or child.shape != parent.shape:
            continue
        diff = child - parent
        if not diff.any():
            continue                       # an SNV-only division: no CN change
        # WGD doubles every non-zero homolog at once; recognise it before per-segment diffing so
        # it is one event rather than 2*n_seg coincidental amplifications.
        if np.array_equal(child, parent * 2) and parent.sum() > 0:
            rows.append((gid, pid, "wgd", None, None, int(child.sum() - parent.sum()), True))
            continue
        for seg in range(n_seg):
            for a, hap in enumerate(("p", "m")):
                d = int(diff[seg, a])
                if d == 0:
                    continue
                rows.append((gid, pid, "amplification" if d > 0 else "deletion",
                             int(seg), hap, d, bool(drv[seg])))
    return pd.DataFrame(rows, columns=_COLUMNS)


def inherited_event_counters(events, clones, children, root):
    """Per-clone multiset of the event ids inherited along its root→clone path.

    Parameters
    ----------
    events : DataFrame
        Output of :func:`cna_event_table`; the row index is the event id.
    clones : sequence of str
    children : dict
        ``node -> [child, ...]`` for the true clone tree.
    root : hashable
    """
    by_clone = {}
    for eid, clone in zip(events.index, events["clone"].astype(str)):
        by_clone.setdefault(clone, []).append(int(eid))

    parent = {c: p for p, ks in children.items() for c in ks}

    def path(node):
        out, cur, seen = [], node, set()
        while cur is not None and cur not in seen:
            seen.add(cur)
            out.append(cur)
            if cur == root:
                break
            cur = parent.get(cur)
        return out

    counters = []
    for c in map(str, clones):
        ctr = Counter()
        for node in path(c):
            # `clone_lineage_tree` attaches a zero-length pseudo-leaf "<gid>#s" under every
            # observed INTERNAL clone, so the leaf set is exactly the observed clones. It is the
            # same clone, and its real node is the very next step of the walk -- counting the
            # pseudo-leaf too would double every event on that clone's own edge.
            if str(node).endswith("#s"):
                continue
            ctr.update(by_clone.get(str(node), []))
        counters.append(ctr)
    return counters


def pairwise_shared_matrix(counters):
    """``mat[i, j] = sum_k min(c_i[k], c_j[k])`` over shared event ids; zero diagonal."""
    n = len(counters)
    mat = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = counters[i], counters[j]
            small, large = (a, b) if len(a) <= len(b) else (b, a)
            v = float(sum(min(cnt, large[k]) for k, cnt in small.items() if k in large))
            mat[i, j] = mat[j, i] = v
    return mat
