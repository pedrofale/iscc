"""Noble et al. (2022) evolutionary-mode indices for the genotype-count engine.

Noble characterise a tumour's *mode of evolution* by a handful of indices read off its clone
phylogeny: the mean number of driver mutations per cell (``n``), the clonal diversity (``D``,
inverse-Simpson over driver-mutation combinations), and the tree-balance index ``J1``.

A *clone* here is a **driver-mutation combination**: the set of mutated driver positions
(oncogene / TSG indices) a genotype carries. Genotypes that differ only in passenger SNVs or
copy number but share the same set of mutated drivers are the same clone, matching Noble's
definition. Driver positions are read from the same oncogene/TSG indices and per-genotype
``get_snvs`` already used by ``GenotypeTumor.make_cell_data``.

``J1`` is the Lemant et al. (2022) "robust, universal" tree-balance index in ``[0, 1]`` (the
metric Noble 2022 uses): a clone-size-weighted average of per-node normalised Shannon entropy
over the *clone* phylogeny. We obtain that phylogeny by contracting the genotype genealogy
(``genotypes_parents``) onto driver-mutation combinations, with node sizes = clone cell counts.
"""
import math

import numpy as np


def inverse_simpson(counts):
    """Inverse-Simpson diversity ``D = 1 / Σ pᵢ²`` over a set of group sizes.

    ``D`` is an effective number of equally abundant groups: ``D = k`` for ``k`` equal-sized
    groups, ``D = 1`` for a single dominant group. Zero/empty input returns ``0.0``.
    """
    c = np.asarray([v for v in counts], dtype=float)
    c = c[c > 0]
    total = c.sum()
    if total == 0:
        return 0.0
    p = c / total
    return float(1.0 / np.square(p).sum())


def _driver_indices(tumor):
    """Flat genome indices of the driver positions (oncogenes + TSGs)."""
    onc = tumor.selection.get_oncogenes()
    tsg = tumor.selection.get_tsgs()
    return np.concatenate([onc, tsg]).astype(int)


def _driver_combo(tumor, gid, driver_idx):
    """``frozenset`` of mutated driver positions (VAF > 0) carried by genotype ``gid``."""
    snv = tumor.genotypes[gid].get_snvs()
    return frozenset(driver_idx[snv[driver_idx] > 0].tolist())


def driver_combination_counts(tumor, counts=None, combo_cache=None):
    """Map each cancer driver-mutation combination -> total cell count.

    A combination is the ``frozenset`` of mutated driver positions (oncogene + TSG flat genome
    indices with VAF > 0) carried by a genotype. Cancer genotypes are pooled by combination and
    weighted by their cell counts; normal cells are excluded.

    ``counts`` overrides the live ``tumor.genotypes_counts`` with any ``{genotype_id: n_cells}``
    mapping -- e.g. a historical snapshot from ``tumor.traces`` -- so the index can be evaluated at
    any generation of the run, not only at its end. The genotype registry retains every genotype
    ever created, so past snapshots resolve. Defaults to the current population.

    ``combo_cache`` is an optional ``{genotype_id: frozenset}`` dict reused across calls. A
    genotype's driver combination never changes, so scanning a whole run of snapshots costs one
    pass over the distinct genotypes rather than one per snapshot.
    """
    driver_idx = _driver_indices(tumor)
    counts = tumor.genotypes_counts if counts is None else counts
    cache = {} if combo_cache is None else combo_cache
    combos = {}
    for gid, cnt in counts.items():
        if not tumor._is_cancer(gid):
            continue
        mutated = cache.get(gid)
        if mutated is None:
            mutated = cache[gid] = _driver_combo(tumor, gid, driver_idx)
        combos[mutated] = combos.get(mutated, 0) + cnt
    return combos


def clonal_diversity(tumor, counts=None):
    """Noble's clonal diversity ``D`` = inverse-Simpson over driver-mutation combinations.

    ``counts`` optionally supplies a population snapshot; see :func:`driver_combination_counts`.
    """
    return inverse_simpson(driver_combination_counts(tumor, counts).values())


def mean_drivers_per_cell(tumor, counts=None):
    """Noble's ``n`` = count-weighted mean number of mutated driver positions per cancer cell.

    Equivalently ``Σ i·pᵢ`` where ``pᵢ`` is the frequency of the driver-mutation combination
    carrying ``i`` drivers (the tree-depth analogue). ``nan`` when there are no cancer cells.
    ``counts`` optionally supplies a population snapshot; see :func:`driver_combination_counts`.
    """
    combos = driver_combination_counts(tumor, counts)
    total = sum(combos.values())
    if total == 0:
        return float("nan")
    return float(sum(len(combo) * cnt for combo, cnt in combos.items()) / total)


def tree_balance_j1(parents, sizes):
    """Lemant et al. (2022) ``J1`` tree-balance index of a rooted tree, in ``[0, 1]``.

    ``J1`` is the size-weighted mean, over internal nodes, of each node's normalised Shannon
    entropy of its child-subtree magnitudes. ``1`` = every internal split perfectly balanced;
    ``0`` = a fully imbalanced caterpillar. It is robust to (and sensitive to) internal node
    sizes, which is why it suits clone phylogenies whose ancestral clones still carry cells.

    Definitions (Lemant et al. 2022):
      * subtree magnitude ``S_i`` = node ``i``'s own size **plus** all descendant sizes;
      * ``S*_i = Σ_{j∈children(i)} S_j`` — descendant magnitude, **excluding** ``i``'s own size;
      * node balance ``W_i = -Σ_j (S_j/S*_i)·log_{d}(S_j/S*_i)`` over the ``d = outdegree(i)``
        children (``W_i = 0`` for ``d < 2``). The node's *own* size never enters its own entropy
        — only its children's subtree magnitudes do — but it does enter ``S_i`` seen by the parent;
      * ``J1 = Σ_i S*_i·W_i / Σ_i S*_i`` over internal nodes with ``S*_i > 0``.

    Args:
        parents: ``dict`` child -> parent; roots are nodes that never appear as a key.
        sizes: ``dict`` node -> non-negative size. Nodes absent here are zero-size (structural).
    Returns:
        ``J1`` as a float, or ``nan`` for a tree with no branching mass (e.g. a single node).
    """
    nodes = set(sizes) | set(parents) | set(parents.values())
    children = {u: [] for u in nodes}
    for child, parent in parents.items():
        children[parent].append(child)
    roots = [u for u in nodes if u not in parents]

    # Subtree magnitudes S_i (own size + descendants), iterative post-order so deep
    # (caterpillar) clone trees don't overflow the recursion limit.
    S = {}
    for root in roots:
        stack = [(root, False)]
        while stack:
            u, expanded = stack.pop()
            if expanded:
                S[u] = float(sizes.get(u, 0.0)) + sum(S[c] for c in children[u])
            else:
                stack.append((u, True))
                for c in children[u]:
                    stack.append((c, False))

    num = den = 0.0
    for i in nodes:
        kids = children[i]
        star = sum(S[c] for c in kids)          # S*_i, excludes node i's own size
        if star <= 0:                           # leaf, or whole subtree is zero-size
            continue
        den += star                             # single-child nodes weigh in but add 0 balance
        if len(kids) < 2:
            continue
        entropy = 0.0
        for c in kids:
            p = S[c] / star
            if p > 0:
                entropy -= p * math.log(p)
        num += star * (entropy / math.log(len(kids)))
    return num / den if den > 0 else float("nan")


def clone_tree(tumor, counts=None, combo_cache=None):
    """Contract the genotype genealogy onto the driver-mutation *clone* phylogeny.

    Each clone node is a maximal connected block of genotypes sharing one driver combination
    (so a convergent combination arising in two lineages stays two nodes, as the tree demands).
    The node is keyed by the block's root genotype id; its size is the summed cell count of the
    block's *extant* genotypes; its parent is the clone node of the block-root's parent genotype.
    Ancestral blocks that are extinct but bridge extant clones are kept (as branch points) with
    whatever extant mass flows through them.

    Returns ``(parents, sizes)`` ready for :func:`tree_balance_j1`.

    ``counts`` optionally supplies a population snapshot in place of the live
    ``tumor.genotypes_counts``; see :func:`driver_combination_counts`, which also documents
    ``combo_cache``.
    """
    driver_idx = _driver_indices(tumor)
    counts = tumor.genotypes_counts if counts is None else counts
    combo_cache = {} if combo_cache is None else combo_cache

    def combo(gid):
        if gid not in combo_cache:
            combo_cache[gid] = _driver_combo(tumor, gid, driver_idx)
        return combo_cache[gid]

    root_cache = {}

    def clone_root(gid):
        """Topmost genotype of ``gid``'s same-combination block (its clone's identity)."""
        if gid in root_cache:
            return root_cache[gid]
        p = tumor.genotypes_parents.get(gid)
        if p is not None and tumor._is_cancer(p) and combo(p) == combo(gid):
            r = clone_root(p)
        else:
            r = gid
        root_cache[gid] = r
        return r

    sizes = {}
    for gid, cnt in counts.items():
        if not tumor._is_cancer(gid):
            continue
        r = clone_root(gid)
        sizes[r] = sizes.get(r, 0) + cnt

    # Walk clone-node parents up to the founder, materialising any extinct bridging blocks.
    parents = {}
    seen = set(sizes)
    stack = list(sizes)
    while stack:
        r = stack.pop()
        p = tumor.genotypes_parents.get(r)
        if p is None or not tumor._is_cancer(p):
            continue                            # founder block / non-cancer ancestor -> a root
        pr = clone_root(p)
        parents[r] = pr
        if pr not in seen:
            seen.add(pr)
            sizes.setdefault(pr, 0)
            stack.append(pr)
    return parents, sizes


def tree_balance(tumor, counts=None):
    """Noble/Lemant ``J1`` tree-balance index of the tumour's driver-clone phylogeny.

    ``counts`` optionally supplies a population snapshot; see :func:`driver_combination_counts`.
    """
    parents, sizes = clone_tree(tumor, counts)
    return tree_balance_j1(parents, sizes)


def mode_indices(tumor, counts=None):
    """Noble ``(n, D, J1)`` evolutionary-mode indices, plus the clone count, in one pass.

    Returns ``dict(n=..., D=..., J1=..., n_clones=...)`` over the tumour's *cancer* population.
    With no cancer cells everything is ``nan`` (``n_clones`` ``0``); ``J1`` is also ``nan`` when
    the clone phylogeny has no branching (a single clone), where tree balance is undefined.
    """
    combos = driver_combination_counts(tumor, counts)
    total = sum(combos.values())
    if total == 0:
        return dict(n=float("nan"), D=float("nan"), J1=float("nan"), n_clones=0)
    n = sum(len(combo) * cnt for combo, cnt in combos.items()) / total
    return dict(
        n=float(n),
        D=inverse_simpson(combos.values()),
        J1=tree_balance(tumor, counts),
        n_clones=len(combos),
    )
