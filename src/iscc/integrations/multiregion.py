"""Multi-region "sample trees are not phylogenies" analysis (Alves, Prieto & Posada 2017).

Alves, Prieto & Posada (*Multiregional Tumor Trees Are Not Phylogenies*, PMC5549612) argue that a
tree built from **bulk multi-region samples** is not a phylogeny: each region is an *admixture* — a
mixture of clones at different proportions — so a "sample tree" drawn from regional mutational/VAF
profiles reflects **similarity, not evolutionary history**. The signature artifacts are *spurious
parallel mutations* (homoplasy the reconstruction infers as independent origins for a mutation that
truly arose once), biased divergence, and reversed mutation ordering. Their proposed fix:
**deconvolve clones per region first**, then build the *clone* tree.

iscc can turn that illustrative argument into a **quantitative benchmark with a ground-truth answer
key**, because it has *both* the true clone phylogeny (``genotypes_parents`` / :mod:`.lineage`) *and*
real spatial admixture (clonal territories that intermix — tune via the cancer ``dispersal_rate``).
This module provides the self-contained machinery (no ete3 / dendropy needed):

  * :func:`true_origin_counts` — the **answer key**: for every locus, the true number of independent
    mutational origins, from Fitch parsimony on the *true* clone tree (pruned to the observed clones).
    Under the engine's per-allele infinite-sites model a locus arises once (origin == 1) unless it is
    genuinely recurrent; the single-origin loci are the clean substrate for scoring *spurious*
    parallelism (an inferred parallel origin for a locus the lineage shows arose once).
  * :func:`region_bulk_profiles` — the admixed observation: pooled bulk-DNA VAF per region (:class:`bulkDNA`).
  * :func:`oracle_clone_profiles` — the *fix*: iscc's per-cell clone truth as an oracle deconvolution
    (the bound on what a Clomial-style method could recover), giving per-clone profiles.
  * :func:`neighbor_joining` / :func:`hamming_nj_tree` — build the tree a study would draw (NJ on a
    region×region or clone×clone genetic distance; Hamming on presence is tree-additive under
    infinite sites).
  * :func:`count_spurious_parallel` — score homoplasy on a reconstructed tree, restricted to the
    truly single-origin loci (so *spurious* excludes genuine recurrence — an honest count).
  * :func:`robinson_foulds` — compare the deconvolved clone tree to the true clone tree on matched
    clone leaves.

See ``validation/validate_multiregion_phylo.py`` for the sweep and figure.
"""
import numpy as np
import pandas as pd

from .lineage import to_lineage_tree


# =============================================================================================
# Tree utilities (self-contained: neighbour joining, Fitch parsimony, Robinson–Foulds)
# =============================================================================================
def neighbor_joining(D):
    """Saitou–Nei neighbour joining on a square distance matrix ``D`` (n x n).

    Returns an adjacency dict ``{node: {neighbour: branch_length}}`` over the ``n`` input leaves
    ``0..n-1`` and the ``n-2`` internal nodes ``n..2n-3`` it creates (an unrooted binary tree).
    """
    D = np.asarray(D, dtype=float).copy()
    n = D.shape[0]
    if n < 2:
        raise ValueError("neighbour joining needs >= 2 leaves")
    adj = {i: {} for i in range(n)}
    active = list(range(n))
    nxt = n
    while len(active) > 2:
        m = len(active)
        r = {i: sum(D[i, j] for j in active if j != i) for i in active}
        best = None
        for a_i, i in enumerate(active):
            for j in active[a_i + 1:]:
                q = (m - 2) * D[i, j] - r[i] - r[j]
                if best is None or q < best[0]:
                    best = (q, i, j)
        _, i, j = best
        u = nxt
        nxt += 1
        di = 0.5 * D[i, j] + (r[i] - r[j]) / (2 * (m - 2))
        dj = D[i, j] - di
        di, dj = max(di, 0.0), max(dj, 0.0)
        adj.setdefault(u, {})
        adj[u][i] = adj[i][u] = di
        adj[u][j] = adj[j][u] = dj
        newd = {k: 0.5 * (D[i, k] + D[j, k] - D[i, j]) for k in active if k not in (i, j)}
        Dn = np.zeros((nxt, nxt))
        Dn[:D.shape[0], :D.shape[1]] = D
        D = Dn
        for k, dk in newd.items():
            D[u, k] = D[k, u] = dk
        active = [k for k in active if k not in (i, j)] + [u]
    a, b = active
    adj[a][b] = adj[b][a] = D[a, b]
    return adj


def root_children(adj, root):
    """Orient an (unrooted) adjacency dict into a ``{node: [children]}`` map rooted at ``root``."""
    children = {}
    seen = {root}
    stack = [root]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                children.setdefault(u, []).append(v)
                stack.append(v)
    return children


def hamming_nj_tree(P):
    """Neighbour-joining tree from a binary presence matrix ``P`` (leaves x loci).

    Uses Hamming distance (number of differing loci), which is the additive path length on the true
    tree under infinite sites — so NJ is statistically consistent for the clone tree. Returns
    ``(children, root)`` where ``root`` is the first internal node id (``P.shape[0]``).
    """
    P = np.asarray(P)
    n = P.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = float((P[i] != P[j]).sum())
    adj = neighbor_joining(D)
    return root_children(adj, n), n


def fitch_length(children, state, root):
    """Fitch small-parsimony length (min # state changes) of a binary character on a rooted tree.

    ``children`` is a ``{node: [children]}`` map; ``state`` gives the observed 0/1 value at each
    *leaf* (missing leaves default to 0). The length is root-independent for a binary character, so
    any rooting gives the true minimum number of origins/losses.
    """
    changes = [0]

    def rec(n):
        kids = children.get(n, [])
        if not kids:
            return {int(state.get(n, 0))}
        sets = [rec(k) for k in kids]
        inter = set(sets[0])
        for s in sets[1:]:
            inter &= s
        if inter:
            return inter
        changes[0] += 1
        u = set()
        for s in sets:
            u |= s
        return u

    rec(root)
    return changes[0]


def bipartitions(children, root, leaf_labels):
    """Non-trivial bipartitions (splits) of a tree, as ``frozenset`` of the smaller side's labels.

    ``leaf_labels`` maps each leaf node to its external label (shared across trees for comparison).
    Returns ``(splits, all_labels)``.
    """
    under = {}

    def dfs(n):
        kids = children.get(n, [])
        if not kids:
            under[n] = frozenset([leaf_labels[n]])
            return
        s = set()
        for k in kids:
            dfs(k)
            s |= under[k]
        under[n] = frozenset(s)

    dfs(root)
    all_labels = under[root]
    splits = set()
    for n, side in under.items():
        if n == root:
            continue
        small = min(side, all_labels - side, key=len)
        if 1 < len(small) < len(all_labels):
            splits.add(frozenset(small))
    return splits, all_labels


def robinson_foulds(children_a, root_a, leafmap_a, children_b, root_b, leafmap_b):
    """Robinson–Foulds comparison of two trees on their *shared* leaf labels.

    Returns a dict with the symmetric-difference distance ``rf``, the number of splits in each tree
    (restricted to shared leaves), and ``recall`` = fraction of tree-B (truth) splits recovered in
    tree A. NJ produces a fully-resolved tree while the true clone tree carries polytomies, so
    ``recall`` (splits of the truth that are recovered) is the cleaner "did we recover history?"
    read; ``rf`` additionally penalises A's extra resolution.
    """
    pa, la = bipartitions(children_a, root_a, leafmap_a)
    pb, lb = bipartitions(children_b, root_b, leafmap_b)
    common = la & lb

    def restrict(parts):
        out = set()
        for s in parts:
            r = frozenset(s & common)
            if 1 < len(r) < len(common):
                out.add(r)
        return out

    pa, pb = restrict(pa), restrict(pb)
    rf = len(pa ^ pb)
    recall = len(pa & pb) / max(len(pb), 1)
    return dict(rf=rf, n_a=len(pa), n_b=len(pb), recall=recall, n_shared=len(common))


# =============================================================================================
# The true clone tree (answer key)
# =============================================================================================
def clone_lineage_tree(tumor, present_clones=None):
    """The true clone genealogy pruned to the *observed* clones.

    Builds the genotype parent tree (:func:`.lineage.to_lineage_tree`), prunes extinct tips that are
    not observed, and attaches a zero-branch pseudo-leaf ``"<gid>#s"`` to every observed clone that
    is *internal* — so the leaf set is exactly the observed clones (an ancestral-but-extant clone is
    both an internal node and a leaf). Returns ``(children, root, leaf_clone)`` where ``leaf_clone``
    maps each leaf node to its clone id.
    """
    lt = to_lineage_tree(tumor)
    parent = lt.parent
    allnodes = set(lt._depth)
    children = {}
    for g in allnodes:
        p = parent.get(g)
        if p in allnodes:
            children.setdefault(p, []).append(g)
    root = "__root__"
    children[root] = list(lt.roots)

    present = set(map(str, present_clones)) if present_clones is not None else set(
        g for g in allnodes if str(getattr(tumor.genotypes.get(g, None), "type", "")) == "cancer")
    # Prune extinct tips (nodes not observed with no surviving children) to a fixpoint.
    changed = True
    while changed:
        changed = False
        for n in list(children.keys()):
            kids = [k for k in children[n] if (children.get(k) or k in present)]
            if len(kids) != len(children[n]):
                changed = True
            if kids:
                children[n] = kids
            else:
                del children[n]
                changed = True
    # Pseudo-leaf for observed internal clones; and register real leaves.
    leaf_clone = {}
    for g in list(children.keys()):
        if g in present:
            ps = "{}#s".format(g)
            children[g] = children[g] + [ps]
            leaf_clone[ps] = g
    reachable = set(children) | {c for ks in children.values() for c in ks}
    for n in reachable:
        if not children.get(n) and n in present:
            leaf_clone[n] = n
    return children, root, leaf_clone


def _clone_presence(cell_data, gid=None):
    """``{clone_id: bool array over loci}`` — a locus is present in a clone if any of its cells carry
    the SNV (constant within a clone genotype). Returns ``(presence, loci, gid, present_clones)``."""
    snv = cell_data["cell_snv"]
    if gid is None:
        gid = cell_data["cell_type"].iloc[:, 0].astype(str).values
    gid = np.asarray(gid).astype(str)
    values = snv.values > 0
    present = list(pd.unique(gid))
    presence = {g: values[gid == g].any(axis=0) for g in present}
    return presence, list(snv.columns), gid, present


def true_origin_counts(tumor, cell_data=None):
    """The **answer key**: per-locus number of independent mutational origins on the true clone tree.

    For each locus, Fitch parsimony on the pruned true clone tree (:func:`clone_lineage_tree`) with
    per-clone presence gives the minimum number of origins consistent with the true topology — 1 for
    a clean single-origin (monophyletic) locus, >= 2 for genuine recurrence. Returns a dict:
    ``origins`` (int array over loci), ``loci`` (columns), ``single`` (origins == 1), plus the
    reusable ``children``/``root``/``leaf_clone`` clone tree and ``carrier_count`` (# clones per locus).
    """
    cell_data = tumor.cell_data if cell_data is None else cell_data
    presence, loci, gid, present = _clone_presence(cell_data)
    children, root, leaf_clone = clone_lineage_tree(tumor, present)
    leaves = list(leaf_clone)
    Pclone = np.array([presence[g] for g in present])          # clones x loci
    carrier_count = Pclone.sum(axis=0)
    L = len(loci)
    origins = np.zeros(L, dtype=int)
    n_leaf = len(leaves)
    for l in range(L):
        s = int(carrier_count[l])
        if s == 0:
            continue
        state = {lf: int(presence[leaf_clone[lf]][l]) for lf in leaves}
        if 0 < sum(state.values()) < n_leaf:
            origins[l] = fitch_length(children, state, root)
        else:
            origins[l] = 1
    return dict(origins=origins, loci=loci, single=(origins == 1), carrier_count=carrier_count,
                children=children, root=root, leaf_clone=leaf_clone, present_clones=present,
                presence=presence, gid=gid)


# =============================================================================================
# Observations: admixed region bulk, and the oracle clone deconvolution (the fix)
# =============================================================================================
def region_bulk_profiles(cell_data, region_series, loci=None, n_reads=4_000_000,
                         presence_thresh=0.05, seed=7):
    """Pooled **bulk-DNA VAF** per multi-region biopsy region (:class:`bulkDNA`) — the admixture.

    Each region's cells are pooled into one deep bulk sample; the copy-number-weighted VAF over the
    mixture is exactly the admixed profile a multi-region study measures. Returns ``(V, P, regions)``
    with ``V`` the (regions x loci) VAF matrix, ``P = V > presence_thresh`` the presence matrix, and
    ``regions`` the sorted region labels aligned to the rows.
    """
    from ..data.dna import bulkDNA

    loci = list(cell_data["cell_snv"].columns) if loci is None else list(loci)
    regions = sorted(pd.unique(region_series.values))
    V = []
    for r in regions:
        cells = list(region_series.index[region_series.values == r])
        b = bulkDNA(breadth="wgs", seed=seed, n_reads=n_reads).run(cell_data, cell_subset=cells)
        V.append(b.observed_data["vaf"].reindex(loci).fillna(0.0).values)
    V = np.asarray(V)
    return V, (V > presence_thresh).astype(int), regions


def oracle_clone_profiles(cell_data, region_series, gid=None, min_cells=3):
    """The *fix* — iscc's per-cell clone truth as an **oracle deconvolution**.

    Assigns every sampled cell to its true clone (the bound on what a Clomial-style deconvolution
    could recover), then forms a consensus presence profile per clone with >= ``min_cells`` sampled
    cells. Returns ``(P_clone, clone_labels)`` — the (clones x loci) presence matrix and clone ids.
    """
    snv = cell_data["cell_snv"]
    if gid is None:
        gid = cell_data["cell_type"].iloc[:, 0].astype(str).values
    gid = np.asarray(gid).astype(str)
    name_to_pos = {c: i for i, c in enumerate(snv.index)}
    sampled = list(region_series.index)
    sg = np.array([gid[name_to_pos[c]] for c in sampled])
    counts = pd.Series(sg).value_counts()
    use = [c for c in counts.index if counts[c] >= min_cells]
    P = []
    for c in use:
        cells = [cn for cn, g in zip(sampled, sg) if g == c]
        rows = snv.loc[cells].values > 0
        P.append((rows.mean(axis=0) > 0.5).astype(int))
    return np.asarray(P), list(use)


# =============================================================================================
# Scoring
# =============================================================================================
def count_spurious_parallel(P, single_mask, children=None, root=None):
    """Count **spurious parallel mutations** on a reconstructed tree with leaves given by rows of ``P``.

    Builds a Hamming NJ tree from the presence matrix ``P`` (leaves x loci) unless a tree is passed,
    then for each locus computes the reconstruction's Fitch length (inferred # origins). A locus is
    *spurious parallel* if it is **truly single-origin** (``single_mask``) yet the reconstruction
    infers >= 2 origins — an admixture artifact, since the lineage shows it arose once. Restricting
    the denominator to single-origin, region-polymorphic loci makes the count honest (genuine
    recurrence is excluded). Returns a dict: ``spurious``, ``denom``, ``rate``, ``origins``.
    """
    P = np.asarray(P)
    n_leaves = P.shape[0]
    if children is None:
        children, root = hamming_nj_tree(P)
    L = P.shape[1]
    origins = np.zeros(L, dtype=int)
    col_sum = P.sum(axis=0)
    poly = (col_sum > 0) & (col_sum < n_leaves)
    for l in np.where(poly)[0]:
        state = {i: int(P[i, l]) for i in range(n_leaves)}
        origins[l] = fitch_length(children, state, root)
    single = np.asarray(single_mask, dtype=bool)
    spurious = int(np.sum(single & (origins >= 2)))
    denom = int(np.sum(single & poly))
    return dict(spurious=spurious, denom=denom, rate=spurious / max(denom, 1), origins=origins)


def ordering_reversal_rate(carrier_count, region_vaf, single_mask, clone_carriers,
                           max_pairs=200, seed=0):
    """Rate at which admixed bulk VAF **reverses** the true ancestor→descendant mutation ordering.

    For truly single-origin loci whose clone-carrier sets are strictly nested (locus A's carriers ⊃
    locus B's ⇒ A is ancestral, hence at least as clonal), a reversal is a pair where the mean
    regional VAF ranks the descendant *above* the ancestor. ``clone_carriers`` maps locus index ->
    frozenset of carrier-clone indices. Returns ``(rate, n_pairs)``.
    """
    single = np.asarray(single_mask, dtype=bool)
    rng = np.random.default_rng(seed)
    idx = np.where(single & (carrier_count > 1) & (carrier_count < carrier_count.max()))[0]
    if len(idx) > max_pairs:
        idx = rng.choice(idx, size=max_pairs, replace=False)
    idx = list(idx)
    pairs = rev = 0
    for a in idx:
        sa = clone_carriers[a]
        for b in idx:
            if a == b:
                continue
            sb = clone_carriers[b]
            if len(sa) > len(sb) and sa >= sb:              # A strict ancestor of B
                pairs += 1
                if region_vaf[a] < region_vaf[b] - 1e-9:
                    rev += 1
    return rev / max(pairs, 1), pairs


# =============================================================================================
# High-level: one multi-region reconstruction vs the truth
# =============================================================================================
def multiregion_phylogeny(tumor, region_series, answer_key=None, min_clone_cells=3,
                          n_reads=4_000_000, presence_thresh=0.05, seed=7):
    """Score a multi-region biopsy: the naive region "sample tree" vs the oracle-deconvolved clone tree.

    Returns a dict with, for the **naive** region tree and the **deconvolved** clone tree, the
    spurious-parallel count/denominator/rate; the Robinson–Foulds comparison of the clone tree to the
    true clone tree; and the number of regions / clones. ``answer_key`` (from
    :func:`true_origin_counts`) is computed if not supplied.
    """
    cd = tumor.cell_data
    ak = true_origin_counts(tumor, cd) if answer_key is None else answer_key
    single = ak["single"]

    V, P_region, regions = region_bulk_profiles(cd, region_series, ak["loci"],
                                                n_reads=n_reads, presence_thresh=presence_thresh,
                                                seed=seed)
    naive = count_spurious_parallel(P_region, single)

    P_clone, clone_labels = oracle_clone_profiles(cd, region_series, ak["gid"],
                                                  min_cells=min_clone_cells)
    fix = count_spurious_parallel(P_clone, single)
    # RF of the deconvolved clone tree vs the true clone tree
    rf = dict(rf=None, recall=None, n_a=None, n_b=None, n_shared=0)
    if len(clone_labels) >= 4:
        rc_children, rc_root = hamming_nj_tree(P_clone)
        leafmap = {i: clone_labels[i] for i in range(len(clone_labels))}
        rf = robinson_foulds(rc_children, rc_root, leafmap,
                             ak["children"], ak["root"], ak["leaf_clone"])
    return dict(n_regions=len(regions), n_clones=len(clone_labels),
                naive=naive, fix=fix, rf=rf, region_vaf=V.mean(axis=0), answer_key=ak)
