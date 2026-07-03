"""Lineage-tree export (DESIGN_features §I).

The engine records the clone genealogy as ``tumor.genotypes_parents`` — a flat ``{child_gid ->
parent_gid}`` map over cancer genotypes (each mutation event spawns a child genotype off its parent
representative). This module materialises that into a first-class tree object with the two things
downstream analyses need:

  * **tree distances** — the number of division/mutation steps between two genotypes on the tree
    (via their lowest common ancestor). This is the "lineage distance" the PEtracer decomposition
    weights cells by, and the metric a phylogenetic-autocorrelation method (Hotspot / PhyloVision)
    runs on;
  * **Newick** — the standard interchange string external phylogenetic tools consume.

The registry ``tumor.genotypes`` keeps *every* genotype ever created (including extinct internal
ancestors), so LCA distances are always well defined even when a present clone's ancestors have died
out. All ids are strings.
"""
import numpy as np


class LineageTree:
    """A clone genealogy: parent map + LCA tree distances + Newick, over genotype ids.

    Parameters
    ----------
    parent : dict[str, str]
        ``child_gid -> parent_gid``. The root(s) are the ids that never appear as a key.
    nodes : list[str]
        The genotypes of interest (e.g. the present cancer clones). Distances/Newick are defined
        over the full ancestry reachable from these via ``parent``, so internal ancestors need not
        be listed here.
    root : str, optional
        The founder id. If omitted it is inferred (the common ancestor / first root encountered).
    """

    def __init__(self, parent, nodes, root=None):
        self.parent = dict(parent)
        self.nodes = list(nodes)
        # Path to root for every node reachable from `nodes` (memoised); this also fixes the depth.
        self._path = {}
        self._depth = {}
        roots = set()
        for g in self.nodes:
            self._path_to_root(g, roots)
        self.roots = sorted(roots)
        self.root = root if root is not None else (self.roots[0] if self.roots else None)

    # -- ancestry ---------------------------------------------------------------------
    def _path_to_root(self, g, roots=None):
        """Return the ancestor chain ``[g, parent(g), ..., root]`` (memoised); records depths."""
        cached = self._path.get(g)
        if cached is not None:
            return cached
        path = []
        cur = g
        seen = set()
        while True:
            path.append(cur)
            seen.add(cur)
            nxt = self.parent.get(cur)
            if nxt is None or nxt in seen:      # root reached (or a defensive cycle guard)
                break
            cur = nxt
        if roots is not None:
            roots.add(path[-1])
        # depth = distance from its root; fill for every node on the chain
        n = len(path)
        for k, node in enumerate(path):
            self._path.setdefault(node, path[k:])
            self._depth.setdefault(node, n - 1 - k)
        return self._path[g]

    def depth(self, g):
        if g not in self._depth:
            self._path_to_root(g)
        return self._depth[g]

    def lca_distance(self, a, b):
        """Tree distance = #edges from ``a`` to ``b`` through their lowest common ancestor.

        Genotypes on different roots (no shared ancestor) are treated as joined through a virtual
        super-root, i.e. the distance is ``depth(a) + depth(b)`` — a conservative separation."""
        if a == b:
            return 0
        pa = self._path_to_root(a)
        ancestors = set(pa)
        da, db = self.depth(a), self.depth(b)
        for x in self._path_to_root(b):
            if x in ancestors:
                return da + db - 2 * self.depth(x)
        return da + db

    def distance_matrix(self, ids=None):
        """Symmetric (n, n) matrix of pairwise LCA tree distances over ``ids`` (default: nodes)."""
        ids = list(self.nodes if ids is None else ids)
        n = len(ids)
        D = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                D[i, j] = D[j, i] = self.lca_distance(ids[i], ids[j])
        return D

    # -- Newick -----------------------------------------------------------------------
    def _children_map(self):
        children = {}
        # every node reachable through the memoised paths (present nodes + their ancestors)
        allnodes = set(self._depth)
        for g in allnodes:
            p = self.parent.get(g)
            if p is not None and p in allnodes:
                children.setdefault(p, []).append(g)
        return children

    def newick(self):
        """Newick string for the tree (unit branch lengths). Multiple roots join at a virtual root."""
        children = self._children_map()

        def render(node):
            kids = sorted(children.get(node, []))
            if not kids:
                return str(node)
            inner = ",".join(f"{render(k)}:1" for k in kids)
            return f"({inner}){node}"

        if len(self.roots) == 1:
            return render(self.roots[0]) + ";"
        inner = ",".join(f"{render(r)}:1" for r in self.roots)
        return f"({inner})root;"


def to_lineage_tree(tumor, cancer_only=True):
    """Build a :class:`LineageTree` from ``tumor.genotypes_parents``.

    Parameters
    ----------
    tumor : GenotypeTumor
        A grown tumour (any model exposing ``genotypes_parents`` + ``genotypes``).
    cancer_only : bool
        Restrict the node set to cancer genotypes (the default; normal cell types don't divide and
        so form no tree). Ancestors are always kept for distances regardless.
    """
    parents = dict(getattr(tumor, "genotypes_parents", {}))
    genos = getattr(tumor, "genotypes", {})
    if cancer_only:
        nodes = [g for g in genos if getattr(genos[g], "type", None) == "cancer"]
    else:
        nodes = list(genos)
    root = getattr(tumor, "founder_id", None)
    return LineageTree(parents, nodes, root=root)


def to_newick(tumor, cancer_only=True):
    """Convenience: the Newick string of the tumour's clone tree."""
    return to_lineage_tree(tumor, cancer_only=cancer_only).newick()
