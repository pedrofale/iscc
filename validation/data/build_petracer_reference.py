"""Build the real PEtracer reference for the Tier-2 lineage+spatial comparison, with provenance.

Mirrors `build_dna_reference.py`: fetch a **real, public, citable** dataset, REDUCE it to the small
reusable summary the comparison consumes, and cache it under `validation/data/` as a compact `.npz`.
Raw sources are 100s of MB (per-tumour MERFISH + lineage trees); we never require/keep them — only
the reduction is cached. `validate_petracer.py --real` loads the cache and falls back to the Tier-1
self-contained benchmark when it is absent (offline / not downloaded), exactly like `validate_dna`.

PEtracer (Weissman lab, *Science* 2025) — mouse syngeneic tumours read jointly for lineage
(prime-editing barcodes → a per-tumour cell tree) and spatial expression (MERFISH panel + coords).

  Sources (Figshare BLOCKS bots — download manually, then pass the local paths):
    * processed MERFISH `AnnData` (h5ad) + per-tumour lineage trees:
        Figshare 10.6084/m9.figshare.28473866
    * scRNA (GEO GSE290975); code/format reference: https://github.com/jweissmanlab/PEtracer-2025
  Cite: Zhang, Yang, … Weissman (2025), *Science* 10.1126/science.adx3800.

The REDUCER (`reduce_petracer`, network-free) computes the SAME statistics as the iscc side so the
two are directly comparable — per **PER TUMOUR** (the model is a multi-site metastasis; multi-site =
RESEARCH_QUESTIONS R9, out of scope, so we never pool across sites):
    * per-gene **lineage** (patristic-tree) and **spatial** (coordinate) autocorrelation (Moran's I);
    * **clone-size** distribution (leaves per clade at a fixed tree cut);
    * **clone-territory** compactness (mean within-clone spatial spread ÷ tumour spread);
    * **tree** leaf count + depth distribution (balance).

Usage (auto-download is blocked; supply local files):
    python validation/data/build_petracer_reference.py --h5ad TUMOR.h5ad --newick TUMOR.nwk \
        --name tumor1 [--clone-key obs_column] [--max-cells 800]
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

from iscc.integrations.petracer import _moran_all  # noqa: E402  (shared Moran's I)

FIGSHARE_DOI = "10.6084/m9.figshare.28473866"
GEO_ACCESSION = "GSE290975"
PAPER_DOI = "10.1126/science.adx3800"

REFERENCE_PREFIX = "petracer_ref_"          # per-tumour cache: petracer_ref_<name>.npz


# --------------------------------------------------------------------------------------
# Newick → patristic leaf distances (no ete3/cassiopeia dependency)
# --------------------------------------------------------------------------------------
class _Tree:
    """Minimal rooted tree: parent map + branch lengths + root, from a Newick string."""

    def __init__(self, parent, blen, root, leaves):
        self.parent = parent          # node -> parent
        self.blen = blen              # node -> branch length to parent (root: 0)
        self.root = root
        self.leaves = leaves          # ordered leaf names
        self._path = {}               # node -> [(ancestor, dist_from_node), ...] to root
        self._depth = {}              # node -> #edges to root

    def _path_to_root(self, n):
        cached = self._path.get(n)
        if cached is not None:
            return cached
        path, cur, dist, d = [], n, 0.0, 0
        seen = set()
        while cur is not None and cur not in seen:
            path.append((cur, dist))
            seen.add(cur)
            dist += self.blen.get(cur, 0.0)
            nxt = self.parent.get(cur)
            if nxt is None:
                break
            cur = nxt
            d += 1
        self._path[n] = path
        self._depth[n] = len(path) - 1
        return path

    def depth(self, n):
        if n not in self._depth:
            self._path_to_root(n)
        return self._depth[n]

    def patristic(self, a, b):
        if a == b:
            return 0.0
        pa = {node: dist for node, dist in self._path_to_root(a)}
        for node, db in self._path_to_root(b):
            if node in pa:
                return pa[node] + db            # dist(a->lca) + dist(b->lca)
        # disjoint (shouldn't happen for a single rooted tree)
        return pa[self._path_to_root(a)[-1][0]] + self._path_to_root(b)[-1][1]

    def clade_at_depth(self, leaf, depth_cut):
        path = self._path_to_root(leaf)           # [(leaf,0), ..., (root, ...)]
        d = self.depth(leaf)
        if d <= depth_cut:
            return leaf
        return path[d - depth_cut][0]


def parse_newick(s):
    """Parse a Newick string into a :class:`_Tree` (branch lengths default to 1, internal nodes get
    synthetic ids). Handles quoted/unquoted labels and ``name:length`` edges."""
    s = s.strip()
    if s.endswith(";"):
        s = s[:-1]
    parent, blen, leaves = {}, {}, []
    counter = [0]

    def new_internal():
        counter[0] += 1
        return f"__node{counter[0]}"

    i = [0]

    def parse_clade():
        node_children = []
        if s[i[0]] == "(":
            i[0] += 1
            while True:
                child = parse_clade()
                node_children.append(child)
                if s[i[0]] == ",":
                    i[0] += 1
                    continue
                if s[i[0]] == ")":
                    i[0] += 1
                    break
        # read label
        j = i[0]
        while j < len(s) and s[j] not in ",():":
            j += 1
        label = s[i[0]:j].strip().strip("'\"")
        i[0] = j
        length = 1.0
        if i[0] < len(s) and s[i[0]] == ":":
            i[0] += 1
            k = i[0]
            while k < len(s) and s[k] not in ",()":
                k += 1
            try:
                length = float(s[i[0]:k])
            except ValueError:
                length = 1.0
            i[0] = k
        name = label if label else new_internal()
        if not node_children:
            leaves.append(name)
        for c in node_children:            # children already recorded their own branch lengths
            parent[c] = name
        blen[name] = length                # this node's branch length to its (future) parent
        return name

    root = parse_clade()
    blen[root] = 0.0
    return _Tree(parent, blen, root, leaves)


# --------------------------------------------------------------------------------------
# Reducer (network-free; the tested core)
# --------------------------------------------------------------------------------------
def _lineage_weights(tree, cells, lengthscale):
    idx = {c: i for i, c in enumerate(cells)}
    n = len(cells)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = tree.patristic(cells[i], cells[j])
    W = np.exp(-D / max(float(lengthscale), 1e-9))
    np.fill_diagonal(W, 0.0)
    return W, D


def reduce_petracer(adata, tree, name="tumor", clone_key=None, depth_cut=3,
                    lineage_lengthscale=None, spatial_lengthscale=None,
                    max_cells=800, seed=0):
    """Reduce one PEtracer tumour (MERFISH ``adata`` + lineage ``tree``) to the comparison summary.

    Uses the cells present in BOTH the AnnData and the tree. Autocorrelation is computed with the
    same Moran's I as the iscc side. Returns a plain dict of arrays/scalars (cacheable to .npz)."""
    import pandas as pd

    leaves = set(tree.leaves)
    obs_names = [str(c) for c in adata.obs_names]
    common = [c for c in obs_names if c in leaves]
    if len(common) < 10:
        raise ValueError(f"only {len(common)} cells shared between AnnData and tree — check ids")
    rng = np.random.default_rng(seed)
    if max_cells and len(common) > max_cells:
        common = sorted(rng.choice(common, max_cells, replace=False).tolist())

    sub = adata[common]
    X = np.asarray(sub.X.todense() if hasattr(sub.X, "todense") else sub.X, dtype=float)
    coords = np.asarray(sub.obsm["spatial"], dtype=float)
    coords = coords + rng.normal(0, 1e-6 + 0.01 * coords.std(), coords.shape)

    # lengthscale defaults: lineage = median patristic NN; spatial = median coordinate NN
    Wl, Dl = _lineage_weights(tree, common, lineage_lengthscale or 1.0)
    if lineage_lengthscale is None:
        nn = Dl + np.eye(len(common)) * Dl.max()
        L = float(np.median(nn.min(axis=1))) or 1.0
        Wl = np.exp(-Dl / L); np.fill_diagonal(Wl, 0.0)
    d2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)
    Ls = spatial_lengthscale or (float(np.median(np.sqrt(d2 + np.eye(len(common)) * d2.max()).min(axis=1))) or 1.0)
    Ws = np.exp(-d2 / (2.0 * Ls * Ls)); np.fill_diagonal(Ws, 0.0)

    I_lin = _moran_all(Wl, X)
    I_sp = _moran_all(Ws, X)

    # clones: an obs column if given, else a tree cut at depth_cut
    if clone_key is not None and clone_key in sub.obs:
        clones = sub.obs[clone_key].astype(str).values
    else:
        clones = np.array([tree.clade_at_depth(c, depth_cut) for c in common])
    clone_sizes = pd.Series(clones).value_counts().values.astype(int)

    # clone-territory compactness: within-clone spatial spread ÷ whole-tumour spread
    tumour_spread = float(np.sqrt(((coords - coords.mean(0)) ** 2).sum(1).mean()))
    spreads = []
    for cl in np.unique(clones):
        pts = coords[clones == cl]
        if len(pts) >= 3:
            spreads.append(np.sqrt(((pts - pts.mean(0)) ** 2).sum(1).mean()))
    territory_ratio = float(np.mean(spreads) / tumour_spread) if spreads and tumour_spread > 0 else float("nan")

    depths = np.array([tree.depth(c) for c in common], dtype=int)
    return dict(
        name=name, source="petracer", n_cells=len(common), n_genes=X.shape[1],
        I_lineage=I_lin, I_spatial=I_sp,
        clone_sizes=clone_sizes, n_clones=len(clone_sizes),
        territory_ratio=territory_ratio, tree_depths=depths,
        median_lineage_autocorr=float(np.median(I_lin)),
        median_spatial_autocorr=float(np.median(I_sp)),
    )


# --------------------------------------------------------------------------------------
# Fetch (network; graceful — Figshare blocks bots, so this returns None + a note)
# --------------------------------------------------------------------------------------
def fetch_petracer(name):
    """Attempt to fetch the processed PEtracer tumour; returns (None, note) — Figshare blocks bots."""
    return None, (f"PEtracer processed data is on Figshare ({FIGSHARE_DOI}), which blocks automated "
                  f"downloads. Download the per-tumour MERFISH h5ad + lineage tree manually and pass "
                  f"--h5ad/--newick. scRNA: GEO {GEO_ACCESSION}. Ref: {PAPER_DOI}.")


# --------------------------------------------------------------------------------------
# Cache I/O (same convention as build_dna_reference)
# --------------------------------------------------------------------------------------
def save_reference(path, ref):
    arrs = {k: np.asarray(v) for k, v in ref.items() if v is not None}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **arrs)


def load_reference(path):
    if not os.path.exists(path):
        return None
    z = np.load(path, allow_pickle=False)
    out = {}
    for k in z.files:
        v = z[k]
        out[k] = (v.item().decode() if isinstance(v.item(), bytes) else v.item()) if v.ndim == 0 else v
    z.close()
    return out


def reference_path(name):
    return os.path.join(HERE, f"{REFERENCE_PREFIX}{name}.npz")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5ad", help="local processed MERFISH AnnData (obsm['spatial'] + panel counts)")
    ap.add_argument("--newick", help="local per-tumour lineage tree (Newick; leaf ids = cell ids)")
    ap.add_argument("--name", default="tumor1", help="tumour label (cache = petracer_ref_<name>.npz)")
    ap.add_argument("--clone-key", default=None, help="obs column with clone/lineage labels")
    ap.add_argument("--depth-cut", type=int, default=3, help="tree depth for clade/clone cut")
    ap.add_argument("--max-cells", type=int, default=800)
    args = ap.parse_args()

    if not (args.h5ad and args.newick):
        _, note = fetch_petracer(args.name)
        print("no local files given.\n" + note)
        return

    import anndata as ad
    adata = ad.read_h5ad(args.h5ad)
    with open(args.newick) as fh:
        tree = parse_newick(fh.read())
    ref = reduce_petracer(adata, tree, name=args.name, clone_key=args.clone_key,
                          depth_cut=args.depth_cut, max_cells=args.max_cells)
    path = reference_path(args.name)
    save_reference(path, ref)
    print(f"reduced {args.name}: {ref['n_cells']} cells x {ref['n_genes']} genes, "
          f"{ref['n_clones']} clones, territory ratio {ref['territory_ratio']:.2f}")
    print(f"  median autocorr: lineage {ref['median_lineage_autocorr']:.3f}  "
          f"spatial {ref['median_spatial_autocorr']:.3f}")
    print("cache ->", path)


if __name__ == "__main__":
    main()
