"""Fig 4D on a real tumour: the lineage-space confound in PEtracer's 4T1 tumour.

Koblan et al. 2025 (Science, doi:10.1126/science.adx3800) traced the lineage of a 4T1 tumour in mouse lung
and imaged it by MERFISH; pycea packages mouse 3 tumour 1 as ``koblan25_tumor.h5td`` (Zenodo record
23195576). Of its 145,954 cells, 10,061 malignant cells carry a lineage on one tree, in sections M3-1, M3-2
and M3-4.

Per section and per random subsample of 600 traced cells (as Fig 4D's 600 cancer cells, ten subsamples),
for each field: Moran's I under the lineage weights exp(-D / 2), D the number of edges between two cells on
the tree (``fig4_lineage``'s form), and under Gaussian spatial weights of width 100 um within the section
(two demes of about 50 um), with raw and with row-standardised weights (``fig4_lineage._moran_rs``), and the
permutation test of the lineage autocorrelation alone (the field shuffled over the same cells, 999 times,
one-sided). Fields:

* environmental by construction, a position that no cell inherits: the distance of the cell to the tumour
  boundary and to the lung boundary;
* heritable by construction, the positive control: Brownian motion along the tree (unit variance per edge,
  one draw per subsample).

    python validation/paper/fig4_petracer.py reduce koblan25_tumor.h5td petracer.npz   # needs treedata
    python validation/paper/fig4_petracer.py analyse petracer.npz $FIG4_DEV/petracer/summary.json
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

SECTIONS = ("M3-1", "M3-2", "M3-4")
FIELDS = ("tumor_boundary_dist", "lung_boundary_dist")
N_CELLS, N_SUB, N_PERM, LIN_SCALE, SP_SCALE = 600, 10, 999, 2.0, 100.0


def reduce(h5td, out):
    """The traced cells and the tree as plain arrays (the only step that needs treedata)."""
    import networkx as nx
    import treedata as td
    t = td.read_h5td(h5td)
    tree = t.obst["tree"]
    nodes = list(nx.topological_sort(tree))
    idx = {n: i for i, n in enumerate(nodes)}
    parent = np.full(len(nodes), -1, dtype=np.int64)
    for u, v in tree.edges:
        parent[idx[v]] = idx[u]
    obs = t.obs
    traced = (obs["tree"].astype(str) == "tree").to_numpy()
    cells = obs.index.to_numpy()[traced]
    np.savez_compressed(out, parent=parent, leaf=np.array([idx[c] for c in cells], dtype=np.int64),
                        section=obs["sample"].astype(str).to_numpy()[traced],
                        xy=np.asarray(t.obsm["spatial"], float)[traced],
                        **{f: obs[f].to_numpy(float)[traced] for f in FIELDS})


def moran(W, z):
    z = np.asarray(z, float) - np.mean(z)
    den = float(z @ z)
    return 0.0 if den <= 0 or W.sum() <= 0 else float(len(z) / W.sum() * (z @ (W @ z)) / den)


def moran_rs(W, z):
    rs = W.sum(axis=1, keepdims=True)
    Wr = np.divide(W, rs, out=np.zeros_like(W), where=rs > 0)
    z = np.asarray(z, float) - np.mean(z)
    den = float(z @ z)
    return 0.0 if den <= 0 else float((z @ (Wr @ z)) / den)


def analyse(npz, out):
    z = np.load(npz, allow_pickle=True)
    parent, leaf, section, xy = z["parent"], z["leaf"], z["section"], z["xy"]
    depth = np.zeros(len(parent), dtype=np.int64)
    for i in range(len(parent)):                 # topological order: a parent precedes its children
        if parent[i] >= 0:
            depth[i] = depth[parent[i]] + 1
    anc = {}

    def ancestors(n):
        if n not in anc:
            out_, x = {}, n
            while x >= 0:
                out_[x] = depth[x]
                x = parent[x]
            anc[n] = out_
        return anc[n]

    def tree_dist(nodes):
        sets = [ancestors(n) for n in nodes]
        D = np.zeros((len(nodes), len(nodes)))
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                common = sets[i].keys() & sets[j].keys()
                D[i, j] = D[j, i] = depth[nodes[i]] + depth[nodes[j]] - 2 * max(sets[i][a] for a in common)
        return D

    res = dict(source="Koblan et al. 2025, PEtracer 4T1 mouse 3 tumour 1 (pycea koblan25, Zenodo 23195576)",
               settings=dict(n_cells=N_CELLS, n_subsamples=N_SUB, n_perm=N_PERM, lineage_width_edges=LIN_SCALE,
                             spatial_width_um=SP_SCALE), sections={})
    for sec in SECTIONS:
        idx = np.flatnonzero(section == sec)
        recs = []
        for s in range(N_SUB):
            rng = np.random.default_rng([s, 0xE4])
            pick = np.sort(rng.choice(idx, size=min(N_CELLS, len(idx)), replace=False))
            nodes = leaf[pick]
            Wl = np.exp(-tree_dist(nodes) / LIN_SCALE)
            np.fill_diagonal(Wl, 0.0)
            p = xy[pick]
            Ws = np.exp(-((p[:, None, :] - p[None, :, :]) ** 2).sum(-1) / (2 * SP_SCALE ** 2))
            np.fill_diagonal(Ws, 0.0)
            steps = rng.normal(size=len(parent))
            bm = np.zeros(len(parent))
            for i in range(len(parent)):
                if parent[i] >= 0:
                    bm[i] = bm[parent[i]] + steps[i]
            fields = {f: z[f][pick] for f in FIELDS}
            fields["heritable_control"] = bm[nodes]
            rec = {}
            for name, v in fields.items():
                obs_rs = moran_rs(Wl, v)
                null = np.array([moran_rs(Wl, rng.permutation(v)) for _ in range(N_PERM)])
                rec[name] = dict(lin=moran(Wl, v), sp=moran(Ws, v), lin_rs=obs_rs, sp_rs=moran_rs(Ws, v),
                                 perm_p_rs=float((1 + (null >= obs_rs).sum()) / (N_PERM + 1)))
            recs.append(rec)
            print(sec, s, json.dumps({k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in rec.items()}),
                  flush=True)
        res["sections"][sec] = dict(
            n_traced=int(len(idx)),
            fields={name: dict(
                lin_rs=float(np.median([r[name]["lin_rs"] for r in recs])),
                sp_rs=float(np.median([r[name]["sp_rs"] for r in recs])),
                lin=float(np.median([r[name]["lin"] for r in recs])),
                sp=float(np.median([r[name]["sp"] for r in recs])),
                share_called_heritable=float(np.mean([r[name]["perm_p_rs"] < 0.05 for r in recs])),
                share_lineage_above_spatial=float(np.mean([r[name]["lin_rs"] > r[name]["sp_rs"] for r in recs])))
                for name in recs[0]})
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("reduce")
    r.add_argument("h5td")
    r.add_argument("out")
    a = sub.add_parser("analyse")
    a.add_argument("npz")
    a.add_argument("out")
    args = ap.parse_args()
    if args.cmd == "reduce":
        reduce(args.h5td, args.out)
    else:
        res = analyse(args.npz, args.out)
        for sec, r_ in res["sections"].items():
            print("==", sec, r_["n_traced"])
            for k, v in r_["fields"].items():
                print(f"   {k:22s} lineage {v['lin_rs']:.3f}  spatial {v['sp_rs']:.3f}  called heritable "
                      f"{v['share_called_heritable']:.2f}  lineage above spatial {v['share_lineage_above_spatial']:.2f}")


if __name__ == "__main__":
    main()
