"""Synthetic inputs with a known answer for the Fig 5 tool images, and the checks on their outputs.

Used by smoke_fig5_tools.sh. Everything but ``h5ad`` is standard library only, so it runs on the
host's python3; ``h5ad`` needs numpy, scipy and anndata (run it in the integration image).

    h5ad   <out.h5ad> <truth.h5ad>   3 patients x 2 batches, a strong per-gene batch shift, and the
                                     same cells without it
    events <out.csv>                 500 patients, 5 events from an MHN with one planted dependency
    trees  <out.csv>                 100 patients' trees, 4 events, one planted order (E0 -> E1)
    check-integration <dir>  check-mhn <prefix>  check-treemhn <prefix>
"""
import csv
import json
import math
import random
import sys


# ------------------------------------------------------------------ scRNA with a batch shift
def make_h5ad(out, truth_out, seed=1):
    import anndata as ad
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp

    rng = np.random.default_rng(seed)
    n_genes, per_patient = 1000, 400
    types = ["cancer", "tcell", "fibroblast"]
    base = rng.normal(np.log(0.3), 1.0, n_genes)
    prog = {t: rng.choice(n_genes, 50, replace=False) for t in types}
    rows = []
    for p in range(3):
        pgenes = rng.choice(n_genes, 30, replace=False)          # the patient's own tumour program
        cgenes = rng.choice(n_genes, 20, replace=False)          # clone c1's program
        for i in range(per_patient):
            t = "cancer" if i < per_patient // 2 else types[1 + i % 2]
            clone = f"P{p}:c{i % 2}" if t == "cancer" else "none"
            b = f"b{rng.integers(2)}"                             # balanced: patients in both batches
            mu = base.copy()
            mu[prog[t]] += 2.0
            if t == "cancer":
                mu[pgenes] += 1.5
                if clone.endswith("c1"):
                    mu[cgenes] += 1.0
            rows.append((mu, dict(batch=b, patient=f"P{p}", cell_type=t, is_cancer=t == "cancer",
                                  clone=clone)))
    mu = np.array([r[0] for r in rows])
    obs = pd.DataFrame([r[1] for r in rows], index=[f"cell{i}" for i in range(len(rows))])
    depth = rng.lognormal(0.0, 0.25, len(obs))[:, None]
    # batch effect: a per-gene log-normal factor and a depth factor per batch (sigma 0.6 / 0.3, the
    # strong level of the Fig 5 design)
    shift = {b: (rng.normal(0, 0.6, n_genes), rng.normal(0, 0.3)) for b in ("b0", "b1")}
    batch_mu = mu + np.array([shift[b][0] + shift[b][1] for b in obs["batch"]])
    for lam, path in ((batch_mu, out), (mu, truth_out)):
        rate = np.exp(lam) * depth * rng.gamma(5.0, 1 / 5.0, lam.shape)   # NB, dispersion 0.2
        X = sp.csr_matrix(rng.poisson(rate).astype(np.float32))
        a = ad.AnnData(X=X, obs=obs.copy(), var=pd.DataFrame(index=[f"g{j}" for j in range(n_genes)]))
        a.layers["counts"] = X.copy()
        a.write_h5ad(path)
    print(f"[smoke] {len(obs)} cells, {n_genes} genes, 3 patients x 2 batches -> {out}, {truth_out}")


# ------------------------------------------------------------------ MHN sampling (stdlib)
def sample_mhn(theta, rng):
    """One observation of an MHN: events accrue at rate exp(theta_ii + sum_{j present} theta_ij)
    until the sampling event (rate 1) stops the process. Returns the events in order."""
    d = len(theta)
    path = []
    while True:
        rates = [0.0 if i in path else math.exp(theta[i][i] + sum(theta[i][j] for j in path))
                 for i in range(d)]
        total = sum(rates) + 1.0
        u = rng.random() * total
        if u < 1.0:
            return path
        u -= 1.0
        for i, r in enumerate(rates):
            if u < r:
                path.append(i)
                break
            u -= r


def planted(d, base, pairs):
    theta = [[0.0] * d for _ in range(d)]
    for i in range(d):
        theta[i][i] = base[i]
    for (i, j), v in pairs.items():
        theta[i][j] = v
    return theta


MHN_TRUE = dict(d=5, base=[-1.0, -2.5, -1.0, -1.2, -1.5], pairs={(1, 0): 2.5})   # E0 raises E1's rate
TREE_TRUE = dict(d=4, base=[-0.5, -4.0, -1.0, -1.0], pairs={(1, 0): 4.0})       # E1 only after E0


def make_events(out, n=500, seed=2):
    rng = random.Random(seed)
    theta = planted(MHN_TRUE["d"], MHN_TRUE["base"], MHN_TRUE["pairs"])
    names = [f"E{i}" for i in range(MHN_TRUE["d"])]
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + names)
        for p in range(n):
            path = set(sample_mhn(theta, rng))
            w.writerow([f"P{p}"] + [int(i in path) for i in range(len(names))])
    print(f"[smoke] {n} patients x {len(names)} events, planted theta[E1, E0] = 2.5 -> {out}")


def make_trees(out, n=100, lineages=2, seed=3):
    """Each patient's tree is the trie of a few MHN paths, as progression.to_mutation_tree builds it
    from clones: root Node_ID 1, Mutation_ID 0, its own parent; event k is Mutation_ID k + 1."""
    rng = random.Random(seed)
    theta = planted(TREE_TRUE["d"], TREE_TRUE["base"], TREE_TRUE["pairs"])
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Patient_ID", "Tree_ID", "Node_ID", "Mutation_ID", "Parent_ID", "n_cells"])
        for p in range(1, n + 1):
            nodes = [(1, 0, 1)]
            child = {}
            for _ in range(lineages):
                parent = 1
                for ev in sample_mhn(theta, rng):
                    key = (parent, ev)
                    if key not in child:
                        child[key] = len(nodes) + 1
                        nodes.append((child[key], ev + 1, parent))
                    parent = child[key]
            for node, mut, par in nodes:
                w.writerow([p, p, node, mut, par, 0])
    print(f"[smoke] {n} patients' trees, {TREE_TRUE['d']} events, planted E0 -> E1 -> {out}")


# ------------------------------------------------------------------ checks
def read_theta(path):
    with open(path) as f:
        r = list(csv.reader(f))
    names = r[0][1:]
    return names, [[float(x) for x in row[1:]] for row in r[1:]]


def pair_ranking(theta):
    """Pairs ranked by the larger |theta| of their two directions (1 = strongest)."""
    d = len(theta)
    pairs = [((i, j), max(abs(theta[i][j]), abs(theta[j][i]))) for i in range(d) for j in range(i + 1, d)]
    return sorted(pairs, key=lambda x: -x[1])


def check_mhn(prefix):
    names, th = read_theta(prefix + ".theta.csv")
    rec = json.load(open(prefix + ".json"))
    rank = [p for p, _ in pair_ranking(th)]
    r = rank.index((0, 1)) + 1
    ok = r == 1 and th[1][0] > 0
    print(f"[smoke] MHN: lambda {rec['lam']:.4g} ({rec['lam_rule']}, penalty {rec['penalty']}); planted "
          f"pair E0-E1 ranks {r} of {len(rank)}; theta[E1,E0] = {th[1][0]:.2f}, theta[E0,E1] = "
          f"{th[0][1]:.2f}; next pair max |theta| {pair_ranking(th)[1][1]:.2f}; "
          f"fit {rec['timings']['total_s']} s -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_treemhn(prefix):
    names, th = read_theta(prefix + ".theta.csv")
    rec = json.load(open(prefix + ".json"))
    rank = [p for p, _ in pair_ranking(th)]
    r = rank.index((0, 1)) + 1
    ok = r == 1 and th[1][0] > 0 and th[1][0] > th[0][1]
    print(f"[smoke] TreeMHN: {rec['n_patients']} patients ({rec['n_dropped_root_only']} root-only "
          f"dropped), {rec['inference']}; planted pair E0-E1 ranks {r} of {len(rank)}; theta[E1,E0] = "
          f"{th[1][0]:.2f} vs theta[E0,E1] = {th[0][1]:.2f}; fit {rec['timings']['total_s']} s "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


def check_integration(d):
    m = {k: json.load(open(f"{d}/{k}.json")) for k in ("none", "harmony", "scvi", "truth")}
    for k, v in m.items():
        x = v["metrics"]
        print(f"[smoke] {k:8s} iLISI {x['ilisi_batch']:.3f} (non-cancer {x['ilisi_batch_noncancer']:.3f})"
              f"  cLISI {x['clisi_celltype']:.3f}  ASW type {x['silhouette_celltype']:.3f}"
              f"  ASW batch {x['silhouette_batch'] if x['silhouette_batch'] is None else round(x['silhouette_batch'], 3)}"
              f"  patient purity (cancer) {x['patient_purity_cancer']:.3f}"
              f"  clone purity {x['clone_purity_cancer']:.3f}  [{x['metrics_backend']}]"
              f"  {v['timings']['total_s']} s" + (f", {v['settings']['scvi']['epochs_trained']} epochs"
                                                  if k == "scvi" else ""))
    il = {k: v["metrics"]["ilisi_batch"] for k, v in m.items()}
    ok = il["harmony"] > il["none"] and il["scvi"] > il["none"] and il["truth"] > il["none"]
    print(f"[smoke] iLISI rises after integration (harmony and scvi above none): {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    cmd, rest = sys.argv[1], sys.argv[2:]
    if cmd == "h5ad":
        make_h5ad(*rest)
    elif cmd == "events":
        make_events(*rest)
    elif cmd == "trees":
        make_trees(*rest)
    else:
        ok = {"check-integration": check_integration, "check-mhn": check_mhn,
              "check-treemhn": check_treemhn}[cmd](*rest)
        sys.exit(0 if ok else 1)
