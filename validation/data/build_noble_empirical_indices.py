"""Build the empirical (n, D, J1) table for real tumours, for the M3a overlay.

Downloads the published real-tumour phylogenies from Noble et al. (2022) "Spatial structure
governs the mode of tumour evolution" (repo: github.com/robjohnnoble/ModesOfEvolution,
RealTumourTreesData/) and computes the three evolutionary-mode indices with **iscc's own**
`indices.py` functions, so the empirical points use exactly the same definitions as the
simulated ones in `validate_evolution_modes.py`.

Tree format: CSV with columns `Parent, Identity, Population` — each node is a clone (a
driver-mutation combination), `Population` is its cell count, and tree depth from the root is the
clone's number of driver mutations. Hence:
  * D  = inverse-Simpson over node Populations            (diversity over driver combinations)
  * n  = Population-weighted mean node depth              (mean drivers per cell)
  * J1 = Lemant tree balance over (parents, Population)   (iscc.inference.indices.tree_balance_j1)

Writes validation/data/noble_empirical_indices.csv.
Usage:  python validation/data/build_noble_empirical_indices.py
"""
import csv
import io
import os
import urllib.request

from iscc.inference.indices import inverse_simpson, tree_balance_j1

RAW = ("https://raw.githubusercontent.com/robjohnnoble/ModesOfEvolution/master/"
       "RealTumourTreesData/{dirn}/{f}.csv")
HERE = os.path.dirname(os.path.abspath(__file__))

# (directory, cancer type, [tumour ids]) — the published compilation.
MANIFEST = [
    ("TRACERx_Trees", "ccRCC (kidney)", ["K136", "K153", "K252", "K255", "K448"]),
    ("TRACERx_Trees", "NSCLC (lung)", ["CRUK0029", "CRUK0062", "CRUK0065", "CRUK0071", "CRUK0096"]),
    ("YatesEtAl_Trees", "breast (WGS)", ["PD9694", "PD9849", "PD9852"]),
    ("MinussiEtAl_Trees", "breast (scDNA)", ["TN1", "TN2", "TN3", "TN4", "TN5", "TN6", "TN7", "TN8"]),
    ("DuranteEtAl_Trees", "uveal melanoma",
     ["UMM059", "UMM061", "UMM062", "UMM063", "UMM064", "UMM065", "UMM066", "UMM069"]),
    ("ZhangEtAl_Trees", "mesothelioma", ["MED001", "MED012", "MED023", "MED024", "MED027", "MED034"]),
    ("AML_Trees", "AML",
     ["AML-02-001", "AML-05-001", "AML-16-001", "AML-33-001",
      "AML-35-001", "AML-55-001", "AML-73-001", "AML-77-001"]),
]


def _read_tree(dirn, tumour):
    url = RAW.format(dirn=dirn, f=tumour)
    text = urllib.request.urlopen(url).read().decode("utf-8-sig")
    parents, sizes = {}, {}
    for row in csv.DictReader(io.StringIO(text)):
        node, parent = row["Identity"], row["Parent"]
        pop = float(row["Population"])
        sizes[node] = pop
        if node != parent:                      # skip the root self-loop (0,0)
            parents[node] = parent
    return parents, sizes


def _depths(parents, sizes):
    """Depth (number of driver events from the root) for every node."""
    roots = [u for u in sizes if u not in parents]
    depth = {r: 0 for r in roots}
    # resolve each node's depth by walking up to a known ancestor
    for node in sizes:
        chain = []
        u = node
        while u not in depth:
            chain.append(u)
            u = parents[u]
        base = depth[u]
        for k, v in enumerate(reversed(chain), start=1):
            depth[v] = base + k
    return depth


def indices_from_tree(parents, sizes):
    pops = list(sizes.values())
    D = inverse_simpson(pops)
    depth = _depths(parents, sizes)
    total = sum(pops)
    n = sum(depth[u] * sizes[u] for u in sizes) / total if total > 0 else float("nan")
    J1 = tree_balance_j1(parents, sizes)
    n_clones = sum(1 for p in pops if p > 0)
    return n, D, J1, n_clones


def main():
    out = os.path.join(HERE, "noble_empirical_indices.csv")
    rows = []
    for dirn, ctype, tumours in MANIFEST:
        for t in tumours:
            try:
                parents, sizes = _read_tree(dirn, t)
                n, D, J1, k = indices_from_tree(parents, sizes)
                rows.append((t, ctype, n, D, J1, k))
                print(f"{t:12s} {ctype:16s} n={n:5.2f} D={D:6.2f} J1={J1:5.3f} clones={k}")
            except Exception as e:                # noqa: BLE001 - report and skip a bad tree
                print(f"  SKIP {t} ({ctype}): {e}")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tumour", "cancer_type", "n", "D", "J1", "n_clones"])
        w.writerows(rows)
    print(f"\n{len(rows)} tumours -> {out}")


if __name__ == "__main__":
    main()
