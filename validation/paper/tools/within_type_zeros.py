"""The scRNA zero fraction at gene means of 1 to 3 UMIs per cell, over all cells and within one cell
type, for the paragraph on Fig 3D: the breast tumour of Fig 3D (``fig2_realism.SCRNA_REF``, typed by
its authors' ``celltype_major``) against the showcase tumour over the human genes at the scRNA settings
fitted to it, on the cells of Fig 3D (``fig2_realism.compute_f``: 3,000 dissociated cells, seed 5).
PBMC3k, typed by marker genes, was the reference until 2026-11-25.

Per group, two statistics:

``median_zero``
    The median zero fraction over the genes whose mean is 1 to 3 UMIs per cell (the paper's number).
``curve_zero``
    The mean of Fig 3D's binned curve (``fig2_realism._count_stats``: median zero fraction in 20 bins of
    log10 gene mean) over its points between 1 and 3 UMIs. It reproduces the whole-population values of
    the v5 draft (0.43 against 0.30), but with one to three points in range it is noisy within a type.

Usage::

    ISCC_PAPER_OUT=... ISCC_PAPER_TIER=full PYTHONPATH=src python validation/paper/tools/within_type_zeros.py OUT.json

About 30 s and 9.5 GB on a laptop (the showcase tumour of v5).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402

import fig2_realism as F2  # noqa: E402
import realistic_regime as RR  # noqa: E402
import registry as REG  # noqa: E402

LO, HI = 1.0, 3.0
TUMOUR = ("R0", "seed1")
N_TYPES = 5                             # the most numerous types of each source


def zeros_at(X):
    X = sp.csr_matrix(X).astype(np.float64)
    m = np.asarray(X.mean(0)).ravel()
    z = 1.0 - np.asarray((X > 0).sum(0)).ravel() / X.shape[0]
    sel = (m >= LO) & (m <= HI)
    ok = m > 0
    lm = np.log10(m[ok])
    b = np.linspace(lm.min(), lm.max(), 20)
    idx = np.digitize(lm, b)
    keep = [i for i in range(1, 20) if (idx == i).sum() > 5]
    cm = np.array([lm[idx == i].mean() for i in keep])
    cz = np.array([np.median(z[ok][idx == i]) for i in keep])
    pts = (cm >= np.log10(LO)) & (cm <= np.log10(HI))
    return dict(n_cells=int(X.shape[0]), n_genes=int(sel.sum()),
                median_zero=float(np.median(z[sel])) if sel.any() else None,
                curve_zero=float(cz[pts].mean()) if pts.any() else None,
                curve_points=[[round(float(10 ** a), 2), round(float(c), 3)] for a, c in zip(cm[pts], cz[pts])])


def main(out):
    from iscc.data import scRNA
    from iscc.integrations import cell_subtypes
    real, est = F2.scrna_reference()
    types = real.obs["celltype_major"].astype(str).to_numpy()
    res = {"real": {"all": zeros_at(real.X), "counts": {k: int((types == k).sum()) for k in np.unique(types)}}}
    for k, _ in sorted(res["real"]["counts"].items(), key=lambda kv: -kv[1])[:N_TYPES]:
        res["real"][k] = zeros_at(real.X[types == k])
    del real
    t = REG.load(*TUMOUR)
    diss = RR.resect(t).dissociate(max_cells=3000)
    expr = F2.human_expression(t)
    del t
    cells = list(diss["cell_exp"].index)
    sim = scRNA(n_cells=len(cells), seed=5, **est.scrna_kwargs()).run(diss, cell_subset=cells, gene_lift=expr)
    gid = diss["cell_type"]["cell_id"].astype(str).reindex(sim.observed_counts.index)
    lab = np.asarray(cell_subtypes(gid.to_numpy()), dtype=str)
    X = sp.csr_matrix(sim.observed_counts.values)
    res["iscc"] = {"all": zeros_at(X), "counts": {k: int((lab == k).sum()) for k in np.unique(lab)}}
    for k, _ in sorted(res["iscc"]["counts"].items(), key=lambda kv: -kv[1])[:N_TYPES]:
        res["iscc"][k] = zeros_at(X[lab == k])
    res["settings"] = dict(gene_mean_umis=[LO, HI], reference=F2.SCRNA_REF, tumour="/".join(TUMOUR),
                           tumour_path=REG.task_path(*TUMOUR), n_dissociated=3000, scrna_seed=5,
                           scrna_kwargs={k: (v if isinstance(v, (int, float, str, bool)) else str(v))
                                         for k, v in est.scrna_kwargs().items()})
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    for src in ("real", "iscc"):
        for k, v in res[src].items():
            if k != "counts":
                print(f"{src:5s} {k:12s} cells {v['n_cells']:5d}  median {v['median_zero']:.3f}  curve "
                      f"{v['curve_zero'] if v['curve_zero'] is None else round(v['curve_zero'], 3)}")


if __name__ == "__main__":
    main(sys.argv[1])
