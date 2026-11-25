"""Coverage CV of the simulated Tapestri panel at two copies, by true copy number and by copy number
called from the coverage as the real panel's was (S11 Fig F).

The real panel's copy number (``validation/data/build_dna_reference.reduce_tapestri``) is 2 x the
cell's normalised amplicon depth over its median across cells, so it is called from the same reads as
the coverage, and loci whose coverage strays are called off two copies and leave the CV at two copies.
Here the simulated coverage of each locus stands for its amplicon's reads, the same rule calls a copy
number, and the CV at two copies is taken over the loci called at two, beside the CV at true two
copies that S11 Fig F shows. Every locus is its own amplicon here, so the coverage and the call share
all their reads; on the real panel several variants share an amplicon, so this bounds the effect from
above.

Usage::

    ISCC_PAPER_OUT=... ISCC_PAPER_TIER=full PYTHONPATH=src python validation/paper/tools/panel_called_cv.py OUT.json

About 9 GB on a laptop (the showcase tumour of v5, 1,500 dissociated cells, 150 cancer cells).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import numpy as np  # noqa: E402

import fig3_biases as F3  # noqa: E402
import realistic_regime as RR  # noqa: E402

CN_CLIP = 8.0


def called_cn(coverage):
    """``reduce_tapestri``'s rule on a cells x loci coverage matrix."""
    reads = np.asarray(coverage, float)
    tot = reads.sum(1, keepdims=True)
    tot[tot == 0] = 1.0
    norm = reads / tot
    med = np.median(norm, axis=0, keepdims=True)
    med[med == 0] = np.nan
    return np.clip(np.nan_to_num(2.0 * norm / med, nan=2.0), 0.0, CN_CLIP)


def main(out):
    from iscc.data import estimate_dna, scDNA
    z = np.load(F3.DNA_REF["Tapestri"], allow_pickle=True)
    est = estimate_dna(z["coverage"], z["alt"], z["cn"], modality="sc", breadth=str(z["breadth"]),
                       depth_model=str(z["depth_model"]), het_mask=z["het_mask"])
    t = F3._load()
    diss = RR.resect(t).dissociate(max_cells=1500)
    gid = diss["cell_type"]["cell_id"].astype(str)
    cancer = [c for c, g in gid.items() if g in t.genotypes and t._is_cancer(g)][:150]
    tg = z["coverage"].shape[1] / t.n_genes
    del t
    cv = F3._f2()._cv_by_state
    real = cv(np.atleast_2d(z["coverage"]), np.atleast_2d(z["cn"])).get(2)
    rows = []
    for x in F3.KAPPA_X:
        s = scDNA(n_cells=len(cancer), breadth=est.breadth, depth_model=est.depth_model, seed=21, target_genes=tg,
                  **{**est.dna_kwargs(), "kappa": est.hypers.kappa * x}).run(diss, cell_subset=cancer)
        C = s.coverage.values.astype(float)
        call = called_cn(C)
        true2 = np.rint(s.true_cn.values.astype(float)) == 2
        both = np.where(true2 & (np.rint(call) == 2), 2.0, -1.0)      # true two copies, also called two
        rows.append(dict(kappa_x=float(x), cv_true_cn2=cv(C, s.true_cn.values).get(2), cv_called_cn2=cv(C, call).get(2),
                         cv_true_and_called_cn2=cv(C, both).get(2),
                         share_called_2=float((np.rint(call) == 2).mean()),
                         share_true_2=float((np.rint(s.true_cn.values.astype(float)) == 2).mean())))
        print(rows[-1], flush=True)
    res = dict(cv_real_cn2=real, kappa_fit=float(est.hypers.kappa), n_loci=int(C.shape[1]), n_cells=len(cancer),
               rows=rows, tumour="/".join(F3.TUMOUR))
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print("real", real)


if __name__ == "__main__":
    main(sys.argv[1])
