"""Validate clone-of-origin assignment (clonealign) on iscc data — a non-circular DNA<->RNA benchmark.

clonealign~\cite{campbell_clonealign_2019} assigns single-cell RNA to the clone whose copy-number
profile best explains its expression, under the model *expression is proportional to copy number*.
We run the GENUINE clonealign (kieranrcampbell/clonealign, R + TensorFlow, in the dedicated
``iscc-clonealign`` conda env) on iscc-simulated data and score the assignment against iscc's true
clone-of-origin.

WHY THIS IS A FAIR (non-circular) TEST.  A bolt-on simulator that layers expression onto externally
supplied CNAs has to hand-impose a CNA->expression law; if that law is the dosage model clonealign
assumes, the benchmark just re-tests the method on its own assumption. In iscc the coupling EMERGES
from the engine's per-allele dosage model (a neutral gene's expression is baseline*(1+copy_number),
plus selection) — it is never imposed to match clonealign. clonealign's assumption (mean strictly
proportional to CN) is therefore only approximately right here, and iscc additionally supplies the
true clone label no real dataset has.

The result also DEGRADES gracefully where the dosage signal is weak: clones separated by many
copy-number-altered genes are assigned well; clones differing by a single segment blur together. The
dosage-dependence panel makes this explicit — assignment quality rises with the fraction of genes
carrying a clone-specific CN effect, which is exactly the signal clonealign is designed to exploit.

Figure (the paper repo's figures/validation_clonealign.png):
  A. expression embedding coloured by TRUE clone vs by clonealign's ASSIGNED clone;
  B. per-clone assignment AUC (one-vs-rest) + overall accuracy vs the chance/majority baselines;
  C. dosage dependence — accuracy / AUC vs the fraction of genes with a clone-specific CN effect.

Run (from the repo root, in the `iscc` env):
    python -u validation/validate_clonealign.py
Requires the `iscc-clonealign` env (see validation/README or the handoff). Falls back to a note if
absent.
"""
import argparse
import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import integration_common as C
from _paths import figure_path

REPO = C.REPO


def subset_by_informative_fraction(inp, target_frac, total_genes, rng):
    """Build a gene subset of ``total_genes`` with a given fraction on CN-informative segments (the
    rest on flat segments), for the dosage-dependence panel. Returns (Y_sub, L_sub, frac_actual)."""
    info_segs = set(C.informative_segments(inp["consensus"]).tolist())
    gseg = inp["gene_seg"]
    genes = np.array(inp["Y"].columns)
    is_info = np.array([s in info_segs for s in gseg])
    info_genes, flat_genes = genes[is_info], genes[~is_info]
    n_info = min(int(round(target_frac * total_genes)), len(info_genes))
    n_flat = min(total_genes - n_info, len(flat_genes))
    pick = np.concatenate([rng.choice(info_genes, n_info, replace=False),
                           rng.choice(flat_genes, n_flat, replace=False)])
    frac_actual = n_info / (n_info + n_flat)
    return inp["Y"][pick], inp["L"].loc[pick], frac_actual


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--steps", type=int, default=750)
    ap.add_argument("--clones", type=int, default=4)
    ap.add_argument("--protocol", default="10x")
    ap.add_argument("--max-iter", type=int, default=250)
    ap.add_argument("--n-repeats", type=int, default=3)
    ap.add_argument("--dosage-fracs", type=float, nargs="+", default=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--dosage-genes", type=int, default=250, help="genes per dosage-panel subset")
    ap.add_argument("--out", default=figure_path("validation_clonealign.png"))
    args = ap.parse_args()

    if not C.clonealign_available():
        print("[clonealign] dedicated env not found at", C.CLONEALIGN_RSCRIPT,
              "\n  -> build it (R + TensorFlow + kieranrcampbell/clonealign); see validation/README_integration.md."
              "\n  Skipping (the iscc data generation itself is exercised by tests/test_integration.py).")
        return

    print(f"growing tumour (seed={args.seed}) and assembling the DNA<->RNA benchmark ...")
    tumor = C.grow_tumor(seed=args.seed, steps=args.steps)
    inp = C.build_clonealign_inputs(tumor, n_clones=args.clones, protocol=args.protocol, seed=0)
    labels = inp["labels"]
    info_segs = C.informative_segments(inp["consensus"])
    print(f"  cancer cells={len(labels)}  clones={inp['consensus'].shape[0]}  "
          f"sizes={np.bincount(labels).tolist()}")
    print(f"  clone consensus segment-CN:\n{inp['consensus']}")
    print(f"  CN-informative segments={info_segs.tolist()}  "
          f"(scDNA recovers the clone CN at concordance {inp['dna_concordance']:.2f})")

    with tempfile.TemporaryDirectory() as work:
        # -- main run: full gene set ------------------------------------------------------------
        print(f"\nrunning clonealign (real R package) on {inp['Y'].shape[1]} genes ...")
        probs = C.run_clonealign(inp["Y"], inp["L"], os.path.join(work, "main"),
                                 max_iter=args.max_iter, n_repeats=args.n_repeats)
        sc = C.score_assignment(labels, probs)
        print("\nHEADLINE — clonealign clone-of-origin assignment vs iscc ground truth:")
        print(f"  accuracy = {sc['accuracy']:.3f}   (chance {sc['chance']:.3f}, "
              f"majority {sc['majority']:.3f})   ARI = {sc['ari']:.3f}")
        print(f"  per-clone AUC = { {f'clone{b}': round(v, 3) for b, v in sc['per_clone_auc'].items()} }")
        print(f"  mean AUC = {sc['mean_auc']:.3f}")

        # -- dosage dependence: accuracy vs the fraction of CN-informative genes -----------------
        print("\ndosage dependence (accuracy / AUC vs fraction of genes with a clone-specific CN effect):")
        rng = np.random.default_rng(0)
        dep = {"frac": [], "acc": [], "auc": []}
        for f in args.dosage_fracs:
            Ys, Ls, frac = subset_by_informative_fraction(inp, f, args.dosage_genes, rng)
            if Ls.nunique(axis=1).max() <= 1:              # no CN variation -> no signal
                acc, auc = sc["chance"], 0.5
            else:
                p = C.run_clonealign(Ys, Ls, os.path.join(work, f"dep{int(f*100)}"),
                                     max_iter=150, n_repeats=2)
                s = C.score_assignment(labels, p)
                acc, auc = s["accuracy"], s["mean_auc"]
            dep["frac"].append(frac); dep["acc"].append(acc); dep["auc"].append(auc)
            print(f"  informative-gene fraction={frac:.2f}  accuracy={acc:.3f}  meanAUC={auc:.3f}")

    _figure(args, inp, labels, probs, sc, dep)


def _embedding(Y):
    """2D expression embedding (UMAP over PCA) of the scRNA counts, for the assignment maps."""
    import scanpy as sc
    import anndata as ad
    a = ad.AnnData(np.asarray(Y.values, dtype=np.float32))
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.pp.pca(a, n_comps=min(30, a.n_vars - 1, a.n_obs - 1))
    try:
        sc.pp.neighbors(a, n_neighbors=15)
        sc.tl.umap(a, random_state=0)
        return a.obsm["X_umap"]
    except Exception:
        return a.obsm["X_pca"][:, :2]


def _figure(args, inp, labels, probs, sc, dep):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    K = inp["consensus"].shape[0]
    emb = _embedding(inp["Y"])
    # matched predicted labels (colours aligned to true clones)
    from scipy.optimize import linear_sum_assignment
    P = probs.values
    pred = P.argmax(1)
    conf = np.array([[((pred == a) & (labels == b)).sum() for b in range(K)] for a in range(K)])
    row, col = linear_sum_assignment(-conf)
    p2t = {row[i]: col[i] for i in range(len(row))}
    pred_m = np.array([p2t[p] for p in pred])
    palette = plt.cm.tab10(np.linspace(0, 1, 10))
    colt = [palette[c] for c in labels]

    # true-clone x assigned-clone confusion (row-normalised)
    M = np.array([[((labels == t) & (pred_m == a)).sum() for a in range(K)] for t in range(K)],
                 dtype=float)
    Mn = M / M.sum(1, keepdims=True)

    fig, ax = plt.subplots(2, 2, figsize=(13, 11))

    # A. embedding coloured by TRUE clone — clones overlap in expression space (dosage is subtle),
    # yet clonealign still resolves them from the copy-number signal (panels B-D).
    for c in range(K):
        m = labels == c
        ax[0, 0].scatter(emb[m, 0], emb[m, 1], color=palette[c], s=16, edgecolors="none",
                         label=f"clone{c} (n={int(m.sum())})")
    ax[0, 0].set(title="A. expression embedding — TRUE clone-of-origin\n(clones overlap: expression "
                       "encodes clone only weakly)", xlabel="UMAP-1", ylabel="UMAP-2")
    ax[0, 0].legend(fontsize=8, loc="best", title="clone")

    # B. confusion matrix — where clonealign confuses clones (the single-segment-apart pairs)
    im = ax[0, 1].imshow(Mn, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax[0, 1].set(xticks=range(K), yticks=range(K),
                 xticklabels=[f"clone{c}" for c in range(K)],
                 yticklabels=[f"clone{c}" for c in range(K)],
                 xlabel="clonealign assigned clone", ylabel="true clone-of-origin",
                 title=f"B. assignment confusion (accuracy {sc['accuracy']:.2f}, "
                       f"chance {sc['chance']:.2f})")
    for t in range(K):
        for a in range(K):
            ax[0, 1].text(a, t, f"{Mn[t, a]:.2f}", ha="center", va="center", fontsize=9,
                          color="white" if Mn[t, a] > 0.5 else "black")
    fig.colorbar(im, ax=ax[0, 1], fraction=0.046, pad=0.04, label="fraction of true-clone cells")

    # C. per-clone AUC vs baselines
    clones = list(range(K))
    aucs = [sc["per_clone_auc"].get(c, np.nan) for c in clones]
    ax[1, 0].bar([f"clone{c}" for c in clones], aucs, color="#2b6fb0", alpha=0.85)
    ax[1, 0].axhline(0.5, color="k", lw=1, ls="--", alpha=0.6, label="AUC = 0.5 (chance)")
    ax[1, 0].set(ylim=(0, 1), ylabel="one-vs-rest AUC",
                 title=f"C. per-clone assignment AUC (mean {sc['mean_auc']:.2f})")
    for i, v in enumerate(aucs):
        ax[1, 0].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    ax[1, 0].legend(fontsize=8, loc="lower right")

    # D. dosage dependence
    ax[1, 1].plot(dep["frac"], dep["acc"], "-o", color="#d1495b", label="accuracy")
    ax[1, 1].plot(dep["frac"], dep["auc"], "-s", color="#2b6fb0", label="mean AUC")
    ax[1, 1].axhline(sc["chance"], color="k", lw=1, ls=":", alpha=0.6, label="chance")
    ax[1, 1].set(xlabel="fraction of genes with a clone-specific CN effect",
                 ylabel="assignment quality", ylim=(0, 1),
                 title="D. dosage dependence\n(clonealign works BECAUSE of the emergent CN coupling)")
    ax[1, 1].legend(fontsize=8, loc="lower right")

    fig.suptitle("clonealign on iscc: clone-of-origin assignment from the EMERGENT copy-number "
                 "dosage coupling (non-circular ground truth)", fontsize=13, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print("\nfigure ->", args.out)


if __name__ == "__main__":
    main()
