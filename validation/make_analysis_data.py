"""Generate the analysis-ready datasets the data-analysis notebooks load.

WHY THIS EXISTS. The tool-benchmark notebooks used to grow their own tumour with `iscc` and then run
the tool in the same notebook. That forces the notebook's kernel to be Python, which in turn forces an
R tool (clonealign, Numbat, RCTD, TreeMHN) to be invoked through a subprocess rather than written as
R. Separating GENERATION from ANALYSIS removes that constraint: this script does the `iscc` part once
and writes plain tables, so an analysis notebook only ever *loads* — and a notebook that only loads
can be an R notebook, with an R kernel and ordinary R code.

It also makes the benchmarks honest in a second way: the analysis side cannot accidentally peek at
anything the tool would not have. What lands on disk is exactly the tool's input plus a separate
ground-truth file used only for scoring.

REGIME. Datasets come from the realistic breach-gated ductal field (`realistic_regime.py`), not the
old toy rig — grid 96 / 5 glands / 6,000 genes at `scale="mid"`, versus the toy's grid 20, one ring
and 600 genes.

Output (default `analysis_data/`, gitignored — the matrices are tens of MB):

    analysis_data/
      manifest.json               what was generated, with params, seeds and versions
      clonealign/
        Y.csv.gz                  cells x genes   scRNA counts        (tool input)
        L.csv.gz                  genes x clones  copy number         (tool input)
        truth.csv                 cell -> true clone                  (SCORING ONLY)
        meta.json                 shapes, clone sizes, the CN consensus

CSV(.gz) on purpose: `read.csv` in R reads `.gz` directly, so the same file serves a Python and an R
notebook without either needing the other's stack.

Usage:  python validation/make_analysis_data.py [--out analysis_data] [--seed 3] [--scale mid]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)


def _clonealign(out_dir, seed, scale, n_clones):
    """scRNA counts + per-gene clone copy number + the true clone of each cell."""
    import integration_common as C

    t = C.grow_tumor(seed=seed, regime="realistic", scale=scale)
    inp = C.build_clonealign_inputs(t, n_clones=n_clones, seed=0)
    Y, L, labels = inp["Y"], inp["L"], np.asarray(inp["labels"])

    os.makedirs(out_dir, exist_ok=True)
    Y.to_csv(os.path.join(out_dir, "Y.csv.gz"))
    L.to_csv(os.path.join(out_dir, "L.csv.gz"))
    # Ground truth lives in its OWN file: the tool reads Y and L, never this.
    pd.DataFrame({"cell": Y.index, "true_clone": labels}).to_csv(
        os.path.join(out_dir, "truth.csv"), index=False)

    sizes = np.bincount(labels, minlength=L.shape[1]).tolist()
    meta = dict(
        n_cells=int(Y.shape[0]), n_genes=int(Y.shape[1]), n_clones=int(L.shape[1]),
        cn_informative_genes=int((L.nunique(axis=1) > 1).sum()),
        clone_sizes=sizes,
        majority_baseline=float(max(sizes) / sum(sizes)),
        chance_baseline=float(1.0 / L.shape[1]),
        # The consensus is what makes the difficulty legible: on the realistic WGD+ field the clones
        # differ in only one or two of the twelve segments, so a pure dosage model has little to go on.
        cn_consensus=inp["consensus"].astype(int).tolist(),
    )
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    return meta


DATASETS = {"clonealign": _clonealign}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "analysis_data"))
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--scale", default="mid", choices=["small", "mid", "cm"])
    ap.add_argument("--clones", type=int, default=4)
    ap.add_argument("--only", default="", help="comma-separated dataset names")
    args = ap.parse_args()

    wanted = [d.strip() for d in args.only.split(",") if d.strip()] or list(DATASETS)
    os.makedirs(args.out, exist_ok=True)
    manifest = {}
    for name in wanted:
        t0 = time.time()
        print(f"generating {name} (regime=realistic, scale={args.scale}, seed={args.seed}) ...",
              flush=True)
        meta = DATASETS[name](os.path.join(args.out, name), args.seed, args.scale, args.clones)
        meta.update(seed=args.seed, scale=args.scale, regime="realistic",
                    seconds=round(time.time() - t0, 1))
        manifest[name] = meta
        print(f"  {name}: {meta['n_cells']:,} cells x {meta['n_genes']:,} genes, "
              f"{meta['n_clones']} clones {meta['clone_sizes']} in {meta['seconds']}s", flush=True)

    try:
        import iscc
        version = getattr(iscc, "__version__", "unknown")
    except Exception:
        version = "unknown"
    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump({"iscc_version": version, "datasets": manifest}, fh, indent=2)
    print(f"\nmanifest -> {os.path.join(args.out, 'manifest.json')}")


if __name__ == "__main__":
    main()
