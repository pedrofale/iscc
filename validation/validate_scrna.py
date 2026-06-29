"""Validate iscc's simulated scRNA-seq against a real 10x reference (PBMC3k), FITTED.

This is the M2 (DESIGN_inference §B) upgrade of the original validation: instead of overlaying
the real data with *hardcoded* assay parameters, we now **estimate** the technical
hyper-parameters from the real dataset (Splatter-style `estimate()`), simulate an iscc tumour
through the scRNA assay *with the fitted params*, and overlay the same count-realism panel. The
improvement is that library size, dispersion and the dropout curve are **learned, not guessed**.

`estimate()` fits the DESIGN_features §B batch model (`BatchHyperParams`). PBMC3k is a single
batch, so the cross-batch hyper-parameters (`sigma_batch`, `depth_batch_sigma`) are not
identifiable here — they fall back to the protocol preset (the recovery test
`tests/test_estimate_scrna.py` exercises those on multi-batch synthetic data). What is fit from
PBMC3k: `mu_lib`, `sigma_lib`, NB `dispersion`, and the logistic `dropout` curve.

Usage:  python validation/validate_scrna.py [--real path.h5ad]
By default the real reference is scanpy's PBMC3k (downloaded on first run).
"""
import argparse
import os

import numpy as np
import yaml

from iscc.tumor.models import GenotypeTumor
from iscc.data import scRNA, estimate
from iscc.validation import compare_plot

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default=None, help="Real reference .h5ad (default: scanpy PBMC3k)")
    ap.add_argument("--out", default=os.path.join(REPO, "manuscript/figures/validation_scrna.png"))
    ap.add_argument("--protocol", default="10x")
    ap.add_argument("--seed", type=int, default=2)
    args = ap.parse_args()

    import scanpy as sc
    if args.real:
        real = sc.read_h5ad(args.real)
    else:
        real = sc.datasets.pbmc3k()

    # --- FIT the technical params from the real dataset (the M2 upgrade) -----------------
    est = estimate(real, protocol=args.protocol)
    print(f"FITTED on real ({real.shape[0]} cells x {real.shape[1]} genes):")
    print(f"  {est}")
    h = est.hypers

    # simulate a tumor with a realistic number of genes, then sequence it with FITTED params
    cfg = yaml.safe_load(open(os.path.join(REPO, "notebooks/example_config.yaml")))
    cfg["spatial_params"].update(structure_radius=0, grid_size=40)
    cfg["genome_params"].update(n_segments=40, segment_size=100)  # 4000 genes
    tmp = os.path.join(REPO, "validation", "_sim_config.yaml")
    yaml.safe_dump(cfg, open(tmp, "w"))
    tumor = GenotypeTumor(config=tmp, seed=args.seed)
    tumor.grow(n_steps=800, seed=args.seed)
    # drive the assay from the fitted hyper-parameters (vs the old hardcoded defaults)
    assay = scRNA(n_cells=600, seed=0, **est.scrna_kwargs())  # scrna_kwargs carries protocol
    assay.run({"cell_exp": tumor.cell_data["cell_exp"]})
    os.remove(tmp)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    _, stats = compare_plot(assay.observed_counts, real, out_path=args.out,
                            sim_label="iscc (fitted)", real_label="PBMC3k (real)")
    s, r = stats["sim"], stats["real"]
    print(f"SIM (fitted): CV(lib)={s['cv_lib']:.2f}  zero_frac={s['mean_zero_frac']:.2f}  "
          f"median_CV2={np.nanmedian(s['gene_cv2']):.1f}")
    print(f"REAL        : CV(lib)={r['cv_lib']:.2f}  zero_frac={r['mean_zero_frac']:.2f}  "
          f"median_CV2={np.nanmedian(r['gene_cv2']):.1f}")
    print(f"figure -> {args.out}")


if __name__ == "__main__":
    main()
