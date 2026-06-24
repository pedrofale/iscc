"""Validate iscc's simulated scRNA-seq against a real 10x reference (PBMC3k).

Generates a tumor with the genotype engine, runs the scRNA assay, and overlays standard
count-realism statistics (library size, mean-variance/overdispersion, dropout, sparsity)
on a real dataset. Produces manuscript/figures/validation_scrna.png.

Usage:  python validation/validate_scrna.py [--real path.h5ad]
By default the real reference is scanpy's PBMC3k (downloaded on first run).
"""
import argparse
import os

import numpy as np
import yaml

from iscc.tumor.models import GenotypeTumor
from iscc.data import scRNA
from iscc.validation import compare_plot

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default=None, help="Real reference .h5ad (default: scanpy PBMC3k)")
    ap.add_argument("--out", default=os.path.join(REPO, "manuscript/figures/validation_scrna.png"))
    ap.add_argument("--seed", type=int, default=2)
    args = ap.parse_args()

    import scanpy as sc
    if args.real:
        real = sc.read_h5ad(args.real)
    else:
        real = sc.datasets.pbmc3k()

    # simulate a tumor with a realistic number of genes, then sequence it
    cfg = yaml.safe_load(open(os.path.join(REPO, "notebooks/example_config.yaml")))
    cfg["spatial_params"].update(structure_radius=0, grid_size=40)
    cfg["genome_params"].update(n_segments=40, segment_size=100)  # 4000 genes
    tmp = os.path.join(REPO, "validation", "_sim_config.yaml")
    yaml.safe_dump(cfg, open(tmp, "w"))
    tumor = GenotypeTumor(config=tmp, seed=args.seed)
    tumor.grow(n_steps=800, seed=args.seed)
    assay = scRNA(n_reads=3000, n_cells=600, dispersion=0.5, lib_size_sigma=0.4, seed=0)
    assay.run({"cell_exp": tumor.cell_data["cell_exp"]})
    os.remove(tmp)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    _, stats = compare_plot(assay.observed_counts, real, out_path=args.out)
    s, r = stats["sim"], stats["real"]
    print(f"SIM : CV(lib)={s['cv_lib']:.2f}  zero_frac={s['mean_zero_frac']:.2f}  "
          f"median_CV2={np.nanmedian(s['gene_cv2']):.1f}")
    print(f"REAL: CV(lib)={r['cv_lib']:.2f}  zero_frac={r['mean_zero_frac']:.2f}  "
          f"median_CV2={np.nanmedian(r['gene_cv2']):.1f}")
    print(f"figure -> {args.out}")


if __name__ == "__main__":
    main()
