"""Harmony (harmonypy) runner — executed in the dedicated ``iscc-harmony`` conda env.

Kept out of the core ``iscc`` env (the clonealign/inferCNV convention, README_integration.md). Reads a
PCA embedding + per-cell batch labels from ``work_dir``, runs harmonypy, writes the corrected embedding.

Build the env:
    conda create -y -n iscc-harmony -c conda-forge python=3.10
    ~/miniconda3/envs/iscc-harmony/bin/pip install harmonypy numpy pandas

Usage (invoked by cohort_common.run_harmony):
    python harmony_runner.py <work_dir>          # reads emb.npy + batch.csv, writes corrected.npy
"""
import os
import sys

import numpy as np
import pandas as pd


def main(work_dir):
    import harmonypy
    emb = np.load(os.path.join(work_dir, "emb.npy"))
    batch = pd.read_csv(os.path.join(work_dir, "batch.csv"))
    meta = pd.DataFrame({"batch": batch.iloc[:, 0].astype(str).values})
    ho = harmonypy.run_harmony(emb, meta, ["batch"])
    corrected = np.asarray(ho.Z_corr).T
    np.save(os.path.join(work_dir, "corrected.npy"), corrected)


if __name__ == "__main__":
    main(sys.argv[1])
