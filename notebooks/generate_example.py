"""Generate the shared example dataset used by the iscc example notebooks.

Runs the full pipeline once into ``notebooks/example_out/``:

    tumor/                 (isccsim)   -- ground-truth tumor state
    sample/                (isccsample)-- a biopsy of the tumor
    data_{scrna,bdna,scdna,visium}/ (isccdata) -- observed molecular data

Idempotent: skips work if ``example_out/tumor`` already exists, unless --force.
Run from anywhere:  python notebooks/generate_example.py [--force] [--steps N]
"""
import argparse
import os
import shutil

import numpy as np
import yaml

from iscc.tumor.models import GenotypeTumor
from iscc.data import ASSAYS

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(HERE, "example_config.yaml")
OUT = os.path.join(HERE, "example_out")

ASSAY_CONFIGS = {
    "scrna": dict(n_reads=4000, n_cells=400),
    "bdna": dict(n_reads=50000, data_mode="counts", fpr=0.02, fnr=0.02),
    "scdna": dict(n_reads=200, n_cells=200, data_mode="counts", fpr=0.01, fnr=0.05),
    "visium": dict(n_reads=4000, n_spots_x=20, n_spots_y=20, spot_radius=1.5),
}


def simulate_tumor(steps=40, seed=2):
    # genotype-level engine (config sets mode: genotype); seed 2 keeps the founder alive
    tumor = GenotypeTumor(config=CONFIG, seed=seed)
    tumor.grow(n_steps=steps, seed=seed)
    return tumor


def sample_cells(tumor_dir, sample_dir, biopsy_type="punch", seed=0):
    import pandas as pd
    from iscc.sample.biopsy.biopsy import Biopsy
    keys = ["cell_evo", "cell_exp", "cell_snv", "cell_cnv", "cell_crd", "cell_type", "cell_deme"]
    src = os.path.join(tumor_dir, "cell_data")
    data = {k: pd.read_csv(os.path.join(src, f"{k}.csv"), index_col=0) for k in keys}
    # spatial biopsy over the cell_crd grid (a punch = disk), not a random subset.
    bx = Biopsy(data, rng=np.random.default_rng(seed))
    chosen, region, geom = bx.sample(biopsy_type=biopsy_type)
    out = os.path.join(sample_dir, "cell_data")
    os.makedirs(out, exist_ok=True)
    for k, df in data.items():
        df.loc[chosen].to_csv(os.path.join(out, f"{k}.csv"))
    if region is not None and len(region):
        region.to_csv(os.path.join(out, "cell_region.csv"))
    with open(os.path.join(sample_dir, "sample_meta.yaml"), "w") as f:
        yaml.safe_dump(dict(method="biopsy", biopsy_type=biopsy_type, biopsy=geom,
                            n_input=int(len(data["cell_snv"])), n_sampled=int(len(chosen)),
                            seed=seed, source=tumor_dir), f)
    return data, chosen


def run_assays(sample_dir):
    import pandas as pd
    src = os.path.join(sample_dir, "cell_data")
    cell_data = {k: pd.read_csv(os.path.join(src, f"{k}.csv"), index_col=0)
                 for k in ["cell_snv", "cell_exp", "cell_cnv", "cell_crd"]}
    for name, cfg in ASSAY_CONFIGS.items():
        assay = ASSAYS[name](**cfg)
        run_kwargs = {}
        if name == "visium":
            run_kwargs["grid_side"] = int(cell_data["cell_crd"].max().max()) + 1
        assay.run(cell_data, **run_kwargs)
        out = os.path.join(OUT, f"data_{name}")
        os.makedirs(out, exist_ok=True)
        assay.write(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="Regenerate even if it exists.")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=2)
    args = ap.parse_args()

    tumor_dir = os.path.join(OUT, "tumor")
    sample_dir = os.path.join(OUT, "sample")

    if os.path.exists(tumor_dir) and not args.force:
        print(f"Example data already present at {OUT} (use --force to regenerate).")
        return
    if args.force and os.path.exists(OUT):
        shutil.rmtree(OUT)

    print("Simulating tumor ...")
    tumor = simulate_tumor(steps=args.steps, seed=args.seed)
    tumor.write(tumor_dir)
    print(f"  tumor size = {tumor.get_tumor_size()} cells")

    print("Sampling (punch biopsy) ...")
    sample_cells(tumor_dir, sample_dir, biopsy_type="punch", seed=args.seed)

    print("Running assays (scrna, bdna, scdna, visium) ...")
    run_assays(sample_dir)

    print(f"Done. Example dataset written to {OUT}")


if __name__ == "__main__":
    main()
