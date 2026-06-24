"""
`isccsample` — sample cells from a simulated tumor (the lab/medical stage).

Stage 2 of the iscc pipeline. Reads the ground-truth `cell_data/` written by
`isccsim` and produces a *sample*: a subset of cells (with their per-cell
ground truth preserved) plus a `sample_meta.yaml` describing how it was taken.
The resulting `cell_data/` is the input to `isccdata`.

NOTE: this is the minimal contract-establishing implementation. Realistic
biopsy (spatial region selection), physical slicing, and dissociation-induced
dropout/doublets are tracked for the sampling-layer milestone; for now every
method captures a random subset of the requested size and records its intent.
"""
import click
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# The full per-cell ground-truth schema written by Tumor.make_cell_data / write.
CELL_DATA_KEYS = [
    "cell_evo", "cell_exp", "cell_snv", "cell_cnv", "cell_crd", "cell_type", "cell_deme",
]


def load_cell_data(tumor_path):
    base = tumor_path
    nested = os.path.join(tumor_path, "cell_data")
    if os.path.isdir(nested):
        base = nested
    data = {}
    for key in CELL_DATA_KEYS:
        f = os.path.join(base, f"{key}.csv")
        if os.path.exists(f):
            data[key] = pd.read_csv(f, index_col=0)
    if not data:
        raise click.ClickException(
            f"No cell_data CSVs found under {tumor_path!r}. Run `isccsim` first."
        )
    return data


@click.command(help="Sample cells from a simulated tumor (stage 2 of the pipeline).")
@click.argument("tumor-path", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--method", default="dissociation",
    type=click.Choice(["dissociation", "biopsy"]),
    help="Sampling method (recorded in sample_meta.yaml).",
)
@click.option("--fraction", default=1.0, type=float, help="Fraction of cells to capture (0-1].")
@click.option("-r", "--random-seed", default=42, help="Random seed.")
@click.option("--log", default=0, help="Logging level. 0 = critical, 1 = info, 2 = debug.")
@click.option("-o", "--output-path", default="./sample_out", help="Output directory.")
def main(tumor_path, method, fraction, random_seed, log, output_path):
    logging.basicConfig(
        level={0: logging.CRITICAL, 1: logging.INFO, 2: logging.DEBUG}.get(log, logging.CRITICAL)
    )
    if not 0 < fraction <= 1:
        raise click.ClickException("--fraction must be in (0, 1].")

    rng = np.random.default_rng(random_seed)
    data = load_cell_data(tumor_path)

    ids = next(iter(data.values())).index
    n = len(ids)
    k = max(1, int(round(fraction * n)))
    chosen = rng.choice(np.asarray(ids), size=min(k, n), replace=False)

    out_dir = Path(output_path) / "cell_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, df in data.items():
        df.loc[chosen].to_csv(out_dir / f"{key}.csv")

    meta = dict(
        method=method,
        fraction=float(fraction),
        n_input=int(n),
        n_sampled=int(len(chosen)),
        seed=int(random_seed),
        source=os.path.abspath(tumor_path),
    )
    with open(Path(output_path) / "sample_meta.yaml", "w") as f:
        yaml.safe_dump(meta, f)

    logging.info("Sampled %d/%d cells via %s -> %s", len(chosen), n, method, output_path)
    print(f"Sampled {len(chosen)}/{n} cells via {method} -> {output_path}")


if __name__ == "__main__":
    main()
