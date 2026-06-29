"""
`isccsample` — sample cells from a simulated tumor (the lab/medical stage).

Stage 2 of the iscc pipeline. Reads the ground-truth `cell_data/` written by
`isccsim` and produces a *sample*: a subset of cells (with their per-cell
ground truth preserved) plus a `sample_meta.yaml` describing how it was taken.
The resulting `cell_data/` is the input to `isccdata`.

Two sampling methods (the F1/F2 sampling layer):

* **biopsy** (F1) — spatial region selection over the `cell_crd` grid:
  `--biopsy-type {needle,punch,multiregion,liquid}`. Multi-region tags each disk
  with its own `region` label (the multi-region-heterogeneity substrate for
  phylogeny); liquid biopsy draws a few circulating cells biased toward
  high-dispersal clones. A per-cell `region` label is written to
  `cell_data/cell_region.csv` and the geometry is recorded in `sample_meta.yaml`.
* **dissociation** (F2) — cell-type-dependent recovery -> composition bias
  (`--dissociation-bias`, `--recovery`), plus an optional dissociation-stress
  expression signature (`--stress-strength`).

Both methods write a valid `cell_data/` that `isccdata` consumes and preserve
per-cell ground truth (subset only; the stress signature, if enabled, perturbs
a small stress-gene set in `cell_exp`).
"""
import click
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from iscc.sample.biopsy.biopsy import Biopsy
from iscc.sample.dissociation.dissociation import Dissociation

# The full per-cell ground-truth schema written by Tumor.make_cell_data / write.
CELL_DATA_KEYS = [
    "cell_evo", "cell_exp", "cell_snv", "cell_cnv", "cell_crd", "cell_type",
    "cell_deme", "cell_rna_vaf",
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


def _parse_center(center):
    if center is None:
        return None
    try:
        parts = [float(x) for x in str(center).replace(" ", "").split(",")]
    except ValueError:
        raise click.ClickException("--center must be 'row,col', e.g. '10,12'.")
    if len(parts) != 2:
        raise click.ClickException("--center must be 'row,col', e.g. '10,12'.")
    return tuple(parts)


def _parse_recovery(recovery):
    """Parse 'cancer=0.9,immune=0.4,...' into a dict; None -> defaults."""
    if not recovery:
        return None
    out = {}
    for item in str(recovery).split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise click.ClickException(
                "--recovery must be 'type=prob,...' e.g. 'immune=0.4,stromal=0.6'."
            )
        t, p = item.split("=", 1)
        out[t.strip()] = float(p)
    return out


@click.command(help="Sample cells from a simulated tumor (stage 2 of the pipeline).")
@click.argument("tumor-path", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--method", default="dissociation",
    type=click.Choice(["dissociation", "biopsy"]),
    help="Sampling method.",
)
@click.option("--fraction", default=1.0, type=float,
              help="Optional uniform thinning of the selected cells (0-1].")
# --- F1 biopsy geometry ---
@click.option("--biopsy-type", default="punch",
              type=click.Choice(["needle", "punch", "multiregion", "liquid"]),
              help="Biopsy geometry (method=biopsy).")
@click.option("--radius", default=None, type=float, help="Disk radius (punch/multiregion).")
@click.option("--width", default=None, type=float, help="Strip width (needle).")
@click.option("--angle", default=0.0, type=float, help="Needle angle in degrees.")
@click.option("--n-regions", default=3, type=int, help="Number of disjoint regions (multiregion).")
@click.option("--n-liquid", default=20, type=int, help="Number of circulating cells (liquid).")
@click.option("--center", default=None, help="Geometry centre as 'row,col' (default: centroid).")
@click.option("--grid-size", default=None, type=int, help="Grid extent (default: infer from coords).")
# --- F2 dissociation ---
@click.option("--dissociation-bias/--no-dissociation-bias", default=True,
              help="Apply cell-type-dependent recovery (method=dissociation).")
@click.option("--recovery", default=None,
              help="Per-type recovery probs, e.g. 'immune=0.4,stromal=0.6'.")
@click.option("--stress-strength", default=0.0, type=float,
              help="Dissociation-stress expression bump on a small gene set.")
@click.option("-r", "--random-seed", default=42, help="Random seed.")
@click.option("--log", default=0, help="Logging level. 0 = critical, 1 = info, 2 = debug.")
@click.option("-o", "--output-path", default="./sample_out", help="Output directory.")
def main(tumor_path, method, fraction, biopsy_type, radius, width, angle,
         n_regions, n_liquid, center, grid_size, dissociation_bias, recovery,
         stress_strength, random_seed, log, output_path):
    logging.basicConfig(
        level={0: logging.CRITICAL, 1: logging.INFO, 2: logging.DEBUG}.get(log, logging.CRITICAL)
    )
    if not 0 < fraction <= 1:
        raise click.ClickException("--fraction must be in (0, 1].")

    rng = np.random.default_rng(random_seed)
    data = load_cell_data(tumor_path)
    n = len(next(iter(data.values())).index)

    meta = dict(
        method=method,
        fraction=float(fraction),
        n_input=int(n),
        seed=int(random_seed),
        source=os.path.abspath(tumor_path),
    )
    region_series = None
    exp_override = None

    if method == "biopsy":
        bx = Biopsy(data, rng=rng, grid_size=grid_size)
        chosen, region_series, geom = bx.sample(
            biopsy_type=biopsy_type, center=_parse_center(center), radius=radius,
            width=width, angle=angle, n_regions=n_regions, n_liquid=n_liquid,
        )
        if len(chosen) == 0:
            raise click.ClickException(
                f"Biopsy geometry ({biopsy_type}) selected 0 cells; widen the region."
            )
        meta["biopsy_type"] = biopsy_type
        meta["biopsy"] = geom
    else:  # dissociation
        # With bias off, every type is recovered equally (no composition shift).
        probs = _parse_recovery(recovery) if dissociation_bias else _uniform_recovery()
        diss = Dissociation(data, rng=rng, recovery_probs=probs,
                            stress_strength=stress_strength)
        chosen, dmeta, exp_override = diss.run()
        meta["dissociation_bias"] = bool(dissociation_bias)
        meta["dissociation"] = dmeta

    # Optional uniform thinning of the selected cells.
    if fraction < 1.0 and len(chosen) > 1:
        k = max(1, int(round(fraction * len(chosen))))
        chosen = pd.Index(rng.choice(np.asarray(chosen), size=k, replace=False))
        if region_series is not None:
            region_series = region_series.loc[chosen]

    out_dir = Path(output_path) / "cell_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, df in data.items():
        src = exp_override if (key == "cell_exp" and exp_override is not None) else df
        src.loc[chosen].to_csv(out_dir / f"{key}.csv")
    if region_series is not None:
        region_series.to_csv(out_dir / "cell_region.csv")

    meta["n_sampled"] = int(len(chosen))
    with open(Path(output_path) / "sample_meta.yaml", "w") as f:
        yaml.safe_dump(meta, f, sort_keys=False)

    logging.info("Sampled %d/%d cells via %s -> %s", len(chosen), n, method, output_path)
    print(f"Sampled {len(chosen)}/{n} cells via {method} -> {output_path}")


def _uniform_recovery():
    """Recovery probs that keep every type equally (no composition bias)."""
    from iscc.sample.dissociation.dissociation import DEFAULT_RECOVERY
    return {t: 1.0 for t in DEFAULT_RECOVERY}


if __name__ == "__main__":
    main()
