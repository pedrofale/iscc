"""
`isccdata` — run a sequencing / spatial-transcriptomics assay on a sampled tumor.

Stage 3 of the iscc pipeline. Reads the per-cell ground truth produced by
`isccsample` (a `cell_data/` directory) and emits observed molecular data
(bulk/single-cell DNA, single-cell RNA, or Visium spots).
"""
from . import ASSAYS, ASSAY_NAMES

import click
import logging
import os
from pathlib import Path

import pandas as pd

def load_cell_data(path):
    """Load the full cell_data dict from a sample directory (or a cell_data/ dir directly).

    Reads every per-cell CSV present, so the optional ground-truth layers (WGD, gene
    programs, allele-specific expression, gland id, microenvironment, region) reach the
    assay whenever the tumor was grown/sampled with them; the base schema loads on its own.
    """
    base = path
    nested = os.path.join(path, "cell_data")
    if os.path.isdir(nested):
        base = nested
    data = {f.stem: pd.read_csv(f, index_col=0) for f in sorted(Path(base).glob("*.csv"))}
    if not data:
        raise click.ClickException(
            f"No cell_data CSVs found under {path!r}. Run `isccsample` first."
        )
    return data


@click.command(help="Run a molecular assay on a sampled tumor (stage 3 of the pipeline).")
@click.argument("sample-path", type=click.Path(exists=True, file_okay=False))
@click.option(
    "-a", "--assay", default="scrna",
    type=click.Choice(sorted(ASSAYS)), help="Assay type.",
)
@click.option(
    "--assay-config", default=None, type=click.Path(exists=True, dir_okay=False),
    help="YAML config of assay parameters (n_reads, n_cells, fpr/fnr, ...).",
)
@click.option(
    "--grid-side", default=0, type=int,
    help="Spatial grid side for Visium. 0 = infer from cell coordinates.",
)
@click.option(
    "--protocol", default="10x", type=str,
    help="scRNA protocol preset: 10x or smartseq3.",
)
@click.option(
    "--breadth", default="wgs", type=click.Choice(["wgs", "wes", "panel"]),
    help="DNA capture breadth (bdna / scdna): wgs, wes, or panel.",
)
@click.option(
    "--depth-model", default="dm", type=click.Choice(["dm", "nb"]),
    help="DNA depth model: dm (Dirichlet-Multinomial, default) or nb (per-bin NB).",
)
@click.option(
    "--batches", default=1, type=int,
    help="scRNA: number of batches (same biology, different technical signature).",
)
@click.option(
    "--batch-seed", default=42, type=int,
    help="scRNA: base seed; batch b uses batch-seed + 1 + b.",
)
@click.option(
    "--design", default="shared", type=click.Choice(["shared", "split"]),
    help="scRNA multi-batch design: shared cells (replicate) or split (balanced).",
)
@click.option(
    "--count-model", default="nb", type=str,
    help="scRNA count emission: nb (default) or dm.",
)
@click.option("--log", default=0, help="Logging level. 0 = critical, 1 = info, 2 = debug.")
@click.option("-o", "--output-path", default="./data_out", help="Output directory.")
def main(sample_path, assay, assay_config, grid_side, protocol, breadth, depth_model,
         batches, batch_seed, design, count_model, log, output_path):
    logging.basicConfig(
        level={0: logging.CRITICAL, 1: logging.INFO, 2: logging.DEBUG}.get(log, logging.CRITICAL)
    )

    config = {}
    if assay_config:
        import yaml
        with open(assay_config) as f:
            config = yaml.safe_load(f) or {}

    cell_data = load_cell_data(sample_path)

    logging.info("Running %s on %d cells", ASSAY_NAMES[assay], len(next(iter(cell_data.values()))))

    # scRNA multi-batch path: one labelled AnnData per batch (+ concatenated benchmark).
    if assay == "scrna" and batches > 1:
        from .rna import run_scrna_batches, concat_batches
        config.pop("protocol", None)
        config.pop("count_model", None)
        assays = run_scrna_batches(
            cell_data, n_batches=batches, base_seed=batch_seed, design=design,
            protocol=protocol, count_model=count_model, **config,
        )
        Path(output_path).mkdir(parents=True, exist_ok=True)
        for a in assays:
            a.write(output_path)
        concat_batches(assays).write_h5ad(os.path.join(output_path, "scrna_all_batches.h5ad"))
        logging.info("Saved %d scRNA batches to %s", batches, output_path)
        print(f"{ASSAY_NAMES[assay]} ({batches} batches, {protocol}) finished -> {output_path}")
        return

    # DNA multi-batch path: one labelled output per batch (same biology, different run).
    if assay in ("bdna", "scdna") and batches > 1:
        from .dna import run_dna_batches
        config.setdefault("breadth", breadth)
        config.setdefault("depth_model", depth_model)
        mode = "bulk" if assay == "bdna" else "sc"
        assays = run_dna_batches(
            cell_data, mode=mode, n_batches=batches, base_seed=batch_seed,
            design=design, **config,
        )
        Path(output_path).mkdir(parents=True, exist_ok=True)
        for a in assays:
            a.write(os.path.join(output_path, a.batch.label))
        logging.info("Saved %d %s batches to %s", batches, ASSAY_NAMES[assay], output_path)
        print(f"{ASSAY_NAMES[assay]} ({batches} batches, {breadth}) finished -> {output_path}")
        return

    if assay == "scrna":
        config.setdefault("protocol", protocol)
        config.setdefault("count_model", count_model)
        config.setdefault("seed", batch_seed)
    if assay in ("bdna", "scdna"):
        config.setdefault("breadth", breadth)
        config.setdefault("depth_model", depth_model)
        config.setdefault("seed", batch_seed)
    if assay == "visium":
        # Visium's compositional per-spot capture defaults to the DM count model (the CLI
        # --count-model default is "nb" for scRNA; only override here if the user passed one).
        config.setdefault("count_model", count_model if count_model != "nb" else "dm")
        config.setdefault("seed", batch_seed)
    if assay == "scspatial":
        config.setdefault("count_model", count_model)
        config.setdefault("seed", batch_seed)
    instrument = ASSAYS[assay](**config)

    run_kwargs = {}
    if assay == "visium":
        if "cell_crd" not in cell_data:
            raise click.ClickException("Visium needs spatial coordinates (cell_crd.csv).")
        if grid_side <= 0:
            grid_side = int(cell_data["cell_crd"].max().max()) + 1
        run_kwargs["grid_side"] = grid_side

    instrument.run(cell_data, **run_kwargs)

    Path(output_path).mkdir(parents=True, exist_ok=True)
    instrument.write(output_path)
    logging.info("Saved %s output to %s", ASSAY_NAMES[assay], output_path)
    print(f"{ASSAY_NAMES[assay]} finished -> {output_path}")


if __name__ == "__main__":
    main()
