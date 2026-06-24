"""
`isccsim` — simulate spatial tumor growth under a configurable evolutionary model.

Stage 1 of the iscc pipeline: grow a tumor and write its ground-truth state
(per-cell genotypes/CNVs/expression, clone trace, spatial grid). Downstream:
`isccsample` (biopsy/dissociation) → `isccdata` (sequencing/spatial assay).

Inspired by Noble et al, 2019; selection follows a CINner-style copy-number model.
"""
from .models.glandular import GlandularTumor
from .models.mixed import MixedTumor
from .models.count import GenotypeTumor

import click
import logging
import yaml

# Map the `mode` field in the sim-config to a tumor model.
TUMOR_MODELS = {
    "glandular": GlandularTumor,      # cell-level agent-based engine
    "mixed": MixedTumor,
    "genotype": GenotypeTumor,        # fast genotype-level (count-based) engine
}


@click.command(help="Simulate spatial tumor growth (stage 1 of the iscc pipeline).")
@click.option(
    "--sim-config",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="YAML config with mode/spatial/genome/selection/cell/deme params.",
)
@click.option("-s", "--steps", default=1000, help="Number of simulation steps.")
@click.option("-r", "--random-seed", default=42, help="Random seed.")
@click.option("--batch-size", default=1, help="Number of demes updated per step.")
@click.option(
    "--log", default=0, help="Logging level. 0 = critical, 1 = info, 2 = debug."
)
@click.option("-o", "--output-path", default="./sim_out", help="Output directory.")
def main(sim_config, steps, random_seed, batch_size, log, output_path):
    logging.basicConfig(
        level={0: logging.CRITICAL, 1: logging.INFO, 2: logging.DEBUG}.get(log, logging.CRITICAL)
    )

    with open(sim_config) as f:
        config = yaml.safe_load(f)

    # Default to the fast genotype-level engine; 'glandular' is the cell-level engine.
    mode = config.get("mode", "genotype")
    if mode not in TUMOR_MODELS:
        raise click.ClickException(
            f"Unknown mode '{mode}'. Choose one of: {sorted(TUMOR_MODELS)}."
        )
    if mode == "mixed":
        raise click.ClickException(
            "The 'mixed' (non-spatial) model is not yet implemented; use 'genotype' or 'glandular'."
        )

    logging.info("Building %s tumor from %s", mode, sim_config)
    tumor = TUMOR_MODELS[mode](config=sim_config, seed=random_seed)

    logging.info("Growing for %d steps (seed=%d)", steps, random_seed)
    tumor.grow(n_steps=steps, seed=random_seed, batch_size=batch_size)

    tumor.write(output_path)
    logging.info("Saved tumor (size=%d) to %s", tumor.get_tumor_size(), output_path)
    print(f"Simulation ({mode}) finished: {tumor.get_tumor_size()} cells -> {output_path}")


if __name__ == "__main__":
    main()
