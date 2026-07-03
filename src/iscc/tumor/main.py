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
from ..treatment.chemotherapy import Chemotherapy
from ..treatment.targeted import TargetedTherapy
from ..treatment.immunotherapy import Immunotherapy

import click
import logging
import yaml

# Map the `mode` field in the sim-config to a tumor model.
TUMOR_MODELS = {
    "glandular": GlandularTumor,      # cell-level agent-based engine
    "mixed": MixedTumor,
    "genotype": GenotypeTumor,        # fast genotype-level (count-based) engine
}


def _gene_coords(flat_indices, segment_size):
    """Convert flat gene indices into (segment, position) coordinates."""
    return [(int(i) // segment_size, int(i) % segment_size) for i in flat_indices]


def build_treatment(kind, tumor, adaptive, start, duration, max_tumor_size):
    """Construct a treatment of the requested kind, targeting the tumor's drivers.

    chemo   -> kills all cancer cells (death-rate therapy)
    targeted-> kills cancer cells expressing oncogenes (death-rate therapy)
    immuno  -> strips immune resistance so the local immune cells kill more (needs
               immune cells present: set spatial_params.immune_density in the config)
    """
    if kind in (None, "none"):
        return None
    kw = dict(adaptive=adaptive, start=start, duration=duration, max_tumor_size=max_tumor_size)
    seg = tumor.selection.segment_size
    if kind == "chemo":
        return Chemotherapy(**kw)
    if kind == "targeted":
        targets = _gene_coords(tumor.selection.get_oncogenes(), seg)
        return TargetedTherapy(targets=targets, **kw)
    if kind == "immuno":
        return Immunotherapy(immune_checkpoints=[], **kw)  # broad: all cancer
    raise click.ClickException(f"Unknown treatment '{kind}'.")


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
    "--treatment", "treatment_kind", default="none",
    type=click.Choice(["none", "chemo", "targeted", "immuno"]),
    help="Apply a therapy during growth (immuno needs spatial_params.immune_density>0).",
)
@click.option("--adaptive", is_flag=True, help="Adaptive dosing (dose only above --max-tumor-size).")
@click.option("--treatment-start", default=0, help="Step at which treatment begins.")
@click.option("--treatment-duration", default=None, type=int, help="Treatment duration in steps (default: until the end).")
@click.option("--max-tumor-size", default=100_000, help="Adaptive-dosing tumor-size threshold.")
@click.option(
    "--log", default=0, help="Logging level. 0 = critical, 1 = info, 2 = debug."
)
@click.option("--no-diagnose", is_flag=True,
              help="Skip the built-in operating-envelope QC that warns on degenerate tumors.")
@click.option("-o", "--output-path", default="./sim_out", help="Output directory.")
def main(sim_config, steps, random_seed, batch_size, treatment_kind, adaptive,
         treatment_start, treatment_duration, max_tumor_size, log, no_diagnose, output_path):
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

    treatment = build_treatment(treatment_kind, tumor, adaptive, treatment_start,
                                treatment_duration, max_tumor_size)
    if treatment is not None:
        logging.info("Applying %s therapy (adaptive=%s, start=%d)",
                     treatment_kind, adaptive, treatment_start)

    logging.info("Growing for %d steps (seed=%d)", steps, random_seed)
    tumor.grow(n_steps=steps, seed=random_seed, batch_size=batch_size, treatment=treatment)

    tumor.write(output_path)
    logging.info("Saved tumor (size=%d) to %s", tumor.get_tumor_size(), output_path)
    msg = f"Simulation ({mode}) finished: {tumor.get_tumor_size()} cells -> {output_path}"
    if treatment is not None:
        msg += f" [treatment: {treatment_kind}]"
    print(msg)

    # Operating-envelope QC: warn (opt-out) if the run produced a degenerate tumor, or surface a
    # size advisory (e.g. "grow to a realistic size"). Read-only — it never touches the simulation
    # output (see DESIGN_operating_envelope.md).
    if not no_diagnose and hasattr(tumor, "diagnose"):
        diag = tumor.diagnose()
        if not diag.ok:
            click.echo("\n" + click.style(
                "warning: this run produced a DEGENERATE tumor (see --no-diagnose to silence)",
                fg="yellow"))
            click.echo(diag.report())
        elif diag.advisories:
            for a in diag.advisories:
                click.echo(click.style(f"note: {a}", fg="yellow"))


if __name__ == "__main__":
    main()
