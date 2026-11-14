"""The constructor's parameter groups, resolved once from the keyword blocks or a YAML config.

:class:`~iscc.tumor.models.count.GenotypeTumor` takes its parameters as nested dicts (the same blocks a
YAML config holds). :func:`resolve_parameters` turns them into two typed groups: the run settings
(:class:`RunSettings`) and the model's parameter blocks (:class:`ModelBlocks`), applying a config file,
the real-genome wiring and a breast-subtype preset, so the construction steps read one resolved source.
"""
from dataclasses import dataclass
from typing import Optional

import yaml

from ...constants import DEFAULT_LAYOUT_SEED


# --- Breast-cancer subtype presets ------------------------------------------------------------
# Selected with ``cell_params.cancer.subtype`` (absent -> no preset -> nothing changes). A preset
# only supplies DEFAULTS: any of the three knobs written explicitly in the config still wins, so a
# config can pick a subtype and then override one number.
#
# The receptor subtypes of breast cancer differ in how chromosomally unstable they are, and that
# difference shows up in exactly three measurable quantities, which are the three knobs here:
#
#   * ``cnv_prob``     — how much of the genome ends up at a non-diploid copy number. Oestrogen-
#                        receptor-positive (luminal) tumours are the quieter end of the spectrum
#                        (roughly a fifth of the genome altered); triple-negative (basal-like)
#                        tumours are the loud end (roughly twice that).
#   * ``wgd_rate``     — how often a whole-genome doubling happens. Doublings are common in breast
#                        cancer and markedly commoner in triple-negative disease than in luminal.
#   * ``mutation_rate`` — ongoing instability: the rate at which new clone-defining events arise at
#                        all, so how much subclonal structure the tumour keeps generating.
#
# Nothing else about the subtypes is modelled here — no receptor signalling, no subtype-specific
# driver genes, no treatment response. These are three instability dials and that is all.
BREAST_SUBTYPES = {
    # cnv_prob is the share of mutating divisions that change copy number. These values give the
    # aneuploidy burden a real breast tumour carries (about a third of the genome altered for the
    # luminal subtype).
    "ER+": dict(cnv_prob=0.045, wgd_rate=0.001, mutation_rate=0.30),
    "TNBC": dict(cnv_prob=0.100, wgd_rate=0.006, mutation_rate=0.45),
}


@dataclass
class RunSettings:
    """How a run is stepped, recorded and seeded (top-level keys of a config)."""
    update_mode: str = "exact"
    tau: float = 1.0
    snapshot_every: int = 1
    spatial_snapshot_every: int = 0
    spatial_snapshot_budget: int = 5_000_000
    coarsen_passengers: bool = False
    max_cells: Optional[int] = None
    germline_params: Optional[dict] = None
    founder_mutations: Optional[dict] = None
    layout_seed: Optional[int] = None


@dataclass
class ModelBlocks:
    """The model's parameter blocks: genome, selection, the cell types, the deme law and the tissue,
    plus the two readout layers. ``subtype`` is the breast-subtype preset taken out of ``cancer``."""
    genome: Optional[dict] = None
    selection: Optional[dict] = None
    cancer: Optional[dict] = None
    deme: Optional[dict] = None
    spatial: Optional[dict] = None
    epithelial: Optional[dict] = None
    stromal: Optional[dict] = None
    immune: Optional[dict] = None
    host: Optional[dict] = None
    microenv: Optional[dict] = None
    expression: Optional[dict] = None
    subtype: Optional[str] = None


def resolve_parameters(config, genome_mode, genome_spec, r, p):
    """Resolve the constructor's arguments into ``(RunSettings, ModelBlocks)``.

    A ``config`` file overrides the keyword blocks; ``genome_mode="real"`` wires the genome and its
    selection from ``genome_spec`` (``selection_params`` such as ``s_arm`` still win); a ``subtype``
    key in the cancer block fills in that preset's instability numbers (explicit values win)."""
    if genome_mode == "real":
        if genome_spec is None:
            raise ValueError("genome_mode='real' requires genome_spec")
        p.genome, p.selection = genome_spec.engine_params(p.selection)
    if config is not None:
        with open(config) as f:
            cfg = yaml.safe_load(f)
        p.genome = cfg["genome_params"]
        p.selection = cfg["selection_params"]
        p.deme = cfg["deme_params"]
        p.spatial = cfg["spatial_params"]
        cp = cfg["cell_params"]
        p.cancer = cp["cancer"]
        p.epithelial = cp.get("epithelial", {})
        p.stromal = cp.get("stromal", {})
        p.immune = cp.get("immune", {})
        p.host = cp.get("host", {})
        r.update_mode = cfg.get("update_mode", r.update_mode)
        r.tau = cfg.get("tau", r.tau)
        r.snapshot_every = cfg.get("snapshot_every", r.snapshot_every)
        r.spatial_snapshot_every = cfg.get("spatial_snapshot_every", r.spatial_snapshot_every)
        r.spatial_snapshot_budget = cfg.get("spatial_snapshot_budget", r.spatial_snapshot_budget)
        r.coarsen_passengers = cfg.get("coarsen_passengers", r.coarsen_passengers)
        r.max_cells = cfg.get("max_cells", r.max_cells)
        r.germline_params = cfg.get("germline_params", r.germline_params)
        r.founder_mutations = cfg.get("founder_mutations", r.founder_mutations)
        p.microenv = cfg.get("microenv_params", p.microenv)
        p.expression = cfg.get("expression_params", p.expression)
        if cfg.get("layout_seed") is not None:
            r.layout_seed = cfg["layout_seed"]
    if r.layout_seed is None:
        r.layout_seed = DEFAULT_LAYOUT_SEED
    # Subtype preset (see BREAST_SUBTYPES). ``subtype`` is a cancer-cell config key that stands in for
    # three instability numbers; it is popped here so the cell class never sees it.
    cancer = dict(p.cancer or {})
    subtype = cancer.pop("subtype", None)
    if subtype is not None:
        if subtype not in BREAST_SUBTYPES:
            raise ValueError(
                f"unknown cancer subtype {subtype!r}; choose one of "
                f"{sorted(BREAST_SUBTYPES)} (or drop the key and set cnv_prob / wgd_rate / "
                f"mutation_rate directly)")
        for k, v in BREAST_SUBTYPES[subtype].items():
            cancer.setdefault(k, v)   # an explicit config value still wins
    p.cancer = cancer
    p.subtype = subtype
    return r, p
