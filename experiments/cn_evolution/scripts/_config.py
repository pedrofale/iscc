"""Build one tumour configuration from the grid axes named in config.yaml.

The Snakefile and every script agree on the merge order here, so a dataset's parameters are
reproducible from its path alone: ``{scenario}/{evo}/{selection}/{genome}/{pop}/seed{N}``.
"""
import copy


def _deep_merge(base, extra):
    out = copy.deepcopy(base)
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def axis(cfg, name, key):
    grid = cfg.get(name) or {}
    if key not in grid:
        raise KeyError(f"{name}: no entry {key!r} (have {sorted(grid)})")
    return grid[key] or {}


def build_tumor_config(cfg, scenario, evo, selection, genome, pop, seed):
    """Merge the six axes into the nested dict ``GenotypeTumor(config=...)`` expects.

    Merge order is defaults -> genome -> selection -> evo -> pop -> scenario, so the tissue
    scenario has the last word on spatial and demographic keys it must control.

    Two traps this function exists to avoid:

    * ``max_birth_rate`` must exceed ``division_rate``. The engine default is 0.3, so a config that
      raises ``division_rate`` without raising the cap silently clamps EVERY clone to the same rate
      and selection becomes invisible — a whole grid axis quietly doing nothing.
    * In a glandular scenario the per-deme capacity comes from ``K_duct`` / ``K_stroma``, and a bare
      ``carrying_capacity`` from the ``pop`` axis is ignored. The demographic axis is therefore
      applied as a SCALING of both compartment capacities, so it stays live in both scenarios.
    """
    out = _deep_merge({}, cfg.get("defaults") or {})
    for name, key in (("genome_grid", genome), ("selection_grid", selection),
                      ("evo_grid", evo), ("pop_grid", pop), ("scenario_grid", scenario)):
        out = _deep_merge(out, axis(cfg, name, key))

    out.setdefault("mode", "genotype")
    cancer = out.setdefault("cell_params", {}).setdefault("cancer", {})
    div = float(cancer.get("division_rate", 0.4))
    if float(cancer.get("max_birth_rate", 0.3)) <= div:
        cancer["max_birth_rate"] = round(min(0.99, div * 2.0 + 0.1), 4)

    spatial = out.setdefault("spatial_params", {})
    deme = out.setdefault("deme_params", {})
    scale = float((axis(cfg, "pop_grid", pop) or {}).get("capacity_scale", 1.0))
    if float(spatial.get("structure_radius", 0)) > 0:
        for key, default in (("K_duct", 12), ("K_stroma", 10)):
            spatial[key] = max(2, int(round(float(spatial.get(key, default)) * scale)))
        deme["carrying_capacity"] = max(2, int(round(
            float(deme.get("carrying_capacity", 10)) * scale)))
    else:
        deme["carrying_capacity"] = max(2, int(round(
            float(deme.get("carrying_capacity", 10)) * scale)))

    steps = int((axis(cfg, "pop_grid", pop) or {}).get("steps", cfg.get("default_steps", 400)))
    out["_steps"] = steps
    out["_seed"] = int(seed)
    return out


def dataset_key(scenario, evo, selection, genome, pop, seed):
    return f"{scenario}/{evo}/{selection}/{genome}/{pop}/seed{int(seed)}"
