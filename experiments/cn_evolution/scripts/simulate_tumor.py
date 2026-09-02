"""Grow one tumour and checkpoint it (workflow stage 1)."""
import argparse, json, os, pickle, sys
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _config import build_tumor_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    for a in ("scenario", "evo", "selection", "genome", "pop"):
        p.add_argument(f"--{a}", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    tc = build_tumor_config(cfg, args.scenario, args.evo, args.selection,
                            args.genome, args.pop, args.seed)
    steps, seed = tc.pop("_steps"), tc.pop("_seed")

    from iscc.tumor.models import GenotypeTumor
    # trace_occupancy is what makes the r/K split (Q3) and the colonisation curve (Q7) measurable;
    # it is inert when off, so the workflow always asks for it.
    t = GenotypeTumor(seed=seed, trace_occupancy=True, **_kwargs(tc))
    t.grow(steps, seed=seed)
    t.make_cell_data()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "tumor.pkl"), "wb") as f:
        pickle.dump(t, f)
    with open(os.path.join(args.out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump({**tc, "steps": steps, "seed": seed}, f, sort_keys=True)
    diag = t.diagnose()
    with open(os.path.join(args.out_dir, "diagnosis.json"), "w") as f:
        json.dump({"ok": bool(diag.ok),
                   "failures": [c.name for c in getattr(diag, "failures", [])],
                   "n_cancer_genotypes": sum(1 for g in t.genotypes_counts if t._is_cancer(g)),
                   "tumor_size": int(t.get_tumor_size())}, f, indent=2)
    print(f"grown: size={t.get_tumor_size()} "
          f"clones={sum(1 for g in t.genotypes_counts if t._is_cancer(g))} diagnose_ok={diag.ok}")


def _kwargs(tc):
    """Map the merged config dict onto GenotypeTumor's constructor keywords."""
    cp = tc.get("cell_params", {}) or {}
    out = dict(
        genome_params=tc.get("genome_params"), selection_params=tc.get("selection_params"),
        cancer_cell_params=cp.get("cancer"), epithelial_cell_params=cp.get("epithelial"),
        stromal_cell_params=cp.get("stromal"), immune_cell_params=cp.get("immune"),
        deme_params=tc.get("deme_params"), spatial_params=tc.get("spatial_params"),
        update_mode=tc.get("update_mode", "exact"), tau=tc.get("tau", 1.0),
        snapshot_every=tc.get("snapshot_every", 1),
        coarsen_passengers=bool(tc.get("coarsen_passengers", False)),
        max_cells=tc.get("max_cells"),
    )
    return {k: v for k, v in out.items() if v is not None}


if __name__ == "__main__":
    main()
