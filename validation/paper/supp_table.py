"""The supplementary data file of runs and costs: every simulation of the paper, from the registry.

Two CSV files, written beside the figures (the paper carries no tables):

- ``runs.csv``: one row per run of the registry (``registry.RUNS``): what the variant changes against
  the paper regime, its scale and target size, how many tumours were grown and reached their target,
  CPU time, peak memory, the commits that grew them, and the figures that read it;
- ``runs_tumours.csv``: one row per tumour: run, task, seed, scale, target, cancer cells reached, time
  in generations, CPU and wall time, peak memory, and commit.

Both read the per-task records the registry writes beside each tumour (``<task>.json``), not the
shared ``manifest.jsonl``, on which concurrent array tasks can race. Each array task is its own process,
so peak memory is per tumour.
Run:  python validation/paper/supp_table.py
"""
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import registry as REG  # noqa: E402
from _paths import paper_dir  # noqa: E402

# What each run changes against the paper regime, and where it is used (kept beside the registry's
# definitions; a run added there without a row here is reported, not silently dropped).
ROWS = {
    "R0": ("Showcase tumour with a metastatic lattice, spatial history every 2 steps",
           "Figs 1, 2a, 2e, 3"),
    "R1": ("Cancer dispersal probability " + ", ".join(f"{d:g}" for d in REG.R1_DISPERSAL),
           "Fig 2f; supplementary territories figure"),
    "R2": ("Selection on or off (driver effect 1.0), passengers not folded",
           "Fig 2f; supplementary copy-number figure"),
    "R3": ("Ductal field or free growth, selection on or off", "Figs 2d, 2f, 4e"),
    "R4": ("Copy-number rate 0, 0.05, 0.15, 0.3; SNV burden 0.05, 0.2, 0.5 at copy-number rate 0.05",
           "Fig 2f; supplementary benchmark-difficulty figure"),
    "R5": ("Whole-genome duplication rate " + ", ".join(f"{w:g}" for w in REG.R5_WGD), "Fig 2f"),
    "R6": ("Six patients sharing one configuration", "Supplementary cohort figure"),
    "R10": ("Compartment barriers on or off, spatial history every 8 steps",
            "Fig 2b; supplementary cell-state figure"),
    "R11": ("Arm-homologous growth genome, copy-number events scaled by 39/12", "Fig 4b-d"),
    "R12": ("Metastatic lattice, twenty seeds", "Fig 2c, 2e"),
    "R13": ("Metastatic lattice with the selection on one trait switched off per arm", "Fig 2c"),
}


def records(run):
    """The per-task record of every tumour of ``run`` grown so far."""
    out = {}
    for task in REG.tasks(run):
        p = os.path.splitext(REG.task_path(run, task))[0] + ".json"
        if os.path.exists(p):
            with open(p) as f:
                out[task] = json.load(f)
    return out


def main():
    root = os.path.join(paper_dir(), "figures", "paper1")
    os.makedirs(root, exist_ok=True)
    runs, tumours = [], []
    for run in REG.RUNS:
        if run not in ROWS:
            print(f"!! {run} has no row in supp_table.ROWS")
            continue
        m = records(run)
        if not m:
            continue
        cpu = np.array([r["cpu_s"] for r in m.values()], dtype=float)
        sizes = sorted({(r["kwargs"]["scale"], r["kwargs"]["target_cancer"]) for r in m.values()})
        runs.append([run, ROWS[run][0], "; ".join(f"{s} {t}" for s, t in sizes), len(REG.tasks(run)), len(m),
                     sum(bool(r.get("reached_target")) for r in m.values()),
                     f"{np.median(cpu):.1f}", f"{cpu.min():.1f}", f"{cpu.max():.1f}",
                     max(r.get("peak_mb", 0) for r in m.values()),
                     " ".join(sorted({str(r.get("commit")) for r in m.values()})), ROWS[run][1]])
        for task, r in m.items():
            kw = r["kwargs"]
            tumours.append([run, task, kw.get("seed"), kw.get("scale"), kw.get("target_cancer"), r.get("cancer"),
                            r.get("time"), r.get("cpu_s"), r.get("wall_s"), r.get("peak_mb"),
                            bool(r.get("reached_target")), r.get("commit")])
    with open(os.path.join(root, "runs.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "variant", "scale_target", "tumours_planned", "tumours_grown", "reached_target",
                    "cpu_median_s", "cpu_min_s", "cpu_max_s", "peak_mb_max", "commits", "used_in"])
        w.writerows(runs)
    with open(os.path.join(root, "runs_tumours.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "task", "seed", "scale", "target_cancer", "cancer_reached", "time_generations",
                    "cpu_s", "wall_s", "peak_mb", "reached_target", "commit"])
        w.writerows(tumours)
    print("wrote", os.path.join(root, "runs.csv"), "and runs_tumours.csv:", len(runs), "runs,", len(tumours), "tumours")


if __name__ == "__main__":
    main()
