"""The paper-1 run registry: every simulation the figures read, defined in one place.

Each run is a named design (the showcase tumour, the dispersal sweep, ...) expanded into tasks, one
per (variant, seed). A task grows a paper-regime tumour (:mod:`regime`), pickles it to
``analysis_data/paper/<tier>/<run>/<task>.pkl`` and writes a manifest line (config, seed, wall time,
peak memory, size, whether it reached its target). Figures and analyses only ever read these files,
so a tumour is grown once however many panels use it, and the manifests become the supplementary
table of benchmark setups.

Usage::

    python validation/paper/registry.py list
    python validation/paper/registry.py run R1 [--jobs 3] [--force]
    python validation/paper/registry.py run all --jobs 3

``ISCC_PAPER_TIER`` (``preview`` or ``full``) selects the scale; see :mod:`regime`.
"""
import argparse
import json
import os
import pickle
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import regime as RG  # noqa: E402

REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(REPO, "analysis_data", "paper", RG.TIER)

T = RG.T
SEEDS = list(range(1, T["seeds"] + 1))
SWEEP, SHOW = T["sweep_scale"], T["showcase_scale"]


def _runs():
    """name -> list of (task_name, grow kwargs). Variants override the paper regime, nothing else."""
    runs = {}
    # R0: the showcase tumour, with its spatial history (one main seed plus replicates).
    runs["R0"] = [(f"seed{s}", dict(seed=s, scale=SHOW, target_cancer=T["showcase_target"],
                                    spatial_snapshot_every=2)) for s in (1, 2, 3)]
    # R1: dispersal sweep, clonal territories through to intermixing.
    runs["R1"] = [(f"disp{d}_seed{s}", dict(seed=s, scale=SWEEP, target_cancer=T["sweep_target"],
                                            spatial_snapshot_every=8, cancer={"dispersal_rate": d}))
                  for d in (0.1, 0.3, 0.9, 2.0, 4.0) for s in SEEDS]
    # R2: neutral against selected growth. Passenger coarsening OFF: the 1/f test needs every
    # passenger tracked exactly, and coarsening re-emits them as an approximate lineage overlay.
    runs["R2"] = [(f"{name}_seed{s}", dict(seed=s, scale=SWEEP, target_cancer=T["sweep_target"],
                                           selection=sel, coarsen=False))
                  # neutral keeps the same driver layout with no fitness effect, so oncogenic content
                  # per segment is identical and only selection differs (as validate_cna does)
                  for name, sel in (("selected", {}), ("neutral", {"driver_effects": 1.0}))
                  for s in SEEDS]
    # R3: tissue architecture x selection (the paper-regime counterpart of Harald's design).
    free = {"n_glands": 1, "structure_radius": 0, "origin_confinement": None}
    runs["R3"] = [(f"{arch}_{name}_seed{s}",
                   dict(seed=s, scale=SWEEP, target_cancer=T["sweep_target"], spatial_snapshot_every=8,
                        selection=sel, spatial=(free if arch == "free" else {})))
                  for arch in ("confined", "free")
                  for name, sel in (("selected", {}), ("neutral", {"prop_driver": 0.0}))
                  for s in SEEDS]
    # R4: copy-number burden sweep, with an SNV-burden control.
    runs["R4"] = ([(f"cnv{c}_seed{s}", dict(seed=s, scale=SWEEP, target_cancer=T["sweep_target"],
                                            cancer={"cnv_prob": c}))
                   for c in (0.0, 0.05, 0.15, 0.3) for s in SEEDS]
                  + [(f"snv{m}_seed{s}", dict(seed=s, scale=SWEEP, target_cancer=T["sweep_target"],
                                              cancer={"cnv_prob": 0.0, "n_snvs_per_allele": m}))
                     for m in (0.05, 0.2, 0.5) for s in SEEDS])
    # R5: whole-genome duplication, prevalence across rates.
    runs["R5"] = [(f"wgd{w}_seed{s}", dict(seed=s, scale=SWEEP, target_cancer=T["sweep_target"],
                                           cancer={"wgd_rate": w}))
                  for w in (0.0, 0.01, 0.03, 0.05, 0.1) for s in range(1, 2 * T["seeds"] + 1)]
    # R6: a cohort of patients sharing one configuration (private evolution seeds).
    runs["R6"] = [(f"patient{s}", dict(seed=100 + s, scale=SWEEP, target_cancer=T["sweep_target"]))
                  for s in range(1, 7)]
    # R10: compartment barriers on against off (the genotype-niche benchmark and Fig 3C).
    off = {"breach_gated_invasion": False, "stromal_hazard": 0.0}
    runs["R10"] = [(f"{name}_seed{s}", dict(seed=s, scale=SWEEP, target_cancer=T["sweep_target"],
                                            spatial=sp))
                   for name, sp in (("barriers", {}), ("open", off)) for s in SEEDS]
    return runs


RUNS = _runs()


def task_path(run, task):
    return os.path.join(OUT, run, f"{task}.pkl")


def _cpu_s():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def _grow_task(run, task, kw):
    from memguard import wait_for_memory
    wait_for_memory(label=f"{run}/{task}")
    t0, c0 = time.time(), _cpu_s()
    t = RG.grow(**kw)
    wall, cpu = time.time() - t0, _cpu_s() - c0
    path = task_path(run, task)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(t, f, protocol=pickle.HIGHEST_PROTOCOL)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2 if sys.platform == "darwin" else 1024)
    # CPU time, not wall time, is the cost: a laptop that sleeps mid-run inflates wall time tenfold.
    rec = dict(run=run, task=task, tier=RG.TIER, kwargs=kw, cpu_s=round(cpu, 1), wall_s=round(wall, 1),
               peak_mb=round(peak),
               cancer=int(t.get_cancer_size()), n_genotypes=len(t.genotypes), time=float(t.time),
               reached_target=bool(t.reached_target))
    with open(os.path.join(OUT, run, "manifest.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def load(run, task):
    """The pickled tumour of one task."""
    with open(task_path(run, task), "rb") as f:
        return pickle.load(f)


def tasks(run):
    return [name for name, _ in RUNS[run]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["list", "run"])
    ap.add_argument("run", nargs="?", default="all")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    names = list(RUNS) if a.run == "all" else a.run.split(",")
    if a.cmd == "list":
        for n in names:
            print(f"{n}: {len(RUNS[n])} tasks ({RG.TIER})")
        return
    todo = [(n, task, kw) for n in names for task, kw in RUNS[n]
            if a.force or not os.path.exists(task_path(n, task))]
    print(f"{len(todo)} tasks to grow ({RG.TIER}, {a.jobs} parallel)", flush=True)
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        futs = {ex.submit(_grow_task, *job): job for job in todo}
        for fut in as_completed(futs):
            n, task, _ = futs[fut]
            try:
                r = fut.result()
                print(f"  {n}/{task}: {r['cancer']} cells, {r['cpu_s']} s CPU, {r['peak_mb']} MB"
                      f"{'' if r['reached_target'] else '  (target not reached)'}", flush=True)
            except Exception as e:  # report and keep going: one failed seed must not hide the rest
                print(f"  !! {n}/{task} FAILED: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
