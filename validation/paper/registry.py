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
    python validation/paper/registry.py count R1,R3       # number of tasks, for a job array
    python validation/paper/registry.py task R1,R3 7      # grow task 7 of that list (0-based)

``ISCC_PAPER_TIER`` (``preview`` or ``full``) selects the scale; see :mod:`regime`.
``ISCC_PAPER_OUT`` moves ``analysis_data/paper`` elsewhere, for example to a cluster's project
storage; the tier is still appended. ``task`` is what a Slurm job array calls, one task per array
element (see ``slurm_registry.sh``).
"""
import argparse
import json
import os
import pickle
import resource
import sys
import time
import multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import regime as RG  # noqa: E402

REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(os.environ.get("ISCC_PAPER_OUT") or os.path.join(REPO, "analysis_data", "paper"),
                   RG.TIER)

T = RG.T
SEEDS = list(range(1, T["seeds"] + 1))
# The dispersal sweep. Under demon's division rule `dispersal_rate` is a per-division probability, so
# the sweep stays within (0, 1]; the old 2.0 and 4.0 arms ran at the cap of 1 (vault:
# decision-noble-division-law). 0.9 is the paper regime's own value. The low end is 0.05, not 0.03: at
# 0.03 two of five preview seeds never left the founding duct in 1,500 generations (Pedro, 2026-11-13).
R1_DISPERSAL = (0.05, 0.1, 0.3, 0.6, 0.9)
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
                  for d in R1_DISPERSAL for s in SEEDS]
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
                  # The SNV-burden control varies SNV burden at the sweep's lowest copy-number rate, not
                  # with copy number off: without copy-number change, invasion hangs on a rare breach
                  # hit surviving in the ducts, and at higher SNV burden half the tumours never invade
                  # (5 of 10 stayed in situ at ~8,000 cells for 1,500 generations). cnv0.0 is the
                  # copy-number-off reference.
                  + [(f"snv{m}_seed{s}", dict(seed=s, scale=SWEEP, target_cancer=T["sweep_target"],
                                              cancer={"cnv_prob": 0.05, "n_snvs_per_allele": m}))
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
    # R11: the human genome layer's tool figure (Fig 5) on an arm-homologous growth genome: one segment
    # per autosomal arm, 6,000 growth genes shared in proportion to each arm's genes, and 240 drivers
    # placed per arm from Cancer Gene Census content (driver_scale 0.4 keeps today's prop_driver 0.04
    # load). A copy-number event alters about one segment, so 1/39 of this genome against 1/12 of the
    # regime's; the copy-number events per division are scaled by 39/12, with the SNV and WGD events per
    # division held fixed (cnv_prob is a weight against snv_prob inside a mutating division, mutation_rate
    # the chance a division mutates at all, wgd_rate per mutating division). Calibrated on three to five
    # seeds per design (vault: exp-r11-arm-calibration). The designs are R0 seed1 and R1 at dispersal 0.9
    # and 0.1; at 0.1, seed 1 is swept by a genome doubling on this genome, so seed 2 is grown beside it.
    from iscc.data.human import arm_genome
    ag = arm_genome(total_genes=6000, driver_scale=0.4)
    f = 39 / 12
    p = 0.045 / (0.045 + 0.5)                      # the ER+ preset's copy-number share of a mutating division
    k = 1.0 - p + f * p
    arm_rates = {"cnv_prob": 0.045 * f, "mutation_rate": RG.CANCER["mutation_rate"] * k, "wgd_rate": 0.001 / k}
    arm = dict(genome=ag["genome_params"], selection=ag["selection_params"])
    runs["R11"] = ([("seed1", dict(seed=1, scale=SHOW, target_cancer=T["showcase_target"],
                                   spatial_snapshot_every=2, cancer=dict(arm_rates), **arm))]
                   + [(f"disp{d}_seed{s}", dict(seed=s, scale=SWEEP, target_cancer=T["sweep_target"],
                                                spatial_snapshot_every=8,
                                                cancer={**arm_rates, "dispersal_rate": d}, **arm))
                      for d, s in ((0.1, 1), (0.1, 2), (0.9, 1))])
    return runs


RUNS = _runs()


def task_path(run, task):
    return os.path.join(OUT, run, f"{task}.pkl")


def _commit():
    if os.environ.get("ISCC_COMMIT"):
        return os.environ["ISCC_COMMIT"]
    try:
        import subprocess
        return subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        p = os.path.join(REPO, "COMMIT")
        return open(p).read().strip() if os.path.exists(p) else None


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
    # `commit` is the iscc commit the tumour was grown from: a git checkout reports it, and a copy without
    # .git (a cluster rsync) reads it from ISCC_COMMIT or a COMMIT file at the repository root.
    rec = dict(run=run, task=task, tier=RG.TIER, commit=_commit(), kwargs=kw, cpu_s=round(cpu, 1),
               wall_s=round(wall, 1),
               peak_mb=round(peak),
               cancer=int(t.get_cancer_size()), n_genotypes=len(t.genotypes), time=float(t.time),
               reached_target=bool(t.reached_target))
    # One record per task beside its pickle, and the run's manifest. Array tasks finish in any order
    # and may race on the shared manifest, so the per-task file is the one to trust; `manifest`
    # rebuilds the shared file from them.
    with open(os.path.splitext(path)[0] + ".json", "w") as f:
        json.dump(rec, f)
    with open(os.path.join(OUT, run, "manifest.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def _grow_safe(job):
    try:
        return job, _grow_task(*job), None
    except Exception as e:
        return job, None, f"{type(e).__name__}: {e}"


def load(run, task):
    """The pickled tumour of one task, with the paper regime's current O2 field.

    The microenvironment is readout-only (it picks its target genes once, at construction, and is
    applied when cell data are materialised), so a stored tumour takes the regime's O2 settings
    without regrowing."""
    with open(task_path(run, task), "rb") as f:
        t = pickle.load(f)
    hyp = (t.microenv_params or {}).get("hypoxia") if isinstance(t.microenv_params, dict) else None
    if hyp is not None:
        hyp.update(RG.O2)
    # Two more readout settings that can post-date a stored tumour: chunked materialisation of the
    # program multiplier, and the bound on a program's log-fold (vault: decision-chunked-materialisation,
    # decision-program-log-fold-bound). Neither touches growth, so they apply without regrowing; on code
    # that predates them the attributes are simply unused.
    if not hasattr(t, "low_memory"):
        t.low_memory = True
    if getattr(t, "programs", None) is not None:
        t.programs.max_log_fold = RG.PROGRAM_MAX_LOG_FOLD
        _apply_dosage(t)
    return t


def _apply_dosage(t):
    """Give a stored tumour the regime's per-gene dosage (``regime.DOSAGE``, calibrated on TCGA breast
    tumours; vault: decision-dosage-total-law). Dosage is a readout drawn from its own sub-stream of the
    layout seed when the program model is built, so a fresh model with the regime's settings carries
    exactly the dosage a tumour grown today would; only its sensitivities and law are copied over, and
    the rest of the stored program model is kept."""
    dosage = getattr(RG, "DOSAGE", None)
    ep = t.expression_params
    if not dosage or not isinstance(ep, dict):
        return
    import copy
    from iscc.tumor.programs import ProgramModel
    ep = copy.deepcopy(ep)
    ep.setdefault("dosage_params", {}).update(dosage)
    fresh = ProgramModel(n_genes=t.n_genes, segment_sizes=t.selection.segment_sizes,
                         layout_seed=t.layout_seed, run_seed=t.seed, expression_params=ep)
    t.programs.dosage_sensitivity = fresh.dosage_sensitivity
    t.programs.dosage_law = getattr(fresh, "dosage_law", "haplotype")
    t.expression_params = ep


def tasks(run):
    return [name for name, _ in RUNS[run]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["list", "run", "count", "task", "manifest"])
    ap.add_argument("run", nargs="?", default="all")
    ap.add_argument("index", nargs="?", type=int, help="task index for `task` (0-based)")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    names = list(RUNS) if a.run == "all" else a.run.split(",")
    if a.cmd == "list":
        for n in names:
            print(f"{n}: {len(RUNS[n])} tasks ({RG.TIER})")
        return
    flat = [(n, task, kw) for n in names for task, kw in RUNS[n]]
    if a.cmd == "count":
        print(len(flat))
        return
    if a.cmd == "task":
        n, task, kw = flat[a.index]
        if not a.force and os.path.exists(task_path(n, task)):
            print(f"  {n}/{task}: already grown", flush=True)
            return
        r = _grow_task(n, task, kw)
        print(f"  {n}/{task}: {r['cancer']} cells, {r['cpu_s']} s CPU, {r['peak_mb']} MB"
              f"{'' if r['reached_target'] else '  (target not reached)'}", flush=True)
        return
    if a.cmd == "manifest":
        for n in names:
            recs = []
            for task, _ in RUNS[n]:
                p = os.path.splitext(task_path(n, task))[0] + ".json"
                if os.path.exists(p):
                    with open(p) as f:
                        recs.append(json.load(f))
            if recs:
                with open(os.path.join(OUT, n, "manifest.jsonl"), "w") as f:
                    f.writelines(json.dumps(r) + "\n" for r in recs)
            print(f"{n}: {len(recs)} of {len(RUNS[n])} tasks grown")
        return
    todo = [(n, task, kw) for n, task, kw in flat if a.force or not os.path.exists(task_path(n, task))]
    print(f"{len(todo)} tasks to grow ({RG.TIER}, {a.jobs} parallel)", flush=True)
    # a fresh process per tumour, so one large genotype registry is not carried into the next task
    with mp.get_context("spawn").Pool(processes=a.jobs, maxtasksperchild=1) as pool:
        for job, r, err in pool.imap_unordered(_grow_safe, todo):
            n, task, _ = job
            if err:                 # report and keep going: one failed seed must not hide the rest
                print(f"  !! {n}/{task} FAILED: {err}", flush=True)
            else:
                print(f"  {n}/{task}: {r['cancer']} cells, {r['cpu_s']} s CPU, {r['peak_mb']} MB"
                      f"{'' if r['reached_target'] else '  (target not reached)'}", flush=True)


if __name__ == "__main__":
    main()
