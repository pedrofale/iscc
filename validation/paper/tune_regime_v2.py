"""Retune the paper regime under demon's division and crowding laws (vault: exp-regime-v2-tuning).

The criteria, the grid and the rule that picks the setting were registered in the vault note before
any run. Each task grows one tumour on the small preview field and records what the criteria need:
the release of the founding acinus, the first cancer cell in the stroma and whether it carries a
breach mutation, deme occupancy against capacity, wall time, and at the end the calibration and
evolutionary-mode summaries. ``v1`` is the old law (the fate split, ``crowding_mode="own"``, offset
streams, a step-counting clock at 400), grown in the same harness at the same seeds.

Usage (one task per Slurm array element)::

    python validation/paper/tune_regime_v2.py count grid
    python validation/paper/tune_regime_v2.py task grid 7
    python validation/paper/tune_regime_v2.py count sweep --overfill 1.5 --d1 5
    python validation/paper/tune_regime_v2.py task sweep 3 --overfill 1.5 --d1 5
    python validation/paper/tune_regime_v2.py collect        # -> <out>/regime-v2-tuning.csv

Outputs go to ``$ISCC_PAPER_OUT/tuning-v2`` (or ``analysis_data/paper/tuning-v2``).
"""
import argparse
import copy
import csv
import glob
import json
import os
import resource
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import regime as RG  # noqa: E402
import registry as REG  # noqa: E402
import sweep_score as SS  # noqa: E402

OUT = os.path.join(os.environ.get("ISCC_PAPER_OUT") or os.path.join(REG.REPO, "analysis_data", "paper"),
                   "tuning-v2")
SEEDS = (1, 2, 3)
SCALE, TARGET, CHUNK = "small", 15_000, 4      # 4 tau steps = 2 generations between checks
DETECTION = 0.05                                # Fig 2a's detection limit (summaries.DETECTION)
OVERFILLS, D1S = (1.1, 1.5, 2.0), (2.2, 5.0, 20.0)
# Second stage, run only because no grid point kept the fraction of genome altered (the registered
# rule): the two points that met every other criterion, at lower mutation rates.
MU_POINTS, MUS = ((1.5, 5.0), (2.0, 5.0)), (0.25, 0.2, 0.15)
CANONICAL_DISPERSAL = 0.9

# The old law, exactly as the paper regime ran it before v2.
V1 = dict(
    deme={"dispersal_law": "compete", "crowding_mode": "own", "crowding_law": "logistic",
          "crowding_overfill": 1.0, "crowding_d1": None, "maximum_death_rate": 2.2},
    spatial={"origin_confinement": dict(RG.ORIGIN_CONFINEMENT, generations=400, clock="steps")},
    rng_streams="offset",
    mutation_rate=0.30,                         # the ER+ preset, as before v2
)


def v2(overfill, d1, mu=None):
    """The new laws at one grid point. The first grid ran at the ER+ preset's mutation rate (0.30),
    so that is the default here even though the regime itself now uses 0.15."""
    return dict(deme={"crowding_overfill": overfill, "crowding_d1": d1, "maximum_death_rate": d1},
                mutation_rate=0.30 if mu is None else mu)


def tasks(stage, overfill=None, d1=None, mu=None):
    """(label, law kwargs, dispersal, seed) for every tumour of a stage."""
    if stage == "grid":
        pts = [("v1", V1)] + [(f"v2_o{o:g}_d{d:g}", v2(o, d)) for o in OVERFILLS for d in D1S]
        return [(lab, kw, CANONICAL_DISPERSAL, s) for lab, kw in pts for s in SEEDS]
    if stage == "mu":
        pts = [(f"v2_o{o:g}_d{d:g}_mu{m:g}", v2(o, d, m)) for o, d in MU_POINTS for m in MUS]
        return [(lab, kw, CANONICAL_DISPERSAL, s) for lab, kw in pts for s in SEEDS]
    if stage == "sweep":
        pts = [("v1", V1), (f"v2_o{overfill:g}_d{d1:g}" + (f"_mu{mu:g}" if mu else ""),
                            v2(overfill, d1, mu))]
        return [(lab, kw, d, s) for lab, kw in pts for d in REG.R1_DISPERSAL for s in SEEDS]
    raise ValueError(stage)


def _occupancy(t):
    occ = []
    for di, deme in enumerate(t.demes):
        n = sum(deme.values())
        if n:
            occ.append(n / float(t._cap_of(di)))
    return np.asarray(occ) if occ else np.zeros(1)


def _stroma(t):
    n = nb = 0
    for di, deme in enumerate(t.demes):
        if t.gland_id[di] != -1:
            continue
        for g, c in deme.items():
            if t._is_cancer(g):
                n += c
                if t.genotypes[g].evolutionary_parameters.get("breach", 0.0) > 0:
                    nb += c
    return n, nb


def run_task(stage, i, overfill=None, d1=None, mu=None):
    from iscc.inference import mode_indices
    label, law, disp, seed = tasks(stage, overfill, d1, mu)[i]
    law = copy.deepcopy(law)
    cancer = {"dispersal_rate": disp}
    if "mutation_rate" in law:
        cancer["mutation_rate"] = law.pop("mutation_rate")
    mon = dict(first_stroma_time=None, breach_frac_at_invasion=None, max_occ=0.0, chunks_over_k=0,
               n_chunks=0)

    def on_chunk(t):
        mon["n_chunks"] += 1
        occ = _occupancy(t)
        mon["max_occ"] = max(mon["max_occ"], float(occ.max()))
        mon["chunks_over_k"] += int((occ > 1.0 + 1e-9).any())
        if mon["first_stroma_time"] is None:
            n, nb = _stroma(t)
            if n > 0:
                mon["first_stroma_time"] = float(t.time)
                mon["breach_frac_at_invasion"] = nb / n

    t0, c0 = time.time(), resource.getrusage(resource.RUSAGE_SELF).ru_utime
    t = RG.grow(seed=seed, scale=SCALE, target_cancer=TARGET, chunk=CHUNK, on_chunk=on_chunk,
                cancer=cancer, **law)
    wall, cpu = time.time() - t0, resource.getrusage(resource.RUSAGE_SELF).ru_utime - c0
    rel = [e["time"] for e in t.events if e.get("event") == "origin_release"]
    release = float(rel[0]) if rel else None
    occ = _occupancy(t)
    full = occ[occ >= 1.0 - 1e-9]
    sc = SS.score_tumour(t)
    m0, md = mode_indices(t, detection=0.0), mode_indices(t, detection=DETECTION)
    rec = dict(stage=stage, label=label, dispersal=disp, seed=seed, mutation_rate=cancer.get("mutation_rate"),
               wall_s=round(wall, 1),
               cpu_s=round(cpu, 1), cancer=int(t.get_cancer_size()), reached_target=bool(t.reached_target),
               time=float(t.time), release_time=release, first_stroma_time=mon["first_stroma_time"],
               in_situ=(mon["first_stroma_time"] - release
                        if release is not None and mon["first_stroma_time"] is not None else None),
               breach_frac_at_invasion=mon["breach_frac_at_invasion"], max_occ=mon["max_occ"],
               frac_chunks_over_k=mon["chunks_over_k"] / max(mon["n_chunks"], 1),
               mean_occ_full_end=float(full.mean()) if full.size else None,
               n_genotypes=len(t.genotypes),
               **{k: sc.get(k) for k in ("frac_genome_altered", "mean_ploidy", "pct_wgd", "n_clones",
                                         "max_cna_freq", "n_truncal_segments", "stroma_pct", "flags")},
               n0=m0["n"], D0=m0["D"], J1=m0["J1"], nd=md["n"], Dd=md["D"])
    os.makedirs(os.path.join(OUT, stage), exist_ok=True)
    with open(os.path.join(OUT, stage, f"{label}_disp{disp:g}_seed{seed}.json"), "w") as f:
        json.dump(rec, f)
    print(json.dumps(rec), flush=True)


def collect():
    rows = []
    for p in sorted(glob.glob(os.path.join(OUT, "*", "*.json"))):
        with open(p) as f:
            rows.append(json.load(f))
    if not rows:
        print("nothing to collect")
        return
    keys = []
    for r in rows:                       # stages add columns (mutation_rate), so take the union
        keys += [k for k in r if k not in keys]
    path = os.path.join(OUT, "regime-v2-tuning.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval="")
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["count", "task", "collect"])
    ap.add_argument("stage", nargs="?", default="grid")
    ap.add_argument("index", nargs="?", type=int)
    ap.add_argument("--overfill", type=float)
    ap.add_argument("--d1", type=float)
    ap.add_argument("--mu", type=float)
    a = ap.parse_args()
    if a.cmd == "collect":
        return collect()
    if a.cmd == "count":
        print(len(tasks(a.stage, a.overfill, a.d1, a.mu)))
        return
    run_task(a.stage, a.index, a.overfill, a.d1, a.mu)


if __name__ == "__main__":
    main()
