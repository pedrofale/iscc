"""Calibrate the copy-number event rate of the arm-homologous growth genome (R11) against the paper
regime's 12 segments.

A copy-number event picks its segment in proportion to the copies on it, so near-uniformly by segment
at diploid, whatever the segment's length: on n segments an event alters about 1/n of the genome. On 39
arms each event therefore alters 12/39 as much as on 12 segments, and the fraction of genome altered
falls. ``cnv_prob`` is a relative weight against ``snv_prob`` inside a mutating division, and
``mutation_rate`` is the probability that a division mutates at all (WGD fires per mutating division),
so scaling the copy-number events per division by ``f`` while holding the SNV and WGD events per
division fixed is::

    cnv_prob      = 0.045 * f                          (ER+ preset 0.045, snv_prob 0.5)
    mutation_rate = 0.15 * (1 - p + f * p)             (p = 0.045 / 0.545, the copy-number share)
    wgd_rate      = 0.001 / (1 - p + f * p)

``f = 39 / 12`` restores the genome altered per division, to first order.

Usage (one task per Slurm array element; ``collect`` writes the CSV)::

    ISCC_PAPER_TIER=full PYTHONPATH=src python validation/paper/tools/calibrate_arm_genome.py count
    ... calibrate_arm_genome.py task <i>        # grows one (design, seed, factor), or scores one stored
                                                # linear-regime tumour at the same design and seed
    ... calibrate_arm_genome.py collect

Writes ``$ISCC_PAPER_OUT/<tier>/calibration-arm/<label>.json`` and ``calibration-arm.csv`` there. Per
tumour: size, time, the in-situ phase (first cancer cell in a stromal deme minus the acinus release,
from snapshots every generation), the share of breach carriers among the stromal cancer cells at that
snapshot, and the regime tuning's summaries (fraction of genome altered, ploidy, WGD share, clones).
"""
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import regime as RG  # noqa: E402
import registry as REG  # noqa: E402
import sweep_score as SS  # noqa: E402

P_CNV = 0.045 / (0.045 + 0.5)
FACTORS = (2.5, 39 / 12, 4.0)
SEEDS = (1, 2, 3)
DESIGNS = {  # name: (scale, target, dispersal); R0's design is the canonical dispersal of the config
    "R0": ("cm", RG.T["showcase_target"], None),
    "R1_disp0.9": ("mid", RG.T["sweep_target"], 0.9),
    "R1_disp0.1": ("mid", RG.T["sweep_target"], 0.1),
}
OUT = os.path.join(REG.OUT, "calibration-arm")


def rates(f):
    k = 1.0 - P_CNV + f * P_CNV
    return {"cnv_prob": 0.045 * f, "mutation_rate": 0.15 * k, "wgd_rate": 0.001 / k}


def _tasks():
    out = []
    for d in DESIGNS:
        for s in SEEDS:
            out.append(("linear", d, s, None))
            for f in FACTORS:
                out.append(("arm", d, s, f))
    return out


def metrics(t):
    rel = [e["time"] for e in t.events if e.get("event") == "origin_release"]
    release = float(rel[0]) if rel else np.nan
    first, breach = np.nan, np.nan
    snap = t.spatial_snapshots() if t.gland_id is not None else pd.DataFrame()
    if len(snap):
        st = snap[(snap["cell_type"] == "cancer") & (t.gland_id[snap["deme"].to_numpy(int)] == -1)]
        if len(st):
            first = float(st["time"].min())
            at = st[st["time"] == first]
            known = [(g, c) for g, c in zip(at["genotype"], at["count"]) if g in t.genotypes]
            n = sum(c for _, c in known)
            nb = sum(c for g, c in known if t.genotypes[g].evolutionary_parameters.get("breach", 0.0) > 0)
            breach = nb / n if n else np.nan
    sc = SS.score_tumour(t)
    return dict(cancer=int(t.get_cancer_size()), time=float(t.time), reached_target=bool(t.reached_target),
                release_time=release, first_stroma_time=first, in_situ=first - release,
                breach_frac_at_invasion=breach,
                **{k: sc.get(k) for k in ("frac_genome_altered", "mean_ploidy", "pct_wgd", "n_clones",
                                          "max_cna_freq", "stroma_pct")})


def run_task(i):
    kind, d, s, f = _tasks()[i]
    scale, target, disp = DESIGNS[d]
    label = f"{kind}_{d}_seed{s}" + ("" if f is None else f"_f{f:.2f}")
    t0 = time.time()
    if kind == "linear":                          # the stored paper-regime tumour at this design and seed
        task = f"seed{s}" if d == "R0" else f"disp{disp}_seed{s}"
        t = REG.load("R0" if d == "R0" else "R1", task)
        rec = dict(kind=kind, design=d, seed=s, factor=np.nan, **metrics(t))
    else:
        from iscc.data.human import arm_genome
        ag = arm_genome(total_genes=6000, driver_scale=0.4)
        cancer = rates(f)
        if disp is not None:
            cancer["dispersal_rate"] = disp
        t = RG.grow(seed=s, scale=scale, target_cancer=target, spatial_snapshot_every=2,
                    genome=ag["genome_params"], selection=ag["selection_params"], cancer=cancer)
        rec = dict(kind=kind, design=d, seed=s, factor=f, **cancer, **metrics(t))
    rec["wall_s"] = round(time.time() - t0, 1)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, label + ".json"), "w") as fh:
        json.dump(rec, fh, default=float)
    print(json.dumps(rec, default=float), flush=True)


def collect():
    rows = []
    for p in sorted(glob.glob(os.path.join(OUT, "*.json"))):
        with open(p) as fh:
            rows.append(json.load(fh))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "calibration-arm.csv"), index=False)
    cols = ["frac_genome_altered", "in_situ", "breach_frac_at_invasion", "pct_wgd", "mean_ploidy", "reached_target"]
    print(df.groupby(["design", "kind", "factor"], dropna=False)[cols].agg(["mean", "min", "max"]).round(3).to_string())


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "count":
        print(len(_tasks()))
    elif cmd == "task":
        run_task(int(sys.argv[2]))
    else:
        collect()
