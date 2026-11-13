"""iscc with the tissue off against demon, at matched settings (the demon match).

The paper builds iscc's cell-level law on demon (Noble et al. 2022): division, death with crowding,
and invasion-mode dispersal. This script grows the same tumour in both and writes, per run, the
growth curve and Noble's evolutionary-mode indices (n, D, J1) over driver-mutation clones, so the
supplementary figure (``supp_demon_match.py``) can compare them.

Matched settings (demon's names first):

* deme carrying capacity K = 16, no normal cells, no tissue structure;
* initial birth rate 1, baseline death rate 0; above K, death 0 + 100 (demon's hard-coded
  ``density_dept_death_rate``) -> iscc ``crowding_mode="lottery"``, ``crowding_law="noble"``,
  ``crowding_d1=100`` with ``crowding_overfill=4`` so the step above K is reached;
* invasion mode (``migration_type 0``): after a division one daughter moves to one of the four
  neighbouring demes with probability 0.1 -> iscc ``dispersal_law="noble"``, ``dispersal_rate=0.1``;
* driver mutations 1e-3 per daughter (2e-3 per division) -> iscc ``mutation_rate`` 0.004 with one
  SNV per mutating division (iscc forces one SNV when every per-allele Poisson draw is zero, so a
  near-zero ``n_snvs_per_allele`` makes it exactly one) x driver share 0.5, on a 48,000-site genome
  so that independent lineages almost never hit the same driver site (iscc's driver sites are
  finite, demon's are not);
* driver effect: demon multiplies the birth rate by 1 + s·Exp(1) (no cap: ``max_relative_birth_rate``
  -1), iscc by 1 + s (CINner's dosage law with copy number off), s = 0 (neutral) or 0.1;
* both stop at 20,000 cancer cells.

Usage (needs the engine options of ``dispersal_law``, iscc PR #12)::

    python demon_match.py demon <neutral|selection> <seed> <outdir> --demon /path/to/demon
    python demon_match.py iscc  <neutral|selection|realised> <seed> <outdir>
    python demon_match.py collect <outdir> <out.csv>
"""
import argparse
import glob
import json
import os
import subprocess
import time

import numpy as np

K = 16
MIGRATION = 0.1
MU_DRIVER_PER_DAUGHTER = 1e-3
N_FINAL = 20_000
# "realised" (iscc only) fixes the effect at demon's realised per-driver effect under selection,
# about 0.2: demon draws 1 + 0.1·Exp(1), and the drivers that establish are the larger ones.
S = {"neutral": 0.0, "selection": 0.1, "realised": 0.2}

_DEMON_KEYS = [
    ("spatial_structure", ["log2_deme_carrying_capacity {log2k}"]),
    ("dispersal", ["migration_type 0", "init_migration_rate {mig}", "migration_edge_only 0",
                   "migration_rate_scales_with_K 0"]),
    ("mutation_rates", ["mu_driver_birth {mu}", "mu_passenger 0", "mu_driver_migration 0",
                        "passenger_pop_threshold -1"]),
    ("fitness_effects", ["normal_birth_rate -1", "baseline_death_rate 0", "s_driver_birth {s}",
                         "s_passenger 0", "s_driver_migration 0", "max_relative_birth_rate -1",
                         "max_relative_migration_rate -1"]),
    ("non_biological_parameters", ["init_pop 1", "matrix_max -1", "max_pop {n}", "max_time 86400",
                                   "max_generations 100000", "seed {seed}", "write_grid 0",
                                   "write_clones_file 0", "write_demes_file 0", "record_matrix 0",
                                   "write_phylo 0", "calculate_total_diversity 0",
                                   "biopsy_size_per_sample 100"]),
]
# demon reads its configuration with Boost's INFO parser: a top-level `parameters` key, then one
# key and value per line.
DEMON_CONFIG = "parameters\n{{\n" + "".join(f"    {sec}\n    {{{{\n" + "".join(f"        {kv}\n" for kv in kvs)
                                + "    }}\n" for sec, kvs in _DEMON_KEYS) + "}}\n"


def _indices_from_tree(parents, sizes, n_drivers):
    """Noble's (n, D, J1) from a driver-clone tree: parent map, populations, drivers per clone."""
    from iscc.inference.indices import inverse_simpson, tree_balance_j1
    pops = {k: v for k, v in sizes.items() if v > 0}
    total = sum(pops.values())
    n = sum(n_drivers[k] * v for k, v in pops.items()) / total
    return dict(n=float(n), D=inverse_simpson(pops.values()), J1=tree_balance_j1(parents, sizes),
                n_clones=len(pops))


def run_demon(cond, seed, outdir, demon):
    d = os.path.join(outdir, f"demon_{cond}_seed{seed}")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "config.dat"), "w") as f:
        f.write(DEMON_CONFIG.format(log2k=int(np.log2(K)), mig=MIGRATION, mu=MU_DRIVER_PER_DAUGHTER,
                                    s=S[cond], n=N_FINAL, seed=seed))
    t0 = time.time()
    subprocess.run([demon, d + "/", "config.dat"], check=True, stdout=subprocess.DEVNULL)
    wall = time.time() - t0
    import pandas as pd
    pops = pd.read_csv(os.path.join(d, "output.dat"), sep="\t")
    props = pd.read_csv(os.path.join(d, "output_driver_genotype_properties.dat"), sep="\t")
    parents = {int(r.Identity): int(r.Parent) for r in props.itertuples() if int(r.Identity) != int(r.Parent)
               and int(r.Parent) >= 0}
    sizes = {int(r.Identity): float(r.Population) for r in props.itertuples()}
    ndrv = {int(r.Identity): int(r.DriverMutations) for r in props.itertuples()}
    rec = dict(model="demon", cond=cond, seed=seed, wall_s=wall, **_indices_from_tree(parents, sizes, ndrv),
               cells=int(pops.NumCells.iloc[-1]), generations=float(pops.Generation.iloc[-1]),
               clone_sizes=sorted((int(v) for v in sizes.values() if v > 0), reverse=True),
               curve=dict(t=pops.Generation.tolist(), cells=pops.NumCells.tolist(),
                          demes=pops["NumNon-EmptyDemes"].tolist()))
    with open(os.path.join(outdir, f"demon_{cond}_seed{seed}.json"), "w") as f:
        json.dump(rec, f)
    return rec


def iscc_tumour(cond, seed):
    from iscc.tumor.models import GenotypeTumor
    genome = {"n_segments": 24, "segment_size": 2000}
    selection = {"prop_driver": 0.5, "prop_dispersal": 0.0, "prop_immune_resistance": 0.0,
                 "prop_treatment_resistance": 0.0, "driver_effects": 1.0 + S[cond],
                 "dispersal_effects": 1.0, "treatment_resistant_effects": 1.0,
                 "immune_resistant_effects": 1.0}
    # One SNV per mutating division: with a near-zero per-allele rate every Poisson draw is zero and
    # the engine places exactly one (its fallback), so drivers arrive at mutation_rate x 0.5.
    cancer = {"division_rate": 1.0, "death_rate": 0.0, "max_birth_rate": 10.0,
              "mutation_rate": 2 * 2 * MU_DRIVER_PER_DAUGHTER, "dispersal_rate": MIGRATION,
              "snv_prob": 1.0, "cnv_prob": 0.0, "wgd_rate": 0.0, "n_snvs_per_allele": 1e-9}
    deme = {"carrying_capacity": K, "initial_cancer_cells": 1, "crowding_mode": "lottery",
            "crowding_law": "noble", "crowding_d1": 100.0, "maximum_death_rate": 100.0,
            "crowding_overfill": 4.0, "dispersal_law": "noble"}
    spatial = {"grid_size": 81, "structure_radius": 0, "stroma_fill_frac": 0.0}
    return GenotypeTumor(seed=seed, genome_params=genome, selection_params=selection,
                         cancer_cell_params=cancer, deme_params=deme, spatial_params=spatial,
                         update_mode="exact", coarsen_passengers=True, rng_streams="independent",
                         snapshot_every=10**12)   # the exact engine fires millions of events; no per-event trace


def run_iscc(cond, seed, outdir, record_every=0.25, n_final=N_FINAL):
    from iscc.inference.indices import driver_combination_counts, mode_indices
    t = iscc_tumour(cond, seed)
    # The exact engine fires one event per step and keeps no clock. Each step's expected waiting
    # time is 1 / (total event rate), so wrapping the per-event update with that increment gives the
    # tumour's age in generations without touching the trajectory (it reads, never draws).
    clock = [0.0]
    update = t.update

    def timed_update(rng):
        rate = float(t.deme_rates.sum())
        if rate > 0:
            clock[0] += 1.0 / rate
        update(rng)

    t.update = timed_update
    t0 = time.time()
    curve = dict(t=[0.0], cells=[t.get_cancer_size()], demes=[1])
    while t.get_cancer_size() < n_final and not t.is_extinct():
        # grow(k) fires k - 1 events; a chunk of about `record_every` generations per curve point
        steps = max(100, int(float(t.deme_rates.sum()) * record_every))
        t.grow(steps + 1, seed=seed, materialize=False)
        curve["t"].append(clock[0])
        curve["cells"].append(int(t.get_cancer_size()))
        curve["demes"].append(int(sum(1 for d in t.demes if any(t._is_cancer(g) for g in d))))
    clock = clock[0]
    wall = time.time() - t0
    m = mode_indices(t)
    sizes = sorted(driver_combination_counts(t).values(), reverse=True)
    rec = dict(model="iscc", cond=cond, seed=seed, wall_s=wall, n=m["n"], D=m["D"], J1=m["J1"],
               n_clones=m["n_clones"], cells=int(t.get_cancer_size()), generations=clock,
               clone_sizes=[int(v) for v in sizes], curve=curve,
               normal_cells=int(sum(c for d in t.demes for g, c in d.items() if not t._is_cancer(g))))
    with open(os.path.join(outdir, f"iscc_{cond}_seed{seed}.json"), "w") as f:
        json.dump(rec, f)
    return rec


def collect(outdir, out_csv):
    """One row per run (indices, size, time) plus a long table of clone sizes, for the figure."""
    import pandas as pd
    recs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(outdir, "*.json")))]
    keep = ["model", "cond", "seed", "n", "D", "J1", "n_clones", "cells", "generations", "wall_s"]
    pd.DataFrame([{k: r[k] for k in keep} for r in recs]).to_csv(out_csv, index=False)
    base = os.path.splitext(out_csv)[0]
    pd.DataFrame([dict(model=r["model"], cond=r["cond"], seed=r["seed"], size=s)
                  for r in recs for s in r["clone_sizes"]]).to_csv(base + "_clones.csv", index=False)
    pd.DataFrame([dict(model=r["model"], cond=r["cond"], seed=r["seed"], t=t, cells=c, demes=d)
                  for r in recs for t, c, d in zip(r["curve"]["t"], r["curve"]["cells"], r["curve"]["demes"])]
                 ).to_csv(base + "_curves.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["demon", "iscc", "collect"])
    ap.add_argument("args", nargs="+")
    ap.add_argument("--demon", default="demon")
    ap.add_argument("--n-final", type=int, default=N_FINAL, help="smaller size, for a quick check (iscc)")
    a = ap.parse_args()
    if a.cmd == "collect":
        collect(*a.args)
        return
    cond, seed, outdir = a.args[0], int(a.args[1]), a.args[2]
    os.makedirs(outdir, exist_ok=True)
    rec = (run_demon(cond, seed, outdir, a.demon) if a.cmd == "demon"
           else run_iscc(cond, seed, outdir, n_final=a.n_final))
    print(json.dumps({k: rec[k] for k in ("model", "cond", "seed", "n", "D", "J1", "cells",
                                           "generations", "wall_s")}))


if __name__ == "__main__":
    main()
