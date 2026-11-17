"""iscc with the tissue off against demon, at matched settings (the demon match and clonal territories).

The paper builds iscc's cell-level law on demon (Noble et al. 2022): division, death with crowding,
and invasion-mode dispersal. This script grows the same tumour in both and writes, per run, the
growth curve, Noble's evolutionary-mode indices (n, D, J1) over driver-mutation clones, the
driver-clone sizes, and the clonal territories (how often neighbouring demes share their dominant
clone), so the supplementary figure (``supp_demon_match.py``) can compare them.

iscc runs on either of its two engines, both on the code after the standard-algorithms change
(vault: decision-standard-simulation-algorithms, decision-paper-regime-v3):

* ``exact``: Gillespie's direct method in demon's hierarchical form, the algorithm demon uses, with
  the drawn exponential waiting times as its clock;
* ``tau``: the paper's engine, binomial tau-leaping with the Cao-Gillespie-Petzold (2006) step at
  epsilon 0.01 (``tau_deaths`` binomial, ``tau_step`` cgp, ``tau_epsilon`` 0.01), recorded every 0.5
  generations.

Matched settings (demon's names first):

* deme carrying capacity K = 16, no normal cells, no tissue structure (``structure_radius`` 0,
  ``stroma_fill_frac`` 0), one founding cell;
* initial birth rate 1, baseline death rate 0; above K, death 0 + 100 (demon's hard-coded
  ``density_dept_death_rate``) -> iscc ``crowding_d1`` = ``maximum_death_rate`` = 100, with
  ``crowding_overfill`` 4 so that a deme can rise above K and meet the step (iscc's lottery crowding
  with demon's two-valued death step is the engine's only law);
* invasion mode (``migration_type 0``): after a division one daughter moves to one of the four
  neighbouring demes with probability 0.1 -> iscc ``dispersal_rate`` 0.1 (demon's invasion rule is
  the engine's only dispersal law); the territory runs vary this rate in both;
* driver mutations: demon draws Poisson(1e-3) drivers for each of the two daughters, 2e-3 per
  division. iscc mutates one daughter: every division is a mutating one (``mutation_rate`` 1, SNVs
  only) and adds Poisson(lambda) point mutations to each of the genome's 48 allele copies, half of
  the sites being drivers, so a division adds Poisson(24 lambda) drivers; lambda = 2e-3 / 24 gives
  demon's 2e-3 per division. (The old match forced exactly one SNV per mutating division at
  ``mutation_rate`` 0.004; ``force_snv`` is gone, and a Poisson count of zero now leaves the genome
  unchanged, as in demon and CINner.) The passenger half of the sites is folded into the parent
  clone (``coarsen_passengers``), so it does not create clones;
* driver effect: demon multiplies the birth rate by 1 + s·Exp(1) per driver (no cap:
  ``max_relative_birth_rate`` -1), iscc by 1 + s (CINner's dosage law with copy number off, capped at
  ``max_birth_rate`` 10, which no run reaches); s = 0 (neutral) or 0.1, and, iscc only, 0.2
  ("realised", about demon's realised per-driver effect under selection, since the larger draws are
  the ones that establish);
* both stop at the first event (exact, demon) or leap (tau) at which the tumour reaches 20,000
  cancer cells (``grow(..., target_cancer=)``).

What cannot be matched: iscc mutates one daughter where demon mutates both (the per-division rate of
new drivers is matched, and two drivers in one division have probability ~1e-6 in both); iscc's
driver sites are finite (24,000 on a 48,000-site genome, so independent lineages almost never hit the
same one); iscc's driver effect is fixed where demon's is drawn; demon's grid grows with the tumour
where iscc's is a fixed 81 x 81 lattice with reflecting edges (the 20,000-cell tumours stay clear of
the edges).

Clonal territories, measured identically for both: a clone is a clade of the driver-genotype tree
holding at least 4% of the cancer cells (as ``common.Clones`` defines it for the paper's figures,
with ``iscc.integrations.clones._clade_roots``), every occupied deme takes its dominant clone
("ancestral" when its cells sit above every clade), and the territory score is the fraction of
neighbouring (von Neumann) occupied deme pairs that share it, over its expectation if the same
dominant clones were placed on the occupied demes at random (the sum of squared deme shares).

The territory sweep runs both at dispersal 0.05, 0.1, 0.3, 0.6 and 0.9 (the paper's R1 values), with
neutral drivers ten times as frequent (``neutral --mu 1e-2``: 1e-2 per daughter in demon, 2e-2 per
division in iscc), so that a 20,000-cell tumour has enough clones above 4% to draw territories (at
the match's rate most have one to three). The marks are neutral for two reasons: without selection
the territories read out dispersal alone, and under selection at that rate demon stops the run
(``Birth rate exceeds death rate``) once a clone's birth rate, multiplied by an unbounded
1 + s·Exp(1) per driver, reaches its crowding death rate d0 + 100, which 8 of its 25 runs at s = 0.1
did before 20,000 cells.

Usage::

    python demon_match.py demon <neutral|selection> <seed> <outdir> --demon /path/to/demon [--migration m] [--mu u] [--map]
    python demon_match.py iscc  <neutral|selection|realised> <seed> <outdir> [--engine exact|tau] [--migration m] [--mu u] [--map]
    python demon_match.py collect <out.csv> <runs.jsonl> [<runs.jsonl> ...]

Each run appends one JSON line to ``<outdir>/runs.jsonl`` (under a file lock, so parallel runs on one
node can share it); ``collect`` turns such files into the per-run CSV, plus ``_clones``, ``_curves``
and ``_maps`` CSVs beside it. The territory sweep's runs (another dispersal or driver rate) keep no
curve or clone sizes.
"""
import argparse
import fcntl
import json
import os
import shutil
import subprocess
import tempfile
import time
from collections import Counter

import numpy as np

K = 16
MIGRATION = 0.1
MU_DRIVER_PER_DAUGHTER = 1e-3
N_FINAL = 20_000
N_SEGMENTS, SEGMENT_SIZE, DRIVER_SHARE = 24, 2000, 0.5
# Poisson mean of new point mutations per allele copy per division: 48 copies (24 segments, two
# homologs) x DRIVER_SHARE of the sites -> drivers per division ~ Poisson(2 x MU_DRIVER_PER_DAUGHTER).
SNVS_PER_ALLELE = 2 * MU_DRIVER_PER_DAUGHTER / (2 * N_SEGMENTS * DRIVER_SHARE)
S = {"neutral": 0.0, "selection": 0.1, "realised": 0.2}
CLONE_MIN_FRAC = 0.04
TAU_RECORD = 0.5                           # generations per tau record (the paper's)

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
                                   "write_clones_file {clones}", "write_demes_file 0", "record_matrix 0",
                                   "write_phylo 0", "calculate_total_diversity 0",
                                   "biopsy_size_per_sample 100"]),
]
# demon reads its configuration with Boost's INFO parser: a top-level `parameters` key, then one
# key and value per line.
DEMON_CONFIG = "parameters\n{{\n" + "".join(f"    {sec}\n    {{{{\n" + "".join(f"        {kv}\n" for kv in kvs)
                                + "    }}\n" for sec, kvs in _DEMON_KEYS) + "}}\n"


def demon_config(cond, seed, migration=MIGRATION, n=N_FINAL, clones=1, mu=MU_DRIVER_PER_DAUGHTER):
    """demon's config.dat for one run (``clones`` writes the per-deme clone file the territories need;
    it only writes, so the trajectory is the same either way)."""
    return DEMON_CONFIG.format(log2k=int(np.log2(K)), mig=migration, mu=mu, s=S[cond], n=n, seed=seed,
                               clones=clones)


# --- shared readouts ------------------------------------------------------------------------------
def _indices_from_tree(parents, sizes, n_drivers):
    """Noble's (n, D, J1) from a driver-clone tree: parent map, populations, drivers per clone."""
    from iscc.inference.indices import inverse_simpson, tree_balance_j1
    pops = {k: v for k, v in sizes.items() if v > 0}
    total = sum(pops.values())
    n = sum(n_drivers[k] * v for k, v in pops.items()) / total
    return dict(n=float(n), D=inverse_simpson(pops.values()), J1=tree_balance_j1(parents, sizes),
                n_clones=len(pops))


def territories(parents, deme_counts, min_frac=CLONE_MIN_FRAC, keep_map=False):
    """Clonal territories of one tumour, defined the same way for both simulators.

    ``parents`` maps a genotype to its parent (the root absent); ``deme_counts`` maps a deme's
    ``(row, col)`` to its ``{genotype: cells}``. Clones are the clades of ``_clade_roots`` holding at
    least ``min_frac`` of the cells; each deme takes its dominant clone (``None``, ancestral, when
    its cells sit above every clade). Returns the same-clone fraction of neighbouring occupied deme
    pairs, its expectation under random placement, the number of clones and of dominant ones, and,
    with ``keep_map``, the dominant clone's size rank per deme (-1 ancestral)."""
    from iscc.integrations.clones import _clade_owner, _clade_roots, _depths
    parents = {str(k): str(v) for k, v in parents.items()}
    counts = Counter()
    for cc in deme_counts.values():
        for g, c in cc.items():
            counts[str(g)] += c
    total = sum(counts.values())
    depth = _depths(parents, counts)
    roots, subtree = _clade_roots(parents, depth, counts, max(1, int(min_frac * total)))
    owner = _clade_owner(parents, depth, roots)
    rank = {g: i for i, g in enumerate(sorted(roots, key=lambda g: (-subtree[g], g)))}
    dom = {}
    for pos, cc in deme_counts.items():
        lab = Counter()
        for g, c in cc.items():
            if c > 0:
                lab[owner.get(str(g))] += c
        if lab:
            dom[pos] = lab.most_common(1)[0][0]
    same = pairs = 0
    for (r, c), lab in dom.items():
        for nb in ((r + 1, c), (r, c + 1)):
            if nb in dom:
                pairs += 1
                same += dom[nb] == lab
    share = Counter(dom.values())
    rec = dict(assortment=same / pairs if pairs else float("nan"),
               assortment_expected=float(sum((v / len(dom)) ** 2 for v in share.values())) if dom else float("nan"),
               n_territory_clones=len(roots), n_dominant=len(share), occupied_demes=len(dom))
    if keep_map:
        rec["map"] = [[int(r), int(c), rank.get(lab, -1)] for (r, c), lab in sorted(dom.items())]
    return rec


def _append(outdir, rec):
    """Append one run to ``outdir/runs.jsonl`` under an exclusive lock (array tasks share the file)."""
    with open(os.path.join(outdir, "runs.jsonl"), "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(rec) + "\n")
        f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)


# --- demon ----------------------------------------------------------------------------------------
def run_demon(cond, seed, outdir, demon, migration=MIGRATION, keep_map=False, n_final=N_FINAL,
              mu=MU_DRIVER_PER_DAUGHTER):
    import pandas as pd
    scratch = os.environ.get("SNIC_TMP") or os.environ.get("TMPDIR") or outdir
    d = tempfile.mkdtemp(prefix=f"demon_{cond}_m{migration}_s{seed}_", dir=scratch)
    try:
        with open(os.path.join(d, "config.dat"), "w") as f:
            f.write(demon_config(cond, seed, migration, n_final, mu=mu))
        t0 = time.time()
        subprocess.run([demon, d + "/", "config.dat"], check=True, stdout=subprocess.DEVNULL)
        wall = time.time() - t0
        pops = pd.read_csv(os.path.join(d, "output.dat"), sep="\t")
        props = pd.read_csv(os.path.join(d, "output_driver_genotype_properties.dat"), sep="\t")
        cl = pd.read_csv(os.path.join(d, "clones.dat"), sep="\t",
                         usecols=["Generation", "DriverGenotype", "X", "Y", "Population"])
    finally:
        shutil.rmtree(d, ignore_errors=True)
    parents = {int(r.Identity): int(r.Parent) for r in props.itertuples() if int(r.Identity) != int(r.Parent)
               and int(r.Parent) >= 0}
    sizes = {int(r.Identity): float(r.Population) for r in props.itertuples()}
    ndrv = {int(r.Identity): int(r.DriverMutations) for r in props.itertuples()}
    cl = cl[cl.Generation == cl.Generation.max()]
    deme_counts = {}
    for (x, y, g), c in cl.groupby(["X", "Y", "DriverGenotype"]).Population.sum().items():
        if c > 0:
            deme_counts.setdefault((int(x), int(y)), {})[int(g)] = int(c)
    rec = dict(model="demon", engine="gillespie", cond=cond, migration=migration, mu=mu, seed=seed, wall_s=wall,
               **_indices_from_tree(parents, sizes, ndrv),
               cells=int(pops.NumCells.iloc[-1]), generations=float(pops.Generation.iloc[-1]),
               **territories(parents, deme_counts, keep_map=keep_map),
               clones_file_cells=int(cl.Population.sum()))
    if migration == MIGRATION and mu == MU_DRIVER_PER_DAUGHTER:
        rec.update(clone_sizes=sorted((int(v) for v in sizes.values() if v > 0), reverse=True),
                   curve=dict(t=pops.Generation.tolist(), cells=pops.NumCells.tolist(),
                              demes=pops["NumNon-EmptyDemes"].tolist()))
    _append(outdir, rec)
    return rec


# --- iscc -----------------------------------------------------------------------------------------
def iscc_tumour(cond, seed, engine="exact", migration=MIGRATION, mu=MU_DRIVER_PER_DAUGHTER):
    from iscc.tumor.models import GenotypeTumor
    genome = {"n_segments": N_SEGMENTS, "segment_size": SEGMENT_SIZE}
    selection = {"prop_driver": DRIVER_SHARE, "prop_dispersal": 0.0, "prop_immune_resistance": 0.0,
                 "prop_treatment_resistance": 0.0, "driver_effects": 1.0 + S[cond],
                 "dispersal_effects": 1.0, "treatment_resistant_effects": 1.0,
                 "immune_resistant_effects": 1.0}
    cancer = {"division_rate": 1.0, "death_rate": 0.0, "max_birth_rate": 10.0,
              "mutation_rate": 1.0, "dispersal_rate": migration,
              "snv_prob": 1.0, "cnv_prob": 0.0, "wgd_rate": 0.0, "n_snvs_per_allele": SNVS_PER_ALLELE * mu / MU_DRIVER_PER_DAUGHTER}
    deme = {"carrying_capacity": K, "initial_cancer_cells": 1, "crowding_d1": 100.0,
            "maximum_death_rate": 100.0, "crowding_overfill": 4.0}
    run = dict(update_mode="exact", snapshot_every=10**12)
    if engine == "tau":
        deme.update(tau_deaths="binomial", tau_step="cgp", tau_epsilon=0.01)   # the paper's tau-leaping
        run = dict(update_mode="tau", tau=TAU_RECORD, snapshot_every=10**12)
    spatial = {"grid_size": 81, "structure_radius": 0, "stroma_fill_frac": 0.0}
    return GenotypeTumor(seed=seed, genome_params=genome, selection_params=selection,
                         cancer_cell_params=cancer, deme_params=deme, spatial_params=spatial,
                         coarsen_passengers=True, **run)


def _occupied(t):
    return int(sum(1 for d in t.demes if any(c > 0 and t._is_cancer(g) for g, c in d.items())))


def run_iscc(cond, seed, outdir, engine="exact", migration=MIGRATION, keep_map=False,
             n_final=N_FINAL, record_every=0.25, mu=MU_DRIVER_PER_DAUGHTER):
    from iscc.inference.indices import driver_combination_counts, mode_indices
    t = iscc_tumour(cond, seed, engine, migration, mu)
    keep_curve = migration == MIGRATION and mu == MU_DRIVER_PER_DAUGHTER
    t0 = time.time()
    curve = dict(t=[0.0], cells=[t.get_cancer_size()], demes=[1])
    while 0 < t.get_cancer_size() < n_final:
        if engine == "exact":
            # grow(k) fires k - 1 events; a chunk of about `record_every` generations per curve point
            steps = max(100, int(float(t.deme_rates.sum()) * record_every))
            t.grow(steps + 1, seed=seed, materialize=False, target_cancer=n_final)
        else:
            t.grow(1, seed=seed, materialize=False, target_cancer=n_final)   # one record, 0.5 generations
        del t.traces[:]                  # per-event count snapshots nothing here reads; bounds memory
        if keep_curve:
            curve["t"].append(float(t.time))
            curve["cells"].append(int(t.get_cancer_size()))
            curve["demes"].append(_occupied(t))
    wall = time.time() - t0
    m = mode_indices(t)
    deme_counts = {}
    for di, deme in enumerate(t.demes):
        cc = {g: c for g, c in deme.items() if c > 0 and t._is_cancer(g)}
        if cc:
            deme_counts[tuple(int(v) for v in t.deme_coords[di])] = cc
    rec = dict(model="iscc", engine=engine, cond=cond, migration=migration, mu=mu, seed=seed, wall_s=wall,
               n=m["n"], D=m["D"], J1=m["J1"], n_clones=m["n_clones"], cells=int(t.get_cancer_size()),
               generations=float(t.time), **territories(t.genotypes_parents, deme_counts, keep_map=keep_map),
               normal_cells=int(sum(c for d in t.demes for g, c in d.items() if not t._is_cancer(g))),
               edge_demes=int(sum(1 for (r, c) in deme_counts if min(r, c) == 0 or max(r, c) == t.grid_size - 1)))
    if keep_curve:
        rec.update(clone_sizes=[int(v) for v in sorted(driver_combination_counts(t).values(), reverse=True)],
                   curve=curve)
    _append(outdir, rec)
    return rec


# --- collect --------------------------------------------------------------------------------------
def collect(out_csv, *jsonl):
    """One row per run (indices, territories, size, time) plus long tables of clone sizes, growth
    curves and dominant-clone maps, for the figure. A run logged twice keeps its latest record."""
    import pandas as pd
    recs = {}
    for path in jsonl:
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                recs[(r["model"], r["engine"], r["cond"], r["migration"], r["mu"], r["seed"])] = r
    recs = [recs[k] for k in sorted(recs, key=str)]
    keep = ["model", "engine", "cond", "migration", "mu", "seed", "n", "D", "J1", "n_clones", "cells",
            "generations", "wall_s", "assortment", "assortment_expected", "n_territory_clones",
            "n_dominant", "occupied_demes"]
    pd.DataFrame([{k: r.get(k) for k in keep} for r in recs]).to_csv(out_csv, index=False)
    base = os.path.splitext(out_csv)[0]
    ids = ["model", "engine", "cond", "migration", "mu", "seed"]
    pd.DataFrame([dict({k: r[k] for k in ids}, size=s)
                  for r in recs for s in r.get("clone_sizes", [])]).to_csv(base + "_clones.csv", index=False)
    pd.DataFrame([dict({k: r[k] for k in ids}, t=round(t, 4), cells=c, demes=d)
                  for r in recs if "curve" in r
                  for t, c, d in zip(r["curve"]["t"], r["curve"]["cells"], r["curve"]["demes"])]
                 ).to_csv(base + "_curves.csv", index=False)
    pd.DataFrame([dict({k: r[k] for k in ids}, row=a, col=b, clone=c)
                  for r in recs for a, b, c in r.get("map", [])]).to_csv(base + "_maps.csv", index=False)
    print(f"{len(recs)} runs -> {out_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["demon", "iscc", "collect"])
    ap.add_argument("args", nargs="+")
    ap.add_argument("--demon", default="demon")
    ap.add_argument("--engine", choices=["exact", "tau"], default="exact", help="iscc's engine")
    ap.add_argument("--migration", type=float, default=MIGRATION, help="dispersal per division (both)")
    ap.add_argument("--mu", type=float, default=MU_DRIVER_PER_DAUGHTER,
                    help="drivers per daughter in demon (iscc: twice this per division)")
    ap.add_argument("--map", action="store_true", help="keep the dominant-clone map of the run")
    ap.add_argument("--n-final", type=int, default=N_FINAL, help="smaller size, for a quick check")
    a = ap.parse_args()
    if a.cmd == "collect":
        collect(a.args[0], *a.args[1:])
        return
    cond, seed, outdir = a.args[0], int(a.args[1]), a.args[2]
    os.makedirs(outdir, exist_ok=True)
    rec = (run_demon(cond, seed, outdir, a.demon, a.migration, a.map, a.n_final, a.mu) if a.cmd == "demon"
           else run_iscc(cond, seed, outdir, a.engine, a.migration, a.map, a.n_final, mu=a.mu))
    print(json.dumps({k: rec[k] for k in ("model", "engine", "cond", "migration", "mu", "seed", "n", "D", "J1",
                                           "cells", "generations", "wall_s", "assortment",
                                           "assortment_expected")}))


if __name__ == "__main__":
    main()
