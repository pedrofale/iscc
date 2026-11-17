"""Fig 5 runs: cohorts through the thin Cohort, one Slurm array task -> ONE summary file.

The design, the tool settings and the planted network were registered before any run. Every
tumour is grown by the paper regime's own ``grow`` (:mod:`regime`) through
:class:`iscc.cohort.Cohort`; only summaries are written, intermediates go to node-local scratch
(``$SNIC_TMP``), never /tmp.

Entry points (``python validation/paper/fig5_cohort.py <cmd> ...``, with ``PYTHONPATH=src``)::

    count part1|part2|part3 [--block B]   number of array tasks
    part1 <task>                          scRNA integration across patients, one cohort per task
    part2 <task>                          planted network: one (cohort, block of patients) per task
    part2-collect                         MHN / TreeMHN / co-occurrence on subsamples + the null
    part3 <task>                          dissemination traits: one (arm, block) per task
    part3-collect                         the same tools on the trait events

Part I (``ISCC_PAPER_TIER=full``: the sweep scale, 96 x 96 demes, 60,000 cancer cells) grows each
cohort's six patients in parallel processes, or loads R6's stored tumours for the first cohort
(seeds 101-106; ``--no-reuse-r6`` regrows them). Loading is identical to regrowing: no growth code
changed between R6's commit (b4f6fb9, regime v4) and this one, and test_paper_regime_golden pins
``regime.grow``. Each patient's primary is dissociated to 1,000 cells; the pooled cells are measured
by scRNA under two designs (confounded: one batch per patient; balanced: the pooled cells split at
random into six batches) at the registered batch-effect levels (the 10x preset; strong: sigma_batch
0.6, depth_batch_sigma 0.3) and the exploratory intermediate one added before any run (0.3 / 0.15;
``LEVEL_STATUS``), and once with no batch effect (one batch, sigma_batch 0, depth_batch_sigma
0), the truth, scored with each design's batch labels. Each set of counts is written to scratch as an
h5ad and integrated by ``tools/integrate.py`` through ``tools/bin/python-integration`` (none,
Harmony, scVI; k = 30), which computes the registered metrics; the summary keeps the metrics, their
ratios to the truth, and (``--embeddings``) 2-D UMAPs for panel b. Clones are the paper's
(``common.Clones``, clades of at least 4% of the cancer cells) keyed by (patient, clone).

Part II grows 3 replicate cohorts of 300 patients with the planted six-event network and one of 300
with the same events and no interactions (each event's own effect kept), on the small field to
15,000 cancer cells. Each task writes, per patient: event cell fractions and presence at 5% and 1%,
the true clone tree collapsed to event acquisitions at 5%, the share of cells at the division cap,
and the cost (CPU, wall, peak memory). ``part2-collect`` fits MHN (sym_sparse, lambda by CV),
TreeMHN (its defaults), CBN (EvAM-Tools, defaults; the first tool to cut) and the co-occurrence log
odds on nested subsamples of 50/100/200/300 of every cohort, and on bootstrap resamples of the
empty-network cohort for the null.

Part III grows the ductal arm (the regime with the metastatic lattice) and a neutral arm (barriers
off, stromal and metastatic hazards 0, transit floor 1, the three trait costs 0), 300 tumours each,
each to its own target drawn uniformly in 2,000-15,000 cancer cells; the events are the three
dissemination traits plus two driver modules with no planted interaction (an epistasis network of two
events, planted empty: it only tracks them). ``part3-collect`` runs the same tools.

Outputs go to ``--out`` (default ``$ISCC_FIG5_OUT``, else ``<registry OUT>/fig5``):
``part1/cohort<c>.json``, ``part2/<arm>_b<block>.json``, ``part2_summary.json``,
``part3/<arm>_b<block>.json``, ``part3_summary.json``. The tool images live in ``$ISCC_TOOLS_DIR``;
a tool whose image is absent is skipped and recorded as skipped.
"""
import argparse
import functools
import json
import math
import multiprocessing as mp
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for p in (os.path.join(REPO, "src"), os.path.dirname(HERE), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import _paths as P  # noqa: E402
import regime as RG  # noqa: E402
import registry as REG  # noqa: E402

TOOLS = os.path.join(HERE, "tools")
BIN = os.path.join(TOOLS, "bin")

# ------------------------------------------------------------------ the registered design
# Part I
COHORTS = {0: tuple(range(101, 107)), 1: tuple(range(107, 113)), 2: tuple(range(113, 119))}
CELLS_PER_PATIENT = 1000
N_BALANCED_BATCHES = 6
LEVELS = {"10x": {}, "intermediate": {"sigma_batch": 0.3, "depth_batch_sigma": 0.15},
          "strong": {"sigma_batch": 0.6, "depth_batch_sigma": 0.3}}
# The intermediate level was added before any run, after the tool smoke test showed Harmony does not
# integrate at the strong level; it is exploratory, with no expectation.
LEVEL_STATUS = {"10x": "registered", "strong": "registered",
                "intermediate": "exploratory: added 2026-11-16 after the tool smoke test, no expectation"}
TRUTH_LEVEL = {"sigma_batch": 0.0, "depth_batch_sigma": 0.0}
METHODS = ("none", "harmony", "scvi")
K = 30

# Part II: the planted network (its E1..E6 are events 0..5)
N_EVENTS, EVENT_SIZE, LETHAL = 6, 10, 2.5
PLANTED = {"effects": 0.05, "interactions": [(0, 1, 0.4)], "exclusive": [(2, 3)],
           "constraints": [(4, 5)]}
EMPTY = {"effects": 0.05}                    # the same events, their own effects, no interaction
def scale_planted(planted, s):
    """The planted network with every fitness term (own effects and pairwise interactions) times s;
    the lethal pair and the order constraint are not fitness terms and are kept (the registered
    cap-share rule halves the event effects, s = 0.5)."""
    out = dict(planted)
    out["effects"] = float(planted["effects"]) * s
    if "interactions" in planted:
        out["interactions"] = [(i, j, float(w) * s) for i, j, w in planted["interactions"]]
    return out


PART2_ARMS = {"planted1": (PLANTED, 20_000), "planted2": (PLANTED, 21_000),
              "planted3": (PLANTED, 22_000), "empty": (EMPTY, 23_000)}
PART2_PATIENTS = 300
PART2_TARGET = 15_000
SIZES = (50, 100, 200, 300)
THRESHOLDS = (0.05, 0.01)
DETECTION = 0.05

# Part III
PART3_ARMS = {"ductal": 30_000, "neutral": 31_000}
PART3_PATIENTS = 300
PART3_TARGETS = (2_000, 15_000)
CONTROL_EVENTS = {"n_events": 2, "event_size": EVENT_SIZE, "planted": {}}
NEUTRAL_SPATIAL = {"breach_gated_invasion": False, "stromal_hazard": 0.0, "met_hazard": 0.0,
                   "met_transit_floor": 1.0}
NEUTRAL_SELECTION = {"breach_cost": 0.0, "stromal_survival_cost": 0.0, "met_survival_cost": 0.0}

# A toy field for smoke tests only (the golden test's): 24 x 24 demes, two glands, the acinus open
# after 10 generations.
TINY = dict(grid_size=24, n_glands=2, min_gland_sep=11,
            origin_confinement=dict(RG.ORIGIN_CONFINEMENT, generations=10))


def out_dir(args):
    d = args.out or os.environ.get("ISCC_FIG5_OUT") or os.path.join(REG.OUT, "fig5")
    os.makedirs(d, exist_ok=True)
    return d


def scratch(prefix):
    """A fresh directory on node-local scratch (never /tmp)."""
    base = os.environ.get("SNIC_TMP") or os.environ.get("ISCC_WORK_DIR")
    if not base:
        sys.exit("no node-local scratch: run inside a Slurm job ($SNIC_TMP) or set ISCC_WORK_DIR")
    os.makedirs(base, exist_ok=True)
    return tempfile.mkdtemp(prefix=prefix, dir=base)


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    part = path + ".part"
    with open(part, "w") as f:
        json.dump(obj, f, default=_jsonable)
    os.replace(part, path)
    return path


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def _cpu():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def _peak_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _commit():
    try:
        return subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None


def blocks(n_patients, block):
    return int(math.ceil(n_patients / block))


def field(args):
    """(scale, spatial overrides) of the growth field: the registered one or the toy field."""
    if args.toy:
        return "small", dict(TINY)
    return args.scale, {}


# ================================================================== Part I
def _part1_patient(job):
    """One patient (a spawned process): load R6's tumour or grow it, dissociate its primary, and
    return only the dissociated cells' expression and labels."""
    seed, i, seeds, scale, target, spatial, n_cells, r6_task = job
    import common as C
    from iscc.cohort import Cohort
    t0, c0 = time.time(), _cpu()
    if r6_task is not None:
        t = REG.load("R6", r6_task)
        source = f"R6/{r6_task}"
    else:
        grow = functools.partial(RG.grow, scale=scale, target_cancer=target,
                                 **({"spatial": spatial} if spatial else {}))
        t = Cohort(grow, patient_seeds=seeds).grow_patient(i)
        source = "grown"
    grow_cpu = _cpu() - c0
    cd = RG.RR.resect(t).dissociate(max_cells=n_cells, layers="scrna")
    gid = cd["cell_type"]["cell_id"].astype(str)
    clones = C.Clones(t)
    is_cancer = np.array([g in t.genotypes and t._is_cancer(g) for g in gid.values])
    ctype = [("cancer" if c else g) for g, c in zip(gid.values, is_cancer)]
    clone = [(clones.of(g) if c else "normal") for g, c in zip(gid.values, is_cancer)]
    return dict(seed=seed, source=source, cancer=int(t.get_cancer_size()),
                reached_target=bool(getattr(t, "reached_target", True)), n_genotypes=len(t.genotypes),
                n_clones=len(clones.labels), cpu_s=round(_cpu() - c0, 1), grow_cpu_s=round(grow_cpu, 1),
                wall_s=round(time.time() - t0, 1), peak_mb=round(_peak_mb()),
                cells=list(gid.index), cell_exp=cd["cell_exp"].values.astype(np.float32),
                genes=list(cd["cell_exp"].columns), genotype=list(gid.values),
                cell_type=ctype, is_cancer=is_cancer.tolist(), clone=clone)


def _scrna_counts(pooled, cells, label, seed, level):
    from iscc.data import scRNA
    a = scRNA(protocol="10x", n_cells=len(cells), batch_label=label, seed=seed, **level)
    a.run(pooled, cell_subset=list(cells))
    return a.observed_counts, a.obs["is_doublet"].to_numpy()


def _write_h5ad(path, counts, batch, meta):
    """Raw counts (cells in ``counts``' order) with the labels integrate.py reads; returns the order."""
    import anndata as ad
    import scipy.sparse as sp
    obs = meta.loc[counts.index, ["patient", "cell_type", "is_cancer", "clone"]].copy()
    obs["batch"] = batch.reindex(counts.index).astype(str).values
    a = ad.AnnData(X=sp.csr_matrix(counts.values.astype(np.float32)), obs=obs,
                   var=pd.DataFrame(index=[str(g) for g in counts.columns]))
    a.write_h5ad(path)
    return list(counts.index)


def _integrate(h5ad, prefix, method, args, log, threads=1):
    py = args.integration_python or os.path.join(BIN, "python-integration")
    cmd = [py, os.path.join(TOOLS, "integrate.py"), h5ad, prefix, "--method", method, "--k", str(K),
           "--batch-key", "batch", "--patient-key", "patient", "--celltype-key", "cell_type",
           "--cancer-key", "is_cancer", "--clone-key", "clone"]
    if args.no_metrics:
        cmd.append("--no-metrics")
    if args.max_epochs:
        cmd += ["--max-epochs", str(args.max_epochs)]
    cmd += ["--threads", str(threads)]
    env = dict(os.environ, ISCC_WORK_DIR=os.path.dirname(h5ad), ISCC_TOOLS_DIR=P.tools_dir(),
               ISCC_THREADS=str(threads))
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    log.append(dict(run=os.path.basename(prefix), method=method, returncode=r.returncode,
                    wall_s=round(time.time() - t0, 1), stderr_tail=r.stderr[-1500:] if r.returncode else ""))
    if r.returncode:
        print(f"[part1] {os.path.basename(prefix)} failed:\n{r.stderr[-3000:]}", file=sys.stderr)
        return None
    with open(prefix + ".json") as f:
        return json.load(f)


def _project(prefix, order, rows, seed=0):
    """2-D view of an embedding (UMAP, else PCA), in the pooled cell order, for panel b."""
    Z = np.load(prefix + ".npy")
    try:
        import umap
        xy = umap.UMAP(n_neighbors=K, random_state=seed).fit_transform(Z)
        how = "umap"
    except Exception:                              # noqa: BLE001
        from sklearn.decomposition import PCA
        xy = PCA(n_components=2, random_state=seed).fit_transform(Z)
        how = "pca"
    pos = {n: i for i, n in enumerate(order)}
    return np.round(xy[[pos[r] for r in rows]], 3).tolist(), how


RATIO_KEYS = ("ilisi_batch_noncancer", "ilisi_batch", "clisi_celltype", "patient_purity_cancer",
              "patient_purity_noncancer", "clone_purity_cancer")


def part1(args):
    from iscc.cohort import clone_key
    from iscc.data.rna import run_scrna_batches
    c = args.task
    seeds = COHORTS[c][:args.n_patients] if args.n_patients else COHORTS[c]
    scale, spatial = field(args)
    scale = scale or RG.T["sweep_scale"]
    target = args.target or RG.T["sweep_target"]
    n_cells = args.cells or CELLS_PER_PATIENT
    r6 = (c == 0 and not args.no_reuse_r6 and not args.toy and scale == RG.T["sweep_scale"]
          and target == RG.T["sweep_target"] and all(REG.grown("R6", f"patient{s - 100}") for s in seeds))
    jobs = [(s, i, list(seeds), scale, target, spatial, n_cells, f"patient{s - 100}" if r6 else None)
            for i, s in enumerate(seeds)]
    t0 = time.time()
    n_procs = max(1, min(len(jobs), args.procs or int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))))
    with mp.get_context("spawn").Pool(n_procs, maxtasksperchild=1) as pool:
        pats = pool.map(_part1_patient, jobs)
    grow_wall = time.time() - t0

    # the pooled cells, namespaced by patient (as iscc.cohort.pool_cell_data does); clones keyed by
    # (patient, clone)
    rows, exps, meta = [], [], []
    for p, d in enumerate(pats):
        names = [f"P{p}::{x}" for x in d["cells"]]
        rows += names
        exps.append(d["cell_exp"])
        meta += [(n, str(p), ct, bool(ic), clone_key(p, cl), g)
                 for n, ct, ic, cl, g in zip(names, d["cell_type"], d["is_cancer"], d["clone"],
                                             d["genotype"])]
    pooled = {"cell_exp": pd.DataFrame(np.vstack(exps).astype(float), index=rows, columns=pats[0]["genes"]),
              "cell_type": pd.DataFrame({"cell_id": [m[5] for m in meta]}, index=rows)}
    meta = pd.DataFrame(meta, columns=["cell", "patient", "cell_type", "is_cancer", "clone",
                                       "genotype"]).set_index("cell")
    del exps

    work = scratch(f"fig5p1-c{c}-")
    log, results, doublets, labels, order = [], {}, {}, {}, {}
    try:
        h5 = {}
        for lname, level in LEVELS.items():
            # confounded: one batch per patient, each its own scRNA run (seed per batch)
            parts, dbl = [], 0
            for p in range(len(pats)):
                cells = list(meta.index[meta["patient"] == str(p)])
                cnt, d = _scrna_counts(pooled, cells, f"batch{p}", 6000 + 100 * c + p, level)
                parts.append(cnt)
                dbl += int(d.sum())
            counts = pd.concat(parts)
            labels["confounded"] = pd.Series([f"batch{q}" for q in meta.loc[counts.index, "patient"]],
                                             index=counts.index)
            doublets[f"confounded/{lname}"] = dbl
            h5[("confounded", lname)] = os.path.join(work, f"confounded_{lname}.h5ad")
            order[h5[("confounded", lname)]] = _write_h5ad(h5[("confounded", lname)], counts,
                                                           labels["confounded"], meta)
            # balanced: the pooled cells split at random into six batches by run_scrna_batches (its
            # split depends only on its base seed, so it is the same at both levels)
            assays = run_scrna_batches(pooled, n_batches=N_BALANCED_BATCHES, design="split",
                                       n_cells=int(math.ceil(len(rows) / N_BALANCED_BATCHES)),
                                       base_seed=7000 + 100 * c, protocol="10x", **level)
            counts = pd.concat([a.observed_counts for a in assays])
            lab = pd.Series(np.concatenate([[a.batch_label] * a.n_cells for a in assays]), index=counts.index)
            if "balanced" in labels:
                assert labels["balanced"].reindex(lab.index).equals(lab), "balanced split moved"
            labels["balanced"] = lab
            doublets[f"balanced/{lname}"] = int(sum(a.obs["is_doublet"].sum() for a in assays))
            h5[("balanced", lname)] = os.path.join(work, f"balanced_{lname}.h5ad")
            order[h5[("balanced", lname)]] = _write_h5ad(h5[("balanced", lname)], counts, lab, meta)
            del counts, assays, parts
        # the truth: every cell measured once, in one batch, with no batch effect, scored with each
        # design's batch labels
        tc, tdbl = _scrna_counts(pooled, rows, "truth", 5000 + c, TRUTH_LEVEL)
        doublets["truth"] = int(tdbl.sum())
        for dname in labels:
            h5[(dname, "truth")] = os.path.join(work, f"truth_{dname}.h5ad")
            order[h5[(dname, "truth")]] = _write_h5ad(h5[(dname, "truth")], tc, labels[dname], meta)
        del tc, pooled

        # every integration at once, one thread each (each run is mostly single-threaded metrics
        # and, for scVI, training on a few thousand cells)
        runs = [(dname, lname, m, h) for (dname, lname), h in h5.items()
                for m in (("none",) if lname == "truth" else (args.methods or METHODS))]
        cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
        workers = max(1, min(len(runs), args.procs or cpus))
        threads = max(1, cpus // workers)
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as ex:
            outs = list(ex.map(lambda r: _integrate(r[3], r[3][:-5] + f"_{r[2]}", r[2], args, log, threads),
                               runs))
        truth = {}
        for (dname, lname, m, h), out in zip(runs, outs):
            pre = h[:-5] + f"_{m}"
            if lname == "truth":
                truth[dname] = dict(prefix=pre, h5ad=h, out=out)
            else:
                results[f"{dname}/{lname}/{m}"] = dict(prefix=pre, h5ad=h, out=out)

        summary = dict(
            part="I", cohort=c, seeds=list(seeds), scale=scale, target_cancer=target,
            toy=bool(args.toy), commit=_commit(), tier=RG.TIER, cells_per_patient=n_cells,
            source=("R6 stored tumours (identical to regrowing: no growth-code change since b4f6fb9)"
                    if r6 else "grown through iscc.cohort.Cohort"),
            k=K, levels=LEVELS, level_status=LEVEL_STATUS, truth_level=TRUTH_LEVEL,
            n_balanced_batches=N_BALANCED_BATCHES, integration_workers=workers,
            doublets=doublets, grow_wall_s=round(grow_wall, 1),
            patients=[{k: v for k, v in d.items() if k not in ("cells", "cell_exp", "genes", "genotype",
                                                                 "cell_type", "is_cancer", "clone")}
                      for d in pats],
            n_cells=int(len(rows)), n_cancer=int(meta["is_cancer"].sum()),
            n_clones=int(meta.loc[meta["is_cancer"], "clone"].nunique()),
            truth={d: ((t["out"] or {}).get("metrics")) for d, t in truth.items()}, runs={}, log=log)
        for key, r in results.items():
            o = r["out"]
            if o is None:
                summary["runs"][key] = dict(error="integration failed; see log")
                continue
            m = o.get("metrics") or {}
            tm = summary["truth"].get(key.split("/")[0]) or {}
            ratio = {f"{k}_retention": (m[k] / tm[k] if m.get(k) is not None and tm.get(k) else None)
                     for k in RATIO_KEYS}
            summary["runs"][key] = dict(metrics=m, retention=ratio, settings=o.get("settings"),
                                        timings=o.get("timings"), versions=o.get("versions"))
        if args.embeddings:
            emb = dict(cells=rows, level=args.embedding_level,
                       labels=dict(batch={d: labels[d].reindex(rows).tolist() for d in labels},
                                   patient=meta["patient"].tolist(), clone=meta["clone"].tolist(),
                                   cell_type=meta["cell_type"].tolist()), xy={})
            shown = {k: r for k, r in results.items() if k.split("/")[1] == args.embedding_level}
            shown.update({f"{d}/truth/none": t for d, t in truth.items()})
            for key, r in shown.items():
                if r["out"] is not None:
                    emb["xy"][key], emb["projection"] = _project(r["prefix"], order[r["h5ad"]], rows)
            summary["embeddings"] = emb
        path = write_json(os.path.join(out_dir(args), "part1", f"cohort{c}.json"), summary)
        print(f"[part1] cohort {c}: {len(rows)} cells, {len(results)} runs -> {path}")
    finally:
        if not args.keep_work:
            shutil.rmtree(work, ignore_errors=True)


# ================================================================== Part II
def _grow_patient_record(job):
    """Grow one patient in a spawned process and summarise it (Parts II and III)."""
    part, arm, i, seed, seeds, grow_kw, n_events = job
    from iscc.cohort import Cohort
    from iscc import integrations as ig
    from iscc.tumor.diagnostics import division_cap_share
    t0, c0 = time.time(), _cpu()
    t = Cohort(functools.partial(RG.grow, **grow_kw), patient_seeds=seeds).grow_patient(i)
    cpu, wall = _cpu() - c0, time.time() - t0
    rec = dict(arm=arm, index=i, seed=seed, target_cancer=grow_kw["target_cancer"],
               reached_target=bool(t.reached_target), cancer=int(t.get_cancer_size()),
               generations=float(t.time), n_genotypes=len(t.genotypes), cpu_s=round(cpu, 1),
               wall_s=round(wall, 1), peak_mb=round(_peak_mb()),
               cap_share=round(float(division_cap_share(t)), 5) if t.get_cancer_size() else None)
    if part == "II":
        frac = ig.event_cell_fractions(t)
        rec["names"] = t.selection.epistasis.event_names()
        rec["fraction"] = [round(float(f), 5) for f in frac]
        tree = ig.to_mutation_tree(t, detection=DETECTION)
        bits = [(c, t.genotypes[g].event_bits) for g, c in t.genotypes_counts.items() if t._is_cancer(g)]
        rec["cells_E2_and_E3"] = int(sum(c for c, b in bits if b >> 2 & 1 and b >> 3 & 1))
        rec["cells_E5_without_E4"] = int(sum(c for c, b in bits if b >> 5 & 1 and not b >> 4 & 1))
    else:
        fr = ig.trait_cell_fractions(t, with_events=True)
        rec["names"] = list(fr.index)
        rec["fraction"] = [round(float(f), 5) for f in fr.values]
        tree = ig.to_trait_tree(t, with_events=True, detection=DETECTION)
        rec["order"] = _trait_order(t)
        rec.update(_compartments(t))
    for thr in THRESHOLDS:
        rec[f"present_{thr:g}"] = [int(f >= thr) for f in rec["fraction"]]
    rec["tree"] = tree[["Node_ID", "Mutation_ID", "Parent_ID", "n_cells"]].astype(int).values.tolist()
    return rec


def _compartments(t):
    """Cancer cells in the primary's glands, in its stroma, and in the metastatic lattice."""
    n_primary = getattr(t, "n_primary_demes", len(t.demes))
    gland = stroma = met = 0
    for di, d in enumerate(t.demes):
        n = sum(c for g, c in d.items() if t._is_cancer(g))
        if di >= n_primary:
            met += n
        elif t.gland_id is not None and t.gland_id[di] >= 0:
            gland += n
        else:
            stroma += n
    return dict(gland_cells=int(gland), stroma_cells=int(stroma), met_cells=int(met))


def _trait_order(t):
    """The true order of the dissemination traits (the answer key of III-3): for each pair of traits,
    the cancer cells carrying both, split by which was acquired first along their lineage."""
    from iscc import integrations as ig
    tbl = ig.trait_table(t, with_events=True)
    names = tbl.attrs["names"]
    traits = [n for n in names if n in ig.TRAIT_EVENTS]
    out = {}
    for a in range(len(traits)):
        for b in range(a + 1, len(traits)):
            ia, ib = names.index(traits[a]), names.index(traits[b])
            first = second = tied = 0
            for groups, n in zip(tbl["event_groups"], tbl["n_cells"]):
                rank = {e: k for k, g in enumerate(groups) for e in g}
                if ia in rank and ib in rank:
                    if rank[ia] < rank[ib]:
                        first += n
                    elif rank[ia] > rank[ib]:
                        second += n
                    else:
                        tied += n
            out[f"{traits[a]}<{traits[b]}"] = dict(first=int(first), second=int(second), tied=int(tied))
    return out


def _part_tasks(arms, n_patients, block):
    nb = blocks(n_patients, block)
    return [(arm, b) for arm in arms for b in range(nb)]


def _run_block(part, arm, b, args, grow_kw_of, seeds, n_events):
    lo, hi = b * args.block, min((b + 1) * args.block, len(seeds))
    jobs = [("II" if part == 2 else "III", arm, i, seeds[i], seeds, grow_kw_of(i), n_events)
            for i in range(lo, hi)]
    recs = []
    t0 = time.time()
    with mp.get_context("spawn").Pool(1, maxtasksperchild=1) as pool:
        for r in pool.imap(_grow_patient_record, jobs):
            recs.append(r)
            print(f"[part{part}] {arm} patient {r['index']} seed {r['seed']}: {r['cancer']} cells, "
                  f"{r['cpu_s']} s CPU, cap share {r['cap_share']}", flush=True)
    return recs, time.time() - t0


def part2(args):
    n = args.n_patients or PART2_PATIENTS
    arm, b = _part_tasks(list(PART2_ARMS), n, args.block)[args.task]
    planted, s0 = PART2_ARMS[arm]
    planted = scale_planted(planted, args.effect_scale)
    seeds = [s0 + i for i in range(n)]
    scale, spatial = field(args)
    scale = scale or "small"
    target = args.target or PART2_TARGET
    ep = {"n_events": N_EVENTS, "event_size": EVENT_SIZE, "lethal_death_rate": LETHAL, "planted": planted}
    sel = {"epistasis_params": ep, "dependency_params": {"gating_mode": "accessibility"}}
    grow_kw = dict(scale=scale, target_cancer=target, selection=sel,
                   **({"spatial": spatial} if spatial else {}))
    recs, wall = _run_block(2, arm, b, args, lambda i: grow_kw, seeds, N_EVENTS)
    summary = dict(part="II", arm=arm, block=b, block_size=args.block, n_patients=n, scale=scale,
                   target_cancer=target, toy=bool(args.toy), commit=_commit(), epistasis_params=ep,
                   dependency_params=sel["dependency_params"], effect_scale=args.effect_scale,
                   detection=DETECTION,
                   thresholds=list(THRESHOLDS), wall_s=round(wall, 1), patients=recs)
    path = write_json(os.path.join(out_dir(args), "part2", f"{arm}_b{b:03d}.json"), summary)
    print(f"[part2] {arm} block {b}: {len(recs)} patients -> {path}")


# ================================================================== Part III
def part3(args):
    n = args.n_patients or PART3_PATIENTS
    arm, b = _part_tasks(list(PART3_ARMS), n, args.block)[args.task]
    s0 = PART3_ARMS[arm]
    seeds = [s0 + i for i in range(n)]
    scale, spatial = field(args)
    scale = scale or "small"
    lo, hi = args.target_range or PART3_TARGETS
    met = dict(RG.met_spatial(scale))
    if args.toy:
        met["met_grid_size"] = TINY["grid_size"]
    sp = {**met, **spatial}
    sel = {**RG.MET_SELECTION, "epistasis_params": CONTROL_EVENTS}
    if arm == "neutral":
        sp.update(NEUTRAL_SPATIAL)
        sel.update(NEUTRAL_SELECTION)
    # each tumour's own target size, drawn from its seed, so a block can be rerun alone
    targets = [int(np.random.default_rng(s).integers(lo, hi + 1)) for s in seeds]

    def grow_kw_of(i):
        return dict(scale=scale, target_cancer=targets[i], selection=sel, spatial=sp)

    recs, wall = _run_block(3, arm, b, args, grow_kw_of, seeds, 2)
    summary = dict(part="III", arm=arm, block=b, block_size=args.block, n_patients=n, scale=scale,
                   target_range=[lo, hi], toy=bool(args.toy), commit=_commit(), selection=sel,
                   spatial=sp, detection=DETECTION, thresholds=list(THRESHOLDS), wall_s=round(wall, 1),
                   patients=recs)
    path = write_json(os.path.join(out_dir(args), "part3", f"{arm}_b{b:03d}.json"), summary)
    print(f"[part3] {arm} block {b}: {len(recs)} patients -> {path}")


# ================================================================== the collectors (II, III)
def _image(name):
    return os.path.exists(os.path.join(P.tools_dir(), f"{name}.sif"))


def _fit_mhn(X, work, tag, seed=0):
    csv = os.path.join(work, f"{tag}.events.csv")
    X.to_csv(csv)
    pre = os.path.join(work, f"{tag}.mhn")
    env = dict(os.environ, ISCC_WORK_DIR=work, ISCC_TOOLS_DIR=P.tools_dir(), ISCC_THREADS="1")
    r = subprocess.run([os.path.join(BIN, "python-mhn"), os.path.join(TOOLS, "mhn_runner.py"), csv, pre,
                        "--seed", str(seed)], capture_output=True, text=True, env=env)
    if r.returncode:
        return dict(error=r.stderr[-1500:])
    with open(pre + ".json") as f:
        rec = json.load(f)
    theta = pd.read_csv(pre + ".theta.csv", index_col=0).values
    return dict(theta=theta.tolist(), lam=rec.get("lam"), cv=_cv_brief(rec.get("cv")),
                timings=rec.get("timings"), versions=rec.get("versions"))


def _cv_brief(cv):
    if not cv:
        return None
    return {k: cv[k] for k in ("nfolds", "steps", "grid", "rule", "seed", "at_grid_edge") if k in cv}


def _fit_treemhn(trees, names, work, tag, seed=0):
    from iscc.integrations import stack_trees
    try:
        df = stack_trees(trees)
    except ValueError as e:
        return dict(error=str(e))
    csv = os.path.join(work, f"{tag}.trees.csv")
    df.to_csv(csv, index=False)
    pre = os.path.join(work, f"{tag}.treemhn")
    env = dict(os.environ, ISCC_WORK_DIR=work, ISCC_TOOLS_DIR=P.tools_dir(), ISCC_THREADS="1")
    r = subprocess.run([os.path.join(BIN, "rscript-treemhn"), os.path.join(TOOLS, "treemhn_runner.R"), csv,
                        pre, "--n-events", str(len(names)), "--events", ",".join(names),
                        "--seed", str(seed)], capture_output=True, text=True, env=env)
    if r.returncode:
        return dict(error=r.stderr[-1500:])
    with open(pre + ".json") as f:
        rec = json.load(f)
    theta = pd.read_csv(pre + ".theta.csv", index_col=0).values
    return dict(theta=theta.tolist(), n_trees=rec.get("n_trees"), inference=rec.get("inference"),
                n_dropped_root_only=rec.get("n_dropped_root_only"), timings=rec.get("timings"),
                versions=rec.get("versions"))


def _fit_cbn(X, work, tag, seed=0):
    """CBN through EvAM-Tools (tools/cbn_runner.R): the fitted poset as (parent, child) edges, the
    root's children left out, plus the model table as evam returns it."""
    csv = os.path.join(work, f"{tag}.cbn.csv")
    X.to_csv(csv)
    pre = os.path.join(work, f"{tag}.cbn")
    env = dict(os.environ, ISCC_WORK_DIR=work, ISCC_TOOLS_DIR=P.tools_dir(), ISCC_THREADS="1")
    r = subprocess.run([os.path.join(BIN, "rscript-evamtools"), os.path.join(TOOLS, "cbn_runner.R"), csv, pre,
                        "--seed", str(seed)], capture_output=True, text=True, env=env)
    if r.returncode or not os.path.exists(pre + ".json"):
        return dict(error=(r.stderr or r.stdout)[-1500:])
    with open(pre + ".json") as f:
        rec = json.load(f)
    if rec.get("error"):
        return dict(error=rec["error"], timings=rec.get("timings"))
    model = rec.get("cbn_model") or []
    edges = []
    for row in model:
        a, b = row.get("From"), row.get("To")
        if a is not None and b is not None and str(a) != "Root":
            edges.append((str(a), str(b)))
    return dict(edges=edges, model=model, returned=rec.get("returned"), timings=rec.get("timings"),
                versions=rec.get("versions"))


def _tree_frame(rows):
    return pd.DataFrame(rows, columns=["Node_ID", "Mutation_ID", "Parent_ID", "n_cells"])


def _pairs(n):
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def pair_scores(theta, symmetric=False):
    """Per pair i < j: the strength (|.| of the larger-magnitude off-diagonal entry), its sign, both
    directed entries, and the rank by strength among all pairs (1 = strongest; ties share the
    smaller rank)."""
    T = np.asarray(theta, dtype=float)
    out = {}
    for i, j in _pairs(T.shape[0]):
        a, b = (T[i, j], T[i, j]) if symmetric else (T[i, j], T[j, i])
        v = a if abs(a) >= abs(b) else b
        out[(i, j)] = dict(strength=float(abs(v)), sign=int(np.sign(v)), ij=float(a), ji=float(b))
    s = sorted(out.values(), key=lambda d: -d["strength"])
    for d in out.values():
        d["rank"] = 1 + sum(1 for e in s if e["strength"] > d["strength"] + 1e-12)
    return out


def _fit_all(X, trees, names, tools, work, tag, seed):
    from iscc.integrations import cooccurrence_scores
    fits = {}
    if "cooc" in tools:
        fits["cooc"] = dict(theta=cooccurrence_scores(X.values).tolist(), symmetric=True)
    if "mhn" in tools:
        fits["mhn"] = _fit_mhn(X, work, tag, seed) if _image("mhn") else dict(skipped="mhn.sif absent")
    if "treemhn" in tools:
        fits["treemhn"] = (_fit_treemhn(trees, names, work, tag, seed) if _image("treemhn")
                           else dict(skipped="treemhn.sif absent"))
    if "cbn" in tools:
        fits["cbn"] = _fit_cbn(X, work, tag, seed) if _image("evamtools") else dict(skipped="evamtools.sif absent")
    for f in fits.values():
        if "theta" in f:
            sc = pair_scores(f["theta"], symmetric=f.get("symmetric", False))
            f["pairs"] = {f"{i},{j}": v for (i, j), v in sc.items()}
            f["top_strength"] = max(v["strength"] for v in sc.values())
    return fits


def _load_part(d, part):
    recs = []
    for fn in sorted(os.listdir(os.path.join(d, part))):
        if fn.endswith(".json"):
            with open(os.path.join(d, part, fn)) as f:
                recs.append(json.load(f))
    return recs


def collect(args, part):
    d = out_dir(args)
    files = _load_part(d, f"part{part}")
    if not files:
        sys.exit(f"no part{part} summaries under {d}")
    by_arm = {}
    for f in files:
        by_arm.setdefault(f["arm"], []).extend(f["patients"])
    tools = args.tools.split(",")
    sizes = [int(s) for s in args.sizes.split(",")] if args.sizes else list(SIZES)
    null_arms = args.null_arms.split(",") if args.null_arms else (["empty"] if part == 2 else ["neutral"])
    work = scratch(f"fig5p{part}-collect-")
    fits, cohorts = [], {}
    jobs = []
    try:
        for arm, pats in sorted(by_arm.items()):
            pats = sorted(pats, key=lambda r: r["index"])
            live = [r for r in pats if r["cancer"] > 0]
            names = live[0]["names"]
            cohorts[arm] = dict(n=len(pats), n_live=len(live),
                                n_reached=int(sum(r["reached_target"] for r in pats)),
                                cap_share_median=float(np.median([r["cap_share"] for r in live
                                                                  if r["cap_share"] is not None])),
                                frequency={f"{thr:g}": np.mean([r[f"present_{thr:g}"] for r in live],
                                                               axis=0).round(4).tolist()
                                           for thr in THRESHOLDS},
                                cpu_s_mean=float(np.mean([r["cpu_s"] for r in pats])), names=names)
            perm = np.random.default_rng(args.seed + sum(map(ord, arm))).permutation(len(live))
            boot = np.random.default_rng(args.seed + 7 + sum(map(ord, arm)))
            for n in sizes:
                if n > len(live):
                    continue
                draws = [("nested", 0, perm[:n])]
                if arm in null_arms:
                    draws += [("bootstrap", k, boot.integers(0, len(live), size=n))
                              for k in range(args.resamples)]
                for kind, k, idx in draws:
                    sub = [live[i] for i in idx]
                    for thr in (THRESHOLDS if args.sensitivity else (DETECTION,)):
                        X = pd.DataFrame([r[f"present_{thr:g}"] for r in sub], columns=names,
                                         index=[f"P{q}" for q in range(len(sub))])
                        trees = [_tree_frame(r["tree"]) for r in sub]
                        tag = f"{arm}_n{n}_{kind}{k}_t{thr:g}"
                        jobs.append((dict(arm=arm, n=n, kind=kind, resample=k, threshold=thr,
                                          event_frequency=X.mean(0).round(4).tolist()),
                                     X, trees if thr == DETECTION else None, names, tag))
        from concurrent.futures import ThreadPoolExecutor
        n_jobs = args.jobs or int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))

        def run(job):
            meta, X, trees, names, tag = job
            t = [x for x in tools if not (x == "treemhn" and trees is None)]
            return dict(meta, fits=_fit_all(X, trees, names, t, work, tag, args.seed))

        with ThreadPoolExecutor(max_workers=max(1, n_jobs)) as ex:
            for r in ex.map(run, jobs):
                fits.append(r)
                print(f"[part{part}-collect] {r['arm']} n={r['n']} {r['kind']}{r['resample']} "
                      f"t={r['threshold']:g}: " + ", ".join(
                          f"{k}:{'ok' if 'theta' in v else list(v)[0]}" for k, v in r["fits"].items()),
                      flush=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    sp = scale_planted(PLANTED, getattr(args, "effect_scale", 1.0))
    truth = (dict(interactions=sp["interactions"], exclusive=sp["exclusive"],
                  constraints=sp["constraints"], effects=sp["effects"],
                  effect_scale=getattr(args, "effect_scale", 1.0),
                  note="events 0..5 are the decision's E1..E6; the empty arm plants effects only")
             if part == 2 else dict(note="no interaction planted; the order of breach, stromal and "
                                         "metastatic survival comes from tissue architecture"))
    summary = dict(part="II" if part == 2 else "III", commit=_commit(), tools=tools, sizes=sizes,
                   null_arms=null_arms, resamples=args.resamples, seed=args.seed,
                   settings=dict(mhn="cMHN, sym_sparse, lambda by 5-fold CV (1-SE), unrounded "
                                     "(decision-fig5-progression-settings)",
                                 cbn="evamtools::evam(methods = 'CBN'), defaults",
                                 treemhn="learn_MHN defaults, no stability selection",
                                 cooc="pairwise log odds, pseudocount 0.5"),
                   truth=truth, cohorts=cohorts, fits=fits)
    path = write_json(os.path.join(d, f"part{part}_summary.json"), summary)
    print(f"[part{part}-collect] {len(fits)} fits -> {path}")


# ================================================================== CLI
def count(args):
    if args.what == "part1":
        return len(COHORTS)
    if args.what == "part2":
        return len(_part_tasks(list(PART2_ARMS), args.n_patients or PART2_PATIENTS, args.block))
    return len(_part_tasks(list(PART3_ARMS), args.n_patients or PART3_PATIENTS, args.block))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--out", default=None, help="summary directory (default $ISCC_FIG5_OUT, "
                                                   "else <registry OUT>/fig5)")
        p.add_argument("--toy", action="store_true", help="the 24 x 24 golden-test field (smoke tests)")
        p.add_argument("--n-patients", type=int, default=None, help="override the cohort size (smoke)")
        p.add_argument("--target", type=int, default=None, help="override the target size (smoke)")
        p.add_argument("--scale", default=None, help="field scale (default: the part's own)")
        p.add_argument("--block", type=int, default=10, help="patients per array task (II, III)")

    c = sub.add_parser("count")
    c.add_argument("what", choices=["part1", "part2", "part3"])
    common(c)
    p1 = sub.add_parser("part1")
    p1.add_argument("task", type=int)
    common(p1)
    p1.add_argument("--cells", type=int, default=None, help="cells per patient (default 1,000)")
    p1.add_argument("--procs", type=int, default=None, help="patients grown in parallel")
    p1.add_argument("--no-reuse-r6", action="store_true", help="regrow cohort 0 instead of loading R6")
    p1.add_argument("--methods", type=lambda s: s.split(","), default=None)
    p1.add_argument("--integration-python", default=None,
                    help="interpreter for integrate.py (default tools/bin/python-integration)")
    p1.add_argument("--no-metrics", action="store_true", help="pass --no-metrics (smoke without the image)")
    p1.add_argument("--max-epochs", type=int, default=None, help="scVI epochs (default scvi-tools' own)")
    p1.add_argument("--embeddings", action="store_true", help="keep 2-D embeddings for panel b")
    p1.add_argument("--embedding-level", default="strong")
    p1.add_argument("--keep-work", action="store_true")
    for name in ("part2", "part3"):
        p = sub.add_parser(name)
        p.add_argument("task", type=int)
        common(p)
        if name == "part3":
            p.add_argument("--target-range", type=lambda s: tuple(int(x) for x in s.split(",")),
                           default=None, help="lo,hi of the uniform target sizes (default 2000,15000)")
        else:
            p.add_argument("--effect-scale", type=float, default=1.0,
                           help="multiply the planted fitness terms (0.5: the registered cap-share rule)")
    for name in ("part2-collect", "part3-collect"):
        p = sub.add_parser(name)
        common(p)
        p.add_argument("--tools", default="cooc,mhn,treemhn,cbn" if name == "part2-collect"
                       else "cooc,mhn,treemhn",
                       help="cbn (EvAM-Tools) is Part II only by default, the first tool to cut")
        p.add_argument("--sizes", default=None, help="cohort sizes (default 50,100,200,300)")
        if name == "part2-collect":
            p.add_argument("--effect-scale", type=float, default=1.0,
                           help="the scale the arms were grown with (recorded in the truth)")
        p.add_argument("--null-arms", default=None, help="arms resampled for the null")
        p.add_argument("--resamples", type=int, default=20, help="bootstrap resamples of a null arm")
        p.add_argument("--sensitivity", action="store_true", help="also fit the 1%% matrices")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--jobs", type=int, default=None)
    args = ap.parse_args(argv)
    if args.cmd == "count":
        print(count(args))
    elif args.cmd == "part1":
        part1(args)
    elif args.cmd == "part2":
        part2(args)
    elif args.cmd == "part3":
        part3(args)
    elif args.cmd == "part2-collect":
        collect(args, 2)
    elif args.cmd == "part3-collect":
        collect(args, 3)


if __name__ == "__main__":
    main()
