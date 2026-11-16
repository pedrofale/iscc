"""Fig 5, the atlas design: one cohort of 50 patients, and every panel from it.

The registered design is the vault note exp-paper1-fig5-cohort, section "Amendment, 2026-11-16
evening: one atlas-scale cohort"; the network is decision-fig5-planted-network (its final amendment)
and the tool settings decision-fig5-progression-settings. The earlier design's driver,
:mod:`fig5_cohort`, supplies the shared pieces (the scRNA pooling and integration runner, the tool
runners, ``pair_scores``, the trait order); nothing here changes what those compute.

Entry points (``python validation/paper/fig5_atlas.py <cmd> ...``, with ``PYTHONPATH=src``)::

    count pilot|grow                    number of array tasks
    pilot <task>                        one pilot tumour: (arm, grid of module sizes, patient)
    pilot-collect                       per arm and grid: event frequencies at 5%, cap share, cost
    grow <task> --sizes s0,...,s6       one cohort patient: events, trees, traits, cells for scRNA
    integrate --level 10x|strong|intermediate
                                        the pooled scRNA of the cohort in both designs + the truth,
                                        integrated by none / Harmony / scVI and scored
    collect                             panels d-e: co-occurrence, MHN, TreeMHN, CBN

**The cohort** (``grow``): 50 patients (seeds :data:`COHORT_SEEDS`) at the sweep scale (``mid`` at
``ISCC_PAPER_TIER=full``: 96 x 96 demes), on the regime's ductal field with its metastatic lattice
(``RG.met_spatial``, ``RG.MET_SELECTION``), each grown to its own size, drawn uniformly in
8,000-60,000 cancer cells from a fixed seed (:data:`TARGET_SEED`). The planted network
(:data:`NETWORK`): seven event modules, E0 (the control: an own effect, no interaction) and E1-E6 as
package indices 0-6; own effects 0.1 and E6's 0.2, counted only once E5 is present (fitness gating);
E1-E2 synergy +0.2; E3-E4 redundancy -0.1; no lethal pair. Module sizes per event come from the
pilot and are given on the command line (``--sizes``); ``--effect-scale`` multiplies every fitness
term (own effects and interactions; the cap-share rule halves them).

**The pilot** (``pilot``, ``pilot-collect``): 15 patients from seeds outside the cohort's
(:data:`PILOT_SEEDS`), targets stratified over the cohort's range, grown in two arms (the network;
the same events and module sizes with every effect and interaction 0) at each of three grids of
module sizes (:data:`PILOT_GRID`). It records ONLY each event's presence at 5% detection, the share
of cancer cells at the division cap, and the cost; it computes no dependency between events.

Outputs (``--out``, default ``$ISCC_FIG5_OUT``, else ``<registry OUT>/fig5-atlas``)::

    pilot/<arm>_<grid>_p<k>.json  pilot/pilot_summary.json
    grow/patient<NN>.json         cells/patient<NN>.npz        (one pair per array task)
    integrate/level_<L>.json      collect_summary.json

Intermediates go to node-local scratch (``$SNIC_TMP``), never /tmp; no tumour is stored.
"""
import argparse
import functools
import json
import math
import os
import shutil
import sys
import time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for p in (os.path.join(REPO, "src"), os.path.dirname(HERE), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import fig5_cohort as F  # noqa: E402
import regime as RG  # noqa: E402
import registry as REG  # noqa: E402

# ------------------------------------------------------------------ the registered design
N_EVENTS = 7
EVENT_NAMES = [f"E{i}" for i in range(N_EVENTS)]
# decision-fig5-planted-network, amendment of 2026-11-16 evening (package indices = event names)
NETWORK = {"effects": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2],
           "interactions": [(1, 2, 0.2), (3, 4, -0.1)],
           "constraints": [(5, 6)]}
NULL = {"effects": 0.0}                           # the pilot's no-effect arm: same events and sizes
GATING = {"gating_mode": "fitness"}
RELATIONS = {"synergy": ((1, 2), +1), "redundancy": ((3, 4), -1), "order": ((5, 6), None)}
E0_PAIRS = [(0, j) for j in range(1, N_EVENTS)]

COHORT_SEEDS = tuple(range(40_001, 40_051))       # 50 patients
TARGET_RANGE = (8_000, 60_000)
TARGET_SEED = 4_000

PILOT_SEEDS = tuple(range(49_001, 49_016))        # 15 patients, outside the cohort's seeds
PILOT_TARGET_SEED = 4_900
PILOT_ARMS = {"network": NETWORK, "null": NULL}
# Module sizes per event (E0..E6): E0-E5 at 1, 2 and 4 genes, and E6 at twice that, since E6 pays
# only once E5 is present and is expected to be rarer. Every arm and grid grows the same 15 patients.
PILOT_GRID = {"g1": (1, 1, 1, 1, 1, 1, 2), "g2": (2, 2, 2, 2, 2, 2, 4), "g4": (4, 4, 4, 4, 4, 4, 8)}

DETECTION = 0.05
THRESHOLDS = (0.05, 0.01)
CELLS_PER_PATIENT = 1000
N_BALANCED_BATCHES = 50
LEVELS = F.LEVELS                                  # 10x, intermediate (exploratory), strong
EMBED_LEVELS = ("10x", "strong")                   # 2-D embeddings kept for panel b
EMBED_CELLS = 10_000                               # cells whose 2-D coordinates are kept
TRAITS = ("breach", "stromal_survival", "met_survival")
PANEL_E_NAMES = ("E0",) + TRAITS
# the names the tools see in panel e: without "_", which EvAM-Tools uses to join merged events
PANEL_E_TOOL = ("E0", "breach", "stromal", "met")


def out_dir(args):
    d = args.out or os.environ.get("ISCC_FIG5_OUT") or os.path.join(REG.OUT, "fig5-atlas")
    os.makedirs(d, exist_ok=True)
    return d


def scaled(planted, s):
    """The planted network with every fitness term (own effects, pairwise interactions) times s;
    the order constraint is not a fitness term and is kept."""
    out = dict(planted)
    eff = planted.get("effects", 0.0)
    out["effects"] = (float(eff) * s if np.ndim(eff) == 0 else [float(e) * s for e in eff])
    if "interactions" in planted:
        out["interactions"] = [(i, j, float(w) * s) for i, j, w in planted["interactions"]]
    return out


def parse_sizes(s):
    sizes = [int(x) for x in str(s).split(",")]
    if len(sizes) != N_EVENTS or min(sizes) < 1:
        raise argparse.ArgumentTypeError(f"--sizes takes {N_EVENTS} positive ints (E0..E6), got {s!r}")
    return sizes


def epistasis(sizes, planted):
    return {"n_events": N_EVENTS, "event_size": [int(x) for x in sizes], "planted": planted}


def field(args):
    """(scale, spatial overrides): the registered sweep scale with the metastatic lattice, or the
    golden test's toy field for smoke tests."""
    if args.toy:
        scale = "small"
        sp = {**RG.met_spatial(scale), **F.TINY, "met_grid_size": F.TINY["grid_size"]}
        return scale, sp
    scale = args.scale or RG.T["sweep_scale"]
    if args.scale is None and RG.TIER != "full":
        sys.exit("the registered sweep scale needs ISCC_PAPER_TIER=full (scale 'mid'); "
                 "pass --scale or --toy for a smoke test")
    return scale, dict(RG.met_spatial(scale))


def cohort_targets(n, lo_hi=None):
    lo, hi = lo_hi or TARGET_RANGE
    return [int(x) for x in np.random.default_rng(TARGET_SEED).integers(lo, hi + 1, size=len(COHORT_SEEDS))][:n]


def pilot_targets(n, lo_hi=None):
    """Stratified uniform over the cohort's range: one target in each of n equal bins, the bins
    assigned to the patients at random (a fixed seed)."""
    lo, hi = lo_hi or TARGET_RANGE
    rng = np.random.default_rng(PILOT_TARGET_SEED)
    u = (np.arange(n) + rng.random(n)) / n
    return [int(round(lo + x * (hi - lo))) for x in u[rng.permutation(n)]]


def grow_one(seeds, targets, i, scale, spatial, ep):
    """Grow patient ``i`` through the thin Cohort, each patient with its own target size."""
    from iscc.cohort import Cohort
    sel = {**RG.MET_SELECTION, "epistasis_params": ep, "dependency_params": dict(GATING)}
    grow = functools.partial(RG.grow, scale=scale, selection=sel, spatial=spatial)
    coh = Cohort(grow, patient_seeds=list(seeds), patient_kwargs=[{"target_cancer": t} for t in targets])
    return coh.grow_patient(i)


def _cost(t, c0, t0):
    return dict(cancer=int(t.get_cancer_size()), reached_target=bool(t.reached_target),
                generations=round(float(t.time), 2), n_genotypes=len(t.genotypes),
                cpu_s=round(F._cpu() - c0, 1), wall_s=round(time.time() - t0, 1),
                peak_mb=round(F._peak_mb()))


# ================================================================== the pilot
def pilot_grid(args):
    """The grids of module sizes a pilot submission grows: PILOT_GRID, or one custom grid
    (``--sizes``, named ``--grid-name``) for a confirming round at chosen sizes."""
    sizes = getattr(args, "sizes", None)
    return {args.grid_name: tuple(sizes)} if sizes else dict(PILOT_GRID)


def pilot_tasks(n, grid=PILOT_GRID):
    return [(arm, g, k) for arm in PILOT_ARMS for g in grid for k in range(n)]


def pilot(args):
    from iscc.integrations import event_cell_fractions
    from iscc.tumor.diagnostics import division_cap_share
    n = args.n_patients or len(PILOT_SEEDS)
    grids = pilot_grid(args)
    arm, grid, k = pilot_tasks(n, grids)[args.task]
    seeds = PILOT_SEEDS[:n]
    targets = ([args.target] * n if args.target else pilot_targets(n, args.target_range))
    scale, spatial = field(args)
    sizes = grids[grid]
    planted = scaled(PILOT_ARMS[arm], args.effect_scale)
    ep = epistasis(sizes, planted)
    t0, c0 = time.time(), F._cpu()
    t = grow_one(seeds, targets, k, scale, spatial, ep)
    present = [int(f >= DETECTION) for f in event_cell_fractions(t)]
    cap = division_cap_share(t) if t.get_cancer_size() else None
    rec = dict(step="pilot", arm=arm, grid=grid, sizes=list(sizes), index=k, seed=seeds[k],
               target_cancer=targets[k], scale=scale, toy=bool(args.toy), commit=F._commit(),
               tier=RG.TIER, effect_scale=args.effect_scale, epistasis_params=ep,
               dependency_params=GATING, detection=DETECTION, names=EVENT_NAMES,
               present=present, cap_share=None if cap is None else round(float(cap), 5),
               **_cost(t, c0, t0))
    path = F.write_json(os.path.join(out_dir(args), "pilot", f"{arm}_{grid}_p{k:02d}.json"), rec)
    print(f"[pilot] {arm} {grid} patient {k} (seed {seeds[k]}, target {targets[k]}): {rec['cancer']} cells, "
          f"{rec['cpu_s']} s CPU, {rec['peak_mb']} MB -> {path}", flush=True)


def _fit_rate(sizes, present):
    """Maximum-likelihood rate per gene of P(present) = 1 - exp(-lam * size), on a grid."""
    s, y = np.asarray(sizes, float), np.asarray(present, float)
    lam = np.exp(np.linspace(np.log(1e-3), np.log(5.0), 2000))
    p = 1 - np.exp(-np.outer(lam, s))
    ll = (y * np.log(np.clip(p, 1e-12, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-12, 1))).sum(1)
    return float(lam[int(np.argmax(ll))])


def pilot_collect(args):
    d = os.path.join(out_dir(args), "pilot")
    recs = []
    for fn in sorted(os.listdir(d)):
        if fn.endswith(".json") and fn != "pilot_summary.json":
            with open(os.path.join(d, fn)) as f:
                recs.append(json.load(f))
    if not recs:
        sys.exit(f"no pilot records under {d}")
    groups = {}
    for r in recs:
        groups.setdefault((r["arm"], r["grid"]), []).append(r)
    cells = {}
    for (arm, grid), rs in sorted(groups.items()):
        rs = sorted(rs, key=lambda r: r["index"])
        live = [r for r in rs if r["cancer"] > 0]
        cap = [r["cap_share"] for r in live if r["cap_share"] is not None]
        P = np.array([r["present"] for r in live]) if live else np.zeros((0, N_EVENTS))
        cpu = np.array([r["cpu_s"] for r in rs])
        cells[f"{arm}/{grid}"] = dict(
            arm=arm, grid=grid, sizes=rs[0]["sizes"], n=len(rs), n_live=len(live),
            n_reached=int(sum(r["reached_target"] for r in rs)),
            frequency=dict(zip(EVENT_NAMES, P.mean(0).round(4).tolist())) if len(P) else None,
            n_present=dict(zip(EVENT_NAMES, P.sum(0).astype(int).tolist())) if len(P) else None,
            cap_share=dict(median=float(np.median(cap)) if cap else None,
                           q25=float(np.percentile(cap, 25)) if cap else None,
                           q75=float(np.percentile(cap, 75)) if cap else None, values=cap),
            cpu_s=dict(median=float(np.median(cpu)), p90=float(np.percentile(cpu, 90)),
                       max=float(cpu.max()), total=float(cpu.sum())),
            wall_s_max=float(max(r["wall_s"] for r in rs)),
            peak_mb_max=int(max(r["peak_mb"] for r in rs)),
            cpu_per_1k_cells=float(np.median([r["cpu_s"] / max(r["cancer"], 1) * 1000 for r in rs])))
    # per arm and event: the rate per gene of 1 - exp(-lam * size) over the grids, and what it
    # predicts at sizes 1..12 (a marginal fit per event; nothing about pairs of events)
    fits = {}
    for arm in sorted({a for a, _ in groups}):
        rs = [r for (a, _), v in groups.items() if a == arm for r in v if r["cancer"] > 0]
        fits[arm] = {}
        for e, name in enumerate(EVENT_NAMES):
            lam = _fit_rate([r["sizes"][e] for r in rs], [r["present"][e] for r in rs])
            fits[arm][name] = dict(rate_per_gene=round(lam, 4), predicted={
                str(s): round(1 - math.exp(-lam * s), 3) for s in range(1, 13)})
    grids = {}
    for r in recs:
        grids[r["grid"]] = r["sizes"]
    summary = dict(step="pilot-collect", commit=F._commit(), detection=DETECTION, grid=grids,
                   names=EVENT_NAMES, n_records=len(recs), cells=cells, fits=fits,
                   note="each event's presence at 5% detection, the cap share and the cost only; "
                        "no dependency between events is computed")
    path = F.write_json(os.path.join(d, "pilot_summary.json"), summary)
    for key, c in cells.items():
        fr = c["frequency"] or {}
        print(f"[pilot] {key:12s} sizes {c['sizes']} n={c['n']} reached {c['n_reached']}: "
              + " ".join(f"{k}={v:.2f}" for k, v in fr.items())
              + f" | cap share median {c['cap_share']['median']} | CPU median {c['cpu_s']['median']:.0f} s,"
                f" max {c['cpu_s']['max']:.0f} s, peak {c['peak_mb_max']} MB")
    for arm, f in fits.items():
        print(f"[pilot] {arm} rate per gene: " + " ".join(f"{k}={v['rate_per_gene']}" for k, v in f.items()))
    print(f"[pilot-collect] {len(recs)} records -> {path}")


# ================================================================== the cohort
def _paths(tbl):
    """Every live cancer genotype's acquisition groups (indices into the trait table's names),
    aggregated: [[groups, n_cells], ...]. Enough to rebuild any tree or order over these names."""
    agg = Counter()
    for groups, n in zip(tbl["event_groups"], tbl["n_cells"]):
        agg[tuple(tuple(int(e) for e in g) for g in groups)] += int(n)
    return [[[list(g) for g in k], int(v)] for k, v in sorted(agg.items(), key=lambda kv: -kv[1])]


def projected_tree(paths, names, keep, detection):
    """The trie (TreeMHN's node table) of the acquisition paths restricted to the events ``keep``
    (names, in their new order; Mutation_ID k + 1 for keep[k]), collapsed at ``detection`` of all
    cancer cells, exactly as :func:`iscc.integrations.to_trait_tree` builds its tree."""
    from iscc.integrations.progression import _trie
    new = {names.index(k): i for i, k in enumerate(keep)}
    total = sum(n for _, n in paths)
    proj = []
    for groups, n in paths:
        order = []
        for g in groups:
            order += sorted(new[e] for e in g if e in new)
        proj.append((tuple(order), n))
    return _trie(proj, total, detection)


def _tree_rows(df):
    return df[["Node_ID", "Mutation_ID", "Parent_ID", "n_cells"]].astype(int).values.tolist()


def _cells(t, n_cells):
    """The dissociated primary (``n_cells`` cells) in the form Part I pools: expression, genotype,
    cell type ("cancer" or the normal type), cancer flag and the patient's clone (clades of at least
    4% of the cancer cells, as Part I)."""
    import common as C
    cd = RG.RR.resect(t).dissociate(max_cells=n_cells, layers="scrna")
    gid = cd["cell_type"]["cell_id"].astype(str)
    clones = C.Clones(t)
    is_cancer = np.array([g in t.genotypes and t._is_cancer(g) for g in gid.values])
    return dict(cells=np.asarray([str(c) for c in gid.index], dtype=str),
                genotype=np.asarray([str(g) for g in gid.values], dtype=str),
                cell_type=np.asarray([("cancer" if c else str(g)) for g, c in zip(gid.values, is_cancer)],
                                     dtype=str),
                is_cancer=np.asarray(is_cancer, dtype=bool),
                clone=np.asarray([(clones.of(g) if c else "normal") for g, c in zip(gid.values, is_cancer)],
                                 dtype=str),
                genes=np.asarray([str(g) for g in cd["cell_exp"].columns], dtype=str),
                exp=cd["cell_exp"].values.astype(np.float32)), len(clones.labels)


def grow_patient(args):
    from iscc import integrations as ig
    from iscc.tumor.diagnostics import division_cap_share
    import common  # noqa: F401  (imported before growth: the cells step needs it)
    n = args.n_patients or len(COHORT_SEEDS)
    i = args.task
    seeds = COHORT_SEEDS[:n]
    targets = [args.target] * n if args.target else cohort_targets(n, args.target_range)
    scale, spatial = field(args)
    planted = scaled(NETWORK, args.effect_scale)
    ep = epistasis(args.sizes, planted)
    t0, c0 = time.time(), F._cpu()
    t = grow_one(seeds, targets, i, scale, spatial, ep)
    cost = _cost(t, c0, t0)
    rec = dict(step="grow", index=i, seed=seeds[i], target_cancer=targets[i], scale=scale,
               toy=bool(args.toy), commit=F._commit(), tier=RG.TIER, sizes=list(args.sizes),
               effect_scale=args.effect_scale, epistasis_params=ep, dependency_params=GATING,
               selection_extra=RG.MET_SELECTION, spatial=spatial, detection=DETECTION,
               thresholds=list(THRESHOLDS), **cost)
    if t.get_cancer_size() > 0:
        tbl = ig.trait_table(t, with_events=True)
        names = list(tbl.attrs["names"])
        frac = ig.trait_cell_fractions(t, with_events=True)
        ev = ig.event_cell_fractions(t)
        rec["event_fraction_check"] = float(np.max(np.abs(frac.values[:N_EVENTS] - ev)))
        rec["names"] = names
        rec["fraction"] = [round(float(x), 5) for x in frac.values]
        for thr in THRESHOLDS:
            rec[f"present_{thr:g}"] = [int(x >= thr) for x in frac.values]
        paths = _paths(tbl)
        rec["paths"] = paths
        rec["tree_events"] = _tree_rows(ig.to_mutation_tree(t, detection=DETECTION))
        rec["tree_traits"] = _tree_rows(projected_tree(paths, names, PANEL_E_NAMES, DETECTION))
        rec["tree_all"] = _tree_rows(ig.to_trait_tree(t, with_events=True, detection=DETECTION))
        rec["order"] = F._trait_order(t)
        rec["cap_share"] = round(float(division_cap_share(t)), 5)
        rec.update(F._compartments(t))
    else:
        rec.update(names=EVENT_NAMES + list(TRAITS), cap_share=None)
    out = out_dir(args)
    tc = time.time()
    cells, n_clones = _cells(t, args.cells or CELLS_PER_PATIENT)
    rec.update(n_clones=n_clones, n_cells_sampled=int(len(cells["cells"])),
               n_cancer_sampled=int(cells["is_cancer"].sum()), cells_wall_s=round(time.time() - tc, 1),
               cells_file=f"cells/patient{i:02d}.npz")
    os.makedirs(os.path.join(out, "cells"), exist_ok=True)
    part = os.path.join(out, "cells", f"patient{i:02d}.part.npz")
    np.savez_compressed(part, **cells)
    os.replace(part, os.path.join(out, "cells", f"patient{i:02d}.npz"))
    rec["cpu_s_total"], rec["peak_mb"] = round(F._cpu() - c0, 1), round(F._peak_mb())
    path = F.write_json(os.path.join(out, "grow", f"patient{i:02d}.json"), rec)
    print(f"[grow] patient {i} (seed {seeds[i]}, target {targets[i]}): {rec['cancer']} cells, "
          f"{rec['cpu_s']} s CPU, cap share {rec['cap_share']}, {rec['n_cells_sampled']} cells kept "
          f"-> {path}", flush=True)


# ================================================================== integration (panels b-c)
def load_cells(out, n=None):
    d = os.path.join(out, "cells")
    files = sorted(fn for fn in os.listdir(d) if fn.startswith("patient") and fn.endswith(".npz")
                   and ".part." not in fn)
    if n:
        files = files[:n]
    pats = []
    for fn in files:
        with np.load(os.path.join(d, fn), allow_pickle=False) as z:
            pats.append({**{k: z[k] for k in z.files}, "patient": int(fn[len("patient"):-len(".npz")])})
    return pats


def pool(pats):
    """The cohort's cells pooled as Part I pools them: names ``P<p>::<cell>``, clones keyed by
    (patient, clone)."""
    from iscc.cohort import clone_key
    genes = pats[0]["genes"]
    rows, exps, meta = [], [], []
    for d in pats:
        if not np.array_equal(d["genes"], genes):
            raise ValueError(f"patient {d['patient']} has another gene set")
        p = d["patient"]
        names = [f"P{p}::{x}" for x in d["cells"]]
        rows += names
        exps.append(d["exp"])
        meta += [(n, str(p), str(ct), bool(ic), clone_key(p, cl), str(g))
                 for n, ct, ic, cl, g in zip(names, d["cell_type"], d["is_cancer"], d["clone"], d["genotype"])]
    pooled = {"cell_exp": pd.DataFrame(np.vstack(exps), index=rows, columns=[str(g) for g in genes]),
              "cell_type": pd.DataFrame({"cell_id": [m[5] for m in meta]}, index=rows)}
    meta = pd.DataFrame(meta, columns=["cell", "patient", "cell_type", "is_cancer", "clone",
                                       "genotype"]).set_index("cell")
    return pooled, meta, rows


def integrate(args):
    from iscc.data.rna import run_scrna_batches
    lname = args.level
    level = LEVELS[lname]
    out = out_dir(args)
    t0 = time.time()
    pats = load_cells(out, args.n_patients)
    if not pats:
        sys.exit(f"no cells under {out}/cells")
    pooled, meta, rows = pool(pats)
    del pats
    n_pat = meta["patient"].nunique()
    work = F.scratch(f"fig5atlas-{lname}-")
    log, labels, order, h5, doublets = [], {}, {}, {}, {}
    try:
        # confounded: one batch per patient, each its own scRNA run (a seed per batch)
        parts, dbl = [], 0
        for p in sorted(meta["patient"].unique(), key=int):
            cells = list(meta.index[meta["patient"] == p])
            cnt, d = F._scrna_counts(pooled, cells, f"batch{p}", 8000 + int(p), level)
            parts.append(cnt)
            dbl += int(d.sum())
        counts = pd.concat(parts)
        del parts
        labels["confounded"] = pd.Series([f"batch{q}" for q in meta.loc[counts.index, "patient"]],
                                         index=counts.index)
        doublets["confounded"] = dbl
        h5["confounded"] = os.path.join(work, "confounded.h5ad")
        order[h5["confounded"]] = F._write_h5ad(h5["confounded"], counts, labels["confounded"], meta)
        del counts
        # balanced: the pooled cells split at random into 50 batches of the same size, each holding
        # every patient (the split depends only on the base seed, so it is the same at every level)
        nb = min(N_BALANCED_BATCHES, len(rows))
        assays = run_scrna_batches(pooled, n_batches=nb, design="split",
                                   n_cells=int(math.ceil(len(rows) / nb)), base_seed=9000,
                                   protocol="10x", **level)
        counts = pd.concat([a.observed_counts for a in assays])
        labels["balanced"] = pd.Series(np.concatenate([[a.batch_label] * a.n_cells for a in assays]),
                                       index=counts.index)
        doublets["balanced"] = int(sum(a.obs["is_doublet"].sum() for a in assays))
        batch_sizes = [int(a.n_cells) for a in assays]
        del assays
        h5["balanced"] = os.path.join(work, "balanced.h5ad")
        order[h5["balanced"]] = F._write_h5ad(h5["balanced"], counts, labels["balanced"], meta)
        del counts
        # the truth: every cell measured once, in one batch, with no batch effect (the same at every
        # level: one seed), scored with each design's batch labels
        tc, tdbl = F._scrna_counts(pooled, rows, "truth", 7000, F.TRUTH_LEVEL)
        doublets["truth"] = int(tdbl.sum())
        for dname in ("confounded", "balanced"):
            h5[f"truth_{dname}"] = os.path.join(work, f"truth_{dname}.h5ad")
            order[h5[f"truth_{dname}"]] = F._write_h5ad(h5[f"truth_{dname}"], tc, labels[dname], meta)
        del tc, pooled
        prep_s = time.time() - t0

        methods = args.methods or F.METHODS
        runs = [(d, m) for d in ("confounded", "balanced") for m in methods]
        runs += [(f"truth_{d}", "none") for d in ("confounded", "balanced")]
        runs.sort(key=lambda r: r[1] != "scvi")           # scVI first: it sets the wall time
        cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
        n_scvi = sum(m == "scvi" for _, m in runs)
        cheap_workers = max(1, min(4, cpus // 4)) if n_scvi else max(1, min(len(runs), cpus))
        scvi_threads = max(1, (cpus - cheap_workers) // n_scvi) if n_scvi else 1
        rargs = argparse.Namespace(integration_python=args.integration_python, no_metrics=args.no_metrics,
                                   max_epochs=args.max_epochs)
        from concurrent.futures import ThreadPoolExecutor

        def run(r):
            d, m = r
            return F._integrate(h5[d], h5[d][:-5] + f"_{m}", m, rargs, log,
                                scvi_threads if m == "scvi" else 1)
        t1 = time.time()
        with ThreadPoolExecutor(max_workers=n_scvi + cheap_workers) as ex:
            outs = list(ex.map(run, runs))
        integrate_s = time.time() - t1
        res = {f"{d}/{m}": dict(prefix=h5[d][:-5] + f"_{m}", h5ad=h5[d], out=o) for (d, m), o in zip(runs, outs)}
        truth = {d: (res.get(f"truth_{d}/none") or {}).get("out") for d in ("confounded", "balanced")}
        summary = dict(
            step="integrate", level=lname, level_params=level, level_status=F.LEVEL_STATUS[lname],
            truth_level=F.TRUTH_LEVEL, commit=F._commit(), k=F.K, n_patients=int(n_pat),
            n_cells=int(len(rows)), n_cancer=int(meta["is_cancer"].sum()),
            n_clones=int(meta.loc[meta["is_cancer"], "clone"].nunique()),
            n_balanced_batches=nb, balanced_batch_sizes=batch_sizes, doublets=doublets,
            cpus=cpus, scvi_threads=scvi_threads, cheap_workers=cheap_workers,
            prep_s=round(prep_s, 1), integrate_s=round(integrate_s, 1), peak_mb=round(F._peak_mb()),
            truth={d: (o or {}).get("metrics") for d, o in truth.items()}, runs={}, log=log)
        for key, r in res.items():
            d, m = key.split("/")
            if d.startswith("truth_"):
                continue
            o = r["out"]
            if o is None:
                summary["runs"][key] = dict(error="integration failed; see log")
                continue
            mt = o.get("metrics") or {}
            tm = summary["truth"].get(d) or {}
            ratio = {f"{k}_retention": (mt[k] / tm[k] if mt.get(k) is not None and tm.get(k) else None)
                     for k in F.RATIO_KEYS}
            summary["runs"][key] = dict(metrics=mt, retention=ratio, settings=o.get("settings"),
                                        timings=o.get("timings"), versions=o.get("versions"))
        if lname in EMBED_LEVELS and not args.no_embeddings:
            te = time.time()
            rng = np.random.default_rng(11)
            keep = np.sort(rng.choice(len(rows), size=min(EMBED_CELLS, len(rows)), replace=False))
            shown = [k for k in res if not k.startswith("truth_balanced")]
            emb = dict(n_cells_total=int(len(rows)), n_shown=int(len(keep)),
                       labels=dict(batch={d: labels[d].reindex([rows[j] for j in keep]).tolist()
                                          for d in labels},
                                   patient=meta["patient"].values[keep].tolist(),
                                   clone=meta["clone"].values[keep].tolist(),
                                   cell_type=meta["cell_type"].values[keep].tolist()),
                       xy={}, note="2-D projection fitted on every cell, coordinates kept for a random "
                                   "subsample; the truth's counts are the same in both designs, so its "
                                   "embedding is shown once (truth/none)")
            for key in shown:
                r = res[key]
                if r["out"] is None:
                    continue
                xy, how = F._project(r["prefix"], order[r["h5ad"]], [rows[j] for j in keep])
                emb["xy"]["truth/none" if key.startswith("truth_") else key] = xy
                emb["projection"] = how
            emb["wall_s"] = round(time.time() - te, 1)
            summary["embeddings"] = emb
        summary["wall_s"] = round(time.time() - t0, 1)
        summary["peak_mb"] = round(F._peak_mb())
        path = F.write_json(os.path.join(out, "integrate", f"level_{lname}.json"), summary)
        print(f"[integrate] {lname}: {len(rows)} cells of {n_pat} patients, {len(runs)} runs in "
              f"{summary['wall_s']} s -> {path}", flush=True)
    finally:
        if not args.keep_work:
            shutil.rmtree(work, ignore_errors=True)


# ================================================================== panels d-e
def load_grown(out):
    d = os.path.join(out, "grow")
    recs = []
    for fn in sorted(os.listdir(d)):
        if fn.startswith("patient") and fn.endswith(".json"):
            with open(os.path.join(d, fn)) as f:
                recs.append(json.load(f))
    return sorted(recs, key=lambda r: r["index"])


def _cbn_has(edges, a, b):
    """Is there a CBN edge a -> b? evam merges events with identical columns into one node
    ("E2_E1_E5"), so a node stands for each event in its name."""
    return any(a in x.split("_") and b in y.split("_") for x, y in edges)


def score_network(fit):
    """Panel d's registered scoring of one fit: each planted relation's sign and magnitude against
    the six E0 pairs (recovered when the sign is the planted one and the magnitude exceeds every E0
    pair's), its rank among the 21 pairs, and for the order Theta[E6,E5] (``ji``) against
    Theta[E5,E6] (``ij``); for CBN, the edges."""
    if "edges" in fit:
        edges = fit["edges"]
        return dict(kind="cbn", edges=edges, order_edge=_cbn_has(edges, "E5", "E6"),
                    reverse_edge=_cbn_has(edges, "E6", "E5"),
                    synergy_edge=_cbn_has(edges, "E1", "E2") or _cbn_has(edges, "E2", "E1"),
                    redundancy_edge=_cbn_has(edges, "E3", "E4") or _cbn_has(edges, "E4", "E3"),
                    e0_edges=[e for e in edges if "E0" in e[0].split("_") or "E0" in e[1].split("_")],
                    recovered=dict(order=_cbn_has(edges, "E5", "E6")))
    if "pairs" not in fit:
        return None
    pairs = fit["pairs"]
    e0 = max(pairs[f"{i},{j}"]["strength"] for i, j in E0_PAIRS)
    sym = fit.get("symmetric", False)
    out = dict(kind="theta", e0_max_strength=e0, relations={}, recovered={})
    for name, ((i, j), sign) in RELATIONS.items():
        p = pairs[f"{i},{j}"]
        rel = dict(pair=[i, j], strength=p["strength"], sign=p["sign"], ij=p["ij"], ji=p["ji"],
                   rank=p["rank"], of=len(pairs), above_e0=p["strength"] > e0)
        if sign is not None:
            rel["sign_ok"] = p["sign"] == sign
            rel["recovered"] = rel["sign_ok"] and rel["above_e0"]
        elif sym:
            rel["recovered"] = None                    # a symmetric score has no direction
        else:
            rel["order_ok"] = p["ji"] > p["ij"]        # Theta[E6,E5] > Theta[E5,E6]
            rel["ji_above_e0"] = p["ji"] > e0
            rel["recovered"] = rel["order_ok"] and rel["ji_above_e0"]
        out["relations"][name] = rel
        out["recovered"][name] = rel["recovered"]
    return out


def score_traits(fit, names=PANEL_E_TOOL):
    """Panel e: the trait-trait dependencies against the E0-trait ones, and the order of breach
    before stromal and metastatic survival (Theta[trait, breach] against Theta[breach, trait]).
    ``names`` are the tools' column names, :data:`PANEL_E_TOOL`."""
    idx = {n: i for i, n in enumerate(names)}
    if "edges" in fit:
        edges = fit["edges"]
        return dict(kind="cbn", edges=edges,
                    breach_to_met=_cbn_has(edges, "breach", "met"),
                    breach_to_stromal=_cbn_has(edges, "breach", "stromal"),
                    met_to_breach=_cbn_has(edges, "met", "breach"),
                    stromal_to_breach=_cbn_has(edges, "stromal", "breach"),
                    e0_edges=[e for e in edges if "E0" in e[0].split("_") or "E0" in e[1].split("_")])
    if "pairs" not in fit:
        return None
    pairs = fit["pairs"]

    def pr(a, b):
        i, j = sorted((idx[a], idx[b]))
        return pairs[f"{i},{j}"], (idx[a] < idx[b])
    traits = names[1:]
    e0 = {t: pr("E0", t)[0] for t in traits}
    e0_max = max(p["strength"] for p in e0.values())
    trait_pairs = {}
    for a, b in ((traits[0], traits[1]), (traits[0], traits[2]), (traits[1], traits[2])):
        p, _ = pr(a, b)
        trait_pairs[f"{a}-{b}"] = dict(strength=p["strength"], sign=p["sign"], ij=p["ij"], ji=p["ji"],
                                       rank=p["rank"], above_e0=p["strength"] > e0_max)
    order = {}
    for t in traits[1:]:
        p, _ = pr(traits[0], t)                    # breach has the lower index: ji = Theta[t, breach]
        order[f"breach<{t}"] = dict(theta_t_breach=p["ji"], theta_breach_t=p["ij"],
                                    order_ok=None if fit.get("symmetric") else p["ji"] > p["ij"],
                                    above_e0=None if fit.get("symmetric") else p["ji"] > e0_max)
    return dict(kind="theta", e0_trait={t: dict(strength=p["strength"], sign=p["sign"], ij=p["ij"],
                                                 ji=p["ji"]) for t, p in e0.items()},
                e0_max_strength=e0_max, trait_pairs=trait_pairs, order=order)


def truth_orders(recs):
    """The answer key of panel e, from each tumour's genealogy: for each pair of [E0, traits], the
    tumours (with any cancer cell carrying both, and with both present at 5%) in which the majority
    of the cells carrying both acquired the first before the second, and the same share by cells."""
    out = {}
    keep = list(PANEL_E_NAMES)
    for a in range(len(keep)):
        for b in range(a + 1, len(keep)):
            ka, kb = keep[a], keep[b]
            tum = dict(any=[0, 0], present=[0, 0])
            cells = [0, 0, 0]
            for r in recs:
                if "paths" not in r:
                    continue
                ia, ib = r["names"].index(ka), r["names"].index(kb)
                first = second = tied = 0
                for groups, n in r["paths"]:
                    rank = {e: k for k, g in enumerate(groups) for e in g}
                    if ia in rank and ib in rank:
                        if rank[ia] < rank[ib]:
                            first += n
                        elif rank[ia] > rank[ib]:
                            second += n
                        else:
                            tied += n
                cells[0] += first
                cells[1] += second
                cells[2] += tied
                if first != second:                    # a majority of the carriers of both
                    k = 0 if first > second else 1
                    tum["any"][k] += 1
                    if r["present_0.05"][ia] and r["present_0.05"][ib]:
                        tum["present"][k] += 1
            out[f"{ka}<{kb}"] = dict(
                tumours_first=tum["any"][0], tumours_second=tum["any"][1],
                share_first=(tum["any"][0] / sum(tum["any"]) if sum(tum["any"]) else None),
                tumours_first_present=tum["present"][0], tumours_second_present=tum["present"][1],
                share_first_present=(tum["present"][0] / sum(tum["present"]) if sum(tum["present"]) else None),
                cells_first=cells[0], cells_second=cells[1], cells_tied=cells[2],
                cell_share_first=(cells[0] / (cells[0] + cells[1]) if cells[0] + cells[1] else None))
    return out


def _tree_frame(rows):
    return F._tree_frame(rows)


def collect(args):
    out = out_dir(args)
    recs = load_grown(out)
    if not recs:
        sys.exit(f"no cohort summaries under {out}/grow")
    live = [r for r in recs if r["cancer"] > 0 and "present_0.05" in r]
    names = live[0]["names"]
    ev_idx = list(range(N_EVENTS))
    pe_idx = [names.index(n) for n in PANEL_E_NAMES]
    tools = args.tools.split(",")
    work = F.scratch("fig5atlas-collect-")
    jobs = []
    try:
        for thr in THRESHOLDS:
            X = pd.DataFrame([[r[f"present_{thr:g}"][k] for k in ev_idx] for r in live], columns=EVENT_NAMES,
                             index=[f"P{r['index']}" for r in live])
            trees = [_tree_frame(r["tree_events"]) for r in live] if thr == DETECTION else None
            jobs.append(("d", thr, X, trees, EVENT_NAMES))
            Xe = pd.DataFrame([[r[f"present_{thr:g}"][k] for k in pe_idx] for r in live],
                              columns=list(PANEL_E_TOOL), index=[f"P{r['index']}" for r in live])
            trees_e = [_tree_frame(r["tree_traits"]) for r in live] if thr == DETECTION else None
            jobs.append(("e", thr, Xe, trees_e, list(PANEL_E_TOOL)))
        from concurrent.futures import ThreadPoolExecutor

        def run(job):
            panel, thr, X, trees, nm = job
            t = [x for x in tools if not (x == "treemhn" and trees is None)]
            fits = F._fit_all(X, trees, nm, t, work, f"{panel}_t{thr:g}", args.seed)
            scores = {k: (score_network(v) if panel == "d" else score_traits(v, nm)) for k, v in fits.items()}
            return dict(panel=panel, threshold=thr, names=nm, n=int(len(X)),
                        frequency=X.mean(0).round(4).tolist(), fits=fits, scores=scores)
        with ThreadPoolExecutor(max_workers=max(1, args.jobs or int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))) as ex:
            results = list(ex.map(run, jobs))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    cap = [r["cap_share"] for r in live if r.get("cap_share") is not None]
    cohort = dict(
        n=len(recs), n_live=len(live), n_reached=int(sum(r["reached_target"] for r in recs)),
        sizes=recs[0]["sizes"], effect_scale=recs[0]["effect_scale"], names=names,
        frequency={f"{thr:g}": dict(zip(names, np.mean([r[f"present_{thr:g}"] for r in live], 0).round(4).tolist()))
                   for thr in THRESHOLDS},
        cap_share=dict(median=float(np.median(cap)) if cap else None, values=cap),
        targets=[r["target_cancer"] for r in recs], cancer=[r["cancer"] for r in recs],
        compartments=[dict(gland=r.get("gland_cells"), stroma=r.get("stroma_cells"), met=r.get("met_cells"))
                      for r in recs],
        cpu_s=dict(median=float(np.median([r["cpu_s"] for r in recs])),
                   max=float(max(r["cpu_s"] for r in recs)), total=float(sum(r["cpu_s"] for r in recs))))
    summary = dict(step="collect", commit=F._commit(), tools=tools, seed=args.seed, detection=DETECTION,
                   settings=dict(mhn="cMHN, sym_sparse, lambda by 5-fold CV (1-SE), unrounded "
                                     "(decision-fig5-progression-settings)",
                                 treemhn="learn_MHN defaults, no stability selection",
                                 cbn="evamtools::evam(methods = 'CBN'), defaults",
                                 cooc="pairwise log odds, pseudocount 0.5"),
                   truth=dict(network=scaled(NETWORK, cohort["effect_scale"]), gating=GATING,
                              sizes=cohort["sizes"], relations={k: dict(pair=list(v[0]), sign=v[1])
                                                                for k, v in RELATIONS.items()},
                              reference="the six pairs of E0, which carries no interaction",
                              orders=truth_orders(live)),
                   cohort=cohort, results=results)
    path = F.write_json(os.path.join(out, "collect_summary.json"), summary)
    for r in results:
        line = []
        for tool, s in r["scores"].items():
            if s is None:
                line.append(f"{tool}: {list(r['fits'][tool])[0]}")
            elif s["kind"] == "cbn":
                line.append(f"{tool}: {len(s['edges'])} edges")
            else:
                line.append(f"{tool}: ok")
        print(f"[collect] panel {r['panel']} t={r['threshold']:g}: " + ", ".join(line), flush=True)
    print(f"[collect] {len(live)} patients -> {path}")


# ================================================================== CLI
def count(args):
    if args.what == "pilot":
        return len(pilot_tasks(args.n_patients or len(PILOT_SEEDS), pilot_grid(args)))
    return args.n_patients or len(COHORT_SEEDS)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--out", default=None, help="summary directory (default $ISCC_FIG5_OUT)")
        p.add_argument("--n-patients", type=int, default=None, help="override the cohort size (smoke)")

    def growth(p):
        p.add_argument("--toy", action="store_true", help="the 24 x 24 golden-test field (smoke tests)")
        p.add_argument("--scale", default=None, help="field scale (default: the sweep scale)")
        p.add_argument("--target", type=int, default=None, help="one target size for all (smoke)")
        p.add_argument("--target-range", type=lambda s: tuple(int(x) for x in s.split(",")), default=None,
                       help="lo,hi of the uniform target sizes (default 8000,60000)")
        p.add_argument("--effect-scale", type=float, default=1.0,
                       help="multiply every planted fitness term (0.5: the cap-share rule)")

    def grid(p):
        p.add_argument("--sizes", type=parse_sizes, default=None,
                       help="one grid of module sizes E0..E6 instead of PILOT_GRID (a confirming round)")
        p.add_argument("--grid-name", default="r2", help="the name of that grid in the records")

    c = sub.add_parser("count")
    c.add_argument("what", choices=["pilot", "grow"])
    common(c)
    grid(c)
    p = sub.add_parser("pilot")
    p.add_argument("task", type=int)
    common(p)
    growth(p)
    grid(p)
    p = sub.add_parser("pilot-collect")
    common(p)
    p = sub.add_parser("grow")
    p.add_argument("task", type=int)
    common(p)
    growth(p)
    p.add_argument("--sizes", type=parse_sizes, required=True, help="module sizes of E0..E6, e.g. 2,1,1,2,2,2,4")
    p.add_argument("--cells", type=int, default=None, help="dissociated primary cells kept (default 1,000)")
    p = sub.add_parser("integrate")
    common(p)
    p.add_argument("--level", required=True, choices=list(LEVELS))
    p.add_argument("--methods", type=lambda s: s.split(","), default=None)
    p.add_argument("--integration-python", default=None,
                   help="interpreter for integrate.py (default tools/bin/python-integration)")
    p.add_argument("--no-metrics", action="store_true")
    p.add_argument("--max-epochs", type=int, default=None, help="scVI epochs (default scvi-tools' own)")
    p.add_argument("--no-embeddings", action="store_true")
    p.add_argument("--keep-work", action="store_true")
    p = sub.add_parser("collect")
    common(p)
    p.add_argument("--tools", default="cooc,mhn,treemhn,cbn")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--jobs", type=int, default=None)
    args = ap.parse_args(argv)
    if args.cmd == "count":
        print(count(args))
    elif args.cmd == "pilot":
        pilot(args)
    elif args.cmd == "pilot-collect":
        pilot_collect(args)
    elif args.cmd == "grow":
        grow_patient(args)
    elif args.cmd == "integrate":
        integrate(args)
    elif args.cmd == "collect":
        collect(args)


if __name__ == "__main__":
    main()
