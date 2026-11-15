"""S1 Fig: tau-leaping against Gillespie's direct method (the exact engine), in the paper regime.

Every tumour in the paper is grown by tau-leaping: binomial deaths (Tian and Burrage 2004; Chatterjee
et al. 2005) and each leap chosen by the epsilon-bounded rule of Cao, Gillespie and Petzold (2006) at
epsilon 0.01 (``regime.DEME``; vault: decision-paper-regime-v3). This figure checks it against the
exact engine, which realises the same continuous-time Markov jump process one event at a time and
keeps Gillespie's clock. The comparison is built to be able to show a bias (vault:
decision-standard-simulation-algorithms, its F4-F6):

* **it varies the leap, not the record interval.** Besides the paper's tau-leaping it runs the same
  rule at a coarser epsilon (0.03, 0.1), the critical-channel rule (Cao, Gillespie and Petzold 2005),
  which fires the channels of demes at or above K one at a time, and the historical leap (capped
  Poisson deaths, a fixed sub-step), whose biases the algorithm reviews of 2026-11-14 measured. The
  rule's bound is max(epsilon c, 1) per (deme, clone) count c, and no count exceeds the slot cap
  floor(1.5 K) = 90 here, so every epsilon at or below 1/90 picks the same leap: epsilon 0.003 is run
  on a few seeds only, to show that it reproduces the paper's runs exactly;
* **it compares at matched size.** Every tumour stops at the first event (exact) or leap (tau) that
  reaches 5,000 cancer cells (``grow(..., target_cancer=)``);
* **it uses independent streams.** Streams are keyed by (seed, purpose, step); the exact and the tau
  arms grow from different growth seeds (``GROW_SEED``) on the same construction seeds, so they share
  the field, germline and founder of a seed but no random draw of growth;
* **it runs the paper's regime, confinement on.** The founding acinus is held for 200 generations of
  Gillespie time (exact) or simulated time (tau) after it is half full, as in every paper tumour.

Two small systems of the algorithm reviews (``wt/audit-algorithms/audit_sim.py``) are re-run with the
paper's rates (division 0.7, death 0.05 up to K and 5 above it, slot cap 1.5 K): one deme held at K
(time-averaged occupancy after a burn-in), and neutral fixation in a full deme of K = 30 (one cell
relabelled as an identical twin; by exchangeability it fixes with probability 1/N0).

Per tumour, at 5,000 cells: drivers per cell and clonal diversity at the 5% detection limit and at
full census (:func:`iscc.inference.mode_indices`), live genotypes, the fraction of genome altered and
mean ploidy (``sweep_score.score_tumour``, as the regime's tuning reads them), the occupancy (cells /
K) of the demes that reached K, the founding acinus's time-averaged occupancy while it is confined
(from 10 generations after it is half full to its release), the time to reach 5,000 cells, the
acinus's release time, and the wall time of growth. Most tumours are still in situ at 5,000 cells, so
the first INSITU_SEEDS of each arm are then grown on, in the same streams, until their first cancer
cell reaches a stroma deme (at most 300 generations after the release): the in-situ phase is that time
minus the release. Full demes and
the stroma are read every half generation in both engines.

Heavy work runs through Slurm (``wt/s1-jobs``); results are a few JSON-lines files per arm.
Run:
    python validation/paper/supp_tau.py run --out DIR --nproc 64 \
        --tumours exact:300,paper:300 --deme exact:200 --fix exact:40000
    python validation/paper/supp_tau.py render --out DIR     # figure via $ISCC_PAPER_DIR
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import realistic_regime as RR  # noqa: E402
import regime as RG  # noqa: E402

SCALE = "small"                  # grid 48, 3 glands: the exact engine's cost grows with the events
TARGET = 5_000                   # cancer cells at which every arm stops
GRID = 0.5                       # generations between the reads of the in-situ phase and full demes
SETTLE = 10.0                    # generations after the acinus is half full before its occupancy counts
T_MAX = 1_500.0                  # generations; a tumour still below target then has stayed in situ
INSITU_MAX = 300.0               # generations after the release that the in-situ phase is followed for
INSITU_SEEDS = 300               # construction seeds 1..INSITU_SEEDS are followed to the first stromal cell
GROW_SEED = {"exact": 10_000_000, "tau": 20_000_000}   # growth seed = construction seed + offset

_TAU_PAPER = {k: RG.DEME[k] for k in ("tau_deaths", "tau_step", "tau_epsilon")}
ARMS = {
    "exact": dict(mode="exact", label="exact (Gillespie)"),
    "paper": dict(mode="tau", deme=dict(_TAU_PAPER), label="paper: binomial, CGP ε 0.01"),
    "eps0.003": dict(mode="tau", deme=dict(_TAU_PAPER, tau_epsilon=0.003), label="CGP ε 0.003"),
    "eps0.03": dict(mode="tau", deme=dict(_TAU_PAPER, tau_epsilon=0.03), label="CGP ε 0.03"),
    "eps0.1": dict(mode="tau", deme=dict(_TAU_PAPER, tau_epsilon=0.1), label="CGP ε 0.1"),
    "critical": dict(mode="tau", deme=dict(_TAU_PAPER, tau_step="cgp_critical"),
                     label="critical channels exact, ε 0.01"),
    "historical": dict(mode="tau", deme=dict(tau_deaths="poisson", tau_step="fixed"),
                       label="historical: Poisson, fixed step"),
}
assert ARMS["paper"]["deme"] == {"tau_deaths": "binomial", "tau_step": "cgp", "tau_epsilon": 0.01}

# The small systems: the paper's cell law on one deme, no mutation, no dispersal.
B0 = float(RR.CANCER["division_rate"])                 # 0.7
D0 = float(RR.CANCER["death_rate"])                    # 0.05
D1 = float(RG.CROWDING_D1)                             # 5
OVERFILL = float(RG.CROWDING_OVERFILL)                 # 1.5
DEME_K = (30, 60)                                      # K_stroma and K_duct
DEME_BURN, DEME_OBS = 50.0, 200.0
FIX_K, FIX_BURN, FIX_TMAX, FIX_BATCH = 30, 30.0, 2_000.0, 200


# ---------------------------------------------------------------------------------- tumours
def _tumour_config(arm):
    a = ARMS[arm]
    return RG.config(SCALE, deme=a.get("deme"), update_mode=a["mode"])


def run_tumour(arm, seed):
    """Grow one paper-regime tumour of ``arm`` from construction seed ``seed`` to TARGET cells, read
    its statistics, then keep growing it until the first cancer cell reaches a stroma deme (the end of
    the in-situ phase), at most INSITU_MAX generations after the acinus's release."""
    from iscc.inference import mode_indices
    from iscc.inference.summaries import cna_summary
    from iscc.tumor.models import GenotypeTumor
    from iscc.tumor.models.materialise import _STREAM_GROW
    import sweep_score as SS

    mode = ARMS[arm]["mode"]
    t = GenotypeTumor(seed=seed, max_cells=1, **_tumour_config(arm))
    gseed = seed + GROW_SEED[mode]
    origin = tuple(t._origin_demes)
    acinus = [0.0, 0.0]                         # integral of occupancy / K over time, and the time
    k_origin = float(sum(t._cap_of(d) for d in origin))
    ever_full, st = set(), {"first_stroma": None, "events": 0, "next_grid": GRID}

    def in_window():
        return t._origin_confined and t._origin_age is not None and t._origin_age >= SETTLE

    def origin_occ():
        return sum(sum(t.demes[d].values()) for d in origin) / k_origin

    def read_grid():
        for di in t._deme_cancer_n:
            if sum(t.demes[di].values()) >= t._cap_of(di):
                ever_full.add(di)
        if st["first_stroma"] is None and any(t.gland_id[di] < 0 for di in t._deme_cancer_n):
            st["first_stroma"] = float(t.time)

    if mode == "tau":
        orig = t._tau_substep

        def substep(rng, dt, demes=None):        # records the state each leap starts from, and dt
            if in_window():
                acinus[0] += origin_occ() * dt
                acinus[1] += dt
            return orig(rng, dt) if demes is None else orig(rng, dt, demes=demes)
        t._tau_substep = substep

    def advance(done, target):
        """Grow until ``done()``, reading the grid every GRID generations."""
        if mode == "exact":
            # grow()'s exact path (one update() per event on the stream keyed by the event index, the
            # target checked before each event) without its per-event trace snapshot; the time a state
            # is held is the waiting time update() draws, so the acinus's occupancy is a time average.
            while not done():
                win = in_window()
                if win:
                    occ0, t0 = origin_occ(), t.time
                t._clock = t.step
                t.update(t._run_rng(gseed, t.step, _STREAM_GROW))
                t.step += 1
                st["events"] += 1
                if win:
                    acinus[0] += occ0 * (t.time - t0)
                    acinus[1] += t.time - t0
                if t.time >= st["next_grid"]:
                    read_grid()
                    st["next_grid"] = (math.floor(t.time / GRID) + 1) * GRID
        else:
            while not done():
                t.grow(n_steps=1, seed=gseed, materialize=False, target_cancer=target)
                t.traces.clear()
                t.trace_times.clear()
                read_grid()

    def release():
        rel = [e["time"] for e in t.events if e.get("event") == "origin_release"]
        return float(rel[0]) if rel else None

    # phase 1: to TARGET cells
    wall0, cpu0 = time.time(), time.process_time()
    advance(lambda: not 0 < t.get_cancer_size() < TARGET or t.time >= T_MAX, TARGET)
    wall, cpu = time.time() - wall0, time.process_time() - cpu0
    cancer = t.get_cancer_size()
    rec = dict(arm=arm, seed=seed, grow_seed=gseed, cancer=int(cancer), reached=bool(cancer >= TARGET),
               time=float(t.time), wall_s=round(wall, 2), cpu_s=round(cpu, 2),
               events=st["events"] or None, leaps=getattr(t, "_n_substeps", None),
               crit_events=getattr(t, "_n_crit_events", None), release=release(),
               invaded=st["first_stroma"] is not None,
               occ_acinus=acinus[0] / acinus[1] if acinus[1] > 0 else None, acinus_window=acinus[1])
    if cancer == 0:
        return rec
    counts = [c for g, c in t.genotypes_counts.items() if c > 0 and t._is_cancer(g)]
    p = np.asarray(counts, float) / cancer
    m5, m0 = mode_indices(t, detection=0.05), mode_indices(t, detection=0.0)
    occ = [sum(t.demes[d].values()) / t._cap_of(d) for d in sorted(ever_full)]
    sc = SS.score_tumour(t)
    rec.update(genotypes=len(counts), top=float(p.max()), n5=m5["n"], D5=m5["D"], clones5=m5["n_clones"],
               n0=m0["n"], D0=m0["D"], clones0=m0["n_clones"], fga=sc["frac_genome_altered"],
               fga_vs2=cna_summary(t)["fga"], ploidy=sc["mean_ploidy"], pct_wgd=sc["pct_wgd"],
               n_full=len(occ), occ_full=float(np.mean(occ)) if occ else None,
               n_cancer_demes=len(t._deme_cancer_n))

    # phase 2: on to the first cancer cell in the stroma (in the same stream, as one growth)
    wall0 = time.time()
    rec["insitu_followed"] = bool(rec["reached"] and seed <= INSITU_SEEDS)
    if rec["insitu_followed"]:
        advance(lambda: (st["first_stroma"] is not None or t.get_cancer_size() == 0
                         or t.time >= (release() or t.time) + INSITU_MAX or t.time >= T_MAX), None)
    rel = release()
    rec.update(release=rel, first_stroma=st["first_stroma"],
               in_situ=(st["first_stroma"] - rel if rec["insitu_followed"] and st["first_stroma"] is not None
                        else None),
               cancer_at_invasion=int(t.get_cancer_size()), wall_insitu_s=round(time.time() - wall0, 2))
    return rec


# ---------------------------------------------------------------------------------- small systems
def _small(arm, seed, n0, K):
    """One deme of the paper's cell law: division B0, death D0 up to K and D1 above it, slot cap
    OVERFILL x K, no mutation and no dispersal (the algorithm reviews' small system)."""
    from iscc.tumor.models import GenotypeTumor
    a = ARMS[arm]
    cancer = dict(division_rate=B0, death_rate=D0, max_birth_rate=2.0, mutation_rate=0.0,
                  n_snvs_per_allele=0.1, dispersal_rate=0.0, cnv_prob=0.0, snv_prob=1.0,
                  wgd_rate=0.0)
    deme = dict(carrying_capacity=K, initial_cancer_cells=n0, maximum_death_rate=D1,
                crowding_overfill=OVERFILL, crowding_d1=D1, **(a.get("deme") or {}))
    sel = dict(prop_driver=0.0, prop_dispersal=0.0, prop_immune_resistance=0.0, prop_met_survival=0.0,
               prop_treatment_resistance=0.0, prop_breach=0.0, prop_stromal_survival=0.0)
    return GenotypeTumor(seed=seed, genome_params=dict(n_segments=2, segment_size=50),
                         selection_params=sel, cancer_cell_params=cancer, deme_params=deme,
                         spatial_params=dict(grid_size=1), update_mode=a["mode"], tau=RR.TAU,
                         snapshot_every=10 ** 9, coarsen_passengers=True)


def _exact_until(T, rng, t_end, hist=None):
    """Run the exact engine to Gillespie time ``t_end``; with ``hist``, add to hist[n] the time the
    deme holds n cells (the waiting time update() draws is the holding time of the state before it)."""
    while T.time < t_end and not T.is_extinct():
        n, t0 = T.get_cancer_size(), T.time
        T.update(rng)
        if hist is not None:
            hist[n] += min(T.time, t_end) - t0


def run_deme(arm, K, rep):
    """Time-averaged occupancy of one deme held at K, after a burn-in."""
    seed = 1 + rep
    T = _small(arm, seed, K, K)
    hist = np.zeros(int(OVERFILL * K) + 2)
    if ARMS[arm]["mode"] == "exact":
        rng = np.random.default_rng([seed, GROW_SEED["exact"], 13])
        _exact_until(T, rng, DEME_BURN)
        _exact_until(T, rng, T.time + DEME_OBS, hist)
    else:
        gseed = seed + GROW_SEED["tau"]
        T.grow(n_steps=int(round(DEME_BURN / RR.TAU)), seed=gseed, materialize=False)
        orig = T._tau_substep

        def substep(rng, dt, demes=None):
            hist[T.get_cancer_size()] += dt
            return orig(rng, dt) if demes is None else orig(rng, dt, demes=demes)
        T._tau_substep = substep
        T.grow(n_steps=int(round(DEME_OBS / RR.TAU)), seed=gseed, materialize=False)
    mean = float(np.dot(np.arange(len(hist)), hist) / hist.sum())
    return dict(arm=arm, K=K, rep=rep, mean=mean, hist=[round(x, 4) for x in hist],
                leaps=getattr(T, "_n_substeps", None))


def _fix_one(arm, seed):
    """Relabel one cell of a full K = FIX_K deme as an identical twin; run until one type is lost."""
    T = _small(arm, seed, FIX_K, FIX_K)
    exact = ARMS[arm]["mode"] == "exact"
    rng = np.random.default_rng([seed, GROW_SEED["exact"], 14])
    gseed = seed + GROW_SEED["tau"]
    if exact:
        _exact_until(T, rng, FIX_BURN)
    else:
        T.grow(n_steps=int(round(FIX_BURN / RR.TAU)), seed=gseed, materialize=False)
    gid = next(g for g in T.demes[0] if T._is_cancer(g))
    N0 = T.demes[0][gid]
    twin = T.genotypes[gid].divide()
    twin.set_genotype_id()
    T._register(twin)
    T.genotypes_parents[twin.genotype_id] = gid
    T._remove(0, gid, 1)
    T._add(0, twin.genotype_id, 1)
    T._refresh_rate(0)
    mid, t_start = twin.genotype_id, T.time
    while T.time - t_start < FIX_TMAX:
        m = T.demes[0].get(mid, 0)
        if m == 0 or m == T.get_cancer_size():
            break
        if exact:
            T.update(rng)
        else:
            T.grow(n_steps=1, seed=gseed, materialize=False)
    m = T.demes[0].get(mid, 0)
    return N0, int(m > 0 and m == T.get_cancer_size()), int(m == 0), T.time - t_start


def run_fix(arm, batch):
    rs = [_fix_one(arm, 1 + batch * FIX_BATCH + i) for i in range(FIX_BATCH)]
    return dict(arm=arm, batch=batch, n=len(rs), fixed=sum(r[1] for r in rs), lost=sum(r[2] for r in rs),
                sum_inv_n0=float(sum(1.0 / r[0] for r in rs)),
                t_fixed=[round(r[3], 3) for r in rs if r[1]])


# ---------------------------------------------------------------------------------- driver
def _job(job):
    kind, arm, i = job
    t0 = time.time()
    try:
        if kind == "tumours":
            out = run_tumour(arm, i)
        elif kind == "deme":
            out = run_deme(arm, *i)
        else:
            out = run_fix(arm, i)
        return job, out, None, time.time() - t0
    except Exception as e:                       # noqa: BLE001 - reported, and retried on a re-run
        import traceback
        return job, None, f"{type(e).__name__}: {e}\n{traceback.format_exc()}", time.time() - t0


def _key(kind, rec):
    if kind == "tumours":
        return rec["seed"]
    if kind == "deme":
        return (rec["K"], rec["rep"])
    return rec["batch"]


def _done(out, kind, arm):
    path = os.path.join(out, f"{kind}_{arm}.jsonl")
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                if line.strip():
                    done.add(_key(kind, json.loads(line)))
    return done


def _parse(spec):
    out = []
    for item in (spec or "").split(","):
        if item:
            arm, n = item.split(":")
            if arm not in ARMS:
                raise SystemExit(f"unknown arm {arm!r}; arms are {sorted(ARMS)}")
            out.append((arm, int(n)))
    return out


def run(a):
    import multiprocessing as mp
    os.makedirs(a.out, exist_ok=True)
    jobs = []
    for arm, n in _parse(a.tumours):
        done = _done(a.out, "tumours", arm)
        jobs += [("tumours", arm, s) for s in range(1, n + 1) if s not in done]
    for arm, n in _parse(a.deme):
        done = _done(a.out, "deme", arm)
        jobs += [("deme", arm, (K, r)) for K in DEME_K for r in range(n) if (K, r) not in done]
    for arm, n in _parse(a.fix):
        done = _done(a.out, "fix", arm)
        jobs += [("fix", arm, b) for b in range(math.ceil(n / FIX_BATCH)) if b not in done]
    # the longest jobs first (tumours followed through the in-situ phase), so the pool does not end on
    # a tail of them
    heavy = {"critical": 0, "exact": 1, "paper": 2, "eps0.003": 2, "eps0.03": 3, "eps0.1": 4,
             "historical": 5}

    def order(j):
        if j[0] == "tumours":
            return (0 if j[2] <= INSITU_SEEDS else 1, heavy[j[1]], j[2])
        return (2 if j[0] == "fix" else 3, heavy[j[1]], 0)
    jobs.sort(key=order)
    print(f"{len(jobs)} jobs on {a.nproc} processes", flush=True)
    files = {}
    t0, n_err = time.time(), 0
    with mp.get_context("fork").Pool(processes=a.nproc, maxtasksperchild=8) as pool:
        for k, (job, rec, err, dt) in enumerate(pool.imap_unordered(_job, jobs, chunksize=1)):
            if err:
                n_err += 1
                print(f"ERROR {job}: {err}", flush=True)
                continue
            kind, arm, _ = job
            f = files.get((kind, arm))
            if f is None:
                f = files[(kind, arm)] = open(os.path.join(a.out, f"{kind}_{arm}.jsonl"), "a")
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if kind == "tumours" or k % 50 == 0:
                print(f"[{time.time() - t0:7.0f}s] {k + 1}/{len(jobs)} {job} {dt:.1f}s", flush=True)
    for f in files.values():
        f.close()
    print(f"done in {time.time() - t0:.0f}s, {n_err} errors", flush=True)


# ---------------------------------------------------------------------------------- statistics
def _load(out, kind):
    rows = {}
    for arm in ARMS:
        path = os.path.join(out, f"{kind}_{arm}.jsonl")
        if os.path.exists(path):
            with open(path) as f:
                rows[arm] = [json.loads(x) for x in f if x.strip()]
    return rows


def _mean_ci(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return dict(mean=float(v.mean()) if len(v) else None, lo=None, hi=None, n=len(v))
    h = 1.96 * v.std(ddof=1) / np.sqrt(len(v))
    return dict(mean=float(v.mean()), lo=float(v.mean() - h), hi=float(v.mean() + h), n=len(v))


def _rel_diff(x, y, n_boot=4000, seed=0):
    """(mean(y) / mean(x) - 1) with a percentile bootstrap 95% CI, x and y resampled independently."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    rng = np.random.default_rng(seed)
    bx = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(1)
    by = y[rng.integers(0, len(y), (n_boot, len(y)))].mean(1)
    r = by / bx - 1
    return dict(rel=float(y.mean() / x.mean() - 1), lo=float(np.quantile(r, 0.025)),
                hi=float(np.quantile(r, 0.975)))


def _wilson(k, n):
    if n == 0:
        return (float("nan"),) * 3
    p, z = k / n, 1.96
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return p, c - h, c + h


def stationary_exact(K):
    """The deme's quasi-stationary law under the exact process (birth B0 n below the slot cap, death
    D0 n at n <= K and D1 n above), by detailed balance, conditioned on n >= 1."""
    cap = int(K * OVERFILL)
    w = np.zeros(cap + 2)
    w[1] = 1.0
    for n in range(2, cap + 1):
        w[n] = w[n - 1] * B0 * (n - 1) / ((D0 if n <= K else D1) * n)
    return w / w.sum()


def _sd_hist(h):
    h = np.asarray(h, float)
    n = np.arange(len(h))
    m = np.dot(n, h) / h.sum()
    return float(np.sqrt(np.dot((n - m) ** 2, h) / h.sum()))


TUMOUR_STATS = [
    ("n5", "drivers per cell, 5% detection"), ("n0", "drivers per cell, full census"),
    ("D5", "clonal diversity, 5% detection"), ("D0", "clonal diversity, full census"),
    ("genotypes", "genotypes"), ("fga", "fraction of genome altered"), ("ploidy", "mean ploidy"),
    ("occ_full", "occupancy of demes that reached K"), ("occ_acinus", "acinus occupancy, confined"),
    ("release", "release of the acinus"), ("growth", "release to 5,000 cells"), ("in_situ", "in-situ phase"),
]


def summarise(out):
    """Every number the figure shows (and the Methods quote), as one JSON-able dict."""
    from scipy.stats import mannwhitneyu
    S = {"target": TARGET, "scale": SCALE, "arms": {a: ARMS[a]["label"] for a in ARMS}}
    tum = _load(out, "tumours")
    for rs in tum.values():
        for r in rs:
            r["growth"] = r["time"] - r["release"] if r.get("release") is not None else None
    if "exact" in tum:
        ok = {a: [r for r in rs if r.get("reached")] for a, rs in tum.items()}
        S["tumours"] = {}
        ex = ok["exact"]
        for arm, rs in ok.items():
            walls = [r["wall_s"] for r in rs]
            d = dict(n=len(rs), n_run=len(tum[arm]),
                     n_extinct=sum(1 for r in tum[arm] if r["cancer"] == 0),
                     n_in_situ=sum(1 for r in tum[arm] if 0 < r["cancer"] < TARGET),
                     cancer=_mean_ci([r["cancer"] for r in rs]),
                     wall_s=dict(median=float(np.median(walls)), mean=float(np.mean(walls)),
                                 q25=float(np.quantile(walls, 0.25)), q75=float(np.quantile(walls, 0.75))))
            d["invaded_at_target"] = float(np.mean([r["invaded"] for r in rs]))
            fol = [r for r in rs if r.get("insitu_followed")]
            d["in_situ_followed"] = len(fol)
            d["in_situ_censored"] = sum(1 for r in fol if r.get("in_situ") is None)
            d["time_to_target"] = _mean_ci([r["time"] for r in rs])
            d["wall_insitu_s"] = float(np.median([r["wall_insitu_s"] for r in fol])) if fol else None
            if arm == "exact":
                d["events"] = float(np.mean([r["events"] for r in rs]))
            elif rs[0].get("leaps"):
                d["leaps_per_gen"] = float(np.mean([r["leaps"] / r["time"] for r in rs]))
            for key, _ in TUMOUR_STATS:
                v = [r.get(key) for r in rs if r.get(key) is not None]
                d[key] = _mean_ci(v)
                if arm != "exact" and v:
                    e = [r.get(key) for r in ex if r.get(key) is not None]
                    d[key].update(vs_exact=_rel_diff(e, v), p=float(mannwhitneyu(e, v).pvalue))
            S["tumours"][arm] = d
        # every epsilon below 1/90 picks the paper's leap: are the runs identical, seed by seed?
        if "eps0.003" in tum and "paper" in tum:
            pp = {r["seed"]: r for r in tum["paper"]}
            keys = ("cancer", "time", "n0", "n5", "D0", "genotypes", "fga", "release", "leaps")
            same = [all(r.get(k) == pp[r["seed"]].get(k) for k in keys)
                    for r in tum["eps0.003"] if r["seed"] in pp]
            S["eps0.003_identical_to_paper"] = dict(n=len(same), identical=int(sum(same)))
    dem = _load(out, "deme")
    if dem:
        S["deme"] = {}
        for K in DEME_K:
            st = stationary_exact(K)
            d = S["deme"][f"K{K}"] = dict(analytic=float(np.dot(np.arange(len(st)), st)),
                                          analytic_sd=_sd_hist(st))
            for arm, rs in dem.items():
                rk = [r for r in rs if r["K"] == K]
                if rk:
                    d[arm] = dict(_mean_ci([r["mean"] for r in rk]),
                                  sd_of_size=_sd_hist(np.sum([r["hist"] for r in rk], axis=0)))
    fix = _load(out, "fix")
    if fix:
        S["fix"] = {}
        for arm, rs in fix.items():
            n, k = sum(r["n"] for r in rs), sum(r["fixed"] for r in rs)
            inv = sum(r["sum_inv_n0"] for r in rs) / n
            p, lo, hi = _wilson(k, n)
            tf = [x for r in rs for x in r["t_fixed"]]
            S["fix"][arm] = dict(n=n, fixed=k, p=p, E_inv_n0=inv, ratio=p / inv, ratio_lo=lo / inv,
                                 ratio_hi=hi / inv, t_fix=_mean_ci(tf),
                                 unresolved=n - k - sum(r["lost"] for r in rs))
    return S


# ---------------------------------------------------------------------------------- figure
def _ci_points(ax, xs, ds, arms, col, mk, key=None):
    for x, d, arm in zip(xs, ds, arms):
        if key is not None:
            d = d[key]
        if d.get("mean") is None:
            continue
        lo = d["lo"] if d.get("lo") is not None else d["mean"]
        hi = d["hi"] if d.get("hi") is not None else d["mean"]
        ax.errorbar(x, d["mean"], yerr=[[d["mean"] - lo], [hi - d["mean"]]], fmt=mk[arm], ms=2.4,
                    color=col[arm], lw=0.7, capsize=0)


def _key_mark(fig, S, x_mm, y_mm, color, marker):
    """A legend key (a marker on a short line) at an mm position from the top-left corner."""
    from matplotlib.lines import Line2D
    W, H = fig.get_size_inches() / S.MM
    fig.lines.append(Line2D([(x_mm - 1.5) / W, x_mm / W, (x_mm + 1.5) / W], [1 - y_mm / H] * 3,
                            transform=fig.transFigure, color=color, lw=0.7, marker=marker,
                            markevery=[1], ms=2.4, figure=fig))


def render(a):
    import style as S
    sm = summarise(a.out)
    with open(os.path.join(a.out, "s1_summary.json"), "w") as f:
        json.dump(sm, f, indent=1)
    S.setup()
    # one colour and one marker per arm (the marker carries identity where two colours are close)
    col = {"exact": S.INK, "paper": S.SOURCE["iscc"], "eps0.003": S.CLONE_FAMILIES[0][2],
           "eps0.03": S.CLONE_FAMILIES[3][1], "eps0.1": S.CLONE_FAMILIES[4][1],
           "critical": S.CLONE_FAMILIES[1][1], "historical": S.CLONE_FAMILIES[2][1]}
    mk = {"exact": "o", "paper": "o", "eps0.003": "o", "eps0.03": "v", "eps0.1": "^", "critical": "D",
          "historical": "s"}
    short = {"exact": "exact", "paper": "paper, ε 0.01", "eps0.003": "ε 0.003", "eps0.03": "ε 0.03",
             "eps0.1": "ε 0.1", "critical": "critical", "historical": "historical"}
    long = {"exact": "exact (Gillespie)", "paper": "paper: binomial, CGP ε 0.01",
            "eps0.03": "CGP ε 0.03", "eps0.1": "CGP ε 0.1", "critical": "critical channels exact",
            "historical": "historical: Poisson, fixed step"}
    order = ["exact", "paper", "eps0.03", "eps0.1", "critical", "historical"]
    fig = S.figure("double", 178)

    # the arms, once for the whole figure
    xk = 5.0
    for arm in order:
        _key_mark(fig, S, xk, 4.0, col[arm], mk[arm])
        S.text_mm(fig, xk + 2.6, 4.0, long[arm], fontsize=5.5, va="center")
        xk += 6.0 + 1.25 * len(long[arm])

    def xaxis(ax, arms):
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels([short[m] for m in arms], rotation=40, ha="right", fontsize=5)
        ax.set_xlim(-0.6, len(arms) - 0.4)

    # (a) one deme at K = 60: the time-averaged law of its size
    dem = _load(a.out, "deme")
    sd = sm.get("deme", {})
    S.panel_letter(fig, "a", 2, 9)
    ax = S.ax_mm(fig, 13, 15, 38, 31)
    K = 60
    st = stationary_exact(K)
    lo_n, hi_n = 36, 72
    nn = np.arange(lo_n, hi_n + 1)
    ax.bar(nn, st[lo_n:hi_n + 1], width=0.9, color=S.BG_CELLS, lw=0)
    for arm in order:
        rs = [r for r in dem.get(arm, []) if r["K"] == K]
        if not rs:
            continue
        h = np.sum([r["hist"] for r in rs], axis=0)
        h = h / h.sum()
        if arm == "exact":
            ax.plot(nn, h[lo_n:hi_n + 1], "o", ms=1.5, color=col[arm], zorder=5)
        else:
            ax.plot(nn, h[lo_n:hi_n + 1], "-", lw=0.7, color=col[arm])
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 1.5)
    ax.set_xlabel("Cells in the deme (K = 60)")
    ax.set_ylabel("Fraction of time")
    ax.set_title("One deme at capacity", loc="left")
    S.stat(ax, "bars: exact law", loc="upper left")

    # (b) the mean size of a deme at K, against the exact law
    S.panel_letter(fig, "b", 57, 9)
    ax = S.ax_mm(fig, 67, 15, 30, 31)
    arms_b = [m for m in order if m in dem]
    for j, arm in enumerate(arms_b):
        for K, dx in ((30, -0.17), (60, 0.17)):
            d = sd.get(f"K{K}", {}).get(arm)
            if d is None:
                continue
            an = sd[f"K{K}"]["analytic"]
            m, lo, hi = (100 * (d[x] - an) / K for x in ("mean", "lo", "hi"))
            ax.errorbar(j + dx, m, yerr=[[m - lo], [hi - m]], fmt=mk[arm], ms=2.4, color=col[arm],
                        mfc=col[arm] if K == 60 else "white", lw=0.7, capsize=0)
    ax.axhline(0, color=S.GREY, lw=0.4)
    xaxis(ax, arms_b)
    ax.set_ylabel("Mean size − exact law (% of K)")
    ax.plot([], [], "o", ms=2.4, color=S.GREY, label="K = 60")
    ax.plot([], [], "o", ms=2.4, color=S.GREY, mfc="white", label="K = 30")
    ax.legend(loc="lower left", fontsize=5)

    # (c) neutral fixation in a full deme of K = 30
    fx = sm.get("fix", {})
    S.panel_letter(fig, "c", 104, 9)
    ax = S.ax_mm(fig, 114, 15, 26, 31)
    arms_c = [m for m in order if m in fx]
    _ci_points(ax, range(len(arms_c)),
               [dict(mean=fx[m]["ratio"], lo=fx[m]["ratio_lo"], hi=fx[m]["ratio_hi"]) for m in arms_c],
               arms_c, col, mk)
    ax.axhline(1, color=S.GREY, lw=0.4)
    xaxis(ax, arms_c)
    ax.set_ylabel("P(fixation) / E[1/N₀]")
    ax.set_title("Neutral fixation, K = 30", loc="left")

    # (d) time to fixation
    S.panel_letter(fig, "d", 146, 9)
    ax = S.ax_mm(fig, 157, 15, 24, 31)
    _ci_points(ax, range(len(arms_c)), [fx[m] for m in arms_c], arms_c, col, mk, key="t_fix")
    if "exact" in fx:
        e = fx["exact"]["t_fix"]
        ax.axhspan(e["lo"], e["hi"], color=S.BG_CELLS, lw=0, zorder=0)
    xaxis(ax, arms_c)
    ax.set_ylabel("Time to fixation (generations)")

    # (e) tumours at 5,000 cells: every statistic of every tau arm against exact
    tum = sm.get("tumours", {})
    S.panel_letter(fig, "e", 2, 63)
    ax = S.ax_mm(fig, 47, 69, 84, 98)
    arms_e = [m for m in order if m in tum and m != "exact"]
    CLIP = 25.0
    for i, (key, lab) in enumerate(TUMOUR_STATS):
        y0 = len(TUMOUR_STATS) - 1 - i
        if i % 2 == 0:
            ax.axhspan(y0 - 0.5, y0 + 0.5, color=S.PASTEL["grey"], lw=0, zorder=0)
        for j, arm in enumerate(arms_e):
            d = tum[arm].get(key, {}).get("vs_exact")
            if d is None:
                continue
            y = y0 + 0.32 - j * 0.64 / max(len(arms_e) - 1, 1)
            m, lo, hi = 100 * d["rel"], 100 * d["lo"], 100 * d["hi"]
            ax.plot([max(lo, -CLIP), min(hi, CLIP)], [y, y], color=col[arm], lw=0.7)
            if abs(m) > CLIP:                        # off the scale: an arrowhead and the value
                s = np.sign(m)
                ax.plot(s * CLIP, y, ">" if s > 0 else "<", ms=2.6, color=col[arm], clip_on=False)
                ax.text(s * (CLIP + 1.5), y, f"{m:+.0f}%", fontsize=4.5, va="center",
                        ha="left" if s > 0 else "right")
            else:
                ax.plot(m, y, mk[arm], ms=2.4, color=col[arm])
    ax.axvspan(-5, 5, color=S.PASTEL["blue"], lw=0, zorder=0, alpha=0.6)
    ax.axvline(0, color=S.INK, lw=0.5)
    ax.set_xlim(-CLIP, CLIP)
    ax.set_ylim(-0.5, len(TUMOUR_STATS) - 0.5)
    ax.set_yticks(range(len(TUMOUR_STATS)))
    ax.set_yticklabels([lab for _, lab in TUMOUR_STATS][::-1], fontsize=5.5)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Difference from exact (% of the exact mean, 95% CI)")
    ex = tum.get("exact", {})
    ax.set_title(f"Paper-regime tumours at {TARGET:,} cells: {ex.get('n', 0):,} exact; "
                 f"in-situ phase on {ex.get('in_situ_followed', 0)}", loc="left")

    # (f) drivers per cell at the 5% detection limit, absolute
    arms_f = [m for m in order if m in tum]
    S.panel_letter(fig, "f", 146, 63)
    ax = S.ax_mm(fig, 157, 69, 24, 40)
    _ci_points(ax, range(len(arms_f)), [tum[m] for m in arms_f], arms_f, col, mk, key="n5")
    if "exact" in tum:
        e = tum["exact"]["n5"]
        ax.axhspan(e["lo"], e["hi"], color=S.BG_CELLS, lw=0, zorder=0)
    ax.set_xticks(range(len(arms_f)))
    ax.set_xticklabels([])
    ax.set_xlim(-0.6, len(arms_f) - 0.4)
    ax.set_ylabel("Drivers per cell, 5% detection")
    ax.set_title("At 5,000 cells", loc="left")

    # (g) wall time of growth to 5,000 cells, per tumour (median and quartiles; n above)
    S.panel_letter(fig, "g", 146, 114)
    ax = S.ax_mm(fig, 157, 120, 24, 40)
    for j, arm in enumerate(arms_f):
        w = tum[arm]["wall_s"]
        ax.errorbar(j, w["median"], yerr=[[w["median"] - w["q25"]], [w["q75"] - w["median"]]],
                    fmt=mk[arm], ms=2.4, color=col[arm], lw=0.7, capsize=0)
        ax.text(j, w["q75"] * 1.3, f"n = {tum[arm]['n']:,}", fontsize=4.2, ha="center", va="bottom",
                rotation=90)
    ax.set_yscale("log")
    ax.set_ylim(top=ax.get_ylim()[1] * 6)
    xaxis(ax, arms_f)
    ax.set_ylabel("Wall time to 5,000 cells (s)")
    path = S.save(fig, "supp_s1_tau", stamp=False)
    print("wrote", path)
    return sm


def main():
    global INSITU_SEEDS
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--nproc", type=int, default=1)
    r.add_argument("--tumours", default="", help="arm:n_seeds,...")
    r.add_argument("--deme", default="", help="arm:n_replicates (per K),...")
    r.add_argument("--fix", default="", help="arm:n_replicates,...")
    r.add_argument("--insitu-seeds", type=int, default=INSITU_SEEDS,
                   help="seeds 1..N are followed past the target to the first stromal cell")
    g = sub.add_parser("render")
    g.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "run":
        INSITU_SEEDS = a.insitu_seeds            # read by run_tumour in the forked workers
        run(a)
    else:
        render(a)


if __name__ == "__main__":
    main()
