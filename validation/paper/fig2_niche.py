"""Figure 2 and S10 Fig: selection depends on the niche, and the tumours resemble real ones.

Fig 2, drawn at half a page (165 by at most 114 mm) by ``render`` below:
  (A) the showcase (R0 seed 1) up to metastasis: five snapshots coloured by clone, and the metastasis;
  (B) traits are selected where they pay (R12 seeds 1-10 against R13's three controls, paired by seed):
      the share of cancer cells carrying each trait in each compartment, in the selected tumour and in
      the control without that trait's selection, side by side (the log2 enrichment is kept in the
      numbers file only);
  (C) metastatic seeding across R12's 20 seeds: the lineages founding each metastasis, and the release,
      first invasion and seeding as fractions of the primary's age;
  (D) against real tumours: fig2_realism.py's panel a, from summaries.py's caches of R1-R4.
S10 Fig (``supp_niche_extra.py``) draws the panels moved out of it with the functions below:
  (A) the showcase's cancer cells in gland, stroma and metastasis over generations, with clone births,
      the first invasion and the first seeding of the metastasis;
  (B, C) escape from the gland across seeds (R10, with barriers against without): generations from
      founding to the first invasion, and how many clones reach the stroma within a window after it;
  (D) confinement shapes the genome (R3): truncal copy-number fraction and clonal diversity, with the
      definitions of the old Fig 4f-g (fig4_space_lineage.py);
  (E) the showcase's clone tree, branches coloured by where their cells live, seeding marked;
  (F) whole-genome-duplication prevalence across rates (fig2_realism.py's panel c, R5).
The speed panel of the earlier Fig 2 is S4 Fig (supp_runtime.py), which reads the same runs from
``comparisons/results/runtime.csv``.

Definitions used throughout:
  * compartments: gland = primary demes of a gland (lumen or wall, ``gland_id >= 0``); stroma =
    primary demes with ``gland_id == -1``; metastasis = the demes of the metastatic lattice (index
    ``>= n_primary_demes``; their ``gland_id`` is also -1, so the split needs both);
  * first invasion: the first snapshot of the spatial history with a cancer cell in a stroma deme (its
    resolution is the snapshot interval, 8 leaps or about 4 generations in R10 and R12);
  * clones: ``common.Clones`` (clades of the true genealogy holding >= 4% of the cancer cells);
  * clones reaching the stroma (S10 C): for every cancer cell in a stroma deme at a snapshot within
    WINDOW generations of the first invasion, the lineage that crossed (its genotype or, if that was
    born in the stroma, its first ancestor born outside it); the count is of the distinct clones of
    those crossing genotypes, clones defined on the whole tumour at the end of the window ("ancestral"
    counts as one). Clones born inside the stroma after a crossing therefore do not add to it;
  * carrier share (Fig 2B): cancer cells with the trait's level > 0 over cancer cells, per compartment, at
    the end of growth; the enrichment in the numbers file is log2 of the selected share over the
    control's at the same seed, with one pseudo-cell added to each side;
  * founding lineage of the metastasis (Fig 2C): for each live metastatic cancer cell, its genotype or the
    first ancestor born in the primary (the genotype that crossed; a mutant born on arrival has its
    parent as the crossing genotype); seeding time = the first seeding event of any founding lineage;
    the number of founding lineages is their effective number, 1 / sum p_i^2 over the metastasis'
    cells (threshold-free: late single-cell arrivals barely count), by genotype and by clone.

Each part is computed once into a small JSON cache under ``<OUT>/derived/fig2/niche_<part>.json``,
one fresh process per tumour, so no process holds two tumours; the render reads only the caches (and
summaries.py's), so every number drawn can be recomputed from them. A part skips tumours already in
its cache, so rerunning it after more tumours have grown adds only the new ones.

Usage (from the iscc repository root, with ISCC_PAPER_OUT and ISCC_PAPER_TIER set as for the registry):
    python validation/paper/fig2_niche.py --part a [--source R0/seed1]
    python validation/paper/fig2_niche.py --part b|c|d|e [--jobs N] [--force]
    python validation/paper/fig2_niche.py                   # render Fig 2 from the caches
    python validation/paper/supp_niche_extra.py             # render S10 Fig from the caches
The cache parts keep their old letters (a: the showcase, b: escape, c: traits, d: confinement, e:
seeding). Fig 2D and S10 Fig F read ``summaries.py``'s caches: ``python validation/paper/summaries.py
R1,R2,R3,R4,R5``.
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import common as C  # noqa: E402
import registry as REG  # noqa: E402
import regime as RG  # noqa: E402
import style as S  # noqa: E402

CACHE = os.path.join(C.DERIVED, "fig2")
COMPS = ("gland", "stroma", "met")
TRAITS = ("breach", "stromal_survival", "met_survival")
# R13's control without each trait's selection (registry.py): breach gate off, stromal hazard off, host
# hazard off with every seeding cell surviving transit.
CONTROL = {"breach": "no_breach_sel", "stromal_survival": "no_stromal_sel", "met_survival": "no_met_sel"}
WINDOW = 10.0            # generations after the first invasion in which crossing lineages count (b ii);
# about a third of the median time from first invasion to the target size in R10's barrier arm
SHOWCASE = "R0/seed1"
STAND_IN = "R12/seed1"   # the showcase panels are built and tested on it until R0 is grown
# Line colours of the three compartments (darker versions of style.COMPARTMENT's tissue fills) and of
# where a clone's cells live in the tree of S10 Fig E.
LINE = {"gland": S.COMPARTMENT["epithelium"], "stroma": "#B8875A", "met": "#3E7CB1"}
WHERE = {"primary": "#8C8C8C", "met": LINE["met"], "both": "#B2182B"}


# --- per-tumour readouts ---------------------------------------------------------------------------
def compartments(t):
    """Compartment of every deme: 0 gland (lumen or wall), 1 stroma, 2 metastasis."""
    n, npd = len(t.demes), t.n_primary_demes
    gland = np.asarray(t.gland_id)[:n] if t.gland_id is not None else np.full(n, -1)
    comp = np.where(gland >= 0, 0, 1)
    comp[npd:] = 2
    return comp


def ord_tables(t):
    """Genotype id and cancer flag of every creation ordinal (the key of the spatial history)."""
    n = max(rep.ord for rep in t.genotypes.values()) + 1
    gid = np.empty(n, dtype=object)
    cancer = np.zeros(n, dtype=bool)
    for g, rep in t.genotypes.items():
        gid[rep.ord] = g
        cancer[rep.ord] = rep.type == "cancer"
    return gid, cancer


def history(t, comp, cancer):
    """(times, cancer cells per compartment) at every stored spatial snapshot."""
    times, cells = [], []
    for s in t.spatial_history:
        m = cancer[s["genotype"]]
        times.append(float(s["time"]))
        cells.append(np.bincount(comp[s["deme"][m]], weights=s["count"][m], minlength=3))
    return np.asarray(times), np.asarray(cells).reshape(-1, 3)


def first_invasion(times, cells):
    """(index, time, previous snapshot's time) of the first snapshot with cancer in the stroma."""
    hit = np.nonzero(cells[:, 1] > 0)[0]
    if not len(hit):
        return None, None, None
    k = int(hit[0])
    return k, float(times[k]), float(times[k - 1]) if k else 0.0


def release_time(t):
    rel = [e["time"] for e in t.events if e.get("event") == "origin_release"]
    return float(rel[0]) if rel else None


def _parent(par, g):
    p = par.get(g)
    return None if p in (None, "None", g) else p


def met_founders(t):
    """``{founding genotype: live metastatic cancer cells}`` and ``{genotype: first seeding time}``.

    The founding genotype of a metastatic cell is its genotype or, if that was born in the metastasis,
    the first ancestor born in the primary: the genotype that crossed (a mutant realised on arrival
    has its parent as the crossing genotype, which is what the seeding event records)."""
    npd = t.n_primary_demes
    par = t.genotypes_parents
    memo = {}

    def founder(g):
        chain = []
        while g not in memo:
            bd = t.genotypes[g].birth_deme
            if bd is None or bd < npd:
                memo[g] = g
                break
            chain.append(g)
            p = _parent(par, g)
            if p is None:
                memo[g] = g
                break
            g = p
        f = memo[g]
        for c in chain:
            memo[c] = f
        return f

    out = Counter()
    for i in range(npd, len(t.demes)):
        for g, c in t.demes[i].items():
            if t._is_cancer(g):
                out[founder(g)] += c
    first = {}
    for e in t.events:
        if e.get("event") == "seeding" and e["genotype"] not in first:
            first[e["genotype"]] = float(e["time"])
    return out, first


def seeding_summary(t, cl):
    """Seeding of the metastasis: founding lineages with their cells, first seeding and clone."""
    founders, first = met_founders(t)
    rows = sorted(([int(c), first.get(f), cl.of(f), f] for f, c in founders.items()), key=lambda r: -r[0])
    seed_times = [r[1] for r in rows if r[1] is not None]
    events = [e for e in t.events if e.get("event") == "seeding"]
    est = None
    if t.traces and "met_counts" in t.traces[0]:
        for i, tr in enumerate(t.traces):
            if sum(c for g, c in tr["met_counts"].items() if t._is_cancer(g)) >= 50:
                est = float(t.trace_times[i]) if i < len(t.trace_times) else None
                break
    return dict(founders=[r[:3] for r in rows[:300]], n_founders=len(rows),
                met_cancer=int(sum(founders.values())),
                t_seed=min(seed_times) if seed_times else None,
                t_first_event=float(events[0]["time"]) if events else None,
                n_seed_events=len(events), n_migrants=int(sum(e.get("n", 1) for e in events)),
                t_established50=est, founder_ids=[r[3] for r in rows[:300]])


def meta(t):
    return dict(time=float(t.time), cancer=int(t.get_cancer_size()),
                reached_target=bool(getattr(t, "reached_target", True)), release=release_time(t))


# --- part a: the showcase's history and clone tree -------------------------------------------------
def _dominant(deme, lab, count, total, offset, n_lab):
    """Sparse map rows [local deme, clone index, alpha %] of the dominant clone of each deme."""
    if not len(deme):
        return []
    key = deme.astype(np.int64) * n_lab + lab
    uk, inv = np.unique(key, return_inverse=True)
    sums = np.bincount(inv, weights=count)
    d, lb = uk // n_lab, uk % n_lab
    order = np.lexsort((-sums, d))
    d, lb, sums = d[order], lb[order], sums[order]
    first = np.r_[True, d[1:] != d[:-1]]
    ncan = np.bincount(np.searchsorted(np.unique(d), d), weights=sums)
    rows = []
    for dd, ll, nc in zip(d[first], lb[first], ncan):
        a = 0.25 + 0.75 * nc / max(total[dd], 1)
        rows.append([int(dd - offset), int(ll), int(round(100 * min(a, 1.0)))])
    return rows


def _maps(t, comp, gid, cancer, lab_index, cl, deme, ords, count):
    """Primary and metastasis maps of one state (a snapshot's arrays, or the live state)."""
    npd = t.n_primary_demes
    total = np.bincount(deme, weights=count, minlength=len(t.demes))
    m = cancer[ords]
    d, o, c = deme[m], ords[m], count[m]
    cache = {}
    lab = np.empty(len(o), dtype=np.int64)
    for i, oo in enumerate(o):
        if oo not in cache:
            cache[oo] = lab_index[cl.of(gid[oo])]
        lab[i] = cache[oo]
    n_lab = len(lab_index)
    p = d < npd
    return (_dominant(d[p], lab[p], c[p], total, 0, n_lab),
            _dominant(d[~p], lab[~p], c[~p], total, npd, n_lab))


def _live_arrays(t):
    deme, ords, count = [], [], []
    for di, dm in enumerate(t.demes):
        for g, c in dm.items():
            deme.append(di)
            ords.append(t.genotypes[g].ord)
            count.append(c)
    return np.asarray(deme), np.asarray(ords), np.asarray(count, dtype=float)


def clone_tree(t, cl, founders, first_seed):
    """The clones' genealogy: clade roots joined at their branch points (unary nodes contracted).

    Each node carries its birth time and the live cancer cells of its subtree in the primary and the
    metastasis; ancestral cells (outside every clone) count at their nearest node above. Founding
    lineages of the metastasis are attached to their clone, or to their nearest node above."""
    par = cl.parents
    below = defaultdict(set)
    top = None
    for r in cl.roots:
        cur = r
        while True:
            p = _parent(par, cur)
            if p is None:
                break
            below[p].add(cur)
            cur = p
        top = cur
    kept = set(cl.roots) | {g for g, ch in below.items() if len(ch) >= 2} | ({top} if top else set())

    memo = {}

    def up(g):
        """Nearest kept ancestor-or-self of g (None above the tree)."""
        chain = []
        while g is not None and g not in kept and g not in memo:
            chain.append(g)
            g = _parent(par, g)
        res = memo.get(g, g)
        for c in chain:
            memo[c] = res
        return res

    nodes = {}
    for g in kept:
        p = _parent(par, g)
        rep = t.genotypes.get(g)
        bt = rep.birth_time if rep is not None and rep.birth_time is not None else 0.0
        nodes[g] = dict(id=g, parent=up(p) if p is not None else None, time=float(bt),
                        label=cl.label.get(g), prim=0, met=0)
    npd = t.n_primary_demes
    for i, dm in enumerate(t.demes):
        key = "met" if i >= npd else "prim"
        for g, c in dm.items():
            if not t._is_cancer(g):
                continue
            lab = cl.of(g)
            n = cl.roots[cl.labels.index(lab)] if lab != C.ANCESTRAL else up(g)
            if n is not None:
                nodes[n][key] += c
    def depth(g):
        k = 0
        while nodes[g]["parent"] is not None:
            g, k = nodes[g]["parent"], k + 1
        return k
    order = sorted(nodes, key=lambda g: -depth(g))                  # children before parents
    sub = {g: [nodes[g]["prim"], nodes[g]["met"]] for g in nodes}
    for g in order:
        p = nodes[g]["parent"]
        if p is not None:
            sub[p][0] += sub[g][0]
            sub[p][1] += sub[g][1]
    for g in nodes:
        nodes[g]["sub_prim"], nodes[g]["sub_met"] = sub[g]
    seeds = []
    for f, c in founders.items():
        lab = cl.of(f)
        n = cl.roots[cl.labels.index(lab)] if lab != C.ANCESTRAL else up(f)
        seeds.append([n, first_seed.get(f), int(c)])
    seeds.sort(key=lambda r: -r[2])
    return dict(nodes=list(nodes.values()), top=top, seeds=seeds[:300])


def compute_a(source):
    run, task = source.split("/")
    t = REG.load(run, task)
    cl = C.Clones(t)
    comp = compartments(t)
    gid, cancer = ord_tables(t)
    labels = cl.labels + [C.ANCESTRAL]
    lab_index = {lab: i for i, lab in enumerate(labels)}
    times, cells = history(t, comp, cancer)
    k_inv, t_inv, _ = first_invasion(times, cells)
    rel = release_time(t)
    seeding = seeding_summary(t, cl) if t.n_primary_demes < len(t.demes) else None
    end = float(t.time)
    t_seed = seeding["t_seed"] if seeding else None
    snaps = t.spatial_history

    def nearest(x):
        return int(np.argmin(np.abs(times - x)))

    moments = []
    if rel is not None:
        before = np.nonzero(times < rel)[0]
        moments.append(("acinus closed", int(before[-1]) if len(before) else 0))
        moments.append(("acinus opens", nearest(rel + 0.5 * ((t_inv if t_inv else end) - rel))))
    if k_inv is not None:
        moments.append(("first breach", k_inv))
        pre = np.nonzero(times < t_seed)[0] if t_seed else []
        k_grow = int(pre[-1]) if len(pre) and pre[-1] > k_inv else nearest(0.5 * (t_inv + end))
        moments.append(("invasive growth", k_grow))
    maps = []
    for name, k in moments:
        s = snaps[k]
        pm, mm = _maps(t, comp, gid, cancer, lab_index, cl, s["deme"].astype(np.int64), s["genotype"],
                       s["count"].astype(float))
        maps.append(dict(name=name, time=float(times[k]), primary=pm, met=mm))
    d, o, c = _live_arrays(t)
    pm, mm = _maps(t, comp, gid, cancer, lab_index, cl, d, o, c)
    maps.append(dict(name="after seeding" if t_seed else "end", time=end, primary=pm, met=mm))
    # the closed acinus is one deme of the field, so its snapshot is cropped to the founding gland; the
    # later ones show the whole field
    G = t.grid_size
    gl = np.asarray(t.gland_id)[:G * G]
    founding = int(t.genotypes[t.founder_id].birth_deme)
    rc = np.asarray(t.deme_coords)[:G * G][gl == gl[founding]] if gl[founding] >= 0 \
        else np.asarray([t.deme_coords[founding]])
    r0, r1, c0, c1 = rc[:, 0].min(), rc[:, 0].max(), rc[:, 1].min(), rc[:, 1].max()
    side = int(min(G, max(r1 - r0, c1 - c0) + 7))
    rs = int(min(max((r0 + r1) // 2 - side // 2, 0), G - side))
    cs = int(min(max((c0 + c1) // 2 - side // 2, 0), G - side))
    crop = [rs, rs + side, cs, cs + side]
    # where the cancer stood in the stroma at the first breach (ringed on that snapshot)
    breach = []
    if k_inv is not None:
        s = snaps[k_inv]
        m = cancer[s["genotype"]] & (comp[s["deme"]] == 1)
        breach = sorted(set(int(x) for x in s["deme"][m]))
    # tissue: 0 stroma, 1 wall, 2 lumen; outline = gland demes touching stroma (drawn over the cancer)
    lumen = set(int(x) for x in np.concatenate([np.asarray(x) for x in t.gland_lumen_demes])) \
        if t.gland_lumen_demes else set()
    tissue = [0 if gl[i] < 0 else (2 if i in lumen else 1) for i in range(G * G)]
    outline = [int(i) for i in np.nonzero(gl >= 0)[0]
               if any(nb < G * G and gl[nb] == -1 for nb in t._neighbors(int(i)))]
    births = [[lab, float(t.genotypes[g].birth_time or 0.0)] for lab, g in zip(cl.labels, cl.roots)]
    founders, first = met_founders(t) if seeding else ({}, {})
    out = dict(source=source, grid=int(G), met_grid=int(getattr(t, "met_grid_size", 0) or 0),
               labels=labels, colours=[cl.colour[lab] for lab in labels], sizes=cl.size,
               hist=dict(time=times.round(2).tolist(), gland=cells[:, 0].astype(int).tolist(),
                         stroma=cells[:, 1].astype(int).tolist(), met=cells[:, 2].astype(int).tolist()),
               snapshot_every=int(t.spatial_snapshot_every), t_inv=t_inv, end=end,
               seeding={k: v for k, v in (seeding or {}).items() if k != "founder_ids"} or None,
               births=births, maps=maps, crop=crop, breach_demes=breach, tissue=tissue, outline=outline,
               tree=clone_tree(t, cl, founders, first), **meta(t))
    return out


# --- parts b-e: one record per tumour --------------------------------------------------------------
def crossers(t, comp):
    """The lineage that crossed into the stroma, for any genotype: the genotype itself or, if it was
    born in a stroma deme, its first ancestor born outside the stroma (a mutant realised in the stroma
    on a crossing division has its parent as the crossing genotype)."""
    par = t.genotypes_parents
    memo = {}

    def f(g):
        chain = []
        while g not in memo:
            bd = t.genotypes[g].birth_deme
            if bd is None or comp[bd] != 1:
                memo[g] = g
                break
            chain.append(g)
            p = _parent(par, g)
            if p is None:
                memo[g] = g
                break
            g = p
        r = memo[g]
        for c in chain:
            memo[c] = r
        return r
    return f


def rec_b(t):
    """R10: the first invasion, and the clones of the lineages that crossed into the stroma within
    WINDOW generations of it (clones born inside the stroma after a crossing do not add to the count)."""
    comp = compartments(t)
    gid, cancer = ord_tables(t)
    times, cells = history(t, comp, cancer)
    k_inv, t_inv, t_prev = first_invasion(times, cells)
    out = dict(meta(t), t_inv=t_inv, t_inv_prev=t_prev, snapshot_every=int(t.spatial_snapshot_every),
               window=WINDOW, n_snapshots=len(times))
    if k_inv is None:
        return out
    snaps = t.spatial_history
    ks = [k for k in range(k_inv, len(snaps)) if times[k] <= t_inv + WINDOW]
    s = snaps[ks[-1]]
    m = cancer[s["genotype"]]
    counts = Counter()
    for o, c in zip(s["genotype"][m], s["count"][m]):
        counts[gid[o]] += int(c)
    cross = crossers(t, comp)
    # stroma cells of each crossing genotype: the most at one snapshot of the window, and at its end
    peak, last = Counter(), Counter()
    for k in ks:
        s = snaps[k]
        m = cancer[s["genotype"]] & (comp[s["deme"]] == 1)
        per = Counter()
        for o, c in zip(s["genotype"][m], s["count"][m]):
            per[cross(gid[o])] += int(c)
        for g, c in per.items():
            peak[g] = max(peak[g], c)
        if k == ks[-1]:
            last = per
    # the clone definition applied to the tumour at the end of the window, and to the grown tumour
    for name, cl in (("then", C.Clones(t, counts=counts)), ("final", C.Clones(t))):
        pk, ls = Counter(), Counter()
        for g, c in peak.items():
            pk[cl.of(g)] += c
        for g, c in last.items():
            ls[cl.of(g)] += c
        out[f"cross_{name}"] = dict(pk)             # clone -> peak stroma cells of its crossing lineages
        out[f"cross_{name}_end"] = dict(ls)         # ... at the end of the window
        out[f"n_clones_{name}"] = len(cl.labels)
    out["n_crossing_genotypes"] = len(peak)
    out["crossing_genotypes_end"] = sum(1 for c in last.values() if c > 0)
    out["window_end"] = float(times[ks[-1]])
    out["cancer_at_window_end"] = int(sum(counts.values()))
    out["stroma_at_window_end"] = int(sum(last.values()))
    return out


def rec_c(t):
    """R12 / R13: cancer cells and trait carriers (level > 0) in each compartment, at the end."""
    comp = compartments(t)
    out = {k: dict(n=0, **{tr: 0 for tr in TRAITS}) for k in COMPS}
    for di, dm in enumerate(t.demes):
        k = COMPS[comp[di]]
        for g, c in dm.items():
            if not t._is_cancer(g):
                continue
            ep = t.genotypes[g].evolutionary_parameters
            out[k]["n"] += int(c)
            for tr in TRAITS:
                if ep.get(tr, 0.0) > 0:
                    out[k][tr] += int(c)
    return dict(meta(t), comp=out)


def rec_d(t):
    """R3: the old Fig 4f-g's truncal copy-number fraction and Noble's D at the 5% detection limit."""
    sys.path.insert(0, os.path.dirname(HERE))
    import sweep_score as SS
    from iscc.inference import mode_indices
    sc = SS.score_tumour(t)
    return dict(meta(t), trunk_fraction=sc["n_truncal_segments"] / float(t.n_segments),
                D=float(mode_indices(t, detection=0.05)["D"]))


def rec_e(t):
    """R12: seeding of the metastasis, with the release and the first invasion for scale."""
    comp = compartments(t)
    gid, cancer = ord_tables(t)
    times, cells = history(t, comp, cancer)
    _, t_inv, _ = first_invasion(times, cells)
    sm = seeding_summary(t, C.Clones(t))
    sm.pop("founder_ids", None)
    prim = sum(c for i in range(t.n_primary_demes) for g, c in t.demes[i].items() if t._is_cancer(g))
    return dict(meta(t), t_inv=t_inv, primary_cancer=int(prim), **sm)


PARTS = {"b": (rec_b, lambda: [("R10", k) for k in REG.tasks("R10")]),
         "c": (rec_c, lambda: [("R12", f"seed{s}") for s in REG.SEEDS] + [("R13", k) for k in REG.tasks("R13")]),
         "d": (rec_d, lambda: [("R3", k) for k in REG.tasks("R3")]),
         "e": (rec_e, lambda: [("R12", k) for k in REG.tasks("R12")])}


def cache_path(part):
    return os.path.join(CACHE, f"niche_{part}.json")


def cached(part):
    p = cache_path(part)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def _write(part, obj):
    os.makedirs(CACHE, exist_ok=True)
    p = cache_path(part)
    with open(p + ".part", "w") as f:
        json.dump(obj, f)
    os.replace(p + ".part", p)


def _one(part, key):
    """One tumour's record, printed as the last line of stdout (run as ``--one <part> <run/task>``)."""
    run, task = key.split("/")
    rec = PARTS[part][0](REG.load(run, task))
    print(json.dumps(rec), flush=True)


def compute_part(part, jobs, force):
    """Each tumour in its own Python process (threads only dispatch), so no process holds two tumours,
    and a process killed for memory reports an error instead of hanging a pool."""
    import subprocess
    from concurrent.futures import ThreadPoolExecutor, as_completed
    have = (cached(part) or {}) if not force else {}
    todo = [f"{r}/{k}" for r, k in PARTS[part][1]()
            if os.path.exists(REG.task_path(r, k)) and f"{r}/{k}" not in have]
    print(f"part {part}: {len(have)} cached, {len(todo)} to compute", flush=True)

    def run_one(key):
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "--one", part, key],
                           capture_output=True, text=True)
        if p.returncode:
            return key, None, f"exit {p.returncode}: {p.stderr[-1500:]}"
        try:
            return key, json.loads(p.stdout.strip().splitlines()[-1]), None
        except Exception as e:
            return key, None, f"unreadable output ({e}): {p.stdout[-500:]}"

    with ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
        for fut in as_completed([ex.submit(run_one, key) for key in todo]):
            key, rec, err = fut.result()
            if err:
                print(f"  !! {key}: {err}", flush=True)
                continue
            have[key] = rec
            _write(part, have)
            print(f"  {key}: ok", flush=True)


# --- render ----------------------------------------------------------------------------------------
# Each panel draws into axes its caller places, so the main figure (render below) and S10 Fig
# (supp_niche_extra.py) lay out the same panels. No text is set below 5.5 pt, the paper's minimum at
# the size the figures are printed.
SMALL = 5.5
NAME = {"gland": "gland", "stroma": "stroma", "met": "metastasis"}


def pending(ax, msg="pending: registry run not grown yet"):
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=SMALL, color=S.GREY, transform=ax.transAxes)
    S.despine(ax, False, False)


def below(ax, mm):
    """A point ``mm`` below the bottom of ``ax``, in its axes fraction (to anchor a legend under it)."""
    H = ax.figure.get_size_inches()[1] / S.MM
    return -mm / (ax.get_position().height * H)


def _hexrgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])


def _rgb(a, rows, primary=True):
    """RGB image of one map from the cached sparse rows."""
    if primary:
        G = a["grid"]
        tis = np.asarray(a["tissue"]).reshape(G, G)
        rgb = np.zeros((G, G, 3))
        for code, key in ((0, "stroma"), (1, "epithelium"), (2, "lumen")):
            rgb[tis == code] = _hexrgb(S.COMPARTMENT[key])
    else:
        G = a["met_grid"]
        rgb = np.tile(_hexrgb(S.COMPARTMENT["met"]), (G, G, 1))
    cols = [_hexrgb(c) for c in a["colours"]]
    for d, lab, alpha in rows:
        r, c = divmod(d, G)
        al = alpha / 100.0
        rgb[r, c] = (1 - al) * rgb[r, c] + al * cols[lab]
    if primary:
        ink = _hexrgb(S.INK)
        for d in a["outline"]:
            r, c = divmod(d, G)
            rgb[r, c] = 0.5 * rgb[r, c] + 0.5 * ink
    return rgb


def _frame(ax, colour=S.GREY, lw=0.4):
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(lw)
        sp.set_color(colour)


ZOOM = 24                                    # demes on the side of a zoomed snapshot


def render_snapshots(fig, x, y, size, gap):
    """Fig 2A: the showcase at five moments coloured by clone, and its metastasis at the last, as one
    row of ``size`` mm maps from (x, y). The rings and the dotted box are explained in the caption."""
    from matplotlib.patches import Rectangle
    a = cached("a")
    axes = [S.ax_mm(fig, x + i * (size + gap), y, size, size) for i in range(6)]
    axes, axm = axes[:5], axes[5]
    if a is None:
        for ax in axes + [axm]:
            pending(ax, "pending: R0")
        return None
    G = a["grid"]
    tis = np.asarray(a["tissue"])

    def window(r, c, side=ZOOM):
        rs = int(min(max(round(r) - side // 2, 0), G - side))
        cs = int(min(max(round(c) - side // 2, 0), G - side))
        return [rs, rs + side, cs, cs + side]

    # Until the tumour spreads it fills the glands and little else, so the whole field looks the same from
    # the acinus opening to invasive growth. The closed acinus is cropped to the founding gland, the first
    # invasion to its breach and invasive growth to the stroma demes holding cancer then (the invasion
    # that grew, near another gland: the first breach did not persist); the whole field at the acinus
    # opening marks the three crops.
    names = [m["name"] for m in a["maps"]]
    crops = {0: a["crop"]}
    if a.get("breach_demes") and "first breach" in names:
        crops[names.index("first breach")] = window(*divmod(a["breach_demes"][0], G))
    if "invasive growth" in names:
        k = names.index("invasive growth")
        d = np.asarray([dd for dd, _, _ in a["maps"][k]["primary"] if tis[dd] == 0], dtype=int)
        if len(d):
            rr, cc = np.divmod(d, G)
            crops[k] = window(rr.mean(), cc.mean())
    # each map is titled with the state it shows: the second is a snapshot after the acinus opened, so
    # it reads "acinus open", not "opens"; the first breach is what the text calls the first invasion
    state = {"acinus opens": "acinus open", "first breach": "first invasion"}
    for i, (ax, m) in enumerate(zip(axes, a["maps"])):
        img = _rgb(a, m["primary"])
        r0, r1, c0, c1 = crops.get(i, [0, G, 0, G])
        ax.imshow(img[r0:r1, c0:c1], interpolation="nearest")
        _frame(ax)
        ax.set_title(f"{state.get(m['name'], m['name'])}\n{m['time']:.0f} generations", linespacing=1.1)
        if i in crops:
            ax.text(0.04, 0.96, str(i + 1), transform=ax.transAxes, ha="left", va="top", fontsize=SMALL,
                    color=S.INK)
        if i == 0:                                    # the closed acinus: ring its cancer demes
            rr, cc = np.divmod(np.asarray([d for d, _, _ in m["primary"]], dtype=int), G)
            ax.scatter(cc - c0, rr - r0, s=18, facecolors="none", edgecolors=S.INK, lw=0.6)
        if m["name"] == "first breach" and a.get("breach_demes"):
            rr, cc = np.divmod(np.asarray(a["breach_demes"]), G)
            ax.scatter(cc - c0, rr - r0, s=18, facecolors="none", edgecolors=LINE["stroma"], lw=0.6)
    for i, (q0, q1, p0, p1) in crops.items():
        axes[1].add_patch(Rectangle((p0 - 0.5, q0 - 0.5), p1 - p0, q1 - q0, fill=False, lw=0.4, ec=S.INK,
                                    ls=":"))
        axes[1].text(p1 + 1, q0, str(i + 1), ha="left", va="top", fontsize=SMALL, color=S.INK)
    if a["met_grid"]:
        axm.imshow(_rgb(a, a["maps"][-1]["met"], primary=False), interpolation="nearest")
        _frame(axm, LINE["met"], 0.6)
        axm.set_title("metastasis", color=LINE["met"])
    else:
        axm.axis("off")
    if a["source"] != SHOWCASE:
        S.text_mm(fig, x + 30, y - 7, f"stand-in: {a['source'].replace('/', ' ')} (R0 not grown yet)",
                  fontsize=SMALL, color="#B2182B", va="bottom")
    sd = a["seeding"] or {}
    return dict(source=a["source"], release=a["release"], first_breach=a["t_inv"], seeding=sd.get("t_seed"),
                end=a["end"], n_clones=len(a["labels"]) - 1, met_cancer=sd.get("met_cancer"),
                snapshot_every=a["snapshot_every"], breach_demes=len(a.get("breach_demes") or []),
                moments=[(m["name"], round(m["time"], 1)) for m in a["maps"]])


def render_history(ax):
    """S10 Fig A: the showcase's cancer cells in each compartment over generations, with clone births,
    the opening of the acinus, the first invasion and the seeding of the metastasis."""
    a = cached("a")
    if a is None:
        return pending(ax, "pending: R0")
    h = a["hist"]
    tt = np.asarray(h["time"])
    for k in COMPS:
        v = np.asarray(h[k], dtype=float)
        if v.max() <= 0:
            continue
        v[v <= 0] = np.nan
        ax.plot(tt, v, color=LINE[k], lw=1.0, marker=".", ms=1.6, label=NAME[k])
    ax.set_yscale("log")
    ax.set_xlim(0, a["end"] * 1.01)
    ax.set_xlabel("Generations since founding")
    ax.set_ylabel("Cancer cells")
    ymax = ax.get_ylim()[1]
    col = dict(zip(a["labels"], a["colours"]))
    for lab, bt in a["births"]:
        ax.plot(bt, ymax * 0.7, "v", ms=3, color=col[lab], mec=S.INK, mew=0.3, clip_on=False)
    ax.plot([], [], "v", ms=3, color=S.GREY, mec=S.INK, mew=0.3, ls="none", label="clone birth")
    marks = [("acinus opens", a["release"], S.GREY, ":"), ("first invasion", a["t_inv"], LINE["stroma"], "-"),
             ("seeding", (a["seeding"] or {}).get("t_seed"), LINE["met"], "-")]
    for name, x, c, ls in marks:
        if x is None:
            continue
        ax.axvline(x, color=c, lw=0.6, ls=ls, zorder=0)
        ax.text(x, ymax * 1.6, name, fontsize=SMALL, color=c, ha="center", va="bottom", clip_on=False)
    ax.legend(loc="upper left", ncol=4)
    return dict(release=a["release"], first_invasion=a["t_inv"], seeding=(a["seeding"] or {}).get("t_seed"),
                births=a["births"])


def render_escape(ax1, ax2):
    """S10 Fig B and C (R10): generations from founding to the first invasion, with barriers and
    without, and how many clones cross into the stroma within WINDOW generations of it."""
    d = cached("b")
    if not d:
        pending(ax1)
        pending(ax2, "")
        return None
    arms = {"barriers": [], "open": []}
    for key, r in d.items():
        arms[key.split("/")[1].split("_")[0]].append(r)
    col = {"barriers": S.SOURCE["iscc"], "open": S.SOURCE["real"]}
    name = {"barriers": "with barriers", "open": "without barriers"}      # the paper's terms for the arms
    out = {}
    rel = [r["release"] for rs in arms.values() for r in rs if r.get("release") is not None]
    for arm, rs in arms.items():
        if not rs:
            continue
        ti = np.sort([r["t_inv"] for r in rs if r.get("t_inv") is not None])
        n = len(rs)
        if len(ti):
            xs = np.r_[0, ti, max(r["time"] for r in rs)]
            ys = np.r_[0, np.arange(1, len(ti) + 1) / n, len(ti) / n]
            ax1.step(xs, ys, where="post", color=col[arm], lw=1.0, label=name[arm])
        out[arm] = dict(n=n, invaded=len(ti), median=float(np.median(ti)) if len(ti) else None,
                        range=(float(ti.min()), float(ti.max())) if len(ti) else None,
                        median_after_release=float(np.median(ti - np.median(rel))) if len(ti) and rel else None)
    if rel:
        ax1.axvline(np.median(rel), color=S.GREY, lw=0.5, ls=":")
        ax1.text(np.median(rel), 1.02, "acinus\nopens", fontsize=SMALL, color=S.GREY, ha="right", va="bottom",
                 transform=ax1.get_xaxis_transform())
        ax1.set_xlim(np.median(rel) - 25, None)
    ax1.set_ylim(0, 1.0)
    ax1.set_xlabel("Generations from founding\nto first invasion")
    ax1.set_ylabel("Share of tumors invaded")
    ax1.legend(loc="lower right")
    # clones reaching the stroma within WINDOW generations of the first invasion
    cats = ["1", "2", "3+"]
    shades = [S.PASTEL["blue"], "#7FB2DA", S.SOURCE["iscc"]]
    for j, (arm, rs) in enumerate(arms.items()):
        ns = [sum(1 for c in r["cross_then"].values() if c > 0) for r in rs if r.get("cross_then")]
        if not ns:
            continue
        fr = [np.mean([n == 1 for n in ns]), np.mean([n == 2 for n in ns]), np.mean([n >= 3 for n in ns])]
        bottom = 0
        for f, c in zip(fr, shades):
            ax2.bar(j, f, bottom=bottom, color=c, width=0.7, lw=0)
            bottom += f
        out[arm]["clones_reaching"] = dict(zip(cats, [round(float(f), 2) for f in fr]))
        out[arm]["clones_reaching_n"] = ns
        out[arm]["clones_reaching_final_def"] = [sum(1 for c in r["cross_final"].values() if c > 0)
                                                 for r in rs if r.get("cross_final")]
        out[arm]["clones_present_window_end"] = [sum(1 for c in r["cross_then_end"].values() if c > 0)
                                                 for r in rs if r.get("cross_then_end")]
        out[arm]["crossing_genotypes"] = [r["n_crossing_genotypes"] for r in rs if "n_crossing_genotypes" in r]
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["with\nbarriers", "without\nbarriers"])
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Share of tumors, by clones\ncrossing into the stroma")
    _key(ax2, shades, cats, "clones")
    return out


def _key(ax, shades, cats, title):
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=lab) for c, lab in zip(shades[::-1], cats[::-1])], title=title,
              loc="upper left", bbox_to_anchor=(1.0, 1.0), title_fontsize=SMALL, handlelength=0.8)


def _enrichment(sel, ctl, comp, trait):
    """log2 enrichment of a trait's carrier share, one pseudo-cell on each side (finite at 0)."""
    a, b = sel["comp"][comp], ctl["comp"][comp]
    if a["n"] == 0 or b["n"] == 0:
        return None
    return float(np.log2((a[trait] + 1) / (a["n"] + 1)) - np.log2((b[trait] + 1) / (b["n"] + 1))), b[trait] == 0


def render_traits(axes, legend_mm):
    """Fig 2B: the share of cancer cells carrying each trait in each compartment (one axes each), in
    the selected tumour and in the control without that trait's selection at the same seed, side by side
    (one dot per seed, a thin line joining the two tumours of a seed, ticks at the medians). The shares
    run from 0 to 1 with several near 0, so the axis is linear below 0.1% and logarithmic above (0 is
    drawn at 0). The log2 enrichment is kept only in the numbers file. The legend sits ``legend_mm``
    below the middle axes."""
    from matplotlib.lines import Line2D
    from matplotlib.ticker import NullLocator
    d = cached("c")
    if not d:
        for ax in axes:
            pending(ax)
        return None
    out = {}
    names = {"breach": "breach", "stromal_survival": "stromal\nsurvival", "met_survival": "metastatic\nsurvival"}
    tcol = {"breach": S.CLONE_FAMILIES[0][1], "stromal_survival": S.CLONE_FAMILIES[2][1],
            "met_survival": S.CLONE_FAMILIES[3][1]}
    off = 0.17                                    # selected (filled) above the trait's row, control (hollow) below
    for j, (ax, comp) in enumerate(zip(axes, COMPS)):
        for i, tr in enumerate(TRAITS):
            vals, shares_s, shares_c, zero = [], [], [], []
            for s in REG.SEEDS:
                sel, ctl = d.get(f"R12/seed{s}"), d.get(f"R13/{CONTROL[tr]}_seed{s}")
                if sel is None or ctl is None:
                    continue
                e = _enrichment(sel, ctl, comp, tr)
                if e is None:
                    continue
                vals.append(e[0])
                zero.append(e[1])
                shares_s.append(sel["comp"][comp][tr] / sel["comp"][comp]["n"])
                shares_c.append(ctl["comp"][comp][tr] / ctl["comp"][comp]["n"])
            if not vals:
                continue
            row = 2 - i
            jit = np.random.default_rng(10 * i + j).uniform(-0.05, 0.05, len(vals))
            ps, pc = 100 * np.asarray(shares_s), 100 * np.asarray(shares_c)
            for a_, b_, k in zip(ps, pc, jit):
                ax.plot([a_, b_], [row + off + k, row - off + k], color=S.BG_CELLS, lw=0.3, zorder=1)
            ax.scatter(ps, row + off + jit, s=5, color=tcol[tr], lw=0, zorder=2)
            ax.scatter(pc, row - off + jit, s=5, facecolors="none", edgecolors=tcol[tr], lw=0.5, zorder=2)
            for med, yc in ((np.median(ps), row + off), (np.median(pc), row - off)):
                ax.plot([med] * 2, [yc - 0.11, yc + 0.11], color=S.INK, lw=0.8, zorder=3)
            v = np.asarray(vals)
            z = np.asarray(zero)
            out[f"{tr}/{comp}"] = dict(n=len(v), median_log2=round(float(np.median(v)), 2),
                                       min=round(float(v.min()), 2), max=round(float(v.max()), 2),
                                       median_share_sel=round(float(np.median(shares_s)), 4),
                                       median_share_ctl=round(float(np.median(shares_c)), 4),
                                       ctl_zero=int(z.sum()))
        ax.set_xscale("symlog", linthresh=0.1, linscale=0.4)
        ax.set_xlim(-0.03, 160)
        ax.set_xticks([0, 1, 10, 100])
        ax.set_xticklabels(["0", "1", "10", "100"])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_ylim(-0.5, 2.5)
        ax.set_yticks([2, 1, 0])
        ax.set_yticklabels([names[tr] for tr in TRAITS] if j == 0 else [])
        ax.set_title(NAME[comp])
        if j == 1:
            ax.set_xlabel("Cancer cells carrying the trait (%)")
            ax.legend(handles=[Line2D([], [], marker="o", ms=2.2, ls="none", color=S.INK, mew=0,
                                      label="selected"),
                               Line2D([], [], marker="o", ms=2.2, ls="none", mfc="none", mec=S.INK, mew=0.5,
                                      label="control without that selection")],
                      loc="upper center", bbox_to_anchor=(0.5, below(ax, legend_mm)), ncol=2)
    return out


def render_confinement(axes):
    """S10 Fig D (R3): the old Fig 4f-g's truncal copy-number fraction and clonal diversity of confined
    and free tumours, neutral and selected."""
    d = cached("d")
    if not d:
        for ax in axes:
            pending(ax)
        return None
    out = {}
    groups = [("confined", "neutral"), ("confined", "selected"), ("free", "neutral"), ("free", "selected")]
    for ax, (col, ylab) in zip(axes, (("trunk_fraction", "Truncal copy number fraction"),
                                      ("D", "Clonal diversity, D"))):
        for j, (arch, sel) in enumerate(groups):
            v = np.asarray([r[col] for k, r in d.items() if k.split("/")[1].startswith(f"{arch}_{sel}_")],
                           dtype=float)
            if not len(v):
                continue
            xx = j + np.random.default_rng(j).uniform(-0.12, 0.12, len(v))
            c = S.CLONE_FAMILIES[0][1] if arch == "confined" else S.CLONE_FAMILIES[2][1]
            ax.scatter(xx, v, s=4, color=c, alpha=0.8 if sel == "selected" else 0.45)
            ax.plot([j - 0.25, j + 0.25], [v.mean()] * 2, color=S.INK, lw=0.8)
            out[f"{col}/{arch}/{sel}"] = dict(n=len(v), mean=round(float(v.mean()), 3),
                                              nonzero=int((v > 0).sum()) if col == "trunk_fraction" else None)
        ax.set_xticks(range(4))
        # four groups on a narrow axis: the full words fit only rotated
        ax.set_xticklabels(["confined, neutral", "confined, selected", "free, neutral", "free, selected"],
                           rotation=90)
        ax.set_ylabel(ylab)
    return out


def _tree_layout(tree):
    nodes = {n["id"]: n for n in tree["nodes"]}
    kids = defaultdict(list)
    for n in nodes.values():
        if n["parent"] is not None:
            kids[n["parent"]].append(n["id"])
    xpos, nxt = {}, [0]

    def place(g):
        ch = sorted(kids[g], key=lambda c: (-(nodes[c]["sub_prim"] + nodes[c]["sub_met"])))
        if not ch:
            xpos[g] = nxt[0]
            nxt[0] += 1
        else:
            for c in ch:
                place(c)
            xpos[g] = float(np.mean([xpos[c] for c in ch]))
    roots = [g for g, n in nodes.items() if n["parent"] is None]
    for r in roots:
        place(r)
    return nodes, kids, xpos


def _where(n):
    p, m = n["sub_prim"], n["sub_met"]
    return "both" if p and m else ("met" if m else "primary")


def render_tree(ax, legend_mm):
    """S10 Fig E: the showcase's clone tree, branches coloured by where their cells live, with the
    seedings that founded at least 1% of the metastasis; the legend sits ``legend_mm`` below the axes."""
    a = cached("a")
    if a is None or not a.get("tree"):
        return pending(ax, "pending: R0")
    nodes, kids, xpos = _tree_layout(a["tree"])
    col = dict(zip(a["labels"], a["colours"]))
    end = a["end"]
    for g, n in nodes.items():
        c = WHERE[_where(n)]
        if n["parent"] is not None:
            p = nodes[n["parent"]]
            ax.plot([xpos[g], xpos[g]], [p["time"], n["time"]], color=c, lw=0.9, solid_capstyle="butt")
            ax.plot([xpos[n["parent"]], xpos[g]], [p["time"]] * 2, color=WHERE[_where(p)], lw=0.9)
        if not kids[g]:
            ax.plot([xpos[g], xpos[g]], [n["time"], end], color=c, lw=1.8, solid_capstyle="butt")
        if n["label"]:
            ax.scatter([xpos[g]], [n["time"]], s=8, color=col[n["label"]], edgecolors=S.INK, lw=0.3, zorder=3)
            if not kids[g]:
                ax.text(xpos[g], end * 1.03, n["label"].replace("clone ", ""), fontsize=SMALL, ha="center",
                        va="top", color=S.INK)
    met_total = max(sum(s[2] for s in a["tree"]["seeds"]), 1)
    shown = [s for s in a["tree"]["seeds"] if s[1] is not None and s[2] / met_total >= 0.01]
    for g, ts, c in shown:
        if g in xpos:
            ax.scatter([xpos[g]], [ts], marker="*", s=14 + 40 * c / met_total, color=WHERE["met"],
                       edgecolors=S.INK, lw=0.3, zorder=4)
    ax.set_ylim(end * 1.02, 0)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.set_xlabel("Clone", labelpad=8)
    ax.set_ylabel("Generations")
    handles = [ax.plot([], [], color=WHERE[k], lw=1.5, label=lab)[0]
               for k, lab in (("primary", "primary only"), ("met", "metastasis only"), ("both", "both"))]
    handles.append(ax.scatter([], [], marker="*", s=16, color=WHERE["met"], edgecolors=S.INK, lw=0.3,
                              label="seeding"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, below(ax, legend_mm)), ncol=4)
    return dict(n_nodes=len(nodes), n_leaves=sum(1 for g in nodes if not kids[g]),
                where=Counter(_where(n) for g, n in nodes.items() if not kids[g]),
                seeds_shown=len(shown), seeds_total=len(a["tree"]["seeds"]))


def render_seeding(ax2, ax3):
    """Fig 2C (R12): the effective number of founding lineages per metastasis, by genotype and by clone,
    and the release, first invasion and first seeding as fractions of the primary's age."""
    from matplotlib.ticker import MaxNLocator
    d = cached("e")
    if not d:
        pending(ax2, "")
        pending(ax3)
        return None
    rows = list(d.values())
    seeded = [r for r in rows if r.get("met_cancer")]

    # the effective number: the inverse Simpson index of the met's cells over its founding lineages
    def eff_n(weights):
        p = np.asarray(list(weights), dtype=float)
        p = p / p.sum()
        return 1.0 / float((p ** 2).sum())
    eff_g = np.asarray([eff_n(f[0] for f in r["founders"]) for r in seeded])
    eff_c = []
    for r in seeded:
        by = Counter()
        for c, _, lab in r["founders"]:
            by[lab] += c
        eff_c.append(eff_n(by.values()))
    eff_c = np.asarray(eff_c)
    nf = np.asarray([r["n_founders"] for r in seeded])
    for j, v in enumerate((eff_g, eff_c)):
        if not len(v):
            continue
        xx = j + np.random.default_rng(5 + j).uniform(-0.15, 0.15, len(v))
        ax2.scatter(xx, v, s=5, color=LINE["met"], lw=0)
        ax2.plot([j - 0.3, j + 0.3], [np.median(v)] * 2, color=S.INK, lw=0.8)
    ax2.axhline(1, color=S.GREY, lw=0.5, ls=":")
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["genotypes", "clones"])
    ax2.set_xlim(-0.7, 1.7)
    ax2.set_ylim(0, max(2.0, 1.1 * float(max(eff_g.max(), eff_c.max()))) if len(eff_g) else 2)
    ax2.yaxis.set_major_locator(MaxNLocator(5, integer=True))
    ax2.set_ylabel("Effective number of\nfounding lineages")
    out = {}
    keys = [("release", "acinus\nopens", S.GREY), ("t_inv", "first\ninvasion", LINE["stroma"]),
            ("t_seed", "first\nseeding", LINE["met"])]
    for j, (k, lab, c) in enumerate(keys):
        v = np.asarray([r[k] / r["time"] for r in rows if r.get(k) is not None], dtype=float)
        if not len(v):
            continue
        xx = j + np.random.default_rng(j).uniform(-0.15, 0.15, len(v))
        ax3.scatter(xx, v, s=5, color=c, lw=0)
        ax3.plot([j - 0.3, j + 0.3], [np.median(v)] * 2, color=S.INK, lw=0.8)
        out[f"frac_{k}"] = dict(n=len(v), median=round(float(np.median(v)), 3), min=round(float(v.min()), 3),
                                max=round(float(v.max()), 3))
    ax3.set_xticks(range(3))
    ax3.set_xticklabels([k[1] for k in keys])
    ax3.set_xlim(-0.6, 2.6)
    ax3.set_ylim(0, 1)
    ax3.set_ylabel("Fraction of the\nage of the primary")
    out.update(n_tumours=len(rows), n_seeded=len(seeded), founders=nf.tolist(),
               effective_genotypes=eff_g.round(2).tolist(), effective_clones=eff_c.round(2).tolist(),
               median_effective_genotypes=float(np.median(eff_g)) if len(eff_g) else None,
               median_effective_clones=float(np.median(eff_c)) if len(eff_c) else None,
               share_effective_genotypes_below_1_5=float(np.mean(eff_g < 1.5)) if len(eff_g) else None,
               founder_clones=[len({f[2] for f in r["founders"]}) for r in seeded],
               met_cancer=[r["met_cancer"] for r in rows])
    return out


def render_noble(ax):
    """Fig 2D: drivers per cell against clonal diversity, iscc against the real tumours of Noble et al.
    (fig2_realism.py's panel a, from summaries.py's caches of R1-R4)."""
    import fig2_realism as FR
    return FR.panel_a(ax)


def render_wgd(ax):
    """S10 Fig F: whole-genome-duplication prevalence across rates against PCAWG (fig2_realism.py's
    panel c, from summaries.py's cache of R5), with the paper's rate marked."""
    import fig2_realism as FR
    out = FR.panel_c(ax)
    w = RG.CANCER["wgd_rate"]
    ax.set_xticks(REG.R5_WGD)
    ax.set_xticklabels([f"{r:g}" for r in REG.R5_WGD], rotation=90)
    ax.axvline(w, color=S.INK, lw=0.5, ls=":")
    ax.text(w, 0.95, " this paper", fontsize=SMALL, color=S.INK, va="top", ha="left")
    return out


def _numbers(out, name):
    for k, v in out.items():
        print(k, json.dumps(v, default=str))
    os.makedirs(CACHE, exist_ok=True)
    with open(os.path.join(CACHE, name), "w") as f:
        json.dump(out, f, default=str, indent=1)


def render():
    """Fig 2, 165 by 110 mm. B spans the height of C and D together, so that its axis and D's share a
    baseline (95 mm) and B's legend sits under both."""
    S.setup()
    fig = S.figure(165, 110)
    out = {}
    S.panel_letter(fig, "A", 1, 1)
    out["A"] = render_snapshots(fig, 6, 8.5, 23.0, 3.4)
    S.panel_letter(fig, "B", 1, 34)
    out["B"] = render_traits([S.ax_mm(fig, 15 + i * 23, 41, 20, 54) for i in range(3)], legend_mm=8)
    S.panel_letter(fig, "C", 86, 34)
    out["C"] = render_seeding(S.ax_mm(fig, 99, 41, 22, 20), S.ax_mm(fig, 132, 41, 31, 20))
    S.panel_letter(fig, "D", 86, 67)
    out["D"] = render_noble(S.ax_mm(fig, 99, 72, 64, 23))
    path = S.save(fig, "fig2_niche")
    print("wrote", path)
    _numbers(out, "niche_render_numbers.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["a", "b", "c", "d", "e"])
    ap.add_argument("--source", default=None, help="run/task of panel a (default R0/seed1, else the stand-in)")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--one", nargs=2, metavar=("PART", "RUN/TASK"), help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.one:
        _one(*a.one)
    elif a.part == "a":
        src = a.source or (SHOWCASE if os.path.exists(REG.task_path(*SHOWCASE.split("/"))) else STAND_IN)
        if not os.path.exists(REG.task_path(*src.split("/"))):
            print(f"part a: {src} not grown yet")
            return
        _write("a", compute_a(src))
        print(f"part a: cached from {src}")
    elif a.part:
        compute_part(a.part, a.jobs, a.force)
    else:
        render()


if __name__ == "__main__":
    main()
