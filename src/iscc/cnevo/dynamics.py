"""Clonal dynamics (Q1), diversity over time (Q2) and r/K demography (Q3).

All three read ``tumor.traces``, which records the **exact** ``{genotype_id: n_cells}`` population
at every snapshot. Combined with ``tumor.genotypes`` — which retains every genotype ever created,
including extinct ancestors — this makes each quantity below computable at every generation of a
finished run with no re-simulation and no reconstruction. (The SISTEM pipeline this mirrors had to
approximate per-clone trajectories from smoothed birth counts, because its log carried only
population totals.)

Every metric is restricted to **cancer** genotypes. A glandular substrate seeds immortal
epithelial and stromal cells, and counting them would swamp the tumour's own dynamics.
"""
import numpy as np
import pandas as pd

from ..inference.indices import (
    clone_tree, driver_combination_counts, inverse_simpson, tree_balance_j1,
)

__all__ = ["clone_size_matrix", "sweep_metrics", "diversity_trajectory", "growth_phase"]

_SWEEP_LOW = 0.3        # max clone frequency must dip below this to arm the detector ...
_SWEEP_HIGH = 0.5       # ... and rise past this to count as a sweep


# --------------------------------------------------------------------------------------
# Shared trace extraction
# --------------------------------------------------------------------------------------
def clone_size_matrix(tumor):
    """Exact per-clone sizes over time, cancer only.

    Returns
    -------
    (clones, sizes, gens) : (list[str], ndarray (C, T), ndarray (T,))
        ``sizes[c, t]`` is the number of cells of clone ``c`` at snapshot ``t``. Generations are
        ``tumor.trace_times`` when tau-leaping recorded them, else the snapshot index.
    """
    traces = tumor.traces or []
    seen = {}
    for snap in traces:
        for gid in snap["genotypes_counts"]:
            if gid not in seen and tumor._is_cancer(gid):
                seen[gid] = len(seen)
    clones = list(seen)
    sizes = np.zeros((len(clones), len(traces)), dtype=float)
    for t, snap in enumerate(traces):
        for gid, cnt in snap["genotypes_counts"].items():
            i = seen.get(gid)
            if i is not None:
                sizes[i, t] = cnt
    times = getattr(tumor, "trace_times", None)
    gens = (np.asarray(times, dtype=float) if times is not None and len(times) == len(traces)
            else np.arange(len(traces), dtype=float))
    return clones, sizes, gens


def _mean_fitness(tumor, clones, sizes):
    """Count-weighted mean heritable division rate per snapshot.

    The engine's *heritable* fitness is the evolved ``division_rate``; the crowding term added to
    death at run time is environmental, not inherited, so it is deliberately excluded — a rising
    series here means selection is acting, not that the tumour got crowded.
    """
    rate = np.array([float(tumor.genotypes[g].evolutionary_parameters["division_rate"])
                     if g in tumor.genotypes else np.nan for g in clones], dtype=float)
    tot = sizes.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(tot > 0, (rate[:, None] * sizes).sum(axis=0) / np.maximum(tot, 1e-12), np.nan)


def _birth_gens(clones, sizes, gens):
    """Generation at which each clone is first seen with a non-zero count."""
    out = np.full(len(clones), np.nan)
    for i in range(len(clones)):
        nz = np.flatnonzero(sizes[i] > 0)
        if nz.size:
            out[i] = gens[nz[0]]
    return out


def _count_sweeps(max_freq):
    """Sawtooth detector: arm below ``_SWEEP_LOW``, fire on rising past ``_SWEEP_HIGH``."""
    fires, armed = [], True
    for i, f in enumerate(max_freq):
        if f < _SWEEP_LOW:
            armed = True
        elif armed and f >= _SWEEP_HIGH:
            fires.append(i)
            armed = False
    return fires


# --------------------------------------------------------------------------------------
# Q1 — clonal dynamics and sweeps
# --------------------------------------------------------------------------------------
def sweep_metrics(tumor):
    """Whole-run clonal-sweep dynamics: coalescent depth, diversity, selection, sweep counts.

    Distinguishes the regimes that matter for a copy-number study: *continual draft* (clone count
    peaks then erodes while mean fitness climbs), *discrete sweeps* (sawtooth max-frequency), and
    *near-neutral* (flat fitness, high stable diversity).

    Two coalescent depths are reported. ``mrca_depth_gens`` is the SISTEM-comparable proxy — how
    long ago the oldest *surviving* clone was born. ``mrca_lca_depth_gens`` is the exact
    genealogical quantity iscc can also give: the birth generation of the true lowest common
    ancestor of the surviving clones, found on ``genotypes_parents``. They differ whenever a
    surviving lineage's common ancestor is older than every surviving clone, which is the normal
    case; the exact one is the honest coalescent depth, the proxy is for comparison with the
    SISTEM findings on their own terms. A small fraction means a shallow genealogy — a big shared
    trunk and few private events, i.e. a hard tree to reconstruct.
    """
    clones, sizes, gens = clone_size_matrix(tumor)
    out = dict(n_snapshots=int(len(gens)))
    if not clones or sizes.size == 0:
        return {**out, "last_gen": float("nan"), "n_surviving_genotypes": 0}

    last_gen = float(gens[-1])
    total = sizes.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        props = np.where(total > 0, sizes / np.maximum(total, 1e-12), 0.0)
    max_freq = props.max(axis=0)
    # inverse Simpson over distinct GENOTYPES (SISTEM's "eff_clones"); Noble's driver-combination
    # diversity D is the other notion, reported per snapshot by `diversity_trajectory`.
    eff_genotypes = np.where(total > 0, 1.0 / np.maximum((props ** 2).sum(axis=0), 1e-12), 0.0)
    clone_count = (sizes > 0).sum(axis=0)
    fitness = _mean_fitness(tumor, clones, sizes)
    births = _birth_gens(clones, sizes, gens)

    surviving = [c for i, c in enumerate(clones) if sizes[i, -1] > 0]
    surv_idx = [i for i in range(len(clones)) if sizes[i, -1] > 0]
    min_birth = float(np.nanmin(births[surv_idx])) if surv_idx else float("nan")

    # established phase = from the last surviving clone's oldest ancestor onward, and only where
    # the population is non-trivial; summary statistics over the founding transient are meaningless.
    est = total >= 0.5 * max(total.max(), 1.0)
    if not est.any():
        est = total > 0

    fires = _count_sweeps(max_freq)
    inter = float(np.mean(np.diff(gens[fires]))) if len(fires) > 1 else float("nan")

    slope = float("nan")
    if est.sum() >= 3 and np.isfinite(fitness[est]).sum() >= 3:
        g, f = gens[est], fitness[est]
        ok = np.isfinite(f)
        if ok.sum() >= 3 and np.ptp(g[ok]) > 0:
            slope = float(np.polyfit(g[ok], f[ok], 1)[0] * 1000.0)

    out.update(
        last_gen=last_gen,
        n_surviving_genotypes=int(len(surviving)),
        min_surviving_birth_gen=min_birth,
        mrca_depth_gens=float(last_gen - min_birth) if np.isfinite(min_birth) else float("nan"),
        mrca_depth_frac=(float((last_gen - min_birth) / last_gen)
                         if np.isfinite(min_birth) and last_gen > 0 else float("nan")),
        peak_genotype_count=int(clone_count.max()),
        peak_genotype_count_gen=float(gens[int(clone_count.argmax())]),
        final_genotype_count=int(clone_count[-1]),
        median_max_clone_freq=float(np.median(max_freq[est])) if est.any() else float("nan"),
        median_eff_genotypes=float(np.median(eff_genotypes[est])) if est.any() else float("nan"),
        fitness_final=float(fitness[-1]),
        fitness_slope_per_1k=slope,
        n_sweeps_detected=int(len(fires)),
        mean_intersweep_gens=inter,
    )
    out.update(_lca_depth(tumor, surviving, clones, births, last_gen))
    return out


def _lca_depth(tumor, surviving, clones, births, last_gen):
    """Birth generation of the true LCA of the surviving clones, and the depth below it."""
    if not surviving:
        return dict(mrca_lca_depth_gens=float("nan"), mrca_lca_depth_frac=float("nan"))
    parents = tumor.genotypes_parents

    def ancestry(g):
        chain, cur, seen = [], str(g), set()
        while cur is not None and cur not in seen:
            seen.add(cur)
            chain.append(cur)
            cur = parents.get(cur)
        return chain

    common = None
    for g in surviving:
        anc = ancestry(g)
        common = set(anc) if common is None else (common & set(anc))
    if not common:
        return dict(mrca_lca_depth_gens=float("nan"), mrca_lca_depth_frac=float("nan"))
    # deepest shared ancestor = the one furthest from the root, i.e. the youngest by creation order
    lca = max(common, key=lambda g: getattr(tumor.genotypes.get(g), "ord", -1))
    idx = {c: i for i, c in enumerate(clones)}
    bg = births[idx[lca]] if lca in idx else 0.0     # ancestor never extant -> born at/near seeding
    if not np.isfinite(bg):
        bg = 0.0
    return dict(mrca_lca_depth_gens=float(last_gen - bg),
                mrca_lca_depth_frac=float((last_gen - bg) / last_gen) if last_gen > 0 else float("nan"))


# --------------------------------------------------------------------------------------
# Q2 — clonal diversity over time
# --------------------------------------------------------------------------------------
def diversity_trajectory(tumor, stride=1):
    """Per-snapshot diversity indices — the trajectory through Noble's ``(n, D, J1)`` mode space.

    ``validate_evolution_modes.py`` places a finished tumour as one point in that space; this is
    the path it took to get there.

    Parameters
    ----------
    stride : int
        Evaluate every ``stride``-th snapshot. ``J1`` requires contracting the genealogy onto the
        driver-clone phylogeny, which is the expensive part; raise this on long runs.

    Two different notions of "how many clones" are reported, because the repo uses both and
    conflating them is easy: ``n_genotypes`` / ``eff_genotypes`` count distinct *genotypes* (the
    engine's lineage entities, and the analogue of SISTEM's clone count), whereas ``n_clones`` and
    ``D`` count distinct *driver combinations*, which is Noble's convention and what
    :func:`~iscc.inference.indices.mode_indices` reports. The last row of this frame equals
    ``mode_indices(tumor)`` on ``n_drivers``/``D``/``J1``/``n_clones``.

    Returns
    -------
    DataFrame with ``gen, n_cells, n_genotypes, eff_genotypes, n_clones, D, n_drivers, J1,
    max_clone_freq, mean_fitness``.
    """
    clones, sizes, gens = clone_size_matrix(tumor)
    traces = tumor.traces or []
    if not traces:
        return pd.DataFrame(columns=["gen", "n_cells", "n_genotypes", "eff_genotypes", "n_clones",
                                     "D", "n_drivers", "J1", "max_clone_freq", "mean_fitness"])
    fitness = _mean_fitness(tumor, clones, sizes) if clones else np.full(len(traces), np.nan)

    cache = {}                       # genotype -> driver combination, reused across snapshots
    rows = []
    for t in range(0, len(traces), max(1, int(stride))):
        counts = {g: c for g, c in traces[t]["genotypes_counts"].items()
                  if tumor._is_cancer(g) and c > 0}
        n_cells = float(sum(counts.values()))
        if n_cells <= 0:
            rows.append((float(gens[t]), 0.0, 0, 0.0, 0, np.nan, np.nan, np.nan, np.nan,
                         fitness[t]))
            continue
        combos = driver_combination_counts(tumor, counts, combo_cache=cache)
        tot = sum(combos.values())
        n_drivers = (sum(len(c) * n for c, n in combos.items()) / tot) if tot else np.nan
        parents, csizes = clone_tree(tumor, counts, combo_cache=cache)
        rows.append((
            float(gens[t]), n_cells, int(len(counts)),
            float(inverse_simpson(counts.values())),
            int(len(combos)),
            float(inverse_simpson(combos.values())),
            float(n_drivers),
            float(tree_balance_j1(parents, csizes)),
            float(max(counts.values()) / n_cells),
            float(fitness[t]),
        ))
    return pd.DataFrame(rows, columns=["gen", "n_cells", "n_genotypes", "eff_genotypes",
                                       "n_clones", "D", "n_drivers", "J1", "max_clone_freq",
                                       "mean_fitness"])


# --------------------------------------------------------------------------------------
# Q3 — r- vs K-phase demography
# --------------------------------------------------------------------------------------
def _smooth(x, w):
    """Centred moving average, edge-preserving."""
    x = np.asarray(x, dtype=float)
    if w <= 1 or x.size < 3:
        return x
    w = int(min(w, x.size if x.size % 2 else x.size - 1))
    if w < 3:
        return x
    k = np.ones(w) / w
    pad = w // 2
    return np.convolve(np.pad(x, pad, mode="edge"), k, mode="valid")[:x.size]


def growth_phase(tumor, smooth_frac=0.05, sat_threshold=0.75):
    """When does the tumour leave free expansion for density-limited growth?

    **The transition is defined by saturation, not by the shape of N(t).** iscc caps each deme at a
    real carrying capacity through a density-dependent crowding law, so "the K phase" has a
    physical meaning here: the generation from which most of the population lives at carrying
    capacity and further growth is confined to the expanding front. ``t_rK`` is the first
    generation at which the ``crowding_index`` -- the cell-weighted mean of a deme's fullness, i.e.
    how full the average cell's own deme is -- reaches ``sat_threshold`` and stays there.

    Crowding is weighted by CELLS, not demes. A density-limited tumour still has a wide rim of
    half-empty front demes, so any deme-weighted fraction stays low however packed the core is;
    weighting by cells tracks where the population actually lives. The founding transient is
    excluded (a lone seeded deme can start at capacity), by ignoring snapshots before the
    population has doubled from its initial size.

    Deriving it from ``d log N / dt`` instead is tempting and wrong: per-capita growth decays under
    *any* sub-exponential growth, so a spatial tumour whose front advances linearly — still filling
    empty space, nowhere near its capacity — registers a spurious transition almost immediately.
    That fallback is used only when the run was grown without ``trace_occupancy=True``, and is
    flagged as such by ``phase_basis == "growth_curve"``; treat those values as approximate and
    biased towards declaring K too early.

    Occupancy fields are ``None`` without ``trace_occupancy`` (see
    :meth:`GenotypeTumor._occupancy_snapshot`). In a glandular substrate the duct and stroma
    compartments have different capacities, so they are reported separately — a single global
    saturation fraction mixes two different denominators.
    """
    traces = tumor.traces or []
    _, sizes, gens = clone_size_matrix(tumor)
    n = sizes.sum(axis=0) if sizes.size else np.zeros(len(traces))
    out = {k: None for k in (
        "n_occupied_demes", "mean_occupancy", "occupancy_ratio", "crowding_index",
        "max_crowding_index", "saturated_deme_frac", "saturated_cell_frac",
        "n_occupied_demes_duct", "mean_occupancy_duct", "saturated_deme_frac_duct",
        "saturated_cell_frac_duct", "crowding_index_duct",
        "n_occupied_demes_stroma", "mean_occupancy_stroma", "saturated_deme_frac_stroma",
        "saturated_cell_frac_stroma", "crowding_index_stroma",
        "t_rK_duct", "t_rK_stroma", "t_escape", "n_glands_colonised")}
    last_gen = float(gens[-1]) if len(gens) else float("nan")
    out.update(last_gen=last_gen,
               n_final=float(n[-1]) if n.size else 0.0,
               n_peak=float(n.max()) if n.size else 0.0,
               occupancy_traced=bool(getattr(tumor, "trace_occupancy", False)))

    traced = bool(getattr(tumor, "trace_occupancy", False)) and bool(traces)

    def series(key):
        v = [s.get(key) for s in traces]
        return np.array([np.nan if x is None else float(x) for x in v], dtype=float)

    if traced:
        out["phase_basis"] = "saturation"
        sat = series("crowding_index")
        t_rK = _sustained_crossing(sat, gens, sat_threshold, after=_post_founding(n))
        out["per_capita_growth_early"] = _transition(n, gens, smooth_frac)[1]
        for key in ("n_occupied_demes", "mean_occupancy", "occupancy_ratio", "crowding_index",
                    "saturated_deme_frac", "saturated_cell_frac"):
            out[key] = float(series(key)[-1])
        out["max_crowding_index"] = float(np.nanmax(sat)) if sat.size else float("nan")
    else:
        out["phase_basis"] = "growth_curve"
        t_rK, early = _transition(n, gens, smooth_frac)
        out["per_capita_growth_early"] = early

    out["t_rK"] = t_rK
    out["frac_gens_in_K"] = (float((last_gen - t_rK) / last_gen)
                             if t_rK is not None and np.isfinite(last_gen) and last_gen > 0
                             else (0.0 if np.isfinite(last_gen) else float("nan")))

    if not traced:
        return out

    if tumor.structure_radius > 0 and "saturated_deme_frac_duct" in traces[-1]:
        for key in ("n_occupied_demes_duct", "mean_occupancy_duct", "saturated_deme_frac_duct",
                    "saturated_cell_frac_duct", "crowding_index_duct",
                    "n_occupied_demes_stroma", "mean_occupancy_stroma",
                    "saturated_deme_frac_stroma", "saturated_cell_frac_stroma",
                    "crowding_index_stroma", "n_glands_colonised"):
            out[key] = float(series(key)[-1])
        for comp in ("duct", "stroma"):
            out[f"t_rK_{comp}"] = _sustained_crossing(
                series(f"crowding_index_{comp}"), gens, sat_threshold,
                after=_post_founding(n))
        hit = np.flatnonzero(series("n_occupied_demes_stroma") > 0)
        out["t_escape"] = float(gens[hit[0]]) if hit.size else None
    return out


def _post_founding(n):
    """First snapshot index at which the population has doubled from its initial size.

    The founding deme can be seeded at capacity, which would otherwise read as "saturated" from
    generation zero and collapse the whole run into the K phase.
    """
    n = np.asarray(n, dtype=float)
    nz = np.flatnonzero(n > 0)
    if nz.size == 0:
        return 0
    start = float(n[nz[0]])
    hit = np.flatnonzero(n >= 2.0 * start)
    return int(hit[0]) if hit.size else int(nz[0])


def _sustained_crossing(x, gens, threshold, persist_frac=0.1, after=0):
    """First generation at or after index ``after`` where ``x`` reaches ``threshold`` and stays.

    "Stays there" means the following ``persist_frac`` of the run does too, so a transient spike --
    a lone deme briefly filling early on -- is not mistaken for the transition.
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0 or not np.any(np.isfinite(x)):
        return None
    need = max(1, int(round(persist_frac * x.size)))
    above = np.nan_to_num(x, nan=0.0) >= threshold
    # `above.size - need` and not `above.size`: a window running off the end is shorter than `need`
    # and would pass trivially, so the last few snapshots would always look like a transition.
    for i in range(max(0, int(after)), max(0, above.size - need) + 1):
        if above[i] and above[i:i + need].all():
            return float(gens[min(i, len(gens) - 1)])
    return None


def _transition(n, gens, smooth_frac):
    """``(t_rK, early_per_capita_growth)`` from a population series; ``t_rK`` is ``None`` if the
    run never leaves the r phase."""
    n = np.asarray(n, dtype=float)
    if n.size < 6 or not np.any(n > 0):
        return None, float("nan")
    w = max(3, int(round(smooth_frac * n.size)))
    ln = np.log(np.maximum(_smooth(n, w), 1e-12))
    dg = np.gradient(np.asarray(gens, dtype=float))
    growth = _smooth(np.gradient(ln) / np.where(dg == 0, 1.0, dg), w)
    grow = np.flatnonzero(n > 0)
    lo, hi = grow[0], grow[-1]
    span = max(3, int(0.2 * (hi - lo + 1)))
    early = float(np.nanmedian(growth[lo:lo + span]))
    if not np.isfinite(early) or early <= 0:
        return None, early
    below = np.flatnonzero(growth[lo:] < 0.5 * early)
    if below.size == 0:
        return None, early
    # require the drop to persist, so a single noisy dip in the r phase is not the transition
    need = max(1, span // 2)
    for start in below:
        window = growth[lo + start: lo + start + need]
        if window.size and np.all(window < 0.5 * early):
            return float(gens[min(lo + start, len(gens) - 1)]), early
    return None, early
