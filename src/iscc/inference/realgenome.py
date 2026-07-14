"""Real-genome (per-arm) ABC wiring for fit-to-real PCAWG + Charm (DESIGN_inference A.4.2-3).

The flagship CINner-parity result: infer the per-arm selection vector ``s_arm`` so that the
simulated per-arm copy-number gain/loss landscape reproduces a real cancer type's, then show the
inferred ``s_arm`` correlates with an independent biological signal (Davoli 2013 Charm scores).

This module mirrors ``inference.tumor`` but for the per-arm copy-number model:
  * :func:`arm_cna_summary` / :func:`arm_summary_vector` — the observed summary is the per-arm
    gain frequency and loss frequency (the directly-PCAWG-comparable statistics; SNV summaries are
    dropped because the v1 arm model has SNV drivers off).
  * :class:`RealGenomeSimulator` — ``theta`` (a dict ``s0..s{n-1}`` of per-arm coefficients) ->
    arm summary vector, by running a short ``GenotypeTumor(genome_mode="real")``. Picklable so the
    ABC engine can parallelise it with ``multiprocessing``.
  * :func:`s_arm_prior` — a Uniform prior over the per-arm coefficients (CINner uses U(0.1, 2.0)).

**Identifiability (A.3):** the CNA *rate* is fixed in the base config and only ``s_arm`` is
inferred, exactly as CINner fixes the missegregation probability to identify selection.
"""
import warnings

import numpy as np

from ..tumor.models import GenotypeTumor
from ..validation import segment_copy_numbers
from .abc import Prior
from .summaries import cna_summary


def arm_cna_summary(tumor):
    """Per-arm gain/loss frequencies of a real-genome tumour (within-tumour cell fractions)."""
    return cna_summary(tumor)


def arm_calls(tumor, thr=0.5):
    """Per-arm gain/loss *calls* for one tumour, relative to its own ploidy (PCAWG-style).

    PCAWG's per-arm gain/loss is a per-*donor* call: an arm is gained if its copy number exceeds
    the donor's ploidy by >0.5 (and lost if below by >0.5), handling whole-genome doublings. We
    mirror that on a simulated tumour using the population-mean copy number per arm and the
    tumour's genome-wide mean as its ploidy. Returns ``(gained, lost)`` boolean-float vectors, or
    ``None`` if the tumour has no cancer cells.
    """
    cn = segment_copy_numbers(tumor, cancer_only=True)
    if cn.sum() == 0:
        return None
    ploidy = float(cn.mean())
    gained = (cn > ploidy + thr).astype(float)
    lost = (cn < ploidy - thr).astype(float)
    return gained, lost


def arm_summary_vector(tumor):
    """Within-tumour per-arm gain/loss *cell fractions* (used for layout/inspection).

    Returns ``(values, names)`` with ``values`` = [gain_freq over arms, loss_freq over arms].
    The fit-to-real summary is the *cohort* version (:func:`cohort_summary_vector`); this
    single-tumour fraction is kept for tests and per-tumour inspection.
    """
    cna = cna_summary(tumor)
    n = len(cna["gain_freq"])
    values = np.concatenate([cna["gain_freq"], cna["loss_freq"]]).astype(float)
    names = [f"gain[{i}]" for i in range(n)] + [f"loss[{i}]" for i in range(n)]
    return values, names


def cohort_summary_vector(tumors, n_arms, thr=0.5):
    """Cohort per-arm gain/loss frequencies, comparable to a PCAWG cohort's per-arm frequencies.

    For each tumour we take the within-tumour per-arm gain/loss *cell fraction* (``cna_summary``)
    and average it across the cohort of independent tumours. Because an arm that is gained in a
    tumour's dominant clone has within-tumour fraction ~1 (and ~0 otherwise), the cohort average
    of that fraction approximates the fraction of *tumours* (PCAWG donors) carrying the gain — the
    real observable — while staying **continuous** (so it is a smooth, identifiable function of
    ``s_arm``, unlike a hard per-tumour binary call which is too sparse for the ABC distance).

    Returns ``(values, names)`` = [gain_freq over arms, loss_freq over arms]; all-``nan`` if every
    tumour in the cohort went extinct.
    """
    names = [f"gain[{i}]" for i in range(n_arms)] + [f"loss[{i}]" for i in range(n_arms)]
    vecs = []
    for t in tumors:
        if t.get_cancer_size() == 0:
            continue
        c = cna_summary(t)
        vecs.append(np.concatenate([c["gain_freq"], c["loss_freq"]]))
    if not vecs:
        return np.full(2 * n_arms, np.nan), names
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(np.vstack(vecs), axis=0).astype(float), names


def default_real_config(genome_spec):
    """Base engine config for real-genome inference sims (CNA rate fixed; only s_arm varies).

    Small grid + modest step budget keep a single sim to ~seconds (the engine does one event per
    update). ``snv_prob`` is ~0 so essentially every event is a copy-number change, and
    ``amp_prob`` is symmetric so the *direction* of each arm's drift is set by ``s_arm``, not by a
    baked-in amplification bias.
    """
    return dict(
        selection_params={},  # filled by GenomeSpec.engine_params (arm mode) + injected s_arm
        cancer_cell_params={
            "max_birth_rate": 0.95, "division_rate": 0.4, "death_rate": 0.02,
            "mutation_rate": 0.3, "dispersal_rate": 0.2,
            "snv_prob": 0.02, "cnv_prob": 0.98, "n_snvs_per_allele": 0.3, "amp_prob": 0.5,
        },
        deme_params={"carrying_capacity": 8, "maximum_death_rate": 1.0},
        spatial_params={"grid_size": 12, "structure_radius": 0, "immune_density": 0.0},
    )


def build_real_tumor(genome_spec, s_arm, seed, base_config=None,
                     update_mode="tau", tau=1.0, snapshot_every=1):
    """Construct a ``GenotypeTumor(genome_mode='real')`` with ``s_arm`` injected (not yet grown).

    Shared by :class:`RealGenomeSimulator` and the tau/exact equivalence harness so both build the
    engine identically (only ``update_mode`` differs).
    """
    cfg = base_config or default_real_config(genome_spec)
    sp = dict(cfg["selection_params"])
    sp["s_arm"] = list(np.asarray(s_arm, dtype=float))
    return GenotypeTumor(
        seed=seed, genome_mode="real", genome_spec=genome_spec,
        selection_params=sp, cancer_cell_params=cfg["cancer_cell_params"],
        deme_params=cfg["deme_params"], spatial_params=cfg["spatial_params"],
        update_mode=update_mode, tau=tau, snapshot_every=snapshot_every,
    )


def grow_to_size(genome_spec, s_arm, seed, target, mode="tau", base_config=None,
                 tau=1.0, max_iter=100000):
    """Grow one real-genome tumour to a target cancer-cell count under ``mode`` ("exact"|"tau").

    Steps the chosen engine one *event* ("exact") or one *generation* ("tau") at a time until the
    cancer population first reaches ``target`` (or goes extinct), then materialises it. Growing both
    engines to the **same size** is the clean equivalence comparison: it controls for developmental
    stage so any difference in the per-arm gain/loss summary is attributable to the update rule
    (tau-leaping vs one-event Gillespie), not to the tumour being caught at a different size.
    """
    t = build_real_tumor(genome_spec, s_arm, seed, base_config=base_config,
                         update_mode=mode, tau=tau)
    rng = np.random.default_rng(seed) if mode == "tau" else t.rng
    for _ in range(max_iter):
        cs = t.get_cancer_size()
        if cs == 0 or cs >= target:
            break
        if mode == "tau":
            t._tau_generation(rng, t.tau)
        else:
            t.update(rng)
            t.step += 1
    t.make_cell_data()
    return t


def matched_size_cohort_vector(genome_spec, s_arm, mode, seeds, target,
                               base_config=None, tau=1.0):
    """Cohort per-arm gain/loss vector for tumours all grown to the same ``target`` size."""
    tumors = [grow_to_size(genome_spec, s_arm, sd, target, mode=mode,
                           base_config=base_config, tau=tau) for sd in seeds]
    vec, _ = cohort_summary_vector(tumors, genome_spec.n_arms)
    survivors = sum(1 for t in tumors if t.get_cancer_size() > 0)
    return vec, survivors


def s_arm_prior(n_arms, low=0.5, high=1.6):
    """Uniform product prior over the per-arm coefficients (CINner-style U(low, high))."""
    return Prior({f"s{i}": (low, high) for i in range(n_arms)})


class PerArmRegressor:
    """Infer the per-arm selection vector by a *pooled* per-arm regression (the M3b estimator).

    The arm-CN fitness factorises over arms (``prod_seg s[seg]**(cn-2)``), so each arm's gain/loss
    frequency is governed mainly by its own coefficient ``s[arm]``. Rather than learn a joint
    ``2*n_arms -> n_arms`` map from the reference table (data-starved in ~39 dimensions, and forced
    to extrapolate at the real observation), we pool **every (reference-sim, arm) example** into a
    single 2-D -> 1-D regressor ``(gain[arm], loss[arm]) -> s[arm]``. This multiplies the training
    set by ``n_arms`` and keeps each per-arm prediction inside the training distribution.

    This is the dependency-light analogue of CINner's per-arm ABC-rf, specialised to the
    factorised arm model.
    """

    def __init__(self, n_arms, n_estimators=400, min_samples_leaf=8, n_jobs=1, random_state=0):
        from sklearn.ensemble import RandomForestRegressor
        self.n_arms = n_arms
        self.rf = RandomForestRegressor(
            n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
            random_state=random_state, n_jobs=n_jobs)

    def _features(self, summaries):
        """Stack a ``(m, 2*n_arms)`` summary block into per-arm ``(m*n_arms, 2)`` (gain, loss)."""
        S = np.atleast_2d(summaries)
        n = self.n_arms
        G, L = S[:, :n], S[:, n:]
        return np.vstack([np.column_stack([G[:, i], L[:, i]]) for i in range(n)])

    def fit(self, theta, summaries):
        """``theta`` ``(m, n_arms)`` true coefficients, ``summaries`` ``(m, 2*n_arms)``."""
        X = self._features(summaries)
        y = np.concatenate([np.asarray(theta)[:, i] for i in range(self.n_arms)])
        ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
        self.rf.fit(X[ok], y[ok])
        return self

    def predict(self, observed):
        """Per-arm ``s`` for one observed summary vector ``(2*n_arms,)`` -> ``(n_arms,)``."""
        feats = self._features(np.asarray(observed)[None, :])   # (n_arms, 2)
        return self.rf.predict(feats)


class RealGenomeSimulator:
    """Callable ``theta(dict s0..s{n-1}) -> cohort per-arm gain/loss vector`` for the ABC engine.

    Each evaluation simulates a small **cohort** of ``cohort_size`` independent tumours under the
    same ``s_arm`` and returns the cohort's per-arm gain/loss frequencies (the PCAWG-comparable
    summary). Picklable so the ABC engine can parallelise it with multiprocessing.

    Set ``update_mode="tau"`` (the default) to grow each cohort tumour with **tau-leaping**
    (DESIGN_scalability §7): the whole tumour is advanced one generation of length ``tau`` per
    ``grow`` step (Poisson births/deaths per clone) instead of one event per update, so a cohort
    that reaches a developed copy-number landscape costs ``#clones x #generations`` rather than
    ``#cells`` events — ~250x faster than the exact one-event engine at the production size. The
    per-arm gain/loss summary the ABC fits to is statistically the same as the exact engine's at a
    matched tumour size (see ``tests/test_realgenome_tau.py`` and the ``--equivalence`` gate in
    ``validate_inference_realgenome.py``). ``update_mode="exact"`` recovers the one-event reference
    engine.

    **Growth is bounded by size, not generations.** Each tumour is grown until its cancer
    population first reaches ``target_size`` (default), then materialised. This is what makes the
    scaled fit feasible *and* trustworthy: (a) the per-sim cost is bounded regardless of seed — the
    tau engine grows exponentially without a hard carrying-capacity cap, so a fixed generation count
    would let lucky seeds explode into multi-minute straggler sims; growing to a size cap removes
    those tails. (b) Comparing tau and exact at the **same size** controls for developmental stage,
    which is precisely the equivalence the gate checks. Pass ``target_size=None`` to fall back to a
    fixed ``n_steps`` budget (generations under tau, per-event steps under exact).
    """

    def __init__(self, genome_spec, base_config=None, n_steps=1000, cohort_size=8,
                 thr=0.5, seed=0, update_mode="tau", tau=1.0, snapshot_every=1,
                 target_size=500):
        self.spec = genome_spec
        self.n_arms = genome_spec.n_arms
        self.base_config = base_config or default_real_config(genome_spec)
        self.n_steps = n_steps
        self.cohort_size = cohort_size
        self.thr = thr
        self.seed = seed
        self.update_mode = update_mode
        self.tau = tau
        self.snapshot_every = snapshot_every
        self.target_size = target_size
        self.names = None

    def _s_arm(self, theta):
        return np.array([theta[f"s{i}"] for i in range(self.n_arms)], dtype=float)

    def simulate_tumor(self, theta, seed):
        if self.target_size is not None:
            return grow_to_size(self.spec, self._s_arm(theta), seed, self.target_size,
                                mode=self.update_mode, base_config=self.base_config, tau=self.tau)
        t = build_real_tumor(
            self.spec, self._s_arm(theta), seed, base_config=self.base_config,
            update_mode=self.update_mode, tau=self.tau, snapshot_every=self.snapshot_every)
        t.grow(n_steps=self.n_steps, seed=seed)
        return t

    def __call__(self, theta, seed=None):
        base = self.seed if seed is None else seed
        offset = abs(hash(tuple(round(theta[f"s{i}"], 5) for i in range(self.n_arms)))) % 100000
        tumors = [self.simulate_tumor(theta, seed=base + offset + r)
                  for r in range(self.cohort_size)]
        vec, self.names = cohort_summary_vector(tumors, self.n_arms, thr=self.thr)
        return vec
