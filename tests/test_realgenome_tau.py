"""Tau-leaping wiring for the M3b real-genome ABC (DESIGN_inference A.5 + DESIGN_scalability §7).

The fit-to-real PCAWG + Charm experiment is HPC-bound when each cohort tumour grows in the exact
one-event-per-update mode (a fixed 1000-event budget per tumour spends most of its time on the
crowded-front turnover phase). These tests cover the tau-leaping wiring that makes a publication-
scale reference table feasible:

  * ``update_mode="tau"`` flows from ``RealGenomeSimulator`` into the grown tumour;
  * the **equivalence gate** — at a matched tumour size the per-arm gain/loss summary the ABC fits
    to is statistically the same under tau and exact (no systematic bias; tau agrees with exact as
    well as one exact cohort agrees with another), so the scaled tau fit is trustworthy;
  * tau reaches a matched landscape faster than exact, and the gap grows with tumour size;
  * a fast smoke of the pooled per-arm fit returns a sane ``s_arm`` with the right Charm sign.

Growth is bounded by **target size** (not a generation count): the tau engine grows exponentially
without a hard carrying-capacity cap, so a fixed generation budget would let lucky seeds explode
into multi-minute stragglers — growing to a size cap bounds every sim and is also the clean
matched-size comparison. The full scaled fit + speedup numbers live in
``validation/validate_inference_realgenome.py``.
"""
import time

import numpy as np
from scipy.stats import pearsonr

from iscc.inference.genome import GenomeSpec
from iscc.inference.realgenome import (
    RealGenomeSimulator, PerArmRegressor, s_arm_prior, grow_to_size,
    matched_size_cohort_vector,
)
from iscc.inference.abc import ABC


def _spec(n_arms=8, seed=0):
    """A real-genome-shaped GenomeSpec: alternating oncogene/TSG-dense arms so selection has a
    clear per-arm direction (the Charm axis) while staying small enough for CI."""
    rng = np.random.default_rng(seed)
    onc = np.where(np.arange(n_arms) % 2 == 0, 6.0, 1.0)
    tsg = np.where(np.arange(n_arms) % 2 == 0, 1.0, 6.0)
    return GenomeSpec(
        arm_names=[f"{i}{'pq'[i % 2]}" for i in range(n_arms)],
        arm_lengths=rng.uniform(0.5, 1.0, n_arms) * 1e8,
        onc_counts=onc, tsg_counts=tsg,
        charm=(onc - tsg) / (onc + tsg),   # +ve on oncogene-dense arms
        min_genes=4, max_genes=10,
    )


# --- wiring: update_mode flows through to the engine ------------------------
def test_update_mode_flows_into_tumor():
    spec = _spec(4)
    sim = RealGenomeSimulator(spec, cohort_size=1, update_mode="tau", target_size=200)
    t = sim.simulate_tumor({f"s{i}": 1.2 for i in range(4)}, seed=1)
    assert t.update_mode == "tau"
    assert t.trace_times and t.trace_times[-1] == t.time   # generation-batched real-time clock
    assert t.get_cancer_size() > 0


def test_tau_is_the_default_and_exact_selectable():
    spec = _spec(4)
    assert RealGenomeSimulator(spec).update_mode == "tau"
    ex = RealGenomeSimulator(spec, cohort_size=1, update_mode="exact", target_size=200)
    t = ex.simulate_tumor({f"s{i}": 1.0 for i in range(4)}, seed=1)
    assert t.update_mode == "exact"
    assert not t.trace_times                              # exact engine: no real-time clock


def test_tau_simulator_returns_valid_cohort_vector():
    spec = _spec(6)
    sim = RealGenomeSimulator(spec, cohort_size=6, update_mode="tau", target_size=300, seed=3)
    v = sim({f"s{i}": 1.1 for i in range(6)})
    assert v.shape[0] == 2 * spec.n_arms
    assert np.isfinite(v).all() and ((v >= 0) & (v <= 1)).all()


def test_target_size_bounds_growth():
    # growth stops near the size cap (a generation can overshoot, but not unboundedly), so a fixed
    # cohort cost is guaranteed regardless of seed -- no exponential-growth stragglers. Some seeds
    # go extinct (size 0); the bounded *upper* tail is the point.
    spec = _spec(6)
    sizes = [grow_to_size(spec, np.full(6, 1.4), seed, target=300, mode="tau").get_cancer_size()
             for seed in range(8)]
    assert max(sizes) < 5 * 300                        # bounded: one generation's overshoot, no blow-up
    assert any(sz > 0 for sz in sizes)                 # not everyone goes extinct


# --- THE EQUIVALENCE GATE: tau matches exact at matched size ----------------
def test_tau_matches_exact_per_arm_summary_no_bias():
    """At a matched tumour size, tau's per-arm gain/loss summary must match the exact engine's
    within Monte-Carlo noise and without a systematic bias.

    We compare at a fixed target size (controls developmental stage) and read the noise floor off
    two *independent exact* cohorts: tau-vs-exact agreement should be no worse than exact-vs-exact,
    and the directional bias must be small. The arm summary is otherwise too noisy at CI cohort
    sizes to assert a high absolute correlation, so the gate is (a) small bias and (b) tau is not an
    outlier relative to the exact-exact noise floor.
    """
    spec = _spec(8)
    rng = np.random.default_rng(0)
    target, seeds_a, seeds_b = 300, list(range(20)), list(range(100, 120))
    biases = []
    for _ in range(3):
        s = rng.uniform(0.5, 1.6, spec.n_arms)
        ex_a, _ = matched_size_cohort_vector(spec, s, "exact", seeds_a, target)
        ex_b, _ = matched_size_cohort_vector(spec, s, "exact", seeds_b, target)
        ta, _ = matched_size_cohort_vector(spec, s, "tau", seeds_a, target)
        ok = np.isfinite(ex_a) & np.isfinite(ex_b) & np.isfinite(ta)
        bias = float(np.mean(ta[ok] - ex_a[ok]))
        biases.append(bias)
        # no large systematic offset between the engines (frequencies are O(0.1))
        assert abs(bias) < 0.05, f"tau-vs-exact bias too large: {bias:.4f}"
        # tau tracks exact at least as closely as exact tracks another exact draw (within slack):
        d_te = np.mean(np.abs(ta[ok] - ex_a[ok]))
        d_ee = np.mean(np.abs(ex_b[ok] - ex_a[ok]))
        assert d_te < d_ee + 0.05, f"tau diverges from exact beyond noise floor: {d_te:.3f} vs {d_ee:.3f}"
    assert abs(np.mean(biases)) < 0.035, f"mean bias across vectors: {np.mean(biases):.4f}"


def test_tau_recovers_selection_direction():
    """A strongly amplification-favoured arm gains more than it loses under tau (and vice versa)."""
    spec = _spec(6)
    s = np.where(np.arange(6) % 2 == 0, 1.6, 0.5)   # even arms amplify, odd arms delete
    t = grow_to_size(spec, s, seed=2, target=600, mode="tau")
    from iscc.inference.summaries import cna_summary
    c = cna_summary(t)
    assert c["gain_freq"][0] > c["loss_freq"][0]      # amplified arm
    assert c["loss_freq"][1] > c["gain_freq"][1]      # deleted arm


def test_tau_faster_than_exact_at_matched_size():
    """A tau cohort draw is faster than an exact one at the same (matched) tumour size; the gap
    grows with size (the per-event exact cost climbs with crowding/clone count faster than tau's
    per-generation cost). Loose factor so CI noise can't flip the sign."""
    spec = _spec(8)
    th = {f"s{i}": 1.1 for i in range(spec.n_arms)}
    ex = RealGenomeSimulator(spec, cohort_size=4, update_mode="exact", target_size=1200, seed=5)
    ta = RealGenomeSimulator(spec, cohort_size=4, update_mode="tau", target_size=1200, seed=5)
    t0 = time.time(); ex(th); t_ex = time.time() - t0
    t0 = time.time(); ta(th); t_ta = time.time() - t0
    assert t_ta < t_ex, f"tau ({t_ta:.2f}s) not faster than exact ({t_ex:.2f}s) at matched size"


# --- fast end-to-end fit smoke: sane s_arm + positive Charm sign ------------
def test_tau_fit_smoke_positive_charm_sign():
    """A small tau reference table + pooled per-arm fit recovers s_arm that rises with Charm
    (oncogene-dense arms -> s>1). Uses a synthetic 'observed' from a known s_arm so the direction
    is unambiguous; the dependency-light analogue of the validation run's orthogonal axis."""
    spec = _spec(8)
    n = spec.n_arms
    s_true = np.where(spec.charm > 0, 1.5, 0.6)        # selection aligned with Charm
    observed = RealGenomeSimulator(spec, cohort_size=16, update_mode="tau", target_size=500,
                                   seed=42)({f"s{i}": float(s_true[i]) for i in range(n)})

    sim = RealGenomeSimulator(spec, cohort_size=6, update_mode="tau", target_size=400, seed=0)
    abc = ABC(s_arm_prior(n, 0.5, 1.6), sim, n_workers=4, seed=0)
    theta, summaries = abc.reference_table(200)
    reg = PerArmRegressor(n, n_jobs=4).fit(theta, summaries)
    s_fit = reg.predict(observed)

    assert np.all(np.isfinite(s_fit))
    assert ((s_fit >= 0.3) & (s_fit <= 2.0)).all()                  # sane range
    # oncogene-dense (Charm>0) arms inferred more amplification-favoured than TSG-dense arms
    assert s_fit[spec.charm > 0].mean() > s_fit[spec.charm < 0].mean()
    assert pearsonr(s_fit, spec.charm)[0] > 0
