"""Fit-to-real PCAWG + Charm correlation — the M3b / CINner-parity result (A.4.2-3).

This is the flagship copy-number inference experiment. Using iscc's **real-genome mode** (the ~39
human chromosome arms as segments, per-arm copy-number selection ``s_arm``; DESIGN_inference A.5):

  1. **Fit-to-real (A.4.2):** RF-ABC infers the per-arm selection vector ``s_arm`` so the simulated
     per-arm gain/loss landscape matches a real cancer type's PCAWG profile, then we re-simulate
     with the fitted ``s_arm`` and show the fitted simulator reproduces the observed per-arm
     gain/loss frequencies.
  2. **Orthogonal (A.4.3):** the inferred ``s_arm`` is correlated with the **independent** Davoli
     2013 Charm score (and, as a second axis, the COSMIC per-arm oncogene-minus-TSG content) —
     iscc recovers the known biology that oncogene-dense arms are selected for amplification and
     TSG-dense arms for deletion (CINner's headline orthogonal validation).

Identifiability (A.3): the CNA *rate* is fixed; only ``s_arm`` is inferred (CINner fixes the
missegregation probability for the same reason). Estimation is dependency-light (numpy + sklearn):
the arm-CN fitness factorises over arms, so we infer the per-arm coefficients with a **pooled
per-arm random-forest regressor** (``PerArmRegressor``) — every (reference-sim, arm) pair is a
training example for one ``(gain, loss) -> s`` map. Reference sims are parallelised with
multiprocessing.

External data: produced + cited by ``validation/data/build_realgenome_data.py`` (cytoBand, COSMIC
CGC, Davoli Charm, PCAWG). Produces the paper repo's figures/validation_inference_realgenome.png.

**Tau-leaping (DESIGN_scalability §7) makes the scaled fit feasible.** Each cohort tumour now grows
with **tau-leaping** (``update_mode="tau"``, the default): the whole tumour is advanced one
generation per step (Poisson births/deaths per clone) instead of one event per update. Growth is
bounded by **size** — every tumour is grown to ``target_size`` cancer cells, then its per-arm
landscape is read — so per-sim cost is bounded regardless of seed (the tau engine grows
exponentially with no hard carrying-cap; a fixed generation budget would let lucky seeds explode
into multi-minute stragglers). At a *matched* tumour size tau and exact produce statistically the
**same** per-arm gain/loss summary (the ``--equivalence`` gate below, and
``tests/test_realgenome_tau.py``; no systematic bias), and tau is faster at every size with the
gap **widening with size** — ~1.2x at 500 cells, ~2.3x at 1500, ~6.3x at 4000 (exact reaches ~15 s
per cohort tumour at 4000 and becomes the HPC bottleneck; tau stays ~2.4 s). ``--update-mode exact``
recovers the one-event reference engine.

**HPC settings for the final figure.** The in-session run already reaches ~26x the prior smoke
scale. For the canonical figure use a larger, more-developed tumour where the tau speedup compounds
and the per-arm landscape is strongest, e.g. ``--n-ref 20000 --cohort-size 16 --target-size 4000
--valid-cohort 200 --n-workers <ncores>`` — feasible on a single node precisely because tau is ~6x+
the exact engine at that size.

Usage:  python validation/validate_inference_realgenome.py [--n-ref N] [--cancer-type T]
                 [--target-size S] [--update-mode tau|exact] [--equivalence] [--quick]
"""
import argparse
import os
import time
import warnings

import numpy as np
from scipy.stats import pearsonr, spearmanr

from iscc.inference.abc import ABC
from iscc.inference.genome import load_default, load_real_cna_profile
from iscc.inference.realgenome import (
    RealGenomeSimulator, s_arm_prior, PerArmRegressor, matched_size_cohort_vector,
)
from _paths import figure_path

warnings.filterwarnings("ignore")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

N_STEPS = 1000            # exact engine fallback: per-event step budget (turnover-heavy)
COHORT_SIZE = 8           # tumours per parameter set (the simulated "cohort", like PCAWG donors)
TARGET_SIZE = 600         # grow each cohort tumour to this cancer-cell count, then read its
                          # per-arm landscape. Bounded by SIZE (not generations): the tau engine
                          # grows exponentially with no hard carrying-cap, so a fixed generation
                          # budget lets lucky seeds explode into multi-minute stragglers; a size cap
                          # bounds every sim and is also the clean matched-size tau/exact comparison.
S_PRIOR = (0.4, 1.8)


def run_fit(cancer_type="BRCA-UK", n_ref=4000, target_size=TARGET_SIZE, cohort_size=COHORT_SIZE,
            valid_cohort=60, n_workers=6, seed=0, update_mode="tau"):
    spec = load_default()
    arm_names = spec.arm_names
    n = spec.n_arms

    # observed PCAWG per-arm gain/loss (cohort frequency over donors) -> the cohort summary layout
    obs_arms, gain, loss = load_real_cna_profile(cancer_type=cancer_type)
    assert obs_arms == arm_names, "arm order mismatch between cna and arms tables"
    observed = np.concatenate([gain, loss]).astype(float)

    sim = RealGenomeSimulator(spec, cohort_size=cohort_size, seed=0,
                              update_mode=update_mode, target_size=target_size)
    prior = s_arm_prior(n, *S_PRIOR)
    abc = ABC(prior, sim, n_workers=n_workers, seed=seed)

    print(f"simulating reference table ({n_ref} param sets x {cohort_size} tumours, {n} arms; "
          f"update_mode={update_mode}, grow-to-size={target_size})...")
    t0 = time.time()
    theta, summaries = abc.reference_table(n_ref)
    ref_secs = time.time() - t0
    print(f"  survivors: {theta.shape[0]}/{n_ref}  ({ref_secs:.0f}s, "
          f"{ref_secs / max(n_ref, 1) * 1000:.0f} ms/param-set)")

    # pooled per-arm regressor: (gain[arm], loss[arm]) -> s[arm]
    reg = PerArmRegressor(n, n_jobs=n_workers).fit(theta, summaries)
    s_fit = reg.predict(observed)

    # fit-to-real check: re-simulate with the fitted s_arm (large cohort) and compare to observed
    print("re-simulating with fitted s_arm (validation cohort)...")
    fit_sim = RealGenomeSimulator(spec, cohort_size=valid_cohort, seed=7,
                                  update_mode=update_mode, target_size=target_size)
    fitted_summary = fit_sim({f"s{i}": float(s_fit[i]) for i in range(n)})
    fit_gain, fit_loss = fitted_summary[:n], fitted_summary[n:]

    return dict(spec=spec, arm_names=arm_names, observed_gain=gain, observed_loss=loss,
                s_fit=s_fit, fit_gain=fit_gain, fit_loss=fit_loss, regressor=reg,
                charm=spec.charm, content=spec.content_score(), cancer_type=cancer_type,
                n_ref=theta.shape[0], cohort_size=cohort_size, target_size=target_size,
                update_mode=update_mode, ref_secs=ref_secs)


def equivalence_and_speedup(spec, n_vectors=3, target=TARGET_SIZE, cohort=16,
                            speedup_sizes=(600, 1500, 4000), seed=0):
    """The correctness gate + speedup, quantified (Tasks 2 & 3 of the tau-scaling work).

    * **Equivalence (matched size):** for several random ``s_arm`` vectors, grow cohorts to the same
      target size under exact and tau and compare the per-arm gain/loss summary. We read the
      Monte-Carlo noise floor off two independent *exact* cohorts (the exact-vs-exact correlation /
      mean-abs-diff), then check tau tracks exact as closely (no systematic bias, agreement within
      that floor). Matched size controls for developmental stage, isolating the update rule.
    * **Speedup (matched size, size-scaling):** wall-time of a tau vs an exact cohort draw grown to
      the same size, at several sizes. Tau is faster at every size and the gap **widens with tumour
      size** (the exact per-event cost climbs with crowding/clone count faster than tau's
      per-generation cost) — which is exactly why the large, deeply-developed tumours are the ones
      that were HPC-bound under exact.
    """
    rng = np.random.default_rng(seed)
    seeds_a = list(range(cohort))
    seeds_b = list(range(1000, 1000 + cohort))
    r_te, r_ee, biases, d_te, d_ee = [], [], [], [], []
    ref_vec = None
    for k in range(n_vectors):
        s = rng.uniform(*S_PRIOR, spec.n_arms)
        ea, _ = matched_size_cohort_vector(spec, s, "exact", seeds_a, target)
        eb, _ = matched_size_cohort_vector(spec, s, "exact", seeds_b, target)
        ta, _ = matched_size_cohort_vector(spec, s, "tau", seeds_a, target)
        ok = np.isfinite(ea) & np.isfinite(eb) & np.isfinite(ta)
        r_te.append(pearsonr(ta[ok], ea[ok])[0])
        r_ee.append(pearsonr(eb[ok], ea[ok])[0])
        biases.append(float(np.mean(ta[ok] - ea[ok])))
        d_te.append(float(np.mean(np.abs(ta[ok] - ea[ok]))))
        d_ee.append(float(np.mean(np.abs(eb[ok] - ea[ok]))))
        if k == 0:
            ref_vec = (ea, ta)   # for the figure panel

    # speedup vs tumour size (matched grow-to-size, both engines, small cohort, timed)
    th = {f"s{i}": 1.1 for i in range(spec.n_arms)}
    scost = []
    for sz in speedup_sizes:
        ex = RealGenomeSimulator(spec, cohort_size=6, update_mode="exact", target_size=sz)
        tas = RealGenomeSimulator(spec, cohort_size=6, update_mode="tau", target_size=sz)
        t0 = time.time(); ex(th); te = time.time() - t0
        t0 = time.time(); tas(th); tt = time.time() - t0
        scost.append((sz, te / 6, tt / 6, te / max(tt, 1e-9)))
    prod = next((c for c in scost if c[0] == target), scost[0])

    return dict(
        r_tau_exact=float(np.mean(r_te)), r_exact_exact=float(np.mean(r_ee)),
        bias=float(np.mean(biases)), mad_tau_exact=float(np.mean(d_te)),
        mad_exact_exact=float(np.mean(d_ee)),
        speedup_by_size=scost,                 # [(size, exact_s/sim, tau_s/sim, speedup), ...]
        t_exact_cohort=prod[1] * 6, t_tau_cohort=prod[2] * 6, speedup=prod[3],
        n_vectors=n_vectors, target=target, cohort=cohort, ref_vec=ref_vec)


def _report(res):
    g_obs, l_obs = res["observed_gain"], res["observed_loss"]
    g_fit, l_fit = res["fit_gain"], res["fit_loss"]
    ok = np.isfinite(g_fit) & np.isfinite(l_fit)
    # fit-to-real: correlation of fitted vs observed across arms (gains and losses pooled)
    obs_all = np.concatenate([g_obs[ok], l_obs[ok]])
    fit_all = np.concatenate([g_fit[ok], l_fit[ok]])
    r_fit = pearsonr(obs_all, fit_all)[0]
    rmse = float(np.sqrt(np.mean((obs_all - fit_all) ** 2)))

    # orthogonal: inferred s_arm vs Charm and vs content (oncogene - TSG)
    s = res["s_fit"]
    charm = res["charm"]
    content = res["content"]
    mc = np.isfinite(charm) & np.isfinite(s)
    r_charm, p_charm = pearsonr(s[mc], charm[mc])
    rho_charm, _ = spearmanr(s[mc], charm[mc])
    r_content, p_content = pearsonr(s, content)
    return dict(r_fit=r_fit, rmse=rmse, r_charm=r_charm, p_charm=p_charm,
                rho_charm=rho_charm, r_content=r_content, p_content=p_content)


def make_figure(res, stats, out_path, equiv=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = res["arm_names"]
    g_obs, l_obs = res["observed_gain"], res["observed_loss"]
    g_fit, l_fit = res["fit_gain"], res["fit_loss"]
    s, charm = res["s_fit"], res["charm"]
    x = np.arange(len(arms))

    ncols = 4 if equiv is not None else 3
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 6))

    # panel 1: per-arm gain/loss, observed vs fitted (loss drawn negative)
    ax = axes[0]
    ax.bar(x - 0.2, g_obs, width=0.4, color="tab:red", alpha=0.6, label="PCAWG gain")
    ax.bar(x - 0.2, -l_obs, width=0.4, color="tab:blue", alpha=0.6, label="PCAWG loss")
    ax.plot(x + 0.2, g_fit, "o", color="darkred", ms=4, label="iscc gain (fitted)")
    ax.plot(x + 0.2, -l_fit, "o", color="darkblue", ms=4, label="iscc loss (fitted)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels(arms, rotation=90, fontsize=6)
    ax.set_ylabel("loss   <-  frequency  ->   gain")
    ax.set_title(f"Per-arm CNA landscape: {res['cancer_type']} vs fitted iscc")
    ax.legend(fontsize=8)

    # panel 2: fit-to-real scatter (observed vs fitted, gains and losses)
    ax = axes[1]
    ax.scatter(g_obs, g_fit, color="tab:red", alpha=0.7, label="gain")
    ax.scatter(l_obs, l_fit, color="tab:blue", alpha=0.7, label="loss")
    lim = [0, max(np.nanmax(g_obs), np.nanmax(l_obs), 0.1) * 1.1]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlabel("observed frequency (PCAWG)"); ax.set_ylabel("fitted frequency (iscc)")
    ax.set_title(f"Fit-to-real\nr={stats['r_fit']:.2f}, RMSE={stats['rmse']:.3f}")
    ax.legend(fontsize=8)

    # panel 3: inferred s_arm vs Davoli Charm (orthogonal)
    ax = axes[2]
    mc = np.isfinite(charm) & np.isfinite(s)
    ax.scatter(charm[mc], s[mc], color="tab:green", alpha=0.7)
    for i in np.where(mc)[0]:
        ax.annotate(arms[i], (charm[i], s[i]), fontsize=5, alpha=0.6)
    if mc.sum() > 1:
        b, a = np.polyfit(charm[mc], s[mc], 1)
        xs = np.linspace(charm[mc].min(), charm[mc].max(), 20)
        ax.plot(xs, a + b * xs, "k--", lw=1)
    ax.axhline(1.0, color="0.6", lw=0.5)
    ax.set_xlabel("Davoli Charm score (OG-TSG; + = amplification-favouring)")
    ax.set_ylabel("inferred s_arm (>1 = amplification-favoured)")
    ax.set_title(f"Orthogonal: inferred selection vs Charm\n"
                 f"Pearson r={stats['r_charm']:.2f} (p={stats['p_charm']:.1e}), "
                 f"Spearman ρ={stats['rho_charm']:.2f}")

    # panel 4 (optional): tau-vs-exact equivalence gate — per-arm summary at matched size
    if equiv is not None:
        ax = axes[3]
        ea, ta = equiv["ref_vec"]
        ok = np.isfinite(ea) & np.isfinite(ta)
        ax.scatter(ea[ok], ta[ok], color="tab:purple", alpha=0.7)
        lim = [0, max(np.nanmax(ea[ok]), np.nanmax(ta[ok]), 0.05) * 1.1]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_xlabel("exact-engine per-arm freq"); ax.set_ylabel("tau-engine per-arm freq")
        sizes_str = ", ".join(f"{int(sz)}c:{sp:.1f}x" for sz, _, _, sp in equiv["speedup_by_size"])
        ax.set_title(
            "Equivalence gate (matched size): tau vs exact summary\n"
            f"r(tau,exact)={equiv['r_tau_exact']:.2f}  vs  noise floor "
            f"r(exact,exact)={equiv['r_exact_exact']:.2f},  bias={equiv['bias']:+.3f}\n"
            f"tau speedup grows with size [{sizes_str}]")

    scale = (f"reference: {res['n_ref']} param sets x {res['cohort_size']} tumours, "
             f"grow-to-{res['target_size']} cells ({res['update_mode']})")
    small = res["n_ref"] * res["cohort_size"] < 20000
    caveat = "  —  SMALL-SCALE SMOKE RUN (publication fit needs HPC scale)" if small else ""
    fig.suptitle("iscc real-genome mode: fit-to-real PCAWG copy-number landscape + "
                 "orthogonal Charm correlation (CINner-parity, tau-leaped)\n" + scale + caveat,
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-ref", type=int, default=4000, help="reference param sets")
    ap.add_argument("--target-size", type=int, default=TARGET_SIZE,
                    help="grow each cohort tumour to this cancer-cell count, then read its landscape")
    ap.add_argument("--cohort-size", type=int, default=COHORT_SIZE)
    ap.add_argument("--valid-cohort", type=int, default=60)
    ap.add_argument("--n-workers", type=int, default=6)
    ap.add_argument("--cancer-type", type=str, default="BRCA-UK")
    ap.add_argument("--update-mode", choices=["tau", "exact"], default="tau",
                    help="tau-leaping (default, fast) or the exact one-event reference engine")
    ap.add_argument("--equivalence", action="store_true",
                    help="also run the tau-vs-exact equivalence gate + speedup benchmark")
    ap.add_argument("--quick", action="store_true",
                    help="small local smoke check (pipeline + direction, not the final figure)")
    args = ap.parse_args()
    if args.quick:
        args.n_ref, args.cohort_size, args.valid_cohort, args.target_size = 300, 6, 16, 400

    equiv = None
    if args.equivalence:
        print("running tau-vs-exact equivalence gate + speedup benchmark...")
        spec0 = load_default()
        equiv = equivalence_and_speedup(
            spec0, n_vectors=2 if args.quick else 3, target=args.target_size,
            cohort=12 if args.quick else 16,
            speedup_sizes=(args.target_size, 1500) if args.quick else (args.target_size, 1500, 4000))
        print(f"  equivalence (matched size {args.target_size}): r(tau,exact)={equiv['r_tau_exact']:.3f}  "
              f"noise floor r(exact,exact)={equiv['r_exact_exact']:.3f}")
        print(f"  bias(tau-exact)={equiv['bias']:+.4f}  "
              f"meanAbsDiff tau-exact={equiv['mad_tau_exact']:.4f} vs exact-exact={equiv['mad_exact_exact']:.4f}")
        for sz, te, tt, sp in equiv["speedup_by_size"]:
            print(f"  speedup @ {int(sz):5d} cells: exact {te:.2f}s/sim -> tau {tt:.2f}s/sim  =>  {sp:.1f}x")

    res = run_fit(cancer_type=args.cancer_type, n_ref=args.n_ref, target_size=args.target_size,
                  cohort_size=args.cohort_size, valid_cohort=args.valid_cohort,
                  n_workers=args.n_workers, update_mode=args.update_mode)
    stats = _report(res)
    out = figure_path("validation_inference_realgenome.png")
    make_figure(res, stats, out, equiv=equiv)

    print("\n=== M3b fit-to-real + Charm summary (tau-leaped) ===")
    print(f"  update mode           : {res['update_mode']}  "
          f"(grow-to-size={res['target_size']}, reference {res['ref_secs']:.0f}s)")
    print(f"  cancer type           : {res['cancer_type']}")
    print(f"  reference table       : {res['n_ref']} param sets x {res['cohort_size']} tumours")
    print(f"  fit-to-real           : r={stats['r_fit']:.3f}  RMSE={stats['rmse']:.3f}")
    print(f"  s_arm vs Charm        : Pearson r={stats['r_charm']:.3f} (p={stats['p_charm']:.2e}), "
          f"Spearman ρ={stats['rho_charm']:.3f}")
    print(f"  s_arm vs OG-TSG count : Pearson r={stats['r_content']:.3f} (p={stats['p_content']:.2e})")
    if equiv is not None:
        print(f"  tau equivalence/speedup: r(tau,exact)={equiv['r_tau_exact']:.2f} "
              f"(floor {equiv['r_exact_exact']:.2f}), bias={equiv['bias']:+.3f}, "
              f"{equiv['speedup']:.1f}x @ {res['target_size']} cells (grows with size)")
    print(f"figure -> {out}")


if __name__ == "__main__":
    main()
