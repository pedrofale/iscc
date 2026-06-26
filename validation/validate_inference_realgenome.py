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
CGC, Davoli Charm, PCAWG). Produces manuscript/figures/validation_inference_realgenome.png.

**Compute note:** a publication-quality fit needs a large reference table (each parameter set is a
*cohort* of tumours and the engine does one event per update — DESIGN_scalability §7), so the
default settings are sized for an **HPC run**. ``--quick`` runs a small local smoke check that
confirms the pipeline and the direction of the result, not the final figure.

Usage:  python validation/validate_inference_realgenome.py [--n-ref N] [--cancer-type T] [--quick]
"""
import argparse
import os
import warnings

import numpy as np
from scipy.stats import pearsonr, spearmanr

from iscc.inference.abc import ABC
from iscc.inference.genome import load_default, load_real_cna_profile
from iscc.inference.realgenome import RealGenomeSimulator, s_arm_prior, PerArmRegressor

warnings.filterwarnings("ignore")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

N_STEPS = 1000
COHORT_SIZE = 10          # tumours per parameter set (the simulated "cohort", like PCAWG donors)
S_PRIOR = (0.4, 1.8)


def run_fit(cancer_type="BRCA-UK", n_ref=3000, n_steps=N_STEPS, cohort_size=COHORT_SIZE,
            valid_cohort=40, n_workers=8, seed=0):
    spec = load_default()
    arm_names = spec.arm_names
    n = spec.n_arms

    # observed PCAWG per-arm gain/loss (cohort frequency over donors) -> the cohort summary layout
    obs_arms, gain, loss = load_real_cna_profile(cancer_type=cancer_type)
    assert obs_arms == arm_names, "arm order mismatch between cna and arms tables"
    observed = np.concatenate([gain, loss]).astype(float)

    sim = RealGenomeSimulator(spec, n_steps=n_steps, cohort_size=cohort_size, seed=0)
    prior = s_arm_prior(n, *S_PRIOR)
    abc = ABC(prior, sim, n_workers=n_workers, seed=seed)

    print(f"simulating reference table ({n_ref} param sets x {cohort_size} tumours, {n} arms)...")
    theta, summaries = abc.reference_table(n_ref)
    print(f"  survivors: {theta.shape[0]}/{n_ref}")

    # pooled per-arm regressor: (gain[arm], loss[arm]) -> s[arm]
    reg = PerArmRegressor(n, n_jobs=n_workers).fit(theta, summaries)
    s_fit = reg.predict(observed)

    # fit-to-real check: re-simulate with the fitted s_arm (large cohort) and compare to observed
    print("re-simulating with fitted s_arm (validation cohort)...")
    fit_sim = RealGenomeSimulator(spec, n_steps=n_steps, cohort_size=valid_cohort, seed=7)
    fitted_summary = fit_sim({f"s{i}": float(s_fit[i]) for i in range(n)})
    fit_gain, fit_loss = fitted_summary[:n], fitted_summary[n:]

    return dict(spec=spec, arm_names=arm_names, observed_gain=gain, observed_loss=loss,
                s_fit=s_fit, fit_gain=fit_gain, fit_loss=fit_loss, regressor=reg,
                charm=spec.charm, content=spec.content_score(), cancer_type=cancer_type,
                n_ref=theta.shape[0], cohort_size=cohort_size, n_steps=n_steps)


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


def make_figure(res, stats, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = res["arm_names"]
    g_obs, l_obs = res["observed_gain"], res["observed_loss"]
    g_fit, l_fit = res["fit_gain"], res["fit_loss"]
    s, charm = res["s_fit"], res["charm"]
    x = np.arange(len(arms))

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

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

    scale = (f"reference: {res['n_ref']} param sets x {res['cohort_size']} tumours, "
             f"{res['n_steps']} steps")
    small = res["n_ref"] * res["cohort_size"] < 20000
    caveat = "  —  SMALL-SCALE SMOKE RUN (publication fit needs HPC scale)" if small else ""
    fig.suptitle("iscc real-genome mode: fit-to-real PCAWG copy-number landscape + "
                 "orthogonal Charm correlation (CINner-parity)\n" + scale + caveat,
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-ref", type=int, default=3000, help="reference param sets (HPC-sized)")
    ap.add_argument("--n-steps", type=int, default=N_STEPS)
    ap.add_argument("--cohort-size", type=int, default=COHORT_SIZE)
    ap.add_argument("--valid-cohort", type=int, default=40)
    ap.add_argument("--n-workers", type=int, default=8)
    ap.add_argument("--cancer-type", type=str, default="BRCA-UK")
    ap.add_argument("--quick", action="store_true",
                    help="small local smoke check (pipeline + direction, not the final figure)")
    args = ap.parse_args()
    if args.quick:
        args.n_ref, args.n_steps, args.cohort_size, args.valid_cohort = 250, 700, 6, 16

    res = run_fit(cancer_type=args.cancer_type, n_ref=args.n_ref, n_steps=args.n_steps,
                  cohort_size=args.cohort_size, valid_cohort=args.valid_cohort,
                  n_workers=args.n_workers)
    stats = _report(res)
    out = os.path.join(REPO, "manuscript/figures/validation_inference_realgenome.png")
    make_figure(res, stats, out)

    print("\n=== M3b fit-to-real + Charm summary ===")
    print(f"  cancer type           : {res['cancer_type']}")
    print(f"  fit-to-real           : r={stats['r_fit']:.3f}  RMSE={stats['rmse']:.3f}")
    print(f"  s_arm vs Charm        : Pearson r={stats['r_charm']:.3f} (p={stats['p_charm']:.2e}), "
          f"Spearman ρ={stats['rho_charm']:.3f}")
    print(f"  s_arm vs OG-TSG count : Pearson r={stats['r_content']:.3f} (p={stats['p_content']:.2e})")
    print(f"figure -> {out}")


if __name__ == "__main__":
    main()
