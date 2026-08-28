"""Parameter recovery for iscc's CNA/SNV rates (DESIGN_inference A.4.1 — the M1 deliverable).

The bar a simulator paper must clear is **parameter estimation from real data**; the lowest-risk,
external-data-free demonstration of that capability is *recovery*: draw known ground-truth
parameters, simulate synthetic tumours, run inference, and show the posterior concentrates on the
truth with calibrated uncertainty. This is the abstract-genome recovery experiment (no download
needed); fit-to-real PCAWG and the Charm correlation come later (M3b).

We infer two of the rates exposed in A.0:
  * ``mutation_rate`` — the per-division probability of a genome-altering event (identified mainly
    by the SNV burden / number of segregating sites),
  * ``amp_prob`` — P(amplification | CNA event), i.e. the amplification-vs-deletion balance
    (identified by the per-segment copy-number gain-vs-loss frequencies).

Method (``iscc.inference``): rejection + regression-adjustment ABC with a random-forest
projection of the summary vector into parameter space (semi-automatic ABC), the dependency-light
Python analogue of CINner's ABC-rf. A single reference table of simulations is reused across all
ground-truth tumours.

Produces the paper repo's figures/validation_inference_recovery.png.
Usage:  python validation/validate_inference_recovery.py [--n-ref N] [--n-truths K] [--quick]
"""
import argparse
import os
import warnings

import numpy as np

from iscc.inference.abc import ABC
from iscc.inference.tumor import TumorSimulator, default_prior
from _paths import figure_path

warnings.filterwarnings("ignore")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PARAMS = ("mutation_rate", "amp_prob")
# ground-truth ranges to draw replicate truths from (inside the prior support)
TRUTH_RANGES = {"mutation_rate": (0.1, 0.55), "amp_prob": (0.15, 0.85)}
N_STEPS = 800
ACCEPT_FRAC = 0.15
LEVEL = 0.9


def run_recovery(n_ref=800, n_truths=24, n_workers=8, seed=0):
    sim = TumorSimulator(n_steps=N_STEPS, seed=0)
    prior = default_prior(PARAMS)
    abc = ABC(prior, sim, n_workers=n_workers, seed=seed)

    print(f"simulating reference table ({n_ref} sims)...")
    reference = abc.reference_table(n_ref)
    print(f"  survivors: {reference[0].shape[0]}/{n_ref}")

    rng = np.random.default_rng(seed + 123)
    truths, maps, rfs, los, his = [], [], [], [], []
    for k in range(n_truths):
        truth = {p: float(rng.uniform(*TRUTH_RANGES[p])) for p in PARAMS}
        # observed = a fresh ground-truth tumour (averaged over a few replicates to tame noise)
        obs = TumorSimulator(n_steps=N_STEPS, seed=10_000 + k, n_replicates=3)(truth)
        post = abc.run(obs, accept_frac=ACCEPT_FRAC, reference=reference, project="rf")
        lo, hi = post.credible_interval(LEVEL)
        truths.append([truth[p] for p in PARAMS])
        maps.append(post.map()); rfs.append(post.rf_estimate)
        los.append(lo); his.append(hi)
        print(f"  truth {k + 1:2d}/{n_truths}: "
              + ", ".join(f"{p}={truth[p]:.3f}->{post.map()[i]:.3f}" for i, p in enumerate(PARAMS)))

    return dict(names=list(PARAMS), truths=np.array(truths), maps=np.array(maps),
                rfs=np.array(rfs), lo=np.array(los), hi=np.array(his),
                reference=reference, abc=abc, prior=prior)


def _example_posterior(res, target=(0.35, 0.7)):
    """One posterior cloud near a chosen truth, for an illustrative scatter panel."""
    truth = dict(zip(res["names"], target))
    obs = TumorSimulator(n_steps=N_STEPS, seed=77, n_replicates=4)(truth)
    post = res["abc"].run(obs, accept_frac=ACCEPT_FRAC, reference=res["reference"], project="rf")
    return target, post


def make_figure(res, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names, truths, maps = res["names"], res["truths"], res["maps"]
    lo, hi = res["lo"], res["hi"]
    n_par = len(names)

    fig, axes = plt.subplots(1, n_par + 1, figsize=(6 * (n_par + 1), 5.2))

    coverage = {}
    for i, name in enumerate(names):
        ax = axes[i]
        t, m = truths[:, i], maps[:, i]
        yerr = np.vstack([m - lo[:, i], hi[:, i] - m])
        ax.errorbar(t, m, yerr=yerr, fmt="o", ms=5, capsize=2, alpha=0.8,
                    color="tab:blue", ecolor="0.7", label=f"posterior (MAP ± {int(LEVEL*100)}% CI)")
        span = [min(t.min(), m.min()), max(t.max(), m.max())]
        ax.plot(span, span, "k--", lw=1, label="truth = estimate")
        r = np.corrcoef(t, m)[0, 1]
        rmse = float(np.sqrt(np.mean((m - t) ** 2)))
        cov = float(np.mean((lo[:, i] <= t) & (t <= hi[:, i])))
        coverage[name] = cov
        ax.set_xlabel(f"true {name}"); ax.set_ylabel(f"inferred {name}")
        ax.set_title(f"{name}\nr={r:.2f}, RMSE={rmse:.3f}, {int(LEVEL*100)}% CI cov={cov:.0%}")
        ax.legend(fontsize=8, loc="upper left")

    # illustrative posterior cloud near a chosen truth
    target, post = _example_posterior(res)
    ax = axes[-1]
    ax.scatter(post.all_theta[:, 0], post.all_theta[:, 1], s=8, color="0.8",
               label="prior draws")
    ax.scatter(post.samples[:, 0], post.samples[:, 1], s=14, color="tab:orange",
               alpha=0.7, label="accepted posterior")
    ax.scatter([target[0]], [target[1]], marker="*", s=320, color="red",
               edgecolor="k", zorder=5, label="truth")
    ax.set_xlabel(names[0]); ax.set_ylabel(names[1])
    ax.set_title("Posterior concentrates on the truth\n(example tumour)")
    ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("iscc recovers known CNA/SNV rates from synthetic tumours (ABC-rf parameter recovery)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    return coverage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-ref", type=int, default=800)
    ap.add_argument("--n-truths", type=int, default=24)
    ap.add_argument("--n-workers", type=int, default=8)
    ap.add_argument("--quick", action="store_true", help="tiny run for a smoke check")
    args = ap.parse_args()
    if args.quick:
        args.n_ref, args.n_truths = 120, 6

    res = run_recovery(n_ref=args.n_ref, n_truths=args.n_truths, n_workers=args.n_workers)
    out = figure_path("validation_inference_recovery.png")
    coverage = make_figure(res, out)

    print("\n=== parameter recovery summary ===")
    for i, name in enumerate(res["names"]):
        t, m = res["truths"][:, i], res["maps"][:, i]
        r = np.corrcoef(t, m)[0, 1]
        rmse = float(np.sqrt(np.mean((m - t) ** 2)))
        print(f"  {name:14s} r={r:.3f}  RMSE={rmse:.3f}  {int(LEVEL*100)}% CI coverage={coverage[name]:.0%}")
    print(f"figure -> {out}")


if __name__ == "__main__":
    main()
