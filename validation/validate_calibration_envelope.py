"""Closing the loop: parameters inferred from data land in the operating envelope's good region.

The operating envelope (DESIGN_operating_envelope.md) reports which parameter ranges yield realistic
tumours; calibration (DESIGN_inference.md) fits those parameters to data. This script verifies the two
agree — *fitting lands you in the good region* — which is the guarantee that makes both useful:

  1. **Prior audit.** Sample the ABC search prior over the evolutionary rates, grow each config, and run
     the read-only QC ``tumor.diagnose()``. Reports the fraction of the prior that is non-degenerate
     and which degenerate zones it leaks into. This is how the envelope *caught* a founder-extinction
     bottleneck in the inference base config (single founder -> ~5-8% silent extinction); with the fix
     (``initial_cancer_cells = 5``) the leak that remains is only the top of the dispersal prior
     (well-mixed), i.e. the search is essentially inside the good region.

  2. **Posterior check.** Run ABC-rf recovery on realistic ground-truth tumours (the M1 experiment) and
     confirm the inferred point estimates (a) fall inside the reported good ranges and (b) regrow into
     NON-degenerate tumours (diagnose passes) — so inference does not drift into a degenerate corner.

Produces the paper repo's figures/validation_calibration_envelope.png.
Usage:  python validation/validate_calibration_envelope.py [--quick]
"""
import argparse
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "validation"))

from iscc.inference.tumor import TumorSimulator, default_prior, default_base_config, PARAM_PATHS  # noqa: E402
from iscc.inference.abc import ABC  # noqa: E402
from _paths import figure_path

OUT = figure_path("validation_calibration_envelope.png")
# the good ranges reported by the operating-envelope sweep (DESIGN_operating_envelope.md tab:envelope)
GOOD_RANGES = {"mutation_rate": (0.02, 4.0), "amp_prob": (0.1, 1.0),
               "dispersal_rate": (0.0, 1.0), "driver_effects": (1.0, 3.0)}
AUDIT_PARAMS = ("mutation_rate", "amp_prob", "dispersal_rate", "driver_effects")
REC_PARAMS = ("mutation_rate", "amp_prob")
TRUTH_RANGES = {"mutation_rate": (0.1, 0.55), "amp_prob": (0.15, 0.85)}


def _diagnose_config(overrides, base, n_steps, seed, initial_cancer_cells=None):
    """Grow a base config with parameter overrides and return the diagnosis (read-only QC)."""
    import copy
    from iscc.tumor.models import GenotypeTumor
    cfg = copy.deepcopy(base)
    if initial_cancer_cells is not None:
        cfg["deme_params"]["initial_cancer_cells"] = initial_cancer_cells
    for name, val in overrides.items():
        section, key = PARAM_PATHS[name]
        cfg[section][key] = float(val)
    t = GenotypeTumor(genome_params=cfg["genome_params"], selection_params=cfg["selection_params"],
                      cancer_cell_params=cfg["cancer_cell_params"], deme_params=cfg["deme_params"],
                      spatial_params=cfg["spatial_params"], seed=seed)
    t.grow(n_steps=n_steps, seed=seed)
    return t.diagnose()


def prior_audit(n=40, n_steps=700, founders=(1, 5), seed=1):
    """Fraction of the ABC prior that grows a non-degenerate tumour, per founder-seeding choice."""
    prior = default_prior(AUDIT_PARAMS)
    rng = np.random.default_rng(seed)
    draws = prior.sample(rng, n)
    base = default_base_config()
    out = {}
    for ic in founders:
        ok = 0
        flags = {}
        for i in range(n):
            theta = {p: float(draws[i, j]) for j, p in enumerate(AUDIT_PARAMS)}
            d = _diagnose_config(theta, base, n_steps, seed=i, initial_cancer_cells=ic)
            if d.ok:
                ok += 1
            for c in d.failures:
                flags[c.name] = flags.get(c.name, 0) + 1
        out[ic] = dict(coverage=ok / n, flags=flags)
        print(f"prior audit (initial_cancer_cells={ic}): {ok}/{n} non-degenerate; leaks={flags}")
    return out


def posterior_check(n_ref=500, n_truths=16, n_steps=800, n_workers=8, seed=0):
    """ABC-rf recovery on realistic truths, then diagnose the tumour regrown at each inferred MAP."""
    sim = TumorSimulator(n_steps=n_steps, seed=0)
    prior = default_prior(REC_PARAMS)
    abc = ABC(prior, sim, n_workers=n_workers, seed=seed)
    print(f"building ABC reference table ({n_ref} sims)...")
    reference = abc.reference_table(n_ref)

    rng = np.random.default_rng(seed + 123)
    truths, maps, map_ok = [], [], []
    for k in range(n_truths):
        truth = {p: float(rng.uniform(*TRUTH_RANGES[p])) for p in REC_PARAMS}
        obs = TumorSimulator(n_steps=n_steps, seed=10_000 + k, n_replicates=3)(truth)
        post = abc.run(obs, accept_frac=0.15, reference=reference, project="rf")
        mp = post.map()
        # regrow at the inferred MAP and run the operating-envelope QC
        d = _diagnose_config(dict(zip(REC_PARAMS, mp)), sim.base_config, n_steps, seed=20_000 + k)
        truths.append([truth[p] for p in REC_PARAMS]); maps.append(mp); map_ok.append(d.ok)
        print(f"  truth {k+1:2d}/{n_truths}: "
              + ", ".join(f"{p}={truth[p]:.3f}->{mp[i]:.3f}" for i, p in enumerate(REC_PARAMS))
              + f"  diagnose={'ok' if d.ok else 'DEGEN'}")
    return dict(names=list(REC_PARAMS), truths=np.array(truths), maps=np.array(maps),
                map_ok=np.array(map_ok))


def in_good_range(name, values):
    lo, hi = GOOD_RANGES[name]
    return (values >= lo) & (values <= hi)


def make_figure(audit, rec, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names, truths, maps, ok = rec["names"], rec["truths"], rec["maps"], rec["map_ok"]
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5))

    # A,B: inferred vs true for each recovered rate, coloured by whether the regrown tumour is OK,
    # with the reported good range shaded.
    for i, name in enumerate(names):
        a = ax[i]
        lo, hi = GOOD_RANGES[name]
        a.axhspan(lo, hi, color="#31a354", alpha=0.10, label="good range")
        t, m = truths[:, i], maps[:, i]
        a.scatter(t[ok], m[ok], c="#31a354", s=45, edgecolor="k", lw=0.4,
                  label="inferred (non-degenerate)")
        if (~ok).any():
            a.scatter(t[~ok], m[~ok], c="#d94801", s=45, edgecolor="k", lw=0.4, label="degenerate")
        span = [min(t.min(), m.min()), max(t.max(), m.max())]
        a.plot(span, span, "k--", lw=1, label="truth = estimate")
        r = np.corrcoef(t, m)[0, 1]
        frac = in_good_range(name, m).mean()
        a.set(xlabel=f"true {name}", ylabel=f"inferred {name}",
              title=f"{name}: r={r:.2f}\n{frac:.0%} of estimates in good range")
        a.legend(fontsize=7, loc="upper left")

    # C: good-region coverage — ABC prior (single founder) vs prior (fixed) vs inferred MAPs.
    labels = ["ABC prior\n(1 founder)", "ABC prior\n(fixed: 5)", "inferred\nMAPs"]
    vals = [audit[1]["coverage"], audit[5]["coverage"], float(ok.mean())]
    colors = ["#fdae6b", "#a1d99b", "#31a354"]
    bars = ax[2].bar(labels, [v * 100 for v in vals], color=colors, edgecolor="k")
    for b, v in zip(bars, vals):
        ax[2].text(b.get_x() + b.get_width() / 2, v * 100 + 1, f"{v:.0%}", ha="center", fontsize=10)
    ax[2].set(ylabel="% non-degenerate", ylim=(0, 108),
              title="Fitting lands in the good region\n(founder fix removes the extinction leak)")
    leak1 = audit[1]["flags"]
    if leak1:
        ax[2].text(0.02, -0.28, "prior leaks: " + ", ".join(f"{k}×{v}" for k, v in leak1.items()),
                   transform=ax[2].transAxes, fontsize=8, color="0.3")

    fig.suptitle("Closing the loop: parameters inferred from data land in the operating envelope",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print("figure ->", out_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--quick", action="store_true", help="tiny run for a smoke check")
    args = ap.parse_args()
    n_ref, n_truths, n_audit = (120, 6, 16) if args.quick else (500, 16, 40)

    audit = prior_audit(n=n_audit)
    rec = posterior_check(n_ref=n_ref, n_truths=n_truths)

    print("\n=== closing the loop ===")
    print(f"  ABC prior non-degenerate: {audit[1]['coverage']:.0%} (1 founder) "
          f"-> {audit[5]['coverage']:.0%} (fixed)")
    print(f"  inferred MAPs non-degenerate: {rec['map_ok'].mean():.0%}")
    for i, name in enumerate(rec["names"]):
        print(f"  {name}: {in_good_range(name, rec['maps'][:, i]).mean():.0%} of estimates in good range")
    make_figure(audit, rec, args.out)


if __name__ == "__main__":
    main()
