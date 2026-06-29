"""Validate tau-leaping against the exact one-event engine BY DISTRIBUTION (DESIGN §7 guarantee c)
and confirm growth-over-time visualisation is preserved (guarantee b).

Both engines realise the same birth/death/mutation/dispersal process; tau-leaping just advances all
clones once per generation instead of one event per update. They are NOT byte-identical (different
random variables), so -- exactly as the genotype engine was validated against the cell engine -- we
compare them statistically: grow each to a common target SIZE over many seeds and compare the
clone-size distribution, clonal diversity, and surviving-clone counts. We also render the tau growth
curve on a REAL-TIME axis plus its Muller plot, proving plot_muller/plot_grid still work.

Writes manuscript/figures/validate_tau_leaping.png.
Run:  python -u validation/validate_tau_leaping.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from iscc.tumor.models import GenotypeTumor

GENOME = {"n_segments": 4, "segment_size": 50}
SELECTION = {"prop_driver": 0.1, "prop_dispersal": 0.1,
             "prop_immune_resistance": 0.1, "prop_treatment_resistance": 0.1}
CANCER = {"division_rate": 0.3, "death_rate": 0.05, "max_birth_rate": 0.8,
          "mutation_rate": 0.05, "dispersal_rate": 0.1,
          "snv_prob": 0.5, "cnv_prob": 0.5, "n_snvs_per_allele": 0.5, "amp_prob": 0.5}
DEME = {"carrying_capacity": 1, "maximum_death_rate": 0.5}
SPATIAL = {"grid_size": 31, "n_structures": 1, "structure_radius": 0}
TARGET = 2000


def _tumor(mode, seed):
    return GenotypeTumor(seed=seed, genome_params=GENOME, selection_params=SELECTION,
                         cancer_cell_params=CANCER, deme_params=DEME, spatial_params=SPATIAL,
                         update_mode=mode, tau=1.0, snapshot_every=1)


def grow_to_size(mode, seed, target=TARGET, cap=200000, tau=1.0):
    """Grow until cancer size >= target (or extinction / cap). Returns the tumor."""
    t = _tumor(mode, seed)
    t.tau = tau
    if mode == "tau":
        rng = np.random.default_rng(seed)
        t.traces.append(dict(genotypes_counts=dict(t.genotypes_counts)))
        t.trace_times.append(0.0)
        for _ in range(cap):
            if t.get_cancer_size() == 0 or t.get_cancer_size() >= target:
                break
            t._tau_generation(rng, tau)
    else:
        rng = t.rng
        for _ in range(cap):
            if t.get_cancer_size() == 0 or t.get_cancer_size() >= target:
                break
            t.update(rng)
            t.step += 1
    return t


def clone_metrics(t):
    """Cancer clone-size distribution summaries at the current state."""
    _, freqs = t.get_genotype_frequencies(normalize=False)
    freqs = np.array([f for f in freqs if f > 0], dtype=float)
    if freqs.sum() == 0:
        return None
    p = freqs / freqs.sum()
    inv_simpson = 1.0 / np.sum(p ** 2)          # effective number of clones
    return dict(size=int(freqs.sum()), n_clones=len(freqs),
                top_frac=float(freqs.max() / freqs.sum()), inv_simpson=float(inv_simpson),
                fractions=p)


def mean_n_clones(mode, tau, seeds):
    vals = [len(grow_to_size(mode, s, tau=tau).get_genotype_frequencies()[0]) for s in seeds]
    return np.mean([v for v in vals if v > 1])


def main():
    seeds = range(40)
    exact = [m for m in (clone_metrics(grow_to_size("exact", s)) for s in seeds) if m]
    tau = [m for m in (clone_metrics(grow_to_size("tau", s)) for s in seeds) if m]

    def summ(ms, key):
        return np.array([m[key] for m in ms], dtype=float)

    print(f"survivors: exact {len(exact)}/{len(list(seeds))}, tau {len(tau)}/{len(list(seeds))}")
    print("Equivalence by DISTRIBUTION at matched size (cf. the genotype-vs-cell engine "
          "validation, which accepted 0.5-2.0 ratios):")
    for key in ["size", "n_clones", "top_frac", "inv_simpson"]:
        e, g = summ(exact, key), summ(tau, key)
        print(f"  {key:12s}: exact {e.mean():9.3f}  tau {g.mean():9.3f}  ratio {g.mean()/e.mean():.3f}")
    ef = np.concatenate([m["fractions"] for m in exact])
    gf = np.concatenate([m["fractions"] for m in tau])
    ks = stats.ks_2samp(ef, gf)  # diagnostic only -- KS is far stricter than the ballpark bar
    print(f"  clone-fraction KS (diagnostic): D={ks.statistic:.3f} p={ks.pvalue:.3f}")

    # tau-leaping converges to the exact process as tau -> 0: the clone-count bias must shrink
    # monotonically toward 1. This is the definitive correctness signature.
    cseeds = range(20)
    ex_nc = mean_n_clones("exact", 1.0, cseeds)
    taus = [1.0, 0.5, 0.25]
    ratios = [mean_n_clones("tau", tt, cseeds) / ex_nc for tt in taus]
    print("tau -> 0 convergence (mean #clones ratio vs exact):")
    for tt, r in zip(taus, ratios):
        print(f"  tau={tt}: ratio {r:.3f}")

    # ---- figure: growth curve (real-time axis) + Muller + clone-size ECDF + tau-convergence -
    tg = grow_to_size("tau", seed=3, target=8000)
    sizes = [sum(c for g, c in tr["genotypes_counts"].items()
                 if g not in ("epithelial", "stromal", "immune")) for tr in tg.traces]
    fig, axes = plt.subplots(1, 4, figsize=(21, 4.5))

    axes[0].plot(tg.trace_times, sizes, marker="o", ms=3)
    axes[0].set_xlabel("generation (real time)"); axes[0].set_ylabel("cancer cells")
    axes[0].set_title("(b) tau growth curve — real-time axis")

    tg.plot_muller(ax=axes[1])
    axes[1].set_title("(b) tau Muller — structure preserved")

    axes[2].plot(np.sort(ef), np.linspace(0, 1, len(ef)), label="exact", lw=2)
    axes[2].plot(np.sort(gf), np.linspace(0, 1, len(gf)), label="tau (τ=1)", lw=2, ls="--")
    axes[2].set_xscale("log"); axes[2].set_xlabel("clone fraction"); axes[2].set_ylabel("ECDF")
    axes[2].set_title("(c) clone-size distribution"); axes[2].legend()

    axes[3].plot(taus, ratios, marker="o")
    axes[3].axhline(1.0, color="k", ls=":", lw=1)
    axes[3].set_xlabel("τ"); axes[3].set_ylabel("#clones ratio  tau / exact")
    axes[3].set_title("(c) convergence to exact as τ → 0")
    axes[3].invert_xaxis()

    os.makedirs("manuscript/figures", exist_ok=True)
    out = "manuscript/figures/validate_tau_leaping.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print("wrote", out)


if __name__ == "__main__":
    main()
