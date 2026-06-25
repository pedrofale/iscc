"""Validate iscc's SNV site-frequency spectrum against neutral-growth theory.

Under neutral tumour growth the bulk variant-allele-frequency (VAF) distribution is
dominated by rare variants and its cumulative form follows the 1/f power law
    M(f) = (mu/beta) (1/f - 1/f_max)
(Williams et al. 2016, Nat Genet) -- i.e. the cumulative number of mutations is linear
in 1/f. We grow neutral tumours (all driver/dispersal/resistance effects = 1, so every
mutation is a passenger), read out the population VAF a bulk WGS assay would see, and
measure the fit of the cumulative spectrum to the 1/f law over the subclonal range.

iscc is spatially explicit, so we also expect (and report) the documented deviation of
spatially constrained tumours away from the well-mixed 1/f expectation -- an excess of
low-frequency and depletion of intermediate-frequency variants (Sun et al. 2017;
Chkhaidze et al. 2019).

Produces manuscript/figures/validation_snv.png.
Usage:  python validation/validate_snv.py
"""
import os
import tempfile

import numpy as np
import yaml

from iscc.tumor.models import GenotypeTumor
from iscc.validation import population_vaf, site_frequency_spectrum, neutral_sfs_rsq

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F_MIN, F_MAX, N_BINS = 0.05, 0.45, 40
SEEDS = range(5)
STEPS = 3000


def _neutral_config():
    """example_config with selection switched off (pure passengers) and a larger arena."""
    base = yaml.safe_load(open(os.path.join(REPO, "notebooks/example_config.yaml")))
    base["spatial_params"].update(structure_radius=0, grid_size=40)
    base["genome_params"].update(n_segments=10, segment_size=200)
    base["deme_params"]["carrying_capacity"] = 10
    base["cell_params"]["cancer"].update(
        division_rate=0.5, death_rate=0.02, mutation_rate=1.0, dispersal_rate=0.2)
    base["selection_params"].update(
        driver_effects=1.0, dispersal_effects=1.0,
        treatment_resistant_effects=1.0, immune_resistant_effects=1.0,
    )
    return base


def main():
    cfg = _neutral_config()
    tmp = os.path.join(tempfile.mkdtemp(), "neutral.yaml")
    yaml.safe_dump(cfg, open(tmp, "w"))

    vafs, rsqs = [], []
    print(f"{'seed':>4} | {'cancer cells':>12} | {'mut sites':>9} | {'R^2 (1/f)':>9}")
    for s in SEEDS:
        t = GenotypeTumor(config=tmp, seed=s)
        t.grow(n_steps=STEPS, seed=s)
        vaf = population_vaf(t)
        r, _ = neutral_sfs_rsq(vaf, F_MIN, F_MAX, N_BINS)
        vafs.append(vaf[vaf > 0])
        rsqs.append(r)
        print(f"{s:>4} | {t.get_cancer_size():>12} | {int((vaf > 0).sum()):>9} | {r:>9.3f}")
    print(f"\nmean R^2 to neutral 1/f law: {np.mean(rsqs):.3f} +/- {np.std(rsqs):.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # (1) VAF distribution -- rare-variant dominated
    allv = np.concatenate(vafs)
    axes[0].hist(allv, bins=50, range=(0, 1), color="tab:blue", alpha=0.8)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("variant allele frequency (f)")
    axes[0].set_ylabel("number of mutations (log)")
    axes[0].set_title("SNV frequency distribution")

    # (2) cumulative SFS vs 1/f with the neutral linear fit
    rep = vafs[0]
    grid, M = site_frequency_spectrum(rep, F_MIN, F_MAX, N_BINS)
    x = 1.0 / grid
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, M, rcond=None)
    axes[1].plot(x, M, "o", ms=4, label="iscc (simulated)")
    axes[1].plot(x, A @ coef, "-", color="tab:red",
                 label=f"neutral 1/f fit (R$^2$={rsqs[0]:.2f})")
    axes[1].set_xlabel("1 / f")
    axes[1].set_ylabel("cumulative # mutations  M(f)")
    axes[1].set_title("Cumulative SFS vs neutral 1/f law")
    axes[1].legend(fontsize=8)

    fig.suptitle("iscc reproduces the neutral SNV site-frequency spectrum")
    fig.tight_layout()
    out = os.path.join(REPO, "manuscript/figures/validation_snv.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"figure -> {out}")


if __name__ == "__main__":
    main()
