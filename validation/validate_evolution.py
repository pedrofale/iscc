"""Validate iscc's evolutionary dynamics: cell dispersal governs the mode of evolution.

Recapitulates the central result of Noble et al. (2022) — that a tumor's spatial structure
(here, the range of cell dispersal) governs its mode of evolution — by sweeping the dispersal
rate and measuring two mode indices over replicates:

  * clonal diversity (Shannon entropy of clone frequencies), and
  * spatial clonal assortment (do neighbouring regions share the dominant clone?).

Low dispersal -> a finely intermixed, highly diverse mosaic of local lineages;
high dispersal -> fewer clones expanding into spatially coherent patches (clonal sweeps).

Produces manuscript/figures/validation_evolution.png.
Usage:  python validation/validate_evolution.py
"""
import os
import tempfile

import numpy as np
import yaml

from iscc.tumor.models import GenotypeTumor
from iscc.validation import clone_diversity, spatial_assortment

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DISPERSALS = [0.05, 0.15, 0.4, 1.0]
SEEDS = range(5)
STEPS = 600


def _mode_indices(t):
    cancer = {g: c for g, c in t.genotypes_counts.items() if t._is_cancer(g)}
    if not cancer:
        return None
    # dominant cancer clone per occupied deme
    dom = {}
    for i, d in enumerate(t.demes):
        cg = {g: c for g, c in d.items() if t._is_cancer(g)}
        if cg:
            dom[i] = max(cg, key=cg.get)
    return dict(
        diversity=clone_diversity(cancer.values()),
        assortment=spatial_assortment(dom, t.grid_size),
        n_clones=len(cancer),
    )


def main():
    base = yaml.safe_load(open(os.path.join(REPO, "notebooks/example_config.yaml")))
    base["spatial_params"].update(structure_radius=0, grid_size=30)
    base["genome_params"].update(n_segments=10, segment_size=100)
    base["cell_params"]["cancer"]["death_rate"] = 0.02

    div_mean, div_sd, ass_mean, ass_sd = [], [], [], []
    print(f"{'dispersal':>9} | {'diversity':>18} | {'assortment':>18} | n_clones")
    for disp in DISPERSALS:
        cfg = yaml.safe_load(yaml.safe_dump(base))
        cfg["cell_params"]["cancer"]["dispersal_rate"] = disp
        tmp = os.path.join(tempfile.mkdtemp(), "c.yaml")
        yaml.safe_dump(cfg, open(tmp, "w"))
        divs, asss, ncl = [], [], []
        for s in SEEDS:
            t = GenotypeTumor(config=tmp, seed=s)
            t.grow(n_steps=STEPS, seed=s)
            m = _mode_indices(t)
            if m:
                divs.append(m["diversity"]); asss.append(m["assortment"]); ncl.append(m["n_clones"])
        div_mean.append(np.mean(divs)); div_sd.append(np.std(divs))
        ass_mean.append(np.nanmean(asss)); ass_sd.append(np.nanstd(asss))
        print(f"{disp:9.2f} | {np.mean(divs):7.2f} +/- {np.std(divs):5.2f} | "
              f"{np.nanmean(asss):7.2f} +/- {np.nanstd(asss):5.2f} | {np.mean(ncl):.0f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = DISPERSALS
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].errorbar(x, div_mean, yerr=div_sd, marker="o", capsize=3)
    axes[0].set_xlabel("cell dispersal rate"); axes[0].set_ylabel("clonal diversity (Shannon)")
    axes[0].set_title("Clonal diversity vs dispersal")
    axes[1].errorbar(x, ass_mean, yerr=ass_sd, marker="o", color="tab:red", capsize=3)
    axes[1].set_xlabel("cell dispersal rate"); axes[1].set_ylabel("spatial clonal assortment")
    axes[1].set_title("Spatial clone structure vs dispersal")
    for ax in axes:
        ax.set_xscale("log")
    fig.suptitle("Cell dispersal governs the mode of tumor evolution (iscc)")
    fig.tight_layout()
    out = os.path.join(REPO, "manuscript/figures/validation_evolution.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"figure -> {out}")


if __name__ == "__main__":
    main()
