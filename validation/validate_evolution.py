"""Validate iscc's evolutionary dynamics in Noble et al. (2022) index space.

Noble et al. (2022) characterise a tumour's *mode of evolution* by indices read off its clone
phylogeny. Here we sweep the cell-dispersal rate and trace the model's trajectory through all
three of those indices over replicates (see DESIGN_inference.md):

  * mean drivers per cell  n   = count-weighted mean number of mutated drivers per cancer cell,
  * clonal diversity       D   = inverse-Simpson over driver-mutation combinations,
  * tree balance           J1  = Lemant et al. (2022) clone-phylogeny balance index in [0, 1].

Low dispersal -> a finely intermixed mosaic of local lineages: many coexisting driver
combinations (high D). High dispersal -> fewer combinations expand into spatially coherent
patches (clonal sweeps), lowering D. The J1 tree-balance index complements this, distinguishing
the branchy phylogenies of well-mixed growth from the imbalanced, sweep-dominated ones. This
recasts the earlier Shannon-diversity / spatial-assortment check onto Noble's (n, D, J1) indices
while keeping the single (glandular) structure.

Produces manuscript/figures/validation_evolution.png.
Usage:  python validation/validate_evolution.py
"""
import os
import tempfile

import numpy as np
import yaml

from iscc.tumor.models import GenotypeTumor
from iscc.inference import mode_indices

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DISPERSALS = [0.05, 0.15, 0.4, 1.0]
SEEDS = range(5)
STEPS = 600


def main():
    base = yaml.safe_load(open(os.path.join(REPO, "notebooks/example_config.yaml")))
    base["spatial_params"].update(structure_radius=0, grid_size=30)
    base["genome_params"].update(n_segments=10, segment_size=100)
    base["cell_params"]["cancer"]["death_rate"] = 0.02

    n_mean, n_sd, d_mean, d_sd, j_mean, j_sd = [], [], [], [], [], []
    print(f"{'dispersal':>9} | {'n (drivers/cell)':>18} | {'D (inv-Simpson)':>18} | "
          f"{'J1 (tree balance)':>18} | n_clones")
    for disp in DISPERSALS:
        cfg = yaml.safe_load(yaml.safe_dump(base))
        cfg["cell_params"]["cancer"]["dispersal_rate"] = disp
        tmp = os.path.join(tempfile.mkdtemp(), "c.yaml")
        yaml.safe_dump(cfg, open(tmp, "w"))
        ns, ds, js, ncl = [], [], [], []
        for s in SEEDS:
            t = GenotypeTumor(config=tmp, seed=s)
            t.grow(n_steps=STEPS, seed=s)
            m = mode_indices(t)
            if m["n_clones"]:
                ns.append(m["n"]); ds.append(m["D"]); ncl.append(m["n_clones"])
                if np.isfinite(m["J1"]):        # nan when a replicate is a single clone
                    js.append(m["J1"])
        n_mean.append(np.mean(ns)); n_sd.append(np.std(ns))
        d_mean.append(np.mean(ds)); d_sd.append(np.std(ds))
        j_mean.append(np.mean(js) if js else np.nan); j_sd.append(np.std(js) if js else 0.0)
        print(f"{disp:9.2f} | {np.mean(ns):7.2f} +/- {np.std(ns):5.2f} | "
              f"{np.mean(ds):7.2f} +/- {np.std(ds):5.2f} | "
              f"{j_mean[-1]:7.2f} +/- {j_sd[-1]:5.2f} | {np.mean(ncl):.0f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = DISPERSALS
    fig, axes = plt.subplots(1, 4, figsize=(21, 4))
    axes[0].errorbar(x, n_mean, yerr=n_sd, marker="o", capsize=3)
    axes[0].set_xlabel("cell dispersal rate"); axes[0].set_ylabel("mean drivers per cell  n")
    axes[0].set_title("Driver load vs dispersal"); axes[0].set_xscale("log")
    axes[1].errorbar(x, d_mean, yerr=d_sd, marker="o", color="tab:red", capsize=3)
    axes[1].set_xlabel("cell dispersal rate"); axes[1].set_ylabel("clonal diversity  D")
    axes[1].set_title("Clonal diversity vs dispersal"); axes[1].set_xscale("log")
    axes[2].errorbar(x, j_mean, yerr=j_sd, marker="o", color="tab:green", capsize=3)
    axes[2].set_xlabel("cell dispersal rate"); axes[2].set_ylabel("tree balance  J1")
    axes[2].set_title("Tree balance vs dispersal"); axes[2].set_xscale("log")
    axes[2].set_ylim(0, 1)
    # trajectory through (n, D) index space, coloured by J1
    sc = axes[3].scatter(n_mean, d_mean, c=j_mean, cmap="viridis", s=80, zorder=3,
                         vmin=0, vmax=1)
    axes[3].plot(n_mean, d_mean, color="0.6", lw=1, zorder=2)
    axes[3].set_xlabel("mean drivers per cell  n"); axes[3].set_ylabel("clonal diversity  D")
    axes[3].set_title("Trajectory in Noble (n, D) space")
    fig.colorbar(sc, ax=axes[3], label="tree balance  J1")
    fig.suptitle("Cell dispersal governs the mode of tumor evolution (iscc, Noble n/D/J1 indices)")
    fig.tight_layout()
    out = os.path.join(REPO, "manuscript/figures/validation_evolution.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"figure -> {out}")


if __name__ == "__main__":
    main()
