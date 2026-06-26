"""M3a: overlay real tumours on iscc's evolutionary-mode (n, D, J1) cloud.

Fit-to-real for the evolution module (DESIGN_inference.md §D.2). We sweep iscc over a grid of
driver-fitness and dispersal values, read off the Noble (n, D, J1) indices for each simulated
tumour (a cloud of points), and overlay the **real** tumours whose indices were computed — with
the *same* `indices.py` definitions — from the published Noble et al. (2022) phylogenies
(`validation/data/noble_empirical_indices.csv`, built by `build_noble_empirical_indices.py`).

Keeping a single (glandular) spatial structure, this is a *coverage / best-fit* result — does
iscc reproduce the region of index space where real solid tumours lie, and at what driver
fitness (cf. Noble's best-fit ~0.2)? — not Noble's multi-structure discrimination claim.

Produces manuscript/figures/validation_evolution_modes.png.
Usage:  python validation/validate_evolution_modes.py
"""
import os

import numpy as np
import pandas as pd

from iscc.tumor.models import GenotypeTumor
from iscc.inference import mode_indices

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMPIRICAL = os.path.join(REPO, "validation/data/noble_empirical_indices.csv")

# Driver-loci density is the axis that moves a tumour through Noble's (n, D) space: scarce driver
# loci (like real recurrent driver genes) -> few combinations -> low D; dense loci -> high D.
# Sweeping it (× dispersal × seeds) traces iscc's trajectory through the index space.
PROP_DRIVERS = [0.02, 0.05, 0.10, 0.15]   # driver loci = prop * (n_segments*segment_size)
DISPERSALS = [0.15, 0.5]
DRIVER_EFFECTS = 1.3
SEEDS = range(8)
STEPS = 1500

GENOME = {"n_segments": 12, "segment_size": 100}
DEME = {"carrying_capacity": 6}
SPATIAL = {"grid_size": 30, "structure_radius": 0}
CANCER = {"division_rate": 0.4, "death_rate": 0.02, "max_birth_rate": 0.98,
          "mutation_rate": 1.0, "dispersal_rate": 0.2}


def _selection(prop_driver):
    return {"prop_driver": prop_driver, "prop_dispersal": 0.0, "prop_immune_resistance": 0.0,
            "prop_treatment_resistance": 0.0, "driver_effects": DRIVER_EFFECTS,
            "dispersal_effects": 1.0, "treatment_resistant_effects": 1.0,
            "immune_resistant_effects": 1.0}


def _simulate():
    rows = []
    for pd_ in PROP_DRIVERS:
        for disp in DISPERSALS:
            cancer = {**CANCER, "dispersal_rate": disp}
            for s in SEEDS:
                t = GenotypeTumor(seed=s, genome_params=GENOME, selection_params=_selection(pd_),
                                  cancer_cell_params=cancer, deme_params=DEME, spatial_params=SPATIAL)
                t.grow(STEPS, seed=s)
                m = mode_indices(t)
                if m["n_clones"] and np.isfinite(m["n"]) and np.isfinite(m["D"]):
                    rows.append(dict(prop_driver=pd_, dispersal=disp, **m))
    return pd.DataFrame(rows)


def _hull_coverage(sim_nd, emp_nd):
    """Fraction of empirical (n, D) points inside the convex hull of the simulated cloud."""
    try:
        from scipy.spatial import Delaunay
        hull = Delaunay(sim_nd)
        return float((hull.find_simplex(emp_nd) >= 0).mean())
    except Exception:
        return float("nan")


def main():
    emp = pd.read_csv(EMPIRICAL)
    sim = _simulate()
    print(f"simulated {len(sim)} tumours; empirical {len(emp)} tumours")

    sim_nd = sim[["n", "D"]].to_numpy()
    emp_nd = emp[["n", "D"]].to_numpy()
    coverage = _hull_coverage(sim_nd, emp_nd)

    # best-fit driver-loci density: which prop_driver subset sits closest to the empirical points
    print(f"{'prop_driver':>12} | {'median nn-dist to empirical':>26}")
    best_pd, best_d = None, np.inf
    for pd_ in PROP_DRIVERS:
        sub = sim[sim.prop_driver == pd_][["n", "D"]].to_numpy()
        dists = np.sqrt(((emp_nd[:, None, :] - sub[None, :, :]) ** 2).sum(-1)).min(1)
        med = float(np.median(dists))
        print(f"{pd_:>12.2f} | {med:>26.2f}")
        if med < best_d:
            best_d, best_pd = med, pd_

    # coverage by cancer type (which real tumours does iscc's region reach)
    print(f"\n{'cancer type':>18} | in-hull")
    try:
        from scipy.spatial import Delaunay
        hull = Delaunay(sim_nd)
        emp = emp.assign(in_hull=hull.find_simplex(emp_nd) >= 0)
        for ct, g in emp.groupby("cancer_type"):
            print(f"{ct:>18} | {g.in_hull.mean():.0%} ({g.in_hull.sum()}/{len(g)})")
    except Exception:
        pass
    print(f"\nempirical tumours inside iscc (n, D) hull: {coverage:.0%}")
    print(f"best-fit driver-loci density (closest to real): prop_driver={best_pd}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    types = emp.cancer_type.unique()
    cmap = plt.get_cmap("tab10")
    for ax, (xcol, ycol, xlab, ylab) in zip(
        axes, [("n", "D", "mean drivers per cell  n", "clonal diversity  D"),
               ("n", "J1", "mean drivers per cell  n", "tree balance  J1")]):
        ax.scatter(sim[xcol], sim[ycol], s=18, c="0.7", alpha=0.5, label="iscc (simulated)", zorder=1)
        for i, ct in enumerate(types):
            e = emp[emp.cancer_type == ct]
            ax.scatter(e[xcol], e[ycol], s=55, color=cmap(i % 10), edgecolor="k",
                       linewidth=0.4, label=ct, zorder=3)
        ax.set_xlabel(xlab); ax.set_ylabel(ylab)
    axes[0].set_title(f"Noble (n, D) space — {coverage:.0%} of real tumours in iscc hull")
    axes[1].set_title("Tree balance (n, J1)"); axes[1].set_ylim(0, 1)
    axes[1].legend(fontsize=7, loc="upper right", framealpha=0.9)
    fig.suptitle("iscc vs real tumours in Noble (n, D, J1) index space "
                 "(single glandular structure; abstract genome)")
    fig.tight_layout()
    out = os.path.join(REPO, "manuscript/figures/validation_evolution_modes.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"figure -> {out}")


if __name__ == "__main__":
    main()
