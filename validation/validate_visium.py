"""Posterior-predictive validation of the Visium ``estimate_visium()`` (DESIGN_inference §C.2, M4).

Same loop as the scRNA (§B) / DNA (§C.1) validations, for the spatial assay: take a Visium dataset,
**estimate** its technical hyper-parameters (``estimate_visium``, method-of-moments + an SE
autocorrelation fit), **re-simulate** a Visium section with the fitted hypers, then recompute the
**same** spatial + count summary statistics and overlay re-simulated vs observed. A tight overlay
means the fitted technical layer reproduces the data's spatial-autocorrelation / depth structure.

The recomputed summaries (the §C.2 validation set):
  * Moran's I of the per-spot log-library          (the spatially-correlated capture field)
  * the spatial autocorrelation correlogram r(d)    (the field_lengthscale)
  * the per-spot total-count distribution           (mu_counts / sigma_counts)
  * spots-per-tissue                                (the spatial extent)

Per the deliverable a synthetic fit->simulate->overlay is acceptable: the "observed" data is itself
an iscc Visium simulation with chosen ground-truth hypers (the stand-in for a real Visium section),
so the figure shows the full round-trip: ground-truth -> observed -> estimate -> re-simulate -> match.

Usage:  python validation/validate_visium.py
Produces manuscript/figures/validation_visium.png.
"""
import argparse
import os

import numpy as np
import pandas as pd

from iscc.data import Visium, morans_i, estimate_visium_from_assay

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Ground-truth technical hypers for the "observed" data (the stand-in for a real Visium section).
TRUE = dict(mu_counts=6000.0, sigma_counts=0.30, field_sigma=0.45, field_lengthscale=7.0,
            edge_sigma=0.25, kappa=80.0)
N_GENES = 50


def make_tissue(grid=34, seed=0):
    """A dense synthetic section: 1-3 cells per integer coordinate, two spatially-segregated clones
    with distinct expression profiles (so the biology is spatially structured, as in real tissue)."""
    rng = np.random.default_rng(seed)
    genes = [f"G_{g}" for g in range(N_GENES)]
    base_a = rng.gamma(2.0, 1.0, N_GENES)
    base_b = rng.gamma(2.0, 1.0, N_GENES)
    ids, coords, exp, ctype = [], [], [], []
    i = 0
    cx = cy = grid / 2.0
    for r in range(grid):
        for c in range(grid):
            clone = "A" if (r - cx) ** 2 + (c - cy) ** 2 < (grid * 0.3) ** 2 else "B"
            base = base_a if clone == "A" else base_b
            for _ in range(int(rng.integers(1, 4))):
                exp.append(base * rng.lognormal(0.0, 0.2, N_GENES))
                coords.append((r, c)); ctype.append(clone); ids.append(f"C{i}"); i += 1
    exp = pd.DataFrame(exp, index=ids, columns=genes)
    return {
        "cell_exp": exp,
        "cell_crd": pd.DataFrame(coords, index=ids, columns=["row", "col"]),
        "cell_type": pd.DataFrame(ctype, index=ids, columns=["cell_id"]),
    }, grid


# --------------------------------------------------------------------------------------
# Summary statistics (recomputed identically on observed + re-simulated)
# --------------------------------------------------------------------------------------
def _log_library_residual(assay):
    """Per-spot log-library residual over on-tissue spots, with its coordinates."""
    occ = assay.obs.n_cells.values > 0
    tot = assay.spot_counts.values.sum(axis=1)[occ]
    coords = assay.spot_coords[occ]
    z = np.log(tot) - np.log(tot).mean()
    return z, coords, tot


def _correlogram(z, coords, nbins=12, max_frac=0.6):
    """Empirical spatial autocorrelation r(d) = <z_i z_j>/Var[z] in distance bins."""
    n = len(z)
    iu = np.triu_indices(n, k=1)
    d = coords[iu[0]] - coords[iu[1]]
    dist = np.sqrt((d * d).sum(axis=1))
    var = float(np.mean(z * z))
    zz = z[iu[0]] * z[iu[1]] / max(var, 1e-12)
    edges = np.linspace(dist.min(), np.percentile(dist, 100 * max_frac), nbins + 1)
    centers, corr = [], []
    for i in range(nbins):
        m = (dist >= edges[i]) & (dist < edges[i + 1])
        if m.sum() >= 20:
            centers.append(0.5 * (edges[i] + edges[i + 1]))
            corr.append(float(zz[m].mean()))
    return np.asarray(centers), np.asarray(corr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(REPO, "manuscript/figures/validation_visium.png"))
    args = ap.parse_args()

    cell_data, grid = make_tissue(seed=args.seed)

    # observed -> estimate -> re-simulate (the round-trip)
    obs = Visium(seed=10, spot_pitch=2.0, spot_radius=1.0, count_model="dm", **TRUE).run(
        cell_data, grid_side=grid)
    est = estimate_visium_from_assay(obs)
    # visium_kwargs() already carries spot_pitch / spot_radius (preset, not fit)
    sim = Visium(seed=11, count_model="dm", **est.visium_kwargs()).run(cell_data, grid_side=grid)
    print(f"Visium fit: {est}")

    zo, co, toto = _log_library_residual(obs)
    zs, cs, tots = _log_library_residual(sim)
    mi_o, mi_s = morans_i(toto, co), morans_i(tots, cs)
    cx_o, cy_o = _correlogram(zo, co)
    cx_s, cy_s = _correlogram(zs, cs)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    ax = axes[0, 0]
    rng = (min(toto.min(), tots.min()), max(toto.max(), tots.max()))
    bins = np.linspace(rng[0], rng[1], 31)
    ax.hist(toto, bins=bins, density=True, alpha=0.5, label="observed", color="tab:gray")
    ax.hist(tots, bins=bins, density=True, alpha=0.5, label="re-simulated (fitted)", color="tab:red")
    ax.set_xlabel("per-spot total counts"); ax.set_ylabel("density")
    ax.set_title(f"Spot-count distribution\n(mu_counts true={TRUE['mu_counts']:.0f}, "
                 f"fit={est.hypers.mu_counts:.0f})")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(cx_o, cy_o, "o-", label="observed", color="tab:gray")
    ax.plot(cx_s, cy_s, "s--", label="re-simulated (fitted)", color="tab:red")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("spot-spot distance"); ax.set_ylabel("autocorrelation r(d)")
    ax.set_title(f"Capture-field spatial autocorrelation\n(field_lengthscale true="
                 f"{TRUE['field_lengthscale']:.1f}, fit={est.hypers.field_lengthscale:.1f})")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    sc = ax.scatter(obs.spot_coords[:, 1], obs.spot_coords[:, 0], c=obs.capture_field,
                    cmap="viridis", s=18)
    ax.set_title(f"Observed capture-efficiency field\n(Moran's I obs={mi_o:.2f}, "
                 f"fit-sim={mi_s:.2f})")
    ax.set_xlabel("col"); ax.set_ylabel("row"); ax.invert_yaxis()
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 1]
    ax.axis("off")
    summary = (
        f"spots-per-tissue:\n"
        f"   observed = {int((obs.obs.n_cells.values>0).sum())}\n"
        f"   re-sim   = {int((sim.obs.n_cells.values>0).sum())}\n\n"
        f"Moran's I (per-spot library):\n"
        f"   observed = {mi_o:.3f}\n   re-sim   = {mi_s:.3f}\n\n"
        f"mu_counts   true={TRUE['mu_counts']:.0f}  fit={est.hypers.mu_counts:.0f}\n"
        f"sigma_counts true={TRUE['sigma_counts']:.2f}  fit={est.hypers.sigma_counts:.2f}\n"
        f"field_sigma  true={TRUE['field_sigma']:.2f}  fit={est.hypers.field_sigma:.2f}\n"
        f"field_length true={TRUE['field_lengthscale']:.1f}  fit={est.hypers.field_lengthscale:.1f}\n"
        f"kappa        true={TRUE['kappa']:.0f}  fit={est.hypers.kappa:.0f}\n\n"
        f"fitted: {est.fitted}"
    )
    ax.text(0.0, 0.5, summary, fontsize=9, family="monospace", va="center")

    fig.suptitle("Visium estimate_visium(): fitted technical params reproduce observed summaries "
                 "(fit -> re-simulate -> overlay)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=120, bbox_inches="tight")

    print(f"Moran's I  observed={mi_o:.3f}  fit-sim={mi_s:.3f}")
    print(f"spots-per-tissue observed={int((obs.obs.n_cells.values>0).sum())}")
    print(f"figure -> {args.out}")


if __name__ == "__main__":
    main()
