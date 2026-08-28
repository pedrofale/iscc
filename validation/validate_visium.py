"""Posterior-predictive validation of the Visium ``estimate_visium()`` (DESIGN_inference §C.2, M4).

Same loop as the scRNA (§B) / DNA (§C.1) validations, for the spatial assay: take a Visium dataset,
**estimate** its technical hyper-parameters (``estimate_visium``, method-of-moments + an SE
autocorrelation fit), **re-simulate** a Visium section with the fitted hypers, then recompute the
**same** spatial + count summary statistics and overlay re-simulated vs observed. A tight overlay
means the fitted technical layer reproduces the data's spatial-autocorrelation / depth structure.

The recomputed summaries (the §C.2 validation set):
  * Moran's I of the per-spot library              (the spatially-correlated capture field)
  * the per-spot total-count distribution           (mu_counts / sigma_counts)
  * spots-per-tissue                                (the spatial extent)

DEFAULTS TO REAL DATA — the spatial analogue of scRNA fitting PBMC3k. It downloads a real 10x Visium
section via ``scanpy.datasets.visium_sge`` (default `V1_Breast_Cancer_Block_A_Section_1`), fits
``estimate_visium`` on it (so we have realistic technical magnitudes), then re-simulates the
technical layer on a synthetic tissue and overlays the summaries. The biology differs (synthetic
tissue), so the TECHNICAL summaries are what's validated — depth distribution + spatial
autocorrelation magnitude. Real coords (full-res pixels) are normalized to SPOT-PITCH units so the
fitted ``field_lengthscale`` is dimensionless (~N spots) and transfers to the grid. ``--synthetic``
runs the offline ground-truth round-trip instead (also the automatic fallback if the download fails).

Usage:  python validation/validate_visium.py [--sample-id ID] [--synthetic]
Produces the paper repo's figures/validation_visium.png.
"""
import argparse
import os

import numpy as np
import pandas as pd

from iscc.data import Visium, morans_i, estimate_visium, estimate_visium_from_assay
from _paths import figure_path

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _to_dense(adata):
    X = adata.X if hasattr(adata, "X") else adata
    return np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)

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


def load_real_visium(sample_id="V1_Breast_Cancer_Block_A_Section_1"):
    """Download a real 10x Visium section via scanpy — the spatial analogue of PBMC3k.

    `estimate_visium` learns the technical magnitudes (per-spot library, the spatially-correlated
    capture field, count overdispersion) from this, so simulated Visium has realistic parameters
    instead of guessed ones. Returns an AnnData (raw spots x genes counts + `obsm['spatial']`), or
    `None` if scanpy / the download is unavailable (the validation then falls back to the synthetic
    round-trip, so it still runs offline / in CI)."""
    try:
        import scanpy as sc
        ad = sc.datasets.visium_sge(sample_id=sample_id)
        ad.var_names_make_unique()
        return ad
    except Exception as e:  # offline, scanpy missing, or 10x host down
        print(f"[real Visium '{sample_id}' unavailable: {e}]\n -> falling back to synthetic round-trip")
        return None


def _median_nn(coords):
    """Median nearest-neighbour spot distance = the spot pitch, used to normalize real Visium
    coordinates (full-res pixels) to dimensionless SPOT-PITCH units so a fitted absolute
    `field_lengthscale` becomes "~N spots" and transfers to the synthetic grid."""
    from scipy.spatial import cKDTree
    d, _ = cKDTree(np.asarray(coords, dtype=float)).query(np.asarray(coords, dtype=float), k=2)
    return float(np.median(d[:, 1]))


def _lib_residual_raw(counts, coords):
    """Per-spot log-library residual over on-tissue spots, from a raw counts matrix + coords."""
    tot = np.asarray(counts).sum(axis=1)
    occ = tot > 0
    tot = tot[occ].astype(float)
    coords = np.asarray(coords, dtype=float)[occ]
    z = np.log(tot) - np.log(tot).mean()
    return z, coords, tot


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


def _make_figure(toto, co, tots, cs, est, mi_o, mi_s, out, header, true=None):
    """2x2 overlay: spot-count distribution + observed/re-sim per-spot-library heatmaps + summary.

    Works for both the real-data fit (`true=None`) and the synthetic round-trip (`true=TRUE`)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = est.hypers
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    ax = axes[0, 0]
    bins = np.linspace(min(toto.min(), tots.min()), max(toto.max(), tots.max()), 31)
    ax.hist(toto, bins=bins, density=True, alpha=0.5, label="observed", color="tab:gray")
    ax.hist(tots, bins=bins, density=True, alpha=0.5, label="re-simulated (fitted)", color="tab:red")
    ax.set(xlabel="per-spot total counts", ylabel="density", title="Spot-count distribution")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    s = ax.scatter(co[:, 1], co[:, 0], c=toto, cmap="viridis", s=10)
    ax.set(xlabel="col", ylabel="row", title=f"Observed per-spot library\n(Moran's I = {mi_o:.2f})")
    ax.invert_yaxis(); fig.colorbar(s, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 0]
    s = ax.scatter(cs[:, 1], cs[:, 0], c=tots, cmap="viridis", s=10)
    ax.set(xlabel="col", ylabel="row",
           title=f"Re-simulated per-spot library (fitted)\n(Moran's I = {mi_s:.2f})")
    ax.invert_yaxis(); fig.colorbar(s, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 1]; ax.axis("off")
    lines = [f"source: {header}", "",
             f"spots-per-tissue: obs={len(toto)}  re-sim={len(tots)}",
             f"Moran's I:        obs={mi_o:.3f}  re-sim={mi_s:.3f}", ""]
    if true is not None:  # synthetic round-trip: show recovered vs ground truth
        lines += [f"mu_counts    true={true['mu_counts']:.0f}  fit={h.mu_counts:.0f}",
                  f"sigma_counts true={true['sigma_counts']:.2f}  fit={h.sigma_counts:.2f}",
                  f"field_sigma  true={true['field_sigma']:.2f}  fit={h.field_sigma:.2f}",
                  f"field_length true={true['field_lengthscale']:.1f}  fit={h.field_lengthscale:.1f}",
                  f"kappa        true={true['kappa']:.0f}  fit={h.kappa:.0f}"]
    else:  # real fit: just the learned magnitudes
        lines += [f"mu_counts    = {h.mu_counts:.0f}",
                  f"sigma_counts = {h.sigma_counts:.2f}",
                  f"field_sigma  = {h.field_sigma:.2f}",
                  f"field_length = {h.field_lengthscale:.1f}",
                  f"kappa        = {h.kappa:.0f}"]
    lines += ["", f"fitted: {est.fitted}"]
    ax.text(0.0, 0.5, "\n".join(lines), fontsize=9, family="monospace", va="center")

    fig.suptitle("Visium estimate_visium(): fitted technical params reproduce observed summaries "
                 "(fit -> re-simulate -> overlay)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-id", default="V1_Breast_Cancer_Block_A_Section_1",
                    help="10x Visium sample for scanpy.datasets.visium_sge (the real default).")
    ap.add_argument("--synthetic", action="store_true",
                    help="Skip the real download; use the synthetic ground-truth round-trip.")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default=figure_path("validation_visium.png"))
    args = ap.parse_args()

    adata = None if args.synthetic else load_real_visium(args.sample_id)
    cell_data, grid = make_tissue(seed=args.seed)

    if adata is not None:
        # FIT ON REAL Visium -> re-simulate the technical layer on a synthetic tissue -> overlay.
        # (Biology differs; the TECHNICAL summaries — depth distribution + spatial autocorrelation —
        # are what's validated, exactly as scRNA fits PBMC3k then simulates on a tumour.)
        counts = _to_dense(adata)
        # Normalize real coords (full-res pixels) to SPOT-PITCH units so the fitted field_lengthscale
        # is dimensionless (~N spots) and transfers to the synthetic grid; then re-simulate at
        # spot_pitch=1.0 so that spot-unit length-scale applies directly.
        coords = np.asarray(adata.obsm["spatial"], dtype=float) / _median_nn(adata.obsm["spatial"])
        est = estimate_visium(adata, count_model="dm", coords=coords)
        kw = {**est.visium_kwargs(), "spot_pitch": 1.0, "spot_radius": 0.5}
        sim = Visium(seed=11, count_model="dm", **kw).run(cell_data, grid_side=grid)
        zo, co, toto = _lib_residual_raw(counts, coords)
        header, true = f"real 10x Visium '{args.sample_id}'", None
    else:
        # SYNTHETIC round-trip: known ground-truth hypers -> observed -> estimate -> re-simulate.
        obs = Visium(seed=10, spot_pitch=2.0, spot_radius=1.0, count_model="dm", **TRUE).run(
            cell_data, grid_side=grid)
        est = estimate_visium_from_assay(obs)
        sim = Visium(seed=11, count_model="dm", **est.visium_kwargs()).run(cell_data, grid_side=grid)
        zo, co, toto = _log_library_residual(obs)
        header, true = "synthetic ground-truth round-trip", TRUE

    zs, cs, tots = _log_library_residual(sim)
    mi_o, mi_s = morans_i(toto, co), morans_i(tots, cs)
    print(f"Visium fit ({header}): {est}")
    _make_figure(toto, co, tots, cs, est, mi_o, mi_s, out=args.out, header=header, true=true)
    print(f"Moran's I  observed={mi_o:.3f}  fit-sim={mi_s:.3f}")
    print(f"spots-per-tissue observed={len(toto)}  re-sim={len(tots)}")
    print(f"figure -> {args.out}")


if __name__ == "__main__":
    main()
