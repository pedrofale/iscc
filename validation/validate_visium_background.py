"""Bleed and ambient RNA from the spots outside the tissue, on a real Visium section and on iscc.

A Visium slide carries 4,992 spots whatever tissue is placed on it, and Space Ranger's raw matrix
keeps all of them with an ``in_tissue`` flag. A spot outside the tissue holds no cells, so whatever it
counts is mRNA that bled from the tissue or ambient RNA floating over the slide. This script fits both
from those spots (``estimate_visium(..., in_tissue=...)``, which calls ``_fit_off_tissue``), then runs
iscc's Visium assay on a section with the same tissue outline, at the preset and at the fitted values,
and computes the same statistics on its output:

  * the off-tissue profile: median spot total, as a fraction of the mean tissue spot, against the
    distance to the nearest tissue spot (in spot pitches);
  * the fitted ambient floor, bleed fraction and bleed kernel sd;
  * the inward profile: tissue spot totals against the distance into the tissue, relative to the
    interior, beside the part of the edge loss bleed alone predicts;
  * the correlation of the ambient gene profile with the pooled tissue profile (iscc's soup).

The real section is 10x Genomics' V1_Breast_Cancer_Block_A_Section_1 (Space Ranger 1.1.0), the same
section Fig 2g fits library depth on. Pass its directory, holding the raw ``.h5`` and
``spatial/tissue_positions_list.csv``:

    python validation/validate_visium_background.py --data-dir <dir> [--out-json fit.json]

Produces the paper repo's figures/validation_visium_background.png.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from iscc.data import Visium, estimate_visium
from iscc.data.estimate_visium import _fit_off_tissue, _pitch
from _paths import figure_path

SAMPLE = "V1_Breast_Cancer_Block_A_Section_1"
SPOT_PITCH = 2.0                                  # iscc's coordinate units per spot pitch (100 um)


def load_raw(data_dir, sample=SAMPLE):
    """All 4,992 spots of a Space Ranger 1.x raw matrix: sparse counts, the tissue flag, and spot
    coordinates rescaled so one spot pitch is ``SPOT_PITCH`` units (iscc's convention)."""
    import h5py
    from scipy import sparse
    with h5py.File(os.path.join(data_dir, f"{sample}_raw_feature_bc_matrix.h5"), "r") as f:
        m = f["matrix"]
        shape = tuple(m["shape"][()])
        X = sparse.csc_matrix((m["data"][()], m["indices"][()], m["indptr"][()]), shape=shape).T.tocsr()
        barcodes = [b.decode() for b in m["barcodes"][()]]
    pos = pd.read_csv(os.path.join(data_dir, "spatial", "tissue_positions_list.csv"), header=None,
                      names=["barcode", "in_tissue", "array_row", "array_col", "pxl_row", "pxl_col"])
    pos = pos.set_index("barcode").loc[barcodes]
    px = pos[["pxl_row", "pxl_col"]].to_numpy(float)
    coords = px / _pitch(px) * SPOT_PITCH
    return X, pos["in_tissue"].to_numpy().astype(bool), coords


def tissue_like(coords, in_tissue, n_genes=200, cells_per_spot=(3, 7), seed=0):
    """A synthetic section with the real section's tissue outline: cells scattered within one spot
    radius of every tissue spot centre, two spatially segregated profiles. Only the assay is under
    test here, so the biology is kept simple."""
    rng = np.random.default_rng(seed)
    genes = [f"G_{g}" for g in range(n_genes)]
    base = {k: rng.gamma(2.0, 1.0, n_genes) for k in ("A", "B")}
    centre = coords[in_tissue].mean(axis=0)
    radius = np.median(np.linalg.norm(coords[in_tissue] - centre, axis=1))
    ids, crd, exp, ctype = [], [], [], []
    for p in coords[in_tissue]:
        clone = "A" if np.linalg.norm(p - centre) < radius else "B"
        for _ in range(int(rng.integers(*cells_per_spot))):
            off = rng.uniform(-0.5, 0.5, 2) * SPOT_PITCH
            crd.append(p + off)
            exp.append(base[clone] * rng.lognormal(0.0, 0.2, n_genes))
            ctype.append(clone)
            ids.append(f"C{len(ids)}")
    lo = coords.min(axis=0) - SPOT_PITCH
    crd = np.asarray(crd) - lo
    cell_data = {
        "cell_exp": pd.DataFrame(exp, index=ids, columns=genes),
        "cell_crd": pd.DataFrame(crd, index=ids, columns=["row", "col"]),
        "cell_type": pd.DataFrame(ctype, index=ids, columns=["cell_id"]),
    }
    side = int(np.ceil((coords.max(axis=0) - lo).max() + SPOT_PITCH))
    return cell_data, side


def iscc_statistics(cell_data, side, seed, **hypers):
    vz = Visium(seed=seed, spot_pitch=SPOT_PITCH, spot_radius=1.0, count_model="dm", **hypers)
    vz.run(cell_data, grid_side=side)
    counts = vz.spot_counts.to_numpy(float)
    in_t = vz.obs["n_cells"].to_numpy() > 0
    return _fit_off_tissue(counts, vz.spot_coords, in_t), counts, in_t


def _figure(real, sims, out):
    import matplotlib.pyplot as plt
    REAL, ISCC, INK = "#9E9E9E", "#4C72B0", "#1A1A1A"        # validation/paper/style.py SOURCE colours
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    ax = axes[0]
    d, v, _ = zip(*real["off_profile"])
    ax.plot(d, v, "o", color=REAL, ms=4, label="real section, median")
    d, v, _ = zip(*real["off_profile_model"])
    ax.plot(d, v, "--", color=INK, lw=1, label=(f"bleed + ambient fitted to it: floor {real['ambient_frac']:.3f}, "
                                                f"bleed {real['bleed_frac']:.2f}, sd {real['sigma_pitch']:.2f} pitch"))
    for name, (st, filled) in sims.items():
        if st is None:
            continue
        d, v, _ = zip(*st["off_profile"])
        shift = 0.08 if filled else -0.08                     # both sit at zero; keep both visible
        ax.plot(np.asarray(d) + shift, v, "s", color=ISCC, ms=3.5, mfc=ISCC if filled else "white",
                label=name + " (all zero)")
    ax.set(xlabel="distance to the nearest tissue spot (spot pitches)",
           ylabel="spot total / mean tissue spot", title="Spots outside the tissue")
    ax.set_yscale("symlog", linthresh=1e-2)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=6.5, loc="center right")
    ax = axes[1]
    if real["in_profile"]:
        d, v, _ = zip(*real["in_profile"])
        ax.plot(d, v, "o", color=REAL, ms=4, label="real section, median")
        d, v, _ = zip(*real["in_profile_bleed"])
        ax.plot(d, v, "--", color=INK, lw=1, label="bleed alone, at the fitted values")
    ax.axhline(1.0, color="#D9D9D9", lw=0.6)
    ax.set(xlabel="distance into the tissue (spot pitches)", ylabel="spot total / interior median",
           title="Tissue spots by depth into the tissue")
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    for a in axes:
        a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    X, in_tissue, coords = load_raw(args.data_dir)
    est = estimate_visium(X, coords=coords, in_tissue=in_tissue)
    real = est.diagnostics["off_tissue"]
    h = est.hypers
    print(f"real: {real['n_in']} tissue spots, {real['n_out']} outside; mean tissue total {real['mu_in']:.0f}")
    print(f"  ambient floor {real['ambient_frac']:.4f} of a tissue spot; bleed fraction "
          f"{real['bleed_frac']:.3f}; kernel sd {real['sigma_pitch']:.2f} pitches "
          f"(diffusion_sigma {h.diffusion_sigma:.2f} units)")
    print(f"  iscc's composition smoothing at that sd keeps {real['smooth_keep']:.2f} of a spot's own "
          f"expression (bleed keeps {1 - real['bleed_frac']:.2f})")
    print(f"  ambient profile vs pooled tissue profile: r = {real['ambient_profile_r']:.3f}")
    print(f"  fitted: {est.fitted}")

    preset = Visium().hypers
    cell_data, side = tissue_like(coords, in_tissue)
    sims = {}
    st_pre, _, _ = iscc_statistics(cell_data, side, seed=1)
    st_fit, _, _ = iscc_statistics(cell_data, side, seed=1, ambient_frac=h.ambient_frac,
                                   diffusion_sigma=h.diffusion_sigma)
    sims[f"iscc at the presets (ambient {preset.ambient_frac}, diffusion {preset.diffusion_sigma})"] = (st_pre, False)
    sims["iscc at the fitted values"] = (st_fit, True)
    for name, (st, _) in sims.items():
        if st is None:
            print(f"{name}: no statistic (no spot outside the tissue)")
            continue
        med = [v for _, v, _ in st["off_profile"]]
        print(f"{name}: off-tissue median totals {np.round(med, 4).tolist()}; fitted ambient "
              f"{st['ambient_frac']:.4f}, bleed {st['bleed_frac']:.3f}")

    out = figure_path("validation_visium_background.png")
    _figure(real, sims, out)
    print(f"wrote {out}")
    if args.out_json:
        res = {"sample": SAMPLE, "real": real,
               "preset": {"ambient_frac": preset.ambient_frac, "diffusion_sigma": preset.diffusion_sigma,
                          "edge_sigma": preset.edge_sigma},
               "fitted_hypers": {"ambient_frac": h.ambient_frac, "diffusion_sigma": h.diffusion_sigma},
               "iscc_preset": st_pre, "iscc_fitted": st_fit, "fitted": est.fitted,
               "in_tissue_fit": {"mu_counts": h.mu_counts, "sigma_counts": h.sigma_counts,
                                 "field_sigma": h.field_sigma, "field_lengthscale": h.field_lengthscale,
                                 "kappa": h.kappa}}
        with open(args.out_json, "w") as fh:
            json.dump(res, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
