"""Spot swapping on a real Visium section and on iscc: the spots outside the tissue, and the tissue edge.

A Visium slide carries 4,992 spots whatever tissue is placed on it, and Space Ranger's raw matrix
keeps all of them with an ``in_tissue`` flag. A spot outside the tissue holds no cells, so whatever it
counts bled from the tissue. This script fits SpotClean's spot-swapping model to those spots
(``estimate_visium(..., in_tissue=...)``, which calls ``_fit_bleed``: a bleeding rate, its distal
share spread over the whole slide, and a Gaussian kernel), leaving out the spots nearest the tissue
(``--min-dist``), then runs iscc's Visium assay on a section with the same tissue outline with the
bleed off (the default) and at the fitted values, and draws:

  a. the spots outside the tissue: median total, as a fraction of the mean tissue spot, against the
     distance to the nearest tissue spot (in spot pitches), real and iscc, with the model's curve on
     the real slide;
  b. the tissue spots: median total relative to the interior, against the distance into the tissue.

It also prints the share of the slide's UMIs counted outside the tissue (SpotClean's paper reports
5-20% across most public datasets).

The real section is 10x Genomics' V1_Breast_Cancer_Block_A_Section_1 (Space Ranger 1.1.0), the same
section Fig 2g fits library depth on. Pass its directory, holding the raw ``.h5`` and
``spatial/tissue_positions_list.csv``:

    python validation/validate_visium_background.py --data-dir <dir> [--min-dist 1.5] [--out-json fit.json]

Produces the paper repo's figures/validation_visium_background.png.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from iscc.data import Visium, estimate_visium
from iscc.data.estimate_visium import _fit_bleed, _pitch
from iscc.data.visium_background import bleed_kernel
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


def _profiles(totals, coords, in_tissue, step=0.5, dmax=8.0):
    """Median spot total outside the tissue by distance to it (as a fraction of the mean tissue spot),
    and inside the tissue by distance into it (relative to the interior, 5+ pitches in)."""
    from scipy.spatial import cKDTree
    xy = coords / _pitch(coords)
    tis = np.asarray(in_tissue, bool)
    y = np.asarray(totals, float) / float(np.mean(totals[tis]))
    d_out = cKDTree(xy[tis]).query(xy[~tis])[0]
    d_in = cKDTree(xy[~tis]).query(xy[tis])[0]
    interior = d_in >= 5.0
    rel = y[tis] / np.median(y[tis][interior]) if interior.any() else None

    def prof(dist, val):
        # bins centred on whole and half pitches, so the hexagonal rings (1, 1.73, 2, ...) of a real
        # slide and of iscc's grid fall in the same bins
        bins = np.arange(0.25, dmax + step, step)
        idx = np.digitize(dist, bins) - 1
        return [(float(0.5 * (bins[i] + bins[i + 1])), float(np.median(val[idx == i])), int((idx == i).sum()))
                for i in range(len(bins) - 1) if (idx == i).sum() >= 5]
    return dict(off=prof(d_out, y[~tis]), inward=(prof(d_in, rel) if rel is not None else None))


def _mean_profiles(profs):
    """Per-bin mean of several slides' median profiles (bins present in every slide)."""
    out = {}
    for key in ("off", "inward"):
        rows = [p[key] for p in profs if p[key]]
        if not rows:
            out[key] = None
            continue
        common = set.intersection(*[{d for d, _, _ in r} for r in rows])
        out[key] = [(d, float(np.mean([v for r in rows for dd, v, _ in r if dd == d])),
                     int(sum(n for r in rows for dd, _, n in r if dd == d))) for d in sorted(common)]
    return out


def model_on_slide(totals, coords, in_tissue, rate, distal, sigma_pitch):
    """The fitted model's expected totals on the real slide: tissue spots keep their counts less what
    bleeds out, every spot receives the kernel-weighted and the distal share of what bled."""
    xy = coords / _pitch(coords)
    src = np.where(in_tissue, np.asarray(totals, float), 0.0)
    W = bleed_kernel(xy, sigma_pitch)
    return (1.0 - rate) * src + rate * (1.0 - distal) * (W @ src) + rate * distal * src.mean()


def iscc_slide(cell_data, side, seed, **hypers):
    # the expected depth: the random capture field (correlated over ~9 pitches) and the per-spot depth
    # noise are switched off, so the profiles show the assay's mechanism, not one field's realisation
    hypers = dict(dict(field_sigma=0.0, sigma_counts=1e-6), **hypers)
    vz = Visium(seed=seed, spot_pitch=SPOT_PITCH, spot_radius=1.0, count_model="dm", **hypers)
    vz.run(cell_data, grid_side=side)
    totals = vz.spot_counts.to_numpy(float).sum(axis=1)
    in_t = vz.obs["n_cells"].to_numpy() > 0
    return totals, np.asarray(vz.spot_coords, float), in_t


def _figure(real, model, sims, fit, out):
    import matplotlib.pyplot as plt
    REAL, ISCC, INK = "#9E9E9E", "#4C72B0", "#1A1A1A"        # validation/paper/style.py SOURCE colours
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    ax = axes[0]
    d, v, _ = zip(*real["off"])
    ax.plot(d, v, "o", color=REAL, ms=4, label="real section, median")
    d, v, _ = zip(*model["off"])
    ax.plot(d, v, "-", color=INK, lw=1,
            label=(f"SpotClean's model fitted beyond {fit['min_dist']:g} pitches: rate {fit['bleed_rate']:.2f}, "
                   f"distal {fit['bleed_distal']:.2f}, sd {fit['sigma_pitch']:.2f} pitches"))
    for name, (st, filled) in sims.items():
        d, v, _ = zip(*st["off"])
        ax.plot(d, v, "s", color=ISCC, ms=3.5, mfc=ISCC if filled else "white", label=name)
    ax.axvspan(0, fit["min_dist"], color="#F2F2F2", lw=0, zorder=0)
    ax.set(xlabel="distance to the nearest tissue spot (spot pitches)",
           ylabel="spot total / mean tissue spot", title="a  Spots outside the tissue")
    ax.set_yscale("symlog", linthresh=1e-2)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=6.5, loc="center right")
    ax = axes[1]
    if real["inward"]:
        d, v, _ = zip(*real["inward"])
        ax.plot(d, v, "o", color=REAL, ms=4, label="real section, median")
    for name, (st, filled) in sims.items():
        if st["inward"]:
            d, v, _ = zip(*st["inward"])
            ax.plot(d, v, "s", color=ISCC, ms=3.5, mfc=ISCC if filled else "white", label=name)
    ax.axhline(1.0, color="#D9D9D9", lw=0.6)
    ax.set(xlabel="distance into the tissue (spot pitches)", ylabel="spot total / interior median",
           title="b  Tissue spots by depth into the tissue")
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    for a in axes:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--min-dist", type=float, default=1.5)
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    X, in_tissue, coords = load_raw(args.data_dir)
    est = estimate_visium(X, coords=coords, in_tissue=in_tissue, bleed_min_dist=args.min_dist)
    fit = est.diagnostics["bleed"]
    h = est.hypers
    totals = np.asarray(X.sum(axis=1), float).ravel()
    real = _profiles(totals, coords, in_tissue)
    model = _profiles(model_on_slide(totals, coords, in_tissue, fit["bleed_rate"], fit["bleed_distal"],
                                     fit["sigma_pitch"]), coords, in_tissue)
    print(f"real: {int(in_tissue.sum())} tissue spots, {int((~in_tissue).sum())} outside; "
          f"{fit['background_share']:.3f} of the slide's UMIs outside the tissue")
    print(f"  SpotClean's model beyond {args.min_dist:g} pitches ({fit['n_fit']} spots): bleeding rate "
          f"{fit['bleed_rate']:.3f}, distal share {fit['bleed_distal']:.3f}, sd {fit['sigma_pitch']:.2f} pitches")
    edge = est.diagnostics.get("edge")
    if edge:
        print(f"  outermost tissue ring at {edge['observed_ring']:.3f} of the interior, bleed alone predicts "
              f"{edge['bleed_predicted_ring']:.3f}; falloff then bleed match it at edge_sigma "
              f"{edge['edge_sigma']:.3f} (edge_reference tissue)")
    print(f"  fitted: {est.fitted}")

    cell_data, side = tissue_like(coords, in_tissue)
    # the fitted setting: the bleed carries the slide-wide floor (no in-spot soup), and the falloff on
    # the tissue's outermost ring is the drop there beyond what the bleed predicts
    bleed_kw = dict(bleed_rate=h.bleed_rate, bleed_distal=h.bleed_distal, bleed_sigma=h.bleed_sigma,
                    ambient_frac=0.0, edge_sigma=h.edge_sigma, edge_reference=h.edge_reference)
    sims, stats = {}, {}
    for name, kw, filled in (("iscc, default assay (no bleed), expected depth", {}, False),
                             ("iscc, bleed and tissue-edge falloff at the fitted values, expected depth",
                              bleed_kw, True)):
        profs, shares, refits = [], [], []
        for seed in range(1, args.n_seeds + 1):
            tot, crd, in_t = iscc_slide(cell_data, side, seed=seed, **kw)
            profs.append(_profiles(tot, crd, in_t))
            shares.append(float(tot[~in_t].sum() / tot.sum()))
            if tot[~in_t].sum() > 0:
                refits.append(_fit_bleed(tot[:, None], crd, in_t, min_dist=args.min_dist))
        sims[name] = (_mean_profiles(profs), filled)
        stats[name] = dict(background_share=float(np.mean(shares)), refits=refits, n_seeds=args.n_seeds)
        print(f"{name}: {np.mean(shares):.3f} of UMIs outside the tissue (mean of {args.n_seeds} seeds)")

    out = figure_path("validation_visium_background.png")
    _figure(real, model, sims, fit, out)
    print(f"wrote {out}")
    if args.out_json:
        res = {"sample": SAMPLE, "bleed_fit": fit, "edge_fit": est.diagnostics.get("edge"),
               "real_profiles": real, "model_profiles": model,
               "iscc": {k: dict(stats[k], profiles=sims[k][0]) for k in sims},
               "fitted": est.fitted, "hypers": h.to_dict(),
               "off_tissue_description": est.diagnostics.get("off_tissue")}
        with open(args.out_json, "w") as fh:
            json.dump(res, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
