"""Validate the ductal-field spatial substrate (DESIGN_ductal_field.md).

A FIELD of many small epithelial-ring glands in moderate-density stroma, grown from ONE cancer founder, with a
low cross-gland (island) dispersal rate abstracting intraductal spread through the out-of-plane ductal
tree. Shows multi-focal, clonally-related DCIS foci arising from a single origin — the ST-realistic
layout (separate foci in stroma; connecting ducts out of plane, not drawn).

Panels (manuscript/figures/validation_ductal_field.png):
  A. GROWTH TIME-SERIES (mandatory): a row of grid snapshots (cancer fraction per deme) from seeding
     to final, over the gland field — cancer appears in one gland then seeds several others; a second
     row shows the gland_id layout and the dominant-clone map, so multi-focal spread from one founder
     is visually obvious.
  B. multi-focal dynamics: the fraction of glands colonised rises over time.
  C. RELATEDNESS: every colonised focus traces to the founder (clonal); between-focus genetic
     divergence > within-focus (the island bottleneck) — a DCIS phylogeography with a known answer.

Run:  python -u validation/validate_ductal_field.py
"""
import argparse
import os

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GENOME = {"n_segments": 6, "segment_size": 60}
CANCER = {"division_rate": 0.7, "death_rate": 0.05, "max_birth_rate": 0.95,
          "mutation_rate": 0.6, "dispersal_rate": 0.35}
SELECTION = {"prop_driver": 0.05, "prop_dispersal": 0.08, "prop_immune_resistance": 0.02,
             "prop_treatment_resistance": 0.02}
DEME = {"carrying_capacity": 20, "initial_cancer_cells": 8, "resident_pressure_ref": 0.2}
GRID = 20
# A few well-separated glands (a legible multi-focal section) and a LOW cross-gland rate, so the field
# colonises gradually, one focus at a time, from the single founder gland — the island signature —
# with each focus founded by a bottleneck so the foci diverge. See the colonisation curve in panel B.
FIELD = {"grid_size": GRID, "n_structures": 1, "structure_radius": 3, "n_glands": 4,
         "gland_radius": 3, "min_gland_sep": 8, "K_duct": 25, "K_stroma": 25,
         "stroma_fill_frac": 0.35, "cross_gland_kappa": 0.04, "cross_gland_lambda": None}
NORMALS = ("epithelial", "stromal", "immune")


def _cancer_frac_grid(t):
    grid = np.full((t.grid_size, t.grid_size), np.nan)
    for di, deme in enumerate(t.demes):
        total = sum(deme.values())
        if total == 0:
            continue
        cancer = sum(c for g, c in deme.items() if t._is_cancer(g))
        r, c = t.deme_coords[di]
        grid[r, c] = cancer / total
    return grid


def _glands_colonised(t):
    out = set()
    for di, deme in enumerate(t.demes):
        if t.gland_id[di] >= 0 and any(t._is_cancer(g) for g in deme):
            out.add(int(t.gland_id[di]))
    return out


def _dominant_clone_grid(t):
    """Per-deme dominant cancer genotype ordinal (-1 where no cancer), for the clonal-structure map."""
    grid = np.full((t.grid_size, t.grid_size), -1.0)
    for di, deme in enumerate(t.demes):
        cancer = {g: c for g, c in deme.items() if t._is_cancer(g)}
        if not cancer:
            continue
        g = max(cancer, key=cancer.get)
        r, c = t.deme_coords[di]
        grid[r, c] = t.genotypes[g].ord
    return grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=int, default=36)
    ap.add_argument("--snaps", type=int, default=6)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(REPO, "manuscript/figures/validation_ductal_field.png"))
    args = ap.parse_args()

    from iscc.tumor.models import GenotypeTumor

    t = GenotypeTumor(seed=args.seed, genome_params=GENOME, selection_params=SELECTION,
                      cancer_cell_params=CANCER, deme_params=DEME, spatial_params=FIELD,
                      update_mode="tau", tau=1.0)
    gland_layout = np.full((t.grid_size, t.grid_size), -1.0)
    for di in range(len(t.demes)):
        r, c = t.deme_coords[di]
        gland_layout[r, c] = t.gland_id[di]

    # grow in equal segments, snapshot the cancer-fraction grid + glands colonised at each
    seg = max(1, args.gens // args.snaps)
    snap_frac, snap_step, invaded = [], [], []
    for k in range(args.snaps):
        t.grow(n_steps=seg, seed=args.seed)
        snap_frac.append(_cancer_frac_grid(t))
        snap_step.append(t.step)
        invaded.append(len(_glands_colonised(t)))
    t.make_cell_data()

    colonised = _glands_colonised(t)
    n_cancer = t.get_cancer_size()
    print(f"ductal field: {t.n_glands} glands on {GRID}^2, {n_cancer} cancer cells from ONE founder")
    print(f"glands colonised over time: {invaded}  (final {len(colonised)}/{t.n_glands})")

    # per-gland cancer counts (scale sanity)
    per_gland = {}
    for di, deme in enumerate(t.demes):
        gi = int(t.gland_id[di])
        if gi >= 0:
            per_gland[gi] = per_gland.get(gi, 0) + sum(c for g, c in deme.items() if t._is_cancer(g))
    counts = np.array([per_gland[g] for g in sorted(colonised)]) if colonised else np.array([0])
    print(f"per-gland cancer counts: median {int(np.median(counts))}, range {int(counts.min())}-{int(counts.max())}")

    # RELATEDNESS: do all cancer genotypes trace to the founder?
    def traces(gid):
        seen = 0
        while gid in t.genotypes_parents and seen < 10 ** 6:
            gid = t.genotypes_parents[gid]; seen += 1
            if gid == t.founder_id:
                return True
        return gid == t.founder_id
    cancer_gids = [g for g in t.genotypes_counts if t._is_cancer(g)]
    frac_related = np.mean([traces(g) for g in cancer_gids]) if cancer_gids else 0.0

    # within- vs between-focus genetic divergence (island bottleneck): per-gland consensus SNV, then
    # mean pairwise Hamming between glands vs mean per-gland spread from its own consensus.
    cd = t.make_cell_data()
    snv = cd["cell_snv"].values > 0
    gland_of = cd["cell_gland"]["gland_id"].values
    types = cd["cell_type"]["cell_id"].values
    is_cancer = np.array([x not in NORMALS for x in types])
    gl = sorted(g for g in colonised)
    consensus, within = {}, []
    for g in gl:
        m = is_cancer & (gland_of == g)
        if m.sum() == 0:
            continue
        con = snv[m].mean(axis=0) > 0.5
        consensus[g] = con
        within.append(np.mean(snv[m] != con))
    keys = list(consensus)
    between = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            between.append(np.mean(consensus[keys[i]] != consensus[keys[j]]))
    within_m = float(np.mean(within)) if within else 0.0
    between_m = float(np.mean(between)) if between else 0.0
    print(f"relatedness: {frac_related*100:.0f}% of cancer genotypes trace to the founder (clonal)")
    print(f"divergence: within-focus {within_m:.3f} vs between-focus {between_m:.3f} SNV fraction "
          f"(island bottleneck -> foci diverge)")

    # ---------------- figure ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncol = args.snaps
    fig, axes = plt.subplots(2, ncol, figsize=(2.7 * ncol, 8.2), constrained_layout=True)

    # Row 1: the cancer-fraction growth time-series (the mandatory grid movie)
    im = None
    for k in range(ncol):
        ax = axes[0, k]
        im = ax.imshow(snap_frac[k], cmap="inferno", vmin=0.0, vmax=1.0)
        ax.set_title(f"gen {snap_step[k]}\n{invaded[k]}/{t.n_glands} glands", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    axes[0, 0].set_ylabel("A. cancer fraction\n(grid growth series)", fontsize=9)
    fig.colorbar(im, ax=axes[0, :].tolist(), fraction=0.015, pad=0.01, label="cancer fraction")

    # Row 2: gland layout, dominant-clone map, colonisation curve, per-gland load, divergence
    axg = axes[1, 0]
    axg.imshow(np.where(gland_layout < 0, np.nan, gland_layout), cmap="tab20")
    axg.set_title("gland_id layout\n(founder in gland 0)", fontsize=9)
    axg.set_xticks([]); axg.set_yticks([])

    axc = axes[1, 1]
    dom = _dominant_clone_grid(t)
    axc.imshow(np.where(dom < 0, np.nan, dom), cmap="nipy_spectral")
    axc.set_title("dominant cancer clone\n(shared origin)", fontsize=9)
    axc.set_xticks([]); axc.set_yticks([])

    axk = axes[1, 2]
    axk.plot(snap_step, invaded, "-o", color="#2c7fb8")
    axk.set_xlabel("generation"); axk.set_ylabel("glands colonised")
    axk.set_title("B. multi-focal spread\nfrom one founder", fontsize=9)

    axr = axes[1, 3] if ncol > 3 else None
    if axr is not None:
        axr.bar(range(len(gl)), [per_gland[g] for g in gl], color="#c51b8a")
        axr.set_xlabel("colonised gland"); axr.set_ylabel("cancer cells")
        axr.set_title(f"per-gland load\n({len(colonised)} foci)", fontsize=9)

    axd = axes[1, 4] if ncol > 4 else None
    if axd is not None:
        axd.bar(["within\nfocus", "between\nfoci"], [within_m, between_m], color=["#7fbf7b", "#af8dc3"])
        axd.set_ylabel("mean SNV difference")
        axd.set_title(f"C. divergence\n(clonal: {frac_related*100:.0f}%)", fontsize=9)

    for k in range(5, ncol):
        axes[1, k].axis("off")

    fig.suptitle("Ductal-field substrate: multi-focal, clonally-related DCIS foci from ONE founder "
                 "via cross-gland (island) dispersal", fontsize=13)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print("\nfigure ->", args.out)


if __name__ == "__main__":
    main()
