"""Validate F8 microenvironment-driven expression (DESIGN_features §H).

Grows one tumour twice at the same seed — with F8 OFF and ON — so the two are byte-identical in
genome/lineage/space and differ ONLY in the expression readout (F8 modulates expression, not
fitness). Shows the two cell-extrinsic fields and their effect:

  A. the hypoxia (O2-derived) field over the tissue — hypoxic dense core, oxygenated rim;
  B. hypoxia tracks local cell density (the mechanism);
  C. the cell-cell-communication field — a neighbourhood-averaged emitter density;
  D. the pure extrinsic effect: the per-cell expression fold-change (ON/OFF) of the hypoxia
     programme is exactly 1 + strength·hypoxia — a graded, spatially-coherent module that no
     amount of lineage structure explains (the cell-intrinsic-vs-extrinsic axis PEtracer measures).

Writes manuscript/figures/validation_microenvironment.png.
Run:  python -u validation/validate_microenvironment.py
"""
import argparse
import os

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# A SOLID tumour (no duct structure) on a grid large enough to leave an oxygenated margin — so the
# hypoxia field shows the classic viable-rim / hypoxic-core gradient rather than a uniformly starved
# tissue. Compact params (cf. tests/conftest.py); F8 modulates the readout only, so growth is fixed.
GENOME = {"n_segments": 10, "segment_size": 100}
SELECTION = {"prop_driver": 0.1, "prop_dispersal": 0.1, "prop_immune_resistance": 0.1,
             "prop_treatment_resistance": 0.1}
CANCER = {"division_rate": 0.6, "death_rate": 0.03, "max_birth_rate": 0.9, "mutation_rate": 0.5,
          "dispersal_rate": 0.35}
# Seed a founder cluster + a grown mass: with real per-deme crowding (DESIGN_crowding.md) demes cap
# near K and the tumour spreads, so a solid, cancer-dense mass (perfused O2 source -> hypoxic core)
# needs more steps than the old overfilling pile.
DEME = {"carrying_capacity": 8, "initial_cancer_cells": 5}
SPATIAL = {"grid_size": 26, "structure_radius": 0, "immune_density": 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=550)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(REPO, "manuscript/figures/validation_microenvironment.png"))
    args = ap.parse_args()

    from iscc.tumor.models import GenotypeTumor
    from iscc.data import morans_i

    mp = {
        "hypoxia": {"strength": 1.0, "n_genes": 60, "o2_consumption": 1.5, "o2_supply": 0.5,
                    "o2_source": "perfused"},
        "cci": {"strength": 0.8, "n_target_genes": 60, "emitter_type": "cancer", "lengthscale": 3.0},
    }

    def grow(microenv_params):
        t = GenotypeTumor(seed=args.seed, genome_params=GENOME, selection_params=SELECTION,
                          cancer_cell_params=CANCER, deme_params=DEME, spatial_params=SPATIAL,
                          microenv_params=microenv_params)
        t.grow(n_steps=args.steps, seed=args.seed)
        return t

    off = grow(None)
    on = grow(mp)
    print(f"tumour: {on.get_tumor_size()} cells on a {SPATIAL['grid_size']}^2 grid")

    # per-deme fields (occupied demes only). Compare the two O2-source models on the SAME tumour.
    dens = on._deme_density()
    occ = np.where(dens > 0)[0]
    coords = np.array([on.deme_coords[d] for d in occ], dtype=float)
    H = dict(o2_consumption=1.5, o2_supply=0.5)
    hyp_u = on._o2_field(k=H["o2_consumption"], s=H["o2_supply"], source="uniform")
    hyp_p = on._o2_field(k=H["o2_consumption"], s=H["o2_supply"], source="perfused")
    cci = on.microenv_truth["cci"]

    mi_u, mi_p = morans_i(hyp_u[occ], coords), morans_i(hyp_p[occ], coords)
    mi_cci = morans_i(cci[occ], coords)

    # the pure extrinsic effect: ON/OFF fold-change of the hypoxia programme, per cell
    hyp_only = np.setdiff1d(on.microenv_truth["hypoxia_genes"], on.microenv_truth["cci_target_genes"])
    on_e, off_e = on.cell_data["cell_exp"].values, off.cell_data["cell_exp"].values
    ratio = np.divide(on_e, off_e, out=np.full_like(on_e, np.nan), where=off_e > 0)
    fold = np.nanmean(ratio[:, hyp_only], axis=1)                 # per-cell mean fold-change
    lvl = on.cell_data["cell_microenv"]["hypoxia_level"].values

    print(f"uniform  O2 source: hypoxia {hyp_u[occ].min():.2f}..{hyp_u[occ].max():.2f}  "
          f"mean {hyp_u[occ].mean():.2f}  Moran's I {mi_u:.2f}")
    print(f"perfused O2 source: hypoxia {hyp_p[occ].min():.2f}..{hyp_p[occ].max():.2f}  "
          f"mean {hyp_p[occ].mean():.2f}  Moran's I {mi_p:.2f}  (stronger core -> comedonecrosis)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 4, figsize=(19, 4.3))
    vmax = max(hyp_u[occ].max(), hyp_p[occ].max())

    sc = ax[0].scatter(coords[:, 1], coords[:, 0], c=hyp_u[occ], cmap="inferno_r", s=26, vmin=0, vmax=vmax)
    ax[0].set_title(f"A. hypoxia — UNIFORM O2 source\n(supplied everywhere) mean {hyp_u[occ].mean():.2f}")
    ax[0].invert_yaxis(); fig.colorbar(sc, ax=ax[0], fraction=0.046)

    sc1 = ax[1].scatter(coords[:, 1], coords[:, 0], c=hyp_p[occ], cmap="inferno_r", s=26, vmin=0, vmax=vmax)
    ax[1].set_title(f"B. hypoxia — PERFUSED O2 source\n(from non-cancer tissue) mean {hyp_p[occ].mean():.2f} — hypoxic core")
    ax[1].invert_yaxis(); fig.colorbar(sc1, ax=ax[1], fraction=0.046)

    sc2 = ax[2].scatter(coords[:, 1], coords[:, 0], c=cci[occ], cmap="viridis", s=26)
    ax[2].set_title(f"C. cell-cell-communication field\nMoran's I = {mi_cci:.2f}")
    ax[2].invert_yaxis(); fig.colorbar(sc2, ax=ax[2], fraction=0.046)

    order = np.argsort(lvl)
    ax[3].scatter(lvl, fold, s=8, alpha=0.4, color="#3b6fb6")
    ax[3].plot(lvl[order], 1.0 + mp["hypoxia"]["strength"] * lvl[order], "k--", lw=1.5,
               label=r"$1+s\cdot$hypoxia")
    ax[3].set(xlabel="cell hypoxia level", ylabel="hypoxia-programme fold-change (ON/OFF)",
              title="D. pure cell-extrinsic effect\n(graded, spatial — not lineage)")
    ax[3].legend()

    fig.suptitle("F8: microenvironment-driven expression — hypoxia + cell-cell communication "
                 "(readout only; growth unchanged)", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print("figure ->", args.out)


if __name__ == "__main__":
    main()
