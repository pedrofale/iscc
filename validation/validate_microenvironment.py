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
DEME = {"carrying_capacity": 6}
SPATIAL = {"grid_size": 26, "structure_radius": 0, "immune_density": 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=320)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(REPO, "manuscript/figures/validation_microenvironment.png"))
    args = ap.parse_args()

    from iscc.tumor.models import GenotypeTumor
    from iscc.data import morans_i

    mp = {
        "hypoxia": {"strength": 1.0, "n_genes": 60, "o2_consumption": 1.5, "o2_supply": 0.5},
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

    # per-deme fields (occupied demes only)
    dens = on._deme_density()
    occ = np.where(dens > 0)[0]
    coords = np.array([on.deme_coords[d] for d in occ], dtype=float)
    hyp = on.microenv_truth["hypoxia"]
    cci = on.microenv_truth["cci"]
    emit = on._emitter_density("cancer")

    mi_hyp = morans_i(hyp[occ], coords)
    mi_cci = morans_i(cci[occ], coords)
    r_dens = np.corrcoef(dens[occ], hyp[occ])[0, 1]
    r_emit = np.corrcoef(emit[occ], cci[occ])[0, 1]

    # the pure extrinsic effect: ON/OFF fold-change of the hypoxia programme, per cell
    hyp_g = on.microenv_truth["hypoxia_genes"]
    cci_g = on.microenv_truth["cci_target_genes"]
    hyp_only = np.setdiff1d(hyp_g, cci_g)
    on_e, off_e = on.cell_data["cell_exp"].values, off.cell_data["cell_exp"].values
    ratio = np.divide(on_e, off_e, out=np.full_like(on_e, np.nan), where=off_e > 0)
    fold = np.nanmean(ratio[:, hyp_only], axis=1)                 # per-cell mean fold-change
    lvl = on.cell_data["cell_microenv"]["hypoxia_level"].values

    print(f"hypoxia field  : Moran's I {mi_hyp:.3f}  corr(density) {r_dens:.3f}  "
          f"range {hyp[occ].min():.2f}..{hyp[occ].max():.2f}")
    print(f"CCI field      : Moran's I {mi_cci:.3f}  corr(emitter) {r_emit:.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 4, figsize=(19, 4.3))

    sc = ax[0].scatter(coords[:, 1], coords[:, 0], c=hyp[occ], cmap="inferno_r", s=26)
    ax[0].set_title(f"A. hypoxia field (O2-derived)\nMoran's I = {mi_hyp:.2f} — hypoxic core, oxygenated rim")
    ax[0].invert_yaxis(); fig.colorbar(sc, ax=ax[0], fraction=0.046)

    ax[1].scatter(dens[occ], hyp[occ], s=14, alpha=0.6, color="#c0413b")
    ax[1].set(xlabel="deme cell density", ylabel="hypoxia",
              title=f"B. hypoxia tracks density\ncorr = {r_dens:.2f}")

    sc2 = ax[2].scatter(coords[:, 1], coords[:, 0], c=cci[occ], cmap="viridis", s=26)
    ax[2].set_title(f"C. cell-cell-communication field\nMoran's I = {mi_cci:.2f}  corr(emitter) = {r_emit:.2f}")
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
