"""Validate the SPATIAL structure of a ductal-field tumour — clonal territories, selection, CNA, and
expression programs — at a mid generation, before the cancer has taken over the whole field.

Motivation: a naive "dominant genotype" map colours by genotype id, but under infinite-sites almost
every division spawns a new (passenger-differentiated) id, so id-space looks like noise even when
spatial clonal structure exists. The right lens is a CONTINUOUS genetic axis (the first principal
component of the per-cell SNV matrix) plus its spatial autocorrelation (Moran's I). This script shows,
on the compartment-on ductal field:

  A. CLONAL TERRITORIES — per-deme mean genetic PC1; positive Moran's I = contiguous genetically
     distinct regions (the structure a genotype-id map hides). Grown at LOW vs HIGH dispersal: lower
     dispersal TENDS to give sharper territories and higher to mix them (the operating-envelope
     relationship, DESIGN_operating_envelope.md) — a tendency over seeds, not a per-run guarantee.
  B. SELECTION — per-deme mean of the heritable compartment traits (breach at the wall, stromal
     survival in the stroma): spatially patchy where selection is acting.
  C. CNA — per-deme mean copy number (ploidy): spatial gain/loss subclones.
  D. EXPRESSION PROGRAMS — the invasive (emt) program is NICHE-driven and lights up on the epithelial
     gland walls (tracking the epithelial fraction — the genetic-vs-niche confound), while the
     proliferation program is GENOTYPE-driven and spatially flatter.

Writes manuscript/figures/validation_spatial_diagnostic.png.
Run:  python -u validation/validate_spatial_diagnostic.py [--gen 22]
"""
import argparse
import os

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NORMALS = ("epithelial", "stromal", "immune")
GENOME = {"n_segments": 6, "segment_size": 60}
SELECTION = {"prop_driver": 0.05, "prop_dispersal": 0.05, "prop_immune_resistance": 0.02,
             "prop_treatment_resistance": 0.02, "prop_breach": 0.03, "prop_stromal_survival": 0.03,
             "breach_effects": 2.2, "stromal_survival_effects": 2.2}
DEME = {"carrying_capacity": 20, "initial_cancer_cells": 8, "resident_pressure_ref": 0.2}
FIELD = {"grid_size": 20, "n_structures": 1, "structure_radius": 3, "n_glands": 4, "gland_radius": 3,
         "min_gland_sep": 8, "K_duct": 25, "K_stroma": 25, "stroma_fill_frac": 0.4,
         "cross_gland_kappa": 0.04, "cross_gland_lambda": None,
         "epithelial_barrier": 1.2, "stromal_hazard": 0.7}
EXPR = {
    "program_params": {"n_programs": 6, "n_genes_per_program": 12, "program_overlap": 0.1,
                       "seeded_programs": ("proliferation", "emt", "hypoxia", "drug_resistance",
                                           "immune_evasion")},
    "coupling_params": {"niche_program_map": {"epithelial": "emt"}, "niche_program_strength": 3.0,
                        "phenotype_program_strength": {"dispersal_rate": 0.6, "__default__": 0.5}},
    "activity_params": {"activity_mean": 1.0, "activity_sd": 0.3, "activity_noise": 0.1},
}
LOW_DISP, HIGH_DISP = 0.12, 0.35


def grow(dispersal, gen, seed):
    from iscc.tumor.models import GenotypeTumor
    cancer = {"division_rate": 0.7, "death_rate": 0.05, "max_birth_rate": 0.95,
              "mutation_rate": 0.6, "dispersal_rate": dispersal}
    t = GenotypeTumor(seed=seed, genome_params=GENOME, selection_params=SELECTION,
                      cancer_cell_params=cancer, deme_params=DEME, spatial_params=FIELD,
                      update_mode="tau", tau=1.0, expression_params=EXPR)
    t.grow(n_steps=gen, seed=seed)
    t.make_cell_data()
    return t


def cancer_mask(t):
    types = t.cell_data["cell_type"]["cell_id"].values
    return np.array([x not in NORMALS for x in types])


def deme_mean_grid(t, per_cell, mask):
    """Per-deme mean of a full-length per-cell array over the masked (cancer) cells; NaN where none."""
    from collections import defaultdict
    demes = t.cell_data["cell_deme"]["deme_id"].values
    grid = np.full((t.grid_size, t.grid_size), np.nan)
    acc = defaultdict(list)
    for i in np.where(mask)[0]:
        acc[int(demes[i])].append(per_cell[i])
    for d, vals in acc.items():
        r, c = t.deme_coords[d]
        grid[r, c] = float(np.mean(vals))
    return grid


def pc1_per_cell(t, mask):
    """First principal component of the cancer-cell SNV matrix, as a FULL-length per-cell array (NaN
    for non-cancer). The dominant genetic axis — smooth in space iff there is spatial clonal structure."""
    X = t.cell_data["cell_snv"].values[mask].astype(float)
    full = np.full(len(mask), np.nan)
    if X.shape[0] >= 2 and X.shape[1] > 0:
        X = X - X.mean(axis=0)
        U, S, _ = np.linalg.svd(X, full_matrices=False)
        full[np.where(mask)[0]] = U[:, 0] * S[0]
    return full


def morans_i_of_grid(t, grid):
    """Moran's I of an occupied-deme grid (spatial autocorrelation; >0 = clustered structure)."""
    from iscc.data import morans_i
    occ = [(r, c) for (r, c) in t.deme_coords if not np.isnan(grid[r, c])]
    if len(occ) < 3:
        return float("nan")
    coords = np.array(occ, dtype=float)
    vals = np.array([grid[r, c] for r, c in occ], dtype=float)
    if np.ptp(vals) == 0:
        return float("nan")
    return float(morans_i(vals, coords))


def occupied_cancer_demes(t):
    return sum(1 for d in t.demes if any(t._is_cancer(g) for g in d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", type=int, default=22, help="generation to snapshot (before takeover)")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(REPO, "manuscript/figures/validation_spatial_diagnostic.png"))
    args = ap.parse_args()

    lo = grow(LOW_DISP, args.gen, args.seed)
    hi = grow(HIGH_DISP, args.gen, args.seed)
    m_lo, m_hi = cancer_mask(lo), cancer_mask(hi)

    pc1_lo = deme_mean_grid(lo, pc1_per_cell(lo, m_lo), m_lo)
    pc1_hi = deme_mean_grid(hi, pc1_per_cell(hi, m_hi), m_hi)
    I_lo, I_hi = morans_i_of_grid(lo, pc1_lo), morans_i_of_grid(hi, pc1_hi)

    cd = lo.cell_data
    pn = list(lo.programs.dictionary.program_names)
    emt_k, pro_k = pn.index("emt"), pn.index("proliferation")
    breach = deme_mean_grid(lo, cd["cell_evo"]["breach"].values, m_lo)
    ss = deme_mean_grid(lo, cd["cell_evo"]["stromal_survival"].values, m_lo)
    burden = deme_mean_grid(lo, (cd["cell_snv"].values > 0).sum(1).astype(float), m_lo)
    ploidy = deme_mean_grid(lo, cd["cell_cnv"].values.mean(1), m_lo)
    emt = deme_mean_grid(lo, cd["cell_program"].values[:, emt_k], m_lo)
    prolif = deme_mean_grid(lo, cd["cell_program"].values[:, pro_k], m_lo)

    # emt vs the epithelial-niche field (the confound), per occupied deme
    epi_field = lo.microenv_truth["epithelial"]
    occ = [i for i in range(len(lo.demes))
           if not np.isnan(emt[lo.deme_coords[i][0], lo.deme_coords[i][1]])]
    emt_vals = np.array([emt[lo.deme_coords[i]] for i in occ])
    epi_vals = np.array([epi_field[i] for i in occ])
    emt_epi_r = float(np.corrcoef(emt_vals, epi_vals)[0, 1]) if np.ptp(epi_vals) > 0 else float("nan")

    n_lo, n_hi = occupied_cancer_demes(lo), occupied_cancer_demes(hi)
    print(f"gen {args.gen}: cancer occupies {n_lo}/{len(lo.demes)} (low disp) and {n_hi}/{len(hi.demes)} "
          f"(high disp) demes — still growing, not taken over")
    print("A. clonal territories (genetic PC1 spatial autocorrelation, Moran's I):")
    print(f"   low dispersal  {LOW_DISP}: I = {I_lo:.2f}   (sharper territories)")
    print(f"   high dispersal {HIGH_DISP}: I = {I_hi:.2f}   (more mixed)")
    print(f"B. selection: breach {np.nanmean(breach):.2f} / stromal_survival {np.nanmean(ss):.2f} "
          f"(mean over colonised demes)")
    print(f"C. CNA: mean ploidy {np.nanmean(ploidy):.2f}, range {np.nanmin(ploidy):.2f}-{np.nanmax(ploidy):.2f}")
    print(f"D. expression: corr(emt, epithelial niche) = {emt_epi_r:.2f} (niche-driven at the wall); "
          f"proliferation spatial range {np.nanmax(prolif) - np.nanmin(prolif):.2f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 4, figsize=(17, 8.4))

    def show(a, g, title, cmap, vmin=None, vmax=None):
        im = a.imshow(g, cmap=cmap, vmin=vmin, vmax=vmax)
        a.set_title(title, fontsize=10); a.set_xticks([]); a.set_yticks([])
        plt.colorbar(im, ax=a, fraction=0.046, pad=0.04)

    show(ax[0, 0], pc1_lo, f"A. genetic PC1 — dispersal {LOW_DISP}\nclonal territories (Moran's I = {I_lo:.2f})", "Spectral")
    show(ax[0, 1], pc1_hi, f"A. genetic PC1 — dispersal {HIGH_DISP}\nmore mixed (Moran's I = {I_hi:.2f})", "Spectral")
    show(ax[0, 2], breach, "B. breach trait\n(selection at the gland wall)", "magma", 0, 1)
    show(ax[0, 3], ss, "B. stromal_survival trait\n(selection in the stroma)", "magma", 0, 1)
    show(ax[1, 0], burden, "genetic distance\n(SNV burden per cell)", "cividis")
    show(ax[1, 1], ploidy, "C. mean copy number\n(spatial CNA structure)", "coolwarm")
    show(ax[1, 2], emt, f"D. emt program (niche-driven)\ncorr with epithelial niche = {emt_epi_r:.2f}", "inferno")
    show(ax[1, 3], prolif, "D. proliferation program\n(genotype-driven, flatter)", "inferno")

    fig.suptitle(f"Spatial diagnostics at gen {args.gen} (cancer growing — {n_lo} & {n_hi} of "
                 f"{len(lo.demes)} demes): clonal territories, selection, CNA, expression programs",
                 fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print("\nfigure ->", args.out)


if __name__ == "__main__":
    main()
