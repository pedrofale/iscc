"""Validate compartment-dependent selection (v1, DESIGN_phenotype_plasticity.md §2) on the DUCTAL
FIELD substrate (DESIGN_ductal_field.md).

Two heritable, gene-based, sequenceable traits (`breach`, `stromal_survival`) attenuate two local
compartment hazards, BOTH keyed to LIVE cell fractions (symmetric with the immune term):
    death += epithelial_barrier · epithelial_fraction(deme) · (1 − breach)          # gland wall
    death += stromal_hazard     · stromal_fraction(deme)    · (1 − stromal_survival) # stroma
So a cancer clone founded in a gland lumen is confined (DCIS) by the epithelial wall and the hostile
stroma until a subclone evolves `breach` (to cross the wall) and `stromal_survival` (to traverse the
stroma) — the DCIS→IDC transition. Cross-gland (island) dispersal spreads confined DCIS foci between
glands (lumen→lumen, no breach). The compartment is also an R13 niche field driving the invasive
program — the genetic-vs-niche expression confound. Count engine only (the cell-engine mirror is
deferred with the count-only substrate).

Panels (the paper repo's figures/validation_compartment.png):
  A. DCIS → IDC (mandatory grid growth series): with the barriers ON, cancer is confined to gland
     lumens (multi-focal DCIS) and only later invades the stroma (IDC) once escape traits evolve;
     the barrier-OFF control invades the stroma immediately.
  B. the escape traits are selected at the invasion front — mean breach / stromal_survival is higher
     in stroma-invading cancer than in confined in-gland cancer (ON), and flat in the OFF control.
  C. the genetic-vs-niche confound — controlling for genotype, invasive-program (emt) activity still
     rises with the epithelial fraction of a cell's niche; iscc knows the genetic vs niche split.

Run:  python -u validation/validate_compartment_selection.py
"""
import argparse
import os

import numpy as np
from _paths import figure_path

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GENOME = {"n_segments": 6, "segment_size": 60}
CANCER = {"division_rate": 0.7, "death_rate": 0.05, "max_birth_rate": 0.95,
          "mutation_rate": 0.6, "dispersal_rate": 0.35}
DEME = {"carrying_capacity": 20, "initial_cancer_cells": 8, "resident_pressure_ref": 0.2}
SELECTION = {"prop_driver": 0.04, "prop_dispersal": 0.0, "prop_immune_resistance": 0.02,
             "prop_treatment_resistance": 0.02, "prop_breach": 0.03, "prop_stromal_survival": 0.03,
             "breach_effects": 2.2, "stromal_survival_effects": 2.2}
GRID = 20
# A few well-separated glands (a legible multi-focal section): a lumen-founded lesion, confined foci
# spread between glands, then a subclone breaks out into the stroma (DCIS -> IDC).
FIELD = {"grid_size": GRID, "n_structures": 1, "structure_radius": 3, "n_glands": 4,
         "gland_radius": 3, "min_gland_sep": 8, "K_duct": 25, "K_stroma": 25,
         "stroma_fill_frac": 0.4, "cross_gland_kappa": 0.05, "cross_gland_lambda": None}
EPI_BARRIER, STRO_HAZARD = 1.2, 0.7
EXPR = {
    "program_params": {"n_programs": 6, "n_genes_per_program": 12, "program_overlap": 0.1,
                       "seeded_programs": ("proliferation", "emt", "hypoxia", "drug_resistance",
                                           "immune_evasion")},
    # emt genetic arm = breach (route 1); prop_dispersal=0, so dispersal->emt is inert. A dedicated
    # breach gain sizes the genetic arm of the confound (breach in [0,1) sweeps to near-fixation, so a
    # larger gain than the [0,1)-scaled default is needed for a substantial, illustrative genetic arm).
    "coupling_params": {"niche_program_map": {"epithelial": "emt"}, "niche_program_strength": 3.0,
                        "phenotype_program_strength": {"breach": 1.0, "__default__": 0.5}},
    "activity_params": {"activity_mean": 1.0, "activity_sd": 0.3, "activity_noise": 0.1},
}
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


def _stroma_cancer_pct(t):
    stroma = ingland = 0
    for di, deme in enumerate(t.demes):
        cc = sum(c for g, c in deme.items() if t._is_cancer(g))
        if t.gland_id[di] == -1:
            stroma += cc
        else:
            ingland += cc
    tot = stroma + ingland
    return 100.0 * stroma / tot if tot else 0.0


def _spatial(barriers_on):
    s = dict(FIELD)
    if barriers_on:
        s["epithelial_barrier"] = EPI_BARRIER
        s["stromal_hazard"] = STRO_HAZARD
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=int, default=36)
    ap.add_argument("--snaps", type=int, default=6)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--out", default=figure_path("validation_compartment.png"))
    args = ap.parse_args()

    from iscc.tumor.models import GenotypeTumor

    def make(barriers_on, expression=None):
        return GenotypeTumor(seed=args.seed, genome_params=GENOME, selection_params=SELECTION,
                             cancer_cell_params=CANCER, deme_params=DEME,
                             spatial_params=_spatial(barriers_on), update_mode="tau", tau=1.0,
                             expression_params=expression)

    t_on = make(True, EXPR)
    t_off = make(False)
    seg = max(1, args.gens // args.snaps)
    snap_on, snap_off, steps, stroma_on, stroma_off = [], [], [], [], []
    for k in range(args.snaps):
        t_on.grow(n_steps=seg, seed=args.seed)
        t_off.grow(n_steps=seg, seed=args.seed)
        snap_on.append(_cancer_frac_grid(t_on))
        snap_off.append(_cancer_frac_grid(t_off))
        steps.append(t_on.step)
        stroma_on.append(_stroma_cancer_pct(t_on))
        stroma_off.append(_stroma_cancer_pct(t_off))
    cd = t_on.make_cell_data()

    print(f"ductal field {FIELD['n_glands']} glands on {GRID}^2; ON cancer={t_on.get_cancer_size()}, "
          f"OFF cancer={t_off.get_cancer_size()}")
    print("\nA. DCIS -> IDC (stroma-invading cancer %, ON vs OFF over time):")
    print("   gen:      " + " ".join(f"{s:5d}" for s in steps))
    print("   ON  (barriers): " + " ".join(f"{v:4.0f}%" for v in stroma_on))
    print("   OFF (control):  " + " ".join(f"{v:4.0f}%" for v in stroma_off))

    # B. the escape traits are selected at the invasion front: in-gland (DCIS) vs stroma (IDC)
    def trait_split(t):
        t.make_cell_data()
        c = t.cell_data
        gland = c["cell_gland"]["gland_id"].values
        types = c["cell_type"]["cell_id"].values
        isc = np.array([x not in NORMALS for x in types])
        br = c["cell_evo"]["breach"].values
        ss = c["cell_evo"]["stromal_survival"].values
        ing = isc & (gland >= 0)
        stro = isc & (gland == -1)
        return {"breach": (float(br[ing].mean()) if ing.sum() else np.nan,
                           float(br[stro].mean()) if stro.sum() else np.nan),
                "stromal_survival": (float(ss[ing].mean()) if ing.sum() else np.nan,
                                     float(ss[stro].mean()) if stro.sum() else np.nan)}
    ts_on, ts_off = trait_split(t_on), trait_split(t_off)
    print("\nB. escape-trait selection (mean trait: in-gland DCIS vs stroma IDC):")
    for ax_ in ("breach", "stromal_survival"):
        print(f"   {ax_:16s} ON in-gland={ts_on[ax_][0]:.3f} stroma={ts_on[ax_][1]:.3f} | "
              f"OFF in-gland={ts_off[ax_][0]:.3f} stroma={ts_off[ax_][1]:.3f}")

    # C. the confound (on the ON tumour, program layer)
    cd = t_on.make_cell_data()
    types = cd["cell_type"]["cell_id"].values
    isc = np.array([x not in NORMALS for x in types])
    demes = cd["cell_deme"]["deme_id"].values
    epi_field = t_on.microenv_truth["epithelial"]
    cell_epi = epi_field[demes]
    emt_k = list(t_on.programs.dictionary.program_names).index("emt")
    emt = cd["cell_program"].values[:, emt_k]
    niche_emt = t_on.programs.niche_drive(
        {"epithelial": epi_field, "stromal": t_on.microenv_truth["stromal"]})[:, emt_k][demes]
    gen_emt = np.zeros(len(types))
    for i, gid in enumerate(types):
        rep = t_on.genotypes[gid]
        if rep.type == "cancer":
            gen_emt[i] = t_on.programs.clone_drive(
                rep.evolutionary_parameters, rep.baseline_rates, rep.get_snvs())[emt_k]
    emt_c, epi_c, gid_c = emt[isc], cell_epi[isc], types[isc]
    resid = emt_c.astype(float).copy()
    for g in np.unique(gid_c):
        m = gid_c == g
        resid[m] = emt_c[m] - emt_c[m].mean()
    r_partial = np.corrcoef(resid, epi_c)[0, 1] if np.ptp(epi_c) > 0 else np.nan
    var_gen, var_niche = float(np.var(gen_emt[isc])), float(np.var(niche_emt[isc]))
    frac_niche = var_niche / (var_gen + var_niche) if (var_gen + var_niche) > 0 else np.nan
    print("\nC. genetic-vs-niche confound (invasive/emt program):")
    print(f"   genotype-controlled corr(emt residual, epithelial fraction) = {r_partial:.2f}")
    print(f"   emt-drive variance: genetic {var_gen:.3f} niche {var_niche:.3f} "
          f"(niche share {frac_niche*100:.0f}%)")

    # ---------------- figure ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncol = args.snaps
    fig = plt.figure(figsize=(2.7 * ncol, 11.5), constrained_layout=True)
    gsfig = fig.add_gridspec(3, ncol)

    im = None
    for k in range(ncol):
        ax = fig.add_subplot(gsfig[0, k])
        im = ax.imshow(snap_on[k], cmap="inferno", vmin=0.0, vmax=1.0)
        ax.set_title(f"gen {steps[k]}\nstroma {stroma_on[k]:.0f}%", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        if k == 0:
            ax.set_ylabel("A. barriers ON:\nDCIS → IDC (confined → stroma)", fontsize=9)

    for k in range(ncol):
        ax = fig.add_subplot(gsfig[1, k])
        ax.imshow(snap_off[k], cmap="inferno", vmin=0.0, vmax=1.0)
        ax.set_title(f"gen {steps[k]}\nstroma {stroma_off[k]:.0f}%", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        if k == 0:
            ax.set_ylabel("A. barriers OFF:\nimmediate invasion (control)", fontsize=9)

    # Row 3 analysis: stroma curve, trait split, confound
    axc = fig.add_subplot(gsfig[2, 0:2])
    axc.plot(steps, stroma_on, "-o", color="#2c7fb8", label="barriers ON (DCIS→IDC)")
    axc.plot(steps, stroma_off, "-s", color="#a6bddb", label="barriers OFF (control)")
    axc.set_xlabel("generation"); axc.set_ylabel("stroma-invading cancer %")
    axc.set_title("A/B. barrier delays stromal invasion\n(the DCIS phase)", fontsize=9)
    axc.legend(fontsize=7)

    axt = fig.add_subplot(gsfig[2, 2:4])
    labels = ["breach", "stromal_survival"]
    xp = np.arange(2); w = 0.2
    axt.bar(xp - 1.5 * w, [ts_on[l][0] for l in labels], w, label="in-gland ON", color="#a6bddb")
    axt.bar(xp - 0.5 * w, [ts_on[l][1] for l in labels], w, label="stroma ON", color="#2c7fb8")
    axt.bar(xp + 0.5 * w, [ts_off[l][0] for l in labels], w, label="in-gland OFF", color="#fbb4b9")
    axt.bar(xp + 1.5 * w, [ts_off[l][1] for l in labels], w, label="stroma OFF", color="#c51b8a")
    axt.set_xticks(xp); axt.set_xticklabels(labels, fontsize=8); axt.set_ylim(0, 1)
    axt.set_ylabel("mean escape trait")
    axt.set_title("B. escape traits selected\nat the invasion front", fontsize=9)
    axt.legend(fontsize=6, ncol=2)

    axd = fig.add_subplot(gsfig[2, 4:ncol]) if ncol > 4 else fig.add_subplot(gsfig[2, 4])
    axd.bar(["genetic\n(clone)", "niche\n(compartment)"], [var_gen, var_niche],
            color=["#7fbf7b", "#af8dc3"])
    axd.set_ylabel("emt drive variance")
    axd.set_title(f"C. genetic vs niche\n(niche {frac_niche*100:.0f}%, r={r_partial:.2f})", fontsize=9)

    fig.suptitle("Compartment-dependent selection on the ductal field: DCIS→IDC, escape-trait "
                 "selection at the front, and the genetic-vs-niche confound", fontsize=13)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print("\nfigure ->", args.out)


if __name__ == "__main__":
    main()
