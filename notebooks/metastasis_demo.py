"""Metastasis module (R9) — end-to-end demo of the full clinical arc on the ductal field:

    grow (DCIS -> IDC)  ->  seed the metastasis  ->  resect the primary  ->  systemic chemo on the met  ->  relapse

Produces the two acceptance figures (both on ONE shared clone colormap, so a clone is the same colour
in the primary grid, the met grid, and both Muller bands):

  * ``metastasis_grids.png``  — the two spatial grids (primary duct field + metastatic deposit) side by
    side at end of growth, coloured by cancer fraction so the minority met cancer is visible.
  * ``metastasis_muller.png`` — the 2-band Muller (primary over metastasis) across the whole arc,
    annotated with the seeding / resection / chemo events. The met founder inherits its (usually minor)
    primary clone's colour.

Run:  python notebooks/metastasis_demo.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from iscc.tumor.models import GenotypeTumor
from iscc.treatment import Surgery, Chemotherapy

OUT = os.path.join(os.path.dirname(__file__), "example_out")
os.makedirs(OUT, exist_ok=True)

GENOME = {"n_segments": 12, "segment_size": 50}
# CINner selection on the ductal field + the two DCIS->IDC escape axes + the met-establishment axis.
SELECTION = {"prop_driver": 0.04, "prop_dispersal": 0.0, "prop_immune_resistance": 0.02,
             "prop_treatment_resistance": 0.03, "prop_breach": 0.03, "prop_stromal_survival": 0.03,
             "prop_met_survival": 0.05, "breach_effects": 2.2, "stromal_survival_effects": 2.2,
             "met_survival_effects": 2.5}
CANCER = {"division_rate": 0.7, "death_rate": 0.05, "max_birth_rate": 0.95, "mutation_rate": 0.6,
          "dispersal_rate": 0.35, "cnv_prob": 0.15, "wgd_rate": 0.05}
DEME = {"carrying_capacity": 20, "initial_cancer_cells": 8, "resident_pressure_ref": 0.2}
SPATIAL = {"grid_size": 26, "structure_radius": 3, "n_glands": 4, "gland_radius": 3, "min_gland_sep": 8,
           "K_duct": 40, "K_stroma": 26, "stroma_fill_frac": 0.3, "cross_gland_kappa": 0.06,
           "epithelial_barrier": 1.2, "stromal_hazard": 0.7,
           # the metastatic deposit
           "met_grid_size": 12, "K_met": 18, "host_fill_frac": 0.35,
           "met_seed_kappa": 0.07, "met_hazard": 0.45, "met_transit_floor": 0.02}


def compartment_cancer(t):
    p = sum(c for i in range(t.n_primary_demes) for g, c in t.demes[i].items() if t._is_cancer(g))
    m = sum(c for i in range(t.n_primary_demes, len(t.demes)) for g, c in t.demes[i].items() if t._is_cancer(g))
    return p, m


def normal_totals(t):
    """Per-snapshot total NORMAL (host/epithelial/stromal/immune) cell counts, primary and met, read
    from the per-compartment traces (which include the normal genotypes)."""
    from iscc.constants import normal_names
    prim, met = [], []
    for tr in t.traces:
        pc, mc = tr.get("primary_counts", tr["genotypes_counts"]), tr.get("met_counts", {})
        prim.append(sum(v for k, v in pc.items() if k in normal_names))
        met.append(sum(v for k, v in mc.items() if k in normal_names))
    return np.array(prim), np.array(met)


def main():
    t = GenotypeTumor(seed=3, genome_params=GENOME, selection_params=SELECTION, cancer_cell_params=CANCER,
                      deme_params=DEME, spatial_params=SPATIAL, update_mode="tau", tau=1.0)

    # 1) grow (DCIS -> IDC) until the metastasis is well established
    seeding_snap = None
    while True:
        p, m = compartment_cancer(t)
        if m >= 250 or p + m >= 5500:
            break
        t.grow(n_steps=2, seed=3)
        if seeding_snap is None and compartment_cancer(t)[1] > 0:
            seeding_snap = len(t.traces)
    p, m = compartment_cancer(t)
    print(f"grown: primary={p} met={m}  seeding@snapshot={seeding_snap}  seeding_events={len(t.events)}")

    # acceptance figure 1 — the two grids, coloured by FUNCTIONAL CLONE (same palette as the Muller)
    t.make_cell_data()
    t.plot_grid_compartments(color=["cancer_frac"])
    plt.gcf().suptitle("primary duct field  +  metastatic deposit  (cancer fraction per deme)")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "metastasis_grids.png"), dpi=95); plt.close("all")

    # 2) resect the primary, then 3) systemic chemo (toxicity kills off-target normal tissue too);
    #    watch the met regress and relapse from the surviving resistant clones.
    resect_snap = len(t.traces)
    t.grow(n_steps=6, seed=3, treatment=Surgery(start=t.step, site="primary"))
    print(f"post-resection: primary={compartment_cancer(t)[0]} met={compartment_cancer(t)[1]}")

    chemo_snap = len(t.traces)
    chemo = Chemotherapy(start=t.step, duration=6, kill_rate=1.4, effectiveness=0.95,
                         toxicity=0.12, sites="both")   # toxicity = off-target ANYTHING (normals too)
    t.grow(n_steps=6, seed=3, treatment=chemo)
    chemo_end_snap = len(t.traces)
    t.grow(n_steps=10, seed=3)          # relapse of the surviving (resistant/persister) met clones
    print(f"post-chemo+relapse: primary={compartment_cancer(t)[0]} met={compartment_cancer(t)[1]}")

    marks = [(seeding_snap or 0, "seeding"), (resect_snap, "resection"),
             (chemo_snap, "chemo start"), (chemo_end_snap, "chemo end")]

    # acceptance figure 2 — the 2-band Muller across the whole arc, coloured by FUNCTIONAL clone so the
    # selective sweeps (DCIS→IDC breach, the met founder, the post-chemo resistant escape) are visible.
    t.plot_muller_compartments(by_drivers=True, min_freq=0.05, mark_generations=marks)
    plt.gcf().suptitle("2-band Muller (functional clones): primary (top) collapses at resection; "
                       "metastasis (bottom) regresses under chemo then relapses")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "metastasis_muller.png"), dpi=95); plt.close("all")

    # figure 3 — total NORMAL (host/epithelial/stromal/immune) cells over the arc: they drop at
    # resection (primary removed) and decline under chemo's off-target toxicity.
    prim_n, met_n = normal_totals(t)
    x = np.arange(len(prim_n))
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(x, prim_n + met_n, color="black", lw=2, label="total normal cells")
    ax.plot(x, prim_n, color="tab:green", lw=1, ls="--", label="primary normals")
    ax.plot(x, met_n, color="tab:blue", lw=1, ls="--", label="met normals")
    ax.axvspan(chemo_snap, chemo_end_snap, color="tab:red", alpha=0.12, label="chemo window")
    ax.axvline(resect_snap, color="k", ls=":", lw=1); ax.text(resect_snap, ax.get_ylim()[1] * 0.95,
                                                              "resection", rotation=90, va="top", fontsize=7)
    ax.set_xlabel("snapshot"); ax.set_ylabel("normal cells"); ax.legend(fontsize=8)
    ax.set_title("Normal tissue: primary removed at resection; met host tissue killed by chemo toxicity")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "metastasis_normals.png"), dpi=95); plt.close("all")
    print("wrote metastasis_grids.png, metastasis_muller.png, metastasis_normals.png to", OUT)


if __name__ == "__main__":
    main()
