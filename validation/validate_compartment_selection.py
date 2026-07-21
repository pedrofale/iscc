"""Validate compartment-dependent selection (v1, DESIGN_phenotype_plasticity.md §2).

A STRUCTURED (gland) tumour: cancer founds in the lumen, must **breach** the epithelial ring, then
**survive** the stroma. Two heritable, gene-based, sequenceable traits (`breach`, `stromal_survival`)
attenuate two matching compartment hazards (`epithelial_barrier`, `stromal_hazard`); the compartment
is read from each deme's LIVE composition, so selection tracks where the resident normals still are.
The compartment is also an R13 niche field driving the invasive (emt) program — the genetic-vs-niche
expression confound.

We exploit the thing that separates iscc from a real experiment: we can **dial one hazard at a time**
and know the ground-truth gene roles. Four arms at several seeds: OFF (no barrier), EPI (epithelial
only), STRO (stromal only), BOTH. Panels (manuscript/figures/validation_compartment.png):

  A. SEQUENTIAL INVASION — the two traits are positively selected by their barriers: population-mean
     breach / stromal_survival, BOTH vs the OFF control, and their spatial sweep over the tumour.
  B. SELECTION RECOVERY — a gene selected specifically when the EPITHELIAL barrier is dialled on is a
     breach gene; specifically by the STROMAL hazard, a stromal-survival gene. AUROC of recovering the
     gene role + its compartment from scDNA frequencies alone, vs the iscc ground-truth roles.
  C. THE CONFOUND — controlling for genotype, invasive-program (emt) activity still rises with the
     epithelial fraction of a cell's niche (env-responsive phenotype), so a naive "invasive expression
     => invasive genotype" call is confounded by location. iscc knows the genetic and the niche
     contribution to the program separately; we quantify the split.

Run:  python -u validation/validate_compartment_selection.py
"""
import argparse
import os

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STRUCTURE_RADIUS = 5
GENOME = {"n_segments": 6, "segment_size": 60}
CANCER = {"division_rate": 0.7, "death_rate": 0.05, "max_birth_rate": 0.95,
          "mutation_rate": 0.5, "dispersal_rate": 0.35}
# resident_pressure_ref below the cancer division so the crowding gate is mild and the *barrier* is the
# compartment-specific selective pressure a clone must clear (raising `breach` / `stromal_survival`).
DEME = {"carrying_capacity": 20, "initial_cancer_cells": 10, "resident_pressure_ref": 0.2}
SELECTION = {"prop_driver": 0.03, "prop_dispersal": 0.05, "prop_immune_resistance": 0.02,
             "prop_treatment_resistance": 0.02, "prop_breach": 0.02, "prop_stromal_survival": 0.02,
             "breach_effects": 2.5, "stromal_survival_effects": 2.5}
# The epithelial ring is one deme thick and the founder is seeded against it, so breach is selected
# in a thin, transient zone; a stronger `epithelial_barrier` sharpens that selection at the ring
# (DESIGN_phenotype_plasticity.md §5, "raise the barrier rather than thicken the geometry"). 1.5 puts
# breach@ring at ~0.78 vs ~0.55 in the barrier-off control while the tumour still invades.
EPI_BARRIER, STRO_HAZARD = 1.5, 0.5
# R13 program layer (for panel C): the compartment (epithelial fraction) drives the seeded `emt`
# program (route 3); the existing dispersal->emt map (route 1) is the GENETIC arm. Both feed ONE
# program — the confound. Readout-only: growth is byte-identical program-on vs -off.
EXPR = {
    "program_params": {"n_programs": 6, "n_genes_per_program": 12, "program_overlap": 0.1,
                       "seeded_programs": ("proliferation", "emt", "hypoxia", "drug_resistance",
                                           "immune_evasion")},
    "coupling_params": {"niche_program_map": {"epithelial": "emt"}, "niche_program_strength": 3.0,
                        "phenotype_program_strength": {"dispersal_rate": 0.6, "__default__": 0.5}},
    "activity_params": {"activity_mean": 1.0, "activity_sd": 0.3, "activity_noise": 0.1},
}
GRID = 24


def _compartment(t):
    center = t.grid_size // 2
    coords = np.array(t.deme_coords, dtype=float)
    d = np.hypot(coords[:, 0] - center, coords[:, 1] - center)
    return np.where(d < STRUCTURE_RADIUS - 0.5, "lumen",
                    np.where(d < STRUCTURE_RADIUS + 0.5, "ring", "stroma"))


def _is_cancer_mask(types):
    return np.array([x not in ("epithelial", "stromal", "immune") for x in types])


def _auroc(labels, scores):
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    ok = ~np.isnan(scores)
    labels, scores = labels[ok], scores[ok]
    P, N = int(labels.sum()), int((~labels).sum())
    if P == 0 or N == 0:
        return np.nan
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    u, inv, cnt = np.unique(scores, return_inverse=True, return_counts=True)
    avg = np.zeros(len(u)); np.add.at(avg, inv, ranks); avg /= cnt
    ranks = avg[inv]
    return (ranks[labels].sum() - P * (P + 1) / 2) / (P * N)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=55)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--out", default=os.path.join(REPO, "manuscript/figures/validation_compartment.png"))
    args = ap.parse_args()

    from iscc.tumor.models import GenotypeTumor

    def grow(seed, eb, sh, expression=None):
        spatial = {"grid_size": GRID, "n_structures": 1, "structure_radius": STRUCTURE_RADIUS,
                   "epithelial_barrier": eb, "stromal_hazard": sh}
        t = GenotypeTumor(seed=seed, genome_params=GENOME, selection_params=SELECTION,
                          cancer_cell_params=CANCER, deme_params=DEME, spatial_params=spatial,
                          update_mode="tau", tau=1.0, expression_params=expression)
        t.grow(n_steps=args.steps, seed=seed)
        t.make_cell_data()
        return t

    ARMS = {"off": (0.0, 0.0), "epi": (EPI_BARRIER, 0.0),
            "stro": (0.0, STRO_HAZARD), "both": (EPI_BARRIER, STRO_HAZARD)}
    COMPS = ("lumen", "ring", "stroma")
    # per-arm: mean cancer SNV frequency (for the selection scan) and per-COMPARTMENT trait means
    # (a compartment trait is selected in its OWN compartment, so a population mean — dominated by the
    # large stroma — would mask the breach signal at the thin ring). Averaged over seeds.
    freq = {a: [] for a in ARMS}
    trait = {a: {"breach": {c: [] for c in COMPS}, "stromal_survival": {c: [] for c in COMPS}}
             for a in ARMS}
    for seed in args.seeds:
        for a, (eb, sh) in ARMS.items():
            t = grow(seed, eb, sh)
            ty = t.cell_data["cell_type"]["cell_id"].values
            isc = _is_cancer_mask(ty)
            freq[a].append((t.cell_data["cell_snv"].values[isc] > 0).mean(axis=0))
            cc = _compartment(t)[t.cell_data["cell_deme"]["deme_id"].values]
            br = t.cell_data["cell_evo"]["breach"].values
            ss = t.cell_data["cell_evo"]["stromal_survival"].values
            for c in COMPS:
                m = isc & (cc == c)
                trait[a]["breach"][c].append(float(br[m].mean()) if m.sum() else np.nan)
                trait[a]["stromal_survival"][c].append(float(ss[m].mean()) if m.sum() else np.nan)
    freq = {a: np.mean(v, axis=0) for a, v in freq.items()}

    def tmean(arm, ax, comp):
        return float(np.nanmean(trait[arm][ax][comp]))

    def tstd(arm, ax, comp):
        return float(np.nanstd(trait[arm][ax][comp]))

    # one BOTH tumour WITH the program layer (growth identical to 'both' above), for the spatial maps
    # and the confound.
    tb = grow(args.seeds[0], EPI_BARRIER, STRO_HAZARD, expression=EXPR)
    cd = tb.cell_data
    types = cd["cell_type"]["cell_id"].values
    isc = _is_cancer_mask(types)
    comp = _compartment(tb)
    demes = cd["cell_deme"]["deme_id"].values
    cell_epi = tb.microenv_truth["epithelial"][demes]
    crd = cd["cell_crd"].values
    br_cell = cd["cell_evo"]["breach"].values
    ss_cell = cd["cell_evo"]["stromal_survival"].values

    # ---------------- headline numbers ----------------
    print(f"tumour (BOTH): {tb.get_cancer_size()} cancer / {tb.get_tumor_size()} total on {GRID}^2, "
          f"seeds={args.seeds}")

    # each trait is selected in its OWN compartment: breach at the epithelial ring, stromal_survival
    # in the stroma. Report the trait there, barriers ON (BOTH) vs the OFF control.
    br_ring_on, br_ring_off = tmean("both", "breach", "ring"), tmean("off", "breach", "ring")
    ss_str_on, ss_str_off = tmean("both", "stromal_survival", "stroma"), tmean("off", "stromal_survival", "stroma")
    print("\nA. each trait is positively selected in ITS compartment (mean over seeds):")
    print(f"   breach @ epithelial ring: BOTH {br_ring_on:.3f}  vs  OFF {br_ring_off:.3f}")
    print(f"   stromal_survival @ stroma: BOTH {ss_str_on:.3f}  vs  OFF {ss_str_off:.3f}")
    print("   sequential-invasion gradient (BOTH, lumen->ring->stroma):")
    print("     breach          " + " ".join(f"{tmean('both','breach',c):.2f}" for c in COMPS))
    print("     stromal_survival " + " ".join(f"{tmean('both','stromal_survival',c):.2f}" for c in COMPS))

    gd = tb.selection.get_gene_data()
    breach_g = gd["breach_types"].values.ravel().astype(bool)
    ss_g = gd["stromal_survival_types"].values.ravel().astype(bool)
    sel_epi = freq["epi"] - freq["off"]     # frequency lift when the EPITHELIAL barrier is dialled on
    sel_stro = freq["stro"] - freq["off"]   # ... when the STROMAL hazard is dialled on
    auroc_breach = _auroc(breach_g, sel_epi)
    auroc_ss = _auroc(ss_g, sel_stro)
    print("\nB. selection recovery — dial one barrier, ask which genes it selects (AUROC vs truth):")
    print(f"   breach genes from the EPITHELIAL-arm frequency lift: AUROC = {auroc_breach:.2f}")
    print(f"   stromal genes from the STROMAL-arm frequency lift:   AUROC = {auroc_ss:.2f}")

    # ---------------- C: the confound ----------------
    emt_k = list(tb.programs.dictionary.program_names).index("emt")
    emt = cd["cell_program"].values[:, emt_k]
    niche_emt = tb.programs.niche_drive(
        {"epithelial": tb.microenv_truth["epithelial"],
         "stromal": tb.microenv_truth["stromal"]})[:, emt_k][demes]
    gen_emt = np.zeros(len(types))
    for i, gid in enumerate(types):
        rep = tb.genotypes[gid]
        if rep.type == "cancer":
            gen_emt[i] = tb.programs.clone_drive(
                rep.evolutionary_parameters, rep.baseline_rates, rep.get_snvs())[emt_k]
    # env-responsive phenotype CONTROLLING for genotype: subtract each genotype's mean emt, then
    # correlate the residual with the epithelial niche. A positive residual correlation means the SAME
    # genotype is more invasive at the epithelial front — pure niche, no genetics.
    emt_c, epi_c, gid_c = emt[isc], cell_epi[isc], types[isc]
    resid = emt_c.copy().astype(float)
    for g in np.unique(gid_c):
        m = gid_c == g
        resid[m] = emt_c[m] - emt_c[m].mean()
    r_partial = np.corrcoef(resid, epi_c)[0, 1] if np.ptp(epi_c) > 0 else np.nan
    var_gen, var_niche = float(np.var(gen_emt[isc])), float(np.var(niche_emt[isc]))
    frac_niche = var_niche / (var_gen + var_niche) if (var_gen + var_niche) > 0 else np.nan
    # a naive invasive-genotype caller thresholds emt; among genotypes seen in both niche halves, how
    # often is the SAME genotype called differently by niche?
    thr = np.median(emt_c)
    hi = epi_c > np.median(epi_c)
    flipped = 0; spanning = 0
    for g in np.unique(gid_c):
        m = gid_c == g
        if (m & hi).sum() and (m & ~hi).sum():
            spanning += 1
            if (emt_c[m & hi] > thr).mean() != (emt_c[m & ~hi] > thr).mean():
                flipped += 1
    print("\nC. genetic-vs-niche confound (invasive/emt program):")
    print(f"   genotype-controlled corr(emt residual, epithelial fraction) = {r_partial:.2f} "
          f"(SAME genotype more invasive at the front)")
    print(f"   emt-drive variance: genetic {var_gen:.3f}  niche {var_niche:.3f}  "
          f"(niche share {frac_niche*100:.0f}%) — iscc knows both")
    if spanning:
        print(f"   naive caller: {flipped}/{spanning} genotypes spanning both niche halves get an "
              f"INCONSISTENT invasive call by location")

    # =========================================================================
    # figure
    # =========================================================================
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from iscc.tumor.models.glandular import bresenham_circumference

    fig, ax = plt.subplots(2, 3, figsize=(16.5, 9.6))
    c0 = tb.grid_size // 2
    ring = np.array([(r, cc) for (r, cc) in bresenham_circumference(c0, c0, STRUCTURE_RADIUS)
                     if 0 <= r < tb.grid_size and 0 <= cc < tb.grid_size])

    # A1/A2 spatial trait maps
    for a, val, name, cmap in ((ax[0, 0], br_cell, "breach", "viridis"),
                               (ax[0, 1], ss_cell, "stromal_survival", "magma")):
        sc = a.scatter(crd[isc, 1], crd[isc, 0], c=val[isc], cmap=cmap, s=10, vmin=0, vmax=1)
        a.scatter(ring[:, 1], ring[:, 0], facecolors="none", edgecolors="tab:green", s=16,
                  linewidths=0.6, label="epithelial ring")
        a.set_title(f"A. {name} over cancer cells\n(the trait sweeps at its own front)")
        a.invert_yaxis(); a.set_aspect("equal"); a.legend(loc="upper right", fontsize=7)
        fig.colorbar(sc, ax=a, fraction=0.046)

    # A3 sequential invasion: mean trait by compartment (BOTH), each trait selected in its own front
    a = ax[0, 2]
    xp = np.arange(len(COMPS)); w = 0.2
    a.bar(xp - 1.5 * w, [tmean("both", "breach", c) for c in COMPS], w,
          yerr=[tstd("both", "breach", c) for c in COMPS], capsize=3, label="breach ON", color="#2c7fb8")
    a.bar(xp - 0.5 * w, [tmean("off", "breach", c) for c in COMPS], w, label="breach OFF", color="#a6bddb")
    a.bar(xp + 0.5 * w, [tmean("both", "stromal_survival", c) for c in COMPS], w,
          yerr=[tstd("both", "stromal_survival", c) for c in COMPS], capsize=3,
          label="strom_surv ON", color="#c51b8a")
    a.bar(xp + 1.5 * w, [tmean("off", "stromal_survival", c) for c in COMPS], w,
          label="strom_surv OFF", color="#fbb4b9")
    a.set_xticks(xp); a.set_xticklabels(COMPS); a.set_ylim(0, 1)
    a.set_ylabel("mean heritable trait")
    a.set_title("A. sequential invasion — each trait\nselected in its compartment (ON) vs flat (OFF)")
    a.legend(fontsize=7, ncol=2)

    # B selection-recovery AUROC
    a = ax[1, 0]
    bars = a.bar(["breach\n(epithelial arm)", "stromal_survival\n(stromal arm)"],
                 [auroc_breach, auroc_ss], color=["#2c7fb8", "#c51b8a"])
    a.axhline(0.5, ls="--", color="k", lw=1, label="chance")
    a.set_ylim(0, 1); a.set_ylabel("AUROC (gene role + compartment)")
    a.set_title("B. selection recovery from scDNA\ndial one barrier -> which genes it selects")
    for b, v in zip(bars, [auroc_breach, auroc_ss]):
        a.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)
    a.legend(fontsize=8)

    # C1 env-responsive emt controlling for genotype
    a = ax[1, 1]
    a.scatter(epi_c, resid, s=7, alpha=0.2, color="#c51b8a")
    if np.ptp(epi_c) > 0:
        z = np.polyfit(epi_c, resid, 1)
        xs = np.linspace(epi_c.min(), epi_c.max(), 20)
        a.plot(xs, np.polyval(z, xs), "k--", lw=1.5, label=f"slope>0 (r={r_partial:.2f})")
    a.axhline(0, color="grey", lw=0.8)
    a.set_xlabel("epithelial fraction of cell's niche")
    a.set_ylabel("emt residual (genotype removed)")
    a.set_title("C. env-responsive phenotype\nSAME genotype, more invasive at the front")
    a.legend(fontsize=8)

    # C2 variance decomposition
    a = ax[1, 2]
    a.bar(["genetic\n(clone)", "niche\n(compartment)"], [var_gen, var_niche],
          color=["#7fbf7b", "#af8dc3"])
    a.set_ylabel("variance of emt drive across cancer cells")
    a.set_title(f"C. genetic vs niche contribution\niscc knows both (niche share {frac_niche*100:.0f}%)")

    fig.suptitle("Compartment-dependent selection (v1): sequential invasion, selection recovery, "
                 "and the genetic-vs-niche expression confound", fontsize=13, y=1.005)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print("\nfigure ->", args.out)


if __name__ == "__main__":
    main()
