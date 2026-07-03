"""Cohort-level ground truth: shared-vs-private states, and the need for personalized medicine (iscc).

iscc runs MANY patients (tumours) separately over ONE shared, config-determined driver landscape (the
seed-decoupling fix, DESIGN_cohort.md §1), pools them into sequencing batches flexibly, and surfaces
cohort-level ground truth no simulator provides and no real cohort can ever give: recurrent-vs-private
drivers, per-cell patient-of-origin, shared-vs-private cell states, and — coupled to the treatment
module — per-patient subgroup + true therapy response.

Figure (manuscript/figures/validation_cohort.png), the strongest first-pass benchmarks:
  A. RECURRENCE IS ONLY WELL-POSED UNDER A SHARED LANDSCAPE. Driver-gene identities are IDENTICAL
     across patients (mean pairwise Jaccard = 1.0) with the fix, versus ~0 without it (per-seed
     layouts) — so cross-patient recurrence / driver detection is meaningful only because of the fix.
     Drivers still sit at higher recurrence than private passengers.
  B. PERSONALIZED MEDICINE. Two molecular subtypes over the SAME landscape: 'resistant' carries a
     pre-existing (SUBCLONAL) resistance subclone, 'sensitive' does not. One therapy eradicates the
     sensitive tumours and the resistant ones relapse — a ground-truth differential response — and the
     resistant subclone (subclonal, single-cell-only) recovers the responsive subtype (a known-answer
     biomarker).
  C. MULTI-PATIENT INTEGRATION. 1:1 pooling -> one batch per patient. Naive embedding OVER-SEPARATES
     the SHARED cell states by patient (low iLISI); a correction restores cross-patient mixing while
     preserving the biology (cell-type ARI) — scored against iscc's shared-vs-private ground truth
     (real Harmony used when the iscc-harmony env is present, else a self-contained baseline).
  D. DEMULTIPLEXING. N:1 pooling -> assign pooled cells back to patient-of-origin from their private
     germline variants (souporcell/vireo-style): accuracy vs the true patient label, far above chance.

Self-contained (numpy/scipy/sklearn); external integration/demux tools are optional and isolated in
their own conda envs. Run:  python -u validation/validate_cohort.py
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cohort_common as cc

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "manuscript/figures/validation_cohort.png"))
    ap.add_argument("--recur-patients", type=int, default=20)
    ap.add_argument("--pm-patients", type=int, default=14)
    ap.add_argument("--int-patients", type=int, default=8)
    ap.add_argument("--demux-patients", type=int, default=8)
    args = ap.parse_args()

    print("=" * 80)
    print("iscc cohort-level ground truth — shared-vs-private states & personalized medicine")
    print("=" * 80)

    # ---- A. recurrence enablement -----------------------------------------------------
    print("\n[A] recurrence / shared-landscape enablement ...")
    rc = cc.recurrence_cohort(n_patients=args.recur_patients)
    ra = cc.recurrence_analysis(rc)
    print(f"    driver-set Jaccard across patients: shared(fix)={ra['jaccard_shared']:.3f}  "
          f"unshared(layout=seed)={ra['jaccard_unshared']:.3f}")
    print(f"    recurrence: drivers={ra['driver_recurrence'].mean():.3f} vs "
          f"passengers={ra['passenger_recurrence'].mean():.3f} (MWU p={ra['mwu_p']:.3g})")

    # ---- B. personalized medicine -----------------------------------------------------
    print("\n[B] personalized medicine / stratification ...")
    pm = cc.pm_cohort(n_patients=args.pm_patients)
    dr = cc.differential_response(pm)
    st = cc.stratification(pm)
    for sg in ("sensitive", "resistant"):
        m = dr["subgroup"] == sg
        print(f"    {sg:9s}: baseline~{dr['baseline'][m].mean():4.0f}  treated~{dr['treated'][m].mean():4.0f}")
    print(f"    stratification AUC — bulk={st['bulk_auc']:.2f}  single-cell subclone={st['singlecell_auc']:.2f}")

    # ---- C. integration ---------------------------------------------------------------
    print("\n[C] multi-patient integration (shared-vs-private) ...")
    ic = cc.integration_cohort(n_patients=args.int_patients)
    ia = cc.integration_analysis(ic)
    print(f"    shared-cell iLISI: naive={ia['naive'][0]:.2f}  corrected={ia['corrected'][0]:.2f}  "
          f"ideal={ia['ideal_ilisi']}  (cell-type ARI {ia['naive'][1]:.2f}->{ia['corrected'][1]:.2f})")
    if "harmony" in ia:
        print(f"    real Harmony iLISI={ia['harmony'][0]:.2f} ARI={ia['harmony'][1]:.2f}")
    else:
        print(f"    (iscc-harmony env absent -> self-contained '{ia['method']}' correction only)")

    # ---- D. demultiplexing ------------------------------------------------------------
    print("\n[D] demultiplexing (N:1 pooling, patient-of-origin) ...")
    dc = cc.demux_cohort(n_patients=args.demux_patients)
    da = cc.demux_analysis(dc)
    print(f"    patient-of-origin accuracy={da['accuracy']:.3f} (chance={da['chance']:.3f}, "
          f"{da['n_cells']} pooled cells)")

    _figure(args.out, ra, dr, st, ia, da)
    print(f"\nsaved figure -> {args.out}")


def _figure(out, ra, dr, st, ia, da):
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))

    # A. recurrence enablement -----------------------------------------------------------
    a = ax[0, 0]
    a.bar([0, 1], [ra["jaccard_unshared"], ra["jaccard_shared"]],
          color=["#c0392b", "#27ae60"], width=0.6)
    a.set_xticks([0, 1]); a.set_xticklabels(["per-seed layout\n(prior sims)", "shared landscape\n(iscc fix)"])
    a.set_ylabel("mean pairwise driver-set Jaccard")
    a.set_ylim(0, 1.08)
    for x, v in zip([0, 1], [ra["jaccard_unshared"], ra["jaccard_shared"]]):
        a.text(x, v + 0.02, f"{v:.2f}", ha="center", fontweight="bold")
    a.set_title("A. Recurrence is well-posed only under a\nshared landscape (driver identities match)")
    # inset: driver vs passenger recurrence
    ai = a.inset_axes([0.58, 0.5, 0.38, 0.42])
    ai.boxplot([ra["passenger_recurrence"], ra["driver_recurrence"]], labels=["pass.", "driv."],
               showfliers=False, widths=0.6)
    ai.set_title(f"recurrence\n(p={ra['mwu_p']:.2g})", fontsize=8)
    ai.tick_params(labelsize=7)

    # B. personalized medicine -----------------------------------------------------------
    b = ax[0, 1]
    order = np.argsort(dr["subgroup"])
    colors = {"sensitive": "#2980b9", "resistant": "#c0392b"}
    xs = np.arange(len(order))
    b.bar(xs, dr["treated"][order], color=[colors[s] for s in dr["subgroup"][order]])
    b.plot(xs, dr["baseline"][order], "k_", ms=12, mew=2, label="baseline (untreated)")
    b.set_xlabel("patient (grouped by subtype)")
    b.set_ylabel("cancer cells after therapy")
    b.set_title("B. One therapy, two fates: sensitive eradicated,\n"
                "resistant relapse (subclonal resistance) — known truth")
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[s]) for s in ("sensitive", "resistant")]
    b.legend(handles + [plt.Line2D([0], [0], color="k", marker="_", ls="", mew=2)],
             ["sensitive", "resistant", "baseline"], fontsize=8, loc="upper center")
    bi = b.inset_axes([0.60, 0.52, 0.36, 0.40])
    y = st["y"]
    bi.scatter(np.where(y == 0)[0] * 0 + 0, st["singlecell"][y == 0], c="#2980b9", s=18)
    bi.scatter(np.where(y == 1)[0] * 0 + 1, st["singlecell"][y == 1], c="#c0392b", s=18)
    bi.set_xticks([0, 1]); bi.set_xticklabels(["sens", "res"], fontsize=7)
    bi.set_title(f"single-cell subclone\nAUC={st['singlecell_auc']:.2f}", fontsize=8)
    bi.tick_params(labelsize=7)

    # C. integration ---------------------------------------------------------------------
    c = ax[1, 0]
    labels, ilisi, ari = ["naive"], [ia["naive"][0]], [ia["naive"][1]]
    labels.append(ia.get("method", "corrected")); ilisi.append(ia["corrected"][0]); ari.append(ia["corrected"][1])
    if "harmony" in ia:
        labels.append("Harmony"); ilisi.append(ia["harmony"][0]); ari.append(ia["harmony"][1])
    xs = np.arange(len(labels))
    c.bar(xs, ilisi, color=["#c0392b"] + ["#27ae60"] * (len(labels) - 1), width=0.6)
    c.axhline(ia["ideal_ilisi"], ls="--", color="gray", label=f"ideal mixing ({ia['ideal_ilisi']})")
    c.axhline(1.0, ls=":", color="k", label="fully separated (1)")
    c.set_xticks(xs); c.set_xticklabels(labels)
    c.set_ylabel("shared-state iLISI (patient mixing)")
    for x, v, r in zip(xs, ilisi, ari):
        c.text(x, v + 0.05, f"iLISI {v:.1f}\nARI {r:.2f}", ha="center", fontsize=8)
    c.legend(fontsize=8, loc="upper left")
    c.set_title("C. Integration must MIX shared states across\npatients without erasing private biology")

    # D. demultiplexing ------------------------------------------------------------------
    d = ax[1, 1]
    d.bar([0, 1], [da["chance"], da["accuracy"]], color=["#7f8c8d", "#27ae60"], width=0.6)
    d.set_xticks([0, 1]); d.set_xticklabels(["chance", "iscc private\nvariants"])
    d.set_ylabel("patient-of-origin accuracy")
    d.set_ylim(0, 1.08)
    for x, v in zip([0, 1], [da["chance"], da["accuracy"]]):
        d.text(x, v + 0.02, f"{v:.2f}", ha="center", fontweight="bold")
    d.set_title(f"D. Demultiplexing (N:1 pool of {da['n_cells']} cells):\n"
                "assign pooled cells to patient-of-origin")

    fig.suptitle("iscc provides cohort-level ground truth: shared-vs-private states, and the need for "
                 "personalized medicine", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
