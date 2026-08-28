"""Cohort-level ground truth: shared-vs-private states, and the need for personalized medicine (iscc).

iscc runs MANY patients (tumours) separately over ONE shared, config-determined driver landscape (the
seed-decoupling fix, DESIGN_cohort.md §1), pools them into sequencing batches flexibly, and surfaces
cohort-level ground truth no simulator provides and no real cohort can ever give: recurrent-vs-private
drivers, per-cell patient-of-origin, shared-vs-private cell states, and — coupled to the treatment
module — per-patient subgroup + true therapy response.

Figure (the paper repo's figures/validation_cohort.png), the strongest first-pass benchmarks:
  A. RECURRENCE IS ONLY WELL-POSED UNDER A SHARED LANDSCAPE. Driver-gene identities are IDENTICAL
     across patients (mean pairwise Jaccard = 1.0) with the fix, versus ~0 without it (per-seed
     layouts) — so cross-patient recurrence / driver detection is meaningful only because of the fix.
     Drivers still sit at higher recurrence than private passengers.
  B. PERSONALIZED MEDICINE (resistance EMERGES, not seeded). Two molecular subtypes over the SAME
     landscape differ only in a resistance EFFECT; during an untreated burn-in, resistance mutations
     arise and drift as neutral standing variation, and adjuvant therapy SELECTS them in the resistant
     subtype (which relapses) while eradicating the sensitive subtype — a ground-truth differential
     response that is a genuine evolutionary outcome. Recovering the responder from molecular data is
     honestly non-predictive at BASELINE (the standing resistance mutations are present in both
     subtypes; only their functional effect differs) but is revealed by the therapy-selected emergent
     signature and the response itself.
  C. MULTI-PATIENT INTEGRATION. 1:1 pooling -> one batch per patient. Naive embedding OVER-SEPARATES
     the SHARED cell states by patient (low iLISI); a correction restores cross-patient mixing while
     preserving the biology (cell-type ARI) — scored against iscc's shared-vs-private ground truth
     (real Harmony used when the iscc-harmony env is present, else a self-contained baseline).
  D. DEMULTIPLEXING, per modality. DNA assays (WGS/WES/scDNA) genotype germline SNPs reliably, so N:1
     pooled DNA is demuxed GENETICALLY (souporcell/vireo/demuxlet) — every cell, cancer AND normal,
     assigned by its germline genotype. Droplet scRNA cannot call germline SNPs reliably (only sparse
     expressed loci), so pooled RNA is demuxed by CELL HASHING (a per-sample HTO/MULTI-seq barcode):
     singlet assignment is near-perfect and the real challenge is DOUBLET detection (hashing is used
     with cell super-loading) — shown as naive accuracy falling with the doublet rate while the
     doublets stay detectable.

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
from _paths import figure_path

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=figure_path("validation_cohort.png"))
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

    # ---- B. personalized medicine (resistance EMERGES) --------------------------------
    print("\n[B] personalized medicine / stratification (emergent resistance) ...")
    pm = cc.pm_cohort(n_patients=args.pm_patients)
    pmres = cc.pm_analysis(pm)
    for sg in ("sensitive", "resistant"):
        m = pmres["subgroup"] == sg
        print(f"    {sg:9s}: baseline~{pmres['baseline'][m].mean():4.0f}  treated~{pmres['treated'][m].mean():4.0f}")
    print(f"    recovery AUC — baseline (non-predictive)={pmres['baseline_auc']:.2f}  "
          f"emergent relapse signature={pmres['relapse_auc']:.2f}  response readout={pmres['response_auc']:.2f}")

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

    # ---- D. demultiplexing (DNA: genetic / RNA: cell hashing) --------------------------
    print("\n[D] demultiplexing — DNA (germline SNPs) vs RNA (cell hashing) ...")
    dc = cc.demux_cohort(n_patients=args.demux_patients)
    dna = cc.dna_demux_analysis(dc)
    rna = cc.rna_hashing_demux_analysis(dc)
    print(f"    DNA genetic demux: accuracy={dna['accuracy']:.3f} "
          f"(cancer={dna['cancer_accuracy']:.3f} normal={dna['normal_accuracy']:.3f}, chance={dna['chance']:.3f}, "
          f"{dna['n_cancer']} cancer + {dna['n_normal']} normal)")
    for r in rna["sweep"]:
        print(f"    RNA cell hashing @ doublet={r['doublet_rate']:.0%}: singlet acc={r['singlet_accuracy']:.3f}  "
              f"naive acc={r['naive_accuracy']:.3f}  doublet-detect AUC={r['doublet_detection_auc']:.3f}")

    _figure(args.out, ra, pmres, ia, dna, rna)
    print(f"\nsaved figure -> {args.out}")


def _figure(out, ra, pm, ia, dna, rna):
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

    # B. personalized medicine (emergent resistance) ------------------------------------
    b = ax[0, 1]
    order = np.argsort(pm["subgroup"])
    colors = {"sensitive": "#2980b9", "resistant": "#c0392b"}
    xs = np.arange(len(order))
    b.bar(xs, pm["treated"][order], color=[colors[s] for s in pm["subgroup"][order]])
    b.plot(xs, pm["baseline"][order], "k_", ms=12, mew=2, label="baseline (untreated)")
    b.set_xlabel("patient (grouped by subtype)")
    b.set_ylabel("cancer cells after therapy")
    b.set_title("B. One therapy, two fates: sensitive eradicated,\n"
                "resistant relapse from EMERGENT resistance — known truth")
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[s]) for s in ("sensitive", "resistant")]
    b.legend(handles + [plt.Line2D([0], [0], color="k", marker="_", ls="", mew=2)],
             ["sensitive", "resistant", "baseline"], fontsize=8, loc="upper center")
    # inset: recovery AUC — baseline is non-predictive; therapy REVEALS the emergent resistance
    bi = b.inset_axes([0.60, 0.50, 0.36, 0.42])
    labs = ["baseline", "relapse", "response"]
    vals = [pm["baseline_auc"], pm["relapse_auc"], pm["response_auc"]]
    bi.bar(range(3), vals, color=["#7f8c8d", "#e67e22", "#27ae60"])
    bi.axhline(0.5, ls=":", color="k", lw=0.8)
    bi.set_xticks(range(3)); bi.set_xticklabels(labs, fontsize=7, rotation=20)
    bi.set_ylim(0, 1.05); bi.set_ylabel("recovery AUC", fontsize=7)
    bi.set_title("recover responders", fontsize=8)
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

    # D. demultiplexing — DNA (genetic) vs RNA (cell hashing) ---------------------------
    d = ax[1, 1]
    rna_dbl = [r["doublet_rate"] for r in rna["sweep"]]
    rna_singlet = [r["singlet_accuracy"] for r in rna["sweep"]]
    rna_naive = [r["naive_accuracy"] for r in rna["sweep"]]
    rna_auc = [r["doublet_detection_auc"] for r in rna["sweep"]]
    # bar: the two modalities' headline accuracy (DNA all-cell genetic; RNA singlet hashing)
    d.bar([0, 1, 2], [dna["chance"], dna["accuracy"], rna_singlet[0]],
          color=["#7f8c8d", "#8e44ad", "#16a085"], width=0.6)
    for x, v, lab in zip([0, 1, 2], [dna["chance"], dna["accuracy"], rna_singlet[0]],
                         ["chance", "DNA genetic\n(germline SNP)", "RNA hashing\n(singlet)"]):
        d.text(x, v + 0.02, f"{v:.2f}", ha="center", fontweight="bold", fontsize=9)
    d.set_xticks([0, 1, 2]); d.set_xticklabels(["chance", "DNA genetic\n(germline SNP)", "RNA hashing\n(singlet)"], fontsize=8)
    d.set_ylabel("patient-of-origin accuracy"); d.set_ylim(0, 1.12)
    d.set_title("D. Demultiplexing: DNA uses germline SNPs,\nRNA uses cell hashing (per-modality method)")
    # inset: the RNA-hashing failure mode is DOUBLETS — naive accuracy falls, but they are detectable
    di = d.inset_axes([0.60, 0.16, 0.36, 0.42])
    di.plot(rna_dbl, rna_naive, "o-", color="#c0392b", ms=4, label="naive acc")
    di.plot(rna_dbl, rna_auc, "s--", color="#2980b9", ms=4, label="doublet AUC")
    di.set_xlabel("doublet rate", fontsize=7); di.set_ylim(0.5, 1.02)
    di.tick_params(labelsize=6); di.legend(fontsize=6, loc="lower left")
    di.set_title("hashing: doublets", fontsize=7)

    fig.suptitle("iscc provides cohort-level ground truth: shared-vs-private states, and the need for "
                 "personalized medicine", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
