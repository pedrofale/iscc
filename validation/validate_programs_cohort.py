"""Benchmark: across a COHORT, can scDEF separate shared programs from patient-specific biology —
and what does batch confounding do to that? (R13, `DESIGN_expression.md` §4.3)

The setup iscc uniquely provides: a cohort shares its landscape (so the program dictionary is
IDENTICAL in every patient — the layout stream) while each patient's evolution is private (so each
carries its OWN CNA landscape). Truth therefore has three tiers, and we know which is which:

  * **programs**    — the K dictionary programs. SCATTERED genes, SHARED by every patient.
  * **patient CNA** — each patient's private copy-number landscape. CONTIGUOUS genes, ONE patient.
                      This is REAL BIOLOGY that simply is not a program.
  * **batch**       — the F3 per-gene technical factor. SCATTERED genes, one batch.

THE POINT — the same statistic, opposite verdicts. In the single-tumour benchmark
(`validate_programs.py`, §4.2) a positionally-clustered factor is a NUISANCE: a copy-number event
wearing a program's label. Here it is SIGNAL THAT MUST SURVIVE: it is the patient-specific biology a
batch-correction step must not delete. Which one it is depends on the question, not on the data.

Two batch designs:
  * **A — pooled / demultiplexed**: all patients in ONE batch. The technical effect is common to all,
    so anything patient-specific is unambiguously biology.
  * **B — one batch per patient**: batch is PERFECTLY confounded with patient. Nothing in the data
    distinguishes "this patient's CNA raised these genes" from "this lane ran hot" — same support.

HYPOTHESIS UNDER TEST (a negative is reportable): in design B, correcting the batch necessarily
removes the genuine patient-specific CNA biology, because they occupy the same subspace — so
`patient_mixing` and `cna_retention` trade off. Design A breaks the confound and keeps both. If
correction preserves `cna_retention` in B, check the batch effect size before believing it.

Usage:
    python validate_programs_cohort.py            # full run
    python validate_programs_cohort.py --quick    # fewer patients/cells/epochs
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import programs_common as pc
from cohort_common import inverse_simpson_lisi

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# A cohort whose patients carry DISTINCTIVE arm-level CNA landscapes (the real-tumour situation: one
# patient's 8q amplification, another's 17p loss), which is what makes "patient-specific biology"
# a real thing to preserve rather than noise. Tuned so per-patient segment CN differs by ~2 copies.
GENOME = {"n_segments": 8, "segment_size": 50}                  # 400 genes over 8 "chromosomes"
SELECTION = {"prop_driver": 0.15, "driver_effects": 1.4, "prop_dispersal": 0.0,
             "prop_immune_resistance": 0.0, "prop_treatment_resistance": 0.0}
CANCER = {"division_rate": 0.6, "death_rate": 0.03, "max_birth_rate": 0.98,
          "mutation_rate": 1.5, "dispersal_rate": 0.3, "cnv_prob": 0.9, "snv_prob": 0.1}
DEME = {"carrying_capacity": 12, "initial_cancer_cells": 8}
SPATIAL = {"grid_size": 20, "structure_radius": 0}
N_PROGRAMS = 6
EXPRESSION = {
    "program_params": {"n_programs": N_PROGRAMS, "n_genes_per_program": 20, "program_overlap": 0.1,
                       "loading_strength": {"mean": 1.0, "sd": 0.3}, "loading_sparsity": 1.0,
                       "program_genomic_scatter": 1.0},
    "activity_params": {"n_active_programs_per_cell": 3, "activity_dist": "lognormal",
                        "activity_mean": 1.0, "activity_sd": 0.5, "activity_noise": 0.2},
    "coupling_params": {"phenotype_program_strength": 0.5},
    "dosage_params": {"dosage_sensitivity_mean": 0.7, "dosage_sensitivity_sd": 0.25,
                      "dosage_saturation": 8, "allele_specific": False},
}


def build_cohort(n_patients=6, steps=15000):
    from iscc.cohort import Cohort
    return Cohort(patient_seeds=list(range(1, n_patients + 1)), genome_params=GENOME,
                  selection_params=SELECTION, cancer_cell_params=CANCER, deme_params=DEME,
                  spatial_params=SPATIAL, grow_steps=steps, expression_params=EXPRESSION).run()


def patient_cn_deviation(cohort):
    """Per-patient per-gene log2 CN deviation from the cohort mean — the patient-specific BIOLOGY
    a correction step must not delete. Returns (n_patients, n_genes)."""
    profiles = []
    for pr in cohort.patients:
        t = pr.tumor
        cd = t.cell_data
        is_cancer = np.array([t.genotypes[g].type == "cancer"
                              for g in cd["cell_type"]["cell_id"].values])
        profiles.append(cd["cell_cnv"].values[is_cancer].mean(0))
    P = np.log2(np.clip(np.array(profiles), 0.1, None))
    return P - P.mean(0, keepdims=True)


def emit(cohort, design, n_cells_per_patient, sigma_batch, depth_batch_sigma, seed=0):
    """Emit scRNA under a batch design. Returns (AnnData, patient labels, batch labels)."""
    from iscc.cohort.batch import run_cohort_batches, concat_cohort_batches
    if design == "pooled":     # A: every patient in ONE batch (demultiplexed)
        kw = dict(mapping="multiplex", capacity=cohort.n_patients)
    else:                      # B: one batch per patient -> batch == patient
        kw = dict(mapping="one_to_one")
    assays, batches, asg = run_cohort_batches(
        cohort, n_cells_per_patient=n_cells_per_patient, sigma_batch=sigma_batch,
        depth_batch_sigma=depth_batch_sigma, **kw)
    comb = concat_cohort_batches(assays)
    patient = comb.obs["patient"].astype(int).values
    batch = np.asarray([asg[p] for p in patient])
    return comb, patient, batch


def batch_center(X_log, batch):
    """The self-contained correction: remove each batch's per-gene mean in log space (the same
    correction `cohort_common.integration_analysis` uses). In design B, batch == patient, so this
    removes the per-PATIENT mean — which is exactly the point being tested."""
    Xc = X_log.copy()
    grand = X_log.mean(0)
    for b in np.unique(batch):
        m = batch == b
        Xc[m] = Xc[m] - Xc[m].mean(0) + grand
    return Xc


def score(inferred, true_loading, cn_dev, patient, segment_sizes, top_n=20, spurious_cosine=0.3):
    """Shared-program recovery, patient-CNA retention, patient mixing, and the positional split."""
    W, A = np.asarray(inferred["loadings"]), np.asarray(inferred["activities"])

    # 1. shared programs: Hungarian-match the true (shared) dictionary
    K_true, K_inf = true_loading.shape[0], W.shape[0]
    C = np.zeros((K_true, K_inf))
    for i in range(K_true):
        for j in range(K_inf):
            C[i, j] = pc._cosine(true_loading[i], W[j])
    from scipy.optimize import linear_sum_assignment
    rows, cols = linear_sum_assignment(-C)
    shared_recovery = float(np.mean([C[i, j] for i, j in zip(rows, cols)]))

    # 2. patient CNA retention: does ANY factor carry this patient's CN deviation?
    retention = []
    for p in range(cn_dev.shape[0]):
        best = 0.0
        for j in range(K_inf):
            if W[j].std() == 0:
                continue
            r = abs(float(np.corrcoef(cn_dev[p], W[j])[0, 1]))
            best = max(best, 0.0 if np.isnan(r) else r)
        retention.append(best)
    cna_retention = float(np.mean(retention))

    # 3. patient mixing in factor-activity space (high = patients merged)
    mixing = float(inverse_simpson_lisi(A, patient, k=min(30, max(5, A.shape[0] // 10))))

    # 4. the positional split: are patient-specific factors contiguous (CNA=biology) or scattered?
    #    Specificity = max_p mean_activity(p) / sum_p mean_activity(p): 1/N when a factor is used
    #    equally by every patient (SHARED), -> 1 when it belongs to one patient. Report the whole
    #    distribution and the max, not just a thresholded count: a hard cut at 2/N returned zero
    #    factors in every arm, which reads as "no patient-specific factors found" when the real
    #    statement is "no factor exceeded an arbitrary cut" — those are different claims.
    n_pat = len(np.unique(patient))
    specs, spec_clust, shared_clust = [], [], []
    for j in range(K_inf):
        aj = A[:, j]
        if aj.sum() <= 0:
            continue
        per_pat = np.array([aj[patient == p].mean() for p in np.unique(patient)])
        specificity = per_pat.max() / max(per_pat.sum(), 1e-9)
        specs.append(specificity)
        cl = pc.positional_clustering(W[j], segment_sizes, top_n=top_n)
        (spec_clust if specificity > 1.5 / n_pat else shared_clust).append(cl)

    return dict(shared_recovery=shared_recovery, cna_retention=cna_retention,
                patient_mixing=mixing, n_inferred=K_inf,
                max_specificity=float(np.max(specs)) if specs else float("nan"),
                mean_specificity=float(np.mean(specs)) if specs else float("nan"),
                shared_baseline=1.0 / n_pat,
                clust_patient_specific=float(np.mean(spec_clust)) if spec_clust else float("nan"),
                clust_shared=float(np.mean(shared_clust)) if shared_clust else float("nan"),
                n_patient_specific=len(spec_clust))


def data_level_confound(cohort, design, args, cn_dev):
    """The TOOL-INDEPENDENT measurement: how much of the apparent patient-specific expression
    deviation is actually the patient's CNA biology?

    corr(patient CN deviation, patient expression deviation) across genes. In the pooled design the
    technical effect is common to all patients, so this is the honest coupling; when each patient is
    its own batch, the batch factor adds patient-specific expression that is NOT CNA-driven, which
    both inflates the deviation and dilutes the correlation. This is the confound itself, measured
    before any tool sees the data — so it cannot be blamed on (or fixed by) the tool.
    """
    comb, patient, batch = emit(cohort, design, args.cells, args.sigma_batch, args.depth_sigma)
    X = np.log1p(np.asarray(comb.X, dtype=float))
    ex = np.array([X[patient == p].mean(0) for p in np.unique(patient)])
    ex_dev = ex - ex.mean(0, keepdims=True)
    r = [float(np.corrcoef(cn_dev[p], ex_dev[p])[0, 1]) for p in range(cn_dev.shape[0])]
    return float(np.mean(r)), float(ex_dev.std())


def run_arm(tag, tool, cohort, design, correct, args, cn_dev, seg_sizes):
    comb, patient, batch = emit(cohort, design, args.cells, args.sigma_batch, args.depth_sigma)
    X = np.asarray(comb.X, dtype=float)
    import anndata as ad
    a = ad.AnnData(X.astype(np.float32))
    a.var_names = list(comb.var_names)
    a.obs_names = [f"C{i}" for i in range(X.shape[0])]
    a.obs["batch"] = [str(b) for b in batch]
    a.layers["counts"] = a.X.copy()

    # `correct` uses scDEF's OWN batch-correction path rather than pre-correcting the counts: a
    # hand-rolled log-space centering exponentiated back to counts clips negatives to zero and
    # destroys the matrix a Poisson-gamma model expects (measured: it flattened shared-program
    # recovery from 0.54 to 0.16, an artefact of the round-trip, not of correction).
    inferred = pc.run_tool(tool, a, k=args.k_factors, seed=0, n_epoch=args.n_epoch,
                           n_iter=args.n_iter, batch_key="batch" if correct else None)
    if inferred is None:
        return None
    s = score(inferred, cohort.patients[0].tumor.program_truth["loading"], cn_dev, patient, seg_sizes)
    s.update(design=design, correct=correct, tag=tag, tool=tool)
    print(f"  {tag:34s} shared_recovery={s['shared_recovery']:.3f} "
          f"cna_retention={s['cna_retention']:.3f} patient_mixing={s['patient_mixing']:.2f} "
          f"max_specificity={s['max_specificity']:.2f} n_specific={s['n_patient_specific']}")
    return s


def make_figure(res, dl, cn_dev, seg_sizes, null_mean, null_p95, n_patients, out):
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.8))
    tags = [r["tag"] for r in res]
    colors = ["#4C72B0" if r["tool"] == "scdef" else "#DD8452" for r in res]
    hatch = ["" if r["design"] == "pooled" else ("//" if not r["correct"] else "xx") for r in res]

    # A — the truth: each patient's private, CONTIGUOUS CNA landscape
    ax = axes[0]
    ax.imshow(cn_dev, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    for off in np.cumsum(seg_sizes)[:-1]:
        ax.axvline(off, color="k", lw=0.4, alpha=0.4)
    ax.set_title("A. Truth: private CNA landscape per patient\n"
                 "(log2 CN deviation — CONTIGUOUS, and real biology)", fontsize=9)
    ax.set_xlabel("gene (genome order)"); ax.set_ylabel("patient")
    ax.set_yticks(range(cn_dev.shape[0]))
    txt = "\n".join(f"{k}: corr(CN,expr)={v[0]:.2f}  sd={v[1]:.2f}" for k, v in dl.items())
    ax.set_xlabel(f"gene (genome order)\ndata-level: {txt}", fontsize=7)

    def bars(ax, key, ttl, ylab, extra=None):
        vals = [r[key] for r in res]
        b = ax.bar(range(len(res)), vals, color=colors)
        for bi, h in zip(b, hatch):
            bi.set_hatch(h)
        ax.set_xticks(range(len(res)))
        ax.set_xticklabels(tags, rotation=28, ha="right", fontsize=6.5)
        ax.set_title(ttl, fontsize=9); ax.set_ylabel(ylab, fontsize=8)
        ax.grid(alpha=0.3, axis="y")
        for i, v in enumerate(vals):
            ax.text(i, v, "n/a" if not np.isfinite(v) else f"{v:.2f}",
                    ha="center", va="bottom", fontsize=6.5)
        if extra:
            extra(ax)

    bars(axes[1], "shared_recovery", "B. SHARED programs recovered", "loading cosine vs truth")
    bars(axes[2], "n_patient_specific", "C. PATIENT-SPECIFIC factors found",
         "# factors concentrated in one patient")

    def nulls(ax):
        ax.axhline(null_mean, color="k", ls=":", label=f"scattered null ({null_mean:.2f}) = batch-like")
        ax.axhline(null_p95, color="k", ls="--", alpha=0.4, label="null 95th pct")
        ax.legend(fontsize=6.5); ax.set_ylim(0, 1.0)
    bars(axes[3], "clust_patient_specific",
         "D. Are those factors CNA biology or batch?", "positional clustering", extra=nulls)

    fig.suptitle("Shared vs patient-specific programs across a cohort — scDEF pools, cNMF fragments, "
                 "and batch confounding dilutes the biology (iscc R13)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nfigure -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "manuscript/figures/validation_programs_cohort.png"))
    ap.add_argument("--patients", type=int, default=6)
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--cells", type=int, default=150)
    ap.add_argument("--sigma-batch", type=float, default=0.6)
    ap.add_argument("--depth-sigma", type=float, default=0.3)
    ap.add_argument("--n-epoch", type=int, default=300)
    ap.add_argument("--n-iter", type=int, default=20)
    ap.add_argument("--k-factors", type=int, default=4 * N_PROGRAMS,
                    help="factors per tool; > n_programs so patient-specific factors CAN appear")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.patients, args.steps, args.cells, args.n_epoch, args.n_iter = 4, 8000, 80, 100, 8

    tools = [t for t in ("scdef", "cnmf")
             if (t == "scdef" and pc.scdef_available()) or (t == "cnmf" and pc.cnmf_available())]
    if not tools:
        print("SKIPPING: no tool env — see validation/README_integration.md")
        return

    print(f"growing cohort: {args.patients} patients ...")
    cohort = build_cohort(args.patients, args.steps)
    print("  cancer sizes:", [int(p.tumor.get_cancer_size()) for p in cohort.patients])
    cn_dev = patient_cn_deviation(cohort)
    seg_sizes = cohort.patients[0].tumor.selection.segment_sizes
    null_mean, null_p95 = pc.scattered_null(seg_sizes, top_n=20)
    print(f"  patient CN deviation: max |log2| = {np.abs(cn_dev).max():.2f}")
    print(f"  scattered null = {null_mean:.2f} (95th {null_p95:.2f})")

    # The confound, measured BEFORE any tool touches the data — so it cannot be blamed on the tool.
    print("\n=== data-level: how much of the patient-specific signal is real CNA biology? ===")
    dl = {}
    for design, label in (("pooled", "A pooled (demux)"), ("separate", "B per-patient batch")):
        r, sd = data_level_confound(cohort, design, args, cn_dev)
        dl[design] = (r, sd)
        print(f"  {label:22s} corr(CN dev, EXPR dev) = {r:.3f} | patient-specific expr sd = {sd:.3f}")
    print("  -> the drop in corr, with a RISE in sd, is the technical effect masquerading as patient biology\n")

    # scDEF gets a batch-corrected arm (it has a native batch path); cNMF has none, so it gets the
    # two designs only — that asymmetry is itself part of the result.
    arms = []
    for tool in tools:
        arms.append((f"{tool}: A pooled", tool, "pooled", False))
        arms.append((f"{tool}: B per-pt batch", tool, "separate", False))
        if tool == "scdef":
            arms.append((f"{tool}: B + batch_key", tool, "separate", True))

    res = []
    for tag, tool, design, correct in arms:
        try:
            r = run_arm(tag, tool, cohort, design, correct, args, cn_dev, seg_sizes)
        except Exception as e:
            print(f"  {tag} FAILED: {str(e)[:180]}")
            continue
        if r:
            res.append(r)
    if res:
        make_figure(res, dl, cn_dev, seg_sizes, null_mean, null_p95, args.patients, args.out)


if __name__ == "__main__":
    main()
