"""Numbat (allele-aware CNA-from-expression) vs inferCNV — does the allele layer actually help?

inferCNV (already benchmarked, `validate_infercnv.py`) infers copy number from EXPRESSION ALONE.
Numbat~\cite{gao_numbat_2023} adds B-allele frequencies (BAF) at phased SNP sites plus a phylogeny
prior — the stronger, allele-aware successor. R13's allele-resolved expression exists precisely to make
this testable: the clean question `iscc` can answer is whether the allele layer improves CNA recovery,
and by how much, in a HEAD-TO-HEAD on the SAME cells.

WHY THIS IS NON-CIRCULAR. The allelic imbalance Numbat reads is not imposed: it EMERGES from the
per-homolog dosage of each clone's CNAs (an amplified or lost homolog pushes `cell_rna_baf` away from
0.5). `iscc` additionally supplies the true per-cell copy number and clone label no real dataset has.

INPUT-INTERFACE SCOPING (documented main risk — see integration_common.build_numbat_inputs). Numbat's
usual allele pipeline (cellsnp-lite pileup + a population phasing panel) has no analogue in an abstract
genome. We feed Numbat the allele counts DIRECTLY: iscc tracks the two homologs, so it knows the true
PHASE (each gene is one phased marker, GT="1|0"), the ALT count is the m-homolog's reads drawn at UMI
depth, and iscc's segments are mapped onto Numbat's chromosomes with a custom gtf. iscc's homolog labels
ARE the ground-truth phasing a real panel only approximates — the honest answer to the scoping question.

Figure (manuscript/figures/validation_numbat.png):
  A. per-segment / clone-level CN correlation — Numbat vs inferCNV (does the allele layer sharpen CN?);
  B. malignant-vs-normal separation (AUC) — Numbat vs inferCNV;
  C. clone recovery — Numbat's clone-assignment ARI + clone-level CN correlation;
  D. the emergent allele signal — pooled segment BAF at copy-number-altered vs balanced segments,
     malignant vs normal (why the allele layer carries information at all).

Run (from the repo root, in the `iscc` env):
    python -u validation/validate_numbat.py            # the paper figure
    python -u validation/validate_numbat.py --quick    # fast smoke
Requires `iscc-numbat` (and `iscc-infercnv` for the head-to-head); missing tools are skipped w/ a note.
"""
import argparse
import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import integration_common as C

REPO = C.REPO
FIG = os.path.join(REPO, "manuscript", "figures", "validation_numbat.png")


def allele_signal(tumor, shared, numbat_in):
    """Pooled segment BAF (|AD/DP - 0.5|) at CN-altered vs balanced segments, malignant vs normal —
    the emergent allelic-imbalance signal Numbat exploits."""
    df = pd.read_csv(os.path.join(numbat_in["work_dir"], "df_allele.csv"))
    mal_ids = set(np.array(shared["cell_ids"])[shared["is_malignant"]])
    df["mal"] = df["cell"].isin(mal_ids)
    consensus = shared["consensus"]                              # clones x segments
    altered_seg = set((np.where(consensus.std(0) > 0)[0] + 1).tolist())  # CHROM = seg+1
    out = {}
    for grp, sub in (("malignant", df[df["mal"]]), ("normal", df[~df["mal"]])):
        pooled = sub.groupby("CHROM").apply(
            lambda x: abs(x["AD"].sum() / max(x["DP"].sum(), 1) - 0.5))
        alt = [pooled[c] for c in pooled.index if c in altered_seg]
        bal = [pooled[c] for c in pooled.index if c not in altered_seg]
        out[grp] = (float(np.mean(alt)) if alt else 0.0, float(np.mean(bal)) if bal else 0.0)
    return out


def run_once(seed, steps, cnv_prob, tmp, min_cells, max_iter):
    cancer = dict(C.CANCER, cnv_prob=cnv_prob, snv_prob=1.0 - cnv_prob)
    tumor = C.grow_tumor_alleles(seed=seed, steps=steps, cancer=cancer)
    shared = C.build_cna_inputs(tumor, n_normal=150, n_clones=4, seed=seed)
    print(f"  tumour seed={seed}: {shared['is_malignant'].sum()} malignant + "
          f"{(~shared['is_malignant']).sum()} normal, {shared['consensus'].shape[0]} clones")

    wd = os.path.join(tmp, f"seed{seed}")
    numbat_in = C.build_numbat_inputs(tumor, shared, wd, seed=seed)

    result = {"seed": seed}
    # --- Numbat (expression + allele) ---
    if C.numbat_available():
        res = C.run_numbat(numbat_in, wd, min_cells=min_cells, ncores=1, max_iter=max_iter)
        scn = C.score_numbat(res, numbat_in)
        result["numbat"] = scn
        print(f"    Numbat:  AUC={scn['malignant_normal_auc']:.3f}  seg_r={scn['mean_segment_r']:.3f}"
              f"  clone_ARI={scn['clone_ari']:.3f}  clone_r={scn.get('clone_level_r', float('nan')):.3f}")
    # --- inferCNV (expression only) on the SAME cells ---
    if C.infercnv_available():
        inf = C.run_infercnv(shared["adata"], os.path.join(wd, "infercnv"))
        # clone labels aligned to malignant cells for the denoised clone-level correlation
        mal_pos = {c: i for i, c in enumerate(shared["mal_cells"])}
        obs_mal = [c for c in inf["obs_names"] if c in mal_pos]
        lab = np.array([shared["clone_labels"][mal_pos[c]] for c in obs_mal])
        # score_infercnv expects clone labels in the malignant-cell order of the AnnData
        mal_mask = inf["cell_type"] == "malignant"
        lab_full = np.array([shared["clone_labels"][mal_pos[c]]
                             for c in np.array(inf["obs_names"])[mal_mask]])
        sci = C.score_infercnv(inf, shared["adata"].obsm["true_seg_cn"], clone_labels=lab_full)
        # a FAIR unsupervised clone-recovery number for inferCNV: cluster its per-cell CNV profile into
        # the same number of clones and score ARI vs the truth (Numbat clusters via its phylogeny).
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_rand_score
        K = shared["consensus"].shape[0]
        seg_mal = inf["seg_score"][mal_mask]
        km = KMeans(n_clusters=K, n_init=10, random_state=0).fit_predict(seg_mal)
        sci["clone_ari"] = float(adjusted_rand_score(lab_full, km))
        result["infercnv"] = sci
        print(f"    inferCNV: AUC={sci['malignant_normal_auc']:.3f}  seg_r={sci['mean_segment_r']:.3f}"
              f"  clone_ARI={sci['clone_ari']:.3f}  clone_r={sci.get('clone_level_r', float('nan')):.3f}")
    result["allele"] = allele_signal(tumor, shared, numbat_in)
    return result


def make_figure(results, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def mean(tool, key, sub=None):
        vals = []
        for r in results:
            if tool in r:
                v = r[tool].get(key)
                vals.append(v)
        vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
        return float(np.mean(vals)) if vals else float("nan")

    c_nb, c_inf = "#2e6f95", "#a0a0a0"
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.3))

    # A. CN correlation head-to-head
    ax = axes[0]
    groups = ["per-segment\n(single cell)", "clone-level\n(denoised)"]
    nb = [mean("numbat", "mean_segment_r"), mean("numbat", "clone_level_r")]
    inf = [mean("infercnv", "mean_segment_r"), mean("infercnv", "clone_level_r")]
    x = np.arange(len(groups)); w = 0.36
    ax.bar(x - w / 2, nb, w, label="Numbat (expr+allele)", color=c_nb)
    ax.bar(x + w / 2, inf, w, label="inferCNV (expr only)", color=c_inf)
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylabel("copy-number correlation r")
    ax.set_title("A. CN recovery: does the allele layer help?")
    ax.legend(frameon=False, fontsize=9); ax.set_ylim(0, 1)

    # B. malignant-vs-normal AUC
    ax = axes[1]
    aucs = [mean("numbat", "malignant_normal_auc"), mean("infercnv", "malignant_normal_auc")]
    ax.bar(["Numbat", "inferCNV"], aucs, color=[c_nb, c_inf])
    ax.axhline(0.5, color="k", lw=0.7, ls="--")
    ax.set_ylabel("malignant-vs-normal AUC"); ax.set_ylim(0, 1)
    ax.set_title("B. Aneuploid-cell detection")

    # C. clone recovery head-to-head (unsupervised ARI vs the true clones)
    ax = axes[2]
    aris = [mean("numbat", "clone_ari"), mean("infercnv", "clone_ari")]
    ax.bar(["Numbat", "inferCNV"], aris, color=[c_nb, c_inf])
    ax.set_ylim(0, 1); ax.set_ylabel("clone-assignment ARI")
    ax.set_title("C. Subclone recovery\n(phylogeny vs expression clustering)")

    # D. emergent allele signal
    ax = axes[3]
    mal_alt = np.mean([r["allele"]["malignant"][0] for r in results if "allele" in r])
    mal_bal = np.mean([r["allele"]["malignant"][1] for r in results if "allele" in r])
    nor_alt = np.mean([r["allele"]["normal"][0] for r in results if "allele" in r])
    nor_bal = np.mean([r["allele"]["normal"][1] for r in results if "allele" in r])
    x = np.arange(2); w = 0.36
    ax.bar(x - w / 2, [mal_alt, nor_alt], w, label="CN-altered segments", color="#d1495b")
    ax.bar(x + w / 2, [mal_bal, nor_bal], w, label="balanced segments", color="#edae49")
    ax.set_xticks(x); ax.set_xticklabels(["malignant", "normal"])
    ax.set_ylabel("pooled segment |BAF - 0.5|")
    ax.set_title("D. The emergent allele signal\n(imbalance tracks the CNAs)")
    ax.legend(frameon=False, fontsize=9)

    fig.suptitle("iscc as non-circular ground truth: allele-aware Numbat vs expression-only inferCNV",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nwrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[3, 7])
    ap.add_argument("--steps", type=int, default=1400)
    ap.add_argument("--cnv-prob", type=float, default=0.65)
    ap.add_argument("--min-cells", type=int, default=20)
    ap.add_argument("--max-iter", type=int, default=2)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if not (C.numbat_available() or C.infercnv_available()):
        print("Neither iscc-numbat nor iscc-infercnv found — nothing to run.")
        print("Build them per validation/README_integration.md, then re-run.")
        return

    seeds = args.seeds[:1] if args.quick else args.seeds
    steps = 900 if args.quick else args.steps
    print(f"Numbat vs inferCNV: numbat={C.numbat_available()} infercnv={C.infercnv_available()} "
          f"seeds={seeds}")
    tmp = tempfile.mkdtemp(prefix="numbat_val_")
    results = []
    for seed in seeds:
        results.append(run_once(seed, steps, args.cnv_prob, tmp, args.min_cells, args.max_iter))

    make_figure(results, FIG)


if __name__ == "__main__":
    main()
