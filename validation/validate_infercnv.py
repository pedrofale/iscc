"""Validate copy-number-from-expression (inferCNV) on iscc data — the second integration benchmark.

inferCNV~\cite{patel_single_2014, gao_copykat_2021} infers copy number from single-cell expression by
smoothing expression along the genome relative to a normal reference: gained regions read high,
deleted regions low. We run the GENUINE ``infercnvpy`` (scverse's Python inferCNV, in the dedicated
``iscc-infercnv`` conda env) on iscc-simulated data and score the inferred CN against iscc's true
per-cell, per-segment copy number.

Like the clonealign benchmark, this is NON-CIRCULAR: the copy-number -> expression coupling inferCNV
inverts is the engine's emergent per-allele dosage (a neutral gene's expression is
baseline*(1+copy_number), plus selection), not a law imposed to match the method. iscc also carries
the true copy number of every cell, which no real dataset provides.

What the benchmark shows:
  * inferCNV separates malignant cells from the diploid normal (epithelial/stromal) compartment
    cleanly (AUC by CNV magnitude);
  * it recovers the clonal CNA structure — the segment gains/losses that distinguish the clones —
    once per-cell noise is averaged within clones;
  * single-cell per-segment CN correlation is real but modest (recovering copy number from expression
    at single-cell resolution is genuinely hard), and is strongest for the largest-amplitude events.

Figure (the paper repo's figures/validation_infercnv.png):
  A. true vs inferred copy number, cells x segments (malignant cells sorted by clone);
  B. per-segment single-cell correlation + the denoised clone-level recovery;
  C. malignant-vs-normal separation by CNV magnitude.

Run (from the repo root, in the `iscc` env):
    python -u validation/validate_infercnv.py
Requires the `iscc-infercnv` env (``pip install infercnvpy`` into a dedicated env). Falls back to a
note if absent.
"""
import argparse
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import integration_common as C
from _paths import figure_path

REPO = C.REPO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--steps", type=int, default=750)
    ap.add_argument("--clones", type=int, default=4)
    ap.add_argument("--normals", type=int, default=200)
    ap.add_argument("--protocol", default="10x")
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--out", default=figure_path("validation_infercnv.png"))
    args = ap.parse_args()

    if not C.infercnv_available():
        print("[infercnv] dedicated env not found at", C.INFERCNV_PYTHON,
              "\n  -> create it (`conda create -n iscc-infercnv python=3.10 && pip install infercnvpy`);"
              " see validation/README_integration.md."
              "\n  Skipping (the iscc data generation itself is exercised by tests/test_integration.py).")
        return

    print(f"growing tumour (seed={args.seed}) and assembling the CNA-from-expression benchmark ...")
    tumor = C.grow_tumor(seed=args.seed, steps=args.steps)
    adata = C.build_infercnv_inputs(tumor, n_normal=args.normals, protocol=args.protocol, seed=0)
    mal = np.asarray(adata.obs["cell_type"] == "malignant")
    true_seg = adata.obsm["true_seg_cn"]
    # group malignant cells into clones on their true segment CN (for the denoised recovery panel).
    labels, consensus = C.define_clones(true_seg[mal], n_clones=args.clones)
    print(f"  malignant={mal.sum()} normal={(~mal).sum()}  clones={consensus.shape[0]} "
          f"sizes={np.bincount(labels).tolist()}")

    with tempfile.TemporaryDirectory() as work:
        print(f"\nrunning infercnvpy (real scverse package), window={args.window} step={args.step} ...")
        res = C.run_infercnv(adata, work, window_size=args.window, step=args.step)
        sc = C.score_infercnv(res, true_seg, clone_labels=labels)

    print("\nHEADLINE — infercnvpy CNA-from-expression vs iscc ground truth:")
    print(f"  malignant-vs-normal separation AUC = {sc['malignant_normal_auc']:.3f}  "
          f"({sc['n_malignant']} malignant vs {sc['n_normal']} normal)")
    print(f"  clone-level CN recovery corr (informative segments) = {sc['clone_level_r']:.3f}")
    print(f"  mean single-cell per-segment correlation = {sc['mean_segment_r']:.3f}")
    print("  per-segment single-cell r:",
          {f"chr{j}": round(v, 3) for j, v in sc["per_segment_r"].items()})

    _figure(args, res, true_seg, labels, sc)


def _figure(args, res, true_seg, labels, sc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mal = res["cell_type"] == "malignant"
    seg = res["seg_score"][mal]
    tru = true_seg[mal]
    order = np.argsort(labels, kind="stable")
    K = int(labels.max()) + 1
    n_seg = seg.shape[1]

    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1])
    axA0 = fig.add_subplot(gs[0, 0]); axA1 = fig.add_subplot(gs[0, 1])
    axB = fig.add_subplot(gs[1, 0]); axC = fig.add_subplot(gs[1, 1])

    # A. true vs inferred copy number, cells x segments (cells sorted by clone)
    im0 = axA0.imshow(tru[order], aspect="auto", cmap="RdBu_r", vmin=0, vmax=4, interpolation="none")
    axA0.set(title="A. TRUE copy number (cells x segments)", xlabel="segment", ylabel="malignant cells (by clone)")
    fig.colorbar(im0, ax=axA0, fraction=0.046, pad=0.04, label="copy number")
    # inferred: z-score per segment for display (removes the shared tumour-vs-normal baseline offset)
    z = (seg - seg.mean(0)) / (seg.std(0) + 1e-9)
    im1 = axA1.imshow(z[order], aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2, interpolation="none")
    axA1.set(title="A'. INFERRED CNV score (z per segment)", xlabel="segment", ylabel="malignant cells (by clone)")
    fig.colorbar(im1, ax=axA1, fraction=0.046, pad=0.04, label="inferred CNV (z)")
    # clone boundaries (cells are sorted by clone id, so boundaries = cumulative clone sizes)
    for b in np.cumsum(np.bincount(labels, minlength=K))[:-1]:
        for ax in (axA0, axA1):
            ax.axhline(b, color="k", lw=0.6, alpha=0.5)

    # B. per-segment single-cell correlation + clone-level recovery
    segs = sorted(sc["per_segment_r"].keys())
    rs = [sc["per_segment_r"][j] for j in segs]
    colors = ["#d1495b" if j in set(sc["informative_segments"].tolist()) else "#9aa4ad" for j in segs]
    axB.bar([f"chr{j}" for j in segs], rs, color=colors)
    axB.axhline(0, color="k", lw=0.8)
    axB.set(ylabel="single-cell Pearson r", ylim=(min(0, min(rs) - 0.05), max(rs) + 0.08),
            title=f"B. inferred-vs-true CN correlation per segment\n(clone-level recovery r="
                  f"{sc['clone_level_r']:.2f}; red = CN-altered segments)")
    axB.tick_params(axis="x", labelrotation=90, labelsize=8)
    axB.text(0.02, 0.92, f"mean single-cell r = {sc['mean_segment_r']:.2f}",
             transform=axB.transAxes, fontsize=9,
             bbox=dict(boxstyle="round", fc="#eef3f8", ec="#2b6fb0"))

    # C. malignant vs normal separation by CNV magnitude
    mag = np.abs(res["x_cnv"]).mean(1)
    axC.hist(mag[mal], bins=35, alpha=0.7, color="#d1495b", label=f"malignant (n={mal.sum()})", density=True)
    axC.hist(mag[~mal], bins=35, alpha=0.7, color="#4b6584", label=f"normal (n={(~mal).sum()})", density=True)
    axC.set(xlabel="per-cell CNV magnitude (mean |inferred CNV|)", ylabel="density",
            title=f"C. malignant vs normal separation\n(AUC = {sc['malignant_normal_auc']:.3f})")
    axC.legend(fontsize=9)

    fig.suptitle("infercnvpy on iscc: copy number recovered from the EMERGENT expression dosage "
                 "(non-circular ground truth)", fontsize=13, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print("\nfigure ->", args.out)


if __name__ == "__main__":
    main()
