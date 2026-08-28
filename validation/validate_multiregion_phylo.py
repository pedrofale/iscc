"""Benchmark: "multi-region tumour trees are not phylogenies" (Alves, Prieto & Posada 2017).

Reproduces AND quantitatively extends Alves, Prieto & Posada (*Multiregional Tumor Trees Are Not
Phylogenies*, PMC5549612 — Posada is the CellCoal author, cited here as posada_cellcoal_2020). Their
claim: bulk multi-region samples are **admixed** — each region is a *mixture* of clones at different
proportions — so a "sample tree" built from regional bulk mutational/VAF profiles reflects
**similarity, not evolutionary history**, producing *spurious parallel mutations*, biased divergence
and reversed ordering. Their fix: **deconvolve clones per region first**, then build the clone tree.

Alves et al. used illustrative simulations only. iscc turns the argument into a **quantitative
benchmark with a ground-truth answer key**, because it uniquely has *both* the true clone phylogeny
(``genotypes_parents`` / ``iscc.integrations.to_newick``) *and* real spatial admixture (clonal
territories that intermix — tuned by the cancer ``dispersal_rate``). The answer key is
``true_origin_counts``: for every locus, the true number of independent origins from Fitch parsimony
on the *true* clone tree. Under the engine's per-allele infinite-sites model most loci arise once;
those single-origin loci are the clean substrate for scoring *spurious* parallelism (an inferred
parallel origin for a mutation the lineage shows arose once — the direct admixture signature).

The demonstration (figure the paper repo's figures/validation_multiregion_phylo.png):
  A. A spatial clone map with the multi-region biopsy overlaid — regions straddle clonal territories,
     so each pooled region is a clone mixture (the admixture).
  B. Naive region "sample tree" spurious-parallelism rate vs the true/deconvolved clone tree: the
     naive tree infers ~20% of single-origin mutations as parallel; the oracle-deconvolved clone tree
     ~0 — even though it has MORE leaves (clones) than the region tree (so it is admixture, not tree
     size). Their fix works.
  C. MORE REGIONS DOESN'T FIX IT: sweep the number of regions K — the spurious rate does not fall to
     zero (it grows), because the problem is admixture, not sampling density. The deconvolved tree
     stays ~0 throughout.
  D. ADMIXTURE DRIVES IT: spurious rate vs the measured per-region clone admixture (effective #clones
     per region) across a ``dispersal_rate`` sweep and seeds — the error scales with admixture.

Prints the headline numbers. Self-contained (neighbour joining / Fitch / Robinson–Foulds in
``iscc.integrations.multiregion``; no ete3 / dendropy needed).

Run:  python -u validation/validate_multiregion_phylo.py
"""
import argparse
import os

import numpy as np
import pandas as pd

from iscc.tumor.models import GenotypeTumor
from iscc.sample.biopsy.biopsy import Biopsy
from iscc import integrations as ig
from _paths import figure_path

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# A solid tumour with ~15-25 sizeable clones over spatial territories. Large genome (18k loci) keeps
# the per-allele infinite-sites model sparse, so most loci are single-origin (a clean answer key).
# With real per-deme crowding (DESIGN_crowding.md) demes cap near K and the tumour spreads by
# dispersal, so we seed a founder cluster (a lone founder goes extinct) and use a moderate K and more
# steps so clones tile the lattice into spatial territories a biopsy disk can straddle.
GENOME = {"n_segments": 60, "segment_size": 300}
SELECTION = {"prop_driver": 0.05, "prop_dispersal": 0.1}
DEME = {"carrying_capacity": 12, "initial_cancer_cells": 5}
SPATIAL = {"grid_size": 40, "structure_radius": 0}
STEPS = 700


def grow(dispersal, seed):
    cancer = {"division_rate": 0.6, "death_rate": 0.05, "max_birth_rate": 0.9,
              "mutation_rate": 0.1, "dispersal_rate": dispersal,
              # SNV-only, infinite-sites-per-allele: a clean single-origin answer key.
              "snv_prob": 1.0, "cnv_prob": 0.0, "n_snvs_per_allele": 1.0}
    t = GenotypeTumor(seed=seed, genome_params=GENOME, selection_params=SELECTION,
                      cancer_cell_params=cancer, deme_params=DEME, spatial_params=SPATIAL)
    t.grow(n_steps=STEPS, seed=seed)
    return t


def region_admixture(cell_data, region_series, gid, min_major=5):
    """Mean effective #clones per region (exp of clone-fraction entropy; rare clones pooled)."""
    name_to_pos = {c: i for i, c in enumerate(cell_data["cell_snv"].index)}
    vc = pd.Series(gid).value_counts()
    major = set(vc[vc >= min_major].index)
    effs = []
    for r in sorted(pd.unique(region_series.values)):
        cells = region_series.index[region_series.values == r]
        cg = [gid[name_to_pos[c]] for c in cells]
        cg = [c if c in major else "minor" for c in cg]
        p = pd.Series(cg).value_counts(normalize=True).values
        effs.append(float(np.exp(-(p * np.log(p)).sum())))
    return float(np.mean(effs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=figure_path("validation_multiregion_phylo.png"))
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    seeds = list(range(1, args.seeds + 1))
    K_grid = [4, 5, 6, 7, 8, 10]
    disp_grid = [0.02, 0.1, 0.3, 0.6]

    print("=" * 78)
    print("Multi-region trees are not phylogenies — quantitative benchmark (iscc)")
    print("=" * 78)

    # ---- Headline reconstruction (dispersal=0.02, K=8), averaged over seeds -------------------
    naive_rates, fix_rates, rfs, recalls, ord_revs = [], [], [], [], []
    ex = None  # a representative run for the spatial panel
    for sd in seeds:
        t = grow(0.02, sd)
        ak = ig.true_origin_counts(t)
        rng = np.random.default_rng(0)
        _, rs, geom = Biopsy(t.cell_data, rng).sample(biopsy_type="multiregion", n_regions=8, radius=1.0)
        res = ig.multiregion_phylogeny(t, rs, answer_key=ak)
        naive_rates.append(res["naive"]["rate"])
        fix_rates.append(res["fix"]["rate"])
        if res["rf"]["rf"] is not None:
            rfs.append(res["rf"]["rf"]); recalls.append(res["rf"]["recall"])
        # ordering reversal (bulk region VAF vs true ancestor->descendant order)
        carriers = {l: frozenset(np.where(np.array([ak["presence"][g][l]
                    for g in ak["present_clones"]]))[0]) for l in np.where(ak["single"])[0]}
        rev, npairs = ig.ordering_reversal_rate(ak["carrier_count"], res["region_vaf"],
                                                ak["single"], carriers, max_pairs=250, seed=sd)
        ord_revs.append(rev)
        if ex is None:
            ex = dict(tumor=t, rs=rs, res=res, ak=ak, single=int(ak["single"].sum()),
                      n_loci=len(ak["loci"]))

    print("\n[A/B] Headline reconstruction (dispersal=0.02, K=8, {} seeds):".format(len(seeds)))
    print("  single-origin loci (answer key) : {} / {}".format(ex["single"], ex["n_loci"]))
    print("  regions={}  deconvolved clones={}".format(ex["res"]["n_regions"], ex["res"]["n_clones"]))
    print("  NAIVE region sample tree  spurious-parallelism rate : {:.3f}  (mean over seeds)".format(np.mean(naive_rates)))
    print("  FIX   deconvolved clone tree spurious rate          : {:.3f}".format(np.mean(fix_rates)))
    print("  -> the clone tree has MORE leaves ({}) than the region tree ({}), yet ~0 spurious"
          .format(ex["res"]["n_clones"], ex["res"]["n_regions"]))
    print("  clone tree vs TRUE clone tree : RF={:.1f}  split-recall={:.2f}".format(np.mean(rfs), np.mean(recalls)))
    print("  bulk-VAF ordering-reversal rate (ancestor<->descendant) : {:.3f}".format(np.mean(ord_revs)))

    # ---- (C) More regions doesn't fix it -----------------------------------------------------
    print("\n[C] More regions doesn't fix it (spurious rate vs #regions):")
    K_naive = {k: [] for k in K_grid}
    K_fix = {k: [] for k in K_grid}
    K_regions = {k: [] for k in K_grid}
    for sd in seeds:
        t = grow(0.02, sd)
        ak = ig.true_origin_counts(t)
        for K in K_grid:
            rng = np.random.default_rng(0)
            _, rs, _ = Biopsy(t.cell_data, rng).sample(biopsy_type="multiregion", n_regions=K, radius=1.0)
            if rs.nunique() < 4:
                continue
            res = ig.multiregion_phylogeny(t, rs, answer_key=ak)
            K_naive[K].append(res["naive"]["rate"])
            K_fix[K].append(res["fix"]["rate"])
            K_regions[K].append(res["n_regions"])
    for K in K_grid:
        if K_naive[K]:
            print("  K~{:2d} (regions~{:.0f}) : naive={:.3f}   deconvolved={:.3f}".format(
                K, np.mean(K_regions[K]), np.mean(K_naive[K]), np.mean(K_fix[K])))

    # ---- (D) Admixture drives it -------------------------------------------------------------
    print("\n[D] Admixture drives it (spurious rate vs measured per-region admixture):")
    admix_x, rate_y, disp_c = [], [], []
    for disp in disp_grid:
        for sd in seeds:
            t = grow(disp, sd)
            ak = ig.true_origin_counts(t)
            gid = ak["gid"]
            rng = np.random.default_rng(0)
            _, rs, _ = Biopsy(t.cell_data, rng).sample(biopsy_type="multiregion", n_regions=8, radius=1.0)
            if rs.nunique() < 4:
                continue
            res = ig.multiregion_phylogeny(t, rs, answer_key=ak)
            admix_x.append(region_admixture(t.cell_data, rs, gid))
            rate_y.append(res["naive"]["rate"])
            disp_c.append(disp)
    admix_x, rate_y, disp_c = np.array(admix_x), np.array(rate_y), np.array(disp_c)
    corr = float(np.corrcoef(admix_x, rate_y)[0, 1]) if len(admix_x) > 2 else float("nan")
    print("  corr(per-region admixture, naive spurious rate) = {:.2f}  (n={})".format(corr, len(admix_x)))

    _figure(args.out, ex, K_grid, K_naive, K_fix, K_regions, admix_x, rate_y, disp_c, corr,
            np.mean(naive_rates), np.mean(fix_rates))
    print("\nSaved figure -> {}".format(args.out))


def _figure(out, ex, K_grid, K_naive, K_fix, K_regions, admix_x, rate_y, disp_c, corr,
            naive_head, fix_head):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(13, 10.5))

    # (A) spatial clone map + biopsy regions
    t, rs = ex["tumor"], ex["rs"]
    cd = t.cell_data
    crd = cd["cell_crd"]
    gid = cd["cell_type"].iloc[:, 0].astype(str).values
    vc = pd.Series(gid).value_counts()
    major = list(vc.index[:12])
    cmap = {g: plt.cm.tab20(i / 12.0) for i, g in enumerate(major)}
    colors = np.array([cmap.get(g, (0.8, 0.8, 0.8, 1.0)) for g in gid])
    jit = np.random.default_rng(0).normal(0, 0.12, size=(len(gid), 2))
    ax[0, 0].scatter(crd["col"].values + jit[:, 1], crd["row"].values + jit[:, 0],
                     c=colors, s=14, alpha=0.85, linewidths=0)
    # region outlines
    name_to_pos = {c: i for i, c in enumerate(cd["cell_snv"].index)}
    for r in sorted(pd.unique(rs.values)):
        cells = rs.index[rs.values == r]
        pos = [name_to_pos[c] for c in cells]
        rr, cc = crd["row"].values[pos], crd["col"].values[pos]
        ax[0, 0].scatter(cc, rr, s=60, facecolors="none", edgecolors="k", linewidths=0.5, alpha=0.5)
        ax[0, 0].text(cc.mean(), rr.mean(), r.replace("region_", "R"), fontsize=9, fontweight="bold",
                      ha="center", va="center")
    ax[0, 0].set_title("A. Clone territories + multi-region biopsy\n(each region pools a clone mixture)")
    ax[0, 0].set_xlabel("grid col"); ax[0, 0].set_ylabel("grid row")
    ax[0, 0].set_aspect("equal")

    # (B) headline bar: naive vs deconvolved spurious rate
    ax[0, 1].bar([0, 1], [naive_head, fix_head], color=["#c0392b", "#27ae60"], width=0.6)
    ax[0, 1].set_xticks([0, 1])
    ax[0, 1].set_xticklabels(["naive\nregion 'sample tree'", "deconvolved\nclone tree (the fix)"])
    ax[0, 1].set_ylabel("spurious parallel-mutation rate")
    for x, v in zip([0, 1], [naive_head, fix_head]):
        ax[0, 1].text(x, v + 0.005, "{:.1%}".format(v), ha="center", fontweight="bold")
    ax[0, 1].set_title("B. Region sample tree invents parallel mutations;\n"
                       "clone deconvolution recovers a single origin")
    ax[0, 1].set_ylim(0, max(naive_head * 1.3, 0.05))

    # (C) more regions doesn't fix it
    Ks = [K for K in K_grid if K_naive[K]]
    xr = [np.mean(K_regions[K]) for K in Ks]
    yn = [np.mean(K_naive[K]) for K in Ks]
    yf = [np.mean(K_fix[K]) for K in Ks]
    ax[1, 0].plot(xr, yn, "o-", color="#c0392b", label="naive region tree")
    ax[1, 0].plot(xr, yf, "s-", color="#27ae60", label="deconvolved clone tree")
    ax[1, 0].set_xlabel("number of biopsy regions (K)")
    ax[1, 0].set_ylabel("spurious parallel-mutation rate")
    ax[1, 0].set_title("C. More regions does NOT fix it\n(admixture, not sampling density)")
    ax[1, 0].legend(); ax[1, 0].set_ylim(bottom=-0.01)

    # (D) admixture drives it
    sc = ax[1, 1].scatter(admix_x, rate_y, c=disp_c, cmap="viridis", s=60, edgecolors="k", linewidths=0.4)
    if len(admix_x) > 2:
        b, a = np.polyfit(admix_x, rate_y, 1)
        xs = np.linspace(admix_x.min(), admix_x.max(), 20)
        ax[1, 1].plot(xs, a + b * xs, "k--", alpha=0.6)
    cb = fig.colorbar(sc, ax=ax[1, 1]); cb.set_label("dispersal_rate")
    ax[1, 1].set_xlabel("measured per-region admixture (effective #clones)")
    ax[1, 1].set_ylabel("naive spurious parallel-mutation rate")
    ax[1, 1].set_title("D. Error scales with admixture\n(corr = {:.2f})".format(corr))

    fig.suptitle("Multi-region bulk 'sample trees' are not phylogenies — and iscc measures by how much",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
