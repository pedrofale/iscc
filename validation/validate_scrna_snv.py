"""scDNA-vs-scRNA SNV-calling benchmark (DESIGN_features §C, F7b).

The payoff of mutation-aware scRNA: a true somatic mutation is recovered by scDNA at its genomic
VAF, but scRNA misses or distorts it because the variant is only visible where the gene is
expressed and is further under-detected by the single `obs_fidelity` knob (monoallelic
expression / bursting / RNA editing / RT error). This script grows one tumour, emits the true
DNA-VAF (`cell_snv`) and the observed scRNA allele matrix (`emit_scrna_reads`, UMI totals
conserved) for the SAME cells/loci, and quantifies the gap:

  * observed RNA-VAF vs true DNA-VAF at mutated, expressed sites (distortion + dropout);
  * scRNA SNV detection rate vs `obs_fidelity`;
  * scRNA SNV detection rate vs gene expression (the gating).

A "site" is a (cell, locus) with a true mutation (DNA-VAF > 0); it is "detected in scRNA" when the
gene is expressed there (UMI > 0) and carries >= `min_alt` alt UMIs.

Usage:  python validation/validate_scrna_snv.py
Produces manuscript/figures/validation_scrna_snv.png.
"""
import argparse
import os

import numpy as np

from iscc.tumor.models import GenotypeTumor
from iscc.data.reads import emit_scrna_reads

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def grow(seed=1, steps=700):
    genome = {"n_segments": 8, "segment_size": 50}
    sel = {"prop_driver": 0.25, "prop_dispersal": 0.0, "prop_immune_resistance": 0.0,
           "prop_treatment_resistance": 0.0, "driver_effects": 1.4, "dispersal_effects": 1.0,
           "treatment_resistant_effects": 1.0, "immune_resistant_effects": 1.0}
    cancer = {"division_rate": 0.4, "death_rate": 0.02, "max_birth_rate": 0.95,
              "mutation_rate": 0.6, "dispersal_rate": 0.2, "snv_prob": 0.7, "cnv_prob": 0.3}
    t = GenotypeTumor(seed=seed, genome_params=genome, selection_params=sel,
                      cancer_cell_params=cancer, deme_params={"carrying_capacity": 8, "initial_cancer_cells": 5},
                      spatial_params={"grid_size": 20, "structure_radius": 0})
    t.grow(steps, seed=seed)
    return t


def detection_rate(res, min_alt=1):
    """Fraction of true-mutation sites (DNA-VAF>0) detected in scRNA (expressed + >=min_alt alt)."""
    dna = res["dna_vaf"].values
    alt = res["alt"].values
    total = res["total"].values
    site = dna > 0
    if site.sum() == 0:
        return np.nan, site
    detected = site & (total > 0) & (alt >= min_alt)
    return detected[site].mean(), site


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(REPO, "manuscript/figures/validation_scrna_snv.png"))
    args = ap.parse_args()

    t = grow(seed=args.seed)
    cd = t.make_cell_data()
    fidelities = [1.0, 0.7, 0.4, 0.2]

    # emit at each fidelity (shared cells via the same seed)
    runs = {f: emit_scrna_reads(cd, obs_fidelity=f, protocol="10x", seed=args.seed,
                                outdir=os.path.join(REPO, "validation", "_scrna_tmp"))
            for f in fidelities}

    print("scDNA-vs-scRNA SNV calling")
    rates = {}
    for f in fidelities:
        r, site = detection_rate(runs[f])
        rates[f] = r
        print(f"  obs_fidelity={f:.1f}: scRNA detection rate of true mutations = {r:.3f} "
              f"({int(site.sum())} true-mutation sites)")

    # --- figure ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))

    # Panel A: observed RNA-VAF vs true DNA-VAF at mutated+expressed sites (obs_fidelity=0.7)
    ref = runs[0.7]
    dna = ref["dna_vaf"].values
    obs = ref["obs_vaf"].values
    tot = ref["total"].values
    m = (dna > 0) & (tot > 0)
    ax[0].scatter(dna[m], obs[m], s=6, alpha=0.25)
    ax[0].plot([0, 1], [0, 1], "k--", lw=1, label="DNA = RNA")
    ax[0].set(xlabel="true DNA-VAF", ylabel="observed scRNA-VAF",
              title="VAF distortion (obs_fidelity=0.7)", xlim=(0, 1), ylim=(0, 1))
    ax[0].legend(frameon=False, fontsize=8)

    # Panel B: detection rate vs obs_fidelity
    ax[1].plot(fidelities, [rates[f] for f in fidelities], "o-")
    ax[1].set(xlabel="obs_fidelity", ylabel="scRNA mutation detection rate",
              title="Detection vs fidelity", ylim=(0, 1))

    # Panel C: detection rate vs expression (UMI) bin, at obs_fidelity=0.7
    alt = ref["alt"].values
    site = dna > 0
    umi = tot[site]; det = (alt[site] >= 1) & (tot[site] > 0)
    if umi.size:
        bins = np.array([0, 1, 3, 6, 12, 25, 1e9])
        which = np.digitize(umi, bins) - 1
        xs, ys = [], []
        for b in range(len(bins) - 1):
            sel = which == b
            if sel.sum() >= 5:
                xs.append(0.5 * (bins[b] + min(bins[b + 1], umi.max() + 1)))
                ys.append(det[sel].mean())
        ax[2].plot(xs, ys, "s-")
    ax[2].set(xlabel="gene UMI count (expression)", ylabel="detection rate",
              title="Detection vs expression (gating)", ylim=(0, 1))

    fig.suptitle("SNV calling from scRNA is hard: expression gating + obs_fidelity distort the VAF",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
