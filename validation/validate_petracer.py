"""Validate the PEtracer lineage-vs-spatial expression decomposition — Tier 1 (self-contained).

PEtracer (Weissman lab, *Science* 2025) decomposes single-cell expression into a **cell-intrinsic**
(heritable, lineage-autocorrelated) and a **cell-extrinsic** (spatial / microenvironment) component
by running a phylogenetic-autocorrelation method on the lineage tree versus the spatial graph. iscc
has BOTH components *with ground truth*: the F8 hypoxia/CCI programmes are extrinsic by construction,
clone-driven expression is intrinsic, and the engine's ``genotypes_parents`` gives the true lineage.

THE FLAGSHIP FINDING — the lineage-space CONFOUND.
In a growing tumour, clonal territories put closely-related cells in the same spatial region, so a
purely ENVIRONMENTAL signal (hypoxia) acquires LINEAGE autocorrelation and a tree-based method
MIS-CLASSIFIES it as heritable. Real PEtracer data cannot catch this (no ground truth); iscc knows
the F8 genes are extrinsic, so it exposes the mis-attribution. The confound is a function of the
lineage-space correlation, which iscc tunes via the cancer ``dispersal_rate``:

  * LOW dispersal  -> clonal territories -> lineage ≈ space -> hypoxia genes look heritable (confound);
  * HIGH dispersal -> intermixed clones  -> lineage ⊥ space -> confound resolved.

Figure (manuscript/figures/validation_petracer.png):
  A. clone spatial map at LOW dispersal — early clades occupy contiguous territories (lineage≈space);
  B. per-gene spatial vs lineage autocorrelation at LOW dispersal, coloured by TRUE category — the
     extrinsic (hypoxia/CCI) genes ride HIGH on the lineage axis = the confound;
  C. confound vs dispersal — extrinsic-gene mis-classification rate tracks the lineage-space
     correlation, both falling as clones intermix;
  D. the RESOLVED case at HIGH dispersal — extrinsic genes drop off the lineage axis.

TIER 2 (real data, ``--real``): places iscc's dispersal sweep and the real PEtracer M2 lineage trees
(reduced by ``validation/data/build_petracer_reference.py``) on ONE lineage-space-coupling axis. The
ground-truth-free confound signature — corr(I_lineage, I_spatial) across genes — is +0.4–0.5 in the
real TERRITORIAL trees and ~0 in the real INTERMIXED tree, and iscc's sweep brackets them: the
confound iscc exposes with ground truth here is measurable in the real data. Falls back to a note
when no reduced reference is cached. Figure: manuscript/figures/validation_petracer_real.png.

Run:  python -u validation/validate_petracer.py            # Tier 1
      python -u validation/validate_petracer.py --real     # + Tier 2 (needs built references)
"""
import argparse
import os

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# A solid tumour dense enough to develop a hypoxic core (an extrinsic gradient) with a rich clone
# tree. Growth is identical F8-on/off (F8 modulates the readout only), so the on/off twins share the
# lineage + space and differ only in expression.
GENOME = {"n_segments": 10, "segment_size": 100}
SELECTION = {"prop_driver": 0.1, "prop_dispersal": 0.1, "prop_immune_resistance": 0.1,
             "prop_treatment_resistance": 0.1}
DEME = {"carrying_capacity": 10}
SPATIAL = {"grid_size": 22, "structure_radius": 0}
MP = {
    "hypoxia": {"strength": 2.5, "n_genes": 60, "o2_consumption": 1.5, "o2_supply": 0.5,
                "o2_source": "perfused"},
    "cci": {"strength": 1.5, "n_target_genes": 60, "emitter_type": "cancer", "lengthscale": 3.0},
}


def grow(dispersal, microenv, steps, seed):
    from iscc.tumor.models import GenotypeTumor
    cancer = {"division_rate": 0.6, "death_rate": 0.03, "max_birth_rate": 0.9,
              "mutation_rate": 0.6, "dispersal_rate": dispersal}
    t = GenotypeTumor(seed=seed, genome_params=GENOME, selection_params=SELECTION,
                      cancer_cell_params=cancer, deme_params=DEME, spatial_params=SPATIAL,
                      microenv_params=microenv)
    t.grow(n_steps=steps, seed=seed)
    return t


def intrinsic_genes(off_tumor, k=60):
    """The truly heritable genes: top-``k`` by between-genotype expression variance in the F8-OFF
    twin (no microenvironment ⇒ all structured variance is clone/CNA-driven = intrinsic)."""
    cd = off_tumor.cell_data
    gid = cd["cell_type"].iloc[:, 0].astype(str).values
    genos = off_tumor.genotypes
    is_cancer = np.array([g in genos and genos[g].type == "cancer" for g in gid])
    means = pd.DataFrame(cd["cell_exp"].values[is_cancer]).groupby(gid[is_cancer]).mean().values
    return np.argsort(means.var(axis=0))[::-1][:k]


def categories(on_tumor, off_tumor, n_genes):
    """True gene categories: extrinsic (F8 hypoxia+CCI), intrinsic (top clone-variance), neutral."""
    ext = np.union1d(on_tumor.microenv_truth["hypoxia_genes"],
                     on_tumor.microenv_truth["cci_target_genes"])
    intr = np.setdiff1d(intrinsic_genes(off_tumor), ext)
    neutral = np.setdiff1d(np.setdiff1d(np.arange(n_genes), ext), intr)
    return {"extrinsic": ext, "intrinsic": intr, "neutral": neutral}


def clade_of(tree, clone, depth_cut):
    """The ancestor of ``clone`` at tree depth ``depth_cut`` (its early clade) — for the map colours."""
    path = tree._path_to_root(clone)               # [clone, ..., root]
    d = tree.depth(clone)
    if d <= depth_cut:
        return clone
    return path[d - depth_cut]                      # step up (d - depth_cut) edges from the clone


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=380)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--low", type=float, default=0.05, help="low dispersal (clonal territories)")
    ap.add_argument("--high", type=float, default=0.6, help="high dispersal (intermixed clones)")
    ap.add_argument("--sweep", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.4, 0.6, 1.0])
    ap.add_argument("--seeds", type=int, default=3, help="replicate tumours per dispersal (averaged)")
    ap.add_argument("--out", default=os.path.join(REPO, "manuscript/figures/validation_petracer.png"))
    ap.add_argument("--real", nargs="*", default=None, metavar="NAME",
                    help="Tier 2: compare against cached real PEtracer tumour(s) "
                         "(petracer_ref_<NAME>.npz). Falls back to a note when absent.")
    args = ap.parse_args()

    from iscc.integrations import decompose_lineage_spatial

    seeds = [args.seed + i for i in range(args.seeds)]

    # -- dispersal sweep: extrinsic-gene mis-classification + lineage-space correlation ----------
    # Averaged over `seeds` replicate tumours (the I_lin>I_sp call is per-gene noisy at a single
    # seed). The first seed's tumour at the low/high dispersal is kept for the figure's map + scatter.
    print(f"dispersal sweep (extrinsic = F8 hypoxia+CCI, truly cell-extrinsic; {args.seeds} seeds):")
    sweep = {k: [] for k in ("disp", "field_lin", "field_sp", "misclass_ext", "misclass_ext_sd",
                             "misclass_intr", "misclass_neu", "I_lin_ext", "I_sp_ext")}
    headline = {}
    for disp in args.sweep:
        rec = {k: [] for k in ("field_lin", "field_sp", "ext", "intr", "neu", "I_lin", "I_sp")}
        for si, seed in enumerate(seeds):
            on = grow(disp, MP, args.steps, seed)
            off = grow(disp, None, args.steps, seed)
            dec = decompose_lineage_spatial(on, seed=0)
            cats = categories(on, off, on.n_genes)
            conf = dec.confusion(cats)
            rec["field_lin"].append(dec.field_autocorr["hypoxia_level"]["lineage"])
            rec["field_sp"].append(dec.field_autocorr["hypoxia_level"]["spatial"])
            rec["ext"].append(conf["extrinsic"]); rec["intr"].append(conf["intrinsic"])
            rec["neu"].append(conf["neutral"])
            rec["I_lin"].append(float(dec.I_lineage[cats["extrinsic"]].mean()))
            rec["I_sp"].append(float(dec.I_spatial[cats["extrinsic"]].mean()))
            if si == 0 and disp in (args.low, args.high):
                headline[disp] = (on, off, dec, cats)
        sweep["disp"].append(disp)
        sweep["field_lin"].append(np.mean(rec["field_lin"])); sweep["field_sp"].append(np.mean(rec["field_sp"]))
        sweep["misclass_ext"].append(np.mean(rec["ext"])); sweep["misclass_ext_sd"].append(np.std(rec["ext"]))
        sweep["misclass_intr"].append(np.mean(rec["intr"])); sweep["misclass_neu"].append(np.mean(rec["neu"]))
        sweep["I_lin_ext"].append(np.mean(rec["I_lin"])); sweep["I_sp_ext"].append(np.mean(rec["I_sp"]))
        print(f"  disp={disp:<5} hypoxia-field autocorr lineage={sweep['field_lin'][-1]:.2f} "
              f"spatial={sweep['field_sp'][-1]:.2f}  "
              f"extrinsic mis-classified heritable={sweep['misclass_ext'][-1]*100:4.0f}%"
              f"±{sweep['misclass_ext_sd'][-1]*100:2.0f}  "
              f"(intrinsic {sweep['misclass_intr'][-1]*100:3.0f}%, neutral {sweep['misclass_neu'][-1]*100:3.0f}%)  "
              f"I_lin_ext={sweep['I_lin_ext'][-1]:.3f}")

    lo, hi = headline[args.low], headline[args.high]
    ilo = args.sweep.index(args.low); ihi = args.sweep.index(args.high)
    print("\nHEADLINE — extrinsic (hypoxia/CCI) genes mis-called HERITABLE by a tree-based method:")
    print(f"  LOW  dispersal ({args.low}, clonal territories): "
          f"{sweep['misclass_ext'][ilo]*100:.0f}%  (the confound; intrinsic genes: "
          f"{sweep['misclass_intr'][ilo]*100:.0f}%)")
    print(f"  HIGH dispersal ({args.high}, intermixed clones):  "
          f"{sweep['misclass_ext'][ihi]*100:.0f}%  (largely resolved)")

    _figure(args, lo, hi, sweep)

    if args.real is not None:
        compare_real(args)


# --------------------------------------------------------------------- Tier 2: real data -------
def iscc_summary(tumor, depth_cut=3, max_cells=800, seed=0):
    """The same summary `reduce_petracer` computes, on an iscc tumour, so real & iscc are directly
    comparable: per-gene lineage/spatial autocorrelation, the lineage-space COUPLING scalar
    (`coord_lineage_autocorr`) + the ground-truth-free CONFOUND signature (`confound_corr` =
    corr(I_lineage, I_spatial) across genes), clone-size + tree-depth distributions, territory."""
    from iscc.integrations import decompose_lineage_spatial
    from iscc.integrations.petracer import _moran_all
    dec = decompose_lineage_spatial(tumor, max_cells=max_cells, seed=seed)
    tree = dec.tree
    cd = tumor.cell_data
    clones = cd["cell_type"].reindex(dec.cells).iloc[:, 0].astype(str).values
    clades = np.array([clade_of(tree, c, depth_cut) for c in clones])
    coords = cd["cell_crd"].reindex(dec.cells).values.astype(float)
    coords = coords + np.random.default_rng(seed).normal(0, 0.15, coords.shape)
    tumour_spread = float(np.sqrt(((coords - coords.mean(0)) ** 2).sum(1).mean()))
    spreads = [np.sqrt(((coords[clades == cl] - coords[clades == cl].mean(0)) ** 2).sum(1).mean())
               for cl in np.unique(clades) if (clades == cl).sum() >= 3]
    # coord lineage autocorr with the SAME lineage weights the decomposition used
    present = list(dict.fromkeys(clones.tolist()))
    gi = {g: i for i, g in enumerate(present)}
    Dg = tree.distance_matrix(present)
    cg = np.array([gi[g] for g in clones])
    Wl = np.exp(-Dg[np.ix_(cg, cg)] / 2.0); np.fill_diagonal(Wl, 0.0)
    coord_lin = float(np.nanmean(_moran_all(Wl, coords)))
    m = np.isfinite(dec.I_lineage) & np.isfinite(dec.I_spatial)
    confound = float(np.corrcoef(dec.I_lineage[m], dec.I_spatial[m])[0, 1])
    return dict(
        I_lineage=dec.I_lineage, I_spatial=dec.I_spatial,
        clone_sizes=pd.Series(clades).value_counts().values.astype(int),
        territory_ratio=float(np.mean(spreads) / tumour_spread) if spreads and tumour_spread > 0 else float("nan"),
        coord_lineage_autocorr=coord_lin, confound_corr=confound,
        tree_depths=np.array([tree.depth(c) for c in clones], dtype=int),
    )


def compare_real(args):
    """Load cached real PEtracer tumours and place iscc on the SAME lineage-space-coupling axis (per
    tumour — the model is a multi-site metastasis, so we never pool sites). The headline: iscc's
    dispersal_rate sweep reproduces the real trees' confound signature. Falls back to a note if absent."""
    import sys
    sys.path.insert(0, os.path.join(REPO, "validation", "data"))
    import build_petracer_reference as B

    names = args.real or []
    if not names:                                       # default: any built caches
        import glob
        names = [os.path.basename(p)[len(B.REFERENCE_PREFIX):-4]
                 for p in sorted(glob.glob(os.path.join(REPO, "validation/data",
                                                        B.REFERENCE_PREFIX + "*.npz")))]
    refs = [(n, B.load_reference(B.reference_path(n))) for n in names]
    present = [(n, r) for n, r in refs if r is not None]
    if not present:
        _, note = B.fetch_petracer(names[0] if names else "tumor1")
        print("\n[Tier 2] no cached real PEtracer reference. " + note +
              "\n  -> Tier 1 (self-contained) stands alone; run "
              "`build_petracer_reference.py --h5td M2_tumor_tracing.h5td` to enable the comparison.")
        return

    # iscc across dispersal (reuse the sweep values) — place each on the coupling axis
    print("\n[Tier 2] iscc dispersal sweep vs real PEtracer trees "
          "(lineage-space coupling → the confound signature):")
    print(f"  {'source':16s} {'cells':>6s} {'coord-lin-coupling':>18s} {'confound corr(Ilin,Isp)':>24s}")
    iscc_pts = []
    for disp in args.sweep:
        s = iscc_summary(grow(disp, MP, args.steps, args.seed))
        iscc_pts.append((disp, s))
        print(f"  {'iscc disp=%.2f' % disp:16s} {len(s['tree_depths']):6d} "
              f"{s['coord_lineage_autocorr']:>+18.3f} {s['confound_corr']:>+24.3f}")
    for n, r in present:
        print(f"  {n:16s} {int(r['n_cells']):6d} {float(r['coord_lineage_autocorr']):>+18.3f} "
              f"{float(r['confound_corr']):>+24.3f}")

    _real_figure(args, iscc_pts, present)


def _real_figure(args, iscc_pts, present):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

    # A. THE HEADLINE — confound signature vs lineage-space coupling, iscc sweep + real trees on one axis
    ax[0].scatter([s["coord_lineage_autocorr"] for _, s in iscc_pts],
                  [s["confound_corr"] for _, s in iscc_pts], s=55, color="#2b6fb0",
                  label="iscc (dispersal sweep)", zorder=2)
    for disp, s in iscc_pts:
        ax[0].annotate(f"disp={disp}", (s["coord_lineage_autocorr"], s["confound_corr"]),
                       fontsize=7, color="#2b6fb0", xytext=(3, 3), textcoords="offset points")
    for n, r in present:
        ax[0].scatter(float(r["coord_lineage_autocorr"]), float(r["confound_corr"]), s=90, marker="*",
                      color="#d1495b", zorder=3, edgecolors="k", linewidths=0.5)
        ax[0].annotate(n.replace("M2-1_", "tree "), (float(r["coord_lineage_autocorr"]),
                       float(r["confound_corr"])), fontsize=7, color="#d1495b",
                       xytext=(4, -8), textcoords="offset points")
    ax[0].axhline(0, color="k", lw=0.5, alpha=0.4)
    ax[0].scatter([], [], s=90, marker="*", color="#d1495b", edgecolors="k",
                  linewidths=0.5, label="real PEtracer trees")
    ax[0].set(xlabel="lineage-space coupling  (coord lineage autocorr)",
              ylabel="confound: corr(I_lineage, I_spatial) across genes",
              title="A. iscc reproduces the real confound regime\n(territories → spatial genes read "
                    "as heritable)")
    ax[0].legend(fontsize=8, loc="upper left")

    # B. per-gene lineage-vs-spatial autocorr for the MOST territorial real tree — the confound cloud
    terr = max(present, key=lambda nr: float(nr[1]["coord_lineage_autocorr"]))
    r = terr[1]
    ax[1].scatter(r["I_spatial"], r["I_lineage"], s=14, alpha=0.6, color="#d1495b")
    lim = [min(r["I_spatial"].min(), r["I_lineage"].min()), max(r["I_spatial"].max(), r["I_lineage"].max())]
    ax[1].plot(lim, lim, "k--", lw=1, alpha=0.5)
    ax[1].set(xlabel="spatial autocorrelation", ylabel="lineage autocorrelation",
              title=f"B. real {terr[0].replace('M2-1_', 'tree ')} (territorial): spatial genes\n"
                    f"carry lineage autocorr  (r={float(r['confound_corr']):+.2f})")

    # C. clone-size distributions — real trees vs iscc
    for disp, s in iscc_pts[:1]:
        ax[2].hist(np.log10(s["clone_sizes"]), bins=12, density=True, alpha=0.5,
                   color="#2b6fb0", label=f"iscc (disp={disp})")
    for n, r in present:
        ax[2].hist(np.log10(np.asarray(r["clone_sizes"])), bins=12, density=True, histtype="step",
                   lw=1.8, label=n.replace("M2-1_", "tree "))
    ax[2].set(xlabel="log10 clade size", ylabel="density", title="C. clade-size distribution")
    ax[2].legend(fontsize=7)

    fig.suptitle("PEtracer (Tier 2): the lineage-space confound iscc exposed with ground truth is "
                 "measurable in REAL PEtracer tumours", fontsize=12, y=1.02)
    fig.tight_layout()
    out = os.path.join(REPO, "manuscript/figures/validation_petracer_real.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("figure ->", out)


def _scatter_panel(ax, dec, cats, title, xlim, ylim):
    colors = {"neutral": "#c9ccd1", "intrinsic": "#2b6fb0", "extrinsic": "#d1495b"}
    for name in ("neutral", "intrinsic", "extrinsic"):
        idx = cats[name]
        ax.scatter(dec.I_spatial[idx], dec.I_lineage[idx], s=18, alpha=0.7,
                   color=colors[name], label=f"{name} (n={len(idx)})",
                   edgecolors="none", zorder=2 if name != "neutral" else 1)
    hi = max(xlim[1], ylim[1])
    ax.plot([0, hi], [0, hi], "k--", lw=1, alpha=0.5, zorder=0,
            label="lineage = spatial")
    ax.set(xlabel="spatial autocorrelation  (Moran's I)",
           ylabel="lineage autocorrelation  (Moran's I)", title=title, xlim=xlim, ylim=ylim)
    ax.legend(fontsize=8, loc="upper right")


def _figure(args, lo, hi, sweep):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    on_lo, off_lo, dec_lo, cats_lo = lo
    dec_hi, cats_hi = hi[2], hi[3]

    fig, ax = plt.subplots(2, 2, figsize=(13, 11))

    # A. clone spatial map at LOW dispersal — early clades occupy contiguous territories
    cd = on_lo.cell_data
    gid = cd["cell_type"].iloc[:, 0].astype(str).values
    genos = on_lo.genotypes
    is_cancer = np.array([g in genos and genos[g].type == "cancer" for g in gid])
    tree = dec_lo.tree
    rng = np.random.default_rng(0)
    crd = cd["cell_crd"].values[is_cancer].astype(float) + rng.normal(0, 0.18, (is_cancer.sum(), 2))
    clones = gid[is_cancer]
    clades = np.array([clade_of(tree, c, depth_cut=2) for c in clones])
    uniq = pd.unique(clades)
    palette = plt.cm.tab20(np.linspace(0, 1, max(len(uniq), 1)))
    cmap = {c: palette[i % len(palette)] for i, c in enumerate(uniq)}
    ax[0, 0].scatter(crd[:, 1], crd[:, 0], c=[cmap[c] for c in clades], s=22, edgecolors="none")
    ax[0, 0].invert_yaxis()
    ax[0, 0].set(title=f"A. clone map — LOW dispersal ({args.low})\nearly clades = contiguous "
                       f"territories  (lineage ≈ space)", xlabel="col", ylabel="row")

    # B & D share axes so the extrinsic genes' drop off the lineage axis is directly comparable
    ymax = max(dec_lo.I_lineage.max(), dec_hi.I_lineage.max()) * 1.08
    xmax = max(dec_lo.I_spatial.max(), dec_hi.I_spatial.max()) * 1.08
    xlim, ylim = (min(-0.02, dec_lo.I_spatial.min() * 1.1), xmax), (-0.02, ymax)

    # B. per-gene decomposition at LOW dispersal — the confound
    _scatter_panel(ax[0, 1], dec_lo, cats_lo,
                   f"B. gene decomposition — LOW dispersal ({args.low})\nextrinsic genes ride HIGH "
                   f"on the lineage axis = CONFOUND", xlim, ylim)

    # C. confound vs dispersal
    disp = np.array(sweep["disp"])
    axc = ax[1, 0]
    axc.errorbar(disp, np.array(sweep["misclass_ext"]) * 100,
                 yerr=np.array(sweep["misclass_ext_sd"]) * 100, fmt="-o", color="#d1495b",
                 capsize=3, label="extrinsic mis-classified heritable (%)")
    axc.set(xlabel="cancer dispersal_rate  (clonal intermixing →)",
            ylabel="extrinsic genes called heritable (%)",
            title="C. confound resolves as clones intermix")
    axc.set_ylim(-5, 105)
    axt = axc.twinx()
    axt.plot(disp, sweep["field_lin"], "-s", color="#4b6584", alpha=0.85,
             label="hypoxia field: lineage autocorr")
    axt.plot(disp, sweep["field_sp"], ":s", color="#4b6584", alpha=0.55,
             label="hypoxia field: spatial autocorr")
    axt.set_ylabel("true hypoxia-field autocorrelation", color="#4b6584")
    axt.tick_params(axis="y", labelcolor="#4b6584")
    axt.set_ylim(bottom=0)
    l1, la1 = axc.get_legend_handles_labels()
    l2, la2 = axt.get_legend_handles_labels()
    axc.legend(l1 + l2, la1 + la2, fontsize=8, loc="upper right")

    # D. resolved case at HIGH dispersal
    _scatter_panel(ax[1, 1], dec_hi, cats_hi,
                   f"D. gene decomposition — HIGH dispersal ({args.high})\nextrinsic genes drop off "
                   f"the lineage axis  (RESOLVED)", xlim, ylim)

    fig.suptitle("PEtracer (Tier 1): lineage-vs-spatial expression decomposition — the "
                 "lineage-space confound, exposed by iscc's ground truth", fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print("figure ->", args.out)


if __name__ == "__main__":
    main()
