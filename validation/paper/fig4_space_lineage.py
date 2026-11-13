"""Figure 4: space misleads lineage inference.

Left, the lineage-space confound (R1, 25 tumours over five dispersal rates):
  (a) one tumour's clones and its hypoxia field;
  (b) the hypoxia field's spatial against lineage autocorrelation (Moran's I on the same cells, with
      the spatial and the tree weights of the PEtracer-style decomposition), every tumour. The field
      is environmental by construction, so any lineage autocorrelation it carries is the confound;
  (c) that lineage share of the field, by dispersal;
  (d) the ground-truth-free signature (correlation of per-gene lineage and spatial autocorrelation)
      against lineage-space coupling, on MERFISH-measured expression (124-gene panel, normalised, as
      the real references were reduced), with three real PEtracer lineages.
Right, tissue architecture and copy-number evolution:
  (e) confined against free growth (R3, paper regime);
  (f)-(g) truncal copy-number fraction and clonal diversity (Noble's D at the 5% detection limit) by
      architecture and selection, from the registry's own R3 tumours;
  (h) tree reconstruction error over its polytomy floor: pending, as iscc has no floor-normalised tree
      score of its own yet.

Why the field and not gene categories: in the paper regime copy-number dosage makes nearly every
gene genuinely heritable, so per-gene "called heritable" rates say little about the environment, and
on measured expression the environmental genes' calls sit at chance (issues note, B1). The field
carries the environment's signal without dosage.

Run:  python validation/paper/fig4_space_lineage.py [--jobs 1]
"""
import argparse
import json
import multiprocessing as mp
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import common as C  # noqa: E402
import registry as REG  # noqa: E402
import style as S  # noqa: E402

REPO = os.path.dirname(os.path.dirname(HERE))
CACHE = os.path.join(C.DERIVED, "petracer_v2")
DISPS = REG.R1_DISPERSAL


def categories(t):
    ext = np.asarray(t.microenv_truth["hypoxia_genes"], dtype=int)
    pt = t.program_truth or {}
    niche_prog = set()
    names = list(pt.get("program_names", []))
    genes = pt.get("program_genes", {})
    for p in ("emt", "hypoxia"):
        if isinstance(genes, dict) and p in genes:
            niche_prog |= set(np.asarray(genes[p], dtype=int).tolist())
        elif p in names and hasattr(genes, "__getitem__"):
            niche_prog |= set(np.asarray(genes[names.index(p)], dtype=int).tolist())
    cd = t.cell_data
    gid = cd["cell_type"].iloc[:, 0].astype(str).values
    is_c = np.array([g in t.genotypes and t.genotypes[g].type == "cancer" for g in gid])
    means = pd.DataFrame(cd["cell_exp"].values[is_c]).groupby(gid[is_c]).mean().values
    order = np.argsort(means.var(axis=0))[::-1]
    excl = set(ext.tolist()) | niche_prog
    intr = np.array([g for g in order if g not in excl][:60])
    neutral = np.setdiff1d(np.arange(t.n_genes), np.union1d(np.union1d(ext, intr), list(niche_prog)))
    return {"extrinsic": ext, "intrinsic": intr, "neutral": neutral}


def measured(t, cells, n_panel=124, seed=3):
    """Log-normalised MERFISH counts of ``cells`` over the top-``n_panel`` genes by mean expression:
    the measurement the real PEtracer references were reduced from (124-gene panel, normalised)."""
    from iscc.data import scSpatial
    a = scSpatial(platform="merfish", n_panel_genes=n_panel, seed=seed).run(t.cell_data, cell_subset=cells)
    X = a.observed_counts.values.astype(float)
    lib = X.sum(1, keepdims=True)
    return pd.DataFrame(np.log1p(X / np.maximum(lib, 1) * np.median(lib)),
                        index=a.observed_counts.index, columns=a.observed_counts.columns)


def _signature(dec):
    m = np.isfinite(dec.I_lineage) & np.isfinite(dec.I_spatial)
    return float(np.corrcoef(dec.I_lineage[m], dec.I_spatial[m])[0, 1])


def analyse(key):
    """One tumour (``key`` = "<run>/<task>", R1 or R3): the hypoxia field's lineage and spatial
    autocorrelation (ground truth), the lineage-space coupling, and the confound signature on
    MERFISH-measured expression."""
    from iscc.integrations import decompose_lineage_spatial
    sys.path.insert(0, os.path.join(REPO, "validation"))
    import validate_petracer as VP
    run, task = key.split("/")
    t = REG.load(run, task)
    t.make_cell_data(max_cells=3000)
    dec = decompose_lineage_spatial(t, max_cells=600, seed=0)
    s = VP.iscc_summary(t, max_cells=600, seed=0)
    meas = decompose_lineage_spatial(t, expression=measured(t, dec.cells), max_cells=600, seed=0)
    cats = categories(t)
    conf = dec.confusion(cats)
    arch = "free" if task.startswith("free") else "ductal"
    disp = float(task.split("_")[0][4:]) if run == "R1" else None
    extra = {}
    if run == "R3":
        # Fig 4f-g: the copy-number structure that architecture sets, from the tumour itself
        import sweep_score as SS
        from iscc.inference import mode_indices
        sc = SS.score_tumour(t)
        extra = dict(selection=task.split("_")[1],
                     trunk_fraction=sc["n_truncal_segments"] / float(t.n_segments),
                     D=float(mode_indices(t, detection=0.05)["D"]))
    rec = dict(run=run, task=task, arch=arch, disp=disp,
               field_lin=dec.field_autocorr["hypoxia_level"]["lineage"],
               field_sp=dec.field_autocorr["hypoxia_level"]["spatial"],
               coupling=float(s["coord_lineage_autocorr"]), territory_ratio=float(s["territory_ratio"]),
               signature_true=_signature(dec), signature_merfish=_signature(meas),
               median_I_lin_merfish=float(np.median(meas.I_lineage)),
               median_I_sp_merfish=float(np.median(meas.I_spatial)),
               # per-gene calls on the true expression, kept for the record (not plotted: copy-number
               # dosage makes nearly every gene heritable in this regime, see the issues note B1)
               called_ext=conf["extrinsic"], called_intr=conf["intrinsic"], called_neu=conf["neutral"],
               **extra)
    os.makedirs(CACHE, exist_ok=True)
    with open(os.path.join(CACHE, f"{run}__{task}.json"), "w") as f:
        json.dump(rec, f)
    return key


def _safe(key):
    from memguard import wait_for_memory
    wait_for_memory(label=key)
    try:
        return analyse(key), None
    except Exception as e:
        return key, f"{type(e).__name__}: {e}"


def compute(jobs):
    todo = [f"{r}/{k}" for r in ("R1", "R3") for k in REG.tasks(r) if os.path.exists(REG.task_path(r, k))
            and not os.path.exists(os.path.join(CACHE, f"{r}__{k}.json"))]
    print(f"{len(todo)} decompositions to compute", flush=True)
    with mp.get_context("spawn").Pool(processes=jobs, maxtasksperchild=1) as pool:
        for task, err in pool.imap_unordered(_safe, todo):
            print(f"  {task}: {err or 'ok'}", flush=True)


def load_cache():
    out = []
    if os.path.isdir(CACHE):
        for f in sorted(os.listdir(CACHE)):
            with open(os.path.join(CACHE, f)) as fh:
                out.append(json.load(fh))
    return out


def real_points():
    """The three cached real PEtracer lineages: (coupling, signature) as the published analysis."""
    sys.path.insert(0, os.path.join(REPO, "validation"))
    pts = []
    for name in ("M2-1_1", "M2-1_2-1", "M2-1_5"):
        p = os.path.join(REPO, "validation", "data", f"petracer_ref_{name}.npz")
        if os.path.exists(p):
            z = np.load(p, allow_pickle=True)
            r = {k: z[k] for k in z.files}
            coupling = float(r.get("coord_lineage_autocorr", np.nan))
            sig = float(r.get("confound_corr", np.nan))
            pts.append((name, coupling, sig))
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args()
    compute(a.jobs)
    S.setup()
    rows = load_cache()
    fig = S.figure("double", 76)
    out = {}
    S.text_mm(fig, 2, 2.5, "Glands make the environment look heritable", fontsize=7,
              va="top", fontweight="bold")
    df = pd.DataFrame(rows)
    # (a) one tumour's clones and its hypoxia field
    S.panel_letter(fig, "a", 2, 7)
    task = "disp0.1_seed1"
    if os.path.exists(REG.task_path("R1", task)):
        t = REG.load("R1", task)
        ax = S.ax_mm(fig, 6, 11, 22, 22)
        ax.imshow(C.tissue_map(t, C.Clones(t)), interpolation="nearest")
        S.despine(ax, False, False)
        ax.set_title("clones", fontsize=5.5)
        hyp = t.microenv_params["hypoxia"]
        h = t._o2_field(D=hyp["o2_diffusion"], k=hyp["o2_consumption"], s=hyp["o2_supply"],
                        source=hyp["o2_source"]).reshape(t.grid_size, t.grid_size)
        ax = S.ax_mm(fig, 31, 11, 22, 22)
        im = ax.imshow(h, cmap="YlOrRd", vmin=0, vmax=1, interpolation="nearest")
        S.despine(ax, False, False)
        ax.set_title("hypoxia field", fontsize=5.5)
        cax = S.ax_mm(fig, 54, 11, 1.5, 22)
        fig.colorbar(im, cax=cax).ax.tick_params(labelsize=4.5, length=1.5)
        del t
    ARCH = {"ductal": ("ductal field", S.CLONE_FAMILIES[0][1]), "free": ("free growth", S.CLONE_FAMILIES[2][1])}
    if len(df):
        df["share"] = df.field_lin / (df.field_lin + df.field_sp)
        # (b) the field's autocorrelation on each axis, every tumour, by architecture
        S.panel_letter(fig, "b", 2, 37)
        ax = S.ax_mm(fig, 9, 41, 22, 22)
        for arch, (lab, col) in ARCH.items():
            sub = df[df.arch == arch]
            ax.scatter(sub.field_sp, sub.field_lin, s=5, color=col, edgecolors=S.INK, lw=0.2,
                       label=f"{lab} ({len(sub)})")
        # Moran's I with these weights can exceed 1, so the axes follow the data
        lim = max(1.1, 1.05 * float(np.nanmax(df[["field_sp", "field_lin"]].to_numpy())))
        ax.plot([0, lim], [0, lim], color=S.INK, lw=0.5, ls=":")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel("Spatial autocorrelation")
        ax.set_ylabel("Lineage autocorrelation")
        ax.set_title("hypoxia field", fontsize=5.5)
        ax.legend(fontsize=4.5, loc="upper left")
        # (c) the lineage share of the environment, by architecture and selection
        S.panel_letter(fig, "c", 34, 37)
        ax = S.ax_mm(fig, 41, 41, 22, 22)
        groups = [("ductal", "R1"), ("ductal", "selected"), ("ductal", "neutral"),
                  ("free", "selected"), ("free", "neutral")]
        df["sel"] = df.task.str.extract(r"_(selected|neutral)_")[0].fillna("R1")
        for k, (arch, sel) in enumerate(groups):
            sub = df[(df.arch == arch) & (df.sel == sel)]
            x = k + np.random.default_rng(k).uniform(-0.15, 0.15, len(sub))
            ax.scatter(x, sub.share, s=4, color=ARCH[arch][1], alpha=0.9 if sel != "neutral" else 0.5)
            ax.plot([k - 0.3, k + 0.3], [sub.share.mean()] * 2, color=S.INK, lw=0.8)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(["ductal\nsweep", "ductal\nsel.", "ductal\nneutral", "free\nsel.",
                            "free\nneutral"], fontsize=4.3)
        ax.set_ylim(0, 0.6)
        ax.set_ylabel("Lineage share of the field")
        g = df.groupby("arch").share
        out["share"] = {a: dict(mean=round(float(v.mean()), 2), min=round(float(v.min()), 2),
                                max=round(float(v.max()), 2), n=int(len(v))) for a, v in g}
        r1 = df[df.run == "R1"]
        out["share_by_disp"] = r1.groupby("disp").share.mean().round(2).to_dict()
        from scipy.stats import mannwhitneyu
        out["mwu_p"] = float(mannwhitneyu(df[df.arch == "ductal"].share, df[df.arch == "free"].share).pvalue)
        # (d) the ground-truth-free signature, measured like the real data
        S.panel_letter(fig, "d", 66, 7)
        ax = S.ax_mm(fig, 73, 11, 26, 26)
        for arch, (lab, col) in ARCH.items():
            sub = df[df.arch == arch]
            ax.scatter(sub.coupling, sub.signature_merfish, s=5, color=col, edgecolors=S.INK, lw=0.2)
        for name, x, y in real_points():
            ax.scatter([x], [y], s=14, color=S.SOURCE["real"], edgecolors=S.INK, lw=0.4, zorder=3)
            ax.annotate(name.replace("M2-1_", "lineage "), (x, y), fontsize=4.2, xytext=(2, 2),
                        textcoords="offset points")
        ax.set_xlabel("Lineage-space coupling")
        ax.set_ylabel("Confound signature")
        ax.set_title("MERFISH-measured; gray: real", fontsize=5.5)
        out["signature"] = {a: (round(float(v.min()), 2), round(float(v.max()), 2))
                            for a, v in df.groupby("arch").signature_merfish}
        out["coupling"] = {a: (round(float(v.min()), 2), round(float(v.max()), 2))
                           for a, v in df.groupby("arch").coupling}
    # right half: architecture and copy-number evolution
    S.text_mm(fig, 108, 2.5, "Tissue architecture sets copy-number structure", fontsize=7, va="top",
              fontweight="bold")
    S.panel_letter(fig, "e", 106, 7)
    for i, arch in enumerate(("confined", "free")):
        ax = S.ax_mm(fig, 110 + i * 25, 11, 22, 22)
        task = f"{arch}_selected_seed1"
        if os.path.exists(REG.task_path("R3", task)):
            t = REG.load("R3", task)
            ax.imshow(C.tissue_map(t, C.Clones(t)), interpolation="nearest")
            del t
        S.despine(ax, False, False)
        ax.set_title(arch, fontsize=5.5)
    H = df[df.run == "R3"].copy() if len(df) and "trunk_fraction" in df else None
    if H is not None and len(H):
        H["arch"] = H.arch.map({"ductal": "confined", "free": "free"})
        H["selection"] = H.selection.map({"neutral": "neutral", "selected": "strong"})
        panels = (("f", "trunk_fraction", "Truncal CN fraction"), ("g", "D", "Clonal diversity, D"),
                  ("h", None, "Tree error over floor"))
        for i, (L, col, ylab) in enumerate(panels):
            S.panel_letter(fig, L, 106 + i * 26, 37)
            ax = S.ax_mm(fig, 113 + i * 26, 41, 18, 22)
            if col is None:
                S.despine(ax, False, False)
                ax.text(0.5, 0.5, "pending: a tree score\nnormalised by its\npolytomy floor",
                        ha="center", va="center", fontsize=4.5, color=S.GREY, transform=ax.transAxes)
                continue
            for j, (arch, sel) in enumerate(((a, s) for a in ("confined", "free")
                                             for s in ("neutral", "strong"))):
                sub = H[(H.arch == arch) & (H.selection == sel)][col].astype(float)
                out[f"{L}_{arch}_{sel}"] = float(sub.mean()) if len(sub) else None
                x = j + np.random.default_rng(j).uniform(-0.12, 0.12, len(sub))
                c = S.CLONE_FAMILIES[0][1] if arch == "confined" else S.CLONE_FAMILIES[2][1]
                ax.scatter(x, sub, s=3, color=c, alpha=0.8 if sel == "strong" else 0.4)
                ax.plot([j - 0.25, j + 0.25], [sub.mean()] * 2, color=S.INK, lw=0.8)
            ax.set_xticks(range(4))
            ax.set_xticklabels(["conf.\nneutral", "conf.\nstrong", "free\nneutral", "free\nstrong"],
                               fontsize=4.3)
            ax.set_ylabel(ylab)
        S.text_mm(fig, 108, 72, f"f-g: registry R3, {len(H)} tumours (paper regime)",
                  fontsize=4.5, color=S.GREY, va="top")
    path = S.save(fig, "fig4_space_lineage")
    print("wrote", path)
    print(out)


if __name__ == "__main__":
    main()
