"""Figure 4: space misleads lineage inference.

Left, the lineage-space confound (R1 dispersal sweep, the PEtracer-style decomposition):
  (a) territorial against intermixed growth;
  (b) per-gene spatial against lineage autocorrelation at low dispersal, by true category;
  (c) the fraction of extrinsic (hypoxia-driven) genes a tree-based method calls heritable, against
      dispersal;
  (d) the ground-truth-free signature against lineage-space coupling: iscc's sweep and three real
      PEtracer lineages.
Right, tissue architecture and copy-number evolution (Harald's benchmark):
  (e) confined against free growth (R3, paper regime);
  (f)-(h) truncal copy-number fraction, clonal diversity and reconstruction error by architecture and
      selection, from Harald's run 1 (vault experiments/data/cn-evolution-local-run-1.csv, 12x12 grid,
      5 seeds per cell), read as is.

Gene categories: extrinsic = the hypoxia field's target genes; intrinsic = the 60 genes with the most
between-genotype variance among the rest, excluding program genes a niche drives. (The published
analysis took intrinsic genes from a twin grown with the microenvironment off; the microenvironment
is readout-only, so the two differ only in which niche-free genes rank highest.)

Run:  python validation/paper/fig4_space_lineage.py [--jobs 1]
"""
import argparse
import io
import json
import multiprocessing as mp
import os
import subprocess
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
CACHE = os.path.join(C.DERIVED, "petracer")
DISPS = (0.1, 0.3, 0.9, 2.0, 4.0)
VAULT = os.path.join(os.path.dirname(REPO), "iscc-markdown")
HARALD_CSV = "experiments/data/cn-evolution-local-run-1.csv"


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


def analyse(task):
    """Decompose one R1 tumour; cache the summary (and per-gene values for seed 1)."""
    from iscc.integrations import decompose_lineage_spatial
    sys.path.insert(0, os.path.join(REPO, "validation"))
    import validate_petracer as VP
    t = REG.load("R1", task)
    t.make_cell_data(max_cells=3000)
    cats = categories(t)
    dec = decompose_lineage_spatial(t, max_cells=600, seed=0)
    conf = dec.confusion(cats)
    rec = dict(task=task, disp=float(task.split("_")[0][4:]), misclass_ext=conf["extrinsic"],
               misclass_intr=conf["intrinsic"], misclass_neu=conf["neutral"],
               field_lin=dec.field_autocorr["hypoxia_level"]["lineage"],
               field_sp=dec.field_autocorr["hypoxia_level"]["spatial"])
    try:
        s = VP.iscc_summary(t, max_cells=600, seed=0)
        rec.update({k: float(v) for k, v in s.items() if np.isscalar(v)})
    except Exception as e:
        rec["summary_error"] = f"{type(e).__name__}: {e}"
    if task.endswith("_seed1"):
        rec["I_lin"] = np.asarray(dec.I_lineage, dtype=float).tolist()
        rec["I_sp"] = np.asarray(dec.I_spatial, dtype=float).tolist()
        rec["cats"] = {k: np.asarray(v, dtype=int).tolist() for k, v in cats.items()}
    os.makedirs(CACHE, exist_ok=True)
    with open(os.path.join(CACHE, f"{task}.json"), "w") as f:
        json.dump(rec, f)
    return task


def _safe(task):
    try:
        return analyse(task), None
    except Exception as e:
        return task, f"{type(e).__name__}: {e}"


def compute(jobs):
    todo = [k for k in REG.tasks("R1") if os.path.exists(REG.task_path("R1", k))
            and not os.path.exists(os.path.join(CACHE, f"{k}.json"))]
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


def harald():
    """Harald's run 1 results, read from the vault's main branch as they are."""
    try:
        txt = subprocess.run(["git", "-C", VAULT, "show", f"origin/main:{HARALD_CSV}"], check=True,
                             capture_output=True, text=True).stdout
        return pd.read_csv(io.StringIO(txt))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args()
    compute(a.jobs)
    S.setup()
    rows = load_cache()
    fig = S.figure("double", 72)
    out = {}
    S.text_mm(fig, 2, 2.5, "Clonal territories make the environment look heritable", fontsize=7,
              va="top", fontweight="bold")
    # (a) maps
    S.panel_letter(fig, "a", 2, 7)
    for i, d in enumerate((0.1, 4.0)):
        ax = S.ax_mm(fig, 6 + i * 25, 11, 22, 22)
        task = f"disp{d}_seed1"
        if os.path.exists(REG.task_path("R1", task)):
            t = REG.load("R1", task)
            ax.imshow(C.tissue_map(t, C.Clones(t)), interpolation="nearest")
            del t
        S.despine(ax, False, False)
        ax.set_title(f"dispersal {d}", fontsize=5.5)
    # (b) per-gene scatter at low dispersal
    S.panel_letter(fig, "b", 2, 37)
    ax = S.ax_mm(fig, 9, 41, 22, 22)
    lo = next((r for r in rows if r["task"] == "disp0.1_seed1" and "I_lin" in r), None)
    if lo:
        Il, Is = np.asarray(lo["I_lin"]), np.asarray(lo["I_sp"])
        for k, col in (("neutral", S.BG_CELLS), ("intrinsic", S.CLONE_FAMILIES[3][1]),
                       ("extrinsic", S.MODALITY["spatial"])):
            idx = np.asarray(lo["cats"][k], dtype=int)
            ax.scatter(Is[idx], Il[idx], s=1.5 if k == "neutral" else 3, color=col, label=k,
                       rasterized=True)
        m = max(np.nanmax(Il), np.nanmax(Is))
        ax.plot([0, m], [0, m], color=S.INK, lw=0.5, ls=":")
        ax.set_xlabel("Spatial autocorrelation")
        ax.set_ylabel("Lineage autocorrelation")
        ax.legend(fontsize=4.5, loc="lower right")
        ax.set_title("dispersal 0.1", fontsize=5.5)
    # (c) misclassification against dispersal
    S.panel_letter(fig, "c", 34, 37)
    ax = S.ax_mm(fig, 41, 41, 22, 22)
    if rows:
        df = pd.DataFrame(rows)
        g = df.groupby("disp")
        for col, name, c in (("misclass_ext", "extrinsic", S.MODALITY["spatial"]),
                             ("misclass_intr", "intrinsic", S.CLONE_FAMILIES[3][1]),
                             ("misclass_neu", "neutral (baseline)", S.GREY)):
            ax.errorbar(g[col].mean().index, g[col].mean().values, yerr=g[col].std().values, color=c,
                        marker="o", ms=2.3, lw=0.8, capsize=1.3, label=name)
        ax.set_xscale("log")
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Dispersal rate")
        ax.set_ylabel("Called heritable")
        ax.legend(fontsize=4.5)
        out["misclass_ext"] = g["misclass_ext"].mean().round(2).to_dict()
        out["misclass_intr"] = g["misclass_intr"].mean().round(2).to_dict()
        out["misclass_neu"] = g["misclass_neu"].mean().round(2).to_dict()
        out["n_per_disp"] = int(g.size().min())
    # (d) real PEtracer against iscc
    S.panel_letter(fig, "d", 66, 7)
    ax = S.ax_mm(fig, 73, 11, 26, 26)
    if rows and "coord_lineage_autocorr" in rows[0]:
        df = pd.DataFrame(rows)
        sigcol = "confound_corr" if "confound_corr" in df else None
        if sigcol:
            ax.scatter(df.coord_lineage_autocorr, df[sigcol], s=4, c=np.log10(df.disp), cmap="cividis",
                       label="iscc")
            for name, x, y in real_points():
                ax.scatter([x], [y], s=12, color=S.SOURCE["real"], edgecolors=S.INK, lw=0.4, zorder=3)
                ax.annotate(name, (x, y), fontsize=4.2, xytext=(2, 2), textcoords="offset points")
            ax.set_xlabel("Lineage-space coupling")
            ax.set_ylabel("Confound signature")
    else:
        ax.text(0.5, 0.5, "pending", transform=ax.transAxes, ha="center", fontsize=5, color=S.GREY)
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
    H = harald()
    if H is not None:
        H["arch"] = H.scenario.str.contains("confined").map({True: "confined", False: "free"})
        panels = (("f", "trunk_fraction", "Truncal CN fraction"), ("g", "D", "Clonal diversity, D"),
                  ("h", "rf_excess", "Tree error over floor"))
        for i, (L, col, ylab) in enumerate(panels):
            S.panel_letter(fig, L, 106 + i * 26, 37)
            ax = S.ax_mm(fig, 113 + i * 26, 41, 18, 22)
            for j, (arch, sel) in enumerate(((a, s) for a in ("confined", "free")
                                             for s in ("neutral", "strong"))):
                sub = H[(H.arch == arch) & (H.selection == sel)][col].astype(float)
                x = j + np.random.default_rng(j).uniform(-0.12, 0.12, len(sub))
                c = S.CLONE_FAMILIES[0][1] if arch == "confined" else S.CLONE_FAMILIES[2][1]
                ax.scatter(x, sub, s=3, color=c, alpha=0.8 if sel == "strong" else 0.4)
                ax.plot([j - 0.25, j + 0.25], [sub.mean()] * 2, color=S.INK, lw=0.8)
            ax.set_xticks(range(4))
            ax.set_xticklabels(["conf.\nneutral", "conf.\nstrong", "free\nneutral", "free\nstrong"],
                               fontsize=4.3)
            ax.set_ylabel(ylab)
        S.text_mm(fig, 108, 67, "f-h: Harald's benchmark, run 1 (12x12 grid, 5 seeds per group), read as is",
                  fontsize=4.5, color=S.GREY, va="top")
    path = S.save(fig, "fig4_space_lineage")
    print("wrote", path)
    print(out)


if __name__ == "__main__":
    main()
