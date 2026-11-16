"""Fig 4e: structure misleads a lineage method, read off the registry's own tumours.

The hypoxia field is environmental by construction, so any lineage autocorrelation it carries is the
confound itself. Per tumour, on 600 of its cancer cells (``iscc.integrations.decompose_lineage_spatial``,
the PEtracer-style decomposition of the old fig4_space_lineage.py, max_cells 3000 then 600, seed 0): the
field's Moran's I under the spatial and under the lineage weights, its lineage share
``I_lin / (I_lin + I_sp)``, and a permutation test of the lineage autocorrelation alone (the field's values
shuffled over the same cells, 999 times; one-sided p).

Tumours: R1 (the dispersal sweep, ductal field) and R3 (confined and free growth, each selected and
neutral). Each tumour's result is cached in ``<out>/e/cache/<run>__<task>.json`` with the size and mtime of
the pickle it was read from, so a regrown tumour is recomputed and the rest are not. ``--part panel`` caches
the map of panel (i) (one R1 tumour's clones and its hypoxia field) in ``<out>/e/panel_i.npz``; ``--part
summary`` writes ``<out>/e/summary.json`` with the numbers the text quotes.

    python validation/paper/fig4_lineage.py --out OUT --jobs 8            # compute, panel and summary
    python validation/paper/fig4_lineage.py --out OUT --part summary      # from the caches only
"""
import argparse
import json
import multiprocessing as mp
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import registry as REG  # noqa: E402

N_PERM = 999
PANEL_TASK = ("R1", "disp0.1_seed1")
GROUPS = [("R1", "ductal", None), ("R3", "ductal", "selected"), ("R3", "ductal", "neutral"),
          ("R3", "free", "selected"), ("R3", "free", "neutral")]


def _moran(W, z):
    z = np.asarray(z, float) - np.mean(z)
    den = float(z @ z)
    return 0.0 if den <= 0 or W.sum() <= 0 else float(len(z) / W.sum() * (z @ (W @ z)) / den)


def _stamp(run, task):
    st = os.stat(REG.task_path(run, task))
    return dict(size=int(st.st_size), mtime=int(st.st_mtime))


def _cache(out, run, task):
    return os.path.join(out, "e", "cache", f"{run}__{task}.json")


def analyse(job):
    run, task, out = job
    from iscc.integrations import decompose_lineage_spatial
    t = REG.load(run, task)
    t.make_cell_data(max_cells=3000)
    dec = decompose_lineage_spatial(t, max_cells=600, seed=0)
    fl = dec.field_autocorr["hypoxia_level"]["lineage"]
    fs = dec.field_autocorr["hypoxia_level"]["spatial"]
    # the lineage weights of the decomposition, rebuilt on its cells, for the permutation test
    cd = t.cell_data
    gid = cd["cell_type"].iloc[:, 0].astype(str)
    gid.index = cd["cell_exp"].index
    clones = gid.reindex(dec.cells).to_numpy()
    present = list(dict.fromkeys(clones.tolist()))
    Dg = dec.tree.distance_matrix(present)
    gi = {g: i for i, g in enumerate(present)}
    cg = np.array([gi[g] for g in clones])
    W = np.exp(-Dg[np.ix_(cg, cg)] / 2.0)
    np.fill_diagonal(W, 0.0)
    f = cd["cell_microenv"]["hypoxia_level"].reindex(dec.cells).to_numpy(float)
    i_obs = _moran(W, f)
    rng = np.random.default_rng([0, 0xE4])
    null = np.array([_moran(W, rng.permutation(f)) for _ in range(N_PERM)])
    arch = "free" if task.startswith("free") else "ductal"
    sel = task.split("_")[1] if run == "R3" else None
    rec = dict(run=run, task=task, arch=arch, selection=sel,
               disp=float(task.split("_")[0][4:]) if run == "R1" else None,
               field_lin=float(fl), field_sp=float(fs), share=float(fl / (fl + fs)) if (fl + fs) else None,
               perm_I=i_obs, perm_matches=bool(abs(i_obs - fl) < 1e-6), perm_null_mean=float(null.mean()),
               perm_null_q95=float(np.quantile(null, 0.95)),
               perm_p=float((1 + (null >= i_obs).sum()) / (N_PERM + 1)),
               n_cells=int(len(dec.cells)), cancer=int(t.get_cancer_size()), pickle=_stamp(run, task),
               lineage_space_corr=float(dec.lineage_space_corr))
    os.makedirs(os.path.dirname(_cache(out, run, task)), exist_ok=True)
    with open(_cache(out, run, task), "w") as fh:
        json.dump(rec, fh)
    return f"{run}/{task}"


def _safe(job):
    try:
        return analyse(job), None
    except Exception as e:
        return f"{job[0]}/{job[1]}", f"{type(e).__name__}: {e}"


def todo(out):
    jobs = []
    for run in ("R1", "R3"):
        for task in REG.tasks(run):
            if not REG.grown(run, task):
                continue
            c = _cache(out, run, task)
            if os.path.exists(c):
                with open(c) as fh:
                    if json.load(fh).get("pickle") == _stamp(run, task):
                        continue
            jobs.append((run, task, out))
    return jobs


def compute(out, jobs):
    work = todo(out)
    print(f"{len(work)} tumours to (re)compute", flush=True)
    with mp.get_context("spawn").Pool(processes=jobs, maxtasksperchild=1) as pool:
        for key, err in pool.imap_unordered(_safe, work):
            print(f"  {key}: {err or 'ok'}", flush=True)


def panel(out):
    """Panel (i): the clone map and the hypoxia field of one ductal-field tumour."""
    import common as C
    run, task = PANEL_TASK
    t = REG.load(run, task)
    rgb = C.tissue_map(t, C.Clones(t))
    # the field the cells carry: the per-deme hypoxia the materialisation reads (microenv_truth)
    t.make_cell_data(max_cells=3000)
    G = t.grid_size
    hd = np.asarray(t.microenv_truth["hypoxia"], float)
    h = np.full((G, G), np.nan)
    for di in range(G * G):
        r, c = t.deme_coords[di]
        h[r, c] = hd[di]
    os.makedirs(os.path.join(out, "e"), exist_ok=True)
    np.savez_compressed(os.path.join(out, "e", "panel_i.npz"), clones=np.asarray(rgb, np.float32),
                        hypoxia=np.asarray(h, np.float32), run=run, task=task)
    print("panel (i) cached:", run, task, flush=True)


def _stalled(run, task):
    """A tumour whose registry record says it never reached its target size (it stalled, in the glands for
    the R1 seeds at about 11,200 cells; vault question 0f)."""
    p = os.path.splitext(REG.task_path(run, task))[0] + ".json"
    try:
        with open(p) as fh:
            return not bool(json.load(fh).get("reached_target", True))
    except OSError:
        return False


def load(out):
    d = os.path.join(out, "e", "cache")
    rows = [json.load(open(os.path.join(d, f))) for f in sorted(os.listdir(d))] if os.path.isdir(d) else []
    df = pd.DataFrame(rows)
    if len(df):
        df["group"] = [f"{r}:{a}:{s}" for r, a, s in zip(df["run"], df["arch"], df["selection"])]
        df["stalled"] = [_stalled(r, t) for r, t in zip(df["run"], df["task"])]
    return df


def summary(out):
    df = load(out)
    if not len(df):
        return {}
    s = {"n_tumours": int(len(df))}

    def block(sub):
        return dict(n=int(len(sub)), n_stalled=int(sub.stalled.sum()),
                    stalled=sorted(f"{r}/{t}" for r, t in zip(sub.run[sub.stalled], sub.task[sub.stalled])),
                    lineage_exceeds_spatial=sorted(f"{r}/{t}" for r, t in
                                                   zip(sub.run[sub.field_lin >= sub.field_sp],
                                                       sub.task[sub.field_lin >= sub.field_sp])),
                    max_lineage=round(float(sub.field_lin.max()), 3),
                    share_min=round(float(sub.share.min()), 3), share_max=round(float(sub.share.max()), 3),
                    share_mean=round(float(sub.share.mean()), 3),
                    spatial_exceeds_lineage_all=bool((sub.field_sp > sub.field_lin).all()),
                    n_spatial_exceeds=int((sub.field_sp > sub.field_lin).sum()),
                    perm_significant_0_05=int((sub.perm_p < 0.05).sum()),
                    perm_rebuilt_matches=bool(sub.perm_matches.all()))
    s["all"] = block(df)
    s["ductal_R1"] = block(df[df.run == "R1"])
    s["ductal_R1_and_R3_confined"] = block(df[df.arch == "ductal"])
    s["free_R3"] = block(df[df.arch == "free"])
    for run, arch, sel in GROUPS:
        sub = df[(df.run == run) & (df.arch == arch) & (df.selection.astype(str) == str(sel))]
        if len(sub):
            s[f"{run}_{arch}_{sel}"] = block(sub)
    s["all_spatial_exceeds_lineage"] = bool((df.field_sp > df.field_lin).all())
    from scipy.stats import mannwhitneyu
    a, b = df[df.run == "R1"].share, df[df.arch == "free"].share
    if len(a) and len(b):
        s["mwu_p_R1_vs_free"] = float(mannwhitneyu(a, b).pvalue)
    with open(os.path.join(out, "e", "summary.json"), "w") as fh:
        json.dump(s, fh, indent=1)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--part", default="all", choices=("all", "compute", "panel", "summary"))
    a = ap.parse_args()
    if a.part in ("all", "compute"):
        compute(a.out, a.jobs)
    if a.part in ("all", "panel"):
        panel(a.out)
    if a.part in ("all", "summary"):
        print(json.dumps(summary(a.out), indent=1), flush=True)


if __name__ == "__main__":
    main()
