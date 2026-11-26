"""Fig 4e: structure misleads a lineage method, read off the registry's own tumours.

The hypoxia field is environmental by construction, so any lineage autocorrelation it carries is the
confound itself. Per tumour, on 600 of its cancer cells (``iscc.integrations.decompose_lineage_spatial``,
the PEtracer-style decomposition of the old fig4_space_lineage.py, max_cells 3000 then 600, seed 0): the
field's Moran's I under the spatial and under the lineage weights, its lineage share
``I_lin / (I_lin + I_sp)``, and a permutation test of the lineage autocorrelation alone (the field's values
shuffled over the same cells, 999 times; one-sided p). The same for a positive control, a trait heritable
by construction: Brownian motion along the true genotype tree (unit variance per edge, seed fixed per
tumour), carried by every cell of a genotype, measured with the same weights on the same cells. And the
Moran's I of both fields over a grid of kernel widths (``LIN_SCALES`` in tree edges, ``SP_SCALES`` in demes,
the spatial coordinates jittered as the decomposition jitters them, with a seed of their own), since the
comparison of a lineage with a spatial autocorrelation depends on the two widths. Every statistic is
also computed with row-standardised weights (``_moran_rs``: each cell's relatives or neighbours averaged,
so the value is the slope of their mean on the cell's own value, within -1 and 1): with raw weights the
cells with the most close relatives dominate, which in a territorial tumour are the cells of its largest
clade, near the mean of any trait, and a trait heritable by construction can then score below an
environmental field.

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
LIN_SCALES = (1.0, 2.0, 5.0, 10.0)      # tree edges; the decomposition's width is 2
SP_SCALES = (1.0, 2.0, 4.0)             # demes; the decomposition's width is 2
PANEL_TASK = ("R1", "disp0.1_seed1")
GROUPS = [("R1", "ductal", None), ("R3", "ductal", "selected"), ("R3", "ductal", "neutral"),
          ("R3", "free", "selected"), ("R3", "free", "neutral")]


def _moran(W, z):
    z = np.asarray(z, float) - np.mean(z)
    den = float(z @ z)
    return 0.0 if den <= 0 or W.sum() <= 0 else float(len(z) / W.sum() * (z @ (W @ z)) / den)


def _brownian(tree, ids, rng):
    """Brownian motion along ``tree`` (unit variance per parent-child edge), read at the genotypes ``ids``:
    a trait heritable by construction, each genotype carrying its parent's value plus a step."""
    parent = {str(k): (None if v is None else str(v)) for k, v in tree.parent.items()}
    val, out = {}, []
    for g in ids:
        path, x = [], str(g)
        while x is not None and x not in val:
            path.append(x)
            x = parent.get(x)
        base = 0.0 if x is None else val[x]
        for y in reversed(path):
            base += float(rng.normal())
            val[y] = base
        out.append(val[str(g)])
    return np.array(out)


def _moran_rs(W, z):
    """Moran's I with row-standardised weights."""
    rs = W.sum(axis=1, keepdims=True)
    Wr = np.divide(W, rs, out=np.zeros_like(W), where=rs > 0)
    z = np.asarray(z, float) - np.mean(z)
    den = float(z @ z)
    return 0.0 if den <= 0 else float((z @ (Wr @ z)) / den)


def _stamp(run, task):
    st = os.stat(REG.task_path(run, task))
    return dict(size=int(st.st_size), mtime=int(st.st_mtime))


def _cache(out, run, task):
    return os.path.join(out, "e", "cache", f"{run}__{task}.json")


def analyse(job):
    run, task, out = job
    from iscc.integrations import decompose_lineage_spatial
    from iscc.integrations.lineage import to_lineage_tree
    t = REG.load(run, task)
    t.make_cell_data(max_cells=3000)
    # the positive control, as one more field of the cells so it takes the decomposition's own weights
    cd0 = t.cell_data
    g_all = cd0["cell_type"].iloc[:, 0].astype(str).to_numpy()
    trait = _brownian(to_lineage_tree(t), g_all, np.random.default_rng([0, 0xB0]))
    cd0["cell_microenv"] = cd0["cell_microenv"].assign(heritable_control=trait)
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
    h = cd["cell_microenv"]["heritable_control"].reindex(dec.cells).to_numpy(float)
    h_obs = _moran(W, h)
    h_null = np.array([_moran(W, rng.permutation(h)) for _ in range(N_PERM)])
    # the two autocorrelations of both fields over the grid of kernel widths
    Dl = Dg[np.ix_(cg, cg)]
    crd = cd["cell_crd"].reindex(dec.cells).to_numpy(float)
    crd = crd + np.random.default_rng([0, 0x5C]).normal(0.0, 0.15, crd.shape)
    d2 = ((crd[:, None, :] - crd[None, :, :]) ** 2).sum(-1)
    widths = {}
    for ls in LIN_SCALES:
        Wl = np.exp(-Dl / ls)
        np.fill_diagonal(Wl, 0.0)
        widths[f"lineage_{ls:g}"] = dict(field=_moran(Wl, f), control=_moran(Wl, h),
                                         field_rs=_moran_rs(Wl, f), control_rs=_moran_rs(Wl, h))
    for ls in SP_SCALES:
        Ws = np.exp(-d2 / (2.0 * ls * ls))
        np.fill_diagonal(Ws, 0.0)
        widths[f"spatial_{ls:g}"] = dict(field=_moran(Ws, f), control=_moran(Ws, h),
                                         field_rs=_moran_rs(Ws, f), control_rs=_moran_rs(Ws, h))
    # the row-standardised statistics at the decomposition's widths, with their permutation tests
    Ws2 = np.exp(-d2 / 8.0)
    np.fill_diagonal(Ws2, 0.0)
    rs = dict(field_lin=_moran_rs(W, f), field_sp=_moran_rs(Ws2, f),
              control_lin=_moran_rs(W, h), control_sp=_moran_rs(Ws2, h))
    for name, z in (("field", f), ("control", h)):
        obs_rs = rs[f"{name}_lin"]
        nul = np.array([_moran_rs(W, rng.permutation(z)) for _ in range(N_PERM)])
        rs[f"{name}_perm_p"] = float((1 + (nul >= obs_rs).sum()) / (N_PERM + 1))
    arch = "free" if task.startswith("free") else "ductal"
    sel = task.split("_")[1] if run == "R3" else None
    rec = dict(run=run, task=task, arch=arch, selection=sel,
               disp=float(task.split("_")[0][4:]) if run == "R1" else None,
               field_lin=float(fl), field_sp=float(fs), share=float(fl / (fl + fs)) if (fl + fs) else None,
               perm_I=i_obs, perm_matches=bool(abs(i_obs - fl) < 1e-6), perm_null_mean=float(null.mean()),
               perm_null_q95=float(np.quantile(null, 0.95)),
               perm_p=float((1 + (null >= i_obs).sum()) / (N_PERM + 1)),
               control_lin=float(dec.field_autocorr["heritable_control"]["lineage"]),
               control_sp=float(dec.field_autocorr["heritable_control"]["spatial"]),
               control_perm_p=float((1 + (h_null >= h_obs).sum()) / (N_PERM + 1)),
               widths=widths, rs=rs,
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
                    rec = json.load(fh)
                if rec.get("pickle") == _stamp(run, task) and "rs" in rec:   # a cache without the control
                    continue                                                  # and its statistics is redone
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
    the R1 seeds at about 11,200 cells)."""
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


def _rs_block(sub):
    """The row-standardised statistics of a group of tumours (the ``rs`` record of each)."""
    r = pd.DataFrame([x for x in sub["rs"]]) if "rs" in sub else pd.DataFrame()
    if not len(r):
        return {}
    return dict(field_lin_mean=round(float(r.field_lin.mean()), 3), field_sp_mean=round(float(r.field_sp.mean()), 3),
                control_lin_mean=round(float(r.control_lin.mean()), 3),
                control_sp_mean=round(float(r.control_sp.mean()), 3),
                field_called_heritable=int((r.field_perm_p < 0.05).sum()),
                control_called_heritable=int((r.control_perm_p < 0.05).sum()),
                field_lineage_above_spatial=int((r.field_lin > r.field_sp).sum()),
                control_lineage_above_spatial=int((r.control_lin > r.control_sp).sum()),
                field_lineage_above_control_lineage=int((r.field_lin > r.control_lin).sum()))


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
                    perm_rebuilt_matches=bool(sub.perm_matches.all()),
                    control_perm_significant_0_05=int((sub.control_perm_p < 0.05).sum()),
                    control_lineage_exceeds_spatial=int((sub.control_lin > sub.control_sp).sum()),
                    control_lineage_mean=round(float(sub.control_lin.mean()), 3),
                    control_spatial_mean=round(float(sub.control_sp.mean()), 3),
                    rs=_rs_block(sub))
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
