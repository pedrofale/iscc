"""S2 Fig: multi-region sample trees are not phylogenies (Alves, Prieto and Posada 2017).

A bulk biopsy region is a mixture of clones, so a tree built from regional mutation profiles reflects
similarity, not descent, and infers parallel origins for mutations that arose once. iscc scores this
against its answer key: the true number of origins of every locus on the true clone tree. Only loci
that arose once are scored (the 6,000-gene genome makes most loci recurrent at tumour scale, see the
Methods).

For every R1 and R3 tumour: eight regions of radius 3 demes, centred on cancer cells of a 3,000-cell
sample, the naive
sample tree (neighbour joining on regional presence) against the tree of the clones deconvolved from
the regions with the oracle, and each region's clone admixture.

Writes ``figures/paper1/supp_s2_multiregion.pdf``; per-tumour results are cached in
``derived/supp_multiregion``.
Run:  python validation/paper/supp_multiregion.py
"""
import json
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

CACHE = os.path.join(C.DERIVED, "supp_multiregion")
N_REGIONS, RADIUS = 8, 3.0
EXAMPLE = ("R1", "disp0.1_seed1")


def admixture(cell_data, regions, gid, min_major=5):
    """Mean effective number of clones per region (exp of the clone-fraction entropy)."""
    pos = {c: i for i, c in enumerate(cell_data["cell_snv"].index)}
    vc = pd.Series(gid).value_counts()
    major = set(vc[vc >= min_major].index)
    eff = []
    for r in sorted(pd.unique(regions.values)):
        cg = [gid[pos[c]] for c in regions.index[regions.values == r]]
        p = pd.Series([g if g in major else "minor" for g in cg]).value_counts(normalize=True).values
        eff.append(float(np.exp(-(p * np.log(p)).sum())))
    return float(np.mean(eff))


def tumour(key):
    from memguard import wait_for_memory
    from iscc import integrations as ig
    from iscc.sample.biopsy.biopsy import Biopsy
    run, task = key
    wait_for_memory(label=f"{run}/{task}")
    try:
        t = REG.load(run, task)
        t.make_cell_data(max_cells=3000)
        ak = ig.true_origin_counts(t)
        # regions are taken from the tumour, as a multi-region study samples it: centres are drawn
        # among the cancer cells only (the sampler draws them from whatever cell_crd holds)
        gid = t.cell_data["cell_type"].iloc[:, 0].astype(str)
        cancer = [c for c, g in gid.items() if g in t.genotypes and t._is_cancer(g)]
        cd = dict(t.cell_data)
        cd["cell_crd"] = t.cell_data["cell_crd"].loc[cancer]
        _, rs, geom = Biopsy(cd, np.random.default_rng(0)).sample(
            biopsy_type="multiregion", n_regions=N_REGIONS, radius=RADIUS)
        res = ig.multiregion_phylogeny(t, rs, answer_key=ak)
        rec = dict(run=run, task=task, arch="free" if task.startswith("free") else "ductal",
                   naive=float(res["naive"]["rate"]), fix=float(res["fix"]["rate"]),
                   n_regions=int(res["n_regions"]), n_clones=int(res["n_clones"]),
                   single=int(ak["single"].sum()), admixture=admixture(t.cell_data, rs, ak["gid"]))
        if key == EXAMPLE:
            rec["map"] = "map.npy"
            np.save(os.path.join(CACHE, "map.npy"), C.tissue_map(t, C.Clones(t)).astype(np.float32))
            rec["centres"] = geom["centers"]
        with open(os.path.join(CACHE, f"{run}__{task}.json"), "w") as f:
            json.dump(rec, f)
        return key, None
    except Exception as e:
        return key, f"{type(e).__name__}: {e}"


def render():
    from scipy.stats import spearmanr, wilcoxon
    S.setup()
    rows = []
    for fn in sorted(os.listdir(CACHE)):
        if fn.endswith(".json"):
            with open(os.path.join(CACHE, fn)) as f:
                rows.append(json.load(f))
    df = pd.DataFrame(rows)
    fig = S.figure("double", 52)
    colour = {"ductal": S.CLONE_FAMILIES[0][1], "free": S.CLONE_FAMILIES[2][1]}
    ex = next((r for r in rows if (r["run"], r["task"]) == EXAMPLE and "centres" in r), None)
    S.panel_letter(fig, "a", 2, 3)
    ax = S.ax_mm(fig, 8, 7, 36, 36)
    if ex:
        ax.imshow(np.load(os.path.join(CACHE, ex["map"])), interpolation="nearest")
        for k, (r, c) in enumerate(ex["centres"]):
            ax.add_patch(__import__("matplotlib").patches.Circle((c, r), RADIUS, fill=False, lw=0.8,
                                                                 color=S.INK))
            ax.text(c, r, str(k + 1), ha="center", va="center", fontsize=4.5, color=S.INK)
    S.despine(ax, False, False)
    ax.set_title("eight regions over the clones", fontsize=5.5)
    S.panel_letter(fig, "b", 52, 3)
    ax = S.ax_mm(fig, 62, 7, 34, 36)
    for _, r in df.iterrows():
        ax.plot([0, 1], [r.naive, r.fix], color=colour[r.arch], lw=0.5, alpha=0.6, marker="o", ms=1.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["region\nsample tree", "deconvolved\nclone tree"], fontsize=5)
    ax.set_xlim(-0.4, 1.4)
    ax.set_ylabel("Single-origin mutations\ncalled parallel")
    S.panel_letter(fig, "c", 108, 3)
    ax = S.ax_mm(fig, 120, 7, 36, 36)
    for arch, sub in df.groupby("arch"):
        ax.scatter(sub.admixture, sub.naive, s=5, color=colour[arch], edgecolors=S.INK, lw=0.2,
                   label=f"{'ductal field' if arch == 'ductal' else 'free growth'} ({len(sub)})")
    ax.set_xlabel("Clones per region (effective)")
    ax.set_ylabel("Called parallel, sample tree")
    ax.legend(fontsize=4.5)
    out = dict(n=len(df), naive_mean=round(float(df.naive.mean()), 3), fix_mean=round(float(df.fix.mean()), 3),
               naive_range=(round(float(df.naive.min()), 3), round(float(df.naive.max()), 3)),
               fix_max=round(float(df.fix.max()), 3),
               wilcoxon_p=float(wilcoxon(df.naive, df.fix).pvalue) if (df.naive != df.fix).any() else None,
               spearman_admixture=tuple(round(float(v), 3) for v in spearmanr(df.admixture, df.naive)),
               single_loci=(int(df.single.min()), int(df.single.max())),
               by_arch=df.groupby("arch")[["naive", "admixture"]].mean().round(3).to_dict())
    path = S.save(fig, "supp_s2_multiregion")
    print("wrote", path)
    print(out)
    return out


def main():
    import multiprocessing as mp
    os.makedirs(CACHE, exist_ok=True)
    todo = [(r, k) for r in ("R1", "R3") for k in REG.tasks(r)
            if os.path.exists(REG.task_path(r, k)) and not os.path.exists(os.path.join(CACHE, f"{r}__{k}.json"))]
    print(f"{len(todo)} tumours", flush=True)
    with mp.get_context("spawn").Pool(processes=1, maxtasksperchild=1) as pool:
        for key, err in pool.imap_unordered(tumour, todo):
            if err:
                print(f"  {key[0]}/{key[1]}: {err}", flush=True)
    render()


if __name__ == "__main__":
    main()
