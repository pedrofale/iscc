"""S1 Fig: tau-leaping against the exact engine, in the paper regime.

The paper grows every tumour with tau-leaping (one leap of all clones per tau generations). The
exact engine realises the same birth, death, mutation and dispersal process one event at a time.
The two are not byte-identical (they draw different random variables), so they are compared by
distribution: each grows the small-scale paper regime to the same number of cancer cells over many
seeds, and the clone-size distribution, clonal diversity and copy-number burden are compared.

The comparison stops at 5,000 cancer cells, because the exact engine's cost grows with the number of
events, and runs with the founding acinus's confinement switched off. The two engines advance the
confinement clock on different scales (the tau engine by one per step of tau generations, the exact
engine by one per event per cancer cell), so with confinement on they open the acinus at different
times and the comparison would test the clock, not the growth algorithm (issues note, N8).

Writes ``figures/paper1/supp_s1_tau.pdf``; per-tumour results are cached in ``derived/supp_tau``.
Run:  python validation/paper/supp_tau.py [--seeds 30]
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import common as C  # noqa: E402
import regime as RG  # noqa: E402
import realistic_regime as RR  # noqa: E402
import style as S  # noqa: E402

CACHE = os.path.join(C.DERIVED, "supp_tau_noconf")
TARGET = 5_000


def grow(mode, seed):
    from iscc.tumor.models import GenotypeTumor
    cfg = RG.config(RG.T["sweep_scale"], spatial={"origin_confinement": None})
    tau = RR.TAU if mode == "tau" else (float(mode[3:]) if mode.startswith("tau") else RR.TAU)
    engine = "tau" if mode.startswith("tau") else "exact"
    t = GenotypeTumor(seed=seed, update_mode=engine, tau=tau, coarsen_passengers=RR.COARSEN,
                      max_cells=1, founder_mutations=RR.FOUNDER_MUTATIONS, germline_params=RR.GERMLINE,
                      **cfg)
    if engine == "tau":
        steps = 0
        while 0 < t.get_cancer_size() < TARGET and steps < 8_000:
            t.grow(n_steps=1, seed=seed)
            steps += 1
    else:
        rng, n = t.rng, 0
        while 0 < t.get_cancer_size() < TARGET and n < 20_000_000:
            t.update(rng)
            t.step += 1
            n += 1
    return t


def metrics(t):
    counts = C.live_cancer_counts(t)
    w = np.array(list(counts.values()), dtype=float)
    if not w.sum():
        return None
    p = w / w.sum()
    cn = np.array([t.genotypes[g].genome_summary["seg_cns"] for g in counts], dtype=float)
    fga = ((cn != 2).mean(1) * w).sum() / w.sum()                     # fraction of segments altered
    from iscc.inference import mode_indices
    m = mode_indices(t)
    return dict(cancer=int(w.sum()), n_genotypes=int(len(w)), top=float(p.max()),
                D_genotypes=float(1 / np.sum(p ** 2)), n_drivers=float(m["n"]), D_drivers=float(m["D"]),
                fga=float(fga), fractions=np.sort(p)[::-1][:200].tolist())


def run_one(job):
    mode, seed = job
    path = os.path.join(CACHE, f"{mode}_{seed}.json")
    if os.path.exists(path):
        return job, None
    try:
        t = grow(mode, seed)
        rec = metrics(t)
        os.makedirs(CACHE, exist_ok=True)
        with open(path, "w") as f:
            json.dump(dict(mode=mode, seed=seed, **(rec or {"extinct": True})), f)
        return job, None
    except Exception as e:
        return job, f"{type(e).__name__}: {e}"


def load():
    out = []
    if os.path.isdir(CACHE):
        for fn in sorted(os.listdir(CACHE)):
            with open(os.path.join(CACHE, fn)) as f:
                out.append(json.load(f))
    return [r for r in out if not r.get("extinct")]


def render():
    import pandas as pd
    from scipy.stats import mannwhitneyu
    S.setup()
    df = pd.DataFrame(load())
    fig = S.figure("double", 52)
    MODES = [m for m in ("exact", "tau0.25", "tau") if m in set(df["mode"])]
    names = {"exact": "exact", "tau0.25": "τ = 0.25", "tau": f"τ = {RR.TAU:g}"}
    col = {"exact": S.GREY, "tau0.25": S.CLONE_FAMILIES[0][2], "tau": S.SOURCE["iscc"]}
    out = {}
    # (a) clone-size distribution, pooled over seeds
    S.panel_letter(fig, "a", 2, 3)
    ax = S.ax_mm(fig, 12, 7, 34, 34)
    for mode in MODES:
        f = np.concatenate([r for r in df[df["mode"] == mode].fractions])
        f = np.sort(f)
        ax.plot(f, np.linspace(0, 1, len(f)), color=col[mode], lw=1.0, label=names[mode])
    ax.set_xscale("log")
    ax.set_xlabel("Genotype fraction of cancer cells")
    ax.set_ylabel("Cumulative fraction")
    ax.legend(fontsize=5)
    # tested per tumour, not on the pooled fractions (which are not independent within a tumour)

    out["cancer"] = df.groupby("mode").cancer.median().to_dict()
    # (b)-(e) per-tumour summaries
    for k, (L, key, lab) in enumerate((("b", "D_drivers", "Driver diversity, D"),
                                       ("c", "n_drivers", "Drivers per cell, n"),
                                       ("d", "top", "Largest genotype fraction"),
                                       ("e", "fga", "Fraction of genome altered"))):
        x0 = 58 + k * 31
        S.panel_letter(fig, L, x0 - 9, 3)
        ax = S.ax_mm(fig, x0, 7, 22, 34)
        for j, mode in enumerate(MODES):
            v = df[df["mode"] == mode][key].values
            ax.scatter(j + np.random.default_rng(j).uniform(-0.15, 0.15, len(v)), v, s=3,
                       color=col[mode], alpha=0.8)
            ax.plot([j - 0.3, j + 0.3], [np.median(v)] * 2, color=S.INK, lw=0.8)
        ax.set_xticks(range(len(MODES)))
        ax.set_xticklabels([names[m] for m in MODES], fontsize=4.5)
        ax.set_xlim(-0.6, len(MODES) - 0.4)
        ax.set_ylabel(lab)
        e = df[df["mode"] == "exact"][key]
        out[key] = {"exact": round(float(e.median()), 3)}
        for m in MODES[1:]:
            v = df[df["mode"] == m][key]
            out[key][m] = round(float(v.median()), 3)
            out[key][f"p_{m}"] = round(float(mannwhitneyu(e, v).pvalue), 3)
    out["n"] = df.groupby("mode").size().to_dict()
    path = S.save(fig, "supp_s1_tau")
    print("wrote", path)
    print(out)
    return out


def main():
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--render-only", action="store_true")
    a = ap.parse_args()
    if not a.render_only:
        jobs = [(m, s) for s in range(1, a.seeds + 1) for m in ("exact", "tau", "tau0.25")]
        with mp.get_context("spawn").Pool(processes=1, maxtasksperchild=4) as pool:
            for job, err in pool.imap_unordered(run_one, jobs):
                if err:
                    print(f"  {job}: {err}", flush=True)
    render()


if __name__ == "__main__":
    main()
