"""S4 Fig: the cohort's ground truth (R6, six patients sharing one configuration).

The driver layout and baseline expression come from a configuration-determined layout seed, so the
same genes are drivers in every patient, while each patient's evolution seed is private. Shown:
(a) each patient's tumour; (b) the copy-number profiles of every patient's clones (at least 4% of
the cancer cells); (c) which driver genes are carried by at least 10% of each patient's cancer cells,
sorted by how many patients share them.

Driver frequencies are read from the genotype counts, not from a sample. The batch-integration and
demultiplexing benchmarks over this cohort need the published tools and are run on the cluster.

Writes ``figures/paper1/supp_s4_cohort.pdf``; per-patient results are cached in ``derived/supp_cohort``.
Run:  python validation/paper/supp_cohort.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import common as C  # noqa: E402
import registry as REG  # noqa: E402
import style as S  # noqa: E402

CACHE = os.path.join(C.DERIVED, "supp_cohort")
HIT = 0.10


def patient(task):
    from memguard import wait_for_memory
    from iscc.inference import driver_combination_counts
    wait_for_memory(label=f"R6/{task}")
    try:
        t = REG.load("R6", task)
        cl = C.Clones(t)
        counts = C.live_cancer_counts(t)
        total = sum(counts.values())
        # clone profiles: the copy number of each clone's root genotype
        prof = {lab: list(map(int, t.genotypes[g].genome_summary["seg_cns"]))
                for lab, g in zip(cl.labels, cl.roots) if g in t.genotypes}
        freq = {}
        for combo, n in driver_combination_counts(t).items():
            for d in combo:
                freq[int(d)] = freq.get(int(d), 0) + n
        hits = sorted(d for d, n in freq.items() if n / total >= HIT)
        rgb = C.tissue_map(t, cl)
        np.save(os.path.join(CACHE, f"{task}_map.npy"), rgb.astype(np.float32))
        rec = dict(task=task, cancer=int(total), n_clones=len(cl.labels), profiles=prof, hits=hits,
                   n_drivers=int(len(np.concatenate([t.selection.get_oncogenes(), t.selection.get_tsgs()]))))
        with open(os.path.join(CACHE, f"{task}.json"), "w") as f:
            json.dump(rec, f)
        return task, None
    except Exception as e:
        return task, f"{type(e).__name__}: {e}"


def render():
    S.setup()
    tasks = [k for k in REG.tasks("R6") if os.path.exists(os.path.join(CACHE, f"{k}.json"))]
    recs = []
    for k in tasks:
        with open(os.path.join(CACHE, f"{k}.json")) as f:
            recs.append(json.load(f))
    fig = S.figure("double", 98)
    S.panel_letter(fig, "a", 2, 3)
    for i, r in enumerate(recs):
        ax = S.ax_mm(fig, 6 + i * 29.5, 7, 26, 26)
        ax.imshow(np.load(os.path.join(CACHE, f"{r['task']}_map.npy")), interpolation="nearest")
        S.despine(ax, False, False)
        ax.set_title(f"patient {i + 1}", fontsize=5.5)
    # (b) clone copy-number profiles, grouped by patient
    S.panel_letter(fig, "b", 2, 39)
    ax = S.ax_mm(fig, 14, 43, 60, 48)
    rows, ylab, bounds = [], [], []
    for i, r in enumerate(recs):
        for lab, p in r["profiles"].items():
            rows.append(p)
            ylab.append(f"P{i + 1} {lab.replace('clone ', 'c')}")
        bounds.append(len(rows))
    ax.imshow(np.array(rows), cmap=S.DIVERGING, norm=S.cn_norm(6), aspect="auto", interpolation="nearest")
    for b in bounds[:-1]:
        ax.axhline(b - 0.5, color=S.INK, lw=0.6)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(ylab, fontsize=3.6)
    ax.set_xlabel("Segment")
    ax.set_title("Copy number of each clone", fontsize=6)
    # (c) driver genes hit, by patient, sorted by recurrence
    S.panel_letter(fig, "c", 84, 39)
    ax = S.ax_mm(fig, 96, 43, 82, 48)
    genes = sorted({d for r in recs for d in r["hits"]})
    H = np.array([[d in set(r["hits"]) for d in genes] for r in recs], dtype=float)
    order = np.lexsort((-H.argmax(0), -H.sum(0)))
    H = H[:, order]
    ax.imshow(H, cmap="Greys", vmin=0, vmax=1.4, aspect="auto", interpolation="nearest")
    rec_n = int((H.sum(0) >= 2).sum())
    ax.axvline(rec_n - 0.5, color=S.SOURCE["iscc"], lw=0.8)
    ax.text(rec_n / 2, -0.8, "recurrent", ha="center", fontsize=4.8, color=S.SOURCE["iscc"])
    ax.text(rec_n + (len(genes) - rec_n) / 2, -0.8, "private", ha="center", fontsize=4.8, color=S.GREY)
    ax.set_yticks(range(len(recs)))
    ax.set_yticklabels([f"patient {i + 1}" for i in range(len(recs))], fontsize=4.8)
    ax.set_xticks([])
    ax.set_xlabel(f"Driver genes carried by at least {HIT:.0%} of a patient's cancer cells")
    path = S.save(fig, "supp_s4_cohort")
    out = dict(n_patients=len(recs), genes_hit=len(genes), recurrent=rec_n,
               clones=[r["n_clones"] for r in recs], drivers_in_layout=recs[0]["n_drivers"] if recs else None,
               profiles_shared=len({tuple(p) for r in recs for p in r["profiles"].values()}) <
               sum(len(r["profiles"]) for r in recs))
    print("wrote", path)
    print(out)
    return out


def main():
    import multiprocessing as mp
    os.makedirs(CACHE, exist_ok=True)
    todo = [k for k in REG.tasks("R6") if not os.path.exists(os.path.join(CACHE, f"{k}.json"))]
    with mp.get_context("spawn").Pool(processes=1, maxtasksperchild=1) as pool:
        for task, err in pool.imap_unordered(patient, todo):
            print(f"  {task}: {err or 'ok'}", flush=True)
    render()


if __name__ == "__main__":
    main()
