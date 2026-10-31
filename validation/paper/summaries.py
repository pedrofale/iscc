"""Small per-tumour statistics, computed once and cached, so figures never reload the registry.

A registry tumour is tens to hundreds of MB; a figure that reads a hundred of them in one process
exhausts a laptop's memory. Each statistic a figure needs from many tumours is computed here once,
written to ``analysis_data/paper/<tier>/summaries/<run>/<task>.json``, and read back as a small
table. ``python summaries.py [RUNS] [--jobs N]`` fills the cache, one short-lived worker process per
task so memory is returned between tumours.

Statistics per tumour:
  * ``n0, D0`` / ``n2, D2``: Noble's evolutionary-mode indices at the full census and at a 2%
    detection limit (combinations below it pooled, not dropped).
  * ``seg_cn``, ``seg_net``: count-weighted mean copy number and net oncogenic content per segment.
  * ``wgd_frac``: fraction of cancer cells carrying a whole-genome duplication.
  * ``assortment``: fraction of neighbouring cancer demes sharing their dominant clone.
  * ``cancer``, ``stroma_frac``: cancer cells, and the share of them in stroma.
"""
import argparse
import json
import os
import sys
from collections import Counter
import multiprocessing as mp

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import registry as REG  # noqa: E402

OUT = os.path.join(REG.OUT, "summaries")
DETECTION = 0.02


def path(run, task):
    return os.path.join(OUT, run, f"{task}.json")


def _mode(t, detection):
    from iscc.inference.indices import driver_combination_counts, inverse_simpson
    combos = driver_combination_counts(t)
    total = sum(combos.values())
    if not total:
        return None, None
    n = sum(len(k) * c for k, c in combos.items()) / total
    if detection > 0:
        kept = [c for c in combos.values() if c / total >= detection]
        pooled = total - sum(kept)
        counts = kept + ([pooled] if pooled else [])
    else:
        counts = list(combos.values())
    return float(n), float(inverse_simpson(counts))


def summarise(run, task):
    import common as C
    t = REG.load(run, task)
    counts = C.live_cancer_counts(t)
    tot = sum(counts.values())
    seg = t.n_segments
    size = t.n_genes // seg
    onc = np.bincount(np.asarray(t.selection.get_oncogenes(), dtype=int) // size, minlength=seg)
    tsg = np.bincount(np.asarray(t.selection.get_tsgs(), dtype=int) // size, minlength=seg)
    w = np.array(list(counts.values()), dtype=float)
    cn = np.array([t.genotypes[g].genome_summary["seg_cns"] for g in counts], dtype=float)
    seg_cn = ((cn * w[:, None]).sum(0) / w.sum()).tolist() if len(w) else [2.0] * seg
    wgd = sum(c for g, c in counts.items() if getattr(t.genotypes[g], "is_wgd", False)) / max(tot, 1)
    n0, D0 = _mode(t, 0.0)
    n2, D2 = _mode(t, DETECTION)
    cl = C.Clones(t)
    G = t.grid_size
    dom, stroma = {}, 0
    gland = np.asarray(t.gland_id) if t.gland_id is not None else None
    for di in range(G * G):
        cc = Counter()
        for g, c in t.demes[di].items():
            if t._is_cancer(g):
                cc[cl.of(g)] += c
        if cc:
            dom[di] = cc.most_common(1)[0][0]
            if gland is not None and gland[di] == -1:
                stroma += sum(cc.values())
    same = pairs = 0
    for di, lab in dom.items():
        for nb in t._neighbors(di):
            if nb in dom and nb > di:
                pairs += 1
                same += dom[nb] == lab
    rec = dict(run=run, task=task, cancer=int(tot), n0=n0, D0=D0, n2=n2, D2=D2, seg_cn=seg_cn,
               seg_net=(onc - tsg).tolist(), wgd_frac=float(wgd),
               assortment=float(same / pairs) if pairs else None,
               stroma_frac=float(stroma / max(tot, 1)), n_clones=len(cl.labels),
               coarsened=bool(t.coarsen_passengers))
    if not t.coarsen_passengers:                 # exact passengers: the neutral spectrum is meaningful
        from iscc.validation import neutral_sfs_rsq, population_vaf, site_frequency_spectrum
        vaf = population_vaf(t)
        r, _ = neutral_sfs_rsq(vaf, 0.12, 0.24, 12)
        grid, M = site_frequency_spectrum(vaf[vaf > 0], 0.12, 0.24, 12)
        rec.update(sfs_rsq=float(r), sfs_grid=np.asarray(grid, dtype=float).tolist(),
                   sfs_M=np.asarray(M, dtype=float).tolist())
    os.makedirs(os.path.dirname(path(run, task)), exist_ok=True)
    with open(path(run, task), "w") as f:
        json.dump(rec, f)
    return rec


def _safe(job):
    r, t = job
    from memguard import wait_for_memory
    wait_for_memory(label=f"{r}/{t}")
    try:
        summarise(r, t)
        return r, t, None
    except Exception as e:
        return r, t, f"{type(e).__name__}: {e}"


def load(run):
    """The cached summaries of a run as a list of dicts (missing tasks skipped)."""
    out = []
    for task in REG.tasks(run):
        p = path(run, task)
        if os.path.exists(p):
            with open(p) as f:
                out.append(json.load(f))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="?", default="all")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    runs = list(REG.RUNS) if a.runs == "all" else a.runs.split(",")
    todo = [(r, t) for r in runs for t in REG.tasks(r)
            if os.path.exists(REG.task_path(r, t)) and (a.force or not os.path.exists(path(r, t)))]
    print(f"{len(todo)} summaries to compute", flush=True)
    # maxtasksperchild=1: a fresh process per tumour, so memory is returned after each one
    with mp.get_context("spawn").Pool(processes=a.jobs, maxtasksperchild=1) as pool:
        for r, t, err in pool.imap_unordered(_safe, todo):
            if err:
                print(f"  !! {r}/{t}: {err}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
