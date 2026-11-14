"""Score the copy-number tools' common output (calls.py) against a dataset's truth, and compare the two
arms of the programme-permutation contrast (P2).

Usage::

    python score_tools.py score   <dataset_dir> <runs_dir> <out.csv>
    python score_tools.py compare <truth_dir> <runs_generated> <runs_permuted> <out.csv>

``<dataset_dir>`` holds ``visium/`` and ``scrna/`` (iscc.data.human writers) and ``truth_extra/``
(make_layer_dataset.py: ``<assay>_barcodes.tsv`` with each barcode's D9 clone, tumour fraction, hypoxia
and epithelial-wall flag, and ``<assay>_arm_cn.tsv`` with its true copy number per arm). Run folders are
named ``<tool>-<assay>[-<variant>]``.

``score`` writes one row per run:
- ``auc``: malignant against normal, from ``p_malignant`` (or the 0/1 call). A spot is malignant when
  most of its cells are tumour.
- ``clone_ari``: the tool's clone labels against the D9 clones, over the tumour barcodes that carry
  both. A spot takes the clone of most of its tumour cells.
- ``cn_r``: Pearson correlation of the tool's per-arm value with the truth over tumour barcodes x arms,
  against ``log2(CN / 2)`` for the log-ratio and relative scales and against CN for ``total_cn``.
- ``wall_s`` from ``run.json``.

``compare`` writes, per tool, the change from the generated to the permuted (or noise) arm, overall and within
the hypoxic core (top decile of hypoxia among tumour barcodes) and at the epithelial wall: the share of
barcodes whose malignant call flips, the ARI between the two arms' clone labels, and the mean absolute
change of the per-arm value.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, roc_auc_score


def _runs(runs_dir):
    """Finished runs only: a run without its run.json is still going, or failed."""
    for name in sorted(os.listdir(runs_dir)):
        p = os.path.join(runs_dir, name, "calls.tsv")
        if os.path.exists(p) and os.path.exists(os.path.join(runs_dir, name, "run.json")):
            assay = "visium" if "visium" in name else "scrna"
            yield name, assay, os.path.join(runs_dir, name)


def _truth(truth_dir, assay):
    bc = pd.read_csv(os.path.join(truth_dir, f"{assay}_barcodes.tsv"), sep="\t", dtype={"clone": str})
    cn = pd.read_csv(os.path.join(truth_dir, f"{assay}_arm_cn.tsv"), sep="\t")
    return bc.set_index("barcode"), cn


def _read(run):
    calls = pd.read_csv(os.path.join(run, "calls.tsv"), sep="\t", dtype={"clone": str}).set_index("barcode")
    p = os.path.join(run, "cn_arm.tsv")
    cn = pd.read_csv(p, sep="\t") if os.path.exists(p) and os.path.getsize(p) > 0 else None
    with open(os.path.join(run, "run.json")) as f:
        info = json.load(f)
    return calls, cn, info


def _score(calls, cn, bc, true_cn):
    out = {}
    y = bc["malignant"].reindex(calls.index)
    ok = y.notna().to_numpy()
    s = calls["p_malignant"] if calls["p_malignant"].notna().any() else calls["malignant"]
    s = s.astype(float)
    ok &= s.notna().to_numpy()
    yy = y[ok].astype(int)
    out["n"] = int(ok.sum())
    out["auc"] = float(roc_auc_score(yy, s[ok])) if yy.nunique() == 2 else np.nan
    out["n_called_malignant"] = int(np.nansum(calls["malignant"].astype(float))) if calls["malignant"].notna().any() else np.nan
    tum = bc.index[bc["malignant"].astype(bool) & bc["clone"].notna()]
    have = calls["clone"].reindex(tum)
    m = have.notna()
    out["n_clones"] = int(calls["clone"].dropna().nunique())
    out["clone_ari"] = (float(adjusted_rand_score(bc.loc[tum[m], "clone"], have[m])) if m.sum() > 1 else np.nan)
    out["n_true_clones"] = int(bc.loc[tum, "clone"].nunique())
    if cn is not None and len(cn):
        scale = cn["scale"].iloc[0]
        tt = true_cn.copy()
        tt["truth"] = tt["cn"] if scale == "total_cn" else np.log2(np.maximum(tt["cn"], 0.05) / 2.0)
        j = cn.merge(tt[["barcode", "arm", "truth"]], on=["barcode", "arm"])
        j = j[j["barcode"].isin(bc.index[bc["malignant"].astype(bool)])]
        out["cn_r"] = float(np.corrcoef(j["value"], j["truth"])[0, 1]) if len(j) > 2 and j["value"].std() > 0 else np.nan
        out["cn_scale"] = scale
    else:
        out["cn_r"], out["cn_scale"] = np.nan, ""
    return out


def score(dataset_dir, runs_dir, out_csv):
    rows = []
    for name, assay, run in _runs(runs_dir):
        bc, true_cn = _truth(os.path.join(dataset_dir, "truth_extra"), assay)
        calls, cn, info = _read(run)
        rows.append(dict(run=name, tool=info.get("tool"), assay=assay, wall_s=info.get("wall_s"),
                         **_score(calls, cn, bc, true_cn)))
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(pd.DataFrame(rows).to_string(index=False))


def _strata(bc):
    tum = bc["malignant"].astype(bool)
    hyp = bc["hypoxia"].astype(float)
    cut = hyp[tum].quantile(0.9) if tum.any() else np.inf
    return {"all": pd.Series(True, index=bc.index),
            "hypoxic_core": tum & (hyp >= cut),
            "epithelial_wall": bc["at_wall"].astype(bool),
            "other_tumour": tum & (hyp < cut) & ~bc["at_wall"].astype(bool)}


def compare(truth_dir, gen_dir, perm_dir, out_csv):
    rows = []
    gen = {n: (a, r) for n, a, r in _runs(gen_dir)}
    for name, assay, rp in _runs(perm_dir):
        if name not in gen:
            continue
        bc, _ = _truth(truth_dir, assay)
        c0, n0, _ = _read(gen[name][1])
        c1, n1, _ = _read(rp)
        for stratum, mask in _strata(bc).items():
            idx = bc.index[mask.to_numpy()]
            r = dict(run=name, assay=assay, stratum=stratum, n=int(len(idx)))
            a, b = c0["malignant"].reindex(idx).astype(float), c1["malignant"].reindex(idx).astype(float)
            both = a.notna() & b.notna()
            r["malignant_flip"] = float((a[both] != b[both]).mean()) if both.any() else np.nan
            ka, kb = c0["clone"].reindex(idx), c1["clone"].reindex(idx)
            kk = ka.notna() & kb.notna()
            r["clone_ari_between_arms"] = float(adjusted_rand_score(ka[kk], kb[kk])) if kk.sum() > 1 else np.nan
            if n0 is not None and n1 is not None and len(n0) and len(n1):
                j = n0.merge(n1, on=["barcode", "arm"], suffixes=("_g", "_p"))
                j = j[j["barcode"].isin(idx)]
                r["cn_mean_abs_change"] = float((j["value_g"] - j["value_p"]).abs().mean()) if len(j) else np.nan
                r["cn_r_between_arms"] = (float(np.corrcoef(j["value_g"], j["value_p"])[0, 1])
                                          if len(j) > 2 and j["value_g"].std() > 0 and j["value_p"].std() > 0 else np.nan)
            rows.append(r)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(pd.DataFrame(rows).to_string(index=False))


# P2 against the noise null: each measure oriented so that a positive difference means the programme
# shuffle changed the tool's answer more than a technical replicate (new assay seeds, new tool seed) did.
ORIENT = {"malignant_flip": 1, "clone_ari_between_arms": -1, "cn_r_between_arms": -1, "cn_mean_abs_change": 1}


def contrast(p2, null, margin=0.02):
    """Per tool run, assay, stratum and measure: P2 and the null per tumour, their oriented difference,
    in how many tumours P2 exceeds the null, and ``exceeds_all``: P2 exceeds the null in every tumour
    with both values, by more than ``margin`` on average."""
    keys = ["tumour", "run", "assay", "stratum"]
    j = p2.merge(null, on=keys, suffixes=("_p2", "_null"))
    rows = []
    for m, sign in ORIENT.items():
        if f"{m}_p2" not in j:
            continue
        d = j[keys + [f"{m}_p2", f"{m}_null"]].dropna()
        d = d.assign(measure=m, diff=sign * (d[f"{m}_p2"] - d[f"{m}_null"]))
        rows.append(d.rename(columns={f"{m}_p2": "p2", f"{m}_null": "null"}))
    long = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not len(long):
        return long, long
    g = long.groupby(["run", "assay", "stratum", "measure"])
    summary = g.agg(n_tumours=("diff", "size"), p2=("p2", "mean"), null=("null", "mean"),
                    mean_diff=("diff", "mean"), min_diff=("diff", "min"),
                    n_exceed=("diff", lambda x: int((x > 0).sum()))).reset_index()
    summary["exceeds_all"] = (summary["n_exceed"] == summary["n_tumours"]) & (summary["mean_diff"] > margin)
    return long, summary


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "score":
        score(*sys.argv[2:5])
    elif cmd == "compare":
        compare(*sys.argv[2:6])
    else:
        sys.exit("usage: score_tools.py score|compare ...")
