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
  most of its cells are tumour. A barcode the tool leaves undefined (an empty call, as CopyKAT's
  "not.defined", or no row at all) counts as a non-malignant call, 0 (``malignancy_score``);
  ``n_undefined`` counts them.
- ``clone_ari``: the tool's clone labels against the D9 clones, over the tumour barcodes that carry
  both. A spot takes the clone of most of its tumour cells.
- ``cn_r``: Pearson correlation of the tool's per-arm value with the truth over tumour barcodes x arms,
  the truth on the tool's scale (``truth_on_scale``): CN for ``total_cn`` (Numbat, CalicoST), CN / 2 for
  ``relative`` (Clonalscope's relative coverage, linear) and ``log2(CN / 2)`` for ``log2ratio`` (CopyKAT,
  inferCNV).
- ``wall_s`` from ``run.json``.

``score_niches`` (Fig 4b) scores one run within each niche of ``niche_masks``: the duct wall, the hypoxic
core and the stroma, each over its malignant and normal barcodes alike.

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


def truth_on_scale(cn, scale):
    """The true copy number on a tool's per-arm scale: CN for ``total_cn``, CN / 2 for ``relative`` (a
    relative coverage, linear in copy number) and log2(CN / 2) for ``log2ratio`` (CN floored at 0.05)."""
    cn = np.asarray(cn, dtype=float)
    if scale == "total_cn":
        return cn
    if scale == "relative":
        return cn / 2.0
    if scale == "log2ratio":
        return np.log2(np.maximum(cn, 0.05) / 2.0)
    raise ValueError(f"unknown per-arm scale {scale!r}")


def malignancy_score(calls, index):
    """A tool's malignancy score on the barcodes of ``index``: ``p_malignant``, else the 0/1 call, with every
    barcode the tool leaves undefined (an empty call, or no row) counted as a non-malignant call, 0.

    Returns ``(score, undefined, absent)`` (Series on ``index``: the score, whether the tool left it
    undefined, whether it has no row at all), or ``(None, None, None)`` when the tool makes no malignancy
    call (CalicoST, Clonalscope)."""
    col = "p_malignant" if calls["p_malignant"].notna().any() else "malignant"
    if not calls[col].notna().any():
        return None, None, None
    s = calls[col].astype(float).reindex(index)
    undefined = s.isna()
    absent = pd.Series(~pd.Index(index).isin(calls.index), index=index)
    return s.fillna(0.0), undefined, absent


def _score(calls, cn, bc, true_cn):
    out = {}
    y = bc["malignant"].dropna().astype(int)
    s, undefined, _ = malignancy_score(calls, y.index)
    out["n"] = int(len(y))
    out["n_undefined"] = int(undefined.sum()) if s is not None else np.nan
    out["auc"] = float(roc_auc_score(y, s)) if s is not None and y.nunique() == 2 else np.nan
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
        tt["truth"] = truth_on_scale(tt["cn"], scale)
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


# Fig 4b's niches. The strata of the P2 contrast below (_strata) hold tumour barcodes only, which is
# enough to see a call change but not to read a malignancy AUC; these masks hold every barcode, malignant
# and normal, of a niche. They need the columns fig4_tools.py writes: ``compartment`` (0 stroma, 1 the
# epithelial wall of a duct, 2 its lumen; a spot takes the compartment most of its cells sit in) and
# ``hypoxia``.
NICHES = ("all", "duct_wall", "hypoxic_core", "stroma")


def niche_masks(bc):
    """``all``; ``duct_wall``: the barcode sits in a duct's epithelial wall; ``hypoxic_core``: its
    hypoxia is at or above the 90th percentile of the malignant barcodes' (the threshold of _strata);
    ``stroma``: it sits in the stroma."""
    tum = bc["malignant"].astype(bool)
    hyp = bc["hypoxia"].astype(float)
    cut = hyp[tum].quantile(0.9) if tum.any() else np.inf
    comp = bc["compartment"].astype(int)
    return {"all": pd.Series(True, index=bc.index), "duct_wall": comp == 1, "hypoxic_core": hyp >= cut,
            "stroma": comp == 0}


def _cn_r(cn, keep, true_cn):
    """Pearson r of a run's per-arm values against the truth over the barcodes in ``keep``."""
    if cn is None or not len(cn):
        return np.nan, ""
    scale = cn["scale"].iloc[0]
    tt = true_cn.copy()
    tt["truth"] = truth_on_scale(tt["cn"], scale)
    j = cn.merge(tt[["barcode", "arm", "truth"]], on=["barcode", "arm"])
    j = j[j["barcode"].isin(keep)]
    ok = len(j) > 2 and j["value"].std() > 0 and j["truth"].std() > 0
    return (float(np.corrcoef(j["value"], j["truth"])[0, 1]) if ok else np.nan), scale


def score_niches(bc, true_cn, run_dir, min_class=10):
    """One run scored in each niche (see score_niches_frames); returns the rows and the run's run.json."""
    calls, cn, info = _read(run_dir)
    return score_niches_frames(bc, true_cn, calls, cn, min_class), info


def score_niches_frames(bc, true_cn, calls, cn, min_class=10):
    """The malignancy AUC (``malignancy_score``: ``p_malignant``, else the 0/1 call, a barcode the tool
    leaves undefined counted as non-malignant) over each niche's scored barcodes when each class holds at
    least ``min_class`` of them, and the per-arm copy-number correlation over its malignant barcodes. With a
    ``scored`` column in ``bc`` only those barcodes are scored, for every tool alike (Fig 4b: the
    epithelial-lineage barcodes); else, with a ``reference`` column, the barcodes not in the tools'
    reference. ``calls`` is indexed by barcode; ``cn`` may be None. ``n_undefined`` (of them
    ``n_undefined_malignant`` truly malignant and ``n_undefined_absent`` without a row) counts the niche's
    scored barcodes the tool left undefined; empty for a tool that makes no malignancy call."""
    s_all, und_all, abs_all = malignancy_score(calls, bc.index)
    rows = []
    masks = niche_masks(bc)            # the niches are defined on every barcode, the hypoxia cut included
    keep = (bc["scored"].astype(int).to_numpy() == 1 if "scored" in bc
            else bc["reference"].astype(int).to_numpy() == 0 if "reference" in bc else None)
    if keep is not None:
        bc = bc[keep]
        masks = {k: m.reindex(bc.index) for k, m in masks.items()}
    for niche, mask in masks.items():
        sub = bc[mask.to_numpy()]
        y = sub["malignant"].astype(int)
        if s_all is None:                      # no malignancy call: nothing is scored
            sc = pd.Series(np.nan, index=sub.index)
            und = dict(n_undefined=np.nan, n_undefined_malignant=np.nan, n_undefined_absent=np.nan)
        else:
            sc = s_all.reindex(sub.index)
            u = und_all.reindex(sub.index).to_numpy()
            und = dict(n_undefined=int(u.sum()), n_undefined_malignant=int(y[u].sum()),
                       n_undefined_absent=int(abs_all.reindex(sub.index).to_numpy().sum()))
        ok = sc.notna().to_numpy()
        n_pos, n_neg = int(y[ok].sum()), int((1 - y[ok]).sum())
        n_epi = int(((1 - y).astype(bool) & (sub["epithelial"].astype(int) > 0)).sum()) if "epithelial" in sub \
            else None
        # with an epithelial column the normal class must hold at least min_class barcodes of normal
        # epithelium (a Visium "normal" spot can be a stroma spot with a minority of cancer cells)
        n_neg_gate = n_neg if n_epi is None else int(((1 - y[ok]).astype(bool)
                                                       & (sub["epithelial"][ok].astype(int) > 0)).sum())
        enough = n_pos >= min_class and n_neg >= min_class and n_neg_gate >= min_class
        auc = (float(roc_auc_score(y[ok], sc[ok])) if enough and sc[ok].nunique() > 1 else (0.5 if enough else np.nan))
        r, scale = _cn_r(cn, sub.index[y.astype(bool).to_numpy()], true_cn)
        rows.append(dict(niche=niche, n=int(len(sub)), n_malignant=int(y.sum()), n_normal=int((1 - y).sum()),
                         n_normal_with_epithelium=n_epi, n_scored=int(ok.sum()), **und, auc=auc, cn_r=r,
                         cn_scale=scale))
    return rows


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
