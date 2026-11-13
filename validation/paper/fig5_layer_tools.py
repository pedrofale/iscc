"""Fig 5, the panels on the human genome layer: published copy-number tools on iscc's Visium and scRNA
over the human genes, and the programme-permutation contrast (P2).

Reads the datasets written by ``tools/make_layer_dataset.py`` (``--arms generated,permuted``) and the
runs of ``tools/submit_p2.sh``, all under ``FIG5_LAYER_DIR``::

    FIG5_LAYER_DIR/<tumour>/{generated,permuted}/{visium,scrna}, <tumour>/truth_extra
    FIG5_LAYER_DIR/runs/<tumour>/{generated,permuted}/<tool>-<assay>[-ref]

scores them with ``tools/score_tools.py`` into ``FIG5_LAYER_DIR/scores/`` and draws
``figures/paper1/fig5_layer_tools.pdf`` through ``validation/_paths.py`` (``ISCC_PAPER_DIR`` picks the
paper checkout). Panels:

a. malignant against normal (AUC), per tool, assay and tumour, programmes as generated;
b. clone recovery against the D9 clones (ARI);
c. copy number per arm against the truth (Pearson r);
d. wall time per run;
e. P2: the share of malignant calls that change when the programmes are shuffled among cells, overall,
   in the hypoxic core and at the epithelial wall;
f. P2: agreement of the clone labels between the two arms (ARI; 1 is unchanged);
g. P2: agreement of the per-arm copy-number calls between the two arms (Pearson r; 1 is unchanged).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tools"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import score_tools as ST  # noqa: E402
import style as S  # noqa: E402

LABEL = {"copykat": "CopyKAT", "copykat-ref": "CopyKAT, normal reference", "clonalscope": "Clonalscope",
         "calicost": "CalicoST", "cancerfinder": "Cancer-Finder", "infercnvpy": "inferCNV (py)",
         "numbat": "Numbat"}
ORDER = ["cancerfinder", "copykat", "copykat-ref", "infercnvpy", "numbat", "clonalscope", "calicost"]
ASSAY = {"visium": S.MODALITY["spatial"], "scrna": S.MODALITY["RNA"]}
MARKERS = ["o", "s", "^", "D"]
STRATA = {"all": ("all barcodes", S.INK), "hypoxic_core": ("hypoxic core", "#D95F02"),
          "epithelial_wall": ("epithelial wall", S.COMPARTMENT["epithelium"])}


def _key(run):
    parts = run.split("-")
    return parts[0] + ("-ref" if parts[-1] == "ref" else "")


def collect(root):
    """Score every tumour's two arms and compare them; returns (scores, changes) and caches CSVs."""
    out = os.path.join(root, "scores")
    os.makedirs(out, exist_ok=True)
    sc, ch = [], []
    for tumour in sorted(os.listdir(os.path.join(root, "runs"))):
        ds = os.path.join(root, tumour)
        for arm in ("generated", "permuted"):
            runs = os.path.join(root, "runs", tumour, arm)
            if not os.path.isdir(runs):
                continue
            f = os.path.join(out, f"{tumour}_{arm}.csv")
            ST.score(ds, runs, f)
            sc.append(pd.read_csv(f).assign(tumour=tumour, arm=arm))
        g, p = (os.path.join(root, "runs", tumour, a) for a in ("generated", "permuted"))
        if os.path.isdir(g) and os.path.isdir(p):
            f = os.path.join(out, f"{tumour}_p2.csv")
            ST.compare(os.path.join(ds, "truth_extra"), g, p, f)
            ch.append(pd.read_csv(f).assign(tumour=tumour))
    scores = pd.concat(sc, ignore_index=True) if sc else pd.DataFrame()
    changes = pd.concat(ch, ignore_index=True) if ch else pd.DataFrame()
    for df in (scores, changes):
        if len(df):
            df["key"] = df["run"].map(_key)
    scores.to_csv(os.path.join(out, "scores.csv"), index=False)
    changes.to_csv(os.path.join(out, "p2_changes.csv"), index=False)
    return scores, changes


def _dots(ax, df, col, tumours, xlim, xlabel, ref=None):
    keys = [k for k in ORDER if k in set(df["key"])]
    for i, k in enumerate(keys):
        for j, tu in enumerate(tumours):
            for assay, colour in ASSAY.items():
                v = df[(df["key"] == k) & (df["tumour"] == tu) & (df["assay"] == assay)][col]
                if len(v) and np.isfinite(v.iat[0]):
                    off = -0.15 if assay == "visium" else 0.15
                    ax.plot(v.iat[0], i + off, MARKERS[j % len(MARKERS)], ms=2.6, mfc=colour, mec="white",
                            mew=0.3, ls="none")
    if ref is not None:
        ax.axvline(ref, color=S.GREY, lw=0.5, ls=":")
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([LABEL[k] for k in keys])
    ax.set_ylim(len(keys) - 0.5, -0.5)
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    S.despine(ax)
    return keys


def _p2(ax, changes, col, xlim, xlabel, ref=None, labels=True):
    keys = [k for k in ORDER if k in set(changes["key"])]
    for i, k in enumerate(keys):
        for s_i, (stratum, (lab, colour)) in enumerate(STRATA.items()):
            v = changes[(changes["key"] == k) & (changes["stratum"] == stratum)]
            for _, r in v.iterrows():
                if np.isfinite(r[col]):
                    y_ = i + (s_i - 1) * 0.22 + (0.06 if r["assay"] == "scrna" else -0.06)
                    ax.plot(r[col], y_, "o", ms=2.4, mfc=colour, mec="white", mew=0.3,
                            alpha=1.0 if r["assay"] == "visium" else 0.55)
    if ref is not None:
        ax.axvline(ref, color=S.GREY, lw=0.5, ls=":")
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([LABEL[k] for k in keys] if labels else [])
    ax.set_ylim(len(keys) - 0.5, -0.5)
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    S.despine(ax)


def figure(root):
    scores, changes = collect(root)
    S.setup()
    fig = S.figure("double", 132)
    gen = scores[scores["arm"] == "generated"] if len(scores) else scores
    tumours = sorted(gen["tumour"].unique()) if len(gen) else []
    top = [("a", "auc", (0, 1.02), "Malignant vs normal (AUC)", 0.5),
           ("b", "clone_ari", (-0.1, 1.02), "Clones vs truth (ARI)", 0.0),
           ("c", "cn_r", (-0.3, 1.02), "Copy number per arm (r)", 0.0),
           ("d", "log_wall", (0, 4.6), "Wall time (log10 s)", None)]
    g = gen.copy()
    if len(g):
        g["log_wall"] = np.log10(g["wall_s"].astype(float).clip(lower=1))
    x = 40
    for i, (letter, col, xlim, xlabel, ref) in enumerate(top):
        w = 32 if i == 0 else 30
        S.panel_letter(fig, letter, (2 if i == 0 else x - 5), 3)
        ax = S.ax_mm(fig, x, 6, w, 42)
        if len(g):
            _dots(ax, g, col, tumours, xlim, xlabel, ref=ref)
            if i:
                ax.set_yticklabels([])
        x += w + 5
    handles = [Line2D([], [], marker="o", ls="none", ms=3, mfc=c, mec="white",
                      label="Visium" if a_ == "visium" else "scRNA") for a_, c in ASSAY.items()]
    handles += [Line2D([], [], marker=MARKERS[j % len(MARKERS)], ls="none", ms=3, mfc=S.GREY, mec="white",
                       label=tu.replace("_", " ")) for j, tu in enumerate(tumours)]
    fig.legend(handles=handles, loc="upper left", ncol=len(handles), fontsize=5, frameon=False,
               handletextpad=0.2, columnspacing=1.0,
               bbox_to_anchor=(40 * S.MM / fig.get_figwidth(), 1 - 58 * S.MM / fig.get_figheight()))
    bottom = [("e", "malignant_flip", "Malignant calls that change\nwhen programmes are shuffled", None),
              ("f", "clone_ari_between_arms", "Clone labels,\nshuffled vs generated (ARI)", 1.0),
              ("g", "cn_r_between_arms", "Copy number per arm,\nshuffled vs generated (r)", 1.0)]
    x = 40
    for i, (letter, col, xlabel, ref) in enumerate(bottom):
        S.panel_letter(fig, letter, (2 if i == 0 else x - 5), 68)
        ax = S.ax_mm(fig, x, 70, 42, 44)
        if len(changes) and changes[col].notna().any():
            hi = max(0.05, float(np.nanmax(changes[col])) * 1.1) if col == "malignant_flip" else 1.05
            lo = 0.0 if col == "malignant_flip" else min(-0.1, float(np.nanmin(changes[col])) - 0.05)
            _p2(ax, changes, col, (lo, hi), xlabel, ref=ref, labels=(i == 0))
            if i == 0:
                ax.legend(handles=[Line2D([], [], marker="o", ls="none", ms=3, mfc=c, mec="white", label=lab)
                                   for lab, c in STRATA.values()], fontsize=5, frameon=False,
                          loc="lower right", handletextpad=0.2, title="faded: scRNA", title_fontsize=4.5)
        x += 42 + 6
    path = S.save(fig, "fig5_layer_tools")
    with open(os.path.join(root, "scores", "fig5_layer_tools.json"), "w") as f:
        json.dump({"figure": path, "tumours": tumours,
                   "scores": scores.round(4).to_dict("records") if len(scores) else [],
                   "p2": changes.round(4).to_dict("records") if len(changes) else []}, f, indent=1, default=str)
    print(path)
    return path


if __name__ == "__main__":
    figure(os.environ.get("FIG5_LAYER_DIR") or sys.exit("set FIG5_LAYER_DIR"))
