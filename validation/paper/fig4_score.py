"""Score Fig 4c and 4d from the per-tumour summaries of fig4_chain.py, and draw the draft panels.

    python validation/paper/fig4_score.py OUT [--figures DIR]

reads ``OUT/c/*/score.json`` (4c) and ``OUT/d/*/spots.tsv.gz`` with ``OUT/c/*/dna_clones.tsv`` (4d),
writes ``OUT/scores_c.csv`` and ``OUT/scores_d.csv``, and draws ``fig4cd_draft.pdf`` and ``.png`` into
``--figures`` (default ``OUT/figures``).

4c, per tumour: clonealign's accuracy on the tumour's scRNA cells given the true clones and profiles, and
given the clones called from simulated scDNA. A cell assigned to a DNA clone counts as right when most of
that clone's DNA cells belong to the cell's true clone, which only a shared answer key can say.

4d, per tumour and tool, over the spots holding cells of a true clone. Each spot has true clone
fractions p (its tumour cells' clones; cells above every clade and normal cells left out). Both tools
give one label per spot, so a tool's fraction vector q is one-hot on the true clone its label stands
for: the clone holding most of the tumour cells of that label's spots (the same rule for every tool).
The per-spot error is the total-variation distance between p and q, 1 - p[called clone], averaged over
spots; the dominant-clone accuracy is the share of spots whose called clone is the spot's commonest; the
ARI compares the raw labels with the commonest clone. For Clonalscope with the DNA prior, cluster k <=
1 + (number of DNA clones) is the prior's row k (row 1 normal, row k the (k-1)th DNA clone), so its
labels can also be named without the truth, through the DNA clone's own mapping (``named_*``).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

TOOLS_D = {"clonalscope": "Clonalscope", "clonalscope_dna": "Clonalscope, DNA prior", "calicost": "CalicoST"}


def _majority_map(labels, counts):
    """label -> index of the true clone holding most of the label's tumour cells."""
    out = {}
    for lab in pd.unique(labels):
        if lab is None or (isinstance(lab, float) and np.isnan(lab)):
            continue
        out[lab] = int(np.argmax(counts[labels == lab].sum(axis=0)))
    return out


def _spot_scores(P, dom, pred_idx, labels):
    from sklearn.metrics import adjusted_rand_score
    got = np.array([P[i, k] if k is not None else 0.0 for i, k in enumerate(pred_idx)])
    lab = np.array(["__none__" if (x is None or (isinstance(x, float) and np.isnan(x))) else str(x)
                    for x in labels])
    return dict(error=float(np.mean(1.0 - got)),
                dominant_accuracy=float(np.mean([k is not None and k == d for k, d in zip(pred_idx, dom)])),
                ari=float(adjusted_rand_score(dom, lab)))


def score_spots(spots, c_dir=None):
    """Per-tool scores of one tumour's spots table (fig4_chain.py spatial)."""
    clones = [c[2:] for c in spots.columns if c.startswith("n:")]
    counts_all = spots[[f"n:{c}" for c in clones]].to_numpy(float)
    m = counts_all.sum(axis=1) > 0
    counts = counts_all[m]
    P = counts / counts.sum(axis=1, keepdims=True)
    dom = P.argmax(axis=1)
    out = dict(n_spots_scored=int(m.sum()), n_true_clones=len(clones),
               mixed_spots=float(np.mean((P > 0).sum(axis=1) > 1)))
    biggest = int(np.argmax(counts.sum(axis=0)))
    out["one_clone_everywhere"] = _spot_scores(P, dom, [biggest] * len(P), np.zeros(len(P)))
    dna = None
    if c_dir and os.path.exists(os.path.join(c_dir, "dna_clones.tsv")):
        dna = pd.read_csv(os.path.join(c_dir, "dna_clones.tsv"), sep="\t", index_col=0)
    for tool in TOOLS_D:
        col = f"call:{tool}"
        if col not in spots:
            continue
        labels = spots.loc[m, col].astype(object).where(spots.loc[m, col].notna(), None).to_numpy()
        labels = np.array([None if x is None else str(x) for x in labels], dtype=object)
        mp = _majority_map(labels, counts)
        pred = [mp.get(x) if x is not None else None for x in labels]
        rec = _spot_scores(P, dom, pred, labels)
        rec["n_labels"] = int(len({x for x in labels if x is not None}))
        rec["unlabelled"] = float(np.mean([x is None for x in labels]))
        if tool == "clonalscope_dna" and dna is not None:
            row_clone = {str(k + 2): dna["true_clone"].iat[k] for k in range(len(dna))}
            idx = {c: i for i, c in enumerate(clones)}
            named = [idx.get(row_clone.get(x)) if x is not None else None for x in labels]
            nrec = _spot_scores(P, dom, named, labels)
            rec.update({f"named_{k}": v for k, v in nrec.items() if k != "ari"})
            rec["share_in_prior_rows"] = float(np.mean([x in row_clone or x == "1" for x in labels]))
        out[tool] = rec
    return out


def collect(out_dir):
    rows_c, rows_d = [], []
    for f in sorted(glob.glob(os.path.join(out_dir, "c", "*", "score.json"))):
        r = json.load(open(f))
        d = r.get("dna", {})
        rows_c.append(dict(tumour=f"{r['run']}_{r['task']}", n_rna_cells=r["n_rna_cells"],
                           n_true_clones=r["n_true_clones"], n_called_clones=d.get("n_called_clones"),
                           acc_true=r["truth_arm"]["accuracy"], acc_dna=r["dna_arm"]["accuracy"],
                           gap=r["gap"], ari_true=r["truth_arm"]["ari"], ari_dna=r["dna_arm"]["ari"],
                           majority=r["majority"], chance=r["chance"],
                           dna_cluster_ari=d.get("cluster_ari"), hmmcopy_concordance=d.get("hmmcopy_bin_concordance"),
                           dna_cells=d.get("n_cells"), dna_tumour_cells=d.get("n_tumour"),
                           dna_noise=d.get("n_noise"), commit=r.get("commit")))
    for f in sorted(glob.glob(os.path.join(out_dir, "d", "*", "spots.tsv.gz"))):
        tag = os.path.basename(os.path.dirname(f))
        sc = score_spots(pd.read_csv(f, sep="\t", dtype={c: str for c in
                                                          ("call:clonalscope", "call:clonalscope_dna", "call:calicost")}),
                         os.path.join(out_dir, "c", tag))
        base = dict(tumour=tag, n_spots=sc["n_spots_scored"], n_true_clones=sc["n_true_clones"],
                    mixed_spots=sc["mixed_spots"])
        rows_d.append(dict(base, tool="one clone everywhere", **sc["one_clone_everywhere"]))
        for tool, name in TOOLS_D.items():
            if tool in sc:
                rows_d.append(dict(base, tool=name, **sc[tool]))
    return pd.DataFrame(rows_c), pd.DataFrame(rows_d)


def figure(c, d, path):
    import matplotlib
    matplotlib.use("Agg")
    import style as S
    S.setup()
    fig = S.figure("double", 62)
    order = sorted(set(c["tumour"]) | set(d.get("tumour", pd.Series(dtype=str))),
                   key=lambda s: (0 if s.endswith("_seed1") and "disp" not in s else 1, s))
    short = {t: t.split("_", 1)[1].replace("disp", "dispersal ").replace("_seed", ", seed ") for t in order}
    DNA = S.MODALITY["DNA"]

    # c: errors carry across modalities
    S.panel_letter(fig, "c", 2, 3)
    ax = S.ax_mm(fig, 14, 8, 70, 38)
    if len(c):
        for i, tu in enumerate(order):
            r = c[c["tumour"] == tu]
            if not len(r):
                continue
            r = r.iloc[0]
            ax.plot([i, i], [r.acc_dna, r.acc_true], color=S.GREY, lw=0.6, zorder=1)
            ax.plot(i - 0.28, r.majority, marker="_", ms=6, mew=0.8, color=S.GREY, ls="none", zorder=1)
            ax.plot(i, r.acc_true, "o", ms=4, mfc=S.INK, mec="white", mew=0.4, zorder=3)
            ax.plot(i, r.acc_dna, "o", ms=4, mfc=DNA, mec="white", mew=0.4, zorder=3)
            change = r.acc_dna - r.acc_true                 # what the DNA step costs (negative: a loss)
            ax.text(i + 0.12, (r.acc_true + r.acc_dna) / 2, f"{change:+.2f}".replace("-", "−"),
                    fontsize=5, color=S.INK, va="center")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([short[t] for t in order], rotation=20, ha="right")
        ax.set_xlim(-0.6, len(order) - 0.4)
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("clonealign accuracy on scRNA")
        from matplotlib.lines import Line2D
        h = [Line2D([], [], marker="o", ls="none", mfc=S.INK, mec="white", ms=4, label="true clones and profiles"),
             Line2D([], [], marker="o", ls="none", mfc=DNA, mec="white", ms=4,
                    label="clones from simulated scDNA (HMMcopy, clustering)"),
             Line2D([], [], marker="_", ls="none", color=S.GREY, ms=6, mew=0.8, label="largest clone (majority)")]
        ax.legend(handles=h, loc="lower left", fontsize=4.8, frameon=False)
    # d: spatial clone mapping with a DNA prior
    S.panel_letter(fig, "d", 96, 3)
    ax = S.ax_mm(fig, 108, 8, 70, 38)
    # validated (dataviz validate_palette, all pairs, light): CVD and normal-vision separation pass
    colours = {"Clonalscope": "#EB6834", "Clonalscope, DNA prior": DNA, "CalicoST": "#1BAF7A"}
    if len(d):
        names = [n for n in colours if n in set(d["tool"])]
        w = 0.22
        for i, tu in enumerate(order):
            g = d[d["tumour"] == tu]
            base = g[g["tool"] == "one clone everywhere"]
            if len(base):
                ax.plot([i - 0.4, i + 0.4], [base["error"].iat[0]] * 2, color=S.GREY, lw=0.6, ls=":", zorder=1)
            for j, n in enumerate(names):
                v = g[g["tool"] == n]["error"]
                if len(v):
                    ax.plot(i + (j - (len(names) - 1) / 2) * w, v.iat[0], "o", ms=4, mfc=colours[n], mec="white",
                            mew=0.4, zorder=3)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([short[t] for t in order], rotation=20, ha="right")
        ax.set_xlim(-0.6, len(order) - 0.4)
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("per-spot clone-fraction error")
        from matplotlib.lines import Line2D
        h = [Line2D([], [], marker="o", ls="none", mfc=colours[n], mec="white", ms=4, label=n) for n in names]
        h.append(Line2D([], [], color=S.GREY, lw=0.6, ls=":", label="largest clone everywhere"))
        ax.legend(handles=h, loc="upper left", fontsize=4.8, frameon=False, ncol=2)
    else:
        ax.text(0.5, 0.5, "pending", transform=ax.transAxes, ha="center", fontsize=6, color=S.GREY)
    fig.text(0.995, 0.004, "draft: R11 tumours of the old registry (paper-v2)" if "paper-v2" in
             str(os.environ.get("ISCC_PAPER_OUT", "paper-v2")) else "draft", ha="right", va="bottom",
             fontsize=4.5, color=S.GREY)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path + ".pdf")
    fig.savefig(path + ".png", dpi=300)
    print("wrote", path + ".pdf", "and .png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--figures", default=None)
    a = ap.parse_args()
    c, d = collect(a.out)
    c.to_csv(os.path.join(a.out, "scores_c.csv"), index=False)
    d.to_csv(os.path.join(a.out, "scores_d.csv"), index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(c.round(3).to_string(index=False))
        print(d.round(3).to_string(index=False))
    figure(c, d, os.path.join(a.figures or os.path.join(a.out, "figures"), "fig4cd_draft"))


if __name__ == "__main__":
    main()
