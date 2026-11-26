"""Score Fig 4c and 4d from the per-tumour summaries of fig4_chain.py, and draw the draft panels.

    python validation/paper/fig4_score.py OUT [--figures DIR]

reads ``OUT/c/*/score.json`` (4c) and ``OUT/d/*/spots.tsv.gz`` with ``OUT/c/*/dna_clones.tsv`` (4d),
writes ``OUT/scores_c.csv`` and ``OUT/scores_d.csv``, and draws ``fig4cd_draft.pdf`` and ``.png`` into
``--figures`` (default ``OUT/figures``).

4c, per tumour: clonealign's accuracy on the tumour's scRNA cells given the true clones and profiles, and
given the clones called from simulated scDNA. A cell assigned to a DNA clone counts as right when most of
that clone's DNA cells belong to the cell's true clone, which only a shared ground truth can say.

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
# the copy-number oracle of fig4_chain.cn_oracle: the true clone profiles given, no clone discovery
ORACLE_D = {"oracle_cn": "copy-number oracle"}
# 4c: a tumour's truth arm counts as above chance when clonealign's balanced accuracy given the true clones
# exceeds chance by at least this much; the DNA step's cost (the gap) is read only there
CHANCE_MARGIN = 0.10


def _reference(diag_dir, tag):
    """The correlation classifier on the true profiles (fig4_diag.py's checks.json), as 4c's reference."""
    p = os.path.join(diag_dir, tag, "checks.json") if diag_dir else ""
    if not p or not os.path.exists(p):
        return dict(acc_corr=np.nan, bal_corr=np.nan)
    sig = json.load(open(p)).get("signal", {}).get("correlation", {})
    return dict(acc_corr=sig.get("accuracy", np.nan), bal_corr=sig.get("balanced_accuracy", np.nan))


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
    """Per-tool scores of one tumour's spots table (fig4_chain.py spatial): against the ground truth's
    clones (``fine``), and against their copy-number groups (``coarse``, from 4c's true_clones.tsv)."""
    out = {"fine": _score_spots(spots, c_dir)}
    tc = os.path.join(c_dir, "true_clones.tsv") if c_dir else None
    if tc and os.path.exists(tc):
        group = pd.read_csv(tc, sep="\t", index_col=0)["group"]
        sp = spots.drop(columns=[c for c in spots.columns if c.startswith("n:")])
        for g in pd.unique(group):
            sp[f"n:{g}"] = spots[[f"n:{c}" for c in group.index[group == g] if f"n:{c}" in spots]].sum(axis=1)
        out["coarse"] = _score_spots(sp, None, group_of=group.to_dict())
    return out


def _score_spots(spots, c_dir=None, group_of=None):
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
    if group_of is not None and dna is not None:
        dna = dna.assign(true_clone=dna["true_clone"].map(group_of))
    for tool in list(TOOLS_D) + list(ORACLE_D):
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


def _tumour(out_dir, tag):
    """wgd_frac and wgd_swept of a tumour, from fig4_chain.py's tumour.json (NaN when absent)."""
    p = os.path.join(out_dir, "c", tag, "tumour.json")
    if not os.path.exists(p):
        return dict(wgd_frac=np.nan, wgd_swept=np.nan)
    r = json.load(open(p))
    return dict(wgd_frac=r["wgd_frac"], wgd_swept=r["wgd_swept"])


def _oracle_set(out_dir, tag):
    """clonealign's accuracy given the true profiles of the clones the DNA step found (NaN when absent)."""
    p = os.path.join(out_dir, "c", tag, "oracle_set.json")
    r = json.load(open(p)) if os.path.exists(p) else {}
    arm = r.get("oracle_set_arm", {})
    return dict(acc_oracle_set=arm.get("accuracy", np.nan), bal_oracle_set=arm.get("balanced_accuracy", np.nan),
                n_found=len(r["clones_found"]) if r else np.nan,
                cells_in_found=r.get("share_of_cells_in_found", np.nan))


def collect(out_dir, diag_dir=None):
    rows_c, rows_d = [], []
    for f in sorted(glob.glob(os.path.join(out_dir, "c", "*", "score.json"))):
        r = json.load(open(f))
        d = r.get("dna", {})
        for res in [x for x in ("fine", "coarse") if x in r and "truth_arm" in r[x]]:
            x = r[res]
            rows_c.append(dict(tumour=f"{r['run']}_{r['task']}", resolution=res,
                               **_tumour(out_dir, f"{r['run']}_{r['task']}"),
                               **(_oracle_set(out_dir, f"{r['run']}_{r['task']}") if res == "fine" else {}),
                               **(_reference(diag_dir, f"{r['run']}_{r['task']}") if res == "fine" else {}),
                               n_rna_cells=r["n_rna_cells"],
                               n_true_clones=x["n_true_clones"], n_called_clones=d.get("n_called_clones"),
                               acc_true=x["truth_arm"]["accuracy"], acc_dna=x["dna_arm"]["accuracy"], gap=x["gap"],
                               bal_true=x["truth_arm"]["balanced_accuracy"],
                               bal_dna=x["dna_arm"]["balanced_accuracy"], gap_balanced=x["gap_balanced"],
                               ari_true=x["truth_arm"]["ari"], ari_dna=x["dna_arm"]["ari"],
                               majority=x["majority"], chance=x["chance"],
                               dna_cluster_ari=d.get("cluster_ari"),
                               hmmcopy_concordance=d.get("hmmcopy_bin_concordance"),
                               hmmcopy_concordance_scaled=d.get("hmmcopy_bin_concordance_scaled"),
                               hmmcopy_wrong_ploidy=d.get("hmmcopy_wrong_ploidy"),
                               dna_cells=d.get("n_cells"), dna_tumour_cells=d.get("n_tumour"),
                               dna_noise=d.get("n_noise"), commit=r.get("commit")))
    for f in sorted(glob.glob(os.path.join(out_dir, "d", "*", "spots.tsv.gz"))):
        tag = os.path.basename(os.path.dirname(f))
        sp = pd.read_csv(f, sep="\t", dtype={c: str for c in ("call:clonalscope", "call:clonalscope_dna",
                                                              "call:calicost", "call:oracle_cn")})
        sc = score_spots(sp, os.path.join(out_dir, "c", tag))
        for res, r in sc.items():
            base = dict(tumour=tag, resolution=res, **_tumour(out_dir, tag), n_spots=r["n_spots_scored"],
                        n_true_clones=r["n_true_clones"],
                        mixed_spots=r["mixed_spots"])
            rows_d.append(dict(base, tool="one clone everywhere", **r["one_clone_everywhere"]))
            if res == "fine":
                # the floor of any caller of one clone per spot: the dominant clone of every spot
                cnt = sp[[c for c in sp.columns if c.startswith("n:")]].to_numpy(float)
                cnt = cnt[cnt.sum(axis=1) > 0]
                rows_d.append(dict(base, tool="dominant clone in every spot",
                                   error=float(np.mean(1.0 - cnt.max(axis=1) / cnt.sum(axis=1)))))
            for tool, name in list(TOOLS_D.items()) + list(ORACLE_D.items()):
                if tool in r:
                    rows_d.append(dict(base, tool=name, **r[tool]))
    c = pd.DataFrame(rows_c)
    if len(c):
        c["above_chance"] = c["bal_true"] >= c["chance"] + CHANCE_MARGIN
    return c, pd.DataFrame(rows_d)


def _panel_c(ax, c, order, short, S, DNA, metric):
    for i, tu in enumerate(order):
        r = c[c["tumour"] == tu]
        if not len(r):
            continue
        r = r.iloc[0]
        a_t, a_d = r[f"{metric}_true"], r[f"{metric}_dna"]
        base = r["majority"] if metric == "acc" else r["chance"]
        ax.plot([i, i], [a_d, a_t], color=S.GREY, lw=0.6, zorder=1)
        ax.plot(i - 0.28, base, marker="_", ms=6, mew=0.8, color=S.GREY, ls="none", zorder=1)
        ax.plot(i, a_t, "o", ms=4, mfc=S.INK, mec="white", mew=0.4, zorder=3)
        ax.plot(i, a_d, "o", ms=4, mfc=DNA, mec="white", mew=0.4, zorder=3)
        a_o = r.get(f"{metric}_oracle_set", np.nan)
        if np.isfinite(a_o):
            ax.plot(i - 0.14, a_o, "o", ms=3.4, mfc="white", mec=S.INK, mew=0.6, zorder=2)
        ax.text(i + 0.1, (a_t + a_d) / 2, f"{a_d - a_t:+.2f}".replace("-", "\u2212"), fontsize=4.6,
                color=S.INK, va="center")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([short[t].replace("\n", " ", 1) for t in order], fontsize=4.4, rotation=40, ha="right",
                       rotation_mode="anchor")
    ax.set_xlim(-0.6, len(order) - 0.3)
    ax.set_ylim(0, 1.02)


def _panel_d(ax, d, order, short, S, colours):
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
    ax.set_xticklabels([short[t] for t in order], fontsize=4.6)
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylim(0, 1.02)


def figure(c, d, path):
    """Draft panels against the ground truth's clones: 4c (clonealign accuracy and balanced accuracy, given
    the true clones against the DNA-called ones) and 4d (per-spot clone-fraction error). The coarse
    resolution stays in the score tables only."""
    import matplotlib
    matplotlib.use("Agg")
    import style as S
    S.setup()
    fig = S.figure("double", 62)
    tumours = set(c["tumour"]) if len(c) else set()
    if len(d):
        tumours |= set(d["tumour"])
    order = sorted(tumours, key=lambda s: (0 if "disp" not in s else 1, s))
    def _short(t):
        task = t.split("_", 1)[1]
        if task.startswith("disp"):
            d_, s_ = task[4:].split("_seed")
            return f"disp. {d_}\nseed {s_}"
        return f"showcase\nseed {task.replace('seed', '')}"
    short = {t: _short(t) for t in order}
    swept = set()
    for frame in (c, d):
        if len(frame) and "wgd_swept" in frame:
            swept |= set(frame.loc[frame["wgd_swept"] == True, "tumour"])  # noqa: E712
    short = {t: v + ("\nWGD swept" if t in swept else "") for t, v in short.items()}
    DNA = S.MODALITY["DNA"]
    # validated (dataviz validate_palette, all pairs, light): CVD and normal-vision separation pass
    colours = {"Clonalscope": "#EB6834", "Clonalscope, DNA prior": DNA, "CalicoST": "#1BAF7A"}
    from matplotlib.lines import Line2D
    y, h = 12, 36
    cc = c[c["resolution"] == "fine"] if len(c) else c
    dd = d[d["resolution"] == "fine"] if len(d) else d
    S.panel_letter(fig, "c", 2, 3)
    for col, (metric, ylab, base) in enumerate((("bal", "Balanced accuracy", "chance"),
                                               ("acc", "Accuracy", "largest clone for every cell"))):
        ax = S.ax_mm(fig, 13 + col * 46, y, 36, h)
        if len(cc):
            _panel_c(ax, cc, order, short, S, DNA, metric)
        ax.set_ylabel(f"clonealign {ylab.lower()}")
        ax.legend(handles=[Line2D([], [], marker="_", ls="none", color=S.GREY, ms=6, mew=0.8, label=base)],
                  loc="upper right", fontsize=4.6, frameon=False)
    fig.legend(handles=[Line2D([], [], marker="o", ls="none", mfc=S.INK, mec="white", ms=4,
                               label="given the true clones and profiles"),
                        Line2D([], [], marker="o", ls="none", mfc=DNA, mec="white", ms=4,
                               label="given the clones called from simulated scDNA (HMMcopy, clustering)"),
                        Line2D([], [], marker="o", ls="none", mfc="white", mec=S.INK, mew=0.6, ms=3.4,
                               label="given the true profiles of only the clones scDNA found")],
               loc="upper left", bbox_to_anchor=(10 / 183, 1 - 1 / 62), ncol=1, fontsize=4.8, frameon=False)
    S.panel_letter(fig, "d", 104, 3)
    ax = S.ax_mm(fig, 116, y, 64, h)
    if len(dd):
        _panel_d(ax, dd, order, short, S, colours)
        names = [n for n in colours if n in set(dd["tool"])]
        hd = [Line2D([], [], marker="o", ls="none", mfc=colours[n], mec="white", ms=4, label=n) for n in names]
        hd.append(Line2D([], [], color=S.GREY, lw=0.6, ls=":", label="largest clone everywhere"))
        fig.legend(handles=hd, loc="upper left", bbox_to_anchor=(112 / 183, 1 - 1 / 62), ncol=2, fontsize=4.8,
                   frameon=False)
    else:
        ax.text(0.5, 0.5, "pending", transform=ax.transAxes, ha="center", fontsize=6, color=S.GREY)
    ax.set_ylabel("Per-spot clone-fraction error")
    src = os.environ.get("ISCC_PAPER_OUT", "")
    fig.text(0.995, 0.004, f"draft; {os.environ.get('FIG4_RUN', 'R14')} tumours from {os.path.basename(src.rstrip('/')) or 'the registry'}",
             ha="right", va="bottom", fontsize=4.5, color=S.GREY)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path + ".pdf")
    fig.savefig(path + ".png", dpi=300)
    print("wrote", path + ".pdf", "and .png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--figures", default=None)
    ap.add_argument("--diag", default=os.environ.get("FIG4_DIAG"), help="fig4_diag.py's output (4c's reference)")
    a = ap.parse_args()
    c, d = collect(a.out, a.diag)
    c.to_csv(os.path.join(a.out, "scores_c.csv"), index=False)
    d.to_csv(os.path.join(a.out, "scores_d.csv"), index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(c.round(3).to_string(index=False))
        print(d.round(3).to_string(index=False))
    figure(c, d, os.path.join(a.figures or os.path.join(a.out, "figures"), "fig4cd_draft"))


if __name__ == "__main__":
    main()
