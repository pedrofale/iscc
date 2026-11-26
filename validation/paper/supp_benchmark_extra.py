"""S12 Fig: the two panels of Fig 4 that the half-page layout moved to the supplement.

  (a) the per-spot clone-fraction error of Clonalscope without and with the DNA prior from the scDNA
      clones, and of CalicoST, against the largest clone in every spot (``fig4_score.collect``:
      ``$FIG4_OUT/d``), Fig 4d before the move;
  (b) the clone map and the hypoxia field of one ductal-field tumour (``fig4_lineage.py --part panel``:
      ``$FIG4_DEV/e/panel_i.npz``), Fig 4e(i) before the move.

Reads the inputs of ``fig4_benchmark.py`` (``FIG4_OUT``, ``FIG4_DIAG``, ``FIG4_DEV``), computes nothing heavy,
and writes ``figures/paper1/supp_benchmark_extra.{pdf,png}`` through ``_paths`` (``ISCC_PAPER_DIR`` picks
the checkout) and every number it draws to ``$FIG4_DEV/supp_benchmark_extra_numbers.json``.

    FIG4_OUT=.../fig4-v4 FIG4_DEV=.../fig4-dev/v4 ISCC_PAPER_DIR=... python validation/paper/supp_benchmark_extra.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import fig4_benchmark as F4  # noqa: E402
import style as S  # noqa: E402

# Drawn at the text width, so it is included at its true size and no text is below 5.5 pt.
W_MM, FIG_H = 165, 50
SMALL = F4.SMALL
TOOL_D = {"Clonalscope": "#EB6834", "Clonalscope, DNA prior": S.MODALITY["DNA"], "CalicoST": "#1BAF7A"}
A_TOP, A_H = 9.0, 31.0


def panel_a(fig, d, nums):
    """The per-spot clone-fraction error per tumour, with the largest clone in every spot as the baseline."""
    S.panel_letter(fig, "a", 0.5, 0.5)
    ax = S.ax_mm(fig, 14.0, A_TOP, 78.0, A_H)
    nums["a"] = {}
    names = list(TOOL_D)
    w = 0.22
    sub_all = d[d["resolution"] == "fine"] if len(d) else d
    for i, task in enumerate(F4.TUMOURS):
        g = sub_all[sub_all["tumour"] == f"{F4.RUN}_{task}"] if len(sub_all) else sub_all
        if not len(g):
            continue
        base = g[g["tool"] == "one clone everywhere"]["error"]
        if len(base):
            ax.plot([i - 0.4, i + 0.4], [base.iat[0]] * 2, color=S.GREY, lw=0.6, ls=":", zorder=1)
        # the floor of a caller of one clone per spot, and the copy-number oracle given the true profiles
        floor = g[g["tool"] == "dominant clone in every spot"]["error"]
        if len(floor):
            ax.plot([i - 0.4, i + 0.4], [floor.iat[0]] * 2, color=S.INK, lw=0.6, zorder=1)
        orc = g[g["tool"] == "copy-number oracle"]["error"]
        if len(orc):
            ax.plot(i + 2 * w, orc.iat[0], "D", ms=3.2, mfc="white", mec=S.INK, mew=0.7, zorder=3)
        rec = {"largest_clone_everywhere": round(float(base.iat[0]), 3) if len(base) else None,
               "dominant_clone_floor": round(float(floor.iat[0]), 3) if len(floor) else None,
               "copy_number_oracle": round(float(orc.iat[0]), 3) if len(orc) else None,
               "mixed_spots": round(float(g["mixed_spots"].iat[0]), 3)}
        for j, n in enumerate(names):
            v = g[g["tool"] == n]["error"]
            if len(v):
                ax.plot(i + (j - 1) * w, v.iat[0], "o", ms=4, mfc=TOOL_D[n], mec="white", mew=0.4, zorder=3)
                rec[n] = round(float(v.iat[0]), 3)
        sj = os.path.join(F4.OUT, "d", f"{F4.RUN}_{task}", "score.json")
        if os.path.exists(sj):
            with open(sj) as f:
                rj = json.load(f)
            rec["commit"] = rj.get("commit")
            rec["reference_protocol"] = rj.get("reference_protocol")
            rec["n_clustered"] = {k: v.get("n_clustered") for k, v in rj.get("runs", {}).items() if "n_clustered" in v}
        nums["a"][task] = rec
    F4.tumour_axis(ax, F4.TUMOURS, A_H)
    ax.set_xlim(-0.6, len(F4.TUMOURS) - 0.4)
    ax.set_ylim(0, 1.02)
    ax.set_yticks((0, 0.5, 1.0))
    ax.set_ylabel("Per-spot clone-fraction error")
    h = [Line2D([], [], marker="o", ls="none", mfc=TOOL_D[n], mec="white", ms=4, label=n) for n in names]
    h.append(Line2D([], [], marker="D", ls="none", mfc="white", mec=S.INK, mew=0.7, ms=3.2,
                    label="copy-number oracle, true profiles"))
    h.append(Line2D([], [], color=S.GREY, lw=0.6, ls=":", label="largest clone in every spot"))
    h.append(Line2D([], [], color=S.INK, lw=0.6, label="dominant clone in every spot"))
    fig.legend(handles=h, loc="upper left", bbox_to_anchor=(14.0 / W_MM, 1 - 0.8 / FIG_H), fontsize=SMALL,
               frameon=False, ncol=2, borderaxespad=0, columnspacing=1.2, handletextpad=0.3)


def panel_b(fig, nums):
    """The clone map and the hypoxia field of one tumour of the dispersal sweep in the ductal field."""
    S.panel_letter(fig, "b", 99.0, 0.5)
    p = os.path.join(F4.DEV, "e", "panel_i.npz")
    if not os.path.exists(p):
        nums["b"] = {}
        return
    z = np.load(p, allow_pickle=True)
    nums["b"] = {"run": str(z["run"]), "task": str(z["task"])}
    for k, (key, title) in enumerate((("clones", "clones"), ("hypoxia", "hypoxia field"))):
        x0 = 104.0 + k * 31.5
        ax = S.ax_mm(fig, x0, 7.0, 28.0, 28.0)
        if key == "clones":
            ax.imshow(z[key], interpolation="nearest")
        else:
            im = ax.imshow(z[key], cmap="YlOrRd", vmin=0, vmax=1, interpolation="nearest")
            cax = S.ax_mm(fig, x0, 36.6, 28.0, 1.3)
            cb = fig.colorbar(im, cax=cax, orientation="horizontal")
            cb.set_ticks((0, 0.5, 1.0))
            cb.ax.tick_params(labelsize=SMALL, length=1.5, pad=1.0)
            cb.set_label("Hypoxia", fontsize=SMALL, labelpad=1.0)
        S.despine(ax, False, False)
        ax.set_title(title, fontsize=6.5, pad=2)


def main():
    import fig4_score as FS
    S.setup()
    fig = S.figure(W_MM, FIG_H)
    nums = {"sources": dict(fig4_out=F4.OUT, diag=F4.DIAG, dev=F4.DEV, tumours=F4.TUMOURS)}
    _, d = FS.collect(F4.OUT, F4.DIAG)
    panel_a(fig, d, nums)
    panel_b(fig, nums)
    path = S.save(fig, "supp_benchmark_extra")
    os.makedirs(F4.DEV, exist_ok=True)
    with open(os.path.join(F4.DEV, "supp_benchmark_extra_numbers.json"), "w") as f:
        json.dump(nums, f, indent=1, default=str)
    print("wrote", path, "and", os.path.join(F4.DEV, "supp_benchmark_extra_numbers.json"))


if __name__ == "__main__":
    main()
