"""Supplementary figure: the structures only iscc produces, beside demon and iscc with the tissue off.

Four maps, each deme coloured by its dominant clone:

(a) demon and (b) iscc with the tissue off, from the territory sweep of ``comparisons/demon_match.py``
    (neutral marks at 2e-2 per division, dispersal 0.1, seed 1, 20,000 cancer cells; read from
    ``comparisons/results/demon_match_maps.csv``): clones on an unstructured lattice;
(c) the primary of a regrown R12 tumour (paper regime v3 with the metastatic lattice, 60,000 cancer
    cells): glands, lumen inside a wall of epithelium, set in stroma, the clones that breached the
    wall and invaded it, and (d) its metastatic deposit in host parenchyma. Read from
    ``comparisons/results/iscc_structures.npz``, written by ``comparisons/iscc_structures.py``.

Clone colours are per tumour: the same colour in two panels is not the same clone. The comparison
does not depend on the paper's scale tier, so the figure carries no preview stamp.

Usage::

    python validation/paper/supp_structures.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import style as S  # noqa: E402
import supp_demon_match as DM  # noqa: E402

RES = os.path.join(HERE, "comparisons", "results")


def main():
    S.setup()
    maps = pd.read_csv(os.path.join(RES, "demon_match_maps.csv"))
    z = np.load(os.path.join(RES, "iscc_structures.npz"))
    meta = json.loads(str(z["meta"]))
    fig = S.figure("double", 64)
    w, y = 40, 9
    xs = [6, 51, 96, 141]
    titles = ["demon", "iscc, tissue off", f"iscc, primary ({meta['run']} {meta['task']})",
              "iscc, metastatic deposit"]
    notes = ["no tissue; 20,000 cells", "no tissue; 20,000 cells",
             f"{meta['n_glands']} glands in stroma; {meta['primary_cancer']:,} cells,\n"
             f"{meta['stroma_cancer'] / max(meta['primary_cancer'], 1):.0%} of them past the gland wall",
             f"host parenchyma; {meta['met_cancer']:,} cells"]
    for k, (x, title, note) in enumerate(zip(xs, titles, notes)):
        ax = S.ax_mm(fig, x, y, w, w)
        S.despine(ax, False, False)
        if k < 2:
            model, engine = ("demon", "gillespie") if k == 0 else ("iscc", "tau")
            sub = maps[(maps.model == model) & (maps.engine == engine) & np.isclose(maps.migration, 0.1)
                       & np.isclose(maps.mu, DM.MU_TERR) & (maps.seed == 1)]
            if len(sub):
                ax.imshow(DM._map_rgb(sub), interpolation="nearest")
        else:
            ax.imshow(z["primary" if k == 2 else "met"], interpolation="nearest")
        ax.set_title(title, fontsize=6.5)
        S.text_mm(fig, x, y + w + 1.5, note, fontsize=5.5, va="top", color=S.GREY)
        S.panel_letter(fig, "abcd"[k], x - 4, 3)
    handles = [Patch(facecolor=S.COMPARTMENT[c], edgecolor=S.GREY, lw=0.3, label=lab) for c, lab in
               (("lumen", "lumen"), ("epithelium", "gland wall"), ("stroma", "stroma"), ("met", "host"))]
    fig.legend(handles=handles, loc="lower right", ncol=4, bbox_to_anchor=(0.99, 0.0), handlelength=1.0,
               title="tissue (clones in colour, per tumour)", title_fontsize=5.5)
    S.save(fig, "supp_structures", stamp=False)


if __name__ == "__main__":
    main()
