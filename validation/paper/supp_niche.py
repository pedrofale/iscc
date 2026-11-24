"""S3 Fig: the genotype-niche split of the invasive program in full.

Reads the per-tumour cache of Fig 3c (``fig3_answer_key.py --compute``): ten R10 tumours, with and
without compartment barriers. Shows one tumour of each kind (the invasive program against the
epithelial niche field) and every tumour's variance split into niche, clone and residual.

Writes ``figures/paper1/supp_s3_niche.pdf``.
Run:  python validation/paper/supp_niche.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import fig3_answer_key as F3  # noqa: E402
import style as S  # noqa: E402


def main():
    S.setup()
    rows = []
    for fn in sorted(os.listdir(F3.C_CACHE)):
        with open(os.path.join(F3.C_CACHE, fn)) as f:
            rows.append(json.load(f))
    df = pd.DataFrame(rows)
    fig = S.figure("double", 50)
    for k, task in enumerate(("barriers_seed1", "open_seed1")):
        r = next((x for x in rows if x["task"] == task and "scatter_niche" in x), None)
        S.panel_letter(fig, "ab"[k], 2 + k * 40, 3)
        ax = S.ax_mm(fig, 12 + k * 40, 7, 28, 32)
        if r is None:
            F3.pending(ax)
            continue
        ax.scatter(r["scatter_niche"], r["scatter_emt"], s=1.0, color=S.COMPARTMENT["epithelium"],
                   rasterized=True)
        ax.set_xlabel("Epithelial niche field")
        ax.set_ylabel("emt program activity" if k == 0 else "")
        ax.set_title(f"{task.split('_')[0]}, seed 1", fontsize=5.5)
    S.panel_letter(fig, "c", 86, 3)
    ax = S.ax_mm(fig, 96, 7, 82, 32)
    df = df.sort_values(["arch", "task"], ascending=[True, True]).reset_index(drop=True)
    x = np.arange(len(df))
    bottom = np.zeros(len(df))
    for comp, col, lab in (("niche", S.COMPARTMENT["epithelium"], "niche"),
                           ("genotype", S.CLONE_FAMILIES[2][1], "clone"),
                           ("residual", S.BG_CELLS, "cell-to-cell")):
        ax.bar(x, df[comp], bottom=bottom, color=col, width=0.7, label=lab)
        bottom += df[comp].values
    ax.set_xticks(x)
    ax.set_xticklabels([f"{a}\n{t.split('seed')[1]}" for a, t in zip(df.arch, df.task)], fontsize=4.3)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Variance of the emt program")
    ax.legend(fontsize=4.8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12), frameon=False)
    path = S.save(fig, "supp_s3_niche")
    print("wrote", path)
    print(df[["task", "n_cells", "n_clones", "niche", "genotype", "residual"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
