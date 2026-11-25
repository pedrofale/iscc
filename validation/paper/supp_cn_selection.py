"""S16 Fig: the mean copy number of each segment against its net oncogenic content (oncogenes minus
tumour suppressors), in the ten tumours of the selection sweep grown under selection and the ten grown
neutrally with the same driver layout (R2), with the Spearman correlation over the segments of each
arm. It draws the cached summaries (``summaries.py``: ``seg_cn``, ``seg_net``), so it needs no tumour.

Usage::

    ISCC_PAPER_OUT=... ISCC_PAPER_TIER=full ISCC_PAPER_DIR=... python validation/paper/supp_cn_selection.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

import style as S  # noqa: E402
import summaries as SM  # noqa: E402

ARMS = (("selected", "selected"), ("neutral", "neutral"))


def main():
    rows = SM.load("R2")
    S.setup()
    fig = S.figure(120, 47)
    out, axes = {}, []
    for k, (prefix, title) in enumerate(ARMS):
        rs = [r for r in rows if r["task"].startswith(prefix + "_")]
        net = np.concatenate([r["seg_net"] for r in rs]).astype(float)
        cn = np.concatenate([r["seg_cn"] for r in rs]).astype(float)
        rho = float(spearmanr(net, cn)[0])
        out[prefix] = dict(tumours=len(rs), segments=len(net), spearman=round(rho, 3))
        ax = S.ax_mm(fig, 12 + k * 56, 6, 46, 32, sharey=axes[0] if axes else None)
        axes.append(ax)
        ax.axhline(2, color=S.BG_CELLS, lw=0.5, zorder=0)
        jitter = np.random.default_rng(0).uniform(-0.18, 0.18, len(net))
        ax.scatter(net + jitter, cn, s=4, color=S.SOURCE["iscc"], alpha=0.7, linewidths=0, zorder=2)
        ax.set_title(f"{title} ({len(rs)} tumors)", fontsize=6, pad=2)
        ax.set_xlabel("oncogenes minus tumor suppressors\nper segment", fontsize=5.5)
        if k == 0:
            ax.set_ylabel("mean copy number of the\ncancer cells", fontsize=5.5)
        else:
            ax.tick_params(labelleft=False)
        ax.tick_params(labelsize=5.5)
        S.stat(ax, f"Spearman {rho:.2f}".replace("-", "\u2212"), loc="upper left")
        S.despine(ax)
    print(S.save(fig, "supp_cn_selection"), out)


if __name__ == "__main__":
    main()
