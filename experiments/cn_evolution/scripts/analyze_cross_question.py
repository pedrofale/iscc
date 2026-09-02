"""Relationships BETWEEN the seven questions.

Deliberately not a dependency of any question. Each metric set is computed on its own terms; the
correlations here are a finding about the simulations, not something baked into how any metric was
defined. Nothing in this script feeds back into the per-question outputs.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _summaries import merged

# (x, y, what a relationship would mean)
PAIRS = [
    ("frac_gens_in_K", "nj_event_rf",
     "Q3 -> Q6: does time spent density-limited make the tree harder to recover?"),
    ("trunk_fraction", "nj_event_rf",
     "Q5 -> Q6: does truncal CN burden drive non-reconstructability?"),
    ("mrca_depth_frac", "trunk_fraction",
     "Q1 -> Q5: does a shallow genealogy show up as a big shared trunk?"),
    ("fitness_slope_per_1k", "mrca_depth_frac",
     "Q1: does active selection collapse the genealogy?"),
    ("D", "nj_event_rf", "Q2 -> Q6: does clonal diversity help reconstruction?"),
    ("fga", "nj_breakpoint_rf", "Q4 -> Q6: does a more altered genome give more usable breakpoints?"),
    ("wgd_frac", "fga", "Q4: does whole-genome doubling inflate the altered fraction?"),
    ("n_clones_requested", "nj_event_rf", "sampling depth -> Q6"),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--min-n", type=int, default=4)
    a = p.parse_args()

    df = merged(a.results_dir)
    if df.empty:
        print("no summary tables found")
        pd.DataFrame(columns=["x", "y", "n", "spearman", "pearson"]).to_csv(
            a.output, sep="\t", index=False)
        return

    from scipy.stats import pearsonr, spearmanr
    rows = []
    for x, y, meaning in PAIRS:
        if x not in df.columns or y not in df.columns:
            continue
        sub = df[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(sub) < a.min_n or sub[x].std() == 0 or sub[y].std() == 0:
            rows.append(dict(x=x, y=y, n=len(sub), spearman=np.nan, pearson=np.nan,
                             meaning=meaning))
            continue
        rows.append(dict(x=x, y=y, n=len(sub),
                         spearman=float(spearmanr(sub[x], sub[y]).correlation),
                         pearson=float(pearsonr(sub[x], sub[y])[0]),
                         meaning=meaning))
    out = pd.DataFrame(rows)
    out.to_csv(a.output, sep="\t", index=False)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n{len(df)} rows -> {a.output}")
    if len(df) < 10:
        print("NOTE: too few runs for these correlations to mean anything -- run the full grid.")


if __name__ == "__main__":
    main()
