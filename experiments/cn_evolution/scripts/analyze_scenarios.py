"""The structured-vs-unstructured contrast, per question, at matched parameters.

The pre-registered expectation (from `origin_confinement`'s own rationale in the engine config) is
that a confined structured field carries a LARGER truncal copy-number layer -- higher
`trunk_fraction`, lower `mrca_depth_frac` -- and is correspondingly HARDER to reconstruct at matched
selection strength. If that does not reproduce, it is a finding about the confinement model, not a
bug to tune away.
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _summaries import merged

METRICS = [
    ("trunk_fraction", "Q5 truncal CN burden"),
    ("mrca_depth_frac", "Q1 coalescent depth"),
    ("nj_event_rf", "Q6 RF from CN events (lower = better)"),
    ("nj_rf_floor", "Q6 polytomy floor for reference"),
    ("phylo_signal_spearman", "Q6 phylogenetic signal"),
    ("frac_gens_in_K", "Q3 fraction of the run density-limited"),
    ("D", "Q2 clonal diversity"),
    ("fga", "Q4 fraction genome altered"),
    ("t_escape", "Q7 stromal escape generation"),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()

    df = merged(a.results_dir)
    if df.empty or "scenario" not in df.columns:
        print("no summary tables found")
        return

    have = [(m, d) for m, d in METRICS if m in df.columns]
    tbl = (df.groupby("scenario")[[m for m, _ in have]]
             .agg(["mean", "count"]))
    tbl.to_csv(a.output, sep="\t")

    print(f"{'metric':<24} {'meaning':<42} " +
          " ".join(f"{s[:20]:>20}" for s in sorted(df["scenario"].unique())))
    for m, meaning in have:
        vals = df.groupby("scenario")[m].mean()
        cells = " ".join(f"{vals.get(s, float('nan')):>20.3f}"
                         for s in sorted(df["scenario"].unique()))
        print(f"{m:<24} {meaning:<42} {cells}")

    n_seeds = df.groupby("scenario")["seed"].nunique().min() if "seed" in df.columns else 0
    print(f"\n{len(df)} rows -> {a.output}")
    if n_seeds < 3:
        print(f"NOTE: {n_seeds} seed(s) per scenario -- indicative only, not a result. "
              "Run the full grid before drawing conclusions.")


if __name__ == "__main__":
    main()
