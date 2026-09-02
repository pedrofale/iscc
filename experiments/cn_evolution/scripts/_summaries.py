"""Read the two-section summary tables the aggregator writes."""
import io
import os

import pandas as pd

AXES = ["scenario", "evo", "selection", "genome", "pop", "seed"]

FILES = {
    "sweep": "sweep_summary.tsv", "diversity": "diversity_summary.tsv",
    "growth": "growth_phase_summary.tsv", "landscape": "landscape_summary.tsv",
    "quality": "quality_summary.tsv", "reconstruction": "reconstruction_summary.tsv",
    "structure": "structure_summary.tsv",
}


def read_detail(path):
    """The per-seed detail section (everything before the summary section)."""
    txt = open(path).read().split("\n# Per-parameter-combo")[0]
    body = txt.split("\n", 1)[1] if txt.startswith("#") else txt
    return pd.read_csv(io.StringIO(body), sep="\t")


def load_all(results_dir):
    """``{name: DataFrame}`` for whichever summary tables exist."""
    out = {}
    for name, fname in FILES.items():
        p = os.path.join(results_dir, fname)
        if os.path.exists(p):
            try:
                df = read_detail(p)
            except Exception as e:
                print(f"  (skipping {fname}: {e})")
                continue
            if not df.empty:
                out[name] = df
    return out


def _join(base, right, on, how):
    """Merge, dropping columns the right frame duplicates.

    Several questions legitimately report a column of the same name (`n_clones` appears in the
    quality, reconstruction and diversity tables), so relying on merge suffixes collides as soon as
    a third table is joined. First writer wins; the per-question tables remain authoritative.
    """
    drop = [c for c in right.columns if c in base.columns and c not in on]
    return base.merge(right.drop(columns=drop), on=on, how=how)


def merged(results_dir):
    """All questions joined on their shared axes — run-level questions broadcast over clone sets."""
    tables = load_all(results_dir)
    if not tables:
        return pd.DataFrame()
    per_k = [t for n, t in tables.items() if "n_clones_requested" in t.columns]
    per_run = [t for n, t in tables.items() if "n_clones_requested" not in t.columns]
    base = per_k[0] if per_k else per_run[0]
    rest_k = per_k[1:] if per_k else []
    rest_run = per_run if per_k else per_run[1:]
    for t in rest_k:
        on = [c for c in AXES + ["n_clones_requested"] if c in base and c in t]
        base = _join(base, t, on, "outer")
    for t in rest_run:
        on = [c for c in AXES if c in base and c in t]
        base = _join(base, t, on, "left")
    return base
