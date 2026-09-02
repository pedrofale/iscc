"""Collect per-run JSON/TSV outputs into one summary table.

One aggregator for all seven questions: the path grammar is the same everywhere, so the dataset
axes are recovered from the path and the payload is flattened alongside them. Emits the same
two-section layout the SISTEM study uses — per-seed detail, then a per-parameter-combo summary —
so the two studies' tables can be read side by side.
"""
import argparse, glob, json, os, re

import numpy as np
import pandas as pd

# results/{scenario}/{evo}/{selection}/{genome}/{pop}/seed{N}/...
KEY = re.compile(r"/(?P<scenario>[^/]+)/(?P<evo>[^/]+)/(?P<selection>[^/]+)/"
                 r"(?P<genome>[^/]+)/(?P<pop>[^/]+)/seed(?P<seed>\d+)/")
AXES = ["scenario", "evo", "selection", "genome", "pop", "seed"]


def _axes_from(path):
    m = KEY.search(path.replace(os.sep, "/"))
    if not m:
        return None
    d = m.groupdict()
    d["seed"] = int(d["seed"])
    k = re.search(r"/K(?P<k>\d+)/", path.replace(os.sep, "/"))
    if k:
        d["n_clones_requested"] = int(k.group("k"))
    return d


def _flatten(obj, prefix=""):
    out = {}
    for k, v in (obj or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "_"))
        elif isinstance(v, list):
            if v and all(isinstance(x, (int, float, type(None))) for x in v):
                arr = np.array([np.nan if x is None else x for x in v], dtype=float)
                out[key + "_mean"] = float(np.nanmean(arr)) if arr.size else np.nan
                out[key + "_max"] = float(np.nanmax(arr)) if arr.size else np.nan
            # non-numeric lists (e.g. the colonisation curve) are series, not summary values
        else:
            out[key] = v
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", required=True)
    p.add_argument("--pattern", required=True,
                   help="glob relative to results-dir, e.g. '*/*/*/*/*/seed*/pop/sweep_metrics.json'")
    p.add_argument("--output", required=True)
    p.add_argument("--kind", choices=["json", "tsv"], default="json")
    p.add_argument("--tsv-summary", default="last",
                   help="for --kind tsv: 'last' row, or 'mean' of every numeric column")
    a = p.parse_args()

    rows = []
    for path in sorted(glob.glob(os.path.join(a.results_dir, a.pattern))):
        axes = _axes_from(path)
        if axes is None:
            continue
        if a.kind == "json":
            with open(path) as f:
                payload = json.load(f)
            rows.append({**axes, **_flatten(payload)})
        else:
            df = pd.read_csv(path, sep="\t")
            if df.empty:
                continue
            vals = (df.iloc[-1].to_dict() if a.tsv_summary == "last"
                    else df.mean(numeric_only=True).to_dict())
            rows.append({**axes, **{str(k): v for k, v in vals.items()}})

    if not rows:
        pd.DataFrame(columns=AXES).to_csv(a.output, sep="\t", index=False)
        print(f"no inputs matched {a.pattern!r}")
        return

    detail = pd.DataFrame(rows).sort_values([c for c in AXES if c in rows[0]])
    group = [c for c in ("scenario", "evo", "selection", "genome", "pop", "n_clones_requested")
             if c in detail.columns]
    # Grouping columns must be excluded from the aggregated set: `n_clones_requested` is both an
    # axis and a numeric column, and `reset_index()` would then try to add it twice.
    numeric = detail.select_dtypes(include="number").columns.difference(["seed"] + group)
    summary = (detail.groupby(group, dropna=False)[list(numeric)].mean().reset_index()
               if group and len(numeric) else pd.DataFrame())
    if len(group):
        summary.insert(len(group), "n_seeds",
                       detail.groupby(group, dropna=False).size().values)

    with open(a.output, "w") as f:
        f.write("# Per-seed detail\n")
        detail.to_csv(f, sep="\t", index=False)
        if not summary.empty:
            f.write("\n# Per-parameter-combo summary (mean over seeds)\n")
            summary.to_csv(f, sep="\t", index=False)
    print(f"{len(detail)} rows -> {a.output}")


if __name__ == "__main__":
    main()
