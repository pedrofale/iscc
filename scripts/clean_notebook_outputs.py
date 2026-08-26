#!/usr/bin/env python
"""Strip terminal progress-bar noise from executed notebooks.

tqdm writes one stream output per redraw, so a 300-epoch fit lands ~600 near-identical
lines in the .ipynb. They are a terminal artifact, not a result: on the rendered docs page
they bury the log lines that matter. This drops them and leaves every other output alone.

Usage:  python scripts/clean_notebook_outputs.py notebooks/*.ipynb
"""
import json
import re
import sys

PROGRESS = re.compile(r"(\d+%\|)|(\d+it \[)|(it/s[,\]])|(s/it[,\]])")


def clean(path):
    nb = json.load(open(path))
    removed = 0
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        kept = []
        for out in cell.get("outputs", []):
            if out.get("output_type") != "stream":
                kept.append(out)
                continue
            lines = [ln for ln in out.get("text", []) if not PROGRESS.search(ln)]
            removed += len(out.get("text", [])) - len(lines)
            if any(ln.strip() for ln in lines):
                out["text"] = lines
                kept.append(out)
        cell["outputs"] = kept
    if removed:
        json.dump(nb, open(path, "w"), indent=1)
    return removed


if __name__ == "__main__":
    for p in sys.argv[1:]:
        n = clean(p)
        print(f"{p}: removed {n} progress lines" if n else f"{p}: clean")
