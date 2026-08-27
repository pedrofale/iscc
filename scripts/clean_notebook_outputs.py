#!/usr/bin/env python
"""Strip terminal noise from executed notebooks.

Two kinds, both artifacts of the machine that ran the notebook rather than results:

* **Progress bars.** tqdm writes one stream output per redraw, so a 300-epoch fit lands ~600
  near-identical lines in the .ipynb. On the rendered docs page they bury the log lines that matter.
* **Third-party import warnings.** dask, numba, anndata and tqdm emit deprecation and environment
  warnings at import. They say nothing about the tumour, and they publish the absolute path of the
  site-packages directory they came from — i.e. the author's home directory — to the docs site.

Warnings raised by ``iscc`` itself are KEPT: those are addressed to the reader. So is every other
output, including the real tools' progress logs (RCTD, Numbat, bwa), which are the evidence that the
tool actually ran.

Usage:  python scripts/clean_notebook_outputs.py notebooks/*.ipynb
"""
import json
import re
import sys

PROGRESS = re.compile(r"(\d+%\|)|(\d+it \[)|(it/s[,\]])|(s/it[,\]])")

# A warning whose FILE lives in site-packages — i.e. raised by a dependency, not by iscc. Matching on
# site-packages is what keeps iscc's own warnings (which come from src/iscc/...) in the notebook.
DEP_WARNING = re.compile(r"site-packages/.*:\d+:\s*\w*(Warning|Error)")
# The indented source line a warning emitter prints under its location line.
WARN_BODY = re.compile(r"^\s+\S")


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
            src = out.get("text", [])
            lines, skip_body = [], False
            for ln in src:
                if PROGRESS.search(ln):
                    continue
                if DEP_WARNING.search(ln):
                    skip_body = True          # drop the location line and its source line
                    continue
                if skip_body and WARN_BODY.match(ln):
                    skip_body = False
                    continue
                skip_body = False
                lines.append(ln)
            removed += len(src) - len(lines)
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
