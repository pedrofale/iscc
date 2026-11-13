"""Collect the runtime logs of ``runtime_job.sh`` into ``results/runtime.csv``.

    python runtime_collect.py <log dir> <out.csv>

One row per log: tool, cancer cells, seed, node, CPU, wall time and peak memory as ``/usr/bin/time -v``
measured them, and whether the run completed. A tool's own JSON line, when it printed one, adds its
stage times (for iscc: grow, section, Visium; for SISTEM: growth).
"""
import glob
import json
import os
import re
import sys

import pandas as pd


def _seconds(s):
    parts = [float(p) for p in s.strip().split(":")]
    return sum(p * 60 ** i for i, p in enumerate(reversed(parts)))


def parse(path):
    text = open(path, errors="replace").read()
    head = re.search(r"TOOL=(\S+) N_CANCER=(\d+) SEED=(\d+) NODE=(\S+) CPU=(.*)", text)
    if not head:
        return None
    rec = dict(tool=head.group(1), n_cancer=int(head.group(2)), seed=int(head.group(3)),
               node=head.group(4), cpu=head.group(5).strip())
    walls = re.findall(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)", text)
    rss = re.findall(r"Maximum resident set size \(kbytes\): (\d+)", text)
    codes = re.findall(r"Exit status: (\d+)", text)
    rec["wall_s"] = _seconds(walls[-1]) if walls else float("nan")
    rec["peak_mb"] = int(rss[-1]) / 1024 if rss else float("nan")
    rec["ok"] = bool(codes) and codes[-1] == "0" and "FAILED" not in text
    for line in text.splitlines():
        if line.startswith("{") and '"tool"' in line:
            j = json.loads(line)
            rec.update({f"stage_{k}": v for k, v in j.items()
                        if k.endswith("_s") or k in ("spots", "section_cells", "n_total", "reached")})
            for k in ("n_cancer", "cells_reached"):       # the size each tool actually reached
                if k in j:
                    rec["n_reached"] = j[k]
    return rec


def main():
    logs, out = sys.argv[1], sys.argv[2]
    rows = [r for r in (parse(p) for p in sorted(glob.glob(os.path.join(logs, "rt-*.out")))) if r]
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"{len(rows)} runs -> {out}")


if __name__ == "__main__":
    main()
