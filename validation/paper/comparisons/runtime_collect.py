"""Collect the runtime logs of ``runtime_job.sh`` into ``results/runtime.csv``.

    python runtime_collect.py <log dir> <out.csv>

One row per log: tool, cancer cells, seed, node, CPU, wall time and peak memory as ``/usr/bin/time -v``
measured them, and whether the run completed. A tool's own JSON line, when it printed one, adds its
stage times (for iscc: grow, section, Visium; for SISTEM: growth).

iscc runs carry a variant: ``before`` (dense cell tables, the runs before chunked materialisation),
``full``, ``visium`` or ``visium-low`` (``make_cell_data(layers="visium")``, plus ``low_memory``),
read from the ``VARIANT=`` header of the chunked-materialisation benchmark's logs or from the
``layers`` and ``low_memory`` fields of iscc's JSON line. When an iscc variant run was logged more
than once, the latest log counts; other tools keep every attempt (a failed one has ``ok`` False).
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
    head = re.search(r"(TOOL|VARIANT)=(\S+) N_CANCER=(\d+) SEED=(\d+) NODE=(\S+) CPU=(.*)", text)
    if not head:
        return None
    tool, variant = (head.group(2), "") if head.group(1) == "TOOL" else ("iscc", head.group(2))
    rec = dict(tool=tool, variant=variant, n_cancer=int(head.group(3)), seed=int(head.group(4)),
               node=head.group(5), cpu=head.group(6).strip(), mtime=os.path.getmtime(path))
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
            if rec["tool"] == "iscc" and not rec["variant"]:
                rec["variant"] = ("before" if "layers" not in j else "full" if j["layers"] is None
                                  else "visium-low" if j.get("low_memory") else "visium")
    return rec


def main():
    """runtime_collect.py <out.csv> <log dir> [<log dir> ...]"""
    out, dirs = sys.argv[1], sys.argv[2:]
    rows = [r for d in dirs for r in (parse(p) for p in sorted(glob.glob(os.path.join(d, "*.out")))) if r]
    df = pd.DataFrame(rows).sort_values("mtime")
    iscc = df[df.tool == "iscc"].drop_duplicates(["variant", "n_cancer", "seed"], keep="last")
    df = pd.concat([iscc, df[df.tool != "iscc"]]).drop(columns="mtime")
    df.sort_values(["tool", "variant", "n_cancer", "seed"]).to_csv(out, index=False)
    print(f"{len(df)} runs -> {out}")


if __name__ == "__main__":
    main()
