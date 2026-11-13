"""CalicoST on a human-gene Visium dataset written by iscc.data.human.write_visium.

Run with CalicoST's environment (``bin/python-calicost``)::

    python calicost.py <dataset_dir> <out_dir> [--n-clones 3] [--code /opt/calicost]

Points CalicoST at ``<dataset>/outs`` (the Space Ranger directory) and ``<dataset>/snp_dir`` (the
phased allele counts, so its BAM preprocessing is skipped) with its GRCh38 resources, runs
``calicost_main.py``, and writes the common output of this directory (see calls.py): calls.tsv
(clone = CalicoST's clone label; no malignant call, since which clone is normal is read from its
copy-number profile), cn_arm.tsv (each spot's clone's total copy number per arm, from
``cnv_seglevel.tsv``, scale "total_cn"), run.json, and CalicoST's own output in native/. The
phylogeny (Startle, which needs CPLEX) is not run.
"""
import argparse
import glob
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import calls as CALLS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("out")
    ap.add_argument("--n-clones", type=int, default=3)
    ap.add_argument("--min-spots-per-clone", type=int, default=20)
    ap.add_argument("--code", default="/opt/calicost")
    ap.add_argument("--seed", type=int, default=0, help="CalicoST's gmm_random_state")
    a = ap.parse_args()
    t0 = time.time()
    native = os.path.join(a.out, "native")
    os.makedirs(native, exist_ok=True)
    res = os.path.join(a.code, "GRCh38_resources")
    with open(os.path.join(a.code, "configuration_cna")) as f:
        template = f.read()
    set_ = {
        "spaceranger_dir": os.path.join(a.dataset, "outs"),
        "snp_dir": os.path.join(a.dataset, "snp_dir"),
        "output_dir": native,
        "geneticmap_file": os.path.join(res, "genetic_map_GRCh38_merged.tab.gz"),
        "hgtable_file": os.path.join(res, "hgTables_hg38_gencode.txt"),
        "filtergenelist_file": os.path.join(res, "ig_gene_list.txt"),
        "filterregion_file": os.path.join(res, "HLA_regions.bed"),
        "n_clones": str(a.n_clones),
        "min_spots_per_clone": str(a.min_spots_per_clone),
        "gmm_random_state": str(a.seed),
    }
    lines = []
    for line in template.splitlines():
        key = line.split(":", 1)[0].strip()
        lines.append(f"{key} : {set_[key]}" if key in set_ and ":" in line else line)
    cfg = os.path.join(native, "configuration_cna")
    with open(cfg, "w") as f:
        f.write("\n".join(lines) + "\n")
    env = dict(os.environ, OMP_NUM_THREADS="1")
    r = subprocess.run([sys.executable, os.path.join(a.code, "src", "calicost", "calicost_main.py"), "-c", cfg],
                       env=env, capture_output=True, text=True)
    with open(os.path.join(native, "calicost.log"), "w") as f:
        f.write(r.stdout + "\n--- stderr ---\n" + r.stderr)
    if r.returncode != 0:
        sys.exit(f"CalicoST failed (exit {r.returncode}); see {native}/calicost.log\n{r.stderr[-2000:]}")

    run_dir = sorted(glob.glob(os.path.join(native, f"clone{a.n_clones}_rectangle*_w*")))[0]
    lab = pd.read_csv(os.path.join(run_dir, "clone_labels.tsv"), sep="\t", index_col=0)
    calls = pd.DataFrame({"barcode": lab.index.astype(str), "malignant": np.nan, "p_malignant": np.nan,
                          "clone": lab["clone_label"].astype(str).to_numpy()})
    cn_arm = None
    seg_files = sorted(glob.glob(os.path.join(run_dir, "cnv*_seglevel.tsv")))
    if seg_files:
        seg = pd.read_csv(seg_files[0], sep="\t")
        arm_table = CALLS.arms(a.dataset)
        mid = ((seg["START"] + seg["END"]) // 2).to_numpy()
        seg["arm"] = CALLS.arm_of(seg["CHR"], mid, arm_table)
        seg["len"] = seg["END"] - seg["START"]
        rows = []
        for clone in calls["clone"].unique():
            a_col, b_col = f"clone{clone} A", f"clone{clone} B"
            if a_col not in seg or b_col not in seg:
                continue
            tot = (seg[a_col] + seg[b_col]) * seg["len"]
            per_arm = (tot.groupby(seg["arm"]).sum() / seg["len"].groupby(seg["arm"]).sum()).drop("", errors="ignore")
            rows.append(pd.DataFrame({"clone": clone, "arm": per_arm.index, "value": per_arm.to_numpy()}))
        if rows:
            per_clone = pd.concat(rows)
            cn_arm = calls[["barcode", "clone"]].merge(per_clone, on="clone").drop(columns="clone").assign(scale="total_cn")
    CALLS.write(a.out, calls, cn_arm, tool="calicost", code=a.code, n_clones=a.n_clones, seed=a.seed,
                min_spots_per_clone=a.min_spots_per_clone, run_dir=os.path.basename(run_dir),
                seglevel=os.path.basename(seg_files[0]) if seg_files else None, phylogeny="not run (Startle needs CPLEX)",
                wall_s=round(time.time() - t0, 1))
    print(f"calicost done: clones {calls['clone'].value_counts().to_dict()}", flush=True)


if __name__ == "__main__":
    main()
