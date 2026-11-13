"""Growth criteria of stored registry tumours, to compare a design on two growth genomes (e.g. R11 on
the arm-homologous genome against R0 / R1 on the paper regime's 12 segments).

Usage::

    ISCC_PAPER_OUT=... ISCC_PAPER_TIER=full PYTHONPATH=src python validation/paper/tools/compare_growth.py \
        OUT.csv R0:seed1 R11:seed1 R1:disp0.1_seed1 R11:disp0.1_seed1 ...

Per tumour: size, time, the in-situ phase (the first time a cancer cell sits in a stromal deme, from the
spatial history, minus the founding acinus's release, from the events; resolution one snapshot
interval), and the regime tuning's summaries (``sweep_score.score_tumour``: fraction of the genome
altered, mean ploidy, share of cells from a doubling, clones, the largest clonal alteration's share,
truncal segments, stroma share).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import registry as REG  # noqa: E402
import sweep_score as SS  # noqa: E402


def in_situ(t):
    rel = [e["time"] for e in t.events if e.get("event") == "origin_release"]
    release = float(rel[0]) if rel else np.nan
    first = np.nan
    if t.gland_id is not None:
        snap = t.spatial_snapshots()
        if len(snap):
            stroma = snap[(snap["cell_type"] == "cancer") & (t.gland_id[snap["deme"].to_numpy(int)] == -1)]
            if len(stroma):
                first = float(stroma["time"].min())
    return release, first, first - release


def main(out, specs):
    rows = []
    for spec in specs:
        run, task = spec.split(":")
        t = REG.load(run, task)
        release, first, phase = in_situ(t)
        sc = SS.score_tumour(t)
        rows.append(dict(run=run, task=task, n_segments=int(t.genome_params["n_segments"]),
                         arm_homologous=t.genome_params.get("arms") is not None,
                         cancer=int(t.get_cancer_size()), time=float(t.time),
                         reached_target=bool(getattr(t, "reached_target", True)),
                         n_genotypes=len(t.genotypes), release_time=release, first_stroma_time=first,
                         in_situ=phase,
                         **{k: sc.get(k) for k in ("frac_genome_altered", "mean_ploidy", "pct_wgd", "n_clones",
                                                   "max_cna_freq", "n_truncal_segments", "stroma_pct")}))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(out, index=False)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
