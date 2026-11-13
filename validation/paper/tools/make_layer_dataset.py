"""Grow a paper-regime tumour on an arm-homologous growth genome and write its human-gene Visium and
scRNA data, with the truth, for the copy-number tools of Fig 5.

Usage::

    ISCC_HUMAN_REF=<refs> PYTHONPATH=src python validation/paper/tools/make_layer_dataset.py OUT \
        [--seed 1] [--scale small] [--target 15000] [--total-genes 6000] [--driver-scale 0.4] \
        [--n-cells 2000] [--max-cells 20000]

Writes ``OUT/visium`` and ``OUT/scrna`` (the layouts of :func:`iscc.data.human.write_visium` and
:func:`iscc.data.human.write_scrna`), ``OUT/ready.tsv`` (:func:`iscc.data.human.check_tool_ready` per
assay) and ``OUT/manifest.json`` (settings, sizes, wall time per step). The tool runners in this
directory read ``OUT/visium`` and ``OUT/scrna``. It needs the paper pipeline (``validation/paper/regime.py``,
from the paper-pipeline pull request) beside it.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
sys.path.insert(0, PAPER)
sys.path.insert(0, os.path.dirname(PAPER))
import pandas as pd  # noqa: E402

import realistic_regime as RR  # noqa: E402
import regime as RG  # noqa: E402
from iscc.data import Visium, scRNA  # noqa: E402
from iscc.data.human import (HUMAN_VISIUM_KAPPA, HumanGenomeLayer, arm_genome,  # noqa: E402
                             check_tool_ready, write_scrna, write_visium)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--scale", default="small")
    ap.add_argument("--target", type=int, default=15_000)
    ap.add_argument("--total-genes", type=int, default=6_000)
    ap.add_argument("--driver-scale", type=float, default=0.4)
    ap.add_argument("--n-cells", type=int, default=2_000)
    ap.add_argument("--max-cells", type=int, default=20_000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    wall, t0 = {}, time.time()

    def lap(name):
        nonlocal t0
        wall[name] = round(time.time() - t0, 1)
        print(f"  {name}: {wall[name]} s", flush=True)
        t0 = time.time()

    cfg = arm_genome(total_genes=a.total_genes, driver_scale=a.driver_scale)
    t = RG.grow(seed=a.seed, scale=a.scale, target_cancer=a.target, genome=cfg["genome_params"],
                selection=cfg["selection_params"])
    lap("grow")
    layer = HumanGenomeLayer(t)
    lap("layer")

    sec = RR.section(t, max_cells=a.max_cells)
    vz = Visium(seed=a.seed, kappa=HUMAN_VISIUM_KAPPA).run(sec, gene_lift=layer.lift)
    av = layer.visium_allele_counts(vz, t, seed=a.seed)
    lap("visium")
    write_visium(os.path.join(a.out, "visium"), vz, layer, allele=av, tumor=t, seed=a.seed)
    lap("write_visium")

    dis = RR.dissociated(t, max_cells=a.max_cells, seed=a.seed)
    sc = scRNA(n_cells=a.n_cells, seed=a.seed).run(dis, gene_lift=layer.lift)
    ac = layer.scrna_allele_counts(sc, dis, t, seed=a.seed)
    lap("scrna")
    write_scrna(os.path.join(a.out, "scrna"), sc, layer, dis, allele=ac, tumor=t, seed=a.seed)
    lap("write_scrna")

    in_tissue = (vz.obs["n_cells"] > 0).to_numpy()
    keep = in_tissue.nonzero()[0]
    av_in = type(av)(av.A[keep].tocsr(), av.B[keep].tocsr(), av.snps, [av.barcodes[i] for i in keep])
    ready = pd.concat([
        check_tool_ready(vz.spot_counts.loc[in_tissue], layer.genes, allele=av_in).assign(assay="visium"),
        check_tool_ready(sc.observed_counts, layer.genes, allele=ac).assign(assay="scrna"),
    ])
    ready.to_csv(os.path.join(a.out, "ready.tsv"), sep="\t", index=False)
    man = dict(vars(a), cancer=int(t.get_cancer_size()), reached_target=bool(t.reached_target),
               n_spots_in_tissue=int(in_tissue.sum()), n_cells_scrna=int(sc.observed_counts.shape[0]),
               n_genes=int(layer.genes.shape[0]), wall_s=wall)
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(man, f, indent=1)
    print(ready.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
