"""Write a tumour's human-gene Visium and scRNA data, with the truth, for the copy-number tools of Fig 5,
optionally in the two arms of the programme-permutation contrast (P2).

Usage::

    # a registry tumour (ISCC_PAPER_OUT / ISCC_PAPER_TIER locate it), both P2 arms
    ISCC_HUMAN_REF=<refs> PYTHONPATH=src python validation/paper/tools/make_layer_dataset.py OUT \
        --run R1 --task disp0.9_seed1 --arms generated,permuted
    # or grow one on an arm-homologous genome
    ... make_layer_dataset.py OUT --grow [--scale small] [--target 15000] [--total-genes 6000]

The tumour is sampled as the paper's figures sample it (``common.assays``): one resection, dissociated
and sliced (``depth_frac`` 0.5) to the tier's ``max_cells``, Visium at the regime's settings
(``regime.VISIUM``, spot swapping on) over the tumour's grid, scRNA on every dissociated cell, with the
paper's assay seeds (Visium ``300 + seed``, scRNA ``100 + seed``). Over the human genes Visium takes
``HUMAN_VISIUM_KAPPA`` and scRNA ``dropout_mid=0`` at ``--n-reads`` UMIs per cell. Every arm then
measures those same cells with the same seeds; the arms differ only in the expression step:
``generated`` keeps each cell's programme activities, ``permuted`` shuffles them among the cells of
each type (``HumanGenomeLayer.expression(permute_programmes=...)``). Writes, per arm,
``OUT/<arm>/visium`` and ``OUT/<arm>/scrna`` (the layouts of :func:`iscc.data.human.write_visium` and
:func:`iscc.data.human.write_scrna`), and once ``OUT/truth_extra`` for score_tools.py:

    <assay>_barcodes.tsv  barcode, malignant, tumour_fraction, clone (the D9 clone: the lineage clade
                          from clones_from_clades; a spot takes its tumour cells' majority clone),
                          hypoxia, epithelial_fraction (of the cell's deme; a spot's mean), at_wall
                          (a tumour barcode whose deme holds at least 20% normal epithelium)
    <assay>_arm_cn.tsv    barcode, arm, cn: true total copy number over the arm's genes (a spot: the
                          mean over its tumour cells, 2 without any)

and ``OUT/manifest.json`` (settings, sizes, wall time per step). A paper-regime tumour has no
arm-homologous genome, so its segments are laid along the arms end to end (``allow_linear``), and its
copy-number segments cross chromosome boundaries; the truth is mapped the same way. The script needs
the paper pipeline (``validation/paper/regime.py`` and ``registry.py``) beside it.
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
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import realistic_regime as RR  # noqa: E402
from iscc.data import Visium, scRNA  # noqa: E402
from iscc.data.human import (HUMAN_VISIUM_KAPPA, HumanGenomeLayer, arm_genome,  # noqa: E402
                             check_tool_ready, write_scrna, write_visium)
from iscc.data.human.writers import barcodes  # noqa: E402
from iscc.integrations import clones_from_clades  # noqa: E402

WALL = 0.2          # a deme with at least this fraction of normal epithelium is at the wall


def _tumour(a):
    if a.grow:
        import regime as RG
        cfg = arm_genome(total_genes=a.total_genes, driver_scale=a.driver_scale)
        return RG.grow(seed=a.seed, scale=a.scale, target_cancer=a.target, genome=cfg["genome_params"],
                       selection=cfg["selection_params"])
    import registry as REG
    return REG.load(a.run, a.task)


def _cell_truth(t, layer, sample, cells, min_frac):
    """Per cell: malignant, D9 clone, hypoxia, epithelial fraction of its deme, and arm copy number."""
    gids = layer._gids(sample, cells)
    tum = layer.is_tumour(gids, t)
    t.cell_data = sample
    n_t = int(tum.sum())
    clones = clones_from_clades(t, min_cells=max(20, int(round(min_frac * n_t))),
                                cell_ids=[c for c, x in zip(cells, tum) if x]).reindex(cells)
    hyp = (sample["cell_microenv"]["hypoxia_level"].reindex(cells).to_numpy(float)
           if "cell_microenv" in sample else np.zeros(len(cells)))
    epi_deme, _ = t._compartment_fields()
    demes = sample["cell_deme"]["deme_id"].reindex(cells).to_numpy(int)
    epi = epi_deme[demes]
    cn_p, cn_m = layer.homolog_cn(gids, t)                           # cells x growth segments
    tot = (cn_p + cn_m)[:, layer.segment]                             # cells x human genes
    arms = layer.genes["arm"].to_numpy()
    arm_names = pd.unique(arms)
    arm_cn = np.stack([tot[:, arms == a].mean(axis=1) for a in arm_names], axis=1)
    return dict(tum=tum, clone=clones.astype(object).to_numpy(), hyp=hyp, epi=epi, arm_cn=arm_cn,
                arm_names=arm_names)


def _write_truth(out, assay, bcs, malignant, frac, clone, hyp, epi, arm_cn, arm_names):
    os.makedirs(out, exist_ok=True)
    pd.DataFrame({"barcode": bcs, "malignant": malignant.astype(int), "tumour_fraction": frac,
                  "clone": clone, "hypoxia": hyp, "epithelial_fraction": epi,
                  "at_wall": (malignant & (epi >= WALL)).astype(int)}).to_csv(
        os.path.join(out, f"{assay}_barcodes.tsv"), sep="\t", index=False)
    pd.DataFrame({"barcode": np.repeat(bcs, len(arm_names)), "arm": np.tile(arm_names, len(bcs)),
                  "cn": arm_cn.ravel()}).to_csv(os.path.join(out, f"{assay}_arm_cn.tsv"), sep="\t", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--run")
    ap.add_argument("--task")
    ap.add_argument("--grow", action="store_true")
    ap.add_argument("--arms", default="generated")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--permute-seed", type=int, default=7)
    ap.add_argument("--scale", default="small")
    ap.add_argument("--target", type=int, default=15_000)
    ap.add_argument("--total-genes", type=int, default=6_000)
    ap.add_argument("--driver-scale", type=float, default=0.4)
    ap.add_argument("--n-reads", type=float, default=10_000)
    ap.add_argument("--max-cells", type=int, default=None, help="default the tier's max_cells")
    ap.add_argument("--max-log-fold", type=float, default=5.0)
    ap.add_argument("--min-clone-frac", type=float, default=0.03)
    a = ap.parse_args()
    if not a.grow and not (a.run and a.task):
        sys.exit("give --run and --task, or --grow")
    os.makedirs(a.out, exist_ok=True)
    wall, t0 = {}, time.time()

    def lap(name):
        nonlocal t0
        wall[name] = round(time.time() - t0, 1)
        print(f"  {name}: {wall[name]} s", flush=True)
        t0 = time.time()

    import regime as RG
    t = _tumour(a)
    if getattr(getattr(t, "programs", None), "max_log_fold", None) is None and a.max_log_fold:
        t.programs.max_log_fold = float(a.max_log_fold)   # registry.load sets it; a grown tumour may not
    lap("load" if not a.grow else "grow")
    layer = HumanGenomeLayer(t, allow_linear=t.genome_params.get("arms") is None)
    max_cells = a.max_cells or RG.T["max_cells"]
    spec = RR.resect(t)
    dis = spec.dissociate(max_cells=max_cells)
    sec = spec.slice(depth_frac=0.5, max_cells=max_cells)
    lap("layer_and_sample")

    ready, sizes = [], {}
    truth_done = set()
    for arm in a.arms.split(","):
        expr = layer.expression(t, permute_programmes=(a.permute_seed if arm == "permuted" else None))
        vz = Visium(seed=300 + a.seed, kappa=HUMAN_VISIUM_KAPPA, **RG.VISIUM).run(
            sec, grid_side=t.grid_size, gene_lift=expr)
        av = layer.visium_allele_counts(vz, t, seed=a.seed)
        write_visium(os.path.join(a.out, arm, "visium"), vz, layer, allele=av, tumor=t, seed=a.seed)
        lap(f"visium_{arm}")
        dcells = list(dis["cell_exp"].index)
        sc = scRNA(n_cells=len(dcells), seed=100 + a.seed, dropout_mid=0.0, n_reads=a.n_reads).run(
            dis, cell_subset=dcells, gene_lift=expr)
        ac = layer.scrna_allele_counts(sc, dis, t, seed=a.seed)
        write_scrna(os.path.join(a.out, arm, "scrna"), sc, layer, dis, allele=ac, tumor=t, seed=a.seed)
        lap(f"scrna_{arm}")
        keep = np.flatnonzero((vz.obs["n_cells"] > 0).to_numpy())
        av_in = type(av)(av.A[keep].tocsr(), av.B[keep].tocsr(), av.snps, [av.barcodes[i] for i in keep])
        ready.append(check_tool_ready(vz.spot_counts.iloc[keep], layer.genes, allele=av_in).assign(assay="visium", arm=arm))
        ready.append(check_tool_ready(sc.counts_csr, layer.genes, allele=ac).assign(assay="scrna", arm=arm))
        sizes[arm] = dict(spots_in_tissue=int(keep.size), cells=int(sc.counts_csr.shape[0]),
                          mean_umis_per_cell=float(sc.obs["n_counts"].mean()),
                          mean_umis_per_spot=float(np.asarray(vz.spot_counts.iloc[keep].sum(axis=1)).mean()))
        if "visium" not in truth_done:                                  # the same cells in every arm
            placed, members = vz._grid["placed"], vz._grid["members"]
            pcells = list(placed["cell_exp"].index)
            ct = _cell_truth(t, layer, placed, pcells, a.min_clone_frac)
            all_bcs = barcodes(len(vz.spot_counts.index), seed=a.seed)
            rows = []
            for s in keep:
                m = members[s]
                tm = m[ct["tum"][m]]
                frac = tm.size / m.size
                cl = pd.Series(ct["clone"][tm]).dropna()
                rows.append((all_bcs[s], frac >= 0.5, frac, cl.mode().iat[0] if len(cl) else None,
                             ct["hyp"][m].mean(), ct["epi"][m].mean(),
                             ct["arm_cn"][tm].mean(axis=0) if tm.size else np.full(len(ct["arm_names"]), 2.0)))
            _write_truth(os.path.join(a.out, "truth_extra"), "visium", [r[0] for r in rows],
                         np.array([r[1] for r in rows]), np.array([r[2] for r in rows]),
                         np.array([r[3] for r in rows], dtype=object), np.array([r[4] for r in rows]),
                         np.array([r[5] for r in rows]), np.stack([r[6] for r in rows]), ct["arm_names"])
            cells = list(sc.obs.index)
            cc = _cell_truth(t, layer, dis, cells, a.min_clone_frac)
            _write_truth(os.path.join(a.out, "truth_extra"), "scrna", barcodes(len(cells), seed=a.seed),
                         cc["tum"], cc["tum"].astype(float), cc["clone"], cc["hyp"], cc["epi"],
                         cc["arm_cn"], cc["arm_names"])
            truth_done.add("visium")
            lap("truth")
    pd.concat(ready).to_csv(os.path.join(a.out, "ready.tsv"), sep="\t", index=False)
    man = dict(vars(a), cancer=int(t.get_cancer_size()), n_genes=int(layer.genes.shape[0]),
               arm_homologous=t.genome_params.get("arms") is not None, sizes=sizes, wall_s=wall)
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(man, f, indent=1, default=str)
    print(json.dumps(sizes), flush=True)


if __name__ == "__main__":
    main()
