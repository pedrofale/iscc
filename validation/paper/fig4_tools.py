"""Fig 4b on one tumour: the single-modality copy-number tools against the truth, by niche.

Builds under ``--work`` (a job's $SNIC_TMP) the tumour's human-gene scRNA of the dissociated cells, drawn as
fig4_chain.py's 4c draws it, and Visium of the 50,000-cell section, drawn as its 4d draws it (the same
samples, seeds and settings, so the same counts); writes both in the tools' input layouts
(``iscc.data.human.write_scrna`` / ``write_visium``) with a truth per barcode; runs the tools of Fig 4b side by
side through ``tools/run_tool.sh`` with the settings of ``tools/submit_tools.sh``; and scores every run in
each niche with ``score_tools.score_niches``.

Tools: on scRNA, CopyKAT (with and without the normal barcodes as its reference), Numbat and inferCNV
(infercnvpy); on Visium, CopyKAT (both ways), CalicoST, Clonalscope and Cancer-Finder. CalicoST and
Clonalscope make no malignancy call and Cancer-Finder no copy-number call; those cells of the score table
stay empty.

Reference protocol (``fig4_chain.cell_classes``, ``visium_reference`` and ``write_reference``, which 4d's
Clonalscope runs share), as the tools are used in practice: a tool that takes a normal reference (CopyKAT with
one, Numbat, inferCNV, Clonalscope, all through ``clonalscope/celltype.tsv``) is given the sample's
microenvironment normals, the cells a user annotates by cell type from markers: every non-epithelial normal
cell (stromal, immune, any other non-epithelial type); on Visium, the spots whose cells are all such
normals (a pathologist's stroma annotation). Normal epithelial cells are never in the reference: a user
cannot tell them from tumour cells. The malignancy AUC is read, for every tool alike, on the
epithelial-lineage barcodes only (``scored``): cancer cells against normal epithelial cells; on Visium the
spots holding any epithelial-lineage cell, a spot malignant when at least half its cells are cancer cells
(the rule of make_layer_dataset.py). No scored barcode is in any reference.

Truth per barcode: ``malignant`` (a cancer cell; a spot whose cells are at least half cancer, as Fig 5 and
make_layer_dataset.py define it), ``compartment`` (the cell's deme: 0 stroma, 1 the epithelial wall of a
duct, 2 its lumen; a spot takes the compartment most of its cells sit in), ``hypoxia`` (the cell's; a spot's
mean) and the true copy number per arm (a spot: the mean over its cancer cells, 2 without any).

Only summaries reach ``--out``/b/<run>_<task>/:

    scores.csv      tool, run, assay, niche, n, n_malignant, n_normal, n_scored, auc, cn_r, cn_scale, wall_s
    runs.json       each run's exit status, wall time and run.json, and the tail of its log when it failed
    barcodes.tsv.gz the truth per barcode (assay, barcode, malignant, tumour_fraction, compartment, hypoxia)
    calls.tsv.gz    every run's calls (run, assay, barcode, malignant, p_malignant, clone)
    cn_arm.tsv.gz   every run's per-arm values (run, assay, barcode, arm, value, scale)
    truth_arm_cn.tsv.gz  the true copy number per arm (assay, barcode, arm, cn)

so that the runs can be scored again without rerunning them (``--rescore``, on the login node: seconds).

    python validation/paper/fig4_tools.py --run R11 --task seed1 --out OUT --work $SNIC_TMP
    python validation/paper/fig4_tools.py --run R11 --task seed1 --out OUT --work - --rescore
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(HERE, "tools")
for p in (HERE, os.path.dirname(HERE), TOOLS):
    if p not in sys.path:
        sys.path.insert(0, p)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import fig4_chain as F  # noqa: E402

# (run name, tool, assay, cores, run_tool.sh arguments after the output directory), longest first; the
# arguments are submit_tools.sh's with TOOL_SEED 0
RUNS = [
    ("numbat-scrna", "numbat", "scrna", 4, ["2", "50", "0"]),
    ("calicost-visium", "calicost", "visium", 1, ["--seed", "0"]),
    ("copykat-scrna", "copykat", "scrna", 3, ["0"]),
    ("copykat-scrna-ref", "copykat", "scrna", 3, ["0", "1"]),
    ("copykat-visium", "copykat", "visium", 2, ["0"]),
    ("copykat-visium-ref", "copykat", "visium", 2, ["0", "1"]),
    ("clonalscope-visium", "clonalscope", "visium", 1, ["20", "200"]),
    ("infercnvpy-scrna", "infercnvpy", "scrna", 1, ["--seed", "0"]),
    ("cancerfinder-visium", "cancerfinder", "visium", 1, []),
]


# the runs whose tool reads the normal reference of clonalscope/celltype.tsv
REFERENCE_RUNS = ("copykat-scrna-ref", "copykat-visium-ref", "numbat-scrna", "infercnvpy-scrna",
                  "clonalscope-visium")


def _write_reference(ds, bc):
    """The reference the tools read (fig4_chain.write_reference): the barcodes of ``bc["reference"]``."""
    F.write_reference(ds, bc["barcode"], bc["reference"])


def _compartments(t):
    """Per deme: 0 stroma, 1 the epithelial wall of a duct, 2 its lumen."""
    n = len(t.demes)
    comp = np.zeros(n, dtype=np.int8)
    if t.gland_id is not None:
        gl = np.asarray(t.gland_id)
        comp[gl >= 0] = 1
        for lumen in (t.gland_lumen_demes or []):
            comp[np.asarray(lumen, dtype=int)] = 2
    return comp


def _arm_cn(t, layer, gids):
    """Per cell, the true total copy number averaged over each arm's human genes (cells x arms)."""
    arms = layer.genes["arm"].to_numpy()
    names = pd.unique(arms)
    uniq, inv = np.unique(np.asarray([str(g) for g in gids]), return_inverse=True)
    lookup = {str(g): g for g in gids}
    cn_p, cn_m = layer.homolog_cn([lookup[u] for u in uniq], t)
    tot = (cn_p + cn_m)[:, layer.segment].astype(float)
    per = np.stack([tot[:, arms == a].mean(axis=1) for a in names], axis=1)
    return per[inv], names


def build(a, work):
    """Write the scRNA and Visium datasets and their truth under ``work``; returns the dataset directories."""
    import regime as RG
    from iscc.data import Visium
    from iscc.data.human import HUMAN_VISIUM_KAPPA, write_scrna, write_visium
    from iscc.data.human.writers import barcodes
    t, layer, dis, sec = F.sample(a.run, a.task, a.section_max_cells)
    comp = _compartments(t)
    truth_dir = os.path.join(work, "truth_extra")
    os.makedirs(truth_dir, exist_ok=True)
    rows_bc = []

    # scRNA of the dissociated cells, as fig4_chain.rna_counts draws it
    sc = F.rna_counts(t, layer, dis, seed=a.seed)
    ac = layer.scrna_allele_counts(sc, dis, t, seed=a.seed)
    ds_rna = os.path.join(work, "scrna")
    write_scrna(ds_rna, sc, layer, dis, allele=ac, tumor=t, seed=a.seed)
    cells = list(sc.obs.index)
    bcs = barcodes(len(cells), seed=a.seed)
    gids = F._gids(dis, cells)
    tum = layer.is_tumour(gids, t)
    demes = dis["cell_deme"]["deme_id"].reindex(cells).to_numpy(int)
    hyp = dis["cell_microenv"]["hypoxia_level"].reindex(cells).to_numpy(float)
    cn, arm_names = _arm_cn(t, layer, gids)
    cls = F.cell_classes(t, gids)
    bc = pd.DataFrame({"barcode": bcs, "malignant": tum.astype(int), "tumour_fraction": tum.astype(float),
                       "compartment": comp[demes], "hypoxia": hyp, "epithelial": (cls == 1).astype(int),
                       "reference": (cls == 0).astype(int), "scored": (cls > 0).astype(int)})
    _write_reference(ds_rna, bc)
    bc.to_csv(os.path.join(truth_dir, "scrna_barcodes.tsv"), sep="\t", index=False)
    pd.DataFrame({"barcode": np.repeat(bcs, len(arm_names)), "arm": np.tile(arm_names, len(bcs)),
                  "cn": cn.ravel()}).to_csv(os.path.join(truth_dir, "scrna_arm_cn.tsv"), sep="\t", index=False)
    rows_bc.append(bc.assign(assay="scrna"))
    del sc, ac

    # Visium of the section, as fig4_chain.run_spatial draws it
    expr = layer.expression(t)
    vz = Visium(seed=300 + a.seed, kappa=HUMAN_VISIUM_KAPPA, **RG.VISIUM).run(sec, grid_side=t.grid_size,
                                                                                 gene_lift=expr)
    av = layer.visium_allele_counts(vz, t, seed=a.seed)
    ds_vis = os.path.join(work, "visium")
    write_visium(ds_vis, vz, layer, allele=av, tumor=t, seed=a.seed)
    placed, members = vz._grid["placed"], vz._grid["members"]
    pcells = list(placed["cell_exp"].index)
    pg = F._gids(placed, pcells)
    ptum = layer.is_tumour(pg, t)
    pcls = F.cell_classes(t, pg)
    pdeme = placed["cell_deme"]["deme_id"].reindex(pcells).to_numpy(int)
    phyp = placed["cell_microenv"]["hypoxia_level"].reindex(pcells).to_numpy(float)
    pcn, _ = _arm_cn(t, layer, pg)
    all_bcs = barcodes(len(vz.spot_counts.index), seed=a.seed)
    keep = np.flatnonzero(np.asarray(vz.obs["n_cells"]) > 0)
    ref = F.visium_reference(pcls, members, keep)
    rows, cn_rows = [], []
    for s, r in zip(keep, ref):
        m = members[s]
        tm = m[ptum[m]]
        frac = tm.size / m.size
        rows.append((all_bcs[s], int(frac >= 0.5), frac, int(np.bincount(comp[pdeme[m]], minlength=3).argmax()),
                     float(phyp[m].mean()), int((pcls[m] == 1).sum()), int(r), int(1 - r)))
        cn_rows.append(pcn[tm].mean(axis=0) if tm.size else np.full(len(arm_names), 2.0))
    bcv = pd.DataFrame(rows, columns=["barcode", "malignant", "tumour_fraction", "compartment", "hypoxia",
                                      "epithelial", "reference", "scored"])
    _write_reference(ds_vis, bcv)
    bcv.to_csv(os.path.join(truth_dir, "visium_barcodes.tsv"), sep="\t", index=False)
    cnv = np.stack(cn_rows)
    pd.DataFrame({"barcode": np.repeat(bcv["barcode"].to_numpy(), len(arm_names)),
                  "arm": np.tile(arm_names, len(bcv)), "cn": cnv.ravel()}).to_csv(
        os.path.join(truth_dir, "visium_arm_cn.tsv"), sep="\t", index=False)
    rows_bc.append(bcv.assign(assay="visium"))
    info = dict(cells_scrna=int(len(bc)), malignant_scrna=int(bc["malignant"].sum()),
                spots_visium=int(len(bcv)), malignant_visium=int(bcv["malignant"].sum()),
                section_cells=int(len(pcells)), mean_cells_per_spot=float(np.mean([members[s].size for s in keep])))
    return {"scrna": ds_rna, "visium": ds_vis}, truth_dir, pd.concat(rows_bc, ignore_index=True), info


def run_tools(datasets, work, total_cores, only=None):
    """Every run of ``RUNS`` side by side within ``total_cores``; returns {run: status}."""
    env = dict(os.environ)
    pending = [r for r in RUNS if only is None or r[0] in only]
    running, status = {}, {}
    free = total_cores
    while pending or running:
        for r in list(pending):
            name, tool, assay, cores, extra = r
            cores = min(cores, total_cores)
            if cores <= free:
                out = os.path.join(work, "runs", name)
                os.makedirs(out, exist_ok=True)
                log = open(os.path.join(work, "runs", f"{name}.log"), "w")
                e = dict(env, SLURM_CPUS_PER_TASK=str(cores))
                p = subprocess.Popen(["/usr/bin/time", "-v", os.path.join(TOOLS, "run_tool.sh"), tool, datasets[assay],
                                      out] + extra, stdout=log, stderr=subprocess.STDOUT, env=e)
                running[name] = (p, log, cores, time.time())
                free -= cores
                pending.remove(r)
                print(f"  start {name} ({cores} cores)", flush=True)
        for name, (p, log, cores, t0) in list(running.items()):
            rc = p.poll()
            if rc is None:
                continue
            log.close()
            wall = round(time.time() - t0, 1)
            text = open(log.name).read()
            rss = [ln for ln in text.splitlines() if "Maximum resident" in ln]
            status[name] = dict(exit=rc, wall_s=wall, max_rss_kb=int(rss[-1].split()[-1]) if rss else None)
            if rc != 0:
                status[name]["log_tail"] = text[-3000:]
            free += cores
            del running[name]
            print(f"  done {name}: exit {rc}, {wall} s", flush=True)
        time.sleep(10)
    return status


def score_dir(out):
    """Score every persisted run of one tumour's directory per niche (``scores.csv``)."""
    import score_tools as ST
    bc_all = pd.read_csv(os.path.join(out, "barcodes.tsv.gz"), sep="\t")
    tcn = pd.read_csv(os.path.join(out, "truth_arm_cn.tsv.gz"), sep="\t")
    calls = pd.read_csv(os.path.join(out, "calls.tsv.gz"), sep="\t", dtype={"clone": str})
    cna = pd.read_csv(os.path.join(out, "cn_arm.tsv.gz"), sep="\t") if os.path.exists(
        os.path.join(out, "cn_arm.tsv.gz")) else pd.DataFrame(columns=["run"])
    with open(os.path.join(out, "runs.json")) as f:
        runs = json.load(f)["runs"]
    rows = []
    for name, tool, assay, _, _ in RUNS:
        c = calls[calls["run"] == name]
        if not len(c):
            continue
        bc = bc_all[bc_all["assay"] == assay].set_index("barcode")
        true_cn = tcn[tcn["assay"] == assay].drop(columns="assay")
        cn = cna[cna["run"] == name].drop(columns=["run", "assay"], errors="ignore")
        rr = ST.score_niches_frames(bc, true_cn, c.drop(columns=["run", "assay"]).set_index("barcode"),
                                    cn if len(cn) else None)
        rj = runs.get(name, {}).get("run_json") or {}
        wall = rj.get("wall_s", runs.get(name, {}).get("wall_s"))
        # a runner that catches the tool's own error still writes calls (Numbat: all zero); such a run is
        # reported as failed, not scored
        status = str(rj.get("status", "ok"))
        failed = status.startswith("error")
        for r in rr:
            if failed:
                r = dict(r, auc=np.nan, cn_r=np.nan)
            rows.append(dict(tool=tool + ("-ref" if name.endswith("-ref") else ""), run=name, assay=assay,
                             wall_s=wall, status="failed: " + status[7:].strip()[:200] if failed else "ok", **r))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "scores.csv"), index=False)
    return df


def main():
    import score_tools as ST
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="R11")
    ap.add_argument("--task", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--section-max-cells", type=int, default=F.SECTION_MAX_CELLS)
    ap.add_argument("--cores", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    ap.add_argument("--only", default=None, help="comma-separated run names (default all)")
    ap.add_argument("--rescore", action="store_true", help="score the persisted runs again, nothing else")
    a = ap.parse_args()
    tag = f"{a.run}_{a.task}"
    out = os.path.join(a.out, "b", tag)
    if a.rescore:
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(score_dir(out).round(3).to_string(index=False))
        return
    os.makedirs(out, exist_ok=True)
    work = os.path.join(a.work, f"b_{tag}")
    os.makedirs(work, exist_ok=True)
    t0 = time.time()
    try:
        datasets, truth_dir, bc_all, info = build(a, work)
        info["build_s"] = round(time.time() - t0, 1)
        print("built:", json.dumps(info), flush=True)
        import gc
        gc.collect()
        cols = ["assay", "barcode", "malignant", "tumour_fraction", "compartment", "hypoxia", "epithelial",
                "reference", "scored"]
        only = None if a.only is None else a.only.split(",")
        merge = only is not None and os.path.exists(os.path.join(out, "calls.tsv.gz"))
        if merge:
            # the runs not redone keep their stored calls: the rebuilt truth must be the one they were made on
            old = pd.read_csv(os.path.join(out, "barcodes.tsv.gz"), sep="\t")
            key = ["assay", "barcode", "malignant", "compartment"] + [
                c for c in ("epithelial", "reference", "scored") if c in old]
            same = (len(old) == len(bc_all)
                    and old[key].reset_index(drop=True).equals(bc_all[key].reset_index(drop=True)))
            if not same:
                raise RuntimeError("the rebuilt truth differs from the stored one: rerun every tool, not --only")
        bc_all[cols].to_csv(os.path.join(out, "barcodes.tsv.gz"), sep="\t", index=False)
        pd.concat([pd.read_csv(os.path.join(truth_dir, f"{x}_arm_cn.tsv"), sep="\t").assign(assay=x)
                   for x in ("scrna", "visium")])[["assay", "barcode", "arm", "cn"]].round(4).to_csv(
            os.path.join(out, "truth_arm_cn.tsv.gz"), sep="\t", index=False)
        status = run_tools(datasets, work, a.cores, only)
        calls, cns = [], []
        for name, tool, assay, _, _ in RUNS:
            st = status.get(name)
            rd = os.path.join(work, "runs", name)
            if st is None or st["exit"] != 0 or not os.path.exists(os.path.join(rd, "run.json")):
                continue
            with open(os.path.join(rd, "run.json")) as f:
                st["run_json"] = json.load(f)
            calls.append(pd.read_csv(os.path.join(rd, "calls.tsv"), sep="\t", dtype={"clone": str}).assign(
                run=name, assay=assay))
            p = os.path.join(rd, "cn_arm.tsv")
            if os.path.exists(p) and os.path.getsize(p) > 0:
                cns.append(pd.read_csv(p, sep="\t").assign(run=name, assay=assay))
        runs_all = status
        if merge:
            redone = set(only)
            oc = pd.read_csv(os.path.join(out, "calls.tsv.gz"), sep="\t", dtype={"clone": str})
            calls.insert(0, oc[~oc["run"].isin(redone)])
            if os.path.exists(os.path.join(out, "cn_arm.tsv.gz")):
                on = pd.read_csv(os.path.join(out, "cn_arm.tsv.gz"), sep="\t")
                cns.insert(0, on[~on["run"].isin(redone)])
            with open(os.path.join(out, "runs.json")) as f:
                runs_all = {k: v for k, v in json.load(f)["runs"].items() if k not in redone}
            runs_all.update(status)
        if calls:
            pd.concat(calls, ignore_index=True)[["run", "assay", "barcode", "malignant", "p_malignant", "clone"]].to_csv(
                os.path.join(out, "calls.tsv.gz"), sep="\t", index=False)
        if cns:
            pd.concat(cns, ignore_index=True)[["run", "assay", "barcode", "arm", "value", "scale"]].round(5).to_csv(
                os.path.join(out, "cn_arm.tsv.gz"), sep="\t", index=False)
        n_ref = {x: dict(reference=int(bc_all[(bc_all.assay == x)]["reference"].sum()),
                         scored=int(bc_all[(bc_all.assay == x)]["scored"].sum()),
                         scored_normal=int(((bc_all.assay == x) & (bc_all.malignant == 0)
                                            & (bc_all.scored == 1)).sum())) for x in ("scrna", "visium")}
        with open(os.path.join(out, "runs.json"), "w") as f:
            json.dump(dict(run=a.run, task=a.task, commit=F._commit(), paper_out=os.environ.get("ISCC_PAPER_OUT"),
                           section_max_cells=a.section_max_cells, data=info,
                           reference_protocol=dict(kind="microenvironment normals (non-epithelial; Visium: "
                                                        "all-normal spots); scored: epithelial-lineage barcodes",
                                                   **n_ref, redone=only),
                           runs=runs_all), f, indent=1, default=str)
        failed = {k: v["exit"] for k, v in status.items() if v["exit"] != 0}
        print("failed runs:", failed or "none", flush=True)
        if calls:
            with pd.option_context("display.width", 200, "display.max_columns", 20):
                print(score_dir(out).round(3).to_string(index=False), flush=True)
    finally:
        if os.environ.get("FIG4_KEEP_WORK") != "1":
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
