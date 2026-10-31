"""Figure 5: copy number seen through expression.

The same copy-number coupling is signal for one method and artefact for another. Every tool is the
published one, run in its own environment through the existing benchmark plumbing
(:mod:`integration_common`, :mod:`programs_common`), on registry tumours of the paper regime:

  (a) clone assignment from copy-number dosage with clonealign (R4, three copy-number rates);
  (b) clonealign accuracy against the fraction of genes carrying a clone-specific copy-number effect;
  (c) copy-number calling from expression: inferCNV (expression only) against Numbat (allele-aware),
      and the allelic-imbalance state after whole-genome doubling (R5);
  (d) program discovery with scDEF and cNMF against copy-number burden, with the SNV-burden control:
      recovery of the true programs, and spurious positionally clustered factors (R4).

clonealign is given the clones' true copy-number profiles (as in the published benchmark); the
notebooks run it on HMMcopy calls instead.

Run:  python validation/paper/fig5_cn_expression.py [--jobs 1] [--only clonealign,cna,programs]
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import tempfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import common as C  # noqa: E402
import registry as REG  # noqa: E402
import style as S  # noqa: E402

CACHE = os.path.join(C.DERIVED, "fig5")
MAX_CELLS = 1500


def _scalars(d):
    out = {}
    for k, v in (d or {}).items():
        if isinstance(v, (int, float, np.integer, np.floating, bool)) and np.isfinite(float(v)):
            out[k] = float(v)
        elif isinstance(v, dict):
            sub = _scalars(v)
            if sub:
                out[k] = sub
    return out


def _materialised(run, task, max_cells=MAX_CELLS):
    t = REG.load(run, task)
    t.make_cell_data(max_cells=max_cells)
    return t


def job_clonealign(task):
    import integration_common as IC
    import validate_clonealign as VC
    t = _materialised("R4", task)
    inp = IC.build_clonealign_inputs(t, n_clones=4, seed=0)
    rec = dict(task=task, cnv=float(task.split("_")[0][3:]), n_cells=int(len(inp["labels"])))
    with tempfile.TemporaryDirectory() as work:
        probs = IC.run_clonealign(inp["Y"], inp["L"], os.path.join(work, "main"), max_iter=200,
                                  n_repeats=2)
        rec["score"] = _scalars(IC.score_assignment(inp["labels"], probs))
        if task.endswith("_seed1"):
            rng = np.random.default_rng(0)
            dial = []
            for f in (0.1, 0.25, 0.5, 1.0):
                Ys, Ls, frac = VC.subset_by_informative_fraction(inp, f, 250, rng)
                if Ls.nunique(axis=1).max() <= 1:
                    continue
                p = IC.run_clonealign(Ys, Ls, os.path.join(work, f"f{f}"), max_iter=200, n_repeats=2)
                s = IC.score_assignment(inp["labels"], p)
                dial.append(dict(frac=float(frac), acc=float(s["accuracy"]), auc=float(s["mean_auc"])))
            rec["dial"] = dial
    return rec


def job_cna(task):
    import integration_common as IC
    t = _materialised("R5", task)
    rec = dict(task=task, wgd=float(task.split("_")[0][3:]))
    shared = IC.build_cna_inputs(t, n_normal=150, n_clones=4, seed=0)
    with tempfile.TemporaryDirectory() as wd:
        numbat_in = IC.build_numbat_inputs(t, shared, wd, seed=0)
        inf = IC.run_infercnv(shared["adata"], os.path.join(wd, "infercnv")) if IC.infercnv_available() else None
        nb = IC.run_numbat(numbat_in, wd, min_cells=20, ncores=1, max_iter=1,
                           min_llr=2.0 if rec["wgd"] > 0 else 5.0) if IC.numbat_available() else None
        if nb is not None:
            rec["numbat"] = _scalars(IC.score_numbat(nb, numbat_in))
        if inf is not None:
            lab = shared.get("clone_labels")
            rec["infercnv"] = _scalars(IC.score_infercnv(inf, shared["adata"].obsm["true_seg_cn"],
                                                         clone_labels=lab))
        if nb is not None and rec["wgd"] > 0:
            rec["imbalance"] = _scalars(IC.score_numbat_imbalance(nb, numbat_in, IC.segment_allele_cn(t),
                                                                  infercnv=inf))
    return rec


def job_programs(task):
    import programs_common as PCm
    t = REG.load("R4", task)
    rec = dict(task=task)
    adata, Z = PCm.counts_anndata(t, seed=0, max_cells=800)
    for tool in ("scdef", "cnmf"):
        inferred = PCm.run_tool(tool, adata, k=2 * PCm.N_PROGRAMS, seed=0, n_epoch=300, n_iter=20)
        if inferred is not None:
            rec[tool] = _scalars(PCm.score_recovery(t.program_truth["loading"], Z, inferred,
                                                    t.selection.segment_sizes))
    rec["fga"] = float(PCm.fga(t))
    rec["snv"] = float(PCm.snv_burden(t))
    return rec


JOBS = {
    "clonealign": (job_clonealign, [f"cnv{c}_seed{s}" for c in (0.05, 0.15, 0.3) for s in (1, 2)]),
    "cna": (job_cna, ["wgd0.0_seed1", "wgd0.05_seed1", "wgd0.1_seed1"]),
    "programs": (job_programs, ["cnv0.0_seed1", "cnv0.15_seed1", "cnv0.3_seed1",
                                "snv0.05_seed1", "snv0.5_seed1"]),
}


def _run(job):
    kind, task = job
    sys.path.insert(0, os.path.dirname(HERE))
    fn = JOBS[kind][0]
    try:
        rec = fn(task)
        os.makedirs(os.path.join(CACHE, kind), exist_ok=True)
        with open(os.path.join(CACHE, kind, f"{task}.json"), "w") as f:
            json.dump(rec, f)
        return kind, task, None
    except Exception as e:
        return kind, task, f"{type(e).__name__}: {e}"


def compute(kinds, jobs):
    run_of = {"clonealign": "R4", "cna": "R5", "programs": "R4"}
    todo = [(k, task) for k in kinds for task in JOBS[k][1]
            if os.path.exists(REG.task_path(run_of[k], task))
            and not os.path.exists(os.path.join(CACHE, k, f"{task}.json"))]
    print(f"{len(todo)} tool runs to compute", flush=True)
    with mp.get_context("spawn").Pool(processes=jobs, maxtasksperchild=1) as pool:
        for kind, task, err in pool.imap_unordered(_run, todo):
            print(f"  {kind}/{task}: {err or 'ok'}", flush=True)


def cached(kind):
    d = os.path.join(CACHE, kind)
    if not os.path.isdir(d):
        return []
    return [json.load(open(os.path.join(d, f))) for f in sorted(os.listdir(d))]


def pending(ax, msg="pending: tool run not computed yet"):
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=5, color=S.GREY, transform=ax.transAxes)
    S.despine(ax, False, False)


def figure():
    S.setup()
    fig = S.figure("double", 74)
    out = {}
    # (a) clonealign accuracy by copy-number rate
    S.panel_letter(fig, "a", 2, 3)
    ax = S.ax_mm(fig, 10, 8, 30, 24)
    ca = cached("clonealign")
    if ca:
        df = pd.DataFrame([dict(cnv=r["cnv"], acc=r["score"].get("accuracy"), chance=r["score"].get("chance"),
                                auc=r["score"].get("mean_auc")) for r in ca])
        g = df.groupby("cnv")
        ax.errorbar(g.acc.mean().index, g.acc.mean().values, yerr=g.acc.std().fillna(0).values, marker="o",
                    ms=2.5, lw=0.8, capsize=1.3, color=S.TOOL["clonealign"], label="clonealign")
        ax.plot(g.chance.mean().index, g.chance.mean().values, color=S.INK, lw=0.5, ls=":", label="chance")
        ax.set_xlabel("Copy-number rate")
        ax.set_ylabel("Clone assignment accuracy")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=4.8)
        out["clonealign"] = df.round(3).to_dict("records")
    else:
        pending(ax)
    # (b) dosage dial
    S.panel_letter(fig, "b", 44, 3)
    ax = S.ax_mm(fig, 52, 8, 30, 24)
    dial = next((r["dial"] for r in ca if r.get("dial")), None)
    if dial:
        d = pd.DataFrame(dial).sort_values("frac")
        ax.plot(d.frac, d.acc, marker="o", ms=2.5, lw=0.8, color=S.TOOL["clonealign"])
        ax.set_xlabel("Genes with a clone-specific\ncopy-number effect")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1)
        out["dial"] = dial
    else:
        pending(ax)
    # (c) inferCNV against Numbat, and the allele layer after WGD
    S.panel_letter(fig, "c", 86, 3)
    ax = S.ax_mm(fig, 94, 8, 24, 24)
    ax2 = S.ax_mm(fig, 124, 8, 16, 24)
    cna = cached("cna")
    if cna:
        base = next((r for r in cna if r["wgd"] == 0.0), None)
        if base:
            vals = []
            for tool, key in (("inferCNV", "infercnv"), ("Numbat", "numbat")):
                sc = base.get(key, {})
                auc = sc.get("auc", sc.get("malignant_auc", np.nan))
                vals.append((tool, auc))
            ax.bar(range(len(vals)), [v for _, v in vals], color=[S.TOOL[t] for t, _ in vals], width=0.6)
            ax.set_xticks(range(len(vals)))
            ax.set_xticklabels([t for t, _ in vals])
            ax.set_ylim(0.4, 1)
            ax.set_ylabel("Malignant vs normal AUC")
        wgd = next((r for r in cna if r["wgd"] > 0 and "imbalance" in r), None)
        if wgd:
            im = wgd["imbalance"]
            nb_auc = im.get("numbat_auc", im.get("auc_numbat", np.nan))
            inf_auc = im.get("infercnv_auc", im.get("auc_infercnv", np.nan))
            ax2.bar([0, 1], [inf_auc, nb_auc], color=[S.TOOL["inferCNV"], S.TOOL["Numbat"]], width=0.6)
            ax2.axhline(0.5, color=S.INK, lw=0.5, ls=":")
            ax2.set_xticks([0, 1])
            ax2.set_xticklabels(["inferCNV", "Numbat"], rotation=90)
            ax2.set_ylim(0.4, 1)
            ax2.set_ylabel("Imbalance AUC, after WGD")
        else:
            pending(ax2, "")
        out["cna"] = cna
    else:
        pending(ax)
        pending(ax2, "")
    # (d) program discovery against burden
    S.panel_letter(fig, "d", 146, 3)
    ax = S.ax_mm(fig, 152, 8, 28, 24)
    pr = cached("programs")
    if pr:
        rows = []
        for r in pr:
            for tool, name in (("scdef", "scDEF"), ("cnmf", "cNMF")):
                if tool in r:
                    sc = r[tool]
                    rows.append(dict(task=r["task"], tool=name, fga=r["fga"],
                                     recovery=sc.get("mean_matched_cosine", sc.get("recovery", np.nan)),
                                     spurious=sc.get("spurious_clustering", sc.get("max_spurious_clustering", np.nan))))
        df = pd.DataFrame(rows)
        for name in ("scDEF", "cNMF"):
            sub = df[(df.tool == name) & df.task.str.startswith("cnv")].sort_values("fga")
            ax.plot(sub.fga, sub.spurious, marker="o", ms=2.5, lw=0.8, color=S.TOOL[name], label=name)
        ax.set_xlabel("Fraction of genome altered")
        ax.set_ylabel("Spurious positional factors")
        ax.legend(fontsize=4.8)
        out["programs"] = df.round(3).to_dict("records")
    else:
        pending(ax)
    path = S.save(fig, "fig5_cn_expression")
    print("wrote", path)
    print(json.dumps(out, default=str)[:3000])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--only", default="clonealign,cna,programs")
    ap.add_argument("--figure-only", action="store_true")
    a = ap.parse_args()
    if not a.figure_only:
        compute(a.only.split(","), a.jobs)
    figure()


if __name__ == "__main__":
    main()
