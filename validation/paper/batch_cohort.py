"""Fig 5's cohort scRNA at the registered batch levels, on top of ``fig5_atlas.py``'s cells.

The batch model and the levels of panels b-c were registered before any run. ``fig5_atlas.py`` is not
changed here; this module supplies the two pieces its cells and integrate steps need:

    dissociation-file --out <json>
        every patient's registered dissociation for ``fig5_atlas.py cells --dissociation-file``: its
        recovery draw (``batch_fits.recovery_draws``, the draws fits.json records) and no stress, so
        the cells are stored unstressed with the patient's recovery applied.
    integrate --out <fig5 out> --fits <fits.json> --level <level> [--methods none,harmony,scvi]
        the pooled cohort's counts in both designs at one level of ``batch_levels.LEVELS`` (fitted,
        shared, the six attribution arms, reference): per-batch overrides in both designs
        (``batch_kwargs`` in the balanced design, one scRNA call per patient in the confounded one),
        each patient's stress applied to the shared stress-gene set as the level says, the reference
        (one batch, no batch effect, no stress) for retention; integrated and scored as Part I.

The stress genes are ONE set shared by every patient, fixed by the cohort's layout seed
(:data:`STRESS_GENE_SEED`, the package's ``DEFAULT_LAYOUT_SEED``, which ``regime.grow`` leaves
GenotypeTumor at) at the package's default size (``choose_stress_genes``: 5% of the genes).

Writes ``<out>/integrate/level_<level>.json`` (the first design's level names, 10x / intermediate /
strong, do not collide with these).
"""
import argparse
import json
import math
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for p in (os.path.join(REPO, "src"), os.path.dirname(HERE), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import batch_fits as BF  # noqa: E402
import batch_levels as BL  # noqa: E402

from iscc.constants import DEFAULT_LAYOUT_SEED as STRESS_GENE_SEED  # noqa: E402

N_BALANCED_BATCHES = 50
EMBED_LEVELS = ("shared", "fitted")
EMBED_CELLS = 10_000
CONFOUNDED_SEED, BALANCED_SEED, REFERENCE_SEED = 8000, 9000, 7000      # as the first design's integrate


def dissociation_file(path, n_patients=BF.N_PATIENTS):
    """``{"<i>": {"recovery_probs": {...}, "stress_strength": 0.0}}`` for ``fig5_atlas.py cells``."""
    draws = BF.recovery_draws(n_patients)
    out = {i: dict(recovery_probs=v, stress_strength=0.0) for i, v in draws["per_patient"].items()}
    out["_rule"] = dict(rule=draws["rule"], sd=draws["sd"], seed=draws["seed"],
                        stress="none at the cells step: applied per level by batch_cohort.py integrate")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    return path


def shared_stress_genes(genes):
    """The cohort's one stress-gene set (see the module notes)."""
    from iscc.sample.dissociation import choose_stress_genes
    return choose_stress_genes(list(genes), None, np.random.default_rng(STRESS_GENE_SEED))


def stressed(exp, patient_of_row, strength, genes):
    """``apply_stress`` per patient, in one pass: the stress-gene columns of each patient's rows times
    ``1 + strength[patient]``; a copy, the input untouched."""
    mult = np.array([1.0 + float(strength[p]) for p in patient_of_row], dtype=np.float64)
    out = exp.copy()
    if np.all(mult == 1.0) or not len(genes):
        return out
    cols = [exp.columns.get_loc(g) for g in genes]
    vals = out.values
    vals[:, cols] = (vals[:, cols] * mult[:, None]).astype(vals.dtype)
    return pd.DataFrame(vals, index=exp.index, columns=exp.columns)


def _patient_meta(d):
    try:
        return json.loads(str(d["meta"]))
    except Exception:                                   # noqa: BLE001
        return {}


def check_cells(pats, fits):
    """Every patient's cells were dissociated with its registered recovery and without stress."""
    rec = fits["recovery"]["per_patient"]
    bad = []
    for d in pats:
        m = _patient_meta(d)
        st = m.get("settings") or {}
        p = str(d["patient"])
        given = st.get("recovery_probs") or m.get("recovery_probs")
        if given != rec.get(p) or float(st.get("stress_strength", 0.0)) != 0.0:
            bad.append(p)
    if bad:
        sys.exit(f"patients {bad[:10]}: the cells were not dissociated with the registered recovery and no "
                 "stress (run fig5_atlas.py cells with batch_cohort.py's dissociation file)")


def level_counts(pooled, meta, rows, fits, lname, nb, scrna_counts):
    """The counts of both designs at one level and the reference's, with the settings used."""
    from iscc.data.rna import run_scrna_batches
    patients = sorted(meta["patient"].unique(), key=int)
    S = {d: BL.level_settings(fits, patients, d, lname, n_balanced_batches=nb) for d in BL.DESIGNS}
    ref = BL.level_settings(fits, patients, "confounded", "reference")
    strength = {p: S["confounded"]["patients"][p]["stress_strength"] for p in patients}
    if strength != {p: S["balanced"]["patients"][p]["stress_strength"] for p in patients}:
        raise RuntimeError("the designs disagree on a patient's stress")
    genes = shared_stress_genes(pooled["cell_exp"].columns)
    out = dict(settings={d: S[d] for d in BL.DESIGNS}, reference_settings=ref, stress_genes=genes,
               stress_gene_seed=STRESS_GENE_SEED, counts={}, labels={}, doublets={})
    if lname != "reference":
        exp = stressed(pooled["cell_exp"], meta.loc[pooled["cell_exp"].index, "patient"].values, strength, genes)
        cd = dict(pooled, cell_exp=exp)
        parts, dbl = [], 0
        for j, p in enumerate(patients):
            cells = list(meta.index[meta["patient"] == p])
            cnt, dd = scrna_counts(cd, cells, S["confounded"]["batch_labels"][j], CONFOUNDED_SEED + int(p),
                                   S["confounded"]["batches"][j])
            parts.append(cnt)
            dbl += int(dd.sum())
        counts = pd.concat(parts)
        out["counts"]["confounded"] = counts
        out["labels"]["confounded"] = pd.Series([f"batch{q}" for q in meta.loc[counts.index, "patient"]],
                                                index=counts.index)
        out["doublets"]["confounded"] = dbl
        assays = run_scrna_batches(cd, n_batches=nb, design="split", n_cells=int(math.ceil(len(rows) / nb)),
                                   base_seed=BALANCED_SEED, protocol="10x", batch_kwargs=S["balanced"]["batches"])
        counts = pd.concat([a.observed_counts for a in assays])
        out["counts"]["balanced"] = counts
        out["labels"]["balanced"] = pd.Series(np.concatenate([[a.batch_label] * a.n_cells for a in assays]),
                                              index=counts.index)
        out["doublets"]["balanced"] = int(sum(a.obs["is_doublet"].sum() for a in assays))
        out["balanced_batch_sizes"] = [int(a.n_cells) for a in assays]
        del assays, exp, cd
    else:   # the balanced split's labels, to score the reference with (the split depends on the seed only)
        sizes = int(math.ceil(len(rows) / nb))
        sel = np.random.default_rng(BALANCED_SEED).choice(np.asarray(pooled["cell_exp"].index),
                                                          size=min(sizes * nb, len(rows)), replace=False)
        lab = np.concatenate([[f"batch{i}"] * len(s) for i, s in enumerate(np.array_split(sel, nb))])
        out["labels"]["balanced"] = pd.Series(lab, index=sel)
        out["labels"]["confounded"] = pd.Series([f"batch{q}" for q in meta.loc[rows, "patient"]], index=rows)
    tc, tdbl = scrna_counts(pooled, rows, "reference", REFERENCE_SEED, ref["batches"][0])
    out["counts"]["reference"] = tc
    out["doublets"]["reference"] = int(tdbl.sum())
    return out


def integrate(args):
    import fig5_atlas as FA
    import fig5_cohort as F
    lname = args.level
    with open(args.fits) as f:
        fits = json.load(f)
    BF.require_ambient_range(fits)
    out = FA.out_dir(args)
    t0 = time.time()
    pats = FA.load_cells(out, args.n_patients)
    if not pats:
        sys.exit(f"no cells under {out}/cells")
    check_cells(pats, fits)
    pooled, meta, rows = FA.pool(pats)
    del pats
    n_pat = meta["patient"].nunique()
    nb = min(args.n_balanced_batches, len(rows))
    work = F.scratch(f"batchcohort-{lname}-")
    log, order, h5 = [], {}, {}
    try:
        made = level_counts(pooled, meta, rows, fits, lname, nb, F._scrna_counts)
        labels = made["labels"]
        for d in BL.DESIGNS:
            if d in made["counts"]:
                h5[d] = os.path.join(work, f"{d}.h5ad")
                order[h5[d]] = F._write_h5ad(h5[d], made["counts"][d], labels[d], meta)
            h5[f"truth_{d}"] = os.path.join(work, f"truth_{d}.h5ad")
            order[h5[f"truth_{d}"]] = F._write_h5ad(h5[f"truth_{d}"], made["counts"]["reference"], labels[d], meta)
        del pooled
        made["counts"] = None
        prep_s = time.time() - t0

        methods = args.methods or F.METHODS
        runs = [(d, m) for d in BL.DESIGNS for m in methods if d in h5]
        runs += [(f"truth_{d}", "none") for d in BL.DESIGNS]
        runs.sort(key=lambda r: r[1] != "scvi")
        cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
        n_scvi = sum(m == "scvi" for _, m in runs)
        cheap_workers = max(1, min(4, cpus // 4)) if n_scvi else max(1, min(len(runs), cpus))
        scvi_threads = max(1, (cpus - cheap_workers) // n_scvi) if n_scvi else 1
        rargs = argparse.Namespace(integration_python=args.integration_python, no_metrics=args.no_metrics,
                                   max_epochs=args.max_epochs)
        from concurrent.futures import ThreadPoolExecutor

        def run(r):
            d, m = r
            return F._integrate(h5[d], h5[d][:-5] + f"_{m}", m, rargs, log, scvi_threads if m == "scvi" else 1)
        t1 = time.time()
        with ThreadPoolExecutor(max_workers=n_scvi + cheap_workers) as ex:
            outs = list(ex.map(run, runs))
        integrate_s = time.time() - t1
        res = {f"{d}/{m}": dict(prefix=h5[d][:-5] + f"_{m}", h5ad=h5[d], out=o) for (d, m), o in zip(runs, outs)}
        truth = {d: (res.get(f"truth_{d}/none") or {}).get("out") for d in BL.DESIGNS}
        summary = dict(
            step="integrate", level=lname, level_status="registered (exp-paper1-fig5-cohort, amendment of "
            "2026-11-16 night and its clarifications)", fits=dict(path=os.path.abspath(args.fits),
                                                                  sha256=BF.file_sha256(args.fits)),
            level_settings=made["settings"], reference_settings=made["reference_settings"],
            stress_genes=made["stress_genes"], stress_gene_seed=made["stress_gene_seed"],
            commit=F._commit(), k=F.K, n_patients=int(n_pat), n_cells=int(len(rows)),
            n_cancer=int(meta["is_cancer"].sum()), n_clones=int(meta.loc[meta["is_cancer"], "clone"].nunique()),
            n_balanced_batches=nb, balanced_batch_sizes=made.get("balanced_batch_sizes"),
            doublets=made["doublets"], cpus=cpus, scvi_threads=scvi_threads, cheap_workers=cheap_workers,
            prep_s=round(prep_s, 1), integrate_s=round(integrate_s, 1),
            truth={d: (o or {}).get("metrics") for d, o in truth.items()}, runs={}, log=log)
        for key, r in res.items():
            d, m = key.split("/")
            if d.startswith("truth_"):
                continue
            o = r["out"]
            if o is None:
                summary["runs"][key] = dict(error="integration failed; see log")
                continue
            mt = o.get("metrics") or {}
            tm = summary["truth"].get(d) or {}
            ratio = {f"{k}_retention": (mt[k] / tm[k] if mt.get(k) is not None and tm.get(k) else None)
                     for k in F.RATIO_KEYS}
            summary["runs"][key] = dict(metrics=mt, retention=ratio, settings=o.get("settings"),
                                        timings=o.get("timings"), versions=o.get("versions"))
        if lname in EMBED_LEVELS and not args.no_embeddings:
            te = time.time()
            rng = np.random.default_rng(11)
            keep = np.sort(rng.choice(len(rows), size=min(EMBED_CELLS, len(rows)), replace=False))
            shown = [k for k in res if not k.startswith("truth_balanced")]
            emb = dict(n_cells_total=int(len(rows)), n_shown=int(len(keep)),
                       labels=dict(batch={d: labels[d].reindex([rows[j] for j in keep]).tolist() for d in labels},
                                   patient=meta["patient"].values[keep].tolist(),
                                   clone=meta["clone"].values[keep].tolist(),
                                   cell_type=meta["cell_type"].values[keep].tolist()),
                       xy={}, note="2-D projection fitted on every cell, coordinates kept for a random "
                                   "subsample; the reference's counts are the same in both designs, so its "
                                   "embedding is shown once (truth/none)")
            for key in shown:
                r = res[key]
                if r["out"] is None:
                    continue
                xy, how = F._project(r["prefix"], order[r["h5ad"]], [rows[j] for j in keep])
                emb["xy"]["truth/none" if key.startswith("truth_") else key] = xy
                emb["projection"] = how
            emb["wall_s"] = round(time.time() - te, 1)
            summary["embeddings"] = emb
        summary["wall_s"] = round(time.time() - t0, 1)
        summary["peak_mb"] = round(F._peak_mb())
        path = F.write_json(os.path.join(out, "integrate", f"level_{lname}.json"), summary)
        print(f"[batch integrate] {lname}: {len(rows)} cells of {n_pat} patients, {len(runs)} runs in "
              f"{summary['wall_s']} s -> {path}", flush=True)
        return path
    finally:
        if not args.keep_work:
            shutil.rmtree(work, ignore_errors=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dissociation-file")
    d.add_argument("--out", required=True)
    d.add_argument("--n-patients", type=int, default=BF.N_PATIENTS)
    p = sub.add_parser("integrate")
    p.add_argument("--out", default=None, help="Fig 5's output directory (default $ISCC_FIG5_OUT)")
    p.add_argument("--n-patients", type=int, default=None)
    p.add_argument("--fits", required=True, help="batch_fits.py's fits.json (with its ambient range if needed)")
    p.add_argument("--level", required=True, choices=list(BL.LEVELS))
    p.add_argument("--n-balanced-batches", type=int, default=N_BALANCED_BATCHES)
    p.add_argument("--methods", type=lambda s: s.split(","), default=None)
    p.add_argument("--integration-python", default=None)
    p.add_argument("--no-metrics", action="store_true")
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--no-embeddings", action="store_true")
    p.add_argument("--keep-work", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "dissociation-file":
        print(dissociation_file(a.out, a.n_patients))
    else:
        return integrate(a)


if __name__ == "__main__":
    main()
