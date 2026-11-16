"""Diagnose Fig 4c's truth arm: why clonealign, given the true clones, can put most cells on one clone.

Rebuilds exactly the inputs fig4_chain.py dna-rna hands clonealign in its truth arm (the same cells,
genes and true profiles) under ``--work`` and writes, to ``--out/<run>_<task>/``:

``checks.json``
    - ``inputs``: the rebuilt cells and truth equal the stored run's (``--stored``, a fig4_chain output
      directory), the copy-number matrix rebuilt gene by gene from each gene's arm equals the one handed
      over, and the stored true profiles equal the rebuilt ones;
    - ``signal``: whether the dosage signal is in the cells. Each cell's log expression (counts over the
      cell's total over these genes, as clonealign's multinomial sees them), centred per gene, is
      correlated with each clone's log copy number centred over clones; ``correlation`` assigns the clone
      with the highest correlation, ``multinomial`` fits clonealign's own likelihood without its latent
      factors by hard EM from that start, and ``centroid`` (an oracle) assigns by correlation with each
      true clone's mean profile. ``dosage`` is the per-gene slope of the clones' mean log expression on
      log copy number (1 is clonealign's assumption);
    - ``genes``: the genes preprocess_for_clonealign keeps, per arm, and on the arms where the clone that
      absorbs the most cells in the stored run differs from the largest clone.

``variant_<name>.json`` (with ``--variant``)
    clonealign rerun with ``--max-iter``, ``--n-repeats`` and ``--K``, scored as fig4_chain.py scores it.

    python validation/paper/fig4_diag.py --run R11 --task disp0.9_seed2 --stored .../fig4-v4/c \\
        --out .../fig4-v4-diag --work $SNIC_TMP [--variant long --max-iter 1000 --n-repeats 3]
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fig4_chain as F  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def _assign_scores(truth, pred):
    s = F.score_assignment(truth, pred)
    return {k: (round(v, 4) if isinstance(v, float) else {kk: round(vv, 3) for kk, vv in v.items()})
            for k, v in s.items()}


def _inputs(a):
    t, layer, dis, sec = F.sample(a.run, a.task, a.section_max_cells)
    clone_of, true_prof, members = F.true_clones(t, layer, [dis, sec])
    X, rna_ids, t_fine = F.rna_inputs(t, layer, dis, clone_of, a)
    gene_seg = layer.genes["segment"].to_numpy()
    g = F.informative_genes(X, gene_seg, true_prof)
    ids = layer.genes["gene_id"].astype(str).to_numpy()[g]
    Y = pd.DataFrame(X[:, g].toarray().astype(float), index=rna_ids, columns=[f"{x}.{i}" for i, x in zip(g, ids)])
    L = pd.DataFrame(true_prof.to_numpy(float)[:, gene_seg[g]].T, index=Y.columns, columns=list(true_prof.index))
    arms = layer.genes["arm"].to_numpy()[g]
    lib = np.asarray(X.sum(axis=1)).ravel()                  # each cell's library over all human genes
    return dict(Y=Y, L=L, truth=t_fine, true_prof=true_prof, arms=arms, layer=layer, g=g, lib=lib)


def _check_inputs(inp, stored_dir):
    Y, L, prof = inp["Y"], inp["L"], inp["true_prof"]
    rec = {"genes_in_same_order": bool(list(Y.columns) == list(L.index))}
    # L rebuilt from each gene's ARM NAME (not its segment index) against the matrix handed over
    L2 = np.stack([prof[arm].to_numpy(float) for arm in inp["arms"]])
    rec["copy_number_matrix_matches_gene_arms"] = bool(np.array_equal(L2, L.to_numpy()))
    rec["gene_ids_match_layer"] = bool(all(c.split(".")[0] == gid for c, gid in
                                           zip(Y.columns, inp["layer"].genes["gene_id"].astype(str).to_numpy()[inp["g"]])))
    if stored_dir:
        rc = pd.read_csv(os.path.join(stored_dir, "rna_cells.tsv.gz"), sep="\t", dtype={"cell": str})
        rec["same_cells_as_stored_run"] = bool(list(rc["cell"]) == list(Y.index))
        rec["same_truth_as_stored_run"] = bool((rc["true_clone"].to_numpy() == inp["truth"]).all())
        tp = pd.read_csv(os.path.join(stored_dir, "true_clones.tsv"), sep="\t", index_col=0)
        rec["same_true_profiles_as_stored_run"] = bool(tp[list(prof.columns)].reindex(prof.index).to_numpy().astype(int).tolist()
                                                       == prof.to_numpy().astype(int).tolist())
    return rec


def _signal(inp):
    Y, L, truth = inp["Y"].to_numpy(), inp["L"], inp["truth"]
    clones = np.asarray(L.columns, dtype=object)
    Lm = L.to_numpy().T                                       # clones x genes
    ok = (Lm > 0).all(axis=0)                                 # clonealign_runner drops copy number 0
    Y, Lm = Y[:, ok], Lm[:, ok]
    s = Y.sum(axis=1, keepdims=True)
    logp = np.log((Y + 0.5) / s)
    r = logp - logp.mean(axis=0, keepdims=True)
    lt = np.log(Lm) - np.log(Lm).mean(axis=0, keepdims=True)

    def corr_pred(prof_rows):
        rc = r - r.mean(axis=1, keepdims=True)
        pk = prof_rows - prof_rows.mean(axis=1, keepdims=True)
        c = (rc @ pk.T) / (np.linalg.norm(rc, axis=1, keepdims=True) * np.linalg.norm(pk, axis=1)[None, :] + 1e-12)
        return c.argmax(axis=1)

    out = {"n_genes": int(ok.sum())}
    pc = corr_pred(lt)
    out["correlation"] = _assign_scores(truth, clones[pc])
    # clonealign's likelihood without latent factors (multinomial, probs ~ mu_g * L_gk), hard EM
    z = pc.copy()
    for _ in range(30):
        mu = np.zeros(Y.shape[1])
        for k in range(len(clones)):
            m = z == k
            if m.any():
                mu += (Y[m] / s[m]).sum(axis=0) / Lm[k]
        mu = np.maximum(mu / max(len(z), 1), 1e-12)
        q = mu[None, :] * Lm
        q /= q.sum(axis=1, keepdims=True)
        ll = Y @ np.log(q).T
        z_new = ll.argmax(axis=1)
        if np.array_equal(z_new, z):
            break
        z = z_new
    out["multinomial"] = _assign_scores(truth, clones[z])
    # oracle: each true clone's mean profile
    cen = np.stack([r[truth == c].mean(axis=0) if (truth == c).any() else np.zeros(r.shape[1]) for c in clones])
    out["centroid_oracle"] = _assign_scores(truth, clones[corr_pred(cen)])
    # realised dosage: per gene, the slope of each clone's pooled expression (its counts over its cells'
    # whole libraries; a mean of logs of sparse counts would be squeezed by the pseudocount) on copy
    # number, both logged, over the clones with at least 30 cells and the genes with at least 50 counts
    lib = inp["lib"]
    big = [k for k, c in enumerate(clones) if (truth == c).sum() >= 30]
    means = np.log(np.stack([Y[truth == clones[k]].sum(axis=0) / lib[truth == clones[k]].sum() for k in big]) + 1e-12)
    lx = np.log(Lm[big])
    var = (lx.var(axis=0) > 0) & (Y.sum(axis=0) >= 50)
    xc = lx[:, var] - lx[:, var].mean(axis=0)
    yc = means[:, var] - means[:, var].mean(axis=0)
    slope = (xc * yc).sum(axis=0) / (xc ** 2).sum(axis=0)
    out["dosage"] = dict(n_genes=int(var.sum()), median_slope=float(np.median(slope)),
                         q25=float(np.quantile(slope, 0.25)), q75=float(np.quantile(slope, 0.75)),
                         share_below_0_3=float(np.mean(slope < 0.3)), share_negative=float(np.mean(slope < 0)))
    return out


def _preprocess_genes(inp, work, stored_dir):
    import integration_common as IC
    d = os.path.join(work, "pp")
    os.makedirs(d, exist_ok=True)
    Y, L = inp["Y"], inp["L"]
    keep = (L.to_numpy() > 0).all(axis=1)                    # the runner's own copy-number-0 filter first
    Y.loc[:, keep].to_csv(os.path.join(d, "Y.csv"))
    L.loc[keep].to_csv(os.path.join(d, "L.csv"))
    expr = ('suppressMessages(library(clonealign)); a <- commandArgs(TRUE); '
            'Y <- as.matrix(read.csv(file.path(a[1], "Y.csv"), row.names = 1, check.names = FALSE)); '
            'L <- as.matrix(read.csv(file.path(a[1], "L.csv"), row.names = 1, check.names = FALSE)); '
            'pp <- preprocess_for_clonealign(Y, L); writeLines(pp$retained_genes, file.path(a[1], "kept.txt"))')
    r = subprocess.run([IC.CLONEALIGN_RSCRIPT, "-e", expr, d], capture_output=True, text=True)
    if r.returncode != 0:
        return {"error": r.stderr[-1500:]}
    kept = set(open(os.path.join(d, "kept.txt")).read().split())
    arms = pd.Series(inp["arms"], index=Y.columns)
    per_arm = pd.DataFrame({"given": arms.value_counts(), "kept": arms[[c in kept for c in Y.columns]].value_counts()}
                           ).fillna(0).astype(int)
    out = {"given": int(len(Y.columns)), "after_cn0_filter": int(keep.sum()), "kept": int(len(kept)),
           "per_arm": per_arm.to_dict(orient="index")}
    prof = inp["true_prof"]
    if stored_dir:
        rc = pd.read_csv(os.path.join(stored_dir, "rna_cells.tsv.gz"), sep="\t")
        wrong = rc[rc["true_clone"] != rc["clonealign_true"]]
        if len(wrong):
            absorber = wrong["clonealign_true"].value_counts().index[0]
            largest = rc["true_clone"].value_counts().index[0]
            diff = [a for a in prof.columns if prof.at[absorber, a] != prof.at[largest, a]]
            out["absorber"], out["largest"] = absorber, largest
            out["arms_absorber_vs_largest"] = {a: dict(cn_absorber=int(prof.at[absorber, a]),
                                                       cn_largest=int(prof.at[largest, a]),
                                                       **per_arm.loc[a].to_dict()) if a in per_arm.index
                                               else dict(cn_absorber=int(prof.at[absorber, a]),
                                                         cn_largest=int(prof.at[largest, a]), given=0, kept=0)
                                               for a in diff}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="R11")
    ap.add_argument("--task", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--stored", default=None, help="the fig4_chain output's c/ directory to compare against")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rna-cells", type=int, default=F.RNA_MAX_CELLS)
    ap.add_argument("--section-max-cells", type=int, default=F.SECTION_MAX_CELLS)
    ap.add_argument("--variant", default=None)
    ap.add_argument("--max-iter", type=int, default=F.CLONEALIGN["max_iter"])
    ap.add_argument("--n-repeats", type=int, default=F.CLONEALIGN["n_repeats"])
    ap.add_argument("--K", default="NA")
    ap.add_argument("--skip-checks", action="store_true")
    a = ap.parse_args()
    tag = f"{a.run}_{a.task}"
    out = os.path.join(a.out, tag)
    os.makedirs(out, exist_ok=True)
    work = os.path.join(a.work, f"diag_{tag}")
    stored = os.path.join(a.stored, tag) if a.stored else None
    inp = _inputs(a)
    print(f"inputs: {inp['Y'].shape[0]} cells x {inp['Y'].shape[1]} genes, {inp['L'].shape[1]} clones", flush=True)
    if not a.skip_checks:
        rec = dict(run=a.run, task=a.task, commit=F._commit(), inputs=_check_inputs(inp, stored))
        print("inputs:", json.dumps(rec["inputs"]), flush=True)
        rec["signal"] = _signal(inp)
        print("signal:", json.dumps({k: v.get("balanced_accuracy", v) if isinstance(v, dict) else v
                                     for k, v in rec["signal"].items()}), flush=True)
        rec["genes"] = _preprocess_genes(inp, work, stored)
        print("genes:", json.dumps({k: v for k, v in rec["genes"].items() if k != "per_arm"}), flush=True)
        with open(os.path.join(out, "checks.json"), "w") as f:
            json.dump(rec, f, indent=1, default=str)
    if a.variant:
        import integration_common as IC
        d = os.path.join(work, f"ca_{a.variant}")
        os.makedirs(os.path.join(d, "in"), exist_ok=True)
        inp["Y"].to_csv(os.path.join(d, "in", "Y.csv"))
        inp["L"].to_csv(os.path.join(d, "in", "L.csv"))
        r = subprocess.run([IC.CLONEALIGN_RSCRIPT, os.path.join(F.VAL, "clonealign_runner.R"), os.path.join(d, "in"),
                            os.path.join(d, "out"), str(a.max_iter), str(a.n_repeats), "1", "1", str(a.K)],
                           capture_output=True, text=True)
        notes = [x for x in r.stdout.splitlines() if x.startswith("clonealign:")]
        print("\n".join(notes), flush=True)
        if r.returncode != 0:
            sys.exit(f"clonealign failed: {r.stderr[-1500:]}")
        P = pd.read_csv(os.path.join(d, "out", "clone_probs.csv"), index_col=0).reindex(inp["Y"].index)
        header_ok = list(P.columns) == list(inp["L"].columns)
        pred = np.asarray(inp["L"].columns, dtype=object)[np.nanargmax(P.to_numpy(float), axis=1)]
        ct = pd.crosstab(pd.Series(inp["truth"], name="true"), pd.Series(pred, name="clonealign"))
        rec = dict(variant=a.variant, max_iter=a.max_iter, n_repeats=a.n_repeats, K=a.K, notes=notes,
                   probs_header_matches_L=bool(header_ok), scores=_assign_scores(inp["truth"], pred),
                   assigned=pd.Series(pred).value_counts().to_dict(), crosstab=ct.to_dict(orient="index"))
        with open(os.path.join(out, f"variant_{a.variant}.json"), "w") as f:
            json.dump(rec, f, indent=1, default=str)
        print("variant:", a.variant, json.dumps({k: rec["scores"][k] for k in ("accuracy", "balanced_accuracy")}),
              json.dumps(rec["assigned"]), flush=True)


if __name__ == "__main__":
    main()
