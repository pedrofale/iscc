"""Fig 5's batch panel: ``iscc``'s batch effects against a real technical-replicate contrast and
against Splatter's batch model.

Registered in exp-paper1-fig5-cohort, "Amendment, 2026-11-16 night", section "The new panel"; the
model is decision-scrna-batches-from-technical-sources. Every fit it uses was made beforehand by
``batch_fits.py`` and is read from fits.json (its SHA-256 is recorded); nothing here is fitted to a
statistic except Splatter's two tuned parameters, as registered.

    PYTHONPATH=src python validation/paper/batch_validation.py run --fits $D/batchval/fits.json \\
        --ding-dir $D/realdata/GSE132044 --pbmcsca-meta $D/realdata/pbmcsca/pbmcsca_meta.tsv.gz \\
        --out $D/batchval/validation.json --work $SNIC_TMP/bv --tools-dir $R/tools
    PYTHONPATH=src python validation/paper/batch_validation.py figure --validation $D/batchval/validation.json

**The contrast.** Ding et al. 2020's PBMC1 in two 10x Chromium v2 runs, A and B: the same blood
sample, so their differences are technical. Cell types are the authors' (``batch_fits.load_ding``);
"Unassigned" cells are left out. Three versions of the A-B contrast:

* **real**;
* **iscc**: the biology is the pooled real cells' mean profile per cell type (each type's mean of
  count / library over runs A and B); each simulated batch holds A's or B's cells, type by type, in
  their real numbers; each batch's technical parameters are the run's own fit in fits.json
  (``estimate_rna`` within types, the run's doublet rate and ambient fraction), ``sigma_batch`` 0.10
  and ``depth_batch_sigma`` 0 (``mu_lib`` is the batch's own, so depth is not counted twice); count
  model NB, protocol 10x. Not tuned to the real effect.
* **Splatter** (Zappia et al. 2017, Splatter 1.34.0 in the datasim-r image): ``splatEstimate`` on the
  pooled real counts; groups in the real types' proportions; ``de.facLoc`` tuned (de.prob,
  de.downProb, de.facScale at Splatter's defaults) so that the cell-type silhouette matches the real
  one, without a batch effect; then two batches of the real runs' sizes with ``batch.facLoc`` 0 and
  ``batch.facScale`` tuned so that S0 matches the real one. Tuning is a grid, a finer grid inside the
  bracketing interval, and linear interpolation, at a tuning seed (:data:`TUNE_SEED`) outside the
  five evaluation seeds.

iscc and Splatter are each run at five seeds (:data:`SEEDS`).

**The statistics** (registered), per cell type, on genes whose mean is at least 0.05 counts per cell
in the type in each of A and B, counts normalised to the dataset's median library size, the fold
change B over A:

* S0, size: the standard deviation over genes of log2 FC; the median over types.
* S1, dependence on expression level: Spearman correlation of the signed log2 FC with the log mean
  (of the two runs' means); the ratio of the log2 FC's standard deviation in the lowest to the highest
  tercile of mean expression; each the median over types.
* S2, detection against mean: the OLS slope of the change in detection rate (share of cells with a
  count, B minus A) on the change in log2 mean, within each tercile of mean expression; the median
  over types, per tercile.
* S3, differences between types: the median over pairs of types of the Pearson correlation of their
  per-gene log2 FCs (genes kept in both).
* S4, embedding: 50 principal components of the log-normalised pooled counts (integrate.py's
  ``lognorm_pca``, as Fig 5): the silhouette by batch and by cell type (sklearn, Euclidean), and
  iLISI over the batch and cLISI over the type (k = 30, scib's normalisation; integrate.py's
  ``lisi_scaled`` in the integration image, as Fig 5).

Fixed here before any statistic (operational details the registration leaves open): cell types
enter the statistics when they have at least :data:`MIN_TYPE_CELLS` cells in each real run;
the comparison uses the genes with at least one count in the pooled real cells.

**Scoring (registered).** Per statistic, iscc reproduces it when the median over its seeds is closer
to the real value than Splatter's median. Statistics with several summaries (S1: 2, S2: 3) are scored
per summary, and the statistic counts as reproduced when every summary is. S0 of iscc is reported
against the real one; if they differ by more than a factor of 1.5 a Splatter version matched to
iscc's S0 is added and the scoring is also reported against it. S4 has no expectation.
"""
import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for p in (os.path.join(REPO, "src"), os.path.dirname(HERE), HERE, os.path.join(HERE, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402

MIN_MEAN = 0.05
MIN_TYPE_CELLS = 20
K = 30
N_PCS = 50
SEEDS = (1, 2, 3, 4, 5)
TUNE_SEED = 101
SIGMA_BATCH = 0.10
SIZE_FACTOR = 1.5
DE_GRID = (0.02, 0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0, 1.3, 1.7, 2.2, 3.0)
BATCH_GRID = (0.0, 0.01, 0.02, 0.04, 0.07, 0.1, 0.15, 0.2, 0.3, 0.45, 0.7)
N_REFINE = 4
SUMMARIES = {"S1": ("S1_spearman", "S1_spread_ratio"),
             "S2": ("S2_slope_low", "S2_slope_mid", "S2_slope_high"),
             "S3": ("S3_median_corr",)}
SPLATTER_IMAGE = "datasim-r.sif"


# ------------------------------------------------------------------ statistics
def contrast_stats(X, batch, ctype, types, curves=True):
    """S0-S3 of one version of the A-B contrast (X cells x genes counts, CSR)."""
    from scipy.stats import spearmanr
    X = sp.csr_matrix(X)
    lib = np.asarray(X.sum(1)).ravel().astype(float)
    med = float(np.median(lib))
    norm = sp.diags(med / np.maximum(lib, 1.0)) @ X
    det = (X > 0).astype(np.float32)
    batch, ctype = np.asarray(batch), np.asarray(ctype)
    per, fcs = {}, {}
    for t in types:
        a = np.flatnonzero((ctype == t) & (batch == "A"))
        b = np.flatnonzero((ctype == t) & (batch == "B"))
        if len(a) < 2 or len(b) < 2:
            per[t] = dict(n_A=int(len(a)), n_B=int(len(b)), skipped=True)
            continue
        mA = np.asarray(norm[a].mean(0)).ravel()
        mB = np.asarray(norm[b].mean(0)).ravel()
        dA = np.asarray(det[a].mean(0)).ravel()
        dB = np.asarray(det[b].mean(0)).ravel()
        keep = (mA >= MIN_MEAN) & (mB >= MIN_MEAN)
        g = np.flatnonzero(keep)
        fc = np.log2(mB[g] / mA[g])
        lm = np.log(0.5 * (mA[g] + mB[g]))
        dd = (dB - dA)[g]
        q1, q2 = np.quantile(lm, [1 / 3, 2 / 3])
        terc = [lm <= q1, (lm > q1) & (lm <= q2), lm > q2]
        slopes = []
        for m in terc:
            x, y = fc[m], dd[m]
            slopes.append(float(np.polyfit(x, y, 1)[0]) if m.sum() >= 3 and np.ptp(x) > 0 else float("nan"))
        rec = dict(n_A=int(len(a)), n_B=int(len(b)), n_genes=int(len(g)), S0=float(np.std(fc, ddof=1)),
                   S1_spearman=float(spearmanr(fc, lm).correlation),
                   S1_spread_ratio=float(np.std(fc[terc[0]], ddof=1) / np.std(fc[terc[2]], ddof=1)),
                   S2_slope_low=slopes[0], S2_slope_mid=slopes[1], S2_slope_high=slopes[2],
                   mean_fc=float(fc.mean()))
        if curves:
            edges = np.quantile(lm, np.linspace(0, 1, 11))
            idx = np.clip(np.searchsorted(edges, lm, side="right") - 1, 0, 9)
            rec["curve"] = dict(log_mean=[float(lm[idx == i].mean()) for i in range(10)],
                                sd_log2fc=[float(np.std(fc[idx == i], ddof=1)) for i in range(10)],
                                mean_log2fc=[float(fc[idx == i].mean()) for i in range(10)])
        per[t] = rec
        fcs[t] = pd.Series(fc, index=g)
    done = [t for t in types if not per[t].get("skipped")]
    pairs = {}
    for i, s in enumerate(done):
        for t in done[i + 1:]:
            j = fcs[s].index.intersection(fcs[t].index)
            if len(j) >= 10:
                pairs[f"{s} | {t}"] = float(np.corrcoef(fcs[s][j], fcs[t][j])[0, 1])
    summ = dict(S0=float(np.median([per[t]["S0"] for t in done])))
    for k in ("S1_spearman", "S1_spread_ratio", "S2_slope_low", "S2_slope_mid", "S2_slope_high"):
        summ[k] = float(np.nanmedian([per[t][k] for t in done]))
    summ["S3_median_corr"] = float(np.median(list(pairs.values()))) if pairs else float("nan")
    return dict(summary=summ, per_type=per, S3_pairs=pairs, median_library=med)


def embedding(X, seed=0):
    """50 PCs of the log-normalised counts (integrate.py's ``lognorm_pca``, as Fig 5)."""
    import anndata as ad
    from integrate import lognorm_pca
    a = ad.AnnData(X=sp.csr_matrix(X, dtype=np.float32))
    pca, n = lognorm_pca(a, N_PCS, seed)
    return pca


def silhouettes(pca, batch, ctype):
    from sklearn.metrics import silhouette_score
    return dict(silhouette_batch=float(silhouette_score(pca, np.asarray(batch))),
                silhouette_celltype=float(silhouette_score(pca, np.asarray(ctype))))


# ------------------------------------------------------------------ the versions
def real_data(ding_dir, labels):
    import batch_fits as BF
    X, genes, obs = BF.load_ding(ding_dir, labels)
    keep = obs["cell_type"].notna().to_numpy() & (obs["cell_type"] != BF.UNASSIGNED).to_numpy()
    X, obs = X[np.flatnonzero(keep)], obs[keep]
    gk = np.flatnonzero(np.asarray(X.sum(0)).ravel() > 0)
    X = X[:, gk].tocsr()
    X.data = X.data.astype(np.int64)
    return X, genes[gk], obs


def synthetic_data(seed=0, n_genes=1500, n_a=900, n_b=600):
    """A small two-run dataset made by iscc itself, for a smoke run of the pipeline (``--synthetic``):
    no real statistic is computed. Four types with log-normal profiles; run B shallower than A."""
    from iscc.data import scRNA
    rng = np.random.default_rng(seed)
    types = ["T", "B", "Mono", "NK"]
    base = rng.lognormal(0.0, 1.5, n_genes)
    prof = {t: base * rng.lognormal(0.0, 0.8, n_genes) for t in types}
    parts, obs = [], []
    for r, n, mu in (("A", n_a, 2500.0), ("B", n_b, 1800.0)):
        ct = rng.choice(types, size=n, p=[0.45, 0.15, 0.3, 0.1])
        names = [f"{r}{i}" for i in range(n)]
        exp = pd.DataFrame(np.vstack([prof[t] for t in ct]), index=names, columns=[f"g{i}" for i in range(n_genes)])
        a = scRNA(protocol="10x", n_cells=n, seed=7 + len(parts), mu_lib=mu, dropout_mid=0.0)
        a.run({"cell_exp": exp}, cell_subset=names)
        parts.append(sp.csr_matrix(a.observed_counts.to_numpy()))
        obs.append(pd.DataFrame({"run": r, "cell_type": ct}, index=names))
    X = sp.vstack(parts).tocsr()
    X.data = X.data.astype(np.int64)
    obs = pd.concat(obs)
    gk = np.flatnonzero(np.asarray(X.sum(0)).ravel() > 0)
    return X[:, gk].tocsr(), np.array([f"g{i}" for i in gk]), obs


def synthetic_fits():
    run = lambda mu: dict(combined=dict(mu_lib=mu, sigma_lib=0.4, dispersion=0.2, dropout_mid=0.0, dropout_shape=1.0),
                          doublet_rate=0.03, ambient_frac=0.03, ambient_source="synthetic")
    return dict(created="synthetic", ding=dict(runs=dict(A=run(2500.0), B=run(1800.0))))


def type_set(obs):
    n = obs.groupby(["cell_type", "run"]).size().unstack(fill_value=0)
    ok = n[(n.get("A", 0) >= MIN_TYPE_CELLS) & (n.get("B", 0) >= MIN_TYPE_CELLS)]
    return sorted(ok.index.astype(str)), {t: {r: int(v) for r, v in row.items()} for t, row in n.iterrows()}


def iscc_settings(fits, run):
    """The run's technical parameters for the iscc version, from fits.json (no tuning)."""
    rr = fits["ding"]["runs"][run]
    c = rr["combined"]
    kw = dict(mu_lib=c["mu_lib"], sigma_lib=c["sigma_lib"], dispersion=c["dispersion"],
              dropout_mid=c["dropout_mid"], dropout_shape=c["dropout_shape"],
              doublet_rate=rr["doublet_rate"], sigma_batch=SIGMA_BATCH, depth_batch_sigma=0.0)
    src = dict(ambient_frac=rr.get("ambient_source"))
    if rr.get("ambient_frac") is not None:
        kw["ambient_frac"] = rr["ambient_frac"]
    else:   # estimate_rna's own rule for what counts cannot identify: carried from the 10x preset
        from iscc.data.rna import PROTOCOL_PRESETS
        kw["ambient_frac"] = float(PROTOCOL_PRESETS["10x"]["ambient_frac"])
        src["ambient_frac"] = "carried from the 10x preset (not identifiable)"
    return {k: float(v) for k, v in kw.items()}, src


def iscc_version(X, obs, fits, seed):
    """iscc's A-B contrast: per-type mean profiles, each run's cells and fitted parameters."""
    from iscc.data import scRNA
    lib = np.asarray(X.sum(1)).ravel().astype(float)
    comp = sp.diags(1.0 / np.maximum(lib, 1.0)) @ X
    ctype = obs["cell_type"].astype(str).to_numpy()
    profiles = {t: np.asarray(comp[np.flatnonzero(ctype == t)].mean(0)).ravel() for t in sorted(set(ctype))}
    cols = [f"g{i}" for i in range(X.shape[1])]
    parts, batch, types, settings = [], [], [], {}
    for r, off in (("A", 1), ("B", 2)):
        sel = np.flatnonzero(obs["run"].to_numpy() == r)
        names = [f"{r}{i}" for i in range(len(sel))]
        exp = pd.DataFrame(np.vstack([profiles[t] for t in ctype[sel]]), index=names, columns=cols)
        kw, src = iscc_settings(fits, r)
        a = scRNA(protocol="10x", n_cells=len(sel), batch_label=r, seed=1000 * seed + off, **kw)
        a.run({"cell_exp": exp}, cell_subset=names)
        parts.append(sp.csr_matrix(a.observed_counts.loc[names].to_numpy()))
        batch += [r] * len(sel)
        types += list(ctype[sel])
        settings[r] = dict(hyperparams=a.hypers.to_dict(), sources=src, seed=1000 * seed + off,
                           n_doublets=int(a.obs["is_doublet"].sum()))
        del exp, a
    return sp.vstack(parts).tocsr(), np.array(batch), np.array(types), settings


def write_bin(X_cells_by_genes, path):
    """Cells x genes CSR as the runner's genes x cells column-compressed int32 binary."""
    X = sp.csr_matrix(X_cells_by_genes)
    with open(path, "wb") as f:
        np.array([X.shape[1], X.shape[0], X.nnz], dtype="<i4").tofile(f)
        X.indptr.astype("<i4").tofile(f)
        X.indices.astype("<i4").tofile(f)
        np.rint(X.data).astype("<i4").tofile(f)


def read_bin(path):
    with open(path, "rb") as f:
        nr, nc, nnz = np.fromfile(f, dtype="<i4", count=3)
        p = np.fromfile(f, dtype="<i4", count=nc + 1)
        i = np.fromfile(f, dtype="<i4", count=nnz)
        x = np.fromfile(f, dtype="<i4", count=nnz)
    return sp.csc_matrix((x.astype(np.int64), i, p), shape=(nr, nc)).T.tocsr()


class Splatter:
    """The R runner in the datasim-r image."""

    def __init__(self, tools_dir, work, types, group_prob, batch_cells):
        self.img = os.path.join(tools_dir, SPLATTER_IMAGE)
        self.work = work
        self.script = os.path.join(HERE, "tools", "batch_splatter.R")
        self.types, self.group_prob, self.batch_cells = list(types), list(group_prob), list(batch_cells)
        self.log = []

    def _r(self, *args):
        cmd = ["apptainer", "exec", "--cleanenv", "-B", "/nobackup", "-B", self.work, self.img,
               "/opt/conda/bin/Rscript", self.script, *args]
        t0 = time.time()
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.log.append(dict(args=list(args)[:1], seconds=round(time.time() - t0, 1), rc=res.returncode))
        sys.stdout.write(res.stdout[-4000:])
        if res.returncode != 0:
            sys.stderr.write(res.stderr[-8000:])
            raise RuntimeError(f"batch_splatter.R {args[0]} failed (rc {res.returncode})")
        return res

    def version(self):
        return subprocess.run(["apptainer", "exec", self.img, "cat", "/opt/versions.txt"],
                              capture_output=True, text=True).stdout.strip()

    def estimate(self, X):
        inp = os.path.join(self.work, "real.bin")
        write_bin(X, inp)
        self.params = os.path.join(self.work, "splat_params.rds")
        info = os.path.join(self.work, "splat_info.json")
        self._r("estimate", inp, self.params, info)
        with open(info) as f:
            return json.load(f)

    def simulate(self, runs, tag):
        spec = dict(group_prob=self.group_prob, batch_cells=self.batch_cells, runs=runs)
        sp_path = os.path.join(self.work, f"spec_{tag}.json")
        with open(sp_path, "w") as f:
            json.dump(spec, f)
        out = os.path.join(self.work, f"sim_{tag}")
        self._r("simulate", self.params, sp_path, out)
        res = {}
        for r in runs:
            X = read_bin(os.path.join(out, f"{r['name']}.bin"))
            cols = pd.read_csv(os.path.join(out, f"{r['name']}_cols.tsv"), sep="\t")
            batch = cols["Batch"].map({"Batch1": "A", "Batch2": "B"}).to_numpy()
            grp = cols["Group"].astype(str).str.replace("Group", "").astype(int).to_numpy()
            ctype = np.array([self.types[g - 1] for g in grp])
            res[r["name"]] = (X, batch, ctype)
        shutil.rmtree(out)
        return res


def _interp(xs, ys, target):
    """x at which the (monotone) curve ys(xs) crosses target, by linear interpolation; with the
    bracketing pair, or None when the target is outside the curve's range."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    for i in range(len(xs) - 1):
        lo, hi = sorted((ys[i], ys[i + 1]))
        if lo <= target <= hi and ys[i + 1] != ys[i]:
            x = xs[i] + (target - ys[i]) * (xs[i + 1] - xs[i]) / (ys[i + 1] - ys[i])
            return float(x), (float(xs[i]), float(xs[i + 1]))
    return None, None


def tune(splat, name, grid, target, stat, fixed):
    """Grid, a finer grid inside the bracket, linear interpolation; returns (value, trace)."""
    trace = []

    def evaluate(values, tag):
        runs = [dict(fixed, name=f"{tag}{i}", seed=TUNE_SEED, **{name: float(v)}) for i, v in enumerate(values)]
        out = splat.simulate(runs, tag)
        for i, v in enumerate(values):
            s = stat(*out[f"{tag}{i}"])
            trace.append(dict(value=float(v), stat=float(s)))
            print(f"[tune {name}] {v:.4g} -> {s:.4f} (target {target:.4f})", flush=True)

    evaluate(grid, f"{name}_g")
    pts = sorted(trace, key=lambda d: d["value"])
    x, br = _interp([d["value"] for d in pts], [d["stat"] for d in pts], target)
    if x is None:
        return None, dict(target=target, trace=trace, matched=False)
    fine = np.linspace(br[0], br[1], N_REFINE + 2)[1:-1]
    evaluate(fine, f"{name}_f")
    pts = sorted(trace, key=lambda d: d["value"])
    x2, br2 = _interp([d["value"] for d in pts], [d["stat"] for d in pts], target)
    return (x2 if x2 is not None else x), dict(target=target, trace=trace, matched=True, bracket=br2 or br)


# ------------------------------------------------------------------ run
def run(a):
    t0 = time.time()
    if os.path.exists(a.out):
        sys.exit(f"{a.out} exists; remove it deliberately to rerun")
    if a.synthetic:
        fits, fits_sha = synthetic_fits(), None
    else:
        with open(a.fits, "rb") as f:
            fits_sha = hashlib.sha256(f.read()).hexdigest()
        with open(a.fits) as f:
            fits = json.load(f)
    os.makedirs(a.work, exist_ok=True)
    work = a.work
    res = dict(registered="exp-paper1-fig5-cohort, Amendment 2026-11-16 night (the new panel)",
               fits=dict(path=os.path.abspath(a.fits), sha256=fits_sha, created=fits.get("created")),
               settings=dict(min_mean=MIN_MEAN, min_type_cells=MIN_TYPE_CELLS, k=K, n_pcs=N_PCS, seeds=SEEDS,
                             tune_seed=TUNE_SEED, sigma_batch=SIGMA_BATCH, size_factor=SIZE_FACTOR,
                             de_grid=DE_GRID, batch_grid=BATCH_GRID, n_refine=N_REFINE,
                             summaries=SUMMARIES),
               host=platform.node(), commit=_commit(), versions={})
    import iscc
    res["versions"].update(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                           iscc=getattr(iscc, "__version__", None))
    try:
        X, genes, obs = synthetic_data() if a.synthetic else real_data(a.ding_dir, a.pbmcsca_meta)
        res["synthetic"] = bool(a.synthetic)
        types, counts = type_set(obs)
        res["data"] = dict(n_cells={r: int((obs["run"] == r).sum()) for r in ("A", "B")}, n_genes=int(X.shape[1]),
                           cells_by_type_and_run=counts, types_scored=types)
        print(f"[bv] real: {X.shape[0]} cells x {X.shape[1]} genes; types scored {types}", flush=True)
        versions, embeds = {}, {}

        def record(name, Xv, batch, ctype, extra=None):
            st = contrast_stats(Xv, batch, ctype, types)
            pca = embedding(Xv)
            st["S4"] = silhouettes(pca, batch, ctype)
            path = os.path.join(work, f"emb_{name}.npz")
            np.savez(path, pca=pca, batch=np.asarray(batch), ctype=np.asarray(ctype))
            embeds[name] = path
            if extra:
                st.update(extra)
            versions[name] = st
            s = st["summary"]
            print(f"[bv] {name}: S0 {s['S0']:.4f} S1 {s['S1_spearman']:.3f}/{s['S1_spread_ratio']:.3f} "
                  f"S2 {s['S2_slope_low']:.3f}/{s['S2_slope_mid']:.3f}/{s['S2_slope_high']:.3f} "
                  f"S3 {s['S3_median_corr']:.3f} sil {st['S4']}", flush=True)
            return st

        real = record("real", X, obs["run"].to_numpy(), obs["cell_type"].astype(str).to_numpy())
        for s in SEEDS:
            Xi, bi, ti, settings = iscc_version(X, obs, fits, s)
            record(f"iscc_s{s}", Xi, bi, ti, dict(settings=settings))
            del Xi

        # Splatter: estimate, tune DE to the cell-type silhouette, tune the batch factor to S0
        ctype = obs["cell_type"].astype(str).to_numpy()
        all_types = sorted(set(ctype))
        prob = [float((ctype == t).mean()) for t in all_types]
        nA, nB = int((obs["run"] == "A").sum()), int((obs["run"] == "B").sum())
        splat = Splatter(a.tools_dir, work, all_types, prob, [nA, nB])
        res["versions"]["splatter_image"] = splat.version()
        est = splat.estimate(X)

        def sil_type(Xv, batch, ct):
            return silhouettes(embedding(Xv), batch, ct)["silhouette_celltype"]

        def s0(Xv, batch, ct):
            return contrast_stats(Xv, batch, ct, types, curves=False)["summary"]["S0"]

        de, de_trace = tune(splat, "de_facLoc", DE_GRID, real["S4"]["silhouette_celltype"], sil_type,
                            dict(batch_facScale=0.0))
        res["splatter"] = dict(estimate=est, group_prob=dict(zip(all_types, prob)), batch_cells=[nA, nB],
                               de_tuning=de_trace, de_facLoc=de)
        if de is None:
            raise RuntimeError("Splatter's cell-type silhouette cannot reach the real one on the DE grid")
        bf, bf_trace = tune(splat, "batch_facScale", BATCH_GRID, real["summary"]["S0"], s0, dict(de_facLoc=de))
        res["splatter"].update(batch_tuning=bf_trace, batch_facScale=bf)
        if bf is None:
            raise RuntimeError("Splatter's S0 cannot reach the real one on the batch grid")
        out = splat.simulate([dict(name=f"splatter_s{s}", seed=s, de_facLoc=de, batch_facScale=bf) for s in SEEDS], "eval")
        for name, (Xs, bs, ts) in out.items():
            record(name, Xs, bs, ts, dict(splatter=dict(de_facLoc=de, batch_facScale=bf)))
        del out

        iscc_s0 = float(np.median([versions[f"iscc_s{s}"]["summary"]["S0"] for s in SEEDS]))
        ratio = iscc_s0 / real["summary"]["S0"]
        res["S0"] = dict(real=real["summary"]["S0"], iscc_median=iscc_s0, ratio=ratio,
                         differs_by_more_than_factor=bool(max(ratio, 1 / ratio) > SIZE_FACTOR),
                         splatter_median=float(np.median([versions[f"splatter_s{s}"]["summary"]["S0"] for s in SEEDS])))
        if res["S0"]["differs_by_more_than_factor"]:
            bf2, bf2_trace = tune(splat, "batch_facScale", BATCH_GRID, iscc_s0, s0, dict(de_facLoc=de))
            res["splatter"].update(batch_tuning_iscc_size=bf2_trace, batch_facScale_iscc_size=bf2)
            if bf2 is not None:
                out = splat.simulate([dict(name=f"splatteriscc_s{s}", seed=s, de_facLoc=de, batch_facScale=bf2)
                                      for s in SEEDS], "evaliscc")
                for name, (Xs, bs, ts) in out.items():
                    record(name, Xs, bs, ts, dict(splatter=dict(de_facLoc=de, batch_facScale=bf2)))
                del out
        res["splatter"]["runner_log"] = splat.log

        # S4's LISI in the integration image (scib-metrics, as Fig 5)
        lisi = run_lisi(embeds, a.tools_dir, work)
        for name, v in lisi.items():
            versions[name]["S4"].update(v)
        res["versions_stats"] = versions
        res["score"] = score(versions, res["S0"]["differs_by_more_than_factor"])
    finally:
        shutil.rmtree(work, ignore_errors=True)
    res["wall_s"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out + ".part", "w") as f:
        json.dump(res, f, indent=1, default=_jsonable)
    os.replace(a.out + ".part", a.out)
    print(f"[bv] wrote {a.out} in {res['wall_s']} s", flush=True)


def run_lisi(embeds, tools_dir, work):
    listing = os.path.join(work, "embeds.json")
    with open(listing, "w") as f:
        json.dump(embeds, f)
    out = os.path.join(work, "lisi.json")
    wrapper = os.path.join(HERE, "tools", "bin", "python-integration")
    # the wrapper binds ISCC_WORK_DIR into the container: the whole work directory, where the
    # embeddings and the listing are
    env = dict(os.environ, ISCC_TOOLS_DIR=tools_dir, ISCC_WORK_DIR=work)
    res = subprocess.run([wrapper, os.path.abspath(__file__), "lisi", listing, out], env=env,
                         capture_output=True, text=True)
    sys.stdout.write(res.stdout[-3000:])
    if res.returncode != 0:
        sys.stderr.write(res.stderr[-6000:])
        raise RuntimeError("LISI in the integration image failed")
    with open(out) as f:
        return json.load(f)


def lisi_cmd(listing, out):
    """Inside the integration image: iLISI (batch) and cLISI (type), integrate.py's lisi_scaled."""
    from integrate import lisi_scaled
    try:
        import scib_metrics  # noqa: F401
        backend = "scib-metrics"
    except Exception:                                   # noqa: BLE001
        backend = "harmonypy.compute_lisi"
    with open(listing) as f:
        embeds = json.load(f)
    res = {}
    for name, path in embeds.items():
        z = np.load(path, allow_pickle=False)
        res[name] = dict(ilisi_batch=lisi_scaled(z["pca"], z["batch"], K, backend, "i"),
                         clisi_celltype=lisi_scaled(z["pca"], z["ctype"], K, backend, "c"),
                         lisi_backend=backend)
        print(f"[lisi] {name}: {res[name]}", flush=True)
    with open(out, "w") as f:
        json.dump(res, f)


def score(versions, size_differs):
    """The registered criterion, per summary and per statistic."""
    def med(prefix, key):
        vals = [versions[f"{prefix}_s{s}"]["summary"][key] for s in SEEDS if f"{prefix}_s{s}" in versions]
        return float(np.median(vals)) if vals else None

    out = {}
    against = ["splatter"] + (["splatteriscc"] if size_differs and "splatteriscc_s1" in versions else [])
    for ref in against:
        per = {}
        for stat, keys in SUMMARIES.items():
            comp = {}
            for k in keys:
                r, i, s = versions["real"]["summary"][k], med("iscc", k), med(ref, k)
                comp[k] = dict(real=r, iscc_median=i, splatter_median=s, iscc_distance=abs(i - r),
                               splatter_distance=abs(s - r), iscc_closer=bool(abs(i - r) < abs(s - r)))
            n = sum(c["iscc_closer"] for c in comp.values())
            per[stat] = dict(summaries=comp, n_closer=n, n_summaries=len(keys), reproduced=bool(n == len(keys)))
        out[f"against_{ref}"] = per
    return out


def _jsonable(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (tuple, set)):
        return list(o)
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def _commit():
    try:
        return subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--fits")
    r.add_argument("--ding-dir")
    r.add_argument("--pbmcsca-meta")
    r.add_argument("--out", required=True)
    r.add_argument("--work", required=True)
    r.add_argument("--tools-dir", required=True)
    r.add_argument("--synthetic", action="store_true",
                   help="smoke run on a small dataset made by iscc (no real statistic is computed)")
    li = sub.add_parser("lisi")
    li.add_argument("listing")
    li.add_argument("out")
    fg = sub.add_parser("figure")
    fg.add_argument("--validation", required=True)
    a = p.parse_args(argv)
    if a.cmd == "run":
        _check_run_args(a)
        run(a)
    elif a.cmd == "lisi":
        lisi_cmd(a.listing, a.out)
    else:
        import batch_figure
        batch_figure.main(a.validation)


def _check_run_args(a):
    if not a.synthetic and not (a.fits and a.ding_dir and a.pbmcsca_meta):
        sys.exit("run needs --fits, --ding-dir and --pbmcsca-meta (or --synthetic)")


if __name__ == "__main__":
    main()
