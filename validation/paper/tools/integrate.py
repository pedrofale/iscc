"""Integrate a cohort's scRNA across batches and score the embedding (Fig 5, Part I).

Runs inside the integration image (images/integration.sh), never in envs/iscc:

    bin/python-integration integrate.py <cells.h5ad> <out_prefix> --method {none,harmony,scvi}

Input: an h5ad of raw counts (``layers['counts']`` if present, else ``X``) with obs columns for the
batch, the patient, the cell type, the cancer flag and the clone (names set by the ``--*-key``
options; the cancer flag may be bool, 0/1 or "True"/"False").

Methods, all on the same genes (every gene unless ``--n-top-genes``):

* ``none``: PCA (``--n-pcs`` 50) of log-normalised counts (total 1e4 per cell, log1p, centred, not
  scaled), the embedding every other method is compared with.
* ``harmony``: harmonypy (Korsunsky et al. 2019) on that PCA with the batch key, package defaults.
* ``scvi``: scVI (Lopez et al. 2018) on the counts with the batch key: 2 layers, 30 latent
  dimensions, scvi-tools' default number of epochs, on CPU.

Output: ``<out_prefix>.npy`` (cells x dimensions, in the h5ad's cell order) and ``<out_prefix>.json``
(the settings, the versions in the image, the timings and, unless ``--no-metrics``, the metrics).

Metrics, on a k-nearest-neighbour graph (``--k`` 30, exact Euclidean, self excluded):

* ``ilisi_batch`` and ``ilisi_batch_noncancer``: scib-style normalised iLISI over the batch key
  (Luecken et al. 2022; median over cells, scaled to [0, 1], 1 = fully mixed), on all cells and on
  the non-cancer cells (k-NN graph built within the subset); perplexity k / 3.
* ``clisi_celltype``: normalised cLISI over the cell type, all cells (1 = types fully separated).
* ``silhouette_celltype`` (scib's ASW label, rescaled to [0, 1]) and ``silhouette_batch`` (scib's
  batch ASW within each cell type, 1 = mixed).
* ``patient_purity_cancer`` / ``patient_purity_noncancer``: the mean share of a cell's k neighbours
  from its own patient, with the graph built within the cancer (non-cancer) cells.
* ``clone_purity_cancer``: within each patient's cancer cells, the mean share of a cell's neighbours
  (k, or the patient's cancer cells less one) from its own clone, averaged over all cancer cells.

Retention against the batch-free truth is a ratio of two of these files and is left to the caller.
LISI comes from scib-metrics; if it cannot be imported, from harmonypy's ``compute_lisi`` with the
same normalisation (recorded in ``metrics_backend``).

Temporary files go to a fresh directory under ``--work-dir`` (default ``$ISCC_WORK_DIR``, else
``$SNIC_TMP``; never /tmp), removed at the end: tempfile, numba, matplotlib, torch, JAX and the
Lightning logs are all pointed there before the libraries are imported.
"""
import argparse
import json
import os
import platform
import shutil
import sys
import tempfile
import time


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("h5ad")
    p.add_argument("out_prefix", help="writes <out_prefix>.npy and <out_prefix>.json")
    p.add_argument("--method", required=True, choices=["none", "harmony", "scvi"])
    p.add_argument("--batch-key", default="batch")
    p.add_argument("--patient-key", default="patient")
    p.add_argument("--celltype-key", default="cell_type")
    p.add_argument("--cancer-key", default="is_cancer")
    p.add_argument("--clone-key", default="clone")
    p.add_argument("--layer", default=None, help="counts layer (default: 'counts' if present, else X)")
    p.add_argument("--n-top-genes", type=int, default=0,
                   help="restrict to this many highly variable genes (seurat_v3 on counts, per batch); "
                        "0 = every gene (default)")
    p.add_argument("--n-pcs", type=int, default=50)
    p.add_argument("--n-latent", type=int, default=30)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--max-epochs", type=int, default=None, help="scVI epochs (default: scvi-tools' own)")
    p.add_argument("--k", type=int, default=30, help="neighbours for the metrics")
    p.add_argument("--no-metrics", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=int(os.environ.get("ISCC_THREADS", "1")))
    p.add_argument("--work-dir", default=os.environ.get("ISCC_WORK_DIR") or os.environ.get("SNIC_TMP"))
    p.add_argument("--keep-work", action="store_true")
    return p.parse_args(argv)


def make_work_dir(base, prefix):
    """A fresh directory under ``base`` for every temporary file, and the environment pointed at it.
    Must run before numba, matplotlib, torch, JAX or scvi are imported."""
    if not base:
        sys.exit("no work directory: pass --work-dir or set ISCC_WORK_DIR / SNIC_TMP (never /tmp)")
    os.makedirs(base, exist_ok=True)
    work = tempfile.mkdtemp(prefix=prefix, dir=base)
    for sub in ("tmp", "numba", "mpl", "cache", "torch", "lightning"):
        os.makedirs(os.path.join(work, sub), exist_ok=True)
    os.environ.update(
        TMPDIR=os.path.join(work, "tmp"), TEMP=os.path.join(work, "tmp"), TMP=os.path.join(work, "tmp"),
        NUMBA_CACHE_DIR=os.path.join(work, "numba"), MPLCONFIGDIR=os.path.join(work, "mpl"),
        XDG_CACHE_HOME=os.path.join(work, "cache"), TORCH_HOME=os.path.join(work, "torch"),
        JAX_PLATFORMS="cpu", PYTHONHASHSEED=os.environ.get("PYTHONHASHSEED", "0"))
    tempfile.tempdir = os.path.join(work, "tmp")
    return work


def set_threads(n):
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[v] = str(n)


def as_bool(series):
    import numpy as np
    if series.dtype == bool:
        return series.values
    s = series.astype(str).str.strip().str.lower()
    out = s.isin(["true", "1", "1.0", "yes", "cancer"])
    bad = ~(out | s.isin(["false", "0", "0.0", "no", "normal", "non-cancer"]))
    if bad.any():
        raise ValueError(f"cancer flag has values that are neither true nor false: {sorted(set(s[bad]))[:5]}")
    return np.asarray(out)


def image_versions():
    try:
        with open("/opt/versions.txt") as f:
            return dict(line.split(" ", 1) for line in f.read().strip().splitlines())
    except OSError:
        return {}


def counts_adata(adata, layer):
    """A copy whose X holds the raw counts."""
    import anndata as ad
    import scipy.sparse as sp
    if layer is None:
        layer = "counts" if "counts" in adata.layers else None
    X = adata.layers[layer] if layer is not None else adata.X
    X = sp.csr_matrix(X, dtype="float32")
    return ad.AnnData(X=X, obs=adata.obs.copy(), var=adata.var[[]].copy()), layer or "X"


def lognorm_pca(adata, n_pcs, seed):
    import scanpy as sc
    a = adata.copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    n = min(n_pcs, min(a.shape) - 1)
    sc.pp.pca(a, n_comps=n, zero_center=True, svd_solver="arpack", random_state=seed)
    return a.obsm["X_pca"].astype("float32"), n


def run_harmony(pca, obs, batch_key, seed, threads):
    import numpy as np
    import harmonypy
    meta = obs[[batch_key]].astype(str)
    ho = harmonypy.run_harmony(pca, meta, [batch_key], random_state=seed, ncores=threads, verbose=False)
    Z = np.asarray(ho.Z_corr)
    if Z.shape[0] != pca.shape[0]:                 # harmonypy < 2 returned dimensions x cells
        Z = Z.T
    return Z.astype("float32"), dict(package_defaults=True, random_state=seed, ncores=threads,
                                     batch_key=batch_key)


def run_scvi(adata, batch_key, n_layers, n_latent, max_epochs, seed, threads, work):
    import torch
    import scvi
    scvi.settings.seed = seed
    scvi.settings.num_threads = threads
    scvi.settings.dl_num_workers = 0
    scvi.settings.logging_dir = os.path.join(work, "lightning")
    torch.set_num_threads(threads)
    a = adata.copy()
    a.obs[batch_key] = a.obs[batch_key].astype(str).astype("category")
    scvi.model.SCVI.setup_anndata(a, batch_key=batch_key)
    model = scvi.model.SCVI(a, n_layers=n_layers, n_latent=n_latent)
    kw = dict(accelerator="cpu", devices=1, enable_progress_bar=False)
    if max_epochs is not None:
        kw["max_epochs"] = max_epochs
    model.train(**kw)
    Z = model.get_latent_representation().astype("float32")
    hist = model.history.get("elbo_train")
    epochs = int(len(hist)) if hist is not None else None
    settings = dict(n_layers=n_layers, n_latent=n_latent, batch_key=batch_key, gene_likelihood="zinb",
                    max_epochs_requested=max_epochs, epochs_trained=epochs,
                    elbo_train_last=float(hist.iloc[-1, 0]) if hist is not None else None,
                    accelerator="cpu", seed=seed, threads=threads)
    try:
        settings["gene_likelihood"] = model.module.gene_likelihood
    except AttributeError:
        pass
    return Z, settings


# ------------------------------------------------------------------ metrics
def knn(X, k):
    """Exact Euclidean k nearest neighbours, self excluded: (indices, distances), each n x k."""
    from sklearn.neighbors import NearestNeighbors
    k = min(k, X.shape[0] - 1)
    d, i = NearestNeighbors(n_neighbors=k).fit(X).kneighbors()   # no query: self is left out
    return i, d


def lisi_scaled(X, labels, k, backend, kind):
    """scib normalisation of the median LISI over cells, perplexity k / 3:
    iLISI (L - 1) / (B - 1), cLISI (C - L) / (C - 1)."""
    import numpy as np
    import pandas as pd
    labels = np.asarray(pd.Categorical(labels).codes)
    n_lab = len(np.unique(labels))
    if n_lab < 2 or len(labels) < 3:
        return None
    if backend == "scib-metrics":
        from scib_metrics import lisi_knn
        from scib_metrics.nearest_neighbors import NeighborsResults
        idx, dist = knn(X, k)
        n = len(labels)
        # scib-metrics' convention: the graph includes each cell's self edge, which it masks out
        idx = np.hstack([np.arange(n)[:, None], idx])
        dist = np.hstack([np.zeros((n, 1)), dist])
        L = np.asarray(lisi_knn(NeighborsResults(indices=idx, distances=dist), labels,
                                perplexity=(idx.shape[1] - 1) / 3))
    else:                                            # harmonypy builds its own graph of 3 x perplexity
        import harmonypy
        L = harmonypy.compute_lisi(X, pd.DataFrame({"l": labels}), ["l"], perplexity=k / 3)[:, 0]
    med = float(np.nanmedian(L))
    return (med - 1) / (n_lab - 1) if kind == "i" else (n_lab - med) / (n_lab - 1)


def purity(X, labels, k):
    import numpy as np
    labels = np.asarray(labels)
    if len(labels) < 2:
        return None, 0
    idx, _ = knn(X, k)
    return float((labels[idx] == labels[:, None]).mean()), len(labels)


def metrics(Z, obs, keys, k):
    import numpy as np
    try:
        import scib_metrics  # noqa: F401
        backend = "scib-metrics"
    except Exception as e:                           # noqa: BLE001
        print(f"[integrate] scib-metrics unavailable ({e}); LISI from harmonypy", file=sys.stderr)
        backend = "harmonypy.compute_lisi"
    batch = obs[keys["batch"]].astype(str).values
    ctype = obs[keys["celltype"]].astype(str).values
    patient = obs[keys["patient"]].astype(str).values
    clone = obs[keys["clone"]].astype(str).values
    cancer = as_bool(obs[keys["cancer"]])
    m = dict(metrics_backend=backend, k=k, perplexity=k / 3, n_cells=int(len(obs)),
             n_cancer=int(cancer.sum()), n_batches=int(len(set(batch))))
    m["ilisi_batch"] = lisi_scaled(Z, batch, k, backend, "i")
    m["ilisi_batch_noncancer"] = lisi_scaled(Z[~cancer], batch[~cancer], k, backend, "i")
    m["clisi_celltype"] = lisi_scaled(Z, ctype, k, backend, "c")
    try:
        import scib_metrics
        m["silhouette_celltype"] = float(scib_metrics.silhouette_label(Z, ctype, rescale=True))
        m["silhouette_batch"] = float(scib_metrics.silhouette_batch(Z, ctype, batch, rescale=True))
        m["silhouette_backend"] = "scib-metrics"
    except Exception as e:                           # noqa: BLE001
        from sklearn.metrics import silhouette_score
        m["silhouette_celltype"] = float((silhouette_score(Z, ctype) + 1) / 2)
        m["silhouette_batch"] = None
        m["silhouette_backend"] = f"sklearn ({e})"
    m["patient_purity_cancer"], _ = purity(Z[cancer], patient[cancer], k)
    m["patient_purity_noncancer"], _ = purity(Z[~cancer], patient[~cancer], k)
    # clone purity within each patient's cancer cells, weighted by the patient's cancer cells
    tot, n = 0.0, 0
    for p in np.unique(patient[cancer]):
        sel = cancer & (patient == p)
        if sel.sum() < 2:
            continue
        v, c = purity(Z[sel], clone[sel], k)
        tot += v * c
        n += c
    m["clone_purity_cancer"] = tot / n if n else None
    # chance levels, to read the purities against
    m["patient_chance_cancer"] = chance(patient[cancer])
    m["patient_chance_noncancer"] = chance(patient[~cancer])
    return m


def chance(labels):
    import numpy as np
    if len(labels) == 0:
        return None
    _, c = np.unique(labels, return_counts=True)
    f = c / c.sum()
    return float((f ** 2).sum())


def main(argv=None):
    args = parse_args(argv)
    args.h5ad, args.out_prefix = os.path.abspath(args.h5ad), os.path.abspath(args.out_prefix)
    work = make_work_dir(args.work_dir, f"integrate-{args.method}-")
    set_threads(args.threads)
    cwd = os.getcwd()
    t0 = time.time()
    timings = {}
    try:
        os.chdir(work)                               # Lightning and anything else write here
        import numpy as np
        import anndata as ad
        adata = ad.read_h5ad(args.h5ad)
        keys = dict(batch=args.batch_key, patient=args.patient_key, celltype=args.celltype_key,
                    cancer=args.cancer_key, clone=args.clone_key)
        missing = [v for v in keys.values() if v not in adata.obs.columns]
        if missing:
            sys.exit(f"obs lacks the columns {missing}; set them with the --*-key options")
        a, layer = counts_adata(adata, args.layer)
        del adata
        genes = "all"
        if args.n_top_genes:
            import scanpy as sc
            sc.pp.highly_variable_genes(a, flavor="seurat_v3", n_top_genes=args.n_top_genes,
                                        batch_key=args.batch_key, subset=True)
            genes = f"{a.n_vars} highly variable (seurat_v3, per batch)"
        timings["read_s"] = round(time.time() - t0, 2)

        t1 = time.time()
        settings = dict(method=args.method, genes=genes, n_genes=int(a.n_vars), counts_from=layer)
        if args.method in ("none", "harmony"):
            pca, n_pcs = lognorm_pca(a, args.n_pcs, args.seed)
            settings.update(normalisation="total 1e4 per cell, log1p", pca=dict(
                n_comps=n_pcs, zero_center=True, scaled=False, svd_solver="arpack"))
            timings["pca_s"] = round(time.time() - t1, 2)
            Z = pca
            if args.method == "harmony":
                t2 = time.time()
                Z, hs = run_harmony(pca, a.obs, args.batch_key, args.seed, args.threads)
                settings["harmony"] = hs
                timings["harmony_s"] = round(time.time() - t2, 2)
        else:
            Z, ss = run_scvi(a, args.batch_key, args.n_layers, args.n_latent, args.max_epochs,
                             args.seed, args.threads, work)
            settings["scvi"] = ss
            timings["scvi_s"] = round(time.time() - t1, 2)
        timings["method_s"] = round(time.time() - t1, 2)

        out = dict(input=args.h5ad, embedding=args.out_prefix + ".npy",
                   shape=list(Z.shape), keys=keys, seed=args.seed, threads=args.threads,
                   settings=settings, versions=image_versions(), host=platform.node(),
                   date=time.strftime("%Y-%m-%d %H:%M:%S"), work_dir=work)
        if not args.no_metrics:
            t3 = time.time()
            out["metrics"] = metrics(Z, a.obs, keys, args.k)
            timings["metrics_s"] = round(time.time() - t3, 2)
        timings["total_s"] = round(time.time() - t0, 2)
        out["timings"] = timings
        os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)
        np.save(args.out_prefix + ".npy", Z)
        with open(args.out_prefix + ".json", "w") as f:
            json.dump(out, f, indent=1, default=float)
        brief = {k: (round(v, 3) if isinstance(v, float) else v)
                 for k, v in out.get("metrics", {}).items()}
        print(f"[integrate] {args.method}: {Z.shape} in {timings['total_s']} s -> {args.out_prefix}.npy")
        print(f"[integrate] metrics {json.dumps(brief)}")
    finally:
        os.chdir(cwd)
        if not args.keep_work:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
