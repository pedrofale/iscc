"""Supplementary figure: Fig 3's measurement statistics for other data simulators.

Each modality is scored on the statistic Fig 3 uses, against the same real reference, with the same
statistic function, so a competitor and ``iscc`` are measured identically:

  (a) WGS: per-locus coverage at the GIAB HG002 benchmark sites (copy number 2) and its CV, for VISOR
      SHORtS reads (pywgsim, aligned by minimap2) from diploid GRCh38 chr1 haplotypes at HG002's mean
      depth (no copy-number change is implanted); iscc's loci are those at its tumour's most common
      copy number (Fig 3b), so every source is scored at one copy-number state;
  (b) scDNA: within-cell coverage CV per copy-number state (``fig2_realism._cv_by_state``), against
      DLP+, for CNAsim in count mode with its coverage non-uniformity set from DLP+;
  (c) scRNA: library size, mean against variance and zero fraction (``fig2_realism._count_stats``),
      against PBMC3k, for Splatter (``splatEstimate``) and scDesign3 (negative binomial by cluster,
      Gaussian copula);
  (d) Visium: per-spot depth and its Moran's I (``iscc.data.morans_i`` on spot-pitch coordinates),
      against the real breast-cancer section, for SRTsim (``srtsim_fit`` on the section, counts placed
      at the section's own spots by each gene's reference rank). stCNASim is not run (see COMPETITORS).

The real and ``iscc`` values are read from Fig 3's cache (``$ISCC_FIG3_CACHE``, parts b-e), so this
figure shows the same numbers as Fig 3. The simulators run in Slurm jobs that write their raw output
to node-local scratch and call this script's reducers there; only the small JSON summaries persist,
in ``$ISCC_DATASIM_DIR`` (default ``<derived>/datasim``). Each summary records the simulator's
version and the settings used.

    python validation/paper/supp_datasim.py prep-scrna DIR           # PBMC3k as a MatrixMarket input
    python validation/paper/supp_datasim.py reduce-scrna MTX NAME    # -> scrna_NAME.json
    python validation/paper/supp_datasim.py prep-visium DIR          # the section's spots and counts
    python validation/paper/supp_datasim.py reduce-visium MTX LOC NAME
    python validation/paper/supp_datasim.py prep-cnasim              # DLP+ depth and Lorenz point
    python validation/paper/supp_datasim.py reduce-cnasim DIR NAME
    python validation/paper/supp_datasim.py hg002-sites BED          # the HG002 sites, re-derived
    python validation/paper/supp_datasim.py reduce-wgs DEPTH BED NAME
    python validation/paper/supp_datasim.py iscc                     # real and iscc from Fig 3
    python validation/paper/supp_datasim.py render
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import common as C  # noqa: E402
import style as S  # noqa: E402

REPO = os.path.dirname(os.path.dirname(HERE))
SUM = os.environ.get("ISCC_DATASIM_DIR") or os.path.join(C.DERIVED, "datasim")
FIG3 = os.environ.get("ISCC_FIG3_CACHE") or os.path.join(C.DERIVED, "fig3")
DNA_REF = os.path.join(REPO, "validation", "data")
MIN_CELLS = 3          # genes kept as simulator input: detected in at least this many cells or spots


def _f2():
    import fig2_realism as F2
    return F2


def _cv(x):
    """Coverage CV over the covered loci, as Fig 3b scores WGS (fig3_biases._cv)."""
    x = np.asarray(x, float)
    x = x[x > 0]
    return float(x.std() / x.mean()) if x.size else float("nan")


def _meta(pairs):
    out = {}
    for p in pairs or []:
        k, _, v = p.partition("=")
        out[k] = v
    return out


def _write(name, d):
    os.makedirs(SUM, exist_ok=True)
    p = os.path.join(SUM, f"{name}.json")
    with open(p, "w") as f:
        json.dump(d, f, default=float)
    print("wrote", p)


def _load(name):
    p = os.path.join(SUM, f"{name}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def _mm_write(path, X):
    """Write a sparse matrix as MatrixMarket (integer counts), the format R's Matrix reads."""
    import scipy.io
    import scipy.sparse as sp
    scipy.io.mmwrite(path, sp.csr_matrix(X).astype(np.int64), field="integer")


def _mm_read(path):
    import scipy.io
    import scipy.sparse as sp
    return sp.csr_matrix(scipy.io.mmread(path))


# --- scRNA: PBMC3k ---------------------------------------------------------------------------------
def prep_scrna(out):
    """PBMC3k's raw counts (genes x cells, genes detected in >= MIN_CELLS cells) and a cluster label
    per cell, for the R simulators. Clusters: k-means (k = 8, the eight PBMC3k cell types of the
    standard tutorial) on 50 principal components of the normalised log counts."""
    import anndata as ad
    import scanpy as sc
    import scipy.sparse as sp
    from sklearn.cluster import KMeans
    os.makedirs(out, exist_ok=True)
    real = ad.read_h5ad(os.path.join(REPO, "data", "pbmc3k_raw.h5ad"))
    X = sp.csr_matrix(real.X)
    keep = np.asarray((X > 0).sum(0)).ravel() >= MIN_CELLS
    dropped = 1.0 - float(X[:, keep].sum() / X.sum())
    a = ad.AnnData(X[:, keep].astype(np.float32))
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, n_top_genes=2000)
    sc.pp.pca(a, n_comps=50, mask_var="highly_variable", random_state=0)
    lab = KMeans(n_clusters=8, n_init=10, random_state=0).fit_predict(a.obsm["X_pca"])
    _mm_write(os.path.join(out, "counts.mtx"), X[:, keep].T)
    np.savetxt(os.path.join(out, "cluster.txt"), lab, fmt="%d")
    info = dict(n_cells=int(X.shape[0]), n_genes_kept=int(keep.sum()), n_genes=int(X.shape[1]),
                umi_frac_dropped=dropped, clusters=np.bincount(lab).tolist())
    json.dump(info, open(os.path.join(out, "prep.json"), "w"))
    print(json.dumps(info))


def reduce_scrna(mtx, name, meta, cells_by_genes=False):
    """Fig 3d's statistics of a simulated count matrix (MatrixMarket, genes x cells)."""
    X = _mm_read(mtx)
    X = X if cells_by_genes else X.T.tocsr()
    d = dict(name=name, n_cells=int(X.shape[0]), n_genes=int(X.shape[1]), **_f2()._count_stats(X))
    d["meta"] = meta
    _write(f"scrna_{name}", d)


# --- Visium: the breast-cancer section -------------------------------------------------------------
def _section():
    sys.path.insert(0, os.path.join(REPO, "validation"))
    import validate_visium as VV
    cwd = os.getcwd()
    os.chdir(REPO)                               # scanpy's dataset cache lives in ./data
    try:
        real = VV.load_real_visium()
    finally:
        os.chdir(cwd)
    coords = np.asarray(real.obsm["spatial"], dtype=float) / VV._median_nn(real.obsm["spatial"])
    return real, coords


def prep_visium(out):
    """The section's in-tissue counts (genes x spots, genes detected in >= MIN_CELLS spots) and its
    spot positions in spot-pitch units, as SRTsim's reference."""
    import scipy.sparse as sp
    os.makedirs(out, exist_ok=True)
    real, coords = _section()
    X = sp.csr_matrix(real.X)
    keep = np.asarray((X > 0).sum(0)).ravel() >= MIN_CELLS
    dropped = 1.0 - float(X[:, keep].sum() / X.sum())
    _mm_write(os.path.join(out, "counts.mtx"), X[:, keep].T)
    with open(os.path.join(out, "loc.csv"), "w") as f:
        f.write("spot,x,y\n")
        for i, (x, y) in enumerate(coords):
            f.write(f"s{i},{x:.6f},{y:.6f}\n")
    info = dict(n_spots=int(X.shape[0]), n_genes_kept=int(keep.sum()), n_genes=int(X.shape[1]),
                umi_frac_dropped=dropped)
    json.dump(info, open(os.path.join(out, "prep.json"), "w"))
    print(json.dumps(info))


def reduce_visium(mtx, loc, name, meta):
    """Fig 3e's statistics: per-spot depth and its Moran's I on the spot-pitch coordinates."""
    import pandas as pd
    from iscc.data import morans_i
    X = _mm_read(mtx).T.tocsr()                  # spots x genes
    xy = pd.read_csv(loc)[["x", "y"]].to_numpy(float)
    tot = np.asarray(X.sum(1)).ravel()
    occ = tot > 0
    d = dict(name=name, n_spots=int(X.shape[0]), n_genes=int(X.shape[1]),
             logdepth=np.log10(tot[occ]).round(4).tolist(), moran=float(morans_i(tot[occ], xy[occ])),
             meta=meta)
    _write(f"visium_{name}", d)


# --- scDNA: DLP+ -----------------------------------------------------------------------------------
def _dlp():
    return np.load(os.path.join(DNA_REF, "dna_ref_dlp_sc_wgs.npz"), allow_pickle=True)


def lorenz_y(reads, cn, x=0.5):
    """Share of the reads in the lowest-covered fraction ``x`` of a cell's bins, after dividing each
    bin's reads by its copy number: the technical non-uniformity CNAsim's Lorenz point sets."""
    ys = []
    for r, c in zip(np.atleast_2d(reads), np.atleast_2d(cn)):
        ok = c > 0
        v = np.sort(r[ok] / c[ok])
        ys.append(v[: int(round(x * v.size))].sum() / v.sum())
    return float(np.median(ys))


def prep_cnasim():
    """DLP+'s mean reads per 500 kb bin, mean copy number and Lorenz point (x = 0.5)."""
    z = _dlp()
    cov, cn = np.asarray(z["coverage"], float), np.rint(np.asarray(z["cn"], float))
    print(json.dumps(dict(mean_reads=float(cov.mean()), mean_cn=float(cn.mean()), lorenz_y=lorenz_y(cov, cn),
                          n_cells=int(cov.shape[0]), n_bins=int(cov.shape[1]))))


def reduce_cnasim(out, name, meta):
    """CNAsim count-mode output -> within-cell coverage CV per copy-number state (Fig 3c)."""
    import pandas as pd
    rc = pd.read_csv(os.path.join(out, "readcounts.tsv"), sep="\t")
    pr = pd.read_csv(os.path.join(out, "profiles.tsv"), sep="\t")
    key = ["CELL", "chrom", "start", "end"]
    m = rc.merge(pr, on=key, how="inner")
    ab = m["CN states"].str.split(",", expand=True).astype(float)
    m["cn"] = ab[0] + ab[1]
    m["reads"] = m["Acount"] + m["Bcount"]
    cells = sorted(m["CELL"].unique())
    m["_bin"] = m.groupby("CELL").cumcount()
    reads = m.pivot(index="CELL", columns="_bin", values="reads").loc[cells].to_numpy(float)
    cn = m.pivot(index="CELL", columns="_bin", values="cn").loc[cells].to_numpy(float)
    cvs = _f2()._cv_by_state(reads, cn)
    states, counts = np.unique(np.rint(cn).astype(int), return_counts=True)
    d = dict(name=name, n_cells=len(cells), n_bins=int(reads.shape[1]), mean_reads=float(reads.mean()),
             lorenz_y=lorenz_y(reads, np.rint(cn)), cv_by_state={str(k): v for k, v in sorted(cvs.items())},
             state_frac={str(int(s)): float(c / cn.size) for s, c in zip(states, counts)}, meta=meta)
    _write(f"scdna_{name}", d)


# --- WGS: GIAB HG002 -------------------------------------------------------------------------------
def hg002_sites(bed):
    """Re-derive the positions of the 20,000 HG002 benchmark sites in the reduction (the reduction
    keeps their coverage but not their positions) by streaming the same VCF through the same parser,
    and check the re-derived coverage equals the stored one."""
    sys.path.insert(0, DNA_REF)
    import build_dna_reference as BDR
    z = np.load(os.path.join(DNA_REF, "dna_ref_hg002_bulk_wgs.npz"), allow_pickle=True)
    n = len(z["coverage"])
    sites, cov = [], []
    for line in BDR._stream_gzip_lines(BDR.GIAB_VCF, max_bytes=80 << 20):
        if not line or line.startswith("#"):
            continue
        f = line.split("\t")
        rec = BDR.parse_giab_record(f)
        if rec is None:
            continue
        sites.append((f[0], int(f[1])))
        cov.append(rec[0])
        if len(sites) >= n:
            break
    same = len(cov) == n and bool(np.array_equal(np.asarray(cov), z["coverage"]))
    with open(bed, "w") as fh:
        for c, p in sites:
            fh.write(f"{c}\t{p - 1}\t{p}\n")
    chroms = sorted({c for c, _ in sites})
    print(json.dumps(dict(n=len(sites), matches_reduction=same, chroms=chroms,
                          first=sites[0][1], last=sites[-1][1])))
    if not same:
        sys.exit("re-derived HG002 sites do not reproduce the stored coverage")


def reduce_wgs(depth, bed, name, meta):
    """Per-site depth (``samtools depth -a -b BED`` output) -> Fig 3b's coverage CV and the per-locus
    coverage over its mean. The simulated genome is diploid at every site, as Fig 3b's copy-number-2
    sites are; a site absent from the depth output counts as zero."""
    import pandas as pd
    s = pd.read_csv(bed, sep="\t", header=None, names=["chrom", "start", "pos"])
    d = pd.read_csv(depth, sep="\t", header=None, names=["chrom", "pos", "depth"])
    x = s.merge(d, on=["chrom", "pos"], how="left").fillna({"depth": 0})["depth"].to_numpy(float)
    out = dict(name=name, n_sites=int(x.size), mean_depth=float(x.mean()), cv=_cv(x),
               norm=(x / x.mean()).round(4).tolist(), meta=meta)
    _write(f"wgs_{name}", out)


# --- real and iscc, from Fig 3's cache -------------------------------------------------------------
def collect_iscc():
    def part(p):
        f = os.path.join(FIG3, f"{p}.json")
        if not os.path.exists(f):
            sys.exit(f"Fig 3 cache part {p} missing in {FIG3}: run fig3_biases.py --part {p} first")
        return json.load(open(f))
    b, c, d, e = (part(p) for p in "bcde")
    dlp = c["left"]["DLP+"]
    out = dict(
        source=FIG3, tumour=b.get("_tumour"),
        wgs=dict(real=dict(norm=np.round(b["hist"]["real"], 4).tolist(), cv=b["cv_real"]),
                 iscc=dict(norm=np.round(b["hist"]["iscc"], 4).tolist(), cv=b["cv_iscc"])),
        scdna=dict(real={k: v[0] for k, v in dlp.items()}, iscc={k: v[1] for k, v in dlp.items()}),
        scrna=dict(real=d["left"]["real"], iscc=d["left"]["iscc"]),
        visium=dict(real=dict(logdepth=e["left"]["toto"], moran=e["left"]["moran_real"]),
                    iscc=dict(logdepth=e["left"]["tots"], moran=e["left"]["moran_iscc"])))
    _write("iscc", out)


# --- render ----------------------------------------------------------------------------------------
COMPETITORS = {"wgs": ["VISOR"], "scdna": ["CNAsim"], "scrna": ["Splatter", "scDesign3"],
               "visium": ["SRTsim"]}
# stCNASim is not run: it needs a STARsolo-aligned seed section and took ~5 h per 600-spot run on
# 2026-11-14. Read from its code (cs/pp.calc_size_factors, not run): spot library sizes are drawn
# independently per spot from a log-normal fitted to the seed, so its depth has no spatial structure.


def _colour(name):
    return S.SOURCE.get(name) or S.TOOL[name]


def _sources(mod, d0):
    """[(label, colour, summary)] for real, iscc and every competitor with a summary."""
    out = [("real", S.SOURCE["real"], d0[mod]["real"]), ("iscc", S.SOURCE["iscc"], d0[mod]["iscc"])]
    for n in COMPETITORS[mod]:
        d = _load(f"{mod}_{n}")
        if d is not None:
            out.append((n, _colour(n), d))
    return out


def _missing(ax, mod):
    """Stop rather than draw a panel without a competitor: the figure carries no "not run" notes."""
    miss = [n for n in COMPETITORS[mod] if _load(f"{mod}_{n}") is None]
    if miss:
        sys.exit(f"{mod}: no summary for {', '.join(miss)} in {SUM}: run it before rendering")


def _density(ax, values, bins, colour, label):
    h, e = np.histogram(values, bins=bins, density=True)
    ax.step(e[:-1], h, where="post", color=colour, lw=0.8, label=label)


def _dots(ax, items, ylabel):
    """One dot per source (label, colour, value), for a scalar statistic."""
    for i, (lab, col, v) in enumerate(items):
        ax.scatter([i], [v], s=12, color=col, zorder=3)
        ax.annotate(f"{v:.2f}", (i, v), xytext=(0, 3), textcoords="offset points", ha="center",
                    va="bottom", fontsize=5, color=col)
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels([it[0] for it in items], rotation=90)
    ax.set_xlim(-0.6, len(items) - 0.4)
    ax.set_ylim(0, 1.25 * max(it[2] for it in items))
    ax.set_ylabel(ylabel)


def render():
    d0 = _load("iscc")
    if d0 is None:
        sys.exit("no iscc.json: run `supp_datasim.py iscc` first")
    S.setup()
    fig = S.figure("double", 150)
    out = {}
    # a. WGS
    S.panel_letter(fig, "a", 2, 4)
    S.text_mm(fig, 10, 4, "WGS: GIAB HG002", fontsize=6.5, va="top")
    ax1, ax2 = S.ax_mm(fig, 14, 11, 38, 26), S.ax_mm(fig, 62, 11, 14, 26)
    src = _sources("wgs", d0)
    bins = np.linspace(0, 3, 61)
    for lab, col, d in src:
        _density(ax1, d["norm"], bins, col, lab)
    ax1.set_xlabel("Coverage / mean coverage")
    ax1.set_ylabel("Density")
    ax1.legend(fontsize=5, loc="upper right")
    _dots(ax2, [(lab, col, d["cv"]) for lab, col, d in src], "Coverage CV")
    _missing(ax1, "wgs")
    out["a"] = {lab: d["cv"] for lab, _, d in src}
    # b. scDNA
    S.panel_letter(fig, "b", 92, 4)
    S.text_mm(fig, 100, 4, "scDNA: DLP+", fontsize=6.5, va="top")
    ax = S.ax_mm(fig, 104, 11, 70, 26)
    src = [(lab, col, d if lab in ("real", "iscc") else d["cv_by_state"]) for lab, col, d in _sources("scdna", d0)]
    for lab, col, d in src:
        k = sorted(d, key=int)
        ax.plot([int(x) for x in k], [d[x] for x in k], "o-", color=col, ms=2.5, lw=0.8, label=lab)
    ax.set_xlabel("Copy number state")
    ax.set_ylabel("Within-cell coverage CV")
    ax.set_ylim(0, None)
    ax.legend(fontsize=5, loc="upper right")
    _missing(ax, "scdna")
    out["b"] = {lab: d for lab, _, d in src}
    # c. scRNA
    S.panel_letter(fig, "c", 2, 50)
    S.text_mm(fig, 10, 50, "scRNA: PBMC3k", fontsize=6.5, va="top")
    axes = [S.ax_mm(fig, 14 + i * 58, 57, 44, 30) for i in range(3)]
    src = _sources("scrna", d0)
    lo = min(min(d["loglib"]) for _, _, d in src)
    hi = max(max(d["loglib"]) for _, _, d in src)
    bins = np.linspace(lo, hi, 40)
    for lab, col, d in src:
        _density(axes[0], d["loglib"], bins, col, lab)
        axes[1].plot(d["curve_m"], d["curve_v"], color=col, lw=0.8)
        axes[2].plot(d["curve_m"], d["curve_z"], color=col, lw=0.8)
    xs = np.linspace(-3.5, 2, 10)
    axes[1].plot(xs, xs, color=S.BG_CELLS, lw=0.5, ls=":", zorder=0, label="Poisson")
    axes[0].set_xlabel("log$_{10}$ UMIs per cell")
    axes[0].set_ylabel("Density")
    axes[0].legend(fontsize=5, loc="upper left")
    axes[1].set_xlabel("log$_{10}$ mean")
    axes[1].set_ylabel("log$_{10}$ variance")
    axes[1].legend(fontsize=5, loc="lower right")
    axes[2].set_xlabel("log$_{10}$ mean")
    axes[2].set_ylabel("Zero fraction")
    _missing(axes[0], "scrna")
    out["c"] = {lab: d["cv_lib"] for lab, _, d in src}
    # d. Visium
    S.panel_letter(fig, "d", 2, 100)
    S.text_mm(fig, 10, 100, "Visium: breast-cancer section", fontsize=6.5, va="top")
    ax1, ax2 = S.ax_mm(fig, 14, 107, 44, 30), S.ax_mm(fig, 72, 107, 14, 30)
    src = _sources("visium", d0)
    lo = min(min(d["logdepth"]) for _, _, d in src)
    hi = max(max(d["logdepth"]) for _, _, d in src)
    bins = np.linspace(lo, hi, 40)
    for lab, col, d in src:
        _density(ax1, d["logdepth"], bins, col, lab)
    ax1.set_xlabel("log$_{10}$ UMIs per spot")
    ax1.set_ylabel("Density")
    ax1.legend(fontsize=5, loc="upper left")
    _dots(ax2, [(lab, col, d["moran"]) for lab, col, d in src], "Moran's I of spot depth")
    ax2.set_ylim(0, 1)
    _missing(ax1, "visium")
    out["d"] = {lab: d["moran"] for lab, _, d in src}
    path = S.save(fig, "supp_datasim", stamp=False)
    print("wrote", path)
    for k, v in out.items():
        print(k, json.dumps(v, default=float))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, args in (("prep-scrna", ["dir"]), ("prep-visium", ["dir"]), ("prep-cnasim", []),
                       ("hg002-sites", ["bed"]), ("iscc", []), ("render", []),
                       ("reduce-scrna", ["mtx", "name"]), ("reduce-visium", ["mtx", "loc", "name"]),
                       ("reduce-cnasim", ["dir", "name"]), ("reduce-wgs", ["depth", "bed", "name"])):
        p = sub.add_parser(name)
        for a in args:
            p.add_argument(a)
        if name.startswith("reduce"):
            p.add_argument("--meta", nargs="*", default=[], help="key=value settings to record")
    a = ap.parse_args()
    meta = _meta(getattr(a, "meta", None))
    {"prep-scrna": lambda: prep_scrna(a.dir), "prep-visium": lambda: prep_visium(a.dir),
     "prep-cnasim": prep_cnasim, "hg002-sites": lambda: hg002_sites(a.bed), "iscc": collect_iscc,
     "render": render,
     "reduce-scrna": lambda: reduce_scrna(a.mtx, a.name, meta),
     "reduce-visium": lambda: reduce_visium(a.mtx, a.loc, a.name, meta),
     "reduce-cnasim": lambda: reduce_cnasim(a.dir, a.name, meta),
     "reduce-wgs": lambda: reduce_wgs(a.depth, a.bed, a.name, meta)}[a.cmd]()


if __name__ == "__main__":
    main()
