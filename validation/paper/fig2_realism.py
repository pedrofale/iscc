"""Figure 2: realistic by each front's own standard.

Top row, the evolution front's tests (registry tumours):
  (a) evolutionary-mode space (Noble's n, D) against real multi-region tumours, at the full census and
      at the 5% detection limit of Noble et al. (drivers below it removed, clones merged into parents);
  (b) copy number follows oncogenic content under selection, not under neutrality (R2);
  (c) whole-genome-duplication prevalence across rates against the PCAWG range (R5);
  (d) the fit to the PCAWG breast-cancer arm landscape: pending its rerun under the corrected model.
No 1/f panel: the 6,000-gene genome saturates at tumour scale, so the infinite-sites test does not apply.
Bottom row, the data front's tests, each fit on a real reference and re-simulated on the showcase
tumour (R0):
  (e) scRNA against PBMC3k: library size, mean-variance, zero fraction;
  (f) DNA against GIAB HG002 (bulk), Tapestri and DLP+ (single cell): coverage dispersion by copy
      number, and Tapestri allelic dropout;
  (g) Visium against a real breast-cancer section: per-spot depth and its spatial autocorrelation.

Run from the iscc repository root (scanpy's data cache is ./data):
    python validation/paper/fig2_realism.py
"""
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import common as C  # noqa: E402
import registry as REG  # noqa: E402
import style as S  # noqa: E402
import realistic_regime as RR  # noqa: E402

REPO = os.path.dirname(os.path.dirname(HERE))
DETECTION = 0.02


def have(run, task):
    return os.path.exists(REG.task_path(run, task))


def pending(ax, msg="pending: registry run not grown yet"):
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=5, color=S.GREY, transform=ax.transAxes,
            wrap=True)
    S.despine(ax, False, False)


# --- a-d read the cached per-tumour summaries (summaries.py), never the tumours themselves ---------
import summaries as SM  # noqa: E402


def panel_a(ax):
    from scipy.spatial import Delaunay
    emp = pd.read_csv(os.path.join(REPO, "validation", "data", "noble_empirical_indices.csv"))
    rows = [r for run in ("R1", "R2", "R3", "R4") for r in SM.load(run)]
    sim = pd.DataFrame(rows)[["n0", "D0", "nd", "Dd"]].dropna() if rows else pd.DataFrame()
    if sim.empty:
        return pending(ax)
    e = emp[["n", "D"]].to_numpy()
    cov = {}
    for k, (nc, dc) in {"census": ("n0", "D0"), "5%": ("nd", "Dd")}.items():
        try:
            cov[k] = float((Delaunay(sim[[nc, dc]].to_numpy()).find_simplex(e) >= 0).mean())
        except Exception:
            cov[k] = float("nan")
    ax.scatter(sim.n0, sim.D0, s=4, facecolors="none", edgecolors=S.SOURCE["iscc"], lw=0.3, alpha=0.5,
               label=f"iscc, full census ({cov['census']:.0%})")
    ax.scatter(sim.nd, sim.Dd, s=4, color=S.SOURCE["iscc"], alpha=0.8,
               label=f"iscc, 5% detection ({cov['5%']:.0%})")
    ax.scatter(emp.n, emp.D, s=6, color=S.SOURCE["real"], edgecolors=S.INK, lw=0.3, label="real tumours")
    ax.set_yscale("log")
    ax.set_xlabel("Drivers per cell, n")
    ax.set_ylabel("Clonal diversity, D")
    ax.legend(fontsize=4.5, loc="upper left", title="real tumours inside\nthe iscc hull",
              title_fontsize=4.5)
    return dict(n_sim=len(sim), coverage=cov, D_median_detected=float(sim.Dd.median()), n_median_detected=float(sim.nd.median()),
                D_median_census=float(sim.D0.median()), D_median_real=float(emp.D.median()))


def panel_c(ax):
    out = {}
    for name, col in (("neutral", S.SOURCE["real"]), ("selected", S.SOURCE["iscc"])):
        rows = [r for r in SM.load("R2") if r["task"].startswith(name)]
        if not rows:
            return pending(ax)
        net = np.concatenate([r["seg_net"] for r in rows]).astype(float)
        cn = np.concatenate([r["seg_cn"] for r in rows]).astype(float)
        r = float(np.corrcoef(net, cn)[0, 1]) if np.std(net) > 0 and np.std(cn) > 0 else float("nan")
        ax.scatter(net + np.random.default_rng(0).uniform(-0.2, 0.2, len(net)), cn, s=2, color=col,
                   alpha=0.7, label=f"{name} (r = {r:.2f})")
        out[name] = r
    ax.axhline(2, color=S.BG_CELLS, lw=0.5, zorder=0)
    ax.set_xlabel("Oncogenes - tumor suppressors\nper segment")
    ax.set_ylabel("Mean copy number")
    ax.legend(fontsize=4.8, loc="upper left")
    return out


def panel_d(ax):
    rows = SM.load("R5")
    if not rows:
        return pending(ax)
    df = pd.DataFrame([dict(rate=float(r["task"].split("_")[0][3:]), frac=r["wgd_frac"]) for r in rows])
    prev = df.assign(wgd=df.frac > 0.5).groupby("rate").wgd.mean()
    ax.axhspan(0.30, 0.50, color=S.PASTEL["yellow"], lw=0, zorder=0)
    ax.text(prev.index.max(), 0.40, "PCAWG", fontsize=4.8, ha="right", va="center", color=S.GREY)
    ax.plot(prev.index, prev.values, marker="o", ms=2.5, color=S.SOURCE["iscc"], lw=0.8)
    ax.set_xlabel("WGD rate per mutating division")
    ax.set_ylabel("Tumours with WGD")
    ax.set_ylim(0, 1)
    return dict(prevalence=prev.round(2).to_dict(), n_per_rate=int(df.groupby("rate").size().min()))


def panel_pcawg(ax):
    """(d) The fit to the PCAWG breast-cancer arm landscape (real-genome mode, ploidy-relative arm
    model): per-arm gain and loss frequencies re-simulated with the fitted selection against the
    observed ones. Reads the JSON that ``validation/validate_inference_realgenome.py --out-json``
    writes into the derived cache as ``fig2/pcawg_fit.json``."""
    p = os.path.join(C.DERIVED, "fig2", "pcawg_fit.json")
    if not os.path.exists(p):
        pending(ax, "PCAWG arm-landscape fit:\npending rerun under the\nploidy-relative arm model")
        return "pending"
    with open(p) as f:
        r = json.load(f)
    for kind, col in (("gain", "#A81B20"), ("loss", "#2F479E")):    # the ends of style.DIVERGING
        obs = np.array([np.nan if v is None else v for v in r[f"observed_{kind}"]], dtype=float)
        fit = np.array([np.nan if v is None else v for v in r[f"fit_{kind}"]], dtype=float)
        ax.scatter(obs, fit, s=4, color=col, edgecolors="none", label=kind)
    vals = [v for k in ("observed_gain", "observed_loss", "fit_gain", "fit_loss") for v in r[k] if v is not None]
    lim = max(0.05, max(vals) * 1.05)
    ax.plot([0, lim], [0, lim], color=S.INK, lw=0.5, ls=":")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("PCAWG arm frequency")
    ax.set_ylabel("Re-simulated frequency")
    st = r["stats"]
    ax.text(0.04, 0.96, f"r = {st['r_fit']:.2f}\nCharm r = {st['r_charm']:.2f}", transform=ax.transAxes,
            fontsize=4.8, va="top")
    ax.legend(fontsize=4.5, loc="lower right", frameon=False)
    return dict(r_fit=round(st["r_fit"], 3), rmse=round(st["rmse"], 3), r_charm=round(st["r_charm"], 3),
                rho_charm=round(st["rho_charm"], 3), r_content=round(st["r_content"], 3),
                n_ref=r["n_ref"], cohort_size=r["cohort_size"], target_size=r["target_size"])


# --- f, g, h: the data front's fit-and-resimulate loops ---------------------------------------------
# Each is computed in its own process (``--part f|g|h``) and cached as small JSON (histograms, binned
# curves, statistics), so no process holds a real reference and a tumour at once for longer than it
# must, and the render step reads only the caches.
import json  # noqa: E402
import subprocess  # noqa: E402
import scipy.sparse as sp  # noqa: E402

PART_CACHE = os.path.join(C.DERIVED, "fig2")


def _count_stats(X):
    """Library sizes and binned mean-variance and dropout curves of a sparse cells x genes matrix."""
    X = X.tocsr().astype(np.float64)
    n = X.shape[0]
    lib = np.asarray(X.sum(1)).ravel()
    m = np.asarray(X.mean(0)).ravel()
    v = np.asarray(X.multiply(X).mean(0)).ravel() - m ** 2
    z = 1.0 - np.asarray((X > 0).sum(0)).ravel() / n
    ok = m > 0
    lm = np.log10(m[ok])
    b = np.linspace(lm.min(), lm.max(), 20)
    idx = np.digitize(lm, b)
    keep = [i for i in range(1, 20) if (idx == i).sum() > 5]
    return dict(loglib=np.log10(lib + 1).tolist(),
                curve_m=[float(lm[idx == i].mean()) for i in keep],
                curve_v=[float(np.median(np.log10(v[ok][idx == i] + 1e-9))) for i in keep],
                curve_z=[float(np.median(z[ok][idx == i])) for i in keep],
                cv_lib=float(lib.std() / lib.mean()))


def compute_f():
    import anndata as ad
    from iscc.data import estimate_rna, scRNA
    real = ad.read_h5ad(os.path.join(REPO, "data", "pbmc3k_raw.h5ad"))
    est = estimate_rna(real)
    real_stats = _count_stats(real.X if sp.issparse(real.X) else sp.csr_matrix(real.X))
    del real
    t = REG.load("R0", "seed1")
    diss = RR.resect(t).dissociate(max_cells=3000)
    del t
    cells = list(diss["cell_exp"].index)
    sim = scRNA(n_cells=len(cells), seed=5, **est.scrna_kwargs()).run(diss, cell_subset=cells)
    del diss
    return dict(real=real_stats, iscc=_count_stats(sp.csr_matrix(sim.observed_counts.values)))


def _cv_by_state(coverage, cn, min_frac=0.05, min_n=20):
    """Within-cell coverage CV per copy-number state, median over cells (as estimate_dna fits it).

    A state counts in a cell only if it holds at least ``min_frac`` of the cell's loci: a sliver
    state (DLP+ one-copy calls hold ~30 of 5,787 bins per cell) is mostly calling error at segment
    edges, not sequencing spread."""
    coverage = np.atleast_2d(np.asarray(coverage, float))
    cn = np.atleast_2d(np.rint(np.asarray(cn, float))).astype(int)
    per = {}
    for X, Cn in zip(coverage, cn):
        for lvl in np.unique(Cn):
            m = Cn == lvl
            c = X[m & (X > 0)]
            if m.sum() >= max(min_n, min_frac * len(Cn)) and c.size >= min_n:
                per.setdefault(int(lvl), []).append(float(c.std() / c.mean()))
    return {lvl: float(np.median(v)) for lvl, v in per.items()}


def compute_g():
    from iscc.data import bulkDNA, estimate_dna, scDNA
    sys.path.insert(0, os.path.join(REPO, "validation"))
    import validate_dna as VD
    t = REG.load("R0", "seed1")
    diss = RR.resect(t).dissociate(max_cells=1500)
    gid = diss["cell_type"]["cell_id"].astype(str)
    cancer = [c for c, g in gid.items() if g in t.genotypes and t._is_cancer(g)][:200]
    out, ado = {}, {}
    for name, f, mod in (("HG002", "dna_ref_hg002_bulk_wgs", "bulk"),
                         ("Tapestri", "dna_ref_tapestri_sc_panel", "sc"), ("DLP+", "dna_ref_dlp_sc_wgs", "sc")):
        z = np.load(os.path.join(REPO, "validation", "data", f + ".npz"), allow_pickle=True)
        # breadth and depth model are chosen, not estimated (estimate_dna defaults to WGS), so they
        # come from the reference: Tapestri is a targeted panel
        kw = dict(modality=mod, breadth=str(z["breadth"]), depth_model=str(z["depth_model"]))
        for k in ("het_mask", "variant_mask"):
            if k in z.files:
                kw[k] = z[k]
        est = estimate_dna(z["coverage"], z["alt"], z["cn"], **kw)
        if mod == "bulk":
            sim = bulkDNA(breadth=est.breadth, depth_model=est.depth_model, seed=11,
                          **est.dna_kwargs()).run(diss)
            cov_s = sim.observed_data["coverage"].values[None, :].astype(float)
            cn_s = np.rint(sim.observed_data["true_cn"].values)[None, :]
        else:
            # a panel observes as many loci as the real panel has amplicons (iscc's default panel is
            # 30 driver loci, too few to measure a spread per copy-number state)
            tg = (z["coverage"].shape[1] / t.n_genes) if est.breadth == "panel" else "all"
            sim = scDNA(n_cells=len(cancer), breadth=est.breadth, depth_model=est.depth_model, seed=21,
                        target_genes=tg, **est.dna_kwargs()).run(diss, cell_subset=cancer)
            cov_s = sim.coverage.values.astype(float)
            cn_s = sim.true_cn.values
        obs_cv = _cv_by_state(np.atleast_2d(z["coverage"]), np.atleast_2d(z["cn"]))
        sim_cv = _cv_by_state(cov_s, cn_s)
        if mod == "sc" and "het_mask" in z.files and np.asarray(z["het_mask"]).any():
            # allelic dropout at heterozygous sites, the single-cell allele layer's headline
            baf_o = VD._baf_at_hets(z["alt"], z["coverage"], z["het_mask"])
            baf_s = VD._baf_at_hets(sim.alt_counts.values, sim.coverage.values, VD._het_obs_assay_sc(sim))
            ado[name] = [float(VD._ado_frac(baf_o)), float(VD._ado_frac(baf_s))]
        out[name] = {str(int(l)): [float(obs_cv[l]), float(sim_cv[l])] for l in sorted(set(obs_cv) & set(sim_cv))}
    out["_ado"] = ado
    return out


def compute_h():
    sys.path.insert(0, os.path.join(REPO, "validation"))
    import validate_visium as VV
    from iscc.data import Visium, estimate_visium, morans_i
    cwd = os.getcwd()
    os.chdir(REPO)                               # scanpy's dataset cache lives in ./data
    try:
        real = VV.load_real_visium()
    finally:
        os.chdir(cwd)
    if real is None:
        return None
    coords = np.asarray(real.obsm["spatial"], dtype=float) / VV._median_nn(real.obsm["spatial"])
    tot = np.asarray(real.X.sum(1)).ravel()
    occ = tot > 0
    mi_o = float(morans_i(tot[occ], coords[occ]))
    est = estimate_visium(real, count_model="dm", coords=coords)
    toto = np.log10(tot[occ]).tolist()
    del real
    t = REG.load("R0", "seed1")
    sec = RR.resect(t).slice(depth_frac=0.5, max_cells=6000)
    kw = {**est.visium_kwargs(), "spot_pitch": 2.0, "spot_radius": 1.0}
    if kw.get("field_lengthscale") is not None:
        kw["field_lengthscale"] = 2.0 * kw["field_lengthscale"]      # spot units -> demes (pitch 2)
    sim = Visium(seed=11, count_model="dm", **kw).run(sec, grid_side=t.grid_size)
    zs, cs, tots = VV._log_library_residual(sim)
    return dict(toto=toto, tots=np.log10(tots).tolist(), moran_real=mi_o, moran_iscc=float(morans_i(tots, cs)))


def _cached(part):
    p = os.path.join(PART_CACHE, f"{part}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def render_f(fig, x, y):
    d = _cached("f")
    axes = [S.ax_mm(fig, x + i * 27, y, 21, 21) for i in range(3)]
    if d is None:
        for ax in axes:
            pending(ax)
        return None
    lr, ls = np.asarray(d["real"]["loglib"]), np.asarray(d["iscc"]["loglib"])
    bins = np.linspace(min(lr.min(), ls.min()), max(lr.max(), ls.max()), 30)
    axes[0].hist(lr, bins=bins, density=True, color=S.SOURCE["real"], alpha=0.6, label="PBMC3k")
    axes[0].hist(ls, bins=bins, density=True, color=S.SOURCE["iscc"], alpha=0.6, label="iscc")
    axes[0].set_xlabel("log$_{10}$ UMIs per cell")
    axes[0].set_ylabel("Density")
    axes[0].legend(fontsize=4.5)
    for key, col in (("real", S.SOURCE["real"]), ("iscc", S.SOURCE["iscc"])):
        axes[1].plot(d[key]["curve_m"], d[key]["curve_v"], color=col, lw=0.9)
        axes[2].plot(d[key]["curve_m"], d[key]["curve_z"], color=col, lw=0.9)
    xs = np.linspace(-3, 2, 10)
    axes[1].plot(xs, xs, color=S.BG_CELLS, lw=0.5, ls=":", zorder=0)
    axes[1].set_xlabel("log$_{10}$ mean")
    axes[1].set_ylabel("log$_{10}$ variance")
    S.stat(axes[1], "dotted: Poisson", "lower right")
    axes[2].set_xlabel("log$_{10}$ mean")
    axes[2].set_ylabel("Zero fraction")
    return dict(cv_lib_real=d["real"]["cv_lib"], cv_lib_iscc=d["iscc"]["cv_lib"])


def render_g(ax):
    d = _cached("g")
    if d is None:
        return pending(ax)
    ado = d.pop("_ado", {})
    for i, (name, levels) in enumerate(d.items()):
        for j, (lvl, (o, s_)) in enumerate(levels.items()):
            ax.plot([i - 0.15, i + 0.15], [o, s_], color=S.GREY, lw=0.4)
            ax.scatter([i - 0.15], [o], s=8, color=S.SOURCE["real"], zorder=3, label="real" if i == j == 0 else None)
            ax.scatter([i + 0.15], [s_], s=8, color=S.SOURCE["iscc"], zorder=3,
                       label="iscc, fitted" if i == j == 0 else None)
    ax.set_xticks(range(len(d)))
    ax.set_xticklabels(list(d))
    ax.set_ylabel("Coverage CV")
    ax.legend(fontsize=4.8, loc="upper left")
    if "Tapestri" in ado:
        o, s_ = ado["Tapestri"]
        S.stat(ax, f"Tapestri allelic dropout:\nreal {o:.2f}, iscc {s_:.2f}", "lower left")
    d["ado"] = ado
    return d


def render_h(fig, x, y):
    d = _cached("h")
    ax1, ax2 = S.ax_mm(fig, x, y, 21, 21), S.ax_mm(fig, x + 26, y, 9, 21)
    if d is None:
        pending(ax1)
        pending(ax2, "")
        return None
    lo, ls = np.asarray(d["toto"]), np.asarray(d["tots"])
    bins = np.linspace(min(lo.min(), ls.min()), max(lo.max(), ls.max()), 30)
    ax1.hist(lo, bins=bins, density=True, color=S.SOURCE["real"], alpha=0.6, label="real section")
    ax1.hist(ls, bins=bins, density=True, color=S.SOURCE["iscc"], alpha=0.6, label="iscc")
    ax1.set_xlabel("log$_{10}$ UMIs per spot")
    ax1.set_ylabel("Density")
    ax1.legend(fontsize=4.5)
    ax2.bar([0, 1], [d["moran_real"], d["moran_iscc"]], color=[S.SOURCE["real"], S.SOURCE["iscc"]], width=0.7)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["real", "iscc"], rotation=90)
    ax2.set_ylabel("Moran's I, spot depth")
    ax2.set_ylim(0, 1)
    return dict(moran_real=d["moran_real"], moran_iscc=d["moran_iscc"])


def render():
    S.setup()
    fig = S.figure("double", 112)
    out = {}
    S.text_mm(fig, 2, 2.5, "The evolution front's tests", fontsize=7, va="top", fontweight="bold")
    # No site-frequency-spectrum (1/f) panel: passengers live on the 6,000-gene genome, which
    # saturates at tumour scale (nearly every site is hit in some lineage), so the spectrum of sites
    # is not a spectrum of mutations and the infinite-sites 1/f test does not apply (issues, B13).
    xs = [12, 58, 104, 150]
    for i, (L, fn) in enumerate(zip("abcd", (panel_a, panel_c, panel_d, None))):
        S.panel_letter(fig, L, xs[i] - 9, 8)
        ax = S.ax_mm(fig, xs[i], 11, 30, 27 if i < 3 else 22)
        if fn is None:
            out[L] = panel_pcawg(ax)
        else:
            out[L] = fn(ax)
    S.text_mm(fig, 2, 57.5, "The data front's tests: fit to a real reference, re-simulate on the tumour",
              fontsize=7, va="top", fontweight="bold")
    S.panel_letter(fig, "e", 2, 63)
    out["e"] = render_f(fig, 10, 67)
    S.panel_letter(fig, "f", 92, 63)
    out["f"] = render_g(S.ax_mm(fig, 100, 67, 28, 21))
    S.panel_letter(fig, "g", 137, 63)
    out["g"] = render_h(fig, 145, 67)
    path = S.save(fig, "fig2_realism")
    print("wrote", path)
    for k, v in out.items():
        print(k, v)


def main():
    import argparse
    from memguard import wait_for_memory
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["f", "g", "h"], help="compute one data-front panel and cache it")
    ap.add_argument("--all", action="store_true", help="compute f, g, h (one process each), then render")
    a = ap.parse_args()
    if a.part:
        res = {"f": compute_f, "g": compute_g, "h": compute_h}[a.part]()
        os.makedirs(PART_CACHE, exist_ok=True)
        if res is not None:
            json.dump(res, open(os.path.join(PART_CACHE, f"{a.part}.json"), "w"))
        print(f"cached {a.part}: {res is not None}")
        return
    if a.all:
        for part in "fgh":
            if _cached(part) is None:
                wait_for_memory(min_avail_gb=2.5, label=f"fig2/{part}")
                subprocess.run([sys.executable, os.path.abspath(__file__), "--part", part], check=False)
    render()


if __name__ == "__main__":
    main()
