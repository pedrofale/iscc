"""Figure 2: realistic by each front's own standard.

Top row, the evolution front's tests (registry tumours):
  (a) evolutionary-mode space (Noble's n, D) against real multi-region tumours, at the full census and
      at the 2% detection limit real data has (combinations below it pooled, not dropped);
  (b) the neutral 1/f site-frequency spectrum (R2 neutral, passengers tracked exactly);
  (c) copy number follows oncogenic content under selection, not under neutrality (R2);
  (d) whole-genome-duplication prevalence across rates against the PCAWG range (R5);
  (e) the fit to the PCAWG breast-cancer arm landscape: pending its rerun under the corrected model.
Bottom row, the data front's tests, each fit on a real reference and re-simulated on the showcase
tumour (R0):
  (f) scRNA against PBMC3k: library size, mean-variance, dropout;
  (g) DNA against GIAB HG002 (bulk), Tapestri and DLP+ (single cell): coverage dispersion by copy number;
  (h) Visium against a real breast-cancer section: per-spot depth and its spatial autocorrelation.

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


# --- a: evolutionary-mode space -------------------------------------------------------------------
def mode_point(t, detection=0.0):
    from iscc.inference.indices import driver_combination_counts, inverse_simpson
    combos = driver_combination_counts(t)
    total = sum(combos.values())
    if not total:
        return np.nan, np.nan
    n = sum(len(k) * c for k, c in combos.items()) / total
    if detection > 0:
        kept = {k: c for k, c in combos.items() if c / total >= detection}
        pooled = total - sum(kept.values())
        counts = list(kept.values()) + ([pooled] if pooled else [])
    else:
        counts = list(combos.values())
    return n, inverse_simpson(counts)


def panel_a(ax):
    from scipy.spatial import Delaunay
    emp = pd.read_csv(os.path.join(REPO, "validation", "data", "noble_empirical_indices.csv"))
    rows = []
    for run in ("R1", "R2", "R3", "R4"):
        for task in REG.tasks(run):
            if have(run, task):
                t = REG.load(run, task)
                rows.append((*mode_point(t, 0.0), *mode_point(t, DETECTION)))
    if not rows:
        return pending(ax)
    sim = pd.DataFrame(rows, columns=["n0", "D0", "n2", "D2"]).dropna()
    e = emp[["n", "D"]].to_numpy()
    cov = {}
    for k, (nc, dc) in {"census": ("n0", "D0"), "2%": ("n2", "D2")}.items():
        try:
            cov[k] = float((Delaunay(sim[[nc, dc]].to_numpy()).find_simplex(e) >= 0).mean())
        except Exception:
            cov[k] = float("nan")
    ax.scatter(sim.n0, sim.D0, s=4, facecolors="none", edgecolors=S.SOURCE["iscc"], lw=0.3, alpha=0.5,
               label=f"iscc, full census ({cov['census']:.0%})")
    ax.scatter(sim.n2, sim.D2, s=4, color=S.SOURCE["iscc"], alpha=0.8,
               label=f"iscc, 2% detection ({cov['2%']:.0%})")
    ax.scatter(emp.n, emp.D, s=6, color=S.SOURCE["real"], edgecolors=S.INK, lw=0.3, label="real tumours")
    ax.set_yscale("log")
    ax.set_xlabel("Drivers per cell, n")
    ax.set_ylabel("Clonal diversity, D")
    ax.legend(fontsize=4.5, loc="upper left", title="real tumours inside\nthe iscc hull",
              title_fontsize=4.5)
    return dict(n_sim=len(sim), coverage=cov, D_median_2pct=float(sim.D2.median()),
                D_median_real=float(emp.D.median()))


# --- b: neutral spectrum --------------------------------------------------------------------------
def panel_b(ax):
    from iscc.validation import neutral_sfs_rsq, population_vaf, site_frequency_spectrum
    tasks = [t for t in REG.tasks("R2") if t.startswith("neutral") and have("R2", t)]
    if not tasks:
        return pending(ax)
    rsq, curves = [], []
    for task in tasks:
        t = REG.load("R2", task)
        if t.coarsen_passengers:
            return pending(ax, "pending: R2 must be regrown with\npassengers tracked exactly")
        vaf = population_vaf(t)
        r, _ = neutral_sfs_rsq(vaf, 0.12, 0.24, 12)
        rsq.append(r)
        curves.append(site_frequency_spectrum(vaf[vaf > 0], 0.12, 0.24, 12))
    grid, M = curves[0]
    x = 1.0 / grid
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, M, rcond=None)
    ax.plot(x, M, "o", ms=2.5, color=S.SOURCE["iscc"])
    ax.plot(x, A @ coef, color=S.INK, lw=0.8)
    ax.set_xlabel("1 / f")
    ax.set_ylabel("Cumulative mutations, M(f)")
    S.stat(ax, f"R$^2$ = {np.mean(rsq):.2f} ± {np.std(rsq):.2f}\n(n = {len(rsq)})", "lower right")
    return dict(rsq_mean=float(np.mean(rsq)), rsq_sd=float(np.std(rsq)), n=len(rsq))


# --- c: copy number against oncogenic content -----------------------------------------------------
def segment_table(t):
    seg = t.selection.n_segments if hasattr(t.selection, "n_segments") else t.n_segments
    size = t.n_genes // seg
    onc = np.bincount(np.asarray(t.selection.get_oncogenes(), dtype=int) // size, minlength=seg)
    tsg = np.bincount(np.asarray(t.selection.get_tsgs(), dtype=int) // size, minlength=seg)
    counts = C.live_cancer_counts(t)
    w = np.array(list(counts.values()), dtype=float)
    cn = np.array([t.genotypes[g].genome_summary["seg_cns"] for g in counts], dtype=float)
    return pd.DataFrame(dict(net=onc - tsg, cn=(cn * w[:, None]).sum(0) / w.sum()))


def panel_c(ax):
    out = {}
    for name, col in (("neutral", S.SOURCE["real"]), ("selected", S.SOURCE["iscc"])):
        tasks = [t for t in REG.tasks("R2") if t.startswith(name) and have("R2", t)]
        if not tasks:
            return pending(ax)
        df = pd.concat([segment_table(REG.load("R2", t)) for t in tasks])
        r = float(np.corrcoef(df.net, df.cn)[0, 1])
        ax.scatter(df.net + np.random.default_rng(0).uniform(-0.2, 0.2, len(df)), df.cn, s=2, color=col,
                   alpha=0.7, label=f"{name} (r = {r:.2f})")
        out[name] = r
    ax.axhline(2, color=S.BG_CELLS, lw=0.5, zorder=0)
    ax.set_xlabel("Oncogenes - tumor suppressors\nper segment")
    ax.set_ylabel("Mean copy number")
    ax.legend(fontsize=4.8, loc="upper left")
    return out


# --- d: whole-genome duplication ------------------------------------------------------------------
def panel_d(ax):
    rows = []
    for task in REG.tasks("R5"):
        if have("R5", task):
            t = REG.load("R5", task)
            counts = C.live_cancer_counts(t)
            tot = sum(counts.values())
            frac = sum(c for g, c in counts.items() if getattr(t.genotypes[g], "is_wgd", False)) / max(tot, 1)
            rows.append((float(task.split("_")[0][3:]), frac))
    if not rows:
        return pending(ax)
    df = pd.DataFrame(rows, columns=["rate", "frac"])
    prev = df.assign(wgd=df.frac > 0.5).groupby("rate").wgd.mean()
    ax.axhspan(0.30, 0.50, color=S.PASTEL["yellow"], lw=0, zorder=0)
    ax.text(prev.index.max(), 0.40, "PCAWG", fontsize=4.8, ha="right", va="center", color=S.GREY)
    ax.plot(prev.index, prev.values, marker="o", ms=2.5, color=S.SOURCE["iscc"], lw=0.8)
    ax.set_xlabel("WGD rate per mutating division")
    ax.set_ylabel("Tumours with WGD")
    ax.set_ylim(0, 1)
    return dict(prevalence=prev.round(2).to_dict(), n_per_rate=int(df.groupby("rate").size().min()))


# --- f: scRNA fit loop ----------------------------------------------------------------------------
def panel_f(fig, x, y):
    import anndata as ad
    from iscc.data import estimate_rna, scRNA
    real = ad.read_h5ad(os.path.join(REPO, "data", "pbmc3k_raw.h5ad"))
    est = estimate_rna(real)
    t = REG.load("R0", "seed1")
    diss = RR.resect(t).dissociate(max_cells=3000)
    cells = list(diss["cell_exp"].index)
    sim = scRNA(n_cells=len(cells), seed=5, **est.scrna_kwargs()).run(diss, cell_subset=cells)
    R = np.asarray(real.X.todense() if hasattr(real.X, "todense") else real.X, dtype=float)
    Ssim = sim.observed_counts.values.astype(float)
    axes = [S.ax_mm(fig, x + i * 27, y, 21, 21) for i in range(3)]
    lr, ls = np.log10(R.sum(1) + 1), np.log10(Ssim.sum(1) + 1)
    bins = np.linspace(min(lr.min(), ls.min()), max(lr.max(), ls.max()), 30)
    axes[0].hist(lr, bins=bins, density=True, color=S.SOURCE["real"], alpha=0.6, label="PBMC3k")
    axes[0].hist(ls, bins=bins, density=True, color=S.SOURCE["iscc"], alpha=0.6, label="iscc")
    axes[0].set_xlabel("log$_{10}$ UMIs per cell")
    axes[0].set_ylabel("Density")
    axes[0].legend(fontsize=4.5)
    for X, col in ((R, S.SOURCE["real"]), (Ssim, S.SOURCE["iscc"])):
        m, v = X.mean(0), X.var(0)
        ok = m > 0
        lm = np.log10(m[ok])
        b = np.linspace(lm.min(), lm.max(), 20)
        idx = np.digitize(lm, b)
        mm = [lm[idx == i].mean() for i in range(1, 20) if (idx == i).sum() > 5]
        vv = [np.median(np.log10(v[ok][idx == i] + 1e-9)) for i in range(1, 20) if (idx == i).sum() > 5]
        zz = [np.median((X[:, ok][:, idx == i] == 0).mean(0)) for i in range(1, 20) if (idx == i).sum() > 5]
        axes[1].plot(mm, vv, color=col, lw=0.9)
        axes[2].plot(mm, zz, color=col, lw=0.9)
    xs = np.linspace(-3, 2, 10)
    axes[1].plot(xs, xs, color=S.BG_CELLS, lw=0.5, ls=":", zorder=0)
    axes[1].set_xlabel("log$_{10}$ mean")
    axes[1].set_ylabel("log$_{10}$ variance")
    S.stat(axes[1], "dotted: Poisson", "lower right")
    axes[2].set_xlabel("log$_{10}$ mean")
    axes[2].set_ylabel("Zero fraction")
    return dict(cv_lib_real=float(np.std(R.sum(1)) / np.mean(R.sum(1))),
                cv_lib_iscc=float(np.std(Ssim.sum(1)) / np.mean(Ssim.sum(1))))


# --- g: DNA fit loop ------------------------------------------------------------------------------
def panel_g(ax):
    from iscc.data import bulkDNA, estimate_dna, scDNA
    sys.path.insert(0, os.path.join(REPO, "validation"))
    import validate_dna as VD
    t = REG.load("R0", "seed1")
    diss = RR.resect(t).dissociate(max_cells=1500)
    gid = diss["cell_type"]["cell_id"].astype(str)
    cancer = [c for c, g in gid.items() if g in t.genotypes and t._is_cancer(g)][:200]
    refs = [("HG002", "dna_ref_hg002_bulk_wgs", "bulk"), ("Tapestri", "dna_ref_tapestri_sc_panel", "sc"),
            ("DLP+", "dna_ref_dlp_sc_wgs", "sc")]
    out = {}
    for i, (name, f, mod) in enumerate(refs):
        z = np.load(os.path.join(REPO, "validation", "data", f + ".npz"), allow_pickle=True)
        kw = dict(modality=mod)
        if "het_mask" in z.files:
            kw["het_mask"] = z["het_mask"]
        if "variant_mask" in z.files:
            kw["variant_mask"] = z["variant_mask"]
        est = estimate_dna(z["coverage"], z["alt"], z["cn"], **kw)
        if mod == "bulk":
            sim = bulkDNA(breadth=est.breadth, depth_model=est.depth_model, seed=11,
                          **est.dna_kwargs()).run(diss)
            cov_s = sim.observed_data["coverage"].values[None, :].astype(float)
            cn_s = np.rint(sim.observed_data["true_cn"].values)[None, :]
        else:
            sim = scDNA(n_cells=len(cancer), breadth=est.breadth, depth_model=est.depth_model, seed=21,
                        **est.dna_kwargs()).run(diss, cell_subset=cancer)
            cov_s = sim.coverage.values.astype(float)
            cn_s = _sim_cn(t, diss, sim, cancer)
        obs_cv = VD._cov_cv_by_cn(np.atleast_2d(z["coverage"]), np.atleast_2d(z["cn"]))
        sim_cv = VD._cov_cv_by_cn(cov_s, cn_s)
        common = sorted(set(obs_cv) & set(sim_cv))
        for j, lvl in enumerate(common):
            ax.plot([i - 0.15, i + 0.15], [obs_cv[lvl], sim_cv[lvl]], color=S.GREY, lw=0.4)
            ax.scatter([i - 0.15], [obs_cv[lvl]], s=8, color=S.SOURCE["real"], zorder=3,
                       label="real" if i == j == 0 else None)
            ax.scatter([i + 0.15], [sim_cv[lvl]], s=8, color=S.SOURCE["iscc"], zorder=3,
                       label="iscc, fitted" if i == j == 0 else None)
        out[name] = {int(l): (round(obs_cv[l], 3), round(sim_cv[l], 3)) for l in common}
    ax.set_xticks(range(len(refs)))
    ax.set_xticklabels([r[0] for r in refs])
    ax.set_ylabel("Coverage CV, per copy-number state")
    ax.legend(fontsize=4.8)
    return out


def _sim_cn(t, diss, sim, cells):
    """True copy number aligned to a simulated DNA object's coverage (cells/pool x loci)."""
    gid = diss["cell_type"]["cell_id"].astype(str)
    genes = list(sim.coverage.columns)
    gidx = np.array([int(g.split("_")[1]) * (t.n_genes // t.n_segments) + int(g.split("_")[2]) for g in genes])
    seg_size = t.n_genes // t.n_segments

    def cn_of(c):
        g = gid[c]
        sc = t.genotypes[g].genome_summary["seg_cns"] if g in t.genotypes else [2] * t.n_segments
        return np.asarray(sc)[gidx // seg_size]
    if cells is None:                                   # bulk: the pooled, count-weighted mean
        allc = list(diss["cell_exp"].index)
        return np.rint(np.mean([cn_of(c) for c in allc], axis=0))[None, :]
    return np.stack([cn_of(c) for c in sim.coverage.index])


# --- h: Visium fit loop ---------------------------------------------------------------------------
def panel_h(fig, x, y):
    sys.path.insert(0, os.path.join(REPO, "validation"))
    import validate_visium as VV
    from iscc.data import Visium, estimate_visium
    from iscc.validation import morans_i
    cwd = os.getcwd()
    os.chdir(REPO)                               # scanpy's dataset cache lives in ./data
    try:
        real = VV.load_real_visium()
    finally:
        os.chdir(cwd)
    ax1, ax2 = S.ax_mm(fig, x, y, 21, 21), S.ax_mm(fig, x + 26, y, 9, 21)
    if real is None:
        pending(ax1, "real Visium section unavailable")
        return None
    counts = VV._to_dense(real)
    coords = np.asarray(real.obsm["spatial"], dtype=float) / VV._median_nn(real.obsm["spatial"])
    est = estimate_visium(real, count_model="dm", coords=coords)
    t = REG.load("R0", "seed1")
    sec = RR.resect(t).slice(depth_frac=0.5, max_cells=6000)
    kw = {**est.visium_kwargs(), "spot_pitch": 2.0, "spot_radius": 1.0}
    if "field_lengthscale" in kw and kw["field_lengthscale"] is not None:
        kw["field_lengthscale"] = 2.0 * kw["field_lengthscale"]      # spot units -> demes (pitch 2)
    sim = Visium(seed=11, count_model="dm", **kw).run(sec, grid_side=t.grid_size)
    zo, co, toto = VV._lib_residual_raw(counts, coords)
    zs, cs, tots = VV._log_library_residual(sim)
    mi_o, mi_s = morans_i(toto, co), morans_i(tots, cs)
    lo, ls = np.log10(toto), np.log10(tots)
    bins = np.linspace(min(lo.min(), ls.min()), max(lo.max(), ls.max()), 30)
    ax1.hist(lo, bins=bins, density=True, color=S.SOURCE["real"], alpha=0.6, label="real section")
    ax1.hist(ls, bins=bins, density=True, color=S.SOURCE["iscc"], alpha=0.6, label="iscc")
    ax1.set_xlabel("log$_{10}$ UMIs per spot")
    ax1.set_ylabel("Density")
    ax1.legend(fontsize=4.5)
    ax2.bar([0, 1], [mi_o, mi_s], color=[S.SOURCE["real"], S.SOURCE["iscc"]], width=0.7)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["real", "iscc"], rotation=90)
    ax2.set_ylabel("Moran's I, spot depth")
    ax2.set_ylim(0, 1)
    return dict(moran_real=float(mi_o), moran_iscc=float(mi_s))


def main():
    S.setup()
    fig = S.figure("double", 112)
    out = {}
    S.text_mm(fig, 2, 2.5, "The evolution front's tests", fontsize=7, va="top", fontweight="bold")
    xs = [10, 47, 84, 121, 158]
    for i, (L, fn) in enumerate(zip("abcde", (panel_a, panel_b, panel_c, panel_d, None))):
        S.panel_letter(fig, L, xs[i] - 8, 8)
        ax = S.ax_mm(fig, xs[i], 11, 27, 27 if i < 4 else 22)
        if fn is None:
            pending(ax, "PCAWG arm-landscape fit:\npending rerun under the\nploidy-relative arm model")
            out["e"] = "pending"
        else:
            out[L] = fn(ax)
    S.text_mm(fig, 2, 57.5, "The data front's tests: fit to a real reference, re-simulate on the tumour",
              fontsize=7, va="top", fontweight="bold")
    S.panel_letter(fig, "f", 2, 63)
    out["f"] = panel_f(fig, 10, 67)
    S.panel_letter(fig, "g", 92, 63)
    ax = S.ax_mm(fig, 100, 67, 28, 21)
    out["g"] = panel_g(ax)
    S.panel_letter(fig, "h", 137, 63)
    out["h"] = panel_h(fig, 145, 67)
    path = S.save(fig, "fig2_realism")
    print("wrote", path)
    for k, v in out.items():
        print(k, v)


if __name__ == "__main__":
    main()
