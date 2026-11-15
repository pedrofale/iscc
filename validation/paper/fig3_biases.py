"""Figure 3: every measurement step adds a bias that is fitted and tunable.

One row per modality. On the left, the fit to a real reference against that reference; on the right,
a sweep or a preset showing the bias can be tuned, with the real value inside the range where a real
value exists.

  (a) sampling: cell-type composition before and after dissociation for three settings of the recovery
      bias, the dissociation-stress signature in the recovered cells, and the clone fractions of the
      whole tumour against a section, a dissociated sample and four multi-region blocks;
  (b) WGS: per-locus coverage of GIAB HG002 against fitted iscc, the coverage spread across the depth
      model's concentration, and iscc's coverage against GC content across the GC-bias strength;
  (c) scDNA: coverage spread per copy-number state (DLP+, Tapestri) and Tapestri allelic dropout, fitted
      against real; the spread across amplification concentration and the dropout across ``ado_rate``;
  (d) scRNA: PBMC3k against fitted iscc (library size, mean-variance, zero fraction); the 10x and
      Smart-seq3 presets and a sweep of the dropout midpoint;
  (e) Visium: per-spot depth and its autocorrelation against a real breast-cancer section; the counts
      outside the tissue against distance to it, real against iscc across the spot-swapping rate.

The left halves of c-e are Fig 2's data-front loops (``fig2_realism.compute_g/f/h``), reused as they
are. Every part is computed in its own process (``--part a|b|c|d|e``) on the stored showcase tumour
(``registry.load("R0", "seed1")`` under ``ISCC_PAPER_OUT`` / ``ISCC_PAPER_TIER``) and cached as small
JSON under ``$ISCC_FIG3_CACHE`` (default ``<derived>/fig3``); the render step reads only the caches.
To re-render from another stored tumour, point ``ISCC_PAPER_OUT`` at it and recompute every part.

Real references: ``validation/data/dna_ref_*.npz``; ``data/pbmc3k_raw.h5ad`` and
``data/V1_Breast_Cancer_Block_A_Section_1`` (scanpy's cache, as Fig 2); the section's raw Space Ranger
matrix (all 4,992 spots) in ``$ISCC_VISIUM_RAW_DIR`` (default ``data/visium_breast_A1``).

Run from the iscc repository root:
    python validation/paper/fig3_biases.py --part a      # ... b, c, d, e
    python validation/paper/fig3_biases.py               # render from the caches
"""
import json
import os
import subprocess
import sys

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import common as C  # noqa: E402
import registry as REG  # noqa: E402
import regime as RG  # noqa: E402
import style as S  # noqa: E402
import realistic_regime as RR  # noqa: E402

REPO = os.path.dirname(os.path.dirname(HERE))
CACHE = os.environ.get("ISCC_FIG3_CACHE") or os.path.join(C.DERIVED, "fig3")
TUMOUR = ("R0", "seed1")
DNA_REF = {name: os.path.join(REPO, "validation", "data", f + ".npz") for name, f in (
    ("HG002", "dna_ref_hg002_bulk_wgs"), ("Tapestri", "dna_ref_tapestri_sc_panel"), ("DLP+", "dna_ref_dlp_sc_wgs"))}
SWEEP = S.CLONE_FAMILIES[0][::-1]          # light to dark iscc blues, for an ordered setting


def _load():
    return REG.load(*TUMOUR)


def _f2():
    import fig2_realism as F2
    return F2


def _cv(x):
    x = np.asarray(x, float)
    x = x[x > 0]
    return float(x.std() / x.mean()) if x.size else float("nan")


# --- a. sampling ---------------------------------------------------------------------------------
RECOVERY = {          # the three settings of the cell-type recovery bias
    "none": {"cancer": 1.0, "epithelial": 1.0, "host": 1.0, "stromal": 1.0, "immune": 1.0},
    "default": None,  # iscc.sample.dissociation.DEFAULT_RECOVERY
    "strong": {"cancer": 0.9, "epithelial": 0.6, "host": 0.6, "stromal": 0.3, "immune": 0.1},
}
STRESS = 1.0
N_BLOCKS, BLOCK_HALF = 4, 8          # multi-region blocks: 17 x 17 demes (about 1 mm at the cm scale)


def _blocks(t, k=N_BLOCKS, half=BLOCK_HALF):
    """``k`` square blocks over cancer-dense demes, as far apart as farthest-point sampling allows."""
    G = t.grid_size
    dens = np.zeros(G * G)
    for di, deme in enumerate(t.demes[:G * G]):
        tot = sum(deme.values())
        if tot:
            dens[di] = sum(c for g, c in deme.items() if t._is_cancer(g)) / tot
    cand = np.where(dens >= 0.5)[0]
    rc = np.array([t.deme_coords[d] for d in cand], float)
    # keep candidates whose block fits on the grid
    ok = (rc.min(1) >= half) & (rc.max(1) < G - half)
    cand, rc = cand[ok], rc[ok]
    centres = [int(np.argmax(dens[cand] + 1e-6 * np.arange(len(cand))))]
    for _ in range(k - 1):
        d = np.min([np.abs(rc - rc[c]).max(1) for c in centres], axis=0)
        centres.append(int(np.argmax(d)))
    blocks, out_rc = [], []
    for c in centres:
        r0, c0 = (int(v) for v in rc[c])
        blocks.append([r * G + cc for r in range(r0 - half, r0 + half + 1) for cc in range(c0 - half, c0 + half + 1)])
        out_rc.append([r0, c0])
    return blocks, out_rc


def _clone_frac(labels, order):
    labels = list(labels)
    n = max(len(labels), 1)
    return [labels.count(L) / n for L in order]


def compute_a():
    from iscc.data import scRNA
    from iscc.sample.dissociation import Dissociation, biological_type
    t = _load()
    clones = C.Clones(t)
    order = clones.labels + [C.ANCESTRAL]
    spec = RR.resect(t)
    # whole tumour: every live cancer cell, from the per-deme counts (no materialisation)
    whole = dict.fromkeys(order, 0)
    for g, c in C.live_cancer_counts(t).items():
        whole[clones.of(g)] += c
    tot = sum(whole.values())
    fr = {"whole tumour": [whole[L] / tot for L in order]}

    def cancer_labels(cd):
        gid = cd["cell_type"]["cell_id"].astype(str)
        return [clones.of(g) for g in gid.values if g in t.genotypes and t._is_cancer(g)]

    sec = spec.slice(depth_frac=0.5, max_cells=6000, layers="visium")
    fr["section"] = _clone_frac(cancer_labels(sec), order)
    del sec
    blocks, centres = _blocks(t)
    for i, b in enumerate(blocks, start=1):
        cd = spec.dissociate(region=b, max_cells=1500, layers="scrna")
        fr[f"block {i}"] = _clone_frac(cancer_labels(cd), order)
        del cd
    diss = spec.dissociate(max_cells=4000, layers="scrna")
    fr["dissociated"] = _clone_frac(cancer_labels(diss), order)

    bio = diss["cell_type"]["cell_id"].astype(str).map(biological_type)
    types = ["cancer", "epithelial", "host", "stromal", "immune"]
    comp = {"tissue": [float((bio == k).mean()) for k in types]}
    for name, probs in RECOVERY.items():
        chosen, meta, _ = Dissociation(diss, rng=np.random.default_rng(3), recovery_probs=probs).run()
        comp[name] = [float((bio.loc[chosen] == k).mean()) for k in types]
        if name == "default":
            comp["_default_probs"] = meta["recovery_probs"]
    # the stress signature: recovered cells (default recovery) through scRNA with and without it
    chosen, meta, exp = Dissociation(diss, rng=np.random.default_rng(4), stress_strength=STRESS).run()
    genes = list(diss["cell_exp"].columns)
    idx = np.array([genes.index(g) for g in meta["stress_genes"]])
    cells = list(chosen[:2000])
    score = {}
    for name, cd in (("off", diss), ("on", {**diss, "cell_exp": exp})):
        X = sp.csr_matrix(scRNA(n_cells=len(cells), seed=5).run(cd, cell_subset=cells).observed_counts.values).tocsc()
        lib = np.asarray(X.sum(1)).ravel()
        score[name] = (np.asarray(X[:, idx].sum(1)).ravel() / np.maximum(lib, 1)).tolist()
    G = t.grid_size
    tmap = (255 * C.tissue_map(t, clones)).astype(np.uint8)
    return dict(_tumour=REG.task_path(*TUMOUR), clone_order=order,
                clone_colour=[clones.colour[L] for L in order], fractions=fr,
                block_centres=centres, block_half=BLOCK_HALF, grid=int(G), map=tmap.tolist(),
                types=types, composition=comp, stress_strength=STRESS, n_stress_genes=len(idx),
                stress_score=score)


# --- b. WGS --------------------------------------------------------------------------------------
KAPPA_X = (0.1, 0.3, 1.0, 3.0, 10.0)           # multiples of the fitted DM concentration
GC_SIGMA = (0.0, 0.1, 0.2, 0.4)


def _bulk(diss, est, seed=11, **over):
    from iscc.data import bulkDNA
    kw = {**est.dna_kwargs(), **over}
    return bulkDNA(breadth=est.breadth, depth_model=est.depth_model, seed=seed, **kw).run(diss).observed_data


def compute_b():
    from iscc.data import estimate_dna
    sys.path.insert(0, os.path.join(REPO, "validation"))
    import validate_dna as VD
    z = np.load(DNA_REF["HG002"], allow_pickle=True)
    est = estimate_dna(z["coverage"], z["alt"], z["cn"], modality="bulk", breadth=str(z["breadth"]),
                       depth_model=str(z["depth_model"]), het_mask=z["het_mask"], variant_mask=z["variant_mask"])
    real = np.asarray(z["coverage"], float)
    t = _load()
    diss = RR.resect(t).dissociate(max_cells=1500, layers="scdna")
    del t
    fit = _bulk(diss, est)
    two = np.rint(fit["true_cn"].values) == 2
    sim = fit["coverage"].values[two].astype(float)
    hist = dict(real=(real / real.mean()).tolist(), iscc=(sim / sim.mean()).tolist())
    sweep = []
    for x in KAPPA_X:
        o = _bulk(diss, est, kappa=est.hypers.kappa * x)
        sweep.append(_cv(o["coverage"].values[np.rint(o["true_cn"].values) == 2]))
    gc = {}
    for s in GC_SIGMA:
        # several batches per strength: each batch draws its own curve (peak and curvature)
        curves = []
        for seed in range(5):
            o = _bulk(diss, est, seed=100 + seed, gc_curve_sigma=s)
            x, m = VD._gc_curve(o["coverage"].values, o["true_cn"].values, o["gc"].values, nbins=10)
            curves.append(m.tolist())
        gc[str(s)] = dict(gc=x.tolist(), curves=curves)
    return dict(_tumour=REG.task_path(*TUMOUR), hist=hist, cv_real=_cv(real), cv_iscc=_cv(sim),
                kappa_fit=float(est.hypers.kappa), kappa_x=list(KAPPA_X), cv_sweep=sweep,
                gc_sigma_fit=float(est.hypers.gc_curve_sigma), gc_fitted="gc_curve_sigma" in est.fitted, gc=gc)


# --- c. scDNA ------------------------------------------------------------------------------------
ADO_RATES = (0.0, 0.15, 0.3, 0.45, 0.6)


def compute_c():
    from iscc.data import estimate_dna, scDNA
    sys.path.insert(0, os.path.join(REPO, "validation"))
    import validate_dna as VD
    left = _f2().compute_g()
    t = _load()
    diss = RR.resect(t).dissociate(max_cells=1500)
    gid = diss["cell_type"]["cell_id"].astype(str)
    cancer = [c for c, g in gid.items() if g in t.genotypes and t._is_cancer(g)][:150]
    n_genes = t.n_genes
    del t
    out = dict(_tumour=REG.task_path(*TUMOUR), left=left, kappa_x=list(KAPPA_X), ado_rates=list(ADO_RATES))
    for name in ("DLP+", "Tapestri"):
        z = np.load(DNA_REF[name], allow_pickle=True)
        kw = dict(modality="sc", breadth=str(z["breadth"]), depth_model=str(z["depth_model"]))
        if "het_mask" in z.files:
            kw["het_mask"] = z["het_mask"]
        est = estimate_dna(z["coverage"], z["alt"], z["cn"], **kw)
        tg = (z["coverage"].shape[1] / n_genes) if est.breadth == "panel" else "all"

        def run(**over):
            return scDNA(n_cells=len(cancer), breadth=est.breadth, depth_model=est.depth_model, seed=21,
                         target_genes=tg, **{**est.dna_kwargs(), **over}).run(diss, cell_subset=cancer)
        real_cv = _f2()._cv_by_state(np.atleast_2d(z["coverage"]), np.atleast_2d(z["cn"])).get(2)
        sweep = [_sc_stats(VD, run(kappa=est.hypers.kappa * x)) for x in KAPPA_X]
        rec = dict(kappa_fit=float(est.hypers.kappa), cv_real_cn2=real_cv, cv_sweep_cn2=[c for c, _ in sweep],
                   ado_sweep_kappa=[a for _, a in sweep], ado_setting=float(est.hypers.ado_rate),
                   ado_fitted="ado_rate" in est.fitted)
        if name == "Tapestri":
            ado = [_sc_stats(VD, run(ado_rate=a))[1] for a in ADO_RATES]
            rec.update(ado_fit=float(est.hypers.ado_rate), ado_sweep=ado,
                       ado_real=VD._ado_frac(VD._baf_at_hets(z["alt"], z["coverage"], z["het_mask"])))
        out[name] = rec
    # The package's own single-cell defaults (scDNA.mode_defaults: kappa 5, ado_rate 0.20, no kappa_loci),
    # recorded for the report; at whole-genome breadth kappa 5 spreads very few reads over thousands of loci.
    try:
        cv, a = _sc_stats(VD, scDNA(n_cells=len(cancer), breadth="wgs", seed=21).run(diss, cell_subset=cancer))
        out["sc_default"] = dict(cv_cn2=cv, ado=a)
    except Exception as e:                       # a diagnostic only; never lose the panel over it
        out["sc_default"] = dict(error=f"{type(e).__name__}: {e}")
    return out


def _sc_stats(VD, s):
    """Coverage CV at two copies (median over cells) and measured allelic dropout of an scDNA run."""
    cv = _f2()._cv_by_state(s.coverage.values.astype(float), s.true_cn.values).get(2)
    ado = VD._ado_frac(VD._baf_at_hets(s.alt_counts.values, s.coverage.values, VD._het_obs_assay_sc(s)))
    return cv, ado


# --- d. scRNA ------------------------------------------------------------------------------------
DROPOUT_MID = (0.0, 0.5, 1.5, 4.0)


def compute_d():
    import anndata as ad
    from iscc.data import estimate_rna, scRNA
    F2 = _f2()
    left = F2.compute_f()
    real = ad.read_h5ad(os.path.join(REPO, "data", "pbmc3k_raw.h5ad"))
    est = estimate_rna(real)
    del real
    t = _load()
    diss = RR.resect(t).dissociate(max_cells=3000, layers="scrna")
    del t
    cells = list(diss["cell_exp"].index)
    presets = {}
    for proto in ("10x", "smartseq3"):
        X = sp.csr_matrix(scRNA(protocol=proto, n_cells=len(cells), seed=5).run(diss, cell_subset=cells).observed_counts.values)
        presets[proto] = F2._count_stats(X)
        presets[proto].pop("loglib")
    sweep = {}
    for mid in DROPOUT_MID:
        kw = {**est.scrna_kwargs(), "dropout_mid": mid}
        X = sp.csr_matrix(scRNA(n_cells=len(cells), seed=5, **kw).run(diss, cell_subset=cells).observed_counts.values)
        st = F2._count_stats(X)
        st.pop("loglib")
        sweep[str(mid)] = st
    return dict(_tumour=REG.task_path(*TUMOUR), left=left, presets=presets, sweep=sweep,
                dropout_mid_fit=float(est.hypers.dropout_mid), dropout_fitted="dropout_mid" in est.fitted)


# --- e. Visium -----------------------------------------------------------------------------------
BLEED_RATES = (0.0, 0.1, 0.31, 0.6)


def compute_e():
    sys.path.insert(0, os.path.join(REPO, "validation"))
    import validate_visium_background as VB
    from iscc.data import estimate_visium
    left = _f2().compute_h()
    raw_dir = os.environ.get("ISCC_VISIUM_RAW_DIR") or os.path.join(REPO, "data", "visium_breast_A1")
    X, in_tissue, coords = VB.load_raw(raw_dir)
    est = estimate_visium(X, coords=coords, in_tissue=in_tissue, bleed_min_dist=1.5)
    fit = est.diagnostics["bleed"]
    totals = np.asarray(X.sum(axis=1), float).ravel()
    del X
    real = VB._profiles(totals, coords, in_tissue)
    model = VB._profiles(VB.model_on_slide(totals, coords, in_tissue, fit["bleed_rate"], fit["bleed_distal"],
                                           fit["sigma_pitch"]), coords, in_tissue)
    # the assay on a section with the real tissue outline (validate_visium_background.tissue_like): the
    # stored tumour's section fills the whole slide with tissue, so it has no spots outside it
    cell_data, side = VB.tissue_like(coords, in_tissue)
    base = {k: v for k, v in RG.VISIUM.items() if k != "bleed_rate"}
    sweep = {}
    for r in BLEED_RATES:
        profs, shares = [], []
        for seed in (1, 2):
            tot, crd, in_t = VB.iscc_slide(cell_data, side, seed=seed, bleed_rate=r, **base)
            profs.append(VB._profiles(tot, crd, in_t))
            shares.append(float(tot[~in_t].sum() / tot.sum()))
        sweep[str(r)] = dict(off=VB._mean_profiles(profs)["off"], share=float(np.mean(shares)))
    return dict(_tumour=REG.task_path(*TUMOUR), left=left, real_off=real["off"], model_off=model["off"],
                real_share=float(fit["background_share"]), fit={k: float(fit[k]) for k in
                                                                  ("bleed_rate", "bleed_distal", "sigma_pitch")},
                regime=RG.VISIUM, sweep=sweep)


# --- render --------------------------------------------------------------------------------------
def _cached(part):
    p = os.path.join(CACHE, f"{part}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def pending(ax, msg="pending"):
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=5, color=S.GREY, transform=ax.transAxes, wrap=True)
    S.despine(ax, False, False)


def _legend(ax, **kw):
    kw.setdefault("fontsize", 4.5)
    ax.legend(**kw)


def render_a(fig, y):
    d = _cached("a")
    y += 4
    axs = [S.ax_mm(fig, 14, y, 24, 21), S.ax_mm(fig, 47, y, 22, 21), S.ax_mm(fig, 100, y, 21, 21),
           S.ax_mm(fig, 141, y, 36, 21)]
    if d is None:
        for ax in axs:
            pending(ax)
        return None
    comp, types = d["composition"], d["types"]
    cols = [S.CLONE_FAMILIES[4][1], S.COMPARTMENT["epithelium"], S.COMPARTMENT["met"], S.CLONE_FAMILIES[5][2],
            S.MODALITY["spatial"]]
    ax = axs[0]
    rows = ["tissue", "none", "default", "strong"]
    for i, r in enumerate(rows):
        left = 0.0
        for k, v in enumerate(comp[r]):
            ax.barh(i, v, left=left, color=cols[k], height=0.7, label=types[k] if i == 0 and v > 0 else None)
            left += v
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(["tissue", "no bias", "default", "strong"])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Cell-type share")
    ax.set_title("recovery bias at dissociation\n(no real reference staged)", fontsize=5.5)
    _legend(ax, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=3, fontsize=4.3)
    ax = axs[1]
    off, on = np.log10(np.asarray(d["stress_score"]["off"]) + 1e-4), np.log10(np.asarray(d["stress_score"]["on"]) + 1e-4)
    bins = np.linspace(min(off.min(), on.min()), max(off.max(), on.max()), 30)
    ax.hist(off, bins=bins, density=True, color=SWEEP[1], alpha=0.8,
            label="stress off")
    ax.hist(on, bins=bins, density=True, color=SWEEP[-1], alpha=0.6, label=f"stress {d['stress_strength']:g}")
    ax.set_xlabel(f"log$_{{10}}$ UMI share in {d['n_stress_genes']} stress\ngenes (a random 5% of genes)")
    ax.set_ylabel("Density")
    _legend(ax, loc="upper left")
    ax.set_title("dissociation stress", fontsize=5.5)
    # clone fractions: map with the blocks, then the stacked bars
    ax = axs[2]
    ax.imshow(np.asarray(d["map"], np.uint8), interpolation="nearest")
    import matplotlib.patches as mp
    h = d["block_half"]
    for k, (r, c) in enumerate(d["block_centres"], start=1):
        ax.add_patch(mp.Rectangle((c - h - 0.5, r - h - 0.5), 2 * h + 1, 2 * h + 1, fill=False, lw=0.6, color=S.INK))
        ax.text(c, r, str(k), ha="center", va="center", fontsize=4.5, color=S.INK)
    S.despine(ax, False, False)
    ax.set_title("four blocks", fontsize=5.5)
    ax = axs[3]
    fr = d["fractions"]
    rows = ["whole tumour", "section", "dissociated"] + [f"block {i}" for i in range(1, len(d["block_centres"]) + 1)]
    for i, r in enumerate(rows):
        left = 0.0
        for k, v in enumerate(fr[r]):
            ax.barh(i, v, left=left, color=d["clone_colour"][k], height=0.7)
            left += v
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Clone share of the cancer cells")
    tv = lambda a, b: 0.5 * float(np.abs(np.asarray(a) - np.asarray(b)).sum())   # noqa: E731
    dist = {r: tv(fr[r], fr["whole tumour"]) for r in rows[1:]}
    blk = [dist[r] for r in rows[3:]]
    ax.set_title(f"distance to the whole: section {dist['section']:.2f},\n"
                 f"dissociated {dist['dissociated']:.2f}, blocks {min(blk):.2f}-{max(blk):.2f}", fontsize=5)
    return dict(composition=comp, tv_to_whole=dist,
                stress_median_ratio=float(np.median(d["stress_score"]["on"]) / np.median(d["stress_score"]["off"])))


def render_b(fig, y):
    d = _cached("b")
    axs = [S.ax_mm(fig, 14, y, 44, 24), S.ax_mm(fig, 100, y, 30, 24), S.ax_mm(fig, 145, y, 32, 24)]
    if d is None:
        for ax in axs:
            pending(ax)
        return None
    ax = axs[0]
    r, s = np.asarray(d["hist"]["real"]), np.asarray(d["hist"]["iscc"])
    bins = np.linspace(0, np.quantile(np.concatenate([r, s]), 0.995), 40)
    ax.hist(r, bins=bins, density=True, color=S.SOURCE["real"], alpha=0.6, label="GIAB HG002")
    ax.hist(s, bins=bins, density=True, color=S.SOURCE["iscc"], alpha=0.6, label="iscc, fitted")
    ax.set_xlabel("Per-locus coverage / mean")
    ax.set_ylabel("Density")
    _legend(ax, loc="upper right")
    S.stat(ax, f"CV real {d['cv_real']:.2f}\niscc {d['cv_iscc']:.2f}", "upper left")
    ax = axs[1]
    ax.plot(d["kappa_x"], d["cv_sweep"], marker="o", ms=2.5, color=S.SOURCE["iscc"], lw=0.8, label="iscc")
    ax.axhline(d["cv_real"], color=S.SOURCE["real"], lw=0.8, ls="--", label="HG002")
    ax.axvline(1.0, color=S.BG_CELLS, lw=0.5, zorder=0)
    ax.set_xscale("log")
    ax.set_xlabel("Depth concentration\n(x fitted)")
    ax.set_ylabel("Coverage CV")
    _legend(ax, loc="upper right")
    ax = axs[2]
    for k, (sig, g) in enumerate(d["gc"].items()):
        cur = np.asarray(g["curves"])
        ax.plot(g["gc"], cur.mean(0), color=SWEEP[k], lw=0.9, label=f"{float(sig):g}")
    ax.set_xlabel("GC content")
    ax.set_ylabel("Coverage per copy\n(mean 1)")
    _legend(ax, loc="lower center", title="GC-bias strength", title_fontsize=4.5, ncol=2)
    ax.set_title("iscc only; real GC curve pending", fontsize=5)
    return dict(cv_real=d["cv_real"], cv_iscc=d["cv_iscc"], cv_sweep=dict(zip(d["kappa_x"], d["cv_sweep"])))


def render_c(fig, y):
    d = _cached("c")
    axs = [S.ax_mm(fig, 14, y, 44, 24), S.ax_mm(fig, 100, y, 30, 24), S.ax_mm(fig, 145, y, 32, 24)]
    if d is None:
        for ax in axs:
            pending(ax)
        return None
    left = {k: v for k, v in d["left"].items() if k in ("Tapestri", "DLP+")}
    ado = d["left"].get("_ado", {})
    ax = axs[0]
    for i, (name, levels) in enumerate(left.items()):
        for j, (lvl, (o, s_)) in enumerate(sorted(levels.items(), key=lambda kv: int(kv[0]))):
            x = i * 11 + int(lvl)
            ax.scatter([x], [o], s=7, color=S.SOURCE["real"], zorder=3, label="real" if i == j == 0 else None)
            ax.scatter([x], [s_], s=7, color=S.SOURCE["iscc"], zorder=3, label="iscc, fitted" if i == j == 0 else None)
            ax.plot([x, x], [o, s_], color=S.GREY, lw=0.4)
    ax.set_xticks([i * 11 + 5 for i in range(len(left))])
    ax.set_xticklabels([f"{n}\n(copy number 1-10)" for n in left], fontsize=4.8)
    ax.set_ylabel("Coverage CV")
    _legend(ax, loc="upper right")
    if "Tapestri" in ado:
        S.stat(ax, f"Tapestri allelic dropout:\nreal {ado['Tapestri'][0]:.2f}, iscc {ado['Tapestri'][1]:.2f}", "upper left")
    ax = axs[1]
    i_fit = d["kappa_x"].index(1.0)
    for name, col in (("DLP+", SWEEP[-1]), ("Tapestri", SWEEP[1])):
        r = d[name]
        cv, ado_ = r["cv_sweep_cn2"], r["ado_sweep_kappa"]
        ax.plot(cv, ado_, color=col, lw=0.6, marker="o", ms=1.4, zorder=2)          # the concentration sweep
        ax.scatter([cv[i_fit]], [ado_[i_fit]], s=16, color=col, zorder=4, label=f"iscc, {name} fit")
    r = d["DLP+"]
    lumpy = d.get("sc_default", {})
    if "cv_cn2" in lumpy:
        # scDNA's own single-cell defaults (kappa 5, ado_rate 0.20), which the code calls lumpy MDA/MALBAC
        ax.scatter([lumpy["cv_cn2"]], [lumpy["ado"]], s=16, facecolors="white", edgecolors=S.SOURCE["iscc"], lw=0.8,
                   zorder=4, label="iscc, lumpy setting (scDNA\ndefaults; not a validated preset)")
    t_ = d["Tapestri"]
    ax.scatter([t_["cv_real_cn2"]], [t_["ado_real"]], s=16, color=S.SOURCE["real"], marker="D", zorder=5,
               label="Tapestri, real")
    ax.axvline(r["cv_real_cn2"], color=S.SOURCE["real"], lw=0.7, ls="--", zorder=1, label="DLP+ real\n(no allele data)")
    ax.set_xscale("log")
    ax.set_xlabel("Coverage CV, two copies")
    ax.set_ylabel("Allelic dropout")
    ax.set_ylim(0, 0.95)
    _legend(ax, loc="upper left", ncol=2, fontsize=3.8, columnspacing=0.6)
    ax.set_title("lines: amplification concentration 0.1-10x fit", fontsize=5)
    ax = axs[2]
    r = d["Tapestri"]
    ax.plot(d["ado_rates"], r["ado_sweep"], marker="o", ms=2.5, color=S.SOURCE["iscc"], lw=0.8, label="iscc")
    ax.axhline(r["ado_real"], color=S.SOURCE["real"], lw=0.8, ls="--", label="Tapestri")
    ax.axvline(r["ado_fit"], color=S.BG_CELLS, lw=0.5, zorder=0)
    ax.set_xlabel("Allelic-dropout rate setting")
    ax.set_ylabel("Allelic dropout")
    _legend(ax, loc="upper left")
    return dict(ado=ado, DLP=d["DLP+"], Tapestri=d["Tapestri"], sc_default=d.get("sc_default"))


def render_d(fig, y):
    d = _cached("d")
    axs = [S.ax_mm(fig, 100, y, 30, 24), S.ax_mm(fig, 145, y, 32, 24)]
    F2 = _f2()
    # the left half is Fig 2's panel, drawn from this figure's cache
    F2._cached = lambda part: (d or {}).get("left") if part == "f" else None
    left = F2.render_f(fig, 14, y + 1.5)
    if d is None:
        for ax in axs:
            pending(ax)
        return None
    real = d["left"]["real"]
    ax = axs[0]
    ax.plot(real["curve_m"], real["curve_z"], color=S.SOURCE["real"], lw=1.2, label="PBMC3k")
    for (proto, st), ls in zip(d["presets"].items(), ("-", "--")):
        ax.plot(st["curve_m"], st["curve_z"], color=S.SOURCE["iscc"], lw=0.9, ls=ls,
                label={"10x": "iscc, 10x", "smartseq3": "iscc, Smart-seq3"}[proto])
    ax.set_xlabel("log$_{10}$ mean")
    ax.set_ylabel("Zero fraction")
    _legend(ax, loc="lower left")
    ax = axs[1]
    ax.plot(real["curve_m"], real["curve_z"], color=S.SOURCE["real"], lw=1.2, label="PBMC3k")
    for k, (mid, st) in enumerate(d["sweep"].items()):
        ax.plot(st["curve_m"], st["curve_z"], color=SWEEP[k], lw=0.9, label=f"{float(mid):g}")
    ax.set_xlabel("log$_{10}$ mean")
    ax.set_ylabel("Zero fraction")
    _legend(ax, loc="lower left", title="dropout midpoint", title_fontsize=4.5)
    return dict(left=left, dropout_mid_fit=d["dropout_mid_fit"], dropout_fitted=d["dropout_fitted"])


def render_e(fig, y):
    d = _cached("e")
    ax = S.ax_mm(fig, 100, y, 46, 24)
    F2 = _f2()
    F2._cached = lambda part: (d or {}).get("left") if part == "h" else None
    left = F2.render_h(fig, 14, y)
    if d is None:
        pending(ax)
        return None
    dist, val, _ = zip(*d["real_off"])
    ax.plot(dist, val, "o", ms=2.2, color=S.SOURCE["real"], label="real section", zorder=3)
    dm, vm, _ = zip(*d["model_off"])
    ax.plot(dm, vm, color=S.INK, lw=0.6, ls=":", label="SpotClean model, fitted")
    for k, (r, st) in enumerate(d["sweep"].items()):
        if not st["off"]:
            continue
        ds, vs, _ = zip(*st["off"])
        ax.plot(ds, vs, marker="s", ms=1.6, color=SWEEP[k], lw=0.8, label=f"iscc, bleed {float(r):g}")
    ax.set_yscale("symlog", linthresh=1e-2)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Distance to the tissue (spot pitches)")
    ax.set_ylabel("Spot total outside the tissue\n/ mean tissue spot")
    _legend(ax, loc="upper left", bbox_to_anchor=(1.02, 1.05))
    ax.set_title("spot swapping (iscc: the real tissue outline, synthetic cells)", fontsize=5)
    S.text_mm(fig, 149, y + 19, "pending: SpotClean recovering the\ntrue counts on an iscc slide\n"
              "(not run; no result on record)", fontsize=5, va="top", color=S.GREY)
    shares = {r: st["share"] for r, st in d["sweep"].items()}
    return dict(left=left, real_share=d["real_share"], iscc_share=shares, fit=d["fit"])


def render():
    S.setup()
    fig = S.figure("double", 190)
    out = {}
    rows = [("a", "Sampling", render_a), ("b", "WGS", render_b), ("c", "scDNA", render_c),
            ("d", "scRNA", render_d), ("e", "Visium", render_e)]
    S.text_mm(fig, 30, 2.5, "fitted to a real reference", fontsize=6.5, va="top", color=S.GREY)
    S.text_mm(fig, 115, 2.5, "tunable: a sweep or a preset, the real value marked", fontsize=6.5, va="top",
              color=S.GREY)
    for i, (L, name, fn) in enumerate(rows):
        y = 9 + i * 37
        S.panel_letter(fig, L, 2, y - 3)
        S.text_mm(fig, 7, y - 2.6, name, fontsize=6.5, va="top", fontweight="bold")
        out[L] = fn(fig, y)
    tum = {p: (_cached(p) or {}).get("_tumour") for p in "abcde"}
    path = S.save(fig, "fig3_biases")
    print("wrote", path)
    print("tumour per part:", tum)
    for k, v in out.items():
        print(k, json.dumps(v, default=str)[:1500])


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=list("abcde"), help="compute one row and cache it")
    ap.add_argument("--all", action="store_true", help="compute every missing part (one process each), then render")
    a = ap.parse_args()
    if a.part:
        res = {"a": compute_a, "b": compute_b, "c": compute_c, "d": compute_d, "e": compute_e}[a.part]()
        os.makedirs(CACHE, exist_ok=True)
        if res is not None:
            json.dump(res, open(os.path.join(CACHE, f"{a.part}.json"), "w"), default=float)
        print(f"cached {a.part}: {res is not None} -> {CACHE}")
        return
    if a.all:
        for part in "abcde":
            if _cached(part) is None:
                subprocess.run([sys.executable, os.path.abspath(__file__), "--part", part], check=False)
    render()


if __name__ == "__main__":
    main()
