"""Figure 3 (every measurement step adds a bias that is fitted and tunable) and S11 Fig (its sweeps).

Figure 3 is one row at half a page: one panel per step or modality against its reference, and one
example of tuning a bias.

  A  sampling: clone shares of the whole primary, a section, a dissociated sample and four blocks;
  B  WGS: per-locus coverage of GIAB HG002 (two copies) against fitted iscc at the loci of the tumour's
     most common copy number, with the CV of each and the share of near-empty HG002 loci;
  C  scDNA: coverage CV per copy-number state, real against fitted iscc, for DLP+ and Tapestri;
  D  scRNA: zero fraction against gene mean, PBMC3k against fitted iscc;
  E  Visium: per-spot depth of a real breast-cancer section against fitted iscc, with Moran's I of each;
  F  tuning: Tapestri allelic dropout against ``ado_rate``, the real value crossed at the fitted setting.

S11 Fig holds the rest: (A-C) the recovery bias, the dissociation stress and the four blocks on the map;
(D, E) the WGS depth concentration and GC-bias sweeps; (F) the scDNA amplification sweep; (G-J) the
scRNA library size, mean-variance, protocol presets and dropout-midpoint sweep; (K) the Visium spot
swapping across its rate.

Parts c-e reuse Fig 2's data-front loops (``fig2_realism.compute_g/f/h``) as they are. Every part is
computed in its own process (``--part a|b|c|d|e``) on the stored showcase tumour
(``registry.load("R0", "seed1")`` under ``ISCC_PAPER_OUT`` / ``ISCC_PAPER_TIER``) and cached as small
JSON under ``$ISCC_FIG3_CACHE`` (default ``<derived>/fig3``); the render step reads only the caches and
writes both figures. To re-render from another stored tumour, point ``ISCC_PAPER_OUT`` at it and
recompute every part.

Real references: ``validation/data/dna_ref_*.npz``; ``data/pbmc3k_raw.h5ad`` and
``data/V1_Breast_Cancer_Block_A_Section_1`` (scanpy's cache, as Fig 2); the section's raw Space Ranger
matrix (all 4,992 spots) in ``$ISCC_VISIUM_RAW_DIR`` (default ``data/visium_breast_A1``).

Run from the iscc repository root:
    python validation/paper/fig3_biases.py --part a      # ... b, c, d, e
    python validation/paper/fig3_biases.py               # render both figures from the caches
"""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
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
    """``k`` square blocks over the cancer mass of the primary, as a multi-region study cuts a tumour
    into regions: the cancer-dense demes (at least half cancer cells) whose block fits on the grid are
    split into ``k`` parts by k-means of their positions, weighted by their cancer cells, and each block
    is centred on the deme of its part nearest the centre of the part, numbered from left to right.
    The blocks therefore sample where most cancer cells are, rather than the densest demes, which are
    the glands."""
    from sklearn.cluster import KMeans
    G = t.grid_size
    dens, cnt = np.zeros(G * G), np.zeros(G * G)
    for di, deme in enumerate(t.demes[:G * G]):
        tot = sum(deme.values())
        if tot:
            cnt[di] = sum(c for g, c in deme.items() if t._is_cancer(g))
            dens[di] = cnt[di] / tot
    cand = np.where(dens >= 0.5)[0]
    rc = np.array([t.deme_coords[d] for d in cand], float)
    # keep candidates whose block fits on the grid
    ok = (rc.min(1) >= half) & (rc.max(1) < G - half)
    cand, rc = cand[ok], rc[ok]
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(rc, sample_weight=cnt[cand])
    blocks, out_rc = [], []
    for j in np.argsort(km.cluster_centers_[:, 1], kind="stable"):
        members = np.flatnonzero(km.labels_ == j)
        m = members[np.argmin(((rc[members] - km.cluster_centers_[j]) ** 2).sum(1))]
        r0, c0 = (int(v) for v in rc[m])
        blocks.append([r * G + cc for r in range(r0 - half, r0 + half + 1) for cc in range(c0 - half, c0 + half + 1)])
        out_rc.append([r0, c0])
    return blocks, out_rc


def _clone_frac(labels, order):
    labels = list(labels)
    n = max(len(labels), 1)
    return [labels.count(L) / n for L in order]


def compute_a():
    from iscc.data import scRNA
    from iscc.sample.dissociation import Dissociation
    t = _load()
    clones = C.Clones(t)
    order = clones.labels + [C.ANCESTRAL]
    spec = RR.resect(t)
    # whole tumour: every live cancer cell of the PRIMARY, from the per-deme counts (no materialisation).
    # The section, the dissociated sample and the blocks are cut from the primary (Resection's default
    # compartment), so a metastatic lattice appended after the primary demes must not count here.
    n_prim = int(getattr(t, "n_primary_demes", 0) or t.grid_size ** 2)
    whole, met = dict.fromkeys(order, 0), 0
    for di, deme in enumerate(t.demes):
        for g, c in deme.items():
            if c > 0 and t._is_cancer(g):
                if di < n_prim:
                    whole[clones.of(g)] += c
                else:
                    met += c
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

    # coarse types: the stromal and immune subtypes count as stroma and immune
    from iscc.integrations import biological_types
    bio = pd.Series(biological_types(diss["cell_type"]["cell_id"].astype(str)), index=diss["cell_type"].index)
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
    return dict(_tumour=REG.task_path(*TUMOUR), n_cancer_primary=int(tot), n_cancer_met=int(met), clone_order=order,
                clone_colour=[clones.colour[L] for L in order], fractions=fr,
                block_centres=centres, block_half=BLOCK_HALF, grid=int(G), map=tmap.tolist(),
                types=types, composition=comp, stress_strength=STRESS, n_stress_genes=len(idx),
                stress_score=score)


# --- b. WGS --------------------------------------------------------------------------------------
KAPPA_X = (0.1, 0.3, 1.0, 3.0, 10.0)           # multiples of the fitted DM concentration
GC_SIGMA = (0.0, 0.1, 0.2, 0.4)
LOW_COV = 0.1                                  # "near-empty" locus: below a tenth of the mean coverage
CN_BAND = 0.02                                 # loci within this of the modal true copy number count as one state


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
    # HG002's sites are all at two copies, so its CV is the depth model's alone. The tumour's loci are
    # compared at one copy number too: those whose true (sample-mean) copy number is within CN_BAND of the
    # most common value. Rounding to two copies instead mixes states on a doubled tumour (on the v4
    # showcase, 17% of loci, true copies 1.98-2.3, CV 0.49 against 0.43 at the mode).
    cn = fit["true_cn"].values.astype(float)
    vals, counts = np.unique(np.round(cn, 2), return_counts=True)
    cn_mode = float(vals[np.argmax(counts)])
    same = np.abs(cn - cn_mode) < CN_BAND
    sim = fit["coverage"].values[same].astype(float)
    hist = dict(real=(real / real.mean()).tolist(), iscc=(sim / sim.mean()).tolist())
    sweep = []
    for x in KAPPA_X:
        o = _bulk(diss, est, kappa=est.hypers.kappa * x)
        sweep.append(_cv(o["coverage"].values[np.abs(o["true_cn"].values.astype(float) - cn_mode) < CN_BAND]))
    gc = {}
    for s in GC_SIGMA:
        # several batches per strength: each batch draws its own curve (peak and curvature)
        curves = []
        for seed in range(5):
            o = _bulk(diss, est, seed=100 + seed, gc_curve_sigma=s)
            x, m = VD._gc_curve(o["coverage"].values, o["true_cn"].values, o["gc"].values, nbins=10)
            curves.append(m.tolist())
        gc[str(s)] = dict(gc=x.tolist(), curves=curves)
    # the CVs match but the shapes do not: HG002 has a spike of near-empty loci that iscc lacks
    low = {k: float((np.asarray(v) < LOW_COV).mean()) for k, v in hist.items()}
    return dict(_tumour=REG.task_path(*TUMOUR), hist=hist, cv_real=_cv(real), cv_iscc=_cv(sim), low_frac=low,
                cn_mode=cn_mode, share_loci_at_mode=float(same.mean()),
                kappa_fit=float(est.hypers.kappa), kappa_x=list(KAPPA_X), cv_sweep=sweep,
                gc_sigma_fit=float(est.hypers.gc_curve_sigma), gc_fitted="gc_curve_sigma" in est.fitted, gc=gc)


# --- c. scDNA ------------------------------------------------------------------------------------
ADO_RATES = (0.0, 0.15, 0.3, 0.45, 0.6)
# Allelic dropout is scored at copy-balanced het sites on both sides, which is what the estimator's
# model assumes (a het BAF around 0.5). Counting every simulated observation with a true alt fraction in
# (0.15, 0.85) as "het" mixes in imbalanced sites: on the doubled v4 showcase 92% of them sit at 1:3,
# 1:2 or 1:4, whose extreme BAFs at panel depth look like dropout (15.5% extreme at ado_rate 0, against
# 2.6% at balanced sites; wt/fig3-jobs/diag/ado_het.py, job 5338433). Real het sites count as balanced
# at a called copy number of two.
BALANCED_AF = 0.05


def _het_balanced_real(z):
    """Real het observations at called copy number two (1:1), as a (cell, locus) mask."""
    cn = np.atleast_2d(np.asarray(z["cn"], float))
    het = np.asarray(z["het_mask"], bool)
    het = np.broadcast_to(het, cn.shape) if het.ndim == 1 else np.atleast_2d(het)
    return het & (np.rint(cn) == 2)


def _het_balanced_sim(s):
    """Simulated observations whose true alt fraction is 0.5 (within BALANCED_AF): 1:1 or 2:2 sites."""
    return np.abs(s.true_alt_fraction.values - 0.5) < BALANCED_AF


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
            fit_run = run()
            rec.update(ado_fit=float(est.hypers.ado_rate), ado_sweep=ado,
                       ado_real=VD._ado_frac(VD._baf_at_hets(z["alt"], z["coverage"], _het_balanced_real(z))),
                       # the earlier measure, every het site on both sides, kept for the record
                       ado_real_allhet=VD._ado_frac(VD._baf_at_hets(z["alt"], z["coverage"], z["het_mask"])),
                       ado_iscc_allhet=VD._ado_frac(VD._baf_at_hets(fit_run.alt_counts.values,
                                                                    fit_run.coverage.values,
                                                                    VD._het_obs_assay_sc(fit_run))),
                       share_balanced_sim=float(_het_balanced_sim(fit_run)[VD._het_obs_assay_sc(fit_run)].mean()))
            # panel c's headline (left["_ado"], computed by fig2_realism over every het site) at balanced sites
            left.setdefault("_ado", {})["Tapestri"] = [rec["ado_real"], _sc_stats(VD, fit_run)[1]]
        out[name] = rec
    # The package's own single-cell defaults (scDNA.mode_defaults), recorded for the report.
    try:
        cv, a = _sc_stats(VD, scDNA(n_cells=len(cancer), breadth="wgs", seed=21).run(diss, cell_subset=cancer))
        out["sc_default"] = dict(cv_cn2=cv, ado=a)
    except Exception as e:                       # a diagnostic only; never lose the panel over it
        out["sc_default"] = dict(error=f"{type(e).__name__}: {e}")
    return out


def _sc_stats(VD, s):
    """Coverage CV at two copies (median over cells) and measured allelic dropout of an scDNA run, at
    copy-balanced het sites."""
    cv = _f2()._cv_by_state(s.coverage.values.astype(float), s.true_cn.values).get(2)
    ado = VD._ado_frac(VD._baf_at_hets(s.alt_counts.values, s.coverage.values, _het_balanced_sim(s)))
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
# Figure 3 is one row at half a page (165 mm wide): the five fits against a real reference (A-E) and one
# tunability example (F, the scDNA allelic dropout against its setting: one line, linear axes, and the
# real value crossed at the fitted setting). Every other sweep and statistic is S11 Fig (render_supp).
MAIN_W, MAIN_H = 165, 48
MAIN_Y, MAIN_AX_H = 9, 29
# per panel: (letter x, axes x, axes width), in mm
MAIN_SLOTS = {"a": (1, 14.5, 15.5), "b": (32, 38.5, 17.5), "c": (58.5, 66, 17.5), "d": (85.5, 93, 17.5),
              "e": (113, 119.5, 17.5), "f": (139.5, 147, 16)}


def _cached(part):
    p = os.path.join(CACHE, f"{part}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def pending(ax, msg="pending"):
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=5.5, color=S.GREY, transform=ax.transAxes, wrap=True)
    S.despine(ax, False, False)


def _legend(ax, **kw):
    kw.setdefault("fontsize", 5.5)
    kw.setdefault("title_fontsize", 5.5)
    return ax.legend(**kw)


def _clone_bars(ax, d):
    """Clone shares of the cancer cells per sample, stacked, with the whole primary first."""
    fr = d["fractions"]
    rows = ["whole tumour", "section", "dissociated"] + [f"block {i}" for i in range(1, len(d["block_centres"]) + 1)]
    for i, r in enumerate(rows):
        left = 0.0
        for k, v in enumerate(fr[r]):
            ax.barh(i, v, left=left, color=d["clone_colour"][k], height=0.7)
            left += v
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r.replace("whole tumour", "whole primary") for r in rows])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["0", "0.5", "1"])
    # the total variation distance of each sample to the whole is printed for the text, not drawn: the
    # panel shows the clone shares it compares side by side
    tv = lambda a, b: 0.5 * float(np.abs(np.asarray(a) - np.asarray(b)).sum())   # noqa: E731
    return {r: tv(fr[r], fr["whole tumour"]) for r in rows[1:]}


def main_a(ax, d):
    tv = _clone_bars(ax, d)
    ax.set_xlabel("Clone share")
    return dict(tv_to_whole=tv)


def main_b(ax, d):
    r, s = np.asarray(d["hist"]["real"]), np.asarray(d["hist"]["iscc"])
    bins = np.linspace(0, np.quantile(np.concatenate([r, s]), 0.995), 30)
    hr, _, _ = ax.hist(r, bins=bins, density=True, color=S.SOURCE["real"], alpha=0.6, label=f"HG002 {d['cv_real']:.2f}")
    ax.hist(s, bins=bins, density=True, color=S.SOURCE["iscc"], alpha=0.6, label=f"iscc {d['cv_iscc']:.2f}")
    ax.set_ylim(0, 1.6 * hr.max())
    _legend(ax, loc="upper left", title="CV", alignment="left")
    # the near-empty loci of HG002 (iscc has none), labelled at their spike
    spike = hr[:max(1, int(np.searchsorted(bins, 2 * LOW_COV)))].max()
    ax.text(0.02, spike + 0.05 * hr.max(), f"{d['low_frac']['real']:.1%}", ha="left", va="bottom",
            fontsize=5.5, color=S.INK)
    ax.set_xlabel("Coverage / mean")
    ax.set_ylabel("Density")
    return dict(cv_real=d["cv_real"], cv_iscc=d["cv_iscc"], low_frac=d["low_frac"])


def main_c(ax, d):
    left = {k: v for k, v in d["left"].items() if k in ("Tapestri", "DLP+")}
    for name, marker in (("Tapestri", "^"), ("DLP+", "o")):
        lv = sorted(left[name], key=int)
        x = [int(v) for v in lv]
        for j, (src, z) in enumerate((("real", 3), ("iscc", 4))):
            y = [left[name][v][j] for v in lv]
            ax.plot(x, y, color=S.SOURCE[src], marker=marker, ms=1.9, lw=0.7, zorder=z,
                    label=("real" if src == "real" else "iscc") if name == "DLP+" else None)
    # the two references labelled beside their own lines: Tapestri (triangles) above, DLP+ (circles) below
    ax.set_ylim(0, 1.0)
    ax.text(0.97, 0.62, "Tapestri", transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=5.5,
            color=S.INK)
    ax.text(0.97, 0.03, "DLP+", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.5, color=S.INK)
    ax.set_xticks([1, 4, 8, 11])
    ax.set_xlabel("Copy number")
    ax.set_ylabel("Coverage CV")
    _legend(ax, loc="upper left")
    return dict(cv=left)


def main_d(ax, d):
    real, sim = d["left"]["real"], d["left"]["iscc"]
    ax.plot(real["curve_m"], real["curve_z"], color=S.SOURCE["real"], lw=1.2, label="PBMC3k")
    ax.plot(sim["curve_m"], sim["curve_z"], color=S.SOURCE["iscc"], lw=0.9, label="iscc")
    ax.set_xlabel("log$_{10}$ mean")
    ax.set_ylabel("Zero fraction")
    ax.set_ylim(0, 1.05)
    _legend(ax, loc="lower left")
    return dict(cv_lib_real=real["cv_lib"], cv_lib_iscc=sim["cv_lib"])


def main_e(ax, d):
    v = d["left"]
    lo, ls = np.asarray(v["toto"]), np.asarray(v["tots"])
    bins = np.linspace(min(lo.min(), ls.min()), max(lo.max(), ls.max()), 30)
    hr, _, _ = ax.hist(lo, bins=bins, density=True, color=S.SOURCE["real"], alpha=0.6,
                       label=f"real {v['moran_real']:.2f}")
    hs, _, _ = ax.hist(ls, bins=bins, density=True, color=S.SOURCE["iscc"], alpha=0.6,
                       label=f"iscc {v['moran_iscc']:.2f}")
    ax.set_ylim(0, 1.6 * max(hr.max(), hs.max()))
    _legend(ax, loc="upper left", title="Moran's I", alignment="left")
    ax.set_xlabel("log$_{10}$ UMIs per spot")
    ax.set_ylabel("Density")
    return dict(moran_real=v["moran_real"], moran_iscc=v["moran_iscc"])


def main_f(ax, d):
    r = d["Tapestri"]
    ado_fit_iscc = d["left"]["_ado"]["Tapestri"][1]
    ax.axvline(r["ado_fit"], color=S.BG_CELLS, lw=0.6, zorder=0)
    ax.axhline(r["ado_real"], color=S.SOURCE["real"], lw=0.8, ls="--", label="Tapestri")
    ax.plot(d["ado_rates"], r["ado_sweep"], marker="o", ms=2.0, color=S.SOURCE["iscc"], lw=0.8, label="iscc")
    ax.scatter([r["ado_fit"]], [ado_fit_iscc], s=14, color=S.SOURCE["iscc"], zorder=4)
    # the real value and iscc at the fitted setting, labelled where they are drawn
    ax.text(0.6, r["ado_real"] - 0.02, f"{r['ado_real']:.2f}", ha="right", va="top", fontsize=5.5, color=S.GREY)
    ax.text(r["ado_fit"] - 0.03, ado_fit_iscc + 0.02, f"{ado_fit_iscc:.2f}", ha="right", va="bottom",
            fontsize=5.5, color=S.INK)
    ax.set_xlim(-0.03, 0.63)
    ax.set_ylim(0, 0.6)
    ax.set_xticks([0, 0.3, 0.6])
    ax.set_xlabel("Dropout setting")
    ax.set_ylabel("Allelic dropout")
    _legend(ax, loc="upper left")
    return dict(ado_real=r["ado_real"], ado_fit=r["ado_fit"], ado_iscc_at_fit=ado_fit_iscc,
                ado_sweep=dict(zip(d["ado_rates"], r["ado_sweep"])))


def render_main():
    S.setup()
    fig = S.figure(MAIN_W, MAIN_H)
    parts = {"a": ("a", "Sampling", main_a), "b": ("b", "WGS", main_b), "c": ("c", "scDNA", main_c),
             "d": ("d", "scRNA", main_d), "e": ("e", "Visium", main_e), "f": ("c", "scDNA tuning", main_f)}
    out = {}
    for L, (part, title, fn) in parts.items():
        lx, ax_x, ax_w = MAIN_SLOTS[L]
        S.panel_letter(fig, L, lx, 1.5)
        ax = S.ax_mm(fig, ax_x, MAIN_Y, ax_w, MAIN_AX_H)
        ax.set_title(title)
        d = _cached(part)
        if d is None:
            pending(ax)
            continue
        out[L] = fn(ax, d)
    path = S.save(fig, "fig3_biases")
    print("wrote", path)
    for k, v in out.items():
        print(k, json.dumps(v, default=str)[:1500])


# --- render: S11 Fig, the sweeps and the further statistics -----------------------------------------
# drawn at the text width (165 mm), as the supplement includes it, so its text keeps its true size
SUPP_W, SUPP_H = 165, 200
SUPP_ROWS = (8, 58, 108, 158)      # top of the axes in each row, in mm
SUPP_AX_H = 32


def supp_a(ax, d):
    comp, types = d["composition"], d["types"]
    cols = [S.CLONE_FAMILIES[4][1], S.COMPARTMENT["epithelium"], S.COMPARTMENT["met"], S.CLONE_FAMILIES[5][2],
            S.MODALITY["spatial"]]
    rows = ["tissue", "none", "default", "strong"]
    for i, r in enumerate(rows):
        left = 0.0
        for k, v in enumerate(comp[r]):
            ax.barh(i, v, left=left, color=cols[k], height=0.7, label=types[k] if i == 0 and v > 0 else None)
            left += v
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(["tissue", "no bias", "default", "strong"])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Cell-type share")
    _legend(ax, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=3)
    return dict(composition=comp)


def supp_b(ax, d):
    off, on = (np.log10(np.asarray(d["stress_score"][k]) + 1e-4) for k in ("off", "on"))
    bins = np.linspace(min(off.min(), on.min()), max(off.max(), on.max()), 30)
    ax.hist(off, bins=bins, density=True, color=SWEEP[1], alpha=0.8, label="stress off")
    ax.hist(on, bins=bins, density=True, color=SWEEP[-1], alpha=0.6, label="stress on")
    ax.set_xlabel("log$_{10}$ UMI share in stress genes")
    ax.set_ylabel("Density")
    _legend(ax, loc="upper left")
    return dict(stress_median_ratio=float(np.median(d["stress_score"]["on"]) / np.median(d["stress_score"]["off"])))


def supp_c(ax, d):
    import matplotlib.patches as mp
    ax.imshow(np.asarray(d["map"], np.uint8), interpolation="nearest")
    h = d["block_half"]
    for k, (r, c) in enumerate(d["block_centres"], start=1):
        ax.add_patch(mp.Rectangle((c - h - 0.5, r - h - 0.5), 2 * h + 1, 2 * h + 1, fill=False, lw=0.6, color=S.INK))
        ax.text(c + h + 2, r, str(k), ha="left", va="center", fontsize=5.5, color=S.INK)
    S.despine(ax, False, False)


def supp_d(ax, d):
    # the depth concentration on its own scale (the sweep multiplies the fitted value), the fitted value marked
    kappa = np.asarray(d["kappa_x"], float) * float(d["kappa_fit"])
    ax.axvline(float(d["kappa_fit"]), color=S.BG_CELLS, lw=0.6, zorder=0)
    ax.axhline(d["cv_real"], color=S.SOURCE["real"], lw=0.8, ls="--", label="HG002")
    ax.plot(kappa, d["cv_sweep"], marker="o", ms=2.5, color=S.SOURCE["iscc"], lw=0.8, label="iscc")
    ax.set_xscale("log")
    ax.set_xlabel("Depth concentration")
    ax.set_ylabel("Coverage CV")
    _legend(ax, loc="upper right")
    return dict(cv_sweep=dict(zip(d["kappa_x"], d["cv_sweep"])))


def supp_e(ax, d):
    for k, (sig, g) in enumerate(d["gc"].items()):
        cur = np.asarray(g["curves"])
        ax.plot(g["gc"], cur.mean(0), color=SWEEP[k], lw=0.9, label=f"{float(sig):g}")
    ax.set_xlabel("GC content")
    ax.set_ylabel("Coverage per copy")
    _legend(ax, loc="lower left", title="GC-bias strength", ncol=2, alignment="left")


def supp_f(ax, d):
    i_fit = d["kappa_x"].index(1.0)
    for name, col in (("DLP+", SWEEP[-1]), ("Tapestri", SWEEP[1])):
        r = d[name]
        cv, ado_ = r["cv_sweep_cn2"], r["ado_sweep_kappa"]
        # the line is the amplification-concentration sweep, the large dot the fitted concentration
        ax.plot(cv, ado_, color=col, lw=0.6, marker="o", ms=1.6, zorder=2, label=f"iscc fitted to {name}")
        ax.scatter([cv[i_fit]], [ado_[i_fit]], s=16, color=col, zorder=4)
    lumpy = d.get("sc_default", {})
    if lumpy.get("cv_cn2") is not None:          # undefined when too few loci at two copies are covered
        # scDNA's own single-cell defaults (scDNA.mode_defaults)
        ax.scatter([lumpy["cv_cn2"]], [lumpy["ado"]], s=16, facecolors="white", edgecolors=S.SOURCE["iscc"], lw=0.8,
                   zorder=4, label="iscc, scDNA defaults")
    t_ = d["Tapestri"]
    ax.scatter([t_["cv_real_cn2"]], [t_["ado_real"]], s=16, color=S.SOURCE["real"], marker="D", zorder=5,
               label="Tapestri, real")
    ax.axvline(d["DLP+"]["cv_real_cn2"], color=S.SOURCE["real"], lw=0.7, ls="--", zorder=1, label="DLP+, real")
    ax.set_xscale("log")
    ax.set_xticks([0.1, 0.2, 0.5, 1, 2])
    ax.set_xticklabels(["0.1", "0.2", "0.5", "1", "2"])
    ax.xaxis.set_minor_formatter(__import__("matplotlib").ticker.NullFormatter())
    ax.set_xlabel("Coverage CV at two copies")
    ax.set_ylabel("Allelic dropout")
    ax.set_ylim(0, 0.6)
    _legend(ax, loc="upper right")
    return dict(DLP=d["DLP+"], Tapestri=d["Tapestri"], sc_default=d.get("sc_default"))


def supp_g(ax, d):
    real, sim = d["left"]["real"], d["left"]["iscc"]
    lr, ls = np.asarray(real["loglib"]), np.asarray(sim["loglib"])
    bins = np.linspace(min(lr.min(), ls.min()), max(lr.max(), ls.max()), 30)
    hr, _, _ = ax.hist(lr, bins=bins, density=True, color=S.SOURCE["real"], alpha=0.6,
                       label=f"PBMC3k {real['cv_lib']:.2f}")
    hs, _, _ = ax.hist(ls, bins=bins, density=True, color=S.SOURCE["iscc"], alpha=0.6,
                       label=f"iscc {sim['cv_lib']:.2f}")
    ax.set_ylim(0, 1.45 * max(hr.max(), hs.max()))
    ax.set_xlabel("log$_{10}$ UMIs per cell")
    ax.set_ylabel("Density")
    _legend(ax, loc="upper left", title="CV", alignment="left")
    return dict(cv_lib_real=real["cv_lib"], cv_lib_iscc=sim["cv_lib"])


def supp_h(ax, d):
    real, sim = d["left"]["real"], d["left"]["iscc"]
    ax.plot(real["curve_m"], real["curve_v"], color=S.SOURCE["real"], lw=1.2, label="PBMC3k")
    ax.plot(sim["curve_m"], sim["curve_v"], color=S.SOURCE["iscc"], lw=0.9, label="iscc")
    xs = np.linspace(min(real["curve_m"]), max(real["curve_m"]), 10)
    ax.plot(xs, xs, color=S.INK, lw=0.5, ls=":", zorder=0, label="Poisson")
    ax.set_xlabel("log$_{10}$ mean")
    ax.set_ylabel("log$_{10}$ variance")
    _legend(ax, loc="upper left")


def supp_i(ax, d):
    real = d["left"]["real"]
    ax.plot(real["curve_m"], real["curve_z"], color=S.SOURCE["real"], lw=1.2, label="PBMC3k")
    for (proto, st), ls in zip(d["presets"].items(), ("-", "--")):
        ax.plot(st["curve_m"], st["curve_z"], color=S.SOURCE["iscc"], lw=0.9, ls=ls,
                label={"10x": "iscc, 10x", "smartseq3": "iscc, Smart-seq3"}[proto])
    ax.set_xlabel("log$_{10}$ mean")
    ax.set_ylabel("Zero fraction")
    _legend(ax, loc="lower left")


def supp_j(ax, d):
    real = d["left"]["real"]
    ref = ax.plot(real["curve_m"], real["curve_z"], color=S.SOURCE["real"], lw=1.2, label="PBMC3k")
    mids = []
    for k, (mid, st) in enumerate(d["sweep"].items()):
        mids += ax.plot(st["curve_m"], st["curve_z"], color=SWEEP[k], lw=0.9, label=f"{float(mid):g}")
    ax.set_xlabel("log$_{10}$ mean")
    ax.set_ylabel("Zero fraction")
    ax.add_artist(_legend(ax, handles=ref, loc="upper right"))
    _legend(ax, handles=mids, loc="lower left", title="Dropout midpoint", alignment="left")
    return dict(dropout_mid_fit=d["dropout_mid_fit"], dropout_fitted=d["dropout_fitted"])


def supp_k(ax, d):
    dist, val, _ = zip(*d["real_off"])
    ref = ax.plot(dist, val, "o", ms=2.2, color=S.SOURCE["real"], label="real section", zorder=3)
    dm, vm, _ = zip(*d["model_off"])
    ref += ax.plot(dm, vm, color=S.INK, lw=0.6, ls=":", label="SpotClean model")
    rates = []
    for k, (r, st) in enumerate(d["sweep"].items()):
        if not st["off"]:
            continue
        ds, vs, _ = zip(*st["off"])
        rates += ax.plot(ds, vs, marker="s", ms=1.6, color=SWEEP[k], lw=0.8, label=f"{float(r):g}")
    ax.set_yscale("symlog", linthresh=1e-2)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Distance to the tissue (spot pitches)")
    # each spot total is divided by the mean spot inside the tissue (validate_visium_background._profiles);
    # the cache keeps no unnormalised totals, so the counts themselves would need part e rerun
    ax.set_ylabel("UMIs per spot outside the tissue,\nrelative to the mean tissue spot")
    first = _legend(ax, handles=ref, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ax.add_artist(first)
    _legend(ax, handles=rates, loc="upper left", bbox_to_anchor=(1.02, 0.72), title="iscc, swapping rate",
            alignment="left")
    return dict(real_share=d["real_share"], iscc_share={r: st["share"] for r, st in d["sweep"].items()}, fit=d["fit"])


def render_supp():
    S.setup()
    fig = S.figure(SUPP_W, SUPP_H)
    y1, y2, y3, y4 = SUPP_ROWS
    h = SUPP_AX_H
    # per panel: (cache part, title, draw, axes box x, y, w, h in mm)
    panels = [
        ("a", "Recovery bias", supp_a, (13, y1, 36, h - 6)),
        ("a", "Dissociation stress", supp_b, (63, y1, 38, h)),
        ("a", "Blocks of Figure 3A", supp_c, (122, y1, h, h)),
        ("b", "WGS depth", supp_d, (10, y2, 35, h)),
        ("b", "WGS GC bias", supp_e, (59, y2, 35, h)),
        ("c", "scDNA amplification", supp_f, (109, y2, 54, h)),
        ("d", "scRNA library size", supp_g, (10, y3, 29, h)),
        ("d", "scRNA mean and variance", supp_h, (51, y3, 29, h)),
        ("d", "scRNA protocols", supp_i, (92, y3, 29, h)),
        ("d", "scRNA dropout", supp_j, (133, y3, 29, h)),
        ("e", "Visium spot swapping", supp_k, (20, y4, 80, h)),
    ]
    out = {}
    for k, (part, title, fn, (x, y, w, hh)) in enumerate(panels):
        L = "abcdefghijk"[k]
        S.panel_letter(fig, L, max(x - 10, 1), y - 6.5)
        ax = S.ax_mm(fig, x, y, w, hh)
        ax.set_title(title)
        d = _cached(part)
        if d is None:
            pending(ax)
            continue
        res = fn(ax, d)
        if res is not None:
            out[L] = res
    path = S.save(fig, "supp_bias_sweeps")
    print("wrote", path)
    for k, v in out.items():
        print(k, json.dumps(v, default=str)[:1500])


def render():
    render_main()
    render_supp()
    print("tumour per part:", {p: (_cached(p) or {}).get("_tumour") for p in "abcde"})


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
