"""Figure 3: what only the join provides, a coupled answer key with a history.

(a) One clone across modalities (R0): its copy number in scDNA, its expression shift against each
    gene's true dosage sensitivity, its allelic imbalance in RNA, and its territory.
(b) From lineage to space (R1): territories at low and high dispersal, and clone assortment against
    dispersal across seeds.
(c) Genotype against niche (R10): the invasive (emt) program of cancer cells against the epithelial
    niche field, and its variance split into the genetic and the niche part.
(d) The history (R0): the tissue at four times from in situ growth to invasion, and when each clone
    arose against the in-situ and invasive burden.
(e) Across patients (R6): copy-number profiles are private, the driver landscape is shared.
(f) Emergent against chained coupling (R0, R4): clone assignment from expression on three versions
    of the same cells. On a chain through a fixed proportional law (the assigner's own); on a chain
    built as OncoSpot builds one (the clone's copy number through a power law, programme states
    shuffled across cells; :mod:`oncospot_law`); and on iscc's data. The assigner is a Poisson
    likelihood under the proportional law (clonealign's assumption), not the published tool; Fig 5
    runs it.

Run:  python validation/paper/fig3_answer_key.py
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
import oncospot_law as OS  # noqa: E402
import realistic_regime as RR  # noqa: E402

SEEDS = REG.SEEDS


def pending(ax, msg="pending: registry run not grown yet"):
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=5, color=S.GREY, transform=ax.transAxes)
    S.despine(ax, False, False)


def have(run, task):
    return os.path.exists(REG.task_path(run, task))


def seg_of_genes(columns):
    return np.array([int(g.split("_")[1]) for g in columns])


def _homologs(t, gid, seg):
    """(paternal, maternal) copy number of one segment in one genotype."""
    g = t.genotypes[gid].genome[seg]
    return len(g["p"]), len(g["m"])


def _dosage(h, s_g):
    """Expression of a gene with sensitivity ``s_g`` over its homologs, as the engine computes it."""
    return sum(np.where(c > 0, 1 + s_g * (c - 1), 0.0) for c in h)


# --- a: one clone across modalities ----------------------------------------------------------------
def panel_a(fig, y):
    t = REG.load("R0", "seed1")
    cl = C.Clones(t)
    A = C.assays("R0", "seed1")
    cn = A["seg_cn_true"]
    cancer = [c for c in A["cells"] if A["is_cancer"][c]]
    lab = pd.Series({c: cl.of(A["genotype"][c]) for c in cancer})
    consensus = cn.loc[cancer].groupby(lab).agg(lambda s: s.mode().iloc[0])
    base = cn.loc[cancer].mode().iloc[0]                       # the tumour's modal profile
    # the clone with the most segments differing from the modal profile (and enough cells)
    big = [l for l in cl.labels if (lab == l).sum() >= 40]
    k = max(big, key=lambda l: ((consensus.loc[l] != base).sum(), (lab == l).sum()))
    segs = np.where(consensus.loc[k].values != base.values)[0]
    colour = cl.colour[k]
    in_k = lab.index[lab == k]
    others = lab.index[(lab != k)]

    # (i) scDNA called copy number per segment
    ax = S.ax_mm(fig, 10, y + 6, 36, 28)
    cov = A["dna_coverage"]
    sid = seg_of_genes(cov.columns)
    segcov = np.stack([cov.values[:, sid == s].mean(1) for s in range(cn.shape[1])], axis=1)
    calls = pd.DataFrame(np.clip(np.rint(2 * segcov / np.median(segcov, axis=1, keepdims=True)), 0, 6),
                         index=A["dna_cells"])
    dk = [c for c in calls.index if c in set(in_k)]
    do = [c for c in calls.index if c in set(others)]
    x = np.arange(cn.shape[1])
    ax.step(x, calls.loc[do].mean(0), where="mid", color=S.GREY, lw=0.8, label="other cancer cells")
    ax.step(x, calls.loc[dk].mean(0), where="mid", color=colour, lw=1.2, label=k)
    ax.plot(x, consensus.loc[k].values, "k_", ms=4, mew=0.8, label="truth")
    ax.set_xlabel("Segment")
    ax.set_ylabel("Copy number (scDNA calls)")
    ax.set_xticks(x[::2])
    ax.legend(loc="upper right", fontsize=4.8)
    ax.set_title("DNA", fontsize=6.5, color=S.MODALITY["DNA"])

    # (ii) expression shift against true dosage sensitivity, genes on the clone's altered segments
    ax = S.ax_mm(fig, 54, y + 6, 36, 28)
    Y = A["rna_counts"]
    Yn = Y.div(Y.sum(1), axis=0) * 1e4
    gseg = seg_of_genes(Y.columns)
    sens = np.asarray(t.program_truth["dosage_sensitivity"], dtype=float)
    pts = []
    for s in segs:
        genes = np.where(gseg == s)[0]
        ref = [c for c in others if cn.loc[c].iloc[s] == base.iloc[s]]
        if len(ref) < 20:
            continue
        m1 = Yn.loc[[c for c in in_k if c in Yn.index]].iloc[:, genes].mean(0).values
        m0 = Yn.loc[[c for c in ref if c in Yn.index]].iloc[:, genes].mean(0).values
        ok = (m0 > 0.05) & (m1 > 0)
        dcn = consensus.loc[k].iloc[s] - base.iloc[s]
        # the engine's law acts per homolog: dosage(c) = 1 + s(c - 1), zero for a lost homolog, so
        # the reference curve needs the homolog copy numbers of the clone and of the reference
        hk = _homologs(t, pd.Series([A["genotype"][c] for c in in_k]).mode().iloc[0], s)
        h0 = _homologs(t, pd.Series([A["genotype"][c] for c in ref]).mode().iloc[0], s)
        pts.append((sens[genes][ok], np.log2(m1[ok] / m0[ok]), dcn, hk, h0))
    gain, loss = "#C66263", "#2F479E"
    xs = np.linspace(0, 1, 50)
    for sg, lfc, dcn, hk, h0 in pts:
        col = gain if dcn > 0 else loss
        ax.scatter(sg, lfc, s=1.2, color=col, alpha=0.5, rasterized=True)
        ax.plot(xs, np.log2(_dosage(hk, xs) / _dosage(h0, xs)), color=col, lw=0.8)
    ax.text(0.03, 0.97, "gain", color=gain, transform=ax.transAxes, va="top", fontsize=5)
    ax.text(0.03, 0.06, "loss", color=loss, transform=ax.transAxes, va="bottom", fontsize=5)
    ax.axhline(0, color=S.BG_CELLS, lw=0.5, zorder=0)
    ax.set_xlabel("True dosage sensitivity of the gene")
    ax.set_ylabel("Expression log$_2$ fold change")
    ax.set_title("RNA: dosage, buffered per gene", fontsize=6.5, color=S.MODALITY["RNA"])

    # (iii) allelic imbalance in RNA on the altered segments
    ax = S.ax_mm(fig, 98, y + 6, 36, 28)
    baf = A["rna_baf"]
    if baf is not None and len(segs):
        bseg = seg_of_genes(baf.columns)
        cols = np.isin(bseg, segs)
        def cell_baf(cells):
            v = baf.loc[[c for c in cells if c in baf.index]].values[:, cols].astype(float)
            return np.abs(np.nanmean(v, axis=1) - 0.5)
        bins = np.linspace(0, 0.5, 26)
        normals = [c for c in A["cells"] if not A["is_cancer"][c]]
        ax.hist(cell_baf(normals), bins=bins, color=S.GREY, alpha=0.6, density=True, label="normal cells")
        ax.hist(cell_baf(in_k), bins=bins, color=colour, alpha=0.8, density=True, label=k)
        ax.set_xlabel("|RNA B-allele fraction - 0.5|")
        ax.set_ylabel("Density")
        ax.legend(fontsize=4.8)
    else:
        pending(ax, "allele layer not materialised")
    ax.set_title("RNA: allelic imbalance", fontsize=6.5, color=S.MODALITY["RNA"])

    # (iv) territory
    ax = S.ax_mm(fig, 142, y + 4, 32, 32)
    rgb = C.tissue_background(t)
    G = t.grid_size
    for di in range(G * G):
        deme = t.demes[di]
        nk = sum(c for g, c in deme.items() if t._is_cancer(g) and cl.of(g) == k)
        nc = sum(c for g, c in deme.items() if t._is_cancer(g))
        r, c = t.deme_coords[di]
        if nk:
            a = 0.3 + 0.7 * nk / max(sum(deme.values()), 1)
            rgb[r, c] = (1 - a) * rgb[r, c] + a * np.array(C._hex(colour))
        elif nc:
            rgb[r, c] = 0.6 * rgb[r, c] + 0.4 * np.array(C._hex(S.BG_CELLS))
    ax.imshow(rgb, interpolation="nearest")
    S.despine(ax, False, False)
    ax.set_title("Space: territory", fontsize=6.5, color=S.MODALITY["spatial"])
    return k


# --- b: lineage to space --------------------------------------------------------------------------
def assortment(t):
    """Fraction of neighbouring cancer-deme pairs whose dominant clone is the same."""
    cl = C.Clones(t)
    G = t.grid_size
    dom = {}
    for di in range(G * G):
        cc = Counter()
        for g, c in t.demes[di].items():
            if t._is_cancer(g):
                cc[cl.of(g)] += c
        if cc:
            dom[di] = cc.most_common(1)[0][0]
    same = tot = 0
    for di, lab in dom.items():
        for nb in t._neighbors(di):
            if nb in dom and nb > di:
                tot += 1
                same += dom[nb] == lab
    return same / max(tot, 1)


def panel_b(fig, y):
    disps = (0.1, 0.3, 0.9, 2.0, 4.0)
    for i, d in enumerate((0.1, 4.0)):
        ax = S.ax_mm(fig, 10 + i * 34, y + 9, 30, 30)
        task = f"disp{d}_seed1"
        if not have("R1", task):
            pending(ax)
            continue
        t = REG.load("R1", task)
        ax.imshow(C.tissue_map(t, C.Clones(t)), interpolation="nearest")
        S.despine(ax, False, False)
        ax.set_title(f"dispersal {d}", fontsize=6)
    ax = S.ax_mm(fig, 84, y + 8, 30, 26)
    import summaries as SM                      # cached per tumour; never reload 25 tumours here
    rows = [(float(r["task"].split("_")[0][4:]), r["assortment"]) for r in SM.load("R1")
            if r.get("assortment") is not None]
    if rows:
        df = pd.DataFrame(rows, columns=["d", "a"])
        g = df.groupby("d")["a"]
        ax.errorbar(g.mean().index, g.mean().values, yerr=g.std().values, color=S.SOURCE["iscc"],
                    marker="o", ms=2.5, lw=0.8, capsize=1.5)
        ax.set_xscale("log")
        ax.set_xlabel("Dispersal rate")
        ax.set_ylabel("Same-clone neighbours")
        S.stat(ax, f"n = {df.groupby('d').size().min()} seeds", "upper right")
    else:
        pending(ax)


# --- c: genotype against niche --------------------------------------------------------------------
C_TASKS = [("R10", f"{arch}_seed{k}") for arch in ("barriers", "open") for k in SEEDS]
C_CACHE = os.path.join(C.DERIVED, "fig3c")


def _r2(y, X):
    X = np.column_stack([np.ones(len(y))] + X)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return 1 - np.var(y - X @ beta) / np.var(y)


def niche_task(key):
    """Variance of the invasive (emt) program across cancer cells split into clone and niche
    (two-factor Shapley: the average gain of each predictor over both orders of adding them)."""
    import json
    from memguard import wait_for_memory
    run, task = key
    wait_for_memory(label=f"{run}/{task}")
    try:
        t = REG.load(run, task)
        cl = C.Clones(t)
        cd = RR.resect(t).dissociate(max_cells=3000)
        gid = cd["cell_type"]["cell_id"].astype(str)
        cancer = np.array([g in t.genotypes and t._is_cancer(g) for g in gid.values])
        emt = cd["cell_program"]["emt"].values[cancer]
        deme = cd["cell_deme"]["deme_id"].values[cancer].astype(int)
        niche = np.asarray(t.microenv_truth["epithelial"])[deme]
        clone = pd.get_dummies(pd.Series([cl.of(g) for g in gid.values[cancer]])).values.astype(float)[:, 1:]
        rg, rn, rb = _r2(emt, [clone]), _r2(emt, [niche]), _r2(emt, [clone, niche])
        rec = dict(run=run, task=task, arch=task.split("_")[0], n_cells=int(cancer.sum()),
                   n_clones=int(clone.shape[1] + 1), genotype=float(0.5 * (rg + rb - rn)),
                   niche=float(0.5 * (rn + rb - rg)), residual=float(1 - rb))
        if task in ("barriers_seed1", "open_seed1"):           # Fig 3c and S3 Fig show these two
            rec.update(scatter_niche=niche.tolist(), scatter_emt=emt.tolist())
        os.makedirs(C_CACHE, exist_ok=True)
        with open(os.path.join(C_CACHE, f"{task}.json"), "w") as f:
            json.dump(rec, f)
        return key, None
    except Exception as e:
        return key, f"{type(e).__name__}: {e}"


def compute_c(jobs=1):
    import multiprocessing as mp
    todo = [k for k in C_TASKS if have(*k) and not os.path.exists(os.path.join(C_CACHE, f"{k[1]}.json"))]
    print(f"{len(todo)} genotype-niche tumours to compute", flush=True)
    with mp.get_context("spawn").Pool(processes=jobs, maxtasksperchild=1) as pool:
        for key, err in pool.imap_unordered(niche_task, todo):
            print(f"  {key[0]}/{key[1]}: {err or 'ok'}", flush=True)


def panel_c(fig, y):
    import json
    ax = S.ax_mm(fig, 126, y + 8, 26, 26)
    axb = S.ax_mm(fig, 160, y + 8, 16, 26)
    rows = []
    if os.path.isdir(C_CACHE):
        for fn in sorted(os.listdir(C_CACHE)):
            with open(os.path.join(C_CACHE, fn)) as f:
                rows.append(json.load(f))
    if not rows:
        pending(ax)
        pending(axb, "")
        return None
    one = next((r for r in rows if r["task"] == "barriers_seed1"), rows[0])
    ax.scatter(one["scatter_niche"], one["scatter_emt"], s=1.0, color=S.COMPARTMENT["epithelium"],
               rasterized=True)
    ax.set_xlabel("Epithelial niche field")
    ax.set_ylabel("emt program activity")
    df = pd.DataFrame(rows)
    for k, (arch, lab) in enumerate((("barriers", "barriers"), ("open", "open"))):
        sub = df[df.arch == arch]
        for comp, dx, col in (("niche", -0.18, S.COMPARTMENT["epithelium"]), ("genotype", 0.18, S.CLONE_FAMILIES[2][1])):
            x = k + dx + np.random.default_rng(k).uniform(-0.05, 0.05, len(sub))
            axb.scatter(x, sub[comp], s=3, color=col, label=comp if k == 0 else None)
            axb.plot([k + dx - 0.12, k + dx + 0.12], [sub[comp].mean()] * 2, color=S.INK, lw=0.8)
    axb.set_xticks([0, 1])
    axb.set_xticklabels(["barriers", "open"], fontsize=4.8)
    axb.set_ylim(0, 1)
    axb.set_ylabel("Variance explained")
    axb.legend(fontsize=4.5, loc="upper right")
    g = df.groupby("arch")
    return {a: dict(genotype=round(float(v.genotype.mean()), 2), niche=round(float(v.niche.mean()), 2),
                    genotype_range=(round(float(v.genotype.min()), 2), round(float(v.genotype.max()), 2)),
                    niche_range=(round(float(v.niche.min()), 2), round(float(v.niche.max()), 2)), n=int(len(v)))
            for a, v in g}


# --- d: the history -------------------------------------------------------------------------------
def panel_d(fig, y):
    t = REG.load("R0", "seed1")
    cl = C.Clones(t)
    snaps = t.spatial_snapshots()
    meta = snaps.groupby("snapshot")[["time"]].first()
    gland = np.asarray(t.gland_id)
    snaps = snaps.assign(stroma=gland[snaps.deme.values] == -1)
    cancer = snaps[snaps.cell_type == "cancer"]
    burden = cancer.groupby(["snapshot", "stroma"])["count"].sum().unstack(fill_value=0)
    burden.index = meta.loc[burden.index, "time"].values
    release = [e["time"] for e in t.events if e.get("event") == "origin_release"]
    t_inv = float(burden.index[(burden.get(True, pd.Series(0, index=burden.index)) > 0).values][0]) \
        if True in burden.columns and (burden[True] > 0).any() else None
    end = float(meta.time.max())
    times = [release[0] if release else end * 0.3, t_inv if t_inv else end * 0.6,
             (t_inv + end) / 2 if t_inv else end * 0.8, end]
    names = ["acinus opens", "first invasion", "invasive growth", "end"]
    for i, (tt, nm) in enumerate(zip(times, names)):
        k = int((meta.time - tt).abs().idxmin())
        ax = S.ax_mm(fig, 10 + i * 28, y + 10, 26, 26)
        ax.imshow(C.tissue_map(t, cl, C.snapshot_counts(t, snaps, k)), interpolation="nearest")
        S.despine(ax, False, False)
        ax.set_title(f"{nm}\n(t = {meta.time.loc[k]:.0f})", fontsize=5.5, linespacing=1.1)
    births = t.genotype_births()
    ax = S.ax_mm(fig, 130, y + 10, 45, 26)
    ax.fill_between(burden.index, 0, burden.get(False, 0), color=S.COMPARTMENT["epithelium"], alpha=0.5,
                    lw=0, label="in ducts")
    if True in burden.columns:
        ax.fill_between(burden.index, burden.get(False, 0), burden.get(False, 0) + burden[True],
                        color="#C8A27A", alpha=0.6, lw=0, label="in stroma")
    ax.set_yscale("symlog", linthresh=100)
    ax.set_xlabel("Time (generations)")
    ax.set_ylabel("Cancer cells")
    ymax = ax.get_ylim()[1]
    for lab_, g in zip(cl.labels, cl.roots):
        if g in births.index:
            ax.plot(births.loc[g, "birth_time"], ymax * 0.5, "v", ms=3, color=cl.colour[lab_],
                    mec=S.INK, mew=0.3)
    if release:
        ax.axvline(release[0], color=S.GREY, lw=0.5, ls="--")
        ax.set_xlim(release[0] - 12, end)
        ax.text(release[0] - 11, ymax * 0.02, "acinus closed\nuntil here", fontsize=4.5, color=S.GREY,
                va="bottom")
    ax.legend(loc="upper left", fontsize=4.8)
    S.stat(ax, "triangles: clone births", "lower right")
    return dict(release=release[0] if release else None, first_invasion=t_inv, end=end)


# --- e: across patients ---------------------------------------------------------------------------
def panel_e(fig, y):
    ax = S.ax_mm(fig, 10, y + 6, 46, 22)
    ax2 = S.ax_mm(fig, 66, y + 6, 40, 22)
    tasks = [task for task in REG.tasks("R6") if have("R6", task)]
    if not tasks:
        pending(ax)
        pending(ax2, "")
        return
    prof, hits = [], []
    for task in tasks:
        t = REG.load("R6", task)
        counts = C.live_cancer_counts(t)
        top = max(counts, key=counts.get)
        prof.append(t.genotypes[top].genome_summary["seg_cns"])
        cd = RR.resect(t).dissociate(max_cells=800)
        gid = cd["cell_type"]["cell_id"].astype(str)
        cancer = np.array([g in t.genotypes and t._is_cancer(g) for g in gid.values])
        drivers = np.concatenate([t.selection.get_oncogenes(), t.selection.get_tsgs()]).astype(int)
        snv = (cd["cell_snv"].values[cancer][:, drivers] > 0).mean(0)
        hits.append(pd.Series(snv >= 0.1, index=drivers))
    ax.imshow(np.array(prof), cmap=S.DIVERGING, norm=S.cn_norm(6), aspect="auto", interpolation="nearest")
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels([f"patient {i + 1}" for i in range(len(tasks))], fontsize=4.8)
    ax.set_xlabel("Segment")
    ax.set_title("Copy number of the dominant clone", fontsize=6)
    H = pd.DataFrame(hits).fillna(False)
    H = H.loc[:, H.any(axis=0)]
    H = H.loc[:, H.sum(axis=0).sort_values(ascending=False).index]
    ax2.imshow(H.values.astype(float), cmap="Greys", vmin=0, vmax=1.4, aspect="auto",
               interpolation="nearest")
    ax2.set_yticks([])
    ax2.set_xlabel("Driver genes hit (≥10% of cells)")
    ax2.set_title("Driver genes hit, from one shared set", fontsize=6)
    ax2.set_xticks([])
    return dict(n_patients=len(tasks), recurrent=int((H.sum(axis=0) >= 2).sum()), hit=int(H.shape[1]))


# --- f: emergent against chained coupling ---------------------------------------------------------
F_TASKS = ([("R0", f"seed{k}") for k in (1, 2, 3)]
           + [("R4", f"cnv{c}_seed{k}") for c in (0.15, 0.3) for k in SEEDS])
F_CACHE = os.path.join(C.DERIVED, "fig3f")


def _assign(counts, prof, gseg, ambient=0.05, n_iter=20):
    """Clone assignment by Poisson likelihood under the proportional dosage law (clonealign's
    assumption): a cell's expected share of gene g in clone k is base_g * CN_gk / 2, renormalised.

    The per-copy baseline base_g is estimated jointly with the assignment, as clonealign does:
    assign, re-estimate each gene's per-copy expression from the cells assigned to each clone,
    repeat. (A plain mean over all cells already contains the majority clone's copy number, which
    then counts twice and pushes cells towards the minority clone.) A small ambient share of the
    pooled profile is mixed in, as droplet data carry ambient RNA, so a clone with a lost segment
    is not ruled out by a few ambient counts there."""
    lib = counts.sum(1, keepdims=True)
    frac = counts / np.maximum(lib, 1)
    pooled = frac.mean(0)
    half = prof.values[:, gseg] / 2.0                                # clones x genes, copies / 2
    base = pooled / np.maximum(half.mean(0), 1e-9)                  # start: clones weighted equally
    k = None
    for _ in range(n_iter):
        P = base[None, :] * half
        P = (1 - ambient) * P / P.sum(1, keepdims=True) + ambient * pooled[None, :] / pooled.sum()
        new = (counts @ np.log(P + 1e-12).T).argmax(1)
        if k is not None and np.array_equal(new, k):
            break
        k = new
        h = half[k]                                                  # each cell's assumed copies
        with np.errstate(invalid="ignore", divide="ignore"):
            per_copy = np.where(h > 0, frac / h, np.nan)
        est = np.nanmean(per_copy, axis=0)
        base = np.where(np.isfinite(est), est, base)
    return prof.index.values[k]


def chained_task(key):
    """Clone assignment on iscc's coupled expression and on two chained versions of the same cells.

    The linear chain pushes each cell's copy number through a fixed proportional law (baseline per
    copy times copy number). The OncoSpot chain follows OncoSpot's law (:mod:`oncospot_law`): the
    cancer baseline times (CN/2)^0.4, with the copy number of the cell's lineage clone (clones under
    1% of the sampled cancer cells merged, each given its modal profile) and the programme vectors
    shuffled across cells. A fourth version, recorded but not drawn, uses OncoSpot's law with a per-
    gene exponent matched to iscc's dosage sensitivity, to separate the dosage law from the coupling.
    All versions are measured by the same scRNA assay with the same seed, so library sizes and count
    noise are identical and only the expression model differs. Each version is scored against the
    copy-number profile its generator used for the cell."""
    import copy
    import json
    from iscc.data import scRNA
    from memguard import wait_for_memory
    run, task = key
    wait_for_memory(label=f"{run}/{task}")
    try:
        t = REG.load(run, task)
        cl = C.Clones(t)
        # installed, so the lineage clones of the OncoSpot arm are read from exactly these cells
        diss = RR.resect(t).dissociate(max_cells=REG.T["max_cells"], install=True)
        gid = diss["cell_type"]["cell_id"].astype(str)
        cancer = [c for c, g in gid.items() if g in t.genotypes and t._is_cancer(g)]
        cn_all = C._seg_cn(t, diss).loc[cancer]
        # the target is each cell's own copy-number profile: a dosage method can at best recover the
        # copy number a cell carries, so cells are scored against the profiles carried by >= 3% of
        # them (lineage clones would mix profiles, which scrambles both arms alike)
        profile = cn_all.apply(lambda r: tuple(r.values), axis=1)
        freq = profile.value_counts()
        common = freq[(freq >= max(20, 0.03 * len(profile)))].index
        if len(common) < 2:
            raise ValueError(f"only {len(common)} copy-number profile carried by >= 3% of cells")
        # OncoSpot gives every cell its lineage clone's copy number: clones as in the paper (lineage
        # clades), with OncoSpot's 1% merge, each carrying its modal profile over the sampled cells
        from iscc.integrations import clones_from_clades
        lin = clones_from_clades(t, min_cells=max(1, int(np.ceil(OS.MERGE_FRAC * len(cancer)))),
                                 cell_ids=cancer).astype(str)
        mode = OS.modal_profiles(lin.to_dict(), profile.to_dict())
        gen_os = {c: mode[lin[c]] for c in cancer}
        # the same cells in every arm: a common profile of their own and of their clone
        common_set = set(common)
        cells = [c for c in cancer if profile[c] in common_set and gen_os[c] in common_set]
        n_dropped = sum(1 for c in cancer if profile[c] in common_set) - len(cells)
        cn = cn_all.loc[cells]
        uniq = {p: f"CN clone {i + 1}" for i, p in enumerate(common)}
        lab = np.array([uniq[profile[c]] for c in cells])
        lab_os = np.array([uniq[gen_os[c]] for c in cells])
        prof = pd.DataFrame({uniq[p]: list(p) for p in common}).T
        exp = diss["cell_exp"].loc[cells]
        gseg = seg_of_genes(exp.columns)
        cn_gene = cn.values[:, gseg].astype(float)
        per_copy = np.median(exp.values / np.maximum(cn_gene / 2.0, 1e-9), axis=0)
        chained = copy.copy(diss)
        chained["cell_exp"] = pd.DataFrame(per_copy[None, :] * cn_gene / 2.0, index=exp.index,
                                           columns=exp.columns)
        # OncoSpot arm: baseline (the diploid cancer expression, as OncoSpot uses the cell of origin's),
        # the clone's copy number through (CN/2)^beta, and each cell's programme state taken from
        # another cell. Without a programme layer the state term is 1.
        cn_os = np.array([gen_os[c] for c in cells], dtype=float)[:, gseg]
        base = np.asarray(t.celltype_exps["cancer"], dtype=float)
        if base.shape[0] != exp.shape[1]:
            raise ValueError("cancer baseline and expression columns differ in length")
        prog = diss.get("cell_program")
        P = getattr(t, "programs", None)
        if prog is not None and P is not None:
            Z = OS.permute_rows(prog.loc[cells].values, np.random.default_rng(2026))
            mult = P.gene_multiplier(Z)
        else:
            mult = None
        s_g = None if P is None else getattr(P, "dosage_sensitivity", None)
        arms = {"oncospot": OS.BETA_DEFAULT}
        if s_g is not None:
            arms["oncospot_matched"] = OS.matched_beta(s_g)
        Y = scRNA(n_cells=len(cells), seed=100).run(diss, cell_subset=cells).observed_counts
        Ych = scRNA(n_cells=len(cells), seed=100).run(chained, cell_subset=cells).observed_counts
        rec = dict(run=run, task=task, n_cells=len(cells), n_clones=int(len(prof)),
                   acc_iscc=float((_assign(Y.values.astype(float), prof, gseg) == lab).mean()),
                   acc_chained=float((_assign(Ych.values.astype(float), prof, gseg) == lab).mean()),
                   majority=float(pd.Series(lab).value_counts(normalize=True).iloc[0]),
                   majority_os=float(pd.Series(lab_os).value_counts(normalize=True).iloc[0]),
                   n_lineage_clones=int(lin[cancer].nunique()), n_dropped=int(n_dropped),
                   n_clone_cn_differs=int((lab != lab_os).sum()),
                   programs_permuted=mult is not None, beta=OS.BETA_DEFAULT,
                   beta_matched_median=(None if s_g is None
                                        else float(np.median(arms["oncospot_matched"]))))
        Yos = {}
        for name, beta in arms.items():
            os_diss = copy.copy(diss)
            os_diss["cell_exp"] = pd.DataFrame(OS.expression(base, cn_os, beta, mult), index=exp.index,
                                               columns=exp.columns)
            Yos[name] = scRNA(n_cells=len(cells), seed=100).run(os_diss, cell_subset=cells).observed_counts
            rec[f"acc_{name}"] = float((_assign(Yos[name].values.astype(float), prof, gseg) == lab_os).mean())
        if (run, task) == ("R0", "seed1"):
            # per-gene response to a gain, gained against diploid cells, on one segment
            # (each arm grouped by the copy number its generator used: the clone's, for OncoSpot)
            cnv = cn.values
            cnv_os = np.array([gen_os[c] for c in cells])
            cand = [k for k in range(cnv.shape[1]) if (cnv[:, k] == 3).sum() >= 30 and (cnv[:, k] == 2).sum() >= 30]
            if cand:
                genes = np.where(gseg == cand[0])[0]
                for name, X, cm in (("iscc", Y.values, cnv), ("chained", Ych.values, cnv),
                                    ("oncospot", Yos["oncospot"].values, cnv_os)):
                    g3, g2 = cm[:, cand[0]] == 3, cm[:, cand[0]] == 2
                    if g3.sum() < 10 or g2.sum() < 10:
                        continue
                    Xn = X[:, genes] / X.sum(1, keepdims=True) * 1e4
                    m3, m2 = Xn[g3].mean(0), Xn[g2].mean(0)
                    ok = (m2 > 0.1) & (m3 > 0)
                    rec[f"lfc_{name}"] = np.log2(m3[ok] / m2[ok]).tolist()
        os.makedirs(F_CACHE, exist_ok=True)
        with open(os.path.join(F_CACHE, f"{run}__{task}.json"), "w") as f:
            json.dump(rec, f)
        return key, None
    except Exception as e:
        return key, f"{type(e).__name__}: {e}"


def compute_f(jobs=1):
    import multiprocessing as mp
    todo = [k for k in F_TASKS if have(*k) and not os.path.exists(os.path.join(F_CACHE, f"{k[0]}__{k[1]}.json"))]
    print(f"{len(todo)} chained-vs-coupled tumours to compute", flush=True)
    with mp.get_context("spawn").Pool(processes=jobs, maxtasksperchild=1) as pool:
        for key, err in pool.imap_unordered(chained_task, todo):
            print(f"  {key[0]}/{key[1]}: {err or 'ok'}", flush=True)


# The three arms, left to right: the assigner's own law, OncoSpot's, iscc. OncoSpot's colour is an
# earthy hue kept clear of the source and modality colours.
OS_COLOUR = "#A6761D"
F_ARMS = (("acc_chained", "linear\nchain", S.GREY), ("acc_oncospot", "OncoSpot\nlaw", OS_COLOUR),
          ("acc_iscc", "iscc", S.SOURCE["iscc"]))


def f_verdict(df, margin=0.05, alpha=0.05):
    """The decision rule written before the run (vault note exp-paper1-fig3f-arms).

    With delta the mean of (OncoSpot arm - iscc) over tumours: delta >= margin and a paired Wilcoxon
    p < alpha means chaining as OncoSpot does it also makes integration easier (the claim holds);
    otherwise the claim narrows to chaining with a fixed proportional law. When OncoSpot's arm is
    clearly below iscc, the arm with iscc-matched exponents says whether the coupling itself costs
    the assigner accuracy beyond the dosage law."""
    from scipy.stats import wilcoxon

    def test(a, b):
        d = df[a] - df[b]
        p = float(wilcoxon(df[a], df[b]).pvalue) if len(df) > 5 and (d != 0).any() else None
        return float(d.mean()), p

    out = {}
    delta, p = test("acc_oncospot", "acc_iscc")
    out.update(delta_oncospot_iscc=round(delta, 3), p_oncospot_iscc=p)
    if delta >= margin and p is not None and p < alpha:
        out["verdict"] = "holds: OncoSpot-style chaining also makes assignment easier than iscc"
    elif delta <= -margin and p is not None and p < alpha:
        out["verdict"] = "narrows: OncoSpot's law makes assignment harder than iscc's"
    else:
        out["verdict"] = "narrows: the OncoSpot arm lands close to iscc; claim limited to a fixed proportional law"
    if "acc_oncospot_matched" in df:
        dm, pm = test("acc_oncospot_matched", "acc_iscc")
        out.update(delta_matched_iscc=round(dm, 3), p_matched_iscc=pm,
                   coupling_costs_accuracy=bool(dm >= margin and pm is not None and pm < alpha))
    return out


def panel_f(fig, y, x0=111):
    import json
    from scipy.stats import wilcoxon
    rows = []
    if os.path.isdir(F_CACHE):
        for fn in sorted(os.listdir(F_CACHE)):
            with open(os.path.join(F_CACHE, fn)) as f:
                rows.append(json.load(f))
    ax = S.ax_mm(fig, x0 + 7, y + 6, 28, 22)
    ax2 = S.ax_mm(fig, x0 + 43, y + 6, 24, 22)
    if not rows:
        pending(ax)
        pending(ax2, "")
        return None
    df = pd.DataFrame(rows)
    arms = [a for a in F_ARMS if a[0] in df]          # caches made before the OncoSpot arm lack it
    for _, r in df.iterrows():
        col = S.GREY if r.run == "R4" else S.INK
        ax.plot(range(len(arms)), [r[a[0]] for a in arms], color=col, lw=0.5, alpha=0.7, marker="o", ms=1.8)
    for x, (c, _, colr) in enumerate(arms):
        ax.plot([x - 0.3, x + 0.3], [df[c].mean()] * 2, color=colr, lw=1.6)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([a[1] for a in arms])
    ax.set_xlim(-0.5, len(arms) - 0.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Clone assignment accuracy")
    S.stat(ax, f"{len(df)} tumours", "lower left")
    one = next((r for r in rows if r["run"] == "R0" and r["task"] == "seed1" and "lfc_iscc" in r), None)
    if one:
        bins = np.linspace(-1.5, 2.0, 36)
        for name, _, col in (("chained", "linear chain", S.GREY), ("oncospot", "OncoSpot law", OS_COLOUR),
                             ("iscc", "iscc", S.SOURCE["iscc"])):
            if f"lfc_{name}" in one:
                lab = {"chained": "linear chain", "oncospot": "OncoSpot law"}.get(name, name)
                ax2.hist(one[f"lfc_{name}"], bins=bins, color=col, alpha=0.55, density=True, label=lab)
        ax2.axvline(np.log2(1.5), color=S.INK, lw=0.5, ls=":")
        ax2.set_xlabel("Per-gene log$_2$ FC, CN 3 vs 2")
        ax2.set_ylabel("Density")
        ax2.legend(fontsize=4.8, loc="upper left", frameon=False)
    else:
        pending(ax2, "no segment with a gain")
    out = dict(n=len(df), n_clones=(int(df.n_clones.min()), int(df.n_clones.max())))
    for c, _, _ in arms:
        out[c] = round(float(df[c].mean()), 3)
        out[c.replace("acc_", "") + "_range"] = (round(float(df[c].min()), 2), round(float(df[c].max()), 2))
    out["n_lower"] = int((df.acc_iscc < df.acc_chained).sum())
    out["wilcoxon_p"] = float(wilcoxon(df.acc_chained, df.acc_iscc).pvalue) if len(df) > 5 else None
    if "acc_oncospot" in df:
        out.update(f_verdict(df))
    if "acc_oncospot_matched" in df:
        out["acc_oncospot_matched"] = round(float(df.acc_oncospot_matched.mean()), 3)
    return out


def preview_f(path):
    """Panel f alone, written to ``path``: a review copy that leaves the paper's Fig 3 untouched until
    the whole figure is regenerated."""
    S.setup()
    fig = S.figure("onehalf", 44)
    S.panel_letter(fig, "f", 1, 1)
    S.text_mm(fig, 6, 2.4, "A proportional chain makes assignment easier", fontsize=7, va="top")
    out = panel_f(fig, 3, x0=0)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=200)
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--compute", action="store_true", help="compute Figs 3c and 3f per tumour, then stop")
    ap.add_argument("--compute-f", action="store_true", help="compute Fig 3f per tumour only, then stop")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--preview-f", metavar="PATH", help="draw panel f alone to PATH, then stop")
    a = ap.parse_args()
    if a.compute or a.compute_f:
        if a.compute:
            compute_c(a.jobs)
        compute_f(a.jobs)
        return
    if a.preview_f:
        import json
        print(json.dumps(preview_f(a.preview_f), indent=1))
        return
    S.setup()
    fig = S.figure("double", 186)
    out = {}
    S.panel_letter(fig, "a", 1, 1)
    S.text_mm(fig, 6, 2.4, "One clone across modalities", fontsize=7, va="top")
    out["a_clone"] = panel_a(fig, 4)
    S.panel_letter(fig, "b", 1, 46)
    S.text_mm(fig, 6, 47.4, "Lineage shapes space", fontsize=7, va="top")
    panel_b(fig, 46)
    S.panel_letter(fig, "c", 121, 46)
    S.text_mm(fig, 126, 47.4, "The key separates genotype from niche", fontsize=7, va="top")
    out["c"] = panel_c(fig, 46)
    S.panel_letter(fig, "d", 1, 92)
    S.text_mm(fig, 6, 93.4, "The answer key has a history", fontsize=7, va="top")
    out["d"] = panel_d(fig, 92)
    S.panel_letter(fig, "e", 1, 142)
    S.text_mm(fig, 6, 143.4, "Across patients", fontsize=7, va="top")
    out["e"] = panel_e(fig, 145)
    S.panel_letter(fig, "f", 111, 142)
    S.text_mm(fig, 116, 143.4, "A proportional chain makes assignment easier", fontsize=7, va="top")
    out["f"] = panel_f(fig, 145)
    path = S.save(fig, "fig3_answer_key")
    print("wrote", path)
    print(out)


if __name__ == "__main__":
    main()
