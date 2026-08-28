"""Validate W0 + W3 (DESIGN_cci_spatial.md): iscc's OWN ligand-receptor database + a receptor-
dependent cell-cell-communication channel, taken END TO END through the REAL CellChat.

THE QUESTION IS AMENABILITY. Does iscc emit data in which a real CCI tool can find the channel we
planted — using its own database format, its own pipeline, its own scoring? This is not a benchmark
of CellChat, and a tool failing to recover something is not the result we are after; it usually means
the signal is not in the form the tool reads (see the three silent failure modes in
DESIGN_cci_spatial.md, every one of which returned a clean null while the channel was working fine).

WHICH ASSAY. CCI inference is overwhelmingly done on DISSOCIATED scRNA — CellPhoneDB, CellChat's
original mode, NATMI, SingleCellSignalR, LIANA all score cell TYPES from mean ligand/receptor
expression, with no positions at all. So the inference arm here is scRNA (iscc's F3 assay, via the
oracle reference over the section's own cells). The Visium section is still built, and it is what
supplies the GROUND-TRUTH panels — the channel drawn in real tissue — plus the cells the reference is
drawn from. Spatial CellChat is available behind --mode spatial/both but is not the default: at true
Visium pitch its filters left 4 of 250 pairs scored, which is a question about the spatial pipeline
rather than about iscc's output.

WHAT IT SHOWS
  1. **The database (W0).** One tumour grown with an F8 CCI channel emits `n_candidate_pairs`
     candidate L-R pairs over its own gene ids; ONE is wired, the rest are unwired decoys. No decoy is
     engineered — pairs drawn at random already land on clone-varying segments by chance, so the
     clone-correlated class is MEASURED (`iscc.integrations.clone_correlation`), never planted.
  2. **The channel (W3).** The wired ligand marks the SENDER cell type (lifted above the tissue's
     ambient level for that gene, with the sender's own receptor suppressed) and the receptor marks
     the receiver population; the per-cell received signal is `ligand_avail[deme]·receptor[cell]`.
  3. **Recovery.** The written database is re-read into CellChat (the round trip of
     `validation/README_cellchat.md`) and we ask: does it rank the WIRED pair above the decoys BY
     COMMUNICATION PROBABILITY? Score by `prob`, not p-value — on group-averaged expression nearly
     every pair earns a "significant" edge, so a dense all-significant network is the background.

SCALE. The realistic breach-gated ductal field at the working point the RCTD deconvolution benchmark
is calibrated to (make_analysis_data.py): 60k cancer cells with a 0.12 immune compartment, Visium v1
geometry (0.55-deme radius at 2-deme pitch, a deme being 50 um => 55 um spots at 100 um pitch). An
earlier toy rig here used ~5k cells and overlapping spots, which that file's own density table calls
degenerate — and CellChat duly dropped three of seven groups for too few cells.

Env: CellChat runs in the dedicated `iscc-cellchat` env (README_cellchat.md); this script stays in the
core `iscc` env and shells out to `cellchat_runner.R`. Without that env the CellChat panels are
skipped and the ground-truth panels still render.

Writes the paper repo's figures/validation_cci.png.
Run:  python -u validation/validate_cci.py
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from _paths import figure_path

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "validation"))
sys.path.insert(0, os.path.join(REPO, "src"))

HOME = os.path.expanduser("~")
CELLCHAT_RSCRIPT = os.environ.get(
    "ISCC_CELLCHAT_RSCRIPT", os.path.join(HOME, "miniconda3/envs/iscc-cellchat/bin/Rscript"))
# Deme -> um. VISIUM_V1 below is a 0.55-deme radius at 2.0-deme pitch, which IS Visium v1 (55 um
# spot, 100 um pitch) only if a deme is 50 um — the ductal-field anchor make_analysis_data.py uses.
DEME_MICRONS = 50.0
# The calibrated Visium v1 geometry (make_analysis_data.VISIUM_V1). An earlier ad-hoc geometry here
# used spot_radius=0.9 at spot_pitch=1.5 — 1.8 demes across on a 1.5-deme pitch, i.e. OVERLAPPING
# spots, which no real Visium array can produce.
VISIUM_V1 = dict(spot_radius=0.55, spot_pitch=2.0, section_radius=None)
# Density decides whether the section means anything (measured table in make_analysis_data.py): at
# ~1.5k cancer cells a spot holds 1.7 cells at 0.98 purity — degenerate. 60k + immune 0.12 gives 0.72
# purity with 75% of spots holding >=2 types, i.e. genuinely mixed spots and no group so rare that
# CellChat drops it.
TARGET_CANCER = 60_000
MAX_CELLS = 40_000
IMMUNE_DENSITY = 0.12


def cellchat_available():
    return os.path.exists(CELLCHAT_RSCRIPT)


def grow(seed, steps, n_candidate_pairs, strength=1.0, emitter="immune", target_cancer=None):
    """Grow the REALISTIC breach-gated ductal field with an F8 CCI channel active.

    Same working point the RCTD deconvolution benchmark is calibrated to (make_analysis_data.py):
    a 60k-cancer-cell field with a 0.12 immune compartment, so Visium spots hold genuine multi-type
    mixtures instead of one cell of one type. The earlier toy rig here produced 134 spots and ~5k
    cells, which is the regime that table calls degenerate — and CellChat duly dropped three of the
    seven groups for having too few cells.
    """
    import deconv_common as D
    import realistic_regime as RR
    microenv = {"cci": {"strength": float(strength), "n_target_genes": 30, "emitter_type": str(emitter),
                        "lengthscale": 3.0, "n_candidate_pairs": int(n_candidate_pairs)}}
    return RR.grow_realistic(seed=seed, scale="mid", target_cancer=target_cancer or TARGET_CANCER,
                             max_cells=MAX_CELLS, spatial={"immune_density": IMMUNE_DENSITY},
                             expression=D.expression_params(), microenv=microenv, materialise=True)


def run_cellchat(work_dir, out_name="cellchat_net.csv", mode="spatial", circle_png=""):
    """Run cellchat_runner.R over CSVs already staged in work_dir; return the net DataFrame.

    ``mode`` is "spatial" (the Visium section, proximity MEASURED) or "rna" (dissociated cells,
    proximity ASSUMED) — the two arms whose disagreement is the point of this validation.
    """
    runner = os.path.join(REPO, "validation", "cellchat_runner.R")
    out_csv = os.path.join(work_dir, out_name)
    r = subprocess.run([CELLCHAT_RSCRIPT, runner, work_dir, out_csv, mode, circle_png],
                       capture_output=True, text=True)
    print(r.stdout)
    if r.returncode != 0:
        print(r.stderr)
        raise RuntimeError("cellchat_runner.R failed")
    return pd.read_csv(out_csv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1800)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--pairs", type=int, default=30, help="n_candidate_pairs in the database")
    ap.add_argument("--strength", type=float, default=8.0, help="CCI channel strength")
    ap.add_argument("--emitter", default="immune", help="emitter cell type (the SENDER population)")
    ap.add_argument("--out", default=figure_path("validation_cci.png"))
    ap.add_argument("--workdir", default=os.path.join(REPO, "validation", "_cci_tmp"))
    ap.add_argument("--sc-cells", type=int, default=4000,
                    help="cells in the dissociated scRNA arm")
    ap.add_argument("--mode", default="rna", choices=["both", "spatial", "rna"],
                    help="which arm(s) to INFER with. The Visium section is always built — it is what "
                         "supplies the ground-truth panels and the oracle scRNA reference — but CCI "
                         "inference defaults to the dissociated scRNA arm, which is how this analysis "
                         "is actually done.")
    ap.add_argument("--target-cancer", type=int, default=TARGET_CANCER,
                    help="cancer-cell target (lower it to smoke-test the code paths quickly)")
    ap.add_argument("--truth-only", action="store_true",
                    help="render the GROUND-TRUTH row only; skip the CellChat run")
    args = ap.parse_args()

    import deconv_common as D
    from iscc.integrations import cci_database, write_cci_database, clone_correlation

    os.makedirs(args.workdir, exist_ok=True)
    t = grow(args.seed, args.steps, args.pairs, args.strength, args.emitter, args.target_cancer)
    truth = t.microenv_truth
    pairs = truth["cci_pairs"]
    wired = int(truth["cci_wired_pair"])
    gene_names = t.selection.get_gene_names()
    wired_name = f"{gene_names[int(pairs[wired,0])]}_{gene_names[int(pairs[wired,1])]}"
    print(f"tumour: {t.get_tumor_size()} cells; database: {len(pairs)} pairs, "
          f"wired = {wired_name} (ligand {truth['cci_ligand']}, receptor {truth['cci_receptor']})")

    # ---- W0: the database, written + asserted complete (the CellChat whitelist round trip) ----
    db = cci_database(t)
    write_cci_database(db, args.workdir)
    print(f"database written to {args.workdir} (geneInfo whitelist asserted complete)")

    # ---- the MEASURED clone-correlation of every candidate (W4's axis; reported, not tuned) ----
    cc = clone_correlation(t.cell_data, pairs)
    cc["wired"] = [i == wired for i in range(len(cc))]
    eta_wired = float(cc.loc[wired, "eta_max"])
    print(f"clone-correlation (eta_max) over candidates: "
          f"min {cc['eta_max'].min():.2f}, median {cc['eta_max'].median():.2f}, "
          f"max {cc['eta_max'].max():.2f}; wired pair {eta_wired:.2f}")

    # ---- the Visium section (deme coords -> um), staged for CellChat ----
    section = D.build_section(t, seed=args.seed, **VISIUM_V1)
    spot_counts = section["spot_counts"]                      # spots x genes
    coords_um = section["coords"] * DEME_MICRONS              # spots x 2, in um
    # per-spot group = dominant population label (normals by type, cancer by CNA clone)
    clone_cats = section["clone_categories"]
    dom = np.asarray(clone_cats)[section["true_clone"].argmax(1)]
    spot_names = section["spot_names"]
    print(f"section: {len(spot_names)} spots, {spot_counts.shape[1]} genes, "
          f"{len(set(dom))} groups")

    net = None
    if args.truth_only or args.mode == "rna":
        if args.truth_only:
            print("truth-only: skipping CellChat.")
    elif cellchat_available():
        pd.DataFrame(spot_counts.values.T, index=spot_counts.columns, columns=spot_names).to_csv(
            os.path.join(args.workdir, "sp_counts.csv"))            # genes x spots
        pd.DataFrame(coords_um, index=spot_names, columns=["x_um", "y_um"]).to_csv(
            os.path.join(args.workdir, "sp_coords.csv"))
        pd.DataFrame({"group": dom}, index=spot_names).to_csv(
            os.path.join(args.workdir, "sp_meta.csv"))
        try:
            net = run_cellchat(args.workdir, "cellchat_net.csv", "spatial",
                               os.path.join(args.workdir, "circle_spatial.png"))
        except Exception as e:                                       # noqa: BLE001
            print(f"CellChat run failed ({e}); rendering the database panels only.")
    else:
        print(f"iscc-cellchat not found at {CELLCHAT_RSCRIPT}; skipping the CellChat panel.")

    # ---- score the recoverability: per-pair max prob, wired rank ----
    pair_prob, wired_rank, n_pairs_seen = None, None, 0
    if net is not None and len(net):
        pp = net.groupby("interaction_name")["prob"].max().sort_values(ascending=False)
        pair_prob = pp
        n_pairs_seen = len(pp)
        if wired_name in pp.index:
            wired_rank = int(pp.index.get_loc(wired_name)) + 1
            print(f"CellChat: {n_pairs_seen} pairs scored; wired pair {wired_name} ranks "
                  f"#{wired_rank}/{n_pairs_seen} by prob (prob={pp[wired_name]:.3f}, "
                  f"top decoy={pp[pp.index != wired_name].max():.3f})")
            verdict = "OUTRANKS decoys" if wired_rank == 1 else \
                      ("beats the median decoy" if wired_rank <= n_pairs_seen / 2 else
                       "does NOT outrank decoys")
            print(f"VERDICT: the wired pair {verdict} at Visium resolution.")
        else:
            print(f"CellChat: wired pair {wired_name} was DROPPED (its L/R genes were not "
                  f"over-expressed / not detected in any spot group). {n_pairs_seen} pairs survived.")

    # ---- GROUND-TRUTH ADJACENCY: which group pairs were ever within signalling range ----------
    # The spatial arm MEASURES proximity; the scRNA arm ASSUMES it. This is the truth both are
    # implicitly claiming about, and only iscc can supply it.
    adj = None
    if not args.truth_only:
        pop = D.population_labels(t, n_clones=4)
        crd = t.cell_data["cell_crd"].values.astype(float) * DEME_MICRONS
        in_sec = np.zeros(len(pop), dtype=bool)
        pos = {c: i for i, c in enumerate(t.cell_data["cell_type"].index)}
        for m in section["members"]:
            for c in m:
                j = pos.get(c)
                if j is not None:
                    in_sec[j] = True
        pitch_um = VISIUM_V1["spot_pitch"] * DEME_MICRONS
        radius = 4.0 * pitch_um                                     # the interaction range CellChat got
        adj = group_adjacency(crd[in_sec], pop.values[in_sec], radius)
        # A CONTACT-scale adjacency as well: at a few hundred um a mixed tissue has every group near
        # every other, so the diffusive range cannot discriminate between methods. Contact scale can.
        contact_adj = group_adjacency(crd[in_sec], pop.values[in_sec], 0.5 * pitch_um)
        print(f"ground-truth adjacency over {int(in_sec.sum())} section cells: "
              f"interaction {radius:.0f} um (min {adj.values.min():.2f}), "
              f"contact {0.5*pitch_um:.0f} um (min {contact_adj.values.min():.2f}, "
              f"pairs <0.25: {int((contact_adj.values < 0.25).sum())}/{contact_adj.size})")

    # ---- ARM 2: DISSOCIATED scRNA — the way CCI is usually actually done ----------------------
    # mode="oracle" takes the SAME cells as the section, so the only variable that changes between
    # the two arms is whether CellChat is given coordinates.
    net_rna, pair_prob_rna = None, None
    if (not args.truth_only) and args.mode in ("both", "rna") and cellchat_available():
        ref = D.build_reference(t, section, mode="oracle", label_by="population",
                                n_cells=args.sc_cells, seed=args.seed)
        sc = ref["counts"]                                          # cells x genes
        pd.DataFrame(sc.values.T, index=sc.columns, columns=sc.index).to_csv(
            os.path.join(args.workdir, "sc_counts.csv"))            # genes x cells
        pd.DataFrame({"group": np.asarray(ref["labels"])}, index=sc.index).to_csv(
            os.path.join(args.workdir, "sc_meta.csv"))
        try:
            net_rna = run_cellchat(args.workdir, "cellchat_net_rna.csv", "rna",
                                   os.path.join(args.workdir, "circle_rna.png"))
        except Exception as e:                                       # noqa: BLE001
            print(f"scRNA CellChat run failed ({e}); rendering the spatial arm only.")
        if net_rna is not None and len(net_rna):
            pair_prob_rna = net_rna.groupby("interaction_name")["prob"].max().sort_values(ascending=False)
            r = (int(pair_prob_rna.index.get_loc(wired_name)) + 1
                 if wired_name in pair_prob_rna.index else None)
            print(f"scRNA: {len(pair_prob_rna)} pairs scored; wired pair "
                  + (f"ranks #{r}/{len(pair_prob_rna)} by prob" if r else "was DROPPED"))

    # ---- THE COMPARISON: does knowing WHERE the cells were change the answer? -----------------
    fid = {}
    if adj is not None:
        true_mat = _true_matrix(dom, section, t, args.emitter)
        for label, n in (("spatial", net), ("scRNA  ", net_rna)):
            f = edge_fidelity(n, adj, true_mat, contact_adj)
            if f is None:
                continue
            fid[label.strip()] = f
            print(f"{label}: {f['n_edges']:3d} group-pair edges | r(inferred, TRUE) = {f['r_true']:+.2f}"
                  f" | r(inferred, adjacency) = {f['r_adj']:+.2f}"
                  f" | prob mass on rarely-touching pairs = {f['mass_low']:.0%}")
            worst = f["edges"].nsmallest(3, "contact")[["source", "target", "prob", "contact"]]
            for _, row in worst.iterrows():
                print(f"      {row['source']:>10} -> {row['target']:<10} prob={row['prob']:.3f} "
                      f"contact-adjacency={row['contact']:.2f}")

    _figure(args.out, cc, wired, pair_prob, wired_name, t, section, net, dom,
            spot_counts, truth, gene_names, args.emitter, args.truth_only,
            net_rna, pair_prob_rna, adj, fid)


def group_adjacency(coords_um, groups, radius_um):
    """GROUND TRUTH "could these two groups even talk?": for each ordered pair (A, B), the fraction of
    A-cells with at least one B-cell within ``radius_um``.

    This is the quantity a dissociated scRNA analysis cannot see and therefore ASSUMES. Every CCI tool
    built for scRNA scores cell types purely on mean ligand/receptor expression, so it will happily
    report communication between two types that never sat within signalling range of each other. Real
    data cannot expose that error — it has no ground truth for who was next to whom. iscc does.
    """
    from scipy.spatial import cKDTree
    groups = np.asarray(groups)
    gs = sorted(set(groups.tolist()))
    out = pd.DataFrame(0.0, index=gs, columns=gs)
    for b in gs:
        mb = groups == b
        if not mb.any():
            continue
        tree = cKDTree(coords_um[mb])
        nb = int(mb.sum())
        for a in gs:
            ma = groups == a
            if not ma.any():
                continue
            k = 2 if (a == b) else 1          # a cell is its own nearest neighbour within its group
            if k > nb:
                out.loc[a, b] = 0.0
                continue
            d, _ = tree.query(coords_um[ma], k=k)
            d = d[:, -1] if k == 2 else d
            out.loc[a, b] = float(np.mean(np.asarray(d) <= radius_um))
    return out


def _true_matrix(dom, section, tumor, emitter):
    """The GROUND-TRUTH sender x receiver matrix: only the emitter sends, and each receiver group's
    weight is its mean received signal (ligand availability x that cell's own receptor level)."""
    groups = sorted(set(np.asarray(dom).tolist()))
    me = tumor.cell_data["cell_microenv"]["cci_level"]
    lvl_by = pd.Series(me.values, index=tumor.cell_data["cell_type"].index)
    spot_lvl = np.array([lvl_by.reindex(list(m)).mean() if len(m) else np.nan
                         for m in section["members"]])
    gt = pd.DataFrame(0.0, index=groups, columns=groups)
    if emitter in gt.index:
        for g in groups:
            sel = np.asarray(dom) == g
            gt.loc[emitter, g] = float(np.nanmean(spot_lvl[sel])) if sel.any() else 0.0
    return gt


def edge_fidelity(net, adj, true_mat, contact_adj=None, low=0.25):
    """Score an inferred network against what the tissue actually permitted.

    A binary "these two groups were NEVER within range" test does not discriminate on a well-mixed
    tissue — at a few hundred um everything is near everything, so the rate collapses to 0% for every
    method. These are graded instead:

      ``r_true``      Pearson r between the inferred sender x receiver strengths and the TRUE ones.
                      The direct answer to "did it recover the real communication structure?".
      ``r_adj``       correlation between inferred strength and ground-truth adjacency — does the
                      method put its mass where cells could actually reach each other?
      ``mass_low``    fraction of total inferred probability mass sitting on group pairs whose
                      CONTACT-scale adjacency is below ``low``. This is the graded form of the
                      phantom-edge idea: communication asserted between groups that rarely touch.
    """
    if net is None or not len(net):
        return None
    e = net.groupby(["source", "target"])["prob"].sum().reset_index()
    pick = lambda M, s, t: (float(M.loc[s, t]) if (s in M.index and t in M.columns) else np.nan)
    e["true"] = [pick(true_mat, s, t) for s, t in zip(e["source"], e["target"])]
    e["adjacency"] = [pick(adj, s, t) for s, t in zip(e["source"], e["target"])]
    ca = contact_adj if contact_adj is not None else adj
    e["contact"] = [pick(ca, s, t) for s, t in zip(e["source"], e["target"])]
    out = {"n_edges": len(e), "edges": e}
    v = e.dropna(subset=["true"])
    out["r_true"] = (float(np.corrcoef(v["prob"], v["true"])[0, 1])
                     if len(v) > 2 and v["true"].std() > 0 and v["prob"].std() > 0 else np.nan)
    v = e.dropna(subset=["adjacency"])
    out["r_adj"] = (float(np.corrcoef(v["prob"], v["adjacency"])[0, 1])
                    if len(v) > 2 and v["adjacency"].std() > 0 and v["prob"].std() > 0 else np.nan)
    tot = e["prob"].sum()
    lowmask = e["contact"] < low
    out["mass_low"] = float(e.loc[lowmask, "prob"].sum() / tot) if tot > 0 else np.nan
    return out


def _spot_group_means(spot_counts, dom, gene):
    """Mean CPM of one gene (by NAME) per spot group — the quantity a CCI tool's gene filter reads."""
    cpm = spot_counts.div(spot_counts.sum(1), axis=0) * 1e4
    return pd.Series(cpm[gene].values).groupby(np.asarray(dom)).mean()


def _net_matrix(net, groups):
    """Inferred sender x receiver strength (summed prob), on a fixed group order."""
    m = pd.DataFrame(0.0, index=groups, columns=groups)
    if net is None or not len(net):
        return m
    g = net.groupby(["source", "target"])["prob"].sum()
    for (src, tgt), v in g.items():
        if src in m.index and tgt in m.columns:
            m.loc[src, tgt] = v
    return m


def _bubble(ax, net, pair_prob, wired_name, title):
    if net is None or not len(net):
        ax.text(0.5, 0.5, "not run", ha="center", va="center"); ax.set_title(title); return
    top = list(pair_prob.index[:8])
    sub = net[net["interaction_name"].isin(top)].copy()
    sub["route"] = sub["source"].astype(str) + "\u2192" + sub["target"].astype(str)
    routes = sorted(sub["route"].unique())
    pv = sub["pval"] if "pval" in sub.columns else pd.Series(0.05, index=sub.index)
    sizes = 14 + 70 * (-np.log10(np.clip(pv.values, 1e-4, 1.0)) / 4.0)
    sp = ax.scatter([routes.index(r) for r in sub["route"]],
                    [top.index(n) for n in sub["interaction_name"]],
                    c=sub["prob"].values, s=sizes, cmap="viridis")
    ax.set_xticks(range(len(routes))); ax.set_xticklabels(routes, rotation=90, fontsize=5)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([("* " + n) if n == wired_name else n for n in top], fontsize=6)
    ax.set_title(title)
    return sp


def _roles(ax, net, emitter, title):
    if net is None or not len(net):
        ax.text(0.5, 0.5, "not run", ha="center", va="center"); ax.set_title(title); return
    o = net.groupby("source")["prob"].sum(); i = net.groupby("target")["prob"].sum()
    gs = sorted(set(o.index) | set(i.index))
    ox = o.reindex(gs).fillna(0.0).values; iy = i.reindex(gs).fillna(0.0).values
    ax.scatter(ox, iy, c=["#d1495b" if g == emitter else "#6c8ebf" for g in gs], s=60)
    for g, x_, y_ in zip(gs, ox, iy):
        ax.annotate(g, (x_, y_), fontsize=6, xytext=(3, 3), textcoords="offset points")
    lim = max(ox.max(), iy.max(), 1e-9) * 1.15
    ax.plot([0, lim], [0, lim], ls="--", lw=0.8, c="0.6")
    ax.set(xlabel="outgoing (sum prob)", ylabel="incoming (sum prob)", title=title)


def _figure(out, cc, wired, pair_prob, wired_name, tumor, section, net, dom,
            spot_counts, truth, gene_names, emitter, truth_only,
            net_rna=None, pair_prob_rna=None, adj=None, fid=None):
    """GROUND TRUTH (top, from the spatial section) vs what CellChat recovers from dissociated scRNA
    (bottom). The question is AMENABILITY: is the planted channel present in iscc's output in the form
    a real CCI tool reads? The spatial section is here to SHOW the truth in tissue, not to be a second
    inference arm."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lig_name = gene_names[int(truth["cci_ligand"])]
    rec_name = gene_names[int(truth["cci_receptor"])]
    groups = sorted(set(np.asarray(dom).tolist()))
    me = tumor.cell_data["cell_microenv"]["cci_level"]
    lvl_by = pd.Series(me.values, index=tumor.cell_data["cell_type"].index)
    spot_lvl = np.array([lvl_by.reindex(list(m)).mean() if len(m) else np.nan
                         for m in section["members"]])

    nrow = 1 if truth_only else 2
    fig, axes = plt.subplots(nrow, 3, figsize=(16.5, 4.9 * nrow), squeeze=False)

    # ---------------- ROW 1 — GROUND TRUTH, drawn from the spatial section --------------------
    ax = axes[0]
    x = np.arange(len(groups)); w = 0.38
    lig_g = _spot_group_means(spot_counts, dom, lig_name).reindex(groups)
    rec_g = _spot_group_means(spot_counts, dom, rec_name).reindex(groups)
    ax[0].bar(x - w/2, lig_g.values, w, label=f"ligand {lig_name}", color="#d1495b")
    ax[0].bar(x + w/2, rec_g.values, w, label=f"receptor {rec_name}", color="#2a9d8f")
    ax[0].set_xticks(x); ax[0].set_xticklabels(groups, rotation=45, ha="right", fontsize=7)
    ax[0].legend(fontsize=7)
    ax[0].set(ylabel="mean CPM per spot group",
              title=f"A. GROUND TRUTH — the wired pair\n(ligand marks the sender '{emitter}')")

    gt = _true_matrix(dom, section, tumor, emitter)
    im = ax[1].imshow(gt.reindex(index=groups, columns=groups).values, cmap="magma", aspect="auto")
    ax[1].set_xticks(range(len(groups))); ax[1].set_xticklabels(groups, rotation=45, ha="right", fontsize=7)
    ax[1].set_yticks(range(len(groups))); ax[1].set_yticklabels(groups, fontsize=7)
    fig.colorbar(im, ax=ax[1], fraction=0.046)
    ax[1].set(xlabel="receiver", ylabel="sender",
              title="B. GROUND TRUTH — sender x receiver\n(only the emitter sends)")

    # Robust colour limits: the receiver population (which carries the receptor) is a small minority
    # here — a confluent IDC has consumed the gland walls, leaving epithelium vestigial — so a handful
    # of high spots otherwise flatten the whole section to black.
    finite = spot_lvl[np.isfinite(spot_lvl)]
    vmax = float(np.percentile(finite, 98)) if finite.size else 1.0
    vmax = vmax if vmax > 0 else float(finite.max() if finite.size else 1.0)
    sc = ax[2].scatter(section["coords"][:, 1], section["coords"][:, 0], c=spot_lvl,
                       cmap="magma", s=5, vmin=0.0, vmax=vmax)
    ax[2].invert_yaxis(); ax[2].set_aspect("equal")
    fig.colorbar(sc, ax=ax[2], fraction=0.046, extend="max")
    pct = 100.0 * float(np.mean(finite > 0)) if finite.size else 0.0
    ax[2].set(title=f"C. GROUND TRUTH in tissue — received CCI signal\n({len(spot_lvl)} spots; "
                    f"{pct:.0f}% carry signal, colour clipped at the 98th pct)")

    if truth_only:
        fig.suptitle("W0+W3 GROUND TRUTH: the planted L-R channel", fontsize=12, y=1.02)
        fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight")
        print("figure ->", out); return

    # ---------------- ROW 2 — what a real CCI tool recovers from dissociated scRNA -------------
    n, pp = (net_rna, pair_prob_rna) if (net_rna is not None and len(net_rna)) else (net, pair_prob)
    ax = axes[1]

    if pp is not None and len(pp):
        vals = pp.values
        names = list(pp.index)
        cols = ["#d1495b" if nm == wired_name else "#6c8ebf" for nm in names]
        ax[0].bar(range(len(vals)), vals, color=cols)
        rank = (names.index(wired_name) + 1) if wired_name in names else None
        sub = (f"the wired pair ranks #{rank} of {len(vals)} screened"
               if rank else "wired pair not scored")
        ax[0].set(xlabel="L-R pair (sorted by prob)", ylabel="communication probability",
                  title=f"D. RECOVERED from scRNA — {sub}\n(red = wired; score by prob, not p-value)")
    else:
        ax[0].text(0.5, 0.5, "CellChat not run", ha="center", va="center")

    _bubble(ax[1], n, pp, wired_name, "E. bubble plot — the field-standard view\n(* = the wired pair)")
    _roles(ax[2], n, emitter, f"F. signalling roles (red = '{emitter}')")

    fig.suptitle("W0+W3: iscc's planted L-R channel (top: GROUND TRUTH, shown on the Visium section) "
                 "and its recovery by real CellChat from dissociated scRNA (bottom)",
                 fontsize=12, y=1.005)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("figure ->", out); return

    # ---------- ROWS 2/3 — the two arms, same layout so they read against row 1 ----------
    for row, (n, pp, label) in enumerate(
            ((net, pair_prob, "SPATIAL (proximity MEASURED)"),
             (net_rna, pair_prob_rna, "scRNA (proximity ASSUMED)")), start=1):
        ax = axes[row]
        m = _net_matrix(n, groups)
        im = ax[0].imshow(m.values, cmap="magma", aspect="auto")
        ax[0].set_xticks(range(len(groups))); ax[0].set_xticklabels(groups, rotation=90, fontsize=6)
        ax[0].set_yticks(range(len(groups))); ax[0].set_yticklabels(groups, fontsize=6)
        fig.colorbar(im, ax=ax[0], fraction=0.046)
        extra = ""
        f = (fid or {}).get("spatial" if row == 1 else "scRNA")
        if f is not None:
            extra = (f"\nr(TRUE)={f['r_true']:+.2f}  r(adjacency)={f['r_adj']:+.2f}"
                     f"  mass off-contact={f['mass_low']:.0%}")
        ax[0].set(xlabel="receiver", ylabel="sender",
                  title=f"{'DG'[row-1]}. INFERRED — {label}{extra}")
        _bubble(ax[1], n, pp, wired_name, f"{'EH'[row-1]}. bubble plot (* = wired)")
        _roles(ax[2], n, emitter, f"{'FI'[row-1]}. signalling roles (red = '{emitter}')")

    fig.suptitle("W0+W3: the planted L-R channel — GROUND TRUTH (top) vs CellChat with positions "
                 "(middle) and without (bottom)", fontsize=12, y=1.005)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("figure ->", out); return

    # ---------------- ROW 2 — WHAT CELLCHAT INFERS -----------------------------------------
    ax = axes[1]
    # D. clone-correlation over candidates (the emergent confound), wired highlighted
    order = np.argsort(cc["eta_max"].values)
    colors = ["#d1495b" if cc["wired"].values[i] else "#6c8ebf" for i in order]
    ax[0].bar(range(len(order)), cc["eta_max"].values[order], color=colors)
    ax[0].set(xlabel="candidate pair (sorted)", ylabel="clone-correlation  eta_max",
              title="D. MEASURED clone-correlation over candidates\n(red = wired; emergent, not planted)")

    # E. the field-standard BUBBLE PLOT: L-R pair x sender->receiver, colour = prob, size = -log10(p)
    if net is not None and len(net):
        top = list(pair_prob.index[:8])
        sub = net[net["interaction_name"].isin(top)].copy()
        sub["route"] = sub["source"].astype(str) + "\u2192" + sub["target"].astype(str)
        routes = sorted(sub["route"].unique())
        pv = sub["pval"] if "pval" in sub.columns else pd.Series(0.05, index=sub.index)
        sizes = 18 + 90 * (-np.log10(np.clip(pv.values, 1e-4, 1.0)) / 4.0)
        xs = [routes.index(r) for r in sub["route"]]
        ys = [top.index(n) for n in sub["interaction_name"]]
        sp = ax[1].scatter(xs, ys, c=sub["prob"].values, s=sizes, cmap="viridis")
        fig.colorbar(sp, ax=ax[1], fraction=0.046, label="communication probability")
        ax[1].set_xticks(range(len(routes)))
        ax[1].set_xticklabels(routes, rotation=90, fontsize=6)
        ax[1].set_yticks(range(len(top)))
        ax[1].set_yticklabels([("* " + n) if n == wired_name else n for n in top], fontsize=7)
        ax[1].set(title="E. CellChat bubble plot (top pairs)\n(* = the wired pair; size = -log10 p)")
    else:
        ax[1].text(0.5, 0.5, "CellChat not run", ha="center", va="center")

    # F. signalling ROLE scatter: outgoing vs incoming strength per group. The planted channel makes
    #    the emitter a pure SENDER, so it should sit hard against the outgoing axis.
    if net is not None and len(net):
        out_s = net.groupby("source")["prob"].sum()
        in_s = net.groupby("target")["prob"].sum()
        gs = sorted(set(out_s.index) | set(in_s.index))
        ox = out_s.reindex(gs).fillna(0.0).values
        iy = in_s.reindex(gs).fillna(0.0).values
        cols = ["#d1495b" if g == emitter else "#6c8ebf" for g in gs]
        ax[2].scatter(ox, iy, c=cols, s=70)
        for g, a, b in zip(gs, ox, iy):
            ax[2].annotate(g, (a, b), fontsize=7, xytext=(3, 3), textcoords="offset points")
        lim = max(ox.max(), iy.max()) * 1.15 + 1e-9
        ax[2].plot([0, lim], [0, lim], ls="--", lw=0.8, c="0.6")
        ax[2].set(xlabel="outgoing strength (sum prob)", ylabel="incoming strength (sum prob)",
                  title=f"F. signalling roles\n(red = '{emitter}', the planted sender)")
    else:
        ax[2].text(0.5, 0.5, "CellChat not run", ha="center", va="center")

    fig.suptitle("W0+W3: iscc's planted L-R channel (top: GROUND TRUTH) vs real CellChat at Visium "
                 "(bottom: INFERRED)", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("figure ->", out)


if __name__ == "__main__":
    main()
