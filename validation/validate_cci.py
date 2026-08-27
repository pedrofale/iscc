"""Validate W0 + W3 (DESIGN_cci_spatial.md): iscc's OWN ligand-receptor database + receptor-dependent
cell-cell communication, taken END TO END through the REAL CellChat at Visium resolution.

WHAT THIS SHOWS
  1. **The database (W0).** One tumour grown with an F8 CCI channel active emits `n_candidate_pairs`
     candidate L-R pairs over its own abstract gene ids. ONE is wired into F8 (`active`); the rest are
     unwired decoys (`candidate`). No decoy is engineered — pairs drawn at random already land on
     clone-varying CNA segments by chance, so the third class, `clone-correlated`, is MEASURED per
     candidate (`iscc.integrations.clone_correlation`), not planted. We report that distribution.
  2. **Receptor-dependence (W3).** The per-cell received signal is `ligand_avail[deme]·receptor[cell]`,
     so it varies cell-to-cell by clone within a deme — the ground truth surfaced as `cell_microenv`.
  3. **The recoverability check.** The written database is re-read into CellChat (the round trip of
     `validation/README_cellchat.md`), the Visium section is run through the spatial pipeline, and we
     ask: does CellChat rank the WIRED pair above the decoys BY COMMUNICATION PROBABILITY? We score by
     `prob`, not by p-value — on group-averaged expression every pair gets a "significant" edge, so a
     dense all-significant network is the expected background (README §8.7).

WHAT THE FIGURE SHOWS. Top row = GROUND TRUTH (the wired pair's L/R by group, the true
sender x receiver matrix -- a single non-zero row, because only the emitter sends -- and the received
signal in space). Bottom row = what CellChat INFERS: the measured clone-correlation over candidates,
the field-standard bubble plot (L-R pair x sender->receiver, colour = prob, size = -log10 p), and the
signalling-role scatter (outgoing vs incoming strength per group).

READ THE BUBBLE PLOT AND THE ROLE SCATTER, NOT JUST THE RANK. Ranking the wired pair #1 says the PAIR
was recovered; it says nothing about DIRECTION. Ground truth is "immune sends to everyone", and the
role scatter shows CellChat naming epithelial the dominant sender instead -- because the ligand is
only ~3x higher in immune than in epithelial, and `prob` multiplies the source's ligand by the
target's receptor, so a group with moderate ligand AND high receptor scores well as a sender. Pair-
level recovery and direction-level recovery are different claims.

UNITS. iscc Visium coordinates are in DEME units; a deme anchors at ~25 um (memory: deme = 3D column
physical scale). We author section coordinates in MICROMETRES (deme × DEME_MICRONS) so CellChat's
`spatial.factors` ratio=1 path applies and the runner computes `scale.distance` from the geometry.

Env: CellChat runs in the dedicated `iscc-cellchat` env (README_cellchat.md); this script stays in the
core `iscc` env and shells out to `cellchat_runner.R`. If that env is absent, the CellChat panel is
skipped and the database + clone-correlation panels still render.

Writes manuscript/figures/validation_cci.png.
Run:  python -u validation/validate_cci.py
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "validation"))
sys.path.insert(0, os.path.join(REPO, "src"))

HOME = os.path.expanduser("~")
CELLCHAT_RSCRIPT = os.environ.get(
    "ISCC_CELLCHAT_RSCRIPT", os.path.join(HOME, "miniconda3/envs/iscc-cellchat/bin/Rscript"))
DEME_MICRONS = 25.0                        # deme -> um (physical anchor; see UNITS above)


def cellchat_available():
    return os.path.exists(CELLCHAT_RSCRIPT)


def grow(seed, steps, n_candidate_pairs, strength=1.0, emitter="immune"):
    """Grow the shared four-type spatial tumour (deconv_common) with an F8 CCI channel active."""
    import deconv_common as D
    from iscc.tumor.models import GenotypeTumor
    microenv = {"cci": {"strength": float(strength), "n_target_genes": 30, "emitter_type": str(emitter),
                        "lengthscale": 3.0, "n_candidate_pairs": int(n_candidate_pairs)}}
    t = GenotypeTumor(seed=seed, genome_params=D.GENOME, selection_params=D.SELECTION,
                      cancer_cell_params=D.CANCER, deme_params=D.DEME, spatial_params=D.SPATIAL,
                      expression_params=D.expression_params(), microenv_params=microenv)
    t.grow(n_steps=steps, seed=seed)
    t.make_cell_data()
    return t


def run_cellchat(work_dir):
    """Write CSVs already staged in work_dir through cellchat_runner.R; return the net DataFrame."""
    runner = os.path.join(REPO, "validation", "cellchat_runner.R")
    out_csv = os.path.join(work_dir, "cellchat_net.csv")
    r = subprocess.run([CELLCHAT_RSCRIPT, runner, work_dir, out_csv],
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
    ap.add_argument("--out", default=os.path.join(REPO, "manuscript/figures/validation_cci.png"))
    ap.add_argument("--workdir", default=os.path.join(REPO, "validation", "_cci_tmp"))
    ap.add_argument("--truth-only", action="store_true",
                    help="render the GROUND-TRUTH row only; skip the CellChat run")
    args = ap.parse_args()

    import deconv_common as D
    from iscc.integrations import cci_database, write_cci_database, clone_correlation

    os.makedirs(args.workdir, exist_ok=True)
    t = grow(args.seed, args.steps, args.pairs, args.strength, args.emitter)
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
    section = D.build_section(t, spot_radius=0.9, spot_pitch=1.5, section_radius=10.0, seed=args.seed)
    spot_counts = section["spot_counts"]                      # spots x genes
    coords_um = section["coords"] * DEME_MICRONS              # spots x 2, in um
    # per-spot group = dominant population label (normals by type, cancer by CNA clone)
    clone_cats = section["clone_categories"]
    dom = np.asarray(clone_cats)[section["true_clone"].argmax(1)]
    spot_names = section["spot_names"]
    print(f"section: {len(spot_names)} spots, {spot_counts.shape[1]} genes, "
          f"{len(set(dom))} groups")

    net = None
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
            net = run_cellchat(args.workdir)
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

    _figure(args.out, cc, wired, pair_prob, wired_name, t, section, net, dom,
            spot_counts, truth, gene_names, args.emitter, args.truth_only)


def _spot_group_means(spot_counts, dom, gene):
    """Mean CPM of one gene (by NAME) per spot group — the quantity a CCI tool's gene filter reads."""
    cpm = spot_counts.div(spot_counts.sum(1), axis=0) * 1e4
    return pd.Series(cpm[gene].values).groupby(np.asarray(dom)).mean()


def _figure(out, cc, wired, pair_prob, wired_name, tumor, section, net, dom,
            spot_counts, truth, gene_names, emitter, truth_only):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nrow = 1 if truth_only else 2
    fig, axes = plt.subplots(nrow, 3, figsize=(16.5, 4.8 * nrow), squeeze=False)
    ax = axes[0]
    lig_name = gene_names[int(truth["cci_ligand"])]
    rec_name = gene_names[int(truth["cci_receptor"])]
    groups = sorted(set(np.asarray(dom).tolist()))

    # ---------------- ROW 1 — GROUND TRUTH -------------------------------------------------
    # A. the mechanism: the wired ligand marks the SENDER type, the receptor the RECEIVERS.
    lig_g = _spot_group_means(spot_counts, dom, lig_name).reindex(groups)
    rec_g = _spot_group_means(spot_counts, dom, rec_name).reindex(groups)
    x = np.arange(len(groups)); w = 0.38
    ax[0].bar(x - w/2, lig_g.values, w, label=f"ligand {lig_name}", color="#d1495b")
    ax[0].bar(x + w/2, rec_g.values, w, label=f"receptor {rec_name}", color="#2a9d8f")
    ax[0].set_xticks(x); ax[0].set_xticklabels(groups, rotation=45, ha="right")
    ax[0].legend(fontsize=8)
    ax[0].set(ylabel="mean CPM per spot group",
              title=f"A. GROUND TRUTH — the wired pair's L/R\n(ligand marks the sender '{emitter}';"
                    " receptor marks receivers)")

    # B. the ground-truth sender -> receiver matrix. Only the emitter type sends, and each receiver
    #    group's weight is its mean RECEIVED signal (ligand availability x its own receptor level).
    me = tumor.cell_data["cell_microenv"]["cci_level"]
    idx = tumor.cell_data["cell_type"].index
    lvl_by = pd.Series(me.values, index=idx)
    members = section["members"]
    spot_lvl = np.array([lvl_by.reindex(list(m)).mean() if len(m) else np.nan for m in members])
    gt = pd.DataFrame(0.0, index=groups, columns=groups)
    if emitter in gt.index:
        for g in groups:
            sel = (np.asarray(dom) == g)
            gt.loc[emitter, g] = np.nanmean(spot_lvl[sel]) if sel.any() else 0.0
    im = ax[1].imshow(gt.values, cmap="magma", aspect="auto")
    ax[1].set_xticks(range(len(groups))); ax[1].set_xticklabels(groups, rotation=45, ha="right")
    ax[1].set_yticks(range(len(groups))); ax[1].set_yticklabels(groups)
    fig.colorbar(im, ax=ax[1], fraction=0.046)
    ax[1].set(xlabel="receiver group", ylabel="sender group",
              title="B. GROUND TRUTH — sender x receiver\n(only the emitter sends; weight = received signal)")

    # C. the ground-truth received signal in space
    coords = section["coords"]
    sc = ax[2].scatter(coords[:, 1], coords[:, 0], c=spot_lvl, cmap="magma", s=28)
    ax[2].invert_yaxis(); fig.colorbar(sc, ax=ax[2], fraction=0.046)
    ax[2].set(title="C. GROUND TRUTH — received CCI signal per spot\n(ligand availability x receptor — W3)")

    if truth_only:
        fig.suptitle("W0+W3 GROUND TRUTH: the planted L-R channel in iscc's simulated Visium section",
                     fontsize=12, y=1.02)
        fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight")
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
