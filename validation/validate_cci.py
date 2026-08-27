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

CAVEAT WE EXPECT TO CONFIRM (DESIGN_cci_spatial.md, "Caveat if W3 ships alone"). At deme resolution the
planted signal is piecewise-constant over ~20-25 um blocks; Visium spots are ~55 um, so it sits below
the observation scale. Moreover F8 modulates the TARGET genes and READS the ligand/receptor — it never
boosts L or R — while CellChat's `prob` scores L/R group-mean expression only. So the wired pair is not
expected to separate from decoys HERE. That is a reported result, not a failure: everything downstream
(W4) then depends on either a target-aware method (COMMOT/NicheNet) or single-cell spatial resolution
(which also needs W2). We state it plainly.

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
    if cellchat_available():
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

    _figure(args.out, cc, wired, pair_prob, wired_name, t, section)


def _figure(out, cc, wired, pair_prob, wired_name, tumor, section):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

    # A. clone-correlation distribution over candidates, wired highlighted
    order = np.argsort(cc["eta_max"].values)
    colors = ["#d1495b" if cc["wired"].values[i] else "#6c8ebf" for i in order]
    ax[0].bar(range(len(order)), cc["eta_max"].values[order], color=colors)
    ax[0].set(xlabel="candidate pair (sorted)", ylabel="clone-correlation  eta_max",
              title="A. MEASURED clone-correlation over candidates\n(red = the wired pair; emergent, not planted)")

    # B. CellChat per-pair communication probability, wired highlighted
    if pair_prob is not None and len(pair_prob):
        vals = pair_prob.values
        names = list(pair_prob.index)
        cols = ["#d1495b" if n == wired_name else "#6c8ebf" for n in names]
        ax[1].bar(range(len(vals)), vals, color=cols)
        rank = (names.index(wired_name) + 1) if wired_name in names else None
        sub = f"wired ranks #{rank}/{len(vals)}" if rank else "wired pair DROPPED by CellChat"
        ax[1].set(xlabel="L-R pair (sorted by prob)", ylabel="communication probability (max over group pairs)",
                  title=f"B. CellChat recoverability at Visium\n{sub} — score by prob, not p-value")
    else:
        ax[1].text(0.5, 0.5, "CellChat not run\n(iscc-cellchat env absent)", ha="center", va="center")
        ax[1].set_title("B. CellChat recoverability at Visium")

    # C. the ground-truth received signal over the section (per spot, mean of member cells)
    me = tumor.cell_data["cell_microenv"]["cci_level"]
    members = section["members"]
    coords = section["coords"]
    idx = tumor.cell_data["cell_type"].index
    lvl_by = pd.Series(me.values, index=idx)
    spot_lvl = np.array([lvl_by.reindex(list(m)).mean() if len(m) else np.nan for m in members])
    sc = ax[2].scatter(coords[:, 1], coords[:, 0], c=spot_lvl, cmap="magma", s=28)
    ax[2].invert_yaxis(); fig.colorbar(sc, ax=ax[2], fraction=0.046)
    ax[2].set(title="C. ground-truth received CCI signal per spot\n(ligand availability × receptor — W3)")

    fig.suptitle("W0+W3: iscc's own L-R database + receptor-dependent CCI, through real CellChat at "
                 "Visium resolution", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("figure ->", out)


if __name__ == "__main__":
    main()
