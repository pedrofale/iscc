"""Generate the analysis-ready datasets the data-analysis notebooks load.

WHY THIS EXISTS. The tool-benchmark notebooks used to grow their own tumour with `iscc` and then run
the tool in the same notebook. That forces the notebook's kernel to be Python, which in turn forces an
R tool (clonealign, Numbat, RCTD, TreeMHN) to be invoked through a subprocess rather than written as
R. Separating GENERATION from ANALYSIS removes that constraint: this script does the `iscc` part once
and writes plain tables, so an analysis notebook only ever *loads* — and a notebook that only loads
can be an R notebook, with an R kernel and ordinary R code.

It also makes the benchmarks honest in a second way: the analysis side cannot accidentally peek at
anything the tool would not have. What lands on disk is exactly the tool's input plus a separate
ground-truth file used only for scoring.

REGIME. `clonealign` and `numbat` come from the realistic breach-gated ductal field
(`realistic_regime.py`) — grid 96 / 5 glands / 6,000 genes at `scale="mid"`, versus the old toy rig's
grid 20, one ring, 600 genes.

**`rctd` does NOT yet**, and `treemhn` is a different kind of dataset entirely (a cohort of mutation
trees, no assay). `rctd` still grows through `deconv_common.grow_tumor`, whose substrate is grid 26 /
one ring / 10 x 30 = 300 genes, because the Visium section geometry (`build_section`: spot radius,
pitch, section radius) is tuned to that grid and does not transfer unchanged. Migrating it is
outstanding — see the per-function note in `_rctd`.

Output (default `analysis_data/`, gitignored — the matrices are tens of MB):

    analysis_data/
      manifest.json               what was generated, with params, seeds and versions
      clonealign/
        Y.csv.gz                  cells x genes   scRNA counts        (tool input)
        L.csv.gz                  genes x clones  copy number         (tool input)
        truth.csv                 cell -> true clone                  (SCORING ONLY)
        meta.json                 shapes, clone sizes, the CN consensus

CSV(.gz) on purpose: `read.csv` in R reads `.gz` directly, so the same file serves a Python and an R
notebook without either needing the other's stack.

Usage:  python validation/make_analysis_data.py [--out analysis_data] [--seed 3] [--scale mid]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)


def _clonealign(out_dir, seed, scale, n_clones):
    """scRNA counts + per-gene clone copy number + the true clone of each cell."""
    import integration_common as C

    t = C.grow_tumor(seed=seed, regime="realistic", scale=scale)
    inp = C.build_clonealign_inputs(t, n_clones=n_clones, seed=0)
    Y, L, labels = inp["Y"], inp["L"], np.asarray(inp["labels"])

    os.makedirs(out_dir, exist_ok=True)
    Y.to_csv(os.path.join(out_dir, "Y.csv.gz"))
    L.to_csv(os.path.join(out_dir, "L.csv.gz"))
    # Ground truth lives in its OWN file: the tool reads Y and L, never this.
    pd.DataFrame({"cell": Y.index, "true_clone": labels}).to_csv(
        os.path.join(out_dir, "truth.csv"), index=False)

    sizes = np.bincount(labels, minlength=L.shape[1]).tolist()
    meta = dict(
        n_cells=int(Y.shape[0]), n_genes=int(Y.shape[1]), n_clones=int(L.shape[1]),
        cn_informative_genes=int((L.nunique(axis=1) > 1).sum()),
        clone_sizes=sizes,
        majority_baseline=float(max(sizes) / sum(sizes)),
        chance_baseline=float(1.0 / L.shape[1]),
        # The consensus is what makes the difficulty legible: on the realistic WGD+ field the clones
        # differ in only one or two of the twelve segments, so a pure dosage model has little to go on.
        cn_consensus=inp["consensus"].astype(int).tolist(),
    )
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    return meta


def _numbat(out_dir, seed, scale, n_clones):
    """Numbat's file inputs: expression counts, a normal reference, phased allele counts, a GTF."""
    import integration_common as C
    import deconv_common as D

    # Numbat is ALLELE-AWARE: it reads per-homolog RNA (`cell_rna_baf`), which only exists when the
    # tumour is grown with allele-specific expression params. Without them build_numbat_inputs fails
    # with KeyError: 'cell_rna_baf'.
    t = C.grow_tumor(seed=seed, regime="realistic", scale=scale,
                     expression=D.expression_params(allele_specific=True))
    shared = C.build_cna_inputs(t, n_normal=150, n_clones=n_clones, seed=seed)
    os.makedirs(out_dir, exist_ok=True)
    # build_numbat_inputs writes count_mat.csv / ref_counts.csv / df_allele.csv / gtf.csv itself —
    # exactly the layout numbat_runner.R reads, so the dataset dir IS the runner's work_dir.
    info = C.build_numbat_inputs(t, shared, out_dir, seed=seed)
    # Ground truth for scoring: which cells are malignant, and which clone each belongs to.
    # (build_numbat_inputs returns these as arrays; it has no single "truth" object.)
    # `is_malignant` is per CELL (aligned to cell_ids); `clone_labels` is per MALIGNANT cell
    # (aligned to mal_cells). They are different lengths by design, so map rather than zip.
    clone_by_cell = dict(zip(info["mal_cells"], np.asarray(info["clone_labels"]).tolist()))
    cell_ids = list(info["cell_ids"])
    pd.DataFrame({"cell": cell_ids,
                  "is_malignant": np.asarray(info["is_malignant"]).astype(int),
                  "clone": [clone_by_cell.get(c, -1) for c in cell_ids]}).to_csv(
        os.path.join(out_dir, "truth.csv"), index=False)
    np.savetxt(os.path.join(out_dir, "truth_consensus_cn.csv"),
               np.asarray(info["consensus"]), fmt="%d", delimiter=",")
    meta = dict(n_clones=int(n_clones), n_cells=int(len(info["cell_ids"])),
                n_genes=int(len(info["genes"])), n_segments=int(info["n_segments"]),
                files=sorted(f for f in os.listdir(out_dir) if f.endswith(".csv")))
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    return meta


def _rctd(out_dir, seed, scale, n_clones):
    """RCTD's inputs: a spatial section (counts + coordinates) and a paired scRNA reference.

    NOT YET MIGRATED to the realistic regime. This grows `deconv_common`'s own substrate — grid 26,
    one epithelial ring, 300 genes — not the ductal field. The blocker is geometric rather than
    conceptual: `build_section`'s spot radius / pitch / section radius are tuned to that grid, so
    pointing it at a grid-96 ductal field would silently change how many cells land in a spot, which
    is the quantity the whole deconvolution benchmark is about. Migrate the geometry with it.
    """
    import deconv_common as D

    t = D.grow_tumor(seed=seed)
    section = D.build_section(t, spot_radius=0.9, spot_pitch=1.5, section_radius=9.0)
    ref = D.build_reference(t, section, mode="oracle", n_cells=450, seed=11)
    os.makedirs(out_dir, exist_ok=True)
    # _write_csvs lays down ref_counts / ref_labels / sp_counts / sp_coords — the four files
    # rctd_runner.R expects.
    D._write_csvs(ref, section, out_dir)
    # True per-spot composition — what RCTD is scored against — in its own file. `true_type` is the
    # spots x cell-type proportion matrix; `true_clone` the same over CNA clones.
    pd.DataFrame(np.asarray(section["true_type"]), index=section["spot_names"],
                 columns=[str(c) for c in section["type_categories"]]).to_csv(
        os.path.join(out_dir, "truth_type_composition.csv"))
    pd.DataFrame(np.asarray(section["true_clone"]), index=section["spot_names"],
                 columns=[str(c) for c in section["clone_categories"]]).to_csv(
        os.path.join(out_dir, "truth_clone_composition.csv"))
    meta = dict(n_spots=int(len(section["spot_names"])),
                n_ref_cells=int(ref["counts"].shape[0]),
                cell_types=[str(c) for c in section["type_categories"]],
                clone_categories=[str(c) for c in section["clone_categories"]])
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    return meta


def _treemhn(out_dir, seed, scale, n_clones):
    """TreeMHN's input: one mutation tree per patient, plus the planted dependency network."""
    import validate_epistasis as VE
    from iscc.integrations import progression as ig
    from iscc.constants import DEFAULT_LAYOUT_SEED

    # A cohort grown over a PLANTED epistasis network — that network is the ground truth. Injecting a
    # non-zero interaction strength is what gives TreeMHN something to recover.
    strength = 2.0
    net = VE.epi(n_interactions=2, strength=strength)
    tumors = VE.run_cohort(20, net, seed0=1 + seed, layout_seed=DEFAULT_LAYOUT_SEED,
                           inject_E=strength)
    trees = ig.to_treemhn_trees(tumors)
    os.makedirs(out_dir, exist_ok=True)
    trees.to_csv(os.path.join(out_dir, "trees.csv"), index=False)
    X = ig.to_mhn_matrix(tumors)                 # MHN's binary-presence view of the same cohort
    X.to_csv(os.path.join(out_dir, "X_presence.csv"))
    with open(os.path.join(out_dir, "truth_network.json"), "w") as fh:
        json.dump({"epistasis_params": net, "interaction_strength": strength}, fh, indent=2)
    meta = dict(n_patients=int(len(tumors)), n_tree_rows=int(trees.shape[0]),
                n_events=int(VE.N_EVENTS), interaction_strength=strength)
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    return meta


DATASETS = {"clonealign": _clonealign, "numbat": _numbat,
            "rctd": _rctd, "treemhn": _treemhn}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "analysis_data"))
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--scale", default="mid", choices=["small", "mid", "cm"])
    ap.add_argument("--clones", type=int, default=4)
    ap.add_argument("--only", default="", help="comma-separated dataset names")
    args = ap.parse_args()

    wanted = [d.strip() for d in args.only.split(",") if d.strip()] or list(DATASETS)
    os.makedirs(args.out, exist_ok=True)
    manifest = {}
    for name in wanted:
        t0 = time.time()
        print(f"generating {name} (regime=realistic, scale={args.scale}, seed={args.seed}) ...",
              flush=True)
        meta = DATASETS[name](os.path.join(args.out, name), args.seed, args.scale, args.clones)
        meta.update(seed=args.seed, scale=args.scale, regime="realistic",
                    seconds=round(time.time() - t0, 1))
        manifest[name] = meta
        # meta differs per dataset (a spatial section has spots, a cohort has patients), so summarise
        # whatever the generator reported rather than assuming one shape.
        shape = ", ".join(f"{k}={v:,}" if isinstance(v, int) else f"{k}={v}"
                          for k, v in meta.items()
                          if isinstance(v, (int, float)) and k != "seconds")
        print(f"  {name}: {shape} in {meta['seconds']}s", flush=True)

    try:
        import iscc
        version = getattr(iscc, "__version__", "unknown")
    except Exception:
        version = "unknown"
    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump({"iscc_version": version, "datasets": manifest}, fh, indent=2)
    print(f"\nmanifest -> {os.path.join(args.out, 'manifest.json')}")


if __name__ == "__main__":
    main()
