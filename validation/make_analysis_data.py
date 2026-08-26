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

`rctd` is migrated too (2026-08-26), with the section geometry re-anchored physically (Visium v1 at
~50 um/deme) and the field grown dense enough for spots to be genuine mixtures — see `_rctd`.
`treemhn` is a different kind of dataset entirely: a cohort of mutation trees, no assay.

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


# Visium v1 geometry in DEME units, from DESIGN_ductal_field.md: a deme is ~50 um in-plane, so the
# 55 um spot is ~1 deme across (radius 0.55) and the 100 um centre-to-centre pitch is 2 demes. The old
# toy values (radius 0.9, pitch 1.5) are NOT a physical target to preserve — at 50 um/deme they
# describe 180 um spots overlapping at 75 um pitch, which no Visium slide has.
VISIUM_V1 = dict(spot_radius=0.55, spot_pitch=2.0, section_radius=None)
# Cancer-cell target for the deconvolution field. Density is what decides whether this benchmark
# means anything: a spot must contain a MIXTURE. Measured on the realistic field at Visium v1 geometry
#   target   cells/spot   spot purity   spots with >=2 types
#    1.5k        1.7          0.98              6%     <- degenerate: nothing to deconvolve
#     20k        4.9          0.96             11%
#     60k        5.0          0.89             34%     <- comparable to the old toy rig (0.88 / 60%)
RCTD_TARGET_CANCER = 60_000
# Immune density. The ductal field ships with no immune compartment, and without one the section is
# essentially cancer + stroma: RCTD then fits TWO types and the benchmark is much weaker than the toy
# rig it replaces. Turning immune on (the same 0.12 deconv_common's rig used) restores a genuinely
# mixed section — and overtakes the toy rig on the measure that matters:
#                          spot purity   >=2 types   >=3 types
#   old toy rig (grid 26)      0.88          60%         10%
#   realistic, no immune       0.89          34%          0%
#   realistic + immune 0.12    0.72          75%         21%
# NOTE epithelial stays vestigial (~0.4%) at this density: a confluent IDC has consumed the gland
# walls. That is biologically coherent, so this is effectively a THREE-type deconvolution.
RCTD_IMMUNE_DENSITY = 0.12


def _rctd(out_dir, seed, scale, n_clones):
    """RCTD's inputs: a spatial section (counts + coordinates) and a paired scRNA reference.

    Migrated to the realistic breach-gated ductal field (2026-08-26). Two things had to move WITH the
    substrate, and neither is a substrate swap:

    * **Geometry.** The section geometry is in deme units, and a deme on the ductal field is ~50 um
      (DESIGN_ductal_field.md), so Visium v1 is a 0.55-deme radius at 2-deme pitch — see VISIUM_V1.
    * **Density.** At the default mid-scale target the field is far too sparse: 1.7 cells per spot,
      98% of spots a single cell type, so RCTD's task degenerates from deconvolution into
      classification and any score is meaningless. Growing to RCTD_TARGET_CANCER restores genuine
      within-spot mixtures.

    Note this resolves ST at DEME resolution: a spot is ~one deme, so each spot's cells are a
    within-deme mixture rather than sub-spot structure. That is the physical tension the design doc
    settles — sub-spot deconvolution would need deme << spot, i.e. ~cell-sized demes, which iscc
    deliberately rejects.
    """
    import deconv_common as D
    import integration_common as C

    t = C.grow_tumor(seed=seed, regime="realistic", scale=scale,
                     target_cancer=RCTD_TARGET_CANCER, max_cells=40_000,
                     spatial={"immune_density": RCTD_IMMUNE_DENSITY},
                     expression=D.expression_params(allele_specific=False))
    section = D.build_section(t, **VISIUM_V1)
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
    import numpy as _np
    T = _np.asarray(section["true_type"]); nmix = (T > 0).sum(1)
    meta = dict(n_spots=int(len(section["spot_names"])),
                n_ref_cells=int(ref["counts"].shape[0]),
                cell_types=[str(c) for c in section["type_categories"]],
                clone_categories=[str(c) for c in section["clone_categories"]],
                cells_per_spot_mean=float(_np.mean(section["n_cells"])),
                # The benchmark is only meaningful where spots are MIXTURES; record it so a
                # degenerate section is visible in the manifest rather than only in the score.
                spot_purity_mean=float(T.max(1).mean()),
                frac_spots_multitype=float((nmix >= 2).mean()))
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    return meta


def _treemhn(out_dir, seed, scale, n_clones):
    """TreeMHN's input: one mutation tree per patient, plus the planted dependency network."""
    import validate_epistasis as VE
    from iscc.integrations import progression as ig
    from iscc.constants import DEFAULT_LAYOUT_SEED

    # A cohort grown over PLANTED ORDERED CONSTRAINTS — a DAG saying which events must precede which.
    #
    # Gating mode is the load-bearing choice. "accessibility" gates the MUTATION PROCESS itself, so
    # the constraint survives into every observable including tree topology, which is what TreeMHN
    # reads. "fitness" gating plants the same DAG but expresses it only through how large the
    # carrying clones grow, leaving no ordering trace — TreeMHN estimates RATES, so it recovers
    # nothing from it (validate_epistasis panel D measures exactly this contrast).
    #
    # These notebooks demonstrate that iscc's output is amenable to the real tools, so the dataset
    # uses the regime where the planted signal is genuinely present in the observable the tool reads.
    # The fitness-gated case is a benchmark result and belongs in validate_epistasis, not here.
    # event_size=8, NOT the default 2. A gated child needs its parent to have occurred first, and at
    # event_size=2 it arises in 0-2 of 40 patients — there is then nothing in the data to recover, no
    # matter how good the tool. validate_epistasis' panel D hits the same wall and makes the same
    # choice: the ordered-constraint half of this benchmark needs a COMMONER alphabet than the
    # presence half. 40 patients for the same reason.
    strength = 2.0
    net = dict(VE.epi(n_interactions=2, strength=strength), event_size=8)
    dep = dict(n_constraints=2, dag_depth=2, dag_branching=1, gating_mode="accessibility")
    tumors = VE.run_cohort(40, net, dependency_params=dep, seed0=1 + seed,
                           layout_seed=DEFAULT_LAYOUT_SEED, inject_E=strength)
    trees = ig.to_treemhn_trees(tumors)
    os.makedirs(out_dir, exist_ok=True)
    trees.to_csv(os.path.join(out_dir, "trees.csv"), index=False)
    X = ig.to_mhn_matrix(tumors)                 # MHN's binary-presence view of the same cohort
    X.to_csv(os.path.join(out_dir, "X_presence.csv"))
    # The REALISED DAG (which event must precede which), not just the spec — this is what a
    # recovered ordering is scored against.
    dag = tumors[0].selection.epistasis.true_dag_edges() if tumors else []
    with open(os.path.join(out_dir, "truth_network.json"), "w") as fh:
        json.dump({"epistasis_params": net, "dependency_params": dep,
                   "interaction_strength": strength,
                   "true_dag_edges": [[int(a), int(b)] for a, b in dag]}, fh, indent=2)
    meta = dict(n_patients=int(len(tumors)), n_tree_rows=int(trees.shape[0]),
                n_events=int(VE.N_EVENTS), interaction_strength=strength,
                gating_mode=dep["gating_mode"], n_constraints=dep["n_constraints"],
                n_dag_edges=int(len(dag)), event_size=int(net["event_size"]))
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    return meta


def _cohort(out_dir, seed, scale, n_clones):
    """A multi-patient scRNA cohort: counts, patient/batch labels, and the true program dictionary.

    The cohort shares ONE program dictionary (every patient draws from the same gene sets) while each
    patient's tumour evolves privately. That is what makes cohort integration measurable: a method
    must pull the shared programs out despite each patient having its own clones, its own copy-number
    landscape and its own batch effect.
    """
    import programs_common as PC

    n_patients = 5
    Xs, obs_rows, loading, Z = [], [], None, []
    for i in range(n_patients):
        t = PC.grow_tumor(seed=seed + i)
        a, z = PC.counts_anndata(t, seed=seed + i, max_cells=300)
        Xs.append(np.asarray(a.X, dtype=np.float32))
        Z.append(np.asarray(z))
        obs_rows += [(f"P{i}", f"batch{i}")] * a.n_obs
        if loading is None:
            # `program_truth` is where iscc records the planted dictionary: the loading matrix
            # (programs x genes) and the per-cell activities. It is the same object validate_programs
            # scores against.
            loading = np.asarray(t.program_truth["loading"])
            var_names = list(a.var_names)
        del t

    X = np.vstack(Xs)
    os.makedirs(out_dir, exist_ok=True)
    cells = [f"C{i}" for i in range(X.shape[0])]
    pd.DataFrame(X, index=cells, columns=var_names).to_csv(
        os.path.join(out_dir, "counts.csv.gz"))
    pd.DataFrame(obs_rows, index=cells, columns=["patient", "batch"]).to_csv(
        os.path.join(out_dir, "obs.csv"))
    # Ground truth: the shared program dictionary, and each cell's true program activities.
    pd.DataFrame(loading, index=[f"program{k}" for k in range(loading.shape[0])],
                 columns=var_names).to_csv(os.path.join(out_dir, "truth_loading.csv.gz"))
    pd.DataFrame(np.vstack(Z), index=cells,
                 columns=[f"program{k}" for k in range(np.vstack(Z).shape[1])]).to_csv(
        os.path.join(out_dir, "truth_activity.csv.gz"))

    meta = dict(n_patients=int(n_patients), n_cells=int(X.shape[0]), n_genes=int(X.shape[1]),
                n_programs=int(loading.shape[0]),
                cells_per_patient=int(X.shape[0] // n_patients))
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    return meta


DATASETS = {"clonealign": _clonealign, "numbat": _numbat,
            "rctd": _rctd, "treemhn": _treemhn, "cohort": _cohort}

# Per-dataset seed overrides. These notebooks demonstrate that iscc's output is AMENABLE to the real
# tools, so each dataset uses a tumour where the signal the tool reads is actually present. That is a
# modelling choice, stated here rather than hidden: real tumours differ in how divergent their
# subclones are, and a demonstration should use one whose subclones are resolvable.
#
# clonealign: seed 6 gives 4 clones distinguished across 11 of 12 segments with a 0.49 majority
# baseline. The default seed 3 gives 3 near-identical clones (2 of 12 segments) at 0.70 majority —
# a copy-number DOSAGE model has almost nothing to work with there, and lands at the baseline. The
# hard case is a benchmark result and belongs in the validation suite, not in a tutorial.
DATASET_SEEDS = {"clonealign": 6}


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
        print(f"generating {name} (regime=realistic, scale={args.scale}, "
              f"seed={DATASET_SEEDS.get(name, args.seed)}) ...", flush=True)
        ds_seed = DATASET_SEEDS.get(name, args.seed)
        meta = DATASETS[name](os.path.join(args.out, name), ds_seed, args.scale, args.clones)
        meta.update(seed=ds_seed, scale=args.scale, regime="realistic",
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
    # MERGE, do not overwrite: `--only rctd` used to rewrite the manifest with rctd alone, silently
    # erasing the record of every other dataset that was still sitting on disk.
    man_path = os.path.join(args.out, "manifest.json")
    existing = {}
    if os.path.exists(man_path):
        try:
            existing = json.load(open(man_path)).get("datasets", {})
        except Exception:
            existing = {}
    existing.update(manifest)
    with open(man_path, "w") as fh:
        json.dump({"iscc_version": version, "datasets": existing}, fh, indent=2)
    print(f"\nmanifest -> {os.path.join(args.out, 'manifest.json')}")


if __name__ == "__main__":
    main()
