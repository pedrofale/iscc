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
    """scRNA counts + per-gene clone copy number + the true clone of each cell.

    The copy number handed to clonealign is CALLED, not iscc's. `build_clonealign_inputs` builds `L`
    from the true per-segment states, which would mean the benchmark is told the answer to half its
    own question; here that matrix is replaced by the per-clone profiles HMMcopy called from
    single-cell read depth in the `dna` dataset. That is the workflow a real study runs — a CNV
    caller first, clonealign second — and it is what makes the assignment score mean anything.
    """
    import integration_common as C

    called_path = os.path.join(os.path.dirname(out_dir), "dna", "hmmcopy", "clone_cn_called.csv")
    if not os.path.exists(called_path):
        raise FileNotFoundError(
            f"clonealign needs HMMcopy's called copy number at {called_path}\n"
            "  generate it first:  python validation/make_analysis_data.py --only dna")

    t = C.grow_tumor(seed=seed, regime="realistic", scale=scale)
    inp = C.build_clonealign_inputs(t, n_clones=n_clones, seed=0)
    Y, labels = inp["Y"], np.asarray(inp["labels"])
    called = pd.read_csv(called_path, index_col=0)
    L = pd.DataFrame(called.values[:, inp["gene_seg"]].T.astype(float),
                     index=list(Y.columns), columns=inp["clone_names"])

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
        cn_called=called.values.astype(int).tolist(),
        cn_source="HMMcopy (analysis_data/dna/hmmcopy)",
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
    """A progression cohort: SCITE-reconstructed mutation trees, plus the cross-sectional view.

    TREES ARE INFERRED, NOT READ. iscc knows every patient's true mutation tree, and an earlier
    version of this dataset handed it straight to TreeMHN — which makes the benchmark vacuous, since
    the method would be estimating rates from the topology it is supposed to infer. Here each
    patient is genotyped single-cell (with dropout and false positives) and SCITE reconstructs the
    tree from that noisy matrix, which is the pipeline a real study runs. iscc's true trees are
    still written out, as ground truth for scoring the reconstruction.

    A cohort grown over PLANTED ORDERED CONSTRAINTS -- a DAG saying which events must precede which.

    Gating mode is the load-bearing choice. "accessibility" gates the MUTATION PROCESS itself, so
    the constraint survives into every observable including tree topology, which is what TreeMHN
    reads. "fitness" gating plants the same DAG but expresses it only through how large the
    carrying clones grow, leaving no ordering trace -- TreeMHN estimates RATES, so it recovers
    nothing from it (validate_epistasis panel D measures exactly this contrast).

    These notebooks demonstrate that iscc's output is amenable to the real tools, so the dataset
    uses the regime where the planted signal is genuinely present in the observable the tool reads.
    The fitness-gated case is a benchmark result and belongs in validate_epistasis, not here.
    event_size=8, NOT the default 2. A gated child needs its parent to have occurred first, and at
    event_size=2 it arises in 0-2 of 40 patients -- there is then nothing in the data to recover, no
    matter how good the tool. validate_epistasis' panel D hits the same wall and makes the same
    choice: the ordered-constraint half of this benchmark needs a COMMONER alphabet than the
    presence half. 40 patients for the same reason.

    That same choice saturates the CROSS-SECTIONAL view: at event_size=8 the planted parents are
    acquired by every patient, and an event present in all of them carries no information for a
    presence/absence method. The continuous cancer-cell-fraction matrix is therefore written
    alongside the binary one, so a detection floor -- which is what a real assay imposes anyway --
    can be applied downstream rather than baked in here at min_freq=0.
    """
    import validate_epistasis as VE
    import scite_common as SCITE
    from iscc.integrations import progression as ig
    from iscc.constants import DEFAULT_LAYOUT_SEED

    strength = 2.0
    net = dict(VE.epi(n_interactions=2, strength=strength), event_size=8)
    dep = dict(n_constraints=2, dag_depth=2, dag_branching=1, gating_mode="accessibility")
    tumors = VE.run_cohort(40, net, dependency_params=dep, seed0=1 + seed,
                           layout_seed=DEFAULT_LAYOUT_SEED, inject_E=strength)
    os.makedirs(out_dir, exist_ok=True)

    # --- the tool's input: trees SCITE reconstructed from noisy single-cell genotypes
    trees, matrices, kept = SCITE.reconstruct_cohort(tumors, seed=seed)
    trees.to_csv(os.path.join(out_dir, "trees.csv"), index=False)

    # SCITE's own input, kept so the reconstruction notebook can re-run it rather than take the
    # trees on faith. Same seeds, so it lands on the same trees.
    sc_dir = os.path.join(out_dir, "sc")
    os.makedirs(sc_dir, exist_ok=True)
    for old in os.listdir(sc_dir):
        os.remove(os.path.join(sc_dir, old))
    event_names = list(tumors[0].selection.epistasis.event_names())
    for pid, obs in enumerate(matrices, start=1):
        pd.DataFrame(obs, index=event_names,
                     columns=[f"cell{c}" for c in range(obs.shape[1])]).to_csv(
            os.path.join(sc_dir, f"P{pid}.csv"))

    # --- ground truth, held back: iscc's own trees for the SAME patients, in the same layout
    truth_trees = ig.to_treemhn_trees([tumors[i] for i in kept], drop_empty=False)
    truth_trees.to_csv(os.path.join(out_dir, "truth_trees.csv"), index=False)

    # --- the cross-sectional view of the same cohort (MHN), binary and continuous
    X = ig.to_mhn_matrix(tumors)
    X.to_csv(os.path.join(out_dir, "X_presence.csv"))
    F = ig.to_cell_fraction_matrix(tumors)
    F.to_csv(os.path.join(out_dir, "X_cellfraction.csv"))

    # The REALISED DAG (which event must precede which), not just the spec -- this is what a
    # recovered ordering is scored against.
    dag = tumors[0].selection.epistasis.true_dag_edges() if tumors else []
    with open(os.path.join(out_dir, "truth_network.json"), "w") as fh:
        json.dump({"epistasis_params": net, "dependency_params": dep,
                   "interaction_strength": strength,
                   "true_dag_edges": [[int(a), int(b)] for a, b in dag]}, fh, indent=2)
    meta = dict(n_patients=int(len(tumors)), n_trees=int(trees["Patient_ID"].nunique()),
                n_tree_rows=int(trees.shape[0]), n_events=int(VE.N_EVENTS),
                interaction_strength=strength, gating_mode=dep["gating_mode"],
                n_constraints=dep["n_constraints"], n_dag_edges=int(len(dag)),
                event_size=int(net["event_size"]),
                trees_from="SCITE", n_cells_sequenced=int(SCITE.N_CELLS_SEQUENCED),
                scite_fd=SCITE.FALSE_POSITIVE, scite_ad=SCITE.DROPOUT,
                event_frequency={k: float(v) for k, v in X.mean(axis=0).items()})
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    return meta


# MHN's cohort, chosen by sweeping for an identifiable cross-section. The binding constraint is that
# the PARENTS must not fix: a column present in every patient has no variance and nothing about it is
# estimable. event_size sets how often an event is acquired, and grow_steps sets how long the cohort
# has to acquire it -- and TIME turned out to be the operative lever. At the treemhn cohort's 500
# steps every parent is at 1.00 whatever event_size is used; at 150 steps with event_size=7 the
# parents sit at 0.78 / 0.83 while the gated children still reach 0.06 / 0.25, which is the window.
# Measured at these values: both planted edges come out as the top two promoting off-diagonals of
# twelve, and the no-DAG control recovers neither.
DNA_N_CELLS = 200            # cells a single-cell DNA run would realistically sequence
DNA_BINS_PER_SEGMENT = 20    # loci are aggregated into bins, as any real CNV pipeline does
DNA_N_NORMALS = 60           # diploid cells sequenced alongside: the ploidy anchor
DNA_N_MUTATIONS = 20         # mutations carried into a tree; SCITE is O(n^2) per MCMC move
DNA_N_BULK_MUTATIONS = 300   # loci carried into the bulk clustering

MHN_EVENT_SIZE = 7
MHN_STEPS = 150
MHN_N_PATIENTS = 300


def _mhn(out_dir, seed, scale, n_clones, event_size=MHN_EVENT_SIZE,
         n_patients=MHN_N_PATIENTS, steps=MHN_STEPS):
    """MHN's input: a cross-sectional cohort whose planted constraints survive into presence.

    A SEPARATE cohort from `treemhn`, and it has to be -- the two observables want the same KIND of
    signal but at opposite densities, and one cohort cannot serve both.

    WHICH SIGNAL. iscc plants two different things, and only one of them is visible to a
    presence/absence method:

      * pairwise fitness epistasis `E` decides how large the clones carrying a combination GROW. It
        does not change the rate at which events arise, so a favoured combination is already present
        at E=0 -- it simply arises many times independently. The binary column is then the same with
        and without the interaction (verified: a planted-E cohort and a matched zero-E control have
        identical presence marginals). This is validate_epistasis' headline finding: the signal lives
        in clone FREQUENCY, and MHN is not an observable that keeps frequency.
      * an ACCESSIBILITY-gated DAG acts on the mutation process itself: a child cannot arise until
        its parent has. That is a hard zero in the joint distribution -- P(child and not parent) = 0
        -- which is exactly what MHN reads.

    So the constraint here is the gated DAG, the same one TreeMHN is scored on, and the difference
    between the two benchmarks is the observable rather than the truth.

    WHICH DENSITY. `treemhn` uses event_size=8 so a gated child occurs often enough to leave an
    ordering trace in tree topology. That saturates presence: all four events turn up somewhere in
    every patient, so every column is constant and nothing is identifiable. Lowering event_size
    makes events rarer per patient and restores the variance, at the cost of needing more patients.

    A CONTROL cohort is generated alongside with the DAG removed and everything else held: same
    seeds, same modules, same growth. Anything the same pipeline reports there is a false positive,
    which is the only honest way to read the main arm.
    """
    import validate_epistasis as VE
    from iscc.integrations import progression as ig
    from iscc.constants import DEFAULT_LAYOUT_SEED

    strength = 2.0
    net = dict(VE.epi(n_interactions=2, strength=strength), event_size=event_size)
    dep = dict(n_constraints=2, dag_depth=2, dag_branching=1, gating_mode="accessibility")
    common = dict(epistasis_params=net, seed0=1 + seed, steps=steps,
                  layout_seed=DEFAULT_LAYOUT_SEED, inject_E=strength)
    tumors = VE.run_cohort(n_patients, dependency_params=dep, **common)
    control = VE.run_cohort(n_patients, dependency_params=None, **common)

    os.makedirs(out_dir, exist_ok=True)
    X = ig.to_mhn_matrix(tumors)
    X.to_csv(os.path.join(out_dir, "X_presence.csv"))
    ig.to_mhn_matrix(control).to_csv(os.path.join(out_dir, "X_presence_control.csv"))
    # The continuous view as well: a detection floor is an assay property, so keeping the raw
    # cancer-cell fractions lets the analysis vary it instead of inheriting min_freq=0 from here.
    ig.to_cell_fraction_matrix(tumors).to_csv(os.path.join(out_dir, "X_cellfraction.csv"))

    dag = tumors[0].selection.epistasis.true_dag_edges() if tumors else []
    with open(os.path.join(out_dir, "truth_network.json"), "w") as fh:
        json.dump({"epistasis_params": net, "dependency_params": dep,
                   "interaction_strength": strength,
                   "true_dag_edges": [[int(a), int(b)] for a, b in dag]}, fh, indent=2)
    meta = dict(n_patients=int(len(tumors)), n_control_patients=int(len(control)),
                n_events=int(VE.N_EVENTS), event_size=int(event_size), grow_steps=int(steps),
                gating_mode=dep["gating_mode"], n_dag_edges=int(len(dag)),
                n_off_diagonal=int(VE.N_EVENTS * (VE.N_EVENTS - 1)),
                event_frequency={k: float(v) for k, v in X.mean(axis=0).items()})
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    return meta


def _dna(out_dir, seed, scale, n_clones):
    """The DNA analysis chain, all from ONE tumour: copy number, single-cell trees, bulk clones.

    Three datasets share a tumour on purpose — it is the same lesion clonealign is run on, so the
    copy-number profiles called here are the ones that benchmark consumes, and the clones a bulk
    caller finds can be held against the clones a single-cell tree finds.

    Nothing a tool reads is copied from iscc's truth. Copy number is CALLED from read depth by
    HMMcopy, trees are reconstructed by SCITE from genotype calls that carry the assay's own dropout,
    and bulk clusters come from PyClone-VI's own fit to read counts.

      hmmcopy/    reads x bins + a bin table with GC and mappability  -> HMMcopy
      scite/      a mutations x cells binary genotype matrix          -> SCITE
      pyclonevi/  per-mutation bulk read counts with called CN        -> PyClone-VI
    """
    import integration_common as C
    import subprocess
    from iscc.data import bulkDNA, scDNA

    t = C.grow_tumor(seed=DATASET_SEEDS["clonealign"], regime="realistic", scale=scale)
    types = C.cell_types(t)
    seg, gene_seg = C.segment_cn(t)
    cancer = types == "cancer"
    cancer_cells = list(np.asarray(t.cell_data["cell_exp"].index)[cancer])
    labels, consensus = C.define_clones(seg[cancer], n_clones=n_clones)
    lab_by_cell = {c: int(labels[i]) for i, c in enumerate(cancer_cells)}
    n_seg = consensus.shape[1]

    # ---------------------------------------------------------------- hmmcopy: CN from read depth
    hm = os.path.join(out_dir, "hmmcopy"); os.makedirs(hm, exist_ok=True)
    rng = np.random.default_rng(seed)
    sub = sorted(rng.choice(cancer_cells, size=min(DNA_N_CELLS, len(cancer_cells)), replace=False),
                 key=cancer_cells.index)
    # Normal cells go into the SAME run, and they are not decoration. HMMcopy calls each cell's own
    # modal state neutral, so a whole-genome-doubled tumour comes back uniformly halved and two
    # clones that differ only in level become identical. Diploid cells sequenced alongside are the
    # anchor that turns relative copy number back into absolute -- which is exactly why real
    # single-cell CNV runs include them.
    normal_pool = list(np.asarray(t.cell_data["cell_exp"].index)[~cancer])
    normals = sorted(rng.choice(normal_pool, size=min(DNA_N_NORMALS, len(normal_pool)),
                                replace=False), key=normal_pool.index)
    # kappa=500 / mu_depth=60, not the scDNA defaults. The defaults model MDA/MALBAC, whose lumpy
    # amplification (kappa=5) leaves the median locus at ZERO reads -- fine for genotyping, useless
    # for depth-based copy number, and HMMcopy's GC loess cannot even fit. Depth-based single-cell
    # CNV is done on DLP+-style uniformly amplified shallow WGS, which is what these values are.
    dna = scDNA(breadth="wgs", kappa=500.0, mu_depth=60.0, seed=200 + seed).run(
        t.cell_data, cell_subset=list(sub) + list(normals))
    loci = list(dna.genes)
    seg_of = np.array([int(g.split("_")[1]) for g in loci])
    # HMMcopy reads a bin table: a contig, a position, GC and mappability. iscc's genome is a run of
    # segments rather than named chromosomes, so each segment is handed over as its own contig --
    # which is also what stops the HMM from smoothing a real breakpoint away across a boundary.
    # Loci are aggregated into coarse BINS, which is what a real single-cell CNV pipeline does and
    # for the same reason: one locus of shallow single-cell coverage is mostly zeros, and copy number
    # is a property of a stretch of genome rather than of a base.
    gc_all, map_all = np.asarray(dna.gc), np.asarray(dna.mappability)
    cov = dna.coverage.values
    rows, groups, names = [], [], []
    for sgm in range(n_seg):
        w = np.where(seg_of == sgm)[0]
        for b, chunk in enumerate(np.array_split(w, DNA_BINS_PER_SEGMENT)):
            rows.append(dict(chr=f"chr{sgm}", start=b * 1000 + 1, end=b * 1000 + 1000,
                             gc=float(gc_all[chunk].mean()), map=float(map_all[chunk].mean())))
            groups.append(chunk); names.append(f"chr{sgm}_b{b}")
    bins = pd.DataFrame(rows, index=pd.Index(names, name="locus"))
    reads = pd.DataFrame(
        np.stack([cov[:, chunk].sum(1) for chunk in groups]),      # bins x cells
        index=bins.index, columns=list(dna.coverage.index)).astype(int)
    bins.to_csv(os.path.join(hm, "bins.csv"))
    reads.to_csv(os.path.join(hm, "reads.csv"))
    # -1 marks a diploid normal: the tool is TOLD which cells are the reference (a real run knows
    # which wells held normal tissue), but never which clone a tumour cell belongs to.
    dna_lab = np.array([lab_by_cell.get(c, -1) for c in dna.cells])
    pd.DataFrame({"cell": dna.cells, "is_normal": (dna_lab < 0).astype(int)}).to_csv(
        os.path.join(hm, "cell_annotation.csv"), index=False)
    pd.DataFrame({"cell": dna.cells, "true_clone": dna_lab}).to_csv(
        os.path.join(hm, "truth_clone.csv"), index=False)
    pd.DataFrame(consensus.astype(int), index=[f"clone{c}" for c in range(consensus.shape[0])],
                 columns=[f"seg{s}" for s in range(n_seg)]).to_csv(
        os.path.join(hm, "truth_consensus.csv"))

    # Run it now, so the CN the downstream datasets use is CALLED copy number rather than iscc's.
    called = os.path.join(hm, "cell_cn_called.csv")
    rscript = os.path.expanduser("~/miniconda3/envs/iscc-hmmcopy/bin/Rscript")
    proc = subprocess.run([rscript, os.path.join(REPO, "validation", "hmmcopy_runner.R"),
                           os.path.join(hm, "reads.csv"), os.path.join(hm, "bins.csv"), called],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"hmmcopy_runner failed: {proc.stderr[-500:]}")
    print("   ", proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else "hmmcopy ok")
    # HMMcopy states are 1..6 for CN 0..5+. It calls each cell's MODAL segment neutral, so on a
    # WGD tumour the level is relative -- the between-clone contrast, which is what clonealign uses,
    # is what survives.
    # Ploidy anchoring: HMMcopy's corrected log2 depth is relative to each cell's own library, so a
    # cell's absolute copy number is 2 x its depth ratio against the diploid normals.
    cop = pd.read_csv(called.replace(".csv", "_copy.csv"), index_col=0)[list(reads.columns)]
    lin = np.power(2.0, cop.values)                                   # back to linear depth
    is_norm = dna_lab < 0
    ref = np.nanmedian(lin[:, is_norm], axis=1, keepdims=True)
    cn_abs = 2.0 * lin / np.where(ref > 0, ref, np.nan)
    seg_called = pd.DataFrame(
        {f"seg{s}": np.nanmedian(cn_abs[(bins["chr"] == f"chr{s}").values], axis=0)
         for s in range(n_seg)}, index=list(reads.columns))
    clone_cn = np.stack([np.rint(np.nanmedian(seg_called.values[dna_lab == c], axis=0))
                         for c in range(consensus.shape[0])])
    pd.DataFrame(clone_cn.astype(int),
                 index=[f"clone{c}" for c in range(clone_cn.shape[0])],
                 columns=[f"seg{s}" for s in range(n_seg)]).to_csv(
        os.path.join(hm, "clone_cn_called.csv"))

    # ---------------------------------------------------------------- scite: trees from genotypes
    sc = os.path.join(out_dir, "scite"); os.makedirs(sc, exist_ok=True)
    gt = scDNA(breadth="wgs", data_mode="binary", kappa=500.0, mu_depth=60.0,
               seed=300 + seed).run(t.cell_data, cell_subset=list(sub))
    obs = gt.observed_snvs
    true_snv = t.cell_data["cell_snv"].loc[list(obs.index), list(obs.columns)].values > 0
    # Pick the mutations a study would actually carry into a tree: present in a real fraction of
    # cells but not in all of them (a truncal mutation separates nobody). Spread the selection ACROSS
    # the frequency range rather than taking the commonest -- mutations that all sit at one frequency
    # are all at the same depth, and the tree collapses to a chain nothing can be ordered within.
    frac = true_snv.mean(0)
    cand = np.where((frac >= 0.05) & (frac <= 0.95))[0]
    cand = cand[np.argsort(-frac[cand])]
    step = max(1, len(cand) // DNA_N_MUTATIONS)
    pick = cand[::step][:DNA_N_MUTATIONS]
    muts = [obs.columns[i] for i in pick]
    obs[muts].T.rename_axis("mutation").to_csv(os.path.join(sc, "sc_mutations.csv"))
    pd.DataFrame(true_snv[:, pick].astype(int), index=obs.index, columns=muts).T.rename_axis(
        "mutation").to_csv(os.path.join(sc, "truth_genotypes.csv"))
    pd.DataFrame({"cell": list(obs.index),
                  "true_clone": [lab_by_cell[c] for c in obs.index]}).to_csv(
        os.path.join(sc, "truth_clone.csv"), index=False)

    # ---------------------------------------------------------------- pyclone-vi: clones from bulk
    pc = os.path.join(out_dir, "pyclonevi"); os.makedirs(pc, exist_ok=True)
    # Bulk on a MACRO-DISSECTED block, not the whole section. Sequencing the raw field gives ~19%
    # tumour content, and at that purity every VAF is squeezed towards zero and only the truncal
    # cluster separates. Real bulk studies dissect to enrich; here the pool is every cancer cell plus
    # enough normal tissue to land near 50%, which is an ordinary purity for a solid tumour.
    n_contam = min(len(normal_pool), len(cancer_cells))
    block = list(cancer_cells) + list(rng.choice(normal_pool, size=n_contam, replace=False))
    bulk = bulkDNA(breadth="wgs", seed=400 + seed).run(t.cell_data, cell_subset=block)
    bd = bulk.observed_data
    somatic = bd[(bd["alt_counts"] > 0) & (bd["coverage"] >= 10)]
    if "is_germline" in somatic:
        somatic = somatic[~somatic["is_germline"].astype(bool)]
    somatic = somatic.loc[[g for g in somatic.index if g in set(t.cell_data["cell_snv"].columns)]]
    keep = somatic.sample(n=min(DNA_N_BULK_MUTATIONS, len(somatic)), random_state=seed).sort_index()
    # Copy number comes from the HMMcopy call above, NOT from iscc. PyClone-VI wants major/minor;
    # depth-based calling gives TOTAL copy number only, so the total is split the conventional way
    # (one minor copy whenever the total allows it) and that assumption is stated rather than hidden.
    tot = np.rint(np.clip(clone_cn.mean(0), 0, None)).astype(int)
    seg_idx = keep["segment"].astype(int).values
    total_cn = np.clip(tot[np.clip(seg_idx, 0, len(tot) - 1)], 1, None)
    minor = np.where(total_cn >= 2, 1, 0)
    pd.DataFrame({
        "mutation_id": keep.index,
        "sample_id": "bulk",
        "ref_counts": (keep["coverage"] - keep["alt_counts"]).astype(int).values,
        "alt_counts": keep["alt_counts"].astype(int).values,
        "major_cn": (total_cn - minor).astype(int),
        "minor_cn": minor.astype(int),
        "normal_cn": 2,
        "tumour_content": float(bulk.purity) if bulk.purity is not None else 1.0,
    }).to_csv(os.path.join(pc, "input.tsv"), sep="\t", index=False)
    # Truth: each mutation's cancer-cell fraction, and its CLONAL IDENTITY. The identity is the SET
    # of clones carrying it, not "the clone that carries it most" -- a truncal mutation is in every
    # clone, and forcing it into one makes the clustering unscoreable (mutations that genuinely
    # belong together get split across labels). Mutations sharing a carrier set are one clone, which
    # is exactly the grouping PyClone-VI estimates.
    snv = t.cell_data["cell_snv"]
    sub_by_clone = [[c for c in cancer_cells if lab_by_cell[c] == k]
                    for k in range(consensus.shape[0])]
    carried = np.stack([(snv.loc[cells_k, list(keep.index)].values > 0).mean(0) > 0.5
                        for cells_k in sub_by_clone])            # clones x mutations
    sig = ["c" + "".join("1" if carried[k, j] else "0" for k in range(carried.shape[0]))
           for j in range(carried.shape[1])]
    codes = {s_: i for i, s_ in enumerate(sorted(set(sig)))}
    ccf = (snv.loc[cancer_cells, list(keep.index)].values > 0).mean(0)
    pd.DataFrame({"mutation_id": keep.index, "true_ccf": ccf,
                  "carrier_clones": sig,
                  "true_cluster": [codes[s_] for s_ in sig]}).to_csv(
        os.path.join(pc, "truth.csv"), index=False)

    meta = dict(
        tumour_seed=int(DATASET_SEEDS["clonealign"]), n_clones=int(consensus.shape[0]),
        n_segments=int(n_seg), n_cancer_cells=int(len(cancer_cells)),
        hmmcopy=dict(n_cells=int(reads.shape[1]), n_bins=int(reads.shape[0])),
        scite=dict(n_cells=int(obs.shape[0]), n_mutations=int(len(muts)),
                   ado_rate=float(gt.hypers.to_dict().get("ado_rate", 0.2))),
        pyclonevi=dict(n_mutations=int(len(keep)),
                       purity=float(bulk.purity) if bulk.purity is not None else None),
        true_clone_sizes=np.bincount(labels, minlength=consensus.shape[0]).tolist(),
    )
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
            "rctd": _rctd, "treemhn": _treemhn, "mhn": _mhn, "cohort": _cohort,
            "dna": _dna}

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
