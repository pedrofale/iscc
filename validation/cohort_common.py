"""Shared data generation + scoring for the multi-patient COHORT benchmarks (DESIGN_cohort.md §3).

Everything here runs in the core ``iscc`` env (numpy/scipy/sklearn only) so the figure is fully
reproducible without any external tool. The heavy integration/demux tools live in their OWN dedicated
conda envs (the clonealign/inferCNV convention, ``validation/README_integration.md``) and are used
ONLY when present — the interpreter path is env-var-overridable and guarded by an ``*_available()``
skip, so the benchmark degrades gracefully to the self-contained baseline.

The point (scMultiSim/SISTEM "we provide the ground truth" convention): iscc uniquely carries the true
recurrent-vs-private driver split, per-cell patient-of-origin, shared-vs-private cell-state labels, and
per-patient subgroup + true therapy response — exactly what a real cohort can never give.
"""
import os
import itertools
import subprocess

import numpy as np
import pandas as pd

from iscc.cohort import Cohort, Subgroup, pool_cell_data, run_cohort_batches, concat_cohort_batches
from iscc.cohort.groundtruth import _cancer_snv
from iscc.tumor.models import GenotypeTumor
from iscc.treatment.chemotherapy import Chemotherapy

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOME = os.path.expanduser("~")

# Dedicated conda envs holding the external integration/demux tools (kept out of the core env).
HARMONY_PYTHON = os.environ.get("ISCC_HARMONY_PYTHON", os.path.join(HOME, "miniconda3/envs/iscc-harmony/bin/python"))
SCVI_PYTHON = os.environ.get("ISCC_SCVI_PYTHON", os.path.join(HOME, "miniconda3/envs/iscc-scvi/bin/python"))
DEMUX_PYTHON = os.environ.get("ISCC_DEMUX_PYTHON", os.path.join(HOME, "miniconda3/envs/iscc-demux/bin/python"))


def harmony_available():
    return os.path.exists(HARMONY_PYTHON)


def scvi_available():
    return os.path.exists(SCVI_PYTHON)


def demux_available():
    return os.path.exists(DEMUX_PYTHON)


# ======================================================================================
# scoring helpers
# ======================================================================================
def inverse_simpson_lisi(emb, labels, mask=None, k=30):
    """LISI = effective number of ``labels`` categories among each cell's k nearest neighbours
    (inverse-Simpson). iLISI on the BATCH/patient label: high = well mixed across patients."""
    from sklearn.neighbors import NearestNeighbors
    labels = np.asarray(labels)
    idx = np.where(mask)[0] if mask is not None else np.arange(len(emb))
    if len(idx) <= k:
        k = max(2, len(idx) - 1)
    nn = NearestNeighbors(n_neighbors=k).fit(emb[idx])
    _, ind = nn.kneighbors(emb[idx])
    lab = labels[idx]
    out = []
    for i in range(len(idx)):
        _, cnt = np.unique(lab[ind[i]], return_counts=True)
        p = cnt / cnt.sum()
        out.append(1.0 / np.sum(p ** 2))
    return float(np.mean(out))


def match_accuracy(pred, truth, K):
    """Best-permutation (Hungarian) accuracy of a clustering ``pred`` vs integer ``truth``."""
    from scipy.optimize import linear_sum_assignment
    conf = np.zeros((K, K))
    for a in range(K):
        for b in range(K):
            conf[a, b] = ((pred == a) & (truth == b)).sum()
    r, c = linear_sum_assignment(-conf)
    m = {r[i]: c[i] for i in range(len(r))}
    return float(np.mean([m.get(p, -1) == t for p, t in zip(pred, truth)]))


# ======================================================================================
# 1) recurrence / driver detection + the shared-vs-unshared enablement contrast
# ======================================================================================
RECUR_GENOME = {"n_segments": 6, "segment_size": 40}
RECUR_SEL = {"prop_driver": 0.12, "prop_dispersal": 0.0, "prop_immune_resistance": 0.0,
             "prop_treatment_resistance": 0.0, "driver_effects": 1.25}
RECUR_DEME = {"carrying_capacity": 6, "initial_cancer_cells": 4}
RECUR_SPATIAL = {"grid_size": 13, "structure_radius": 3}
RECUR_CANCER = {"division_rate": 0.6, "death_rate": 0.06, "max_birth_rate": 0.98,
                "mutation_rate": 1.0, "dispersal_rate": 0.3}


def recurrence_cohort(n_patients=24, steps=300):
    return Cohort(patient_seeds=list(range(1, n_patients + 1)), genome_params=RECUR_GENOME,
                  selection_params=RECUR_SEL, cancer_cell_params=RECUR_CANCER, deme_params=RECUR_DEME,
                  spatial_params=RECUR_SPATIAL, grow_steps=steps).run()


def _jaccard(a, b):
    a, b = set(int(x) for x in a), set(int(x) for x in b)
    return len(a & b) / len(a | b) if (a | b) else 1.0


def recurrence_analysis(cohort, clonal_thresh=0.05):
    """Returns the recurrence table + the shared-vs-unshared driver-identity contrast (the headline:
    cross-patient recurrence is only well-posed BECAUSE the layout fix shares driver identities)."""
    from iscc.cohort import recurrence_table
    from scipy.stats import mannwhitneyu
    rt = recurrence_table(cohort, clonal_thresh=clonal_thresh)
    drv = rt[rt.is_driver].recurrence.values
    pas = rt[~rt.is_driver].recurrence.values
    _, mwu_p = mannwhitneyu(drv, pas, alternative="greater")

    # shared (the fix, default) vs unshared (layout_seed = evolution seed) driver-set Jaccard
    onc_shared = [list(p.tumor.selection.get_oncogenes()) for p in cohort.patients]
    j_shared = np.mean([_jaccard(a, b) for a, b in itertools.combinations(onc_shared, 2)])
    seeds = cohort.patient_seeds[:min(6, cohort.n_patients)]
    onc_unshared = [list(GenotypeTumor(seed=s, layout_seed=s, genome_params=RECUR_GENOME,
                                       selection_params=RECUR_SEL, cancer_cell_params=RECUR_CANCER,
                                       deme_params=RECUR_DEME, spatial_params=RECUR_SPATIAL
                                       ).selection.get_oncogenes()) for s in seeds]
    j_unshared = np.mean([_jaccard(a, b) for a, b in itertools.combinations(onc_unshared, 2)])
    return dict(table=rt, driver_recurrence=drv, passenger_recurrence=pas, mwu_p=float(mwu_p),
                jaccard_shared=float(j_shared), jaccard_unshared=float(j_unshared))


# ======================================================================================
# 2) personalized medicine / stratification (uses the treatment module)
# ======================================================================================
# A larger genome (2400 loci) so the specific resistance loci are NOT saturated by genome-wide random
# mutation — the pre-existing resistant subclone is then a distinct, detectable minority signature.
PM_GENOME = {"n_segments": 12, "segment_size": 200}
PM_SEL = {"prop_driver": 0.05, "prop_dispersal": 0.0, "prop_immune_resistance": 0.0,
          "prop_treatment_resistance": 0.05, "driver_effects": 1.15, "treatment_resistant_effects": 1.0}
PM_DEME = {"carrying_capacity": 8, "initial_cancer_cells": 6}
PM_SPATIAL = {"grid_size": 15, "structure_radius": 4}
PM_CANCER = {"division_rate": 0.6, "death_rate": 0.05, "max_birth_rate": 0.98,
             "mutation_rate": 0.5, "dispersal_rate": 0.3}


def pm_cohort(n_patients=16, steps=260, n_loci=6, subclone_cells=2):
    """Two subgroups over the shared landscape: 'sensitive' and 'resistant'. The resistant subtype
    carries a PRE-EXISTING RESISTANT SUBCLONE (biologically faithful: resistance usually pre-exists as
    a rare subclone, not clonally) at the shared treatment-resistance loci + a high resistance effect.
    A therapy eradicates the sensitive tumours; the resistant ones relapse from the selected subclone.
    The subclone is SUBCLONAL at baseline (a minority of cells) — below bulk VAF prominence, detectable
    at single-cell resolution: iscc's ground truth shows the actionable subclone bulk would miss."""
    sel = Cohort(patient_seeds=[1], genome_params=PM_GENOME, selection_params=PM_SEL,
                 cancer_cell_params=PM_CANCER, deme_params=PM_DEME, spatial_params=PM_SPATIAL).selection
    loci = tuple(int(x) for x in sel.get_treatment_resistant()[:n_loci])
    subs = [Subgroup("sensitive", {"treatment_resistant_effects": 1.0}, therapy_response=1),
            Subgroup("resistant", {"treatment_resistant_effects": 4.0},
                     subclone_mutations=loci, subclone_cells=subclone_cells, therapy_response=0)]
    co = Cohort(patient_seeds=list(range(1, n_patients + 1)), genome_params=PM_GENOME,
                selection_params=PM_SEL, cancer_cell_params=PM_CANCER, deme_params=PM_DEME,
                spatial_params=PM_SPATIAL, subgroups=subs, grow_steps=steps)
    co._resistance_loci = loci
    return co


def _chemo():
    return Chemotherapy(start=0, effectiveness=0.95, toxicity=0.01, kill_rate=1.8, rate_multiplier=2.5)


def differential_response(cohort):
    """Per-patient baseline and treated cancer size (fresh, identical-setup runs differing only in the
    therapy) — the ground-truth differential response by subgroup."""
    base = np.array([cohort.grow_patient(i).get_cancer_size() for i in range(cohort.n_patients)], float)
    treated = np.array([cohort.grow_patient(i, treatment=_chemo()).get_cancer_size()
                        for i in range(cohort.n_patients)], float)
    subgroup = np.array([cohort.subgroup_assignment[i] for i in range(cohort.n_patients)])
    return dict(baseline=base, treated=treated, subgroup=subgroup)


def stratification(cohort):
    """Recover the therapy-responsive subgroup from the BASELINE molecular profile — contrasting a BULK
    biomarker (mean VAF at the resistance loci) with a SINGLE-CELL one (fraction of cells co-mutated at
    the resistance loci = the pre-existing resistant subclone). The subclone is subclonal, so the
    single-cell signature is the clean, specific marker of a non-responder — stratification with a known
    answer key. Also reports a cross-validated whole-profile classifier."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import roc_auc_score, accuracy_score
    if not cohort.patients:
        cohort.run()
    loci = list(getattr(cohort, "_resistance_loci", ()))
    feats, y, bulk, singlecell = [], [], [], []
    for i, pr in enumerate(cohort.patients):
        cs, _ = _cancer_snv(pr)
        feats.append(cs.mean(0) if cs.shape[0] else np.zeros(pr.tumor.n_genes))
        y.append(0 if cohort.subgroup_assignment[i] == "sensitive" else 1)
        bulk.append(float(cs[:, loci].mean()) if (cs.shape[0] and loci) else 0.0)             # bulk mean VAF
        singlecell.append(float((cs[:, loci] > 0).all(1).mean()) if (cs.shape[0] and loci) else 0.0)  # subclone frac
    X = np.vstack(feats)
    y = np.array(y)
    bulk = np.array(bulk)
    singlecell = np.array(singlecell)
    clf = LogisticRegression(max_iter=1000, C=1.0)
    proba = cross_val_predict(clf, X, y, cv=min(5, cohort.n_patients // 2), method="predict_proba")[:, 1]
    auc = lambda s: roc_auc_score(y, s) if 0 < y.sum() < len(y) else float("nan")
    return dict(X=X, y=y, proba=proba, clf_auc=float(auc(proba)),
                clf_acc=float(accuracy_score(y, (proba >= 0.5).astype(int))),
                bulk=bulk, singlecell=singlecell, bulk_auc=float(auc(bulk)),
                singlecell_auc=float(auc(singlecell)), resistance_loci=loci)


# ======================================================================================
# 3) multi-patient batch integration (shared-vs-private)
# ======================================================================================
INT_GENOME = {"n_segments": 6, "segment_size": 40}
INT_SEL = {"prop_driver": 0.1, "prop_dispersal": 0.0, "prop_immune_resistance": 0.0,
           "prop_treatment_resistance": 0.0, "driver_effects": 1.2}
INT_DEME = {"carrying_capacity": 6, "initial_cancer_cells": 4}
INT_SPATIAL = {"grid_size": 13, "structure_radius": 3}
INT_CANCER = {"division_rate": 0.6, "death_rate": 0.05, "max_birth_rate": 0.98,
              "mutation_rate": 1.2, "dispersal_rate": 0.3}


def integration_cohort(n_patients=8, steps=280):
    return Cohort(patient_seeds=list(range(1, n_patients + 1)), genome_params=INT_GENOME,
                  selection_params=INT_SEL, cancer_cell_params=INT_CANCER, deme_params=INT_DEME,
                  spatial_params=INT_SPATIAL, grow_steps=steps).run()


def _coarse_types(cohort, gids):
    g2t = {}
    for pr in cohort.patients:
        for g, rep in pr.tumor.genotypes.items():
            g2t[g] = rep.type
    return np.array([g2t.get(g, "?") for g in gids])


def _embed(Xlog, n=20):
    """z-score + PCA of an already log-normalized matrix."""
    from sklearn.decomposition import PCA
    Xz = (Xlog - Xlog.mean(0)) / (Xlog.std(0) + 1e-8)
    return PCA(n_components=min(n, Xz.shape[1] - 1), random_state=0).fit_transform(Xz)


def integration_analysis(cohort, sigma_batch=0.6, depth_batch_sigma=0.3, n_cells_per_patient=60):
    """1:1 pooling -> one batch per patient. Score shared-state batch mixing (iLISI on the SHARED
    normal cells) and biology preservation (cell-type ARI) for the naive embedding vs a batch-centered
    correction (self-contained) and, when the ``iscc-harmony`` env is present, real Harmony."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    assays, batches, asg = run_cohort_batches(cohort, mapping="one_to_one",
                                              n_cells_per_patient=n_cells_per_patient,
                                              sigma_batch=sigma_batch, depth_batch_sigma=depth_batch_sigma)
    comb = concat_cohort_batches(assays)
    X = np.asarray(comb.X, dtype=float)
    patient = comb.obs["patient"].astype(int).values
    coarse = _coarse_types(cohort, comb.obs["clone"].values)
    shared = np.isin(coarse, ["epithelial", "stromal", "immune"])
    K_ct = len(set(coarse))

    def scores(emb):
        ilisi = inverse_simpson_lisi(emb, patient, mask=shared, k=30)
        km = KMeans(n_clusters=max(2, K_ct), n_init=10, random_state=0).fit_predict(emb)
        ari = adjusted_rand_score(coarse, km)      # biology (cell type) preserved?
        return ilisi, ari

    Xlog = np.log1p(X)
    emb_naive = _embed(Xlog)
    # self-contained "correction": remove the additive per-batch (patient) mean per gene IN LOG SPACE
    Xc = Xlog.copy()
    grand = Xlog.mean(0)
    for b in np.unique(patient):
        m = patient == b
        Xc[m] = Xc[m] - Xc[m].mean(0) + grand
    emb_corr = _embed(Xc)

    out = dict(n_patients=cohort.n_patients, n_shared=int(shared.sum()), n_cells=len(patient),
               naive=scores(emb_naive), corrected=scores(emb_corr), ideal_ilisi=cohort.n_patients,
               method="batch-centered")
    if harmony_available():
        try:
            emb_h = run_harmony(emb_naive, patient)
            out["harmony"] = scores(emb_h)
        except Exception as e:            # never let the optional tool break the figure
            out["harmony_error"] = str(e)
    return out


def run_harmony(emb, batch_labels, work_dir=None):
    """Run REAL Harmony (harmonypy) in the ``iscc-harmony`` env on a PCA embedding. Returns the
    corrected embedding. Shells out to ``harmony_runner.py`` so harmonypy stays out of the core env."""
    import tempfile
    if not harmony_available():
        raise RuntimeError(f"harmony env not found at {HARMONY_PYTHON}")
    work_dir = work_dir or tempfile.mkdtemp(prefix="iscc_harmony_")
    np.save(os.path.join(work_dir, "emb.npy"), np.asarray(emb, float))
    pd.Series(np.asarray(batch_labels)).to_csv(os.path.join(work_dir, "batch.csv"), index=False)
    runner = os.path.join(REPO, "validation", "harmony_runner.py")
    subprocess.run([HARMONY_PYTHON, runner, work_dir], check=True, capture_output=True, text=True)
    return np.load(os.path.join(work_dir, "corrected.npy"))


# ======================================================================================
# 4) demultiplexing (N:1 pooling, patient-of-origin from private germline variants)
# ======================================================================================
DEMUX_GENOME = {"n_segments": 12, "segment_size": 400}
DEMUX_SEL = {"prop_driver": 0.03, "prop_dispersal": 0.0, "prop_immune_resistance": 0.0,
             "prop_treatment_resistance": 0.0, "driver_effects": 1.2}
DEMUX_DEME = {"carrying_capacity": 12, "initial_cancer_cells": 6}
DEMUX_SPATIAL = {"grid_size": 15, "structure_radius": 0}
DEMUX_CANCER = {"division_rate": 0.6, "death_rate": 0.05, "max_birth_rate": 0.98,
                "mutation_rate": 0.7, "dispersal_rate": 0.3}


def demux_cohort(n_patients=8, steps=300, n_germline=40):
    """All-cancer tumours with per-patient PRIVATE germline markers (the individual background genetic
    demux exploits) — pooled N:1, the patient-of-origin answer key is the private-variant fingerprint."""
    return Cohort(patient_seeds=list(range(1, n_patients + 1)), genome_params=DEMUX_GENOME,
                  selection_params=DEMUX_SEL, cancer_cell_params=DEMUX_CANCER, deme_params=DEMUX_DEME,
                  spatial_params=DEMUX_SPATIAL, grow_steps=steps, n_germline_markers=n_germline).run()


def demux_analysis(cohort, capacity=None, n_cells_per_patient=120):
    """Pool all patients into one batch (N:1) and assign pooled cells back to patient-of-origin from
    their private-variant genotype (souporcell/vireo-style). Returns accuracy vs the true patient."""
    from sklearn.cluster import AgglomerativeClustering
    K = cohort.n_patients
    pooled, meta = pool_cell_data([cohort.patients[p] for p in range(K)],
                                  n_cells_per_patient=n_cells_per_patient)
    cancer = meta["cell_type"].values == "cancer"
    v = (pooled["cell_snv"].loc[meta.index[cancer]].values > 0).astype(float)
    truth = meta["patient"].values[cancer].astype(int)
    freq = v.mean(0)
    keep = (freq >= 0.01) & (freq <= 0.95)
    pred = AgglomerativeClustering(n_clusters=K).fit_predict(v[:, keep])
    acc = match_accuracy(pred, truth, K)
    out = dict(accuracy=acc, chance=1.0 / K, n_cells=int(cancer.sum()), n_sites=int(keep.sum()),
               pred=pred, truth=truth, method="private-variant clustering (oracle calls)")
    return out
