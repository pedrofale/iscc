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
# 2) personalized medicine / stratification (uses the treatment module) — resistance EMERGES
# ======================================================================================
# Nothing is seeded: resistance ARISES from mutation + selection. Two subtypes share the landscape and
# differ ONLY in an effect scalar (``treatment_resistant_effects``). A larger genome (1200 loci) keeps
# the shared resistance loci from being saturated by genome-wide passenger mutation, so the emergent
# resistant clone is a distinct signature.
PM_GENOME = {"n_segments": 10, "segment_size": 120}
PM_SEL = {"prop_driver": 0.04, "prop_dispersal": 0.0, "prop_immune_resistance": 0.0,
          "prop_treatment_resistance": 0.06, "driver_effects": 1.15, "treatment_resistant_effects": 1.0}
PM_DEME = {"carrying_capacity": 10, "initial_cancer_cells": 8}
PM_SPATIAL = {"grid_size": 17, "structure_radius": 0}
PM_CANCER = {"division_rate": 0.6, "death_rate": 0.08, "max_birth_rate": 0.98,
             "mutation_rate": 0.9, "dispersal_rate": 0.3}
PM_STEPS = 520
PM_THERAPY_START = 200      # untreated burn-in during which resistance emerges as standing variation


def pm_cohort(n_patients=14, steps=PM_STEPS):
    """Two molecular subtypes over ONE shared landscape, 'sensitive' and 'resistant', differing only in
    ``treatment_resistant_effects``. Resistance is NOT seeded — it EMERGES: during an untreated burn-in,
    treatment-resistance mutations arise at the shared resistance loci and drift as neutral standing
    variation; ADJUVANT therapy then SELECTS them in the resistant subtype (which relapses) while they
    stay inert in the sensitive subtype (which is eradicated). The differential response is therefore a
    genuine evolutionary outcome, not an imposed one."""
    sel = Cohort(patient_seeds=[1], genome_params=PM_GENOME, selection_params=PM_SEL,
                 cancer_cell_params=PM_CANCER, deme_params=PM_DEME, spatial_params=PM_SPATIAL).selection
    loci = [int(x) for x in sel.get_treatment_resistant()]
    subs = [Subgroup("sensitive", {"treatment_resistant_effects": 1.0}, therapy_response=1),
            Subgroup("resistant", {"treatment_resistant_effects": 6.0}, therapy_response=0)]
    co = Cohort(patient_seeds=list(range(1, n_patients + 1)), genome_params=PM_GENOME,
                selection_params=PM_SEL, cancer_cell_params=PM_CANCER, deme_params=PM_DEME,
                spatial_params=PM_SPATIAL, subgroups=subs, grow_steps=steps)
    co._resistance_loci = loci
    return co


def _adjuvant_chemo():
    return Chemotherapy(start=PM_THERAPY_START, effectiveness=0.95, toxicity=0.01,
                        kill_rate=1.8, rate_multiplier=2.5)


def _cancer_snv_of(tumor):
    tumor.make_cell_data()
    cs = tumor.cell_data["cell_snv"].values
    gid = tumor.cell_data["cell_type"].iloc[:, 0].astype(str).values
    can = np.array([tumor.genotypes[g].type == "cancer" for g in gid], dtype=bool)
    return cs[can] if can.size else cs[:0]


def pm_analysis(cohort):
    """Grow each patient untreated (baseline) and under ADJUVANT therapy, and score:
      * the ground-truth DIFFERENTIAL RESPONSE (treated cancer size by subtype);
      * RECOVERY of the responsive subtype from molecular data — honestly contrasting the non-predictive
        BASELINE (standing resistance mutations are present in BOTH subtypes; only their functional
        effect differs, so a bulk baseline call cannot separate them) with the EMERGENT signature the
        therapy reveals (the relapsed tumour is clonally enriched for the selected resistance mutations)
        and the response readout itself (who benefits, the known answer)."""
    from sklearn.metrics import roc_auc_score
    y = np.array([0 if cohort.subgroup_assignment[i] == "sensitive" else 1
                  for i in range(cohort.n_patients)])
    loci = list(getattr(cohort, "_resistance_loci", ()))
    auc = lambda s: float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")

    baseline_sizes, treated_sizes, baseline_bm, relapse_bm = [], [], [], []
    for i in range(cohort.n_patients):
        base = cohort.grow_patient(i)                                   # untreated
        treat = cohort.grow_patient(i, treatment=_adjuvant_chemo())     # adjuvant therapy
        baseline_sizes.append(base.get_cancer_size())
        treated_sizes.append(treat.get_cancer_size())
        cb, ct = _cancer_snv_of(base), _cancer_snv_of(treat)
        # baseline "standing resistance" = any resistance-locus mutation present (emergent, neutral)
        baseline_bm.append(float((cb[:, loci] > 0).any(1).mean()) if (cb.shape[0] and loci) else 0.0)
        # emergent signature therapy REVEALS = fraction of surviving cells carrying a resistance mutation
        relapse_bm.append(float((ct[:, loci] > 0).any(1).mean()) if (ct.shape[0] and loci) else 0.0)

    baseline_sizes = np.array(baseline_sizes, float)
    treated_sizes = np.array(treated_sizes, float)
    baseline_bm = np.array(baseline_bm)
    relapse_bm = np.array(relapse_bm)
    subgroup = np.array([cohort.subgroup_assignment[i] for i in range(cohort.n_patients)])
    return dict(subgroup=subgroup, y=y, baseline=baseline_sizes, treated=treated_sizes,
                baseline_bm=baseline_bm, relapse_bm=relapse_bm,
                response_auc=auc(treated_sizes), baseline_auc=auc(baseline_bm),
                relapse_auc=auc(relapse_bm), resistance_loci=loci)


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
DEMUX_DEME = {"carrying_capacity": 10, "initial_cancer_cells": 6}
# A REALISTIC pool: a solid tumour with a normal microenvironment (epithelial/stromal ring +
# infiltrating immune cells) — so the demux operates on cancer AND normal cells, exactly like a real
# multiplexed single-cell run. Genetic demux (souporcell/vireo) assigns EVERY cell to its individual
# via germline variants, which iscc now carries in all cell types (tumour and normal).
DEMUX_SPATIAL = {"grid_size": 17, "structure_radius": 5, "immune_density": 0.5}
DEMUX_CANCER = {"division_rate": 0.6, "death_rate": 0.05, "max_birth_rate": 0.98,
                "mutation_rate": 0.7, "dispersal_rate": 0.3}
DEMUX_IMMUNE = {"division_rate": 0.0, "death_rate": 0.1, "dispersal_rate": 0.1}


def demux_cohort(n_patients=8, steps=320, n_germline=40):
    """Solid tumours WITH a normal compartment (epithelial/stromal + immune), each patient carrying its
    own PRIVATE germline markers (the individual background genetic demux exploits). Pooled N:1, the
    patient-of-origin answer key is the private germline fingerprint carried by EVERY cell."""
    return Cohort(patient_seeds=list(range(1, n_patients + 1)), genome_params=DEMUX_GENOME,
                  selection_params=DEMUX_SEL, cancer_cell_params=DEMUX_CANCER, deme_params=DEMUX_DEME,
                  spatial_params=DEMUX_SPATIAL, immune_cell_params=DEMUX_IMMUNE, grow_steps=steps,
                  n_germline_markers=n_germline).run()


def demux_analysis(cohort, capacity=None, n_cells_per_patient=120):
    """Pool all patients into one batch (N:1) and assign EVERY pooled cell — cancer AND normal — back to
    its patient-of-origin from its private germline variants (souporcell/vireo-style). Returns overall
    accuracy and the per-compartment breakdown (normal cells are demuxable ONLY because germline is
    carried by every cell of the individual, not just the tumour)."""
    from sklearn.cluster import AgglomerativeClustering
    K = cohort.n_patients
    pooled, meta = pool_cell_data([cohort.patients[p] for p in range(K)],
                                  n_cells_per_patient=n_cells_per_patient)
    types = meta["cell_type"].values
    v = (pooled["cell_snv"].values > 0).astype(float)           # ALL pooled cells (cancer + normal)
    truth = meta["patient"].values.astype(int)
    freq = v.mean(0)
    keep = (freq >= 0.01) & (freq <= 0.95)
    pred = AgglomerativeClustering(n_clusters=K).fit_predict(v[:, keep])
    is_cancer = types == "cancer"
    is_normal = np.isin(types, ["epithelial", "stromal", "immune"])
    out = dict(accuracy=match_accuracy(pred, truth, K),
               cancer_accuracy=match_accuracy(pred[is_cancer], truth[is_cancer], K) if is_cancer.any() else float("nan"),
               normal_accuracy=match_accuracy(pred[is_normal], truth[is_normal], K) if is_normal.any() else float("nan"),
               chance=1.0 / K, n_cells=len(truth), n_cancer=int(is_cancer.sum()), n_normal=int(is_normal.sum()),
               n_sites=int(keep.sum()), pred=pred, truth=truth,
               method="private germline clustering (cancer + normal cells)")
    return out
