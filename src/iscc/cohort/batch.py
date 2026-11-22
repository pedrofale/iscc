"""Flexible patient->batch multiplexing for the cohort.

A user-specified assignment of patients to sequencing batches, emitted by reusing the scRNA
multi-batch / "confounded" machinery (:mod:`iscc.data.rna`):

  * **1:1** (``mapping="one_to_one"``)  — each patient is its own batch (one biology per batch): the
    multi-patient integration case (biological variation IS the batch variation).
  * **N:1** (``mapping="multiplex"``, ``capacity=k``) — pool up to ``k`` patients into one batch
    (cell-hashing / genetic-multiplexing style): each batch then carries BOTH biological (several
    patients) AND one technical signature (one ``Batch`` realization), and — because patients share a
    lane — a **demultiplexing** ground truth (patient-of-origin per pooled cell).
  * an explicit ``{patient_idx: batch_id}`` dict is also accepted.

Each batch is one ``scRNA`` realization (one ``Batch`` seed = one technical signature) over the POOLED
cells of its patients, so cells from different patients in the same batch share the technical
signature while cells of the same patient across batches (N:1 never does this, but 1:1 replicates
could) would not — the standard batch-effect structure integration methods must untangle.
"""
from collections import defaultdict

import numpy as np
import pandas as pd

# per-cell matrices carried through pooling (gene-columned ones align on the shared genome)
_POOL_KEYS = ("cell_evo", "cell_snv", "cell_cnv", "cell_exp", "cell_crd", "cell_type",
             "cell_deme", "cell_rna_vaf", "cell_microenv")


def assign_batches(n_patients, mapping="one_to_one", capacity=None, explicit=None):
    """Assign the patients of a cohort to sequencing batches.

    Parameters
    ----------
    n_patients : int
        Number of patients.
    mapping : {"one_to_one", "multiplex"}, default "one_to_one"
        ``"one_to_one"`` gives each patient its own batch; ``"multiplex"`` (also ``"n_to_one"`` or
        ``"pool"``) pools ``capacity`` consecutive patients per batch.
    capacity : int, optional
        Patients per batch under ``"multiplex"``. Default None pools every patient into one batch.
    explicit : dict, optional
        ``{patient_idx: batch_id}``, used instead of ``mapping`` when given.

    Returns
    -------
    (dict, dict)
        ``assignment``, ``{patient_idx: batch_id}``, and ``batches``, ``{batch_id: [patient_idx,
        ...]}``, both ordered.
    """
    if explicit is not None:
        assignment = {int(k): int(v) for k, v in dict(explicit).items()}
    elif mapping == "one_to_one":
        assignment = {i: i for i in range(n_patients)}
    elif mapping in ("multiplex", "n_to_one", "pool"):
        cap = int(capacity) if capacity else n_patients
        if cap < 1:
            raise ValueError("capacity must be >= 1")
        assignment = {i: i // cap for i in range(n_patients)}
    else:
        raise ValueError(f"unknown mapping {mapping!r}; use 'one_to_one', 'multiplex', or explicit")
    batches = defaultdict(list)
    for p in range(n_patients):
        batches[assignment[p]].append(p)
    return assignment, dict(sorted(batches.items()))


def _namespace(name, pid):
    return f"P{pid}::{name}"


def pool_cell_data(patient_results, n_cells_per_patient=None, base_seed=0):
    """Concatenate several patients' ``cell_data`` into one pooled ``cell_data`` for a shared batch.

    Cell names are namespaced ``P{patient_id}::{orig}`` so that they never collide.

    Parameters
    ----------
    patient_results : sequence of PatientResult
        The patients to pool, for example ``cohort.patients[i]`` of a
        [`Cohort`][iscc.cohort.Cohort]. The per-cell tables of each are built first if missing.
    n_cells_per_patient : int, optional
        Subsample each patient to this many cells (seeded). Default None keeps every cell.
    base_seed : int, default 0
        Seed of the subsampling.

    Returns
    -------
    (dict, pandas.DataFrame)
        The pooled ``cell_data`` and ``meta``, one row per pooled cell (indexed by the namespaced
        name) with its ``patient``, ``subgroup``, ``orig_cell``, ``clone`` (genotype id) and
        ``cell_type`` (the type of its genotype): the cohort ground truth of the batch.
    """
    rng = np.random.default_rng(base_seed)
    per_key = {k: [] for k in _POOL_KEYS}
    meta_rows = []
    for pr in patient_results:
        cd = pr.cell_data
        idx = np.asarray(cd["cell_exp"].index)
        if n_cells_per_patient is not None and len(idx) > n_cells_per_patient:
            idx = rng.choice(idx, size=int(n_cells_per_patient), replace=False)
        idx = list(idx)
        ns = {c: _namespace(c, pr.patient_id) for c in idx}
        for k in _POOL_KEYS:
            df = cd.get(k)
            if df is None:
                continue
            sub = df.loc[idx].rename(index=ns)
            per_key[k].append(sub)
        clone = cd["cell_type"].loc[idx].iloc[:, 0].astype(str).values
        genos = pr.tumor.genotypes
        # coarse cell type (cancer/epithelial/stromal/immune) — the SHARED-state axis for the
        # shared-vs-private integration ground truth; clone/patient is the PRIVATE axis.
        ctype = [genos[g].type if g in genos else "?" for g in clone]
        for c, cl, ct in zip(idx, clone, ctype):
            meta_rows.append((_namespace(c, pr.patient_id), pr.patient_id, pr.subgroup, c, cl, ct))

    pooled = {}
    for k, parts in per_key.items():
        if parts:
            pooled[k] = pd.concat(parts, axis=0)
    meta = pd.DataFrame(
        meta_rows, columns=["cell", "patient", "subgroup", "orig_cell", "clone", "cell_type"]
    ).set_index("cell")
    return pooled, meta


def run_cohort_batches(cohort, mapping="one_to_one", capacity=None, explicit=None,
                       n_cells_per_patient=100, protocol="10x", count_model="nb",
                       base_seed=1000, batch_kwargs=None, **scrna_kwargs):
    """Measure a grown cohort by scRNA in sequencing batches, with the chosen patient-to-batch mapping.

    Parameters
    ----------
    cohort : Cohort
        A grown [`Cohort`][iscc.cohort.Cohort] (after ``run``, or loaded).
    mapping, capacity, explicit
        The patient-to-batch mapping, as in [`assign_batches`][iscc.cohort.assign_batches].
    n_cells_per_patient : int, default 100
        Cells drawn from each patient into its batch.
    protocol : {"10x", "smartseq3"}, default "10x"
        The scRNA protocol preset of every batch.
    count_model : {"nb", "dm"}, default "nb"
        The count model of [`scRNA`][iscc.data.scRNA].
    base_seed : int, default 1000
        Batch ``b`` draws its cells and its technical noise from ``base_seed + b``.
    batch_kwargs : list of dict, optional
        Each batch's own technical settings, one mapping per batch in batch order, applied on top of
        ``scrna_kwargs``, as in [`run_scrna_batches`][iscc.data.run_scrna_batches] (an empty mapping
        keeps the shared settings).
    **scrna_kwargs
        Further keywords of [`scRNA`][iscc.data.scRNA], shared by every batch.

    Returns
    -------
    (list of scRNA, dict, dict)
        ``assays``, one run [`scRNA`][iscc.data.scRNA] per batch, each with ``obs`` carrying the
        ``patient``, ``subgroup`` and ``clone`` of every cell and a ``batch_id``; and ``batches``
        and ``assignment``, the maps of [`assign_batches`][iscc.cohort.assign_batches].
    """
    from ..data import scRNA
    from ..data.rna import _check_batch_kwargs

    if not cohort.patients:
        raise RuntimeError("run the cohort first (cohort.run()) before emitting batches")
    assignment, batches = assign_batches(cohort.n_patients, mapping, capacity, explicit)
    per_batch = _check_batch_kwargs(batch_kwargs, len(batches))
    assays = []
    for (bid, pids), own in zip(batches.items(), per_batch):
        pooled, meta = pool_cell_data([cohort.patients[p] for p in pids],
                                      n_cells_per_patient=n_cells_per_patient, base_seed=base_seed + bid)
        assay = scRNA(n_cells=len(meta), protocol=protocol, count_model=count_model,
                      batch_label=f"batch{bid}", seed=base_seed + bid, **{**scrna_kwargs, **own})
        assay.run(pooled, cell_subset=list(meta.index))
        m = meta.reindex(assay.obs.index)
        assay.obs["patient"] = m["patient"].values
        assay.obs["subgroup"] = m["subgroup"].values
        assay.batch_id = bid
        assays.append(assay)
    return assays, batches, assignment


def concat_cohort_batches(assays):
    """Concatenate the batches of [`run_cohort_batches`][iscc.cohort.run_cohort_batches] into one
    AnnData, keeping the patient and subgroup of every cell.

    Parameters
    ----------
    assays : list of scRNA
        The run batches.

    Returns
    -------
    anndata.AnnData
        The cells of every batch, with ``obs["batch_index"]`` naming the batch.
    """
    import anndata as ad

    adatas = [a.to_anndata() for a in assays]
    combined = ad.concat(adatas, join="outer", label="batch_index", index_unique="-")
    combined.uns["protocol"] = adatas[0].uns.get("protocol")
    combined.uns["count_model"] = adatas[0].uns.get("count_model")
    combined.uns["n_batches"] = len(adatas)
    combined.uns["source"] = "iscc-cohort"
    return combined
