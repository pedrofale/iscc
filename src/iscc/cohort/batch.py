"""Flexible patient->batch multiplexing for the cohort (DESIGN_cohort.md §2.2).

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
    """Return ``(assignment, batches)``: ``assignment`` = {patient_idx: batch_id}; ``batches`` =
    {batch_id: [patient_idx, ...]} (both ordered)."""
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

    Cell names are namespaced ``P{patient_id}::{orig}`` so they never collide. Returns
    ``(pooled_cell_data, meta)`` where ``meta`` is a per-cell DataFrame (indexed by the namespaced
    name) carrying ``patient``, ``subgroup``, and ``orig_cell`` — the cohort ground truth attached to
    the emitted batch. ``n_cells_per_patient`` optionally subsamples each patient (seeded)."""
    rng = np.random.default_rng(base_seed)
    per_key = {k: [] for k in _POOL_KEYS}
    meta_rows = []
    for pr in patient_results:
        cd = pr.tumor.cell_data
        if cd is None:
            pr.tumor.make_cell_data()
            cd = pr.tumor.cell_data
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
                       base_seed=1000, **scrna_kwargs):
    """Emit per-batch scRNA over the cohort with the chosen patient->batch mapping.

    Returns ``(assays, batches, assignment)``: ``assays`` is one run ``scRNA`` per batch (each with
    ``.obs`` carrying ``patient`` / ``subgroup`` / ``clone`` ground truth and ``.batch_id``);
    ``batches`` / ``assignment`` are the maps from :func:`assign_batches`.
    """
    from ..data import scRNA

    if not cohort.patients:
        raise RuntimeError("run the cohort first (cohort.run()) before emitting batches")
    assignment, batches = assign_batches(cohort.n_patients, mapping, capacity, explicit)
    assays = []
    for bid, pids in batches.items():
        pooled, meta = pool_cell_data([cohort.patients[p] for p in pids],
                                      n_cells_per_patient=n_cells_per_patient, base_seed=base_seed + bid)
        assay = scRNA(n_cells=len(meta), protocol=protocol, count_model=count_model,
                      batch_label=f"batch{bid}", seed=base_seed + bid, **scrna_kwargs)
        assay.run(pooled, cell_subset=list(meta.index))
        m = meta.reindex(assay.obs.index)
        assay.obs["patient"] = m["patient"].values
        assay.obs["subgroup"] = m["subgroup"].values
        assay.batch_id = bid
        assays.append(assay)
    return assays, batches, assignment


def concat_cohort_batches(assays):
    """Concatenate per-batch AnnData into one labelled cohort AnnData (patient/subgroup preserved)."""
    import anndata as ad

    adatas = [a.to_anndata() for a in assays]
    combined = ad.concat(adatas, join="outer", label="batch_index", index_unique="-")
    combined.uns["protocol"] = adatas[0].uns.get("protocol")
    combined.uns["count_model"] = adatas[0].uns.get("count_model")
    combined.uns["n_batches"] = len(adatas)
    combined.uns["source"] = "iscc-cohort"
    return combined
