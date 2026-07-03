"""Cohort-level ground truth (DESIGN_cohort.md §2.3).

Exactly the structure that no simulator provides and that real cohorts can never give:

  * :func:`recurrence_table`          — recurrent-vs-private drivers (per-gene recurrence across patients);
  * :func:`true_recurrent_drivers`    — the known recurrent driver set (onc ∪ tsg, the answer key);
  * :func:`private_mutation_table`    — per-patient private (singleton) mutations;
  * :func:`shared_private_labels`     — per-cell (shared cell-state, private patient) label pair;
  * :func:`subgroup_response_table`   — per-patient subgroup + true therapy response.

Everything is read from the grown cohort (per-patient ``cell_data`` + the shared ``Selection``), so it
is genuine ground truth, not a re-inference.
"""
import numpy as np
import pandas as pd


def _cancer_snv(pr):
    """(cancer-cell SNV matrix, gene_names) for one patient."""
    cd = pr.tumor.cell_data
    snv = cd["cell_snv"]
    genos = pr.tumor.genotypes
    gid = cd["cell_type"].iloc[:, 0].astype(str).values
    is_cancer = np.array([(g in genos and genos[g].type == "cancer") for g in gid])
    return snv.values[is_cancer], list(snv.columns)


def patient_gene_mutated(cohort, clonal_thresh=0.05):
    """Boolean (n_patients x n_genes): is gene g mutated in patient p?

    "Mutated in patient" = a fraction >= ``clonal_thresh`` of the patient's CANCER cells carry a
    non-zero SNV VAF at that gene (so a private single-cell artefact is not counted, but a real
    subclonal/clonal driver is)."""
    genes = list(cohort.selection.get_gene_names())
    M = np.zeros((cohort.n_patients, len(genes)), dtype=bool)
    for i, pr in enumerate(cohort.patients):
        cs, _ = _cancer_snv(pr)
        if cs.shape[0] == 0:
            continue
        frac = (cs > 0).mean(axis=0)
        M[i] = frac >= clonal_thresh
    return M, genes


def recurrence_table(cohort, clonal_thresh=0.05):
    """Per-gene recurrence across the cohort with the true driver-role annotation.

    Columns: ``is_oncogene``, ``is_tsg``, ``is_driver``, ``n_patients_mutated``, ``recurrence``
    (fraction of patients), ``is_recurrent`` (>= 2 patients). The recurrent DRIVERS should sit at high
    recurrence and the private passengers at low — the answer key for the recurrence/driver-detection
    benchmark."""
    M, genes = patient_gene_mutated(cohort, clonal_thresh=clonal_thresh)
    sel = cohort.selection
    onc = set(int(x) for x in sel.get_oncogenes())
    tsg = set(int(x) for x in sel.get_tsgs())
    n_mut = M.sum(axis=0)
    df = pd.DataFrame({
        "gene": genes,
        "is_oncogene": [i in onc for i in range(len(genes))],
        "is_tsg": [i in tsg for i in range(len(genes))],
        "n_patients_mutated": n_mut,
        "recurrence": n_mut / max(cohort.n_patients, 1),
    })
    df["is_driver"] = df["is_oncogene"] | df["is_tsg"]
    df["is_recurrent"] = df["n_patients_mutated"] >= 2
    return df.set_index("gene")


def true_recurrent_drivers(cohort):
    """The known driver gene indices (onc ∪ tsg) — the recurrent-driver answer key."""
    sel = cohort.selection
    return np.array(sorted(set(int(x) for x in sel.get_oncogenes()) |
                           set(int(x) for x in sel.get_tsgs())), dtype=int)


def private_mutation_table(cohort, clonal_thresh=0.05):
    """Per-patient PRIVATE (mutated in exactly one patient) genes. Returns a DataFrame with one row
    per (patient, gene) private mutation plus a driver flag."""
    M, genes = patient_gene_mutated(cohort, clonal_thresh=clonal_thresh)
    n_mut = M.sum(axis=0)
    private_gene = np.where(n_mut == 1)[0]
    sel = cohort.selection
    drivers = set(int(x) for x in sel.get_oncogenes()) | set(int(x) for x in sel.get_tsgs())
    rows = []
    for g in private_gene:
        p = int(np.where(M[:, g])[0][0])
        rows.append((cohort.patients[p].patient_id, genes[g], int(g), g in drivers))
    return pd.DataFrame(rows, columns=["patient", "gene", "gene_idx", "is_driver"])


def shared_private_labels(meta):
    """Per-cell (shared_state, private_state) labels from a pooled ``meta`` (from
    :func:`iscc.cohort.batch.pool_cell_data`) or any obs-like frame with ``cell_type`` + ``patient``.

    ``shared_state`` = coarse cell type (cancer/epithelial/stromal/immune) — the axis integration
    SHOULD align across patients; ``private_state`` = patient (the axis it should PRESERVE). Returns a
    DataFrame indexed like ``meta`` with ``shared_state`` and ``private_state`` columns."""
    ct = meta["cell_type"].astype(str).values
    pat = meta["patient"].astype(str).values
    out = pd.DataFrame(index=meta.index)
    out["shared_state"] = ct
    out["private_state"] = pat
    return out


def subgroup_response_table(cohort):
    """Per-patient subgroup + true therapy response (the stratification / biomarker answer key)."""
    rows = []
    for i in range(cohort.n_patients):
        sub = cohort.subgroup_of(i)
        rows.append((i, cohort.patient_seeds[i], sub.name, sub.therapy_response))
    return pd.DataFrame(rows, columns=["patient", "seed", "subgroup", "therapy_response"]).set_index("patient")
