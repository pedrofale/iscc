"""iscc.cohort — the multi-patient COHORT layer (DESIGN_cohort.md).

Run many patients (tumours) separately over a SHARED, config-determined driver landscape (the engine
seed-decoupling fix, DESIGN_cohort.md §1), pool them into sequencing batches flexibly (1:1 or N:1
multiplexing), and surface cohort-level ground truth — recurrent-vs-private drivers, per-cell
patient-of-origin, shared-vs-private cell states, and (coupled to :mod:`iscc.treatment`) per-patient
subgroup + true therapy response. This is the ground truth for benchmarking multi-patient integration,
demultiplexing, recurrence/driver detection, and personalized medicine / patient stratification.

    from iscc.cohort import Cohort, Subgroup, run_cohort_batches
    cohort = Cohort(patient_seeds=range(20), genome_params=..., selection_params=..., ...).run()
    assays, batches, assignment = run_cohort_batches(cohort, mapping="multiplex", capacity=4)
"""
from .cohort import Cohort, Subgroup, PatientResult
from .batch import (assign_batches, pool_cell_data, run_cohort_batches, concat_cohort_batches)
from .groundtruth import (recurrence_table, true_recurrent_drivers, patient_gene_mutated,
                          private_mutation_table, shared_private_labels, subgroup_response_table)

__all__ = [
    "Cohort", "Subgroup", "PatientResult",
    "assign_batches", "pool_cell_data", "run_cohort_batches", "concat_cohort_batches",
    "recurrence_table", "true_recurrent_drivers", "patient_gene_mutated",
    "private_mutation_table", "shared_private_labels", "subgroup_response_table",
]
