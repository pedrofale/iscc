"""iscc.cohort — the multi-patient COHORT layer.

Run many patients (tumours) separately over a SHARED, config-determined driver landscape (set by
the layout seed, not the per-patient evolution seed), pool them into sequencing batches flexibly
(1:1 or N:1 multiplexing), and surface cohort-level ground truth — recurrent-vs-private drivers, per-cell
patient-of-origin, shared-vs-private cell states, and (coupled to :mod:`iscc.treatment`) per-patient
subgroup + true therapy response. This is the ground truth for benchmarking multi-patient integration,
demultiplexing, recurrence/driver detection, and personalized medicine / patient stratification.

    import functools
    from iscc.cohort import Cohort, grow_tumor, run_cohort_batches
    grow = functools.partial(grow_tumor, genome_params=..., selection_params=..., grow_steps=300)
    cohort = Cohort(grow, patient_seeds=range(20)).run()
    assays, batches, assignment = run_cohort_batches(cohort, mapping="multiplex", capacity=4)

``Cohort`` is a thin utility: it calls one tumour's growth function once per patient, with the
patient's seed and the shared layout seed (see :mod:`iscc.cohort.cohort`).
"""
from .cohort import (Cohort, Subgroup, PatientResult, grow_tumor, shared_selection, assign_subgroups,
                     draw_germline_markers, patient_overrides, clone_key)
from .batch import (assign_batches, pool_cell_data, run_cohort_batches, concat_cohort_batches)
from .groundtruth import (recurrence_table, true_recurrent_drivers, patient_gene_mutated,
                          private_mutation_table, shared_private_labels, subgroup_response_table)
from .hashing import emit_cell_hashtags, demux_hashtags

__all__ = [
    "Cohort", "Subgroup", "PatientResult", "grow_tumor", "shared_selection", "assign_subgroups",
    "draw_germline_markers", "patient_overrides", "clone_key",
    "assign_batches", "pool_cell_data", "run_cohort_batches", "concat_cohort_batches",
    "recurrence_table", "true_recurrent_drivers", "patient_gene_mutated",
    "private_mutation_table", "shared_private_labels", "subgroup_response_table",
    "emit_cell_hashtags", "demux_hashtags",
]
