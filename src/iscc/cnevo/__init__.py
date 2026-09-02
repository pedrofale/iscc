"""Copy-number evolution analysis over iscc simulations.

Seven questions, each with its own metric set, computed from one grown tumour. They are siblings,
not steps: none is an input to another, and each is reported on its own terms.

===  ==========================================================  ==========================
Q    Question                                                    Entry point
===  ==========================================================  ==========================
Q1   Clonal dynamics and sweeps                                  :func:`sweep_metrics`
Q2   Clonal diversity over time                                  :func:`diversity_trajectory`
Q3   Demography: r- vs K-phase                                   :func:`growth_phase`
Q4   Copy-number landscape                                       :func:`cn_landscape`
Q5   CN data quality of a sampled clone set                      :func:`data_quality`
Q6   Tree-reconstruction potential                               :func:`reconstruction_potential`
Q7   Spatial structure and multi-focality (glandular runs only)  :func:`spatial_structure`
===  ==========================================================  ==========================

Q1-Q4 describe the whole run and need no sampling. Q5-Q6 take a set of sampled clones, chosen with
:func:`select_clones`. Q7 returns ``None`` unless the tumour has a glandular substrate.

The ground truth all of them read is exact: ``tumor.traces`` holds per-generation clone counts and
``tumor.genotypes`` retains every genotype ever created, so ancestral copy number and the CNA event
log are recovered rather than inferred.

Grow the tumour with ``trace_occupancy=True`` to get the per-deme occupancy Q3 and Q7 use; without
it those fields are ``None`` and everything else still works.
"""
from .dynamics import clone_size_matrix, diversity_trajectory, growth_phase, sweep_metrics
from .events import cna_event_table, inherited_event_counters, pairwise_shared_matrix
from .landscape import cn_landscape
from .phylo import (
    data_quality, nj_rf, normalized_rf, reconstruction_potential, true_clone_tree,
)
from .profile import (
    breakpoint_sets, clone_segment_cn, segment_allele_cn, segment_cn, segment_coordinates,
    select_clones, to_medicc2_input,
)
from .structure import is_structured, spatial_structure

__all__ = [
    # Q1-Q3
    "sweep_metrics", "diversity_trajectory", "growth_phase", "clone_size_matrix",
    # Q4
    "cn_landscape",
    # Q5-Q6
    "data_quality", "reconstruction_potential", "true_clone_tree", "normalized_rf", "nj_rf",
    # Q7
    "spatial_structure", "is_structured",
    # substrate
    "segment_cn", "segment_allele_cn", "clone_segment_cn", "segment_coordinates",
    "breakpoint_sets", "select_clones", "to_medicc2_input",
    "cna_event_table", "inherited_event_counters", "pairwise_shared_matrix",
]
