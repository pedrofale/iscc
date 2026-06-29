"""CNA validation: copy number tracks oncogenic content under selection, not drift.

A fast check of the cancer CNA selection signature (oncogene amplification / TSG deletion;
Beroukhim et al. 2010, Davoli et al. 2013): under iscc's CINner-style selection the per-segment
copy number correlates with net oncogenic content, while under neutral drift it does not. The
full sweep + figure is in validation/validate_cna.py.
"""
import numpy as np

from iscc.tumor.models import GenotypeTumor
from iscc.validation import (segment_copy_numbers, segment_driver_content,
                             cna_amplification_signature)

GENOME = {"n_segments": 12, "segment_size": 50}
DEME = {"carrying_capacity": 5}
SPATIAL = {"grid_size": 19, "structure_radius": 0}
# snv_prob is held low so this test isolates the *CNA* selection signature (oncogene
# amplification / TSG deletion). With the genome-wide, ploidy-scaled SNV model
# (n_snvs_per_allele), abundant driver SNVs accumulate across all oncogene loci and saturate
# fitness, washing out the marginal advantage of amplifying oncogene-rich segments — a real
# dynamic, but a confound for a test whose job is the CNA signature. Weighting events toward
# CNAs makes the amplification signal observable (pure-CNA s_corr≈0.42; here ≈0.34).
CANCER = {"division_rate": 0.4, "death_rate": 0.02, "max_birth_rate": 0.95,
          "mutation_rate": 0.7, "dispersal_rate": 0.2, "snv_prob": 0.2, "cnv_prob": 0.8}


def _selection(driver_effects):
    return {"prop_driver": 0.2, "prop_dispersal": 0.0, "prop_immune_resistance": 0.0,
            "prop_treatment_resistance": 0.0, "driver_effects": driver_effects,
            "dispersal_effects": 1.0, "treatment_resistant_effects": 1.0,
            "immune_resistant_effects": 1.0}


def _grow(driver_effects, seed, steps=900):
    t = GenotypeTumor(seed=seed, genome_params=GENOME, selection_params=_selection(driver_effects),
                      cancer_cell_params=CANCER, deme_params=DEME, spatial_params=SPATIAL)
    t.grow(steps, seed=seed)
    return t


def test_segment_metrics_basics():
    t = _grow(1.5, seed=0)
    cn = segment_copy_numbers(t)
    net = segment_driver_content(t)
    assert cn.shape == (GENOME["n_segments"],)
    assert net.shape == (GENOME["n_segments"],)
    assert np.all(cn >= 0)              # copy numbers are non-negative
    assert np.all(cn < t.selection.max_cn + 1)
    # net content is bounded by the per-segment driver count
    assert np.all(np.abs(net) <= GENOME["segment_size"])


def test_copy_number_tracks_oncogenic_content_under_selection():
    neutral = [_grow(1.0, s) for s in range(5)]
    selected = [_grow(1.6, s) for s in range(5)]

    _, _, n_slope, n_corr = cna_amplification_signature(neutral)
    _, _, s_slope, s_corr = cna_amplification_signature(selected)

    # selection amplifies oncogene-rich segments; neutral drift does not
    assert s_corr > 0.2
    assert s_slope > 0
    assert s_corr > n_corr + 0.1
