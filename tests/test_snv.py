"""SNV site-frequency-spectrum validation: neutral growth -> 1/f power law.

A fast check that, under neutral dynamics (every mutation a passenger), iscc's bulk VAF
spectrum is dominated by rare variants and its cumulative form fits the neutral 1/f power
law (Williams et al. 2016). The full multi-seed sweep + figure is in
validation/validate_snv.py.
"""
import numpy as np

from conftest import GENOME_PARAMS, CANCER_CELL_PARAMS, DEME_PARAMS
from iscc.tumor.models import GenotypeTumor
from iscc.validation import population_vaf, site_frequency_spectrum, neutral_sfs_rsq

NEUTRAL_SELECTION = {
    "prop_driver": 0.1, "prop_dispersal": 0.0,
    "prop_immune_resistance": 0.0, "prop_treatment_resistance": 0.0,
    "driver_effects": 1.0, "dispersal_effects": 1.0,
    "treatment_resistant_effects": 1.0, "immune_resistant_effects": 1.0,
}
GENOME = {"n_segments": 6, "segment_size": 150}
CANCER = {**CANCER_CELL_PARAMS, "death_rate": 0.02, "mutation_rate": 1.0, "dispersal_rate": 0.2}
DEME = {"carrying_capacity": 10}
SPATIAL = {"grid_size": 25, "n_structures": 1, "structure_radius": 0}


def _neutral_tumor(seed, steps=1200):
    t = GenotypeTumor(
        seed=seed, genome_params=GENOME, selection_params=NEUTRAL_SELECTION,
        cancer_cell_params=CANCER, deme_params=DEME, spatial_params=SPATIAL,
    )
    t.grow(n_steps=steps, seed=seed)
    return t


def test_neutral_snv_spectrum_is_one_over_f():
    t = _neutral_tumor(seed=0)
    vaf = population_vaf(t)
    nz = vaf[vaf > 0]
    assert nz.size > 100  # the tumor accumulated a substantial mutation load

    # rare-variant dominated: most mutations are at low frequency
    assert (nz < 0.1).mean() > 0.8

    # cumulative spectrum is monotone non-increasing in f
    _, M = site_frequency_spectrum(vaf, 0.05, 0.45, 40)
    assert np.all(np.diff(M) <= 0)

    # and well-approximated by the neutral 1/f power law
    rsq, slope = neutral_sfs_rsq(vaf, 0.05, 0.45, 40)
    assert rsq > 0.7
    assert slope > 0  # more mutations at lower frequency


def test_population_vaf_bounds_and_cancer_only():
    t = _neutral_tumor(seed=1)
    vaf_cancer = population_vaf(t, cancer_only=True)
    assert np.all((vaf_cancer >= 0) & (vaf_cancer <= 1))
    # including normal (wild-type) cells only dilutes VAFs, never raises them
    vaf_all = population_vaf(t, cancer_only=False)
    assert np.all(vaf_all <= vaf_cancer + 1e-9)
