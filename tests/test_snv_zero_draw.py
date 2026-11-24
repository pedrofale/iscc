"""An SNV event whose per-allele Poisson draws all come up 0 leaves the genome unchanged.

Neither demon nor CINner forces a mutation, so neither does iscc: the division returns ``False`` and
the engine adds the daughter to the parent clone.
"""
import numpy as np

from conftest import GENOME_PARAMS, CANCER_CELL_PARAMS, DEME_PARAMS
from iscc.tumor.models import GenotypeTumor

SELECTION = {"prop_driver": 0.1, "prop_dispersal": 0.0,
             "prop_immune_resistance": 0.0, "prop_treatment_resistance": 0.0}
SPATIAL = {"grid_size": 15, "n_structures": 1, "structure_radius": 0}


def _n_mutated(cell):
    return int(sum(a.sum() for seg in cell.genome for hap in ("p", "m") for a in seg[hap]))


def test_an_empty_snv_draw_leaves_the_genome_unchanged():
    t = GenotypeTumor(seed=3, genome_params=GENOME_PARAMS, selection_params=SELECTION,
                      cancer_cell_params=CANCER_CELL_PARAMS, deme_params=DEME_PARAMS,
                      spatial_params=SPATIAL)
    child = t.genotypes[t.founder_id].divide()
    before = _n_mutated(child)
    res = child.mutate(np.random.default_rng(0), t.selection, n_snvs_per_allele=0.0,
                       snv_prob=1.0, cnv_prob=0.0, wgd_rate=0.0)
    assert res is False
    assert _n_mutated(child) == before and child._n_new_snv == 0
