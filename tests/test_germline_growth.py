"""The germline does not change growth: its sites stay in the mutation pool.

Leaving germline positions out of the pool of later point mutations shrank it by the germline sites,
which changed every later draw and raised the chance that a mutation hits a driver, so growth from one
seed depended on the germline density (483 against 231 generations in one tutorial tumour).
"""
import numpy as np

from conftest import SELECTION_PARAMS, CANCER_CELL_PARAMS, DEME_PARAMS
from iscc.tumor import GenotypeTumor
from iscc.tumor.components.cell import Cell


def _tumour(het, mode="exact"):
    Cell._genotype_counter = 0
    return GenotypeTumor(seed=5, genome_params={"n_segments": 4, "segment_size": 100},
                         selection_params=SELECTION_PARAMS,
                         cancer_cell_params={**CANCER_CELL_PARAMS, "mutation_rate": 0.5},
                         deme_params={**DEME_PARAMS, "carrying_capacity": 12},
                         spatial_params={"grid_size": 12}, update_mode=mode, tau=0.5,
                         germline_params={"het_frac": het, "hom_frac": 0.33}, coarsen_passengers=True)


def test_mutation_pool_of_the_founder_is_the_same_with_and_without_germline():
    bare, germ = _tumour(0.0), _tumour(0.3)
    assert len(germ.germline_sites) > 0
    for t_bare, t_germ in [(bare, germ)]:
        fb, fg = t_bare.genotypes[t_bare.founder_id], t_germ.genotypes[t_germ.founder_id]
        for seg in range(fb.n_segments):
            for hap in ("p", "m"):
                for ab, ag in zip(fb.genome[seg][hap], fg.genome[seg][hap]):
                    np.testing.assert_array_equal(
                        fb._available_sites(t_bare.selection, seg, ab, fb.event_bits, hap),
                        fg._available_sites(t_germ.selection, seg, ag, fg.event_bits, hap))


def test_exact_growth_does_not_depend_on_the_germline_density():
    sizes = []
    for het in (0.0, 0.005, 0.3):
        t = _tumour(het)
        s = []
        for _ in range(6):
            t.grow(n_steps=15, seed=5, materialize=False)
            s.append(t.get_cancer_size())
        sizes.append(s)
    assert sizes[0] == sizes[1] == sizes[2]
