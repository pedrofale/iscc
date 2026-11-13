"""`force_snv`: whether an SNV event that draws no SNVs still changes the genome.

The historical default places one SNV when every per-allele Poisson draw comes up 0. Neither demon
nor CINner forces a mutation, so the paper regime turns it off; then such a division leaves the
genome unchanged and the engine adds the daughter to the parent clone.
"""
import numpy as np

from conftest import GENOME_PARAMS, CANCER_CELL_PARAMS, DEME_PARAMS
from iscc.tumor.models import GenotypeTumor

SELECTION = {"prop_driver": 0.1, "prop_dispersal": 0.0,
             "prop_immune_resistance": 0.0, "prop_treatment_resistance": 0.0}
SPATIAL = {"grid_size": 15, "n_structures": 1, "structure_radius": 0}


def _tumor(**cancer):
    return GenotypeTumor(seed=3, genome_params=GENOME_PARAMS, selection_params=SELECTION,
                         cancer_cell_params={**CANCER_CELL_PARAMS, **cancer},
                         deme_params=DEME_PARAMS, spatial_params=SPATIAL)


def _n_mutated(cell):
    return int(sum(a.sum() for seg in cell.genome for hap in ("p", "m") for a in seg[hap]))


def _empty_snv_event(t, force):
    """One SNV event on a daughter of the founder whose per-allele draws are all 0."""
    child = t.genotypes[t.founder_id].divide()
    child.force_snv = force
    before = _n_mutated(child)
    res = child.mutate(np.random.default_rng(0), t.selection, n_snvs_per_allele=0.0,
                       snv_prob=1.0, cnv_prob=0.0, wgd_rate=0.0)
    return res, _n_mutated(child) - before, child


def test_default_forces_one_snv():
    t = _tumor()
    assert t.genotypes[t.founder_id].force_snv is True
    res, added, child = _empty_snv_event(t, force=True)
    assert res is True
    assert added == 1 and child._n_new_snv == 1


def test_off_leaves_the_genome_unchanged():
    t = _tumor(force_snv=False)
    assert t.genotypes[t.founder_id].force_snv is False      # the config reaches the founder
    res, added, child = _empty_snv_event(t, force=False)
    assert res is False                                        # the engine adds it to the parent
    assert added == 0 and child._n_new_snv == 0


def test_off_lowers_the_mutation_supply_at_a_low_snv_rate():
    def n_genotypes(force):
        t = _tumor(force_snv=force, n_snvs_per_allele=0.002, snv_prob=1.0, cnv_prob=0.0)
        t.grow(n_steps=150, seed=3)
        return len(t.genotypes)
    assert n_genotypes(False) < n_genotypes(True)
