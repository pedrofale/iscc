"""Evolutionary-dynamics validation: cell dispersal governs the mode of evolution.

A focused, fast check that the mode gradient holds — low dispersal yields more clonal
diversity and a more finely intermixed (less spatially assorted) tumor than high dispersal —
recapitulating Noble et al. (2022). The full sweep + figure is in validation/validate_evolution.py.
"""
import numpy as np

from conftest import GENOME_PARAMS, SELECTION_PARAMS, CANCER_CELL_PARAMS, DEME_PARAMS
from iscc.tumor.models import GenotypeTumor
from iscc.validation import clone_diversity, spatial_assortment

SPATIAL = {"grid_size": 15, "n_structures": 1, "structure_radius": 0}
CANCER = {**CANCER_CELL_PARAMS, "death_rate": 0.02}


def _run(dispersal, seed, steps=250):
    cancer = {**CANCER, "dispersal_rate": dispersal}
    t = GenotypeTumor(
        seed=seed, genome_params=GENOME_PARAMS, selection_params=SELECTION_PARAMS,
        cancer_cell_params=cancer, deme_params=DEME_PARAMS, spatial_params=SPATIAL,
    )
    t.grow(n_steps=steps, seed=seed)
    cancer_counts = {g: c for g, c in t.genotypes_counts.items() if t._is_cancer(g)}
    dom = {}
    for i, d in enumerate(t.demes):
        cg = {g: c for g, c in d.items() if t._is_cancer(g)}
        if cg:
            dom[i] = max(cg, key=cg.get)
    return clone_diversity(cancer_counts.values()), spatial_assortment(dom, t.grid_size)


def test_dispersal_governs_mode():
    seeds = [0, 1, 2]
    low = [_run(0.05, s) for s in seeds]      # low dispersal -> diverse, intermixed
    high = [_run(1.0, s) for s in seeds]      # high dispersal -> fewer, coherent clones

    low_div = np.mean([d for d, _ in low])
    high_div = np.mean([d for d, _ in high])
    low_ass = np.nanmean([a for _, a in low])
    high_ass = np.nanmean([a for _, a in high])

    # low dispersal is more diverse and less spatially assorted than high dispersal
    assert low_div > high_div
    assert low_ass < high_ass
