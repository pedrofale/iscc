"""Companion test for validation/validate_spatial_diagnostic.py.

Exercises the diagnostic's core claims on a small ductal-field tumour: (a) genetic PC1 has POSITIVE
spatial autocorrelation (clonal territories exist — the thing a genotype-id map hides), and (b) the
invasive (emt) program is niche-driven (higher where the epithelial fraction is higher).
"""
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "validation"))
import validate_spatial_diagnostic as vsd  # noqa: E402


def _occupied(grid, coords):
    return [i for i, (r, c) in enumerate(coords) if not np.isnan(grid[r, c])]


def test_spatial_clonal_territories_and_niche_expression():
    t = vsd.grow(vsd.LOW_DISP, gen=18, seed=3)
    mask = vsd.cancer_mask(t)
    assert mask.sum() > 20                                   # enough cancer to have structure

    # genetic PC1 -> full-length per-cell, finite on cancer cells
    pc1 = vsd.pc1_per_cell(t, mask)
    assert pc1.shape[0] == len(mask)
    assert np.isfinite(pc1[mask]).all()

    # per-deme PC1 has POSITIVE spatial autocorrelation => spatial clonal territories
    grid = vsd.deme_mean_grid(t, pc1, mask)
    moran = vsd.morans_i_of_grid(t, grid)
    assert np.isfinite(moran) and moran > 0.1

    # the emt program is niche-driven: per-deme emt tracks the epithelial fraction
    cd = t.cell_data
    emt_k = list(t.programs.dictionary.program_names).index("emt")
    emt = vsd.deme_mean_grid(t, cd["cell_program"].values[:, emt_k], mask)
    epi = t.microenv_truth["epithelial"]
    occ = _occupied(emt, t.deme_coords)
    ev = np.array([emt[t.deme_coords[i]] for i in occ])
    pv = np.array([epi[i] for i in occ])
    assert np.ptp(pv) > 0 and np.corrcoef(ev, pv)[0, 1] > 0.0
