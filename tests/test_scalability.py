"""Tests for the spatial-scale scalability features (DESIGN_scalability.md §8-9):

- **passenger coarsening** (``coarsen_passengers``): folds neutral passenger mutations during growth
  so #genotypes tracks distinct CLONES, not cells. OFF by default (byte-identical to the exact
  per-genotype engine — see test_count_engine / test_tau_leaping, which stay green). ON, it is
  statistically equivalent (clonal selection intact) but not byte-identical, and per-cell passengers
  are reconstructed at materialisation.
- **assay-memory cap** (``max_cells`` / ``make_cell_data(region=...)``): materialise a representative
  subsample (or a dense spatial window) instead of every cell, so the per-cell frames stay bounded at
  cm-scale.
"""
import numpy as np
import pytest

from conftest import GENOME_PARAMS, SELECTION_PARAMS, CANCER_CELL_PARAMS, DEME_PARAMS
from iscc.tumor.models import GenotypeTumor

SPATIAL = {"grid_size": 24, "structure_radius": 0}
CANCER = {**CANCER_CELL_PARAMS, "death_rate": 0.05, "n_snvs_per_allele": 0.1, "cnv_prob": 0.05}


def _grow(seed=1, steps=45, coarsen=False, max_cells=None, mode="tau"):
    t = GenotypeTumor(
        seed=seed, genome_params=GENOME_PARAMS, selection_params=SELECTION_PARAMS,
        cancer_cell_params=CANCER, deme_params=DEME_PARAMS, spatial_params=SPATIAL,
        update_mode=mode, tau=0.5, snapshot_every=1,
        coarsen_passengers=coarsen, max_cells=max_cells)
    t.grow(n_steps=steps, seed=seed)
    return t


# --- passenger coarsening -----------------------------------------------------------------------
def test_coarsen_off_is_the_default():
    t = GenotypeTumor(seed=1, genome_params=GENOME_PARAMS, selection_params=SELECTION_PARAMS,
                      cancer_cell_params=CANCER, deme_params=DEME_PARAMS, spatial_params=SPATIAL)
    assert t.coarsen_passengers is False


def test_coarsening_is_reproducible():
    a = _grow(seed=3, coarsen=True)
    b = _grow(seed=3, coarsen=True)
    assert a.get_tumor_size() == b.get_tumor_size()
    assert np.array_equal(a.cell_data["cell_snv"].values, b.cell_data["cell_snv"].values)
    assert np.array_equal(a.cell_data["cell_cnv"].values, b.cell_data["cell_cnv"].values)


def test_coarsening_folds_passengers_and_reduces_genotypes():
    """The whole point: at a comparable tumour size, coarsening has FAR fewer distinct genotypes
    (neutral passengers are folded into their parent clone) and records the fold count."""
    off = _grow(seed=5, coarsen=False)
    on = _grow(seed=5, coarsen=True)
    # comparable sizes (statistically equivalent dynamics)
    assert 0.5 < on.get_tumor_size() / off.get_tumor_size() < 2.0
    # coarsening actually folded passenger divisions ...
    assert on._n_folded > 0
    # ... and so carries many fewer distinct genotypes than the exact per-genotype engine
    assert len(on.genotypes_counts) < 0.75 * len(off.genotypes_counts)


def test_coarsening_statistically_equivalent_sizes():
    seeds = range(8)
    on = [_grow(s, coarsen=True).get_cancer_size() for s in seeds]
    off = [_grow(s, coarsen=False).get_cancer_size() for s in seeds]
    on_mean = np.mean([x for x in on if x > 0])
    off_mean = np.mean([x for x in off if x > 0])
    assert 0.5 < on_mean / off_mean < 2.0


def test_coarsening_preserves_selection():
    """Selection must survive coarsening: with drivers, the cell-weighted division rate evolves
    ABOVE the configured baseline (fitter clones dominate) — the same as with coarsening off."""
    for coarsen in (False, True):
        t = _grow(seed=2, steps=60, coarsen=coarsen)
        base = t.genotypes[t.founder_id].baseline_rates["division_rate"]
        gids = [g for g in t.genotypes_counts if t._is_cancer(g)]
        counts = np.array([t.genotypes_counts[g] for g in gids], dtype=float)
        divs = np.array([t.genotypes[g].evolutionary_parameters["division_rate"] for g in gids])
        cw = (divs * counts).sum() / counts.sum()
        assert cw > base + 1e-6, f"coarsen={coarsen}: no selection (cw {cw} <= base {base})"


def test_passenger_reconstruction_gives_realistic_tmb():
    """Coarsening drops passenger genotypes during growth but re-emits per-cell passenger SNVs at
    materialisation, so cancer cells still carry a neutral mutation burden beyond their drivers."""
    t = _grow(seed=4, steps=55, coarsen=True)
    assert t._pass_load, "expected some folded passenger burden"
    snv = t.cell_data["cell_snv"].values
    ct = t.cell_data["cell_type"]["cell_id"]
    is_can = ct.map(lambda g: t.genotypes[g].type == "cancer").values
    tmb = (snv[is_can] > 0).sum(axis=1)
    assert tmb.mean() > 0  # cancer cells carry SNVs (drivers + reconstructed passengers)
    # reconstruction is reproducible
    t2 = _grow(seed=4, steps=55, coarsen=True)
    assert np.array_equal(snv, t2.cell_data["cell_snv"].values)


# --- assay-memory cap ---------------------------------------------------------------------------
def test_max_cells_caps_materialisation():
    full = _grow(seed=1, coarsen=True, max_cells=None)
    n_full = full.get_tumor_size()
    cap = max(50, n_full // 4)
    capped = _grow(seed=1, coarsen=True, max_cells=cap)
    # same tumour (growth is identical; only materialisation differs)
    assert capped.get_tumor_size() == n_full
    n_mat = capped.cell_data["cell_snv"].shape[0]
    assert n_mat <= n_full
    assert n_mat < 1.3 * cap  # bounded near the cap (Binomial subsample)
    assert n_mat > 0.3 * cap
    # all frames agree in row count
    for k, df in capped.cell_data.items():
        assert df.shape[0] == n_mat


def test_subsample_preserves_celltype_proportions():
    """The subsample is representative: the cancer fraction of a capped materialisation matches the
    ground-truth cancer fraction of the whole tumour, within sampling noise."""
    t = _grow(seed=6, coarsen=True, max_cells=None)
    true_frac = t.get_cancer_size() / t.get_tumor_size()
    t2 = _grow(seed=6, coarsen=True, max_cells=max(80, t.get_tumor_size() // 3))
    ct = t2.cell_data["cell_type"]["cell_id"]
    sub_frac = ct.map(lambda g: t2.genotypes[g].type == "cancer").mean()
    assert abs(sub_frac - true_frac) < 0.15


def test_region_materialisation_is_dense_and_local():
    """make_cell_data(region=...) materialises ONLY the given demes, at FULL local density — what a
    spatial (Visium) assay needs to sample a dense window without a diluting whole-tumour subsample."""
    t = _grow(seed=1, coarsen=True, max_cells=200)
    G = t.grid_size
    center = (G // 2, G // 2)
    window = t.primary_window(side=8, center=center)
    cd = t.make_cell_data(region=window)
    demes = set(cd["cell_deme"]["deme_id"])
    assert demes.issubset(set(window))                      # nothing outside the window
    # full local density: the window's materialised cell count == the true count in those demes
    true_in_window = sum(sum(t.demes[d].values()) for d in window)
    assert cd["cell_snv"].shape[0] == true_in_window
