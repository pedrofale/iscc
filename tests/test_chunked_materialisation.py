"""Chunked materialisation and ``make_cell_data(layers=...)`` change memory, never values.

``make_cell_data`` builds the gene-wide frames (SNV, copy number, expression, the allele split, RNA
VAF and BAF) chunk by chunk, and ``layers=`` builds only the frames an assay reads; both leave every
value byte-identical. ``low_memory`` also chunks the program multiplier, which changes only the last
bits of some values and no longer depends on the chunk size. The config
switches on every layer that writes at materialisation (a ``max_cells`` cap, coarsened passengers,
germline and founder variants, hypoxia, CCI, the program layer with allele-specific dosage), and the
chunk is made smaller than one run of identical cells, so a chunk boundary falls inside runs, inside
genotypes and inside demes.
"""
import numpy as np
import pytest

import iscc.tumor.models.count as count
from iscc.data import Visium
from iscc.sample import Resection
from iscc.tumor.components.cell import Cell
from iscc.tumor.models import GenotypeTumor

GENOME = {"n_segments": 6, "segment_size": 40}
SELECTION = {"prop_driver": 0.2, "prop_dispersal": 0.1}
CANCER = {"division_rate": 1.0, "death_rate": 0.02, "mutation_rate": 1.0, "dispersal_rate": 0.3}
DEME = {"carrying_capacity": 20, "initial_cancer_cells": 10}
SPATIAL = {"grid_size": 12, "structure_radius": 3}
MICROENV = {
    "hypoxia": {"strength": 0.8, "n_genes": 10, "o2_source": "perfused", "o2_supply": 0.3},
    "cci": {"strength": 0.6, "n_target_genes": 10, "emitter_type": "cancer", "n_candidate_pairs": 4},
}
EXPRESSION = {
    "program_params": {"n_programs": 4, "n_genes_per_program": 10},
    "activity_params": {"n_active_programs_per_cell": 2, "activity_noise": 0.2},
    "dosage_params": {"dosage_sensitivity_mean": 0.7, "dosage_sensitivity_sd": 0.25,
                      "allele_specific": True},
}


def _tumour(programs=True, max_cells=None):
    Cell._genotype_counter = 0
    t = GenotypeTumor(seed=7, genome_params=GENOME, selection_params=SELECTION,
                      cancer_cell_params=CANCER, deme_params=DEME, spatial_params=SPATIAL,
                      update_mode="tau", tau=0.5, coarsen_passengers=True, max_cells=max_cells,
                      microenv_params=MICROENV, expression_params=EXPRESSION if programs else None,
                      founder_mutations={"n_passenger": 6, "n_driver": 1},
                      germline_params={"het_frac": 0.05, "hom_frac": 0.3})
    t.grow(n_steps=30, seed=7, materialize=False)
    return t


def _same(a, b):
    assert list(a) == list(b)                     # the same frames, in the same order
    for k in a:
        assert a[k].index.equals(b[k].index) and a[k].columns.equals(b[k].columns), k
        assert list(a[k].dtypes) == list(b[k].dtypes), k
        if a[k].dtypes.nunique() == 1 and np.issubdtype(a[k].dtypes.iloc[0], np.number):
            va, vb = np.ascontiguousarray(a[k].values), np.ascontiguousarray(b[k].values)
            assert va.tobytes() == vb.tobytes(), k     # byte-identical, NaNs included
        else:
            assert a[k].equals(b[k]), k


def _cell_data(t, monkeypatch, chunk_rows, **kw):
    if chunk_rows is None:                        # one chunk: every row at once, as before chunking
        monkeypatch.setattr(count, "_MATERIALISE_CHUNK_BYTES", 1 << 60)
    else:
        monkeypatch.setattr(count, "_MATERIALISE_CHUNK_BYTES", 1)
        monkeypatch.setattr(count, "_MATERIALISE_MIN_ROWS", chunk_rows)
    return t.make_cell_data(**kw)


@pytest.mark.parametrize("programs", [True, False])
@pytest.mark.parametrize("max_cells", [None, 300])
def test_chunking_leaves_every_frame_byte_identical(monkeypatch, programs, max_cells):
    t = _tumour(programs=programs, max_cells=max_cells)
    whole = _cell_data(t, monkeypatch, None)
    # A branching tumour uncapped; under the cap a subsample that is still many cells and genotypes.
    assert len(whole["cell_type"]) > 100
    assert whole["cell_type"]["cell_id"].nunique() > (20 if max_cells is None else 3)
    assert ("cell_exp_p" in whole) == programs and "cell_snv_p" in whole
    chunked = _cell_data(t, monkeypatch, 7)
    _same(whole, chunked)


def test_layers_build_only_the_requested_frames_with_the_same_values(monkeypatch):
    t = _tumour()
    full = _cell_data(t, monkeypatch, 7)
    small = ("cell_evo", "cell_crd", "cell_type", "cell_deme")
    for layers, frames in (("visium", {"cell_exp"}),
                           ("scdna", {"cell_snv", "cell_cnv"}),
                           (("cell_rna_baf", "cell_snv_p"), {"cell_rna_baf", "cell_snv_p"})):
        part = _cell_data(t, monkeypatch, 7, layers=layers)
        built = set(part) & set(count.CELL_LAYERS)
        assert built == frames, layers
        assert all(k in part for k in small)
        _same({k: full[k] for k in part}, part)
    with pytest.raises(ValueError, match="unknown layer"):
        t.make_cell_data(layers="cell_bogus")


def test_a_visium_section_built_with_only_its_layers_gives_the_same_spots():
    t = _tumour()
    full = Resection(t).slice(depth_frac=0.5)
    thin = Resection(t).slice(depth_frac=0.5, layers="visium")
    assert "cell_snv" in full and "cell_snv" not in thin
    a = Visium(seed=3).run(full, grid_side=t.grid_size)
    b = Visium(seed=3).run(thin, grid_side=t.grid_size)
    assert np.array_equal(a.spot_counts.values, b.spot_counts.values)
    assert np.array_equal(a.raw_expr.values, b.raw_expr.values)


def test_summing_float32_rows_in_float64_matches_summing_a_float64_copy():
    # Visium.run pools each spot's cells with sum(dtype=float) instead of first copying the whole
    # float32 expression frame to float64; the sums are the same to the bit.
    rng = np.random.default_rng(0)
    for n, g in ((1, 5), (7, 300), (250, 1000), (1000, 37)):
        x = rng.gamma(0.3, 2.0, size=(n, g)).astype(np.float32)
        idx = rng.choice(n, size=max(1, n // 2), replace=False)
        assert np.array_equal(x[idx].sum(axis=0, dtype=float), x.astype(float)[idx].sum(axis=0))


def test_low_memory_does_not_depend_on_the_chunk_and_matches_to_rounding(monkeypatch):
    t = _tumour()
    default = _cell_data(t, monkeypatch, 7)
    a = _cell_data(t, monkeypatch, 7, low_memory=True)
    b = _cell_data(t, monkeypatch, 13, low_memory=True)
    _same(a, b)                                   # the chunk no longer matters
    for k in ("cell_exp_p", "cell_exp_m", "cell_rna_baf"):
        assert np.allclose(a[k].values, default[k].values, rtol=1e-12, atol=0), k
    x, y = a["cell_exp"].values, default["cell_exp"].values
    assert np.allclose(x, y, rtol=1e-6, atol=0)   # float32: at most a rounding step apart
    for k in ("cell_snv", "cell_cnv", "cell_rna_vaf", "cell_program"):
        assert np.array_equal(a[k].values, default[k].values), k    # the multiplier touches none

    # The same switch, as the tumour's own setting.
    t.low_memory = True
    _same(a, _cell_data(t, monkeypatch, 7))
