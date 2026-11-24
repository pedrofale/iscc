"""Immune subtypes: the immune cells of each deme split into T and NK cells (genotype ``immune``, the only
ones that kill), myeloid cells and B and plasma cells by ``immune_composition``; each subtype has its own
baseline, and the coarse cell type of all three is ``immune``."""
import numpy as np
import pytest

from conftest import GENOME_PARAMS, SELECTION_PARAMS, CANCER_CELL_PARAMS
from iscc.integrations import biological_types, cell_subtypes
from iscc.tumor.models import GenotypeTumor

FIELD = {"grid_size": 24, "structure_radius": 3, "n_glands": 1, "K_duct": 20, "K_stroma": 20,
         "stroma_fill_frac": 0.3, "immune_density": 0.5}
MIX = {"immune": 0.5, "myeloid": 0.3, "bplasma": 0.2}


def _tumour(spatial, **kw):
    return GenotypeTumor(seed=1, genome_params=GENOME_PARAMS, selection_params=SELECTION_PARAMS,
                         cancer_cell_params=CANCER_CELL_PARAMS,
                         deme_params={"carrying_capacity": 20, "initial_cancer_cells": 5},
                         spatial_params=spatial, **kw)


def test_each_stroma_deme_splits_its_immune_cells():
    t = _tumour({**FIELD, "immune_composition": MIX})
    stroma = [d for i, d in enumerate(t.demes) if t.gland_id[i] == -1]
    totals = [sum(d.get(g, 0) for g in MIX) for d in stroma]
    assert set(totals) == {10}                                   # 0.5 of K_stroma, whatever the split
    shares = np.array([[d.get(g, 0) for g in MIX] for d in stroma]).sum(axis=0) / sum(totals)
    np.testing.assert_allclose(shares, list(MIX.values()), atol=0.03)
    assert len({tuple(d.get(g, 0) for g in MIX) for d in stroma}) > 1   # patchy, not one split everywhere


def test_default_is_all_t_and_nk_cells_and_unchanged_landscape():
    plain = _tumour(dict(FIELD))
    mixed = _tumour({**FIELD, "immune_composition": MIX})
    assert "myeloid" not in plain.genotypes and "myeloid" not in plain.celltype_exps
    assert {"myeloid", "bplasma"} <= set(mixed.celltype_exps)
    for ct in plain.celltype_exps:                                # every other baseline is the same
        np.testing.assert_array_equal(plain.celltype_exps[ct], mixed.celltype_exps[ct])


def test_only_t_and_nk_cells_count_toward_killing():
    t = _tumour({**FIELD, "immune_composition": {"myeloid": 0.6, "bplasma": 0.4}})
    for deme in t.demes:
        total, n_normal, n_immune, *_ = t._deme_comp(deme)
        assert n_immune == 0
        assert n_normal == sum(c for g, c in deme.items() if not t._is_cancer(g))   # every immune cell holds a slot
    assert not t._has_immune and t._immune_fraction(t.demes[0]) == 0.0


def test_bad_composition_is_refused():
    with pytest.raises(ValueError, match="immune_composition"):
        _tumour({**FIELD, "immune_composition": {"immune": 0.5, "neutrophil": 0.5}})
    with pytest.raises(ValueError, match="immune_composition"):
        _tumour({**FIELD, "immune_composition": {"immune": 0.5, "myeloid": 0.2}})


def test_labels():
    ids = ["immune", "myeloid", "bplasma", "stromal", "epithelial", "host", "c12"]
    assert list(biological_types(ids)) == ["immune", "immune", "immune", "stromal", "epithelial", "host", "cancer"]
    assert list(cell_subtypes(ids)) == ["T_NK", "myeloid", "B_plasma", "fibroblast", "epithelial", "host", "cancer"]


def test_materialised_subtypes_take_their_own_baseline():
    expr = {"program_params": {"n_programs": 6, "n_genes_per_program": 8}}
    t = _tumour({**FIELD, "immune_composition": MIX}, expression_params=expr)
    t.grow(n_steps=40, seed=1)
    cd = t.make_cell_data()
    gids = cd["cell_type"]["cell_id"].astype(str)
    exp = cd["cell_exp"]
    for g in ("myeloid", "bplasma"):
        cells = gids.index[gids == g]
        assert len(cells) > 0
        # a normal cell's expression is its baseline times its program factor, so the subtypes differ
        # from the T and NK cells in their mean profile
    m = {g: exp.loc[gids.index[gids == g]].mean(axis=0).to_numpy() for g in ("immune", "myeloid", "bplasma")}
    assert np.corrcoef(m["immune"], m["myeloid"])[0, 1] < 0.9
    assert np.corrcoef(m["myeloid"], m["bplasma"])[0, 1] < 0.9


def test_tissue_map_shows_the_share_of_immune_cells():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = _tumour({**FIELD, "immune_composition": MIX})
    ax = t.plot_tissue(color="immune")
    shown = np.ma.filled(ax.get_images()[0].get_array(), np.nan).ravel()
    expected = np.array([sum(c for g, c in d.items() if t.genotypes[g].type == "immune") / sum(d.values())
                         if d else np.nan for d in t.demes[:t.grid_size ** 2]])
    np.testing.assert_allclose(shown, expected)                   # all three subtypes, per deme
    assert np.nanmax(shown) > 0 and np.nanmin(shown) == 0         # immune cells in the stroma, none in the duct
    assert [x.get_text() for x in ax.get_legend().get_texts()] == ["cancer cells"]
    plt.close("all")
