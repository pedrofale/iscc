"""Stromal subtypes: the stromal cells of each deme split into fibroblasts (genotype ``stromal``),
endothelial cells and perivascular cells by ``stromal_composition``; every subtype counts as stroma for
the stromal hazard and for eviction, so the split changes the data and not the growth."""
import numpy as np
import pytest

from conftest import GENOME_PARAMS, SELECTION_PARAMS, CANCER_CELL_PARAMS
from iscc.integrations import biological_types, cell_subtypes
from iscc.tumor.models import GenotypeTumor

# stroma demes seeded full, so that cancer cells entering the stroma must evict stromal cells
FIELD = {"grid_size": 16, "structure_radius": 3, "n_glands": 1, "K_duct": 20, "K_stroma": 20,
         "stroma_fill_frac": 1.0, "stromal_hazard": 0.3}
MIX = {"stromal": 0.35, "endothelial": 0.4, "pvl": 0.25}


def _tumour(spatial, seed=3):
    return GenotypeTumor(seed=seed, genome_params=GENOME_PARAMS, selection_params=SELECTION_PARAMS,
                         cancer_cell_params=CANCER_CELL_PARAMS,
                         deme_params={"carrying_capacity": 20, "initial_cancer_cells": 5},
                         spatial_params=spatial)


def _stroma(t):
    return np.array([sum(c for g, c in d.items() if t.genotypes[g].type == "stromal") for d in t.demes])


def _cancer(t):
    return np.array([sum(c for g, c in d.items() if t._is_cancer(g)) for d in t.demes])


def test_each_stroma_deme_splits_its_stromal_cells():
    t = _tumour({**FIELD, "stromal_composition": MIX})
    stroma = [d for i, d in enumerate(t.demes) if t.gland_id[i] == -1]
    assert {sum(d.get(g, 0) for g in MIX) for d in stroma} == {20}
    shares = np.array([[d.get(g, 0) for g in MIX] for d in stroma]).sum(axis=0) / (20 * len(stroma))
    np.testing.assert_allclose(shares, list(MIX.values()), atol=0.03)
    assert all(t.genotypes[g].type == "stromal" for g in MIX)
    assert {"endothelial", "pvl"} <= set(t.celltype_exps)


def test_the_split_leaves_growth_unchanged():
    plain, mixed = _tumour(dict(FIELD)), _tumour({**FIELD, "stromal_composition": MIX})
    seeded = _stroma(plain).sum()
    for t in (plain, mixed):
        t.grow(n_steps=60_000, target_cancer=1500, materialize=False)
    assert _stroma(plain).sum() < seeded                         # cancer evicted stromal cells
    assert plain.time == mixed.time
    np.testing.assert_array_equal(_cancer(plain), _cancer(mixed))
    np.testing.assert_array_equal(_stroma(plain), _stroma(mixed))
    clones = lambda t: sorted(c for g, c in t.genotypes_counts.items() if t._is_cancer(g))
    assert clones(plain) == clones(mixed)


def test_a_grown_tumour_takes_the_split_afterwards():
    t = _tumour(dict(FIELD))
    t.grow(n_steps=60_000, target_cancer=1500, materialize=False)
    before, hazard = _stroma(t), [t._stromal_fraction(d) for d in t.demes]
    t.split_stroma(MIX)
    np.testing.assert_array_equal(_stroma(t), before)            # the same stromal cells, now by subtype
    assert [t._stromal_fraction(d) for d in t.demes] == hazard    # and the same stromal hazard
    counts = np.array([sum(d.get(g, 0) for d in t.demes) for g in MIX], dtype=float)
    np.testing.assert_allclose(counts / counts.sum(), list(MIX.values()), atol=0.03)
    with pytest.raises(ValueError, match="already split"):
        t.split_stroma(MIX)


def _same_genome(a, b):
    return all(np.array_equal(x, y) for sa, sb in zip(a.genome, b.genome) for h in ("p", "m")
               for x, y in zip(sa[h], sb[h]))


@pytest.mark.parametrize("when", ["seeding", "afterwards"])
def test_the_subtypes_carry_the_germline_of_the_fibroblasts(when):
    germline = {"het_frac": 0.2, "seed": 5}
    spatial = {**FIELD, "stromal_composition": MIX} if when == "seeding" else dict(FIELD)
    t = GenotypeTumor(seed=3, genome_params=GENOME_PARAMS, selection_params=SELECTION_PARAMS,
                      cancer_cell_params=CANCER_CELL_PARAMS,
                      deme_params={"carrying_capacity": 20, "initial_cancer_cells": 5},
                      spatial_params=spatial, germline_params=germline)
    assert len(t.germline_sites) > 0
    if when == "afterwards":
        t.grow(n_steps=2_000, materialize=False)
        t.split_stroma(MIX)
    fibroblast = t.genotypes["stromal"]
    assert any(x.any() for seg in fibroblast.genome for h in ("p", "m") for x in seg[h])   # the germline
    for g in ("endothelial", "pvl"):
        assert _same_genome(t.genotypes[g], t.genotypes["stromal"])


def test_default_is_all_fibroblasts_and_bad_composition_is_refused():
    t = _tumour(dict(FIELD))
    assert not {"endothelial", "pvl"} & set(t.genotypes)
    assert "endothelial" not in t.celltype_exps
    with pytest.raises(ValueError, match="stromal_composition"):
        _tumour({**FIELD, "stromal_composition": {"stromal": 0.5, "adipocyte": 0.5}})
    with pytest.raises(ValueError, match="stromal_composition"):
        _tumour({**FIELD, "stromal_composition": {"stromal": 0.5, "pvl": 0.2}})


def test_labels():
    ids = ["stromal", "endothelial", "pvl", "immune", "epithelial", "c12"]
    assert list(biological_types(ids)) == ["stromal", "stromal", "stromal", "immune", "epithelial", "cancer"]
    assert list(cell_subtypes(ids)) == ["fibroblast", "endothelial", "PVL", "T_NK", "epithelial", "cancer"]


def test_materialised_subtypes_take_their_own_baseline():
    expr = {"program_params": {"n_programs": 6, "n_genes_per_program": 8}}
    t = GenotypeTumor(seed=3, genome_params=GENOME_PARAMS, selection_params=SELECTION_PARAMS,
                      cancer_cell_params=CANCER_CELL_PARAMS,
                      deme_params={"carrying_capacity": 20, "initial_cancer_cells": 5},
                      spatial_params={**FIELD, "stromal_composition": MIX}, expression_params=expr)
    t.grow(n_steps=40, seed=1)
    cd = t.make_cell_data()
    gids = cd["cell_type"]["cell_id"].astype(str)
    m = {g: cd["cell_exp"].loc[gids.index[gids == g]].mean(axis=0).to_numpy() for g in MIX}
    assert np.corrcoef(m["stromal"], m["endothelial"])[0, 1] < 0.9
    assert np.corrcoef(m["endothelial"], m["pvl"])[0, 1] < 0.9


def test_recovery_given_for_the_stroma_applies_to_its_subtypes():
    from iscc.sample.dissociation import DEFAULT_RECOVERY, merge_recovery
    assert merge_recovery()["endothelial"] == DEFAULT_RECOVERY["endothelial"]
    probs = merge_recovery({"stromal": 0.05})
    assert probs["stromal"] == probs["endothelial"] == probs["pvl"] == 0.05
    assert merge_recovery({"stromal": 0.05, "pvl": 0.5})["pvl"] == 0.5
