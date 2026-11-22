"""Batches from named technical sources (paper 1, Fig 5; decision-scrna-batches-from-technical-sources).

- ``run_scrna_batches(batch_kwargs=...)`` gives each batch its own technical settings on top of the
  shared ones, without moving any cell between batches; ``concat_batches`` records them.
- ``atlas_scrna_batches`` draws each batch the settings fitted to one tumour of Wu et al. 2021, which
  run as ``batch_kwargs``.
- the scRNA assay itself takes per-call hyper-parameter overrides (every ``RNABatchHyperParams``
  field), which the per-batch settings rely on.
- the dissociation-stress signature is two separable steps (choose the genes, apply the bump), and
  ``Dissociation`` still draws exactly what it drew before.
- ``Resection.dissociate`` applies per-patient recovery and stress; the recovered cells and the
  stress genes do not depend on the stress strength, so a patient can be re-dissociated at several
  levels with its recovery fixed.
"""
import numpy as np
import pandas as pd
import pytest

from conftest import GENOME_PARAMS, SELECTION_PARAMS, CANCER_CELL_PARAMS, DEME_PARAMS
from iscc.data import scRNA
from iscc.data.batch import RNABatchHyperParams
from iscc.data.rna import (run_scrna_batches, concat_batches, atlas_scrna_batches, BATCH_OVERRIDE_KEYS,
                           ATLAS_AMBIENT_RANGE)
from iscc.sample import Resection
from iscc.sample.dissociation.dissociation import (
    Dissociation, DEFAULT_RECOVERY, apply_stress, choose_stress_genes)
from iscc.tumor.models import GenotypeTumor


def _cell_data(n_cells=400, n_genes=60, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.lognormal(0.0, 1.0, size=n_genes)
    exp = pd.DataFrame(base[None, :] * rng.beta(0.5, 1.0, size=(n_cells, n_genes)),
                       index=[f"C{i}" for i in range(n_cells)],
                       columns=[f"g{i}" for i in range(n_genes)])
    types = rng.choice(["1", "2", "immune", "stromal", "epithelial"], size=n_cells)
    return {"cell_exp": exp, "cell_type": pd.DataFrame({"cell_id": types}, index=exp.index)}


# --- the assay takes per-call overrides ----------------------------------------------------------
def test_scrna_takes_every_technical_hyperparameter_per_call():
    fields = set(RNABatchHyperParams().to_dict()) - {"protocol"}
    # every field but `protocol` is a constructor override, and the per-batch keys are among them
    import inspect
    params = set(inspect.signature(scRNA.__init__).parameters)
    assert fields <= params
    assert set(BATCH_OVERRIDE_KEYS) <= params
    a = scRNA(protocol="10x", mu_lib=1234.0, dispersion=0.7, ambient_frac=0.01, dropout_mid=0.0,
              doublet_rate=0.02, sigma_lib=0.2, sigma_batch=0.1, depth_batch_sigma=0.0)
    h = a.hypers
    assert (h.mu_lib, h.dispersion, h.ambient_frac, h.dropout_mid, h.doublet_rate, h.sigma_lib,
            h.depth_batch_sigma) == (1234.0, 0.7, 0.01, 0.0, 0.02, 0.2, 0.0)


# --- per-batch settings in run_scrna_batches -------------------------------------------------------
def test_batch_kwargs_set_each_batchs_own_parameters():
    cd = _cell_data()
    per = [dict(mu_lib=800.0, depth_batch_sigma=0.0, doublet_rate=0.0),
           dict(mu_lib=6000.0, dispersion=0.05, depth_batch_sigma=0.0, doublet_rate=0.0),
           {}]
    assays = run_scrna_batches(cd, n_batches=3, design="split", n_cells=120, base_seed=7,
                               dropout_mid=0.0, batch_kwargs=per)
    assert assays[0].hypers.mu_lib == 800.0 and assays[1].hypers.mu_lib == 6000.0
    assert assays[1].hypers.dispersion == 0.05
    # the third batch keeps the shared settings (preset + the shared dropout override)
    assert assays[2].hypers.mu_lib == scRNA(protocol="10x").hypers.mu_lib
    assert all(a.hypers.dropout_mid == 0.0 for a in assays)
    lib = [a.observed_counts.sum(axis=1).mean() for a in assays]
    assert lib[1] > 4 * lib[0]                       # depth follows each batch's own mu_lib


def test_batch_kwargs_do_not_move_cells_and_empty_overrides_change_nothing():
    cd = _cell_data()
    plain = run_scrna_batches(cd, n_batches=3, design="split", n_cells=100, base_seed=3)
    same = run_scrna_batches(cd, n_batches=3, design="split", n_cells=100, base_seed=3,
                             batch_kwargs=[{}, {}, {}])
    other = run_scrna_batches(cd, n_batches=3, design="split", n_cells=100, base_seed=3,
                              batch_kwargs=[dict(mu_lib=500.0), {}, dict(ambient_frac=0.3)])
    for a, b, c in zip(plain, same, other):
        assert list(a.observed_counts.index) == list(c.observed_counts.index)
        pd.testing.assert_frame_equal(a.observed_counts, b.observed_counts)
    # a batch without an override is drawn exactly as before (its own seed, the shared settings)
    pd.testing.assert_frame_equal(plain[1].observed_counts, other[1].observed_counts)


def test_batch_kwargs_are_checked():
    cd = _cell_data()
    with pytest.raises(ValueError, match="one mapping per batch"):
        run_scrna_batches(cd, n_batches=3, design="split", n_cells=50, batch_kwargs=[{}, {}])
    with pytest.raises(ValueError, match="not per-batch technical"):
        run_scrna_batches(cd, n_batches=2, design="split", n_cells=50,
                          batch_kwargs=[{"n_cells": 3}, {}])
    with pytest.raises(ValueError, match="not per-batch technical"):
        run_scrna_batches(cd, n_batches=2, design="split", n_cells=50,
                          batch_kwargs=[{}, {"seed": 1}])
    for alias in ("n_reads", "lib_size_sigma"):
        with pytest.raises(ValueError, match="alias"):
            run_scrna_batches(cd, n_batches=2, design="split", n_cells=50,
                              batch_kwargs=[{alias: 1000.0}, {}])


def test_concat_records_each_batchs_hyperparameters():
    cd = _cell_data()
    assays = run_scrna_batches(cd, n_batches=2, design="split", n_cells=60, base_seed=1,
                               batch_kwargs=[dict(mu_lib=900.0), dict(mu_lib=3000.0)])
    ad = concat_batches(assays)
    hp = ad.uns["batch_hyperparams"]
    assert hp["batch0"]["mu_lib"] == 900.0 and hp["batch1"]["mu_lib"] == 3000.0


# --- settings fitted to an atlas ------------------------------------------------------------------
def _atlas_table():
    from importlib import resources
    with resources.files("iscc.data").joinpath("reference", "wu2021_scrna_batches.tsv").open() as f:
        return pd.read_csv(f, sep="\t", comment="#", index_col="tumor")


def test_atlas_batches_take_one_fitted_tumour_each():
    table = _atlas_table()
    assert len(table) == 26                                  # every tumour of the atlas
    s = atlas_scrna_batches(40, seed=1)
    assert len(s) == 40 and s.index.name == "tumor" and s.index.isin(table.index).all()
    for col in ["mu_lib", "sigma_lib", "dispersion", "dropout_mid", "dropout_shape", "doublet_rate",
                "stress_strength"]:
        np.testing.assert_allclose(s[col].to_numpy(), table.loc[s.index, col].to_numpy())
    lo, hi = ATLAS_AMBIENT_RANGE
    assert s["ambient_frac"].between(lo, hi).all()
    assert set(s.columns) - {"stress_strength"} <= set(BATCH_OVERRIDE_KEYS)


def test_atlas_batches_follow_the_seed():
    pd.testing.assert_frame_equal(atlas_scrna_batches(6, seed=3), atlas_scrna_batches(6, seed=3))
    assert not atlas_scrna_batches(6, seed=3).equals(atlas_scrna_batches(6, seed=4))
    with pytest.raises(ValueError, match="at least 1"):
        atlas_scrna_batches(0)


def test_atlas_batches_run_as_batch_kwargs():
    cd = _cell_data()
    s = atlas_scrna_batches(3, seed=0)
    assays = run_scrna_batches(cd, n_batches=3, design="split", n_cells=100, base_seed=2,
                               batch_kwargs=s.drop(columns="stress_strength").to_dict("records"))
    for a, (_, row) in zip(assays, s.iterrows()):
        assert a.hypers.mu_lib == row["mu_lib"] and a.hypers.ambient_frac == row["ambient_frac"]


# --- the stress signature in two steps -------------------------------------------------------------
def test_choose_stress_genes_is_dissociations_own_draw():
    cd = _cell_data(n_genes=80)
    for seed in (1, 2, 3):
        d = Dissociation(cd, rng=np.random.default_rng(seed), stress_strength=0.4)
        _, meta, _ = d.run()
        rng = np.random.default_rng(seed)
        rng.random(len(cd["cell_type"]))              # the recovery draw comes first
        assert choose_stress_genes(cd["cell_exp"].columns, None, rng) == meta["stress_genes"]
        assert len(meta["stress_genes"]) == 4          # 5% of 80 genes
        assert meta["stress_genes_source"] == "random"


def test_apply_stress_scales_only_the_named_genes_and_rows():
    exp = _cell_data(n_cells=10, n_genes=6)["cell_exp"]
    out = apply_stress(exp, 0.5, ["g1", "g4"], cells=["C0", "C3"])
    np.testing.assert_allclose(out.loc[["C0", "C3"], ["g1", "g4"]], exp.loc[["C0", "C3"], ["g1", "g4"]] * 1.5)
    rest = out.drop(index=["C0", "C3"])
    pd.testing.assert_frame_equal(rest, exp.drop(index=["C0", "C3"]))
    pd.testing.assert_frame_equal(out[["g0", "g2", "g3", "g5"]], exp[["g0", "g2", "g3", "g5"]])
    pd.testing.assert_frame_equal(apply_stress(exp, 0.0, ["g1"]), exp)
    with pytest.raises(ValueError, match="not columns"):
        apply_stress(exp, 0.5, ["FOS"])


def test_given_stress_genes_replace_the_random_set_and_keep_the_recovery():
    cd = _cell_data()
    a, meta_a, _ = Dissociation(cd, rng=np.random.default_rng(9), stress_strength=0.3).run()
    b, meta_b, exp_b = Dissociation(cd, rng=np.random.default_rng(9), stress_strength=0.3,
                                    stress_genes=["g0", "g5"]).run()
    assert list(a) == list(b)
    assert meta_b["stress_genes"] == ["g0", "g5"] and meta_b["stress_genes_source"] == "given"
    np.testing.assert_allclose(exp_b.loc[b, ["g0", "g5"]], cd["cell_exp"].loc[b, ["g0", "g5"]] * 1.3)


# --- per-patient settings through Resection.dissociate ---------------------------------------------
@pytest.fixture(scope="module")
def tumour():
    t = GenotypeTumor(seed=2, genome_params=GENOME_PARAMS, selection_params=SELECTION_PARAMS,
                      cancer_cell_params={**CANCER_CELL_PARAMS, "death_rate": 0.05},
                      deme_params=DEME_PARAMS, spatial_params={"grid_size": 20, "structure_radius": 3},
                      update_mode="tau", tau=0.5, coarsen_passengers=True)
    t.grow(n_steps=30, seed=2)
    return t


def test_plain_dissociate_is_unchanged(tumour):
    spec = Resection(tumour)
    a = spec.dissociate(max_cells=300, layers="scrna")
    assert spec.dissociation_meta is None
    b = Resection(tumour).dissociate(max_cells=300, layers="scrna", recovery_probs=None,
                                     stress_strength=0.0)
    assert list(a["cell_type"].index) == list(b["cell_type"].index)


def test_dissociate_applies_recovery_and_stress(tumour):
    probs = {t: 0.5 for t in DEFAULT_RECOVERY}
    plain = Resection(tumour).dissociate(max_cells=400, layers="scrna", recovery_probs=probs, seed=11)
    spec = Resection(tumour)
    cd = spec.dissociate(max_cells=400, layers="scrna", recovery_probs=probs, stress_strength=0.8,
                         seed=11)
    meta = spec.dissociation_meta
    # enough cells are materialised that max_cells survive a recovery of one half, and max_cells are kept
    assert meta["n_materialised"] == meta["n_input"]
    assert abs(meta["n_input"] - 880) <= 0.02 * 880              # 1.1 x 400 / 0.5, rounded per deme
    assert 400 < meta["n_recovered"] < meta["n_input"]
    assert meta["recovery_probs"]["immune"] == 0.5 and meta["stress_strength"] == 0.8
    kept = cd["cell_type"].index
    assert len(kept) == 400 and list(kept) == list(plain["cell_type"].index)
    assert all(len(df) == len(kept) for df in cd.values())     # every table subset to the same cells
    g = meta["stress_genes"]
    assert g and set(g) <= set(cd["cell_exp"].columns)
    np.testing.assert_allclose(cd["cell_exp"].loc[kept, g], plain["cell_exp"].loc[kept, g] * 1.8, rtol=1e-6)
    other = [c for c in cd["cell_exp"].columns if c not in g]
    np.testing.assert_allclose(cd["cell_exp"].loc[kept, other], plain["cell_exp"].loc[kept, other])


def test_cancer_region_is_the_cancer_demes_and_a_rim(tumour):
    spec = Resection(tumour)
    core = spec.cancer_region(margin=0)
    rim = spec.cancer_region(margin=1)
    has_cancer = [i for i, d in enumerate(tumour.demes[:tumour.grid_size ** 2])
                  if any(tumour.genotypes[g].type == "cancer" for g in d)]
    assert core == has_cancer and set(core) < set(rim)
    G = tumour.grid_size
    for i in rim:                                             # every rim deme touches a cancer deme
        r, c = divmod(i, G)
        assert any(divmod(j, G)[0] in (r - 1, r, r + 1) and divmod(j, G)[1] in (c - 1, c, c + 1)
                   for j in core)
    cd = spec.dissociate(region=rim, max_cells=200, layers="scrna", recovery_probs="default", seed=0)
    assert set(cd["cell_deme"]["deme_id"]) <= set(rim)


def test_dissociate_returns_max_cells_under_the_default_recovery(tumour):
    spec = Resection(tumour)
    cd = spec.dissociate(max_cells=300, layers="scrna", recovery_probs="default", seed=3)
    meta = spec.dissociation_meta
    assert len(cd["cell_type"]) == 300 and meta["n_recovered"] >= 300
    assert meta["n_materialised"] > 300


def test_recovery_and_stress_genes_do_not_depend_on_the_strength(tumour):
    probs = {"immune": 0.3, "stromal": 0.5}
    runs = []
    for s in (0.0, 0.4, 1.5):
        spec = Resection(tumour)
        cd = spec.dissociate(max_cells=400, layers="scrna", recovery_probs=probs, stress_strength=s,
                             seed=5)
        runs.append((list(cd["cell_type"].index), spec.dissociation_meta))
    assert runs[0][0] == runs[1][0] == runs[2][0]
    assert runs[1][1]["stress_genes"] == runs[2][1]["stress_genes"]
    assert runs[0][1]["stress_genes"] == []                    # no stress, no genes drawn
    # stress alone recovers every cell
    spec = Resection(tumour)
    cd = spec.dissociate(max_cells=400, layers="scrna", stress_strength=0.2, seed=5)
    assert spec.dissociation_meta["n_recovered"] == spec.dissociation_meta["n_input"]


def test_dissociate_leaves_the_tumour_alone_unless_installed(tumour):
    before = getattr(tumour, "cell_data", None)
    Resection(tumour).dissociate(max_cells=200, layers="scrna", recovery_probs="default", seed=1)
    assert getattr(tumour, "cell_data", None) is before
    with pytest.raises(ValueError):
        Resection(tumour).dissociate(max_cells=200, layers="scrna", recovery_probs="bogus")
