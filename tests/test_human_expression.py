"""Human-gene expression from atlas baselines and Hallmark programmes (L3), and the program bound.

What must hold:

1. **Program bound.** `program_params["max_log_fold"]` unset leaves the multiplier exactly
   `exp(Z·loading)` (and warns past a 10^6-fold multiplier); set, it never exceeds `e^c`, and the
   dictionary itself is unchanged.
2. **Atlas baselines.** Each type's baseline sums to 1 over the GRCh38-2020-A genes; cancer cells take
   their cell of origin's.
3. **Only genotype, niche and programmes separate cancer from epithelium.** Off the programme genes, a
   cancer cell's human-gene expression divided by the epithelial baseline is its dosage factor:
   1 on a balanced diploid arm, constant along an arm.
4. **Per-type pooling is exact.** A Visium spot's pooled expression equals the sum of its cells'
   expression computed one by one.
5. **The assays run on it.** scRNA counts come out sparse over the human genes; Visium spots too.
"""
import warnings

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from iscc.data import Visium, scRNA
from iscc.data.human import (HUMAN_VISIUM_KAPPA, HumanGenomeLayer, arm_genome, load_atlas_baselines,
                             load_hallmark_programs, load_reference)
from iscc.tumor.models import GenotypeTumor
from iscc.tumor.programs import ProgramModel

EXPR = {
    "program_params": {"n_programs": 6, "n_genes_per_program": 12, "program_overlap": 0.1,
                       "loading_strength": {"mean": 1.0, "sd": 0.3}, "loading_sparsity": 1.0,
                       "program_genomic_scatter": 1.0, "max_log_fold": 5.0},
    "activity_params": {"n_active_programs_per_cell": 3, "activity_dist": "lognormal",
                        "activity_mean": 1.0, "activity_sd": 0.5, "activity_noise": 0.2},
    "dosage_params": {"dosage_sensitivity_mean": 0.7, "dosage_sensitivity_sd": 0.25,
                      "dosage_saturation": 8, "allele_specific": True},
    "cell_of_origin": "epithelial",
}


def _model(**pp):
    params = {"n_programs": 4, "n_genes_per_program": 10, "loading_sparsity": 1.0, **pp}
    return ProgramModel(200, [100, 100], layout_seed=3, expression_params={"program_params": params})


def test_bound_is_opt_in_and_saturates():
    free, bounded = _model(), _model(max_log_fold=5.0)
    assert np.array_equal(free.dictionary.loading, bounded.dictionary.loading)
    Z = np.full((2, 4), 0.5)
    assert np.allclose(free.gene_multiplier(Z), np.exp(Z @ free.dictionary.loading))
    big = np.full((1, 4), 50.0)
    assert bounded.gene_multiplier(big).max() <= np.exp(5.0) + 1e-9
    with pytest.warns(UserWarning, match="max_log_fold"):
        free.gene_multiplier(big)
    with pytest.raises(ValueError):
        _model(max_log_fold=0)


@pytest.fixture(scope="module")
def reference():
    ref = load_reference(snps=False)
    rows = np.flatnonzero(ref.gene_arm() >= 0)
    g = ref.genes.iloc[rows]
    pos = np.where(g["strand"].to_numpy() == "+", g["end"].to_numpy() - 10, g["start"].to_numpy() + 10)
    ref.snps = pd.DataFrame({"chrom": g["chrom"].to_numpy(), "pos": pos, "ref": "A", "alt": "G",
                             "af": 0.5, "gene": rows})
    return ref


@pytest.fixture(scope="module")
def grown(reference):
    cfg = arm_genome(genes_per_segment=8, driver_scale=0.1, reference=reference)
    t = GenotypeTumor(seed=5, genome_params=cfg["genome_params"],
                      selection_params={"prop_driver": 0.0, **cfg["selection_params"]},
                      cancer_cell_params={"division_rate": 0.6, "death_rate": 0.03, "mutation_rate": 1.0,
                                          "dispersal_rate": 0.4, "cnv_prob": 0.6},
                      deme_params={"carrying_capacity": 10, "initial_cancer_cells": 10},
                      spatial_params={"grid_size": 12, "structure_radius": 3}, expression_params=EXPR)
    t.grow(n_steps=40, seed=5)
    layer = HumanGenomeLayer(t, reference=reference)
    return t, layer, layer.expression(t, chunk=64)


def test_packaged_tables():
    tab, info = load_atlas_baselines()
    assert len(tab) == 36601 and set(tab.columns) == {"epithelial", "stromal", "immune"}
    assert np.allclose(tab.sum(axis=0), 1.0, atol=1e-4)
    assert set(info["type"]) == set(tab.columns)
    hp = load_hallmark_programs()
    assert {"proliferation", "emt", "hypoxia"} <= set(hp["programme"])


def test_cancer_takes_its_origins_baseline(grown):
    _, _, expr = grown
    assert np.array_equal(expr.B["cancer"], expr.B["epithelial"])
    assert not np.array_equal(expr.B["stromal"], expr.B["epithelial"])
    assert {"proliferation", "emt", "hypoxia"} <= set(expr.programme_genes)


def test_cancer_differs_from_epithelium_only_by_its_genotype(grown):
    t, layer, expr = grown
    cd = t.cell_data
    types = expr._types(cd, list(cd["cell_exp"].index))
    cells = list(cd["cell_exp"].index[types == "cancer"][:30])
    E = expr.cell_expression(cd, cells)
    off = np.setdiff1d(np.arange(len(expr.genes)), expr.hp)
    off = off[expr.B["epithelial"][off] > 0]
    ratio = E[:, off] / expr.B["epithelial"][off][None, :]
    F, _, _ = expr._factors(cd, cells)
    assert np.allclose(ratio, F[:, layer.cover[off]])     # nothing but the genotype/niche factor
    gids = layer._gids(cd, cells)
    cn_p, cn_m = layer.homolog_cn(gids, t)
    seg = layer.segment[off]
    for i in range(len(cells)):
        bal = (cn_p[i, seg] == 1) & (cn_m[i, seg] == 1)
        if bal.any():
            assert np.allclose(ratio[i, bal], 1.0, rtol=0.05)   # balanced diploid arm, no niche here


def test_per_type_pooling_is_exact(grown):
    t, layer, expr = grown
    vz = Visium(seed=2, kappa=HUMAN_VISIUM_KAPPA)
    vz.place_grid(t.cell_data, grid_side=12)
    raw = expr.pool_spots(vz._grid)
    cells = np.asarray(vz._grid["placed"]["cell_exp"].index)
    for s, m in enumerate(vz._grid["members"]):
        if m.size:
            brute = expr.cell_expression(vz._grid["placed"], list(cells[m])).sum(axis=0)
            assert np.allclose(raw[s], brute, rtol=1e-9, atol=1e-12)
            break


def test_assays_run_on_human_expression(grown):
    t, layer, expr = grown
    sc = scRNA(n_cells=50, seed=1).run(t.cell_data, gene_lift=expr)
    assert sp.issparse(sc.counts_csr) and sc.counts_csr.shape == (50, len(expr.genes))
    assert sc.obs["n_counts"].min() > 0
    ac = layer.scrna_allele_counts(sc, t.cell_data, t)
    assert ac.A.shape[0] == 50
    vz = Visium(seed=2, kappa=HUMAN_VISIUM_KAPPA).run(t.cell_data, grid_side=12, gene_lift=expr)
    assert vz.spot_counts.shape[1] == len(expr.genes)
    av = layer.visium_allele_counts(vz, t)
    tissue = np.asarray(vz.obs["n_cells"]) > 0
    assert av.total[tissue].sum() > 0
