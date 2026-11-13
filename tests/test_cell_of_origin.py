"""`expression_params["cell_of_origin"]`: cancer cells share the baseline of the normal type they came from.

What must hold:

1. **Off => unchanged.** Without the key, the cancer baseline is its own draw, as before.
2. **On => shared.** With it, the cancer baseline equals the origin type's, and every other type's
   baseline is untouched.
3. **Readout only.** Growth, the genotype registry and the layout gene sets are byte-identical with
   the key on or off at a given seed.
4. **It reaches the data.** Materialised cancer expression follows the origin's baseline.
"""
import numpy as np
import pytest
from scipy.stats import spearmanr

from iscc.tumor.models import GenotypeTumor

GENOME = {"n_segments": 6, "segment_size": 40}
SELECTION = {"prop_driver": 0.2}
DEME = {"carrying_capacity": 8, "initial_cancer_cells": 8}
SPATIAL = {"grid_size": 12, "structure_radius": 3}
CANCER = {"division_rate": 0.6, "death_rate": 0.03, "mutation_rate": 1.0, "dispersal_rate": 0.4}
EXPRESSION_PARAMS = {
    "program_params": {"n_programs": 6, "n_genes_per_program": 12, "program_overlap": 0.1,
                       "loading_strength": {"mean": 1.0, "sd": 0.3}, "loading_sparsity": 1.0,
                       "program_genomic_scatter": 1.0},
    "activity_params": {"n_active_programs_per_cell": 3, "activity_dist": "lognormal",
                        "activity_mean": 1.0, "activity_sd": 0.5, "activity_noise": 0.2},
    "dosage_params": {"dosage_sensitivity_mean": 0.7, "dosage_sensitivity_sd": 0.25,
                      "dosage_saturation": 8, "allele_specific": True},
}
MICROENV = {"hypoxia": {"n_genes": 10, "strength": 1.0}}


def build(seed=1, steps=0, **kw):
    t = GenotypeTumor(seed=seed, genome_params=GENOME, selection_params=SELECTION,
                      cancer_cell_params=CANCER, deme_params=DEME, spatial_params=SPATIAL, **kw)
    if steps:
        t.grow(n_steps=steps, seed=seed)
    return t


def cancer_mask(t):
    ct = t.cell_data["cell_type"]["cell_id"].values
    return np.array([t.genotypes[g].type == "cancer" for g in ct])


def with_origin(origin):
    return dict(EXPRESSION_PARAMS, cell_of_origin=origin)


def test_off_keeps_an_independent_cancer_baseline():
    t = build(expression_params=EXPRESSION_PARAMS)
    assert not np.array_equal(t.celltype_exps["cancer"], t.celltype_exps["epithelial"])


def test_on_shares_the_origin_baseline_and_leaves_the_others():
    off = build(expression_params=EXPRESSION_PARAMS)
    on = build(expression_params=with_origin("epithelial"))
    assert np.array_equal(on.celltype_exps["cancer"], on.celltype_exps["epithelial"])
    assert on.celltype_exps["cancer"] is not on.celltype_exps["epithelial"]
    for ct in ("epithelial", "stromal", "immune"):
        assert np.array_equal(on.celltype_exps[ct], off.celltype_exps[ct])


def test_growth_and_layout_are_identical_on_or_off():
    off = build(seed=4, steps=3000, expression_params=EXPRESSION_PARAMS, microenv_params=MICROENV)
    on = build(seed=4, steps=3000, expression_params=with_origin("epithelial"),
               microenv_params=MICROENV)
    assert off.get_tumor_size() == on.get_tumor_size()
    np.testing.assert_array_equal(off.cell_data["cell_snv"].values, on.cell_data["cell_snv"].values)
    np.testing.assert_array_equal(off.cell_data["cell_cnv"].values, on.cell_data["cell_cnv"].values)
    assert np.array_equal(off._hypoxia_genes, on._hypoxia_genes)
    assert np.array_equal(off.programs.dosage_sensitivity, on.programs.dosage_sensitivity)


def test_cancer_expression_follows_the_origin():
    def rho(t):
        exp = t.cell_data["cell_exp"].values
        mean_cancer = exp[cancer_mask(t)].mean(axis=0)
        return spearmanr(mean_cancer, t.celltype_exps["epithelial"]).statistic

    off = build(seed=5, steps=3000, expression_params=EXPRESSION_PARAMS)
    on = build(seed=5, steps=3000, expression_params=with_origin("epithelial"))
    assert rho(on) > 0.8
    assert rho(off) < 0.3


@pytest.mark.parametrize("origin", ["cancer", "neuron"])
def test_rejects_a_non_normal_origin(origin):
    with pytest.raises(ValueError, match="cell_of_origin"):
        build(expression_params=with_origin(origin))
