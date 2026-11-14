"""The per-gene dosage model (`expression_params["dosage_params"]`): its prior and its law.

What must hold:

1. **Default unchanged.** `s_g` is the clipped normal drawn as before, from the same layout
   sub-stream, and the law is the "haplotype" law.
2. **The "total" law** buffers losses as it buffers gains, leaves a copy-neutral loss of
   heterozygosity at the diploid level, and keeps the allelic split of the "haplotype" law.
3. **Readout only.** Growth is byte-identical whatever the prior or the law, since dosage acts at
   materialisation.
"""
import numpy as np
import pytest

from iscc.constants import LAYOUT_OFFSET_PROGRAMS
from iscc.tumor.models import GenotypeTumor
from iscc.tumor.programs import ProgramModel

N_GENES, SIZES, LAYOUT = 600, [100] * 6, 11
GENOME = {"n_segments": 6, "segment_size": 40}
SELECTION = {"prop_driver": 0.2}
DEME = {"carrying_capacity": 8, "initial_cancer_cells": 8}
SPATIAL = {"grid_size": 12, "structure_radius": 3}
CANCER = {"division_rate": 0.6, "death_rate": 0.03, "mutation_rate": 1.0, "dispersal_rate": 0.4}
DEFAULT = {"dosage_sensitivity_mean": 0.7, "dosage_sensitivity_sd": 0.25, "dosage_saturation": 8,
           "allele_specific": True}
TOTAL = dict(DEFAULT, dosage_law="total", dosage_sensitivity_mean=0.86, dosage_sensitivity_sd=0.61)


def model(dosage):
    return ProgramModel(N_GENES, SIZES, LAYOUT, expression_params={"dosage_params": dosage})


def test_default_draw_is_unchanged():
    _, _, sg_ss = np.random.SeedSequence(LAYOUT + LAYOUT_OFFSET_PROGRAMS).spawn(3)
    expected = np.clip(np.random.default_rng(sg_ss).normal(0.7, 0.25, size=N_GENES), 0.0, 1.0)
    np.testing.assert_array_equal(model(DEFAULT).dosage_sensitivity, expected)


def _cell_with(cn):
    """A founder copy whose segments carry the given (paternal, maternal) copy numbers."""
    t = GenotypeTumor(seed=1, genome_params=GENOME, selection_params=SELECTION,
                      cancer_cell_params=CANCER, deme_params=DEME, spatial_params=SPATIAL)
    cell = t.genotypes[t.founder_id].divide()
    genome = []
    for seg, (cp, cm) in enumerate(cn):
        allele = cell.genome[seg]["p"][0]
        genome.append({"p": [allele.copy() for _ in range(cp)], "m": [allele.copy() for _ in range(cm)]})
    cell.genome = genome
    cell.baseline_exp = np.ones(cell.n_genes)
    return cell


def test_total_law_buffers_losses_and_keeps_allelic_ratios():
    # segments: diploid, one-copy loss, one-copy gain, copy-neutral LOH, homozygous deletion, two gains
    cell = _cell_with([(1, 1), (1, 0), (2, 1), (2, 0), (0, 0), (2, 2)])
    s = np.full(cell.n_genes, 0.4)
    hp, hm = cell.get_exp_alleles(dosage_sensitivity=s, dosage_saturation=8)
    tp, tm = cell.get_exp_alleles(dosage_sensitivity=s, dosage_saturation=8, dosage_law="total")
    seg = np.repeat(np.arange(6), 40)
    def level(e):
        return np.array([e[seg == k].mean() for k in range(6)])
    np.testing.assert_allclose(level(hp + hm), [1.0, 0.5, 1.2, 0.7, 0.0, 1.4])
    np.testing.assert_allclose(level(tp + tm), [1.0, 0.8, 1.2, 1.0, 0.0, 1.4])
    with np.errstate(invalid="ignore"):
        np.testing.assert_allclose(level(tp / (tp + tm))[[0, 2, 5]], level(hp / (hp + hm))[[0, 2, 5]])
    # without a sensitivity model both laws are plain linear dosage
    a = cell.get_exp_alleles()
    b = cell.get_exp_alleles(dosage_law="total")
    np.testing.assert_array_equal(a[0], b[0]) and np.testing.assert_array_equal(a[1], b[1])
    with pytest.raises(ValueError):
        cell.get_exp_alleles(dosage_sensitivity=s, dosage_law="power")


def test_law_is_read_from_the_config():
    assert model(DEFAULT).dosage_law == "haplotype"
    assert model(TOTAL).dosage_law == "total"
    with pytest.raises(ValueError):
        model(dict(DEFAULT, dosage_law="power"))


def test_growth_is_identical_whatever_the_prior():
    def build(dosage):
        t = GenotypeTumor(seed=2, genome_params=GENOME, selection_params=SELECTION,
                          cancer_cell_params=CANCER, deme_params=DEME, spatial_params=SPATIAL,
                          expression_params={"dosage_params": dosage})
        t.grow(n_steps=40, seed=2)
        t.make_cell_data()
        return t
    a, b = build(DEFAULT), build(TOTAL)
    assert a.get_tumor_size() == b.get_tumor_size()
    np.testing.assert_array_equal(a.cell_data["cell_snv"].values, b.cell_data["cell_snv"].values)
    np.testing.assert_array_equal(a.cell_data["cell_cnv"].values, b.cell_data["cell_cnv"].values)
    assert not np.array_equal(a.programs.dosage_sensitivity, b.programs.dosage_sensitivity)


def test_law_reaches_the_materialised_expression():
    # same prior, different law: the s_g draws, the growth and the programs are identical, so the
    # expression frames differ only where the law does (copy losses), and only if make_cell_data's
    # chunked path passes the law on
    def build(dosage):
        t = GenotypeTumor(seed=2, genome_params=GENOME, selection_params=SELECTION,
                          cancer_cell_params=CANCER, deme_params=DEME, spatial_params=SPATIAL,
                          expression_params={"dosage_params": dosage})
        t.grow(n_steps=40, seed=2)
        t.make_cell_data(layers=["cell_cnv", "cell_exp"])
        return t
    a, b = build(DEFAULT), build(dict(DEFAULT, dosage_law="total"))
    np.testing.assert_array_equal(a.programs.dosage_sensitivity, b.programs.dosage_sensitivity)
    assert (a.cell_data["cell_cnv"].values < 2).any()           # the tumour carries a loss
    assert not np.array_equal(a.cell_data["cell_exp"].values, b.cell_data["cell_exp"].values)
