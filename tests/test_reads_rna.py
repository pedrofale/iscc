"""F7b mutation-aware scRNA read/count tests (DESIGN_features §C).

CI-light (scReadSim is an optional binary): the count-level layer is asserted directly, and the
read shell-out is skipped gracefully. The load-bearing invariants:
  * UMI totals conserved (alt + ref == expression count), so reads never invent coverage;
  * the single `obs_fidelity` knob distorts the observed VAF (under-detection);
  * an unexpressed gene is uncallable (0 UMIs -> 0 alt);
  * the engine `cell_rna_vaf` equals DNA-VAF at neutral loci and diverges at drivers.
"""
import numpy as np
import pandas as pd
import pytest

from iscc.data.reads import (
    distort_vaf, observed_allele_counts, emit_scrna_reads, ScReadSimAdapter, ScrnaAlleleCounts,
)
from iscc.tumor.models import GenotypeTumor


# ------------------------------------------------------------------- count-level core --------
class TestObservedAlleleCounts:
    def _frames(self, n_cells=40, n_genes=12, umi=50, vaf=0.5, seed=0):
        cells = [f"C{i}" for i in range(n_cells)]
        genes = [f"G_0_{g}" for g in range(n_genes)]
        expr = pd.DataFrame(np.full((n_cells, n_genes), umi), index=cells, columns=genes)
        rvaf = pd.DataFrame(np.full((n_cells, n_genes), vaf), index=cells, columns=genes)
        return expr, rvaf

    def test_umi_totals_conserved_exactly(self):
        expr, rvaf = self._frames()
        out = observed_allele_counts(expr, rvaf, obs_fidelity=0.6,
                                     rng=np.random.default_rng(1))
        assert np.array_equal((out.alt.values + out.ref.values), expr.values)  # alt+ref == total
        assert np.array_equal(out.total.values, expr.values)

    def test_fidelity_one_tracks_true_vaf(self):
        expr, rvaf = self._frames(umi=200, vaf=0.5)
        out = observed_allele_counts(expr, rvaf, obs_fidelity=1.0, error_rate=0.0,
                                     rng=np.random.default_rng(2))
        assert abs(out.obs_vaf.values.mean() - 0.5) < 0.03         # faithful

    def test_low_fidelity_underdetects(self):
        expr, rvaf = self._frames(umi=200, vaf=0.5)
        out = observed_allele_counts(expr, rvaf, obs_fidelity=0.3, error_rate=0.0,
                                     rng=np.random.default_rng(3))
        assert abs(out.obs_vaf.values.mean() - 0.15) < 0.03        # v_true * fidelity = 0.5*0.3

    def test_unexpressed_gene_uncallable(self):
        expr, rvaf = self._frames(umi=50, vaf=0.5)
        expr.iloc[:, 0] = 0                                        # gene 0 not expressed
        out = observed_allele_counts(expr, rvaf, obs_fidelity=1.0, error_rate=0.01,
                                     rng=np.random.default_rng(4))
        assert out.alt.values[:, 0].sum() == 0 and out.ref.values[:, 0].sum() == 0
        assert np.all(out.obs_vaf.values[:, 0] == 0.0)            # uncallable

    def test_distort_vaf_monotone_and_clipped(self):
        v = np.array([0.0, 0.5, 1.0])
        assert np.allclose(distort_vaf(v, 1.0), v)
        assert np.allclose(distort_vaf(v, 0.5), [0.0, 0.25, 0.5])
        assert np.all(distort_vaf(v, 5.0) <= 1.0)                  # clipped


# ------------------------------------------------------------------- scReadSim adapter --------
def test_screadsim_command_built():
    cmd = ScReadSimAdapter().build_command("counts.csv", "ref.fa", "tmpl.bam", "out/sim")
    assert cmd[cmd.index("--count") + 1] == "counts.csv"
    assert cmd[cmd.index("--ref") + 1] == "ref.fa"
    assert cmd[cmd.index("--bam") + 1] == "tmpl.bam"


# ----------------------------------------------------------------- engine integration ---------
def _grow_tumor(seed=1, steps=500):
    GENOME = {"n_segments": 6, "segment_size": 40}
    SEL = {"prop_driver": 0.3, "prop_dispersal": 0.0, "prop_immune_resistance": 0.0,
           "prop_treatment_resistance": 0.0, "driver_effects": 1.3, "dispersal_effects": 1.0,
           "treatment_resistant_effects": 1.0, "immune_resistant_effects": 1.0}
    C = {"division_rate": 0.4, "death_rate": 0.02, "max_birth_rate": 0.95,
         "mutation_rate": 0.6, "dispersal_rate": 0.2, "snv_prob": 0.7, "cnv_prob": 0.3}
    t = GenotypeTumor(seed=seed, genome_params=GENOME, selection_params=SEL,
                      cancer_cell_params=C, deme_params={"carrying_capacity": 6},
                      spatial_params={"grid_size": 14, "structure_radius": 0})
    t.grow(steps, seed=seed)
    return t


def test_cell_rna_vaf_matches_dna_at_neutral_diverges_at_drivers():
    t = _grow_tumor()
    cd = t.make_cell_data()
    assert "cell_rna_vaf" in cd
    dna = cd["cell_snv"].values
    rna = cd["cell_rna_vaf"].values
    eff = np.concatenate(t.selection.mut_effects)
    neutral = eff == 1.0
    # neutral loci: RNA-VAF == DNA-VAF exactly
    assert np.allclose(rna[:, neutral], dna[:, neutral])
    # oncogene loci (e>1): expressed mutant fraction inflated where there is signal
    onc = eff > 1.0
    m = dna[:, onc] > 0
    if m.any():
        assert rna[:, onc][m].mean() > dna[:, onc][m].mean()


def test_emit_scrna_reads_counts_only_path(tmp_path):
    t = _grow_tumor()
    cd = t.make_cell_data()
    res = emit_scrna_reads(cd, obs_fidelity=0.5, protocol="10x", seed=3,
                           outdir=str(tmp_path))
    # UMI totals conserved against the F3 count matrix; no binary/reference -> counts-only.
    assert np.array_equal(res["alt"].values + res["ref"].values, res["total"].values)
    assert res["status"].startswith("counts-only")
    assert res["dna_vaf"].shape == res["obs_vaf"].shape


def test_emit_scrna_requires_rna_vaf():
    t = _grow_tumor()
    cd = t.make_cell_data()
    del cd["cell_rna_vaf"]
    with pytest.raises(KeyError):
        emit_scrna_reads(cd, outdir="/tmp/iscc_should_not_exist")
