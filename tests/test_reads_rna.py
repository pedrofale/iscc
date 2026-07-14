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

import gzip
import os

from iscc.data.reads import (
    distort_vaf, observed_allele_counts, emit_scrna_reads, ScReadSimAdapter, ScrnaAlleleCounts,
    SyntheticTranscriptome, write_scrna_fastq,
)
from iscc.tumor.models import GenotypeTumor


def _read_fastq(path):
    """Return the list of (id, seq) records from a gzipped FASTQ."""
    recs = []
    with gzip.open(path, "rt") as fh:
        lines = fh.read().splitlines()
    for i in range(0, len(lines), 4):
        recs.append((lines[i], lines[i + 1]))
    return recs


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
                      cancer_cell_params=C, deme_params={"carrying_capacity": None},  # well-mixed: cross-modality read checks need a grown population, not spatial capping
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


def test_observed_vaf_diverges_from_dna_at_low_expression(tmp_path):
    """cell_rna_vaf == DNA-VAF at neutral loci is the EXPECTED fraction; the OBSERVED scRNA-VAF is
    depth-sampled at the gene's expression, so it drops out at low expression and only matches
    DNA-VAF where the gene is well expressed."""
    t = _grow_tumor()
    cd = t.make_cell_data()
    res = emit_scrna_reads(cd, obs_fidelity=1.0, protocol="10x", seed=1, outdir=str(tmp_path))
    eff = np.concatenate(t.selection.mut_effects)
    neutral = eff == 1.0
    dna = res["dna_vaf"].values
    obs = res["obs_vaf"].values
    tot = res["total"].values
    site = (dna > 0) & neutral[None, :]
    u, d, o = tot[site], dna[site], obs[site]
    # unexpressed neutral sites: all drop out (observed VAF 0 != DNA-VAF > 0)
    unexpr = u == 0
    assert unexpr.sum() > 0 and np.all(o[unexpr] == 0.0)
    # well-expressed neutral sites: observed VAF tracks DNA-VAF
    welln = u >= 10
    if welln.sum() > 30:
        assert abs(o[welln].mean() - d[welln].mean()) < 0.1


def test_snv_scales_expression_depth_by_gene_type():
    """The SNV affects the expression DEPTH too (not just the VAF): an oncogene mutation raises
    expression (better detection), a TSG mutation lowers it (worse) — controlling for copy number."""
    t = _grow_tumor()
    cd = t.make_cell_data()
    exp, snv, cnv = cd["cell_exp"].values, cd["cell_snv"].values, cd["cell_cnv"].values
    eff = np.concatenate(t.selection.mut_effects)

    def ratio(cols):
        mut_e, wt_e = [], []
        for g in cols:
            cn2 = np.abs(cnv[:, g] - 2) < 0.01
            mut, wt = cn2 & (snv[:, g] > 0), cn2 & (snv[:, g] == 0)
            if mut.sum() > 10 and wt.sum() > 10:
                mut_e.append(exp[mut, g].mean()); wt_e.append(exp[wt, g].mean())
        return np.mean(mut_e) / np.mean(wt_e) if mut_e else np.nan

    assert ratio(np.where(eff > 1)[0]) > 1.0      # oncogene SNV -> more expression
    assert ratio(np.where(eff < 1)[0]) < 1.0      # TSG SNV -> less expression


# ----------------------------------------------------------------- actual reads (FASTQ) -------
class TestScrnaFastq:
    def _counts(self, n_cells=8, n_genes=4, umi=6, vaf=1.0):
        cells = [f"C{i}" for i in range(n_cells)]
        genes = [f"G_0_{g}" for g in range(n_genes)]
        alt = pd.DataFrame(np.full((n_cells, n_genes), umi), index=cells, columns=genes)
        ref = pd.DataFrame(np.zeros((n_cells, n_genes), dtype=int), index=cells, columns=genes)
        return alt, ref, genes

    def test_read_count_equals_total_umis(self, tmp_path):
        alt, ref, genes = self._counts(umi=6)            # 8*4*6 = 192 alt UMIs, 0 ref
        tx = SyntheticTranscriptome(genes, seed=0, read_length=60)
        fq = write_scrna_fastq(alt, ref, tx, str(tmp_path), error_rate=0.0, seed=1)
        assert fq["n_reads"] == int(alt.values.sum() + ref.values.sum())   # totals conserved
        assert len(_read_fastq(fq["R1"])) == fq["n_reads"]
        assert len(_read_fastq(fq["R2"])) == fq["n_reads"]

    def test_alt_reads_carry_alt_base(self, tmp_path):
        alt, ref, genes = self._counts(n_cells=4, n_genes=1, umi=10)
        tx = SyntheticTranscriptome(genes, seed=0, read_length=60)
        # all-alt vs all-ref: the variant base at var_pos differs
        fq_a = write_scrna_fastq(alt, ref, tx, str(tmp_path / "a"), error_rate=0.0, seed=1)
        fq_r = write_scrna_fastq(ref, alt, tx, str(tmp_path / "r"), error_rate=0.0, seed=1)
        vp = tx.var_pos
        alt_bases = {seq[vp] for _, seq in _read_fastq(fq_a["R2"])}
        ref_bases = {seq[vp] for _, seq in _read_fastq(fq_r["R2"])}
        assert alt_bases and ref_bases and alt_bases.isdisjoint(ref_bases)   # alt != ref base

    def test_barcode_per_cell_in_r1(self, tmp_path):
        alt, ref, genes = self._counts(n_cells=5, n_genes=2, umi=4)
        tx = SyntheticTranscriptome(genes, seed=0, read_length=40)
        fq = write_scrna_fastq(alt, ref, tx, str(tmp_path), error_rate=0.0, seed=2,
                               barcode_len=16, umi_len=12)
        # one distinct 16bp barcode per cell; R1 length == barcode+umi
        bcs = {seq[:16] for _, seq in _read_fastq(fq["R1"])}
        assert len(bcs) == 5
        assert all(len(seq) == 28 for _, seq in _read_fastq(fq["R1"]))

    def test_error_rate_is_single_source(self, tmp_path):
        """The FASTQ's proportion of reads not matching the true sequence == error_rate (the per-base
        sequencing error is the ONLY read-error source — not doubled by also folding it into the
        alt/ref split). Hom-ref molecular content -> non-ref fraction at the variant site ~ e."""
        n_cells, n_genes, umi, e = 40, 3, 100, 0.05
        cells = [f"C{i}" for i in range(n_cells)]
        genes = [f"G_0_{g}" for g in range(n_genes)]
        alt = pd.DataFrame(np.zeros((n_cells, n_genes), dtype=int), index=cells, columns=genes)
        ref = pd.DataFrame(np.full((n_cells, n_genes), umi), index=cells, columns=genes)
        tx = SyntheticTranscriptome(genes, seed=0, read_length=50)
        vp = tx.var_pos
        fq = write_scrna_fastq(alt, ref, tx, str(tmp_path), error_rate=e, seed=1)
        nonref = tot = 0
        for rid, seq in _read_fastq(fq["R2"]):
            g = rid.split(":")[1]
            nonref += int(seq[vp] != tx.seq[g][vp]); tot += 1
        assert abs(nonref / tot - e) < 0.015          # ~e, NOT ~2e

    def test_emit_scrna_reads_writes_fastq(self, tmp_path):
        t = _grow_tumor()
        cd = t.make_cell_data()
        res = emit_scrna_reads(cd, obs_fidelity=0.6, protocol="10x", seed=1,
                               emit_fastq=True, read_length=60, outdir=str(tmp_path))
        assert res["status"] == "emitted:synthetic-fastq"
        assert len(res["fastq"]) == 2 and all(os.path.exists(p) for p in res["fastq"])
        # read count == conserved UMI total (alt + ref over all cells/genes)
        assert res["n_reads"] == int(res["alt"].values.sum() + res["ref"].values.sum())
