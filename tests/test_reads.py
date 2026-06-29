"""F7 read-emission tests (DESIGN_features §C C2/C3).

CI stays light: DWGSIM / ART / bwa / samtools are OPTIONAL binaries, so the shell-out is
skipped gracefully when absent and the bespoke layers are asserted instead:
  * the shared `variants.inject` seam (total conserved exactly; alt fraction in expectation;
    error floor) — the reusable piece, tested hardest;
  * per-cell FASTA reflects CNAs (amplified repeated / deletion removed) and SNVs (correct
    base substituted at the right loci);
  * coverage allocation ∝ copy number (the C1 reuse);
  * the simulator adapters build the right command + wire FASTQ paths, skipping if absent;
  * the synthetic reference is deterministic in seed and the real-genome seam is present.
"""
import os

import numpy as np
import pandas as pd
import pytest

from iscc.data.reads import (
    inject, AlleleSplit, SyntheticReference, RealGenomeReference, build_cell_fasta,
    coverage_budget, emit_reads, DwgsimAdapter, ArtAdapter, find_binary, run_binary,
    MissingBinaryError,
)
from iscc.data.reads.dna import _segment_cn


N_SEG, SZ = 4, 25
N_GENES = N_SEG * SZ
GENES = [f"G_{s}_{p}" for s in range(N_SEG) for p in range(SZ)]


def make_cell_data(n_cells=30, amp_seg=1, amp_cn=8.0, del_seg=3, het_loci=(5, 30), seed=0):
    """cell_data with one amplified segment, one deleted segment, a few het loci."""
    cells = [f"C{i}" for i in range(n_cells)]
    cnv = np.full((n_cells, N_GENES), 2.0)
    cnv[:, amp_seg * SZ:(amp_seg + 1) * SZ] = amp_cn
    cnv[:, del_seg * SZ:(del_seg + 1) * SZ] = 0.0
    af = np.zeros((n_cells, N_GENES))
    for l in het_loci:
        af[:, l] = 0.5
    return {
        "cell_cnv": pd.DataFrame(cnv, index=cells, columns=GENES),
        "cell_snv": pd.DataFrame(af, index=cells, columns=GENES),
        "cell_type": pd.DataFrame(["cloneA"] * n_cells, index=cells, columns=["cell_id"]),
    }


# ---------------------------------------------------------------- variants.inject (seam) ----
class TestInject:
    def test_total_conserved_exactly_scalar(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            s = inject(137, 0.37, error_rate=0.01, rng=rng)
            assert s.alt + s.ref == 137
            assert s.total == 137
            assert 0 <= s.alt <= 137

    def test_total_conserved_exactly_array(self):
        rng = np.random.default_rng(1)
        totals = np.array([0, 1, 10, 100, 1000, 5000])
        s = inject(totals, 0.5, error_rate=0.001, rng=rng)
        assert np.array_equal(s.alt + s.ref, totals)
        assert s.alt.shape == totals.shape
        assert np.all(s.alt >= 0) and np.all(s.ref >= 0)

    def test_observed_fraction_matches_in_expectation(self):
        rng = np.random.default_rng(2)
        totals = np.full(2000, 500)
        s = inject(totals, 0.3, error_rate=0.0, rng=rng)
        obs = s.alt.sum() / s.total.sum()
        assert abs(obs - 0.3) < 0.01

    def test_error_floor_applied(self):
        """Even at true VAF 0, the per-base error floor emits some alt reads (and vice versa)."""
        rng = np.random.default_rng(3)
        totals = np.full(5000, 200)
        s0 = inject(totals, 0.0, error_rate=0.02, rng=rng)
        obs0 = s0.alt.sum() / s0.total.sum()
        assert abs(obs0 - 0.02) < 0.005          # floor ~ error_rate
        s1 = inject(totals, 1.0, error_rate=0.02, rng=rng)
        obs_ref = s1.ref.sum() / s1.total.sum()
        assert abs(obs_ref - 0.02) < 0.005       # symmetric on the ref side

    def test_zero_total(self):
        s = inject(0, 0.5, rng=np.random.default_rng(0))
        assert s.alt == 0 and s.ref == 0 and s.total == 0

    def test_negative_total_rejected(self):
        with pytest.raises(ValueError):
            inject(np.array([-1, 5]), 0.5, rng=np.random.default_rng(0))

    def test_generic_for_rna_reuse(self):
        """Same call shape works with UMI totals + observed RNA-VAF (RNA reuses it unchanged)."""
        rng = np.random.default_rng(7)
        umi = np.array([3, 8, 20, 0, 5])      # per cell-gene UMI counts from a count matrix
        v_obs = np.array([0.5, 0.0, 0.9, 0.5, 1.0])
        s = inject(umi, v_obs, error_rate=0.005, rng=rng)
        assert np.array_equal(s.alt + s.ref, umi)   # UMI totals conserved


# ------------------------------------------------------------- per-cell FASTA (CNA + SNV) ----
class TestCellFasta:
    def _records(self, tmp_path, amp_cn=4.0):
        cd = make_cell_data(amp_cn=amp_cn)
        ref = SyntheticReference(GENES, seed=1, locus_length=40)
        cnv = cd["cell_cnv"].loc["C0", GENES].values
        af = cd["cell_snv"].loc["C0", GENES].values
        p = os.path.join(tmp_path, "cell.fa")
        recs = build_cell_fasta(ref, cnv, af, p, name="C0")
        return ref, recs, p

    def test_amplification_repeats_segment(self, tmp_path):
        ref, recs, _ = self._records(tmp_path, amp_cn=6.0)
        amp_copies = [k for k in recs if k.startswith("C0_seg1_")]
        dip_copies = [k for k in recs if k.startswith("C0_seg0_")]
        assert len(amp_copies) == 6           # CN6 amplicon -> 6 sequence copies
        assert len(dip_copies) == 2           # diploid -> 2 copies
        # the repeated copies are the same base sequence (modulo SNVs)
        assert recs["C0_seg1_cp0"][:20] == recs["C0_seg1_cp1"][:20]

    def test_deletion_removes_segment(self, tmp_path):
        ref, recs, _ = self._records(tmp_path)
        assert not any(k.startswith("C0_seg3_") for k in recs)   # seg3 deleted (CN0)

    def test_snv_substitutes_correct_base(self, tmp_path):
        ref, recs, _ = self._records(tmp_path)
        # het locus 5 is in segment 0 (CN2): exactly one of the two copies carries the alt base.
        gl = 5
        pos = ref.locus_local_pos[gl]
        ref_base = ref.base_seq[0][pos]
        bases_at_site = [recs[f"C0_seg0_cp{cp}"][pos] for cp in range(2)]
        assert ref_base in bases_at_site                # one copy keeps the ref allele
        assert any(b != ref_base for b in bases_at_site)  # the other carries the substituted alt

    def test_fasta_written(self, tmp_path):
        _, _, p = self._records(tmp_path)
        assert os.path.exists(p)
        with open(p) as fh:
            head = fh.read(1)
        assert head == ">"

    def test_segment_cn_rounds(self):
        assert _segment_cn(np.array([2.0, 2.0, 2.0]), [0, 1, 2]) == 2
        assert _segment_cn(np.array([0.0, 0.0]), [0, 1]) == 0


# ----------------------------------------------------------- coverage allocation ∝ CN -------
class TestCoverage:
    def test_segment_budget_tracks_copy_number(self):
        cd = make_cell_data(amp_cn=8.0)
        per_seg, assay = coverage_budget(cd, breadth="wgs", modality="bulk", seed=2)
        # amplicon seg1 (CN8) gets ~4x the reads of a diploid seg (CN2); seg3 (CN0) gets ~0.
        ratio = per_seg[1] / max(per_seg[0], 1)
        assert 3.0 < ratio < 5.5
        assert per_seg[3] == 0

    def test_single_cell_budget(self):
        cd = make_cell_data(amp_cn=6.0)
        per_seg, assay = coverage_budget(cd, breadth="wgs", modality="sc", seed=2, n_cells=10)
        assert per_seg[1] > per_seg[0]       # amplicon still draws more reads
        assert per_seg[3] == 0


# ------------------------------------------------------------------- simulator adapters -----
class TestAdapters:
    def test_dwgsim_command(self):
        cmd = DwgsimAdapter().build_command("ref.fa", "out/sim", 30.0,
                                            read_length=150, error_rate=0.001)
        assert cmd[-2:] == ["ref.fa", "out/sim"]      # ref then output prefix, wired correctly
        assert "-C" in cmd and cmd[cmd.index("-C") + 1] == "30.0"
        assert "150" in cmd

    def test_dwgsim_error_rate_wired(self):
        # a non-default error_rate must reach DWGSIM's -e/-E (the read-level base-error floor).
        cmd = DwgsimAdapter().build_command("ref.fa", "out/sim", 30.0, error_rate=0.07)
        assert cmd[cmd.index("-e") + 1] == "0.07"
        assert cmd[cmd.index("-E") + 1] == "0.07"

    def test_art_command(self):
        cmd = ArtAdapter().build_command("ref.fa", "out/sim", 25.0, read_length=100)
        assert "-i" in cmd and cmd[cmd.index("-i") + 1] == "ref.fa"
        assert "-o" in cmd and cmd[cmd.index("-o") + 1] == "out/sim"
        assert "-f" in cmd and cmd[cmd.index("-f") + 1] == "25.0"

    def test_art_accepts_error_rate_and_ignores(self):
        # uniform adapter API: ART takes error_rate but draws errors from the -ss profile.
        cmd = ArtAdapter().build_command("ref.fa", "out/sim", 25.0, error_rate=0.05)
        assert "0.05" not in cmd and "-ss" in cmd


# ------------------------------------------------------ allele layer (ADO / overdispersion) --
class TestAlleleLayer:
    """The read path consumes the count model's allele hypers (ado_rate, beta_binom_conc,
    error_rate) so reads reproduce the scDNA artifacts `estimate_dna` is fit to."""

    @staticmethod
    def _batch(ado_rate=0.0, beta_binom_conc=30.0, error_rate=0.0, seed=0, n=2000):
        from iscc.data.batch import DNABatch, DNABatchHyperParams
        h = DNABatchHyperParams(ado_rate=ado_rate, beta_binom_conc=beta_binom_conc,
                                error_rate=error_rate)
        gc = np.full(n, 0.45); mapp = np.ones(n); ct = np.zeros(n, dtype=bool)
        return DNABatch(h, seed=seed).realize(gc, mapp, ct)

    def test_bulk_returns_true_fraction(self):
        from iscc.data.reads import effective_alt_fraction
        af = np.array([0.0, 0.5, 1.0, 0.5])
        out = effective_alt_fraction(af, batch=self._batch(ado_rate=0.9), modality="bulk")
        assert np.allclose(out, af)  # bulk ignores single-cell artifacts

    def test_sc_ado_collapses_hets_to_monoallelic(self):
        from iscc.data.reads import effective_alt_fraction
        af = np.full(2000, 0.5)
        out = effective_alt_fraction(af, batch=self._batch(ado_rate=0.8, beta_binom_conc=1e6,
                                                           n=2000), modality="sc")
        mono = np.mean((out < 1e-6) | (out > 1 - 1e-6))
        assert 0.7 < mono < 0.9                      # ~ado_rate of het loci drop to 0/1
        # no ADO + tight concentration -> hets stay ~0.5
        out0 = effective_alt_fraction(af, batch=self._batch(ado_rate=0.0, beta_binom_conc=1e6,
                                                            n=2000), modality="sc")
        assert np.mean(np.abs(out0 - 0.5) < 0.05) > 0.9

    def test_sc_overdispersion_lumpier_at_low_conc(self):
        from iscc.data.reads import effective_alt_fraction
        af = np.full(2000, 0.5)
        tight = effective_alt_fraction(af, batch=self._batch(beta_binom_conc=1000.0, n=2000),
                                       modality="sc")
        lumpy = effective_alt_fraction(af, batch=self._batch(beta_binom_conc=1.0, n=2000),
                                       modality="sc")
        assert np.std(lumpy) > np.std(tight)         # small conc = lumpier allele balance

    def test_emit_wires_error_rate_into_command(self, tmp_path):
        res = emit_reads(make_cell_data(amp_cn=6.0), modality="bulk", breadth="wgs",
                         outdir=str(tmp_path), seed=3)
        cmd = res["command"]
        assert "-e" in cmd and float(cmd[cmd.index("-e") + 1]) >= 0.0

    def test_sc_molecular_content_excludes_sequencing_error(self):
        """The per-base error is the simulator's job (DWGSIM -e), not the FASTA molecular content:
        the single-cell allele balance at a hom-ref/hom-alt locus stays ~0 / ~1 even at a high
        error_rate, so the error floor is NOT also baked into copy multiplicity (no double-count)."""
        from iscc.data.reads import effective_alt_fraction
        out0 = effective_alt_fraction(np.zeros(3000), modality="sc",
                                      batch=self._batch(beta_binom_conc=1e6, error_rate=0.05, n=3000))
        assert out0.mean() < 5e-3                  # hom-ref stays ~0, NOT ~0.05
        out1 = effective_alt_fraction(np.ones(3000), modality="sc",
                                      batch=self._batch(beta_binom_conc=1e6, error_rate=0.05, n=3000))
        assert out1.mean() > 1 - 5e-3              # hom-alt stays ~1, NOT ~0.95

    def test_emit_calls_binary_with_wired_paths(self, tmp_path, monkeypatch):
        """Monkeypatch the binary present + capture subprocess: assert command + FASTQ wiring."""
        import iscc.data.reads.base as base

        calls = {}
        monkeypatch.setattr(base, "find_binary", lambda name: "/usr/bin/" + name)

        def fake_run(cmd, **kw):
            calls["cmd"] = cmd
            class R: returncode = 0
            return R()
        monkeypatch.setattr(base.subprocess, "run", fake_run)

        adapter = DwgsimAdapter()
        fastqs = adapter.emit("cell.fa", "out/cell.sim", 30.0)
        assert calls["cmd"][0] == "/usr/bin/dwgsim"
        assert "cell.fa" in calls["cmd"] and "out/cell.sim" in calls["cmd"]
        # no real files on disk -> falls back to the expected dwgsim FASTQ names
        assert any("read1" in f for f in fastqs) and any("read2" in f for f in fastqs)

    def test_run_binary_missing_raises(self, monkeypatch):
        import iscc.data.reads.base as base
        monkeypatch.setattr(base, "find_binary", lambda name: None)
        with pytest.raises(MissingBinaryError):
            run_binary("dwgsim", ["ref.fa", "out"])


# ----------------------------------------------------------------------- references ---------
class TestReferences:
    def test_synthetic_deterministic_in_seed(self):
        a = SyntheticReference(GENES, seed=123)
        b = SyntheticReference(GENES, seed=123)
        c = SyntheticReference(GENES, seed=999)
        assert a.base_seq[0] == b.base_seq[0]            # same seed -> identical reference
        assert a.base_seq[0] != c.base_seq[0]            # different seed -> different
        assert set(a.seg_ids) == set(range(N_SEG))
        assert all(set(b.segments[s]) for s in b.seg_ids)

    def test_synthetic_locus_sites_inside_sequence(self):
        ref = SyntheticReference(GENES, seed=1, locus_length=30)
        for gl, pos in ref.locus_local_pos.items():
            seg = next(s for s in ref.seg_ids if gl in ref.segments[s])
            assert 0 <= pos < len(ref.base_seq[seg])

    def test_real_genome_seam_present(self, tmp_path):
        """The real-genome backend ingests a user FASTA via the same interface (drop-in seam)."""
        fa = os.path.join(tmp_path, "genome.fa")
        with open(fa, "w") as fh:
            for s in range(N_SEG):
                fh.write(f">chr{s}\n" + "ACGT" * 50 + "\n")
        ref = RealGenomeReference(fa, GENES)
        assert set(ref.seg_ids) == set(range(N_SEG))
        assert all(len(ref.base_seq[s]) == 200 for s in ref.seg_ids)
        # interface parity with the synthetic backend: every locus maps inside its contig.
        for gl, pos in ref.locus_local_pos.items():
            seg = next(s for s in ref.seg_ids if gl in ref.segments[s])
            assert 0 <= pos < len(ref.base_seq[seg])


# ------------------------------------------------------------------ end-to-end (guarded) ----
class TestEmitReads:
    def test_emit_skips_gracefully_without_binary(self, tmp_path):
        cd = make_cell_data(amp_cn=6.0)
        res = emit_reads(cd, modality="bulk", breadth="wgs", outdir=str(tmp_path), seed=3)
        # bespoke layers always produced:
        assert os.path.exists(res["fasta"][0])
        assert res["per_segment_coverage"][1] > res["per_segment_coverage"][0]
        assert res["command"][0] == "dwgsim" or res["command"][-1].endswith("sim")
        assert "allele_split" in res
        if find_binary("dwgsim") is None:
            assert res["status"].startswith("skipped")
            assert res["fastq"] == []
        else:                                            # binary present -> reads emitted
            assert res["status"].startswith("emitted")
            assert len(res["fastq"]) >= 1

    def test_emit_sc_per_cell_fasta(self, tmp_path):
        cd = make_cell_data(amp_cn=4.0)
        res = emit_reads(cd, modality="sc", breadth="wgs", outdir=str(tmp_path),
                         seed=3, n_cells=5)
        assert len(res["fasta"]) == 5                    # one FASTA per cell
        assert all(os.path.exists(p) for p in res["fasta"])

    def test_allele_split_conserves_segment_budget(self, tmp_path):
        cd = make_cell_data(amp_cn=6.0)
        res = emit_reads(cd, modality="bulk", outdir=str(tmp_path), seed=4)
        for seg, (alt, ref) in res["allele_split"].items():
            assert alt + ref == res["per_segment_coverage"][seg]
