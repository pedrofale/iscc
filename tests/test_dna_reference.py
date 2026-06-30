"""Tests for the real-DNA reference reducers + cache (validation/data/build_dna_reference.py).

CI-light and **network-free**: the pure reducers (`parse_giab_record`, `reduce_dlp_records`,
`reduce_tapestri`) are exercised on tiny in-memory fixtures (the shape of each real source), the
fetchers' graceful fallback is checked by monkeypatching the stream to fail, and `estimate_dna` is
run on the reduced fixtures to confirm sane hypers with **honest `.fitted` flags** per regime —
bulk: ado/beta_binom NOT fit; single-cell with het reads: they ARE fit. If the real `.npz` caches
have been built (committed under validation/data/), they are additionally loaded + fit; otherwise
that check skips so CI passes offline.
"""
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "validation", "data"))
import build_dna_reference as B  # noqa: E402

from iscc.data import estimate_dna  # noqa: E402


# --------------------------------------------------------------------------------------
# Pure reducers on tiny fixtures
# --------------------------------------------------------------------------------------
class TestGiabReducer:
    def _line(self, ref, alt, gt, ad, dp=40, filt="PASS"):
        return f"chr1\t100\t.\t{ref}\t{alt}\t50\t{filt}\tinfo\tGT:DP:AD\t{gt}:{dp}:{ad}".split("\t")

    def test_het_snv_parsed(self):
        cov, alt, is_het = B.parse_giab_record(self._line("A", "G", "0/1", "18,22"))
        assert cov == 40 and alt == 22 and is_het is True

    def test_hom_alt_is_not_het(self):
        cov, alt, is_het = B.parse_giab_record(self._line("A", "G", "1/1", "1,39"))
        assert cov == 40 and alt == 39 and is_het is False

    def test_phased_het_recognized(self):
        assert B.parse_giab_record(self._line("C", "T", "0|1", "20,20"))[2] is True

    def test_indel_and_nonpass_and_missing_ad_dropped(self):
        assert B.parse_giab_record(self._line("AT", "A", "0/1", "10,10")) is None      # indel
        assert B.parse_giab_record(self._line("A", "G", "0/1", "10,10", filt="fail")) is None
        bad = "chr1\t100\t.\tA\tG\t50\tPASS\tinfo\tGT:DP\t0/1:40".split("\t")           # no AD
        assert B.parse_giab_record(bad) is None


class TestDlpReducer:
    def test_matrix_shapes_and_values(self):
        from collections import OrderedDict
        rows = OrderedDict()
        # two cells, three 500 kb bins on chr1; bin2 is a deletion (state 1), bin3 amplified (state 4)
        for cell in ("cA", "cB"):
            rows[cell] = [(("1", 1), 100, 2), (("1", 500001), 40, 1), (("1", 1000001), 200, 4)]
        cov, cn = B.reduce_dlp_records(rows)
        assert cov.shape == (2, 3) and cn.shape == (2, 3)
        assert cov[0].tolist() == [100, 40, 200]
        assert cn[0].tolist() == [2, 1, 4]

    def test_empty_input(self):
        cov, cn = B.reduce_dlp_records({})
        assert cov.size == 0 and cn.size == 0


class TestTapestriReducer:
    def test_reduction_shapes_alt_and_het(self):
        # 4 cells x 3 variants; AF is a 0–100 percentage -> alt = round(AF/100 * DP)
        dp = np.array([[40, 50, 60], [30, 0, 80], [40, 40, 40], [20, 20, 20]])
        af = np.array([[50, 0, 100], [48, 0, 95], [50, 50, 0], [52, 50, 50]], float)
        ngt = np.array([[1, 0, 2], [1, 3, 2], [1, 1, 0], [1, 1, 1]])     # 3 == missing
        var_amplicon = np.array([b"AMP1", b"AMP1", b"AMP2"])
        amplicon_ids = np.array([b"AMP1", b"AMP2"])
        amplicon_reads = np.array([[100, 100], [100, 100], [200, 100], [50, 100]], float)
        cov, alt, cn, het = B.reduce_tapestri(dp, af, ngt, var_amplicon, amplicon_reads, amplicon_ids)
        assert cov.shape == alt.shape == cn.shape == het.shape == (4, 3)
        assert alt[0, 0] == 20 and alt[0, 2] == 60                       # round(50/100*40), 100/100*60
        assert het.tolist()[0] == [True, False, False]                  # NGT==1 only
        # missing genotype (cell1, var1) -> zero coverage so it drops out of fits
        assert cov[1, 1] == 0 and alt[1, 1] == 0
        assert np.all(cn >= 0) and np.all(cn <= 8)


# --------------------------------------------------------------------------------------
# Cache round-trip + fetcher graceful fallback
# --------------------------------------------------------------------------------------
def test_save_load_roundtrip(tmp_path):
    ref = dict(modality="sc", breadth="panel", depth_model="dm",
               coverage=np.arange(6).reshape(2, 3).astype(np.int32),
               alt=np.zeros((2, 3), np.int32), cn=np.full((2, 3), 2.0, np.float32),
               het_mask=np.ones((2, 3), bool), variant_mask=None,
               source="unit-test source")
    path = os.path.join(tmp_path, "ref.npz")
    B.save_reference(path, ref)
    out = B.load_reference(path)
    assert out["modality"] == "sc" and out["breadth"] == "panel"
    assert out["source"] == "unit-test source"
    assert out["variant_mask"] is None
    np.testing.assert_array_equal(out["coverage"], ref["coverage"])


def test_load_missing_returns_none():
    assert B.load_reference("/no/such/reference.npz") is None


def test_fetch_falls_back_to_none_on_stream_failure(monkeypatch):
    def boom(*a, **k):
        raise OSError("offline")
    monkeypatch.setattr(B, "_stream_gzip_lines", boom)
    assert B.fetch_hg002_bulk() is None
    assert B.fetch_dlp_sc() is None


def test_fetch_hg002_reduces_monkeypatched_stream(monkeypatch):
    """Feed synthetic VCF lines through the real fetch path (no network) -> well-formed bulk ref."""
    rng = np.random.default_rng(0)
    lines = ["##header", "#CHROM\tPOS"]
    for _ in range(400):
        dp = int(rng.integers(25, 45))
        a = int(rng.binomial(dp, 0.5))                                   # het ~0.5
        lines.append(f"chr1\t1\t.\tA\tG\t50\tPASS\ti\tGT:DP:AD\t0/1:{dp}:{dp - a},{a}")
    monkeypatch.setattr(B, "_stream_gzip_lines", lambda *a, **k: iter(lines))
    ref = B.fetch_hg002_bulk(max_sites=400)
    assert ref["modality"] == "bulk" and ref["breadth"] == "wgs"
    assert ref["coverage"].shape == (400,)
    assert np.all(ref["cn"] == 2.0) and ref["het_mask"].all()


# --------------------------------------------------------------------------------------
# estimate_dna on the reduced fixtures: sane hypers + honest per-regime flags
# --------------------------------------------------------------------------------------
def _bulk_fixture(n=2000, seed=1):
    rng = np.random.default_rng(seed)
    cov = rng.poisson(60, n).astype(int) + 1
    alt = rng.binomial(cov, 0.5)                                          # all germline hets
    return dict(coverage=cov, alt=alt, cn=np.full(n, 2.0),
                het_mask=np.ones(n, bool), variant_mask=np.ones(n, bool))


def _sc_fixture(cells=80, loci=120, seed=2):
    rng = np.random.default_rng(seed)
    cn = np.where(rng.random((cells, loci)) < 0.3, 4.0, 2.0)
    cov = rng.poisson(40 * cn / 2.0).astype(int) + 1
    het = rng.random((cells, loci)) < 0.4
    af = np.where(het, 0.5, 0.0)
    alt = rng.binomial(cov, af)
    return dict(coverage=cov, alt=alt, cn=cn, het_mask=het)


def test_bulk_fit_flags_honest():
    fx = _bulk_fixture()
    est = estimate_dna(fx["coverage"], fx["alt"], fx["cn"], modality="bulk", breadth="wgs",
                       variant_mask=fx["variant_mask"])
    assert "mu_depth" in est.fitted
    assert est.hypers.mu_depth == pytest.approx(float(fx["coverage"].mean()), rel=1e-6)
    # single-cell-only fields are NOT fit on bulk; doublet is always prior-only
    assert "ado_rate" not in est.fitted and "beta_binom_conc" not in est.fitted
    assert "doublet_rate" not in est.fitted


def test_sc_fit_flags_honest_with_hets():
    fx = _sc_fixture()
    est = estimate_dna(fx["coverage"], fx["alt"], fx["cn"], modality="sc", breadth="wgs",
                       het_mask=fx["het_mask"])
    assert "mu_depth" in est.fitted and "kappa" in est.fitted
    # het reads present -> the per-cell amplification/dropout layer IS fit
    assert "ado_rate" in est.fitted
    assert 0.0 <= est.hypers.ado_rate <= 1.0


def test_sc_without_hets_does_not_fit_ado():
    fx = _sc_fixture()
    est = estimate_dna(fx["coverage"], np.zeros_like(fx["alt"]), fx["cn"], modality="sc",
                       breadth="wgs", het_mask=None)
    assert "ado_rate" not in est.fitted and "beta_binom_conc" not in est.fitted


# --------------------------------------------------------------------------------------
# Real caches, when built (committed under validation/data/) — else skipped (offline CI)
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["hg002", "dlp", "tapestri"])
def test_real_cache_loads_and_fits_if_present(name):
    path = B.reference_path(name)
    if not os.path.exists(path):
        pytest.skip(f"{name} cache not built (run build_dna_reference.py)")
    ref = B.load_reference(path)
    est = estimate_dna(ref["coverage"], ref["alt"], ref["cn"], modality=ref["modality"],
                       breadth=ref["breadth"], depth_model=ref["depth_model"],
                       het_mask=ref["het_mask"], variant_mask=ref["variant_mask"])
    assert "mu_depth" in est.fitted and est.hypers.mu_depth > 0
    if ref["modality"] == "bulk":
        assert "ado_rate" not in est.fitted
    if name == "tapestri":                                # rich single-cell panel: ADO IS fit
        assert "ado_rate" in est.fitted and "beta_binom_conc" in est.fitted
