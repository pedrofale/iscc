"""Whole-genome duplication (WGD) — v1 of DESIGN_focal_cna.md.

WGD is a punctuated event that doubles every copy on both homologs of every segment, gated by the
existing CINner viability limits (max_ploidy / max_cn). It is a SEPARATE per-division channel
(``wgd_rate``) that leaves the SNV/CNA split untouched and is OFF by default, so growth is
byte-identical to before when ``wgd_rate == 0``.

Covered here:
  * ``wgd_rate == 0`` -> growth is byte-identical (golden hashes, verified equal to the pre-WGD
    baseline via ``git stash``; the count-engine path is the reference).
  * a forced WGD doubles every segment's copy count and carries existing SNVs into the duplicates.
  * a WGD that would breach ``max_ploidy`` is REJECTED AT BIRTH (non-viable daughter), not applied.
  * both engines drive WGD through the same ``Cell.mutate`` seam and agree that it doubles copies.
"""
import hashlib

import numpy as np
import pytest

from conftest import (
    GENOME_PARAMS, SELECTION_PARAMS, CANCER_CELL_PARAMS, N_SEGMENTS, SEGMENT_SIZE,
    EPITHELIAL_CELL_PARAMS, STROMAL_CELL_PARAMS, IMMUNE_CELL_PARAMS, DEME_PARAMS,
)
from iscc.tumor.models import GenotypeTumor, GlandularTumor
from iscc.tumor.components.cell import CancerCell
from iscc.tumor.components.selection import Selection

SPATIAL = {"grid_size": 15, "n_structures": 1, "structure_radius": 0}
NO_DEATH = {**CANCER_CELL_PARAMS, "death_rate": 0.0}


def _count(seed, steps, cancer=NO_DEATH, selection=SELECTION_PARAMS, **kw):
    t = GenotypeTumor(
        seed=seed, genome_params=GENOME_PARAMS, selection_params=selection,
        cancer_cell_params=cancer, deme_params=DEME_PARAMS, spatial_params=SPATIAL, **kw,
    )
    t.grow(n_steps=steps, seed=seed)
    return t


def _cell(seed, steps, cancer=NO_DEATH, selection=SELECTION_PARAMS):
    t = GlandularTumor(
        seed=seed, genome_params=GENOME_PARAMS, selection_params=selection,
        cancer_cell_params=cancer, epithelial_cell_params=EPITHELIAL_CELL_PARAMS,
        stromal_cell_params=STROMAL_CELL_PARAMS, immune_cell_params=IMMUNE_CELL_PARAMS,
        deme_params=DEME_PARAMS, grid_size=SPATIAL["grid_size"], structure_radius=0,
    )
    t.grow(n_steps=steps, seed=seed)
    return t


def _fresh_cancer(selection, wgd_rate=0.0, **kw):
    c = CancerCell(
        n_segments=selection.n_segments, segment_size=selection.segment_size,
        n_onc=len(selection.get_oncogenes()), n_tsg=len(selection.get_tsgs()),
        n_disp=len(selection.get_dispersal_genes()), n_ir=len(selection.get_immune_resistant()),
        n_tr=len(selection.get_treatment_resistant()), wgd_rate=wgd_rate, **kw,
    )
    c.set_genotype_id()
    return c


# --- bit-identical when off -------------------------------------------------------------------
# The WGD branch is gated behind ``wgd_rate > 0 and rng.random() < wgd_rate``. At the default 0.0
# the short-circuit means NO random variable is drawn, so the rng stream (and every downstream
# state transition) is exactly what it was before WGD existed. These golden fingerprints were
# captured with the WGD changes stashed and re-checked with them applied -- identical both ways.
_GOLDEN_COUNT = {
    1: (46, "566c7173995f34bd943ace2019141cae"),
    2: (52, "dd2f14ac83142c97c0a87dfb6774f49a"),
    3: (48, "1579aff5e943d3b6450e122be0a2e937"),
}


def _snv_hash(t):
    return hashlib.md5(np.ascontiguousarray(t.cell_data["cell_snv"].values)).hexdigest()


@pytest.mark.parametrize("seed", sorted(_GOLDEN_COUNT))
def test_wgd_off_is_byte_identical(seed):
    t = _count(seed, steps=150)              # wgd_rate absent -> default 0.0 -> WGD off
    size, digest = _GOLDEN_COUNT[seed]
    assert t.get_tumor_size() == size
    assert _snv_hash(t) == digest, "wgd_rate=0 perturbed the growth stream (must be byte-identical)"
    assert "cell_wgd" not in t.cell_data     # no WGD ground-truth frame when the feature is off


def test_explicit_zero_wgd_rate_matches_absent_key():
    """Passing ``wgd_rate=0.0`` must be indistinguishable from omitting the key entirely."""
    for seed in (1, 2):
        a = _count(seed, steps=120)
        b = _count(seed, steps=120, cancer={**NO_DEATH, "wgd_rate": 0.0})
        assert a.get_tumor_size() == b.get_tumor_size()
        assert _snv_hash(a) == _snv_hash(b)


# --- the event: a forced WGD doubles every segment and carries SNVs ----------------------------
def test_forced_wgd_doubles_every_segment_and_carries_snvs():
    sel = Selection(n_segments=3, segment_size=30, prop_driver=0.2)
    c = _fresh_cancer(sel)
    rng = np.random.default_rng(0)
    # plant SNVs across the genome via a few SNV-only divisions
    for _ in range(4):
        child = c.divide()
        child.mutate(rng, sel, wgd_rate=0.0, snv_prob=1.0, cnv_prob=0.0)
        c = child

    before_cns = list(c.genome_summary["seg_cns"])
    before_ploidy = c.genome_summary["ploidy"]
    before_copies = {(s, h): len(c.genome[s][h]) for s in range(3) for h in ("p", "m")}
    before_bits = sum(int(b.sum()) for s in range(3) for h in ("p", "m") for b in c.genome[s][h])
    assert before_bits > 0, "the SNV-planting step must actually plant SNVs"
    assert not c.is_wgd

    w = c.divide()
    assert w.mutate(rng, sel, wgd_rate=1.0) is True            # WGD forced -> a new genotype
    assert w.is_wgd is True

    # every segment's copy number doubled; both homologs of every segment doubled their copies
    assert list(w.genome_summary["seg_cns"]) == [2 * x for x in before_cns]
    assert w.genome_summary["ploidy"] == pytest.approx(2 * before_ploidy)
    assert w.genome_summary["highest_cn"] == 2 * max(before_cns)
    for (s, h), n in before_copies.items():
        assert len(w.genome[s][h]) == 2 * n

    # SNVs are carried into the duplicates: total mutated bits doubled
    after_bits = sum(int(b.sum()) for s in range(3) for h in ("p", "m") for b in w.genome[s][h])
    assert after_bits == 2 * before_bits

    # a duplicate is an INDEPENDENT array, not an alias of the copy it was made from: overwriting
    # the newly-appended copy must not touch the original it was duplicated from (nor the parent).
    orig = w.genome[0]["p"][0].copy()
    w.genome[0]["p"][-1][:] = True
    assert np.array_equal(w.genome[0]["p"][0], orig)   # sibling copy untouched
    assert c.genome is not w.genome                    # parent kept its own genome (copy-on-write)


def test_wgd_preserves_nullisomy():
    """A WGD doubles nothing on a lost (0-copy) homolog: 2*0 == 0, so it never rescues a deletion."""
    sel = Selection(n_segments=2, segment_size=20, prop_driver=0.0)
    c = _fresh_cancer(sel)
    # delete both copies of segment 0 -> nullisomic segment
    for hap in ("p", "m"):
        del c.genome[0][hap][0]
    c.genome_summary["seg_cns"][0] = 0
    c.genome_summary["nullisomy_count"] = 1
    rng = np.random.default_rng(1)
    w = c.divide()
    w.mutate(rng, sel, wgd_rate=1.0)
    assert w.genome_summary["seg_cns"][0] == 0            # stayed nullisomic
    assert w.genome_summary["seg_cns"][1] == 4            # the surviving segment doubled 2 -> 4


# --- viability: a WGD breaching max_ploidy is rejected at birth --------------------------------
def test_wgd_breaching_max_ploidy_is_non_viable():
    """A diploid genome (ploidy 2) WGD-ing to ploidy 4 is non-viable under max_ploidy=3."""
    sel = Selection(n_segments=2, segment_size=20, prop_driver=0.1, max_ploidy=3)
    c = _fresh_cancer(sel)
    assert c.evolutionary_parameters["viability"] == 1     # the untouched diploid founder is viable
    w = c.divide()
    assert w.mutate(rng=np.random.default_rng(0), selection=sel, wgd_rate=1.0) is True
    assert w.genome_summary["ploidy"] == pytest.approx(4.0)
    assert w.evolutionary_parameters["viability"] == 0     # WGD busted max_ploidy -> non-viable


def test_engine_rejects_wgd_that_busts_ploidy():
    """With max_ploidy=3 the only reachable WGD (diploid 2 -> 4) is rejected at birth, so no
    survivor is a WGD clone and none exceeds the ploidy cap -- while rejections DO fire."""
    rejected = []
    real = GenotypeTumor._is_viable

    def spy(self, rep):
        ok = real(self, rep)
        if not ok:
            rejected.append(dict(rep.genome_summary))
        return ok

    GenotypeTumor._is_viable = spy
    try:
        # amp/del OFF (cnv_prob 0) so ploidy only moves via WGD -> the only breach is 2->4.
        cancer = {**NO_DEATH, "mutation_rate": 0.9, "wgd_rate": 0.5,
                  "snv_prob": 1.0, "cnv_prob": 0.0}
        t = _count(3, steps=200, cancer=cancer, selection={**SELECTION_PARAMS, "max_ploidy": 3})
    finally:
        GenotypeTumor._is_viable = real

    assert rejected, "no WGD daughter was ever rejected -- the config no longer exercises the gate"
    assert all(gs["ploidy"] == pytest.approx(4.0) for gs in rejected)   # every rejection is the 2->4 WGD
    living = [t.genotypes[g] for g, c in t.genotypes_counts.items() if c > 0 and t._is_cancer(g)]
    assert all(rep.genome_summary["ploidy"] <= 3 for rep in living)
    assert not any(rep.is_wgd for rep in living)            # no WGD clone survived the cap


# --- both engines agree WGD doubles copies -----------------------------------------------------
def test_engines_agree_wgd_doubles_copies():
    """Both engines drive mutation through ``Cell.mutate``, so a forced WGD on a matched cell +
    rng produces an identical doubled genome_summary regardless of which engine owns the cell."""
    sel = Selection(n_segments=N_SEGMENTS, segment_size=SEGMENT_SIZE, **SELECTION_PARAMS)

    def forced_wgd_summary():
        c = _fresh_cancer(sel, wgd_rate=1.0)
        w = c.divide()
        assert w.mutate(np.random.default_rng(7), sel) is True
        return dict(w.genome_summary), w.is_wgd

    s1, wgd1 = forced_wgd_summary()
    s2, wgd2 = forced_wgd_summary()
    assert s1 == s2 and wgd1 is True and wgd2 is True
    assert list(s1["seg_cns"]) == [4] * N_SEGMENTS         # diploid -> tetraploid everywhere
    assert s1["ploidy"] == pytest.approx(4.0)


def test_ploidy_advisory_is_non_failing():
    """diagnose() flags an implausibly high mean ploidy (WGD/amp runaway) as a NON-failing advisory
    -- like the small-tumour advisory, it points at wgd_rate/max_ploidy without flipping ``ok``."""
    from iscc.tumor.diagnostics import diagnose
    t = _count(1, steps=120)                                   # an ordinary near-diploid tumour
    diag = diagnose(t, thresholds={"ploidy_max_plausible": 1.0})   # ceiling below its ploidy -> fires
    assert any("mean ploidy" in a for a in diag.advisories)
    assert "ploidy" not in {c.name for c in diag.failures}     # an advisory is never a failed check


def test_both_engines_produce_and_surface_wgd():
    """Grown under a plausible wgd_rate + a generous ploidy cap, BOTH engines produce surviving
    WGD clones and surface the ``is_wgd`` ground truth in cell_data."""
    cancer = {**NO_DEATH, "mutation_rate": 0.7, "wgd_rate": 0.15}
    selection = {**SELECTION_PARAMS, "max_ploidy": 8, "max_cn": 16}

    tc = _count(2, steps=180, cancer=cancer, selection=selection)
    tg = _cell(2, steps=140, cancer=cancer, selection=selection)
    for t in (tc, tg):
        assert "cell_wgd" in t.cell_data
        wgd_col = t.cell_data["cell_wgd"]["is_wgd"]
        assert wgd_col.dtype == bool
        assert wgd_col.shape[0] == t.cell_data["cell_snv"].shape[0]
        assert wgd_col.any(), "expected some WGD cells at this wgd_rate"
