"""Coalescent SNV overlay (v1, DESIGN_snv_coalescent.md §9 step 1).

The count engine coarsens pure-passenger divisions into a per-clone counter (``_pass_load``) instead
of spawning a genotype each. ``GenotypeTumor._reconstruct_passengers`` re-emits those passengers at
materialisation. This module tests the v1 "star-at-founding" overlay that REPLACED the old per-cell
INDEPENDENT reconstruction (which re-drew each cell's passengers from scratch, destroying the
passenger genealogy) with a LINEAGE-CONSISTENT, allele-resolved one:

  * clade-clonal "stem" passengers born on a clone's edge are shared by its whole subtree, so cells of
    the same clade share passenger SNVs and a passenger-only tree recovers the clone-tree clades;
  * multiplicity is propagated through the exact per-clone allele-specific CN events (WGD / amp / del);
  * the primary output is allele-resolved (``cell_snv_p`` + ``cell_snv_m`` == ``cell_snv``), with the
    collapsed ``cell_snv`` kept back-compatible in shape/semantics.

Acceptance (see the DESIGN §8 validation plan): the headline is driver<->passenger consistency — the
new overlay's cophenetic correlation (single-cell Hamming distance vs the true lineage distance) is
MATERIALLY higher than the old independent baseline, which is ~noise for the passenger layer.
"""
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "validation"))
import realistic_regime as RR  # noqa: E402

from iscc.tumor.models import GenotypeTumor  # noqa: E402
from iscc.integrations.lineage import to_lineage_tree  # noqa: E402

RUN_SLOW = os.environ.get("ISCC_RUN_SLOW") == "1"


# --------------------------------------------------------------------------- shared small tumour
def _grow_small(seed=3, target=3000, wgd=0.1, max_cells=4000):
    """A small realistic breach-gated ductal-field tumour, coarsening ON + WGD ON (so there is a
    folded passenger burden and copy-number churn for the multiplicity test). Materialised."""
    return RR.grow_realistic(
        seed=seed, target_cancer=target, scale="small", coarsen=True,
        cancer={"mutation_rate": 0.6, "n_snvs_per_allele": 0.1, "wgd_rate": wgd},
        materialise=True, max_cells=max_cells)


@pytest.fixture(scope="module")
def tumor():
    return _grow_small()


# --------------------------------------------------------------------------- baseline (pre-change)
def old_independent_reconstruction(t, seed_off=909090):
    """The pre-change per-cell-INDEPENDENT reconstruction, recomputed on the SAME materialised cells
    for a fair before/after baseline: each cancer cell draws ``Poisson(load/count)`` neutral sites at
    VAF ``1/cn``, on top of the exact clonal genome (``get_snvs``)."""
    cd = t.cell_data
    types = cd["cell_type"]["cell_id"].astype(str).values
    n_cells, n_genes = cd["cell_snv"].shape
    snv = np.zeros((n_cells, n_genes))
    for i, g in enumerate(types):
        snv[i] = t.genotypes[g].get_snvs()
    prng = np.random.default_rng(t.seed + seed_off)
    neutral = t._neutral_gene_ids
    seg_of = t._gene_segment
    n_neutral = len(neutral)
    mus = {}
    for g in set(types):
        if g in t._pass_load and t._is_cancer(g):
            cnt = t.genotypes_counts.get(g, 0)
            mus[g] = (t._pass_load[g] / cnt) if cnt else 0.0
    for i, g in enumerate(types):
        mu = mus.get(g, 0.0)
        if mu <= 0:
            continue
        k = int(prng.poisson(mu))
        if k <= 0:
            continue
        pos = prng.choice(neutral, size=min(k, n_neutral), replace=False)
        cns = t.genotypes[g].genome_summary["seg_cns"]
        for p in pos:
            cn = cns[seg_of[p]]
            snv[i, p] = (1.0 / cn) if cn > 0 else 0.0
    return snv


def cophenetic_r(t, snv, n=250, seed=0):
    """Pearson r between observed neutral-site Hamming distance and true lineage (LCA) distance over a
    random sample of cancer cells — the DESIGN §8 "cophenetic jump" metric (as in the phylo probe)."""
    from scipy.stats import pearsonr
    cd = t.cell_data
    types = cd["cell_type"]["cell_id"].astype(str).values
    idx = np.arange(len(types))
    is_can = np.array([t._is_cancer(g) for g in types])
    cidx = idx[is_can]
    rng = np.random.default_rng(seed)
    if len(cidx) > n:
        cidx = np.sort(rng.choice(cidx, size=n, replace=False))
    gid_s = types[cidx]
    lt = to_lineage_tree(t)
    uniq = list(dict.fromkeys(gid_s.tolist()))
    Du = lt.distance_matrix(ids=uniq)
    gi = {g: i for i, g in enumerate(uniq)}
    cg = np.array([gi[g] for g in gid_s])
    Dlca = Du[cg][:, cg]
    neutral = t._neutral_gene_ids
    present = (snv[np.ix_(cidx, neutral)] > 0.1).astype(int)
    info = present.std(0) > 0
    if info.sum() < 2:
        return float("nan")
    P = present[:, info]
    Dobs = (P[:, None, :] != P[None, :, :]).sum(-1).astype(float)
    iu = np.triu_indices(len(cidx), 1)
    return pearsonr(Dlca[iu], Dobs[iu])[0]


# =========================================================================== tests
def test_allele_resolved_frames_sum_to_collapsed(tumor):
    """The overlay's PRIMARY output is allele-resolved: cell_snv_p + cell_snv_m == cell_snv exactly,
    and both new frames are present with the right shape (only when the overlay ran)."""
    cd = tumor.cell_data
    assert "cell_snv_p" in cd and "cell_snv_m" in cd
    snv = cd["cell_snv"].values
    p = cd["cell_snv_p"].values
    m = cd["cell_snv_m"].values
    assert p.shape == snv.shape == m.shape
    np.testing.assert_allclose(p + m, snv, atol=1e-9)
    assert snv.min() >= 0.0 and snv.max() <= 1.0
    # the overlay actually did something: cancer cells carry a neutral passenger burden
    types = cd["cell_type"]["cell_id"].astype(str).values
    is_can = np.array([tumor._is_cancer(g) for g in types])
    neutral = tumor._neutral_gene_ids
    burden = (snv[np.ix_(is_can, neutral)] > 1e-9).sum(1)
    assert burden.mean() > 1.0


def test_overlay_is_reproducible():
    """Same tumour + same sample -> byte-identical reconstructed genomes (dedicated seeded rng)."""
    a = _grow_small()
    b = _grow_small()
    for key in ("cell_snv", "cell_snv_p", "cell_snv_m"):
        assert np.array_equal(a.cell_data[key].values, b.cell_data[key].values), key


def test_fallback_is_byte_identical_with_coarsening_off():
    """With coarsening OFF the overlay is a no-op: no allele frames are added and cell_snv is exactly
    the stacked exact-genome VAFs (the untouched pre-change path)."""
    t = GenotypeTumor(seed=5, genome_params={"n_segments": 3, "segment_size": 30},
                      selection_params={"prop_driver": 0.1, "prop_dispersal": 0.1},
                      cancer_cell_params={"division_rate": 0.6, "death_rate": 0.1,
                                          "max_birth_rate": 0.9, "mutation_rate": 0.6,
                                          "dispersal_rate": 0.2},
                      deme_params={"carrying_capacity": 8, "initial_cancer_cells": 5},
                      spatial_params={"grid_size": 8, "structure_radius": 0},
                      update_mode="tau", tau=0.5, coarsen_passengers=False)
    t.grow(n_steps=40, seed=5)
    cd = t.make_cell_data()
    assert "cell_snv_p" not in cd and "cell_snv_m" not in cd
    types = cd["cell_type"]["cell_id"].astype(str).values
    expected = np.array([t.genotypes[g].get_snvs() for g in types])
    np.testing.assert_array_equal(cd["cell_snv"].values, expected)


def test_multiplicity_is_not_all_one_under_wgd(tumor):
    """On a WGD tumour some passenger sites sit at multiplicity >= 2 (pre-WGD / amplified) rather than
    all at 1 — recovering the mutation-timing signal the flat ``1/cn`` placeholder destroyed."""
    cd = tumor.cell_data
    types = cd["cell_type"]["cell_id"].astype(str).values
    is_can = np.array([tumor._is_cancer(g) for g in types])
    neutral = tumor._neutral_gene_ids
    snv = cd["cell_snv"].values
    cnv = cd["cell_cnv"].values
    # variant copies = VAF * total copy number, at neutral sites of cancer cells
    mult = np.round(snv[np.ix_(is_can, neutral)] * cnv[np.ix_(is_can, neutral)]).astype(int)
    variant = mult[mult > 0]
    assert variant.size > 0
    frac_multi = (variant >= 2).mean()
    assert frac_multi > 0.02, f"expected a non-trivial multiplicity>=2 tail, got {frac_multi:.3f}"


def test_driver_passenger_consistency_headline(tumor):
    """HEADLINE (DESIGN §6 invariant): a tree built from the reconstructed passenger SNVs recovers the
    true clone-tree structure. Quantified as the cophenetic correlation between single-cell Hamming
    distance and the true lineage distance — the NEW lineage-consistent overlay is MATERIALLY higher
    than the OLD per-cell-independent baseline (which carries only the sparse clonal-genome signal)."""
    new = tumor.cell_data["cell_snv"].values
    old = old_independent_reconstruction(tumor)
    r_new = cophenetic_r(tumor, new)
    r_old = cophenetic_r(tumor, old)
    assert r_new > r_old + 0.15, f"overlay did not improve clade recovery: new={r_new:.3f} old={r_old:.3f}"
    assert r_new > 0.7, f"absolute clade recovery too low: {r_new:.3f}"


@pytest.mark.skipif(not RUN_SLOW, reason="slow cm-ish verification; set ISCC_RUN_SLOW=1 to run")
def test_cophenetic_jump_at_scale():
    """The cophenetic jump persists at a larger (mid) scale, where the passenger layer dominates the
    signal and the old independent draw is close to noise."""
    t = RR.grow_realistic(seed=3, target_cancer=20000, scale="mid", coarsen=True,
                          cancer={"mutation_rate": 0.6, "n_snvs_per_allele": 0.1, "wgd_rate": 0.05},
                          materialise=True, max_cells=12000)
    new = t.cell_data["cell_snv"].values
    old = old_independent_reconstruction(t)
    r_new = cophenetic_r(t, new)
    r_old = cophenetic_r(t, old)
    assert r_new > r_old + 0.1, f"new={r_new:.3f} old={r_old:.3f}"
