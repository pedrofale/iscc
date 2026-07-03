"""Multi-region "sample trees are not phylogenies" benchmark tests (Alves, Prieto & Posada 2017).

Covers ``iscc.integrations.multiregion`` — the self-contained machinery (neighbour joining, Fitch
parsimony, Robinson–Foulds) and the benchmark itself on a small deterministic tumour:

  * the naive region "sample tree" invents *spurious parallel mutations* (> 0) — mutations the true
    lineage shows arose once but the admixed region tree infers as independent origins;
  * the oracle-deconvolved *clone* tree removes almost all of them (their fix), even though it has
    MORE leaves than the region tree — so the artifact is admixture, not tree size;
  * more regions does NOT drive the naive rate to zero.

No optional tree dependency (ete3 / dendropy) is required — everything is numpy-only, so these tests
never skip on a missing dep; a guard test asserts the module imports and runs dependency-free.
"""
import numpy as np
import pandas as pd
import pytest

from iscc.tumor.models import GenotypeTumor
from iscc.sample.biopsy.biopsy import Biopsy
from iscc.integrations import multiregion as mr


# Small, fast, deterministic tumour: ~1600 loci keeps the per-allele infinite-sites model sparse
# (clean single-origin answer key); low dispersal + capacity gives spatial clone territories.
GENOME = {"n_segments": 20, "segment_size": 80}
SELECTION = {"prop_driver": 0.05, "prop_dispersal": 0.1}
DEME = {"carrying_capacity": 4}
SPATIAL = {"grid_size": 24, "structure_radius": 0}
CANCER = {"division_rate": 0.6, "death_rate": 0.05, "max_birth_rate": 0.9, "mutation_rate": 0.15,
          "dispersal_rate": 0.03, "snv_prob": 1.0, "cnv_prob": 0.0, "n_snvs_per_allele": 1.0}
STEPS = 250
SEED = 7


@pytest.fixture(scope="module")
def tumor():
    t = GenotypeTumor(seed=SEED, genome_params=GENOME, selection_params=SELECTION,
                      cancer_cell_params=CANCER, deme_params=DEME, spatial_params=SPATIAL)
    t.grow(n_steps=STEPS, seed=SEED)
    return t


@pytest.fixture(scope="module")
def answer_key(tumor):
    return mr.true_origin_counts(tumor)


def _biopsy(tumor, k, seed=0):
    rng = np.random.default_rng(seed)
    _, region_series, _ = Biopsy(tumor.cell_data, rng).sample(
        biopsy_type="multiregion", n_regions=k, radius=1.0)
    return region_series


# ---------------------------------------------------------------------------------------------
# Tree primitives
# ---------------------------------------------------------------------------------------------
def test_neighbor_joining_recovers_known_split():
    # Two cherries: (A,B) and (C,D) far apart -> the AB|CD split must appear.
    labels = ["A", "B", "C", "D"]
    D = np.array([[0, 1, 5, 5],
                  [1, 0, 5, 5],
                  [5, 5, 0, 1],
                  [5, 5, 1, 0]], dtype=float)
    adj = mr.neighbor_joining(D)
    root = 4                                     # first internal node NJ creates
    children = mr.root_children(adj, root)
    leafmap = {0: "A", 1: "B", 2: "C", 3: "D"}
    splits, allL = mr.bipartitions(children, root, leafmap)
    assert frozenset({"C", "D"}) in splits or frozenset({"A", "B"}) in splits


def test_fitch_length_compatible_vs_homoplastic():
    # tree: root -> {i0(->L0,L1), i1(->L2,L3)}
    children = {"r": ["i0", "i1"], "i0": ["L0", "L1"], "i1": ["L2", "L3"]}
    # {L0,L1}=1 is a clade -> single change
    assert mr.fitch_length(children, {"L0": 1, "L1": 1, "L2": 0, "L3": 0}, "r") == 1
    # {L0,L2}=1 crosses the split -> two independent changes (homoplasy)
    assert mr.fitch_length(children, {"L0": 1, "L1": 0, "L2": 1, "L3": 0}, "r") == 2


def test_robinson_foulds_identical_tree_is_zero():
    children = {"r": ["i0", "i1"], "i0": ["A", "B"], "i1": ["C", "D"]}
    leafmap = {"A": "A", "B": "B", "C": "C", "D": "D"}
    rf = mr.robinson_foulds(children, "r", leafmap, children, "r", leafmap)
    assert rf["rf"] == 0
    assert rf["recall"] == 1.0


def test_hamming_nj_tree_groups_identical_leaves():
    # leaves 0,1 identical; 2,3 identical but different from 0,1
    P = np.array([[1, 1, 0, 0, 0],
                  [1, 1, 0, 0, 0],
                  [0, 0, 1, 1, 1],
                  [0, 0, 1, 1, 1]])
    children, root = mr.hamming_nj_tree(P)
    leafmap = {i: str(i) for i in range(4)}
    splits, _ = mr.bipartitions(children, root, leafmap)
    assert frozenset({"0", "1"}) in splits or frozenset({"2", "3"}) in splits


# ---------------------------------------------------------------------------------------------
# Answer key (true origins)
# ---------------------------------------------------------------------------------------------
def test_answer_key_has_single_origin_loci(answer_key):
    ak = answer_key
    assert ak["origins"].shape[0] == len(ak["loci"])
    # every carried locus has >= 1 origin; the tree yields a substantial single-origin set
    carried = ak["carrier_count"] > 0
    assert np.all(ak["origins"][carried] >= 1)
    assert int(ak["single"].sum()) > 50
    # single-origin subset is exactly origins == 1
    assert np.array_equal(ak["single"], ak["origins"] == 1)


# ---------------------------------------------------------------------------------------------
# The benchmark
# ---------------------------------------------------------------------------------------------
def test_naive_region_tree_has_spurious_parallelisms(tumor, answer_key):
    rs = _biopsy(tumor, 6)
    res = mr.multiregion_phylogeny(tumor, rs, answer_key=answer_key)
    # the naive region "sample tree" invents parallel mutations for truly single-origin loci
    assert res["naive"]["spurious"] > 0
    assert res["naive"]["rate"] > 0.02


def test_deconvolution_removes_spurious_parallelisms(tumor, answer_key):
    rs = _biopsy(tumor, 6)
    res = mr.multiregion_phylogeny(tumor, rs, answer_key=answer_key)
    # the fix: clone deconvolution drops the spurious rate far below the naive region tree ...
    assert res["fix"]["rate"] < res["naive"]["rate"] / 2.0
    assert res["fix"]["rate"] < 0.05
    # ... even though the clone tree has MORE leaves than the region tree (=> admixture, not size)
    assert res["n_clones"] > res["n_regions"]


def test_deconvolved_clone_tree_recovers_true_topology(tumor, answer_key):
    rs = _biopsy(tumor, 6)
    res = mr.multiregion_phylogeny(tumor, rs, answer_key=answer_key)
    # the deconvolved clone tree recovers most true clone splits (unlike the region sample tree)
    assert res["rf"]["recall"] is not None
    assert res["rf"]["recall"] > 0.6


def test_more_regions_does_not_fix_it(tumor, answer_key):
    # the naive spurious rate does not vanish as regions are added; the fix stays ~0 throughout
    naive, fix = {}, {}
    for k in (4, 6):
        rs = _biopsy(tumor, k)
        res = mr.multiregion_phylogeny(tumor, rs, answer_key=answer_key)
        naive[k] = res["naive"]["rate"]
        fix[k] = res["fix"]["rate"]
    assert naive[4] > 0 and naive[6] > 0
    # more regions does not reduce the naive artifact toward zero (here it does not shrink at all)
    assert naive[6] >= naive[4] - 0.02
    # the deconvolved tree stays far below the naive tree at every K
    assert fix[4] < naive[4] and fix[6] < naive[6]


def test_runs_without_optional_tree_dependency():
    # the whole analysis is numpy-only: no ete3 / dendropy import path exists to fail on.
    import sys
    import importlib
    assert "ete3" not in sys.modules and "dendropy" not in sys.modules
    importlib.reload(mr)  # re-import cleanly; must not raise on a missing optional dep
    assert hasattr(mr, "neighbor_joining") and hasattr(mr, "robinson_foulds")
