"""The shared clone definition: a clone is a lineage clade with enough sampled cells.

``iscc.integrations.clones`` is what every assay notebook uses to turn the engine's raw per-cell
genotype id into a readable, stable clone annotation. These tests pin the properties the notebooks
rely on: the labels are deterministic and size-ordered (so a clone keeps its colour across notebooks
and reruns), they really are clades of the true lineage tree, only cancer cells get one, coarser
``min_cells`` gives fewer clones, and the summary table agrees with the labels cell for cell.

Runs on the shared realistic regime at the ``small`` preset (a shrunken genome keeps the dense
per-cell frames tiny) grown far enough to invade — a monoclonal lesion has nothing to label.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "validation"))
import realistic_regime as R  # noqa: E402

from conftest import GENOME_PARAMS, SELECTION_PARAMS, CANCER_CELL_PARAMS  # noqa: E402
from iscc.integrations import clones_from_clades, clone_summary  # noqa: E402
from iscc.integrations.clones import UNASSIGNED  # noqa: E402
from iscc.tumor.models import GenotypeTumor  # noqa: E402

MIN_CELLS = 20          # the sampled tumour below holds ~800 cancer cells


@pytest.fixture(scope="module")
def tumor():
    """One invaded ductal-field tumour with a branched clone tree, sampled to ~2500 cells."""
    return R.grow_realistic(seed=3, target_cancer=12_000, scale="small",
                            genome={"n_segments": 4, "segment_size": 50},
                            max_cells=2500, materialise=True)


@pytest.fixture(scope="module")
def clones(tumor):
    return clones_from_clades(tumor, min_cells=MIN_CELLS)


# --- helpers ---------------------------------------------------------------------------------
def _genotypes(tumor, index=None):
    g = tumor.cell_data["cell_type"]["cell_id"].astype(str)
    return g if index is None else g.loc[index]


def _ancestors(tumor, gid):
    """``[gid, parent, ..., founder]``."""
    out, cur, seen = [], gid, set()
    while cur is not None and cur not in seen:
        out.append(cur)
        seen.add(cur)
        cur = tumor.genotypes_parents.get(cur)
    return out


def _clone_labels(clones):
    return [lab for lab in clones.cat.categories if lab != UNASSIGNED]


# --- determinism / ordering ------------------------------------------------------------------
def test_labels_are_deterministic(tumor, clones):
    again = clones_from_clades(tumor, min_cells=MIN_CELLS)
    assert clones.equals(again)
    assert clones.attrs["clade_root"] == again.attrs["clade_root"]


def test_labels_do_not_depend_on_cell_order(tumor, clones):
    shuffled = list(clones.index[::-1])
    other = clones_from_clades(tumor, min_cells=MIN_CELLS, cell_ids=shuffled)
    # Same cells, listed backwards: every cell must keep its label, and the clades their identity.
    assert other.reindex(clones.index).astype(object).equals(clones.astype(object))
    assert other.attrs["clade_root"] == clones.attrs["clade_root"]


def test_clones_are_ordered_by_size(clones):
    labels = _clone_labels(clones)
    assert labels == [f"clone_{i}" for i in range(1, len(labels) + 1)]
    sizes = [int((clones == lab).sum()) for lab in labels]
    assert sizes == sorted(sizes, reverse=True)
    assert list(clones.cat.categories)[-1] == UNASSIGNED  # the unassigned bucket always sorts last


def test_label_prefix_is_configurable(tumor):
    named = clones_from_clades(tumor, min_cells=MIN_CELLS, label_prefix="subclone")
    assert [lab for lab in named.cat.categories if lab != UNASSIGNED][0] == "subclone_1"


# --- the clade definition --------------------------------------------------------------------
def test_every_clone_is_a_clade_of_the_lineage_tree(tumor, clones):
    genotype = _genotypes(tumor, clones.index)
    for label, root in clones.attrs["clade_root"].items():
        inside = (clones == label).values
        # every cell of the clone descends from the clade root ...
        for gid in pd.unique(genotype.values[inside]):
            assert root in _ancestors(tumor, gid), f"{label}: {gid} is not below {root}"
        # ... and no cell outside it does (the clades are disjoint, complete subtrees)
        for gid in pd.unique(genotype.values[~inside]):
            assert root not in _ancestors(tumor, gid), f"{label}: {gid} below {root} but unlabelled"


def test_every_clone_holds_at_least_min_cells(clones):
    for label in _clone_labels(clones):
        assert int((clones == label).sum()) >= MIN_CELLS


def test_more_min_cells_gives_fewer_clones(tumor):
    counts = [len(_clone_labels(clones_from_clades(tumor, min_cells=m)))
              for m in (5, 10, 20, 30, 50, 200)]
    assert counts == sorted(counts, reverse=True)
    assert counts[0] > counts[-1]          # the sweep really does coarsen this tumour
    assert counts[-1] >= 1


def test_min_cells_beyond_the_whole_tumour_collapses_to_one_clone(tumor, clones):
    n_cancer = int(clones.notna().sum())
    one = clones_from_clades(tumor, min_cells=10 * n_cancer)
    assert _clone_labels(one) == ["clone_1"]
    assert int((one == "clone_1").sum()) == n_cancer     # nothing left over in `other`
    assert UNASSIGNED not in list(one.cat.categories)


def test_unassigned_cells_sit_above_every_clade(tumor, clones):
    genotype = _genotypes(tumor, clones.index)
    roots = set(clones.attrs["clade_root"].values())
    for gid in pd.unique(genotype.values[(clones == UNASSIGNED).values]):
        assert not roots & set(_ancestors(tumor, gid))


# --- cancer only -----------------------------------------------------------------------------
def test_only_cancer_cells_are_labelled(tumor, clones):
    genotype = _genotypes(tumor, clones.index)
    is_cancer = np.array([tumor._is_cancer(g) for g in genotype.values])
    assert is_cancer.sum() > 0 and (~is_cancer).sum() > 0      # a mixed microenvironment
    assert clones.notna().values.tolist() == is_cancer.tolist()


def test_normal_cells_are_nan(tumor, clones):
    genotype = _genotypes(tumor, clones.index)
    normals = clones[[not tumor._is_cancer(g) for g in genotype.values]]
    assert len(normals) and normals.isna().all()


# --- subsets, errors -------------------------------------------------------------------------
def test_cell_ids_subset_defines_clones_on_that_subset(tumor, clones):
    subset = list(clones.index[:1500])
    sub = clones_from_clades(tumor, min_cells=MIN_CELLS, cell_ids=subset)
    assert list(sub.index) == subset
    assert len(_clone_labels(sub)) >= 1
    # clades are counted over the subset only, so they can only get coarser
    assert len(_clone_labels(sub)) <= len(_clone_labels(clones))


def test_unknown_cell_ids_are_rejected(tumor, clones):
    with pytest.raises(KeyError, match="not in this tumour"):
        clones_from_clades(tumor, cell_ids=list(clones.index[:5]) + ["not-a-cell"])


def test_min_cells_must_be_positive(tumor):
    with pytest.raises(ValueError, match="min_cells"):
        clones_from_clades(tumor, min_cells=0)


def test_unmaterialised_tumour_raises_a_clear_error():
    t = _small_tumour()
    t.cell_data = None                      # the memory-safe cm-scale default
    with pytest.raises(ValueError, match="make_cell_data"):
        clones_from_clades(t)


# --- the summary table -----------------------------------------------------------------------
def test_clone_summary_matches_the_labels(tumor, clones):
    summary = clone_summary(tumor, clones)
    assert list(summary.index) == [lab for lab in clones.cat.categories
                                   if int((clones == lab).sum())]
    for label in summary.index:
        assert summary.loc[label, "n_cells"] == int((clones == label).sum())
    assert summary["n_cells"].sum() == int(clones.notna().sum())


def test_clone_summary_reports_the_shared_state_of_each_clone(tumor, clones):
    summary = clone_summary(tumor, clones)
    genotype = _genotypes(tumor, clones.index)
    for label in _clone_labels(clones):
        rep = summary.loc[label, "genotype"]
        assert rep in tumor.genotypes and tumor._is_cancer(rep)
        # the representative is ancestral to every sampled cell of the clone (their shared state)
        for gid in pd.unique(genotype.values[(clones == label).values]):
            assert rep in _ancestors(tumor, gid)
        assert summary.loc[label, "n_genotypes"] >= 1
    # the one-line trait string agrees with the tallies it summarises
    onc = summary.loc[summary["n_mut_onc"].fillna(0) > 0]
    assert all("oncogene" in t for t in onc["traits"])
    none = summary.loc[_clone_labels(clones)]
    none = none[none[[c for c in none.columns if c.startswith("n_mut_")]].sum(axis=1) == 0]
    assert all(t == "none" for t in none["traits"])


def test_clone_summary_survives_a_plain_series(tumor, clones):
    # `attrs` do not survive every pandas round-trip (e.g. going through an AnnData obs column), so
    # the summary must still rebuild each clone's representative from the tree.
    plain = pd.Series(clones.astype(object).values, index=clones.index, name="clone")
    assert not plain.attrs
    rebuilt = clone_summary(tumor, plain)
    expected = clone_summary(tumor, clones)
    assert list(rebuilt.index) == list(expected.index)
    assert rebuilt["n_cells"].tolist() == expected["n_cells"].tolist()
    assert rebuilt["genotype"].tolist() == expected["genotype"].tolist()


# --- metastasis: one clone tree across both compartments --------------------------------------
def _small_tumour(spatial=None, seed=3):
    spatial = {"grid_size": 16, "structure_radius": 3, "n_glands": 3, "gland_radius": 3,
               "min_gland_sep": 8, "K_duct": 28, "K_stroma": 18, "stroma_fill_frac": 0.3,
               "cross_gland_kappa": 0.05, "epithelial_barrier": 1.2, "stromal_hazard": 0.7,
               **(spatial or {})}
    selection = {**SELECTION_PARAMS, "prop_breach": 0.2, "prop_stromal_survival": 0.2,
                 "prop_met_survival": 0.2, "breach_effects": 2.0, "stromal_survival_effects": 2.0,
                 "met_survival_effects": 2.2}
    t = GenotypeTumor(seed=seed, genome_params=GENOME_PARAMS, selection_params=selection,
                      cancer_cell_params=CANCER_CELL_PARAMS,
                      deme_params={"carrying_capacity": 16, "initial_cancer_cells": 6,
                                   "resident_pressure_ref": 0.2},
                      spatial_params=spatial, update_mode="tau", tau=1.0)
    t.grow(n_steps=2, seed=seed)
    return t


@pytest.fixture(scope="module")
def met_tumor():
    """A tumour that has seeded a metastatic deposit, grown until the met holds cells."""
    t = _small_tumour({"met_grid_size": 10, "K_met": 16, "host_fill_frac": 0.4,
                       "met_seed_kappa": 0.08, "met_hazard": 0.5, "met_transit_floor": 0.03})
    for _ in range(70):
        met = sum(c for i in range(t.n_primary_demes, len(t.demes))
                  for g, c in t.demes[i].items() if t._is_cancer(g))
        if met >= 40 or t.get_cancer_size() >= 4000:
            break
        t.grow(n_steps=2, seed=t.seed)
    t.make_cell_data()
    return t


def test_clones_span_primary_and_metastasis(met_tumor):
    compartment = met_tumor.cell_data["cell_compartment"]["compartment"]
    clones = clones_from_clades(met_tumor, min_cells=5)
    genotype = _genotypes(met_tumor, clones.index)
    cancer = np.array([met_tumor._is_cancer(g) for g in genotype.values])
    assert (cancer & (compartment.values == 1)).sum() > 0     # the deposit really is colonised
    # Both sites are labelled from the ONE shared clone tree, so a met cell is never left out.
    assert clones[cancer].notna().all()
    assert len(_clone_labels(clones)) >= 1
    summary = clone_summary(met_tumor, clones)
    assert summary["n_cells"].sum() == int(cancer.sum())
