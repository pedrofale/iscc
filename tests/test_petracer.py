"""PEtracer lineage-vs-spatial decomposition tests (Tier 1) — DESIGN_features §H/§I.

Covers the `iscc.integrations` export seam and the flagship finding:
  * the lineage tree from `genotypes_parents` is well-formed (founder root, non-negative LCA
    distances, symmetric matrix, parent-child distance 1) and round-trips to Newick;
  * `to_anndata` packs cell_data with spatial coords + clone identity;
  * the decomposition RECOVERS the intrinsic-vs-extrinsic split under intermixing (high dispersal),
    and EXPOSES THE CONFOUND under clonal territories (low dispersal): a purely extrinsic hypoxia
    field acquires lineage autocorrelation and is mis-classified as heritable.

Deterministic small tumours (seed fixed) so the signal is stable and the suite stays fast.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

from iscc.tumor.models import GenotypeTumor
from iscc.integrations import (to_lineage_tree, to_newick, to_anndata,
                               decompose_lineage_spatial)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "validation", "data"))
import build_petracer_reference as B  # noqa: E402  (Tier-2 reducer + Newick parser)

GENOME = {"n_segments": 4, "segment_size": 25}
SELECTION = {"prop_driver": 0.1, "prop_dispersal": 0.1, "prop_immune_resistance": 0.1,
             "prop_treatment_resistance": 0.1}
DEME = {"carrying_capacity": 20, "initial_cancer_cells": 5}
SPATIAL = {"grid_size": 16, "structure_radius": 0}
MP = {"hypoxia": {"strength": 2.5, "n_genes": 20, "o2_consumption": 1.5, "o2_supply": 0.5,
                  "o2_source": "perfused"},
      "cci": {"strength": 1.5, "n_target_genes": 20, "emitter_type": "cancer", "lengthscale": 2.5}}
# With real per-deme crowding (DESIGN_crowding.md) a low-dispersal tumour SPREADS one deme at a time
# and caps near K instead of overfilling, so the territories carry fewer cells; we raise K and the
# step count so each clonal territory holds enough cells for the lineage-vs-space signal to stabilise
# (small capped tumours make the field autocorrelation noisy).
STEPS = 800
SEED = 3


def _grow(dispersal, microenv):
    cancer = {"division_rate": 0.6, "death_rate": 0.03, "max_birth_rate": 0.9,
              "mutation_rate": 0.6, "dispersal_rate": dispersal}
    t = GenotypeTumor(seed=SEED, genome_params=GENOME, selection_params=SELECTION,
                      cancer_cell_params=cancer, deme_params=DEME, spatial_params=SPATIAL,
                      microenv_params=microenv)
    t.grow(n_steps=STEPS, seed=SEED)
    return t


@pytest.fixture(scope="module")
def low():
    return _grow(0.05, MP), _grow(0.05, None)          # (on, off) — clonal territories


@pytest.fixture(scope="module")
def high():
    return _grow(0.6, MP), _grow(0.6, None)            # (on, off) — intermixed clones


def _categories(on, off, n_genes):
    ext = np.union1d(on.microenv_truth["hypoxia_genes"], on.microenv_truth["cci_target_genes"])
    cd = off.cell_data
    gid = cd["cell_type"].iloc[:, 0].astype(str).values
    isc = np.array([g in off.genotypes and off.genotypes[g].type == "cancer" for g in gid])
    means = pd.DataFrame(cd["cell_exp"].values[isc]).groupby(gid[isc]).mean().values
    intr = np.setdiff1d(np.argsort(means.var(axis=0))[::-1][:20], ext)
    neutral = np.setdiff1d(np.setdiff1d(np.arange(n_genes), ext), intr)
    return {"extrinsic": ext, "intrinsic": intr, "neutral": neutral}


# --------------------------------------------------------------------- lineage tree -----------
class TestLineageTree:
    def test_root_is_founder_at_depth_zero(self, low):
        on, _ = low
        tree = to_lineage_tree(on)
        assert tree.root == on.founder_id
        assert tree.depth(tree.root) == 0
        assert on.founder_id in tree.roots

    def test_distance_matrix_metric(self, low):
        on, _ = low
        tree = to_lineage_tree(on)
        ids = tree.nodes[:25]
        D = tree.distance_matrix(ids)
        assert np.allclose(D, D.T)                     # symmetric
        assert np.allclose(np.diag(D), 0.0)            # zero self-distance
        assert (D >= 0).all()                          # non-negative
        assert tree.lca_distance(ids[0], ids[0]) == 0

    def test_parent_child_distance_is_one(self, low):
        on, _ = low
        tree = to_lineage_tree(on)
        # a child sits exactly one edge from its parent, and one deeper
        child = next(c for c in tree.nodes if tree.parent.get(c) in set(tree.nodes))
        parent = tree.parent[child]
        assert tree.lca_distance(child, parent) == 1
        assert tree.depth(child) == tree.depth(parent) + 1

    def test_newick_wellformed(self, low):
        on, _ = low
        nw = to_newick(on)
        assert nw.endswith(";")
        assert nw.count("(") == nw.count(")")
        assert str(on.founder_id) in nw

    def test_cancer_only_excludes_normal_types(self, low):
        on, _ = low
        tree = to_lineage_tree(on)
        assert all(on.genotypes[g].type == "cancer" for g in tree.nodes)


# --------------------------------------------------------------------- anndata export ---------
class TestAnnData:
    def test_shapes_and_spatial(self, low):
        on, _ = low
        ad = to_anndata(on.cell_data)
        assert ad.n_obs == on.cell_data["cell_exp"].shape[0]
        assert ad.n_vars == on.n_genes
        assert "spatial" in ad.obsm and ad.obsm["spatial"].shape == (ad.n_obs, 2)
        assert "clone" in ad.obs.columns
        assert np.allclose(ad.X, on.cell_data["cell_exp"].values, atol=1e-4)

    def test_microenv_ground_truth_attached(self, low):
        on, _ = low
        ad = to_anndata(on.cell_data)
        assert {"hypoxia_level", "cci_level"} <= set(ad.obs.columns)


# --------------------------------------------------------------------- decomposition ----------
class TestDecomposition:
    def test_confound_under_clonal_territories(self, low):
        # LOW dispersal: the true hypoxia FIELD (purely spatial) acquires LINEAGE autocorrelation
        # exceeding its spatial autocorrelation, and its genes are (mis-)called heritable.
        on, off = low
        dec = decompose_lineage_spatial(on, seed=0)
        fld = dec.field_autocorr["hypoxia_level"]
        assert fld["lineage"] > fld["spatial"]                 # the confound (field looks heritable)
        conf = dec.confusion(_categories(on, off, on.n_genes))
        assert conf["extrinsic"] >= 0.9                        # extrinsic genes mis-called heritable
        assert conf["intrinsic"] >= 0.9                        # intrinsic genes correctly heritable

    def test_split_recovered_under_intermixing(self, high):
        # HIGH dispersal: the hypoxia field is now correctly MORE spatial than lineage, and fewer
        # extrinsic genes are mis-called heritable.
        on, off = high
        dec = decompose_lineage_spatial(on, seed=0)
        fld = dec.field_autocorr["hypoxia_level"]
        assert fld["spatial"] > fld["lineage"]                 # resolved: field reads as spatial
        conf = dec.confusion(_categories(on, off, on.n_genes))
        assert conf["intrinsic"] >= 0.9                        # intrinsic still correctly heritable

    def test_confound_decreases_with_dispersal(self, low, high):
        # The headline: the extrinsic field's apparent heritability (lineage autocorr) and the
        # extrinsic-gene mis-classification rate both fall as clones intermix.
        d_lo = decompose_lineage_spatial(low[0], seed=0)
        d_hi = decompose_lineage_spatial(high[0], seed=0)
        assert (d_lo.field_autocorr["hypoxia_level"]["lineage"]
                > d_hi.field_autocorr["hypoxia_level"]["lineage"] + 0.02)
        c_lo = d_lo.confusion(_categories(low[0], low[1], low[0].n_genes))
        c_hi = d_hi.confusion(_categories(high[0], high[1], high[0].n_genes))
        assert c_lo["extrinsic"] > c_hi["extrinsic"]

    def test_observed_counts_supported(self, low):
        # The decomposition also runs on F9 OBSERVED counts (adds realistic noise); the confound
        # (field lineage > spatial under territories) survives.
        from iscc.data import scSpatial
        on, _ = low
        adata = scSpatial(platform="merfish", n_panel_genes=60, seed=1).run(on.cell_data).to_anndata()
        obs = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)
        dec = decompose_lineage_spatial(on, expression=obs, seed=0)
        assert dec.I_lineage.shape[0] == adata.n_vars
        assert dec.field_autocorr["hypoxia_level"]["lineage"] > dec.field_autocorr["hypoxia_level"]["spatial"]

    def test_subsampling_caps_cells(self, low):
        on, _ = low
        dec = decompose_lineage_spatial(on, max_cells=50, seed=0)
        assert len(dec.cells) <= 50


# --------------------------------------------------------------------- Tier 2: real-data reducer
# Network-free: the Newick parser + `reduce_petracer` are exercised on tiny in-memory fixtures (the
# shape of the real PEtracer source); the fetcher's graceful fallback is checked directly.
class TestTier2Newick:
    def test_patristic_distances(self):
        t = B.parse_newick("((c0:1,c1:1)A:1,(c2:1,c3:1)B:1)root;")
        assert set(t.leaves) == {"c0", "c1", "c2", "c3"}
        assert t.patristic("c0", "c0") == 0.0
        assert t.patristic("c0", "c1") == 2.0            # sibling: up 1 + down 1
        assert t.patristic("c0", "c2") == 4.0            # across the root: 1+1 + 1+1
        assert t.root == "root"

    def test_branch_lengths_and_depth(self):
        t = B.parse_newick("(a:2.5,b:0.5)r;")
        assert t.patristic("a", "b") == 3.0
        assert t.depth("a") == 1 and t.depth("r") == 0

    def test_topology_only_defaults_to_unit_lengths(self):
        t = B.parse_newick("((a,b),c);")               # no branch lengths -> default 1
        assert t.patristic("a", "b") == 2.0
        assert t.clade_at_depth("a", 1) != "a"          # a's depth-1 ancestor is its clade

    def test_nx_to_tree(self):
        # the PEtracer obst format: a networkx rooted DiGraph with `length` edge weights
        import networkx as nx
        g = nx.DiGraph()
        g.add_edge("r", "i", length=2)
        g.add_edge("i", "a", length=1)
        g.add_edge("i", "b", length=1)
        g.add_edge("r", "c", length=3)
        t = B.nx_to_tree(g)
        assert t.root == "r"
        assert set(t.leaves) == {"a", "b", "c"}
        assert t.patristic("a", "b") == 2.0            # siblings under i: 1 + 1
        assert t.patristic("a", "c") == 2 + 1 + 3      # up to r (1+2) then down to c (3)

    def test_fetch_is_graceful(self):
        got, note = B.fetch_petracer("tumor1")
        assert got is None and "Figshare" in note


class TestTier2Reducer:
    def _fixture(self):
        """A tiny MERFISH-like AnnData + matching cell tree: two clades in two spatial territories.
        gene0 tracks the clade (heritable+territorial), gene1 is a spatial gradient, gene2 is noise."""
        import anndata as ad
        rng = np.random.default_rng(0)
        cellsA = [f"cA{i}" for i in range(8)]
        cellsB = [f"cB{i}" for i in range(8)]
        cells = cellsA + cellsB
        coords = np.array([[0.0, 0.0]] * 8 + [[10.0, 10.0]] * 8) + rng.normal(0, 0.2, (16, 2))
        g_clade = np.array([1.0] * 8 + [5.0] * 8)
        X = np.c_[g_clade, coords[:, 0], rng.normal(5, 1, 16)]
        adata = ad.AnnData(X=X.astype(np.float32), obs=pd.DataFrame(index=cells),
                           var=pd.DataFrame(index=["g_clade", "g_spatial", "g_noise"]))
        adata.obsm["spatial"] = coords
        nwk = ("(" + "(" + ",".join(f"{c}:1" for c in cellsA) + ")A:3,"
               + "(" + ",".join(f"{c}:1" for c in cellsB) + ")B:3)root;")
        return adata, B.parse_newick(nwk)

    def test_reduce_shapes_and_stats(self):
        adata, tree = self._fixture()
        ref = B.reduce_petracer(adata, tree, name="toy", depth_cut=1, max_cells=100, seed=0)
        assert ref["n_cells"] == 16 and ref["n_genes"] == 3 and ref["n_clones"] == 2
        assert list(ref["clone_sizes"]) == [8, 8]
        assert ref["I_lineage"].shape == (3,) and ref["I_spatial"].shape == (3,)
        # tight clades -> strong territories (within-clade spread << tumour spread)
        assert ref["territory_ratio"] < 0.2
        # the clade-driven gene is both lineage- and spatially-autocorrelated; noise gene is neither
        assert ref["I_lineage"][0] > 0.3
        assert abs(ref["I_lineage"][2]) < 0.3

    def test_reduce_confound_fields(self):
        # the two clades sit in two spatial territories -> lineage-space coupling is high and the
        # (ground-truth-free) confound signature corr(I_lineage, I_spatial) is positive.
        adata, tree = self._fixture()
        ref = B.reduce_petracer(adata, tree, name="toy", depth_cut=1)
        assert ref["coord_lineage_autocorr"] > 0.2       # coords are lineage-autocorrelated
        assert ref["confound_corr"] > 0.0                # spatial genes carry lineage autocorr
        assert np.isfinite(ref["coord_spatial_autocorr"])

    def test_reduce_rejects_disjoint_ids(self):
        adata, tree = self._fixture()
        adata.obs_names = [f"x{i}" for i in range(adata.n_obs)]   # no overlap with the tree leaves
        with pytest.raises(ValueError):
            B.reduce_petracer(adata, tree, name="toy")

    def test_reference_roundtrip(self, tmp_path):
        adata, tree = self._fixture()
        ref = B.reduce_petracer(adata, tree, name="toy", depth_cut=1)
        p = str(tmp_path / "petracer_ref_toy.npz")
        B.save_reference(p, ref)
        r2 = B.load_reference(p)
        assert r2["name"] == "toy" and int(r2["n_cells"]) == 16
        assert np.allclose(r2["I_lineage"], ref["I_lineage"])
