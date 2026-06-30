"""F6 spatial (Visium) assay tests (DESIGN_features §D "spatial" row).

Internal-validation targets from the F6 spec:
  * spot->cell aggregation: a spot's counts track the pooled expression of its member cells;
    n_cells sits in a sensible range for the radius; off-tissue (empty) spots emit nothing;
  * the HEADLINE piece — the capture-efficiency field is spatially autocorrelated: Moran's I > 0
    and rises with `field_lengthscale`; reproducible from the seed;
  * lateral mRNA diffusion raises neighbour-spot expression correlation;
  * both count models (NB, DM) run and the AnnData carries obsm['spatial'] + ground-truth obs +
    uns hyperparams.
"""
import numpy as np
import pandas as pd
import pytest

from iscc.data import (
    Visium, VisiumBatch, VisiumBatchHyperParams, morans_i,
)

GENES = [f"G_{g}" for g in range(20)]
G_HALF = len(GENES) // 2


def make_block_cell_data(grid=24, lo=2, hi=22, seed=0):
    """Two spatially-segregated clones with disjoint expression blocks, on a sub-region of the grid.

    Clone A (rows < grid/2) expresses only the first gene block; clone B the second. Cells are placed
    only for coordinates in ``[lo, hi)`` so the grid corners are OFF-TISSUE (empty spots). One cell
    per integer coordinate -> with a sub-pitch radius each spot aggregates a single coordinate.
    """
    rng = np.random.default_rng(seed)
    ids, coords, exp, ctype = [], [], [], []
    i = 0
    for r in range(lo, hi):
        for c in range(lo, hi):
            clone = "A" if r < grid / 2 else "B"
            e = np.zeros(len(GENES))
            if clone == "A":
                e[:G_HALF] = rng.gamma(3.0, 1.0, G_HALF)
            else:
                e[G_HALF:] = rng.gamma(3.0, 1.0, len(GENES) - G_HALF)
            exp.append(e); coords.append((r, c)); ctype.append(clone)
            ids.append(f"C{i}"); i += 1
    exp = pd.DataFrame(exp, index=ids, columns=GENES)
    return {
        "cell_exp": exp,
        "cell_crd": pd.DataFrame(coords, index=ids, columns=["row", "col"]),
        "cell_type": pd.DataFrame(ctype, index=ids, columns=["cell_id"]),
    }


# ----------------------------------------------------------------- aggregation ----------------
class TestSpotAggregation:
    def test_pure_clone_spot_tracks_member_expression(self):
        """A spot whose members are all one clone expresses only that clone's gene block (ambient
        off): the spot counts track the pooled member expression, not the whole-tissue average."""
        cd = make_block_cell_data()
        a = Visium(seed=1, spot_pitch=2.0, spot_radius=0.6, ambient_frac=0.0,
                   diffusion_sigma=0.0).run(cd, grid_side=24)
        pureA = a.obs.index[(a.obs.clone == "A") & (a.obs.clone_frac == 1.0) & (a.obs.n_cells > 0)]
        assert len(pureA) > 10
        block = a.spot_counts.loc[pureA].values
        assert block[:, G_HALF:].sum() == 0          # no clone-B genes in a pure clone-A spot
        assert block[:, :G_HALF].sum() > 0           # clone-A genes expressed

    def test_n_cells_in_sensible_range(self):
        cd = make_block_cell_data()
        a = Visium(seed=1, spot_pitch=2.0, spot_radius=0.6).run(cd, grid_side=24)
        occ = a.obs.n_cells[a.obs.n_cells > 0]
        assert occ.min() >= 1 and occ.max() <= 10    # ~1-10 cells / spot for this radius

    def test_empty_spots_handled(self):
        """Off-tissue spots (no member cells) emit zero counts and are flagged n_cells==0."""
        cd = make_block_cell_data(lo=6, hi=18)        # cells only in the centre -> empty corners
        a = Visium(seed=2, spot_pitch=2.0, spot_radius=0.6, ambient_frac=0.05).run(cd, grid_side=24)
        empty = a.obs.index[a.obs.n_cells == 0]
        assert len(empty) > 0
        assert a.spot_counts.loc[empty].values.sum() == 0   # empty spots produce nothing

    def test_membership_records_ground_truth(self):
        cd = make_block_cell_data()
        a = Visium(seed=1, spot_pitch=2.0, spot_radius=0.6).run(cd, grid_side=24)
        # each occupied spot records its member ids, dominant clone and clone fraction
        occ = np.where(a.obs.n_cells.values > 0)[0]
        for s in occ[:5]:
            assert len(a.spot_members[s]) == a.obs.n_cells.values[s]
            assert a.obs.clone.values[s] in ("A", "B")
            assert 0.0 < a.obs.clone_frac.values[s] <= 1.0


# ----------------------------------------------------------------- capture field --------------
class TestCaptureField:
    def _coords(self, side=15):
        xs = np.arange(side)
        return np.array([(r, c) for r in xs for c in xs], dtype=float)

    def _field(self, lengthscale, seed=1, field_sigma=0.4, edge_sigma=0.0):
        coords = self._coords()
        h = VisiumBatchHyperParams(field_lengthscale=lengthscale, field_sigma=field_sigma,
                                   edge_sigma=edge_sigma)
        b = VisiumBatch(h, seed=seed).realize([f"G{i}" for i in range(8)], np.ones(8), coords)
        return b.capture_field, coords

    def test_field_is_positive_and_spatially_autocorrelated(self):
        field, coords = self._field(5.0)
        assert np.all(field > 0)                      # smooth POSITIVE field
        assert morans_i(field, coords) > 0.1          # positive spatial autocorrelation

    def test_morans_i_rises_with_lengthscale(self):
        f_short, coords = self._field(2.0)
        f_long, _ = self._field(10.0)
        assert morans_i(f_long, coords) > morans_i(f_short, coords) > 0

    def test_reproducible_in_seed(self):
        f1, _ = self._field(5.0, seed=3)
        f2, _ = self._field(5.0, seed=3)
        f3, _ = self._field(5.0, seed=4)
        assert np.array_equal(f1, f2)                 # same seed -> identical field
        assert not np.array_equal(f1, f3)             # different seed -> different signature

    def test_realized_assay_field_is_autocorrelated(self):
        cd = make_block_cell_data()
        a = Visium(seed=7, spot_pitch=2.0, spot_radius=0.6, field_lengthscale=6.0,
                   field_sigma=0.4).run(cd, grid_side=24)
        assert morans_i(a.capture_field, a.spot_coords) > 0.1


# ----------------------------------------------------------------- diffusion ------------------
class TestDiffusion:
    def _adjacent_corr(self, diffusion_sigma):
        cd = make_block_cell_data()
        a = Visium(seed=2, spot_pitch=2.0, spot_radius=0.6, diffusion_sigma=diffusion_sigma,
                   ambient_frac=0.0, field_sigma=0.0, edge_sigma=0.0).run(cd, grid_side=24)
        counts = a.spot_counts.values
        comp = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
        coords = a.spot_coords
        corrs = []
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                d2 = float(((coords[i] - coords[j]) ** 2).sum())
                if 1.5 < d2 < 5.0 and comp[i].sum() > 0 and comp[j].sum() > 0:
                    corrs.append(np.corrcoef(comp[i], comp[j])[0, 1])
        return float(np.nanmean(corrs))

    def test_diffusion_raises_neighbour_correlation(self):
        assert self._adjacent_corr(2.0) > self._adjacent_corr(0.0)


# ----------------------------------------------------------------- count models + AnnData -----
class TestCountModelsAndAnnData:
    @pytest.mark.parametrize("count_model", ["nb", "dm"])
    def test_count_model_runs(self, count_model):
        cd = make_block_cell_data()
        a = Visium(seed=3, count_model=count_model, spot_pitch=2.0, spot_radius=0.6).run(
            cd, grid_side=24)
        assert a.spot_counts.values.sum() > 0
        assert (a.spot_counts.values >= 0).all()

    def test_anndata_structure(self):
        cd = make_block_cell_data()
        a = Visium(seed=3, spot_pitch=2.0, spot_radius=0.6).run(cd, grid_side=24)
        ad = a.to_anndata()
        assert ad.shape == (len(a.spot_names), len(GENES))
        assert "spatial" in ad.obsm and ad.obsm["spatial"].shape == (len(a.spot_names), 2)
        for col in ("n_cells", "clone", "library", "batch"):
            assert col in ad.obs.columns          # ground-truth obs
        assert ad.uns["assay"] == "visium"
        assert ad.uns["hyperparams"]["mu_counts"] == a.hypers.mu_counts
        assert ad.uns["count_model"] == "dm"

    def test_seed_reproducible_counts(self):
        cd = make_block_cell_data()
        a1 = Visium(seed=5, spot_pitch=2.0, spot_radius=0.6).run(cd, grid_side=24)
        a2 = Visium(seed=5, spot_pitch=2.0, spot_radius=0.6).run(cd, grid_side=24)
        assert np.array_equal(a1.spot_counts.values, a2.spot_counts.values)
