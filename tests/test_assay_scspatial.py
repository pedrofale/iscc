"""F9 single-cell spatial (imaging) assay tests (DESIGN_features F9 milestone).

Internal-validation targets for the F9 spec:
  * SINGLE-CELL resolution — one observation per input cell (no spot aggregation), each carrying
    its own `cell_crd` coordinates (the imaging headline vs Visium/F6);
  * TRANSCRIPTOME COVERAGE — the measured gene set is the requested panel (`panel` list or
    `n_panel_genes` top-N), whole-transcriptome by default;
  * counts track the cell's own expression (a pure clone cell expresses only its gene block);
  * DATA DISTRIBUTION knobs behave (`mu_counts` sets per-cell depth);
  * both count models (NB, DM) run; AnnData carries obsm['spatial'] + ground-truth obs + uns;
  * reproducible from the seed.
"""
import numpy as np
import pandas as pd
import pytest

from iscc.data import scSpatial, ASSAYS

GENES = [f"G_{g}" for g in range(20)]
G_HALF = len(GENES) // 2


def make_block_cell_data(grid=24, lo=2, hi=22, seed=0):
    """Two spatially-segregated clones with disjoint expression blocks (one cell per coordinate)."""
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


# ------------------------------------------------------------- single-cell resolution ---------
class TestSingleCellResolution:
    def test_one_observation_per_cell(self):
        cd = make_block_cell_data()
        a = scSpatial(seed=1).run(cd)
        assert a.observed_counts.shape[0] == cd["cell_exp"].shape[0]      # no aggregation
        assert list(a.observed_counts.index) == list(cd["cell_exp"].index)

    def test_coordinates_retained_and_match_input(self):
        cd = make_block_cell_data()
        a = scSpatial(seed=1).run(cd)
        assert {"row", "col"} <= set(a.obs.columns)
        got = a.obs[["row", "col"]].loc[cd["cell_crd"].index]
        assert np.array_equal(got.values, cd["cell_crd"][["row", "col"]].values)

    def test_cell_subset_measures_only_those_cells(self):
        cd = make_block_cell_data()
        subset = list(cd["cell_exp"].index[:30])
        a = scSpatial(seed=1).run(cd, cell_subset=subset)
        assert list(a.observed_counts.index) == subset


# ------------------------------------------------------------- transcriptome coverage ---------
class TestPanel:
    def test_whole_transcriptome_by_default(self):
        cd = make_block_cell_data()
        a = scSpatial(seed=1).run(cd)
        assert list(a.genes) == GENES

    def test_n_panel_genes_selects_top_n(self):
        cd = make_block_cell_data()
        a = scSpatial(n_panel_genes=8, seed=1).run(cd)
        assert len(a.genes) == 8
        assert a.observed_counts.shape[1] == 8
        assert set(a.genes) <= set(GENES)                                # genome order preserved

    def test_explicit_panel(self):
        cd = make_block_cell_data()
        panel = ["G_1", "G_5", "G_10"]
        a = scSpatial(panel=panel, seed=1).run(cd)
        assert list(a.genes) == panel

    def test_empty_panel_raises(self):
        cd = make_block_cell_data()
        with pytest.raises(ValueError):
            scSpatial(panel=["not_a_gene"], seed=1).run(cd)


# ------------------------------------------------------------- counts / distribution ----------
class TestCounts:
    def test_pure_clone_cell_expresses_only_its_block(self):
        """With ambient off, a clone-A cell has zero counts in the clone-B gene block."""
        cd = make_block_cell_data()
        a = scSpatial(seed=2, ambient_frac=0.0).run(cd)
        cloneA = a.obs.index[a.obs.clone == "A"]
        block = a.observed_counts.loc[cloneA].values
        assert block[:, G_HALF:].sum() == 0                              # no clone-B genes
        assert block[:, :G_HALF].sum() > 0

    def test_mu_counts_sets_depth(self):
        cd = make_block_cell_data()
        lo = scSpatial(seed=3, mu_counts=200.0, ambient_frac=0.0).run(cd)
        hi = scSpatial(seed=3, mu_counts=1000.0, ambient_frac=0.0).run(cd)
        assert hi.observed_counts.values.sum(1).mean() > lo.observed_counts.values.sum(1).mean()

    @pytest.mark.parametrize("count_model", ["nb", "dm"])
    def test_count_models_run(self, count_model):
        cd = make_block_cell_data()
        a = scSpatial(seed=3, count_model=count_model).run(cd)
        assert a.observed_counts.values.sum() > 0
        assert (a.observed_counts.values >= 0).all()


# ------------------------------------------------------------- AnnData + reproducibility ------
class TestAnnDataAndSeed:
    def test_anndata_structure(self):
        cd = make_block_cell_data()
        a = scSpatial(n_panel_genes=12, seed=3).run(cd)
        ad = a.to_anndata()
        assert ad.shape == (cd["cell_exp"].shape[0], 12)
        assert "spatial" in ad.obsm and ad.obsm["spatial"].shape == (cd["cell_exp"].shape[0], 2)
        for col in ("clone", "row", "col", "n_counts", "platform"):
            assert col in ad.obs.columns
        assert ad.uns["assay"] == "scspatial"
        assert ad.uns["n_panel_genes"] == 12

    def test_registered_in_assays(self):
        assert ASSAYS["scspatial"] is scSpatial

    def test_seed_reproducible(self):
        cd = make_block_cell_data()
        a1 = scSpatial(seed=5).run(cd)
        a2 = scSpatial(seed=5).run(cd)
        a3 = scSpatial(seed=6).run(cd)
        assert np.array_equal(a1.observed_counts.values, a2.observed_counts.values)
        assert not np.array_equal(a1.observed_counts.values, a3.observed_counts.values)
