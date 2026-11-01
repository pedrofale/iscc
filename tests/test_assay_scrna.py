"""Tests for the scRNA assay batch-effect model (DESIGN_features §B / milestone F3)."""
import hashlib

import numpy as np
import pandas as pd
import pytest

from conftest import N_GENES


def _plain_cell_data(n_cells=600, n_genes=40, seed=0):
    """Expression table with no ground-truth annotation, fixed shape and seed.

    Used by the byte-identity regression test, so its construction must never change.
    """
    rng = np.random.default_rng(seed)
    gene_base = rng.lognormal(0.0, 1.0, size=n_genes)
    noise = rng.beta(0.5, 1.0, size=(n_cells, n_genes))
    exp = pd.DataFrame(gene_base[None, :] * noise,
                       index=[f"C{i}" for i in range(n_cells)],
                       columns=[f"g{i}" for i in range(n_genes)])
    return {"cell_exp": exp}


def _counts_hash(counts):
    return hashlib.sha256(np.ascontiguousarray(counts.astype(np.int64))).hexdigest()


def _icc_by_group(values, groups):
    """One-way random-effects ICC of log `values`: the share of the variance between groups.

    ICC ~ 0 means group membership carries no information about the value; ICC > 0 means
    members of a group are more alike than members of different groups.
    """
    y = np.log(np.asarray(values, dtype=float))
    g = np.asarray(groups)
    labels, sizes = np.unique(g, return_counts=True)
    means = np.array([y[g == lab].mean() for lab in labels])
    grand = y.mean()
    ms_between = float((sizes * (means - grand) ** 2).sum() / (len(labels) - 1))
    ss_within = float(sum(((y[g == lab] - m) ** 2).sum() for lab, m in zip(labels, means)))
    ms_within = ss_within / (len(y) - len(labels))
    var_between = (ms_between - ms_within) / sizes.mean()
    return var_between / (var_between + ms_within), var_between, ms_within


@pytest.fixture
def labelled_cell_data():
    """cell_data with ground truth (clone + coords) like Tumor.make_cell_data output."""
    rng = np.random.default_rng(0)
    n_cells = 60
    names = [f"C{i}" for i in range(n_cells)]
    cols = [f"G_{s}_{p}" for s in range(2) for p in range(N_GENES // 2)]
    # per-gene baseline (real expression has gene-level structure) * per-cell noise
    gene_base = rng.lognormal(mean=0.0, sigma=1.0, size=N_GENES)
    noise = rng.beta(0.5, 1.0, size=(n_cells, N_GENES))
    exp = pd.DataFrame(gene_base[None, :] * noise, index=names, columns=cols)
    clones = rng.choice(["g0", "g1", "g2"], size=n_cells)
    cell_type = pd.DataFrame({"cell_id": clones}, index=names)
    cell_crd = pd.DataFrame(
        {"row": rng.integers(0, 10, n_cells), "col": rng.integers(0, 10, n_cells)}, index=names
    )
    return {"cell_exp": exp, "cell_type": cell_type, "cell_crd": cell_crd}


# --------------------------------------------------------------------------------------
# Batch realization (batch.py)
# --------------------------------------------------------------------------------------
class TestBatch:
    def test_beta_per_gene_shared_and_lognormal(self, labelled_cell_data):
        from iscc.data.batch import Batch, RNABatchHyperParams
        genes = labelled_cell_data["cell_exp"].columns
        b = Batch(RNABatchHyperParams(sigma_batch=0.3), seed=1).realize(genes, np.ones(N_GENES))
        assert b.beta.shape == (N_GENES,)
        assert (b.beta > 0).all()                       # lognormal -> strictly positive

    def test_library_factors_centered_on_depth(self):
        from iscc.data.batch import Batch, RNABatchHyperParams
        b = Batch(RNABatchHyperParams(mu_lib=5000.0, sigma_lib=0.3, depth_batch_sigma=0.0),
                  seed=2).realize([f"g{i}" for i in range(5)], np.ones(5))
        lib = b.library_factors(5000)
        assert lib.std() > 0                            # varies cell to cell
        assert 0.8 * 5000 < lib.mean() < 1.25 * 5000    # centered on depth

    def test_composition_rows_sum_to_one(self):
        from iscc.data.batch import Batch, RNABatchHyperParams
        b = Batch(RNABatchHyperParams(), seed=3).realize([f"g{i}" for i in range(4)], np.ones(4))
        probs = np.array([[0.25, 0.25, 0.25, 0.25], [0.7, 0.1, 0.1, 0.1]])
        comp = b.composition(probs)
        assert np.allclose(comp.sum(axis=1), 1.0)

    def test_dm_count_model_emits_fixed_total(self):
        # The DM seam is now implemented (F6 wired it for the Visium assay): a fixed library total
        # partitioned across genes via Dirichlet(kappa*comp) -> Multinomial, so each cell's counts
        # sum to round(lib) exactly (the compositional alternative to the independent-per-gene NB).
        from iscc.data.batch import Batch, RNABatchHyperParams
        b = Batch(RNABatchHyperParams(kappa=50.0), seed=4).realize([f"g{i}" for i in range(3)],
                                                                 np.ones(3))
        comp = np.full((2, 3), 1 / 3)
        counts = b.emit(comp, np.array([100.0, 100.0]), count_model="dm")
        assert counts.shape == (2, 3)
        assert np.all(counts.sum(axis=1) == 100)        # fixed total per cell (compositional)

    def test_unknown_count_model_raises(self):
        from iscc.data.batch import Batch, RNABatchHyperParams
        b = Batch(RNABatchHyperParams(), seed=5).realize(["g0"], np.ones(1))
        with pytest.raises(ValueError):
            b.emit(np.ones((1, 1)), np.array([10.0]), count_model="bogus")


# --------------------------------------------------------------------------------------
# scRNA assay
# --------------------------------------------------------------------------------------
class TestScRNAAssay:
    def test_output_shape_and_integer_counts(self, labelled_cell_data):
        from iscc.data.rna import scRNA
        a = scRNA(protocol="10x", n_cells=30).run(labelled_cell_data)
        assert a.observed_counts.shape == (30, N_GENES)
        vals = a.observed_counts.values
        assert (vals >= 0).all() and np.issubdtype(vals.dtype, np.integer)

    def test_seed_reproducible_and_seed_changes_signature(self, labelled_cell_data):
        from iscc.data.rna import scRNA
        sub = labelled_cell_data["cell_exp"].index[:25]
        a1 = scRNA(protocol="10x", seed=7).run(labelled_cell_data, cell_subset=sub)
        a2 = scRNA(protocol="10x", seed=7).run(labelled_cell_data, cell_subset=sub)
        a3 = scRNA(protocol="10x", seed=8).run(labelled_cell_data, cell_subset=sub)
        # same hypers + same seed -> identical technical signature
        assert np.array_equal(a1.observed_counts.values, a2.observed_counts.values)
        # different seed -> different signature (different per-gene beta etc.)
        assert not np.array_equal(a1.observed_counts.values, a3.observed_counts.values)
        assert not np.allclose(a1.batch.beta, a3.batch.beta)

    def test_ground_truth_surfaced_in_obs(self, labelled_cell_data):
        from iscc.data.rna import scRNA
        a = scRNA(protocol="10x", n_cells=30, batch_label="B0").run(labelled_cell_data)
        for col in ["batch", "protocol", "n_counts", "n_genes", "is_doublet", "clone", "row", "col"]:
            assert col in a.obs.columns
        assert (a.obs["batch"] == "B0").all()
        # clone label matches the source ground truth for the sampled cells
        src = labelled_cell_data["cell_type"]["cell_id"]
        assert (a.obs["clone"].values == src.reindex(a.obs.index).astype(str).values).all()

    def test_smartseq3_deeper_and_sparser_than_10x(self, labelled_cell_data):
        """Smart-seq3 preset: higher sensitivity (deeper) and lower dropout (fewer zeros)."""
        from iscc.data.rna import scRNA
        sub = labelled_cell_data["cell_exp"].index
        tenx = scRNA(protocol="10x", seed=1).run(labelled_cell_data, cell_subset=sub)
        ss3 = scRNA(protocol="smartseq3", seed=1).run(labelled_cell_data, cell_subset=sub)
        assert ss3.observed_counts.values.sum() > tenx.observed_counts.values.sum()
        ss3_zero = (ss3.observed_counts.values == 0).mean()
        tenx_zero = (tenx.observed_counts.values == 0).mean()
        assert ss3_zero < tenx_zero

    def test_batch_factor_strength_changes_composition(self, labelled_cell_data):
        """Larger sigma_batch => larger per-gene technical fold-changes (wider beta spread)."""
        from iscc.data.rna import scRNA
        sub = labelled_cell_data["cell_exp"].index
        weak = scRNA(protocol="10x", sigma_batch=0.01, seed=2).run(labelled_cell_data, cell_subset=sub)
        strong = scRNA(protocol="10x", sigma_batch=0.8, seed=2).run(labelled_cell_data, cell_subset=sub)
        assert np.log(strong.batch.beta).std() > np.log(weak.batch.beta).std()

    def test_to_anndata_structure(self, labelled_cell_data):
        from iscc.data.rna import scRNA
        a = scRNA(protocol="10x", n_cells=20).run(labelled_cell_data)
        adata = a.to_anndata()
        assert adata.shape == (20, N_GENES)
        assert "batch" in adata.obs.columns
        assert adata.uns["protocol"] == "10x"
        assert "spatial" in adata.obsm
        assert adata.uns["hyperparams"]["sigma_batch"] == a.hypers.sigma_batch

    def test_write_creates_csv_and_h5ad(self, labelled_cell_data, tmp_path):
        from iscc.data.rna import scRNA
        a = scRNA(protocol="10x", n_cells=15, batch_label="run1").run(labelled_cell_data)
        a.write(str(tmp_path))
        assert (tmp_path / "umis.csv").exists()
        assert (tmp_path / "run1.h5ad").exists()


# --------------------------------------------------------------------------------------
# Plate / well structure (plate protocols: Smart-seq3)
# --------------------------------------------------------------------------------------
class TestPlateLayout:
    def test_well_grid_matches_standard_plate_formats(self):
        from iscc.data.batch import well_grid
        assert [well_grid(n) for n in (6, 12, 24, 48, 96, 384, 1536)] == [
            (2, 3), (3, 4), (4, 6), (6, 8), (8, 12), (16, 24), (32, 48)]

    def test_non_standard_well_count_still_fits(self):
        from iscc.data.batch import well_grid
        rows, cols = well_grid(10)
        assert rows * cols >= 10

    def test_row_labels(self):
        from iscc.data.batch import row_label
        assert [row_label(i) for i in (0, 15, 25, 26, 31)] == ["A", "P", "Z", "AA", "AF"]

    def test_cells_split_evenly_one_per_well(self):
        from iscc.data.batch import assign_plates
        lay = assign_plates(1000, n_wells=384, n_plates=4)
        assert lay.n_plates == 4 and lay.n_rows == 16 and lay.n_cols == 24
        sizes = np.bincount(lay.plate, minlength=4)
        assert sizes.tolist() == [250, 250, 250, 250]
        # one cell per well: every (plate, well) pair is distinct
        assert len(set(zip(lay.plate.tolist(), lay.well.tolist()))) == 1000
        # wells are filled in reading order within each plate
        assert lay.well[:3].tolist() == [0, 1, 2]
        assert lay.row[:3].tolist() == [0, 0, 0] and lay.col[:3].tolist() == [0, 1, 2]
        assert lay.well_ids()[:3].tolist() == ["A01", "A02", "A03"]

    def test_plate_count_grows_when_cells_do_not_fit(self):
        """Capacity rule: more cells than wells grows the plate count (and says so)."""
        from iscc.data.batch import assign_plates
        with pytest.warns(UserWarning, match="growing to 6 plates"):
            lay = assign_plates(500, n_wells=96, n_plates=2)
        assert lay.n_plates == 6
        assert np.bincount(lay.plate).max() <= 96          # never more cells than wells


class TestPlatesInAssay:
    def test_smartseq3_surfaces_plate_and_well_in_obs(self, labelled_cell_data):
        from iscc.data.rna import scRNA
        a = scRNA(protocol="smartseq3", n_cells=60, n_plates=3, batch_label="B0",
                  seed=1).run(labelled_cell_data)
        for col in ["plate", "plate_index", "well", "well_row", "well_col"]:
            assert col in a.obs.columns
        assert sorted(set(a.obs["plate"])) == ["B0_P0", "B0_P1", "B0_P2"]
        assert (a.obs["plate_index"].values == [0] * 20 + [1] * 20 + [2] * 20).all()
        assert (a.obs["well_row"] < 16).all() and (a.obs["well_col"] < 24).all()
        assert len(set(zip(a.obs["plate"], a.obs["well"]))) == 60
        # and the same columns come out of to_anndata
        adata = a.to_anndata()
        assert {"plate", "well", "well_row", "well_col"} <= set(adata.obs.columns)
        assert adata.uns["plate_layout"] == dict(n_plates=3, n_wells=384, n_rows=16, n_cols=24)

    def test_10x_has_no_plate_structure(self, labelled_cell_data):
        """Plates are a plate-protocol concept: the droplet path is untouched."""
        from iscc.data.rna import scRNA
        a = scRNA(protocol="10x", n_cells=30, seed=1).run(labelled_cell_data)
        assert a.plates is None
        assert not any(c.startswith(("plate", "well")) for c in a.obs.columns)
        assert "plate_layout" not in a.to_anndata().uns
        assert a.hypers.n_wells == 0 and a.hypers.plate_sigma == 0.0

    def test_capacity_warning_from_the_assay(self):
        from iscc.data.rna import scRNA
        with pytest.warns(UserWarning, match="do not fit"):
            a = scRNA(protocol="smartseq3", n_cells=300, n_wells=96, n_plates=2,
                      seed=1).run(_plain_cell_data(300))
        assert a.plates.n_plates == 4          # grown from the requested 2

    def test_plate_columns_survive_h5ad_roundtrip(self, labelled_cell_data, tmp_path):
        import anndata as ad
        from iscc.data.rna import scRNA
        a = scRNA(protocol="smartseq3", n_cells=40, n_plates=2, batch_label="run1",
                  seed=1).run(labelled_cell_data)
        a.write(str(tmp_path))
        back = ad.read_h5ad(str(tmp_path / "run1.h5ad"))
        assert (back.obs["plate"].astype(str).values == a.obs["plate"].values).all()
        assert (back.obs["well"].astype(str).values == a.obs["well"].values).all()
        assert int(back.uns["plate_layout"]["n_wells"]) == 384

    def test_negative_plate_params_rejected(self):
        from iscc.data.rna import scRNA
        with pytest.raises(ValueError, match="non-negative"):
            scRNA(protocol="smartseq3", n_plates=-1)


class TestPlateNestedDepth:
    """The scientific content: cells sharing a plate share a depth offset."""

    CELLS, PLATES = 1200, 20

    def _icc(self, plate_sigma, seed=3):
        from iscc.data.rna import scRNA
        a = scRNA(protocol="smartseq3", n_cells=self.CELLS, n_plates=self.PLATES,
                  plate_sigma=plate_sigma, seed=seed).run(_plain_cell_data(self.CELLS))
        return _icc_by_group(a.obs["n_counts"].values, a.obs["plate"].values)

    def test_same_plate_cells_have_correlated_library_sizes(self):
        icc, var_between, var_within = self._icc(0.30)
        assert icc > 0.15                       # plate explains a real share of the depth
        assert var_between > 0.05               # ~ plate_sigma^2 = 0.09
        assert var_within > 0.15                # cell-to-cell spread is still the bulk of it

    def test_no_plate_correlation_when_plate_sigma_is_zero(self):
        icc, var_between, _ = self._icc(0.0)
        assert abs(icc) < 0.02                  # plate membership says nothing about depth
        assert var_between < 0.01

    def test_icc_grows_with_plate_sigma(self):
        assert self._icc(0.05)[0] < self._icc(0.15)[0] < self._icc(0.30)[0]

    def test_plate_factor_is_shared_by_every_cell_on_the_plate(self):
        from iscc.data.rna import scRNA
        a = scRNA(protocol="smartseq3", n_cells=200, n_plates=4, plate_sigma=0.2,
                  seed=5).run(_plain_cell_data(200))
        assert a.batch.plate_factor.shape == (4,)
        assert (a.batch.plate_factor > 0).all()

    def test_plate_sigma_zero_makes_the_layout_inert(self):
        """With no plate effect the counts do not depend on the plate/well arrangement."""
        from iscc.data.rna import scRNA
        cd = _plain_cell_data(600)
        base = scRNA(protocol="smartseq3", n_cells=600, seed=3).run(cd).observed_counts.values
        for kwargs in (dict(n_plates=2), dict(n_plates=9), dict(n_wells=96)):
            alt = scRNA(protocol="smartseq3", n_cells=600, plate_sigma=0.0, seed=3,
                        **kwargs).run(cd).observed_counts.values
            assert np.array_equal(base, alt)

    def test_default_smartseq3_counts_unchanged_by_the_plate_model(self):
        """Regression guard: the presets reproduce the pre-plate-model output exactly."""
        from iscc.data.rna import scRNA
        cd = _plain_cell_data(600)
        ss3 = scRNA(protocol="smartseq3", n_cells=600, seed=1).run(cd).observed_counts.values
        tenx = scRNA(protocol="10x", n_cells=600, seed=1).run(cd).observed_counts.values
        assert (_counts_hash(ss3), int(ss3.sum())) == (
            "655f81aa1744a65441f209212283a2c7b1b5fc4cca6dc7e81731aa1fd8899e15", 4455795)
        assert (_counts_hash(tenx), int(tenx.sum())) == (
            "ca7778be26c8af4aed58c6d2333304e836c7faa3311613b70e80efa48dab202c", 2415897)


class TestCallerPlateLayout:
    """`plate_layout`: the caller decides which plate (and well) each cell is sorted into."""

    N = 600

    def _regions(self):
        cd = _plain_cell_data(self.N)
        cells = list(cd["cell_exp"].index)
        region = {c: "A" if i < self.N // 2 else "B" for i, c in enumerate(cells)}
        return cd, cells, region

    def _region_and_plate_gaps(self, layout, seeds=range(20)):
        """Per seed: the A - B gap in mean log depth, and the realized log plate-offset ratio."""
        from iscc.data.rna import scRNA
        cd, _, region = self._regions()
        gaps, offsets = [], []
        for seed in seeds:
            a = scRNA(protocol="smartseq3", n_cells=self.N, plate_sigma=0.15,
                      plate_layout=layout, seed=seed).run(cd)
            y = np.log(a.obs["n_counts"].values)
            in_a = np.array([region[c] == "A" for c in a.obs.index])
            on_p1 = a.obs["plate"].values == f"batch{seed}_P1"
            gaps.append((y[in_a].mean() - y[~in_a].mean(), y[on_p1].mean() - y[~on_p1].mean()))
            offsets.append(np.log(a.batch.plate_factor[0] / a.batch.plate_factor[1]))
        gaps = np.array(gaps)
        return gaps[:, 0], gaps[:, 1], np.array(offsets)

    def test_region_confounded_layout_puts_the_plate_effect_on_the_region(self):
        _, cells, region = self._regions()
        confounded = {c: 1 if region[c] == "A" else 2 for c in cells}
        region_gap, _, offset = self._region_and_plate_gaps(confounded)
        assert offset.std() > 0.1                          # the plates really differ in depth
        # ... and the region gap IS the plate offset, seed by seed (cell noise SE ~0.04)
        assert np.corrcoef(region_gap, offset)[0, 1] > 0.9
        assert np.abs(region_gap - offset).max() < 0.15

    def test_balanced_layout_keeps_the_plate_effect_off_the_region(self):
        _, cells, region = self._regions()
        balanced = {c: 1 + i % 2 for i, c in enumerate(cells)}   # each region half on each
        region_gap, plate_gap, offset = self._region_and_plate_gaps(balanced)
        assert offset.std() > 0.1
        assert np.corrcoef(plate_gap, offset)[0, 1] > 0.9      # the plate effect is still there
        assert np.abs(region_gap).max() < 0.15                  # but the regions do not see it
        assert region_gap.std() < 0.3 * offset.std()

    def test_a_map_of_the_default_layout_reproduces_the_default_counts(self):
        """The map only replaces WHERE cells sit: same placement, same counts."""
        from iscc.data.rna import scRNA
        cd = _plain_cell_data(self.N)
        kw = dict(protocol="smartseq3", n_cells=self.N, plate_sigma=0.15, seed=4)
        default = scRNA(**kw).run(cd)
        layout = dict(zip(default.obs.index, zip(default.obs["plate_index"], default.obs["well"])))
        mapped = scRNA(plate_layout=layout, **kw).run(cd)
        assert np.array_equal(default.observed_counts.values, mapped.observed_counts.values)
        assert (default.obs["plate"] == mapped.obs["plate"]).all()
        assert (default.obs["well"] == mapped.obs["well"]).all()

    def test_pairs_put_each_cell_in_its_well(self, labelled_cell_data):
        from iscc.data.rna import scRNA
        cells = list(labelled_cell_data["cell_exp"].index)
        layout = {c: ("plateA" if i % 2 else "plateB", i // 2) for i, c in enumerate(cells)}
        layout[cells[0]] = ("plateB", "P24")                    # a well name, the last well
        a = scRNA(protocol="smartseq3", n_cells=60, plate_layout=layout, batch_label="run",
                  seed=1).run(labelled_cell_data)
        assert a.obs.loc[cells[0], "plate"] == "run_plateB"
        assert a.obs.loc[cells[0], "well"] == "P24"
        assert a.obs.loc[cells[3], "plate"] == "run_plateA"
        assert a.obs.loc[cells[3], "well"] == "A02"             # index 1
        assert a.obs.loc[cells[3], "plate_index"] == 0          # labels index in sorted order
        assert a.to_anndata().uns["plate_layout"]["plate_labels"] == ["plateA", "plateB"]

    def test_labels_fill_wells_in_assay_order(self, labelled_cell_data):
        from iscc.data.rna import scRNA
        cells = list(labelled_cell_data["cell_exp"].index)
        a = scRNA(protocol="smartseq3", n_cells=60,
                  plate_layout={c: i % 3 for i, c in enumerate(cells)},
                  batch_label="B", seed=1).run(labelled_cell_data)
        assert sorted(set(a.obs["plate"])) == ["B_P0", "B_P1", "B_P2"]
        for p in range(3):
            on = a.obs[a.obs["plate_index"] == p]
            assert list(on["well"]) == [f"A{k + 1:02d}" for k in range(20)]

    def test_split_batches_each_sort_their_own_cells_by_the_map(self, labelled_cell_data):
        from iscc.data.rna import run_scrna_batches, concat_batches
        cells = list(labelled_cell_data["cell_exp"].index)
        layout = {c: 1 if i < 30 else 2 for i, c in enumerate(cells)}
        assays = run_scrna_batches(labelled_cell_data, n_batches=2, base_seed=12, design="split",
                                   protocol="smartseq3", n_cells=30, plate_layout=layout,
                                   plate_sigma=0.2)
        for b, a in enumerate(assays):
            expect = [f"batch{b}_P{layout[c]}" for c in a.obs.index]
            assert list(a.obs["plate"]) == expect
            assert a.plates.n_plates == 2                       # plate indices are run-wide
        combined = concat_batches(assays)
        assert len(set(zip(combined.obs["plate"].astype(str),
                           combined.obs["well"].astype(str)))) == combined.n_obs

    def test_custom_layout_survives_h5ad_roundtrip(self, labelled_cell_data, tmp_path):
        import anndata as ad
        from iscc.data.rna import scRNA
        cells = list(labelled_cell_data["cell_exp"].index)
        a = scRNA(protocol="smartseq3", n_cells=60, batch_label="run1",
                  plate_layout={c: "left" if i < 25 else "right" for i, c in enumerate(cells)},
                  seed=1).run(labelled_cell_data)
        a.write(str(tmp_path))
        back = ad.read_h5ad(str(tmp_path / "run1.h5ad"))
        assert (back.obs["plate"].astype(str).values == a.obs["plate"].values).all()
        assert list(back.uns["plate_layout"]["plate_labels"]) == ["left", "right"]

    @pytest.mark.parametrize("layout,match", [
        (lambda cells: {c: 1 for c in cells[:-1]}, "no plate for 1"),
        (lambda cells: {c: (1, 0) for c in cells}, "share plate P1 well A01"),
        (lambda cells: {c: (1, 384 + i) for i, c in enumerate(cells)}, "not on a 384-well"),
        (lambda cells: {c: (1, "Q01" if i == 0 else i) for i, c in enumerate(cells)},
         "not on a 384-well"),
        (lambda cells: {c: (1, "A25") for c in cells}, "column 25"),
        (lambda cells: {c: (1, "7A") for c in cells}, "neither a 0-based index"),
        (lambda cells: {c: (1, i) if i else 1 for i, c in enumerate(cells)}, "mixes"),
        (lambda cells: {c: None for c in cells}, "missing plate label"),
        (lambda cells: {c: (1, i) if i < 30 else ("P1", i) for i, c in enumerate(cells)},
         "collide"),
    ])
    def test_bad_layouts_are_rejected(self, labelled_cell_data, layout, match):
        from iscc.data.rna import scRNA
        cells = list(labelled_cell_data["cell_exp"].index)
        with pytest.raises(ValueError, match=match):
            scRNA(protocol="smartseq3", n_cells=60, plate_layout=layout(cells),
                  seed=1).run(labelled_cell_data)

    def test_a_plate_holds_at_most_n_wells_cells(self):
        from iscc.data.rna import scRNA
        cd = _plain_cell_data(200)
        cells = list(cd["cell_exp"].index)
        layout = {c: "big" if i < 120 else "small" for i, c in enumerate(cells)}
        with pytest.raises(ValueError, match="plate big holds 120 .* 96 wells"):
            scRNA(protocol="smartseq3", n_cells=200, n_wells=96, plate_layout=layout,
                  seed=1).run(cd)
        # the same map fits once the assayed cells do (capacity is about the cells sorted)
        a = scRNA(protocol="smartseq3", n_wells=96, plate_layout=layout,
                  seed=1).run(cd, cell_subset=cells[30:150])
        assert np.bincount(a.plates.plate).tolist() == [90, 30]

    def test_layout_needs_plates_and_a_consistent_plate_count(self, labelled_cell_data):
        from iscc.data.rna import scRNA
        cells = list(labelled_cell_data["cell_exp"].index)
        layout = {c: i % 2 for i, c in enumerate(cells)}
        with pytest.raises(ValueError, match="plate protocol"):
            scRNA(protocol="10x", plate_layout=layout)
        with pytest.raises(ValueError, match="n_plates=3 contradicts"):
            scRNA(protocol="smartseq3", n_cells=60, n_plates=3, plate_layout=layout,
                  seed=1).run(labelled_cell_data)
        with pytest.raises(TypeError, match="dict or pandas Series"):
            scRNA(protocol="smartseq3", n_cells=60, plate_layout=[0, 1] * 30,
                  seed=1).run(labelled_cell_data)


# --------------------------------------------------------------------------------------
# Multi-batch generation (DESIGN_features §B.3)
# --------------------------------------------------------------------------------------
class TestMultiBatch:
    def test_shared_design_same_biology_different_technical(self, labelled_cell_data):
        from iscc.data.rna import run_scrna_batches
        assays = run_scrna_batches(labelled_cell_data, n_batches=2, base_seed=10,
                                   design="shared", protocol="10x", n_cells=30)
        a0, a1 = assays
        # same cells measured in both batches (exact correspondence)
        assert list(a0.observed_counts.index) == list(a1.observed_counts.index)
        # same biology -> per-gene mean expression correlated across batches
        m0 = a0.observed_counts.values.mean(axis=0)
        m1 = a1.observed_counts.values.mean(axis=0)
        assert np.corrcoef(m0, m1)[0, 1] > 0.5
        # different technical signature
        assert not np.array_equal(a0.observed_counts.values, a1.observed_counts.values)
        assert set(a0.obs["batch"]) == {"batch0"} and set(a1.obs["batch"]) == {"batch1"}

    def test_split_design_partitions_cells(self, labelled_cell_data):
        from iscc.data.rna import run_scrna_batches
        assays = run_scrna_batches(labelled_cell_data, n_batches=3, base_seed=11,
                                   design="split", protocol="10x", n_cells=15)
        idx = [set(a.observed_counts.index) for a in assays]
        # disjoint across batches (balanced one-tumour design)
        assert idx[0].isdisjoint(idx[1]) and idx[0].isdisjoint(idx[2])

    def test_concat_batches_labelled(self, labelled_cell_data):
        from iscc.data.rna import run_scrna_batches, concat_batches
        assays = run_scrna_batches(labelled_cell_data, n_batches=2, base_seed=12,
                                   design="shared", protocol="smartseq3", n_cells=20)
        combined = concat_batches(assays)
        assert combined.n_obs == 40
        assert set(combined.obs["batch"]) == {"batch0", "batch1"}
        assert combined.uns["n_batches"] == 2

    def test_plate_ids_survive_concatenation(self, labelled_cell_data):
        """Two batches are two plate runs: their plates stay distinct after concat."""
        from iscc.data.rna import run_scrna_batches, concat_batches
        assays = run_scrna_batches(labelled_cell_data, n_batches=2, base_seed=12,
                                   design="split", protocol="smartseq3", n_cells=30,
                                   n_plates=2, plate_sigma=0.2)
        combined = concat_batches(assays)
        for col in ["plate", "well", "well_row", "well_col"]:
            assert col in combined.obs.columns
        assert sorted(set(combined.obs["plate"].astype(str))) == [
            "batch0_P0", "batch0_P1", "batch1_P0", "batch1_P1"]
        # every cell keeps its own well: no id is lost or merged by the concat
        assert len(set(zip(combined.obs["plate"].astype(str),
                           combined.obs["well"].astype(str)))) == combined.n_obs

    def test_unknown_design_raises(self, labelled_cell_data):
        from iscc.data.rna import run_scrna_batches
        with pytest.raises(ValueError):
            run_scrna_batches(labelled_cell_data, design="bogus")
