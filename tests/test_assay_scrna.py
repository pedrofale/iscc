"""Tests for the scRNA assay batch-effect model (DESIGN_features §B / milestone F3)."""
import numpy as np
import pandas as pd
import pytest

from conftest import N_GENES


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

    def test_unknown_design_raises(self, labelled_cell_data):
        from iscc.data.rna import run_scrna_batches
        with pytest.raises(ValueError):
            run_scrna_batches(labelled_cell_data, design="bogus")
