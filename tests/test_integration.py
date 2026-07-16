"""Multi-modal integration-benchmark tests: clonealign (DNA<->RNA) + inferCNV (CNA-from-expression).

Two layers:
  * the iscc-side data generation and scoring (``validation/integration_common.py``) are ALWAYS
    exercised — a small deterministic tumour must yield clones with distinct segmental CNAs, well-
    formed tool inputs, and scoring helpers that behave;
  * the REAL external tools run only when their dedicated conda env is present (``iscc-clonealign`` /
    ``iscc-infercnv``); otherwise the test skips, so the core ``iscc`` suite stays green without the
    heavy R+TensorFlow / infercnvpy stacks.

The point mirrors the PEtracer benchmark (test_petracer.py): iscc supplies a *non-circular* ground
truth — the copy-number -> expression coupling both methods exploit EMERGES from the engine's dosage
model, it is not imposed to match the method.
"""
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "validation"))
import integration_common as C  # noqa: E402

# A small, fast, deterministic tumour: a few segments so distinct CNAs accumulate, a normal
# compartment (structure_radius>0) for the inferCNV reference, grown just long enough for subclones.
GENOME = {"n_segments": 6, "segment_size": 40}
# Real per-deme crowding (DESIGN_crowding.md) caps demes near K, so the cancer spreads across the
# gland rather than piling up; grow a larger gland (bigger grid/duct + K + steps) so each clone holds
# enough cells for scDNA to recover its consensus copy number cleanly.
SPATIAL = {"grid_size": 18, "structure_radius": 5}
DEME = {"carrying_capacity": 15, "initial_cancer_cells": 4}
CANCER = {"division_rate": 0.6, "death_rate": 0.03, "max_birth_rate": 0.98,
          "mutation_rate": 1.2, "dispersal_rate": 0.5}


@pytest.fixture(scope="module")
def tumor():
    return C.grow_tumor(seed=0, steps=650, genome=GENOME, spatial=SPATIAL, deme=DEME, cancer=CANCER)


@pytest.fixture(scope="module")
def ca_inputs(tumor):
    return C.build_clonealign_inputs(tumor, n_clones=3, seed=0)


# --------------------------------------------------------------------- data generation --------
class TestDataGeneration:
    def test_clones_have_distinct_cnas(self, ca_inputs):
        cons = ca_inputs["consensus"]
        assert cons.shape[0] >= 2                                  # multiple clones
        # clones differ on >=1 segment, and there is at least one gain and one loss vs diploid
        assert len(C.informative_segments(cons)) >= 1
        assert (cons >= 3).any() or (cons <= 1).any()
        assert len(set(map(tuple, cons))) == cons.shape[0]         # profiles are distinct

    def test_clonealign_inputs_wellformed(self, ca_inputs):
        Y, L = ca_inputs["Y"], ca_inputs["L"]
        assert (Y.values >= 0).all() and Y.shape[0] > 30           # counts, enough cells
        assert list(L.index) == list(Y.columns)                    # genes align (Y cols = L rows)
        assert L.shape[1] == ca_inputs["consensus"].shape[0]       # one column per clone
        assert set(np.unique(ca_inputs["labels"])) <= set(range(L.shape[1]))

    def test_scdna_recovers_clone_cn(self, ca_inputs):
        # the clone CN profiles clonealign consumes are ones the scDNA modality actually supports.
        assert ca_inputs["dna_concordance"] > 0.6

    def test_infercnv_inputs_have_normals_and_coords(self, tumor):
        adata = C.build_infercnv_inputs(tumor, n_normal=80, seed=0)
        assert {"malignant", "normal"} <= set(adata.obs["cell_type"])
        assert (adata.obs["cell_type"] == "normal").sum() > 0
        assert {"chromosome", "start", "end"} <= set(adata.var.columns)
        assert adata.obsm["true_seg_cn"].shape == (adata.n_obs, tumor.n_segments)


# --------------------------------------------------------------------- scoring helpers --------
class TestScoring:
    def test_score_assignment_perfect(self):
        true = np.array([0, 0, 1, 1, 2, 2])
        # probabilities that put all mass on a (relabelled) permutation of the true clone
        probs = pd.DataFrame(np.eye(3)[[2, 2, 0, 0, 1, 1]], columns=["a", "b", "c"])
        sc = C.score_assignment(true, probs)
        assert sc["accuracy"] == pytest.approx(1.0)
        assert sc["ari"] == pytest.approx(1.0)

    def test_score_assignment_chance(self):
        rng = np.random.default_rng(0)
        true = rng.integers(0, 3, size=300)
        probs = pd.DataFrame(rng.dirichlet(np.ones(3), size=300), columns=["a", "b", "c"])
        sc = C.score_assignment(true, probs)
        assert sc["accuracy"] < 0.55                                # near chance, not inflated
        assert 0.4 < sc["mean_auc"] < 0.6


# --------------------------------------------------------------------- real clonealign ---------
@pytest.mark.skipif(not C.clonealign_available(),
                    reason="iscc-clonealign env (R + TensorFlow + clonealign) not installed")
class TestClonealignReal:
    def test_assignment_beats_chance(self, ca_inputs):
        with tempfile.TemporaryDirectory() as work:
            probs = C.run_clonealign(ca_inputs["Y"], ca_inputs["L"], work,
                                     max_iter=120, n_repeats=2)
        sc = C.score_assignment(ca_inputs["labels"], probs)
        assert probs.shape == (ca_inputs["Y"].shape[0], ca_inputs["consensus"].shape[0])
        assert sc["accuracy"] > 1.5 * sc["chance"]                  # meaningfully above chance
        assert sc["mean_auc"] > 0.7                                 # dosage signal recovered


# --------------------------------------------------------------------- real infercnvpy ---------
@pytest.mark.skipif(not C.infercnv_available(),
                    reason="iscc-infercnv env (infercnvpy) not installed")
class TestInferCNVReal:
    def test_recovers_cnas_and_separates_normal(self, tumor):
        adata = C.build_infercnv_inputs(tumor, n_normal=120, seed=0)
        mal = np.asarray(adata.obs["cell_type"] == "malignant")
        labels, _ = C.define_clones(adata.obsm["true_seg_cn"][mal], n_clones=3)
        with tempfile.TemporaryDirectory() as work:
            res = C.run_infercnv(adata, work, window_size=15, step=2)
        sc = C.score_infercnv(res, adata.obsm["true_seg_cn"], clone_labels=labels)
        assert sc["malignant_normal_auc"] > 0.8                     # malignant vs normal separates
        assert sc["clone_level_r"] > 0.5                            # clonal CNA structure recovered
        assert sc["mean_segment_r"] > 0.1                           # single-cell CN signal is real


# ==================================================================================================
# Numbat (allele-aware CNA-from-expression) — demo 2
# --------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def allele_tumor():
    # the same small gland, grown with the R13 allele layer ON so cell_rna_baf exists for Numbat
    return C.grow_tumor_alleles(seed=0, steps=650, genome=GENOME, spatial=SPATIAL, deme=DEME,
                                cancer=dict(CANCER, cnv_prob=0.6, snv_prob=0.4))


@pytest.fixture(scope="module")
def numbat_bundle(allele_tumor):
    shared = C.build_cna_inputs(allele_tumor, n_normal=80, n_clones=3, seed=0)
    work = tempfile.mkdtemp(prefix="numbat_test_")
    numbat_in = C.build_numbat_inputs(allele_tumor, shared, work, seed=0)
    return allele_tumor, shared, numbat_in


class TestNumbatDataGen:
    def test_allele_layer_materialises(self, allele_tumor):
        assert "cell_rna_baf" in allele_tumor.cell_data          # R13 allele layer is on
        assert "cell_exp_p" in allele_tumor.cell_data and "cell_exp_m" in allele_tumor.cell_data

    def test_emergent_allelic_imbalance_tracks_cnas(self, allele_tumor):
        # the signal Numbat exploits: BAF deviates from 0.5 exactly where copy number is altered,
        # and NOT (much) where it is balanced — an emergent, non-imposed consequence of dosage.
        # Use PER-CELL copy number (a gene altered in one clone only is balanced on the cancer average
        # but altered in that clone's cells, so a cell-averaged mask contaminates the balanced set).
        types = C.cell_types(allele_tumor)
        cnv = allele_tumor.cell_data["cell_cnv"].values
        baf = allele_tumor.cell_data["cell_rna_baf"].values
        cancer = types == "cancer"
        cn_c = cnv[cancer]
        baf_c = np.abs(baf[cancer] - 0.5)
        altered = np.abs(cn_c - 2.0) > 0.5                       # per (cell, gene)
        assert altered.sum() > 0
        dev_alt = baf_c[altered].mean()
        dev_bal = baf_c[~altered].mean()
        assert dev_alt > 2 * dev_bal                             # imbalance localises to the CNAs

    def test_numbat_inputs_wellformed(self, numbat_bundle):
        _, shared, numbat_in = numbat_bundle
        wd = numbat_in["work_dir"]
        df = pd.read_csv(os.path.join(wd, "df_allele.csv"))
        gtf = pd.read_csv(os.path.join(wd, "gtf.csv"))
        cmat = pd.read_csv(os.path.join(wd, "count_mat.csv"), index_col=0)
        # df_allele has Numbat's schema, ALT<=DP, GT phased, CHROM in the segment range
        assert {"cell", "snp_id", "CHROM", "POS", "cM", "REF", "ALT", "AD", "DP", "GT", "gene"} \
            <= set(df.columns)
        assert (df["AD"] <= df["DP"]).all() and (df["DP"] > 0).all()
        assert set(df["GT"].unique()) <= {"1|0", "0|1"}
        assert df["CHROM"].min() >= 1 and df["CHROM"].max() <= shared["n_segments"]
        # count_mat genes x cells; gtf maps every gene to a chromosome + length
        assert cmat.shape[1] == len(shared["cell_ids"])
        assert {"gene", "gene_start", "gene_end", "CHROM", "gene_length"} <= set(gtf.columns)
        assert len(gtf) == cmat.shape[0]

    def test_score_numbat_perfect(self, numbat_bundle):
        # an oracle "prediction" (true malignancy, true clone, true CN) must score at the ceiling
        _, shared, numbat_in = numbat_bundle
        cells = list(numbat_in["cell_ids"])
        id_pos = {c: i for i, c in enumerate(cells)}
        mal_pos = {c: i for i, c in enumerate(numbat_in["mal_cells"])}
        clone = {c: (int(numbat_in["clone_labels"][mal_pos[c]]) if c in mal_pos else -1) for c in cells}
        p_aneu = {c: (1.0 if numbat_in["is_malignant"][id_pos[c]] else 0.0) for c in cells}
        seg_cn = np.stack([numbat_in["true_seg_cn"][id_pos[c]] for c in cells]).astype(float)
        res = dict(clone=clone, p_aneuploid=p_aneu, seg_cn=seg_cn, cells=cells)
        sc = C.score_numbat(res, numbat_in)
        assert sc["malignant_normal_auc"] == pytest.approx(1.0)
        assert sc["clone_ari"] == pytest.approx(1.0)
        assert sc["mean_segment_r"] > 0.9


@pytest.mark.skipif(not C.numbat_available(),
                    reason="iscc-numbat env (R + numbat) not installed")
class TestNumbatReal:
    def test_numbat_recovers_cnas(self, numbat_bundle):
        _, shared, numbat_in = numbat_bundle
        res = C.run_numbat(numbat_in, numbat_in["work_dir"], min_cells=15, max_iter=2)
        sc = C.score_numbat(res, numbat_in)
        assert res["seg_cn"].shape[1] == numbat_in["n_segments"]
        # the allele+expression HMM recovers real per-segment copy-number signal
        assert sc["mean_segment_r"] > 0.3
        assert sc["n_malignant"] > 0
