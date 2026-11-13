"""The human genome layer (``iscc.data.human``): real genes, germline SNPs and tool-ready files.

What must hold:

1. **Reference.** The packaged gene table is 10x's GRCh38-2020-A set (36,601 genes) and the arms are
   the 39 modelled autosomal arms.
2. **Arm genome.** ``arm_genome`` gives one segment per arm at any resolution, driver counts from the
   Census that do not change with resolution, and a configuration the engine grows.
3. **Growth is untouched.** ``segment_drivers`` unset leaves the driver layout byte-identical, and the
   assays without ``gene_lift`` are unchanged (covered by the existing assay suites).
4. **Resolution-free map.** The same human genes come out at every resolution; each growth gene covers
   a contiguous stretch of one arm, in genomic order.
5. **Allele counts follow the homologs.** A homolog lost in a genotype contributes no SNP UMIs from
   that arm; a balanced diploid splits evenly.
6. **Writers.** The files have the shapes and conventions CalicoST, Clonalscope and CopyKAT read.
"""
import gzip
import os

import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from iscc.data import Visium, scRNA
from iscc.data.human import (HUMAN_VISIUM_KAPPA, HumanGenomeLayer, arm_genome, check_tool_ready,
                             load_reference, write_scrna, write_visium)
from iscc.tumor.components.selection import Selection
from iscc.tumor.models import GenotypeTumor


@pytest.fixture(scope="module")
def reference():
    """The packaged reference with one synthetic SNP near each modelled gene's 3' end (positions are
    not real; the tests check mechanics, not SNP density)."""
    ref = load_reference(snps=False)
    arm_of = ref.gene_arm()
    rows = np.flatnonzero(arm_of >= 0)
    g = ref.genes.iloc[rows]
    pos = np.where(g["strand"].to_numpy() == "+", g["end"].to_numpy() - 10, g["start"].to_numpy() + 10)
    ref.snps = pd.DataFrame({"chrom": g["chrom"].to_numpy(), "pos": pos, "ref": "A", "alt": "G",
                             "af": 0.5, "gene": rows})
    return ref


def grow(cfg, seed=3, steps=40):
    t = GenotypeTumor(seed=seed, genome_params=cfg["genome_params"],
                      selection_params={"prop_driver": 0.0, **cfg["selection_params"]},
                      cancer_cell_params={"division_rate": 0.6, "death_rate": 0.03, "mutation_rate": 1.0,
                                          "dispersal_rate": 0.4},
                      deme_params={"carrying_capacity": 10, "initial_cancer_cells": 10},
                      spatial_params={"grid_size": 12, "structure_radius": 3})
    t.grow(n_steps=steps, seed=seed)
    return t


@pytest.fixture(scope="module")
def tumour(reference):
    return grow(arm_genome(genes_per_segment=6, driver_scale=0.1, reference=reference))


def test_reference_is_the_10x_gene_set_and_the_39_arms(reference):
    assert len(reference.genes) == 36601
    assert reference.genes["gene_id"].is_unique
    assert len(reference.arms) == 39
    assert set(reference.arms["arm"]).isdisjoint({"13p", "14p", "15p", "21p", "22p"})
    assert (reference.gene_arm() >= 0).sum() == 35061


@pytest.mark.parametrize("kw", [dict(total_genes=600), dict(total_genes=6000, weight="length"),
                                dict(genes_per_segment=25)])
def test_arm_genome_has_one_segment_per_arm(reference, kw):
    cfg = arm_genome(reference=reference, **kw)
    gp, sel = cfg["genome_params"], cfg["selection_params"]
    assert gp["n_segments"] == 39 and gp["arms"] == reference.arms["arm"].tolist()
    assert len(gp["segment_sizes"]) == 39 and min(gp["segment_sizes"]) >= 4
    for (o, t), n in zip(sel["segment_drivers"], gp["segment_sizes"]):
        assert 0 <= o + t <= n


def test_driver_counts_do_not_change_with_resolution(reference):
    lo = arm_genome(total_genes=2000, reference=reference)["selection_params"]["segment_drivers"]
    hi = arm_genome(total_genes=20000, reference=reference)["selection_params"]["segment_drivers"]
    assert lo == hi                                     # uncapped at both resolutions
    assert np.sum(hi, axis=0).tolist() == [297, 293]


def test_segment_drivers_places_exact_counts_and_unset_is_unchanged():
    base = Selection(n_segments=3, segment_sizes=[20, 30, 40], prop_driver=0.2, layout_seed=7)
    again = Selection(n_segments=3, segment_sizes=[20, 30, 40], prop_driver=0.2, layout_seed=7,
                      segment_drivers=None)
    for a, b in zip(base.driver_types, again.driver_types):
        assert np.array_equal(a, b)
    exact = Selection(n_segments=3, segment_sizes=[20, 30, 40], layout_seed=7,
                      segment_drivers=[(2, 1), (0, 3), (5, 5)])
    assert [(int((d == 1).sum()), int((d == -1).sum())) for d in exact.driver_types] == [(2, 1), (0, 3), (5, 5)]
    with pytest.raises(ValueError):
        Selection(n_segments=2, segment_sizes=[3, 3], segment_drivers=[(2, 2), (0, 0)])


@pytest.mark.parametrize("n", [4, 50])
def test_map_is_resolution_free_and_in_genomic_order(reference, n):
    layer = HumanGenomeLayer(genome_params=arm_genome(genes_per_segment=n, reference=reference)["genome_params"],
                             reference=reference, seed=0)
    assert len(layer.genes) == 35061
    for arm, g in layer.genes.groupby("arm", sort=False):
        mids = (g["start"] + g["end"]) // 2
        cov = g["growth_gene"].to_numpy()[np.argsort(mids.to_numpy(), kind="stable")]
        assert np.all(np.diff(cov) >= 0)               # slices in genomic order
        assert len(set(g["segment"])) == 1             # one arm, one segment


def test_a_genome_without_arms_needs_allow_linear_and_warns(reference):
    gp = {"n_segments": 12, "segment_size": 50}
    with pytest.raises(ValueError):
        HumanGenomeLayer(genome_params=gp, reference=reference)
    with pytest.warns(UserWarning):
        layer = HumanGenomeLayer(genome_params=gp, reference=reference, allow_linear=True)
    assert len(layer.genes) == 35061 and layer.cover.max() < 600


def test_growth_is_the_same_with_and_without_the_layer(reference):
    cfg = arm_genome(genes_per_segment=6, driver_scale=0.1, reference=reference)
    a, b = grow(cfg), grow(cfg)
    HumanGenomeLayer(b, reference=reference)
    assert a.get_tumor_size() == b.get_tumor_size()
    assert np.array_equal(a.cell_data["cell_cnv"].values, b.cell_data["cell_cnv"].values)


def test_lost_homolog_gives_no_snp_umis_from_that_arm(reference, tumour):
    layer = HumanGenomeLayer(tumour, reference=reference, snp_coverage=1.0)
    cd = tumour.cell_data
    cells = list(cd["cell_exp"].index)
    gids = layer._gids(cd, cells)
    cn_p, cn_m = layer.homolog_cn(gids, tumour)
    counts = layer.lift(cd["cell_exp"].to_numpy() * 50).round().astype(int)
    frac = layer.hap_fraction(cd, cells, tumour)
    ac = layer.allele_counts(counts, frac, cells, seed=1)
    seg_of_snp = layer.segment[pd.Index(layer.genes["gene_id"]).get_indexer(ac.snps["gene_id"])]
    A, B = ac.A.toarray(), ac.B.toarray()
    lost_p = cn_p[:, seg_of_snp] == 0                  # (cells, snps)
    lost_m = cn_m[:, seg_of_snp] == 0
    assert A[lost_p].sum() == 0 and B[lost_m].sum() == 0
    bal = (cn_p[:, seg_of_snp] == 1) & (cn_m[:, seg_of_snp] == 1)
    share = A[bal].sum() / max(A[bal].sum() + B[bal].sum(), 1)
    assert 0.45 < share < 0.55


def test_assays_count_human_genes_and_writers_produce_tool_inputs(reference, tumour, tmp_path):
    layer = HumanGenomeLayer(tumour, reference=reference)
    cd = tumour.cell_data
    sc = scRNA(n_cells=40, seed=1).run(cd, gene_lift=layer.lift)
    assert sc.observed_counts.shape == (40, 35061)
    assert list(sc.observed_counts.columns[:3]) == layer.genes["gene_id"].tolist()[:3]
    write_scrna(str(tmp_path / "sc"), sc, layer, cd, allele=layer.scrna_allele_counts(sc, cd, tumour),
                tumor=tumour)
    vz = Visium(seed=2, kappa=HUMAN_VISIUM_KAPPA).run(cd, grid_side=12, gene_lift=layer.lift)
    assert vz.spot_counts.shape[1] == 35061
    av = layer.visium_allele_counts(vz, tumour)
    write_visium(str(tmp_path / "vz"), vz, layer, allele=av, tumor=tumour)

    outs = tmp_path / "vz" / "outs"
    n_tissue = int((vz.obs["n_cells"] > 0).sum())
    with h5py.File(outs / "filtered_feature_bc_matrix.h5") as f:
        assert tuple(f["matrix/shape"][()]) == (35061, n_tissue)
        assert f["matrix/features/id"][0].decode() == layer.genes["gene_id"].iat[0]
        assert len(f["matrix/indptr"]) == n_tissue + 1
    pos = pd.read_csv(outs / "spatial" / "tissue_positions.csv")
    assert list(pos.columns) == ["barcode", "in_tissue", "array_row", "array_col",
                                 "pxl_row_in_fullres", "pxl_col_in_fullres"]
    assert len(pos) == len(vz.spot_counts) and pos["in_tissue"].sum() == n_tissue
    assert ((pos["array_row"] + pos["array_col"]) % 2 == 0).all()      # Visium hex parity
    snp = tmp_path / "vz" / "snp_dir"
    A = sp.load_npz(snp / "cell_snp_Aallele.npz")
    ids = np.load(snp / "unique_snp_ids.npy", allow_pickle=True)
    bcs = pd.read_csv(snp / "barcodes.txt", header=None)[0]
    assert A.shape == (len(bcs), len(ids)) == (n_tissue, len(ids))
    assert all(int(s.split("_")[0]) in range(1, 23) for s in ids[:50])
    with gzip.open(tmp_path / "sc" / "outs" / "filtered_feature_bc_matrix" / "features.tsv.gz", "rt") as f:
        first = f.readline().rstrip("\n").split("\t")
    assert first[2] == "Gene Expression" and first[0].startswith("ENSG")
    bed = pd.read_csv(tmp_path / "sc" / "clonalscope" / "genes.bed", sep="\t")
    assert list(bed.columns) == ["chr", "start", "end", "geneID"] and len(bed) == 35061
    cells = pd.read_csv(tmp_path / "sc" / "truth" / "cells.tsv", sep="\t")
    assert len(cells) == 40 and cells["barcode"].is_unique
    seg = pd.read_csv(tmp_path / "vz" / "truth" / "cn_segments.tsv", sep="\t")
    assert set(seg["arm"]) == set(reference.arms["arm"])

    ready = check_tool_ready(sc.observed_counts, layer.genes)
    assert set(ready["tool"]) == {"copykat", "calicost", "clonalscope", "cancerfinder"}
