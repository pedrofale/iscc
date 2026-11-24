"""Genome-wide passengers at infinite sites (``iscc.tumor.models.passengers``).

The gene-level SNVs of the engine repeat across lineages; the passengers of this layer each have a
site of their own, so the carriers of a passenger form one clade, and bulk WGS reads them beside the
gene loci without changing anything it read before.
"""
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "validation"))
import realistic_regime as RR  # noqa: E402

from iscc.data import bulkDNA  # noqa: E402
from iscc.sample import Resection  # noqa: E402
from iscc.tumor.models import passengers  # noqa: E402


def _grow(**cancer):
    return RR.grow_realistic(seed=3, target_cancer=2000, scale="small", coarsen=True,
                             cancer={"mutation_rate": 0.6, "n_snvs_per_allele": 0.1, **cancer},
                             materialise=True, max_cells=3000)


@pytest.fixture(scope="module")
def tumor():
    return _grow(wgd_rate=0.1)


@pytest.fixture(scope="module")
def tumor_no_cn():
    """No copy-number events, so no passenger is ever lost from part of its clade."""
    return _grow(wgd_rate=0.0, cnv_prob=0.0)


def _catalogue(cd):
    lin = cd["cell_lineage"]
    return lin.attrs["passengers"], lin


def test_on_by_default_and_other_frames_unchanged(tumor):
    cd_on = tumor.cell_data
    cat, lin = _catalogue(cd_on)
    assert len(cat) > 0 and len(lin) == len(cd_on["cell_type"])
    keep = tumor.genome_passengers
    try:
        tumor.genome_passengers = passengers.resolve(False)
        cd_off = tumor.make_cell_data(max_cells=3000)
    finally:
        tumor.genome_passengers = keep
        tumor.make_cell_data(max_cells=3000)
    assert "cell_lineage" not in cd_off
    for key in ("cell_snv", "cell_cnv", "cell_type"):
        assert cd_off[key].equals(cd_on[key])


def test_pooled_counts_are_the_sum_over_cells(tumor):
    cat, lin = _catalogue(tumor.cell_data)
    alt, total, carriers, n_cancer = cat.pooled(lin)
    M = cat.cell_vaf(lin)
    ci = np.array([cat._clone_ix[c] for c in lin["lineage_clone"]])
    cn = cat.cn[ci][:, cat.segment]
    assert np.allclose(np.asarray(M.multiply(cn).sum(0)).ravel(), alt)
    assert np.allclose(total, cn.sum(0))
    assert np.array_equal(np.asarray((M > 0).sum(0)).ravel(), carriers)
    assert n_cancer == int(lin["lineage_leaf"].ge(0).sum())


def test_every_passenger_marks_one_clade(tumor_no_cn):
    """Without copy-number losses, any two passengers are carried by nested or disjoint sets of
    cells (no site is mutated twice), and the founder's passengers are in every cancer cell."""
    cat, lin = _catalogue(tumor_no_cn.cell_data)
    cancer = lin["lineage_leaf"].to_numpy() >= 0
    X = (cat.cell_vaf(lin)[cancer] > 0).astype(np.int64).tocsc()
    size = np.asarray(X.sum(0)).ravel()
    pick = np.flatnonzero(size >= 2)
    pick = pick[np.random.default_rng(0).permutation(len(pick))[:600]]
    sub = X[:, pick]
    inter = (sub.T @ sub).toarray()
    s = size[pick]
    smaller = np.minimum.outer(s, s)
    assert np.all((inter == 0) | (inter == smaller))
    truncal = cat.kind == 0
    assert truncal.sum() == passengers.DEFAULT_TRUNCAL
    assert np.all(size[truncal] == cancer.sum())


def test_yule_branches_give_the_neutral_spectrum():
    """A Yule genealogy of n leaves has on average 2n / (k (k + 1)) clades of k leaves, so equal
    passengers per branch give the 1/f spectrum of neutral growth."""
    n, reps = 200, 400
    rng = np.random.default_rng(1)
    hist = np.zeros(n + 1)
    for _ in range(reps):
        a, b = passengers._yule_edges(n, rng.random(n - 1))
        hist += np.bincount(b - a, minlength=n + 1)
    hist /= reps
    assert hist[1] == n
    k = np.arange(2, 11)
    assert np.allclose(hist[k], 2 * n / (k * (k + 1)), rtol=0.1)


def test_bulk_wgs_reads_the_passengers(tumor_no_cn, tmp_path):
    w = bulkDNA(seed=1).run(tumor_no_cn.cell_data)
    p = w.passenger_data
    assert p is not None and len(p) == len(_catalogue(tumor_no_cn.cell_data)[0])
    assert abs(p["coverage"].mean() / w.observed_data["coverage"].mean() - 1) < 0.1
    assert p["true_ccf"].between(0, 1).all()
    assert (p.loc[p["kind"] == "truncal", "true_ccf"] == 1).all()
    # a clonal passenger on a diploid segment sits at about half the purity
    clonal = p.loc[(p["kind"] == "truncal") & np.isclose(p["true_cn"], 2)]
    assert abs(clonal["vaf"].mean() - w.purity / 2) < 0.03
    assert bulkDNA(seed=1, breadth="wes").run(tumor_no_cn.cell_data).passenger_data is None
    w.write(str(tmp_path), write_h5ad=False)
    assert (tmp_path / "passengers.csv").exists()


def test_the_layer_survives_dissociation(tumor):
    spec = Resection(tumor)
    diss = spec.dissociate(spec.cancer_region(margin=1), seed=0)
    cat, lin = _catalogue(diss)
    assert len(lin) == len(diss["cell_type"])
    w = bulkDNA(seed=2).run(diss)
    assert w.passenger_data is not None and w.passenger_data["true_ccf"].between(0, 1).all()


def test_reproducible(tumor):
    cat1, _ = _catalogue(tumor.make_cell_data(max_cells=3000))
    cat2, _ = _catalogue(tumor.make_cell_data(max_cells=3000))
    for attr in ("segment", "homolog", "position", "kind", "car_p", "car_m", "w_p", "w_a", "w_b"):
        assert np.array_equal(getattr(cat1, attr), getattr(cat2, attr))


def test_settings():
    assert passengers.resolve(None) == {"rate": passengers.DEFAULT_RATE, "truncal": passengers.DEFAULT_TRUNCAL}
    assert passengers.resolve(False) is None
    assert passengers.resolve({"rate": 0, "truncal": 0}) is None
    assert passengers.resolve({"rate": 5})["rate"] == 5.0
    with pytest.raises(TypeError):
        passengers.resolve({"mu": 1})
    with pytest.raises(ValueError):
        passengers.resolve({"rate": -1})


def test_human_positions_lie_on_the_arm_of_their_segment():
    from iscc.data.human import HumanGenomeLayer, arm_genome, load_reference
    from iscc.tumor.models import GenotypeTumor
    ref = load_reference(snps=False)
    cfg = arm_genome(genes_per_segment=6, driver_scale=0.1, reference=ref)
    t = GenotypeTumor(seed=3, genome_params=cfg["genome_params"],
                      selection_params={"prop_driver": 0.0, **cfg["selection_params"]},
                      cancer_cell_params={"division_rate": 0.6, "death_rate": 0.03, "mutation_rate": 1.0,
                                          "dispersal_rate": 0.4},
                      deme_params={"carrying_capacity": 10, "initial_cancer_cells": 10},
                      spatial_params={"grid_size": 12, "structure_radius": 3})
    t.grow(n_steps=40, seed=3)
    cat, _ = _catalogue(t.make_cell_data())
    sites = cat.sites()
    pos = HumanGenomeLayer(t, reference=ref).passenger_positions(sites)
    arms = ref.arms.set_index("arm")
    arm = np.asarray(cfg["genome_params"]["arms"])[sites["segment"].to_numpy()]
    assert (pos["chrom"].to_numpy() == arms.loc[arm, "chrom"].to_numpy()).all()
    assert (pos["pos"].to_numpy() >= arms.loc[arm, "start"].to_numpy()).all()
    assert (pos["pos"].to_numpy() < arms.loc[arm, "end"].to_numpy()).all()
