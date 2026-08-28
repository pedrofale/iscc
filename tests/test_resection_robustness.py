"""Robustness of the sampling API (Resection / Biopsy) + Visium placement (review fixes).

- ``Resection(compartment=)`` selects primary / met / both (default primary), so a metastatic
  tumour no longer silently drops its met demes from a sample.
- taking a sample does not REPLACE the tumour's own ``cell_data``: ``dissociate`` / ``slice`` used
  to leave the last cut installed on the tumour, so ``to_anndata(tumor)`` silently exported the
  thin section instead of the tumour-level table.
- a liquid biopsy draws circulating TUMOUR cells: every normal type — host parenchyma included —
  stays out of the pool.
- an empty part (e.g. ``bisect(frac=0)`` or an empty ``region``) placed on a Visium slide renders a
  blank slide instead of crashing on an empty-array reduction.
- ``Visium.section_image`` returns exactly the H&E the assay attaches (same placement/frame).
"""
import numpy as np
import pandas as pd
import pytest

from conftest import GENOME_PARAMS, SELECTION_PARAMS, CANCER_CELL_PARAMS, DEME_PARAMS
from iscc.constants import normal_names
from iscc.tumor.models import GenotypeTumor
from iscc.sample import Resection
from iscc.sample.biopsy.biopsy import Biopsy, NORMAL_TYPES
from iscc.data import Visium

PRIMARY = {"grid_size": 24, "structure_radius": 0}
MET = {"grid_size": 16, "structure_radius": 0, "met_grid_size": 8, "K_met": 16,
       "host_fill_frac": 0.4, "met_seed_kappa": 0.08, "met_hazard": 0.5, "met_transit_floor": 0.03}
CANCER = {**CANCER_CELL_PARAMS, "death_rate": 0.05, "n_snvs_per_allele": 0.1, "cnv_prob": 0.05}


def _tumor(spatial, steps=40, seed=1):
    t = GenotypeTumor(seed=seed, genome_params=GENOME_PARAMS, selection_params=SELECTION_PARAMS,
                      cancer_cell_params=CANCER, deme_params=DEME_PARAMS, spatial_params=spatial,
                      update_mode="tau", tau=0.5, snapshot_every=1, coarsen_passengers=True)
    t.grow(n_steps=steps, seed=seed)
    return t


# --- compartment selection ----------------------------------------------------------------------
def test_compartment_default_is_primary_and_bisect_partitions_it():
    t = _tumor(PRIMARY)
    spec = Resection(t)
    assert spec.compartment == "primary"
    assert spec._all_demes() == list(range(t.grid_size ** 2))
    part, rem = spec.bisect(frac=0.42, axis="x")
    assert len(part) + len(rem) == t.grid_size ** 2
    assert set(part).isdisjoint(rem)


def test_compartment_met_without_metastasis_raises():
    t = _tumor(PRIMARY)
    with pytest.raises(ValueError, match="no metastasis"):
        Resection(t, compartment="met")


def test_compartment_invalid_raises():
    t = _tumor(PRIMARY)
    with pytest.raises(ValueError):
        Resection(t, compartment="bogus")


def test_compartment_both_without_met_is_just_primary():
    t = _tumor(PRIMARY)
    assert Resection(t, compartment="both")._all_demes() == list(range(t.grid_size ** 2))


def test_compartment_selects_the_right_deme_blocks_on_a_met_tumour():
    t = _tumor(MET, steps=20)
    n_prim, m = t.n_primary_demes, t.met_grid_size
    assert n_prim == t.grid_size ** 2 and m == 8
    assert Resection(t, "primary")._all_demes() == list(range(n_prim))
    assert Resection(t, "met")._all_demes() == list(range(n_prim, n_prim + m * m))
    assert Resection(t, "both")._all_demes() == list(range(n_prim + m * m))
    # bisect on the met stays within the met block
    part, rem = Resection(t, "met").bisect(frac=0.5)
    assert all(i >= n_prim for i in part + rem)


# --- a sample does not become the tumour ---------------------------------------------------------
def test_sampling_leaves_the_tumours_own_cell_data_alone():
    """``make_cell_data`` installs what it materialises; a Resection must not let that through.

    Before the fix, ``spec.slice(...)`` left the thin section sitting in ``tumor.cell_data``, so the
    next ``to_anndata(tumor)`` / ``write_h5ad`` exported the section — fewer cells, silently — even
    though the caller never asked for the tumour's table to change.
    """
    t = _tumor(PRIMARY)
    before = t.cell_data
    n_before = len(before["cell_exp"])
    spec = Resection(t)
    cut, remainder = spec.bisect(frac=0.5)

    cd = spec.dissociate(cut, max_cells=20000)
    assert t.cell_data is before and len(t.cell_data["cell_exp"]) == n_before

    section = spec.slice(remainder, depth_frac=0.5, max_cells=20000)
    assert t.cell_data is before and len(t.cell_data["cell_exp"]) == n_before

    # the samples are real pieces of the specimen, not the whole table handed back
    assert 0 < len(cd["cell_exp"]) < n_before
    assert 0 < len(section["cell_exp"]) < n_before


def test_a_counts_only_tumour_stays_counts_only():
    """The cm-scale path: nothing is materialised, and sampling must not change that."""
    t = _tumor(PRIMARY)
    t.cell_data = None
    cd = Resection(t).dissociate(max_cells=5000)
    assert t.cell_data is None
    assert len(cd["cell_exp"]) > 0


def test_install_opts_into_making_the_sample_the_tumours_table():
    t = _tumor(PRIMARY)
    spec = Resection(t)
    cut, _ = spec.bisect(frac=0.5)
    cd = spec.dissociate(cut, max_cells=20000, install=True)
    assert t.cell_data is cd


# --- liquid biopsy: normal cells are not circulating TUMOUR cells --------------------------------
def _liquid_cell_data(n_cancer=6, n_host=200, seed=0):
    """A cell table dominated by host parenchyma, with a handful of cancer cells."""
    rng = np.random.default_rng(seed)
    types = ["7"] * n_cancer + ["host"] * n_host
    names = [f"C{i}" for i in range(len(types))]
    crd = pd.DataFrame({"row": rng.integers(0, 20, len(types)),
                        "col": rng.integers(0, 20, len(types))}, index=names)
    return {"cell_crd": crd,
            "cell_type": pd.DataFrame({"cell_id": types}, index=names),
            "cell_evo": pd.DataFrame({"n_mut_disp": [1] * n_cancer + [0] * n_host},
                                     index=names)}


def test_normal_types_tracks_the_shared_list():
    """A hand-written copy of the normal-type names went stale when host parenchyma was added."""
    assert set(NORMAL_TYPES) == set(normal_names)
    assert "host" in NORMAL_TYPES


def test_liquid_biopsy_pool_excludes_host_parenchyma():
    cd = _liquid_cell_data()
    bx = Biopsy(cd, rng=np.random.default_rng(0))
    chosen, region, geom = bx.sample("liquid", n_liquid=6)
    picked = cd["cell_type"].loc[chosen, "cell_id"].astype(str)
    assert len(picked) == 6
    assert (picked == "7").all(), f"host cells reached the CTC pool: {sorted(set(picked))}"
    assert (region == "blood").all()


# --- empty part does not crash the slide ---------------------------------------------------------
def test_empty_section_renders_blank_slide_without_crashing():
    t = _tumor(PRIMARY)
    empty = Resection(t).slice(region=[], depth_frac=0.5)
    vz = Visium(section_frac=1.0, spot_radius=0.55, seed=0)
    vz.place_grid(empty)                       # must not raise
    ad = vz.to_anndata()
    assert int((ad.obs["in_tissue"] == 1).sum()) == 0
    assert ad.n_obs > 0


# --- section_image preview equals the assayed H&E ------------------------------------------------
def test_section_image_matches_assay_he():
    t = _tumor(PRIMARY)
    spec = Resection(t)
    _, rem = spec.bisect(frac=0.4)
    section = spec.slice(rem, depth_frac=0.5, max_cells=20000)
    ctr = section["cell_crd"][["row", "col"]].to_numpy(float).mean(axis=0)
    vz = Visium(section_frac=1.0, spot_pitch=2.0, spot_radius=0.55,
                placement=(float(ctr[0]), float(ctr[1])), rotation=90, seed=0)
    preview = vz.section_image(section)
    vz.run()                                   # reuses the placement section_image set
    u = vz.to_anndata().uns["spatial"]
    he = u[list(u)[0]]["images"]["hires"]
    assert preview.shape == he.shape and np.allclose(preview, he)
