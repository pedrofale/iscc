"""Robustness of the Resection API + Visium placement (review fixes).

- ``Resection(compartment=)`` selects primary / met / both (default primary), so a metastatic
  tumour no longer silently drops its met demes from a sample.
- an empty part (e.g. ``bisect(frac=0)`` or an empty ``region``) placed on a Visium slide renders a
  blank slide instead of crashing on an empty-array reduction.
- ``Visium.section_image`` returns exactly the H&E the assay attaches (same placement/frame).
"""
import numpy as np
import pytest

from conftest import GENOME_PARAMS, SELECTION_PARAMS, CANCER_CELL_PARAMS, DEME_PARAMS
from iscc.tumor.models import GenotypeTumor
from iscc.sample import Resection
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
