"""F8 microenvironment-driven expression tests (DESIGN_features §H).

Validates the invariants and the mechanism:
  * OFF (no microenv_params) is reproducible, and enabling F8 does NOT change growth — F8
    modulates the expression READOUT only, so the tumour (cell_snv/cnv/type) is byte-identical
    on/off at a seed (the modifier draws from a dedicated rng);
  * control (non-program) genes are exactly unmodified;
  * hypoxia / CCI program genes are multiplied by exactly (1 + strength * field) per deme;
  * the hypoxia field is high in the dense core (correlates with deme density) and spatially
    autocorrelated; the CCI field tracks emitter density;
  * ground truth (programs + per-deme fields + per-cell levels) is surfaced.
"""
import numpy as np
import pytest

from conftest import GENOME_PARAMS, SELECTION_PARAMS, CANCER_CELL_PARAMS, DEME_PARAMS
from iscc.tumor.models import GenotypeTumor
from iscc.data import morans_i

SPATIAL = {"grid_size": 18, "structure_radius": 0}
NO_DEATH = {**CANCER_CELL_PARAMS, "death_rate": 0.0}     # deterministic fill for a density gradient
MP = {
    "hypoxia": {"strength": 0.8, "n_genes": 40, "o2_consumption": 1.5, "o2_supply": 0.3},
    "cci": {"strength": 0.6, "n_target_genes": 40, "emitter_type": "cancer", "lengthscale": 2.5},
}
STEPS = 160


def _grow(microenv_params=None, seed=3):
    # Pin layout_seed to the evolution seed so this fixture reproduces the exact tumour it was
    # calibrated against (the driver/dispersal-gene layout that yields a solid hypoxic-core mass).
    # After the layout/evolution seed decoupling (DESIGN_cohort.md §1) the layout defaults to a shared
    # constant; here we want THIS seed's original layout so the spatial-gradient assertions hold.
    t = GenotypeTumor(seed=seed, layout_seed=seed, genome_params=GENOME_PARAMS,
                      selection_params=SELECTION_PARAMS, cancer_cell_params=NO_DEATH,
                      deme_params=DEME_PARAMS, spatial_params=SPATIAL, microenv_params=microenv_params)
    t.grow(n_steps=STEPS, seed=seed)
    return t


@pytest.fixture(scope="module")
def off():
    return _grow(None)


@pytest.fixture(scope="module")
def on():
    return _grow(MP)


def _ratio(on, off):
    on_e, off_e = on.cell_data["cell_exp"].values, off.cell_data["cell_exp"].values
    return np.divide(on_e, off_e, out=np.full_like(on_e, np.nan), where=off_e > 0)


# --------------------------------------------------------------------- invariants -------------
class TestInvariants:
    def test_off_reproducible(self):
        a, b = _grow(None), _grow(None)
        assert np.array_equal(a.cell_data["cell_exp"].values, b.cell_data["cell_exp"].values)

    def test_growth_unchanged_on_off(self, on, off):
        # F8 touches only the expression readout: the tumour (genome + spatial layout) is identical
        # per cell. (Genotype-id *labels* are a session-global counter, so we compare genome/position,
        # not the arbitrary label.)
        assert on.get_tumor_size() == off.get_tumor_size()
        assert np.array_equal(on.cell_data["cell_snv"].values, off.cell_data["cell_snv"].values)
        assert np.array_equal(on.cell_data["cell_cnv"].values, off.cell_data["cell_cnv"].values)
        assert np.array_equal(on.cell_data["cell_crd"].values, off.cell_data["cell_crd"].values)
        assert np.array_equal(on.cell_data["cell_deme"].values, off.cell_data["cell_deme"].values)

    def test_off_has_no_microenv_key(self, off):
        assert "cell_microenv" not in off.cell_data


# --------------------------------------------------------------------- the modifier -----------
class TestModifier:
    def test_control_genes_unmodified(self, on, off):
        prog = np.union1d(on.microenv_truth["hypoxia_genes"], on.microenv_truth["cci_target_genes"])
        ctrl = np.setdiff1d(np.arange(on.n_genes), prog)
        assert np.array_equal(on.cell_data["cell_exp"].values[:, ctrl],
                              off.cell_data["cell_exp"].values[:, ctrl])

    def test_hypoxia_ratio_is_exact(self, on, off):
        r = _ratio(on, off)
        hyp_only = np.setdiff1d(on.microenv_truth["hypoxia_genes"], on.microenv_truth["cci_target_genes"])
        expected = (1.0 + MP["hypoxia"]["strength"] * on.cell_data["cell_microenv"]["hypoxia_level"].values)
        sub = r[:, hyp_only]
        m = ~np.isnan(sub)
        assert np.allclose(sub[m], np.broadcast_to(expected[:, None], sub.shape)[m], atol=1e-9)

    def test_cci_ratio_is_exact(self, on, off):
        r = _ratio(on, off)
        cci_only = np.setdiff1d(on.microenv_truth["cci_target_genes"], on.microenv_truth["hypoxia_genes"])
        expected = (1.0 + MP["cci"]["strength"] * on.cell_data["cell_microenv"]["cci_level"].values)
        sub = r[:, cci_only]
        m = ~np.isnan(sub)
        assert np.allclose(sub[m], np.broadcast_to(expected[:, None], sub.shape)[m], atol=1e-9)


# --------------------------------------------------------------------- the fields -------------
class TestFields:
    def _occupied(self, on):
        dens = on._deme_density()
        occ = np.where(dens > 0)[0]
        coords = np.array([on.deme_coords[d] for d in occ], dtype=float)
        return occ, dens, coords

    def test_hypoxia_high_in_dense_core(self, on):
        occ, dens, _ = self._occupied(on)
        hyp = on.microenv_truth["hypoxia"]
        # hypoxia rises with local density (core more hypoxic than rim)
        assert np.corrcoef(dens[occ], hyp[occ])[0, 1] > 0.3

    def test_hypoxia_field_spatially_autocorrelated(self, on):
        occ, _, coords = self._occupied(on)
        assert morans_i(on.microenv_truth["hypoxia"][occ], coords) > 0.1

    def test_cci_tracks_emitter_density(self, on):
        occ, _, _ = self._occupied(on)
        emit = on._emitter_density("cancer")
        assert np.corrcoef(emit[occ], on.microenv_truth["cci"][occ])[0, 1] > 0.5

    def test_hypoxia_field_in_unit_range(self, on):
        h = on.microenv_truth["hypoxia"]
        assert (h >= 0).all() and (h <= 1).all()

    def test_perfused_source_more_hypoxic_than_uniform(self, on):
        # "perfused": O2 supplied only by non-cancer tissue (supply <= uniform everywhere), so a
        # solid cancer mass is MORE hypoxic in its interior -> the DCIS comedonecrosis mechanism.
        hu = on._o2_field(source="uniform")
        hp = on._o2_field(source="perfused")
        occ = on._deme_density() > 0
        assert (hp >= hu - 1e-6).all()              # monotone: never less hypoxic than uniform
        assert hp[occ].mean() > hu[occ].mean()      # strictly more hypoxic within the tumour
        assert (hp >= 0).all() and (hp <= 1).all()


# --------------------------------------------------------------------- ground truth -----------
class TestGroundTruth:
    def test_programs_and_levels_surfaced(self, on):
        t = on.microenv_truth
        assert len(t["hypoxia_genes"]) == MP["hypoxia"]["n_genes"]
        assert len(t["cci_target_genes"]) == MP["cci"]["n_target_genes"]
        cm = on.cell_data["cell_microenv"]
        assert {"hypoxia_level", "cci_level"} <= set(cm.columns)
        assert len(cm) == on.get_tumor_size()

    def test_disabled_component_has_no_effect(self):
        # only hypoxia enabled -> CCI genes unmodified vs a fully-off run
        only_hyp = _grow({"hypoxia": MP["hypoxia"]})
        base = _grow(None)
        r = _ratio(only_hyp, base)
        cci_genes = only_hyp.microenv_truth["cci_target_genes"]
        hyp_genes = only_hyp.microenv_truth["hypoxia_genes"]
        cci_only = np.setdiff1d(cci_genes, hyp_genes)
        sub = r[:, cci_only]
        assert np.allclose(sub[~np.isnan(sub)], 1.0, atol=1e-9)      # CCI off -> no change
