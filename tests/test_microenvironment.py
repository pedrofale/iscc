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
# The conftest genome has 40 genes. Keep the hypoxia and CCI target sets SMALL and (mostly) disjoint
# so there is a non-empty pool left for the L-R database — which also makes the control / hyp_only /
# cci_only gene sets below non-empty, so the ratio and control tests actually bite (with a 40-gene
# hypoxia + 40-gene CCI set they covered every gene and were vacuous).
MP = {
    "hypoxia": {"strength": 0.8, "n_genes": 10, "o2_consumption": 1.5, "o2_supply": 0.3},
    # `n_candidate_pairs` is the ONE new W0 parameter: a database of 8 L-R pairs, of which row 0 is the
    # wired channel and the other 7 are unwired decoys. The receptor-dependence (W3) is driven by the
    # wired pair's ligand/receptor genes.
    "cci": {"strength": 0.6, "n_target_genes": 10, "emitter_type": "cancer", "lengthscale": 2.5,
            "n_candidate_pairs": 8},
}
# Real per-deme crowding (DESIGN_crowding.md) caps demes near K and makes a NO_DEATH tumour spread
# by dispersal rather than pile up, so it needs more steps to occupy enough demes for the hypoxia
# FIELD to develop a stable spatial gradient (core–rim contrast + spatial autocorrelation).
STEPS = 400


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
        # The modulated set is hypoxia genes + CCI target genes + the WIRED pair's ligand/receptor
        # (which the field up-regulates so the channel is visible to an L-R tool).
        t = on.microenv_truth
        prog = np.union1d(t["hypoxia_genes"], t["cci_target_genes"])
        prog = np.union1d(prog, [t["cci_ligand"], t["cci_receptor"]])
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
        t = on.microenv_truth
        cci_only = np.setdiff1d(t["cci_target_genes"], t["hypoxia_genes"])
        # the wired L/R carry the field boost too, so they are not pure target genes
        cci_only = np.setdiff1d(cci_only, [t["cci_ligand"], t["cci_receptor"]])
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


# ------------------------------------------------------------- W3 receptor-dependence ----------
class TestReceptorDependence:
    """W3 (DESIGN_cci_spatial.md): the CCI effect is `strength · ligand_avail[deme] · receptor[cell]`.

    So the per-cell received signal (`cci_level`) is the ligand availability at the cell's deme times
    the cell's OWN receptor level — it decomposes exactly that way, and it varies cell-to-cell by
    clone WITHIN a deme (the per-cell heterogeneity that makes W2 unnecessary at Visium)."""

    def test_cci_level_decomposes_into_ligand_avail_times_receptor(self, on):
        t = on.microenv_truth
        lig = t["cci"]                                    # per-deme ligand availability
        rl = t["cci_receptor_level"]                      # gid -> normalised receptor level
        cm = on.cell_data["cell_microenv"]
        dcol = on.cell_data["cell_deme"]["deme_id"].values
        gids = on.cell_data["cell_type"]["cell_id"].values
        expected = lig[dcol] * np.array([rl.get(g, 0.0) for g in gids])
        assert np.allclose(cm["cci_level"].values, expected, atol=1e-9)

    def test_received_signal_varies_within_a_deme(self, on):
        # WITHIN a single deme, cells of different clones get DIFFERENT received signal (the receptor
        # term) — the deme-constant field alone could not produce this. Find a multi-clone deme.
        df = on.cell_data
        dcol = df["cell_deme"]["deme_id"].values
        gids = df["cell_type"]["cell_id"].values
        lvl = df["cell_microenv"]["cci_level"].values
        import collections
        by_deme = collections.defaultdict(set)
        spread_found = False
        for d, g, v in zip(dcol, gids, lvl):
            by_deme[d].add(g)
        for d, clones in by_deme.items():
            if len(clones) >= 2:
                vals = lvl[dcol == d]
                if vals.max() - vals.min() > 1e-9:
                    spread_found = True
                    break
        assert spread_found, "expected within-deme variation in the received signal across clones"

    def test_receptor_level_averages_about_one(self, on):
        # NORMALISATION: receptor level is divided by the population mean receptor expression, so the
        # count-weighted mean over all materialised cells is ~1 (keeps `strength` calibrated).
        rl = on.microenv_truth["cci_receptor_level"]
        gids = on.cell_data["cell_type"]["cell_id"].values
        vals = np.array([rl.get(g, 0.0) for g in gids])
        assert 0.5 < vals.mean() < 1.5


# ------------------------------------------------------------- W0 the L-R database --------------
class TestDatabase:
    """W0 (DESIGN_cci_spatial.md): iscc emits its own candidate ligand-receptor database over its own
    abstract gene ids. One wired pair, the rest unwired decoys; the whitelist is complete."""

    def test_database_has_n_candidate_pairs(self, on):
        pairs = on.microenv_truth["cci_pairs"]
        assert pairs.shape == (MP["cci"]["n_candidate_pairs"], 2)

    def test_wired_pair_is_row_zero_and_matches_channel(self, on):
        t = on.microenv_truth
        assert t["cci_wired_pair"] == 0
        assert int(t["cci_pairs"][0, 0]) == t["cci_ligand"]
        assert int(t["cci_pairs"][0, 1]) == t["cci_receptor"]

    def test_candidates_are_not_hypoxia_genes(self, on):
        # candidates must not be moved by the unrelated hypoxia programme (that would boost a DECOY
        # and blur the benchmark). Overlap with the CCI target set is allowed and harmless.
        t = on.microenv_truth
        genes = set(t["cci_pairs"].ravel().tolist())
        assert genes.isdisjoint(set(np.asarray(t["hypoxia_genes"]).tolist()))

    def test_wired_ligand_and_receptor_carry_the_signal(self, on, off):
        # THE POINT OF THE FEATURE: an L-R tool scores a pair from the LIGAND's and RECEPTOR's own
        # expression, so the channel has to mark those two genes or it is invisible. The ligand marks
        # the SENDER type and the receptor the RECEIVER population, both by a flat 1 + strength — a
        # between-group contrast, which is what these tools' differential-expression filters read.
        t = on.microenv_truth
        r = _ratio(on, off)
        # `cell_type` holds the GENOTYPE ID (cancer cells appear under their clone id, not the
        # string "cancer"), so map it through the genotypes to get the cell type.
        gids = on.cell_data["cell_type"].iloc[:, 0].values
        is_emitter = np.array([on.genotypes[g].type == t["cci_emitter_type"] for g in gids])
        f = 1.0 + MP["cci"]["strength"]
        targets = set(np.asarray(t["cci_target_genes"]).tolist())
        for g, boosted in ((t["cci_ligand"], is_emitter), (t["cci_receptor"], ~is_emitter)):
            if g in targets:
                continue                                    # also a target -> carries both factors
            sub = r[:, g]
            m = ~np.isnan(sub)
            # This fixture's tumour is pure cancer, so with emitter_type="cancer" only the SENDER
            # side exists; guard each side so the test asserts what the fixture actually exercises.
            # The receiver side is covered end-to-end by validation/validate_cci.py.
            if (m & boosted).any():
                assert np.allclose(sub[m & boosted], f, atol=1e-9)
            if (m & ~boosted).any():
                assert np.allclose(sub[m & ~boosted], 1.0, atol=1e-9)
        assert is_emitter.any()                              # the sender side is exercised

    def test_decoy_pairs_are_unmodified(self, on, off):
        # an unwired decoy gets NO boost — that is what makes the wired pair distinguishable
        t = on.microenv_truth
        r = _ratio(on, off)
        prog = np.union1d(t["hypoxia_genes"], t["cci_target_genes"])
        decoys = [g for g in t["cci_pairs"][1:].ravel().tolist() if g not in set(prog.tolist())]
        assert decoys, "expected at least one decoy gene outside the modulated sets"
        sub = r[:, decoys]
        assert np.allclose(sub[~np.isnan(sub)], 1.0, atol=1e-9)

    def test_pairs_are_strict_1to1_distinct_genes(self, on):
        genes = on.microenv_truth["cci_pairs"].ravel()
        assert len(set(genes.tolist())) == len(genes)      # every ligand/receptor is a distinct gene

    def test_cellchat_database_shape_and_whitelist(self, on):
        from iscc.integrations import cci_database, referenced_genes
        db = cci_database(on)
        inter = db["interaction"]
        assert len(inter) == MP["cci"]["n_candidate_pairs"]
        assert set(inter["annotation"]) == {"Secreted Signaling"}
        assert int(inter["wired"].sum()) == 1               # exactly one active pair
        # geneInfo is the whitelist and MUST cover every referenced gene (CellChat §4.1)
        assert referenced_genes(db) <= set(db["geneInfo"]["Symbol"])

    def test_write_database_round_trips(self, on, tmp_path):
        from iscc.integrations import write_cci_database
        paths, expected = write_cci_database(on, str(tmp_path))
        import pandas as pd
        gi = pd.read_csv(paths["geneInfo"])
        assert expected <= set(gi["Symbol"].astype(str))
        # complex/cofactor written with >=2 columns (never 1 — CellChat §8.1)
        cx = pd.read_csv(paths["complex"])
        assert cx.shape[1] >= 2 and len(cx) == 0

    def test_clone_correlation_reports_all_pairs(self, on):
        from iscc.integrations import clone_correlation
        cc = clone_correlation(on.cell_data, on.microenv_truth["cci_pairs"])
        assert len(cc) == MP["cci"]["n_candidate_pairs"]
        assert (cc["eta_ligand"].between(0, 1)).all() and (cc["eta_receptor"].between(0, 1)).all()
