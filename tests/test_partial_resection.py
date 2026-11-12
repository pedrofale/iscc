"""``Surgery(region=..., margin=...)`` — a partial resection with ground-truth margins.

``Surgery(site)`` with no region removes every cell of the compartment, so a resected primary is
cured by construction; that stays the default and must stay byte-identical. With a ``region`` only
those demes are cleared (cancer and normal residents), everything outside keeps growing, and the
tumour records the ground truth of the cut: the cancer cells removed, the cancer cells in the rim
of ``margin`` demes just inside the cut (the true margin status), and the cancer cells left behind,
by their distance from the cut.
"""
import hashlib

import numpy as np
import pytest

from conftest import GENOME_PARAMS, SELECTION_PARAMS, CANCER_CELL_PARAMS
from iscc.tumor.components.cell import Cell
from iscc.tumor.models import GenotypeTumor
from iscc.treatment import Surgery

DEME = {"carrying_capacity": 10, "initial_cancer_cells": 5}
SPATIAL = {"grid_size": 24, "n_structures": 1, "structure_radius": 0}
# The metastasis tests' structured field + deposit: the only setting where a whole-primary
# resection leaves something to keep growing.
MET_DEME = {"carrying_capacity": 16, "initial_cancer_cells": 6, "resident_pressure_ref": 0.2}
MET_SPATIAL = {"grid_size": 16, "structure_radius": 3, "n_glands": 3, "gland_radius": 3,
               "min_gland_sep": 8, "K_duct": 28, "K_stroma": 18, "stroma_fill_frac": 0.3,
               "cross_gland_kappa": 0.05, "epithelial_barrier": 1.2, "stromal_hazard": 0.7,
               "met_grid_size": 10, "K_met": 16, "host_fill_frac": 0.4, "met_seed_kappa": 0.08,
               "met_hazard": 0.5, "met_transit_floor": 0.03}
MET_SEL = {**SELECTION_PARAMS, "prop_breach": 0.2, "prop_stromal_survival": 0.2,
           "prop_met_survival": 0.2, "breach_effects": 2.0, "stromal_survival_effects": 2.0,
           "met_survival_effects": 2.2}


def _build(seed=3, mode="tau", spatial=SPATIAL, deme=DEME, selection=SELECTION_PARAMS, **kw):
    # Genotype ids come from a process-wide counter and the digests include them.
    Cell._genotype_counter = 2_000_000
    return GenotypeTumor(seed=seed, genome_params=GENOME_PARAMS, selection_params=selection,
                         cancer_cell_params=CANCER_CELL_PARAMS, deme_params=deme,
                         spatial_params=spatial, update_mode=mode, **kw)


def _digest(t):
    blob = repr((t.step, t.traces, sorted(t.genotypes_counts.items()),
                 sorted(t.genotypes_parents.items()), t.events, t.deme_rates.tolist()))
    return hashlib.md5(blob.encode()).hexdigest()


def _met_tumour():
    t = _build(spatial=MET_SPATIAL, deme=MET_DEME, selection=MET_SEL, tau=1.0)
    for _ in range(40):
        met = sum(c for i in range(t.n_primary_demes, len(t.demes))
                  for g, c in t.demes[i].items() if t._is_cancer(g))
        if met >= 30:
            break
        t.grow(n_steps=2, seed=3)
    return t


def _whole_primary_met():
    t = _met_tumour()
    t.grow(n_steps=3, seed=3, treatment=Surgery(start=t.step, site="primary"))
    t.grow(n_steps=3, seed=3)
    return t


def _whole_met():
    t = _met_tumour()
    t.grow(n_steps=3, seed=3, treatment=Surgery(start=t.step + 1, site="met"))
    t.grow(n_steps=3, seed=3)
    return t


def _whole_primary_tau():
    t = _build()
    t.grow(n_steps=20, seed=3, treatment=Surgery(start=12))
    return t


def _whole_primary_exact():
    t = _build(mode="exact")
    t.grow(n_steps=300, seed=3, treatment=Surgery(start=150))
    return t


WHOLE_SCENARIOS = {"primary_with_met": _whole_primary_met, "met": _whole_met,
                   "primary_tau": _whole_primary_tau, "primary_exact": _whole_primary_exact}

# Recorded on the code BEFORE partial resection existed (feat/mechanistic-drug + perf/materialize-opt-in).
WHOLE_DIGESTS = {
    "primary_with_met": "d0263fdab083025d2aaa924809ff79c1",
    "met": "fa9f77542abc9ef32e33dad285602c7c",
    "primary_tau": "abe717192f39eb1a683300438deae286",
    "primary_exact": "b6af752cf0468c33f912ee5f3e68ef23",
}


@pytest.mark.parametrize("name", sorted(WHOLE_SCENARIOS))
def test_whole_compartment_resection_is_byte_identical(name):
    """(a) region=None removes the whole compartment exactly as before."""
    assert _digest(WHOLE_SCENARIOS[name]()) == WHOLE_DIGESTS[name]


# --- partial resection ------------------------------------------------------------------------------
# A dispersive lesion, so it spans enough demes to cut through.
LESION_CANCER = {"division_rate": 0.6, "death_rate": 0.05, "max_birth_rate": 1.0,
                 "mutation_rate": 0.3, "dispersal_rate": 0.4}


def _lesion(target=800, mode="tau", **kw):
    Cell._genotype_counter = 2_000_000
    t = GenotypeTumor(seed=3, genome_params=GENOME_PARAMS, selection_params=SELECTION_PARAMS,
                      cancer_cell_params=LESION_CANCER, deme_params=DEME, spatial_params=SPATIAL,
                      update_mode=mode, **kw)
    steps = 2 if mode == "tau" else 200
    for _ in range(60):
        if t.get_cancer_size() >= target:
            break
        t.grow(n_steps=steps, seed=3)
    return t


def _cancer_by_deme(t):
    return {i: sum(c for g, c in d.items() if c > 0 and t._is_cancer(g))
            for i, d in enumerate(t.demes[:t.n_primary_demes])}


def _centre(t):
    occ = {i: c for i, c in _cancer_by_deme(t).items() if c}
    rc = np.array([t.deme_coords[i] for i in occ], dtype=float)
    w = np.array(list(occ.values()), dtype=float)
    cen = (rc * w[:, None]).sum(0) / w.sum()
    return cen, float(np.sqrt(((rc - cen) ** 2).sum(1)).max())


def _half(t):
    """The demes left of a straight cut through the lesion's centre (column < k)."""
    k = int(round(_centre(t)[0][1]))
    return k, [i for i in range(t.n_primary_demes) if t.deme_coords[i][1] < k]


def test_a_region_through_the_lesion_removes_its_half_and_reports_the_cut():
    """(b) Cut straight through the lesion: the region's cancer cells go, the other half stays, the
    margin is positive and the residual is non-zero — all checked against the geometry of a straight
    cut (rim = the `margin` columns left of the cut, residual distance = c - k + 1)."""
    t = _lesion()
    k, region = _half(t)
    before = _cancer_by_deme(t)
    by_col = lambda lo, hi: sum(c for i, c in before.items() if lo <= t.deme_coords[i][1] < hi)
    margin = 2
    rec = Surgery(start=t.step, region=region, margin=margin).discrete_event(t, t.step)

    assert sum(rec["removed"].values()) == by_col(0, k) > 100
    assert all(sum(c for c in t.demes[i].values()) == 0 for i in region)     # residents too
    assert all(t.deme_rates[i] == 0 for i in region)
    assert sum(_cancer_by_deme(t).values()) == by_col(k, t.grid_size) > 100   # the other half

    assert rec["margin_positive"]
    assert sum(rec["margin_cancer"].values()) == by_col(k - margin, k)
    res = rec["residual"]
    assert res["n"].sum() == by_col(k, t.grid_size)
    cols = np.array([t.deme_coords[d][1] for d in res["deme"]])
    assert np.array_equal(res["distance"].values, cols - k + 1.0)
    assert sorted(set(res["genotype"])) == sorted(g for g in t.genotypes_counts
                                                  if t.genotypes_counts[g] and t._is_cancer(g))
    assert t.resections[-1] is rec and np.array_equal(rec["region"], region)
    ev = t.events[-1]
    assert ev["event"] == "resection" and ev["n_cancer"] == by_col(0, k)
    assert ev["margin_cancer"] == by_col(k - margin, k) and ev["residual"] == by_col(k, t.grid_size)


def test_a_region_enclosing_the_lesion_with_a_wide_margin_is_clear():
    """(c) A disc around the whole lesion, `margin` demes wider than its furthest cell: nothing in
    the rim (negative margin), nothing left behind, and the tumour is gone."""
    t = _lesion(target=300)
    (r0, c0), radius = _centre(t)
    margin = 3
    G = t.grid_size
    rr, cc = np.mgrid[0:G, 0:G]
    disc = (rr - r0) ** 2 + (cc - c0) ** 2 <= (radius + margin + 1.5) ** 2   # a (side, side) mask
    assert not disc.all()                                                   # there is a cut
    n0 = t.get_cancer_size()
    rec = Surgery(start=t.step, region=disc, margin=margin).discrete_event(t, t.step)
    assert sum(rec["removed"].values()) == n0
    assert not rec["margin_positive"] and rec["margin_cancer"] == {}
    assert rec["residual"].empty and t.get_cancer_size() == 0


def test_residual_cells_regrow_after_a_partial_resection():
    """(d) What the cut leaves behind keeps growing, and recolonises the cavity."""
    t = _lesion()
    _, region = _half(t)
    rec = Surgery(start=t.step, region=region, margin=2).discrete_event(t, t.step)
    left = rec["residual"]["n"].sum()
    t.grow(n_steps=6, seed=3)
    assert t.get_cancer_size() > 1.5 * left
    assert sum(c for i in region for g, c in t.demes[i].items() if t._is_cancer(g)) > 0
    # the regrowth descends from the residual clones
    residual = set(rec["residual"]["genotype"])
    for g, c in t.genotypes_counts.items():
        if c and t._is_cancer(g):
            while g not in residual:
                g = t.genotypes_parents[g]
            assert g in residual


def test_surgery_in_grow_fires_once_and_returns_the_record():
    t = _lesion()
    _, region = _half(t)
    s = Surgery(start=t.step, region=region, margin=1)
    t.grow(n_steps=3, seed=3, treatment=s)
    assert len(t.resections) == 1 and t.resections[0]["step"] == t.step - 3
    assert s.discrete_event(t, t.step) is None                              # one-shot


def test_partial_resection_on_the_exact_engine_keeps_every_rate_current():
    t = _lesion(target=300, mode="exact")
    _, region = _half(t)
    Surgery(start=t.step, region=region).discrete_event(t, t.step)
    assert np.allclose(t.deme_rates, [t._deme_rate(i) for i in range(len(t.demes))])
    n = t.get_cancer_size()
    t.grow(n_steps=300, seed=3)
    assert t.get_cancer_size() > n


def test_partial_resection_marks_cell_data_stale():
    """Both routes: a direct resection marks the tables stale, and a grow(materialize=False) through
    a resection leaves them stale; ensure_cell_data then rebuilds them from the cut tumour."""
    t = _lesion()
    _, region = _half(t)
    Surgery(start=t.step, region=region).discrete_event(t, t.step)
    assert t._cell_data_stale
    cd = t.ensure_cell_data(max_cells=100_000)
    assert len(cd["cell_type"]) == t.get_tumor_size()
    assert not set(region) & set(cd["cell_deme"]["deme_id"])               # nothing in the cavity

    t = _lesion()
    _, region = _half(t)
    t.grow(n_steps=2, seed=3, treatment=Surgery(start=t.step, region=region), materialize=False)
    assert t._cell_data_stale and len(t.resections) == 1
    assert len(t.ensure_cell_data(max_cells=100_000)["cell_type"]) == t.get_tumor_size()


def test_a_mechanistic_drug_after_a_partial_resection_acts_on_the_residual():
    from iscc.treatment.chemotherapy import Chemotherapy
    t = _lesion()
    _, region = _half(t)
    rec = Surgery(start=t.step, region=region).discrete_event(t, t.step)
    left = rec["residual"]["n"].sum()
    tx = Chemotherapy(response="mechanistic", kill_mode="proliferation", kill_rate=1.5,
                      toxicity=0.0)
    t.grow(n_steps=4, seed=3, treatment=tx)
    assert 0 < t.get_cancer_size() < 0.5 * left


def test_partial_resection_of_the_metastasis_cuts_on_the_met_grid():
    t = _met_tumour()
    lo, side = t.n_primary_demes, t.met_grid_size
    primary_before = sum(_cancer_by_deme(t).values())
    region = [lo + r * side + c for r in range(side) for c in range(side // 2)]
    rec = Surgery(start=t.step, site="met", region=region, margin=1).discrete_event(t, t.step)
    assert sum(_cancer_by_deme(t).values()) == primary_before                # primary untouched
    res = rec["residual"]
    assert (res["deme"] >= lo).all()                                         # met cells only
    assert np.array_equal(res["distance"].values, res["col"].values - side // 2 + 1.0)


@pytest.mark.parametrize("kw,match", [
    (dict(margin=-1, region=[0]), "non-negative"),
    (dict(margin=2), "needs a region"),
])
def test_surgery_arguments_are_validated(kw, match):
    with pytest.raises(ValueError, match=match):
        Surgery(**kw)


@pytest.mark.parametrize("region,match", [
    ([10_000], "outside the primary"),
    (np.ones(7, dtype=bool), "one entry per primary deme"),
    (np.zeros(24 * 24, dtype=bool), "selects no deme"),
])
def test_bad_regions_are_rejected(region, match):
    t = _build()
    with pytest.raises(ValueError, match=match):
        Surgery(region=region).discrete_event(t, 0)
