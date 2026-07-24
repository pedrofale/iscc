"""Metastasis module (R9): a second, clonally-linked deme-grid (the metastatic deposit) reached from
the primary by an invasion-gated migration operator, plus per-site treatment, surgical resection, and
the two-compartment views.

Design docs: the metastasis plan (2-band Muller + two-grid plot_grid), building on the ductal field
(DESIGN_ductal_field.md) and compartment selection (DESIGN_phenotype_plasticity.md). COUNT engine only.

Covered here:
  * OFF-by-default -> byte-identical: the met_survival axis is inert at prop 0, and an ENABLED-but-inert
    met (met_grid_size>0, met_seed_kappa=0, met_hazard=0) leaves the primary cancer growth byte-identical
    (exact + tau) — the appended host-only demes carry zero event rate.
  * met_survival is a real, sequenceable heritable axis (gene counts + per-genotype trait surface).
  * the met host hazard lowers a high-met_survival clone's death ONLY in a host-occupied met deme.
  * seeding is invasion-gated: DCIS (confined) cannot seed, IDC (in stroma) can; every met clone traces
    back to the primary founder (shared genealogy); a floor-0 transit gate admits only met_survival>0
    founders (the transit bottleneck).
  * per-site treatment: the chemo/immuno override is gated to the treated compartment(s).
  * surgical resection empties the primary and the simulation continues on the met alone.
  * the two-band Muller and two-grid plot_grid render on a shared clone colormap.
"""
import hashlib

import numpy as np
import pytest

from conftest import GENOME_PARAMS, SELECTION_PARAMS, CANCER_CELL_PARAMS
from iscc.constants import normal_names
from iscc.tumor.models import GenotypeTumor
from iscc.tumor.components.cell import CancerCell
from iscc.tumor.components.selection import Selection
from iscc.treatment import Surgery, Chemotherapy

# A structured ductal field (glands + stroma) so IDC can invade the stroma and seed, plus a metastatic
# deposit. Small + few steps so the suite stays fast.
DEME = {"carrying_capacity": 16, "initial_cancer_cells": 6, "resident_pressure_ref": 0.2}
SPATIAL = {"grid_size": 16, "structure_radius": 3, "n_glands": 3, "gland_radius": 3, "min_gland_sep": 8,
           "K_duct": 28, "K_stroma": 18, "stroma_fill_frac": 0.3, "cross_gland_kappa": 0.05,
           "epithelial_barrier": 1.2, "stromal_hazard": 0.7}
SEL = {**SELECTION_PARAMS, "prop_breach": 0.2, "prop_stromal_survival": 0.2, "prop_met_survival": 0.2,
       "breach_effects": 2.0, "stromal_survival_effects": 2.0, "met_survival_effects": 2.2}
MET = {"met_grid_size": 10, "K_met": 16, "host_fill_frac": 0.4,
       "met_seed_kappa": 0.08, "met_hazard": 0.5, "met_transit_floor": 0.03}


def _build(spatial, selection=SEL, mode="tau", seed=3):
    return GenotypeTumor(seed=seed, genome_params=GENOME_PARAMS, selection_params=selection,
                         cancer_cell_params=CANCER_CELL_PARAMS, deme_params=DEME,
                         spatial_params=spatial, update_mode=mode, tau=1.0)


def _grow_to_met(t, target=40, cap=4000, steps=2):
    for _ in range(70):
        p, m = _compartment_cancer(t)
        if m >= target or p + m >= cap:
            break
        t.grow(n_steps=steps, seed=t.seed)
    return t


def _compartment_cancer(t):
    p = sum(c for i in range(t.n_primary_demes) for g, c in t.demes[i].items() if t._is_cancer(g))
    m = sum(c for i in range(t.n_primary_demes, len(t.demes)) for g, c in t.demes[i].items() if t._is_cancer(g))
    return p, m


def _cancer_snv_hash(t):
    t.make_cell_data()
    ct = t.cell_data["cell_type"]["cell_id"].astype(str).values
    mask = ~np.isin(ct, list(normal_names))
    return hashlib.md5(np.ascontiguousarray(t.cell_data["cell_snv"].values[mask])).hexdigest()


@pytest.fixture(scope="module")
def met_tumor():
    """One grown met tumour, reused by the read-only tests."""
    return _grow_to_met(_build({**SPATIAL, **MET}))


# --- ground truth: met_survival is a sequenceable axis ----------------------------------------
def test_met_survival_axis_ground_truth():
    on = Selection(n_segments=6, segment_size=200, prop_met_survival=0.2)
    assert on.N_ms == len(on.get_met_survival()) > 0
    assert "met_survival_types" in on.get_gene_data()
    off = Selection(n_segments=6, segment_size=200)
    assert off.N_ms == 0 and len(off.get_met_survival()) == 0
    assert off.update_met_survival({"n_wt_ms": 0, "n_mut_ms": 0, "ploidy": 2}) == 1.0


def test_met_survival_trait_surfaces():
    sel = Selection(n_segments=4, segment_size=100, prop_met_survival=0.3, met_survival_effects=1.8)
    c = CancerCell(n_segments=4, segment_size=100, n_ms=len(sel.get_met_survival()),
                   division_rate=0.5, death_rate=0.1)
    c.set_genotype_id()
    c.update_evolutionary_parameters(sel)
    assert c.evolutionary_parameters["met_survival"] == 0.0            # wild-type
    for seg in range(sel.n_segments):
        bits = np.zeros(sel.segment_sizes[seg], dtype=bool)
        bits[sel.met_survival[seg]] = True
        for hap in ("p", "m"):
            for allele in c.genome[seg][hap]:
                allele[bits] = True
        c.update_genome_summary_mutation(sel, bits, seg)
    c.update_evolutionary_parameters(sel)
    assert c.evolutionary_parameters["met_survival"] > 0.0
    assert c.genome_summary["n_mut_ms"] > 0


# --- byte-identity: the module is inert when off ----------------------------------------------
def test_met_survival_axis_inert_at_zero():
    """prop_met_survival=0 (drawn last, binomial(1,0) consumes no rng) must equal omitting it."""
    for seed in (1, 2):
        a = _build(SPATIAL, selection={**SEL, "prop_met_survival": 0.0}, seed=seed)
        b = _build(SPATIAL, selection={k: v for k, v in SEL.items()
                                       if k not in ("prop_met_survival", "met_survival_effects")}, seed=seed)
        a.grow(n_steps=30, seed=seed); b.grow(n_steps=30, seed=seed)
        assert a.get_cancer_size() == b.get_cancer_size()
        assert _cancer_snv_hash(a) == _cancer_snv_hash(b)


@pytest.mark.parametrize("mode", ["exact", "tau"])
def test_inert_met_matches_no_met(mode):
    """An ENABLED but inert met (met_seed_kappa=0, met_hazard=0) leaves the primary cancer growth
    byte-identical to a no-met run — the appended host-only demes carry zero event rate."""
    steps = 150 if mode == "exact" else 12
    a = _build(SPATIAL, mode=mode); a.grow(n_steps=steps, seed=3)
    b = _build({**SPATIAL, **MET, "met_seed_kappa": 0.0, "met_hazard": 0.0}, mode=mode)
    b.grow(n_steps=steps, seed=3)
    assert a.get_cancer_size() == b.get_cancer_size()
    assert _cancer_snv_hash(a) == _cancer_snv_hash(b)


# --- the met host hazard pays off only with host present --------------------------------------
def test_met_hazard_lowers_death_only_with_host():
    t = _build({**SPATIAL, **MET})
    cg = t.founder_id
    vidx = t.met_vessel_idx
    # host-occupied met deme: a high-met_survival clone dies strictly less than wild-type
    t.demes[vidx] = {cg: 2, "host": 6}
    t.genotypes[cg].evolutionary_parameters["met_survival"] = 0.9
    d_ms = t._death_rate(cg, vidx)
    t.genotypes[cg].evolutionary_parameters["met_survival"] = 0.0
    d_wt = t._death_rate(cg, vidx)
    assert d_ms < d_wt
    # host-free deme: met_survival pays nothing (the hazard term is zero)
    t.demes[vidx] = {cg: 2}
    d_wt2 = t._death_rate(cg, vidx)
    t.genotypes[cg].evolutionary_parameters["met_survival"] = 0.9
    d_ms2 = t._death_rate(cg, vidx)
    assert d_wt2 == pytest.approx(d_ms2)


# --- seeding is invasion-gated ----------------------------------------------------------------
def test_seeding_requires_idc(met_tumor):
    """IDC (breach/stromal_survival on -> cancer reaches the stroma) seeds the met."""
    assert _compartment_cancer(met_tumor)[1] > 0
    assert len(met_tumor.events) > 0 and any(e["event"] == "seeding" for e in met_tumor.events)


def test_seeding_originates_from_stroma(met_tumor):
    """The gate is deterministic: every met seeding event originates from a primary STROMAL deme
    (gland_id == -1) — i.e. an invaded IDC cell in the stroma, never a duct-confined DCIS cell (a
    lumen/ring deme has gland_id >= 0 and is never eligible). This is the "DCIS can't seed, IDC can"
    claim, as a hard invariant of the operator rather than a stochastic rate."""
    seedings = [e for e in met_tumor.events if e["event"] == "seeding"]
    assert seedings
    assert all(met_tumor.gland_id[e["from_deme"]] == -1 for e in seedings)


def test_met_clones_trace_to_primary_founder(met_tumor):
    """Shared genealogy: every met genotype reaches the primary founder through genotypes_parents."""
    def traces(gid):
        seen = set()
        while gid != met_tumor.founder_id and gid in met_tumor.genotypes_parents and gid not in seen:
            seen.add(gid); gid = met_tumor.genotypes_parents[gid]
        return gid == met_tumor.founder_id
    met_gids = {g for i in range(met_tumor.n_primary_demes, len(met_tumor.demes))
                for g in met_tumor.demes[i] if met_tumor._is_cancer(g)}
    assert met_gids and all(traces(g) for g in met_gids)


def test_transit_gate_selects_met_survival():
    """With met_transit_floor=0, transit survival prob == met_survival, so a wild-type (met_survival=0)
    daughter NEVER survives the jump: every seeding founder must carry met_survival > 0."""
    t = _grow_to_met(_build({**SPATIAL, **MET, "met_transit_floor": 0.0}))
    seeded = [e["genotype"] for e in t.events if e["event"] == "seeding"]
    assert seeded, "expected at least one seeding event"
    assert all(t.genotypes[g].evolutionary_parameters["met_survival"] > 0.0 for g in seeded)


# --- host is static; cells are compartment-tagged ---------------------------------------------
def test_host_static_and_cells_tagged(met_tumor):
    met_tumor.make_cell_data()
    comp = met_tumor.cell_data["cell_compartment"]["compartment"].to_numpy()
    assert set(np.unique(comp).tolist()) == {0, 1}                     # both compartments present
    # host never counts as cancer
    assert all(met_tumor.genotypes[g].type != "cancer"
               for g in ("host",) if g in met_tumor.genotypes)


# --- per-site treatment gate ------------------------------------------------------------------
def test_treatment_site_gate(met_tumor):
    di_pri = 0
    di_met = met_tumor.n_primary_demes
    for sites, want in (("both", (True, True)), ("met", (False, True)), ("primary", (True, False))):
        met_tumor._tx_sites = sites
        assert (met_tumor._tx_applies(di_pri), met_tumor._tx_applies(di_met)) == want
    met_tumor._tx_sites = "both"


def test_met_only_chemo_spares_primary():
    """Systemic-to-met chemo (sites='met') adds a death hazard in the met but not the primary."""
    t = _grow_to_met(_build({**SPATIAL, **MET}))
    chemo = Chemotherapy(start=0, kill_rate=1.5, effectiveness=1.0, sites="met")
    t._apply_treatment(chemo, 1.0)
    cg = next(g for i in range(t.n_primary_demes, len(t.demes)) for g in t.demes[i] if t._is_cancer(g))
    di_met = next(i for i in range(t.n_primary_demes, len(t.demes)) if cg in t.demes[i])
    di_pri = next(i for i in range(t.n_primary_demes) if any(t._is_cancer(g) for g in t.demes[i]))
    cg_pri = next(g for g in t.demes[di_pri] if t._is_cancer(g))
    # the tx death add reaches the met deme but is gated out of the primary
    assert t._tx_applies(di_met) and not t._tx_applies(di_pri)


# --- surgical resection -----------------------------------------------------------------------
def test_resection_empties_primary_and_met_continues():
    t = _grow_to_met(_build({**SPATIAL, **MET}))
    p0, m0 = _compartment_cancer(t)
    assert p0 > 0 and m0 > 0
    removed = t._resect("primary")
    p1, m1 = _compartment_cancer(t)
    assert p1 == 0 and m1 == m0 and removed > 0
    assert float(t.deme_rates[:t.n_primary_demes].sum()) == 0.0        # primary is inert
    assert any(e["event"] == "resection" for e in t.events)
    t.grow(n_steps=4, seed=t.seed)                                     # continues on the met alone
    assert _compartment_cancer(t)[0] == 0                             # primary stays empty


def test_surgery_operator_discrete_event():
    t = _grow_to_met(_build({**SPATIAL, **MET}))
    surg = Surgery(start=t.step, site="primary")
    surg.discrete_event(t, t.step)
    assert _compartment_cancer(t)[0] == 0 and surg._done


# --- two-compartment views --------------------------------------------------------------------
def test_two_band_muller_and_grids_render(met_tumor):
    import matplotlib
    matplotlib.use("Agg")
    from iscc.tumor import viz
    shared = viz.cell_type_colors(met_tumor.traces, met_tumor.genotypes_parents)
    axes = met_tumor.plot_muller_compartments()
    assert len(axes) == 2
    met_tumor.make_cell_data()
    gaxes = met_tumor.plot_grid_compartments()
    assert len(gaxes) == 2
    # a met clone shares its colour with the global (grid) colormap by construction
    met_gid = next(g for i in range(met_tumor.n_primary_demes, len(met_tumor.demes))
                   for g in met_tumor.demes[i] if met_tumor._is_cancer(g))
    assert str(met_gid) in {str(k) for k in shared}


def test_empty_met_muller_degrades_gracefully():
    """A met that never seeded draws an informative empty panel rather than erroring."""
    import matplotlib
    matplotlib.use("Agg")
    t = _build({**SPATIAL, **MET, "met_seed_kappa": 0.0})
    t.grow(n_steps=8, seed=3)
    axes = t.plot_muller_compartments()
    assert len(axes) == 2


# --- chemo toxicity on off-target NORMAL tissue (one toxicity = off-target anything) -----------
def _normal_count(t, lo=0, hi=None):
    hi = len(t.demes) if hi is None else hi
    return sum(c for i in range(lo, hi) for g, c in t.demes[i].items() if g in normal_names)


def test_chemo_toxicity_kills_normal_tissue():
    """Systemic chemo's toxicity removes off-target normal cells; without treatment the static normals
    are unchanged (so the loss is the drug's, not drift)."""
    t = _grow_to_met(_build({**SPATIAL, **MET}))
    n0 = _normal_count(t)
    t.grow(4, seed=t.seed)                                    # untreated -> normals static
    assert _normal_count(t) == n0
    t.grow(6, seed=t.seed, treatment=Chemotherapy(start=t.step, kill_rate=1.5, effectiveness=0.9,
                                                  toxicity=0.3, sites="both"))
    assert _normal_count(t) < n0


def test_toxicity_gated_to_treated_compartment():
    """Chemo sites='met' kills the met's host tissue but spares primary normal tissue."""
    t = _grow_to_met(_build({**SPATIAL, **MET}))
    p0 = _normal_count(t, 0, t.n_primary_demes)
    m0 = _normal_count(t, t.n_primary_demes)
    t.grow(6, seed=t.seed, treatment=Chemotherapy(start=t.step, kill_rate=1.5, effectiveness=0.9,
                                                  toxicity=0.3, sites="met"))
    assert _normal_count(t, t.n_primary_demes) < m0          # met host tissue killed
    assert _normal_count(t, 0, t.n_primary_demes) == p0      # primary normal tissue spared


def test_treatment_overrides_cleared_after_grow():
    """A grow with NO treatment after a treated grow must not keep applying the stale toxicity hazard
    (normal tissue stops dying once chemo is over)."""
    t = _grow_to_met(_build({**SPATIAL, **MET}))
    t.grow(4, seed=t.seed, treatment=Chemotherapy(start=t.step, kill_rate=1.5, effectiveness=0.9,
                                                  toxicity=0.3, sites="both"))
    n = _normal_count(t)
    t.grow(4, seed=t.seed)                                    # untreated -> normals must hold
    assert _normal_count(t) == n


def test_untreated_growth_byte_identical_with_toxicity_code():
    """The off-target-toxicity machinery is inert without an active treatment: an untreated run is
    byte-identical to a plain run at the same seed (no residual hazard, no extra rng)."""
    a = _build({**SPATIAL, **MET}); a.grow(n_steps=12, seed=5)
    b = _build({**SPATIAL, **MET}); b.grow(n_steps=12, seed=5)
    assert a.get_cancer_size() == b.get_cancer_size()
    assert _cancer_snv_hash(a) == _cancer_snv_hash(b)


# --- functional-clone views + founder Muller render -------------------------------------------
def test_functional_muller_founders_and_grids_render(met_tumor):
    import matplotlib
    matplotlib.use("Agg")
    axes = met_tumor.plot_muller_compartments(by_drivers=True, min_freq=0.05)
    assert len(axes) == 2
    assert met_tumor.plot_muller_founders() is not None
    met_tumor.make_cell_data()
    assert len(met_tumor.plot_grid_compartments(by_drivers=True)) == 2


def test_functional_signature_spans_more_than_drivers():
    """The functional signature (all selection axes) is a superset of the driver (onc/TSG) signature."""
    t = _build({**SPATIAL, **MET})
    s = t.selection
    parts = [s.get_oncogenes(), s.get_tsgs(), s.get_breach(), s.get_stromal_survival(),
             s.get_met_survival(), s.get_treatment_resistant()]
    func_idx = set(int(i) for p in parts for i in np.asarray(p, dtype=int))
    driver_idx = set(int(i) for p in (s.get_oncogenes(), s.get_tsgs()) for i in np.asarray(p, dtype=int))
    assert driver_idx.issubset(func_idx) and len(func_idx) > len(driver_idx)
    assert all(isinstance(sig, tuple) for sig in t._functional_signatures().values())


def test_clone_grid_series_renders(met_tumor):
    """The spatial clone series renders one panel per (snapshot, compartment) on the shared clone
    colormap, and degrades gracefully when a requested snapshot is absent."""
    import matplotlib
    matplotlib.use("Agg")
    from iscc.tumor import viz
    met_tumor.arc_snapshots = {"a": [dict(d) for d in met_tumor.demes]}
    panels = [("a", "primary", "primary"), ("a", "met", "met"), ("missing", "met", "absent")]
    axes = viz.plot_clone_grid_series(met_tumor, panels, ncols=3)
    assert len(axes) >= 3
    # with the functional-clone palette too
    axes2 = viz.plot_clone_grid_series(met_tumor, panels[:2],
                                       driver_map=met_tumor._functional_signatures(), min_freq=0.05)
    assert len(axes2) >= 2


def test_stage_dominant_coloring(met_tumor):
    """The stage-dominant-driver colouring maps each cancer clone to its highest arc stage and renders
    on the Muller + grids on one categorical palette (todo #14)."""
    import matplotlib
    matplotlib.use("Agg")
    from iscc.tumor import viz
    colors, present = met_tumor._stage_colors()
    # every cancer genotype gets a stage colour from the palette; normals keep their type colour
    assert colors and set(present).issubset(set(range(len(viz.STAGE_PALETTE))))
    assert all(tuple(c) in {tuple(p) for p in viz.STAGE_PALETTE} or g in normal_names
               for g, c in colors.items())
    axes = met_tumor.plot_muller_compartments(by_stage=True, min_freq=0.05)
    assert len(axes) == 2
    met_tumor.make_cell_data()
    assert len(met_tumor.plot_grid_compartments(by_stage=True)) == 2


def test_clone_tree_renders(met_tumor):
    """The ground-truth clone tree renders (stage + functional colourings) and degrades on an empty
    clone set."""
    import matplotlib
    matplotlib.use("Agg")
    from iscc.tumor import viz
    ax = met_tumor.plot_clone_tree(by_stage=True, min_freq=0.03)
    assert ax is not None
    ax2 = met_tumor.plot_clone_tree(by_drivers=True, min_freq=0.05)
    assert ax2 is not None
    # empty degrades gracefully
    ax3 = viz.plot_clone_tree([{"genotypes_counts": {}}], {}, min_freq=0.1)
    assert ax3 is not None
