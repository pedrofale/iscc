"""Landing clinical-arc feature (metastasis module, R9): the config-driven timed treatment SCHEDULE
that grows a primary+metastasis through grow -> surgery/resection -> chemotherapy -> relapse and
captures a compartment TRAJECTORY, plus the `isccgif --compartment` composite renderer.

Reproduces the docs landing hero from the ordinary CLI:
    isccsim --sim-config configs/landing.yaml -o out
    isccgif out --compartment --splash -o docs/assets/landing_hero.gif
"""
import os
from types import SimpleNamespace

import numpy as np
import yaml
import pytest
from click.testing import CliRunner

from conftest import GENOME_PARAMS, SELECTION_PARAMS, CANCER_CELL_PARAMS
from iscc.constants import normal_names
from iscc.tumor.models import GenotypeTumor
from iscc.tumor import arc, viz
from iscc.tumor.main import main as sim_main
from iscc.visualization.animate import main as gif_main
from iscc.visualization import compartment as comp

# A small structured ductal field + metastatic deposit so the arc runs fast.
DEME = {"carrying_capacity": 16, "initial_cancer_cells": 6, "resident_pressure_ref": 0.2}
SPATIAL = {"grid_size": 14, "structure_radius": 3, "n_glands": 3, "gland_radius": 3, "min_gland_sep": 7,
           "K_duct": 24, "K_stroma": 16, "stroma_fill_frac": 0.3, "cross_gland_kappa": 0.05,
           "epithelial_barrier": 1.2, "stromal_hazard": 0.7,
           "met_grid_size": 10, "K_met": 16, "host_fill_frac": 0.4,
           "met_seed_kappa": 0.08, "met_hazard": 0.5, "met_transit_floor": 0.03}
SEL = {**SELECTION_PARAMS, "prop_breach": 0.2, "prop_stromal_survival": 0.2, "prop_met_survival": 0.2,
       "breach_effects": 2.0, "stromal_survival_effects": 2.0, "met_survival_effects": 2.2}

SCHEDULE = {
    "seed": 3, "min_freq": 0.05,
    "capture": {"pre_seed_every": 2, "post_seed_every": 1},
    "phases": [
        {"op": "grow", "until": {"met_cancer": 30, "total_cancer": 2500},
         "phase": "growth", "seeded_phase": "metastatic seeding"},
        {"op": "surgery", "site": "primary", "settle": 2, "phase": "resection"},
        {"op": "chemotherapy", "steps": 6, "duration": 6, "kill_rate": 2.0,
         "effectiveness": 0.95, "toxicity": 0.1, "sites": "both", "phase": "chemotherapy"},
        {"op": "grow", "steps": 8, "phase": "relapse"},
    ],
}


def _build():
    return GenotypeTumor(seed=3, genome_params=GENOME_PARAMS, selection_params=SEL,
                         cancer_cell_params=CANCER_CELL_PARAMS, deme_params=DEME,
                         spatial_params=SPATIAL, update_mode="tau", tau=1.0)


def _primary_cancer(demes, n_primary):
    return sum(c for i in range(n_primary) for g, c in demes[i].items() if g not in normal_names)


@pytest.fixture(scope="module")
def arc_run():
    """One executed arc reused across the read-only tests."""
    t = _build()
    frames, marks, seed, min_freq = arc.execute_schedule(t, SCHEDULE)
    traj = arc.build_trajectory(t, frames, marks, min_freq)
    return t, frames, marks, traj


# --- schedule parsing + execution -------------------------------------------------------------
def test_schedule_runs_all_phases(arc_run):
    t, frames, marks, traj = arc_run
    assert frames and all({"demes", "cursor", "phase"} <= set(f) for f in frames)
    # every scheduled phase surfaces in the captured frames
    assert {"growth", "resection", "chemotherapy", "relapse"} <= {f["phase"] for f in frames}
    # the reveal cursor is monotone non-decreasing (each frame reveals at least as much)
    cursors = [f["cursor"] for f in frames]
    assert cursors == sorted(cursors)


def test_schedule_records_clinical_event_marks(arc_run):
    t, frames, marks, traj = arc_run
    labels = [l for _, l in marks]
    assert labels[-3:] == ["resection", "chemo start", "chemo end"]   # execution order
    assert "seeding" in labels                                        # the met seeded
    # a resection surgery + a seeding hop are recorded on the engine too
    assert any(e["event"] == "resection" for e in t.events)
    assert any(e["event"] == "seeding" for e in t.events)


def test_surgery_phase_empties_primary(arc_run):
    """The resection frame (and everything after) has zero primary cancer — the primary is resected
    and the arc continues on the metastasis alone."""
    t, frames, marks, traj = arc_run
    n_primary = traj["n_primary_demes"]
    resect_frames = [f for f in frames if f["phase"] == "resection"]
    assert resect_frames
    assert all(_primary_cancer(f["demes"], n_primary) == 0 for f in resect_frames)
    # relapse (post-chemo) is met-only too
    assert all(_primary_cancer(f["demes"], n_primary) == 0
               for f in frames if f["phase"] == "relapse")


def test_unknown_schedule_op_raises():
    t = _build()
    with pytest.raises(ValueError):
        arc.execute_schedule(t, {"seed": 3, "phases": [{"op": "radiotherapy"}]})


# --- drug escape: the stage label must agree with the engine's protection -----------------------
def _rep(**ep):
    """A minimal cancer genotype stand-in: only what ``stage_of`` reads."""
    ep.setdefault("division_rate", 1.0)
    return SimpleNamespace(type="cancer", evolutionary_parameters=ep,
                           baseline_rates={"division_rate": 1.0})


@pytest.mark.parametrize("tr, dt, expect", [
    (0.0, 1.0, "tolerance"),     # PURE PERSISTER: takes zero drug in the engine (max(tr, dt) = 1)
    (0.3, 0.9, "tolerance"),     # tolerance is what keeps it alive -> it is the persister colour
    (1.0, 0.0, "resistance"),
    (0.9, 0.3, "resistance"),    # resistance already covers the drug; the persister state is idle
    (1.0, 1.0, "resistance"),    # tie -> resistance, matching _apply_treatment's `dt > tr` cost test
])
def test_drug_escape_stage_matches_engine_protection(tr, dt, expect):
    """A cell's drug-escape stage must follow the protection the ENGINE actually gives it, which is
    ``max(treatment_resistance, drug_tolerance)`` (``_apply_treatment`` / ``_tx_death_add_for``) — not
    ``treatment_resistance`` alone. The pure persister (tr=0, dt=1) is the case that matters: it takes
    ZERO drug, so labelling it a drug-sensitive stage would paint the residual-disease floor as
    sensitive in every stage/colour/Muller/GIF product.

    Tolerance keeps its own stage rather than folding into resistance, so the two remain distinguishable
    in figures: only resistance regrows under the drug, tolerance merely waits the dose out."""
    assert arc.stage_of(_rep(treatment_resistance=tr, drug_tolerance=dt)) == expect
    # the engine-side twin (plot_tissue(color="stage") / by_stage Mullers) indexes viz.STAGE_PALETTE;
    # both readings must land on the SAME colour or the two product families disagree.
    code = GenotypeTumor._stage_of(None, _rep(treatment_resistance=tr, drug_tolerance=dt))
    assert tuple(viz.STAGE_PALETTE[code]) == tuple(arc.STAGE_COL[expect])


def test_drug_escape_stage_ranks_above_the_cascade_and_tolerates_missing_key():
    """Drug escape outranks the metastatic cascade (it is the last sweep of the arc), a clone with
    neither drug trait still reads its cascade stage, and a genotype minted before the tolerance axis
    existed (no ``drug_tolerance`` key at all) behaves exactly as it always did."""
    assert arc.stage_of(_rep(treatment_resistance=0.0, drug_tolerance=0.8,
                             met_survival=0.5, stromal_survival=0.3)) == "tolerance"
    assert arc.stage_of(_rep(treatment_resistance=0.0, drug_tolerance=0.0, met_survival=0.5)) == "met"
    assert arc.stage_of(_rep(treatment_resistance=0.5)) == "resistance"        # legacy rep, no dt key
    assert arc.stage_of(_rep(breach=0.2)) == "breach"


def test_stage_palettes_stay_index_aligned():
    """``arc`` (string stages, the GIF) and ``viz``/``GenotypeTumor._stage_of`` (integer stages, the
    engine plots) are two spellings of ONE scheme; a stage added to one must be added to the other, and
    every stage needs its own colour or two stages become indistinguishable."""
    assert len(viz.STAGE_NAMES) == len(viz.STAGE_PALETTE)
    assert len({tuple(c) for c in viz.STAGE_PALETTE}) == len(viz.STAGE_PALETTE)
    # STAGE_ORDER is the legend (every stage but "none"); the palette carries "none" at index 0.
    assert len(arc.STAGE_ORDER) == len(viz.STAGE_PALETTE) - 1
    assert set(arc.STAGE_ORDER) | {"none"} == set(arc.STAGE_COL) == set(arc.STAGE_LABEL) | {"none"}
    # the integer stage of each string stage indexes the matching palette colour
    for code, st in enumerate(["none"] + list(arc.STAGE_ORDER)):
        assert tuple(viz.STAGE_PALETTE[code]) == tuple(arc.STAGE_COL[st])


# --- shared clone colormap: grids and Mullers agree by construction ---------------------------
def test_grid_and_muller_share_one_colormap(arc_run):
    """Grids and Mullers share ONE per-cell (founder-clone) colormap: the SAME ``colors`` dict, keyed by
    genotype, drives both, so a clone is the same colour everywhere. Each cancer clone takes the colour of
    ITS OWN cascade stage (``stage_of``) — its trait mutations — NOT a band-dominant/founder-washed colour;
    a low ``min_freq`` (schedule) then keeps the polyclonal invasion sweep's clones as their own coloured
    Muller bands instead of folding them into their grey founder, so the Muller matches the grid."""
    t, frames, marks, traj = arc_run
    colors = traj["colors"]
    checked = 0
    for gid, rep in t.genotypes.items():
        if getattr(rep, "type", None) != "cancer":
            continue
        st = arc.stage_of(rep)
        assert tuple(colors[str(gid)]) == tuple(arc.STAGE_COL["none" if st is None else st])
        checked += 1
    assert checked > 0


def test_trajectory_is_engine_free_and_roundtrips(tmp_path, arc_run):
    t, frames, marks, traj = arc_run
    p = arc.write_trajectory(str(tmp_path), traj)
    assert p.endswith(arc.TRAJECTORY_FILE)
    loaded = arc.read_trajectory(str(tmp_path))
    # everything the renderer needs, and nothing that is a live engine object
    assert {"frames", "traces", "genotypes_parents", "colors", "driver_map", "deme_coords",
            "gid_ord", "n_primary_demes", "grid_size", "met_grid_size", "marks"} <= set(loaded)
    assert loaded["frames"][0]["demes"]                     # deme snapshots survived the pickle


def test_frame_cell_data_matches_engine_ordering(arc_run):
    """The offline grid reconstruction reproduces the engine's own make_cell_data materialisation
    (cell count + per-deme cell ordering), so the section-sampled grids are identical."""
    t, frames, marks, traj = arc_run
    # replay the final frame's demes on the engine and compare cell tables
    t.demes = [dict(d) for d in frames[-1]["demes"]]
    from collections import Counter
    t.genotypes_counts = Counter()
    for d in t.demes:
        for g, c in d.items():
            t.genotypes_counts[g] += c
    t.cell_data = None
    t.make_cell_data()
    eng = t.cell_data
    got = comp.frame_cell_data(frames[-1]["demes"], traj["deme_coords"],
                               traj["n_primary_demes"], traj["gid_ord"])
    assert len(got["cell_type"]) == len(eng["cell_type"])
    assert list(got["cell_type"]["cell_id"].astype(str)) == list(eng["cell_type"]["cell_id"].astype(str))
    assert list(got["cell_compartment"]["compartment"]) == list(eng["cell_compartment"]["compartment"])


# --- the composite renderer -------------------------------------------------------------------
@pytest.mark.parametrize("splash", [True, False])
def test_compartment_render_produces_valid_gif(tmp_path, arc_run, splash):
    from PIL import Image
    t, frames, marks, traj = arc_run
    out = str(tmp_path / ("hero.gif" if splash else "full.gif"))
    mb = comp.render_animation(traj, out, splash=splash, poster=False)
    assert os.path.exists(out) and mb > 0
    im = Image.open(out)
    assert getattr(im, "n_frames", 1) == len(frames)
    # the splash renders at the hero resolution, the full at the labelled resolution
    assert im.size == ((1890, 1065) if splash else (2142, 1207))


# --- CLI: isccsim schedule -> trajectory, then isccgif --compartment --------------------------
@pytest.fixture(scope="module")
def sim_out(tmp_path_factory):
    """Run the isccsim schedule CLI once and return its output dir."""
    cfg = {
        "mode": "genotype", "update_mode": "tau", "tau": 1.0, "snapshot_every": 1,
        "genome_params": GENOME_PARAMS, "selection_params": SEL,
        "cell_params": {"cancer": CANCER_CELL_PARAMS},
        "deme_params": DEME, "spatial_params": SPATIAL, "schedule": SCHEDULE,
    }
    d = tmp_path_factory.mktemp("landing")
    cfg_path = str(d / "landing.yaml")
    yaml.safe_dump(cfg, open(cfg_path, "w"))
    out = str(d / "out")
    r = CliRunner().invoke(sim_main, ["--sim-config", cfg_path, "-o", out], catch_exceptions=False)
    assert r.exit_code == 0, r.output
    return out


def test_isccsim_schedule_writes_trajectory_and_layout(sim_out):
    files = set(os.listdir(sim_out))
    assert arc.TRAJECTORY_FILE in files
    # the canonical layout is still written so downstream isccsample/isccdata work
    assert {"cell_data", "gene_data", "parents.csv", "trace_counts.csv"} <= files
    assert "events.csv" in files          # the seeding/resection annotations


@pytest.mark.parametrize("splash", [True, False])
def test_isccgif_compartment_cli(tmp_path, sim_out, splash):
    from PIL import Image
    out_gif = str(tmp_path / ("hero.gif" if splash else "full.gif"))
    args = [sim_out, "--compartment", "-o", out_gif] + (["--splash"] if splash else [])
    r = CliRunner().invoke(gif_main, args, catch_exceptions=False)
    assert r.exit_code == 0, r.output
    assert os.path.exists(out_gif)
    assert Image.open(out_gif).n_frames > 1


def test_default_animate_still_requires_count_args(sim_out):
    """The historical single-grid mode is unchanged: without --compartment it still needs the
    genotype-count/parent arguments (additive, non-breaking)."""
    r = CliRunner().invoke(gif_main, [sim_out], catch_exceptions=False)
    assert r.exit_code != 0
    assert "GENOTYPE_COUNTS" in r.output or "compartment" in r.output


def test_r15_proliferation_cost_and_driver_load():
    """R15 compartment-context fitness. (a) proliferation_cost: each *_cost lowers a trait-carrying
    clone's division as a multiplicative penalty, and is exactly 1.0 (byte-identical) when all costs are
    0. (b) the driver LOAD counts the dissemination traits toward max_mut_drivers, so a clone that has
    activated every trait is non-viable under a realistic ceiling."""
    from iscc.tumor.components.selection import Selection
    ep = {"breach": 0.6, "stromal_survival": 0.5, "met_survival": 0.9, "treatment_resistance": 0.6}
    assert Selection().proliferation_cost(ep) == 1.0                       # off -> no-op
    # breach_cost 0.5 on breach 0.6 -> factor 1 - 0.3 = 0.70; the others still 0
    assert abs(Selection(breach_cost=0.5).proliferation_cost(ep) - 0.70) < 1e-9
    # driver load = onc/TSG distinct drivers + one per activated dissemination trait
    gs = {"ploidy": 2, "highest_cn": 2, "nullisomy_count": 0, "n_mutated_drivers": 2,
          "n_mut_breach": 1, "n_mut_ss": 1, "n_mut_ms": 1, "n_mut_tr": 1}          # load = 2 + 4 = 6
    assert Selection(max_mut_drivers=6).update_viability(gs) == 1
    assert Selection(max_mut_drivers=5).update_viability(gs) == 0            # the super-clone dies
    assert Selection(max_mut_drivers=1000).update_viability(gs) == 1         # default ceiling: viable


def test_breach_gated_invasion_requires_breach():
    """breach_gated_invasion makes crossing the basement membrane (a duct->stroma dispersal hop) require
    the `breach` trait. With the gate ON and NO breach genes (prop_breach 0), cancer cannot reach the
    stroma at all; with it OFF, dispersal leaks cells across. Off by default => byte-identical otherwise."""
    spatial = {"grid_size": 22, "structure_radius": 2, "n_glands": 3, "gland_radius": 3,
               "min_gland_sep": 7, "K_duct": 24, "K_stroma": 16, "stroma_fill_frac": 0.3,
               "stromal_hazard": 0.6}
    sel = {**SELECTION_PARAMS, "prop_breach": 0.0}       # no breach genes -> breach trait always 0

    def stroma_cancer(gate):
        t = GenotypeTumor(seed=3, genome_params=GENOME_PARAMS, selection_params=sel,
                          cancer_cell_params=CANCER_CELL_PARAMS, deme_params=DEME,
                          spatial_params={**spatial, "breach_gated_invasion": gate},
                          update_mode="tau", tau=1.0)
        t.grow(n_steps=70, seed=3)
        return sum(c for di, d in enumerate(t.demes) if t.gland_id[di] < 0
                   for g, c in d.items() if t._is_cancer(g))

    assert stroma_cancer(gate=True) == 0      # the gate blocks invasion when no cell can breach
    assert stroma_cancer(gate=False) > 0      # without the gate, dispersal leaks cancer into the stroma


def test_landing_config_is_valid_and_schedule_shaped():
    """The committed configs/landing.yaml parses, encodes the full arc, and IS the tutorial's biology
    (notebooks/example_config.yaml) + metastasis + treatment on the SAME grid and seed."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = yaml.safe_load(open(os.path.join(root, "configs", "landing.yaml")))
    tut = yaml.safe_load(open(os.path.join(root, "notebooks", "example_config.yaml")))
    assert cfg["mode"] == "genotype" and cfg["update_mode"] == "tau" and cfg["tau"] == 0.5
    sch = cfg["schedule"]
    # min_freq is LOW (0.002) so the polyclonal invasion sweep keeps its own green Muller bands
    assert sch["seed"] == 2 and sch["min_freq"] == 0.002      # SAME seed as the tutorial
    ops = [p["op"] for p in sch["phases"]]
    assert ops == ["grow", "surgery", "chemotherapy", "grow"]
    # the shared biology is IDENTICAL to the tutorial's (grid, invasion, per-cell rates)
    assert cfg["spatial_params"]["grid_size"] == tut["spatial_params"]["grid_size"]
    assert cfg["selection_params"]["prop_breach"] == tut["selection_params"]["prop_breach"] == 0.02
    assert cfg["spatial_params"]["breach_gated_invasion"] is tut["spatial_params"]["breach_gated_invasion"] is True
    assert cfg["cell_params"]["cancer"] == tut["cell_params"]["cancer"]
    # R15 go-or-grow: the invasion costs + the driver-load ceiling are shared with the tutorial
    for k in ("breach_cost", "stromal_survival_cost", "max_mut_drivers"):
        assert cfg["selection_params"][k] == tut["selection_params"][k]
    # the landing-only ADDITIONS: metastasis + treatment (off in the tutorial) + the met compartment,
    # with a LOW met-survival cost (so the met establishes) and a HIGH resistance cost (rare pre-chemo)
    assert cfg["selection_params"]["prop_met_survival"] > 0 > -cfg["selection_params"]["prop_treatment_resistance"]
    assert 0 < cfg["selection_params"]["met_survival_cost"] < cfg["selection_params"]["treatment_resistance_cost"]
    assert cfg["spatial_params"]["met_hazard"] == 3.5 and cfg["spatial_params"]["met_grid_size"]
