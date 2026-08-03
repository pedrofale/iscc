"""Config-driven CLINICAL ARC for the metastasis module (R9): a timed treatment schedule that grows a
primary + metastasis through the full clinical story and captures a compartment TRAJECTORY that
``isccgif --compartment`` renders offline (grids + centered Mullers on one shared clone colormap).

This promotes the arc + shared-clone-colouring logic that used to live only in the bespoke
``notebooks/landing_animation.py`` into the package, so the landing hero is reproducible from the
ordinary CLI:

    isccsim --sim-config configs/landing.yaml -o out
    isccgif out --compartment --splash -o docs/assets/landing_hero.gif

The schedule (under the ``schedule:`` key of a sim-config) is a list of PHASES executed in order:

    schedule:
      seed: 3                 # evolution seed for the whole arc (overrides --random-seed)
      min_freq: 0.02          # clone size-merge threshold (grids + Mullers must agree)
      capture: {pre_seed_every: 2, post_seed_every: 1}
      phases:
        - {op: grow, until: {met_cancer: 800, total_cancer: 11000},
           phase: growth, seeded_phase: metastatic seeding}
        - {op: surgery, site: primary, settle: 3, phase: resection}
        - {op: chemotherapy, steps: 16, duration: 16, kill_rate: 2.8, effectiveness: 0.98,
           toxicity: 0.12, sites: both, phase: chemotherapy}
        - {op: grow, steps: 26, phase: relapse}

Each captured frame is a full per-compartment deme-grid snapshot + the reveal cursor (trace length) +
its phase; the clinical events (seeding / resection / chemo start+end) are recorded as trace-index
marks. Everything the renderer needs (frames, traces, genealogy, the shared ``gid -> rgba`` colormap,
the functional-clone ``driver_map``) is bundled into ONE pickle so the GIF renders with no engine.
"""
import os
import pickle

from .. import treatment as _treatment
from ..constants import normal_names


# --- shared clone colouring (identical in the grids and the Mullers) ----------------------------
# The four selective sweeps of the metastatic arc + proliferation + none. A clone is coloured by its
# STAGE-DOMINANT selective trait (the last sweep it activated), the SAME colour in the primary grid,
# the met grid, and both Muller bands.
STAGE_COL = {
    "none":       (0.50, 0.53, 0.60, 1.0),   # grey
    "prolif":     (0.23, 0.51, 0.84, 1.0),   # blue   — proliferation (onc/TSG)
    "breach":     (0.98, 0.62, 0.09, 1.0),   # orange — escape the duct (breach)
    "stromal":    (0.18, 0.75, 0.44, 1.0),   # green  — survive the stroma
    "met":        (0.68, 0.40, 0.86, 1.0),   # purple — establish the metastasis
    "resistance": (0.94, 0.24, 0.28, 1.0),   # red    — escape chemo (resistance)
}
STAGE_ORDER = ["prolif", "breach", "stromal", "met", "resistance"]
STAGE_LABEL = {"prolif": "proliferation", "breach": "duct escape", "stromal": "stromal survival",
               "met": "met establishment", "resistance": "chemo resistance"}
# muted normal tissue: barely above the dark page colour so the cancer pops and the grids blend into
# the splash background (no filled-square seam).
NORMAL_MUTED = {"epithelial": (0.15, 0.29, 0.23, 1.0), "stromal": (0.18, 0.15, 0.18, 1.0),
                "host": (0.13, 0.16, 0.21, 1.0), "immune": (0.24, 0.22, 0.11, 1.0)}

# per-phase caption for the fully-labelled (non-splash) variant
PHASE_CAPTION = {
    "growth": "① proliferation in the ducts (DCIS)",
    "metastatic seeding": "①→② breach the duct wall, survive the stroma, seed the met (IDC)",
    "resection": "the primary is surgically resected",
    "chemotherapy": "systemic chemotherapy — the sensitive metastasis regresses",
    "relapse": "④ resistant clones relapse the metastasis",
}

TRAJECTORY_FILE = "compartment_trajectory.pkl"


def stage_of(rep):
    """The cascade stage of a cancer clone: which trait MUTATIONS it carries, taken as the furthest
    along the metastatic arc — resistance > met > stromal > breach > proliferation > none. ``None`` for
    normal cells.

    No value thresholds: a clone is at a stage iff it carries ≥1 mutation giving that trait (the trait
    value is >0 iff a gene for it is mutated). A cell that carries several trait mutations is coloured by
    the last one in the arc. (Because a clone can carry multiple traits at once — asexual linked
    selection means a sweeping clone hitchhikes them together — the same clone can be, e.g., both
    met-survival and resistance; it then reads as the later stage, resistance.)"""
    if getattr(rep, "type", None) != "cancer":
        return None
    ep = rep.evolutionary_parameters
    base_div = getattr(rep, "baseline_rates", {}).get("division_rate", ep["division_rate"])
    if ep.get("treatment_resistance", 0) > 0:
        return "resistance"
    if ep.get("met_survival", 0) > 0:
        return "met"
    if ep.get("stromal_survival", 0) > 0:
        return "stromal"
    if ep.get("breach", 0) > 0:
        return "breach"
    if ep["division_rate"] > base_div + 1e-9:
        return "prolif"
    return "none"


def story_colors(t, min_freq=None):
    """{str(gid) -> rgba}: ONE shared clone colormap for the grids AND the Mullers.

    PER-CELL colouring: every cancer genotype takes the colour of ITS OWN cascade stage (``stage_of`` —
    the trait MUTATIONS it carries: green=stromal, purple=met, red=resistance, blue=proliferation,
    grey=no trait), so a clone shows the trait it ACTUALLY HAS in the primary grid, the met grid, and the
    Muller alike. This is the "colour cells/clones by whether they carry each trait's mutations" view: a
    small breach/stromal/… sub-clone keeps its own trait colour instead of being folded (band-dominant or
    by founder) into a bulk that is mostly stage 'none' and washed grey. ``min_freq`` is accepted for
    call-site compatibility but unused (no band merging happens in the colour map). Normals stay muted."""
    out = {}
    for gid, rep in t.genotypes.items():
        sgid = str(gid)
        if getattr(rep, "type", None) != "cancer":
            out[sgid] = NORMAL_MUTED.get(getattr(rep, "type", None), (0.7, 0.7, 0.7, 1.0))
            continue
        st = stage_of(rep)
        out[sgid] = STAGE_COL["none" if st is None else st]
    for n in normal_names:
        out[n] = NORMAL_MUTED[n]
    return out


def _compartment_cancer(t):
    """(primary_cancer, met_cancer) live cancer-cell counts, split by compartment."""
    p = sum(c for i in range(t.n_primary_demes) for g, c in t.demes[i].items() if t._is_cancer(g))
    m = sum(c for i in range(t.n_primary_demes, len(t.demes)) for g, c in t.demes[i].items()
            if t._is_cancer(g))
    return p, m


def execute_schedule(t, schedule):
    """Run the timed treatment schedule on tumor ``t``, capturing a compartment trajectory.

    Returns ``(frames, marks, seed, min_freq)`` where ``frames`` is a list of
    ``dict(demes=[{gid: count}, ...], cursor=int, phase=str)`` (one per captured generation) and
    ``marks`` is a list of ``(trace_index, label)`` clinical-event marks. The dynamics ALWAYS advance
    one generation per ``grow`` call (calibration-stable), exactly as the reference arc; the capture
    cadence is a cosmetic trim, never a dynamics change."""
    seed = int(schedule.get("seed", t.seed))
    min_freq = float(schedule.get("min_freq", 0.02))
    cap = schedule.get("capture", {}) or {}
    pre_every = int(cap.get("pre_seed_every", 1))
    post_every = int(cap.get("post_seed_every", 1))
    phases = schedule.get("phases", [])

    frames, marks = [], []
    seeding_snap = None
    # mark "seeding" when the met is ESTABLISHED (has reached this many cancer cells), not when the FIRST
    # cell arrives -- early seeds often land and die under the met-host hazard; the indicator should point
    # at the seed that actually sticks and grows the deposit. Default 1 = the first cell (old behaviour).
    established_at = int(schedule.get("seeding_establishes_at", 1))

    def capture(phase):
        frames.append(dict(demes=[dict(d) for d in t.demes], cursor=len(t.traces), phase=phase))

    def note_seeding():
        nonlocal seeding_snap
        if seeding_snap is None and _compartment_cancer(t)[1] >= established_at:
            seeding_snap = len(t.traces)
            marks.append((seeding_snap, "seeding"))

    first = True
    for spec in phases:
        op = spec.get("op", "grow")
        if op == "grow":
            label = spec.get("phase", "growth")
            seeded_label = spec.get("seeded_phase")
            until = spec.get("until")
            n_steps = spec.get("steps")
            if first:
                capture(label)                       # the initial (t=0) frame
                first = False
            gi = 0
            while True:
                if until is not None:
                    p, m = _compartment_cancer(t)
                    stop = (("met_cancer" in until and m >= until["met_cancer"])
                            or ("total_cancer" in until and p + m >= until["total_cancer"])
                            or ("primary_cancer" in until and p >= until["primary_cancer"]))
                    if stop:
                        break
                elif n_steps is not None:
                    if gi >= n_steps:
                        break
                else:
                    break
                t.grow(n_steps=1, seed=seed)
                note_seeding()
                gi += 1
                do_cap = ((seeding_snap is not None and gi % post_every == 0)
                          or (seeding_snap is None and gi % pre_every == 0))
                if do_cap:
                    lbl = seeded_label if (seeding_snap is not None and seeded_label) else label
                    capture(lbl)
        elif op == "surgery":
            first = False
            label = spec.get("phase", "resection")
            site = spec.get("site", "primary")
            settle = int(spec.get("settle", 0))
            marks.append((len(t.traces), "resection"))
            t.grow(n_steps=1, seed=seed,
                   treatment=_treatment.Surgery(start=t.step, site=site))
            capture(label)
            for _ in range(settle):
                t.grow(n_steps=1, seed=seed)
                capture(label)
        elif op in ("chemotherapy", "chemo"):
            first = False
            label = spec.get("phase", "chemotherapy")
            n_steps = int(spec.get("steps", spec.get("duration", 1)))
            marks.append((len(t.traces), "chemo start"))
            chemo = _treatment.Chemotherapy(
                start=t.step, duration=spec.get("duration"),
                kill_rate=float(spec.get("kill_rate", 1.5)),
                effectiveness=float(spec.get("effectiveness", 0.9)),
                toxicity=float(spec.get("toxicity", 0.1)),
                sites=spec.get("sites", "both"))
            for _ in range(n_steps):
                t.grow(n_steps=1, seed=seed, treatment=chemo)
                capture(label)
            marks.append((len(t.traces), "chemo end"))
        else:
            raise ValueError(f"unknown schedule op {op!r} (expected grow / surgery / chemotherapy)")

    return frames, marks, seed, min_freq


def build_trajectory(t, frames, marks, min_freq):
    """Bundle everything ``isccgif --compartment`` needs into ONE self-contained, engine-free dict.

    The per-frame deme snapshots + the ``gid -> creation-ordinal`` map let the renderer rebuild each
    grid's cells in the exact order the engine materialises them; ``colors`` (the shared stage-dominant
    clone map) + ``driver_map`` + ``min_freq`` reproduce the shared palette across the grids and both
    centered Mullers."""
    return dict(
        frames=frames,
        marks=[(int(i), str(l)) for i, l in marks],
        traces=t.traces,
        genotypes_parents={str(k): str(v) for k, v in dict(t.genotypes_parents).items()},
        grid_size=int(t.grid_size),
        met_grid_size=int(t.met_grid_size),
        n_primary_demes=int(t.n_primary_demes),
        deme_coords=[(int(r), int(c)) for r, c in t.deme_coords],
        gid_ord={str(g): int(rep.ord) for g, rep in t.genotypes.items()},
        colors={str(g): tuple(float(x) for x in c) for g, c in story_colors(t, min_freq).items()},
        driver_map={str(g): tuple(int(i) for i in v)
                    for g, v in t._functional_signatures().items()},
        min_freq=float(min_freq),
        stage_order=list(STAGE_ORDER),
        stage_legend=[(STAGE_LABEL[s], STAGE_COL[s]) for s in STAGE_ORDER],
        phase_caption=dict(PHASE_CAPTION),
    )


def write_trajectory(output_path, trajectory):
    """Persist the trajectory pickle under ``output_path`` (the isccsim output dir)."""
    os.makedirs(output_path, exist_ok=True)
    path = os.path.join(output_path, TRAJECTORY_FILE)
    with open(path, "wb") as f:
        pickle.dump(trajectory, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def read_trajectory(path):
    """Load a trajectory pickle. ``path`` may be the isccsim output dir OR the pickle file itself."""
    if os.path.isdir(path):
        path = os.path.join(path, TRAJECTORY_FILE)
    with open(path, "rb") as f:
        return pickle.load(f)
