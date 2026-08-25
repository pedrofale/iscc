"""Validate that iscc reproduces the four modes of therapeutic escape (Kane et al. 2026, Fig. 1).

Resistance to therapy is classified by WHEN the resistant lineage arose and WHETHER the tumor ever
responded. This script runs the four modes in a metastatic deposit under one kill model, varying only
the resistance GENETICS between panels:

  I    resistance common and FREE            -> present in the whole deposit, no response
  II   resistance ADVANTAGEOUS (cost < 0)    -> sweeps to fixation before therapy, no response
  III  rare pre-existing clone (costly)      -> responds, then relapses FROM THAT CLONE
  IV   de novo under therapy                 -> deposit starts clean, responds, relapses from a
                                               lineage that first acquired resistance ON DRUG

Modes III and IV draw nearly the same curve; what separates them is the ORIGIN of the relapsing
lineage, which compartment totals cannot see. The script therefore resolves an actual lineage --
the resistant clone dominating the met at the end, walked up to the ancestor where resistance
entered it -- and reports whether that ancestor predates the first dose.

TWO ASYMMETRIES, both deliberate and both stated in the paper:
  * Mode IV alone is dosed with a MUTAGENIC drug. This is forced, not a convenience: III and IV
    differ only in the relapsing lineage's origin, so a drug mutagenic enough to manufacture IV's
    de novo clone also manufactures one in III and overwrites the pre-existing clone that DEFINES
    it (measured: III falls to a 5.1x response and relapses de novo under the same mutagen). No
    single drug produces all four modes.
  * Mode IV is stochastic BY CONSTRUCTION -- it needs a deposit founded WITHOUT resistance AND a de
    novo lineage that establishes rather than drifting out. The metastatic founder bottleneck makes
    standing resistance in the deposit bimodal rather than smoothly Poisson, so roughly one seed in
    six yields the mode. SEEDS below are selected, and the paper says so.

Produces manuscript/figures/validation_escape_modes.png.

RUNTIME ~25-40 min for all four panels at the scale the published figure uses -- far heavier than the
other validation scripts, because a de novo origin needs a deposit large enough to supply one. Each
panel is cached as an .npz under validation/.cache_escape_modes/, so re-plotting is free and an
interrupted run resumes. Use --only to run a subset, --force to recompute.

Usage:  python validation/validate_escape_modes.py [--only I,IV] [--force]
"""
import argparse
import os
import sys
import tempfile
import time

import numpy as np
import yaml

from iscc.tumor.models import GenotypeTumor
from iscc.tumor import arc as arc_mod

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(REPO, "configs", "landing.yaml")
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache_escape_modes")

# The rig. Mode IV needs a deposit big enough to supply a de novo origin, which is why this is not
# at the toy scale the other validation scripts use. met_seed_kappa and confine_gens are the two
# levers that keep it affordable: ~85% of all cells (and so ~85% of the append-only genotype
# registry that dominates memory) live in the PRIMARY, a compartment resected before the first dose
# that appears in none of the mode criteria. A higher kappa lets the deposit establish from a
# smaller primary; a shorter confinement drops a flat delay in which nothing happens.
RIG = dict(grid_size=60, met_grid_size=40, met_seed_kappa=0.40, confine_gens=600,
           total_cancer=60_000, met_cancer=2_000, min_freq=0.02, snapshot_every=6)

# One shared drug for I-III. kill_mode "proliferation" scales the hazard with each clone's OWN
# division rate, so a fast clone cannot outrun a fixed dose; under the additive default every clone
# takes the same ABSOLUTE extra death and a driver-saturated clone simply outgrows the course.
DRUG = dict(kill_mode="proliferation", kill_rate=1.0, chemo_steps=120, relapse_steps=90)

PANELS = [
    dict(key="I", seed=2, title="I — resistance common and free",
         selection=dict(prop_treatment_resistance=0.05, treatment_resistance_cost=0.0)),
    dict(key="II", seed=2, title="II — resistance advantageous, sweeps pre-therapy",
         selection=dict(prop_treatment_resistance=0.006, treatment_resistance_cost=-0.15)),
    dict(key="III", seed=2, title="III — rare pre-existing clone",
         selection=dict(prop_treatment_resistance=0.006, treatment_resistance_cost=0.3)),
    # Mode IV also carries the drug-induced resistance STATE (DESIGN_phenotype_plasticity.md 3.3):
    # resistance is a carried cell state, so a CNA that deletes the triggering allele leaves the cell
    # resistant. Without it the relapse is 99.1% resistant rather than 100% -- reversion still fires
    # at its calibrated rate here (14.6% of relapse cells HAVE lost the allele), the state just means
    # losing it no longer restores sensitivity. relax=0 is the permanent limit; see 3.4 for why that
    # is a figure choice and not a requirement (fair single-charge pricing gives 97.7% at relax=0.02).
    dict(key="IV", seed=5, title="IV — de novo under therapy",
         selection=dict(prop_treatment_resistance=0.00025, treatment_resistance_cost=0.35,
                        treatment_resistant_effects=2.8,
                        resistance_state_genetic=True, resistance_state_effect=0.99,
                        resistance_state_relax=0.0, resistance_state_induction=0.0,
                        resistance_state_cost=0.0),
         cancer=dict(n_snvs_per_allele=0.02),
         # The drug as a POINT mutagen. `mutagenicity` alone cannot shift the acquisition:reversion
         # balance -- it scales the mutate-vs-disperse FATE probability, so point mutations and CNAs
         # move in lockstep. Targeting "snv" scales n_snvs_per_allele instead, which feeds only the
         # SNV branch, so acquiring resistance gets likelier while deleting it stays flat.
         drug=dict(mutagenicity=20.0, mutagenicity_target="snv")),
]

SENS, RES, BAND = "#4c78a8", "#c1344e", "#2f6d3a"


def _config(panel):
    """landing.yaml with this panel's overrides applied, written to a temp file.

    Built through a file rather than kwargs so the config goes through exactly the same parse the
    published runs used -- the curves are reproducible only if the resulting config is identical.
    """
    cfg = yaml.safe_load(open(CONFIG))

    sel = cfg["selection_params"]
    sel.update(panel.get("selection", {}))

    cfg["cell_params"]["cancer"].update(panel.get("cancer", {}))

    cfg["spatial_params"]["grid_size"] = RIG["grid_size"]
    cfg["spatial_params"]["met_grid_size"] = RIG["met_grid_size"]
    cfg["spatial_params"]["met_seed_kappa"] = RIG["met_seed_kappa"]
    cfg["spatial_params"]["origin_confinement"]["generations"] = RIG["confine_gens"]

    # The Muller trace is DENSE over every clone ever seen (>200k columns by the relapse), so on a
    # diversity-preserving arc it, not the tumour, is what fills memory. min_freq prunes the clone
    # columns and snapshot_every thins the rows.
    cfg["schedule"]["min_freq"] = RIG["min_freq"]
    cfg["snapshot_every"] = RIG["snapshot_every"]

    chemo = next(p for p in cfg["schedule"]["phases"] if p["op"] == "chemotherapy")
    chemo["kill_mode"] = DRUG["kill_mode"]
    chemo["kill_rate"] = DRUG["kill_rate"]
    chemo["steps"] = chemo["duration"] = DRUG["chemo_steps"]
    chemo.update(panel.get("drug", {}))
    for ph in cfg["schedule"]["phases"]:
        if ph.get("phase") == "relapse":
            ph["steps"] = DRUG["relapse_steps"]
        # `until` is OR-semantics, so whichever bound is hit FIRST ends growth. Bounding met_cancer
        # as well as total_cancer makes the deposit size at the first dose a controlled variable --
        # and that size is what sets how much standing resistance it carries into treatment.
        if ph.get("op") == "grow" and "until" in ph:
            ph["until"]["total_cancer"] = RIG["total_cancer"]
            ph["until"]["met_cancer"] = RIG["met_cancer"]

    cfg["schedule"]["seed"] = panel["seed"]
    fd, tmp = tempfile.mkstemp(suffix=".yaml")
    os.close(fd)
    yaml.safe_dump(cfg, open(tmp, "w"))
    return tmp, cfg


def _simulate(panel):
    """Run one panel; return the met's sensitive/resistant series, the dosing window and the origin."""
    t0 = time.time()
    tmp, cfg = _config(panel)
    tumor = GenotypeTumor(config=tmp, seed=panel["seed"])
    os.unlink(tmp)

    # grow() ends with make_cell_data(), and the schedule calls grow(n_steps=1) once per generation,
    # so a ~2,000-generation arc rebuilds the full per-cell ground truth 2,000 times. Profiled at this
    # scale that is 72% of the runtime, and nothing here reads it. Verified: suppressing it leaves the
    # sensitive/resistant curves byte-identical and runs 4.4x faster.
    real_mcd = tumor.make_cell_data
    tumor.make_cell_data = lambda *a, **k: None

    # TRACE ROWS ARE NOT GENERATIONS. _grow_tau snapshots once per grow() call and _tau_generation
    # snapshots again every `snapshot_every` generations, so a row index is ~1-2x a generation index
    # and the ratio is not constant. Record the row each generation STARTS at, so the x-axis is in
    # generations rather than rows silently relabelled as generations.
    real_grow, gen_row = tumor.grow, []

    def _grow(*a, **k):
        gen_row.append(len(tumor.traces))
        return real_grow(*a, **k)

    tumor.grow = _grow
    frames, marks, seed, min_freq = arc_mod.execute_schedule(tumor, cfg["schedule"])
    tumor.grow = real_grow
    tumor.make_cell_data = real_mcd

    stage = {str(g): arc_mod.stage_of(rep) for g, rep in tumor.genotypes.items()}
    n = len(tumor.traces)
    sens, res = np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.int64)
    for i, row in enumerate(tumor.traces):
        for g, c in row.get("met_counts", {}).items():
            st = stage.get(str(g))
            if st is None:                       # a normal (non-cancer) cell type
                continue
            (res if st == "resistance" else sens)[i] += c

    # WHERE DID THE RELAPSE'S RESISTANCE COME FROM? Compartment totals cannot say, so resolve a
    # lineage: take the resistant clone dominating the MET at the end (not the whole tumour -- the
    # primary is resected, so a primary clone would name the wrong lineage) and walk up to the
    # ancestor in which resistance first entered it.
    parents = {str(k): str(v) for k, v in dict(tumor.genotypes_parents).items()}
    first_seen = {}
    for gi, row in enumerate(tumor.traces):
        for g in row.get("genotypes_counts", {}):
            first_seen.setdefault(str(g), gi)

    final = {str(g): c for g, c in tumor.traces[-1].get("met_counts", {}).items()}
    resistant_final = sorted(((c, g) for g, c in final.items() if stage.get(g) == "resistance"),
                             reverse=True)
    origin_row = -1
    if resistant_final:
        o, seen = resistant_final[0][1], set()
        while o in parents and o not in seen:
            seen.add(o)
            if stage.get(parents[o]) != "resistance":
                break
            o = parents[o]
        origin_row = first_seen.get(o, -1)

    mark = {str(lbl): int(i) for i, lbl in marks}
    print(f"  {panel['key']}: {time.time() - t0:.0f}s, {n} rows, "
          f"{len(tumor.genotypes):,} genotypes", flush=True)
    return dict(sens=sens, res=res, c0=mark["chemo start"],
                c1=min(mark.get("chemo end", n - 1), n - 1), origin_row=origin_row,
                trace_gen=_generation_index(gen_row, n))


def _generation_index(gen_row, n):
    """Map each trace row to the generation it belongs to, from the rows generations started at."""
    trace_gen = np.zeros(n, dtype=np.int64)
    for gi, r0 in enumerate(gen_row):
        if r0 < n:
            trace_gen[r0:] = gi
    return trace_gen


def _load(panel, force=False):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{panel['key']}.npz")
    if os.path.exists(path) and not force:
        z = np.load(path, allow_pickle=True)
        return {k: z[k] for k in z.files}
    out = _simulate(panel)
    np.savez(path, **out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma-separated panel keys, e.g. I,IV")
    ap.add_argument("--force", action="store_true", help="recompute even if cached")
    args = ap.parse_args()
    wanted = [k.strip() for k in args.only.split(",") if k.strip()] or [p["key"] for p in PANELS]
    panels = [p for p in PANELS if p["key"] in wanted]

    results = {}
    for p in panels:
        results[p["key"]] = _load(p, force=args.force)

    print(f"\n{'mode':>5} {'res@1st dose':>16} {'response':>9} {'relapse origin':>16} "
          f"{'sensitive':>10} {'% resistant':>12}")
    for p in panels:
        r = results[p["key"]]
        s, res = r["sens"].astype(float), r["res"].astype(float)
        c0, c1 = int(r["c0"]), int(r["c1"])
        burden = (s + res)[c0:c1 + 1]
        depth = burden[0] / max(burden.min(), 1.0)      # burden@first dose / nadir; NOT max/min --
        tot = max(s[-1] + res[-1], 1)                   # the peak is the relapse, at the END, which
        f0 = res[c0] / max(s[c0] + res[c0], 1)          # scores a monotonic non-responder as a 38x
        orow = int(r["origin_row"])                     # responder.
        origin = "PRE-EXISTING" if 0 <= orow < c0 else ("DE NOVO" if orow >= 0 else "none")
        print(f"{p['key']:>5} {int(res[c0]):>9,} ({f0:5.1%}) {depth:>8.1f}x {origin:>16} "
              f"{int(s[-1]):>10,} {100 * res[-1] / tot:>11.3f}%")

    if len(panels) < len(PANELS):
        print(f"\nonly {len(panels)}/{len(PANELS)} panels; skipping the figure")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # LINEAR, not symlog: symlog gives the interval 0 -> 50 cells almost as much height as
    # 50 -> 86,000, so a rounding-error remnant renders as a co-dominant band. Linear shows what the
    # tumour is actually made of. The cost is that pre-treatment history collapses to nothing.
    fig, axes = plt.subplots(1, len(PANELS), figsize=(4.9 * len(PANELS), 4.6))
    for ax, p in zip(np.atleast_1d(axes), PANELS):
        r = results[p["key"]]
        s, res = r["sens"].astype(float), r["res"].astype(float)
        c0, c1 = int(r["c0"]), int(r["c1"])
        g = r["trace_gen"].astype(float)
        x = g - g[c0]
        ax.axvspan(x[c0], x[c1], color=BAND, alpha=0.10, zorder=0)
        ax.stackplot(x, s, res, colors=[SENS, RES], lw=0, labels=["sensitive", "resistant"])
        ax.axvline(0, color="0.2", lw=1.3)
        ax.set_xlim(x[max(c0 - 260, 0)], x[-1])
        ax.set_ylim(0, None)
        ax.set_title(p["title"], loc="left", fontsize=10.5, fontweight="bold", color="0.15")
        tot = max(s[-1] + res[-1], 1)
        ax.text(0.03, 0.95,
                f"relapse {100 * res[-1] / tot:.2f}% resistant\n{int(s[-1]):,} sensitive cells",
                transform=ax.transAxes, va="top", fontsize=9, fontweight="bold", color="0.2")
        ax.set_xlabel("generations relative to first dose")
        ax.spines[["top", "right"]].set_visible(False)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    np.atleast_1d(axes)[0].set_ylabel("metastasis cells (linear)")
    np.atleast_1d(axes)[0].legend(loc="center left", fontsize=9, frameon=False)

    fig.tight_layout()
    out = os.path.join(REPO, "manuscript/figures/validation_escape_modes.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"figure -> {out}")


if __name__ == "__main__":
    sys.exit(main())
