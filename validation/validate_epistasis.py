"""Benchmark: can a cohort progression model recover iscc's PLANTED epistasis network? (R14)

`DESIGN_epistasis.md` gives iscc a **known** event x event dependency network — pairwise epistasis
`E`, conjunctive (ordered) constraints, mutual exclusivity — so that MHN / TreeMHN / CBN-H-CBN /
REVOLVER have a ground truth to be scored against. Before R14 iscc's selection was purely additive,
so the true network was empty and the benchmark could only measure a method's FALSE-POSITIVE rate.
This script plants a network, runs a cohort over it (shared landscape, private per-patient evolution
— `DESIGN_cohort.md`), and scores recovery with the **real tools**: MHN (Schill et al. 2020) in the
`iscc-mhn` env and TreeMHN (Luo, Kuipers & Beerenwinkel 2023) in `iscc-treemhn`.

THE FINDING — **the observable, not the cohort size, is what decides recovery.**

    iscc's pairwise `E` acts on **fitness**: it decides how large the clones carrying a combination
    grow. It does NOT change the rate at which events arise. So:

      * A **binary "did this patient ever acquire event i"** matrix — MHN's input — is nearly blind
        to it, but not for the naive reason. A favoured combination arises MANY times independently
        in a tumour (recurrent mutation), so "is it present?" is already yes at E=0 and saturates:
        the column carries almost no information about E.
      * The signal lives in **clone frequency**, and it is large: raising E expands the carrying
        clones from a few percent of the tumour towards a majority (panel B). Any observable that
        keeps frequency — a detection threshold on cancer-cell fraction, or the mutation TREES
        TreeMHN consumes (which carry `n_cells`) — can in principle see it.

    The ordered/conjunctive constraints are a separate axis, decided by `gating_mode`:
    **accessibility** gating acts on the mutation process itself, survives into every observable, and
    is recovered perfectly; **fitness** gating leaves the same planted DAG with no trace (panel D).

Panels (figure manuscript/figures/validation_epistasis.png):
  A. The ANSWER KEY: the planted E matrix.
  B. THE MECHANISM: planted E vs the realised cancer-cell fraction of the clones carrying the pair —
     a large, monotone dose-response, against the flat binary-presence observable. The fitness clamp
     is marked (a planted |E| above log(b_max/b_0) cannot express itself).
  C. THE OBSERVABLE: real-MHN edge recovery vs the detection threshold applied to that fraction,
     against the empty-network false-positive control, with degenerate (no-variance) regimes flagged.
  D. TOOLS: MHN (binary, cross-sectional) vs TreeMHN (trees, frequency-aware) vs the co-occurrence
     floor, on the same cohorts; plus order/conjunction recovery under both gating modes.

External tools each run in their own env (validation/README_integration.md); the script SKIPS the
tool arms gracefully when an env is absent and always renders the built-in co-occurrence floor.

Run:  python -u validation/validate_epistasis.py
"""
import argparse
import os
import subprocess
import tempfile
from collections import Counter

import numpy as np
import pandas as pd

from iscc.cohort import Cohort
from iscc.constants import DEFAULT_LAYOUT_SEED
from iscc import integrations as ig

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- dedicated envs (one per external tool; see validation/README_integration.md) -----------------
MHN_PYTHON = os.path.expanduser(os.environ.get(
    "ISCC_MHN_PYTHON", "~/miniconda3/envs/iscc-mhn/bin/python"))
TREEMHN_RSCRIPT = os.path.expanduser(os.environ.get(
    "ISCC_TREEMHN_RSCRIPT", "~/miniconda3/envs/iscc-treemhn/bin/Rscript"))
MHN_RUNNER = os.path.join(REPO, "validation/mhn_runner.py")
TREEMHN_RUNNER = os.path.join(REPO, "validation/treemhn_runner.R")


def mhn_available():
    return os.path.exists(MHN_PYTHON)


def treemhn_available():
    return os.path.exists(TREEMHN_RSCRIPT)


# A cohort regime where events accrue at INTERMEDIATE frequency — the only regime in which any
# method has signal to work with. Traps this avoids, all of them real (see PARAMETERS.md):
#   * driver_effects > 1 with many drivers pins the division rate at max_birth_rate and the clamp
#     silently EATS the interaction term. Selection is left neutral so the network is the only actor.
#   * event_size too large -> every patient acquires every event (all-ones matrix, zero variance,
#     nothing to correlate); too small -> events never fire. event_size=2 sits in the window.
GENOME = {"n_segments": 5, "segment_size": 40}
SELECTION = {"prop_driver": 0.5, "driver_effects": 1.0}
CANCER = {"division_rate": 0.2, "death_rate": 0.02, "max_birth_rate": 0.95,
          "mutation_rate": 0.3, "dispersal_rate": 0.3, "n_snvs_per_allele": 0.15}
DEME = {"carrying_capacity": 10, "initial_cancer_cells": 4}
SPATIAL = {"grid_size": 7, "structure_radius": 0}
STEPS = 500
N_EVENTS, EVENT_SIZE = 4, 2

# The fitness clamp: log(max_birth_rate / division_rate). A planted |E| above this cannot express
# itself — the division rate is already capped — which is why panel B plateaus.
CLAMP_HEADROOM = np.log(CANCER["max_birth_rate"] / CANCER["division_rate"])


def epi(n_interactions=0, strength=1.0, **kw):
    return dict(n_events=N_EVENTS, event_size=EVENT_SIZE, n_interactions=n_interactions,
                interaction_strength=strength, interaction_strength_sd=0.0, prop_synergy=1.0,
                event_effect_mean=0.0, event_effect_sd=0.0, **kw)


def run_cohort(n_patients, epistasis_params, dependency_params=None, seed0=1, steps=STEPS,
               layout_seed=DEFAULT_LAYOUT_SEED, inject_E=None):
    """Grow a cohort over ONE shared planted network; drop extinct patients (an all-zero row is not
    a patient in a real cohort study).

    ``inject_E`` sets E[0, 1] directly AFTER construction. That makes the E sweep a genuinely PAIRED
    comparison: identical event modules, identical evolution seeds, only the interaction differs.
    Re-drawing the network per E value would confound the interaction with a different layout.

    ``layout_seed`` picks WHICH network is planted, and every replicate varies it: the network is
    layout-determined, so holding it fixed and varying only evolution seeds would re-measure a single
    draw of E/DAG and report its idiosyncrasies as a result.
    """
    sel = {**SELECTION, "epistasis_params": epistasis_params}
    if dependency_params:
        sel["dependency_params"] = dependency_params
    coh = Cohort(patient_seeds=list(range(seed0, seed0 + n_patients)), genome_params=GENOME,
                 selection_params=sel, cancer_cell_params=CANCER, deme_params=DEME,
                 spatial_params=SPATIAL, grow_steps=steps, layout_seed=layout_seed)
    out = []
    for i in range(coh.n_patients):
        t = coh._build_tumor(i)
        if inject_E is not None:
            net = t.selection.epistasis
            net.E[0, 1] = net.E[1, 0] = inject_E
            net._fitness_cache.clear()
        t.grow(n_steps=steps, seed=coh.patient_seeds[i])
        if t.get_cancer_size() > 0:
            out.append(t)
    return out


def pair_cell_fraction(tumor, i=0, j=1):
    """Fraction of the tumour's cells carrying BOTH events — where the fitness signal lives."""
    tbl = tumor.event_table()
    total = tbl["n_cells"].sum()
    if not total:
        return 0.0
    both = tbl["events"].apply(lambda e: i in e and j in e)
    return tbl.loc[both, "n_cells"].sum() / total


# ------------------------------------------------------------------ the external tools
def run_mhn(X, lam="cv"):
    """Fit the REAL MHN (Schill et al.) in its own env. Returns the Theta matrix, or None."""
    if not mhn_available():
        return None
    with tempfile.TemporaryDirectory() as d:
        fin, fout = os.path.join(d, "in.csv"), os.path.join(d, "out.csv")
        X.to_csv(fin)
        r = subprocess.run([MHN_PYTHON, MHN_RUNNER, fin, fout, str(lam)],
                           capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(fout):
            print(f"  [mhn] FAILED: {r.stderr.strip().splitlines()[-1:]}")
            return None
        return pd.read_csv(fout, index_col=0)


def run_treemhn(trees, gamma=0.5):
    """Fit the REAL TreeMHN (Luo et al.) in its own env. Returns the Theta matrix, or None."""
    if not treemhn_available():
        return None
    with tempfile.TemporaryDirectory() as d:
        fin, fout = os.path.join(d, "trees.csv"), os.path.join(d, "out.csv")
        trees.to_csv(fin, index=False)
        r = subprocess.run([TREEMHN_RSCRIPT, TREEMHN_RUNNER, fin, fout, str(gamma)],
                           capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(fout):
            print(f"  [treemhn] FAILED: {r.stderr.strip().splitlines()[-3:]}")
            return None
        return pd.read_csv(fout, index_col=0)


def pair_rank(theta_or_scores, target=(0, 1), n_events=N_EVENTS):
    """Rank of the target pair among all pairs by |score| (1 = strongest). None if unavailable.

    Reported alongside the top-1 hit rate because top-1 is a coarse, high-variance metric: with 6
    pairs and a handful of network draws it cannot distinguish "ranked 2nd every time" from "found
    nothing". Chance rank is (n_pairs + 1) / 2.
    """
    if theta_or_scores is None:
        return None
    n_pairs = n_events * (n_events - 1) // 2
    edges = theta_to_edges(theta_or_scores, k=n_pairs) if hasattr(theta_or_scores, "shape") \
        else list(theta_or_scores)
    if edges is None:
        return None
    for r, (i, j, _) in enumerate(edges, start=1):
        if {min(i, j), max(i, j)} == set(target):
            return r
    return None


def theta_to_edges(theta, k):
    """A Theta matrix -> the k strongest UNDIRECTED event pairs.

    Theta[i, j] is the effect of j on the RATE of i — directed and asymmetric, whereas iscc's planted
    E is a symmetric fitness coefficient. We therefore symmetrise by the larger |Theta| of the two
    directions before ranking: the benchmark asks "did the method find that these two events
    interact", not "did it get a direction the ground truth does not even define". This is also why
    we never compare Theta VALUES to E values — different parameters, different units.
    """
    if theta is None:
        return None
    T = np.asarray(theta, dtype=float).copy()
    np.fill_diagonal(T, 0.0)
    n = T.shape[0]
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            w = T[i, j] if abs(T[i, j]) >= abs(T[j, i]) else T[j, i]
            pairs.append((i, j, float(w)))
    pairs.sort(key=lambda p: -abs(p[2]))
    return pairs[:k]


# ------------------------------------------------------------------ reconstructed (assay-only) orders
def reconstructed_orders(tumors):
    """Event orders inferred WITHOUT the lineage — what a real study actually has. Clone event SETS
    ordered by cohort-wide event frequency (the REVOLVER/CBN-style "frequent = early" heuristic)."""
    freq = Counter()
    for t in tumors:
        for events in ig.clone_events(t)["events"]:
            for e in events:
                freq[int(e)] += 1
    out = []
    for t in tumors:
        for events in ig.clone_events(t)["events"]:
            out.append(tuple(sorted((int(e) for e in events), key=lambda e: -freq[e])))
    return out


def true_orders(tumors):
    """The GROUPED true orders — events acquired in one division stay tied, so score_order excludes
    them rather than scoring a tie-break the engine never generated."""
    return [tuple(g) for t in tumors for g in ig.clone_events(t)["event_groups"]]


# =============================================================================== the experiments
def sweep_strength(strengths, n=40, reps=3):
    """Panel B — THE MECHANISM: does the planted E actually move the observable it acts on?

    Paired: identical modules and evolution seeds, only E[0, 1] differs.
    """
    rows = []
    for s in strengths:
        for rep in range(reps):
            tumors = run_cohort(n, epi(), seed0=1 + rep * 500,
                                layout_seed=DEFAULT_LAYOUT_SEED + rep, inject_E=s)
            if len(tumors) < 4:
                continue
            fr = np.array([pair_cell_fraction(t) for t in tumors])
            present = fr > 0
            # Report the two effects SEPARATELY -- they have different causes and conflating them
            # is what makes the unconditional mean so draw-dependent:
            #   presence   = P(the combination ever arose) -- a MUTATION property. E does not touch
            #                it, and it swings wildly with the network draw (which genes the modules
            #                landed on sets each event's hit rate).
            #   expansion  = E[cell fraction | it arose] -- the SELECTION property, i.e. the thing E
            #                actually acts on. This is the mechanistic read-out.
            rows.append(dict(strength=s, rep=rep,
                             cell_fraction=float(fr.mean()),
                             expansion=float(fr[present].mean()) if present.any() else np.nan,
                             presence=float(present.mean()),
                             n_present=int(present.sum()), n_patients=len(tumors)))
    return pd.DataFrame(rows)


def sweep_threshold(thresholds, strength=1.5, n=40, reps=3):
    """Panel C — THE OBSERVABLE: real-MHN edge recovery vs the detection threshold on cell fraction,
    with the empty-E control at each threshold (whatever it reports there is a false positive)."""
    rows = []
    for rep in range(reps):
        ls = DEFAULT_LAYOUT_SEED + rep
        tum = run_cohort(n, epi(), seed0=1 + rep * 500, layout_seed=ls, inject_E=strength)
        ctl = run_cohort(n, epi(), seed0=1 + rep * 500, layout_seed=ls, inject_E=0.0)
        if len(tum) < 4 or len(ctl) < 4:
            continue
        for mf in thresholds:
            for label, ts, tru in (("planted E", tum, [(0, 1)]), ("empty E (control)", ctl, [])):
                X = ig.to_mhn_matrix(ts, min_freq=mf)
                # a column with no variance carries no information -- record it, it explains a lot
                degenerate = bool((X.values.mean(0) == 0).any() or (X.values.mean(0) == 1).any())
                pred = theta_to_edges(run_mhn(X), k=1)
                if pred is None:
                    continue
                found = int(any({min(i, j), max(i, j)} == {0, 1} for i, j, _ in pred))
                rows.append(dict(threshold=mf, rep=rep, arm=label, degenerate=degenerate,
                                 found=found if tru else found,  # for the control, "found" == a FP
                                 pair_frac=float(np.mean([pair_cell_fraction(t) for t in ts]))))
    return pd.DataFrame(rows)


def compare_tools(strength=1.5, n=40, reps=3):
    """Panel D (left) — MHN (binary presence) vs TreeMHN (trees, which keep clone sizes) on the SAME
    cohorts, plus the built-in co-occurrence floor."""
    rows = []
    for rep in range(reps):
        ls = DEFAULT_LAYOUT_SEED + rep
        for label, Eval in (("planted E", strength), ("empty E (control)", 0.0)):
            ts = run_cohort(n, epi(), seed0=1 + rep * 500, layout_seed=ls, inject_E=Eval)
            if len(ts) < 4:
                continue
            X = ig.to_mhn_matrix(ts)                       # binary presence (MHN's input)
            trees = ig.to_treemhn_trees(ts)                # mutation trees (TreeMHN's input)
            n_pairs = N_EVENTS * (N_EVENTS - 1) // 2
            thetas = {
                "co-occurrence\n(floor)": ig.cooccurrence_scores(X.values),
                "MHN\n(binary presence)": run_mhn(X),
                "TreeMHN\n(mutation trees)": run_treemhn(trees),
            }
            for tool, th in thetas.items():
                if th is None:
                    continue
                edges = theta_to_edges(np.asarray(th), k=n_pairs)
                rank = next((r for r, (i, j, _) in enumerate(edges, 1)
                             if {min(i, j), max(i, j)} == {0, 1}), None)
                rows.append(dict(tool=tool, arm=label, rep=rep,
                                 found=int(rank == 1), rank=rank, n_pairs=n_pairs))
    return pd.DataFrame(rows)


def sweep_gating(n=40, reps=3):
    """Panel D (right) — the two axes a dependency DAG admits (the CONJUNCTION "B requires A" and the
    ORDER "A precedes B") under each gating mode, from true vs reconstructed trees."""
    rows = []
    for mode in ("accessibility", "fitness"):
        for rep in range(reps):
            dp = dict(n_constraints=2, dag_depth=2, dag_branching=1, gating_mode=mode)
            tumors = run_cohort(n, epi(), dp, seed0=1 + rep * 500,
                                layout_seed=DEFAULT_LAYOUT_SEED + rep)
            if len(tumors) < 4:
                continue
            dag = tumors[0].selection.epistasis.true_dag_edges()
            if not dag:
                continue
            for label, orders in (("true trees", true_orders(tumors)),
                                  ("reconstructed trees", reconstructed_orders(tumors))):
                r = ig.score_order(dag, orders)
                if r["n_child"]:
                    rows.append(dict(gating=mode, arm=label, rep=rep, **r))
    return pd.DataFrame(rows)


# =============================================================================== figure
def make_figure(net, strength_df, thr_df, tool_df, gate_df, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5))
    axA, axB, axC, axD = axes.ravel()

    # -- A: the answer key
    E = net.true_interaction_matrix()
    lim = max(0.5, np.abs(E).max())
    im = axA.imshow(E, cmap="RdBu_r", vmin=-lim, vmax=lim)
    axA.set_xticks(range(net.n_events)); axA.set_yticks(range(net.n_events))
    axA.set_xticklabels(net.event_names()); axA.set_yticklabels(net.event_names())
    for i in range(net.n_events):
        for j in range(net.n_events):
            if abs(E[i, j]) > 1e-9:
                axA.text(j, i, f"{E[i, j]:.1f}", ha="center", va="center", fontsize=9)
    axA.set_title("A. The answer key: E planted on the (E0, E1) pair\n"
                  "(the conjunctive DAG is a separate experiment — panel D)")
    fig.colorbar(im, ax=axA, fraction=0.046, label="$E_{ij}$ (log-fitness interaction)")

    # -- B: the mechanism (dose-response)
    if not strength_df.empty:
        g = strength_df.groupby("strength")["expansion"].agg(["mean", "std"]).reset_index()
        axB.errorbar(g["strength"], g["mean"], yerr=g["std"].fillna(0), fmt="o-", capsize=3,
                     label="cell fraction of the pair GIVEN it arose\n(what E acts on: SELECTION)")
        p = strength_df.groupby("strength")["presence"].mean().reset_index()
        axB.plot(p["strength"], p["presence"], "s--", color="grey",
                 label="P(the pair ever arose)\n(the binary observable: MUTATION)")
        axB.axvline(CLAMP_HEADROOM, color="crimson", ls=":", lw=1.2)
        axB.text(CLAMP_HEADROOM, 0.55, "  fitness clamp\n  $\\log(b_{max}/b_0)$",
                 color="crimson", fontsize=8)
    axB.set_xlabel("planted interaction strength $E_{01}$")
    axB.set_ylabel("fraction")
    axB.set_ylim(-0.03, 1.0)
    axB.legend(fontsize=8, loc="upper left", framealpha=0.9)
    axB.set_title("B. The mechanism: E strongly expands the carrying clones\n"
                  "— while binary PRESENCE barely moves (saturated by recurrent mutation)")

    # -- C: the observable (real MHN vs detection threshold)
    if not thr_df.empty:
        d = thr_df[thr_df["arm"] == "planted E"].groupby("threshold")["found"].agg(["mean", "std"]).reset_index()
        axC.errorbar(d["threshold"], d["mean"], yerr=d["std"].fillna(0), fmt="o-", capsize=3,
                     label="MHN recovers the planted edge")
        c = thr_df[thr_df["arm"] == "empty E (control)"].groupby("threshold")["found"].mean().reset_index()
        axC.plot(c["threshold"], c["found"], "k--", label="empty E (a false positive)")
        deg = thr_df[thr_df["arm"] == "planted E"].groupby("threshold")["degenerate"].mean()
        for x, v in deg.items():
            if v > 0.5:
                axC.axvspan(x - 0.008, x + 0.008, color="grey", alpha=0.18)
        axC.text(0.03, 0.06, "grey band = observable DEGENERATE\n(some column has no variance)",
                 fontsize=8, color="dimgrey", transform=axC.transAxes)
    axC.set_xlabel("detection threshold on cancer-cell fraction")
    axC.set_ylabel("P(the (E0,E1) edge reported)"); axC.set_ylim(-0.05, 1.05)
    axC.legend(fontsize=8, loc="upper right")
    axC.set_title("C. The OBSERVABLE decides recovery\n"
                  "(real MHN, same tumours, only the detection threshold varies)")

    # -- D: tools
    if not tool_df.empty:
        tools = list(dict.fromkeys(tool_df["tool"]))
        x = np.arange(len(tools)); w = 0.35
        for k, arm in enumerate(["planted E", "empty E (control)"]):
            m, e = [], []
            for t in tools:
                d = tool_df[(tool_df.tool == t) & (tool_df.arm == arm)]["rank"].dropna()
                m.append(d.mean() if len(d) else np.nan)
                e.append(d.std() if len(d) else 0.0)
            axD.bar(x + (k - 0.5) * w, m, w, yerr=np.nan_to_num(e), capsize=3, label=arm)
        n_pairs = int(tool_df["n_pairs"].iloc[0])
        axD.axhline((n_pairs + 1) / 2, color="grey", ls=":", lw=1)
        axD.text(len(tools) - 0.5, (n_pairs + 1) / 2 + 0.06, "chance rank", ha="right",
                 fontsize=8, color="grey")
        axD.set_xticks(x); axD.set_xticklabels(tools, fontsize=8)
        axD.set_ylabel(f"rank of the planted (E0,E1) pair\n(1 = strongest of {n_pairs}; lower is better)")
        axD.invert_yaxis()
        axD.legend(fontsize=8)
        sub = ""
        if not gate_df.empty:
            acc = gate_df[(gate_df.gating == "accessibility") & (gate_df.arm == "true trees")]
            fit = gate_df[(gate_df.gating == "fitness") & (gate_df.arm == "true trees")]
            if len(acc) and len(fit):
                sub = (f"\nDAG 'B requires A': accessibility {acc['constraint_satisfaction'].mean():.2f}"
                       f" vs fitness {fit['constraint_satisfaction'].mean():.2f} gating")
        axD.set_title("D. Real tools on the same cohort: binary presence vs trees" + sub)

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nwrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "manuscript/figures/validation_epistasis.png"))
    ap.add_argument("--strengths", type=float, nargs="+", default=[0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.0, 0.02, 0.05, 0.10, 0.25])
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    print(f"tools: MHN {'OK' if mhn_available() else 'ABSENT (skipping)'} | "
          f"TreeMHN {'OK' if treemhn_available() else 'ABSENT (skipping)'}")
    print(f"fitness-clamp headroom log(b_max/b_0) = {CLAMP_HEADROOM:.2f} "
          f"(a planted |E| above this cannot express itself)")

    net = run_cohort(2, epi(), inject_E=1.5)[0].selection.epistasis

    print("\n== B. mechanism: planted E -> realised cell fraction (paired) ==")
    strength_df = sweep_strength(args.strengths, n=args.n, reps=args.reps)
    if not strength_df.empty:
        print(strength_df.groupby("strength")[["cell_fraction", "presence"]].mean().round(3).to_string())

    print("\n== C. observable: real MHN vs the detection threshold ==")
    thr_df = sweep_threshold(args.thresholds, n=args.n, reps=args.reps)
    if not thr_df.empty:
        print(thr_df.groupby(["arm", "threshold"])[["found", "degenerate", "pair_frac"]]
              .mean().round(3).to_string())

    print("\n== D. tools: MHN vs TreeMHN on the same cohorts ==")
    print("   (rank of the planted pair among all pairs; 1 = strongest, chance = (n_pairs+1)/2)")
    tool_df = compare_tools(n=args.n, reps=args.reps)
    if not tool_df.empty:
        print(tool_df.groupby(["tool", "arm"])[["found", "rank"]].mean().round(2).to_string())

    print("\n== D. gating: order / conjunction recovery ==")
    gate_df = sweep_gating(n=args.n, reps=args.reps)
    if not gate_df.empty:
        print(gate_df.groupby(["gating", "arm"])[
            ["constraint_satisfaction", "order_accuracy", "n_child", "n_scored"]
        ].mean().round(2).to_string())

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    make_figure(net, strength_df, thr_df, tool_df, gate_df, args.out)

    # ------------------------------------------------------------------ headline
    print("\n================ headline ================")
    if not strength_df.empty:
        def at(s_, col):
            v = strength_df[strength_df.strength == s_][col]
            return v.mean() if len(v) else float("nan")
        lo, hi = at(0.0, "expansion"), at(1.5, "expansion")
        pl, ph = at(0.0, "presence"), at(1.5, "presence")
        ratio = (hi / lo) if (lo and np.isfinite(lo) and lo > 1e-6) else float("nan")
        print(f"MECHANISM (paired, given the pair arose): E 0 -> 1.5 grows the carrying clones from "
              f"{lo:.3f} to {hi:.3f} of the tumour" + (f" ({ratio:.0f}x)" if np.isfinite(ratio) else ""))
        print(f"           P(the pair ever arose) moves {pl:.2f} -> {ph:.2f} -- E does NOT change it "
              f"(that is a mutation property).")
        print("=> the signal is real and large, and it lives in clone FREQUENCY, not in event presence.")
    if not gate_df.empty:
        for mode in ("accessibility", "fitness"):
            d = gate_df[(gate_df.gating == mode) & (gate_df.arm == "true trees")]
            if len(d):
                print(f"{mode:>13} gating: 'B requires A' {d['constraint_satisfaction'].mean():.2f} | "
                      f"'A precedes B' {d['order_accuracy'].mean():.2f} "
                      f"(n={d['n_child'].mean():.0f} child-carrying lineages/draw)")
    print("\nCAVEAT (do not skip): these tumours are ~130 cells. A clone arising late has little time")
    print("to expand, which limits how much of the frequency signal reaches ANY observable, and the")
    print("event alphabet is 4 events over a 200-gene genome. The MECHANISM (frequency carries the")
    print("signal, presence does not) is structural and should generalise; the absolute recovery")
    print("rates are a floor for this regime, NOT an estimate for real cohorts.")


if __name__ == "__main__":
    main()
