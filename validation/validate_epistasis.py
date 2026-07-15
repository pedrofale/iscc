"""Benchmark: can a cohort progression model recover iscc's PLANTED epistasis network? (R14)

`DESIGN_epistasis.md` gives iscc a **known** event x event dependency network — pairwise epistasis
`E`, conjunctive (ordered) constraints, mutual exclusivity — so that MHN / TreeMHN / CBN-H-CBN /
REVOLVER have a ground truth to be scored against. Before R14 iscc's selection was purely additive,
so the true network was empty and the benchmark could only measure a method's FALSE-POSITIVE rate.
This script plants a network, runs a cohort over it (shared landscape, private per-patient evolution
— `DESIGN_cohort.md`), and scores recovery against the answer key.

THE HEADLINE FINDING — and it is a NEGATIVE one, reported as such:

    iscc's pairwise `E` acts on **fitness** (how large a clone grows). MHN/CBN model the **rate of
    event acquisition**. These are different generative stories, and a cross-sectional "which events
    did this patient ever acquire" matrix is largely BLIND to the difference: whether a clone is 1%
    or 60% of the tumour, the event is equally "present". So fitness epistasis is recovered barely
    above the empty-network false-positive floor (panel B), no matter how strong it is planted
    (panel C) — until a DETECTION THRESHOLD is imposed, which is the only channel through which
    "grew bigger" becomes "was observed".

    The ordered/conjunctive constraints under ACCESSIBILITY gating are a different story: they act
    on the mutation process itself, so they survive into both the cross-sectional matrix (as a hard
    implication child => parent) and the mutation trees, and are recovered near-perfectly (panel D).

That contrast is the useful result for the paper: it says *which* planted structure these tools can
see, and it is only visible because iscc knows the answer. Do not read panel B as "MHN is bad" — read
it as "cross-sectional event presence is the wrong observable for fitness epistasis".

Panels (figure manuscript/figures/validation_epistasis.png):
  A. The ANSWER KEY: the planted E matrix + the conjunctive DAG.
  B. COHORT SIZE sweep: edge precision/recall for fitness-gated pairwise E at two detection
     thresholds, against the EMPTY-network false-positive floor (the sanity control).
  C. INTERACTION STRENGTH sweep: recovery vs |E| — flat, which is the point.
  D. ORDER recovery: accessibility vs fitness gating, on TRUE trees vs RECONSTRUCTED trees (does
     tree-inference error destroy the progression signal? — an iscc-only question).

The scoring seam is `iscc.integrations.progression` (`to_mhn_matrix` / `to_treemhn_trees` /
`score_edges` / `score_order`), so a real MHN/TreeMHN run in its own `iscc-mhn` / `iscc-treemhn` env
plugs into the same scoring. The built-in co-occurrence baseline (the DISCOVER/MEGSA log-odds
statistic) is used here so the figure renders without R; it is the FLOOR, not a stand-in for MHN.

Run:  python -u validation/validate_epistasis.py
"""
import argparse
import os
from collections import Counter

import numpy as np
import pandas as pd

from iscc.cohort import Cohort
from iscc.constants import DEFAULT_LAYOUT_SEED
from iscc import integrations as ig

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# A cohort regime where events actually accrue at INTERMEDIATE frequency across patients — the only
# regime in which any cross-sectional method has signal to work with. Two traps this avoids, both
# worth knowing about (they are in PARAMETERS.md):
#   * driver_effects > 1 with many drivers pins the division rate at max_birth_rate, and the clamp
#     silently EATS the interaction term. Selection is left neutral so the network is the only actor.
#   * a high mutation rate saturates every event in every patient (all-ones matrix -> zero variance
#     -> nothing to correlate), while a low one never fires them at all.
GENOME = {"n_segments": 5, "segment_size": 40}
SELECTION = {"prop_driver": 0.5, "driver_effects": 1.0}
CANCER = {"division_rate": 0.2, "death_rate": 0.02, "max_birth_rate": 0.95,
          "mutation_rate": 0.3, "dispersal_rate": 0.3, "n_snvs_per_allele": 0.15}
DEME = {"carrying_capacity": 10, "initial_cancer_cells": 4}
SPATIAL = {"grid_size": 7, "structure_radius": 0}
STEPS = 500

N_EVENTS = 5
EVENT_SIZE = 6


def run_cohort(n_patients, epistasis_params, dependency_params=None, seed0=1, steps=STEPS,
               layout_seed=DEFAULT_LAYOUT_SEED):
    """Grow a cohort over ONE shared planted network; drop extinct patients (an extinct tumour is an
    all-zero row, and is not a patient in a real cohort study).

    ``layout_seed`` picks WHICH network is planted. Every replicate below varies it, because the
    network is layout-determined: holding it fixed and varying only the evolution seeds would
    re-measure a single draw of E/DAG over and over and report its idiosyncrasies as a result. Event
    acquisition rates are strongly heterogeneous (an emergent consequence of the CNA landscape — a
    module on a frequently-amplified segment is hit more often, since SNVs are drawn per allele), so
    for any ONE dependency DAG the order metric is largely decided by whether the parent happens to
    be the faster event. Only averaging over networks measures the method rather than the draw.
    """
    sel = {**SELECTION, "epistasis_params": epistasis_params}
    if dependency_params:
        sel["dependency_params"] = dependency_params
    coh = Cohort(patient_seeds=list(range(seed0, seed0 + n_patients)), genome_params=GENOME,
                 selection_params=sel, cancer_cell_params=CANCER, deme_params=DEME,
                 spatial_params=SPATIAL, grow_steps=steps, layout_seed=layout_seed)
    return [p.tumor for p in coh.run().patients if p.tumor.get_cancer_size() > 0]


def epi(n_interactions=2, strength=1.0, **kw):
    return dict(n_events=N_EVENTS, event_size=EVENT_SIZE, n_interactions=n_interactions,
                interaction_strength=strength, interaction_strength_sd=0.05, prop_synergy=1.0,
                event_effect_mean=0.0, event_effect_sd=0.0, **kw)


def recover_edges(tumors, k, min_freq=0.0):
    """The cross-sectional baseline: rank event pairs by |log odds ratio| and take the top k."""
    X = ig.to_mhn_matrix(tumors, min_freq=min_freq)
    S = ig.cooccurrence_scores(X.values)
    return ig.top_edges(S, k=k), S, X


# --------------------------------------------------------------- reconstructed (assay-only) orders
def reconstructed_orders(tumors, min_freq=0.0):
    """Event orders inferred WITHOUT the lineage — the information a real study actually has.

    A real cohort sees each tumour's clones and their event SETS, not the order they arose in. The
    standard heuristic (REVOLVER/CBN-style) is that a more frequent event is the earlier one, so
    each clone's set is ordered by cohort-wide event frequency. Scoring this against iscc's true
    lineage order measures how much of the progression signal survives inference — a question only a
    simulator that knows both can ask.
    """
    freq = Counter()
    for t in tumors:
        for events in ig.clone_events(t, min_freq=min_freq)["events"]:
            for e in events:
                freq[int(e)] += 1
    out = []
    for t in tumors:
        for events in ig.clone_events(t, min_freq=min_freq)["events"]:
            out.append(tuple(sorted((int(e) for e in events), key=lambda e: -freq[e])))
    return out


def true_orders(tumors, min_freq=0.0):
    """The GROUPED true orders — events acquired in one division stay tied, so score_order can
    exclude them instead of scoring a tie-break the engine never generated."""
    return [tuple(g) for t in tumors for g in ig.clone_events(t, min_freq=min_freq)["event_groups"]]


# =============================================================================== the experiments
def sweep_cohort_size(sizes, strength=1.0, reps=3):
    """Panel B: does pooling more patients recover fitness-gated pairwise E? Against the empty-E
    floor (whatever the method reports there is by construction a false positive)."""
    rows = []
    for n in sizes:
        for rep in range(reps):
            ls = DEFAULT_LAYOUT_SEED + rep      # a different planted network per replicate
            ep = epi(n_interactions=2, strength=strength)
            tumors = run_cohort(n, ep, seed0=1 + rep * 1000, layout_seed=ls)
            if len(tumors) < 4:
                continue
            net = tumors[0].selection.epistasis
            truth = net.true_edges()
            for mf, label in ((0.0, "all clones"), (0.10, "detectable clones (>=10%)")):
                pred, _, _ = recover_edges(tumors, k=len(truth), min_freq=mf)
                r = ig.score_edges(truth, pred)
                rows.append(dict(n=n, rep=rep, arm=label, **r))
            # the EMPTY-network control: same regime, no planted edges
            ctrl_tumors = run_cohort(n, epi(n_interactions=0), seed0=1 + rep * 1000, layout_seed=ls)
            if len(ctrl_tumors) >= 4:
                pred, _, _ = recover_edges(ctrl_tumors, k=len(truth), min_freq=0.0)
                r = ig.score_edges(ctrl_tumors[0].selection.epistasis.true_edges(), pred)
                rows.append(dict(n=n, rep=rep, arm="empty E (control)", **r,))
    return pd.DataFrame(rows)


def sweep_strength(strengths, n=30, reps=3):
    """Panel C: recovery vs how hard the network is planted."""
    rows = []
    for s in strengths:
        for rep in range(reps):
            tumors = run_cohort(n, epi(n_interactions=2, strength=s), seed0=1 + rep * 1000,
                                layout_seed=DEFAULT_LAYOUT_SEED + rep)
            if len(tumors) < 4:
                continue
            net = tumors[0].selection.epistasis
            truth = net.true_edges()
            pred, _, _ = recover_edges(tumors, k=len(truth), min_freq=0.0)
            rows.append(dict(strength=s, rep=rep, **ig.score_edges(truth, pred)))
    return pd.DataFrame(rows)


def sweep_gating(n=40, reps=3):
    """Panel D: the two axes a dependency DAG can be scored on — the CONJUNCTION ("B requires A")
    and the ORDER ("A came first") — under each gating mode, from true vs reconstructed trees.

    Uses a SHORTER, wider event alphabet than panels B/C (4 events of 8 genes, not 5 of 6). Under
    accessibility gating a gated child needs its parent first, so acquiring it is a two-step event:
    with the panel-B alphabet only a couple of lineages per cohort ever get there, and an accuracy
    computed on n=2 is not a result. Watch ``n_child`` in the printout — it is the power this panel
    actually has.
    """
    rows = []
    for mode in ("accessibility", "fitness"):
        for rep in range(reps):
            ep = dict(epi(n_interactions=0), n_events=4, event_size=8)
            dp = dict(n_constraints=2, dag_depth=2, dag_branching=1, gating_mode=mode)
            tumors = run_cohort(n, ep, dp, seed0=1 + rep * 1000,
                                layout_seed=DEFAULT_LAYOUT_SEED + rep)
            if len(tumors) < 4:
                continue
            dag = tumors[0].selection.epistasis.true_dag_edges()
            for label, orders in (("true trees", true_orders(tumors)),
                                  ("reconstructed trees", reconstructed_orders(tumors))):
                r = ig.score_order(dag, orders)
                if r["n_child"]:
                    rows.append(dict(gating=mode, arm=label, rep=rep, **r))
    return pd.DataFrame(rows)


# =============================================================================== figure
def make_figure(net, size_df, strength_df, gate_df, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
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
                axA.text(j, i, f"{E[i, j]:.1f}", ha="center", va="center", fontsize=8)
    axA.set_title("A. The answer key: the planted E of panels B/C\n"
                  "(the conjunctive DAG is a separate experiment — panel D)")
    fig.colorbar(im, ax=axA, fraction=0.046, label="$E_{ij}$ (log-fitness interaction)")

    # -- B: cohort size
    for arm, style in (("all clones", "o-"), ("detectable clones (>=10%)", "s-"),
                       ("empty E (control)", "k--")):
        d = size_df[size_df["arm"] == arm]
        if d.empty:
            continue
        g = d.groupby("n")["recall"].agg(["mean", "std"]).reset_index()
        axB.errorbar(g["n"], g["mean"], yerr=g["std"].fillna(0), fmt=style, capsize=3, label=arm)
    n_pairs = net.n_events * (net.n_events - 1) // 2
    chance = len(net.true_edges()) / n_pairs   # take top-k of n_pairs at random -> E[recall] = k/n_pairs
    axB.axhline(chance, color="grey", ls=":", lw=1)
    axB.text(size_df["n"].max(), chance + 0.02, f"chance (top {len(net.true_edges())} of {n_pairs} pairs)",
             ha="right", fontsize=8, color="grey")
    axB.set_xlabel("patients in the cohort"); axB.set_ylabel("edge recall vs planted E")
    axB.set_ylim(-0.05, 1.05); axB.legend(fontsize=8)
    axB.set_title("B. Pooling more patients does NOT recover fitness epistasis\n"
                  "(recall sits at chance: cross-sectional presence is blind to clone size)")

    # -- C: interaction strength
    g = strength_df.groupby("strength")["recall"].agg(["mean", "std"]).reset_index()
    axC.errorbar(g["strength"], g["mean"], yerr=g["std"].fillna(0), fmt="o-", capsize=3)
    axC.axhline(chance, color="grey", ls=":", lw=1)
    axC.text(strength_df["strength"].max(), chance + 0.02, "chance", ha="right", fontsize=8, color="grey")
    axC.set_xlabel("planted interaction strength $|E_{ij}|$")
    axC.set_ylabel("edge recall vs planted E"); axC.set_ylim(-0.05, 1.05)
    axC.set_title("C. ...and neither does planting it harder\n(the observable, not the effect size, is the limit)")

    # -- D: the two axes of the dependency DAG, under each gating mode
    bars = [("constraint_satisfaction", "true trees", 'B requires A\n(true trees)'),
            ("order_accuracy", "true trees", 'A precedes B\n(true trees)'),
            ("order_accuracy", "reconstructed trees", 'A precedes B\n(reconstructed)')]
    x = np.arange(len(bars))
    width = 0.35
    for k, mode in enumerate(["accessibility", "fitness"]):
        means, errs = [], []
        for metric, arm, _ in bars:
            d = gate_df[(gate_df.gating == mode) & (gate_df.arm == arm)][metric]
            means.append(d.mean())
            errs.append(np.nan_to_num(d.std()))
        axD.bar(x + (k - 0.5) * width, means, width, yerr=errs, capsize=3, label=f"{mode} gating")
    axD.set_xticks(x); axD.set_xticklabels([b[2] for b in bars], fontsize=8)
    axD.set_ylabel("accuracy vs the planted DAG"); axD.set_ylim(0, 1.05)
    axD.axhline(0.5, color="grey", ls=":", lw=1)
    axD.text(len(bars) - 0.5, 0.52, "chance", ha="right", fontsize=8, color="grey")
    axD.legend(fontsize=8)
    axD.set_title("D. Ordered constraints ARE recoverable — if they gate ACCESSIBILITY\n"
                  "(fitness gating enforces no conjunction, and leaves no order to recover)")

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nwrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "manuscript/figures/validation_epistasis.png"))
    ap.add_argument("--sizes", type=int, nargs="+", default=[10, 20, 40, 80])
    ap.add_argument("--strengths", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0])
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    print("== the planted network (the answer key) ==")
    net = run_cohort(2, epi())[0].selection.epistasis
    print("  edges:", [(i, j, round(w, 2)) for i, j, w in net.true_edges()])

    print("\n== B. cohort-size sweep ==")
    size_df = sweep_cohort_size(args.sizes, reps=args.reps)
    print(size_df.groupby(["arm", "n"])[["precision", "recall"]].mean().round(2).to_string())

    print("\n== C. interaction-strength sweep ==")
    strength_df = sweep_strength(args.strengths, reps=args.reps)
    print(strength_df.groupby("strength")[["precision", "recall"]].mean().round(2).to_string())

    print("\n== D. gating / order recovery ==")
    gate_df = sweep_gating(reps=args.reps)
    print(gate_df.groupby(["gating", "arm"])[
        ["constraint_satisfaction", "order_accuracy", "n_child", "n_scored", "n_tied"]
    ].mean().round(2).to_string())

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    make_figure(net, size_df, strength_df, gate_df, args.out)

    # ------------------------------------------------------------------ headline numbers
    def m(df, **q):
        d = df
        for k, v in q.items():
            d = d[d[k] == v]
        return d
    print("\n================ headline ================")
    big = max(args.sizes)
    r_fit = m(size_df, arm="all clones", n=big)["recall"].mean()
    r_det = m(size_df, arm="detectable clones (>=10%)", n=big)["recall"].mean()
    r_ctl = m(size_df, arm="empty E (control)", n=big)["precision"].mean()
    print(f"pairwise E, {big} patients: recall {r_fit:.2f} (all clones), {r_det:.2f} (detectable only)")
    print(f"empty-E control precision (the false-positive floor): {r_ctl:.2f}")
    print("=> fitness epistasis is ~invisible to a cross-sectional event-presence matrix.")
    for mode in ("accessibility", "fitness"):
        d = m(gate_df, gating=mode, arm="true trees")
        r = m(gate_df, gating=mode, arm="reconstructed trees")
        spread = ", ".join(f"{v:.2f}" for v in d["order_accuracy"])
        print(f"{mode:>13} gating: 'B requires A' {d['constraint_satisfaction'].mean():.2f} | "
              f"'A precedes B' {d['order_accuracy'].mean():.2f} (true trees) -> "
              f"{r['order_accuracy'].mean():.2f} (reconstructed); "
              f"n={d['n_child'].mean():.0f} child-carrying lineages/draw")
        print(f"{'':>15} per-network-draw order accuracy: [{spread}]")
    print("=> the conjunction is recoverable ONLY under accessibility gating. Under fitness gating the\n"
          "   child arises freely, so the DAG leaves NO trace to recover: 'B requires A' is violated in\n"
          "   most lineages, and the 'order' merely reports which event is intrinsically faster (event\n"
          "   acquisition rates are heterogeneous, emerging from the CNA landscape), which is why it\n"
          "   swings either side of chance depending on the network draw -- see the per-draw spread.")


if __name__ == "__main__":
    main()
