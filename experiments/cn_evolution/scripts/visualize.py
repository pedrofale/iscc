"""Render the figures for one tumour (``--stage pop``) or one sampled clone set (``--stage sim``)."""
import argparse, json, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import load_tumor

from iscc.cnevo import (
    clone_segment_cn, cna_event_table, inherited_event_counters, pairwise_shared_matrix,
    segment_coordinates, spatial_structure, true_clone_tree,
)
from iscc.cnevo import viz as V


def _save(fig_dir, name):
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, name), dpi=110)
    plt.close("all")


def pop_figures(t, out, diversity, growth, landscape, landscape_summary, max_clones=500):
    traj = pd.read_csv(diversity, sep="\t")
    with open(growth) as f:
        gp = json.load(f)
    lt = pd.read_csv(landscape, sep="\t")
    with open(landscape_summary) as f:
        ls = json.load(f)
    occ = {k: [s.get(k) for s in t.traces]
           for k in ("crowding_index", "crowding_index_duct", "crowding_index_stroma")
           if t.traces and k in t.traces[-1]}

    V.plot_diversity_over_time(traj, gp);            _save(out, "diversity_over_time.jpg")
    V.plot_mode_space_trajectory(traj);              _save(out, "mode_space_trajectory.jpg")
    V.plot_growth_phase(traj, gp, occ or None);      _save(out, "growth_phase.jpg")
    V.plot_cn_landscape_over_time(lt);               _save(out, "cn_landscape.jpg")
    V.plot_segment_recurrence(ls);                   _save(out, "segment_recurrence.jpg")

    st = spatial_structure(t)
    if st is not None:
        V.plot_colonisation_curve(st);               _save(out, "colonisation.jpg")
        V.plot_focus_divergence(st);                 _save(out, "focus_divergence.jpg")

    # The engine's own muller / clone-tree plots do not scale, and the quantity that drives their
    # cost is the GENOTYPE REGISTRY -- every genotype ever created, which `genotypes_parents` spans --
    # not the live clone count. A run with only 415 live clones but an 11k registry still hangs
    # plot_muller past a minute (by_drivers=True does not help: the cost is the expansion, not the
    # colouring), and a CNA-heavy run mints a genotype per event and reaches 10^5. Gating on live
    # clones lets exactly those runs through, so gate on the registry. The cnevo figures above carry
    # the seven questions and are unaffected; these are a bonus, skipped when too costly.
    n_clones = sum(1 for g in t.genotypes_counts if t._is_cancer(g))
    n_registry = len(t.genotypes)
    try:
        t.plot_tissue();                             _save(out, "tissue.jpg")
    except Exception as e:
        print(f"  (plot_tissue skipped: {e})")
    if n_registry > max_clones:
        print(f"  (engine muller/clone-tree skipped: genotype registry {n_registry} "
              f"> --max-clones {max_clones}; {n_clones} live clones)")
        return
    for name, fn in (("muller.jpg", t.plot_muller), ("clone_tree.jpg", t.plot_clone_tree)):
        try:
            fn();                                    _save(out, name)
        except Exception as e:                # needs >1 clone / a live population
            print(f"  ({name} skipped: {e})")


def sim_figures(t, out, clone_names):
    with open(clone_names) as f:
        clones = json.load(f)["clones"]
    children, root, leaf_of = true_clone_tree(t, clones)
    order = V.tree_leaf_order(children, root, leaf_of)
    if not order:
        print("  (no clones on the tree: no CN figures)")
        return
    ids, total, _ = clone_segment_cn(t, order)
    coords = segment_coordinates(t)
    V.plot_cn_heatmap(total, ids, coords);           _save(out, "ground_truth_cn.jpg")
    V.plot_pairwise_cn_distance(total, ids);         _save(out, "cn_pairwise_distance.jpg")
    events = cna_event_table(t, order)
    ctr = inherited_event_counters(events, [leaf_of[c] for c in order], children, root)
    V.plot_pairwise_shared_events(pairwise_shared_matrix(ctr), ids)
    _save(out, "cna_events_pairwise_shared.jpg")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["pop", "sim"], required=True)
    p.add_argument("--tumor", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--diversity"); p.add_argument("--growth")
    p.add_argument("--landscape"); p.add_argument("--landscape-summary")
    p.add_argument("--clone-names")
    p.add_argument("--max-clones", type=int, default=2000,
                   help="skip the engine's muller/clone-tree plots above this GENOTYPE REGISTRY "
                        "size (every genotype ever created, not the live clone count)")
    p.add_argument("--done")
    a = p.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    t = load_tumor(a.tumor)
    if a.stage == "pop":
        pop_figures(t, a.out_dir, a.diversity, a.growth, a.landscape, a.landscape_summary,
                    max_clones=a.max_clones)
    else:
        sim_figures(t, a.out_dir, a.clone_names)
    if a.done:
        open(a.done, "w").close()
    print("figures written")


if __name__ == "__main__":
    main()
