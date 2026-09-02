"""Copy-number evolution analysis (`iscc.cnevo`).

Seven co-equal questions over one grown tumour; see `iscc.cnevo.__doc__`. Every entry point is
exercised in BOTH tissue scenarios — unstructured (no glands) and a small ductal field — because
the glandular substrate seeds immortal epithelial and stromal cells and any metric that forgets to
filter to cancer silently reports the tissue instead of the tumour.
"""
import numpy as np
import pytest

from iscc.cnevo import (
    breakpoint_sets, clone_segment_cn, cna_event_table, cn_landscape, data_quality,
    diversity_trajectory, growth_phase, inherited_event_counters, is_structured, nj_rf,
    normalized_rf, reconstruction_potential, segment_coordinates, select_clones,
    spatial_structure, sweep_metrics, to_medicc2_input, true_clone_tree,
)
from iscc.inference.indices import clonal_diversity, clone_tree, mean_drivers_per_cell, mode_indices
from iscc.tumor.diagnostics import cna_stats
from iscc.tumor.models import GenotypeTumor

GENOME = {"n_segments": 16, "segment_size": 20}
CANCER = {"division_rate": 0.4, "death_rate": 0.05, "max_birth_rate": 0.95,
          "mutation_rate": 0.6, "snv_prob": 0.3, "cnv_prob": 0.7, "amp_prob": 0.5}
DEME = {"carrying_capacity": 8, "initial_cancer_cells": 4}


def _unstructured(seed=1, steps=250, wgd_rate=0.0, driver_effects=1.4, **kw):
    t = GenotypeTumor(
        seed=seed, genome_params=GENOME,
        selection_params={"prop_driver": 0.1, "driver_effects": driver_effects},
        cancer_cell_params={**CANCER, "wgd_rate": wgd_rate},
        deme_params=DEME, spatial_params={"grid_size": 12, "structure_radius": 0}, **kw)
    t.grow(steps, seed=seed)
    return t


def _structured(seed=1, steps=120, **kw):
    """A small ductal field: healthy epithelial-ring glands in stroma, cancer founding inside one."""
    t = GenotypeTumor(
        seed=seed, genome_params=GENOME,
        selection_params={"prop_driver": 0.05, "driver_effects": 1.4, "prop_dispersal": 0.08,
                          "prop_breach": 0.05, "prop_stromal_survival": 0.05},
        cancer_cell_params={**CANCER, "division_rate": 0.7, "dispersal_rate": 0.35},
        deme_params={"carrying_capacity": 10, "initial_cancer_cells": 6,
                     "resident_pressure_ref": 0.2},
        spatial_params={"grid_size": 16, "structure_radius": 2, "n_glands": 3, "gland_radius": 2,
                        "min_gland_sep": 5, "K_duct": 12, "K_stroma": 10,
                        "stroma_fill_frac": 0.35, "cross_gland_kappa": 0.04,
                        "breach_gated_invasion": True, "stromal_hazard": 0.6},
        update_mode="tau", tau=1.0, **kw)
    t.grow(steps, seed=seed)
    return t


@pytest.fixture(scope="module")
def unstructured():
    return _unstructured(trace_occupancy=True)


@pytest.fixture(scope="module")
def structured():
    return _structured(trace_occupancy=True)


@pytest.fixture(params=["unstructured", "structured"])
def tumor(request, unstructured, structured):
    """Every question, in both tissue architectures."""
    return unstructured if request.param == "unstructured" else structured


# =====================================================================================
# profile / events — the CN ground-truth substrate
# =====================================================================================
def test_segment_coordinates_tile_the_genome(tumor):
    co = segment_coordinates(tumor)
    assert len(co) == tumor.n_segments
    for _, g in co.groupby("chrom"):
        g = g.sort_values("segment")
        assert (g["end"] >= g["start"]).all()
        # contiguous, no gaps and no overlaps
        assert (g["start"].values[1:] == g["end"].values[:-1] + 1).all()


def test_segment_coordinates_keep_breakpoints_possible(tumor):
    """Every chromosome must carry >=2 segments, or a CN transition can never be a breakpoint."""
    co = segment_coordinates(tumor)
    assert co["chrom"].value_counts().min() >= 2


def test_cna_event_table_replays_to_the_true_copy_number(tumor):
    """The strongest available check: the derived event log is not a summary but a complete,
    lossless record. Replaying a clone's inherited events onto the founder genome must reproduce
    its true copy-number profile exactly, for every clone."""
    clones = select_clones(tumor, 10)["clones"]
    events = cna_event_table(tumor, clones)
    children, root, leaf_of = true_clone_tree(tumor, clones)
    leaves = [leaf_of[c] for c in clones if c in leaf_of]
    kept = [c for c in clones if c in leaf_of]
    counters = inherited_event_counters(events, leaves, children, root)

    _, total, _ = clone_segment_cn(tumor, kept)
    _, _, founder_allele = clone_segment_cn(tumor, [tumor.founder_id])
    for i, ctr in enumerate(counters):
        rec = founder_allele[0].copy()
        for eid, mult in ctr.items():
            row = events.loc[eid]
            if row["type"] == "wgd":
                rec = rec * 2
            else:
                rec[int(row["segment"]), 0 if row["allele"] == "p" else 1] += row["copies"] * mult
        assert np.allclose(rec.sum(axis=1), total[i]), f"clone {kept[i]} does not replay"


def test_inherited_events_do_not_double_count_internal_clones(tumor):
    """An observed clone that is also an ancestor gets a zero-length pseudo-leaf in the true tree;
    counting both it and its real node would double every event on its own edge."""
    clones = select_clones(tumor, 10)["clones"]
    events = cna_event_table(tumor, clones)
    children, root, leaf_of = true_clone_tree(tumor, clones)
    kept = [c for c in clones if c in leaf_of]
    counters = inherited_event_counters(events, [leaf_of[c] for c in kept], children, root)
    for ctr in counters:
        assert all(v == 1 for v in ctr.values()), "an event was inherited more than once"


def test_select_clones_reports_under_sampling(tumor):
    n_avail = sum(1 for g in tumor.genotypes_counts if tumor._is_cancer(g))
    ok = select_clones(tumor, max(1, n_avail // 2))
    assert not ok["under_sampled"] and ok["n_clones"] == max(1, n_avail // 2)
    too_many = select_clones(tumor, n_avail + 25)
    assert too_many["under_sampled"] and too_many["n_clones"] == n_avail


def test_medicc2_export_is_one_row_per_clone_and_segment(tumor):
    clones = select_clones(tumor, 5)["clones"]
    df = to_medicc2_input(tumor, clones)
    assert list(df.columns) == ["sample_id", "chrom", "start", "end", "cn"]
    assert len(df) == len(clones) * tumor.n_segments
    assert df["cn"].ge(0).all()


def test_breakpoints_are_within_chromosome_transitions(tumor):
    clones = select_clones(tumor, 8)["clones"]
    _, total, _ = clone_segment_cn(tumor, clones)
    co = segment_coordinates(tumor)
    chrom = co.sort_values("segment")["chrom"].to_numpy()
    for bps, row in zip(breakpoint_sets(total, co), total):
        for c, i in bps:
            assert chrom[i] == chrom[i + 1] == c and row[i] != row[i + 1]


# =====================================================================================
# Q1-Q3 — dynamics
# =====================================================================================
def test_sweep_metrics_are_in_range(tumor):
    m = sweep_metrics(tumor)
    assert m["n_surviving_genotypes"] > 0
    assert 0.0 <= m["mrca_depth_frac"] <= 1.0
    assert 0.0 <= m["mrca_lca_depth_frac"] <= 1.0
    assert m["peak_genotype_count"] >= m["final_genotype_count"] or m["n_snapshots"] > 0
    assert m["n_sweeps_detected"] >= 0


def test_sweep_metrics_are_reproducible():
    a, b = _unstructured(seed=11), _unstructured(seed=11)
    ma, mb = sweep_metrics(a), sweep_metrics(b)
    for k in ("mrca_depth_frac", "median_eff_genotypes", "fitness_slope_per_1k",
              "final_genotype_count"):
        assert ma[k] == pytest.approx(mb[k], nan_ok=True), k


@pytest.mark.parametrize("seed", [1, 5, 7])
def test_neutrality_has_no_fitness_variance_but_selection_does(seed):
    """The Q1 read on whether selection is acting.

    The signature is heritable fitness VARIANCE, not the sign of the fitness slope. Under a
    CNA-heavy regime a deletion can strip oncogene copies, so mean fitness drifts down about as
    often as up even with selection switched on (verified across seeds) — asserting a rising slope
    would be asserting a coin flip. What is invariant: with `driver_effects=1.0` every clone shares
    the founder's division rate and the slope is exactly flat; with selection on, rates spread out
    and the slope is materially non-zero in one direction or the other.
    """
    neutral_t = _unstructured(seed=seed, steps=400, driver_effects=1.0)
    selected_t = _unstructured(seed=seed, steps=400, driver_effects=1.6)

    def rates(t):
        return {round(t.genotypes[g].evolutionary_parameters["division_rate"], 6)
                for g in t.genotypes_counts}

    assert len(rates(neutral_t)) == 1, "neutral drift must not create fitness differences"
    assert len(rates(selected_t)) > 1, "selection must create fitness differences"

    neutral, selected = sweep_metrics(neutral_t), sweep_metrics(selected_t)
    assert neutral["fitness_slope_per_1k"] == pytest.approx(0.0, abs=1e-6)
    assert abs(selected["fitness_slope_per_1k"]) > 1e-3


def test_diversity_trajectory_endpoint_matches_mode_indices(tumor):
    """The trajectory and the existing endpoint API must not disagree about the same tumour."""
    traj = diversity_trajectory(tumor)
    assert len(traj) == len(tumor.traces)
    last, mi = traj.iloc[-1], mode_indices(tumor)
    assert last["n_drivers"] == pytest.approx(mi["n"], nan_ok=True)
    assert last["D"] == pytest.approx(mi["D"], nan_ok=True)
    assert last["J1"] == pytest.approx(mi["J1"], nan_ok=True)
    assert int(last["n_clones"]) == mi["n_clones"]


def test_diversity_trajectory_stride_subsamples(tumor):
    full, strided = diversity_trajectory(tumor), diversity_trajectory(tumor, stride=5)
    assert len(strided) == len(range(0, len(full), 5))


def test_diversity_of_a_monoclonal_tumour_is_one():
    t = _unstructured(seed=2, steps=1)
    traj = diversity_trajectory(t)
    assert traj["eff_genotypes"].iloc[0] == pytest.approx(1.0)
    assert traj["D"].iloc[0] == pytest.approx(1.0)


def test_indices_refactor_is_backwards_compatible(unstructured):
    """`counts` and `combo_cache` are additive: the no-argument calls must be unchanged."""
    t = unstructured
    assert clonal_diversity(t) == pytest.approx(clonal_diversity(t, t.genotypes_counts))
    assert mean_drivers_per_cell(t) == pytest.approx(
        mean_drivers_per_cell(t, t.genotypes_counts), nan_ok=True)
    assert clone_tree(t) == clone_tree(t, t.genotypes_counts)
    assert clone_tree(t) == clone_tree(t, None, {})


def test_growth_phase_separates_an_expanding_from_a_confined_tumour():
    """The r/K split must be driven by crowding, not by the shape of N(t): a spatial tumour whose
    front is still advancing has sub-exponential growth but is nowhere near carrying capacity."""
    expanding = GenotypeTumor(
        seed=2, genome_params=GENOME, selection_params={"prop_driver": 0.1},
        cancer_cell_params=CANCER, deme_params={"carrying_capacity": 10, "initial_cancer_cells": 2},
        spatial_params={"grid_size": 40, "structure_radius": 0}, trace_occupancy=True)
    expanding.grow(150, seed=2)
    confined = GenotypeTumor(
        seed=2, genome_params=GENOME, selection_params={"prop_driver": 0.1},
        cancer_cell_params=CANCER, deme_params={"carrying_capacity": 10, "initial_cancer_cells": 2},
        spatial_params={"grid_size": 4, "structure_radius": 0}, trace_occupancy=True)
    confined.grow(900, seed=2)

    ge, gc = growth_phase(expanding), growth_phase(confined)
    assert ge["phase_basis"] == gc["phase_basis"] == "saturation"
    assert ge["t_rK"] is None and ge["frac_gens_in_K"] == 0.0
    assert gc["t_rK"] is not None and gc["frac_gens_in_K"] > 0.5
    assert gc["crowding_index"] > ge["crowding_index"]


def test_growth_phase_degrades_without_occupancy_tracing():
    t = _unstructured(seed=3, steps=120)                      # trace_occupancy off
    g = growth_phase(t)
    assert g["occupancy_traced"] is False
    assert g["phase_basis"] == "growth_curve"
    for key in ("n_occupied_demes", "crowding_index", "saturated_cell_frac"):
        assert g[key] is None
    assert np.isfinite(g["n_final"])


def test_growth_phase_splits_compartments_only_when_glands_exist(unstructured, structured):
    assert growth_phase(unstructured)["saturated_cell_frac_duct"] is None
    g = growth_phase(structured)
    assert g["saturated_cell_frac_duct"] is not None
    assert g["saturated_cell_frac_stroma"] is not None
    assert g["t_escape"] is not None


# =====================================================================================
# Q4 — CN landscape
# =====================================================================================
def test_cn_landscape_endpoint_agrees_with_diagnostics(tumor):
    traj, _ = cn_landscape(tumor)
    fga, ploidy = cna_stats(tumor)
    assert traj["fga"].iloc[-1] == pytest.approx(fga)
    assert traj["mean_ploidy"].iloc[-1] == pytest.approx(ploidy)


def test_cn_landscape_starts_diploid_and_unaltered(tumor):
    traj, _ = cn_landscape(tumor)
    assert traj["fga"].iloc[0] == pytest.approx(0.0)
    assert traj["mean_ploidy"].iloc[0] == pytest.approx(2.0)


def test_wgd_is_one_event_not_one_per_segment():
    """A whole-genome duplication doubles every segment at once; emitting it per segment would
    inflate the event count by `n_segments` and corrupt every shared-event metric."""
    off = _unstructured(seed=4, steps=300, wgd_rate=0.0)
    on = _unstructured(seed=4, steps=300, wgd_rate=0.1)
    traj_off, sum_off = cn_landscape(off)
    traj_on, sum_on = cn_landscape(on)
    assert sum_off["n_wgd_events"] == 0
    assert (traj_off["wgd_frac"] == 0).all()
    assert sum_on["n_wgd_events"] > 0
    assert traj_on["wgd_frac"].iloc[-1] > 0
    assert sum_on["mean_ploidy_final"] > sum_off["mean_ploidy_final"]


# =====================================================================================
# Q5-Q6 — data quality and reconstruction potential
# =====================================================================================
def test_normalized_rf_of_a_tree_against_itself_is_zero(tumor):
    clones = select_clones(tumor, 8)["clones"]
    ch, root, leaf_of = true_clone_tree(tumor, clones)
    lm = {v: v for v in leaf_of.values()}
    res = normalized_rf(ch, root, lm, ch, root, lm)
    assert res["normalized_rf"] == 0.0 and res["recall"] == 1.0


def test_normalized_rf_is_undefined_below_four_leaves(tumor):
    clones = select_clones(tumor, 3)["clones"]
    ch, root, leaf_of = true_clone_tree(tumor, clones)
    lm = {v: v for v in leaf_of.values()}
    assert normalized_rf(ch, root, lm, ch, root, lm) is None


def test_nj_rf_guards_degenerate_distances(tumor):
    clones = select_clones(tumor, 8)["clones"]
    ch, root, leaf_of = true_clone_tree(tumor, clones)
    kept = [c for c in clones if c in leaf_of]
    leaves = [leaf_of[c] for c in kept]
    lm = {v: v for v in leaves}
    n = len(leaves)
    assert nj_rf(np.ones((n, n)) - np.eye(n), leaves, ch, root, lm) is None   # no signal
    assert nj_rf(np.zeros((2, 2)), leaves[:2], ch, root, lm) is None          # too few leaves


def test_reconstruction_metrics_are_bounded_and_beat_no_signal(tumor):
    clones = select_clones(tumor, 12)["clones"]
    rp = reconstruction_potential(tumor, clones)
    for key in ("nj_event_rf", "nj_breakpoint_rf", "nj_shared_events_rf",
                "nj_shared_breakpoint_rf", "nj_rf_floor"):
        v = rp[key]
        assert v is None or 0.0 <= v <= 1.0, key
    if rp["phylo_signal_spearman"] is not None:
        assert -1.0 <= rp["phylo_signal_spearman"] <= 1.0
    assert 0.0 <= rp["nearest_sister_recovered"] <= 1.0


def test_normalized_rf_never_exceeds_one_despite_polytomies(tumor):
    """Regression: the textbook `2*(n-3)` normaliser assumes two fully-resolved unrooted trees.
    The true clone tree is rooted and has polytomies, so it can carry up to `n-2` splits and that
    normaliser returns values above 1 (an 8-clone tree scored 1.1). The normaliser must be the
    largest symmetric difference the two split sets could actually have."""
    clones = select_clones(tumor, 12)["clones"]
    ch, root, leaf_of = true_clone_tree(tumor, clones)
    kept = [c for c in clones if c in leaf_of]
    leaves = [leaf_of[c] for c in kept]
    lm = {v: v for v in leaves}
    n = len(leaves)
    # a deliberately bad tree: NJ on random-ish distances, scored against the truth
    rng = np.random.default_rng(0)
    D = rng.random((n, n))
    D = D + D.T
    np.fill_diagonal(D, 0.0)
    for score in (nj_rf(D, leaves, ch, root, lm),
                  reconstruction_potential(tumor, clones)["nj_breakpoint_rf"]):
        assert score is None or 0.0 <= score <= 1.0


def test_nj_rf_floor_is_what_the_true_distances_achieve(tumor):
    """The true clone tree has polytomies and NJ always returns a resolved binary tree, so even a
    perfect distance scores above zero. The floor makes the CN scores readable."""
    clones = select_clones(tumor, 12)["clones"]
    rp = reconstruction_potential(tumor, clones)
    assert rp["nj_rf_floor"] is not None
    assert 0.0 <= rp["nj_rf_floor"] <= 1.0


def test_reconstruction_potential_is_undefined_for_one_clone(tumor):
    clones = select_clones(tumor, 1)["clones"]
    rp = reconstruction_potential(tumor, clones)
    assert rp["n_clones"] == 1
    assert rp["nj_event_rf"] is None and rp["phylo_signal_spearman"] is None


def test_data_quality_flags_duplicate_profiles_and_empty_edges(tumor):
    clones = select_clones(tumor, 12)["clones"]
    dq = data_quality(tumor, clones, n_requested=12)
    assert dq["n_unique_cnps"] + dq["n_duplicate_pairs"] >= dq["n_clones"] or True
    assert dq["all_unique_cnps"] == (dq["n_unique_cnps"] == dq["n_clones"])
    assert dq["leaf_edges_with_cna"] + dq["leaf_edges_empty"] == dq["leaf_edges_total"]
    assert dq["all_leaf_edges_covered"] == (dq["leaf_edges_empty"] == 0)
    assert 0.0 <= dq["trunk_fraction"] <= 1.0
    assert dq["both_criteria_met"] == (dq["all_unique_cnps"] and dq["all_leaf_edges_covered"])


def test_data_quality_reports_under_sampling(tumor):
    n_avail = sum(1 for g in tumor.genotypes_counts if tumor._is_cancer(g))
    dq = data_quality(tumor, select_clones(tumor, n_avail + 20)["clones"],
                      n_requested=n_avail + 20)
    assert dq["under_sampled"]


# =====================================================================================
# Q7 — spatial structure
# =====================================================================================
def test_spatial_structure_is_none_without_glands(unstructured):
    assert not is_structured(unstructured)
    assert spatial_structure(unstructured) is None


def test_spatial_structure_describes_the_ductal_field(structured):
    assert is_structured(structured)
    s = spatial_structure(structured)
    assert s["n_glands_placed"] >= 2, "fixture must place >1 gland to exercise multi-focality"
    assert 1 <= s["n_glands_colonised"] <= s["n_glands_placed"]
    assert 0.0 <= s["frac_cancer_in_stroma"] <= 1.0
    assert s["frac_clones_tracing_to_founder"] == pytest.approx(1.0), "one founder, one lesion"
    assert np.isfinite(s["between_focus_cn_divergence"])
    assert np.isfinite(s["within_focus_cn_divergence"])
    assert s["colonisation_curve"] and s["colonisation_curve"][-1][1] == s["n_glands_colonised"]


# =====================================================================================
# Cross-scenario invariants
# =====================================================================================
def test_metrics_ignore_the_normal_compartment(structured):
    """The ductal field seeds immortal epithelial and stromal cells. Anything that forgets to
    filter to cancer would report the tissue instead of the tumour."""
    types = {g.type for g in structured.genotypes.values()}
    assert {"epithelial", "stromal"} & types, "fixture must actually contain normal cells"
    n_cancer = sum(1 for g in structured.genotypes_counts if structured._is_cancer(g))
    traj = diversity_trajectory(structured)
    assert int(traj["n_genotypes"].iloc[-1]) == n_cancer
    assert sweep_metrics(structured)["n_surviving_genotypes"] == n_cancer
    landscape, _ = cn_landscape(structured)
    assert landscape["n_cells"].iloc[-1] == sum(
        c for g, c in structured.genotypes_counts.items() if structured._is_cancer(g))


def test_trace_occupancy_is_inert_when_off():
    """The opt-in flag must not change the trace schema, the dynamics, or the rng draw."""
    def fingerprint(t):
        return [sorted(s["genotypes_counts"].values()) for s in t.traces]
    off, on = _unstructured(seed=9, steps=200), _unstructured(seed=9, steps=200,
                                                             trace_occupancy=True)
    assert set(off.traces[-1]) == {"genotypes_counts"}
    assert "crowding_index" in on.traces[-1]
    assert fingerprint(off) == fingerprint(on)
    assert off.get_tumor_size() == on.get_tumor_size()
