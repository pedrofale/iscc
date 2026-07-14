"""ABC inference layer: exposed rates (A.0), summary statistics (A.2), ABC engine (A.1).

The tumour-driven parameter recovery (A.4.1) itself lives in
``validation/validate_inference_recovery.py`` (it needs thousands of sims); here we unit-test the
machinery on hand-built tumours and a fast analytic toy model.
"""
import numpy as np
import pytest

from conftest import (
    GENOME_PARAMS, SELECTION_PARAMS, CANCER_CELL_PARAMS, DEME_PARAMS, N_SEGMENTS, SEGMENT_SIZE,
)
from iscc.tumor.components.cell import CancerCell
from iscc.tumor.components.selection import Selection
from iscc.tumor.models import GenotypeTumor
from iscc.inference import cna_summary, snv_summary, summary_vector, Prior, ABC
from iscc.inference.tumor import TumorSimulator, default_prior

SPATIAL = {"grid_size": 12, "structure_radius": 0}
CANCER = {**CANCER_CELL_PARAMS, "death_rate": 0.02}


def _selection(seed=0):
    return Selection(n_segments=N_SEGMENTS, segment_size=SEGMENT_SIZE,
                     rng=np.random.default_rng(seed), **SELECTION_PARAMS)


def _cancer_cell(selection, **overrides):
    return CancerCell(
        n_segments=N_SEGMENTS, segment_size=SEGMENT_SIZE,
        n_onc=len(selection.get_oncogenes()), n_tsg=len(selection.get_tsgs()),
        n_disp=len(selection.get_dispersal_genes()),
        n_ir=len(selection.get_immune_resistant()),
        n_tr=len(selection.get_treatment_resistant()),
        **{**CANCER_CELL_PARAMS, **overrides},
    )


# --- A.0: SNV/CNA rates are exposed parameters ------------------------------
def test_cancer_cell_exposes_event_rates():
    c = _cancer_cell(_selection(), snv_prob=0.3, cnv_prob=0.7, n_snvs_per_allele=4, amp_prob=0.9)
    assert (c.snv_prob, c.cnv_prob, c.n_snvs_per_allele, c.amp_prob) == (0.3, 0.7, 4, 0.9)


def test_divide_inherits_event_rates():
    c = _cancer_cell(_selection(), snv_prob=0.2, cnv_prob=0.8, amp_prob=0.1)
    child = c.divide()
    assert (child.snv_prob, child.cnv_prob, child.amp_prob) == (0.2, 0.8, 0.1)


def test_amp_prob_one_only_amplifies():
    # snv_prob=0 forces CNA events; amp_prob=1 forces amplifications -> copy number only grows.
    sel = _selection()
    c = _cancer_cell(sel, snv_prob=0.0, cnv_prob=1.0, amp_prob=1.0)
    rng = np.random.default_rng(0)
    for _ in range(15):
        c.mutate(rng, sel)
    assert c.genome_summary["ploidy"] > 2.0
    assert min(c.genome_summary["seg_cns"]) >= 2


def test_amp_prob_zero_only_deletes():
    sel = _selection()
    c = _cancer_cell(sel, snv_prob=0.0, cnv_prob=1.0, amp_prob=0.0)
    rng = np.random.default_rng(0)
    for _ in range(10):
        c.mutate(rng, sel)
    assert c.genome_summary["ploidy"] < 2.0


def test_event_rates_thread_through_config():
    cancer = {**CANCER, "amp_prob": 0.123, "n_snvs_per_allele": 7}
    t = GenotypeTumor(genome_params=GENOME_PARAMS, selection_params=SELECTION_PARAMS,
                      cancer_cell_params=cancer, deme_params=DEME_PARAMS, spatial_params=SPATIAL,
                      seed=0)
    founder = t.genotypes[t.founder_id]
    assert founder.amp_prob == 0.123 and founder.n_snvs_per_allele == 7


# --- A.2: summary statistics ------------------------------------------------
def _grown_tumor(seed=0, amp_prob=0.5, steps=300):
    cancer = {**CANCER, "mutation_rate": 0.5, "amp_prob": amp_prob}
    # well-mixed (no crowding ceiling) + seeded cluster: these exercise the CNA/SNV summary-stat
    # functions on a grown population, not spatial capping (DESIGN_crowding.md well-mixed mode).
    t = GenotypeTumor(genome_params=GENOME_PARAMS, selection_params=SELECTION_PARAMS,
                      cancer_cell_params=cancer,
                      deme_params={"carrying_capacity": None, "initial_cancer_cells": 5},
                      spatial_params=SPATIAL, seed=seed)
    t.grow(n_steps=steps, seed=seed)
    return t


def test_cna_summary_shapes_and_ranges():
    t = _grown_tumor()
    s = cna_summary(t)
    assert s["gain_freq"].shape == (t.n_segments,)
    assert s["loss_freq"].shape == (t.n_segments,)
    assert np.all((s["gain_freq"] >= 0) & (s["gain_freq"] <= 1))
    assert np.all((s["loss_freq"] >= 0) & (s["loss_freq"] <= 1))
    assert 0.0 <= s["fga"] <= 1.0
    assert s["ploidy_mean"] > 0


def test_amp_prob_raises_gain_over_loss():
    # population grown with amplification-biased CNAs should show more gains than losses
    gains, losses = [], []
    for seed in range(4):
        s = cna_summary(_grown_tumor(seed=seed, amp_prob=0.95))
        gains.append(s["gain_freq"].sum()); losses.append(s["loss_freq"].sum())
    assert np.mean(gains) > np.mean(losses)


def test_snv_summary_and_vector():
    t = _grown_tumor()
    s = snv_summary(t)
    assert 0.0 <= s["n_sites"] <= 1.0
    assert 0.0 <= s["mean_vaf"] <= 1.0
    vec, names = summary_vector(t)
    assert vec.shape == (len(names),)
    assert all(np.isfinite(vec[:t.n_segments]))   # gain freqs always finite when cancer present


def test_summaries_nan_when_extinct():
    t = GenotypeTumor(genome_params=GENOME_PARAMS, selection_params=SELECTION_PARAMS,
                      cancer_cell_params=CANCER, deme_params=DEME_PARAMS, spatial_params=SPATIAL,
                      seed=0)
    # never grown -> founder only; force extinction by emptying counts
    t.genotypes_counts.clear()
    s = cna_summary(t)
    assert np.isnan(s["fga"])
    assert np.all(np.isnan(s["gain_freq"]))


# --- A.1: ABC engine on a fast analytic toy model ---------------------------
def _toy_simulate(theta):
    # identifiable summaries: each parameter appears directly, plus an interaction term
    a, b = theta["a"], theta["b"]
    rng = np.random.default_rng(int((a * 1e4 + b * 1e2) * 1000) % (2**32))
    noise = rng.normal(0, 0.02, size=3)
    return np.array([a, b, a * b]) + noise


def test_prior_sampling_ranges():
    prior = Prior({"a": (0.0, 1.0), "b": (-2.0, 2.0)})
    samples = prior.sample(np.random.default_rng(0), 500)
    assert samples.shape == (500, 2)
    assert samples[:, 0].min() >= 0.0 and samples[:, 0].max() <= 1.0
    assert samples[:, 1].min() >= -2.0 and samples[:, 1].max() <= 2.0


def test_abc_recovers_toy_parameters():
    prior = Prior({"a": (0.0, 1.0), "b": (0.0, 1.0)})
    abc = ABC(prior, _toy_simulate, n_workers=1, seed=0)
    truth = {"a": 0.7, "b": 0.3}
    observed = np.array([truth["a"], truth["b"], truth["a"] * truth["b"]])
    post = abc.run(observed, n_samples=1500, accept_frac=0.1)
    est = post.map()
    assert abs(est[0] - 0.7) < 0.07
    assert abs(est[1] - 0.3) < 0.07
    # RF point estimate is also close
    assert abs(post.rf_estimate[0] - 0.7) < 0.1
    assert abs(post.rf_estimate[1] - 0.3) < 0.1


def test_abc_credible_interval_covers_truth():
    prior = Prior({"a": (0.0, 1.0), "b": (0.0, 1.0)})
    abc = ABC(prior, _toy_simulate, n_workers=1, seed=1)
    truth = {"a": 0.4, "b": 0.6}
    observed = np.array([0.4, 0.6, 0.24])
    post = abc.run(observed, n_samples=1500, accept_frac=0.1)
    lo, hi = post.credible_interval(0.9)
    assert lo[0] <= 0.4 <= hi[0]
    assert lo[1] <= 0.6 <= hi[1]


def test_default_prior_and_simulator_smoke():
    # one short tumour simulation produces a finite summary vector of the advertised length
    sim = TumorSimulator(n_steps=120, seed=0)
    prior = default_prior(("mutation_rate", "amp_prob"))
    assert prior.names == ["mutation_rate", "amp_prob"]
    vec = sim({"mutation_rate": 0.3, "amp_prob": 0.5})
    assert vec.ndim == 1 and len(vec) == len(sim.names)
