"""Real-genome mode + per-arm selection (M3b, DESIGN_inference A.5).

Unit tests for the engine generalization (variable segment sizes, per-arm copy-number selection),
the GenomeSpec wiring, the per-arm summary, and a tiny end-to-end ABC on a toy arm genome. The
full fit-to-real PCAWG + Charm experiment lives in validation/validate_inference_realgenome.py.
"""
import numpy as np
import pytest

from iscc.tumor.components.cell import CancerCell
from iscc.tumor.components.selection import Selection
from iscc.tumor.models import GenotypeTumor
from iscc.inference.genome import GenomeSpec
from iscc.inference.realgenome import (
    RealGenomeSimulator, arm_summary_vector, s_arm_prior, default_real_config, PerArmRegressor,
)
from iscc.inference.abc import ABC


def _toy_spec(n_arms=4, s_arm=None):
    """A tiny GenomeSpec so engine tests run fast."""
    return GenomeSpec(
        arm_names=[f"{i}p" for i in range(n_arms)],
        arm_lengths=np.linspace(1.0, 4.0, n_arms),
        onc_counts=np.array([5, 0, 2, 0][:n_arms]),
        tsg_counts=np.array([0, 5, 0, 2][:n_arms]),
        charm=np.array([1.0, -1.0, 0.5, -0.5][:n_arms]),
        s_arm=s_arm, min_genes=4, max_genes=10,
    )


# --- variable segment sizes -------------------------------------------------
def test_variable_segment_sizes_genome_shape():
    sizes = [4, 7, 10]
    c = CancerCell(n_segments=3, segment_sizes=sizes)
    assert c.segment_sizes == sizes
    assert c.n_genes == sum(sizes)
    assert [len(c.genome[s]["p"][0]) for s in range(3)] == sizes
    assert c.get_snvs().shape[0] == sum(sizes)
    assert c.get_cnvs().shape[0] == sum(sizes)


def test_uniform_default_matches_scalar():
    # segment_sizes=None must reproduce the uniform scalar behaviour (abstract mode unchanged).
    c = CancerCell(n_segments=3, segment_size=5)
    assert c.segment_sizes == [5, 5, 5]
    assert c.n_genes == 15
    assert list(c._seg_offsets) == [0, 5, 10, 15]


def test_variable_sizes_snv_indexing():
    sizes = [4, 6]
    sel = Selection(n_segments=2, segment_sizes=sizes, prop_driver=0.0,
                    rng=np.random.default_rng(0))
    c = CancerCell(n_segments=2, segment_sizes=sizes, snv_prob=1.0, cnv_prob=0.0, n_events=1)
    rng = np.random.default_rng(1)
    for _ in range(8):
        c.mutate(rng, sel)
    snv = c.get_snvs()
    assert snv.shape[0] == sum(sizes)
    assert (snv > 0).any()                              # some sites mutated


# --- per-arm selection mode -------------------------------------------------
def test_arm_fitness_neutral_baseline():
    sel = Selection(n_segments=3, segment_size=4, selection_mode="arm",
                    s_arm=[1.5, 0.5, 1.2], rng=np.random.default_rng(0))
    gs = {"seg_cns": [2, 2, 2]}
    assert sel.update_division_rate(gs) == pytest.approx(1.0)   # diploid baseline -> neutral


def test_arm_fitness_direction():
    sel = Selection(n_segments=2, segment_size=4, selection_mode="arm",
                    s_arm=[1.5, 0.5], rng=np.random.default_rng(0))
    # amplifying the s>1 arm is beneficial; amplifying the s<1 arm is deleterious
    assert sel.update_division_rate({"seg_cns": [3, 2]}) > 1.0
    assert sel.update_division_rate({"seg_cns": [2, 3]}) < 1.0
    # deleting the s<1 arm is beneficial
    assert sel.update_division_rate({"seg_cns": [2, 1]}) > 1.0
    # exact value: 1.5**(3-2) * 0.5**(2-2) = 1.5
    assert sel.update_division_rate({"seg_cns": [3, 2]}) == pytest.approx(1.5)


def test_s_arm_length_validation():
    with pytest.raises(ValueError):
        Selection(n_segments=3, selection_mode="arm", s_arm=[1.0, 1.0])


# --- GenomeSpec wiring ------------------------------------------------------
def test_genomespec_engine_params():
    spec = _toy_spec(4)
    gp, sp = spec.engine_params({})
    assert gp["n_segments"] == 4
    assert gp["segment_sizes"] == list(spec.segment_sizes)
    assert sp["selection_mode"] == "arm"
    assert sp["prop_driver"] == 0.0
    assert len(sp["s_arm"]) == 4


def test_genomespec_injected_s_arm_wins():
    spec = _toy_spec(4)
    _, sp = spec.engine_params({"s_arm": [1.0, 2.0, 0.5, 1.1]})
    assert sp["s_arm"] == [1.0, 2.0, 0.5, 1.1]


def test_real_genome_tumor_runs_and_responds():
    # a strong-amplification arm should gain copies; a strong-deletion arm should lose them.
    s = [2.0, 0.4, 1.0, 1.0]
    spec = _toy_spec(4, s_arm=s)
    t = GenotypeTumor(
        seed=3, genome_mode="real", genome_spec=spec,
        selection_params={"s_arm": s},
        cancer_cell_params={"max_birth_rate": 0.95, "division_rate": 0.4, "death_rate": 0.02,
                            "mutation_rate": 0.4, "dispersal_rate": 0.2,
                            "snv_prob": 0.0, "cnv_prob": 1.0, "n_events": 1, "amp_prob": 0.5},
        deme_params={"carrying_capacity": 8, "maximum_death_rate": 0.5},
        spatial_params={"grid_size": 10, "structure_radius": 0, "immune_density": 0.0},
    )
    t.grow(n_steps=900, seed=3)
    assert t.get_cancer_size() > 0
    assert t.selection.selection_mode == "arm"
    assert t.n_segments == 4
    from iscc.inference.summaries import cna_summary
    c = cna_summary(t)
    # amplification-favoured arm 0 gains more than it loses; deletion-favoured arm 1 loses more
    assert c["gain_freq"][0] > c["loss_freq"][0]
    assert c["loss_freq"][1] > c["gain_freq"][1]


# --- per-arm summary + simulator --------------------------------------------
def test_arm_summary_vector_layout():
    spec = _toy_spec(4, s_arm=np.ones(4))
    sim = RealGenomeSimulator(spec, n_steps=300, seed=0)
    t = sim.simulate_tumor({f"s{i}": 1.0 for i in range(4)}, seed=0)
    vals, names = arm_summary_vector(t)
    assert vals.shape[0] == 2 * spec.n_arms
    assert names[0] == "gain[0]" and names[spec.n_arms] == "loss[0]"


def test_simulator_callable_returns_cohort_frequencies():
    # cohort summary over several tumours; an occasional extinct tumour is dropped from the cohort.
    spec = _toy_spec(3, s_arm=np.ones(3))
    sim = RealGenomeSimulator(spec, n_steps=500, seed=1, cohort_size=6)
    v = sim({f"s{i}": 1.0 for i in range(3)})
    assert v.shape[0] == 6
    assert np.isfinite(v).all()
    assert ((v >= 0) & (v <= 1)).all()             # frequencies in [0, 1]


# --- pooled per-arm regressor (the M3b estimator), on synthetic data --------
def test_per_arm_regressor_recovers_monotone_map():
    # synthetic reference where gain rises and loss falls with s (the engine's qualitative law);
    # the pooled regressor must recover s from (gain, loss) and respect the direction.
    rng = np.random.default_rng(0)
    n_arms, m = 6, 400
    theta = rng.uniform(0.4, 1.8, size=(m, n_arms))
    gain = np.clip((theta - 0.4) / 1.4 + rng.normal(0, 0.05, theta.shape), 0, 1)
    loss = np.clip((1.8 - theta) / 1.4 + rng.normal(0, 0.05, theta.shape), 0, 1)
    summaries = np.concatenate([gain, loss], axis=1)

    reg = PerArmRegressor(n_arms, n_jobs=1).fit(theta, summaries)
    # an arm observed with high gain / low loss -> s>1; low gain / high loss -> s<1
    obs = np.concatenate([np.full(n_arms, 0.9), np.full(n_arms, 0.1)])
    s_hi = reg.predict(obs)
    obs2 = np.concatenate([np.full(n_arms, 0.1), np.full(n_arms, 0.9)])
    s_lo = reg.predict(obs2)
    assert s_hi.mean() > 1.2 > s_lo.mean()
    # recovery on a held-out row
    s_true = theta[0]
    summ_row = summaries[0]
    s_hat = reg.predict(summ_row)
    assert np.corrcoef(s_true, s_hat)[0, 1] > 0.5


# --- tiny end-to-end ABC recovers the selection direction -------------------
def test_abc_recovers_arm_direction():
    # 3-arm toy: arm 0 strongly amplified, arm 2 strongly deleted; arm 1 neutral.
    spec = _toy_spec(3, s_arm=np.ones(3))
    truth = {"s0": 1.7, "s1": 1.0, "s2": 0.45}
    sim = RealGenomeSimulator(spec, n_steps=700, seed=0, cohort_size=6)
    observed = RealGenomeSimulator(spec, n_steps=700, seed=999, cohort_size=12)(truth)

    prior = s_arm_prior(3, 0.4, 1.8)
    abc = ABC(prior, sim, n_workers=4, seed=0)
    reference = abc.reference_table(300)
    post = abc.run(observed, accept_frac=0.1, reference=reference, project="rf")
    s_hat = dict(zip(post.names, post.rf_estimate))
    # the amplified arm is inferred > the deleted arm (direction recovered)
    assert s_hat["s0"] > s_hat["s2"]
