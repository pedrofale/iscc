"""Fig 5's batch levels (validation/paper/batch_levels.py) follow the registered rules.

exp-paper1-fig5-cohort, "Amendment, 2026-11-16 night": whole fitted vectors drawn with replacement
and recorded; fitted / shared / six attribution arms / reference; sigma_batch the 10x preset everywhere but
the reference; depth_batch_sigma 0 where mu_lib is drawn per batch; recovery the patient's own at
every level.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "validation", "paper"))
import batch_levels as BL  # noqa: E402
from iscc.data.rna import BATCH_OVERRIDE_KEYS, PROTOCOL_PRESETS  # noqa: E402


def _fits(n_tumours=7, ambient=True, n_patients=12):
    rng = np.random.default_rng(0)
    vec = {}
    for i in range(n_tumours):
        mid = 0.0 if i % 3 == 0 else float(rng.uniform(0.1, 1.0))
        vec[f"T{i}"] = dict(mu_lib=float(rng.uniform(1500, 8000)), sigma_lib=float(rng.uniform(0.3, 0.7)),
                            dispersion=float(rng.uniform(0.1, 0.8)), dropout_mid=mid,
                            dropout_shape=(float(rng.uniform(0.5, 3)) if mid > 0 else None),
                            ambient_frac=(float(rng.uniform(0.01, 0.1)) if ambient else None),
                            doublet_rate=float(rng.uniform(0.01, 0.06)),
                            stress_strength=float(max(rng.normal(0.1, 0.2), 0.0)))
    rec = {str(p): {"cancer": 0.9, "immune": float(rng.uniform(0.2, 0.6)), "stromal": 0.6,
                    "epithelial": 0.8, "host": 0.8} for p in range(n_patients)}
    needs = [t for t, v in vec.items() if v["ambient_frac"] is None]
    return {"wu": {"vectors": vec, "ambient": {"needs_range": needs}}, "recovery": {"per_patient": rec},
            "ambient_range": None}


def _with_range(f, lo=0.03, hi=0.11):
    f = dict(f)
    f["ambient_range"] = dict(rho_low=lo, rho_high=hi, ambient_frac_low=lo,
                              ambient_frac_high=hi, source="test")
    return f


PATIENTS = list(range(10))


def test_every_level_and_design_is_built_with_valid_overrides():
    f = _fits()
    for d in BL.DESIGNS:
        res = BL.levels(f, PATIENTS, d)
        assert set(res) == set(BL.LEVELS) and len(BL.LEVELS) == 9
        for lv, s in res.items():
            for kw in s["batches"]:
                assert set(kw) <= set(BATCH_OVERRIDE_KEYS)
            assert set(s["patients"]) == {str(p) for p in PATIENTS}


def test_fitted_level_takes_whole_vectors_and_records_the_draws():
    f = _fits()
    s = BL.level_settings(f, PATIENTS, "balanced", "fitted", n_balanced_batches=15)
    assert len(s["batches"]) == 15 == len(s["batch_tumours"])
    for kw, t in zip(s["batches"], s["batch_tumours"]):
        v = f["wu"]["vectors"][t]
        for p in ("mu_lib", "sigma_lib", "dispersion", "ambient_frac", "doublet_rate", "dropout_mid"):
            assert kw[p] == pytest.approx(v[p])
        if v["dropout_mid"] > 0:
            assert kw["dropout_shape"] == pytest.approx(v["dropout_shape"])
        assert kw["sigma_batch"] == BL.SIGMA_BATCH and kw["depth_batch_sigma"] == 0.0
    # drawn with replacement, reproducibly, and recorded
    again = BL.level_settings(f, PATIENTS, "balanced", "fitted", n_balanced_batches=15)
    assert again["batch_tumours"] == s["batch_tumours"]
    assert s["draws"]["seed"] == BL.DRAW_SEED
    assert len(set(s["batch_tumours"])) < 15                     # 15 draws from 7 tumours repeat
    # each patient's stress is its own drawn tumour's
    for p, ps in s["patients"].items():
        assert ps["stress_strength"] == pytest.approx(f["wu"]["vectors"][ps["stress_tumour"]]["stress_strength"])


def test_confounded_batches_take_their_patients_tumour():
    f = _fits()
    s = BL.level_settings(f, PATIENTS, "confounded", "fitted")
    assert s["batch_labels"] == [f"batch{p}" for p in PATIENTS]
    for p, t in zip(PATIENTS, s["batch_tumours"]):
        assert s["patients"][str(p)]["drawn_tumour"] == t
    # the patients' draws do not depend on the design
    b = BL.level_settings(f, PATIENTS, "balanced", "fitted")
    assert {p: v["drawn_tumour"] for p, v in s["patients"].items()} == \
           {p: v["drawn_tumour"] for p, v in b["patients"].items()}


def test_shared_level_is_the_median_everywhere():
    f = _fits()
    med = BL.median_vector(f["wu"]["vectors"])
    s = BL.level_settings(f, PATIENTS, "balanced", "shared")
    assert all(kw == s["batches"][0] for kw in s["batches"])
    kw = s["batches"][0]
    for p in ("mu_lib", "sigma_lib", "dispersion", "ambient_frac", "doublet_rate", "dropout_mid"):
        assert kw[p] == pytest.approx(med[p])
    assert kw["sigma_batch"] == BL.SIGMA_BATCH
    assert kw["depth_batch_sigma"] == PROTOCOL_PRESETS["10x"]["depth_batch_sigma"]   # mu_lib not drawn
    assert all(v["stress_strength"] == pytest.approx(med["stress_strength"]) for v in s["patients"].values())
    # the shape's median is over the tumours that have dropout
    shapes = [v["dropout_shape"] for v in f["wu"]["vectors"].values() if v["dropout_mid"] > 0]
    assert med["dropout_shape"] == pytest.approx(np.median(shapes))


@pytest.mark.parametrize("arm", list(BL.SOURCES))
def test_each_arm_varies_only_its_source_from_the_fitted_draws(arm):
    f = _fits()
    fit = BL.level_settings(f, PATIENTS, "balanced", "fitted")
    sh = BL.level_settings(f, PATIENTS, "balanced", "shared")
    s = BL.level_settings(f, PATIENTS, "balanced", arm)
    own = BL.SOURCES[arm]
    for kw, kf, ks in zip(s["batches"], fit["batches"], sh["batches"]):
        for p in BL.TECH:
            if p == "dropout_shape" and kw["dropout_mid"] == 0:
                continue                                      # unused without dropout
            assert kw[p] == pytest.approx(kf[p] if p in own else ks[p]), (arm, p)
        assert kw["sigma_batch"] == BL.SIGMA_BATCH
        assert kw["depth_batch_sigma"] == (0.0 if arm == "depth" else ks["depth_batch_sigma"])
    # patients keep the shared (median) stress in the technical arms
    assert {p: v["stress_strength"] for p, v in s["patients"].items()} == \
           {p: v["stress_strength"] for p, v in sh["patients"].items()}


def test_stress_arm_varies_only_the_patients():
    f = _fits()
    fit = BL.level_settings(f, PATIENTS, "confounded", "fitted")
    sh = BL.level_settings(f, PATIENTS, "confounded", "shared")
    s = BL.level_settings(f, PATIENTS, "confounded", "stress")
    assert s["batches"] == sh["batches"]
    for p in s["patients"]:
        assert s["patients"][p]["stress_strength"] == fit["patients"][p]["stress_strength"]


def test_reference_is_one_batch_without_a_batch_effect_and_without_stress():
    f = _fits()
    for d in BL.DESIGNS:
        s = BL.level_settings(f, PATIENTS, d, "reference")
        assert len(s["batches"]) == 1
        kw = s["batches"][0]
        assert kw["sigma_batch"] == 0.0 and kw["depth_batch_sigma"] == 0.0
        assert all(v["stress_strength"] == 0.0 for v in s["patients"].values())


def test_recovery_is_the_patients_own_at_every_level():
    f = _fits()
    rec = None
    for d in BL.DESIGNS:
        for lv, s in BL.levels(f, PATIENTS, d).items():
            r = {p: v["recovery_probs"] for p, v in s["patients"].items()}
            assert r == {str(p): f["recovery"]["per_patient"][str(p)] for p in PATIENTS}
            rec = rec or r
            assert r == rec


def test_unfitted_ambient_needs_the_registered_range_in_the_fits():
    f = _fits(ambient=False)
    with pytest.raises(RuntimeError, match="registered range"):
        BL.level_settings(f, PATIENTS, "balanced", "fitted")
    s = BL.level_settings(_with_range(f), PATIENTS, "balanced", "fitted")
    assert all(0.03 <= kw["ambient_frac"] <= 0.11 for kw in s["batches"])   # the soup share itself
    assert all(src.startswith("drawn") for src in s["batch_ambient_source"])
    # one draw per batch, the same at every level that varies the ambient fraction
    arm = BL.level_settings(_with_range(f), PATIENTS, "balanced", "ambient")
    assert [kw["ambient_frac"] for kw in arm["batches"]] == [kw["ambient_frac"] for kw in s["batches"]]
    assert len({kw["ambient_frac"] for kw in s["batches"]}) == len(s["batches"])
    # with nothing fitted, the shared level sits at the middle of the range
    sh = BL.level_settings(_with_range(f), PATIENTS, "balanced", "shared")
    assert sh["batches"][0]["ambient_frac"] == pytest.approx(0.07)


def test_partly_fitted_ambient_mixes_fits_and_draws():
    f = _fits()
    f["wu"]["vectors"]["T1"]["ambient_frac"] = None
    f["wu"]["ambient"]["needs_range"] = ["T1"]
    with pytest.raises(RuntimeError):
        BL.level_settings(f, PATIENTS, "balanced", "fitted", n_balanced_batches=30)
    s = BL.level_settings(_with_range(f), PATIENTS, "balanced", "fitted", n_balanced_batches=30)
    for kw, t, src in zip(s["batches"], s["batch_tumours"], s["batch_ambient_source"]):
        if t == "T1":
            assert src.startswith("drawn")
        else:
            assert src == "fitted" and kw["ambient_frac"] == pytest.approx(f["wu"]["vectors"][t]["ambient_frac"])
    # the median is over the fitted values only
    fitted = [v["ambient_frac"] for t, v in f["wu"]["vectors"].items() if t != "T1"]
    assert s["median_vector"]["ambient_frac"] == pytest.approx(np.median(fitted))


def test_too_many_patients_for_the_recovery_draws():
    f = _fits(n_patients=5)
    with pytest.raises(ValueError, match="recovery"):
        BL.level_settings(f, PATIENTS, "balanced", "fitted")


def test_the_overrides_drive_run_scrna_batches():
    import pandas as pd
    from iscc.data.rna import run_scrna_batches
    f = _fits()
    s = BL.level_settings(f, PATIENTS, "balanced", "fitted", n_balanced_batches=3)
    rng = np.random.default_rng(1)
    exp = pd.DataFrame(rng.lognormal(0, 1, (90, 30)), index=[f"C{i}" for i in range(90)],
                       columns=[f"g{i}" for i in range(30)])
    assays = run_scrna_batches({"cell_exp": exp}, n_batches=3, design="split", n_cells=30,
                               batch_kwargs=s["batches"])
    for a, kw in zip(assays, s["batches"]):
        assert a.hypers.mu_lib == pytest.approx(kw["mu_lib"])
        assert a.hypers.depth_batch_sigma == 0.0 and a.hypers.sigma_batch == BL.SIGMA_BATCH
