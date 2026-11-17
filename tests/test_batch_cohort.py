"""Fig 5's cohort scRNA at the registered batch levels (validation/paper/batch_cohort.py), end to end
on a toy cohort: cells stored as ``fig5_atlas.py cells`` stores them (unstressed, the registered
recovery in their record), counts made at two levels in both designs with per-batch overrides and
the shared stress genes, the reference, and integrate.py run on every design (method ``none``,
without metrics, in this environment's interpreter).
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "validation", "paper"))
import batch_cohort as BC  # noqa: E402
import batch_fits as BF  # noqa: E402

N_PAT, N_CELLS, N_GENES = 3, 60, 80


def _toy_fits(rec):
    vec = {f"T{i}": dict(mu_lib=1500.0 + 1000 * i, sigma_lib=0.3 + 0.1 * i, dispersion=0.2 + 0.1 * i,
                         dropout_mid=(0.0 if i == 0 else 0.4), dropout_shape=(None if i == 0 else 1.5),
                         ambient_frac=0.02 + 0.02 * i, doublet_rate=0.01 * (i + 1), stress_strength=0.5 * i)
           for i in range(3)}
    return {"wu": {"vectors": vec, "ambient": {"needs_range": []}}, "ding": {"needs_range": []},
            "recovery": {"per_patient": rec}, "ambient_range": None}


def _write_cells(out, rec):
    rng = np.random.default_rng(0)
    genes = np.array([f"G{i}" for i in range(N_GENES)])
    os.makedirs(os.path.join(out, "cells"))
    for p in range(N_PAT):
        ct = rng.choice(["cancer", "immune", "stromal"], size=N_CELLS)
        meta = dict(settings=dict(recovery_probs=rec[str(p)], stress_strength=0.0), step="cells")
        np.savez_compressed(
            os.path.join(out, "cells", f"patient{p:02d}.npz"), meta=np.asarray(json.dumps(meta)),
            cells=np.array([f"C{i}" for i in range(N_CELLS)]), genotype=np.where(ct == "cancer", "7", ct),
            cell_type=ct, is_cancer=ct == "cancer", clone=np.where(ct == "cancer", "clone 1", "normal"),
            genes=genes, exp=rng.lognormal(0, 1, (N_CELLS, N_GENES)).astype(np.float32))


@pytest.fixture
def cohort(tmp_path, monkeypatch):
    monkeypatch.setenv("ISCC_WORK_DIR", str(tmp_path / "work"))
    rec = BF.recovery_draws(N_PAT)["per_patient"]
    out = str(tmp_path / "fig5")
    _write_cells(out, rec)
    fits = str(tmp_path / "fits.json")
    with open(fits, "w") as f:
        json.dump(_toy_fits(rec), f)
    return out, fits, rec


def _run(out, fits, level):
    return BC.main(["integrate", "--out", out, "--fits", fits, "--level", level, "--n-balanced-batches", "4",
                    "--methods", "none", "--no-metrics", "--no-embeddings",
                    "--integration-python", sys.executable])


def test_integrate_at_two_levels_end_to_end(cohort):
    out, fits, rec = cohort
    res = {}
    for level in ("fitted", "shared"):
        path = _run(out, fits, level)
        with open(path) as f:
            res[level] = json.load(f)
        s = res[level]
        assert s["level"] == level and s["n_cells"] == N_PAT * N_CELLS
        for key in ("confounded/none", "balanced/none"):
            assert "error" not in s["runs"][key], s["log"]
        assert all(r["returncode"] == 0 for r in s["log"])
        # per-batch overrides in both designs, recorded
        assert len(s["level_settings"]["confounded"]["batches"]) == N_PAT
        assert len(s["level_settings"]["balanced"]["batches"]) == 4
        assert s["reference_settings"]["batches"][0]["sigma_batch"] == 0.0
        assert len(s["stress_genes"]) == 4                     # 5% of 80 genes, one shared set
    fit, sh = res["fitted"]["level_settings"], res["shared"]["level_settings"]
    mus = {kw["mu_lib"] for kw in fit["balanced"]["batches"]}
    assert len({kw["mu_lib"] for kw in sh["balanced"]["batches"]}) == 1
    assert mus <= {1500.0, 2500.0, 3500.0}
    assert res["fitted"]["stress_genes"] == res["shared"]["stress_genes"]
    assert all(kw["depth_batch_sigma"] == 0.0 for kw in fit["confounded"]["batches"])
    assert all(kw["depth_batch_sigma"] == 0.05 for kw in sh["confounded"]["batches"])


def test_the_reference_level_runs_alone(cohort):
    out, fits, _ = cohort
    with open(_run(out, fits, "reference")) as f:
        s = json.load(f)
    assert s["runs"] == {} and {r["run"] for r in s["log"]} == {"truth_confounded_none", "truth_balanced_none"}


def test_cells_without_the_registered_recovery_are_refused(cohort, tmp_path):
    out, fits, rec = cohort
    bad = dict(rec)
    bad["1"] = {k: 0.5 for k in rec["1"]}
    with open(fits) as f:
        fj = json.load(f)
    fj["recovery"]["per_patient"] = bad
    with open(fits, "w") as f:
        json.dump(fj, f)
    with pytest.raises(SystemExit, match="registered recovery"):
        _run(out, fits, "shared")


def test_stress_is_applied_per_patient_to_the_shared_genes():
    exp = pd.DataFrame(np.ones((4, 5)), index=list("abcd"), columns=[f"g{i}" for i in range(5)])
    got = BC.stressed(exp, np.array(["0", "0", "1", "2"]), {"0": 0.5, "1": 0.0, "2": 2.0}, ["g1", "g3"])
    from iscc.sample.dissociation import apply_stress
    want = apply_stress(apply_stress(exp, 0.5, ["g1", "g3"], cells=["a", "b"]), 2.0, ["g1", "g3"], cells=["d"])
    pd.testing.assert_frame_equal(got, want)
    assert (exp.values == 1).all()                              # the input is untouched


def test_dissociation_file_matches_the_fits_recovery(tmp_path):
    path = BC.dissociation_file(str(tmp_path / "d.json"), n_patients=5)
    with open(path) as f:
        d = json.load(f)
    rec = BF.recovery_draws(5)["per_patient"]
    for i in range(5):
        assert d[str(i)] == dict(recovery_probs=rec[str(i)], stress_strength=0.0)
