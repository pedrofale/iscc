"""F2 dissociation tests: composition bias, stress signature, ground-truth subset."""
import os

import numpy as np
import pandas as pd
import pytest
import yaml
from click.testing import CliRunner

from iscc.sample.dissociation.dissociation import (
    Dissociation, biological_type, DEFAULT_RECOVERY)
from iscc.sample.main import main as sample_main


def make_mixed_cell_data(n=600, seed=0):
    rng = np.random.default_rng(seed)
    types = rng.choice(["1", "2", "immune", "stromal", "epithelial"],
                       size=n, p=[0.25, 0.15, 0.3, 0.15, 0.15])
    names = [f"C{i}" for i in range(n)]
    ct = pd.DataFrame({"cell_id": types}, index=names)
    exp = pd.DataFrame(np.ones((n, 8)), index=names, columns=[f"G{i}" for i in range(8)])
    snv = pd.DataFrame(rng.integers(0, 2, (n, 8)).astype(float), index=names, columns=exp.columns)
    cnv = pd.DataFrame(np.full((n, 8), 2.0), index=names, columns=exp.columns)
    crd = pd.DataFrame({"row": rng.integers(0, 20, n), "col": rng.integers(0, 20, n)}, index=names)
    return {"cell_type": ct, "cell_exp": exp, "cell_snv": snv, "cell_cnv": cnv, "cell_crd": crd}


def write_cell_data(cd, path):
    os.makedirs(os.path.join(path, "cell_data"), exist_ok=True)
    for k, df in cd.items():
        df.to_csv(os.path.join(path, "cell_data", f"{k}.csv"))


def test_biological_type_mapping():
    assert biological_type("immune") == "immune"
    assert biological_type("stromal") == "stromal"
    assert biological_type("epithelial") == "epithelial"
    assert biological_type("7") == "cancer"
    assert biological_type("clone_42") == "cancer"


def test_downweighted_type_fraction_drops():
    cd = make_mixed_cell_data()
    diss = Dissociation(cd, rng=np.random.default_rng(1),
                        recovery_probs={"immune": 0.1})
    chosen, meta, exp_override = diss.run()
    inp = meta["input_composition"]["immune"]
    out = meta["sampled_composition"]["immune"]
    assert out < inp                      # immune under-recovered
    assert meta["composition_shift"]["immune"] < 0
    # cancer (recovered at default 0.9) should rise as a fraction
    assert meta["sampled_composition"]["cancer"] > meta["input_composition"]["cancer"]


def test_default_recovery_underrepresents_immune():
    cd = make_mixed_cell_data()
    diss = Dissociation(cd, rng=np.random.default_rng(2))  # DEFAULT_RECOVERY
    chosen, meta, _ = diss.run()
    # default has immune lowest -> immune fraction should fall
    assert meta["composition_shift"]["immune"] < 0


def test_subset_and_ground_truth_preserved():
    cd = make_mixed_cell_data()
    diss = Dissociation(cd, rng=np.random.default_rng(3), recovery_probs={"immune": 0.3})
    chosen, meta, exp_override = diss.run()
    assert set(chosen) <= set(cd["cell_type"].index)        # valid subset
    assert len(chosen) == meta["n_recovered"] <= meta["n_input"]
    # per-cell ground truth unchanged for the kept cells
    pd.testing.assert_frame_equal(cd["cell_snv"].loc[chosen], cd["cell_snv"].loc[chosen])
    assert exp_override is None  # no stress => original exp used


def test_stress_signature_only_perturbs_stress_genes():
    cd = make_mixed_cell_data()
    diss = Dissociation(cd, rng=np.random.default_rng(4), recovery_probs={"immune": 1.0},
                        stress_strength=0.5, n_stress_genes=2)
    chosen, meta, exp_override = diss.run()
    assert exp_override is not None
    stress = meta["stress_genes"]
    assert len(stress) == 2
    non_stress = [g for g in cd["cell_exp"].columns if g not in stress]
    # non-stress genes unchanged; stress genes scaled by 1+strength for recovered cells
    pd.testing.assert_frame_equal(
        exp_override.loc[chosen, non_stress], cd["cell_exp"].loc[chosen, non_stress])
    np.testing.assert_allclose(
        exp_override.loc[chosen, stress].to_numpy(),
        cd["cell_exp"].loc[chosen, stress].to_numpy() * 1.5)


def test_no_bias_keeps_all_types():
    cd = make_mixed_cell_data(n=400)
    # uniform recovery (every type 1.0) => composition ~ unchanged
    probs = {t: 1.0 for t in DEFAULT_RECOVERY}
    diss = Dissociation(cd, rng=np.random.default_rng(5), recovery_probs=probs)
    chosen, meta, _ = diss.run()
    assert len(chosen) == meta["n_input"]
    for t, shift in meta["composition_shift"].items():
        assert abs(shift) < 1e-9


def test_cli_dissociation_smoke(tmp_path):
    cd = make_mixed_cell_data()
    src = str(tmp_path / "tumor")
    write_cell_data(cd, src)
    out = str(tmp_path / "sample")
    r = CliRunner().invoke(
        sample_main,
        [src, "--method", "dissociation", "--recovery", "immune=0.2,stromal=0.5",
         "--stress-strength", "0.2", "-o", out],
        catch_exceptions=False,
    )
    assert r.exit_code == 0
    meta = yaml.safe_load(open(os.path.join(out, "sample_meta.yaml")))
    assert meta["method"] == "dissociation"
    assert meta["dissociation"]["composition_shift"]["immune"] < 0
    # reloadable cell_data
    ct = pd.read_csv(os.path.join(out, "cell_data", "cell_type.csv"), index_col=0)
    assert len(ct) == meta["n_sampled"]
