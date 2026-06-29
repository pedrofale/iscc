"""F1 biopsy tests: spatial geometry, multi-region heterogeneity, liquid bias."""
import os

import numpy as np
import pandas as pd
import pytest
import yaml
from click.testing import CliRunner

from iscc.sample.biopsy.biopsy import Biopsy
from iscc.sample.main import main as sample_main


def make_cell_data(n_per_clone=40, grid=40, seed=0):
    """Synthetic cell_data with 4 spatially-clustered clones (one per quadrant)
    plus a sprinkling of normal cells, so a small region sees few clones."""
    rng = np.random.default_rng(seed)
    rows, cols, types, disp = [], [], [], []
    centers = {"1": (10, 10), "2": (10, 30), "3": (30, 10), "4": (30, 30)}
    for clone, (cr, cc) in centers.items():
        rows += list(np.clip(rng.normal(cr, 3, n_per_clone), 0, grid - 1))
        cols += list(np.clip(rng.normal(cc, 3, n_per_clone), 0, grid - 1))
        types += [clone] * n_per_clone
        # clones 3,4 are high-dispersal; 1,2 low.
        disp += list(rng.integers(2, 4, n_per_clone) if clone in ("3", "4")
                     else rng.integers(0, 1, n_per_clone))
    # a few normal cells scattered
    for t in ("immune", "stromal", "epithelial"):
        rows += list(rng.uniform(0, grid - 1, 10))
        cols += list(rng.uniform(0, grid - 1, 10))
        types += [t] * 10
        disp += [0] * 10
    n = len(types)
    names = [f"C{i}" for i in range(n)]
    crd = pd.DataFrame({"row": np.round(rows).astype(int),
                        "col": np.round(cols).astype(int)}, index=names)
    ct = pd.DataFrame({"cell_id": types}, index=names)
    evo = pd.DataFrame({"n_mut_disp": disp,
                        "dispersal_rate": rng.random(n)}, index=names)
    exp = pd.DataFrame(rng.random((n, 6)), index=names,
                       columns=[f"G{i}" for i in range(6)])
    snv = pd.DataFrame(rng.integers(0, 2, (n, 6)).astype(float),
                       index=names, columns=exp.columns)
    cnv = pd.DataFrame(np.full((n, 6), 2.0), index=names, columns=exp.columns)
    return {"cell_crd": crd, "cell_type": ct, "cell_evo": evo,
            "cell_exp": exp, "cell_snv": snv, "cell_cnv": cnv}


def write_cell_data(cd, path):
    os.makedirs(os.path.join(path, "cell_data"), exist_ok=True)
    for k, df in cd.items():
        df.to_csv(os.path.join(path, "cell_data", f"{k}.csv"))


# --------------------------------------------------------------- geometry
def test_punch_cells_inside_disk():
    cd = make_cell_data()
    bx = Biopsy(cd, rng=np.random.default_rng(1), grid_size=40)
    chosen, region, geom = bx.sample("punch", center=(10, 10), radius=6)
    sub = cd["cell_crd"].loc[chosen]
    c = np.array(geom["center"])
    d = np.sqrt((sub.row - c[0]) ** 2 + (sub.col - c[1]) ** 2)
    assert (d <= geom["radius"] + 1e-9).all()
    assert (region == "punch").all()


def test_needle_cells_inside_strip():
    cd = make_cell_data()
    bx = Biopsy(cd, rng=np.random.default_rng(1), grid_size=40)
    chosen, region, geom = bx.sample("needle", center=(20, 20), width=4, angle=0.0)
    sub = cd["cell_crd"].loc[chosen]
    c = np.array(geom["center"])
    theta = np.deg2rad(geom["angle"])
    d = np.array([np.cos(theta), np.sin(theta)])
    rel = sub[["row", "col"]].to_numpy() - c
    perp = np.abs(rel[:, 0] * d[1] - rel[:, 1] * d[0])
    assert (perp <= geom["width"] / 2 + 1e-9).all()


def test_multiregion_disjoint_and_labels_match_coords():
    cd = make_cell_data()
    bx = Biopsy(cd, rng=np.random.default_rng(3), grid_size=40)
    chosen, region, geom = bx.sample("multiregion", n_regions=4, radius=6)
    assert geom["k"] >= 2
    labels = set(region.unique())
    assert labels == {f"region_{i}" for i in range(geom["k"])}
    centers = np.array(geom["centers"])
    # every cell's region label matches the disk it falls in
    for name, lab in region.items():
        ri = int(lab.split("_")[1])
        r, c = cd["cell_crd"].loc[name, ["row", "col"]]
        d = np.hypot(r - centers[ri][0], c - centers[ri][1])
        assert d <= geom["radius"] + 1e-9
    # disjoint: centres pairwise >= 2*radius apart
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            assert np.hypot(*(centers[i] - centers[j])) >= 2 * geom["radius"] - 1e-9


def test_region_heterogeneity_subset_of_clones():
    cd = make_cell_data()
    bx = Biopsy(cd, rng=np.random.default_rng(5), grid_size=40)
    all_clones = set(cd["cell_type"]["cell_id"].unique())
    chosen, region, geom = bx.sample("punch", center=(10, 10), radius=6)
    region_clones = set(cd["cell_type"].loc[chosen, "cell_id"].unique())
    assert region_clones < all_clones  # strict subset
    # the punch over clone "1" should not contain the far clone "4"
    assert "4" not in region_clones


def test_multiregion_union_recovers_more_than_single():
    cd = make_cell_data()
    bx = Biopsy(cd, rng=np.random.default_rng(7), grid_size=40)
    chosen, region, geom = bx.sample("multiregion", n_regions=4, radius=6)
    ct = cd["cell_type"].loc[chosen, "cell_id"]
    union = ct.nunique()
    per_region = [ct[region.values == lab].nunique() for lab in region.unique()]
    assert union >= max(per_region)


# --------------------------------------------------------------- liquid biopsy
def test_liquid_enriched_for_high_dispersal():
    cd = make_cell_data()
    evo = cd["cell_evo"]
    liq_means, unif_means = [], []
    for s in range(20):
        bx = Biopsy(cd, rng=np.random.default_rng(s), grid_size=40)
        chosen, _, _ = bx.sample("liquid", n_liquid=20)
        liq_means.append(evo.loc[chosen, "n_mut_disp"].mean())
        unif = np.random.default_rng(100 + s).choice(evo.index, 20, replace=False)
        unif_means.append(evo.loc[unif, "n_mut_disp"].mean())
    assert np.mean(liq_means) > np.mean(unif_means)


def test_liquid_only_cancer_cells():
    cd = make_cell_data()
    bx = Biopsy(cd, rng=np.random.default_rng(2), grid_size=40)
    chosen, region, geom = bx.sample("liquid", n_liquid=30)
    sampled_types = cd["cell_type"].loc[chosen, "cell_id"]
    assert not sampled_types.isin(["immune", "stromal", "epithelial"]).any()
    assert (region == "blood").all()


def test_liquid_uniform_when_signal_absent():
    cd = make_cell_data()
    cd = {k: v for k, v in cd.items() if k != "cell_evo"}  # drop dispersal signal
    bx = Biopsy(cd, rng=np.random.default_rng(2), grid_size=40)
    chosen, region, geom = bx.sample("liquid", n_liquid=10)
    assert geom["dispersal_signal"] is None
    assert len(chosen) == 10


# --------------------------------------------------------------- CLI smoke
@pytest.mark.parametrize("biopsy_type", ["punch", "needle", "multiregion", "liquid"])
def test_cli_biopsy_writes_reloadable_output(tmp_path, biopsy_type):
    cd = make_cell_data()
    src = str(tmp_path / "tumor")
    write_cell_data(cd, src)
    out = str(tmp_path / f"sample_{biopsy_type}")
    r = CliRunner().invoke(
        sample_main,
        [src, "--method", "biopsy", "--biopsy-type", biopsy_type,
         "--radius", "8", "--width", "6", "--n-regions", "3", "--n-liquid", "15",
         "-o", out],
        catch_exceptions=False,
    )
    assert r.exit_code == 0
    meta = yaml.safe_load(open(os.path.join(out, "sample_meta.yaml")))
    assert meta["biopsy_type"] == biopsy_type
    assert meta["n_sampled"] <= meta["n_input"]
    # region labels persisted and reload aligned to cell_data
    region = pd.read_csv(os.path.join(out, "cell_data", "cell_region.csv"), index_col=0)
    crd = pd.read_csv(os.path.join(out, "cell_data", "cell_crd.csv"), index_col=0)
    assert list(region.index) == list(crd.index)
    assert region["region"].notna().all()
