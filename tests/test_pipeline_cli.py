"""End-to-end CLI pipeline test: isccsim -> isccsample -> isccdata.

Guards the stage-2 work (entry points + I/O schema) so the three commands stay
runnable and their on-disk contract stays stable.
"""
import os

import yaml
import pytest
from click.testing import CliRunner

from iscc.tumor.main import main as sim_main
from iscc.sample.main import main as sample_main
from iscc.data.main import main as data_main

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TUMORCFG = os.path.join(REPO_ROOT, "src/iscc/tumor/tumorconfigs/glandular.yaml")
ASSAYCFG = os.path.join(REPO_ROOT, "src/iscc/data/assayconfigs")


@pytest.fixture
def small_sim_config(tmp_path):
    cfg = yaml.safe_load(open(TUMORCFG))
    cfg["spatial_params"].update(grid_size=12, structure_radius=2, n_structures=1)
    cfg["genome_params"].update(n_segments=3, segment_size=30)
    cfg["cell_params"]["cancer"].update(mutation_rate=0.2)
    p = tmp_path / "sim.yaml"
    yaml.safe_dump(cfg, open(p, "w"))
    return str(p)


@pytest.fixture
def tumor_dir(tmp_path, small_sim_config):
    out = str(tmp_path / "tumor")
    r = CliRunner().invoke(
        sim_main, ["--sim-config", small_sim_config, "-s", "20", "-o", out],
        catch_exceptions=False,
    )
    assert r.exit_code == 0
    return out


@pytest.fixture
def sample_dir(tmp_path, tumor_dir):
    out = str(tmp_path / "sample")
    r = CliRunner().invoke(
        sample_main, [tumor_dir, "--method", "biopsy", "--fraction", "0.5", "-o", out],
        catch_exceptions=False,
    )
    assert r.exit_code == 0
    return out


def test_isccsim_writes_canonical_layout(tumor_dir):
    files = set(os.listdir(tumor_dir))
    assert {"cell_data", "gene_data", "genotypes.csv", "parents.csv",
            "trace_counts.csv", "grid.csv"} <= files
    cd = set(os.listdir(os.path.join(tumor_dir, "cell_data")))
    assert {"cell_snv.csv", "cell_cnv.csv", "cell_exp.csv", "cell_crd.csv",
            "cell_type.csv", "cell_deme.csv", "cell_evo.csv"} <= cd


def test_isccsample_subsets_and_writes_meta(sample_dir):
    meta = yaml.safe_load(open(os.path.join(sample_dir, "sample_meta.yaml")))
    assert meta["method"] == "biopsy"
    assert meta["n_sampled"] <= meta["n_input"]
    assert os.path.isdir(os.path.join(sample_dir, "cell_data"))


@pytest.mark.parametrize(
    "assay,cfg,expected",
    [
        ("scrna", "scrna.yaml", {"umis.csv"}),
        ("bdna", "bdna.yaml", {"counts.csv"}),
        ("scdna", "scdna.yaml", {"alt_counts.csv", "coverage.csv"}),
        ("visium", "visium.yaml", {"spot_umi.csv", "spot_crd.csv"}),
    ],
)
def test_isccdata_runs_each_assay(tmp_path, sample_dir, assay, cfg, expected):
    out = str(tmp_path / f"data_{assay}")
    r = CliRunner().invoke(
        data_main,
        [sample_dir, "-a", assay, "--assay-config", os.path.join(ASSAYCFG, cfg), "-o", out],
        catch_exceptions=False,
    )
    assert r.exit_code == 0
    assert expected <= set(os.listdir(out))
