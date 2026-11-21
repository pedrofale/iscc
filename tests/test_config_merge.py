"""A config and the keyword arguments beside it combine: blocks merge, explicit run settings win.

Before, a config replaced every keyword block, so ``GenotypeTumor(config=..., selection_params=...)``
silently dropped the selection parameters.
"""
import copy

import yaml

from conftest import GENOME_PARAMS, SELECTION_PARAMS, CANCER_CELL_PARAMS, DEME_PARAMS
from iscc.tumor import GenotypeTumor
from iscc.tumor.models.params import ModelBlocks, RunSettings, resolve_parameters

CONFIG = {"genome_params": GENOME_PARAMS,
          "selection_params": {**SELECTION_PARAMS, "driver_effects": 1.03},
          "deme_params": DEME_PARAMS,
          "spatial_params": {"grid_size": 12, "origin_confinement": {"dispersal_factor": 0.0,
                                                                     "generations": 200}},
          "cell_params": {"cancer": CANCER_CELL_PARAMS},
          "update_mode": "tau", "tau": 0.5}
NO_RUN = {f: None for f in RunSettings.__dataclass_fields__}


def _resolve(config=CONFIG, run=None, **blocks):
    return resolve_parameters(config, "abstract", None, {**NO_RUN, **(run or {})}, ModelBlocks(**blocks))


def test_a_block_beside_the_config_changes_only_its_keys():
    _, p = _resolve(selection={"driver_effects": 1.1})
    assert p.selection["driver_effects"] == 1.1
    assert p.selection["prop_driver"] == SELECTION_PARAMS["prop_driver"]


def test_nested_dicts_merge():
    _, p = _resolve(spatial={"origin_confinement": {"generations": 50}})
    assert p.spatial["origin_confinement"] == {"dispersal_factor": 0.0, "generations": 50}
    assert p.spatial["grid_size"] == 12


def test_explicit_run_settings_replace_the_config_and_the_rest_come_from_it():
    r, _ = _resolve(run={"update_mode": "exact"})
    assert r.update_mode == "exact" and r.tau == 0.5
    r, _ = _resolve()
    assert r.update_mode == "tau" and r.snapshot_every == RunSettings.snapshot_every


def test_readout_layers_stay_off_unless_one_side_sets_them():
    _, p = _resolve()
    assert p.expression is None and p.microenv is None
    _, p = _resolve(expression={"cell_of_origin": "epithelial"})
    assert p.expression == {"cell_of_origin": "epithelial"}


def test_a_dict_config_is_not_changed(tmp_path):
    before = copy.deepcopy(CONFIG)
    _resolve(selection={"driver_effects": 1.1}, spatial={"origin_confinement": {"generations": 5}})
    assert CONFIG == before


def test_a_path_and_a_dict_give_the_same_tumour(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(CONFIG))
    a = GenotypeTumor(config=str(path), seed=1, selection_params={"prop_driver": 0.2})
    b = GenotypeTumor(config=CONFIG, seed=1, selection_params={"prop_driver": 0.2})
    assert a.selection.prop_driver == b.selection.prop_driver == 0.2
    assert a.update_mode == b.update_mode == "tau"


def test_without_a_config_the_defaults_are_unchanged():
    r, _ = resolve_parameters(None, "abstract", None, NO_RUN, ModelBlocks(genome=GENOME_PARAMS))
    assert r == RunSettings(layout_seed=r.layout_seed)
