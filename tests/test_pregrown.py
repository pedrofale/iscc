"""The pregrown loaders find, check and load their files."""
import functools
import json
from importlib import resources

import pytest

from iscc import pregrown
from iscc.cohort import Cohort, grow_tumor
from iscc.data.human.reference import _sha256
from iscc.tumor import GenotypeTumor
from test_save_load import _grow, PARAMS


@pytest.fixture
def shelf(tmp_path, monkeypatch):
    """A cache folder holding one tumour and one cohort, and a manifest naming them."""
    tumour = _grow().save(tmp_path / "ductal.iscc")
    c = Cohort(functools.partial(grow_tumor, grow_steps=30, **PARAMS), patient_seeds=[1, 2]).run()
    c.save(tmp_path / "cohort.iscc")
    files = {"ductal": {"file": "ductal.iscc", "sha256": _sha256(tumour), "kind": "GenotypeTumor"},
             "cohort": {"file": "cohort.iscc", "sha256": _sha256(tmp_path / "cohort.iscc"),
                        "kind": "Cohort"}}
    monkeypatch.setattr(pregrown, "_manifest", lambda: {"base_url": None, "files": files})
    monkeypatch.setenv(pregrown.ENV_VAR, str(tmp_path))
    return tmp_path, files


def test_loads_a_tumour_and_a_cohort_from_the_cache(shelf):
    assert isinstance(pregrown.ductal(), GenotypeTumor)
    assert isinstance(pregrown.cohort(), Cohort)


def test_a_changed_file_is_refused(shelf):
    folder, _ = shelf
    with open(folder / "ductal.iscc", "ab") as f:
        f.write(b"x")
    with pytest.raises(ValueError, match="checksum"):
        pregrown.ductal()


def test_a_missing_file_without_an_address_says_how_to_provide_it(shelf):
    folder, _ = shelf
    (folder / "ductal.iscc").unlink()
    with pytest.raises(FileNotFoundError, match="ISCC_PREGROWN_DIR"):
        pregrown.ductal()


def test_shipped_manifest_is_valid_json():
    m = json.loads(resources.files("iscc.pregrown").joinpath("manifest.json").read_text())
    assert set(m) == {"base_url", "files"}


def test_pregrown_is_reachable_from_a_plain_import():
    import iscc
    assert iscc.pregrown.ductal is pregrown.ductal
