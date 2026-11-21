"""Tumours and a cohort grown once with ``iscc``, to generate data from.

Each function returns a grown [`GenotypeTumor`][iscc.tumor.GenotypeTumor] (or, for the cohort, a
[`Cohort`][iscc.cohort.Cohort]) that can be resected, sampled and measured as one grown in the session.
Growth from the same seed can differ between processor architectures, so these tumours, grown once
on one machine, are what makes an analysis repeatable everywhere.

The first call downloads the file, checks its SHA-256 checksum against the one that ships with the
package and keeps it in a cache folder: ``ISCC_PREGROWN_DIR`` if it is set, otherwise
``iscc/pregrown`` in the user cache folder (``$XDG_CACHE_HOME``, by default ``~/.cache``). Later
calls load the cached file.

The four tumours grow in the same ductal field of the paper, on the arm-level human genome of
[`arm_genome`][iscc.data.human.arm_genome] with the epithelium as the cell of origin, and differ in
their evolution. ``scripts/build_pregrown.py`` grows them from ``configs/pregrown/``.
"""
import json
import os
import tempfile
import urllib.request
from importlib import resources
from pathlib import Path

from ..data.human.reference import _sha256

ENV_VAR = "ISCC_PREGROWN_DIR"

__all__ = ["ductal", "neutral", "sweeps", "cin", "cohort"]


def _manifest():
    return json.loads(resources.files(__package__).joinpath("manifest.json").read_text())


def _cache_dir():
    if os.environ.get(ENV_VAR):
        return Path(os.environ[ENV_VAR])
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "iscc" / "pregrown"


def _path(name):
    """The checked local file of the pregrown object ``name``, downloaded first if needed."""
    manifest = _manifest()
    entry = manifest["files"].get(name)
    if entry is None:
        raise KeyError(f"no pregrown {name!r} in this version of iscc")
    folder = _cache_dir()
    path = folder / entry["file"]
    if not path.exists():
        if not manifest.get("base_url"):
            raise FileNotFoundError(f"{entry['file']} is not in {folder} and this version of iscc "
                                    f"names no download address; set {ENV_VAR} to a folder that "
                                    f"holds it")
        folder.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=folder, delete=False) as tmp:
            urllib.request.urlretrieve(manifest["base_url"] + entry["file"], tmp.name)
        os.replace(tmp.name, path)
    got = _sha256(path)
    if got != entry["sha256"]:
        raise ValueError(f"{path}: checksum {got} does not match the expected {entry['sha256']}; "
                         f"delete the file to download it again")
    return path


def _grown_elsewhere(**kwargs):
    """The growth function of the pregrown cohort, which grows no further patients."""
    raise RuntimeError("the pregrown cohort was grown with the paper's regime by "
                       "scripts/build_pregrown.py of the iscc repository; build a Cohort with your "
                       "own growth function to grow more patients")


def _load(name):
    entry = _manifest()["files"].get(name, {})
    if entry.get("kind") == "Cohort":
        from ..cohort import Cohort
        return Cohort.load(_path(name))
    from ..tumor import GenotypeTumor
    return GenotypeTumor.load(_path(name))


def ductal():
    """The tumour of the paper's regime: growth inside the glands, breach of the duct wall and
    invasion of the stroma, with selection by niche.

    Returns
    -------
    GenotypeTumor
    """
    return _load("ductal")


def neutral():
    """A tumour whose drivers have no effect on fitness, so its clones branch without sweeps. It
    keeps the niche traits of the paper's regime, which it needs to leave the glands.

    Returns
    -------
    GenotypeTumor
    """
    return _load("neutral")


def sweeps():
    """A tumour with fewer and stronger drivers than the paper's regime, so that new drivers sweep
    and a few large, nested clones dominate.

    Returns
    -------
    GenotypeTumor
    """
    return _load("sweeps")


def cin():
    """A tumour with a high rate of copy-number events and of genome doubling under per-arm
    copy-number selection, so that its clones differ mostly in copy number.

    Returns
    -------
    GenotypeTumor
    """
    return _load("cin")


def cohort():
    """Six patients grown over one configuration of the paper's regime, for analyses across
    patients such as batch integration.

    Returns
    -------
    Cohort
    """
    return _load("cohort")
