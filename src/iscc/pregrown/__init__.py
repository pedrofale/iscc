"""Tumors and a cohort grown once with ``iscc``, to generate data from.

Each function returns a grown [`GenotypeTumor`][iscc.tumor.GenotypeTumor] (or, for the cohort, a
[`Cohort`][iscc.cohort.Cohort]) that can be resected, sampled and measured as one grown in the session.
Growth from the same seed can differ between processor architectures, so these tumors, grown once
on one machine, are what makes an analysis repeatable everywhere.

The first call downloads the file, checks its SHA-256 checksum against the one that ships with the
package and keeps it in a cache folder: ``ISCC_PREGROWN_DIR`` if it is set, otherwise
``iscc/pregrown`` in the user cache folder (``$XDG_CACHE_HOME``, by default ``~/.cache``). Later
calls load the cached file.

The five tumors grow in the ductal field of the paper, at its regime on the arm-level human genome of
[`arm_genome`][iscc.data.human.arm_genome] (the paper's run R11) with the epithelium as the cell of
origin, and differ in their evolution. [`ductal`][iscc.pregrown.ductal] grows without immune cells; the
other four tumors and the cohort grow with the regime's immune cells in the stroma, so that the data
generated from them hold the immune cells of a breast tumor. Their stroma holds fibroblasts,
endothelial cells and perivascular cells in the shares of a breast cancer atlas
(``stromal_composition``). They carry germline variants at the density of a genotyping
array (30% of the neutral genes heterozygous), so that the DNA assays carry the B-allele signal of
allele-specific callers, and no segment at zero copies. ``validation/paper/build_pregrown.py`` in the
repository grows them.
"""
import json
import os
import tempfile
import urllib.request
from importlib import resources
from pathlib import Path

from ..data.human.reference import _sha256

ENV_VAR = "ISCC_PREGROWN_DIR"

__all__ = ["ductal", "neutral", "sweeps", "cin", "immune", "cohort"]


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
                       "validation/paper/build_pregrown.py of the iscc repository; build a Cohort with your "
                       "own growth function to grow more patients")


def _load(name):
    entry = _manifest()["files"].get(name, {})
    if entry.get("kind") == "Cohort":
        from ..cohort import Cohort
        return Cohort.load(_path(name))
    from ..tumor import GenotypeTumor
    return GenotypeTumor.load(_path(name))


def ductal():
    """The tumor of the paper's regime: growth inside the glands, breach of the duct wall and
    invasion of the stroma, with selection by niche.

    200,284 cancer cells on the 1 cm field (grid 170, eight glands) after 299 generations; driver
    clones sweep in situ for about 200 generations before cancer cells invade the stroma from
    several glands, one large lesion and smaller ones, and some glands still hold small in-situ
    tumors. 7% of the cancer cells have a doubled genome. About 40 MB.

    Returns
    -------
    GenotypeTumor
        The grown tumor, read from the cache folder.
    """
    return _load("ductal")


def neutral():
    """A tumor grown without selection: drivers without effect, no niche traits, and the duct wall
    and the stroma open to every cell.

    The stroma holds the regime's immune cells, which kill nothing, since immune killing selects
    resistance. 200,002 cancer cells on the 1 cm field after 247 generations, in six lesions that are
    each a mosaic of clones drifting without sweeps. A negative control for methods that look for
    selection or subclonal structure. About 15 MB.

    Returns
    -------
    GenotypeTumor
        The grown tumor, read from the cache folder.
    """
    return _load("neutral")


def sweeps():
    """A tumor with fewer and stronger drivers than the paper's regime, so that new drivers sweep
    and a few large, nested clones dominate.

    Grown with the immune cells of [`immune`][iscc.pregrown.immune]. 200,003 cancer cells on the 1 cm
    field after 310 generations, mostly in one large lesion through which a late driver clone swept,
    and 96% of the cancer cells have a doubled genome. About 96 MB.

    Returns
    -------
    GenotypeTumor
        The grown tumor, read from the cache folder.
    """
    return _load("sweeps")


def cin():
    """A chromosomally unstable tumor: four times the copy-number events and five times the genome
    doubling of the paper's regime, so that its clones differ mostly in copy number.

    Grown with the immune cells of [`immune`][iscc.pregrown.immune]. 200,045 cancer cells on the 1 cm
    field after 284 generations; 96% of the cancer cells have a doubled genome (mean ploidy 4.0), and
    lesions grow from several glands. About 81 MB.

    Returns
    -------
    GenotypeTumor
        The grown tumor, read from the cache folder.
    """
    return _load("cin")


def immune():
    """The tumor of the paper's regime with immune cells in the stroma: T and NK cells, which kill the
    cancer cells around them unless their immune resistance protects them and whose interferon-gamma
    induces the interferon response in nearby cancer cells, and myeloid cells and B and plasma cells,
    which kill nothing.

    Grown like [`ductal`][iscc.pregrown.ductal], with six immune cells in every stroma deme, split into
    the three subtypes with the shares of a breast cancer atlas, and none in the ducts. The immune cells
    neither divide nor move, and each cancer birth that overfills a deme removes one of them, so they
    stay in the stroma and at the invasive front and are lost from tumor nests; the human genome layer
    gives each subtype its baseline from the Human Breast Cell Atlas.

    200,064 cancer cells on the 1 cm field after 538 generations, against 299 for
    [`ductal`][iscc.pregrown.ductal], since the immune cells kill cancer cells in the stroma. Two
    large lesions, each dominated by one driver clone, grew from the glands, and every cancer cell has a
    doubled genome (mean ploidy 4.4). About 57 MB.

    Returns
    -------
    GenotypeTumor
        The grown tumor, read from the cache folder.
    """
    return _load("immune")


def cohort():
    """Six patients grown over one configuration of the paper's regime with immune cells (that of
    [`immune`][iscc.pregrown.immune]), for analyses across patients such as batch integration, where
    immune cells are the commonest cell state the patients share.

    Each patient grows from its own seed to 60,000 cancer cells on the 96 by 96 field, the scale of
    the paper's seeding runs. Four patients reached that size after 242 to 299 generations; the other
    two took 426 and 512, and every cancer cell of these two has a doubled genome. About 96 MB.

    Returns
    -------
    Cohort
        The six grown patients, read from the cache folder.
    """
    return _load("cohort")
