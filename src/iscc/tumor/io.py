"""Saving and loading grown tumours and cohorts.

A saved file is a short header (the format version, the ``iscc`` version that wrote it and the class
of the object) followed by the object, pickled and compressed with lzma. The per-cell tables
(``cell_data``) are left out, since ``make_cell_data`` rebuilds them from seeded streams: a loaded
tumour rebuilds them at once, as ``grow`` does, and a loaded cohort rebuilds those of a patient when
they are first read. Everything else is kept, so a loaded tumour can be grown further, sampled and
measured exactly as the saved one.

Genotype ids come from a process-wide counter (``Cell._genotype_counter``) that starts at zero in
every process, so loading moves it past every id the loaded tumours hold; otherwise a clone born
after loading could take the id of one already in the genealogy.

Clones share most of their state with their parents, so the pickle compresses well: a
25,000-cell tumour of the tutorial configuration takes about 5 MB, against about 140 MB unpacked.
"""
import json
import lzma
import pickle
import warnings
from importlib.metadata import PackageNotFoundError, version

MAGIC = b"ISCC-SAVE\n"
FORMAT = 1


def iscc_version():
    try:
        return version("iscc")
    except PackageNotFoundError:
        return "unknown"


def _tumours(obj):
    """The tumours an object holds: itself, or the patients of a cohort."""
    if hasattr(obj, "patients"):
        return [p.tumor for p in obj.patients]
    return [obj]


def save(obj, path):
    """Write ``obj`` (a tumour or a cohort) to ``path`` and return ``path``."""
    tumours = [t for t in _tumours(obj) if hasattr(t, "cell_data")]
    kept = [t.cell_data for t in tumours]
    built = obj.cell_data is not None and not getattr(obj, "_cell_data_stale", False) \
        if not hasattr(obj, "patients") else None
    try:
        for t in tumours:
            t.cell_data = None
        payload = lzma.compress(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL), preset=6)
    finally:
        for t, c in zip(tumours, kept):
            t.cell_data = c
    header = {"format": FORMAT, "iscc": iscc_version(), "kind": type(obj).__name__,
              "cell_data": built}
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(json.dumps(header).encode() + b"\n")
        f.write(payload)
    return path


def read_header(path):
    """The header of a saved file, as a dict."""
    with open(path, "rb") as f:
        if f.readline() != MAGIC:
            raise ValueError(f"{path} is not a file written by iscc's save")
        return json.loads(f.readline())


def load(path, cls):
    """Read an object of class ``cls`` saved with :func:`save`."""
    with open(path, "rb") as f:
        if f.readline() != MAGIC:
            raise ValueError(f"{path} is not a file written by iscc's save")
        header = json.loads(f.readline())
        payload = f.read()
    if header["format"] != FORMAT:
        raise ValueError(f"{path} has save format {header['format']}; this iscc reads format {FORMAT}")
    if header["kind"] != cls.__name__:
        raise TypeError(f"{path} holds a {header['kind']}, not a {cls.__name__}: "
                        f"load it with {header['kind']}.load")
    if header["iscc"] != iscc_version():
        warnings.warn(f"{path} was saved with iscc {header['iscc']} and is loaded with iscc "
                      f"{iscc_version()}", stacklevel=3)
    try:
        obj = pickle.loads(lzma.decompress(payload))
    except (AttributeError, ImportError, TypeError) as e:
        raise ValueError(f"{path} was saved with iscc {header['iscc']} and cannot be loaded with "
                         f"iscc {iscc_version()}; load it with iscc {header['iscc']}") from e
    _advance_genotype_counter(obj)
    return obj


def _advance_genotype_counter(obj):
    """Move the process-wide genotype counter past every numeric genotype id of ``obj``."""
    from .components.cell import Cell
    top = Cell._genotype_counter
    for t in _tumours(obj):
        for ids in (getattr(t, "genotypes", {}), getattr(t, "genotypes_parents", {})):
            top = max([top] + [int(g) for g in ids if str(g).isdigit()])
    Cell._genotype_counter = top
