"""Shared IO for the analysis scripts.

Not named `_io`: that shadows CPython's built-in `_io` module on the script path and every
import of it fails with a confusing "unknown location" ImportError.
"""
import json, pickle

import numpy as np


def load_tumor(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _plain(o):
    """JSON-safe: numpy scalars/arrays, NaN -> null, tuples -> lists."""
    if isinstance(o, dict):
        return {str(k): _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return None if not np.isfinite(v) else v
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _plain(o.tolist())
    return o


def write_json(obj, path):
    with open(path, "w") as f:
        json.dump(_plain(obj), f, indent=2, sort_keys=True)
