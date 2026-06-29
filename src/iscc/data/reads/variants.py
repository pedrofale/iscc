"""Shared variant-injection seam (DESIGN_features §C, §E `reads/variants.py`).

The one genuinely modality-agnostic piece of read realism: given the molecules at a locus
and a per-locus alt-fraction, partition them into **alt-carrying** vs **ref-carrying**
counts, **preserving the total exactly**. Keyed on a generic ``(total, alt_fraction)`` so it
is reused unchanged by both read backends:

    DNA  — ``total`` = coverage at the locus (∝ copy number, from the C1 model),
           ``alt_fraction`` = DNA-VAF (`cell_snv`, the true alt fraction).
    RNA  — ``total`` = UMI count from the F3 expression matrix (conserved),
           ``alt_fraction`` = observed RNA-VAF (`cell_rna_vaf` × `obs_fidelity`).

Only the count split lives here; the sequence-level placement (which base, which read) is the
adapter's job. This keeps the "unify at the count matrix + a shared variant layer" contract.
"""
from __future__ import annotations

from collections import namedtuple

import numpy as np

#: Result of a split. ``alt + ref == total`` element-wise, always.
AlleleSplit = namedtuple("AlleleSplit", ["alt", "ref", "total"])


def effective_alt_prob(alt_fraction, error_rate):
    """Fold a symmetric per-base error floor into the alt probability.

    ``p = v*(1-e) + (1-v)*e`` — false-alt reads on true-ref molecules (the floor) and
    false-ref reads on true-alt molecules — so even ``v=0`` emits alt at rate ``e`` and
    ``v=1`` emits ref at rate ``e``. Clipped to ``[0, 1]``.
    """
    v = np.asarray(alt_fraction, dtype=float)
    e = np.asarray(error_rate, dtype=float)
    return np.clip(v * (1.0 - e) + (1.0 - v) * e, 0.0, 1.0)


def inject(total, alt_fraction, error_rate=0.0, rng=None):
    """Partition `total` molecules at a locus into alt vs ref, preserving the total.

    ``alt ~ Binomial(total, p_eff)`` with ``p_eff`` the error-floored alt probability;
    ``ref = total - alt`` so the total is conserved **exactly** (no molecules created or
    destroyed). Scalars and arrays both work (broadcast like numpy); the observed alt
    fraction equals `alt_fraction` in expectation (up to the error floor).

      total         int or array of non-negative molecule counts (coverage / UMIs).
      alt_fraction  in [0, 1], the per-locus true alt fraction (broadcast to `total`).
      error_rate    per-base error floor (scalar or array).
      rng           a numpy Generator (defaults to a fresh default_rng()).

    Returns an `AlleleSplit(alt, ref, total)` of integer arrays (or scalars for scalar input).
    """
    if rng is None:
        rng = np.random.default_rng()
    total_arr = np.asarray(total)
    scalar = total_arr.ndim == 0
    n = np.asarray(total_arr, dtype=np.int64)
    if np.any(n < 0):
        raise ValueError("`total` must be non-negative")
    p = np.broadcast_to(effective_alt_prob(alt_fraction, error_rate), n.shape)
    alt = rng.binomial(n, p)
    ref = n - alt
    if scalar:
        return AlleleSplit(int(alt), int(ref), int(n))
    return AlleleSplit(alt.astype(np.int64), ref.astype(np.int64), n)
