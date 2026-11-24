"""OncoSpot's malignant expression law, for the OncoSpot arm of Fig 3f.

OncoSpot (Bhattarai et al. 2026; code ``compbio-mallory/OncoSpot``, ``malignant_counts.py``) draws the
mean expression of gene g in malignant cell i as

    mu_gi = L_i * exp(alpha_g + beta_g * log(CN_gc / 2) + eta_gi),    mu_gi = 0 at CN_gc = 0,

where alpha_g is the baseline of the cell of origin, CN_gc the copy number of the cell's clone (one
profile per clone, clones under 1% of malignant cells merged into a relative), beta_g a dosage
exponent and eta_gi a programme-state term whose state vector is drawn independently of the cell's
lineage and position. Only HGSOC has per-gene exponents (calibrated on SA1053, global 0.415); CRC and
HCC use a fixed beta = 0.4 (``parameters/CRC/malignant/expression/expression_metadata.yaml``, marked
"illustrative_positive_fixed_coefficient"). There is no calibration for breast, so the fixed 0.4 is
the default here.
"""
from collections import Counter

import numpy as np

BETA_DEFAULT = 0.4
MERGE_FRAC = 0.01


def expression(baseline, cn, beta=BETA_DEFAULT, state_multiplier=None):
    """Mean expression under OncoSpot's law, (n_cells, n_genes).

    ``baseline`` is the diploid expression of each gene (n_genes,), ``cn`` the total copy number of
    each cell's clone at each gene (n_cells, n_genes), ``beta`` a scalar or per-gene exponent, and
    ``state_multiplier`` exp(eta), (n_cells, n_genes), or None for no programme term."""
    cn = np.asarray(cn, dtype=float)
    beta = np.broadcast_to(np.asarray(beta, dtype=float), cn.shape[1:])
    with np.errstate(divide="ignore"):
        dosage = np.where(cn > 0, np.exp(beta * np.log(np.maximum(cn, 1e-12) / 2.0)), 0.0)
    mu = np.asarray(baseline, dtype=float)[None, :] * dosage
    if state_multiplier is not None:
        mu = mu * np.asarray(state_multiplier, dtype=float)
    return mu


def modal_profiles(clone_of, profile_of):
    """The most common copy-number profile of each clone.

    ``clone_of`` maps cell -> clone and ``profile_of`` cell -> hashable profile, over the same cells.
    Ties go to the profile that sorts first, so the result does not depend on cell order."""
    counts = {}
    for cell, clone in clone_of.items():
        counts.setdefault(clone, Counter())[profile_of[cell]] += 1
    out = {}
    for clone, c in counts.items():
        top = max(c.values())
        out[clone] = min(p for p, n in c.items() if n == top)
    return out


def permute_rows(state, rng):
    """The rows of ``state`` shuffled across cells: each cell takes another cell's programme vector,
    so programme state no longer follows lineage or position (OncoSpot samples it per cell)."""
    state = np.asarray(state)
    return state[rng.permutation(state.shape[0])]


# Allelic states (paternal, maternal copies) over which a power law is matched to iscc's dosage.
_STATES = ((1, 0), (2, 1), (2, 2), (3, 1), (3, 0), (2, 0))


def matched_beta(s):
    """Per-gene exponent whose power law (CN/2)^beta best matches iscc's allele-specific dosage.

    iscc expresses each homolog at ``(base/2) * (1 + s_g * (c - 1))`` when it has c > 0 copies, so a
    gene's total relative to the diploid is ``sum_a [c_a > 0] (1 + s_g (c_a - 1)) / 2``. The exponent
    is the least-squares slope of its log on log(CN/2), through the origin, over a few common allelic
    states. s_g = 1 (full linear dosage) gives exactly 1."""
    s = np.asarray(s, dtype=float)
    x = np.array([np.log((p + m) / 2.0) for p, m in _STATES])
    y = np.array([[np.log(sum((1 + si * (c - 1)) for c in (p, m) if c > 0) / 2.0) for p, m in _STATES]
                  for si in np.atleast_1d(s)])
    beta = y @ x / (x @ x)
    return beta.reshape(s.shape)
