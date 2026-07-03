"""Cell / sample hashing (HTO) — the RNA-modality demultiplexer (DESIGN_cohort.md §2.2/§3).

Genetic demultiplexing (souporcell/vireo) recovers patient-of-origin from **germline SNPs**, which
works for **DNA** assays (WGS/WES/scDNA cover the genome/exome broadly) but NOT reliably for droplet
**scRNA** — 10x reads only sparsely cover expressed loci, so germline genotyping per cell is mostly
dropout. In practice pooled scRNA is therefore demultiplexed by **cell hashing**: before pooling, each
sample's cells are labelled with a sample-specific oligo (an antibody-conjugated HTO in Cell Hashing,
a lipid-anchored barcode in MULTI-seq) that is captured alongside the mRNA and read out as an extra
per-cell feature. The dominant hashtag per cell then names its patient.

This module models that readout and the standard HTODemux-style calling. Each patient is one hashtag;
each cell gets a strong count of its OWN hashtag (per-cell staining efficiency varies), a low ambient
background of ALL hashtags (the pooled soup, ~pool composition), and — for a doublet fraction — a
second patient's hashtag (two cells in one droplet). Singlet assignment is then near-perfect; the real
challenge (and the reason hashing is used with cell super-loading) is detecting the DOUBLETS.
"""
import numpy as np


def emit_cell_hashtags(patient_ids, mu_signal=800.0, staining_sigma=0.4, ambient_frac=0.08,
                       doublet_rate=0.1, seed=0):
    """Cell-hashing HTO readout for a pool of cells with known patient-of-origin.

    Parameters
    ----------
    patient_ids : array-like
        true patient-of-origin per pooled cell (each distinct value gets its own hashtag).
    mu_signal : float
        mean own-hashtag UMI count per cell.
    staining_sigma : float
        per-cell staining-efficiency LogNormal sd (cells stain unevenly).
    ambient_frac : float
        fraction of ``mu_signal`` present as ambient hashtag background, spread across ALL hashtags in
        proportion to pool composition (the shared soup — the main singlet-misassignment source).
    doublet_rate : float
        fraction of barcodes that are doublets (carry a second patient's hashtag).

    Returns
    -------
    counts : ndarray (n_cells, n_hashtags)   HTO counts
    hashtags : ndarray                        the distinct patient id per hashtag column
    is_doublet : ndarray (bool)               true doublet barcodes (the ground truth to detect)
    """
    rng = np.random.default_rng(seed)
    patient_ids = np.asarray(patient_ids)
    n = len(patient_ids)
    hashtags = np.unique(patient_ids)
    K = len(hashtags)
    idx = {p: i for i, p in enumerate(hashtags)}
    own = np.array([idx[p] for p in patient_ids])

    comp = np.bincount(own, minlength=K) / max(n, 1)               # ambient soup ~ pool composition
    exp = np.zeros((n, K), dtype=float)
    exp[np.arange(n), own] += mu_signal * rng.lognormal(0.0, staining_sigma, size=n)  # own hashtag
    exp += (ambient_frac * mu_signal) * comp[None, :]             # ambient background of all hashtags

    is_doublet = rng.random(n) < doublet_rate
    for c in np.where(is_doublet)[0]:
        q = int(rng.integers(0, K))
        if q == own[c]:
            q = (q + 1) % K
        exp[c, q] += mu_signal * rng.lognormal(0.0, staining_sigma)  # a second cell's hashtag
    counts = rng.poisson(exp)
    return counts, hashtags, is_doublet


def _clr(counts):
    """Centered-log-ratio normalize HTO counts per cell (the standard hashing normalization)."""
    x = counts.astype(float) + 1.0
    g = np.exp(np.log(x).mean(axis=1, keepdims=True))
    return np.log(x / g)


def demux_hashtags(counts):
    """HTODemux-style demux: assign each cell to its dominant (CLR-normalized) hashtag and score its
    doublet likelihood by the GAP between its top two hashtags (a small gap = two co-dominant hashtags
    = a doublet). Returns ``(assignment, doublet_score)`` where ``assignment`` indexes the hashtag
    column (map back through ``hashtags``) and higher ``doublet_score`` = more doublet-like."""
    Z = _clr(counts)
    order = np.argsort(-Z, axis=1)
    top = order[:, 0]
    rows = np.arange(Z.shape[0])
    gap = Z[rows, order[:, 0]] - Z[rows, order[:, 1]]
    doublet_score = -gap
    return top, doublet_score
