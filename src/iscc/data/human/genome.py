"""An arm-homologous growth genome: one growth segment per human chromosome arm.

The growth model keeps a coarse genome (``genome_params`` / ``selection_params`` of
:class:`~iscc.tumor.models.GenotypeTumor`); the human genome layer lifts it to real genes when cells
are measured. For that lift to be realistic, each growth segment has to be one contiguous human
region inside one chromosome, so a copy-number event acts on one arm. :func:`arm_genome` builds such a
genome at any resolution: how many growth genes each arm gets is the user's trade-off between the
detail of the dynamics and their cost, and the layer works whatever it is.
"""
import numpy as np

from .reference import load_reference

WEIGHTS = ("genes", "length", "uniform")


def arm_genome(total_genes=6000, genes_per_segment=None, weight="genes", min_genes=4,
               driver_scale=1.0, reference=None):
    """Genome and driver parameters for a growth genome with one segment per chromosome arm.

    Parameters
    ----------
    total_genes : int, default 6000
        Growth genes over the 39 arms, shared out by ``weight``. Ignored when ``genes_per_segment``
        is given.
    genes_per_segment : int, optional
        The same number of growth genes on every arm, instead of ``total_genes``.
    weight : {"genes", "length", "uniform"}, default "genes"
        How ``total_genes`` is shared between arms: in proportion to each arm's number of
        ``GRCh38-2020-A`` genes, to its length in base pairs, or equally.
    min_genes : int, default 4
        Floor on an arm's growth genes, so a small arm still holds its drivers.
    driver_scale : float, default 1.0
        Drivers per arm = ``driver_scale`` times the arm's Cancer Gene Census oncogenes and
        suppressors (297 and 293 over the 39 arms), rounded. The count does not change with the
        resolution, so neither does an arm's driver load; it is capped at the arm's growth genes.
    reference : HumanReference, optional
        Default the packaged one.

    Returns
    -------
    dict
        ``{"genome_params": ..., "selection_params": ...}``. ``genome_params`` holds ``n_segments``,
        ``segment_sizes``, ``segment_size`` (their rounded mean) and ``arms`` (the arm names, which
        the layer reads); ``selection_params`` holds ``segment_drivers``, one ``(oncogenes,
        suppressors)`` pair per arm. Merge the latter into the configuration's own
        ``selection_params``; ``prop_driver`` is then unused.
    """
    ref = reference or load_reference()
    arms = ref.arms
    n = len(arms)
    if genes_per_segment is not None:
        sizes = np.full(n, int(genes_per_segment))
    else:
        if weight not in WEIGHTS:
            raise ValueError(f"weight must be one of {WEIGHTS}, got {weight!r}")
        if weight == "genes":
            arm_of = ref.gene_arm()
            w = np.bincount(arm_of[arm_of >= 0], minlength=n).astype(float)
        elif weight == "length":
            w = (arms["end"] - arms["start"]).to_numpy(float)
        else:
            w = np.ones(n)
        sizes = np.maximum(np.round(w / w.sum() * int(total_genes)).astype(int), int(min_genes))
    n_onc = np.round(driver_scale * arms["n_onc"].to_numpy(float)).astype(int)
    n_tsg = np.round(driver_scale * arms["n_tsg"].to_numpy(float)).astype(int)
    over = n_onc + n_tsg - sizes
    for a in np.flatnonzero(over > 0):               # cap at the arm's genes, keeping the split
        tot = n_onc[a] + n_tsg[a]
        n_onc[a] = int(np.floor(n_onc[a] * sizes[a] / tot))
        n_tsg[a] = int(sizes[a] - n_onc[a])
    sizes = [int(s) for s in sizes]
    return {
        "genome_params": {"n_segments": n, "segment_sizes": sizes,
                          "segment_size": int(round(float(np.mean(sizes)))),
                          "arms": arms["arm"].tolist()},
        "selection_params": {"segment_drivers": [[int(o), int(t)] for o, t in zip(n_onc, n_tsg)]},
    }
