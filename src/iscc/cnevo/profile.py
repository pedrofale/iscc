"""Copy-number ground truth: extraction, genomic coordinates, clone selection, export.

The engine stores a genotype's genome as a list over segments of ``{'p': [bitset, ...],
'm': [bitset, ...]}`` — one boolean array per *physical allele copy*, per homolog — so a
segment's copy number is ``len(genome[seg]['p']) + len(genome[seg]['m'])`` and the per-homolog
split is available for free. ``genome_summary["seg_cns"]`` caches the total. Copy number is
therefore **segment-granular**: every CNA the engine applies changes exactly one segment by one
copy on one homolog (:mod:`iscc.tumor.components.cell`), so a segment is the natural "bin".

This module is the substrate the rest of :mod:`iscc.cnevo` reads CN from. It works for *extant
and ancestral* genotypes alike, because ``tumor.genotypes`` retains every genotype ever created —
so the ancestral copy-number profiles are the true ones, not a parsimony reconstruction.
"""
import numpy as np
import pandas as pd

__all__ = [
    "segment_cn", "segment_allele_cn", "clone_segment_cn", "segment_coordinates",
    "breakpoint_sets", "select_clones", "to_medicc2_input",
]


# --------------------------------------------------------------------------------------
# Per-cell views (promoted from validation/integration_common.py)
# --------------------------------------------------------------------------------------
def segment_cn(tumor):
    """Per-cell per-segment total copy number ``(n_cells, n_segments)``, plus the gene→segment map.

    CN is constant within a segment, so the first gene of each segment is taken. Requires
    ``tumor.cell_data`` to have been materialised (``make_cell_data``).

    Returns
    -------
    (seg, gene_seg) : (ndarray, ndarray)
        ``seg`` is ``(n_cells, n_segments)`` float; ``gene_seg`` maps each gene to its segment.
    """
    n_seg = tumor.n_segments
    sizes = tumor.selection.segment_sizes
    offs = np.concatenate([[0], np.cumsum(sizes)]).astype(int)
    cnv = tumor.cell_data["cell_cnv"].values
    seg = np.stack([cnv[:, offs[s]] for s in range(n_seg)], axis=1).astype(float)
    gene_seg = np.concatenate([np.full(sizes[s], s) for s in range(n_seg)]).astype(int)
    return seg, gene_seg


def segment_allele_cn(tumor):
    """Per-cell per-segment allele-specific copy number ``cell_id -> (n_seg, 2)`` array ``(p, m)``.

    ``cell_cnv`` carries TOTAL CN only, so it cannot express allelic imbalance: a 4+0 segment
    reads as total 4, identical to a balanced 2+2. These per-homolog counts can. A cell whose
    genotype has no materialised genome maps to ``None``.
    """
    gid = tumor.cell_data["cell_type"].iloc[:, 0].astype(str).values
    idx = np.asarray(tumor.cell_data["cell_type"].index)
    g = tumor.genotypes
    cache, out = {}, {}
    for cell, gg in zip(idx, gid):
        if gg not in cache:
            rep = g.get(gg)
            cache[gg] = (np.array([(len(s["p"]), len(s["m"])) for s in rep.genome], dtype=int)
                         if rep is not None and hasattr(rep, "genome") else None)
        out[cell] = cache[gg]
    return out


# --------------------------------------------------------------------------------------
# Per-clone views (extant OR ancestral)
# --------------------------------------------------------------------------------------
def clone_segment_cn(tumor, clones=None):
    """Per-clone total and allele-specific segment CN, straight off the genotype registry.

    Unlike :func:`segment_cn` this needs no ``cell_data`` and works for **ancestral** genotypes
    (extinct internal nodes of the genealogy), which is what makes the ancestral CN profiles exact.

    Parameters
    ----------
    tumor : GenotypeTumor
    clones : sequence of str, optional
        Genotype ids. Defaults to the extant cancer genotypes, ordered by descending cell count.

    Returns
    -------
    (clone_ids, total_cn, allele_cn) : (list[str], ndarray (C, S), ndarray (C, S, 2))
    """
    if clones is None:
        clones = [g for g in tumor.genotypes_counts if tumor._is_cancer(g)]
        clones = sorted(clones, key=lambda g: -tumor.genotypes_counts[g])
    clones = [str(c) for c in clones]
    n_seg = tumor.n_segments
    total = np.zeros((len(clones), n_seg), dtype=float)
    allele = np.zeros((len(clones), n_seg, 2), dtype=float)
    for i, gid in enumerate(clones):
        rep = tumor.genotypes.get(gid)
        if rep is None:
            total[i] = np.nan
            allele[i] = np.nan
            continue
        if hasattr(rep, "genome") and rep.genome:
            pm = np.array([(len(s["p"]), len(s["m"])) for s in rep.genome], dtype=float)
            allele[i] = pm
            total[i] = pm.sum(axis=1)
        else:                                    # no materialised genome: fall back to the summary
            total[i] = np.asarray(rep.genome_summary["seg_cns"], dtype=float)
            allele[i] = np.nan
    return clones, total, allele


# --------------------------------------------------------------------------------------
# Genomic coordinates
# --------------------------------------------------------------------------------------
def segment_coordinates(tumor, bin_bp=1_000_000, n_chroms=22):
    """Map each segment to a ``(chrom, start, end)`` interval.

    Abstract genome (``genome_mode="abstract"``): segments are distributed as evenly as possible
    over at most ``n_chroms`` synthetic chromosomes and laid end to end within each, every segment
    ``segment_size * bin_bp`` long. Several segments MUST share a chromosome: a copy-number
    breakpoint is a transition between adjacent segments of the same chromosome, so one segment
    per chromosome would make :func:`breakpoint_sets` empty by construction and every
    breakpoint-based metric degenerate. The chromosome count is therefore capped at
    ``n_segments // 2`` so every chromosome carries at least two segments.

    Real genome (``genome_mode="real"``): segments ARE chromosome arms, so the real chromosome and
    the arm's true base-pair length are used, with p and q laid end to end on their chromosome.

    Returns
    -------
    DataFrame with columns ``segment, chrom, start, end`` (1-based inclusive ``start``, inclusive
    ``end``, matching the SISTEM ``observed_CNPs.tsv`` convention).
    """
    n_seg = int(tumor.n_segments)
    sizes = list(tumor.selection.segment_sizes)
    spec = getattr(tumor, "genome_spec", None)

    if getattr(tumor, "genome_mode", "abstract") == "real" and spec is not None:
        rows, cursor = [], {}
        for s, (name, length) in enumerate(zip(spec.arm_names, spec.arm_lengths)):
            nm = str(name)
            chrom = "chr" + (nm[:-1] if nm[-1] in "pq" else nm)
            start = cursor.get(chrom, 0) + 1
            end = start + int(length) - 1
            cursor[chrom] = end
            rows.append((s, chrom, start, end))
        return pd.DataFrame(rows, columns=["segment", "chrom", "start", "end"])

    # At least two segments per chromosome, or there are no within-chromosome CN transitions
    # and `breakpoint_sets` is empty by construction (see this function's docstring).
    n_chroms = max(1, min(int(n_chroms), n_seg // 2))
    # even split: the first `n_seg % n_chroms` chromosomes carry one extra segment
    base, extra = divmod(n_seg, n_chroms)
    chrom_of, s = [], 0
    for c in range(n_chroms):
        k = base + (1 if c < extra else 0)
        chrom_of.extend([c + 1] * k)
        s += k
    rows, cursor = [], {}
    for seg in range(n_seg):
        chrom = f"chr{chrom_of[seg]}"
        length = int(sizes[seg]) * int(bin_bp)
        start = cursor.get(chrom, 0) + 1
        end = start + length - 1
        cursor[chrom] = end
        rows.append((seg, chrom, start, end))
    return pd.DataFrame(rows, columns=["segment", "chrom", "start", "end"])


def breakpoint_sets(total_cn, coords):
    """Per-clone set of copy-number transition points, as ``{(chrom, segment_index)}``.

    A breakpoint sits between segment ``i`` and ``i+1`` where the CN changes. Transitions are only
    counted **within** a chromosome — a CN change across a chromosome boundary is not a breakpoint.
    """
    chrom = coords.sort_values("segment")["chrom"].to_numpy()
    total_cn = np.asarray(total_cn, dtype=float)
    out = []
    for row in total_cn:
        d = np.where((row[:-1] != row[1:]) & (chrom[:-1] == chrom[1:]))[0]
        out.append({(chrom[i], int(i)) for i in d})
    return out


# --------------------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------------------
def select_clones(tumor, n_clones, min_cells=1, strategy="largest", seed=0):
    """Choose ``n_clones`` extant cancer clones — one representative cell each.

    The analogue of SISTEM's ``uniform-clone`` sampling: one cell per distinct clone maximises CN
    diversity, and since iscc's genealogy is over clones rather than cells this is also the leaf
    set of the true tree.

    Parameters
    ----------
    strategy : {"largest", "random"}
        ``"largest"`` takes the most abundant clones (deterministic, and the ones a real assay
        would actually catch); ``"random"`` samples uniformly among eligible clones.

    Returns
    -------
    dict with ``clones`` (list[str]), ``n_requested``, ``n_clones``, ``under_sampled``.
    """
    eligible = [g for g, c in tumor.genotypes_counts.items()
                if tumor._is_cancer(g) and c >= int(min_cells)]
    n_clones = int(n_clones)
    if strategy == "random":
        rng = np.random.default_rng(seed)
        order = list(rng.permutation(np.asarray(sorted(eligible))))
    elif strategy == "largest":
        order = sorted(eligible, key=lambda g: (-tumor.genotypes_counts[g], g))
    else:
        raise ValueError(f"unknown strategy {strategy!r}")
    chosen = [str(g) for g in order[:n_clones]]
    return dict(clones=chosen, n_requested=n_clones, n_clones=len(chosen),
                under_sampled=len(chosen) < n_clones)


# --------------------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------------------
def to_medicc2_input(tumor, clones, path=None, coords=None):
    """Write ground-truth total CN as a MEDICC2 ``--input-type tsv`` table.

    Columns ``sample_id, chrom, start, end, cn`` — consumed with ``-a cn --total-copy-numbers``.
    """
    coords = segment_coordinates(tumor) if coords is None else coords
    clone_ids, total, _ = clone_segment_cn(tumor, clones)
    coords = coords.sort_values("segment")
    rows = []
    for i, gid in enumerate(clone_ids):
        for _, c in coords.iterrows():
            rows.append((gid, c["chrom"], int(c["start"]), int(c["end"]),
                         int(round(float(total[i, int(c["segment"])])))))
    df = pd.DataFrame(rows, columns=["sample_id", "chrom", "start", "end", "cn"])
    if path is not None:
        df.to_csv(path, sep="\t", index=False)
    return df
