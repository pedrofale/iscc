"""Noble et al. (2022) evolutionary-mode indices for the genotype-count engine.

Noble characterise a tumour's *mode of evolution* by a handful of indices read off its clone
phylogeny: the mean number of driver mutations per cell (``n``), the clonal diversity (``D``,
inverse-Simpson over driver-mutation combinations), and the tree-balance index ``J1``. This
module implements ``n`` and ``D``; the more involved ``J1`` tree-balance metric lands in a
later milestone (M0b).

A *clone* here is a **driver-mutation combination**: the set of mutated driver positions
(oncogene / TSG indices) a genotype carries. Genotypes that differ only in passenger SNVs or
copy number but share the same set of mutated drivers are the same clone, matching Noble's
definition. Driver positions are read from the same oncogene/TSG indices and per-genotype
``get_snvs`` already used by ``GenotypeTumor.make_cell_data``.
"""
import numpy as np


def inverse_simpson(counts):
    """Inverse-Simpson diversity ``D = 1 / Σ pᵢ²`` over a set of group sizes.

    ``D`` is an effective number of equally abundant groups: ``D = k`` for ``k`` equal-sized
    groups, ``D = 1`` for a single dominant group. Zero/empty input returns ``0.0``.
    """
    c = np.asarray([v for v in counts], dtype=float)
    c = c[c > 0]
    total = c.sum()
    if total == 0:
        return 0.0
    p = c / total
    return float(1.0 / np.square(p).sum())


def driver_combination_counts(tumor):
    """Map each cancer driver-mutation combination -> total cell count.

    A combination is the ``frozenset`` of mutated driver positions (oncogene + TSG flat genome
    indices with VAF > 0) carried by a genotype. Cancer genotypes are pooled by combination and
    weighted by their cell counts; normal cells are excluded.
    """
    onc = tumor.selection.get_oncogenes()
    tsg = tumor.selection.get_tsgs()
    driver_idx = np.concatenate([onc, tsg]).astype(int)
    combos = {}
    for gid, cnt in tumor.genotypes_counts.items():
        if not tumor._is_cancer(gid):
            continue
        snv = tumor.genotypes[gid].get_snvs()
        mutated = frozenset(driver_idx[snv[driver_idx] > 0].tolist())
        combos[mutated] = combos.get(mutated, 0) + cnt
    return combos


def clonal_diversity(tumor):
    """Noble's clonal diversity ``D`` = inverse-Simpson over driver-mutation combinations."""
    return inverse_simpson(driver_combination_counts(tumor).values())


def mean_drivers_per_cell(tumor):
    """Noble's ``n`` = count-weighted mean number of mutated driver positions per cancer cell.

    Equivalently ``Σ i·pᵢ`` where ``pᵢ`` is the frequency of the driver-mutation combination
    carrying ``i`` drivers (the tree-depth analogue). ``nan`` when there are no cancer cells.
    """
    combos = driver_combination_counts(tumor)
    total = sum(combos.values())
    if total == 0:
        return float("nan")
    return float(sum(len(combo) * cnt for combo, cnt in combos.items()) / total)


def mode_indices(tumor):
    """Noble ``(n, D)`` evolutionary-mode indices, plus the clone count, in one pass.

    Returns ``dict(n=..., D=..., n_clones=...)`` over the tumour's *cancer* population. With no
    cancer cells, ``n`` and ``D`` are ``nan`` and ``n_clones`` is ``0``. ``J1`` is added in M0b.
    """
    combos = driver_combination_counts(tumor)
    total = sum(combos.values())
    if total == 0:
        return dict(n=float("nan"), D=float("nan"), n_clones=0)
    n = sum(len(combo) * cnt for combo, cnt in combos.items()) / total
    return dict(n=float(n), D=inverse_simpson(combos.values()), n_clones=len(combos))
