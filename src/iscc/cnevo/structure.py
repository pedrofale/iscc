"""Spatial structure and multi-focality (Q7) — structured scenarios only.

In the glandular substrate a lesion founds inside one epithelial-ring gland and spreads two ways:
locally, by breaching the basement membrane into the stroma (the DCIS→IDC step), and at a distance,
by cross-gland "island" dispersal that seeds other glands through the out-of-plane ductal tree.
Each seeding is a founding bottleneck, so the resulting foci are clonally related but genetically
diverged — a multi-region phylogeography with a known answer.

Nothing here is defined without glands, so every entry point returns ``None`` when
``structure_radius == 0``. That is deliberate rather than an omission: the workflow writes a null
payload for unstructured runs so both scenarios keep one summary-table schema.

Where ``validation/validate_ductal_field.py`` measures this on SNV presence, this module measures
it on **copy number**, so it lines up with the rest of :mod:`iscc.cnevo`.
"""
import numpy as np

from ..inference.indices import inverse_simpson
from .profile import clone_segment_cn

__all__ = ["is_structured", "spatial_structure"]


def is_structured(tumor):
    """Whether this tumour has a glandular substrate at all."""
    return bool(getattr(tumor, "structure_radius", 0) > 0
                and getattr(tumor, "gland_id", None) is not None)


def _cancer_by_gland(tumor):
    """``{gland_id: {genotype: n_cells}}`` over cancer cells; gland ``-1`` is stroma."""
    out = {}
    for di, _ in enumerate(tumor.demes):
        deme = tumor.demes[di]
        if not deme:
            continue
        g = int(tumor.gland_id[di])
        for gid, cnt in deme.items():
            if cnt > 0 and tumor._is_cancer(gid):
                out.setdefault(g, {})[gid] = out.setdefault(g, {}).get(gid, 0) + cnt
    return out


def _consensus_cn(tumor, genotype_counts):
    """Count-weighted mean segment CN over a set of ``{genotype: n_cells}``."""
    gids = [g for g in genotype_counts if g in tumor.genotypes]
    if not gids:
        return None
    _, total, _ = clone_segment_cn(tumor, gids)
    w = np.array([genotype_counts[g] for g in gids], dtype=float)
    if w.sum() <= 0 or not np.isfinite(total).all():
        return None
    return (total * w[:, None]).sum(axis=0) / w.sum()


def spatial_structure(tumor):
    """Multi-focal structure of a glandular run; ``None`` for an unstructured one.

    Returns a dict with the colonisation history (``n_glands``, ``n_glands_colonised``,
    ``colonisation_curve``, ``t_escape``), the invasive fraction (``frac_cancer_in_stroma``), the
    island-bottleneck signature (``within_focus_cn_divergence`` vs ``between_focus_cn_divergence``),
    clonal relatedness (``frac_clones_tracing_to_founder``) and per-focus admixture
    (``focus_clone_admixture``, mean effective clones per focus).

    ``colonisation_curve`` and ``t_escape`` need the run to have been grown with
    ``trace_occupancy=True``; they are ``None`` otherwise, while everything else still resolves
    from the final state.
    """
    if not is_structured(tumor):
        return None

    by_gland = _cancer_by_gland(tumor)
    glands = sorted(g for g in by_gland if g >= 0)
    stroma = by_gland.get(-1, {})
    n_cancer = float(sum(sum(d.values()) for d in by_gland.values()))

    out = dict(
        n_glands=int(tumor.n_glands),
        n_glands_placed=int(np.unique(tumor.gland_id[tumor.gland_id >= 0]).size),
        n_glands_colonised=len(glands),
        n_cancer_cells=n_cancer,
        frac_cancer_in_stroma=(float(sum(stroma.values()) / n_cancer) if n_cancer > 0
                               else float("nan")),
    )

    # --- island bottleneck: within- vs between-focus CN divergence ---
    cons = {g: _consensus_cn(tumor, by_gland[g]) for g in glands}
    cons = {g: c for g, c in cons.items() if c is not None}
    if len(cons) >= 2:
        keys = sorted(cons)
        d = [float(np.abs(cons[a] - cons[b]).mean())
             for i, a in enumerate(keys) for b in keys[i + 1:]]
        out["between_focus_cn_divergence"] = float(np.mean(d))
    else:
        out["between_focus_cn_divergence"] = float("nan")

    within = []
    for g in glands:
        c = cons.get(g)
        if c is None:
            continue
        gids = [x for x in by_gland[g] if x in tumor.genotypes]
        _, total, _ = clone_segment_cn(tumor, gids)
        w = np.array([by_gland[g][x] for x in gids], dtype=float)
        if w.sum() > 0 and np.isfinite(total).all():
            within.append(float((np.abs(total - c).mean(axis=1) * w).sum() / w.sum()))
    out["within_focus_cn_divergence"] = float(np.mean(within)) if within else float("nan")

    # --- clonal relatedness of the whole field ---
    parents, founder = tumor.genotypes_parents, tumor.founder_id
    cancer_gids = [g for g in tumor.genotypes_counts if tumor._is_cancer(g)]

    def traces_to_founder(g):
        cur, seen = str(g), set()
        while cur != founder and cur in parents and cur not in seen:
            seen.add(cur)
            cur = parents[cur]
        return cur == founder

    out["frac_clones_tracing_to_founder"] = (
        float(np.mean([traces_to_founder(g) for g in cancer_gids])) if cancer_gids
        else float("nan"))

    # --- per-focus admixture ---
    adm = [float(inverse_simpson(by_gland[g].values())) for g in glands if by_gland[g]]
    out["focus_clone_admixture"] = float(np.mean(adm)) if adm else float("nan")

    # --- colonisation history (needs occupancy tracing) ---
    traces = tumor.traces or []
    times = getattr(tumor, "trace_times", None)
    gens = (list(map(float, times)) if times is not None and len(times) == len(traces)
            else list(range(len(traces))))
    if traces and "n_glands_colonised" in (traces[-1] or {}):
        curve = [(float(gens[i]), int(s.get("n_glands_colonised", 0) or 0))
                 for i, s in enumerate(traces)]
        out["colonisation_curve"] = curve
        stroma_n = [float(s.get("n_occupied_demes_stroma", 0) or 0) for s in traces]
        hit = [i for i, v in enumerate(stroma_n) if v > 0]
        out["t_escape"] = float(gens[hit[0]]) if hit else None
    else:
        out["colonisation_curve"] = None
        out["t_escape"] = None
    return out
