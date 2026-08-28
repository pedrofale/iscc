"""One shared definition of a *clone*: a clade of the true lineage tree with enough sampled cells.

Every notebook that annotates cells needs to say "this cell belongs to clone X", and the engine does
not hand that over directly. The raw ``cell_type``/``cell_id`` label is the *genotype* id, and a
genotype is far too fine-grained to be a clone: a few hundred sampled cells routinely carry a
hundred-odd distinct genotypes, so colouring by genotype gives a confetti plot with no readable
structure. Ad-hoc alternatives (merging by driver signature, clustering copy-number profiles, cutting
a distance matrix at K) each answer a different question and disagree with each other.

This module fixes the definition on the ground truth the simulator actually has — the lineage:

    a **clone** is a clade of the true clone tree (``tumor.genotypes_parents``) that contains at
    least ``min_cells`` of the SAMPLED cells.

The clades are chosen so that they are disjoint and as large as possible: the tree is cut at the
points where it genuinely branches into two well-sampled lineages, and each cut piece is one clone.
Concretely, a node "qualifies" when its subtree holds at least ``min_cells`` sampled cells; the clones
are the *deepest* qualifying nodes (so a clone is never emitted together with a sub-clone of itself),
each grown back up the tree as far as it can go without swallowing a sibling clone. Sampled cells that
sit above every clone — the ancestral backbone and small side branches — share one ``"other"`` label.

Because the labels are derived from the tree and ordered by size (the biggest clade is ``clone_1``),
they are deterministic: the same tumour and the same ``min_cells`` always give the same labels, so a
clone keeps its colour across notebooks, figures and reruns. Only cancer cells get a clone; normal
cells (epithelial, stromal, immune, host) are left as ``NaN``.

Typical use::

    clones = clones_from_clades(tumor, min_cells=50)      # pandas Series over sampled cells
    adata.obs["clone"] = clones                            # stable, readable colour annotation
    clone_summary(tumor, clones)                           # what each clone is, one row per clone
"""
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

#: Label given to sampled cancer cells that fall outside every clone (the ancestral backbone and the
#: small side branches that never reached ``min_cells``).
UNASSIGNED = "other"

# Mutated-driver tallies reported per clone: (summary column, Selection accessor, plain-English name
# used in the one-line `traits` string). Same quantities `make_cell_data` writes into `cell_evo`.
_TRAITS = (
    ("n_mut_onc", "get_oncogenes", "oncogene"),
    ("n_mut_tsg", "get_tsgs", "TSG"),
    ("n_mut_disp", "get_dispersal_genes", "dispersal"),
    ("n_mut_ir", "get_immune_resistant", "immune-resistance"),
    ("n_mut_tr", "get_treatment_resistant", "treatment-resistance"),
    ("n_mut_breach", "get_breach", "breach"),
    ("n_mut_ss", "get_stromal_survival", "stromal-survival"),
    ("n_mut_ms", "get_met_survival", "met-survival"),
)


# ---------------------------------------------------------------------------------- cell bookkeeping
def _cell_genotypes(tumor, cell_ids=None):
    """``Series`` cell id -> genotype id over the sampled cells (optionally a given subset)."""
    cell_data = getattr(tumor, "cell_data", None)
    if not cell_data or "cell_type" not in cell_data:
        raise ValueError(
            "this tumour has no per-cell data, and clone labels are defined over sampled cells. "
            "Materialise a sample first — tumor.make_cell_data(max_cells=...) for a whole-tumour "
            "subsample, or cut a piece with iscc.sample.Resection (dissociate / slice) — then call "
            "clones_from_clades again."
        )
    genotype = cell_data["cell_type"]["cell_id"].astype(str)
    if cell_ids is not None:
        wanted = pd.Index([str(c) for c in cell_ids])
        missing = wanted.difference(genotype.index)
        if len(missing):
            raise KeyError(
                f"{len(missing)} of the requested cell ids are not in this tumour's sampled cells "
                f"(first few: {list(missing[:5])}). They may come from a different sample — "
                "cell ids are re-generated every time cell data is materialised."
            )
        genotype = genotype.loc[wanted]
    return genotype


def _is_cancer(tumor, gid):
    """Whether genotype ``gid`` is a cancer genotype (normal cell types never get a clone)."""
    genotypes = getattr(tumor, "genotypes", None)
    if genotypes is not None and gid in genotypes:
        if hasattr(tumor, "_is_cancer"):
            return bool(tumor._is_cancer(gid))
        return getattr(genotypes[gid], "type", None) == "cancer"
    # No genotype registry (e.g. a table loaded from disk): normal cells are named cell types, every
    # other id is a cancer clone.
    from iscc.constants import normal_names
    return gid not in set(normal_names)


# ------------------------------------------------------------------------------------ tree walking
def _depths(parents, seeds):
    """Depth (edges below its root) of every node in ``seeds`` and of all their ancestors."""
    depth = {}
    for g in seeds:
        chain, cur, seen = [], g, set()
        while cur is not None and cur not in depth:
            chain.append(cur)
            seen.add(cur)
            nxt = parents.get(cur)
            # `nxt in seen` cannot happen in a genealogy; it is a guard so a corrupted parent map
            # degrades to "treat this node as a root" instead of looping forever.
            cur = None if (nxt is None or nxt in seen) else nxt
        base = -1 if cur is None else depth[cur]     # depth of the parent of the last chain entry
        for k, node in enumerate(reversed(chain)):
            depth[node] = base + 1 + k
    return depth


def _subtree_counts(parents, depth, counts):
    """Sampled cells in each node's subtree, accumulated deepest-first up the parent map."""
    subtree = Counter()
    for gid, n in counts.items():
        subtree[gid] += n
    for gid in sorted(depth, key=lambda g: (-depth[g], g)):
        parent = parents.get(gid)
        if parent is not None and depth.get(parent, depth[gid]) < depth[gid]:
            subtree[parent] += subtree[gid]
    return subtree


def _children(parents, depth):
    children = defaultdict(list)
    for gid in depth:
        parent = parents.get(gid)
        if parent is not None and depth.get(parent, depth[gid]) < depth[gid]:
            children[parent].append(gid)
    return children


def _clade_roots(parents, depth, counts, min_cells):
    """The clone clades: disjoint, each holding >= ``min_cells`` sampled cells where possible.

    A node qualifies when its subtree holds at least ``min_cells`` sampled cells; roots always
    qualify, which is what makes an over-large ``min_cells`` collapse to a single whole-tumour clone
    instead of no clones at all. Qualifying nodes are closed upwards (a parent's subtree is at least
    as big as a child's), so the deepest qualifying nodes are exactly the leaves of that set: they
    are pairwise non-nested, hence disjoint clades. Each is then grown back up the tree while its
    ancestor still contains only that one clade, which keeps the clades as LARGE as possible (fewer
    cells stranded in ``other``) without ever merging two of them.
    """
    subtree = _subtree_counts(parents, depth, counts)
    children = _children(parents, depth)
    qualifies = {g for g in depth if subtree[g] >= min_cells or depth[g] == 0}
    leaves = [g for g in qualifies if not any(c in qualifies for c in children[g])]

    # How many clades sit inside each node's subtree — the stop condition for growing a clade upward.
    n_clades = Counter({g: 1 for g in leaves})
    for gid in sorted(depth, key=lambda g: (-depth[g], g)):
        parent = parents.get(gid)
        if parent is not None and depth.get(parent, depth[gid]) < depth[gid]:
            n_clades[parent] += n_clades[gid]

    roots = []
    for leaf in leaves:
        node = leaf
        while True:
            parent = parents.get(node)
            if parent is None or parent not in depth or n_clades[parent] != 1:
                break
            node = parent
        roots.append(node)
    return roots, subtree


def _clade_owner(parents, depth, roots):
    """Map every node to the clade it belongs to (its nearest clade-root ancestor, or ``None``)."""
    root_set = set(roots)
    owner = {}
    for gid in sorted(depth, key=lambda g: (depth[g], g)):     # parents before children
        if gid in root_set:
            owner[gid] = gid
        else:
            owner[gid] = owner.get(parents.get(gid))
    return owner


# --------------------------------------------------------------------------------------- public API
def clones_from_clades(tumor, min_cells=50, cell_ids=None, label_prefix="clone"):
    """Label each sampled cell with the lineage clade (the clone) it belongs to.

    Parameters
    ----------
    tumor : GenotypeTumor
        A grown tumour whose per-cell data has been materialised (``make_cell_data`` or any
        ``iscc.sample`` cut). Both compartments of a tumour with a metastasis are covered: the
        deposit shares the primary's clone tree, so a clone label spans the two sites.
    min_cells : int, optional
        Smallest number of SAMPLED cells a clade must hold to be called a clone (default 50). Bigger
        values give fewer, coarser clones. Larger than the whole sample, everything collapses into a
        single clone.
    cell_ids : iterable, optional
        Restrict to these cells (they must be part of the current sample). Clade sizes are then
        counted over this subset, so clones are defined on exactly the cells being annotated.
    label_prefix : str, optional
        Prefix of the generated labels (default ``"clone"`` -> ``clone_1``, ``clone_2``, ...).

    Returns
    -------
    pandas.Series
        Indexed by cell id, dtype ``category``. Cancer cells get ``clone_1`` (the largest clade),
        ``clone_2``, ... or ``"other"`` when they sit above every clade; normal cells get ``NaN``.
        ``Series.attrs["clade_root"]`` maps each label to the genotype id at the top of its clade.

    Notes
    -----
    Labels are ordered by clade size and derived from the tree alone, so they are stable across
    reruns and across notebooks: ``clone_1`` is the same lineage everywhere.
    """
    min_cells = int(min_cells)
    if min_cells < 1:
        raise ValueError(f"min_cells must be at least 1, got {min_cells}")

    genotype = _cell_genotypes(tumor, cell_ids)
    cancer = {g: _is_cancer(tumor, g) for g in pd.unique(genotype.values)}
    is_cancer = np.array([cancer[g] for g in genotype.values], dtype=bool)
    counts = Counter(genotype.values[is_cancer])

    labels = np.full(len(genotype), None, dtype=object)
    clade_root = {}
    if counts:
        parents = {str(k): str(v) for k, v in dict(getattr(tumor, "genotypes_parents", {})).items()}
        depth = _depths(parents, counts)
        roots, subtree = _clade_roots(parents, depth, counts, min_cells)
        # Largest clade first, so `clone_1` is the dominant lineage. Registration order then id break
        # ties, keeping the labelling reproducible for equal-sized clades.
        genotypes = getattr(tumor, "genotypes", {})
        roots = sorted(roots, key=lambda g: (-subtree[g], getattr(genotypes.get(g), "ord", 0), g))
        clade_root = {f"{label_prefix}_{i}": g for i, g in enumerate(roots, start=1)}
        label_of_root = {g: lab for lab, g in clade_root.items()}
        owner = _clade_owner(parents, depth, roots)
        by_genotype = {g: label_of_root.get(owner.get(g), UNASSIGNED) for g in counts}
        labels[is_cancer] = [by_genotype[g] for g in genotype.values[is_cancer]]

    categories = list(clade_root)
    if UNASSIGNED in labels:
        categories.append(UNASSIGNED)
    clones = pd.Series(pd.Categorical(labels, categories=categories), index=genotype.index,
                       name="clone")
    clones.attrs["clade_root"] = clade_root
    clones.attrs["min_cells"] = min_cells
    clones.attrs["unassigned"] = UNASSIGNED
    return clones


def _traits_string(row):
    """One-line summary of the trait mutations a clone carries, e.g. ``"2 oncogene, 1 breach"``."""
    parts = [f"{int(row[col])} {name}" for col, _, name in _TRAITS
             if row.get(col) is not None and not pd.isna(row.get(col)) and int(row[col]) > 0]
    return ", ".join(parts) if parts else "none"


def _ancestors(parents, gid):
    """``{gid, parent, ..., root}`` — the node and every ancestor of it."""
    out, cur = [], gid
    seen = set()
    while cur is not None and cur not in seen:
        out.append(cur)
        seen.add(cur)
        cur = parents.get(cur)
    return out


def _lca(parents, gids):
    """Lowest common ancestor of ``gids`` in the parent map (``None`` if they share no ancestor)."""
    node = None
    for gid in gids:
        if node is None:
            node = gid
            continue
        ancestors = set(_ancestors(parents, node))
        node = next((a for a in _ancestors(parents, gid) if a in ancestors), None)
        if node is None:
            return None
    return node


def clone_summary(tumor, clones):
    """One row per clone: how big it is, which genotype represents it, and what it carries.

    ``clones`` is the Series from :func:`clones_from_clades`. The representative ``genotype`` of a
    clone is the deepest genotype ancestral to every one of its sampled cells, so the mutation
    tallies and evolutionary parameters reported next to it are the clone's SHARED state — the
    traits every cell in the clone inherited, and nothing a sub-lineage picked up on its own. The
    ``traits`` column renders those tallies as one readable line (``"2 oncogene, 1 breach"``), so a
    notebook can print which trait mutations separate the clones without unpacking the columns.

    The ``other`` bucket (cells above every clade) appears as a final row with a cell count but no
    representative genotype, so the counts add up to the sampled cancer cells.
    """
    genotype = _cell_genotypes(tumor, clones.index)
    parents = {str(k): str(v) for k, v in dict(getattr(tumor, "genotypes_parents", {})).items()}
    genotypes = getattr(tumor, "genotypes", {})
    selection = getattr(tumor, "selection", None)
    clade_root = dict(clones.attrs.get("clade_root", {}))

    if hasattr(clones, "cat"):
        labels = list(clones.cat.categories)
    else:
        # A plain (non-categorical) Series has lost the size ordering — rebuild it, `other` last.
        counts = clones.dropna().value_counts()
        labels = sorted(counts.index, key=lambda lab: (lab == UNASSIGNED, -counts[lab], str(lab)))
    labels = [lab for lab in labels if (clones == lab).any()]

    trait_idx = {col: (getattr(selection, accessor)() if hasattr(selection, accessor) else None)
                 for col, accessor, _ in _TRAITS}
    evo_keys, rows = [], []
    for label in labels:
        members = genotype[(clones == label).values]
        member_genotypes = pd.unique(members.values)
        # The clade's representative: the common ancestor of everything sampled inside it. That is
        # the clade root itself as soon as the clade holds two divergent sub-lineages, and otherwise
        # a node below it — the deepest state the whole clone actually shares. The `other` bucket is
        # not a clade, so it has no representative.
        rep_gid = (_lca(parents, member_genotypes)
                   if label != UNASSIGNED and len(member_genotypes) else None)
        if rep_gid is None and label != UNASSIGNED:
            rep_gid = clade_root.get(label)
        row = {"n_cells": int(len(members)),
               "n_genotypes": int(len(member_genotypes)),
               "genotype": rep_gid if rep_gid is not None else pd.NA}
        rep = genotypes.get(rep_gid) if rep_gid is not None else None
        if rep is not None:
            snv = rep.get_snvs()
            for col, _, _ in _TRAITS:
                idx = trait_idx[col]
                row[col] = int((snv[idx] > 0).sum()) if idx is not None and len(idx) else 0
            for key, value in dict(getattr(rep, "evolutionary_parameters", {})).items():
                row[key] = float(value)
                if key not in evo_keys:
                    evo_keys.append(key)
        row["traits"] = _traits_string(row) if rep is not None else pd.NA
        rows.append(row)

    columns = (["n_cells", "n_genotypes", "genotype"] + [c for c, _, _ in _TRAITS] + evo_keys
               + ["traits"])
    summary = pd.DataFrame(rows, index=pd.Index(labels, name="clone"), columns=columns)
    for col, _, _ in _TRAITS:
        summary[col] = summary[col].astype("Int64")
    return summary
