"""Cohort progression-model seam: MHN / TreeMHN / CBN / REVOLVER input + scoring.

The companion to :mod:`iscc.tumor.components.epistasis`. That module PLANTS a known event x event
network; this one exports a cohort in the shape those tools consume, and scores what they recover
against the planted answer key.

Three export shapes:

  * :func:`to_mhn_matrix` — the **cross-sectional** patients x events binary matrix (MHN, DISCOVER,
    MEGSA): "did patient p ever acquire event i?". Throws away order — the price MHN pays.
  * :func:`to_mutation_tree` / :func:`to_treemhn_trees` — the per-patient **mutation tree** (TreeMHN,
    REVOLVER): the trie of the event orders realised along the surviving lineages. iscc has these
    exactly, which is why TreeMHN is the flagship of this benchmark.
  * :func:`to_cbn_poset` — the observed event sets for CBN/H-CBN's poset fit.

Event presence takes an explicit cancer-cell-fraction threshold, ``min_freq``, with no default: the
old default of 0 (any surviving cell) saturates the matrix in a mutation-rich regime, so every caller
states its detection limit. :data:`DETECTION_LIMIT` (5%, the detection limit of the paper's Fig 2f) is
the one paper 1's Fig 5 uses. The tree exports take ``detection``, the same limit applied to the nodes
of the collapsed clone tree.

**Traits as events.** :func:`to_trait_matrix`, :func:`trait_table` and :func:`to_trait_trees` export the
heritable dissemination traits (breach, stromal survival, metastatic survival) in the same shapes, read
from the genotype counts and the genealogy rather than from an epistasis network, so a progression
model can be asked about an order that tissue architecture imposes without any planted interaction.

Scoring (:func:`score_edges`, :func:`score_order`, :func:`score_exclusivity`) is deliberately generic:
it takes an edge/order list from ANY tool and the ground truth from
``tumor.epistasis_ground_truth()``, so a new tool only needs an adapter to its output format.

:func:`cooccurrence_scores` is a built-in, dependency-free baseline (the pairwise log odds ratio —
essentially the DISCOVER/MEGSA statistic). It is not a replacement for MHN: it cannot separate a
direct interaction from one induced through a shared ancestor. It is here so the benchmark always
has a floor to compare a real tool against, and so the validation figure renders without R.
"""
from collections import Counter

import numpy as np
import pandas as pd

#: The cancer-cell fraction at which an event counts as present in paper 1's Fig 5: the detection
#: limit of Fig 2f (vault: decision-fig5-planted-network). Pass it as ``min_freq`` / ``detection``.
DETECTION_LIMIT = 0.05

#: The heritable dissemination traits Fig 5 reads as events, in column order.
TRAIT_EVENTS = ("breach", "stromal_survival", "met_survival")

# A trait is acquired when a genotype carries >= 1 mutated copy of one of the trait's genes: the
# genome-summary tally of each trait axis.
_TRAIT_TALLY = {"breach": "n_mut_breach", "stromal_survival": "n_mut_ss", "met_survival": "n_mut_ms",
                "treatment_resistance": "n_mut_tr", "immune_resistance": "n_mut_ir",
                "dispersal": "n_mut_disp", "drug_tolerance": "n_mut_dt"}


# ------------------------------------------------------------------ export: cross-sectional (MHN)
def clone_events(tumor, min_clone_freq=0.0):
    """Per-clone event tuples with their cell counts.

    ``min_clone_freq`` drops clones smaller than this fraction of the tumour — a CLONE-level filter
    (useful for tree exports, where a tiny clone is a tip you may not have resolved). It is NOT a
    detection threshold for an EVENT: an event carried by fifty 1%-clones is at 50% cell fraction and
    trivially detectable, yet no clone clears a 10% clone filter. Use ``event_cell_fractions`` for
    event detection.
    """
    net = tumor.selection.epistasis
    if net is None:
        raise ValueError("tumour has no epistasis network; nothing to export "
                         "(set selection_params['epistasis_params'])")
    tbl = tumor.event_table()
    if tbl.empty:
        return tbl
    total = tbl["n_cells"].sum()
    if min_clone_freq > 0 and total > 0:
        tbl = tbl[tbl["n_cells"] / total >= min_clone_freq]
    return tbl


def event_cell_fractions(tumor, n_events=None):
    """Per-event fraction of cancer CELLS carrying it, summed ACROSS clones.

    This is the quantity a real assay measures (a variant's cancer-cell fraction / VAF), and it is
    the observable that actually carries the selection signal: iscc's ``E`` acts on how large the
    carrying clones grow, and clones carrying a favoured combination expand. Aggregating across
    clones is essential — a favoured combination typically arises MANY times independently, so its
    cells are spread over dozens of small lineages rather than concentrated in one big one.
    """
    net = tumor.selection.epistasis
    if net is None:
        raise ValueError("tumour has no epistasis network; nothing to export "
                         "(set selection_params['epistasis_params'])")
    n = n_events if n_events is not None else net.n_events
    tbl = tumor.event_table()
    frac = np.zeros(n)
    if tbl.empty:
        return frac
    total = tbl["n_cells"].sum()
    if total <= 0:
        return frac
    for events, cnt in zip(tbl["events"], tbl["n_cells"]):
        for e in events:
            frac[e] += cnt
    return frac / total


def _present(frac, min_freq):
    """1 where a cell fraction reaches the detection floor; a floor of 0 means any cell at all."""
    frac = np.asarray(frac, dtype=float)
    if min_freq <= 0:
        return (frac > 0).astype(int)
    return (frac >= min_freq).astype(int)


def patient_event_vector(tumor, min_freq, n_events=None):
    """Binary event vector for ONE patient: 1 where the event's CELL FRACTION >= ``min_freq``.

    ``min_freq`` is the detection floor of the assay, applied to the event's cancer-cell fraction
    (see :func:`event_cell_fractions`) — not to any single clone's size. It has no default: at
    ``min_freq=0`` every event ever acquired by any surviving cell counts as observed, which is both
    more than a real study sees AND, in a mutation-rich regime, **saturated**: a recurrent event is
    present in almost every patient regardless of selection, so the column carries no signal. Sweep
    this threshold rather than assuming it (see validation/validate_epistasis.py);
    :data:`DETECTION_LIMIT` is paper 1's.
    """
    net = tumor.selection.epistasis
    n = n_events if n_events is not None else net.n_events
    return _present(event_cell_fractions(tumor, n_events=n), min_freq)


def to_mhn_matrix(tumors, min_freq):
    """Patients x events binary DataFrame — MHN's (and DISCOVER/MEGSA's) input.

    ``tumors`` is any iterable of grown tumours sharing one network (i.e. one ``Cohort``).
    ``min_freq`` is the per-event cancer-cell-fraction detection floor (no default); see
    :func:`patient_event_vector`.
    """
    tumors = list(tumors)
    if not tumors:
        raise ValueError("no tumours given")
    net = tumors[0].selection.epistasis
    rows = [patient_event_vector(t, min_freq, n_events=net.n_events) for t in tumors]
    return pd.DataFrame(rows, columns=net.event_names(),
                        index=[f"P{i}" for i in range(len(tumors))])


def to_cell_fraction_matrix(tumors):
    """Patients x events CONTINUOUS cancer-cell-fraction matrix — the frequency-aware observable.

    The binary matrix above is what MHN consumes; this is what the selection signal actually lives
    in. Keeping both makes the comparison in validate_epistasis.py possible: same tumours, same
    network, two observables.
    """
    tumors = list(tumors)
    if not tumors:
        raise ValueError("no tumours given")
    net = tumors[0].selection.epistasis
    rows = [event_cell_fractions(t, n_events=net.n_events) for t in tumors]
    return pd.DataFrame(rows, columns=net.event_names(),
                        index=[f"P{i}" for i in range(len(tumors))])


# ------------------------------------------------------------------ export: mutation trees (TreeMHN)
def _trie(paths, total, detection=0.0):
    """TreeMHN's node table for a set of ``(acquisition path, n_cells)``: the trie of the paths.

    ``detection`` collapses the tree at a cell-fraction limit: a node is kept when the cells whose
    path passes through it reach ``detection`` of ``total`` (the tumour's cancer cells), and the cells
    of a dropped node are attributed to its deepest kept ancestor. A node's cells include its
    descendants', as in Noble et al.'s detection limit (see iscc.inference.indices.detect_combinations),
    so a node below the limit merges into its parent. ``detection=0`` keeps every node."""
    keep = None
    if detection > 0 and total > 0:
        through = Counter()
        for path, n in paths:
            for k in range(1, len(path) + 1):
                through[path[:k]] += n
        keep = {p for p, c in through.items() if c / total >= detection}
    nodes = [dict(Node_ID=1, Mutation_ID=0, Parent_ID=1, n_cells=0)]  # root: its own parent (TreeMHN)
    children = {}  # (parent Node_ID, mutation) -> Node_ID
    for path, n in paths:
        parent_node = 1
        for k, ev in enumerate(path):
            if keep is not None and path[:k + 1] not in keep:
                break                    # a node's cells shrink down the path: nothing deeper is kept
            key = (parent_node, int(ev))
            node = children.get(key)
            if node is None:
                node = len(nodes) + 1
                nodes.append(dict(Node_ID=node, Mutation_ID=int(ev) + 1,
                                  Parent_ID=parent_node, n_cells=0))
                children[key] = node
            parent_node = node
        # attribute the clone's cells to the deepest node of its path that is kept
        nodes[parent_node - 1]["n_cells"] += int(n)
    return pd.DataFrame(nodes)


def _tree_from_table(tbl, min_clone_freq, detection):
    """The trie of an event table's ``event_order`` paths (see :func:`_trie`)."""
    total = int(tbl["n_cells"].sum()) if not tbl.empty else 0
    if min_clone_freq > 0 and total > 0:
        tbl = tbl[tbl["n_cells"] / total >= min_clone_freq]
    paths = [(tuple(int(e) for e in order), int(n))
             for order, n in zip(tbl["event_order"], tbl["n_cells"])]
    return _trie(paths, total, detection)


def to_mutation_tree(tumor, min_clone_freq=0.0, detection=0.0):
    """The patient's **mutation tree** as a trie over the realised event orders.

    Each surviving clone contributes the path ``event_order`` from the root; shared prefixes are
    merged, so the result is the tree of "which event followed which" in THIS patient — exactly
    TreeMHN's / REVOLVER's input. The root is ``Mutation_ID = 0`` (the normal genotype) and events
    are numbered ``1..n_events`` (TreeMHN's convention: 0 is reserved for the root).

    This is the true clone tree collapsed to event acquisitions: ``event_order`` is each genotype's
    lineage read off the genealogy (every event acquisition mints a genotype), and parallel lineages
    that acquired the same events in the same order become one node, which is what TreeMHN's tree (a
    set of acquisition orders) can express. Events acquired in one division are tied and appear in
    index order, an arbitrary tie-break (see ``event_groups``).

    ``detection`` is the clone-size limit of the collapsed tree: a node is kept when the cancer cells
    descending through it (its own and its descendants') reach this fraction of the tumour, and a
    node below it merges into its parent. Paper 1's Fig 5 uses :data:`DETECTION_LIMIT` (clones of at
    least 5% of cancer cells). A genotype is far finer than a clone — with passengers unfolded, which
    epistasis forces, almost no single genotype holds 5% of a tumour — so this is the limit to use for
    a clone tree; ``min_clone_freq`` is the older GENOTYPE-level filter (drop genotypes below a
    fraction of the tumour before building the trie), kept for its callers.

    ``n_cells`` is emitted for the caller's own analysis, but note that **TreeMHN does not read it**:
    its ``input_tree_df`` accepts ONLY ``Patient_ID``/``Tree_ID``/``Node_ID``/``Mutation_ID``/
    ``Parent_ID`` and errors on any other column (its ``weights`` argument is a per-TREE weight for
    tree uncertainty, not a clone size). So what this export gives a tree-based method over the binary
    matrix is **event ORDER**, not frequency — which is exactly what it can and cannot recover.

    Follows TreeMHN's own input convention exactly (its README): the root is ``Node_ID = 1`` with
    ``Mutation_ID = 0`` and is **its own parent** (``Parent_ID = 1``) — not 0, which its
    ``sort_one_tree_df`` cannot resolve.

    Returns a DataFrame with ``Node_ID``, ``Mutation_ID``, ``Parent_ID``, ``n_cells``.
    """
    return _tree_from_table(clone_events(tumor), min_clone_freq, detection)


def stack_trees(frames, drop_empty=True, what="event"):
    """Per-patient tree frames (from :func:`to_mutation_tree` / :func:`to_trait_tree`) stacked with
    ``Patient_ID`` / ``Tree_ID``, as :func:`to_treemhn_trees` does. For trees stored one patient at a
    time (one Slurm array task per patient) and assembled later, without the tumours."""
    frames = [df for df in frames if not (drop_empty and len(df) < 2)]
    if not frames:
        raise ValueError(f"no patient acquired any {what}, so there are no mutation trees to export; "
                         "raise event_size / mutation_rate / grow_steps, or lower the detection limit")
    out = []
    for i, df in enumerate(frames):
        df = df.copy()
        df.insert(0, "Patient_ID", i + 1)
        df.insert(1, "Tree_ID", i + 1)
        out.append(df)
    return pd.concat(out, ignore_index=True)


def to_treemhn_trees(tumors, min_clone_freq=0.0, drop_empty=True, detection=0.0):
    """Every patient's mutation tree in one tidy frame (a ``Patient_ID`` column added) — the shape
    TreeMHN's R ``input_tree`` list is built from. ``min_clone_freq`` and ``detection`` are
    :func:`to_mutation_tree`'s.

    A patient that acquired NO event yields a tree with only the root and no edges, which TreeMHN
    rejects outright ("Tree with ID n does not contain edges from the root"). Such a patient carries
    no ordering information, so with ``drop_empty`` (the default) it is omitted and the remaining
    patients are RENUMBERED contiguously from 1 — TreeMHN indexes patients by position, so a gap in
    the IDs is not safe to leave behind. Set ``drop_empty=False`` to keep them and handle it yourself.

    Note this makes the tree export's patient set potentially SMALLER than the cross-sectional
    matrix's; when comparing the two observables, compare on the tumours each actually saw.
    """
    return stack_trees([to_mutation_tree(t, min_clone_freq=min_clone_freq, detection=detection)
                         for t in tumors], drop_empty, "event")


def to_cbn_poset(tumors, min_freq):
    """CBN/H-CBN input: the cross-sectional binary genotype matrix (same shape as MHN's), which
    H-CBN fits a poset to. Kept as its own name because the two tools' semantics differ even though
    the file format coincides. ``min_freq`` as in :func:`to_mhn_matrix`."""
    return to_mhn_matrix(tumors, min_freq)


# ------------------------------------------------------------------ traits as events
def _trait_names(traits, with_events, tumor):
    traits = tuple(traits)
    unknown = [t for t in traits if t not in _TRAIT_TALLY]
    if unknown:
        raise ValueError(f"unknown trait(s) {unknown}; known: {sorted(_TRAIT_TALLY)}")
    events = []
    if with_events:
        net = tumor.selection.epistasis
        if net is None:
            raise ValueError("with_events=True needs an epistasis network on the tumour")
        events = net.event_names()
    return list(events) + list(traits)


def _carried_bits(rep, traits, n_events):
    """Bitmask of what genotype ``rep`` carries: its epistasis events in the low ``n_events`` bits,
    then one bit per trait with >= 1 mutated copy of the trait's genes."""
    bits = rep.event_bits if n_events else 0
    gs = rep.genome_summary
    for k, name in enumerate(traits):
        if gs.get(_TRAIT_TALLY[name], 0) > 0:
            bits |= 1 << (n_events + k)
    return bits


def trait_table(tumor, traits=TRAIT_EVENTS, with_events=False):
    """Per-genotype trait acquisitions, in :meth:`GenotypeTumor.event_table`'s shape.

    One row per live cancer genotype, in every compartment (primary and metastasis): its cell count,
    the traits it has acquired (``events``, indices into ``traits``) and the order its lineage acquired
    them, read off the genealogy (``tumor.genotypes_parents``). A trait is acquired at the first
    genotype of the lineage that carries a mutated copy of one of its genes — under
    ``trait_source="mutation"`` (the paper regime) exactly where the trait turns on. Acquisitions are
    MONOTONE, as epistasis events are and as MHN/TreeMHN assume: a lineage that later loses every
    mutated copy (a deletion) still counts as having acquired the trait. ``event_groups`` holds the
    traits acquired in one division as one tied group, ``event_order`` flattens them in index order.

    ``with_events=True`` prepends the tumour's epistasis events (``E0``, ``E1``, ...), so a cohort can
    be read on traits and event modules together; trait ``k`` is then index ``n_events + k``.
    """
    names = _trait_names(traits, with_events, tumor)
    traits = tuple(traits)
    n_events = len(names) - len(traits)
    cols = ["genotype_id", "n_cells", "events", "event_order", "event_groups"]
    parents, genos = tumor.genotypes_parents, tumor.genotypes
    acquired, groups = {}, {}                       # gid -> carried-so-far bitmask, acquisition groups

    def resolve(gid):
        chain, cur = [], gid
        while cur is not None and cur not in acquired:
            chain.append(cur)
            p = parents.get(cur)
            cur = p if (p is not None and p in genos and genos[p].type == "cancer") else None
        base_bits = acquired[cur] if cur is not None else 0
        base_groups = groups[cur] if cur is not None else ()
        for g in reversed(chain):
            new = _carried_bits(genos[g], traits, n_events) & ~base_bits
            if new:
                base_groups = base_groups + (tuple(i for i in range(len(names)) if new >> i & 1),)
                base_bits |= new
            acquired[g], groups[g] = base_bits, base_groups

    rows = []
    for gid, count in tumor.genotypes_counts.items():
        if not tumor._is_cancer(gid) or count <= 0:
            continue
        resolve(gid)
        grp = groups[gid]
        rows.append(dict(genotype_id=gid, n_cells=int(count),
                         events=tuple(i for i in range(len(names)) if acquired[gid] >> i & 1),
                         event_order=tuple(e for g in grp for e in g), event_groups=grp))
    tbl = pd.DataFrame(rows, columns=cols)
    tbl.attrs["names"] = names
    return tbl


def trait_cell_fractions(tumor, traits=TRAIT_EVENTS, with_events=False):
    """Per-trait fraction of the tumour's cancer cells (primary and metastasis together) whose lineage
    has acquired it; see :func:`trait_table`. Returned as a Series indexed by name."""
    tbl = trait_table(tumor, traits=traits, with_events=with_events)
    names = tbl.attrs["names"]
    frac = np.zeros(len(names))
    total = tbl["n_cells"].sum() if not tbl.empty else 0
    if total > 0:
        for events, cnt in zip(tbl["events"], tbl["n_cells"]):
            for e in events:
                frac[e] += cnt
        frac = frac / total
    return pd.Series(frac, index=names)


def to_trait_matrix(tumors, traits=TRAIT_EVENTS, min_freq=DETECTION_LIMIT, with_events=False):
    """Patients x traits binary DataFrame (MHN's input): 1 where the trait is carried by at least
    ``min_freq`` of the tumour's cancer cells, primary and metastasis together (default
    :data:`DETECTION_LIMIT`; 0 means any cell). ``with_events=True`` adds the epistasis events as
    leading columns, at the same threshold."""
    tumors = list(tumors)
    if not tumors:
        raise ValueError("no tumours given")
    fr = [trait_cell_fractions(t, traits=traits, with_events=with_events) for t in tumors]
    return pd.DataFrame([_present(f.values, min_freq) for f in fr], columns=list(fr[0].index),
                        index=[f"P{i}" for i in range(len(tumors))])


def to_trait_tree(tumor, traits=TRAIT_EVENTS, detection=DETECTION_LIMIT, with_events=False):
    """The patient's true clone tree collapsed to trait acquisitions, in :func:`to_mutation_tree`'s
    format (``Mutation_ID = k + 1`` for trait ``k`` of :func:`trait_table`'s names; 0 is the root).
    ``detection`` is the clone-size limit, as in :func:`to_mutation_tree` (default
    :data:`DETECTION_LIMIT`)."""
    return _tree_from_table(trait_table(tumor, traits=traits, with_events=with_events), 0.0, detection)


def to_trait_trees(tumors, traits=TRAIT_EVENTS, detection=DETECTION_LIMIT, with_events=False,
                   drop_empty=True):
    """Every patient's trait tree in one frame, as :func:`to_treemhn_trees` stacks event trees
    (patients with no trait above the limit dropped and the rest renumbered, with ``drop_empty``)."""
    return stack_trees([to_trait_tree(t, traits=traits, detection=detection, with_events=with_events)
                         for t in tumors], drop_empty, "trait")


# ------------------------------------------------------------------ a dependency-free baseline
def cooccurrence_scores(matrix, pseudocount=0.5):
    """Pairwise **log odds ratio** of co-occurrence — the DISCOVER/MEGSA-style statistic.

    ``> 0`` = the pair co-occurs more than chance (looks synergistic), ``< 0`` = looks mutually
    exclusive. A Haldane–Anscombe ``pseudocount`` keeps empty cells finite.

    This is a MARGINAL statistic: it cannot tell a direct interaction from one induced by a shared
    dependency, which is precisely the confound MHN's regularized fit is built to remove. Treat it
    as the floor of the benchmark, not as a method.
    """
    X = np.asarray(matrix, dtype=float)
    n_events = X.shape[1]
    S = np.zeros((n_events, n_events))
    for i in range(n_events):
        for j in range(i + 1, n_events):
            a = np.sum((X[:, i] == 1) & (X[:, j] == 1)) + pseudocount  # both
            b = np.sum((X[:, i] == 1) & (X[:, j] == 0)) + pseudocount
            c = np.sum((X[:, i] == 0) & (X[:, j] == 1)) + pseudocount
            d = np.sum((X[:, i] == 0) & (X[:, j] == 0)) + pseudocount  # neither
            S[i, j] = S[j, i] = np.log((a * d) / (b * c))
    return S


def top_edges(score_matrix, k):
    """The ``k`` strongest pairs of a symmetric score matrix as ``(i, j, score)``, ``i < j``,
    ranked by |score| — the usual way a continuous method is turned into an edge set."""
    n = score_matrix.shape[0]
    pairs = [(i, j, float(score_matrix[i, j])) for i in range(n) for j in range(i + 1, n)]
    pairs.sort(key=lambda p: -abs(p[2]))
    return pairs[:k]


# ------------------------------------------------------------------ scoring vs the planted truth
def score_edges(true_edges, pred_edges, match_sign=False):
    """Precision / recall / F1 of recovered interaction edges against the planted ones.

    ``true_edges`` / ``pred_edges`` are ``(i, j)`` or ``(i, j, weight)``; pairs are compared
    UNDIRECTED (epistasis is symmetric). With ``match_sign=True`` an edge only counts as recovered
    if the sign of its weight also matches — the stricter, more meaningful test, since calling a
    synergy an antagonism is a real error, not a near-miss.
    """
    def key(e):
        i, j = int(e[0]), int(e[1])
        pair = (min(i, j), max(i, j))
        if match_sign:
            if len(e) < 3:
                raise ValueError("match_sign=True needs weighted edges (i, j, weight)")
            return pair + (np.sign(e[2]),)
        return pair

    T, P = {key(e) for e in true_edges}, {key(e) for e in pred_edges}
    tp = len(T & P)
    precision = tp / len(P) if P else 0.0
    recall = tp / len(T) if T else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return dict(precision=precision, recall=recall, f1=f1,
                tp=tp, fp=len(P - T), fn=len(T - P), n_true=len(T), n_pred=len(P))


def _ranks(order):
    """Event -> rank, from either a flat sequence ``(0, 2, 1)`` or a GROUPED one ``((0, 2), (1,))``.

    Grouped input is the honest form: events acquired in the same division are TIED and share a rank,
    so a tie can be detected rather than silently resolved by whatever order they were listed in.
    """
    ranks = {}
    for k, item in enumerate(order):
        if isinstance(item, (tuple, list, set, frozenset)):
            for e in item:
                ranks[int(e)] = k
        else:
            ranks[int(item)] = k
    return ranks


def score_order(true_dag_edges, trees):
    """Score a planted dependency DAG ``A -> B`` against the realised lineages, on TWO distinct axes.

    These must not be conflated, because the two gating modes fail on different ones:

      * ``order_accuracy`` — among lineages carrying BOTH events, the fraction in which ``A`` really
        did precede ``B``. This is the ordering question.
      * ``constraint_satisfaction`` — among lineages carrying ``B`` at all, the fraction that also
        carry ``A``. This is the *conjunctive* question (CBN's poset: "B requires A"), and it is
        exactly what accessibility gating enforces and fitness gating does not.

    A lineage with ``B`` but no ``A`` violates the constraint but says nothing about order, so it is
    counted in ``n_child_without_parent`` and excluded from ``order_accuracy``. Folding it in as an
    ordering error (an easy mistake) would report fitness gating as *worse than chance* at ordering,
    when in truth its ordering is chance and it simply does not enforce the conjunction.

    ``trees`` is an iterable of event orders — ideally the GROUPED
    ``tumor.event_table()['event_groups']``, in which events acquired in one division share a rank.
    Tied pairs (acquired together, so no order exists to recover) are excluded and reported as
    ``n_tied``: scoring them would credit or penalise a method for an ordering the simulator never
    generated. Flat sequences are accepted, but there ties are indistinguishable from real orderings —
    a silent bias whose direction depends on the tie-break — so prefer the grouped form.
    """
    ok = total = tied = uninformative = 0
    n_child = n_child_without_parent = 0
    for (parent, child) in true_dag_edges:
        seen = 0
        for order in trees:
            pos = _ranks(order)
            if child not in pos:
                continue            # lineage never acquired the child: uninformative
            n_child += 1
            if parent not in pos:
                n_child_without_parent += 1   # violates the conjunction; says nothing about order
                continue
            if pos[parent] == pos[child]:
                tied += 1           # acquired in the same division: no order to score
                continue
            seen += 1
            total += 1
            if pos[parent] < pos[child]:
                ok += 1
        if seen == 0:
            uninformative += 1
    return dict(order_accuracy=ok / total if total else float("nan"),
                constraint_satisfaction=(1 - n_child_without_parent / n_child) if n_child else float("nan"),
                n_scored=total, n_tied=tied, n_child=n_child,
                n_child_without_parent=n_child_without_parent, n_uninformative=uninformative)


def score_exclusivity(true_exclusive_pairs, score_matrix):
    """Sign recovery for the planted mutually-exclusive pairs: the fraction whose measured
    co-occurrence score is negative (i.e. correctly called exclusive rather than co-occurring)."""
    if not true_exclusive_pairs:
        return dict(sign_accuracy=float("nan"), n_pairs=0)
    hits = sum(1 for (i, j) in true_exclusive_pairs if score_matrix[i, j] < 0)
    return dict(sign_accuracy=hits / len(true_exclusive_pairs), n_pairs=len(true_exclusive_pairs))
