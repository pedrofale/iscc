"""Cohort progression-model seam: MHN / TreeMHN / CBN / REVOLVER input + scoring (R14).

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

Scoring (:func:`score_edges`, :func:`score_order`, :func:`score_exclusivity`) is deliberately generic:
it takes an edge/order list from ANY tool and the ground truth from
``tumor.epistasis_ground_truth()``, so a new tool only needs an adapter to its output format.

:func:`cooccurrence_scores` is a built-in, dependency-free baseline (the pairwise log odds ratio —
essentially the DISCOVER/MEGSA statistic). It is not a replacement for MHN: it cannot separate a
direct interaction from one induced through a shared ancestor. It is here so the benchmark always
has a floor to compare a real tool against, and so the validation figure renders without R.
"""
import numpy as np
import pandas as pd


# ------------------------------------------------------------------ export: cross-sectional (MHN)
def clone_events(tumor, min_freq=0.0):
    """Per-clone event tuples with their cell counts, for clones above ``min_freq`` of the tumour."""
    net = tumor.selection.epistasis
    if net is None:
        raise ValueError("tumour has no epistasis network; nothing to export "
                         "(set selection_params['epistasis_params'])")
    tbl = tumor.event_table()
    if tbl.empty:
        return tbl
    total = tbl["n_cells"].sum()
    if min_freq > 0 and total > 0:
        tbl = tbl[tbl["n_cells"] / total >= min_freq]
    return tbl


def patient_event_vector(tumor, min_freq=0.0, n_events=None):
    """Binary event vector for ONE patient: 1 where any surviving clone carries the event.

    ``min_freq`` is the detection floor — a clone below this fraction of the tumour is treated as
    undetected, which is what a real bulk/single-cell assay would do. It matters: at ``min_freq=0``
    every event ever acquired by any surviving cell is "observed", which is more than any real
    cohort study sees.
    """
    net = tumor.selection.epistasis
    n = n_events if n_events is not None else net.n_events
    vec = np.zeros(n, dtype=int)
    tbl = clone_events(tumor, min_freq=min_freq)
    for events in tbl.get("events", []):
        for e in events:
            vec[e] = 1
    return vec


def to_mhn_matrix(tumors, min_freq=0.0):
    """Patients x events binary DataFrame — MHN's (and DISCOVER/MEGSA's) input.

    ``tumors`` is any iterable of grown tumours sharing one network (i.e. one ``Cohort``).
    """
    tumors = list(tumors)
    if not tumors:
        raise ValueError("no tumours given")
    net = tumors[0].selection.epistasis
    rows = [patient_event_vector(t, min_freq=min_freq, n_events=net.n_events) for t in tumors]
    return pd.DataFrame(rows, columns=net.event_names(),
                        index=[f"P{i}" for i in range(len(tumors))])


# ------------------------------------------------------------------ export: mutation trees (TreeMHN)
def to_mutation_tree(tumor, min_freq=0.0):
    """The patient's **mutation tree** as a trie over the realised event orders.

    Each surviving clone contributes the path ``event_order`` from the root; shared prefixes are
    merged, so the result is the tree of "which event followed which" in THIS patient — exactly
    TreeMHN's / REVOLVER's input. The root is ``Mutation_ID = 0`` (the normal genotype) and events
    are numbered ``1..n_events`` (TreeMHN's convention: 0 is reserved for the root).

    Returns a DataFrame with ``Node_ID``, ``Mutation_ID``, ``Parent_ID``, ``n_cells``.
    """
    tbl = clone_events(tumor, min_freq=min_freq)
    nodes = [dict(Node_ID=1, Mutation_ID=0, Parent_ID=0, n_cells=0)]  # root
    children = {}  # (parent Node_ID, mutation) -> Node_ID
    for _, row in tbl.iterrows():
        parent_node = 1
        for ev in row["event_order"]:
            key = (parent_node, int(ev))
            node = children.get(key)
            if node is None:
                node = len(nodes) + 1
                nodes.append(dict(Node_ID=node, Mutation_ID=int(ev) + 1,
                                  Parent_ID=parent_node, n_cells=0))
                children[key] = node
            parent_node = node
        # attribute the clone's cells to the leaf it ends at
        nodes[parent_node - 1]["n_cells"] += int(row["n_cells"])
    return pd.DataFrame(nodes)


def to_treemhn_trees(tumors, min_freq=0.0):
    """Every patient's mutation tree in one tidy frame (a ``Patient_ID`` column added) — the shape
    TreeMHN's R ``input_tree`` list is built from."""
    frames = []
    for i, t in enumerate(tumors):
        df = to_mutation_tree(t, min_freq=min_freq)
        df.insert(0, "Patient_ID", i + 1)
        df.insert(1, "Tree_ID", i + 1)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def to_cbn_poset(tumors, min_freq=0.0):
    """CBN/H-CBN input: the cross-sectional binary genotype matrix (same shape as MHN's), which
    H-CBN fits a poset to. Kept as its own name because the two tools' semantics differ even though
    the file format coincides."""
    return to_mhn_matrix(tumors, min_freq=min_freq)


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
