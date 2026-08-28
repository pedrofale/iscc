"""Reconstruct each patient's mutation tree with SCITE, instead of reading iscc's own.

WHY. TreeMHN's input is a mutation tree per patient. iscc knows every patient's true tree exactly
(`integrations.progression.to_treemhn_trees`), and handing that straight to TreeMHN makes the
benchmark vacuous: the method is then estimating rates from the topology it was supposed to infer.
A real study never has it. So the trees TreeMHN reads are INFERRED here, by SCITE (Jahn, Kuipers &
Beerenwinkel 2016) — the standard single-cell mutation-tree reconstructor, and TreeMHN's own
assumed upstream — from a noisy single-cell genotype matrix.

The noise is the point. A single-cell assay misses real mutations (allelic dropout) and calls
absent ones (false positives), so the matrix SCITE sees is a corrupted view of the truth and the
tree it returns is an ESTIMATE. SCITE is told the same error rates the matrix was corrupted with,
which is what a study does after estimating them from its own data.

SCITE is a command-line program with no library API — its CLI is its interface — so it is run here,
in the generator, and never from a notebook.
"""
import os
import re
import shutil
import subprocess
import tempfile

import numpy as np
import pandas as pd

#: Single-cell genotyping error rates. Dropout dominates and is the reason a tree has to be
#: inferred rather than read off: at 20% a fifth of the real calls are simply missing.
FALSE_POSITIVE = 0.01
DROPOUT = 0.20
N_CELLS_SEQUENCED = 500


def scite_binary():
    """Locate the compiled `scite`, preferring the dedicated env."""
    env_bin = os.path.expanduser("~/miniconda3/envs/iscc-scite/bin/scite")
    return env_bin if os.path.exists(env_bin) else shutil.which("scite")


def single_cell_matrix(tumor, n_cells=N_CELLS_SEQUENCED, fd=FALSE_POSITIVE, ad=DROPOUT, seed=0):
    """A noisy events x cells binary genotype matrix, SCITE's input format.

    Cells are drawn from the patient's clones in proportion to clone size — the sampling a
    single-cell experiment does — and then corrupted: a carried event is lost with probability
    ``ad``, an absent one called with probability ``fd``. Returns ``(observed, true)``; only
    ``observed`` is ever given to SCITE.
    """
    net = tumor.selection.epistasis
    n = net.n_events
    tbl = tumor.event_table()
    if tbl.empty or tbl["n_cells"].sum() <= 0:
        return np.zeros((n, 0), int), np.zeros((n, 0), int)

    rng = np.random.default_rng(seed)
    weights = tbl["n_cells"].to_numpy(dtype=float)
    picks = rng.choice(len(tbl), size=n_cells, p=weights / weights.sum())

    true = np.zeros((n, n_cells), dtype=int)
    for c, row in enumerate(picks):
        for e in tbl["events"].iloc[row]:
            true[int(e), c] = 1

    obs = true.copy()
    obs[(true == 1) & (rng.random(true.shape) < ad)] = 0
    obs[(true == 0) & (rng.random(true.shape) < fd)] = 1
    return obs, true


def run_scite(obs, names, fd=FALSE_POSITIVE, ad=DROPOUT, seed=1, chain_length=100_000):
    """Run SCITE on an events x cells matrix; return its ML tree as (parent, child) name pairs.

    ``Root`` is SCITE's own label for the normal genotype at the top of the tree.
    """
    binary = scite_binary()
    if binary is None:
        raise RuntimeError("scite not found — build it with `clang++ -O2 -std=c++11 *.cpp -o scite` "
                           "in a clone of github.com/cbg-ethz/SCITE and put it on PATH "
                           "(or in ~/miniconda3/envs/iscc-scite/bin)")
    n, m = obs.shape
    with tempfile.TemporaryDirectory() as tmp:
        mat = os.path.join(tmp, "sc.txt")
        nam = os.path.join(tmp, "names.txt")
        np.savetxt(mat, obs, fmt="%d")
        with open(nam, "w") as fh:
            fh.write("\n".join(names) + "\n")
        cmd = [binary, "-i", mat, "-n", str(n), "-m", str(m), "-r", "1", "-l", str(chain_length),
               "-fd", str(fd), "-ad", str(ad), "-names", nam, "-max_treelist_size", "1",
               "-seed", str(seed)]
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp)
        if proc.returncode != 0:
            raise RuntimeError(f"scite failed ({proc.returncode}): {proc.stderr[-400:]}")
        gv = [f for f in os.listdir(tmp) if f.endswith(".gv")]
        if not gv:
            raise RuntimeError(f"scite wrote no .gv tree; stdout: {proc.stdout[-400:]}")
        text = open(os.path.join(tmp, sorted(gv)[0])).read()
    return parse_gv(text)


def parse_gv(text):
    """(parent, child) name pairs from a SCITE GraphViz tree."""
    edges = []
    for line in text.splitlines():
        m = re.match(r"\s*(\S+)\s*->\s*(\S+)\s*;", line)
        if m:
            edges.append((m.group(1), m.group(2)))
    return edges


def edges_to_treemhn(edges, names, patient_id):
    """A SCITE tree in TreeMHN's `input_tree_df` layout.

    TreeMHN's convention (its README): the root is ``Node_ID = 1`` with ``Mutation_ID = 0`` and is
    **its own parent**; events are numbered 1..n. SCITE labels mutations by the names it was given,
    so the mapping back to event indices is by position in ``names``.

    ``Tree_ID`` must be UNIQUE PER PATIENT, not a constant. Node ids restart at 1 in every tree, so
    a shared Tree_ID makes TreeMHN read the whole cohort as one tree; its ``remove_duplicates`` then
    resolves a parent by ``which(Node_ID == j)``, matches one row per patient, and dies with
    "the condition has length > 1". iscc's own ``to_treemhn_trees`` numbers them per patient too.
    """
    idx = {nm: i + 1 for i, nm in enumerate(names)}          # E0 -> Mutation_ID 1
    node_of = {"Root": 1}
    rows = [dict(Patient_ID=patient_id, Tree_ID=patient_id, Node_ID=1, Mutation_ID=0, Parent_ID=1)]
    # SCITE's edge list is not guaranteed to be parent-before-child, so walk it from the root.
    children = {}
    for parent, child in edges:
        children.setdefault(parent, []).append(child)
    stack = ["Root"]
    while stack:
        parent = stack.pop()
        for child in children.get(parent, []):
            if child in node_of:                              # already placed (defensive)
                continue
            node = len(rows) + 1
            node_of[child] = node
            rows.append(dict(Patient_ID=patient_id, Tree_ID=patient_id, Node_ID=node,
                             Mutation_ID=idx[child], Parent_ID=node_of[parent]))
            stack.append(child)
    return pd.DataFrame(rows)


def reconstruct_cohort(tumors, seed=0, **kw):
    """SCITE-inferred trees for a whole cohort, plus the matrices it read.

    A patient whose inferred tree has no edges below the root carries no ordering information and
    TreeMHN rejects it outright, so it is dropped and the rest renumbered contiguously — TreeMHN
    indexes patients by position, so a gap in the IDs is not safe to leave behind.
    """
    net = tumors[0].selection.epistasis
    names = list(net.event_names())
    frames, matrices, kept = [], [], []
    for i, t in enumerate(tumors):
        obs, _ = single_cell_matrix(t, seed=seed + i, **kw)
        if obs.shape[1] == 0:
            continue
        edges = run_scite(obs, names, seed=1 + i)
        df = edges_to_treemhn(edges, names, patient_id=len(frames) + 1)
        if len(df) < 2:
            continue
        frames.append(df)
        matrices.append(obs)
        kept.append(i)
    if not frames:
        raise ValueError("SCITE reconstructed no non-trivial tree for any patient")
    return pd.concat(frames, ignore_index=True), matrices, kept
