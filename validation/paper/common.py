"""Shared readouts for the paper-1 figures: clones, tissue maps and cached assay samples.

Every figure reads a registry tumour through these helpers, so a "clone", a tissue colour or a
sampled data set means the same thing in every panel:

* **Clones** are lineage clades of the true genealogy, defined exactly as
  :func:`iscc.integrations.clones_from_clades` defines them for sampled cells, but on the tumour's
  live genotype counts. ``clone 1`` is the largest clade; anything above every clade is
  ``ancestral``.
* **Tissue maps** colour each deme by its dominant clone, blended into the tissue colour (lumen,
  epithelial wall, stroma) by the deme's cancer fraction, from the live state or from any snapshot
  of the spatial history.
* **Assay samples** (a dissociated sample through scDNA and scRNA, a thin section through Visium)
  are generated once per tumour and cached under ``analysis_data/paper/<tier>/derived/``.
"""
import os
import pickle
import sys
from collections import Counter

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import regime as RG  # noqa: E402
import registry as REG  # noqa: E402
import style as S  # noqa: E402
import realistic_regime as RR  # noqa: E402
from iscc.integrations.clones import _clade_owner, _clade_roots, _depths  # noqa: E402

DERIVED = os.path.join(REG.OUT, "derived")
ANCESTRAL = "ancestral"


# --- clones --------------------------------------------------------------------------------------
def live_cancer_counts(t):
    return {g: c for g, c in t.genotypes_counts.items() if c > 0 and t._is_cancer(g)}


def parents_of(t):
    return {str(k): str(v) for k, v in t.genotypes_parents.items()}


class Clones:
    """The clone partition of a tumour: clade roots, their labels and colours, and a lookup that
    works for any genotype ever created (so snapshots of the past use the same labels)."""

    def __init__(self, t, min_frac=0.04):
        counts = live_cancer_counts(t)
        self.parents = parents_of(t)
        depth = _depths(self.parents, counts)
        min_cells = max(1, int(min_frac * sum(counts.values())))
        roots, subtree = _clade_roots(self.parents, depth, counts, min_cells)
        roots = sorted(roots, key=lambda g: (-subtree[g], t.genotypes[g].ord if g in t.genotypes else 0))
        self.roots = roots
        self.label = {g: f"clone {i}" for i, g in enumerate(roots, start=1)}
        self.labels = [self.label[g] for g in roots]
        palette = ([fam[1] for fam in S.CLONE_FAMILIES[:-1]] + [fam[0] for fam in S.CLONE_FAMILIES[:-1]]
                   + [fam[2] for fam in S.CLONE_FAMILIES[:-1]])          # 15 before any colour repeats
        self.colour = {lab: palette[i % len(palette)] for i, lab in enumerate(self.labels)}
        self.colour[ANCESTRAL] = S.CLONE_FAMILIES[-1][2]
        self.size = {self.label[g]: subtree[g] for g in roots}
        self._cache = {}

    def of(self, gid):
        """Clone label of any genotype: its nearest clade-root ancestor-or-self, else ancestral."""
        gid = str(gid)
        if gid in self._cache:
            return self._cache[gid]
        chain, cur, lab = [], gid, ANCESTRAL
        while cur is not None:
            if cur in self.label:
                lab = self.label[cur]
                break
            if cur in self._cache:
                lab = self._cache[cur]
                break
            chain.append(cur)
            nxt = self.parents.get(cur)
            cur = None if nxt in (None, cur) else nxt
        for c in chain:
            self._cache[c] = lab
        self._cache[gid] = lab
        return lab

    def tree(self):
        """Edges between clone roots: each clone's parent is the clone owning its root's parent
        (``ancestral`` for clones that branch off above every other clone)."""
        return {self.label[g]: self.of(self.parents.get(g)) if self.parents.get(g) else ANCESTRAL
                for g in self.roots}


# --- tissue maps ---------------------------------------------------------------------------------
def tissue_background(t):
    """(G, G, 3) RGB of the primary's normal tissue: lumen white, epithelial wall, stroma."""
    G = t.grid_size
    rgb = np.tile(np.array(_hex(S.COMPARTMENT["stroma"])), (G, G, 1)).astype(float)
    gland = np.asarray(t.gland_id)[:G * G] if t.gland_id is not None else np.full(G * G, -1)
    lumen = set(int(d) for d in np.concatenate([np.asarray(x) for x in (t.gland_lumen_demes or [])]))\
        if t.gland_lumen_demes else set()
    for di in range(G * G):
        r, c = t.deme_coords[di]
        if gland[di] >= 0:
            rgb[r, c] = _hex(S.COMPARTMENT["lumen"] if di in lumen else S.COMPARTMENT["epithelium"])
    return rgb


def tissue_map(t, clones, deme_counts=None, min_alpha=0.25, outline=True):
    """RGB map of the primary: each deme's dominant clone over the tissue, by cancer fraction.

    ``deme_counts`` is ``{deme: {gid: count}}`` (default: the live state); pass a snapshot from
    :func:`snapshot_counts` to draw the past."""
    G = t.grid_size
    rgb = tissue_background(t)
    demes = deme_counts if deme_counts is not None else {i: d for i, d in enumerate(t.demes[:G * G])}
    for di, deme in demes.items():
        if di >= G * G:
            continue
        cancer = Counter()
        total = 0
        for g, c in deme.items():
            total += c
            if t._is_cancer(g):
                cancer[clones.of(g)] += c
        n = sum(cancer.values())
        if not n:
            continue
        lab = cancer.most_common(1)[0][0]
        a = min_alpha + (1 - min_alpha) * n / max(total, 1)
        r, c = t.deme_coords[di]
        rgb[r, c] = (1 - a) * rgb[r, c] + a * np.array(_hex(clones.colour[lab]))
    if outline and t.gland_id is not None:
        # the gland walls over the cancer, so the ducts stay visible once cancer fills them
        gland = np.asarray(t.gland_id)[:G * G]
        ink = np.array(_hex(S.INK))
        for di in np.where(gland >= 0)[0]:
            if any(nb < G * G and gland[nb] == -1 for nb in t._neighbors(int(di))):
                r, c = t.deme_coords[di]
                rgb[r, c] = 0.5 * rgb[r, c] + 0.5 * ink
    return rgb


def snapshot_counts(t, snapshots, k):
    """``{deme: {gid: count}}`` of the k-th recorded spatial snapshot (``snapshots`` is
    ``t.spatial_snapshots()``, passed in so callers build the long table once)."""
    sub = snapshots[snapshots.snapshot == k]
    out = {}
    for d, g, c in zip(sub.deme.values, sub.genotype.values, sub["count"].values):
        out.setdefault(int(d), {})[g] = int(c)
    return out


def _hex(h):
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]


# --- assay samples -------------------------------------------------------------------------------
def assays(run, task, seed=0, n_dna=400, force=False):
    """A dissociated sample through scDNA (cancer cells) and scRNA (all cells), and a thin section
    through Visium, for one registry tumour. Cached; returns a dict of plain tables."""
    path = os.path.join(DERIVED, f"{run}_{task}_assays.pkl")
    if os.path.exists(path) and not force:
        with open(path, "rb") as f:
            return pickle.load(f)
    from iscc.data import Visium, scDNA, scRNA
    t = REG.load(run, task)
    spec = RR.resect(t)
    diss = spec.dissociate(max_cells=RG.T["max_cells"])
    cells = list(diss["cell_exp"].index)
    gid = diss["cell_type"]["cell_id"].astype(str)
    is_cancer = np.array([t._is_cancer(g) if g in t.genotypes else False for g in gid.values])
    cancer_cells = [c for c, k in zip(cells, is_cancer) if k]
    rng = np.random.default_rng(seed)
    dna_cells = list(rng.choice(cancer_cells, size=min(n_dna, len(cancer_cells)), replace=False))
    dna = scDNA(n_cells=len(dna_cells), breadth="wgs", kappa=500.0, mu_depth=60.0,
                seed=200 + seed).run(diss, cell_subset=dna_cells)
    rna = scRNA(n_cells=len(cells), seed=100 + seed).run(diss, cell_subset=cells)
    sec = spec.slice(depth_frac=0.5, max_cells=RG.T["max_cells"])
    vis = Visium(seed=300 + seed, **RG.VISIUM).run(sec, grid_side=t.grid_size)
    out = dict(
        cells=cells, genotype=gid.to_dict(), is_cancer=dict(zip(cells, is_cancer)),
        cell_type=_coarse_type(diss), seg_cn_true=_seg_cn(t, diss),
        dna_cells=list(dna.cells), dna_coverage=dna.coverage.astype(np.float32),
        rna_counts=rna.observed_counts.astype(np.int32),
        programs=diss.get("cell_program"), rna_baf=diss.get("cell_rna_baf"),
        visium_counts=vis.spot_counts.astype(np.int32), visium_coords=np.asarray(vis.spot_coords),
        visium_members=[list(m) for m in vis.spot_members],
        section_genotype=sec["cell_type"]["cell_id"].astype(str).to_dict(),
        section_coords=sec["cell_crd"].copy(), grid_size=int(t.grid_size))
    os.makedirs(DERIVED, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    return out


def _coarse_type(cd):
    ids = cd["cell_type"]["cell_id"].astype(str)
    names = {"epithelial", "stromal", "immune", "host"}
    return {c: (g if g in names else "cancer") for c, g in ids.items()}


def _seg_cn(t, cd):
    """True per-segment copy number of every sampled cell (cells x segments)."""
    gid = cd["cell_type"]["cell_id"].astype(str)
    rows = {c: t.genotypes[g].genome_summary["seg_cns"] if g in t.genotypes else None
            for c, g in gid.items()}
    n = t.n_segments
    return pd.DataFrame({c: (r if r is not None else [2] * n) for c, r in rows.items()}).T
