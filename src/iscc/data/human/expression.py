"""Expression over the human genes from reference-atlas baselines and Hallmark programmes (L3).

On the growth genes a cell's expression is, per homolog, its type's baseline times the program
multiplier times what its genotype and niche do (dosage, SNV effects, hypoxia, CCI). This module keeps
the last part, read off each cell's growth-gene expression as

    F[c, g] = cell_exp[c, g] / (growth baseline[type(c), g] * growth program multiplier[c, g]),

and rebuilds the rest on the human genes:

    exp[c, h] = B[type(c), h] * F[c, cover(h)] * exp(bounded Σ_k z[c, k] L[k, h]).

``B`` is a cell type's composition in the Human Breast Cell Atlas, scaled by its relative RNA content;
the cancer takes its cell of origin's, so a malignant cell differs from normal epithelium only by
copy number, SNVs, programmes and niche. ``L`` loads iscc's named programmes on their Hallmark gene
sets, with the growth dictionary's loading distribution; programmes without a Hallmark counterpart
load on random human genes. The cell's program activities ``z`` are the growth model's.

Nothing here builds a cells × human-genes matrix: :meth:`HumanExpression.cell_expression` works on
chunks of cells, and :meth:`HumanExpression.pool_spots` pools a Visium section per cell type on the
growth genes and lifts the pools, computing only the programme genes cell by cell.
"""
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .reference import load_atlas_baselines, load_hallmark_programs

TINY = 1e-30


class HumanExpression:
    """The human-gene expression of a tumour's cells, for the assays' ``gene_lift``.

    Pass it as ``gene_lift`` to :meth:`scRNA.run <iscc.data.scRNA.run>` (counts are drawn in chunks
    of cells and kept sparse) or :meth:`Visium.run <iscc.data.Visium.run>` (spots are pooled per cell
    type). Build it with :meth:`HumanGenomeLayer.expression`.

    Parameters
    ----------
    layer : HumanGenomeLayer
    tumor : GenotypeTumor
        The grown tumour: its growth baselines, program model and genotypes are read.
    baselines : pandas.DataFrame, optional
        Gene_id-indexed composition per cell type (columns ``epithelial``, ``stromal``, ``immune``);
        default the packaged Human Breast Cell Atlas table, scaled by each type's RNA content.
    programmes : pandas.DataFrame, optional
        ``programme``, ``gene_id`` rows seeding named programmes; default the packaged Hallmark table.
    seed : int, optional
        Seed of the human programme loadings; default the layer's seed.
    chunk : int, default 512
        Cells processed at a time.

    Attributes
    ----------
    genes : pandas.Index
        Ensembl IDs of the human genes (the layer's).
    B : dict
        Baseline per cell type over the human genes; ``B["cancer"]`` is the cell of origin's.
    programme_genes : dict
        Human-gene indices each program loads on.
    """

    def __init__(self, layer, tumor, baselines=None, programmes=None, seed=None, chunk=512):
        self.layer, self.tumor = layer, tumor
        self.genes = layer.lift.genes
        self.cover = layer.cover
        self.chunk = int(chunk)
        H = len(self.genes)
        if baselines is None:
            tab, info = load_atlas_baselines()
            scale = dict(zip(info["type"], info["rna_content"]))
        else:
            tab, scale = baselines, {}
        tab = tab.reindex(self.genes).fillna(0.0)
        self.B = {t: tab[t].to_numpy(float) * float(scale.get(t, 1.0)) for t in tab.columns}
        self.B.setdefault("host", self.B.get("stromal"))      # no breast counterpart: stroma stands in
        origin = (getattr(tumor, "expression_params", None) or {}).get("cell_of_origin")
        if origin is None:
            warnings.warn("the tumour has no expression_params['cell_of_origin']; cancer cells take the "
                          "epithelial baseline", stacklevel=2)
            origin = "epithelial"
        self.origin = origin
        self.B["cancer"] = self.B[origin]
        self.growth_base = {t: np.asarray(v, float) for t, v in tumor.celltype_exps.items()}
        self.P = getattr(tumor, "programs", None)
        if self.P is None:
            warnings.warn("the tumour has no expression layer (expression_params): cancer cells are then "
                          "on the legacy scale (baseline x (1 + copies)) and there are no programmes",
                          stacklevel=2)
        self.max_log_fold = getattr(self.P, "max_log_fold", None)
        self.programme_genes = {}
        self.hp = np.array([], dtype=int)
        self.Lh = np.zeros((0, 0))
        if self.P is not None and self.P.dictionary is not None:
            self._build_programmes(programmes, layer.seed if seed is None else int(seed), H)
        self._spot = None

    # -- programmes -----------------------------------------------------------------------------
    def _build_programmes(self, programmes, seed, H):
        d = self.P.dictionary
        table = load_hallmark_programs() if programmes is None else programmes
        index = pd.Index(self.genes)
        rng = np.random.default_rng([seed, 0x4C3])
        L = sp.lil_matrix((d.n_programs, H))
        for k, name in enumerate(d.program_names):
            ids = table.loc[table["programme"] == name, "gene_id"]
            g = index.get_indexer(ids)
            g = np.unique(g[g >= 0])
            if g.size == 0:                    # no Hallmark counterpart: random human genes, at the
                n = max(1, int(round(len(d.program_genes[k]) * H / self.P.n_genes)))  # same density
                g = np.sort(rng.choice(H, size=min(n, H), replace=False))
            self.programme_genes[name] = g
            L[k, g] = d._draw_loadings(len(g), rng)
        L = L.tocsc()
        self.hp = np.flatnonzero(np.asarray((abs(L)).sum(axis=0)).ravel() > 0)
        self.Lh = L[:, self.hp].toarray()

    def _human_programme(self, Z):
        x = np.asarray(Z, float) @ self.Lh
        if self.max_log_fold is not None:
            c = self.max_log_fold
            x = c * np.tanh(x / c)
        return np.exp(x)

    # -- per-cell factors -----------------------------------------------------------------------
    def _types(self, cell_data, cells):
        gids = cell_data["cell_type"].iloc[:, 0].reindex(cells).to_numpy()
        return np.array([getattr(self.tumor.genotypes.get(g), "type", "cancer") for g in gids])

    def _factors(self, cell_data, cells):
        """``(F, types, Z)``: the genotype-and-niche factor per growth gene, the cell types, and the
        program activities (None without programmes)."""
        ce = cell_data["cell_exp"].reindex(cells).to_numpy(np.float64)
        types = self._types(cell_data, cells)
        base = np.empty_like(ce)
        for t in np.unique(types):
            base[types == t] = self.growth_base[t]
        Z = None
        if self.P is not None and self.P.dictionary is not None and "cell_program" in cell_data:
            Z = cell_data["cell_program"].reindex(cells).to_numpy(np.float64)
            base *= self.P.gene_multiplier(Z)
        F = np.ones_like(ce)
        ok = ce > 0
        F[ok] = ce[ok] / np.maximum(base[ok], TINY)
        F[(~ok) & (base > TINY)] = 0.0           # truly silent (e.g. both homologs lost)
        return F, types, Z

    def cell_expression(self, cell_data, cells):
        """Expression of ``cells`` over the human genes, ``(len(cells), n_human)``; call it on chunks."""
        F, types, Z = self._factors(cell_data, list(cells))
        out = np.empty((len(types), len(self.genes)))
        for t in np.unique(types):
            r = types == t
            out[r] = F[r][:, self.cover] * self.B[t][None, :]
        if Z is not None and self.hp.size:
            out[:, self.hp] *= self._human_programme(Z)
        return out

    # -- Visium ---------------------------------------------------------------------------------
    def pool_spots(self, grid):
        """Pooled human-gene expression of each spot of a placed Visium grid, ``(n_spots, n_human)``.

        Cells are pooled per type on the growth genes, then lifted with their type's baseline; the
        programme genes are summed cell by cell. Also keeps the per-type pools of homolog-``p``
        expression for :meth:`spot_hap_fraction`."""
        placed, members = grid["placed"], grid["members"]
        cells = np.asarray(placed["cell_exp"].index)
        n_spots = len(members)
        occ = np.flatnonzero([m.size > 0 for m in members])
        rows = np.concatenate([np.full(members[s].size, i) for i, s in enumerate(occ)]) if occ.size else np.array([], int)
        cols = np.concatenate([members[s] for s in occ]) if occ.size else np.array([], int)
        M = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(occ), len(cells))).tocsc()
        G = self.layer.n_growth
        pools, pools_p = {}, {}
        raw_hp = np.zeros((len(occ), self.hp.size))
        for lo in range(0, len(cells), self.chunk):
            sl = slice(lo, lo + self.chunk)
            chunk_cells = cells[sl]
            F, types, Z = self._factors(placed, chunk_cells)
            frac = self.layer.hap_fraction(placed, chunk_cells, self.tumor)
            Mc = M[:, sl]
            for t in np.unique(types):
                r = np.flatnonzero(types == t)
                Mt = Mc[:, r]
                pools[t] = pools.get(t, np.zeros((len(occ), G))) + Mt @ F[r]
                pools_p[t] = pools_p.get(t, np.zeros((len(occ), G))) + Mt @ (F[r] * frac[r])
            if self.hp.size:
                X = F[:, self.cover[self.hp]]
                for t in np.unique(types):
                    X[types == t] *= self.B[t][self.hp][None, :]
                if Z is not None:
                    X *= self._human_programme(Z)
                raw_hp += Mc @ X
        raw_occ = np.zeros((len(occ), len(self.genes)))
        for t, pool in pools.items():
            raw_occ += pool[:, self.cover] * self.B[t][None, :]
        if self.hp.size:
            raw_occ[:, self.hp] = raw_hp
        raw = np.zeros((n_spots, len(self.genes)))
        raw[occ] = raw_occ
        self._spot = {"occ": occ, "n_spots": n_spots, "pools": pools, "pools_p": pools_p}
        return raw

    def spot_hap_fraction(self, spot_rows, human_genes):
        """Fraction from homolog ``p`` of a spot's expression of a human gene, for entries
        ``(spot_rows[i], human_genes[i])`` (after :meth:`pool_spots`)."""
        if self._spot is None:
            raise ValueError("pool a Visium section first (Visium.run with gene_lift=this)")
        pos = np.full(self._spot["n_spots"], -1)
        pos[self._spot["occ"]] = np.arange(len(self._spot["occ"]))
        r = pos[np.asarray(spot_rows)]
        if (r < 0).any():
            raise ValueError("allele counts at a spot without cells")
        g = self.cover[np.asarray(human_genes)]
        num = np.zeros(len(r))
        den = np.zeros(len(r))
        for t, pool in self._spot["pools"].items():
            w = self.B[t][np.asarray(human_genes)]
            num += w * self._spot["pools_p"][t][r, g]
            den += w * pool[r, g]
        return np.divide(num, den, out=np.full(len(r), 0.5), where=den > 0)
