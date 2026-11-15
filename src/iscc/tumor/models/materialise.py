"""Materialisation: from per-deme genotype counts to per-cell ground-truth tables, with the
microenvironment fields and the coalescent re-creation of folded passengers.

A mixin of :class:`~iscc.tumor.models.count.GenotypeTumor`; it holds no state of its own."""

import warnings
from collections import Counter
import numpy as np
import pandas as pd

from ...constants import LAYOUT_OFFSET_F8_PROGRAMS


# Fraction of a clone's realized folded passenger burden (``_pass_load``) that is treated as
# CLADE-CLONAL (a "stem" born on the clone's founding edge, inherited by the whole subtree) vs.
# PRIVATE (per-cell singletons). 0.35 was chosen empirically: high enough that the shared clade
# signal dominates the per-cell noise (so a passenger-only tree recovers the clone-tree clades),
# low enough to keep a realistic per-cell singleton tail (the SFS is not collapsed to pure clonal
# markers).
F_STEM_DEFAULT = 0.35

# Every clone-split that separates sampled cells should be MARKABLE, so each clone on a sampled
# lineage gets at least this many stem markers even when its folded burden rounds to zero.
STEM_FLOOR = 2

# Per-clone cap on stem markers, so the finite neutral-site budget spreads across many splits
# instead of saturating on one early high-burden clone (raise the genome size for deeper trees).
STEM_CAP = 6

# Dedicated RNG offset for the passenger reconstruction (added to ``self.seed``) — a fixed constant
# so the overlay is reproducible and independent of the growth stream.
PASSENGER_RNG_OFFSET = 909090

# Purpose keys of the per-run streams (see ``_run_rng``).
_STREAM_GROW, _STREAM_PASSENGERS, _STREAM_DEPTH, _STREAM_SUBSAMPLE = 0, 1, 2, 3

# The gene-wide per-cell frames (cells × genes) that `make_cell_data` can build, and the ones each of
# iscc's assay classes reads (`layers=` takes either). The reads layer additionally reads
# `cell_rna_vaf` (scRNA and Visium reads) and `cell_snv`.
CELL_LAYERS = ("cell_snv", "cell_cnv", "cell_exp", "cell_snv_p", "cell_snv_m", "cell_rna_vaf",
               "cell_exp_p", "cell_exp_m", "cell_rna_baf")

ASSAY_LAYERS = {
    "visium": ("cell_exp",),
    "imaging": ("cell_exp",),
    "scrna": ("cell_exp", "cell_snv", "cell_rna_vaf"),
    "scdna": ("cell_snv", "cell_cnv"),
}

# Gene-wide rows are built per chunk of cells of about this many float64 bytes (and at least this
# many cells), so the transient working set of materialisation stays small whatever the number of cells.
_MATERIALISE_CHUNK_BYTES = 64 << 20

_MATERIALISE_MIN_ROWS = 256


def _resolve_layers(layers):
    """The set of gene-wide frames ``make_cell_data(layers=...)`` builds (see ``CELL_LAYERS``)."""
    if layers is None:
        return frozenset(CELL_LAYERS)
    if isinstance(layers, str):
        layers = (layers,)
    out = set()
    for name in layers:
        if name in ASSAY_LAYERS:
            out.update(ASSAY_LAYERS[name])
        elif name in CELL_LAYERS:
            out.add(name)
        else:
            raise ValueError(f"unknown layer {name!r}: give frames from {CELL_LAYERS} "
                             f"or assays from {tuple(ASSAY_LAYERS)}")
    return frozenset(out)


def _program_multiplier(P, Z):
    """``exp(Z · loading)`` for a chunk of cells, summed over programs one at a time, so each value
    is the same whichever chunk its cell falls in (see ``make_cell_data``). The log-fold is bounded as
    in ``ProgramModel.gene_multiplier``: ``x -> c·tanh(x / c)`` when ``max_log_fold`` is ``c``."""
    if P.dictionary is None or Z.shape[1] == 0:
        return np.ones((Z.shape[0], P.n_genes))
    loading = P.dictionary.loading
    acc = np.zeros((Z.shape[0], loading.shape[1]))
    for k in range(Z.shape[1]):
        acc += Z[:, k:k + 1] * loading[k]
    c = getattr(P, "max_log_fold", None)
    if c is not None:
        return np.exp(c * np.tanh(acc / c))
    return np.exp(acc)


class MaterialiseMixin:
    def _init_microenv_genes(self):
        """Designate the hypoxia-responsive and CCI target genes and the ligand-receptor database."""
        # Microenvironment program designation: pick the hypoxia-responsive and CCI-receptor-target
        # gene sets (the ground-truth cell-extrinsic modules). A DEDICATED rng (not self.rng) keeps
        # the growth trajectory the same whether the layer is on or off at a given seed.
        # WHICH genes respond to hypoxia is a property of the GENOME, not of a run, so the stream is
        # the LAYOUT one: two patients in a cohort share their niche programs exactly as they share
        # their oncogenes.
        self._hypoxia_genes = np.array([], dtype=int)
        self._cci_target_genes = np.array([], dtype=int)
        # iscc's OWN ligand-receptor database. `_cci_pairs` is (n_candidate_pairs x 2) of gene
        # indices (col 0 = ligand, col 1 = receptor); row 0 is the WIRED pair the CCI channel is
        # driven by, every other row is an unwired decoy. Empty (and ligand/receptor None) unless a
        # CCI channel is active.
        self._cci_pairs = np.empty((0, 2), dtype=int)
        self._cci_ligand = None
        self._cci_receptor = None
        if self.microenv_params:
            prog_rng = np.random.default_rng(self.layout_seed + LAYOUT_OFFSET_F8_PROGRAMS)
            hyp = ((self.microenv_params.get("hypoxia") or {}) if isinstance(self.microenv_params, dict) else {})
            cci = ((self.microenv_params.get("cci") or {}) if isinstance(self.microenv_params, dict) else {})
            n_hyp = int(hyp.get("n_genes", 0))
            n_cci = int(cci.get("n_target_genes", 0))
            if n_hyp > 0:
                self._hypoxia_genes = prog_rng.choice(self.n_genes, size=min(n_hyp, self.n_genes),
                                                      replace=False)
            if n_cci > 0:
                self._cci_target_genes = prog_rng.choice(self.n_genes, size=min(n_cci, self.n_genes),
                                                         replace=False)
            # Draw the L-R database from the SAME layout sub-stream, AFTER the target-gene draws, so
            # it is part of the shared landscape (pooling patients onto ONE landscape is only
            # justified against a shared network). `n_candidate_pairs` defaults to 1 -> a database
            # holding only the wired pair.
            # The WIRED pair's ligand and receptor are themselves MODULATED by the CCI field (see
            # `_microenv_deme_mod`): that is what puts the L-R signal in the two genes every CCI tool
            # actually reads. Only the hypoxia set is excluded from the pool, so a candidate's
            # expression is never moved by the unrelated hypoxia programme (which would boost a DECOY
            # and blur the benchmark); overlap with the CCI target set is harmless and allowed.
            if float(cci.get("strength", 0.0)) != 0.0 and n_cci > 0:
                n_pairs = max(1, int(cci.get("n_candidate_pairs", 1)))
                # Candidates must be genes that are actually EXPRESSED. Baseline expression is
                # beta(0.1, 1.0) per cell type — CDF x^0.1, so the median gene sits at ~1e-3 and is
                # effectively silent. Drawing a ligand/receptor uniformly lands on a silent gene most
                # of the time, and the field boost is MULTIPLICATIVE: 0 * (1 + strength*field) = 0, so
                # the channel would be undetectable and the wired pair would be handicapped against
                # decoys that happen to be well expressed. Restricting the pool to genes above the
                # genome-mean baseline (~top 20%) avoids both. A designation rule, not a parameter.
                base_expr = np.mean([self.celltype_exps[ct] for ct in self.celltype_exps], axis=0)
                pool = np.setdiff1d(np.flatnonzero(base_expr > base_expr.mean()),
                                    self._hypoxia_genes)
                if len(pool) < 2 * n_pairs:                  # small genomes: take the best expressed
                    ranked = np.argsort(base_expr)[::-1]
                    ranked = ranked[~np.isin(ranked, self._hypoxia_genes)]
                    pool = np.sort(ranked[:2 * n_pairs])
                k = min(2 * n_pairs, len(pool) - (len(pool) % 2))     # even: 2 distinct genes/pair
                if k >= 2:
                    picks = prog_rng.choice(pool, size=k, replace=False)
                    self._cci_pairs = picks.reshape(-1, 2).astype(int)
                    self._cci_ligand = int(self._cci_pairs[0, 0])
                    self._cci_receptor = int(self._cci_pairs[0, 1])


    # Computed once per make_cell_data (a snapshot), so tau-leaping-compatible. These modulate the
    # expression READOUT only; the microenvironment does not feed back into fitness.
    def _deme_density(self):
        """Per-deme occupancy = total cells / carrying capacity (drives O2 consumption)."""
        cc = self._cap
        return np.array([sum(d.values()) / cc for d in self.demes], dtype=float)

    def _emitter_density(self, emitter_type, weight=None):
        """Per-deme density of a ligand-emitting cell type (e.g. immune).

        ``weight`` (a ``gid -> per-cell level`` map) turns the bare count into per-deme LIGAND
        availability: each emitter cell contributes its ligand EXPRESSION instead of 1. ``None``
        gives the plain count fraction, which the O2 perfusion source uses."""
        cc = self._cap
        out = np.zeros(len(self.demes))
        for i, deme in enumerate(self.demes):
            acc = 0.0
            for gid, c in deme.items():
                if self.genotypes[gid].type == emitter_type:
                    acc += c * (1.0 if weight is None else float(weight.get(gid, 0.0)))
            out[i] = acc / cc
        return out

    def _o2_field(self, D=1.0, k=1.0, s=0.2, source="uniform", n_iter=500, tol=1e-5):
        """Steady-state O2 on the deme grid (BioFVM-style), returned as hypoxia = 1 - O2.

        Solves D∇²O2 + supply·(1-O2) - k·density·O2 = 0 by Jacobi relaxation with zero-flux edges.
        O2 ∈ [0,1] (a weighted average of neighbours and the supply target 1), so hypoxia ∈ [0,1].

        `source` sets where O2 comes from:
          * ``"uniform"`` — supplied everywhere (a well-vascularised-tissue assumption). Gives a
            hypoxic core only when the tumour has an oxygenated (empty/normal) margin.
          * ``"perfused"`` — supplied ONLY by non-cancer (perfused stroma / empty) tissue:
            supply ∝ (1 - cancer density). A solid cancer mass or a cancer-filled duct then develops
            a hypoxic core from the O2 diffusion limit (comedonecrosis in DCIS) even with no empty
            margin, because the vasculature does not live inside the tumour.
        """
        # The metastatic deposit (met_grid_size > 0) is a second lattice appended after the primary
        # demes, so each lattice gets its own field; with the met off there is one block.
        dens_all = self._deme_density()
        perf_all = (np.clip(1.0 - self._emitter_density("cancer"), 0.0, 1.0)
                    if source == "perfused" else None)
        blocks = [(0, self.grid_size)]
        if getattr(self, "_met_enabled", False):
            blocks.append((self.n_primary_demes, self.met_grid_size))
        out = []
        for base, G in blocks:
            sl = slice(base, base + G * G)
            supply = s * perf_all[sl].reshape(G, G) if perf_all is not None else s
            out.append(self._o2_relax(dens_all[sl].reshape(G, G), supply, G, D, k, n_iter, tol))
        return np.concatenate(out)

    @staticmethod
    def _o2_relax(dens, supply, G, D, k, n_iter, tol):
        """Jacobi relaxation of one lattice's O2 field (see `_o2_field`); returns hypoxia, raveled."""
        nn = np.full((G, G), 4.0)
        nn[0, :] -= 1; nn[-1, :] -= 1; nn[:, 0] -= 1; nn[:, -1] -= 1
        O2 = np.ones((G, G))
        for _ in range(n_iter):
            nb = np.zeros((G, G))
            nb[1:, :] += O2[:-1, :]; nb[:-1, :] += O2[1:, :]
            nb[:, 1:] += O2[:, :-1]; nb[:, :-1] += O2[:, 1:]
            new = (D * nb + supply) / (D * nn + supply + k * dens)
            if np.max(np.abs(new - O2)) < tol:
                O2 = new
                break
            O2 = new
        return np.clip(1.0 - O2.ravel(), 0.0, 1.0)

    def _cci_field(self, emitter_type="immune", lengthscale=2.0, weight=None):
        """Per-deme ligand signal = neighbourhood-averaged emitter LIGAND availability (Gaussian
        `lengthscale`). ``weight`` (gid -> ligand level) weights each emitter by its ligand
        expression rather than its bare density; ``None`` -> the plain density field."""
        emit = self._emitter_density(emitter_type, weight=weight)
        coords = np.array(self.deme_coords, dtype=float)
        diff = coords[:, None, :] - coords[None, :, :]
        d2 = np.sum(diff * diff, axis=-1)
        L = max(float(lengthscale), 1e-9)
        W = np.exp(-d2 / (2.0 * L * L))
        wsum = W.sum(axis=1)
        return np.divide(W @ emit, wsum, out=np.zeros_like(emit), where=wsum > 0)

    def _microenv_deme_mod(self, exp_cache=None):
        """Microenvironment expression modifier: the HYPOXIA per-deme x gene matrix
        (n_demes x n_genes), or None.

        Hypoxia stays a genuine per-deme row: `mod[deme, hypoxia_genes] *= 1 + strength·hypoxia[deme]`.

        CCI is RECEPTOR-DEPENDENT, so it is NOT folded into this
        matrix — its multiplier is per-(deme, GENOTYPE) and is applied at materialisation. Here we
        compute the two inputs that multiplier needs and stash them on `self.microenv_truth`:
          * ``cci`` — ligand availability per deme: the smoothed emitter field, but each emitter
            weighted by its LIGAND expression (the wired pair's ligand gene) instead of a bare count;
          * ``cci_receptor_level`` — each genotype's expression of the wired RECEPTOR gene.
        The received signal on cell c is ``cci[deme(c)]·receptor_level[gid(c)]`` and the CCI
        multiplier on its target genes is ``1 + strength·received``.

        NORMALISATION (a definition, not a knob — getting it wrong silently rescales the calibrated
        ``strength``). The unweighted ``_cci_field`` is a density in [0,1] (a fraction of carrying
        capacity), which is what gives ``strength`` a stable meaning; weighting by RAW ligand
        expression and multiplying by RAW receptor expression breaks that bound. So each term is
        divided by its population mean: the ligand weight by the emitter cells' mean ligand
        expression (so ``cci`` reduces EXACTLY to the unweighted density field when ligand is uniform
        across genotypes), the receptor by ALL cells' mean receptor expression (so ``receptor_level``
        averages ~1). Both then average to ~1, so ``strength`` keeps the magnitude it has on the
        unweighted density field.
        """
        mp = self.microenv_params
        if not mp:
            return None
        hyp = (mp.get("hypoxia") or {}) if isinstance(mp, dict) else {}
        cci = (mp.get("cci") or {}) if isinstance(mp, dict) else {}
        n_demes = len(self.demes)
        mod = np.ones((n_demes, self.n_genes))
        hypoxia = np.zeros(n_demes)
        if len(self._hypoxia_genes) and float(hyp.get("strength", 0.0)) != 0.0:
            hypoxia = self._o2_field(D=float(hyp.get("o2_diffusion", 1.0)),
                                     k=float(hyp.get("o2_consumption", 1.0)),
                                     s=float(hyp.get("o2_supply", 0.2)),
                                     source=hyp.get("o2_source", "uniform"))
            mod[:, self._hypoxia_genes] *= (1.0 + float(hyp["strength"]) * hypoxia[:, None])

        # --- CCI: receptor-dependent channel (the wired pair) ---------------------------------
        cci_signal = np.zeros(n_demes)                    # per-deme ligand availability (normalised)
        receptor_level = {}
        cci_strength = float(cci.get("strength", 0.0))
        etype = cci.get("emitter_type", "immune")         # bound even when the channel is off
        # AMBIENT level of the ligand gene: the highest baseline any cell type has for it. The sender
        # is lifted ABOVE this rather than multiplied from its own baseline — otherwise whether the
        # channel is visible depends on the drawn gene's luck (a sender whose own baseline is far
        # below another type's stays unmarked even after a large boost). `strength` is the margin
        # over ambient.
        lig_ref = 1.0
        if self._cci_ligand is not None and self.celltype_exps:
            lig_ref = float(max(float(e[self._cci_ligand]) for e in self.celltype_exps.values()))
            lig_ref = lig_ref if lig_ref > 0 else 1.0
        cci_on = (len(self._cci_target_genes) and cci_strength != 0.0
                  and self._cci_ligand is not None and exp_cache)
        if cci_on:
            lig, rec = self._cci_ligand, self._cci_receptor
            lig_raw = {gid: float(exp_cache[gid][lig]) for gid in exp_cache}
            rec_raw = {gid: float(exp_cache[gid][rec]) for gid in exp_cache}
            # count-weighted population means over the FULL deme composition (unmaterialised
            # emitters still shape the field; genotypes we hold no expression for are skipped).
            lig_s = lig_n = rec_s = rec_n = 0.0
            for deme in self.demes:
                for gid, c in deme.items():
                    if gid not in lig_raw:
                        continue
                    rec_s += c * rec_raw[gid]; rec_n += c
                    if self.genotypes[gid].type == etype:
                        lig_s += c * lig_raw[gid]; lig_n += c
            mean_lig = (lig_s / lig_n) if lig_n > 0 else 1.0
            mean_rec = (rec_s / rec_n) if rec_n > 0 else 1.0
            mean_lig = mean_lig if mean_lig > 0 else 1.0
            mean_rec = mean_rec if mean_rec > 0 else 1.0
            weight = {gid: lig_raw[gid] / mean_lig for gid in lig_raw}
            receptor_level = {gid: rec_raw[gid] / mean_rec for gid in rec_raw}
            cci_signal = self._cci_field(etype, float(cci.get("lengthscale", 2.0)), weight=weight)

        self.microenv_truth = dict(
            hypoxia_genes=np.asarray(self._hypoxia_genes),
            cci_target_genes=np.asarray(self._cci_target_genes),
            hypoxia=hypoxia, cci=cci_signal,
            # Ground truth: the candidate L-R database, which pair is wired, and the
            # per-genotype receptor level that (with `cci`) makes the received signal per cell.
            cci_pairs=np.asarray(self._cci_pairs, dtype=int),
            cci_wired_pair=(0 if len(self._cci_pairs) else -1),
            cci_ligand=self._cci_ligand, cci_receptor=self._cci_receptor,
            cci_strength=cci_strength, cci_receptor_level=receptor_level,
            cci_emitter_type=etype, cci_ligand_ref=lig_ref)
        return mod

    def _hap_cns(self, gid):
        """Per-segment ``(cn_p, cn_m)`` allele-specific copy numbers of clone ``gid`` — the number of
        physical copies on each homolog, read straight off the genome copy lists (their lengths)."""
        g = self.genotypes[gid]
        return [(len(g.genome[s]['p']), len(g.genome[s]['m'])) for s in range(self.n_segments)]

    def _reconstruct_passengers(self, snv_mat, types):
        """Overlay a LINEAGE-CONSISTENT, allele-resolved neutral (passenger) SNV layer on the sampled
        cells.

        For each cancer clone ``Y`` whose subtree is a PROPER subset of the sample (a split between
        sampled cells) we place a set of STEM passengers "born on ``Y``'s founding edge"
        (count ≈ ``F_STEM``·``_pass_load[Y]``, floored so every split is markable, capped so the
        finite neutral-site budget spreads across splits). This layer is deliberately SUBCLONAL only:
        the clonal (truncal) burden of a tumour is carried by the founder genome itself (see
        ``founder_mutations``), so that it is present whether or not passengers are coarsened. A
        marker picks a neutral position (globally unique so clades stay homoplasy-free in the collapsed view), a
        homolog ∝ its copy number, and starts at multiplicity 1. Its multiplicity is then propagated
        down every descendant clone through the exact per-clone allele-specific CN events (WGD ×2;
        amplification +1 w.p. mult/cn; deletion −1 w.p. mult/cn, lost at 0 — the marginal of the
        engine's per-copy process). A materialised cell of clone ``Z`` therefore carries the stem
        passengers of ``Z`` AND all its ancestors (each at its propagated multiplicity) plus a private
        per-cell tail (``Poisson((1−F_STEM)·load/count)`` singletons). Cells in the same clade
        SHARE passenger SNVs, so a passenger-only tree recovers the clone-tree clades.

        Modifies ``snv_mat`` in place (the collapsed VAF = total variant copies / seg_cn) and RETURNS
        the allele-resolved ``(snv_p, snv_m)`` homolog-VAF frames (the primary truth;
        ``snv_p + snv_m == snv_mat``), or ``None`` when there is nothing to overlay (coarsening off /
        nothing folded / no neutral sites / no sampled cancer). Dedicated seeded rng -> reproducible."""
        if (not self.coarsen_passengers or not self._pass_load or not len(self._neutral_gene_ids)
                or snv_mat.shape[0] == 0):
            return None
        neutral = self._neutral_gene_ids
        # Keep the somatic overlay off the loci that are already spoken for: the patient's inherited
        # (germline) variants, and the mutations the founder cell carried when it transformed (which
        # are in every cancer cell and form the clonal peak). A reconstructed passenger never shares a
        # position with either, so a caller can tell the layers apart.
        for taken in (getattr(self, "germline_sites", None), getattr(self, "seeded_truncal_sites", None)):
            if taken is not None and len(taken):
                neutral = neutral[~np.isin(neutral, taken)]
        seg_of = self._gene_segment
        n_neutral = len(neutral)
        if not n_neutral:
            return None

        # ---- materialised cancer clones + their ancestor closure (the sampled clone sub-tree) ------
        per_clone_cells = Counter(types)
        mat_cancer = [g for g in per_clone_cells if self._is_cancer(g)]
        if not mat_cancer:
            return None
        parents = self.genotypes_parents
        closure = set()
        for g in mat_cancer:
            cur = g
            while cur is not None and cur not in closure:
                closure.add(cur)
                cur = parents.get(cur)
        # children within the closure + roots (founder, or any closure node whose parent is outside)
        children = {g: [] for g in closure}
        roots = []
        for g in closure:
            p = parents.get(g)
            if p in closure:
                children[p].append(g)
            else:
                roots.append(g)
        for g in closure:                       # deterministic child order (by birth ordinal)
            children[g].sort(key=lambda c: self.genotypes[c].ord)
        roots.sort(key=lambda c: self.genotypes[c].ord)

        hap_cn = {g: self._hap_cns(g) for g in closure}   # gid -> [(cn_p, cn_m), ...] per segment
        seg_cn = {g: [cp + cm for (cp, cm) in hap_cn[g]] for g in closure}

        # Pre-order (root-first) traversal of the closure, built iteratively so tree DEPTH never hits
        # the recursion limit (a long linear driver chain can be deep). Reused for subtree sums and
        # multiplicity descent below.
        order = []
        stack = list(reversed(roots))
        while stack:
            g = stack.pop()
            order.append(g)
            stack.extend(reversed(children[g]))

        # subtree sampled-cell counts (a clone whose subtree is ALL sampled cancer cells marks no
        # observable split, so it is skipped to save the site budget for informative splits).
        total_cancer = sum(per_clone_cells[g] for g in mat_cancer)
        subtree = {g: per_clone_cells.get(g, 0) for g in closure}
        for g in reversed(order):                 # children precede parents in reverse pre-order
            p = parents.get(g)
            if p in subtree:
                subtree[p] += subtree[g]

        prng = self._run_rng(self.seed, PASSENGER_RNG_OFFSET, _STREAM_PASSENGERS)

        # ---- allocate stem markers to clones, drawing globally-unique neutral positions -----------
        # Position-level global uniqueness (one stem SNV per neutral position across the whole tree)
        # keeps the collapsed clades homoplasy-free; homolog is still resolved per SNV. A shuffled
        # position order + pointer makes the draw O(1) and deterministic. Informative splits are
        # marked first (by how balanced the split is) so a tight budget covers the big clades.
        pos_order = prng.permutation(neutral)
        pos_ptr = 0
        f_stem = F_STEM_DEFAULT
        born = {g: [] for g in closure}          # gid -> list of (pos, hom, seg)

        def _mark(g, n_markers, limit):
            """Place ``n_markers`` markers born on clone ``g``: an unused neutral position each (taken
            from the shuffled order, stopping at ``limit``), on a homolog drawn ∝ its copy number."""
            nonlocal pos_ptr
            cn_g = hap_cn[g]
            for _ in range(int(n_markers)):
                if pos_ptr >= limit:
                    return
                pos = int(pos_order[pos_ptr]); pos_ptr += 1
                s = int(seg_of[pos])
                cp, cm = cn_g[s]
                if cp + cm == 0:
                    continue                     # nullisomic segment: no copy to sit on, skip
                if cp == 0:
                    hom = 'm'
                elif cm == 0:
                    hom = 'p'
                else:
                    hom = 'p' if prng.random() < cp / (cp + cm) else 'm'
                born[g].append((pos, hom, s))

        # ---- stems: one budget per clone-split that separates sampled cells ------------------------
        # A clone whose subtree is EVERY sampled cancer cell marks no observable split, so it is
        # skipped and its sites stay available for the splits that do carry information. The clonal
        # layer of a real tumour is not built here at all — it is inherited from the founder genome
        # (``founder_mutations``), which is why it survives with coarsening switched off.
        # Total order (final `ord` tiebreaker) so allocation is independent of set-iteration order
        # (string hashing is per-process randomised) -> reproducible.
        informative = [g for g in closure if 0 < subtree[g] < total_cancer]
        informative.sort(key=lambda g: (min(subtree[g], total_cancer - subtree[g]),
                                        self._pass_load.get(g, 0), self.genotypes[g].ord),
                         reverse=True)
        for g in informative:
            if pos_ptr >= n_neutral:
                break
            load = self._pass_load.get(g, 0)
            _mark(g, min(max(int(round(f_stem * load)), STEM_FLOOR), STEM_CAP), n_neutral)

        # ---- propagate stem multiplicity down the closure, record per-clone stem genome ----------
        # A stem SNV is (pos, hom, seg, mult). At each clone we hold the surviving inherited SNVs +
        # those born there (mult 1); each child edge applies that edge's exact CN event to every
        # active SNV on the affected (seg, hom). clone_stem[Z] is the shared stem layer of clone Z.
        clone_stem = {}

        def _edge_event(parent_g, child_g):
            """The CN event on edge parent->child as ``('wgd',)`` or ``('cn', changes)`` where changes
            is a list of ``(seg, hom, delta, cn_before)``. Each registered child is ONE mutate() = one
            event: a CNV changes exactly one (seg, hom) by ±1, a WGD doubles EVERY homolog (so it
            changes many), an SNV changes none. Reading it off the exact per-homolog copy numbers, more
            than one changed (seg, hom) therefore uniquely identifies a WGD — no reliance on the
            monotone ``is_wgd`` flag (which cannot flag a repeat WGD)."""
            cp, cc = hap_cn[parent_g], hap_cn[child_g]
            changes = []
            for s in range(self.n_segments):
                dp = cc[s][0] - cp[s][0]
                dm = cc[s][1] - cp[s][1]
                if dp:
                    changes.append((s, 'p', dp, cp[s][0]))
                if dm:
                    changes.append((s, 'm', dm, cp[s][1]))
            if len(changes) > 1:
                return ('wgd',)
            return ('cn', changes)

        def _apply(active, event):
            """Return the child's active stem SNVs after applying `event` (mult propagation)."""
            ev = event[0]
            out = []
            if ev == 'wgd':
                for (pos, hom, s, mult) in active:
                    out.append((pos, hom, s, mult * 2))
                return out
            changes = event[1]
            for (pos, hom, s, mult) in active:
                m = mult
                for (cs, chom, delta, cn_before) in changes:
                    if cs != s or chom != hom:
                        continue
                    step = 1 if delta > 0 else -1
                    cn = cn_before
                    for _ in range(abs(delta)):
                        if cn <= 0:
                            break
                        if step > 0:             # amplification: duplicated copy is a mutant w.p. m/cn
                            if prng.random() < m / cn:
                                m += 1
                            cn += 1
                        else:                    # deletion: removed copy is a mutant w.p. m/cn
                            if prng.random() < m / cn:
                                m -= 1
                            cn -= 1
                if m > 0:
                    out.append((pos, hom, s, m))
            return out

        active_in = {g: [] for g in roots}        # stem SNVs entering each clone from its parent
        for g in order:                           # pre-order: a parent is processed before its children
            active = active_in.pop(g, [])
            for (pos, hom, s) in born[g]:
                active.append((pos, hom, s, 1))
            if g in per_clone_cells:
                clone_stem[g] = active
            for c in children[g]:
                active_in[c] = _apply(active, _edge_event(g, c))

        # ---- assemble per-clone overlay vectors, then per-cell rows (base alleles + stem + private) -
        snv_p = np.zeros_like(snv_mat)
        snv_m = np.zeros_like(snv_mat)
        base_p, base_m = {}, {}          # per-clone allele-resolved base (drivers + genome passengers)
        stem_p, stem_m = {}, {}          # per-clone stem overlay (homolog VAF added on top of base)
        # per-clone allele fraction CAPS: the most a homolog can contribute at a locus is cn_h/seg_cn
        # (all its copies mutated). Enforced at the end so a collision (an overlay SNV landing where the
        # base genome already mutated the same homolog+position) merges to "all copies mutated" instead
        # of over-counting past the physical copy number -> guarantees snv_p+snv_m == snv_mat and VAF<=1.
        seg_of_arr = np.asarray(seg_of)
        capfrac_p, capfrac_m = {}, {}
        for g in per_clone_cells:
            rep = self.genotypes[g]
            vp, vm = rep.get_snvs_alleles()
            base_p[g], base_m[g] = vp, vm
            sp = np.zeros_like(vp); sm = np.zeros_like(vm)
            if g in clone_stem:
                scn = seg_cn[g]
                for (pos, hom, s, mult) in clone_stem[g]:
                    cn = scn[s]
                    if cn > 0:
                        (sp if hom == 'p' else sm)[pos] += mult / cn
            stem_p[g], stem_m[g] = sp, sm
            hp = hap_cn.get(g) or self._hap_cns(g)   # cancer closure cached; normals computed here
            cp_seg = np.asarray([cp for (cp, cm) in hp], dtype=float)
            cm_seg = np.asarray([cm for (cp, cm) in hp], dtype=float)
            cn_seg = cp_seg + cm_seg
            denom = np.maximum(cn_seg[seg_of_arr], 1e-9)
            nz = cn_seg[seg_of_arr] > 0
            capfrac_p[g] = np.where(nz, cp_seg[seg_of_arr] / denom, 0.0)
            capfrac_m[g] = np.where(nz, cm_seg[seg_of_arr] / denom, 0.0)

        for i, g in enumerate(types):
            snv_p[i] = base_p[g] + stem_p[g]
            snv_m[i] = base_m[g] + stem_m[g]

        # private per-cell singletons (Poisson((1-f_stem)*load/count)); NOT inherited, so drawn fresh
        # per cell on sites unused by that cell (per-cell infinite sites).
        for i, g in enumerate(types):
            if g not in per_clone_cells or g not in self._pass_load or not self._is_cancer(g):
                continue
            cnt = self.genotypes_counts.get(g, 0)
            if not cnt:
                continue
            mu = (1.0 - f_stem) * self._pass_load[g] / cnt
            if mu <= 0:
                continue
            k = int(prng.poisson(mu))
            if k <= 0:
                continue
            used = set(np.flatnonzero(snv_p[i] + snv_m[i]).tolist())
            pool = neutral if not used else neutral[~np.isin(neutral, list(used))]
            if not len(pool):
                continue
            picks = prng.choice(pool, size=min(k, len(pool)), replace=False)
            scn = seg_cn.get(g)
            cn_g = hap_cn.get(g)
            for pos in picks:
                pos = int(pos); s = int(seg_of[pos])
                cn = scn[s]
                if cn <= 0:
                    continue
                cp, cm = cn_g[s]
                if cp + cm == 0:
                    continue
                if cp == 0:
                    hom_p = False
                elif cm == 0:
                    hom_p = True
                else:
                    hom_p = prng.random() < cp / (cp + cm)
                inc = 1.0 / cn
                if hom_p:
                    snv_p[i, pos] += inc
                else:
                    snv_m[i, pos] += inc

        # Clip each homolog to its physical fraction, then DERIVE the collapsed frame from the alleles
        # (single source of truth) so snv_p + snv_m == snv_mat exactly and every VAF is in [0, 1].
        for i, g in enumerate(types):
            np.minimum(snv_p[i], capfrac_p[g], out=snv_p[i])
            np.minimum(snv_m[i], capfrac_m[g], out=snv_m[i])
            snv_mat[i] = snv_p[i] + snv_m[i]
        return snv_p, snv_m

    def _materialize_plan(self, max_cells, region=None, depth_frac=None):
        """Per-deme ``{gid: n_to_materialise}`` for ``make_cell_data``.

        With no cap (or a tumour under it) every cell in scope is materialised (``n = count``).
        Above the cap, each (deme, genotype) bucket keeps ``Binomial(count,
        cap/total)`` cells — a representative subsample preserving spatial + clonal + cell-type
        proportions in expectation, the biopsy an assay actually takes — so the dense per-cell frames
        never exceed ~``max_cells`` rows regardless of tumour size. Deterministic (dedicated seeded
        rng).

        ``region`` (an iterable of deme indices) restricts materialisation to those demes, at FULL
        local density unless the region itself exceeds the cap. This is an IN-PLANE cut: it selects a
        2-D patch of the tissue (e.g. one part of a resected specimen). Non-region demes materialise
        nothing.

        ``depth_frac`` is the orthogonal DEPTH cut (a "parallel-axis" cut): each deme keeps only
        ``Binomial(count, depth_frac)`` of its cells, so the 2-D structure is left intact but every
        deme-column's occupancy drops. Because a deme's ``K`` stands for the cells in its 3-D column,
        this is exactly cutting through the depth — a thin histology/Visium SLICE of a tissue block
        keeps the whole 2-D field but only ~one layer of cells. Applied once up-front (dedicated seeded
        rng) so it composes deterministically with ``region`` and ``max_cells``."""
        reg = None if region is None else set(int(d) for d in region)

        # DEPTH cut: thin every deme's column to a fraction of its cells (a 2-D slice), up-front.
        if depth_frac is not None:
            drng = self._run_rng(self.seed, 20240731, _STREAM_DEPTH)
            src = []
            for deme in self.demes:
                out = {}
                for g, c in deme.items():
                    n = int(drng.binomial(c, depth_frac))
                    if n:
                        out[g] = n
                src.append(out)
        else:
            src = self.demes

        if depth_frac is None and reg is None:
            total = self.get_tumor_size()
        else:
            scope = range(len(src)) if reg is None else [d for d in reg if 0 <= d < len(src)]
            total = sum(sum(src[d].values()) for d in scope)
        subsample = max_cells is not None and total > max_cells and total > 0
        if not subsample and reg is None and depth_frac is None:
            return [dict(d) for d in self.demes], False, total
        frac = (max_cells / total) if subsample else 1.0
        mrng = self._run_rng(self.seed, 20240730, _STREAM_SUBSAMPLE)
        plan = []
        for di, deme in enumerate(src):
            if reg is not None and di not in reg:
                plan.append({})
                continue
            if not subsample:
                plan.append(dict(deme))
                continue
            keep = {}
            for gid, c in deme.items():
                n = int(mrng.binomial(c, frac))
                if n:
                    keep[gid] = n
            plan.append(keep)
        return plan, subsample, total

    def truncal_sites(self, ccf=0.95, cell_ids=None):
        """Locus indices of the tumour's **trunk** — the somatic variants essentially every cancer
        cell carries — MEASURED from the cells rather than looked up.

        The founder starts with a blank genome, so a tumour's clonal layer is not planted: it fixes
        in the founding duct before the lesion spreads. There is nothing to look up, and
        :attr:`seeded_truncal_sites` is empty unless ``founder_mutations`` was configured. This finds
        the trunk the way a study does — the somatic sites carried by at least ``ccf`` of cancer
        cells. Germline variants are excluded: they sit in every cell, tumour and normal alike, so
        they would otherwise pass the same test.

        Parameters
        ----------
        ccf : float, default 0.95
            Minimum cancer-cell fraction for a site to count as truncal. 0.95 is the usual clonal
            cutoff. The count is sensitive to it — a convention, not a constant of the tumour — so
            report the cutoff with the number.
        cell_ids : sequence of str, optional
            Restrict to these cells (e.g. one sample's). Defaults to every materialised cell.

        Returns
        -------
        numpy.ndarray
            Sorted locus indices, i.e. positions along the gene axis of ``cell_data["cell_snv"]``.
        """
        if self.cell_data is None:
            raise ValueError("cell_data is not materialised; call make_cell_data() first")
        if getattr(self, "_cell_data_stale", False):
            raise ValueError("cell_data is stale (grown with materialize=False); "
                             "call ensure_cell_data() first")
        snv = self.cell_data["cell_snv"]
        gids = self.cell_data["cell_type"].iloc[:, 0]
        if cell_ids is not None:
            snv, gids = snv.loc[list(cell_ids)], gids.loc[list(cell_ids)]
        is_cancer = np.array([self._is_cancer(g) for g in gids], dtype=bool)
        if not is_cancer.any():
            return np.array([], dtype=int)
        carried = (snv.values[is_cancer] > 0).mean(axis=0)
        germline = np.zeros(snv.shape[1], dtype=bool)
        sites = getattr(self, "germline_sites", None)
        if sites is not None and np.size(sites):
            germline[np.asarray(sites, dtype=int)] = True
        return np.flatnonzero((carried >= ccf) & ~germline)

    def primary_window(self, side, center=None):
        """Deme indices of a rectangular window of the PRIMARY grid, centred on ``center``
        ``(row, col)`` or the grid centre. Pass to ``make_cell_data(region=...)`` to materialise a
        dense spatial patch (e.g. a Visium slide's footprint) of a cm-scale tumour at real cell
        density, rather than thinning the whole tumour to a uniform subsample.

        Parameters
        ----------
        side : int or tuple of int
            Window size in demes: an int for a ``side``×``side`` square, or ``(rows, cols)`` for a
            rectangle. A real capture area is rarely square — the Visium v1 slide is 78×64 spots,
            i.e. wider than it is tall — so a rectangle is what fills one.
        center : tuple of int, optional
            ``(row, col)`` centre. Defaults to the grid centre. A window that would overhang the
            grid is SHIFTED back onto it, so the full requested size is always returned (clipped
            only if it is larger than the grid) — size a window to a capture area and you get that
            area, not a sliver of it.
        """
        G = self.grid_size
        n_r, n_c = (side, side) if np.isscalar(side) else side
        n_r, n_c = min(int(n_r), G), min(int(n_c), G)
        cr, cc = center if center is not None else (G // 2, G // 2)
        r0 = min(max(0, int(cr) - n_r // 2), G - n_r)
        c0 = min(max(0, int(cc) - n_c // 2), G - n_c)
        return [r * G + c for r in range(r0, r0 + n_r) for c in range(c0, c0 + n_c)]

    def make_cell_data(self, cell_prefix="C", max_cells=None, region=None, depth_frac=None,
                       layers=None, **kwargs):
        """Expand the per-deme genotype counts into per-cell ground-truth tables.

        Materialises one row per cell (each genotype's count becomes that many identical
        cells) into ``self.cell_data`` — a dict of dataframes keyed by cell name:
        evolutionary parameters, expression, SNV and CNV matrices, grid coordinates,
        cell-type and deme labels, plus any optional layers that are enabled (compartment,
        allele-resolved expression / RNA BAF, program activity). Called automatically at the
        end of ``grow``.

        Parameters
        ----------
        cell_prefix : str, optional
            Prefix for the generated cell names (default ``"C"``).
        max_cells : int, optional
            Cap on the number of cells materialised (defaults to the tumour's ``self.max_cells``).
            When the tumour is larger, a representative subsample is materialised instead of every
            cell (see ``_materialize_plan``) so the dense frames stay memory-bounded at cm-scale.
            ``None`` materialises every cell.
        region : iterable of int, optional
            Deme indices to restrict materialisation to, at FULL local density (subject to
            ``max_cells``). Use this for a SPATIAL assay (e.g. a Visium slide) that samples a small
            window of a cm-scale tumour and needs that window at real cell density — a uniform
            whole-tumour subsample would thin it to ~1 cell/spot. ``None`` materialises the whole
            tumour.
        depth_frac : float, optional
            DEPTH cut (a "parallel-axis" cut): keep only this fraction of EACH deme's cells, leaving
            the 2-D structure intact but lowering every deme-column's occupancy. Because ``K`` stands
            for the deme's 3-D column, this is a thin histology/Visium SLICE of a tissue block — the
            whole 2-D field, one thin layer. ``None`` keeps the full column; ``0.5`` is a half-depth
            slice. Composes with ``region`` (an in-plane cut) and ``max_cells``.
        layers : str or iterable of str, optional
            Which gene-wide frames (cells × genes) to build: names from ``CELL_LAYERS`` (``"cell_exp"``,
            ``"cell_snv"``, ``"cell_cnv"``, ...) and/or assay names from ``ASSAY_LAYERS``
            (``"visium"``, ``"scrna"``, ``"scdna"``, ``"imaging"``), which stand for the frames that
            assay reads. These frames dominate memory, so a Visium section that needs only
            expression is several times smaller with ``layers="visium"``. The small per-cell frames
            (coordinates, types, demes, evolutionary parameters, and the optional truth layers) are
            always built. ``None`` (default) builds every frame.

        Returns
        -------
        dict
            ``self.cell_data`` — the per-cell dataframes.
        """
        if max_cells is None:
            max_cells = self.max_cells
        mat_plan, subsampled, want = self._materialize_plan(max_cells, region=region,
                                                            depth_frac=depth_frac)
        # Only for an explicit SECTION (region / depth_frac). A plain whole-tumour materialisation
        # being capped is the documented representative-biopsy behaviour, and warning on it would
        # fire on every cm-scale grow(). The trap is asking for a physical section and silently
        # getting a memory budget instead: the cap, not depth_frac, then sets the density, and a
        # Visium section capped this way reads ~1 cell/spot — a budget artefact that looks like a
        # modelling result. Note `max_cells=None` means "the tumour's own cap", NOT "uncapped".
        if subsampled and (region is not None or depth_frac is not None):
            warnings.warn(
                f"max_cells={max_cells} capped this section: {want:,} cells were selected by "
                f"region/depth_frac but only ~{max_cells:,} are materialised ({max_cells / want:.1%}). "
                f"The cap, not depth_frac, is setting the cell density. Pass a larger max_cells "
                f"(~{int(want * 1.2):,}) to let the physical section govern.",
                stacklevel=2)
        layers = _resolve_layers(layers)
        gene_names = self.selection.get_gene_names()
        onc_idx, tsg_idx = self.selection.get_oncogenes(), self.selection.get_tsgs()
        disp_idx, ir_idx, tr_idx = (self.selection.get_dispersal_genes(),
                                    self.selection.get_immune_resistant(),
                                    self.selection.get_treatment_resistant())
        breach_idx, ss_idx = self.selection.get_breach(), self.selection.get_stromal_survival()
        ms_idx = self.selection.get_met_survival()
        dt_idx = self.selection.get_drug_tolerance()
        P = self.programs
        # Genotypes that will be materialised — the whole plan without a cap, the subsample's
        # genotypes under one. At cm-scale iterating every genotype here is itself an
        # O(#genotypes × n_genes) memory/time wall, so the cap must bound it too.
        mat_gids = set()
        for keep in mat_plan:
            mat_gids.update(keep)

        # MEMORY. A section of a cm-scale tumour holds nearly as many genotypes as cells, so caching
        # every genotype's gene-wide rows (SNV, copy number, expression and its allele split) up front
        # costs as much as the cell tables themselves, and stacking per-cell rows in float64 before
        # the final cast costs several times more. Only the SMALL per-genotype quantities (the
        # evolutionary parameters with their driver tallies, and the program drive) are kept for all
        # genotypes; the gene-wide rows are built per chunk of cells and written straight into the
        # requested frames. Every per-genotype function here is deterministic, so a genotype seen
        # again in a later chunk gives the same row, and the frames equal stacking every row at once.
        def genotype_exp(gid):
            """(exp, exp_p, exp_m) of genotype ``gid``; the allele split is None without programs."""
            rep = self.genotypes[gid]
            if P is None:
                # no program layer: cancer reads `get_exp`, normals their type baseline
                if rep.type == "cancer":
                    rep.baseline_exp = self.celltype_exps["cancer"]
                    return rep.get_exp(self.selection.mut_effects), None, None
                return self.celltype_exps[rep.type], None, None
            rep.baseline_exp = self.celltype_exps[rep.type]
            if rep.type == "cancer":
                ep, em = rep.get_exp_alleles(dosage_sensitivity=P.dosage_sensitivity,
                                             snv_exp_effect=P.snv_exp_effect,
                                             dosage_saturation=P.dosage_saturation,
                                             dosage_law=getattr(P, "dosage_law", "haplotype"))
            else:
                # Normal cells are diploid and unmutated: a flat allele split.
                half = self.celltype_exps[rep.type] / 2.0
                ep, em = half.copy(), half.copy()
            return ep + em, ep, em

        evo_cache, drive_cache = {}, {}
        for gid in mat_gids:
            rep = self.genotypes[gid]
            snv = rep.get_snvs()
            if P is None:
                if rep.type == "cancer":
                    rep.baseline_exp = self.celltype_exps["cancer"]
            else:
                rep.baseline_exp = self.celltype_exps[rep.type]
                # The clone drive reads the EVOLVED per-clone phenotype, so whatever moved fitness —
                # CINner drivers, and the epistasis multiplier — propagates into the programs by
                # construction; it also reads the clone's SNV VAFs. Normal cells carry no drive.
                drive = (P.clone_drive(rep.evolutionary_parameters, rep.baseline_rates, snv)
                         if rep.type == "cancer" else np.zeros(P.n_programs))
                drive_cache[gid] = drive + P.celltype_bias(rep.type)
            # per-genotype evolutionary parameters + unique-mutated-driver tallies
            evo = dict(rep.evolutionary_parameters)
            evo["n_mut_onc"] = int((snv[onc_idx] > 0).sum())
            evo["n_mut_tsg"] = int((snv[tsg_idx] > 0).sum())
            evo["n_mut_disp"] = int((snv[disp_idx] > 0).sum())
            evo["n_mut_ir"] = int((snv[ir_idx] > 0).sum())
            evo["n_mut_tr"] = int((snv[tr_idx] > 0).sum())
            evo["n_mut_breach"] = int((snv[breach_idx] > 0).sum())
            evo["n_mut_ss"] = int((snv[ss_idx] > 0).sum())
            evo["n_mut_ms"] = int((snv[ms_idx] > 0).sum())
            evo["n_mut_dt"] = int((snv[dt_idx] > 0).sum())
            evo_cache[gid] = evo

        # Microenvironment: per-deme expression modifier (None -> layer off). The CCI ligand/receptor
        # levels are the only part of each genotype's expression it reads, so only those two genes
        # are kept per genotype.
        lig, rec = self._cci_ligand, self._cci_receptor
        cci_levels = ({} if lig is None or rec is None else
                      {g: {lig: e[lig], rec: e[rec]} for g in mat_gids for e in (genotype_exp(g)[0],)})
        deme_mod = self._microenv_deme_mod(cci_levels)
        # CCI is receptor-dependent, so its multiplier is per-(deme, genotype), not a column of
        # `deme_mod` (which carries hypoxia only). Extract the pieces the loop needs: the target
        # genes, the calibrated strength, the per-deme ligand availability and the per-genotype
        # receptor level. `None` when no CCI channel is wired -> the loop applies hypoxia alone.
        cci_apply = None
        if deme_mod is not None and self.microenv_truth.get("cci_ligand") is not None:
            _ct = np.asarray(self.microenv_truth["cci_target_genes"], dtype=int)
            _cs = float(self.microenv_truth["cci_strength"])
            _rl = self.microenv_truth["cci_receptor_level"]
            if _ct.size and _cs != 0.0 and _rl:
                cci_apply = (_ct, _cs, self.microenv_truth["cci"], _rl,
                             int(self.microenv_truth["cci_ligand"]),
                             int(self.microenv_truth["cci_receptor"]),
                             self.microenv_truth["cci_emitter_type"],
                             float(self.microenv_truth.get("cci_ligand_ref", 1.0)))

        # Niche -> program: the microenvironment fields drive per-deme program activity, generalising
        # the hard-wired hypoxia/CCI gene sets (which still apply via `deme_mod`; the two compose).
        # The COMPARTMENT (epithelial / stromal live fraction) is a niche field too: mapping it to the
        # seeded invasive/emt program in `niche_program_map` makes the SAME clone express the invasive
        # program more at the epithelial front than in the stroma — the genetic-vs-niche confound.
        # `niche_drive` is a zeros matrix (or None) unless `niche_program_map` names a field.
        niche_drive = None
        if P is not None:
            fields = {}
            # `deme_mod is not None` means the microenvironment layer is on (microenv_truth then
            # carries hypoxia/cci).
            if deme_mod is not None:
                fields["hypoxia"] = self.microenv_truth["hypoxia"]
                fields["cci"] = self.microenv_truth["cci"]
            if self.structure_radius > 0:
                epi_frac, stro_frac = self._compartment_fields()
                fields["epithelial"] = epi_frac
                fields["stromal"] = stro_frac
                # Record the compartment field as ground truth (create microenv_truth if the
                # microenvironment layer is off).
                if getattr(self, "microenv_truth", None) is None:
                    self.microenv_truth = {}
                self.microenv_truth["epithelial"] = epi_frac
                self.microenv_truth["stromal"] = stro_frac
            if fields:
                niche_drive = P.niche_drive(fields)

        # The cell plan: one run per (deme, genotype), each `count` identical cells from `start` on.
        runs, rows_evo, crd, types, demes_col, names = [], [], [], [], [], []
        i = 0
        for deme_idx, keep in enumerate(mat_plan):
            if not keep:
                continue
            r, c = self.deme_coords[deme_idx]
            for gid in sorted(keep.keys(), key=lambda g: self.genotypes[g].ord):
                runs.append((deme_idx, gid, i, keep[gid]))
                for _ in range(keep[gid]):
                    rows_evo.append(evo_cache[gid])
                    crd.append((r, c)); types.append(gid); demes_col.append(deme_idx)
                    names.append(f"{cell_prefix}{i}"); i += 1
        n_cells, n_genes = i, self.n_genes

        # The per-CELL program activity. `z` is per-cell (cycling is a cell state, not a clone
        # constant) while the genotype drive is per-clone — so the drive sets each clone's MEAN and
        # `activity_noise` supplies the within-clone spread. Drawn for all cells at once, so the stream
        # does not depend on how the expression below is chunked.
        Z = None
        if P is not None and n_cells:
            drive = np.empty((n_cells, P.n_programs))
            for deme_idx, gid, start, count in runs:
                drive_row = drive_cache[gid]
                if niche_drive is not None:
                    drive_row = drive_row + niche_drive[deme_idx]
                drive[start:start + count] = drive_row
            Z = P.sample_z(drive, P.activity_rng())

        # The gene-wide frames, allocated at their final dtype and filled chunk by chunk.
        # DTYPES. These frames are (n_cells x n_genes) and dominate the memory at cm-scale, so they
        # are stored no wider than the values need. Copy number is a small non-negative integer
        # (int16 covers 0..32767 against a realistic ceiling of ~10): LOSSLESS at a quarter of int64.
        # Expression is float32 — a level, nowhere near float64's 15 significant digits.
        # `cell_snv` STAYS float64 on purpose. Several suites pin a golden md5 of its raw bytes to
        # assert that turning a feature off leaves the growth stream byte-identical; narrowing it
        # would force those baselines to be regenerated, and a re-blessed baseline is exactly how a
        # real perturbation would slip past them.
        allele = P is not None and P.allele_specific
        want_snv = bool(layers & {"cell_snv", "cell_snv_p", "cell_snv_m", "cell_rna_vaf"})
        snv_mat = np.empty((n_cells, n_genes)) if want_snv else None
        cnv_mat = np.empty((n_cells, n_genes), dtype=np.int16) if "cell_cnv" in layers else None
        exp_mat = np.empty((n_cells, n_genes), dtype=np.float32) if "cell_exp" in layers else None
        ep_mat = np.empty((n_cells, n_genes)) if allele and "cell_exp_p" in layers else None
        em_mat = np.empty((n_cells, n_genes)) if allele and "cell_exp_m" in layers else None
        baf_mat = np.empty((n_cells, n_genes)) if allele and "cell_rna_baf" in layers else None
        want_exp = any(m is not None for m in (exp_mat, ep_mat, em_mat, baf_mat))
        # The program multiplier exp(z · loading) is built per chunk of cells, like every gene-wide
        # frame, summing over programs in a fixed order so a value does not depend on its chunk. A
        # large section then never holds a cells x genes float64 table beyond the frames returned.
        step = max(_MATERIALISE_MIN_ROWS, _MATERIALISE_CHUNK_BYTES // (8 * max(n_genes, 1)))
        first = 0                                       # first run that can overlap the chunk
        for s in range(0, n_cells, step):
            e = min(n_cells, s + step)
            rows = {}                                   # this chunk's per-genotype rows
            if want_exp:
                buf = np.empty((e - s, n_genes))        # float64, as the stacked rows were
                if P is not None:
                    buf_m = np.empty((e - s, n_genes))
            while runs[first][2] + runs[first][3] <= s:
                first += 1
            for deme_idx, gid, start, count in runs[first:]:
                if start >= e:
                    break
                lo, hi = max(s, start), min(e, start + count)
                if hi <= lo:
                    continue
                rep = self.genotypes[gid]
                # A genotype spans many demes, so its rows are built once per chunk, not once per run.
                got = rows.get(gid)
                if got is None:
                    got = rows[gid] = dict(
                        snv=rep.get_snvs() if want_snv else None,
                        cnv=rep.get_cnvs() if cnv_mat is not None else None,
                        exp=genotype_exp(gid) if want_exp else None)
                if want_snv:
                    snv_mat[lo:hi] = got["snv"]
                if cnv_mat is not None:
                    cnv_mat[lo:hi] = got["cnv"]
                if not want_exp:
                    continue
                exp_g, ep_g, em_g = got["exp"]
                # The receptor-dependent CCI multiplier for this (deme, genotype) — the same
                # scalar for every cell of the clone here (ligand availability at the deme × the
                # clone's own receptor level). `deme_mod` carries hypoxia; this carries CCI.
                cci_f = None
                if cci_apply is not None:
                    _ct, _cs, _la, _rl, _lig, _rec, _etype, _lref = cci_apply
                    cci_f = 1.0 + _cs * float(_la[deme_idx]) * float(_rl.get(gid, 0.0))
                if P is None:
                    if deme_mod is None:
                        exp_row = exp_g
                    else:
                        exp_row = exp_g * deme_mod[deme_idx]           # fresh array (hypoxia)
                        if cci_f is not None:
                            exp_row[_ct] *= cci_f                      # receptor-dependent CCI
                            # THE L-R SIGNAL ITSELF, split by WHO SENDS AND WHO RECEIVES. Every CCI
                            # tool scores a pair from the ligand's and receptor's own expression, and
                            # CellChat's gene filter is a DIFFERENTIAL-EXPRESSION test BETWEEN CELL
                            # GROUPS — so a deme-wide boost is invisible to it (it lifts every group
                            # in the niche together and leaves the group contrast flat). Put the
                            # ligand on the EMITTER cells and the receptor on everyone else, both
                            # scaled by the local field: that is the canonical "ligand-high group
                            # adjacent to receptor-high group" pattern these tools are built to find,
                            # and it is what a sender/receiver pair actually looks like in tissue.
                            # The LIGAND is a marker of the SENDER TYPE: a sender expresses it
                            # because of what it is, not because of how crowded its deme is (density
                            # governs how much signal REACHES receivers, which is what `_la` already
                            # does for the downstream targets). A flat elevation makes it a clean
                            # cell-type marker, which is the between-group contrast the tools' DE
                            # filter looks for; scaling it by the local density instead washes it out
                            # to ~1.8x and the pair is rejected on the ligand.
                            # The RECEPTOR is up-regulated where signal ARRIVES, so it stays graded
                            # by the local field.
                            if rep.type == _etype:
                                exp_row[_lig] = _lref * (1.0 + _cs)   # sender clears ambient by margin
                                exp_row[_rec] /= (1.0 + _cs)          # and is not listening to itself
                            else:
                                exp_row[_rec] *= (1.0 + _cs)
                    buf[lo - s:hi - s] = exp_row
                else:
                    mod_row = 1.0 if deme_mod is None else deme_mod[deme_idx]
                    ep_row, em_row = ep_g * mod_row, em_g * mod_row
                    if cci_f is not None:                              # keep alleles in step
                        ep_row[_ct] *= cci_f; em_row[_ct] *= cci_f
                        # The L-R signal, as in the no-program path above, split across the alleles.
                        if rep.type == _etype:
                            half = 0.5 * _lref * (1.0 + _cs)
                            ep_row[_lig] = em_row[_lig] = half
                            ep_row[_rec] /= (1.0 + _cs); em_row[_rec] /= (1.0 + _cs)
                        else:
                            ep_row[_rec] *= (1.0 + _cs); em_row[_rec] *= (1.0 + _cs)
                    buf[lo - s:hi - s] = ep_row
                    buf_m[lo - s:hi - s] = em_row
            if not want_exp:
                continue
            if P is None:
                exp_mat[s:e] = buf
                continue
            # The per-cell program activity scales both alleles; total expression is their sum.
            prog_mult = _program_multiplier(P, Z[s:e])        # (chunk cells, n_genes)
            buf *= prog_mult
            buf_m *= prog_mult
            tot = buf + buf_m
            if exp_mat is not None:
                exp_mat[s:e] = tot
            if ep_mat is not None:
                ep_mat[s:e] = buf
            if em_mat is not None:
                em_mat[s:e] = buf_m
            if baf_mat is not None:
                # BAF in RNA: the fraction of a gene's expression coming from the `p` homolog. 0.5 at
                # a balanced diploid locus; an allelic imbalance (an amplified or lost homolog, or an
                # NMD SNV in cis) pushes it away from 0.5. This is the layer Numbat / CalicoST read,
                # and summing p+m (as `get_exp` does) loses it.
                baf_mat[s:e] = np.divide(buf, tot, out=np.full_like(tot, 0.5, dtype=float), where=tot > 0)

        idx = pd.Index(names)
        # SNV matrix, with coarsened-away passenger SNVs re-emitted per cell (no-op when coarsening is
        # off or nothing was folded). It modifies `snv_mat` in place, so cell_rna_vaf reads it after.
        snv_alleles = self._reconstruct_passengers(snv_mat, types) if want_snv else None

        def _frame(mat, dtype):
            return pd.DataFrame(np.asarray(mat, dtype=dtype), index=idx, columns=gene_names)

        self.cell_data = dict(cell_evo=pd.DataFrame(rows_evo, index=idx))
        if "cell_snv" in layers:
            self.cell_data["cell_snv"] = _frame(snv_mat, np.float64)
        if cnv_mat is not None:
            self.cell_data["cell_cnv"] = _frame(cnv_mat, np.int16)
        if exp_mat is not None:
            self.cell_data["cell_exp"] = _frame(exp_mat, np.float32)
        self.cell_data["cell_crd"] = pd.DataFrame(crd if crd else np.empty((0, 2)), index=idx,
                                                  columns=["row", "col"]).astype(int)
        self.cell_data["cell_type"] = pd.DataFrame(types, index=idx, columns=["cell_id"])
        self.cell_data["cell_deme"] = pd.DataFrame(demes_col, index=idx, columns=["deme_id"])
        # Allele-resolved neutral SNV layer (the PRIMARY truth of the coalescent passenger overlay):
        # per-homolog VAF frames whose sum is exactly `cell_snv`. Only present when the overlay
        # actually ran (coarsening on + folded burden + neutral sites).
        if snv_alleles is not None:
            snv_p, snv_m = snv_alleles
            if "cell_snv_p" in layers:
                self.cell_data["cell_snv_p"] = _frame(snv_p, np.float64)
            if "cell_snv_m" in layers:
                self.cell_data["cell_snv_m"] = _frame(snv_m, np.float64)
        # cell_rna_vaf: the EXPECTED allele FRACTION in RNA (not an observed VAF). With m
        # mutant + w wt copies at a locus and per-locus expression effect e (selection.mut_effects:
        # oncogene=2, TSG=0.5, else 1), the fraction of expression from mutant alleles is
        # (m·e·base)/(m·e·base + w·base) = (m·e)/(m·e + w) = (v·e)/(v·e + (1-v)) with v = DNA-VAF
        # (cell_snv). The per-gene baseline expression CANCELS in this fraction, which is why at a
        # neutral locus (e=1) the *expected fraction* equals DNA-VAF; e>1 inflates, e<1 deflates.
        # This is NOT what a caller observes: the OBSERVED scRNA-VAF is this fraction SAMPLED at a
        # depth equal to the gene's expression (the UMI count, which carries the per-gene
        # baseline + cell-type + CNV + SNV-effect scaling). Per-gene expression varies enormously,
        # so most loci drop out or are noisy and the observed VAF does NOT match DNA-VAF at neutral
        # loci — the read layer (reads/rna.py) does the depth-aware sampling and the obs_fidelity
        # distortion; both deliberately live outside the engine.
        if "cell_rna_vaf" in layers:
            flat_eff = (np.concatenate(self.selection.mut_effects)
                        if self.selection.mut_effects else np.ones(n_genes))
            rna_vaf = np.empty((n_cells, n_genes))
            for s in range(0, n_cells, step):
                v = snv_mat[s:s + step]
                num = v * flat_eff
                denom = num + (1.0 - v)
                rna_vaf[s:s + step] = np.divide(num, denom, out=np.zeros_like(v, dtype=float),
                                                where=denom > 0)
            self.cell_data["cell_rna_vaf"] = _frame(rna_vaf, np.float64)
        # WGD ground truth: the per-genotype `is_wgd` flag surfaced per cell, so downstream CNA / BAF
        # benchmarks (e.g. Numbat) can score WGD detection against the truth. Present only when WGD
        # is enabled. `types` is the per-cell genotype id.
        if self._cancer_params is not None and self._cancer_params.get("wgd_rate", 0):
            self.cell_data["cell_wgd"] = pd.DataFrame(
                {"is_wgd": [bool(self.genotypes[g].is_wgd) for g in types]}, index=idx)
        # Drug-induced resistance-state ground truth: the per-genotype carried state level surfaced per cell, so a downstream method can be scored on
        # telling a non-genetic tolerant STATE from a resistance DRIVER (the Whiting & Graham
        # plasticity-vs-genetic split, with iscc's lineage tracing as ground truth). Present only when
        # the feature is on.
        if self.selection.resistance_state_on:
            self.cell_data["cell_resistance_state"] = pd.DataFrame(
                {"resistance_state": [float(getattr(self.genotypes[g], "resistance_state", 0.0))
                                      for g in types]}, index=idx)
        # Microenvironment: surface the per-cell cell-extrinsic levels (ground truth for the
        # intrinsic-vs-extrinsic decomposition benchmark). Present only when the layer is on.
        # `cci_level` is the per-cell RECEIVED signal: ligand availability at the cell's deme × the
        # cell's own receptor level — so it varies cell-to-cell by clone within a deme, not just
        # deme-to-deme. It is exactly the quantity the CCI target-gene fold-change reads
        # (fold = 1 + strength·cci_level), and the ground truth a CCI method is scored against.
        if deme_mod is not None:
            dcol = np.asarray(demes_col, dtype=int)
            hyp, lig = self.microenv_truth["hypoxia"], self.microenv_truth["cci"]
            rl = self.microenv_truth.get("cci_receptor_level", {})
            if dcol.size:
                recv = lig[dcol] * np.array([float(rl.get(g, 0.0)) for g in types])
            else:
                recv = np.array([])
            self.cell_data["cell_microenv"] = pd.DataFrame(
                {"hypoxia_level": hyp[dcol] if dcol.size else np.array([]),
                 "cci_level": recv}, index=idx)

        # Ductal-field ground truth: the gland each cell sits in (-1 for stroma). Only added when a
        # gland field was seeded.
        if self.gland_id is not None:
            gcol = np.asarray(demes_col, dtype=int)
            self.cell_data["cell_gland"] = pd.DataFrame(
                {"gland_id": self.gland_id[gcol] if gcol.size else np.array([], dtype=int)}, index=idx)

        # Metastasis ground truth: the compartment each cell sits in (0 = primary, 1 = met). Only
        # added when the met is enabled. Flows through the sample/data
        # glob loaders automatically, so an assay can subset to a single site.
        if self._met_enabled:
            ccol = np.asarray(demes_col, dtype=int)
            self.cell_data["cell_compartment"] = pd.DataFrame(
                {"compartment": ((ccol >= self.n_primary_demes).astype(int) if ccol.size
                                 else np.array([], dtype=int))}, index=idx)

        # The allele layers + the RNA BAF, and the per-cell program activity. Only added when the
        # program layer is on.
        if P is not None:
            for name, mat in (("cell_exp_p", ep_mat), ("cell_exp_m", em_mat), ("cell_rna_baf", baf_mat)):
                if mat is not None:
                    self.cell_data[name] = pd.DataFrame(mat, index=idx, columns=gene_names)
            if Z is not None and P.n_programs:
                self.cell_data["cell_program"] = pd.DataFrame(
                    Z, index=idx, columns=list(P.dictionary.program_names))
            self.program_truth = P.truth()
        self._cell_data_stale = False
        return self.cell_data

    def ensure_cell_data(self, **kwargs):
        """Materialise ``cell_data`` only if it is missing or stale, and return it.

        The counterpart of ``grow(materialize=False)``: step the tumour without rebuilding the
        per-cell tables, then call this right before sampling. When the tables are current it
        returns them as they are, without rebuilding.

        Parameters
        ----------
        **kwargs
            Passed to [`make_cell_data`][iscc.tumor.GenotypeTumor.make_cell_data] (``max_cells``,
            ``region``, ``depth_frac``, ...) and used only when it materialises. To cut a
            specific section of a tumour whose tables are current, call ``make_cell_data``
            directly.

        Returns
        -------
        dict
            ``self.cell_data`` — the per-cell dataframes.
        """
        if self.cell_data is None or getattr(self, "_cell_data_stale", False):
            self.make_cell_data(**kwargs)
        return self.cell_data
