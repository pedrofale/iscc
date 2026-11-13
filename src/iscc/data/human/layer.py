"""The human genome layer: lift a coarse growth genome to real genes, germline SNPs and allele counts.

Growth never reads this module. It maps each growth gene to a stretch of a human chromosome arm, so
that every human gene in that stretch inherits the growth gene's copy number per homolog and its
expression, and it gives the patient a germline of real heterozygous SNPs. The assays then draw
counts over the human genes (``scRNA.run(..., gene_lift=layer.lift)``, ``Visium.run(...,
gene_lift=layer.lift)``), and :meth:`HumanGenomeLayer.allele_counts` adds per-haplotype counts at the
SNPs that a 3' library covers.

With ``layer.lift`` a human gene simply takes the expression of the growth gene that covers it.
:meth:`HumanGenomeLayer.expression` is the realistic path: reference-atlas baselines per cell type and
Hallmark programmes on the human genes (:class:`~iscc.data.human.HumanExpression`); only what the
genotype and the niche do is read off the growth genes.
"""
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .reference import load_reference


def as_csr(counts):
    """A barcodes x genes count table as a CSR matrix: a scipy matrix, a dense or sparse pandas
    DataFrame, or an array."""
    if sp.issparse(counts):
        return counts.tocsr()
    if isinstance(counts, pd.DataFrame):
        try:
            return counts.sparse.to_coo().tocsr()
        except AttributeError:
            return sp.csr_matrix(counts.to_numpy())
    return sp.csr_matrix(np.asarray(counts))


class GeneLift:
    """Lift a matrix over growth genes to one over human genes.

    ``lift(X)`` returns ``X[:, cover]``: column ``h`` is the growth gene that covers human gene
    ``h``. The map is linear, so lifting before or after pooling cells into spots gives the same
    result, which lets :class:`~iscc.data.Visium` pool on the small growth axis first.

    Attributes
    ----------
    cover : numpy.ndarray
        Growth-gene index of each human gene.
    genes : pandas.Index
        Ensembl IDs of the human genes, the column labels of a lifted matrix.
    """

    def __init__(self, cover, genes):
        self.cover = np.asarray(cover, dtype=np.int64)
        self.genes = pd.Index(genes, name="gene_id")

    def __call__(self, X):
        return np.asarray(X)[:, self.cover]


@dataclass
class AlleleCounts:
    """Per-haplotype UMI counts at heterozygous SNPs, for cells or spots.

    Attributes
    ----------
    A, B : scipy.sparse.csr_matrix
        Barcodes x SNPs. ``A`` counts UMIs from haplotype 1 (the ``p`` homolog), ``B`` from haplotype
        2 (``m``): CalicoST's ``cell_snp_Aallele`` / ``cell_snp_Ballele`` convention with an exact
        phase.
    snps : pandas.DataFrame
        One row per column: ``snp_id`` (``chr_pos_ref_alt``, chromosome as an integer), ``chrom``,
        ``pos``, ``ref``, ``alt``, ``gene_id``, ``alt_hap`` (the homolog carrying ALT).
    barcodes : list of str
        Row labels (cell or spot names of the assay).
    """
    A: sp.csr_matrix
    B: sp.csr_matrix
    snps: pd.DataFrame
    barcodes: list

    @property
    def total(self):
        """SNP-covering UMIs per barcode."""
        return np.asarray((self.A + self.B).sum(axis=1)).ravel()

    def ref_alt(self):
        """``(AD, DP)``: ALT-allele and total depth per barcode and SNP (cellsnp-lite's convention)."""
        alt_on_p = (self.snps["alt_hap"].to_numpy() == "p")
        D = sp.diags(alt_on_p.astype(float))
        AD = self.A @ D + self.B @ sp.diags((~alt_on_p).astype(float))
        return sp.csr_matrix(AD).astype(np.int64), (self.A + self.B).astype(np.int64)

    def drop_empty(self):
        """The same counts without the SNPs no barcode covers."""
        keep = np.flatnonzero(np.asarray((self.A + self.B).sum(axis=0)).ravel() > 0)
        return AlleleCounts(self.A[:, keep].tocsr(), self.B[:, keep].tocsr(),
                            self.snps.iloc[keep].reset_index(drop=True), list(self.barcodes))


class HumanGenomeLayer:
    """Map a growth genome onto the human genome and give the patient a germline.

    Built from a grown tumour (or its ``genome_params``), it fixes which human genes each growth
    gene covers and which real SNPs are heterozygous, and how they are phased. It is a deterministic
    function of the growth genome and ``seed``; growth never reads it, so a tumour grows identically
    with or without it.

    With an arm-homologous genome (``genome_params["arms"]``, as :func:`~iscc.data.human.arm_genome`
    builds), growth segment ``s`` is arm ``arms[s]``, and growth gene ``j`` of its ``n`` genes covers
    the ``j``-th of ``n`` equal slices of the arm, whatever ``n`` is. Any other genome is laid along
    the 39 arms end to end (``allow_linear=True``): its segments then straddle chromosome boundaries,
    which no real tumour shows, so that path warns.

    Parameters
    ----------
    tumor : GenotypeTumor, optional
        The grown tumour; its ``genome_params`` and ``layout_seed`` are read. Give ``genome_params``
        instead to build the map without one.
    genome_params : dict, optional
        ``n_segments``, ``segment_sizes`` (or ``segment_size``) and, for an arm-homologous genome,
        ``arms``.
    reference : HumanReference, optional
        Default :func:`~iscc.data.human.load_reference` (with SNPs if ``ISCC_HUMAN_REF`` points at
        them).
    seed : int, optional
        Seed of the germline draw. Default the tumour's ``layout_seed``, so patients grown from one
        configuration share a germline; pass a per-patient seed to give each its own.
    snp_window : int, default 1000
        A 3' library covers a gene's last ``snp_window`` bases (strand-aware); only heterozygous
        SNPs there can show allele counts.
    snp_coverage : float, default 0.1
        Probability that a UMI of a gene covers a given SNP in its window (about read length over
        window). A preset, not fitted.
    snp_concentration : float, optional
        Beta-binomial concentration of the haplotype split at a SNP (allelic-expression noise).
        Default None draws it binomially around the expected haplotype fraction.
    allow_linear : bool, default False
        Allow a genome without ``arms`` (see above).
    """

    def __init__(self, tumor=None, genome_params=None, reference=None, seed=None, snp_window=1000,
                 snp_coverage=0.1, snp_concentration=None, allow_linear=False):
        if genome_params is None:
            if tumor is None:
                raise ValueError("give a tumor or its genome_params")
            genome_params = tumor.genome_params
        self.reference = reference or load_reference()
        n_seg = int(genome_params["n_segments"])
        sizes = genome_params.get("segment_sizes")
        if sizes is None:
            sizes = [int(genome_params.get("segment_size", 1000))] * n_seg
        self.segment_sizes = np.asarray([int(s) for s in sizes])
        self.seg_offsets = np.concatenate([[0], np.cumsum(self.segment_sizes)]).astype(int)
        self.n_growth = int(self.seg_offsets[-1])
        self.arms = genome_params.get("arms")
        if self.arms is None and not allow_linear:
            raise ValueError("the growth genome has no 'arms'; build it with arm_genome(), or pass "
                             "allow_linear=True to lay its segments along the arms end to end")
        if self.arms is not None and len(self.arms) != n_seg:
            raise ValueError(f"{len(self.arms)} arms for {n_seg} segments")
        if self.arms is None:
            warnings.warn("laying a non-arm genome along the human arms: its copy-number segments "
                          "will cross chromosome boundaries", stacklevel=2)
        self.seed = int(seed if seed is not None else getattr(tumor, "layout_seed", 0) or 0)
        self.snp_window = int(snp_window)
        self.snp_coverage = float(snp_coverage)
        self.snp_concentration = None if snp_concentration is None else float(snp_concentration)
        self._build_grid()
        self._build_gene_map()
        self.lift = GeneLift(self.cover, self.genes["gene_id"].to_numpy())
        self.germline = self._draw_germline() if self.reference.snps is not None else None

    # -- the map --------------------------------------------------------------------------------
    def _build_grid(self):
        """Pieces ``(growth gene, arm, lo, hi)`` of the arms each growth gene covers (bp, 0-based)."""
        arms = self.reference.arms
        arm_index = {a: i for i, a in enumerate(arms["arm"])}
        rows = []
        if self.arms is not None:
            for s, name in enumerate(self.arms):
                if name not in arm_index:
                    raise ValueError(f"unknown arm {name!r}")
                a = arm_index[name]
                lo, hi = int(arms["start"].iat[a]), int(arms["end"].iat[a])
                n = int(self.segment_sizes[s])
                edges = lo + np.round(np.arange(n + 1) * (hi - lo) / n).astype(np.int64)
                for j in range(n):
                    rows.append((self.seg_offsets[s] + j, a, edges[j], edges[j + 1]))
        else:
            lengths = (arms["end"] - arms["start"]).to_numpy(np.int64)
            cum = np.concatenate([[0], np.cumsum(lengths)])
            g_edges = np.round(np.arange(self.n_growth + 1) * cum[-1] / self.n_growth).astype(np.int64)
            cuts = np.union1d(g_edges, cum)
            for lo, hi in zip(cuts[:-1], cuts[1:]):
                if hi <= lo:
                    continue
                k = min(int(np.searchsorted(g_edges, lo, side="right") - 1), self.n_growth - 1)
                a = int(np.searchsorted(cum, lo, side="right") - 1)
                off = int(arms["start"].iat[a]) - int(cum[a])
                rows.append((k, a, lo + off, hi + off))
        grid = pd.DataFrame(rows, columns=["growth_gene", "arm", "lo", "hi"])
        grid["segment"] = np.searchsorted(self.seg_offsets, grid["growth_gene"], side="right") - 1
        self.grid = grid

    def _build_gene_map(self):
        ref = self.reference
        arm_of = ref.gene_arm()
        mid = ((ref.genes["start"].to_numpy() + ref.genes["end"].to_numpy()) // 2).astype(np.int64)
        cover = np.full(len(ref.genes), -1, dtype=np.int64)
        for a, piece in self.grid.groupby("arm", sort=False):
            piece = piece.sort_values("lo")
            on = np.flatnonzero(arm_of == a)
            k = np.searchsorted(piece["lo"].to_numpy(), mid[on], side="right") - 1
            k = np.clip(k, 0, len(piece) - 1)
            cover[on] = piece["growth_gene"].to_numpy()[k]
        keep = np.flatnonzero(cover >= 0)
        self.ref_rows = keep                               # rows of reference.genes that are emitted
        self.genes = ref.genes.iloc[keep].reset_index(drop=True)
        self.cover = cover[keep]
        self.segment = np.searchsorted(self.seg_offsets, self.cover, side="right") - 1
        self.genes["arm"] = ref.arms["arm"].to_numpy()[arm_of[keep]]
        self.genes["growth_gene"] = self.cover
        self.genes["segment"] = self.segment

    # -- germline -------------------------------------------------------------------------------
    def _draw_germline(self):
        """Heterozygous SNPs of this patient, phased onto the homologs (exactly)."""
        snps = self.reference.snps
        row_to_gene = np.full(len(self.reference.genes), -1, dtype=np.int64)
        row_to_gene[self.ref_rows] = np.arange(len(self.ref_rows))
        gene = row_to_gene[snps["gene"].to_numpy()]
        cand = snps.loc[gene >= 0].copy()
        cand["gene"] = gene[gene >= 0]
        rng = np.random.default_rng([self.seed, 0x6E])
        af = np.nan_to_num(cand["af"].to_numpy(float), nan=0.0)
        het = rng.random(len(cand)) < 2.0 * af * (1.0 - af)
        g = cand.loc[het].reset_index(drop=True)
        g["alt_hap"] = np.where(rng.random(len(g)) < 0.5, "p", "m")
        genes = self.genes
        strand = genes["strand"].to_numpy()[g["gene"]]
        start = genes["start"].to_numpy()[g["gene"]]
        end = genes["end"].to_numpy()[g["gene"]]
        pos = g["pos"].to_numpy()
        w = self.snp_window
        g["observable"] = np.where(strand == "+", pos > end - w, pos < start + w)
        g["gene_id"] = genes["gene_id"].to_numpy()[g["gene"]]
        chrom_num = g["chrom"].str.replace("chr", "", regex=False)
        g["snp_id"] = chrom_num + "_" + g["pos"].astype(str) + "_" + g["ref"] + "_" + g["alt"]
        g = g.assign(_c=chrom_num.astype(int)).sort_values(["_c", "pos"]).drop(columns="_c")
        return g.reset_index(drop=True)[["snp_id", "chrom", "pos", "ref", "alt", "af", "gene",
                                         "gene_id", "alt_hap", "observable"]]

    # -- per-cell truth -------------------------------------------------------------------------
    @staticmethod
    def _gids(cell_data, cells=None):
        t = cell_data["cell_type"].iloc[:, 0]
        return (t if cells is None else t.reindex(cells)).astype(object).to_numpy()

    def homolog_cn(self, gids, tumor):
        """Copy number per homolog and growth segment for each genotype: ``(cn_p, cn_m)``, two
        ``(len(gids), n_segments)`` int arrays. Normal cells are ``(1, 1)``."""
        gids = list(gids)
        cn_p = np.ones((len(gids), len(self.segment_sizes)), dtype=np.int16)
        cn_m = np.ones_like(cn_p)
        for i, gid in enumerate(gids):
            g = tumor.genotypes.get(gid)
            if g is None or getattr(g, "type", "cancer") != "cancer":
                continue
            for s in range(len(self.segment_sizes)):
                cn_p[i, s] = len(g.genome[s]["p"])
                cn_m[i, s] = len(g.genome[s]["m"])
        return cn_p, cn_m

    def hap_fraction(self, cell_data, cells, tumor=None):
        """Expected fraction of each growth gene's expression from homolog ``p``, per cell.

        Read from ``cell_exp_p`` / ``cell_exp_m`` when the expression layer is allele-resolved;
        otherwise from the cell's copy number per homolog (linear dosage), which needs ``tumor``.
        Returns a ``(len(cells), n_growth)`` float32 array; 0.5 where a gene is not expressed.
        """
        cells = list(cells)
        if "cell_exp_p" in cell_data:
            ep = cell_data["cell_exp_p"].reindex(cells).to_numpy(np.float32)
            em = cell_data["cell_exp_m"].reindex(cells).to_numpy(np.float32)
            tot = ep + em
            return np.divide(ep, tot, out=np.full_like(tot, 0.5), where=tot > 0)
        if tumor is None:
            raise ValueError("this cell_data has no allele-resolved expression; pass the tumor so the "
                             "haplotype fraction can be read from each cell's copy number")
        gids = self._gids(cell_data, cells)
        uniq, inv = np.unique(gids.astype(str), return_inverse=True)
        lookup = {str(g): g for g in gids}
        cn_p, cn_m = self.homolog_cn([lookup[u] for u in uniq], tumor)
        tot = (cn_p + cn_m).astype(np.float32)
        seg_frac = np.divide(cn_p.astype(np.float32), tot, out=np.full_like(tot, 0.5), where=tot > 0)
        gene_frac = np.repeat(seg_frac, self.segment_sizes, axis=1)
        return gene_frac[inv]

    # -- allele counts --------------------------------------------------------------------------
    def allele_counts(self, counts, frac_p, barcodes, seed=0):
        """Per-haplotype UMI counts at the observable heterozygous SNPs.

        Each UMI of a gene covers a given SNP in the gene's 3' window with probability
        ``snp_coverage``; a covering UMI comes from homolog ``p`` with the barcode's expected
        fraction for the gene (binomial, or beta-binomial with ``snp_concentration``).

        Parameters
        ----------
        counts : pandas.DataFrame, numpy.ndarray or scipy.sparse matrix
            Barcodes x human genes UMI counts, in the layer's gene order.
        frac_p : numpy.ndarray or callable
            Barcodes x growth genes expected fraction from homolog ``p`` (see :meth:`hap_fraction`), or
            a function ``frac_p(barcode_rows, human_genes)`` giving it per entry (a Visium spot whose
            cell types weight its genes differently, :meth:`HumanExpression.spot_hap_fraction`).
        barcodes : list of str
            Row labels.
        seed : int, default 0
            Seed of the coverage and allele draws.

        Returns
        -------
        AlleleCounts
        """
        if self.germline is None:
            raise ValueError("no SNPs: run iscc.data.human.fetch_snps() and point ISCC_HUMAN_REF at "
                             "it, or pass load_reference(snps=...) to the layer")
        obs = self.germline.loc[self.germline["observable"]].reset_index(drop=True)
        C = as_csr(counts)
        n_bc = C.shape[0]
        # SNPs grouped by gene: for every nonzero (barcode, gene), one draw per SNP of the gene
        order = np.argsort(obs["gene"].to_numpy(), kind="stable")
        snp_gene = obs["gene"].to_numpy()[order]
        g_lo = np.searchsorted(snp_gene, np.arange(C.shape[1]), side="left")
        g_hi = np.searchsorted(snp_gene, np.arange(C.shape[1]), side="right")
        k = g_hi - g_lo
        Ccoo = C.tocoo()
        has = k[Ccoo.col] > 0
        r, g, n = Ccoo.row[has], Ccoo.col[has], Ccoo.data[has].astype(np.int64)
        reps = k[g]
        rows_e = np.repeat(r, reps)
        n_e = np.repeat(n, reps)
        starts = np.repeat(g_lo[g], reps)
        within = np.arange(reps.sum()) - np.repeat(np.cumsum(reps) - reps, reps)
        snp_e = order[starts + within]
        rng = np.random.default_rng([self.seed, int(seed), 0xA1])
        cov = rng.binomial(n_e, self.snp_coverage)
        nz = cov > 0
        rows_e, snp_e, cov = rows_e[nz], snp_e[nz], cov[nz]
        human = obs["gene"].to_numpy()[snp_e]
        if callable(frac_p):
            f = np.asarray(frac_p(rows_e, human), dtype=float)
        else:
            f = np.asarray(frac_p)[rows_e, self.cover[human]].astype(float)
        if self.snp_concentration:
            c = self.snp_concentration
            f = rng.beta(np.maximum(f * c, 1e-6), np.maximum((1.0 - f) * c, 1e-6))
        a = rng.binomial(cov, np.clip(f, 0.0, 1.0))
        shape = (n_bc, len(obs))
        A = sp.csr_matrix((a, (rows_e, snp_e)), shape=shape)
        B = sp.csr_matrix((cov - a, (rows_e, snp_e)), shape=shape)
        A.sum_duplicates(); B.sum_duplicates()
        snps = obs[["snp_id", "chrom", "pos", "ref", "alt", "gene_id", "alt_hap"]]
        return AlleleCounts(A, B, snps, list(barcodes))

    def scrna_allele_counts(self, assay, cell_data, tumor=None, seed=0):
        """Allele counts for an :class:`~iscc.data.scRNA` assay run with ``gene_lift=self.lift``."""
        cells = list(assay.observed_counts.index)
        frac = self.hap_fraction(cell_data, cells, tumor)
        counts = getattr(assay, "counts_csr", None)
        return self.allele_counts(assay.observed_counts if counts is None else counts, frac, cells, seed=seed)

    def visium_allele_counts(self, assay, tumor=None, seed=0):
        """Allele counts for a :class:`~iscc.data.Visium` assay run with ``gene_lift=self.lift``.

        A spot's haplotype fraction for a gene pools its cells' allele-resolved expression, after the
        same lateral diffusion the assay applied to the counts; with :class:`HumanExpression` as the
        lift, each cell type weighs in with its own baseline for the gene."""
        lift = getattr(assay, "gene_lift", None)
        if lift is not None and hasattr(lift, "spot_hap_fraction"):
            return self.allele_counts(assay.spot_counts, lift.spot_hap_fraction,
                                      list(assay.spot_counts.index), seed=seed)
        g = assay._grid
        placed = g["placed"]
        cells = np.asarray(placed["cell_exp"].index)
        exp = placed["cell_exp"].to_numpy(np.float64)
        frac = self.hap_fraction(placed, cells, tumor)
        exp_p = exp * frac
        n_spots = len(g["members"])
        raw = np.zeros((n_spots, exp.shape[1]))
        raw_p = np.zeros_like(raw)
        for s, m in enumerate(g["members"]):
            if m.size:
                raw[s] = exp[m].sum(axis=0)
                raw_p[s] = exp_p[m].sum(axis=0)
        raw = assay._diffuse(raw, g["spot_coords"])
        raw_p = assay._diffuse(raw_p, g["spot_coords"])
        spot_frac = np.divide(raw_p, raw, out=np.full_like(raw, 0.5), where=raw > 0)
        return self.allele_counts(assay.spot_counts, spot_frac, list(assay.spot_counts.index), seed=seed)

    def expression(self, tumor, **kwargs):
        """The human-gene expression of ``tumor``'s cells from atlas baselines and Hallmark programmes
        (:class:`~iscc.data.human.HumanExpression`; keyword arguments go to it). Pass it to the assays
        as ``gene_lift``."""
        from .expression import HumanExpression
        return HumanExpression(self, tumor, **kwargs)

    # -- truth ----------------------------------------------------------------------------------
    def cn_segments(self, gids, tumor):
        """True copy number per homolog of each genotype, as human intervals.

        Returns a table with ``genotype``, ``chrom``, ``start``, ``end`` (0-based, half-open),
        ``arm``, ``cn_p``, ``cn_m``, ``cn``: one row per genotype and stretch of a growth segment on
        one arm (one row per arm for an arm-homologous genome)."""
        gids = list(dict.fromkeys(gids))
        cn_p, cn_m = self.homolog_cn(gids, tumor)
        arms = self.reference.arms
        pieces = (self.grid.groupby(["segment", "arm"], sort=True)
                  .agg(lo=("lo", "min"), hi=("hi", "max")).reset_index())
        rows = []
        for i, gid in enumerate(gids):
            for pc in pieces.itertuples(index=False):
                p, m = int(cn_p[i, pc.segment]), int(cn_m[i, pc.segment])
                rows.append((str(gid), arms["chrom"].iat[pc.arm], int(pc.lo), int(pc.hi),
                             arms["arm"].iat[pc.arm], p, m, p + m))
        return pd.DataFrame(rows, columns=["genotype", "chrom", "start", "end", "arm",
                                           "cn_p", "cn_m", "cn"])

    def gene_cn(self, gids, tumor):
        """Total copy number of every human gene for each genotype: ``(len(gids), n_genes)`` int16."""
        cn_p, cn_m = self.homolog_cn(list(gids), tumor)
        return (cn_p + cn_m)[:, self.segment]

    @staticmethod
    def is_tumour(gids, tumor):
        """Whether each genotype is a cancer genotype."""
        out = []
        for gid in gids:
            g = tumor.genotypes.get(gid)
            out.append(bool(g is not None and getattr(g, "type", None) == "cancer"))
        return np.asarray(out)
