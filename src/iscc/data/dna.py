"""Bulk + single-cell DNA-seq count/coverage simulation (DESIGN_features §C C1 / §D, F4+F5).

Replaces the old uniform-multinomial + fpr/fnr placeholders with a realistic,
**copy-number-scaled** coverage model shared by both modalities. The two assays differ only
in the *regime* of the same shared core (see `batch.DNABatch`):

    BULK        — pool the sampled cells, one high read budget, large-kappa (≈ multinomial/
                  Poisson) compositional depth, Binomial allele layer.
    SINGLE-CELL — per-cell low read budget, small-kappa (lumpy MDA/MALBAC amplification)
                  compositional depth nested in the run batch, Beta-Binomial allele layer
                  with explicit ADO, and doublets.

Orthogonal to bulk-vs-single-cell is **capture breadth** ∈ {wgs, wes, panel}, which sets
(i) the observed locus set, (ii) the depth regime (wgs low / wes moderate / panel very high),
and (iii) the dominant capture bias (wgs GC; wes per-target; panel per-amplicon). Breadth
applies identically to bulk and single-cell.

Ground truth consumed (per-locus, from the tumour engine, see `tumor.make_cell_data`):
  * `cell_cnv[c, l]` — total copy number at locus l in cell c (constant within a segment).
  * `cell_snv[c, l]` — TRUE alt fraction (VAF) at locus l in cell c (0 / 0.5 / 1.0 / ...).

Outputs are **count-level** matrices (coverage / alt-count / VAF / CNA log-ratio) plus the
surfaced ground truth (true CN, true alt fraction, ADO mask). Raw FASTQ is F7 (out of scope).
"""
from .assay import Assay
from .batch import DNABatch, DNABatchHyperParams, DNA_DEPTH_MODELS

import os

import numpy as np
import pandas as pd


# Capture-breadth presets (DESIGN_features §D): observed-locus regime + depth regime +
# dominant capture bias. `locus_frac` / `panel_size` drive locus selection; the rest are
# DNABatchHyperParams. Single-cell scales the depth down (see `scDNA.depth_scale`).
DNA_BREADTH_PRESETS = {
    # WGS: all loci, low per-locus depth, GC bias dominant, no targeted capture.
    "wgs":   dict(mu_depth=30.0,   capture_sigma=0.0,  gc_curve_sigma=0.20,
                  locus_frac=1.0,  panel_size=None),
    # WES: exon/gene subset, moderate on-target depth, per-target capture variability.
    "wes":   dict(mu_depth=120.0,  capture_sigma=0.30, gc_curve_sigma=0.12,
                  locus_frac=0.30, panel_size=None),
    # Panel: a small designated set (drivers), very deep, per-amplicon efficiency dominant.
    "panel": dict(mu_depth=1500.0, capture_sigma=0.45, gc_curve_sigma=0.05,
                  locus_frac=None, panel_size=30),
}
_HYPER_KEYS = set(DNABatchHyperParams().to_dict().keys())


def genome_features(genes, seed=20240601):
    """Stable per-locus GC content, mappability and C-site flags (a genome property).

    Deterministic in the gene set/order, so the SAME genome yields the same features across
    batches and breadths (only the per-batch GC *curve* differs). Returns arrays aligned to
    `genes`: gc ∈ ~[0.2, 0.8] (centred ~0.45), mappability ∈ (0, 1] (mostly near 1),
    ct_sites (bool, FFPE C>T-eligible loci).
    """
    rng = np.random.default_rng(seed)
    n = len(genes)
    gc = 0.2 + 0.6 * rng.beta(4.0, 5.0, size=n)
    mappability = np.clip(rng.beta(9.0, 1.0, size=n), 0.05, 1.0)
    ct_sites = rng.random(n) < 0.25
    return gc, mappability, ct_sites


def _segment_of(gene):
    """Parse the segment index from an engine gene name `G_{seg}_{pos}` (else -1)."""
    s = str(gene).split("_")
    if len(s) >= 3 and s[0] == "G":
        try:
            return int(s[1])
        except ValueError:
            return -1
    return -1


class DNA(Assay):
    """Base DNA assay: breadth + the shared `DNABatch` coverage core.

    Holds the breadth / locus-selection logic and assembles the per-locus ground truth
    (copy number + true alt fraction) that the bulk / single-cell subclasses pass through
    the shared depth + allele layers. Subclasses set the amplification regime via
    `mode_defaults` (large-kappa Binomial for bulk; small-kappa Beta-Binomial + ADO for sc).

    Backward-compatible with the old placeholder signature: `n_reads` (now the total read
    budget), `target_genes`, `data_mode`, and `fpr` (now mapped to `error_rate`); `fnr` is
    accepted but unused by the realistic model.
    """

    mode = "bulk"
    depth_scale = 1.0  # multiplies the breadth depth regime (single-cell lowers this)

    def __init__(self, breadth="wgs", target_genes="all", data_mode="counts",
                 depth_model="dm", n_reads=None, seed=42, batch_label=None,
                 fpr=None, fnr=None, **hyper_overrides):
        super(DNA, self).__init__(seed=seed, protocol=f"dna-{self.mode}-{breadth}")
        if breadth not in DNA_BREADTH_PRESETS:
            raise ValueError(
                f"unknown breadth {breadth!r}; choose from {sorted(DNA_BREADTH_PRESETS)}"
            )
        if depth_model not in DNA_DEPTH_MODELS:
            raise ValueError(
                f"unknown depth_model {depth_model!r}; choose from {sorted(DNA_DEPTH_MODELS)}"
            )
        self.breadth = breadth
        self.target_genes = target_genes
        self.data_mode = data_mode
        self.depth_model = depth_model
        self.n_reads = n_reads
        self.batch_label = batch_label
        self.fnr = fnr  # accepted for back-compat; the realistic model uses error_rate
        # back-compat: fpr -> sequencing error rate
        if fpr is not None and "error_rate" not in hyper_overrides:
            hyper_overrides["error_rate"] = float(fpr)
        self.hypers = self._build_hypers(breadth, depth_model, hyper_overrides)

    # -- hyper-parameter assembly ------------------------------------------------------
    def mode_defaults(self):
        """Amplification-regime defaults for this modality (overridden by subclasses)."""
        return {}

    def _build_hypers(self, breadth, depth_model, overrides):
        preset = DNA_BREADTH_PRESETS[breadth]
        params = {k: v for k, v in preset.items() if k in _HYPER_KEYS}
        params.update(self.mode_defaults())
        params["breadth"] = breadth
        params["depth_model"] = depth_model
        params["mu_depth"] = float(params.get("mu_depth", 30.0)) * self.depth_scale
        for k, v in overrides.items():
            if k not in _HYPER_KEYS:
                raise TypeError(f"unknown DNA hyper-parameter {k!r}")
            if k not in ("breadth", "depth_model"):
                v = float(v)
            params[k] = v
        return DNABatchHyperParams(**params)

    # -- locus selection (breadth + target_genes) --------------------------------------
    def _select_loci(self, genes):
        """Observed-locus positions for this breadth (DESIGN_features §D).

        `target_genes` overrides the breadth default: a list selects exactly those loci, a
        float a random fraction; 'all' defers to the breadth regime (wgs=all, wes=fraction,
        panel=small driver set). Selection is seeded so it is stable across batches.
        """
        genes = list(genes)
        n = len(genes)
        rng = np.random.default_rng(self.seed + 7919)  # locus selection (not technical)
        tg = self.target_genes
        if isinstance(tg, (list, tuple, np.ndarray)):
            keep = [g for g in tg if g in set(genes)]
            idx = [genes.index(g) for g in keep]
        elif isinstance(tg, float):
            k = max(1, int(round(tg * n)))
            idx = sorted(rng.choice(n, size=k, replace=False))
        else:  # 'all' -> breadth regime
            preset = DNA_BREADTH_PRESETS[self.breadth]
            if preset["locus_frac"] is not None and preset["locus_frac"] >= 1.0:
                idx = list(range(n))
            elif preset["panel_size"] is not None:
                k = min(int(preset["panel_size"]), n)
                idx = sorted(rng.choice(n, size=k, replace=False))
            else:
                k = max(1, int(round(preset["locus_frac"] * n)))
                idx = sorted(rng.choice(n, size=k, replace=False))
        return np.asarray(idx, dtype=int)

    @staticmethod
    def _ground_truth(cell_data, cells, genes):
        """Per-cell (cells × loci) copy-number and true-alt-fraction matrices.

        Falls back to diploid (CN=2) when `cell_cnv` is absent (e.g. minimal test fixtures).
        """
        snv = cell_data["cell_snv"].loc[cells, genes].values.astype(float)
        cnv_df = cell_data.get("cell_cnv")
        if cnv_df is not None:
            cnv = cnv_df.loc[cells, genes].values.astype(float)
        else:
            cnv = np.full_like(snv, 2.0)
        return cnv, snv

    def _make_batch(self, genes):
        """Build + realize the run batch for the observed loci (stable genome features)."""
        gc, mapp, ct = genome_features(genes)
        label = self.batch_label if self.batch_label is not None else f"dnabatch{self.seed}"
        batch = DNABatch(self.hypers, seed=self.seed, label=label).realize(gc, mapp, ct)
        self.gc, self.mappability = gc, mapp
        return batch


class bulkDNA(DNA):
    """Bulk DNA-seq: pool the sampled cells, high read budget, large-kappa compositional
    depth, Binomial allele layer (DESIGN_features F4).

    Coverage at a locus ∝ the POOLED total copy number (Σ_c CN_cl) × per-locus efficiency,
    so amplified segments draw proportionally more reads, and at a fixed budget compositionally
    reduce the coverage of others (the DM coupling). The pooled true alt fraction is the
    copy-number-weighted mean of the per-cell VAFs.
    """

    mode = "bulk"
    depth_scale = 1.0

    def mode_defaults(self):
        # large kappa ≈ multinomial/Poisson (un-amplified bulk); deep beta-binom conc unused.
        return dict(kappa=2000.0, nb_dispersion=0.1, ado_rate=0.0,
                    beta_binom_conc=200.0, doublet_rate=0.0)

    def run(self, cell_data, cell_subset=None, **kwargs):
        snv = cell_data["cell_snv"]
        cells = np.asarray(list(cell_subset)) if cell_subset is not None else np.asarray(snv.index)
        genes_all = list(snv.columns)
        obs = self._select_loci(genes_all)
        genes = [genes_all[i] for i in obs]

        cnv, af = self._ground_truth(cell_data, cells, genes)            # (n_cells, n_loci)
        total_copies = cnv.sum(axis=0)                                   # pooled copies
        alt_copies = (af * cnv).sum(axis=0)
        true_af = np.divide(alt_copies, total_copies,
                            out=np.zeros_like(alt_copies), where=total_copies > 0)
        true_cn = total_copies / max(len(cells), 1)                      # mean CN across pool

        batch = self._make_batch(genes)
        weights = total_copies * batch.efficiency
        N = int(self.n_reads) if self.n_reads is not None else int(round(self.hypers.mu_depth * len(genes)))
        coverage = batch.emit_depth(weights, N).astype(int)
        alt = batch.alleles_binomial(coverage, true_af).astype(int)
        vaf = np.divide(alt, coverage, out=np.zeros(len(coverage)), where=coverage > 0)

        # CNA log-ratio: depth corrected for capture efficiency, normalized to its median.
        corr = np.divide(coverage, batch.efficiency, out=np.zeros(len(coverage)),
                         where=batch.efficiency > 0)
        med = np.median(corr[corr > 0]) if np.any(corr > 0) else 0.0
        log2_ratio = np.where((corr > 0) & (med > 0), np.log2(np.maximum(corr, 1e-9) / med), np.nan)

        self.batch = batch
        self.n_cells_pooled = len(cells)
        self.observed_data = pd.DataFrame(
            {
                "coverage": coverage,
                "alt_counts": alt,
                "vaf": vaf,
                "true_cn": true_cn,
                "true_alt_fraction": true_af,
                "log2_ratio": log2_ratio,
                "segment": [_segment_of(g) for g in genes],
                "gc": self.gc,
                "mappability": self.mappability,
            },
            index=pd.Index(genes, name="locus"),
        )
        return self

    def to_anndata(self):
        """One 'pseudobulk sample' observation × loci (var) AnnData (X = coverage)."""
        import anndata as ad

        df = self.observed_data
        adata = ad.AnnData(
            X=df["coverage"].values.astype(np.float32)[None, :],
            obs=pd.DataFrame({"batch": [self.batch.label], "breadth": [self.breadth],
                              "n_cells_pooled": [self.n_cells_pooled]},
                             index=pd.Index([self.batch.label], name="sample")),
            var=df.drop(columns=["coverage"]).copy(),
        )
        adata.layers["alt_counts"] = df["alt_counts"].values.astype(np.float32)[None, :]
        adata.layers["vaf"] = df["vaf"].values.astype(np.float32)[None, :]
        adata.uns["assay"] = "bulkDNA"
        adata.uns["breadth"] = self.breadth
        adata.uns["hyperparams"] = self.hypers.to_dict()
        return adata

    def write(self, out_path, write_h5ad=True, **kwargs):
        os.makedirs(out_path, exist_ok=True)
        # `counts.csv` kept as the stable on-disk contract (CLI/pipeline test).
        self.observed_data.to_csv(os.path.join(out_path, "counts.csv"))
        if write_h5ad:
            try:
                self.to_anndata().write_h5ad(os.path.join(out_path, f"{self.batch.label}.h5ad"))
            except Exception:
                pass


class cfDNA(DNA):
    """Cell-free DNA (liquid biopsy) — placeholder (DESIGN_features §A liquid biopsy / F2)."""

    def run(self, cell_data, **kwargs):
        raise NotImplementedError("cfDNA (liquid biopsy) is not implemented yet (see §A).")


class scDNA(DNA):
    """Single-cell DNA-seq: per-cell low budget, small-kappa (lumpy amplification) depth
    nested in the run batch, Beta-Binomial alleles + explicit ADO + doublets (F5).

    Each cell gets its own DM draw (the per-cell amplification profile, the dominant
    single-cell noise) on weights ∝ that cell's copy number. ADO (one allele lost at a het
    locus) is a separate Bernoulli layer — the dominant single-cell allele artifact — applied
    before the Beta-Binomial alt-count draw.
    """

    mode = "sc"
    depth_scale = 0.3  # single-cell coverage is far shallower than bulk for the same breadth

    def __init__(self, n_cells=100, **dna_kwargs):
        super(scDNA, self).__init__(**dna_kwargs)
        self.n_cells = int(n_cells)

    def mode_defaults(self):
        # small kappa = lumpy MDA/MALBAC amplification; beta-binom alleles + ADO + doublets.
        return dict(kappa=5.0, nb_dispersion=0.5, ado_rate=0.20,
                    beta_binom_conc=30.0, doublet_rate=0.02)

    def _select_cells(self, cells_index, cell_subset):
        if cell_subset is not None:
            return np.asarray(list(cell_subset))
        n = min(self.n_cells, len(cells_index))
        rng = np.random.default_rng(self.seed)
        return rng.choice(np.asarray(cells_index), size=n, replace=False)

    def _apply_doublets(self, cnv, af, cells, rng):
        """Merge a `doublet_rate` fraction of cells with a random partner (copy-number level).

        Combined CN = CN_a + CN_b; combined alt fraction = (alt_a + alt_b) / combined CN.
        """
        n = len(cells)
        is_doublet = np.zeros(n, dtype=bool)
        rate = self.hypers.doublet_rate
        if rate <= 0 or n < 2:
            return cnv, af, is_doublet
        n_dbl = int(round(rate * n))
        if n_dbl == 0:
            return cnv, af, is_doublet
        cnv = cnv.copy()
        af = af.copy()
        idx = rng.choice(n, size=n_dbl, replace=False)
        partners = rng.choice(n, size=n_dbl, replace=True)
        for i, p in zip(idx, partners):
            if p == i:
                p = (p + 1) % n
            comb_cn = cnv[i] + cnv[p]
            comb_alt = af[i] * cnv[i] + af[p] * cnv[p]
            af[i] = np.divide(comb_alt, comb_cn, out=np.zeros_like(comb_cn), where=comb_cn > 0)
            cnv[i] = comb_cn
            is_doublet[i] = True
        return cnv, af, is_doublet

    def run(self, cell_data, cell_subset=None, **kwargs):
        snv = cell_data["cell_snv"]
        cells = self._select_cells(snv.index, cell_subset)
        self.n_cells = len(cells)
        genes_all = list(snv.columns)
        obs = self._select_loci(genes_all)
        genes = [genes_all[i] for i in obs]

        cnv, af_true = self._ground_truth(cell_data, cells, genes)       # (n_cells, n_loci)
        batch = self._make_batch(genes)
        cnv, af_dbl, is_doublet = self._apply_doublets(cnv, af_true, cells, batch.rng)

        # per-cell amplification: independent DM draw per cell on CN-driven weights.
        N_c = int(round(self.hypers.mu_depth * len(genes)))
        eff = batch.efficiency[None, :]
        coverage = np.vstack([
            batch.emit_depth(cnv[i] * batch.efficiency, N_c) for i in range(self.n_cells)
        ]).astype(int)

        # allele layer: ADO (one allele lost at a het locus) then Beta-Binomial + error.
        af_obs, ado_mask = batch.apply_ado(af_dbl)
        alt = batch.alleles_betabinom(coverage, af_obs).astype(int)
        vaf = np.divide(alt, coverage, out=np.zeros(coverage.shape), where=coverage > 0)

        self.batch = batch
        self.cells = cells
        self.genes = genes
        self.is_doublet = is_doublet
        self.ado_mask = pd.DataFrame(ado_mask, index=cells, columns=genes)
        self.true_cn = pd.DataFrame(cnv, index=cells, columns=genes)
        self.true_alt_fraction = pd.DataFrame(af_dbl, index=cells, columns=genes)
        self.coverage = pd.DataFrame(coverage, index=cells, columns=genes)
        self.alt_counts = pd.DataFrame(alt, index=cells, columns=genes)
        self.vaf = pd.DataFrame(vaf, index=cells, columns=genes)

        if self.data_mode == "binary":
            # genotype call after ADO, flipped at the per-locus error rate.
            base = (af_obs > 0).astype(int)
            flips = batch.rng.random(base.shape) < batch.error[None, :]
            obs_snv = np.where(flips, 1 - base, base)
            self.observed_snvs = pd.DataFrame(obs_snv.astype(int), index=cells, columns=genes)

        self._build_obs(cell_data, cells)
        return self

    def _build_obs(self, cell_data, cells):
        obs = pd.DataFrame(index=pd.Index(cells, name="cell"))
        obs["batch"] = self.batch.label
        obs["breadth"] = self.breadth
        obs["is_doublet"] = self.is_doublet
        obs["mean_coverage"] = self.coverage.mean(axis=1).values
        obs["ado_frac"] = self.ado_mask.mean(axis=1).values
        ctype = cell_data.get("cell_type")
        if ctype is not None:
            obs["clone"] = ctype.reindex(cells).iloc[:, 0].astype(str).values
        crd = cell_data.get("cell_crd")
        if crd is not None:
            crd = crd.reindex(cells)
            obs["row"] = crd["row"].values
            obs["col"] = crd["col"].values
        self.obs = obs

    def to_anndata(self):
        """cells × loci AnnData: X = coverage; alt/vaf/true_cn/true_af/ado in layers."""
        import anndata as ad

        adata = ad.AnnData(
            X=self.coverage.values.astype(np.float32),
            obs=self.obs.copy(),
            var=pd.DataFrame({"segment": [_segment_of(g) for g in self.genes],
                              "gc": self.gc, "mappability": self.mappability},
                             index=pd.Index(self.genes, name="locus")),
        )
        adata.layers["alt_counts"] = self.alt_counts.values.astype(np.float32)
        adata.layers["vaf"] = self.vaf.values.astype(np.float32)
        adata.layers["true_cn"] = self.true_cn.values.astype(np.float32)
        adata.layers["true_alt_fraction"] = self.true_alt_fraction.values.astype(np.float32)
        adata.layers["ado"] = self.ado_mask.values.astype(np.float32)
        if {"row", "col"} <= set(self.obs.columns):
            adata.obsm["spatial"] = self.obs[["row", "col"]].values.astype(float)
        adata.uns["assay"] = "scDNA"
        adata.uns["breadth"] = self.breadth
        adata.uns["hyperparams"] = self.hypers.to_dict()
        return adata

    def write(self, out_path, write_h5ad=True, **kwargs):
        os.makedirs(out_path, exist_ok=True)
        if self.data_mode == "binary":
            self.observed_snvs.to_csv(os.path.join(out_path, "observed_snvs.csv"))
        # `coverage.csv` + `alt_counts.csv` kept as the stable on-disk contract.
        self.coverage.to_csv(os.path.join(out_path, "coverage.csv"))
        self.alt_counts.to_csv(os.path.join(out_path, "alt_counts.csv"))
        if write_h5ad:
            try:
                self.to_anndata().write_h5ad(os.path.join(out_path, f"{self.batch.label}.h5ad"))
            except Exception:
                pass


# --------------------------------------------------------------------------------------
# Multi-batch DNA (DESIGN_features §B.3 / §D): same biology, different technical signature.
# --------------------------------------------------------------------------------------
def run_dna_batches(cell_data, mode="bulk", n_batches=2, base_seed=42, design="shared",
                    breadth="wgs", n_cells=100, **dna_kwargs):
    """Generate `n_batches` DNA batches that share biology but differ technically.

    `mode` ∈ {"bulk", "sc"}. designs mirror scRNA (`run_scrna_batches`):
      * "shared" — every batch measures the SAME cells (replicate concordance).
      * "split"  — the sampled cells are partitioned across batches (single-cell only;
                   bulk pools all cells so "split" falls back to "shared").
    Returns the list of run assay instances (each with `.batch.label` and `.to_anndata()`).
    """
    cls = {"bulk": bulkDNA, "sc": scDNA}[mode]
    snv = cell_data["cell_snv"]
    sel_rng = np.random.default_rng(base_seed)

    if mode == "sc":
        n_avail = len(snv.index)
        if design == "split":
            n = min(n_cells * n_batches, n_avail)
            chosen = sel_rng.choice(np.asarray(snv.index), size=n, replace=False)
            subsets = [np.asarray(s) for s in np.array_split(chosen, n_batches)]
        else:
            n = min(n_cells, n_avail)
            chosen = sel_rng.choice(np.asarray(snv.index), size=n, replace=False)
            subsets = [chosen for _ in range(n_batches)]
    else:
        subsets = [None for _ in range(n_batches)]  # bulk pools all cells

    assays = []
    for i, subset in enumerate(subsets):
        kw = dict(dna_kwargs)
        if mode == "sc":
            kw["n_cells"] = len(subset)
        assay = cls(breadth=breadth, seed=base_seed + 1 + i, batch_label=f"batch{i}", **kw)
        assay.run(cell_data, cell_subset=subset)
        assays.append(assay)
    return assays
