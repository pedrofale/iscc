"""Summary statistics for ABC inference on the genotype-count engine (DESIGN_inference A.2).

ABC compresses a whole simulated tumour to a short vector of summaries; the inference engine
(``abc.py``) then matches simulated vectors to an observed one. The summaries here are the
CINner analogue — per-segment copy-number gain/loss frequencies plus fraction-genome-altered
and ploidy — together with site-frequency-spectrum features of the SNV burden. They reuse the
existing validation metrics (``segment_copy_numbers``, ``population_vaf``, ``neutral_sfs_rsq``)
so the inference layer and the qualitative validations measure the same quantities.

A "frequency" here is *across the cancer-cell population* of a single tumour: e.g. the gain
frequency of a segment is the fraction of cancer cells whose copy number of that segment exceeds
the diploid baseline. This mirrors CINner's per-chromosome-arm gain/loss frequencies summed over
PCAWG samples, at the single-tumour level used for parameter recovery.
"""
import numpy as np

from ..validation import population_vaf, neutral_sfs_rsq


def _cancer_seg_cn_matrix(tumor):
    """Per-cancer-genotype segment copy numbers and their cell counts.

    Returns ``(cn, weights)`` where ``cn`` is ``(n_genotypes, n_segments)`` and ``weights`` the
    matching cell counts. Empty arrays when there are no cancer cells.
    """
    cn, w = [], []
    for gid, cnt in tumor.genotypes_counts.items():
        if not tumor._is_cancer(gid):
            continue
        cn.append(np.asarray(tumor.genotypes[gid].genome_summary["seg_cns"], dtype=float))
        w.append(cnt)
    if not cn:
        return np.empty((0, tumor.n_segments)), np.empty((0,))
    return np.vstack(cn), np.asarray(w, dtype=float)


def cna_summary(tumor, baseline=2.0):
    """Per-segment CNA gain/loss frequencies + genome-level CNA load (the CINner analogue).

    Returns ``dict`` with:
      * ``gain_freq`` ``(n_segments,)`` — fraction of cancer cells with ``CN > baseline``;
      * ``loss_freq`` ``(n_segments,)`` — fraction with ``CN < baseline``;
      * ``fga`` — fraction genome altered = mean over cells of the fraction of segments off-baseline;
      * ``ploidy_mean`` / ``ploidy_std`` — population mean/SD of per-cell mean copy number.
    All ``nan`` (and FGA/ploidy ``nan``) when there are no cancer cells.
    """
    cn, w = _cancer_seg_cn_matrix(tumor)
    nseg = tumor.n_segments
    if cn.shape[0] == 0:
        nanseg = np.full(nseg, np.nan)
        return dict(gain_freq=nanseg, loss_freq=nanseg.copy(),
                    fga=float("nan"), ploidy_mean=float("nan"), ploidy_std=float("nan"))
    p = w / w.sum()
    gain_freq = (p[:, None] * (cn > baseline)).sum(axis=0)
    loss_freq = (p[:, None] * (cn < baseline)).sum(axis=0)
    altered_frac = (cn != baseline).mean(axis=1)            # per genotype, over segments
    fga = float((p * altered_frac).sum())
    ploidy = cn.mean(axis=1)                                # per genotype
    ploidy_mean = float((p * ploidy).sum())
    ploidy_std = float(np.sqrt((p * (ploidy - ploidy_mean) ** 2).sum()))
    return dict(gain_freq=gain_freq, loss_freq=loss_freq,
                fga=fga, ploidy_mean=ploidy_mean, ploidy_std=ploidy_std)


def snv_summary(tumor, vaf_quantiles=(0.25, 0.5, 0.75, 0.9)):
    """SNV burden / site-frequency-spectrum features of the cancer population's bulk VAFs.

    Returns ``dict`` with:
      * ``n_sites`` — number of segregating sites (population VAF > 0), normalised by genome size;
      * ``mean_vaf`` — mean VAF over segregating sites;
      * ``vaf_quantiles`` — the requested quantiles of the segregating-site VAFs;
      * ``sfs_rsq`` — goodness-of-fit of the cumulative SFS to the neutral 1/f law
        (``validation.neutral_sfs_rsq``); a selection vs neutrality signal.
    """
    vaf = population_vaf(tumor, cancer_only=True)
    seg = vaf[vaf > 0]
    genome_size = tumor.n_segments * tumor.segment_size
    if seg.size == 0:
        return dict(n_sites=0.0, mean_vaf=float("nan"),
                    vaf_quantiles=np.full(len(vaf_quantiles), np.nan), sfs_rsq=float("nan"))
    rsq, _ = neutral_sfs_rsq(seg)
    return dict(
        n_sites=float(seg.size) / genome_size,
        mean_vaf=float(seg.mean()),
        vaf_quantiles=np.quantile(seg, vaf_quantiles),
        sfs_rsq=float(rsq),
    )


def summary_vector(tumor, include_snv=True, baseline=2.0):
    """Flatten a tumour's CNA (+ optional SNV) summaries into one numeric vector + names.

    Returns ``(values, names)`` with ``values`` a 1-D ``float`` array and ``names`` the matching
    labels (so a summary table stays interpretable). Non-finite entries (e.g. an extinct tumour)
    are passed through as ``nan`` and handled by the ABC engine's distance scaling.
    """
    cna = cna_summary(tumor, baseline=baseline)
    values, names = [], []
    for i, g in enumerate(cna["gain_freq"]):
        values.append(g); names.append(f"gain_freq[{i}]")
    for i, l in enumerate(cna["loss_freq"]):
        values.append(l); names.append(f"loss_freq[{i}]")
    values += [cna["fga"], cna["ploidy_mean"], cna["ploidy_std"]]
    names += ["fga", "ploidy_mean", "ploidy_std"]
    if include_snv:
        snv = snv_summary(tumor)
        values += [snv["n_sites"], snv["mean_vaf"], snv["sfs_rsq"]]
        names += ["n_sites", "mean_vaf", "sfs_rsq"]
        for q, v in zip((0.25, 0.5, 0.75, 0.9), snv["vaf_quantiles"]):
            values.append(float(v)); names.append(f"vaf_q{int(q * 100)}")
    return np.asarray(values, dtype=float), names
