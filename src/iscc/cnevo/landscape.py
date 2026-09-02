"""The copy-number landscape and how it evolves (Q4).

FGA, ploidy, LOH, nullisomy and whole-genome-doubling burden are the summary statistics a
copy-number study reports about a tumour. Because ``tumor.genotypes`` retains every genotype and
``tumor.traces`` carries exact per-generation counts, each of them is available not just at the
end of the run but **at every generation**, count-weighted, with no re-simulation.

The trajectory's last row reproduces :func:`iscc.tumor.diagnostics.cna_stats` by construction —
the same segment-length-weighted FGA and the same count-weighted ploidy — so this module extends
the existing endpoint diagnostics rather than offering a second, quietly different definition of
them. The endpoint block additionally reuses :func:`iscc.inference.summaries.cna_summary` for the
per-segment gain/loss recurrence spectrum.
"""
import numpy as np
import pandas as pd

from ..inference.summaries import cna_summary
from ..validation import segment_driver_content
from .events import cna_event_table

__all__ = ["cn_landscape"]


def _genotype_cn_cache(tumor):
    """``gid -> (seg_cns, allele_cn or None, ploidy, is_wgd)`` for every cancer genotype."""
    cache = {}
    for gid, rep in tumor.genotypes.items():
        if not tumor._is_cancer(gid):
            continue
        gs = getattr(rep, "genome_summary", None)
        if gs is None:
            continue
        seg = np.asarray(gs["seg_cns"], dtype=float)
        allele = (np.array([(len(s["p"]), len(s["m"])) for s in rep.genome], dtype=float)
                  if getattr(rep, "genome", None) else None)
        cache[gid] = (seg, allele, float(gs.get("ploidy", np.nan)),
                      bool(getattr(rep, "is_wgd", False)))
    return cache


def cn_landscape(tumor, baseline=2.0, stride=1):
    """Copy-number landscape over time, plus an endpoint summary.

    Parameters
    ----------
    baseline : float
        Copy number counted as unaltered (2 = diploid).
    stride : int
        Evaluate every ``stride``-th snapshot.

    Returns
    -------
    (trajectory, summary) : (DataFrame, dict)
        ``trajectory`` has ``gen, n_cells, fga, mean_ploidy, ploidy_sd, mean_cn, max_cn,
        frac_segments_loh, frac_segments_nullisomy, wgd_frac``.
        ``summary`` carries the endpoint spectrum (``gain_freq`` / ``loss_freq`` per segment),
        the CN-selection correlation, and event-rate statistics.
    """
    traces = tumor.traces or []
    cache = _genotype_cn_cache(tumor)
    sizes = np.asarray(tumor.selection.segment_sizes, dtype=float)
    genome_len = float(sizes.sum()) if sizes.size else 1.0
    times = getattr(tumor, "trace_times", None)
    gens = (np.asarray(times, dtype=float) if times is not None and len(times) == len(traces)
            else np.arange(len(traces), dtype=float))

    rows = []
    for t in range(0, len(traces), max(1, int(stride))):
        counts = {g: c for g, c in traces[t]["genotypes_counts"].items()
                  if c > 0 and g in cache}
        n = float(sum(counts.values()))
        if n <= 0:
            rows.append((float(gens[t]), 0.0, *([np.nan] * 8)))
            continue
        fga = ploidy = ploidy2 = mean_cn = loh = null = wgd = 0.0
        mx = -np.inf
        for gid, cnt in counts.items():
            seg, allele, gp, is_wgd = cache[gid]
            w = cnt / n
            # segment-LENGTH-weighted altered fraction: matches diagnostics.cna_stats
            fga += w * float(sizes[seg != baseline].sum()) / genome_len
            ploidy += w * gp
            ploidy2 += w * gp * gp
            mean_cn += w * float(seg.mean())
            mx = max(mx, float(seg.max()))
            null += w * float((seg <= 0).mean())
            if allele is not None:
                imbalanced_zero = ((allele.min(axis=1) <= 0) & (seg > 0))
                loh += w * float(imbalanced_zero.mean())
            else:
                loh = np.nan
            wgd += w * float(is_wgd)
        var = max(ploidy2 - ploidy * ploidy, 0.0)
        rows.append((float(gens[t]), n, fga, ploidy, float(np.sqrt(var)), mean_cn,
                     mx if np.isfinite(mx) else np.nan, loh, null, wgd))

    traj = pd.DataFrame(rows, columns=["gen", "n_cells", "fga", "mean_ploidy", "ploidy_sd",
                                       "mean_cn", "max_cn", "frac_segments_loh",
                                       "frac_segments_nullisomy", "wgd_frac"])

    endpoint = cna_summary(tumor, baseline=baseline)
    events = cna_event_table(tumor)
    last_gen = float(gens[-1]) if len(gens) else float("nan")
    n_ev = int(len(events))
    n_amp = int((events["type"] == "amplification").sum()) if n_ev else 0
    n_del = int((events["type"] == "deletion").sum()) if n_ev else 0
    n_wgd = int((events["type"] == "wgd").sum()) if n_ev else 0

    summary = dict(
        gain_freq=np.asarray(endpoint["gain_freq"]).tolist(),
        loss_freq=np.asarray(endpoint["loss_freq"]).tolist(),
        fga_endpoint=float(endpoint["fga"]),
        ploidy_mean_endpoint=float(endpoint["ploidy_mean"]),
        ploidy_std_endpoint=float(endpoint["ploidy_std"]),
        n_events=n_ev, n_amplifications=n_amp, n_deletions=n_del, n_wgd_events=n_wgd,
        amp_del_ratio=(float(n_amp) / n_del) if n_del else float("nan"),
        driver_event_frac=(float(events["driver"].mean()) if n_ev else float("nan")),
        event_rate_per_gen=(n_ev / last_gen) if last_gen and np.isfinite(last_gen) and last_gen > 0
        else float("nan"),
        cn_selection_corr=_cn_selection_corr(tumor),
    )
    if not traj.empty:
        summary.update(fga_final=float(traj["fga"].iloc[-1]),
                       mean_ploidy_final=float(traj["mean_ploidy"].iloc[-1]),
                       wgd_frac_final=float(traj["wgd_frac"].iloc[-1]))
    return traj, summary


def _cn_selection_corr(tumor):
    """Correlation of per-segment population mean CN with net oncogenic content.

    Positive means selection is visibly shaping the CN landscape in this regime — oncogene-rich
    segments amplified, TSG-rich ones lost (Beroukhim 2010 / Davoli 2013, the signature
    ``validation/validate_cna.py`` checks). ``nan`` if either side is constant, which is the
    normal outcome for a neutral run.
    """
    from ..validation import segment_copy_numbers
    net = np.asarray(segment_driver_content(tumor), dtype=float)
    cn = np.asarray(segment_copy_numbers(tumor), dtype=float)
    if net.size < 2 or net.std() == 0 or cn.std() == 0 or not np.all(np.isfinite(cn)):
        return float("nan")
    return float(np.corrcoef(net, cn)[0, 1])
