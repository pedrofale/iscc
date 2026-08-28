"""One scorer for the truncal-CNA calibration, so every configuration is judged the same way.

The question this module answers is: **did the duct constraint produce a truncal copy-number
alteration, and is the tumour that produced it still a realistic breast tumour?**

Both halves matter. A configuration that sweeps a segment to fixation by making the lesion
monoclonal, or diploid, or never letting it out of the duct, has not demonstrated anything — it has
just broken the biology in a different place. So :func:`score_tumour` returns the SWEEP numbers and
the REALISM numbers side by side, and :func:`summarise` prints them as one table.

Everything is read straight from the per-genotype counts (``genotypes_counts`` / ``genotypes`` /
``demes``), the same read-only interface ``iscc.tumor.diagnostics`` uses. That makes the score
**exact over every cancer cell** — not over a materialised subsample — and cheap enough to call
dozens of times: scoring is milliseconds, so a calibration sweep pays only for growth.
"""
from collections import Counter

import numpy as np

try:                                        # validation/ on sys.path (the convention in this dir)
    import realistic_regime as RR
except ImportError:                         # repo root on sys.path
    from validation import realistic_regime as RR

#: A segment state at or above this frequency among cancer cells counts as TRUNCAL.
TRUNCAL_FREQ = 0.9

#: Realism windows for a breast carcinoma. A configuration outside these is flagged, not rejected —
#: the flag is a note in the table, the caller decides.
TARGETS = {
    "frac_genome_altered": (0.20, 0.40),
    "mean_ploidy": (2.4, 3.0),
    "pct_wgd": (20.0, 50.0),
}

#: Clone-count window. Below the low bound the lesion is degenerate (a handful of clones, so any
#: "sweep" is trivial); above the high bound the genotype registry has exploded and the run is
#: neither tractable nor meaningful.
CLONE_RANGE = (5, 50_000)

#: A lesion with less than this percent of its cancer in the stroma has not invaded — the breach gate
#: never opened, so the DCIS->IDC biology is absent however well the tumour swept.
MIN_STROMA_PCT = 1.0

#: Default growth targets per scale for :func:`grow_and_score` — big enough for the frequencies to
#: mean something, small enough to iterate on.
DEFAULT_TARGET = {"small": 12_000, "mid": 40_000, "cm": 150_000}


def _cancer_clones(t):
    """``[(genotype_id, n_cells), ...]`` for the live CANCER genotypes only."""
    return [(g, int(c)) for g, c in t.genotypes_counts.items() if c > 0 and t._is_cancer(g)]


def score_tumour(t, truncal_freq=TRUNCAL_FREQ):
    """Score one grown :class:`GenotypeTumor` for SWEEP and for REALISM.

    Does not need ``cell_data``: every number below is computed from the per-genotype cell counts, so
    it covers every cancer cell exactly and costs nothing to run on a cm-scale tumour.

    Returns
    -------
    dict
        Sweep keys

        ``max_cna_freq``
            Over segments, the largest fraction of cancer cells sharing ONE non-diploid state
            (copy number != 2) at that segment. This is the headline number: 1.0 means a
            copy-number alteration is in every cancer cell, ~0.4 means the best any alteration
            managed was a minority subclone.
        ``n_truncal_segments``
            How many segments carry such a state at or above ``truncal_freq``.
        ``max_cna_freq_vs_clonal`` / ``n_truncal_segments_vs_clonal``
            The same two numbers measured against ``clonal_ploidy`` (the population's modal copy
            number) instead of against 2. A tumour that sweeps a whole-genome doubling reads as
            twelve truncal segments on the plain measure while carrying no segment-level event at
            all; these two see through that, so a sweep that is real shows up on BOTH pairs.
        ``max_snv_freq``
            Largest fraction of cancer cells carrying a point mutation at any one site. The control:
            truncal SNVs already work through founder seeding, so this should sit at ~1.0 and its
            falling is a sign the configuration broke something basic.
        ``clonal_cn_profile``
            Per-segment list of the copy number shared by more than 90% of cancer cells, with
            ``None`` in the segments that have no such consensus (``None`` for the whole profile
            when there are no cancer cells). This is the profile a bulk caller would report as the
            tumour's clonal state — the truncal alterations are its non-``clonal_ploidy`` entries.
        ``clonal_ploidy``
            The population's modal segment copy number (2 for a diploid-baseline tumour, 4 after a
            clonal doubling).

        Realism keys

        ``frac_genome_altered``
            Mean over cancer cells of the fraction of the genome sitting off THAT cell's own modal
            copy number (segment-length weighted, so it is a genome fraction rather than a segment
            count). Breast target ~0.2-0.4.
        ``mean_ploidy``
            Cell-count-weighted mean ploidy. Breast target ~2.4-3.0.
        ``pct_wgd``
            Percent of cancer cells descended from a whole-genome doubling. Breast target ~20-50%.
        ``n_clones``
            Live cancer genotype count (see :data:`CLONE_RANGE`).
        ``stroma_pct``
            Percent of cancer cells in the stroma — the DCIS->IDC invasion readout. A configuration
            that never invades has broken the biology even if it sweeps.
        ``n_glands_colonised``, ``n_cancer``, ``purity_of_whole_lesion``
            Glands containing any cancer; total cancer cells; cancer as a fraction of ALL live cells
            in the field (what bulk-sequencing the whole specimen would give).
        ``flags``
            Space-joined short names of the realism windows this tumour falls outside — the only
            non-numeric entry besides ``clonal_cn_profile``. Empty string means nothing tripped.
    """
    clones = _cancer_clones(t)
    n_cancer = int(sum(c for _, c in clones))
    n_seg = int(t.n_segments)
    seg_sizes = np.asarray(t.selection.segment_sizes, dtype=float)
    genome_len = float(seg_sizes.sum()) if seg_sizes.size else 1.0

    if n_cancer == 0:
        nan = float("nan")
        return dict(max_cna_freq=nan, n_truncal_segments=0, max_cna_freq_vs_clonal=nan,
                    n_truncal_segments_vs_clonal=0, max_snv_freq=nan, clonal_cn_profile=None,
                    clonal_ploidy=nan, frac_genome_altered=nan, mean_ploidy=nan, pct_wgd=nan,
                    n_clones=0, stroma_pct=nan, n_glands_colonised=0, n_cancer=0,
                    purity_of_whole_lesion=nan, flags="extinct")

    # --- one pass over the clones: per-segment state counts + the realism aggregates -------------
    seg_states = [Counter() for _ in range(n_seg)]   # segment -> {copy number: n cancer cells}
    pop_cn = Counter()                               # copy number -> genome-length-weighted cells
    fga = ploidy = wgd = 0.0
    for gid, cnt in clones:
        gs = t.genotypes[gid].genome_summary
        cns = np.asarray(gs["seg_cns"], dtype=int)
        for s in range(n_seg):
            seg_states[s][int(cns[s])] += cnt
        # The cell's OWN baseline is its modal copy number, so a doubled cell is not counted as
        # having 100% of its genome altered. Length-weighted, so an unequal (real-genome) segment
        # map still yields a genome FRACTION.
        hist = np.bincount(cns, weights=seg_sizes)
        modal = int(np.argmax(hist))
        fga += cnt * float(seg_sizes[cns != modal].sum()) / genome_len
        ploidy += cnt * float(gs["ploidy"])
        if getattr(t.genotypes[gid], "is_wgd", False):
            wgd += cnt
        for cn, w in enumerate(hist):          # reuse the length-weighted per-clone histogram
            if w:
                pop_cn[cn] += cnt * float(w)

    clonal_ploidy = int(max(pop_cn.items(), key=lambda kv: kv[1])[0])

    # --- sweep ------------------------------------------------------------------------------------
    def _best(baseline):
        """(largest non-``baseline`` state frequency, #segments at/above the truncal threshold)."""
        best, n_trunc = 0.0, 0
        for states in seg_states:
            off = [c for cn, c in states.items() if cn != baseline]
            if not off:
                continue
            f = max(off) / n_cancer
            best = max(best, f)
            if f >= truncal_freq:
                n_trunc += 1
        return float(best), int(n_trunc)

    max_cna_freq, n_truncal = _best(2)
    max_cna_clonal, n_truncal_clonal = _best(clonal_ploidy)

    profile = []
    for states in seg_states:
        cn, c = max(states.items(), key=lambda kv: kv[1])
        profile.append(int(cn) if c / n_cancer > 0.9 else None)

    # Largest fraction of cancer cells carrying a point mutation at any one site. `get_snvs()` is the
    # per-genotype VAF vector, so `> 0` is "this clone carries the variant"; the founder's own
    # mutations are in every clone and put this at ~1.0.
    n_genes = int(getattr(t, "n_genes", t.n_segments * t.segment_size))
    carriers = np.zeros(n_genes, dtype=float)
    for gid, cnt in clones:
        carriers += cnt * (t.genotypes[gid].get_snvs() > 0)
    max_snv_freq = float(carriers.max() / n_cancer)

    # --- realism ----------------------------------------------------------------------------------
    fga /= n_cancer
    ploidy /= n_cancer
    pct_wgd = 100.0 * wgd / n_cancer
    n_clones = len(clones)
    has_field = getattr(t, "gland_id", None) is not None
    stroma_pct = RR.stroma_cancer_pct(t) if has_field else float("nan")
    n_glands = len(RR.glands_colonised(t)) if has_field else 0
    total = int(t.get_tumor_size())
    purity = float(n_cancer / total) if total else float("nan")

    flags = []
    if n_clones < CLONE_RANGE[0]:
        flags.append("degenerate-clones")
    if n_clones > CLONE_RANGE[1]:
        flags.append("clone-explosion")
    if has_field and stroma_pct < MIN_STROMA_PCT:
        flags.append("no-invasion")
    for key, value in (("frac_genome_altered", fga), ("mean_ploidy", ploidy), ("pct_wgd", pct_wgd)):
        lo, hi = TARGETS[key]
        short = {"frac_genome_altered": "fga", "mean_ploidy": "ploidy", "pct_wgd": "wgd"}[key]
        if value < lo:
            flags.append(short + "-low")
        elif value > hi:
            flags.append(short + "-high")

    return dict(
        max_cna_freq=max_cna_freq, n_truncal_segments=n_truncal,
        max_cna_freq_vs_clonal=max_cna_clonal, n_truncal_segments_vs_clonal=n_truncal_clonal,
        max_snv_freq=max_snv_freq, clonal_cn_profile=profile, clonal_ploidy=clonal_ploidy,
        frac_genome_altered=float(fga), mean_ploidy=float(ploidy), pct_wgd=float(pct_wgd),
        n_clones=int(n_clones), stroma_pct=float(stroma_pct), n_glands_colonised=int(n_glands),
        n_cancer=int(n_cancer), purity_of_whole_lesion=purity, flags=" ".join(flags),
    )


# Table layout: (column header, score key, format). `truncD` is the vs-clonal truncal count, so a
# clonal doubling (which fills `trunc` with all 12 segments while altering no segment relative to the
# tumour's own baseline) is visible as trunc high / truncD zero.
_COLUMNS = (
    ("n_cancer", "n_cancer", "{:d}"),
    ("clones", "n_clones", "{:d}"),
    ("maxCNA", "max_cna_freq", "{:.3f}"),
    ("trunc", "n_truncal_segments", "{:d}"),
    ("truncD", "n_truncal_segments_vs_clonal", "{:d}"),
    ("maxSNV", "max_snv_freq", "{:.3f}"),
    ("FGA", "frac_genome_altered", "{:.3f}"),
    ("ploidy", "mean_ploidy", "{:.2f}"),
    ("%WGD", "pct_wgd", "{:.1f}"),
    ("stroma%", "stroma_pct", "{:.1f}"),
    ("glands", "n_glands_colonised", "{:d}"),
    ("purity", "purity_of_whole_lesion", "{:.2f}"),
    ("flags", "flags", "{}"),
)


def _cell(fmt, value):
    if value is None:
        return "n/a"
    if isinstance(value, float) and np.isnan(value):
        return "n/a"
    try:
        return fmt.format(value)
    except (ValueError, TypeError):
        return str(value)


def summarise(rows):
    """A compact aligned text table over ``[(label, score_dict), ...]``, ready to paste in a report.

    The first two numeric columns are the verdict — ``maxCNA`` is the largest fraction of cancer
    cells sharing one copy-number state and ``trunc`` how many segments cleared 0.9 — and everything
    right of ``maxSNV`` says whether the tumour that produced them is still realistic.
    """
    rows = list(rows)
    header = ["config"] + [h for h, _, _ in _COLUMNS]
    table = [header]
    for label, s in rows:
        table.append([str(label)] + [_cell(fmt, s.get(key)) for _, key, fmt in _COLUMNS])
    widths = [max(len(r[i]) for r in table) for i in range(len(header))]
    # Label column left-aligned (names read better flush left), numbers right-aligned.
    def line(cells):
        out = [cells[0].ljust(widths[0])] + [c.rjust(w) for c, w in zip(cells[1:], widths[1:])]
        return "  ".join(out).rstrip()
    lines = [line(header), "  ".join("-" * w for w in widths)]
    lines += [line(r) for r in table[1:]]
    lo, hi = TARGETS["frac_genome_altered"]
    p_lo, p_hi = TARGETS["mean_ploidy"]
    w_lo, w_hi = TARGETS["pct_wgd"]
    lines.append(f"targets: FGA {lo:.1f}-{hi:.1f}  ploidy {p_lo:.1f}-{p_hi:.1f}  "
                 f"%WGD {w_lo:.0f}-{w_hi:.0f}  clones {CLONE_RANGE[0]}-{CLONE_RANGE[1]}  "
                 f"stroma% >= {MIN_STROMA_PCT:.0f}")
    lines.append("trunc = segments with a non-diploid state in >=90% of cancer cells; "
                 "truncD = same vs the tumour's own modal ploidy")
    return "\n".join(lines)


def grow_and_score(label, seed=3, scale="small", target_cancer=None, materialise=False,
                   truncal_freq=TRUNCAL_FREQ, **regime_overrides):
    """Grow one realistic tumour under ``regime_overrides`` and score it: ``(label, score_dict)``.

    A thin wrapper over ``realistic_regime.grow_realistic`` — ``deme=``, ``selection=``, ``cancer=``,
    ``spatial=``, ``genome=`` and the rest pass straight through, so a calibration reads as one line
    per configuration::

        rows = [grow_and_score("shipped"),
                grow_and_score("fixed crowding", deme={"crowding_mode": "fixed"})]
        print(summarise(rows))

    ``target_cancer`` defaults per scale (see :data:`DEFAULT_TARGET`). ``materialise`` is off because
    the score reads the counts directly — turning it on only costs time and adds subsampling noise to
    nothing, but it is here for a caller that wants ``cell_data`` on the same grown tumour.
    """
    if target_cancer is None:
        target_cancer = DEFAULT_TARGET.get(scale)
        if target_cancer is None:
            raise ValueError(f"no default target_cancer for scale {scale!r}; pass one explicitly")
    t = RR.grow_realistic(seed=seed, target_cancer=target_cancer, scale=scale,
                          materialise=materialise, **regime_overrides)
    return label, score_tumour(t, truncal_freq=truncal_freq)
