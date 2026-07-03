"""Operating-envelope QC — read-only diagnosis of a grown tumour (DESIGN_operating_envelope.md).

After growth, :func:`diagnose` computes phenotype metrics and flags each **degenerate regime**
(extinct / monoclonal / hypermutated / well-mixed / no-microenvironment-gradient / CNA-runaway /
trivial-genome) against an overridable threshold, with an ACTIONABLE hint pointing at the culprit
knob. It answers the user question "did my run produce a crappy tumour?" and protects the headline
benchmarks: the *well-mixed* and *no-gradient* regimes silently break the PEtracer lineage–space
confound and the multi-region-phylogeny demos, so those get their own flags.

This is a **read-only readout** (like the F8 microenvironment ground truth): it never draws from the
tumour's rng and never mutates the counts, so it cannot change simulation output. It reads the grown
state through the genotype-count interface (``genotypes_counts`` / ``genotypes`` / ``_is_cancer`` /
``demes`` / ``deme_coords`` / ``selection``), which both the count and cell engines expose.
"""
import numpy as np

from ..validation import clone_diversity, population_vaf, neutral_sfs_rsq

# Default thresholds (DESIGN_operating_envelope.md). Lenient on purpose: flag the clearly broken,
# not the merely unusual. Overridable per call: ``tumor.diagnose(thresholds={"shannon_min": 1.0})``.
DEFAULT_THRESHOLDS = {
    "min_cancer": 25,        # < this many cancer cells -> extinct / too small (hard fail)
    "min_realistic": 1000,   # < this many cancer cells -> advisory: grow to a realistic size
    "shannon_min": 0.5,      # clonal Shannon diversity below this -> monoclonal
    "subclone_freq": 0.01,   # a genotype counts as a "subclone" above this frequency
    "tmb_min": 1.0,          # mean mutated sites/cell below this -> no-mutation monoclonal
    "tmb_frac_max": 0.5,     # fraction of the genome mutated/cell above this -> hypermutated mush
    "sweep_enrichment": 1.5, # driver-vs-passenger VAF ratio above this -> a selective sweep
    "confinement_min": 0.1,  # clone spatial confinement below this (>=2 subclones) -> well-mixed
    "contrast_min": 0.05,    # hypoxia core-rim contrast below this -> no O2 gradient
    "fga_max": 0.95,         # fraction-genome-altered above this -> CNA runaway
    "min_genes": 100,        # fewer than this many genes -> trivial genome
    "vaf_1f_rsq": None,      # reported only (context-dependent under selection); not a hard gate
}


class Check:
    """One QC check: whether the tumour passed a threshold, with an actionable hint if not."""

    def __init__(self, name, ok, value, threshold, hint="", skipped=False):
        self.name = name
        self.ok = bool(ok)
        self.value = value
        self.threshold = threshold
        self.hint = hint
        self.skipped = bool(skipped)

    def __repr__(self):
        if self.skipped:
            return f"Check({self.name}: skipped)"
        return f"Check({self.name}: {'ok' if self.ok else 'FAIL'}, value={self.value})"


class TumorDiagnosis:
    """Structured QC report: metric values + per-regime checks. ``ok`` iff nothing failed."""

    def __init__(self, metrics, checks, advisories=None):
        self.metrics = dict(metrics)
        self.checks = list(checks)
        # Non-failing notes (e.g. "small tumour: grow to a realistic size"). A small-but-healthy
        # tumour is not degenerate, so advisories do NOT flip ``ok`` — they only surface in the report.
        self.advisories = list(advisories or [])

    @property
    def ok(self):
        return all(c.ok for c in self.checks if not c.skipped)

    @property
    def failures(self):
        return [c for c in self.checks if not c.ok and not c.skipped]

    def __getitem__(self, key):
        return self.metrics[key]

    def report(self):
        lines = ["iscc tumour diagnosis " + ("[OK]" if self.ok else "[DEGENERATE]")]
        m = self.metrics
        lines.append(
            f"  metrics: N={m['n_cancer']} shannon={m['shannon']:.2f} "
            f"subclones={m['n_subclones']} TMB={m['tmb']:.1f} ({m['tmb_frac']*100:.0f}% of genome) "
            f"driver_enrich={_fmt(m['driver_enrichment'])} confinement={_fmt(m['clone_confinement'])} "
            f"fga={_fmt(m['fga'])} ploidy={_fmt(m['mean_ploidy'])} "
            f"hypoxia_contrast={_fmt(m['hypoxia_contrast'])} 1/f_R2={_fmt(m['vaf_1f_rsq'])}")
        for c in self.checks:
            if c.skipped:
                lines.append(f"  [ -- ] {c.name}: n/a")
            elif c.ok:
                lines.append(f"  [ ok ] {c.name}")
            else:
                lines.append(f"  [FAIL] {c.name}: {_fmt(c.value)} (threshold {_fmt(c.threshold)}) "
                             f"-> {c.hint}")
        for a in self.advisories:
            lines.append(f"  [note] {a}")
        return "\n".join(lines)

    def __str__(self):
        return self.report()


def _fmt(x):
    if x is None:
        return "n/a"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, float) and (np.isnan(x)):
        return "n/a"
    return f"{x:.3f}"


# --- metrics -----------------------------------------------------------------
def _cancer_gids_counts(tumor):
    return [(g, c) for g, c in tumor.genotypes_counts.items() if tumor._is_cancer(g)]


def tmb_stats(tumor):
    """Count-weighted mean mutated sites per cancer cell, and its fraction of the genome.

    A "mutated site" is a gene with non-zero SNV VAF in the genotype's representative cell
    (``rep.get_snvs() > 0``). Read directly from the genotype registry (no per-cell
    materialisation), so it is cheap and does not depend on ``make_cell_data``.
    """
    n_genes = getattr(tumor, "n_genes", tumor.n_segments * tumor.segment_size)
    tot = 0
    muts = 0.0
    for gid, cnt in _cancer_gids_counts(tumor):
        rep = tumor.genotypes[gid]
        muts += cnt * int((rep.get_snvs() > 0).sum())
        tot += cnt
    if tot == 0:
        return 0.0, 0.0
    tmb = muts / tot
    return float(tmb), float(tmb / max(n_genes, 1))


def driver_enrichment(tumor):
    """Mean population VAF at driver sites / mean VAF at passenger sites.

    Positive selection makes driver mutations rise to higher frequency than passengers, so a ratio
    well above 1 means selection is *detectable* (a proxy for the driver-enrichment-vs-neutral
    metric). Returns nan when there are no drivers or no mutations to compare.
    """
    sel = tumor.selection
    drivers = np.concatenate([sel.get_oncogenes(), sel.get_tsgs()]) if (
        len(sel.get_oncogenes()) or len(sel.get_tsgs())) else np.array([], dtype=int)
    if drivers.size == 0:
        return float("nan")
    vaf = population_vaf(tumor)
    n = vaf.shape[0]
    mask = np.zeros(n, dtype=bool)
    mask[drivers[drivers < n]] = True
    d = vaf[mask]
    p = vaf[~mask]
    if d.size == 0 or p.size == 0 or p.mean() == 0:
        return float("nan")
    return float(d.mean() / p.mean())


def clone_confinement(tumor, subclone_freq=0.01):
    """Spatial confinement of subclones: 1 ⇒ tight clonal territories, ~0 ⇒ well-mixed.

    For each cancer subclone above ``subclone_freq`` we take its count-weighted spatial spread (RMS
    distance of its cells from the clone centroid) relative to the whole tumour's spread; a clone
    confined to a compact patch has a small ratio, a clone smeared across the whole lesion has a
    ratio near 1. We return ``1 - (size-weighted mean ratio)``, so **high ⇒ territories** and
    **~0 ⇒ well-mixed** — the regime (dispersal_rate too high vs division_rate) that breaks the
    PEtracer lineage–space confound and the multi-region-phylogeny benchmarks.

    We use spatial confinement rather than a naive Moran's I of clone labels because, in this
    engine, raising the dispersal rate ALSO lowers the mutation branch probability
    ``mut/(mut+dispersal)`` and so reduces the number of clones — a label-Moran's I is confounded by
    that (fewer, larger clones can look *more* autocorrelated), whereas per-clone spatial spread
    isolates mixing cleanly. Returns nan with fewer than two subclones or an unresolved geometry.
    """
    gc = _cancer_gids_counts(tumor)
    total_cancer = sum(c for _, c in gc)
    if total_cancer == 0:
        return float("nan")
    clones = [(g, c) for g, c in gc if c / total_cancer >= subclone_freq]
    if len(clones) < 2:
        return float("nan")

    # per-clone (coords, weights) over occupied demes, and the whole-tumour spread.
    per = {g: ([], []) for g, _ in clones}
    keep = {g for g, _ in clones}
    all_coords, all_w = [], []
    for i, deme in enumerate(tumor.demes):
        rc = tumor.deme_coords[i]
        for gid, cnt in deme.items():
            if gid not in keep:
                continue
            per[gid][0].append(rc)
            per[gid][1].append(cnt)
            all_coords.append(rc)
            all_w.append(cnt)
    if not all_coords:
        return float("nan")
    ac = np.array(all_coords, dtype=float)
    aw = np.array(all_w, dtype=float)
    tumor_centroid = (ac * aw[:, None]).sum(0) / aw.sum()
    tumor_spread = np.sqrt((((ac - tumor_centroid) ** 2).sum(1) * aw).sum() / aw.sum())
    if tumor_spread == 0:
        return float("nan")

    num = wsum = 0.0
    for gid, cnt in clones:
        pc = np.array(per[gid][0], dtype=float)
        pw = np.array(per[gid][1], dtype=float)
        if pw.sum() == 0:
            continue
        cen = (pc * pw[:, None]).sum(0) / pw.sum()
        spread = np.sqrt((((pc - cen) ** 2).sum(1) * pw).sum() / pw.sum())
        num += cnt * (spread / tumor_spread)
        wsum += cnt
    if wsum == 0:
        return float("nan")
    return float(1.0 - num / wsum)


def cna_stats(tumor):
    """Count-weighted fraction-genome-altered (copy number != 2) and mean ploidy over cancer cells."""
    tot = 0
    fga = 0.0
    ploidy = 0.0
    seg_sizes = np.asarray(tumor.selection.segment_sizes, dtype=float)
    n_genes = float(seg_sizes.sum()) if seg_sizes.size else 1.0
    for gid, cnt in _cancer_gids_counts(tumor):
        gs = tumor.genotypes[gid].genome_summary
        seg_cns = np.asarray(gs["seg_cns"], dtype=float)
        altered = seg_sizes[seg_cns != 2].sum()
        fga += cnt * (altered / n_genes)
        ploidy += cnt * float(gs["ploidy"])
        tot += cnt
    if tot == 0:
        return float("nan"), float("nan")
    return float(fga / tot), float(ploidy / tot)


def hypoxia_contrast(tumor, core_frac=0.3, rim_frac=0.3):
    """Core–rim hypoxia contrast: mean hypoxia in inner demes − mean in outer demes.

    Uses the tumour's configured hypoxia parameters to solve the O2 field (``_o2_field``), then
    splits occupied demes by radial distance from the cancer-mass centroid into an inner ``core``
    and outer ``rim`` and returns the mean-hypoxia difference. ~0 ⇒ no gradient (tumour small
    relative to the O2 diffusion length). Returns nan when the microenvironment/hypoxia is off or
    the tumour is too small to split.
    """
    mp = getattr(tumor, "microenv_params", None)
    if not mp:
        return float("nan")
    hyp = (mp.get("hypoxia") or {}) if isinstance(mp, dict) else {}
    if not hyp:
        return float("nan")
    field = tumor._o2_field(D=float(hyp.get("o2_diffusion", 1.0)),
                            k=float(hyp.get("o2_consumption", 1.0)),
                            s=float(hyp.get("o2_supply", 0.2)),
                            source=hyp.get("o2_source", "uniform"))
    # occupied demes weighted by cancer presence (centroid = cancer mass)
    occ = np.array([i for i, d in enumerate(tumor.demes) if sum(d.values()) > 0])
    if occ.size < 6:
        return float("nan")
    coords = np.array([tumor.deme_coords[i] for i in occ], dtype=float)
    cancer = np.array([sum(c for g, c in tumor.demes[i].items() if tumor._is_cancer(g))
                       for i in occ], dtype=float)
    if cancer.sum() == 0:
        return float("nan")
    centroid = (coords * cancer[:, None]).sum(0) / cancer.sum()
    radius = np.sqrt(((coords - centroid) ** 2).sum(1))
    order = np.argsort(radius)
    n = occ.size
    core = order[: max(1, int(core_frac * n))]
    rim = order[-max(1, int(rim_frac * n)):]
    h = field[occ]
    return float(h[core].mean() - h[rim].mean())


# --- diagnosis ---------------------------------------------------------------
def diagnose(tumor, thresholds=None, verbose=False):
    """Compute phenotype metrics and flag degenerate regimes. See DESIGN_operating_envelope.md.

    Read-only: never draws from the tumour's rng or mutates the counts, so it cannot alter output.
    Returns a :class:`TumorDiagnosis`. ``thresholds`` overrides individual defaults.
    """
    th = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        th.update(thresholds)

    n_cancer = int(tumor.get_cancer_size())
    n_genes = int(getattr(tumor, "n_genes", tumor.n_segments * tumor.segment_size))
    gids, counts = tumor.get_genotype_frequencies(normalize=False)
    shannon = float(clone_diversity(counts)) if len(counts) else 0.0
    tot = float(np.sum(counts)) if len(counts) else 0.0
    n_subclones = int(sum(1 for c in counts if tot > 0 and c / tot >= th["subclone_freq"]))

    tmb, tmb_frac = tmb_stats(tumor)
    enrich = driver_enrichment(tumor)
    confinement = clone_confinement(tumor, subclone_freq=th["subclone_freq"])
    fga, ploidy = cna_stats(tumor)
    contrast = hypoxia_contrast(tumor)
    try:
        rsq, _ = neutral_sfs_rsq(population_vaf(tumor)) if n_cancer > 0 else (float("nan"), 0.0)
    except Exception:
        rsq = float("nan")

    metrics = dict(n_cancer=n_cancer, n_genes=n_genes, shannon=shannon, n_subclones=n_subclones,
                   tmb=tmb, tmb_frac=tmb_frac, driver_enrichment=enrich,
                   clone_confinement=confinement, fga=fga, mean_ploidy=ploidy,
                   hypoxia_contrast=contrast, vaf_1f_rsq=rsq)

    checks = []
    advisories = []
    extinct = n_cancer < th["min_cancer"]
    checks.append(Check(
        "extinct", not extinct, n_cancer, th["min_cancer"],
        "tumour extinct/too small: lower death_rate or raise division_rate / initial_cancer_cells / "
        "carrying_capacity"))

    # Size advisory (non-failing): a small-but-healthy tumour is not degenerate, but realistic
    # analyses expect ~10^3--10^4 cancer cells. In the exact engine tumour size grows roughly one
    # cell per step, so reaching thousands needs enough steps (and grid_size^2 * carrying_capacity
    # to hold them); tau-leaping (update_mode="tau") scales to large tumours cheaply. Applies to
    # both structured (structure_radius > 0) and unstructured (structure_radius = 0) simulations.
    if not extinct and n_cancer < th["min_realistic"]:
        advisories.append(
            f"small tumour ({n_cancer} cancer cells): for realistic analyses grow to "
            f"~10^3-10^4 cells -- run more steps (exact engine: size ~ #steps) or use tau-leaping, "
            f"and ensure grid_size^2 x carrying_capacity exceeds the target size")

    # trivial genome (config-time)
    checks.append(Check(
        "trivial_genome", n_genes >= th["min_genes"], n_genes, th["min_genes"],
        "genome too small to carry realistic variation: raise n_segments / segment_size"))

    if extinct:
        # everything downstream is undefined on an empty tumour; skip the phenotype checks.
        for name in ("monoclonal", "low_mutation", "hypermutated", "well_mixed", "cna_runaway"):
            checks.append(Check(name, True, None, None, skipped=True))
        checks.append(Check("no_gradient", True, None, None, skipped=True))
        diag = TumorDiagnosis(metrics, checks, advisories)
        if verbose:
            print(diag.report())
        return diag

    # monoclonal — low diversity, with the hint attributing the culprit.
    if tmb < th["tmb_min"]:
        mono_hint = "monoclonal (no mutations): raise mutation_rate / n_snvs_per_allele"
    elif not np.isnan(enrich) and enrich > th["sweep_enrichment"]:
        mono_hint = "monoclonal (selective sweep): lower driver_effects / prop_driver"
    else:
        mono_hint = "monoclonal: raise mutation_rate (or lower driver_effects if a sweep)"
    checks.append(Check("monoclonal", shannon >= th["shannon_min"], shannon, th["shannon_min"],
                        mono_hint))

    checks.append(Check("low_mutation", tmb >= th["tmb_min"], tmb, th["tmb_min"],
                        "too few mutations/cell: raise mutation_rate / n_snvs_per_allele"))
    checks.append(Check("hypermutated", tmb_frac <= th["tmb_frac_max"], tmb_frac,
                        th["tmb_frac_max"],
                        "hypermutated mush (broken 1/f tail): lower mutation_rate / n_snvs_per_allele"))

    # well-mixed — only meaningful with >=2 spatial subclones (else it is monoclonal, not mixed).
    if n_subclones >= 2 and not np.isnan(confinement):
        checks.append(Check("well_mixed", confinement >= th["confinement_min"], confinement,
                            th["confinement_min"],
                            "well-mixed (no clonal territories): lower dispersal_rate relative to "
                            "division_rate -- BREAKS the PEtracer & multi-region benchmarks"))
    else:
        checks.append(Check("well_mixed", True, confinement, th["confinement_min"], skipped=True))

    checks.append(Check("cna_runaway", np.isnan(fga) or fga <= th["fga_max"], fga, th["fga_max"],
                        "CNA runaway / saturated genome: lower amp_prob / max_cn"))

    # no microenvironment gradient — only when hypoxia is enabled.
    if np.isnan(contrast):
        checks.append(Check("no_gradient", True, contrast, th["contrast_min"], skipped=True))
    else:
        checks.append(Check("no_gradient", contrast >= th["contrast_min"], contrast,
                            th["contrast_min"],
                            "no O2 gradient: grow a larger tumour or raise o2_consumption k / lower "
                            "o2_diffusion D"))

    diag = TumorDiagnosis(metrics, checks, advisories)
    if verbose:
        print(diag.report())
    return diag
