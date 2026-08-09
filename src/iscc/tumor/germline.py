"""The two variant layers a tumour already has before it starts growing.

GERMLINE (inherited) variants — the constant genetic background every cell of a patient carries.

A germline variant sits in EVERY cell of an individual, tumour and normal alike, which is exactly
what makes it useful to a DNA assay: heterozygous germline sites read at a variant allele fraction
of ~0.5 wherever the two parental copies are balanced, and move away from 0.5 only where a somatic
copy-number change has unbalanced them. That constant, cell-type-independent layer is the anchor a
purity / cancer-cell-fraction analysis calibrates against (a bulk sample's germline heterozygous
sites report the mixture, not the tumour), the B-allele-frequency track allele-specific copy-number
callers read, and the private background genetic demultiplexing uses to assign cells to individuals.

Two calls, made once per patient and applied BEFORE growth so every descendant inherits them:

* :func:`draw_germline_sites` picks the loci. They are drawn from NEUTRAL positions only — genes
  with no role in any selection axis — so the germline is a pure observational layer and can never
  perturb fitness or the growth trajectory.
* :func:`apply_germline` writes the bits into every genotype registered on the tumour: the cancer
  founder and each normal (epithelial / stromal / immune / host) genotype.

Because the variants are written into the genome itself rather than added at assay time, later
somatic events treat them like any other variant: a copy-number gain duplicates them, a loss can
delete them, and a whole-genome duplication doubles them. The germline therefore sits UNDERNEATH the
somatic copy-number landscape and picks up its allele-specific behaviour for free.

FOUNDER (truncal) mutations — the burden the transformed cell already carried at the moment it
became a tumour. A simulation that starts at transformation with a blank genome has no such layer,
and the consequence is not subtle: truncal means "present in the most recent common ancestor", the
most recent common ancestor of the tumour IS the founder, so an empty founder genome makes it
impossible for ANY site to be clonal. A purely expanding, spatially structured population never
sweeps a later mutation to fixation either, so nothing fills the gap: real tumours put a large share
of their mutation burden in the clonal peak, and without this layer the simulated site-frequency
spectrum has no peak at all. :func:`seed_founder_mutations` supplies it — the clock-like mutations
the normal lineage accumulated before transformation, plus the transforming driver hits — written
into the founder genome itself, so every cancer cell inherits them by descent rather than by any
assay-time reconstruction.
"""
import numpy as np

__all__ = ["neutral_positions", "driver_positions", "draw_germline_sites", "apply_germline",
           "seed_founder_mutations"]


def neutral_positions(selection):
    """Flat gene indices that carry NO selective role — not a driver (oncogene or tumour
    suppressor) and not part of any heritable trait axis (dispersal, immune / treatment resistance,
    breach, stromal survival, metastatic survival).

    This is the pool germline variants are drawn from, so that an inherited variant is guaranteed
    never to change a cell's division, death, dispersal or resistance.
    """
    pool = []
    for seg in range(selection.n_segments):
        size = int(selection.segment_sizes[seg])
        functional = np.zeros(size, dtype=bool)
        for idx in (selection.drivers[seg], selection.dispersal[seg],
                    selection.immune_resistance[seg], selection.treatment_resistance[seg],
                    selection.breach[seg], selection.stromal_survival[seg],
                    selection.met_survival[seg]):
            if len(idx):
                functional[np.asarray(idx, dtype=int)] = True
        pool.append(int(selection._seg_offsets[seg]) + np.flatnonzero(~functional))
    return np.concatenate(pool) if pool else np.array([], dtype=int)


def driver_positions(selection):
    """Flat gene indices of the DRIVER genes — oncogenes and tumour suppressors.

    These are the genes whose mutation changes how fast a cell divides or dies, which is what makes
    them the right home for the transforming hits a founder cell carries. Deliberately NOT the whole
    functional set: the dissemination and resistance traits (breach, stromal survival, dispersal,
    immune / treatment resistance, metastatic survival) are acquired LATER, during progression, and
    handing them to the founder would start the tumour already invasive or already drug-resistant.
    """
    pool = []
    for seg in range(selection.n_segments):
        idx = np.asarray(selection.drivers[seg], dtype=int)
        if len(idx):
            pool.append(int(selection._seg_offsets[seg]) + idx)
    return np.concatenate(pool) if pool else np.array([], dtype=int)


def draw_germline_sites(selection, het_frac=5e-3, hom_frac=0.33, seed=None):
    """Draw one patient's germline variant loci.

    Parameters
    ----------
    selection : Selection
        The gene-role layout; only its NEUTRAL positions are eligible (see :func:`neutral_positions`).
    het_frac : float, optional
        Density of HETEROZYGOUS germline sites — the fraction of neutral genome positions that carry
        a variant on one parental copy. These are the sites a purity / allelic-imbalance analysis
        reads, so this is the knob that matters; the default 5e-3 is a few sites per thousand, the
        order of magnitude of common human single-nucleotide polymorphism density.
    hom_frac : float, optional
        Fraction of the drawn sites that are HOMOZYGOUS (present on both parental copies, so read at
        a variant allele fraction of ~1.0 and carrying no allelic-imbalance information). The
        remaining sites are heterozygous, at the density ``het_frac`` asks for.
    seed : int, optional
        Seed for the draw. Germline variants are PRIVATE TO AN INDIVIDUAL, so two patients should be
        given different seeds (and the same patient re-simulated with the same seed keeps its
        background genotype).

    Returns
    -------
    tuple
        ``(sites, zygosity, homologs)`` — flat gene indices in ascending order, ``"het"`` / ``"hom"``
        per site, and the parental copy the variant sits on: ``"p"`` or ``"m"`` for a heterozygous
        site (chosen at random, so the B-allele track is not systematically one-sided) and
        ``"both"`` for a homozygous one.
    """
    if not 0.0 <= hom_frac < 1.0:
        raise ValueError(f"hom_frac must be in [0, 1), got {hom_frac}")
    if het_frac < 0.0:
        raise ValueError(f"het_frac must be >= 0, got {het_frac}")
    pool = neutral_positions(selection)
    empty = (np.array([], dtype=int), np.array([], dtype="<U4"), np.array([], dtype="<U4"))
    if not len(pool):
        return empty
    n_het = int(round(het_frac * len(pool)))
    # `het_frac` fixes the HETEROZYGOUS density, so the total draw is scaled up to leave room for the
    # homozygous share on top of it (rather than carving it out of the same budget).
    n_total = min(int(round(n_het / (1.0 - hom_frac))), len(pool))
    n_het = min(n_het, n_total)
    n_hom = n_total - n_het
    if n_total <= 0:
        return empty

    rng = np.random.default_rng(seed)
    sites = rng.choice(pool, size=n_total, replace=False).astype(int)
    zygosity = np.array(["hom"] * n_hom + ["het"] * n_het, dtype="<U4")
    rng.shuffle(zygosity)                       # which of the drawn sites are homozygous
    # A heterozygous variant sits on ONE parental copy, picked at random per site: always using the
    # same homolog would make every patient's B-allele track perfectly one-sided.
    picks = np.where(rng.random(n_total) < 0.5, "p", "m")
    homologs = np.where(zygosity == "hom", "both", picks).astype("<U4")

    order = np.argsort(sites, kind="stable")
    return sites[order], zygosity[order], homologs[order]


def _set_bits(rep, selection, sites, homologs, update_summary):
    """Write the germline bits into one genotype's genome, on EVERY existing copy of the chosen
    parental homolog(s). ``update_summary`` refreshes the driver/trait tallies, which is only needed
    for the cancer founder (the static normal genotypes carry no selection and read their variants
    straight off the genome)."""
    offs = np.asarray(rep._seg_offsets, dtype=int)
    for seg in range(rep.n_segments):
        lo, hi = int(offs[seg]), int(offs[seg + 1])
        in_seg = (sites >= lo) & (sites < hi)
        if not in_seg.any():
            continue
        seg_pos = sites[in_seg] - lo
        seg_hom = homologs[in_seg]
        for hap in ("p", "m"):
            take = (seg_hom == hap) | (seg_hom == "both")
            if not take.any():
                continue
            pos = seg_pos[take]
            for allele in rep.genome[seg][hap]:
                new = np.zeros(int(rep.segment_sizes[seg]), dtype=bool)
                new[pos] = True
                new &= ~allele                  # only positions this copy did not already carry
                if not new.any():
                    continue
                allele |= new                   # in place: the genotype owns its bitsets pre-growth
                if update_summary:
                    rep.update_genome_summary_mutation(selection, new, seg)


def apply_germline(tumor, sites, zygosity, homologs):
    """Seed the drawn germline variants into EVERY genotype registered on ``tumor`` — the cancer
    founder and each normal (epithelial / stromal / immune / host) genotype — so that every cell the
    simulation ever produces inherits them.

    Call this before growth, while the registry holds only the founders: the bits then descend
    through division and are reshaped by somatic copy-number events exactly like somatic variants.
    The ground truth is recorded on the tumour as ``germline_sites`` / ``germline_zygosity`` /
    ``germline_homolog``, and as a per-gene mask on the gene-role layout (so it is written out with
    the rest of the gene annotation).
    """
    sites = np.asarray(sites, dtype=int)
    zygosity = np.asarray(zygosity, dtype="<U4")
    homologs = np.asarray(homologs, dtype="<U4")
    selection = tumor.selection

    for rep in tumor.genotypes.values():
        is_cancer = getattr(rep, "type", None) == "cancer"
        _set_bits(rep, selection, sites, homologs, update_summary=is_cancer)
        if is_cancer:
            # No-op for a neutral draw, but a caller may plant a variant on a driver on purpose;
            # refreshing keeps the clone's rates consistent with its genome either way.
            rep.update_evolutionary_parameters(selection)

    tumor.germline_sites = sites
    tumor.germline_zygosity = zygosity
    tumor.germline_homolog = homologs
    # Per-gene mask on the shared gene-role layout: 0 = none, 1 = heterozygous, 2 = homozygous.
    mask = getattr(selection, "germline_types", None)
    if mask is None or len(mask) != selection.n_genes:
        mask = np.zeros(selection.n_genes, dtype=int)
        selection.germline_types = mask
    mask[:] = 0
    if len(sites):
        mask[sites] = np.where(zygosity == "hom", 2, 1)

    # Planting a variant on a driver can move the founder's rates, so any already-computed per-deme
    # event rate has to be rebuilt. Skipped when the tumour has not reached that point yet (the
    # in-constructor path), where the rates are computed fresh straight afterwards.
    _refresh_deme_rates(tumor)


def _refresh_deme_rates(tumor):
    """Rebuild the per-deme event rates after the founder's genome (and so possibly its division /
    death rates) has changed. A no-op before the constructor has computed them for the first time."""
    if getattr(tumor, "deme_rates", None) is not None:
        tumor.deme_rates = np.array([tumor._deme_rate(i) for i in range(len(tumor.demes))],
                                    dtype=float)


def seed_founder_mutations(tumor, n_passenger=40, n_driver=2, seed=None):
    """Give the FOUNDER cancer genotype the mutations it already carried when it transformed — the
    tumour's TRUNCAL (fully clonal) layer.

    A tumour's clonal peak is not made during growth. It is inherited: the cell that transformed had
    already spent a lifetime accumulating clock-like mutations as an ordinary dividing cell, and it
    transformed because a handful of driver lesions landed on top of that. Both are present in the
    most recent common ancestor of every tumour cell, so both are clonal from the first division.
    Writing them into the founder genome here — before any growth — is what makes them clonal by
    DESCENT: every cancer cell inherits them, no normal cell has them, and later copy-number events
    and whole-genome duplications reshape them exactly as they reshape any other variant.

    Parameters
    ----------
    tumor : GenotypeTumor
        The tumour whose founder is to be seeded. Call before growth, while the founder genotype is
        still the only cancer genotype registered.
    n_passenger : int, optional
        Number of NEUTRAL sites to plant — the pre-transformation, clock-like burden. Drawn from the
        same neutral pool the germline uses, excluding any position the germline already took, so the
        two layers never collide and a caller can tell an inherited variant from a truncal somatic
        one. This is the knob that sets what share of a sampled cell's mutated sites is clonal; the
        default 40 puts roughly half of a sampled cell's mutated sites in the clonal peak at the
        shipped tutorial settings, and more than that in a small, barely-diversified lesion — which
        is what such a lesion really looks like.
    n_driver : int, optional
        Number of SNVs to plant in DRIVER genes (oncogenes / tumour suppressors) — the transforming
        hits. Unlike the passengers these DO change the founder's fitness, which is the point: a
        transformed cell divides faster than the normal one it came from. Set to 0 for a founder that
        is genetically an ordinary cell apart from its passenger burden. Dissemination and resistance
        traits are never seeded (see :func:`driver_positions`) — those are acquired during
        progression, and a founder that already carried them would start out invasive or resistant.
    seed : int, optional
        Seed for the draw. Two tumours built with the same seed get the same truncal layer.

    Returns
    -------
    dict
        ``n_passenger`` / ``n_driver`` actually planted (a small genome can run out of eligible
        positions), ``n_sites`` their total, and ``sites`` the flat gene indices in ascending order.

    Notes
    -----
    Every variant is HETEROZYGOUS: it goes on ONE randomly chosen parental homolog, on one copy, so
    it sits at a variant allele fraction of 0.5 in the diploid founder and then rides through later
    gains, losses and whole-genome duplications at the correct multiplicity. (One consequence worth
    knowing when reading the output: after a whole-genome duplication a truncal variant sits on ~1.5
    copies on average, so a caller that assumes multiplicity 1 reports a cancer-cell fraction near
    1.5 rather than 1.0. That is a real property of multiplicity-1 calling on a part-duplicated
    tumour, not an artefact.)

    The ground truth is recorded on the tumour as ``seeded_truncal_sites``, ``seeded_truncal_homolog`` and
    ``seeded_truncal_is_driver``, so a plot can label the clonal peak and split it into its driver and
    passenger parts.
    """
    n_passenger = max(0, int(n_passenger))
    n_driver = max(0, int(n_driver))
    selection = tumor.selection
    rng = np.random.default_rng(seed)

    # The germline is drawn from the same neutral pool, so exclude whatever it took: an inherited
    # variant and a truncal somatic one must never land on the same position.
    taken = np.asarray(getattr(tumor, "germline_sites", ()), dtype=int)
    neutral = neutral_positions(selection)
    drivers = driver_positions(selection)
    if len(taken):
        neutral = neutral[~np.isin(neutral, taken)]
        drivers = drivers[~np.isin(drivers, taken)]

    n_pass = min(n_passenger, len(neutral))
    n_drv = min(n_driver, len(drivers))
    pass_sites = (rng.choice(neutral, size=n_pass, replace=False).astype(int)
                  if n_pass else np.array([], dtype=int))
    drv_sites = (rng.choice(drivers, size=n_drv, replace=False).astype(int)
                 if n_drv else np.array([], dtype=int))

    sites = np.concatenate([pass_sites, drv_sites]).astype(int)
    is_driver = np.concatenate([np.zeros(n_pass, dtype=bool), np.ones(n_drv, dtype=bool)])
    # One randomly chosen parental copy per site: always using the same homolog would put the whole
    # truncal layer on one allele, and every later allele-specific readout would inherit that bias.
    homologs = np.where(rng.random(len(sites)) < 0.5, "p", "m").astype("<U4")

    order = np.argsort(sites, kind="stable")
    sites, is_driver, homologs = sites[order], is_driver[order], homologs[order]

    founder = tumor.genotypes[tumor.founder_id]
    _set_bits(founder, selection, sites, homologs, update_summary=True)
    # The driver hits legitimately move the founder's division / death rates — a transformed cell is
    # fitter than the normal cell it came from — so its evolved rates (and every per-deme event rate
    # built from them) are refreshed.
    founder.update_evolutionary_parameters(selection)

    tumor.seeded_truncal_sites = sites
    tumor.seeded_truncal_homolog = homologs
    tumor.seeded_truncal_is_driver = is_driver
    _refresh_deme_rates(tumor)

    return {"n_passenger": int(n_pass), "n_driver": int(n_drv), "n_sites": int(len(sites)),
            "sites": sites}
