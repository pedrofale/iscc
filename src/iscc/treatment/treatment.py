class Treatment(object):
    """Shared base of the therapies, with the dosing schedule and the drug response.

    The therapies ([`Chemotherapy`][iscc.treatment.Chemotherapy],
    [`TargetedTherapy`][iscc.treatment.TargetedTherapy],
    [`Immunotherapy`][iscc.treatment.Immunotherapy],
    [`Surgery`][iscc.treatment.Surgery]) share the parameters below, and each subclass adds its
    own targeting parameters. Pass an instance to
    [`GenotypeTumor.grow`][iscc.tumor.GenotypeTumor.grow] as ``treatment=...``.

    While a dose is active, the growth engine adds a death hazard to each clone (chemotherapy and
    targeted therapy) or lowers its immune resistance (immunotherapy). With ``response="label"``,
    a cancer clone that is a target takes the dose ``effectiveness * (1 - r)`` and any other clone
    ``toxicity * (1 - r)``, where ``r`` is the largest of the treatment resistance, the drug
    tolerance and the carried resistance state of the clone. Normal cells take ``toxicity``. A dose
    ``x`` adds the death hazard ``x * max(kill_rate - d, 0)`` under ``kill_mode="additive"`` and
    ``x * kill_rate * b`` under ``kill_mode="proliferation"``, where ``d`` and ``b`` are the death
    and division rates of the clone.

    Parameters
    ----------
    adaptive : bool, optional
        Dose only while the tumour exceeds ``max_tumor_size`` (adaptive therapy). Default False,
        which doses throughout the active window.
    start : int, optional
        First step at which the therapy is active (default 0).
    duration : int, optional
        Number of steps the therapy stays active (default ``None``, until the run ends).
    dosage_decay : float, optional
        Not read by the growth engine (default 0.5).
    rounds : int, optional
        Not read by the growth engine (default 4).
    rate_multiplier : float, optional
        Not read by the growth engine, whose hazard is set by ``kill_rate`` (default 2.0).
    toxicity : float, optional
        Share of the full dose taken by a cell that is not a target, a normal cell or a cancer
        clone outside the targets (default 0.1).
    effectiveness : float, optional
        Share of the full dose taken by a cancer clone that is a target, before its resistance is
        applied (default 0.9).
    kill_rate : float, optional
        Sets the death hazard of a fully sensitive cell under the full dose (default 1.5), as
        ``kill_rate - d`` under ``kill_mode="additive"`` and ``kill_rate * b`` under
        ``kill_mode="proliferation"``.
    max_tumor_size : int, optional
        Size threshold that gates dosing when ``adaptive=True`` (default 100000).
    sites : {"both", "primary", "met"}, optional
        Compartment(s) the therapy acts on (default ``"both"``, systemic): ``"primary"``
        (neoadjuvant or local) or ``"met"`` (adjuvant after primary resection).
    mutagenicity : float, optional
        Therapy-induced mutator: an exposed cancer clone's mutation rate is multiplied by this
        once, on first exposure, and its descendants inherit it (default 1.0 = off).
    kill_mode : {"additive", "proliferation"}, optional
        How the hazard scales with the clone. ``"additive"`` (default) adds a hazard that does not
        depend on the division rate, and ``"proliferation"`` scales it by the division rate of the
        clone, as cytotoxic chemotherapy kills cells as they replicate.
    mutagenicity_mode : {"uniform", "dose"}, optional
        Who gets the mutator: every exposed cancer clone (``"uniform"``, default), or a share
        scaled by the dose the clone receives (``"dose"``).
    mutagenicity_target : {"all", "snv"}, optional
        What the mutator accelerates: every mutation (``"all"``, default) or point mutations
        only (``"snv"``), leaving the copy-number rate where it was.
    response : {"label", "mechanistic"}, optional
        How the response of a cancer clone is decided. With ``"label"`` (default), the resistance
        labels ``treatment_resistance``, ``drug_tolerance`` and the carried resistance state scale
        down the dose the clone takes. With ``"mechanistic"``, the labels are ignored and the
        response follows from the genotype of the clone and its expression alone. A clone is
        on target only if it carries the therapy's targets (see
        [`TargetedTherapy`][iscc.treatment.TargetedTherapy]); off-target clones take
        ``toxicity`` as before. With ``target_genes`` its dose also scales with its expression
        of those genes, and with ``kill_mode="proliferation"`` a broad drug kills in proportion
        to division rate. Under ``mutagenicity_mode="dose"`` the mutator follows the same
        expression scaling instead of ``1 - treatment_resistance``.
    target_genes : list, optional
        Mechanistic mode only: genes the drug acts through, as ``(segment, position)``
        coordinates or flat gene indices. The dose of an on-target clone is multiplied by its mean
        expression of these genes relative to that of the founding clone, so a clone at the level
        of the founder takes the nominal dose and one at twice the level twice the dose. Expression is
        the clone's dosage-driven baseline times its mean program activity (the same law
        ``make_cell_data`` uses, without the per-cell noise and the niche).
    expression_threshold : float, optional
        Mechanistic mode only, with ``target_genes``: the expression below which the drug has
        no effect on a clone at all (default 0).
    amplification_threshold : int, optional
        Mechanistic mode only: a clone also carries a target when the target's segment has more
        copies than this. Default None: only a copy with the mutation counts.
    """

    # Which evolutionary parameter the therapy modifies. Death-rate therapies
    # (chemo, targeted) kill sensitive cells; immunotherapy instead strips immune
    # resistance so the local immune microenvironment can kill the cell.
    affects = "death_rate"

    def __init__(self, adaptive=False, start=0, duration=None,
                 dosage_decay=0.5, rounds=4,
                 rate_multiplier=2., toxicity=0.1, effectiveness=0.9,
                 kill_rate=1.5, max_tumor_size=100_000, sites="both",
                 mutagenicity=1.0, kill_mode="additive",
                 mutagenicity_mode="uniform", mutagenicity_target="all",
                 response="label", target_genes=None, expression_threshold=None,
                 amplification_threshold=None):
        self.adaptive = adaptive
        self.start = start
        self.duration = duration
        self.rate_multiplier = rate_multiplier
        self.toxicity = toxicity
        self.effectiveness = effectiveness
        # Death hazard imposed on a fully sensitive cell under full dose (genotype
        # engine). Set above the max_birth_rate cap so even high-fitness (driver-amplified)
        # sensitive clones regress, while resistant clones (treatment_resistance -> 1) escape.
        self.kill_rate = kill_rate
        self.dosage_decay = dosage_decay
        self.rounds = rounds
        self.max_tumor_size = max_tumor_size
        # Which compartment(s) the therapy acts on: "both" (systemic, default), "met" (adjuvant after
        # primary resection), or "primary" (neoadjuvant / local). The engine gates the per-genotype
        # death/immune override by the deme's compartment (GenotypeTumor._death_rate).
        self.sites = sites
        # HOW the kill scales with the cell (genotype engine). "additive" (default) gives every clone
        # the same absolute extra death, so survival under drug is decided by birth rate and a
        # fast-dividing clone simply outruns the dose -- the drug then selects for the fittest clones
        # and the lesion regrows mid-course. "proliferation" scales the hazard by the clone's own
        # division rate (net = b*(1 - kill_rate) - death), so at kill_rate 1.0 every clone declines at
        # its death rate whatever its fitness, and above 1.0 the faster ones decline faster. That is
        # also the biology: cytotoxic chemotherapy kills cells as they replicate, which is why it
        # spares quiescent tissue and why fast-growing tumours respond best.
        self.kill_mode = kill_mode
        # THERAPY-INDUCED MUTATOR PHENOTYPE. Cytotoxic therapy damages DNA and can permanently impair
        # repair (temozolomide-driven mismatch-repair loss in glioma is the classic case; platinums
        # leave lasting mutational signatures), so an exposed cell's mutation rate rises and STAYS
        # risen — and its descendants inherit it. Multiplies `mutation_rate` ONCE per genotype on
        # first exposure, never compounding per step and never reverting when the drug stops.
        # 1.0 (default) leaves every genotype untouched.
        # NOTE the effect saturates: iscc draws mutation as an alternative FATE to dispersal
        # (mut_prob = mutation_rate / (mutation_rate + dispersal_rate)), so mut_prob -> 1 at most,
        # i.e. at the shipped rates the mutant fraction can rise 0.25 -> 1.0, a ceiling of 4x.
        self.mutagenicity = float(mutagenicity)
        # WHO gets the mutator phenotype. "uniform" (default) raises it for every cancer clone in a
        # treated compartment, regardless of resistance — so a clone taking ZERO drug still mutates
        # faster, permanently. It is the only drug effect in the engine NOT scaled by
        # (1 - treatment_resistance): the kill, immunotherapy's resistance stripping and the persister
        # cost all are. The elevated rate is also inherited by a resistant clone, where it multiplies
        # the CNA rate that DELETES the resistance allele (reversion).
        # "dose" scales the boost by the dose the clone actually receives,
        #     factor = 1 + (mutagenicity - 1) * (1 - tr),
        # exactly as `_kill_amount` scales the hazard: a sensitive cell gets the full multiplier, a
        # fully resistant one gets none. Mechanistically this says the mutagenesis comes from drug
        # that reaches the DNA — true of efflux/detoxification resistance, weaker for target-site
        # resistance where the drug still enters.
        self.mutagenicity_mode = str(mutagenicity_mode)
        # WHAT the mutator accelerates. `mutagenicity` multiplies `mutation_rate`, i.e. the chance a
        # division takes the MUTATION fate at all; the SNV/CNA split happens downstream on
        # snv_prob/cnv_prob, so with "all" (default) point mutations and copy-number events scale
        # TOGETHER. That couples the two processes this model cares about: resistance is ACQUIRED by
        # an SNV landing on a resistance locus and LOST by a CNA deleting the copy that carries it,
        # so turning the mutator up changes the scale of both, not their balance.
        # "snv" makes the drug a POINT MUTAGEN rather than a clastogen: cnv_prob is lowered by
        # exactly the factor that keeps the ABSOLUTE per-division CNA rate where it was, so extra
        # mutating divisions all become SNVs. This is a claim about the AGENT — platinums and
        # alkylators are base-damaging and leave substitution signatures, which is mechanistically
        # distinct from inducing chromosome missegregation.
        self.mutagenicity_target = str(mutagenicity_target)
        # HOW a cancer clone's response is decided (genotype engine). "label" (default) reads the
        # resistance labels, as above. "mechanistic" ignores
        # them and derives the response from the genotype (does it carry the target?) and its
        # expression (how much target does it make?). The knobs below only mean something there.
        if response not in ("label", "mechanistic"):
            raise ValueError(f"response must be 'label' or 'mechanistic', got {response!r}")
        self.response = response
        given = [name for name, set_ in (("target_genes", bool(target_genes)),
                                         ("expression_threshold", expression_threshold is not None),
                                         ("amplification_threshold",
                                          amplification_threshold is not None)) if set_]
        if given and response != "mechanistic":
            raise ValueError(f"{', '.join(given)} only apply with response='mechanistic'")
        if expression_threshold is not None and not target_genes:
            raise ValueError("expression_threshold needs target_genes")
        self.target_genes = [tuple(g) if isinstance(g, (tuple, list)) else int(g)
                             for g in (target_genes or [])]
        self.expression_threshold = (0.0 if expression_threshold is None
                                     else float(expression_threshold))
        self.amplification_threshold = (None if amplification_threshold is None
                                        else int(amplification_threshold))
        self.dosage_trace = []  # list of (step, dosage) for every queried step

    def discrete_event(self, tumor, step):
        """One-shot state edits that are NOT rate modifiers (e.g. surgical resection). The base
        treatment does nothing; `Surgery` overrides this. The engine calls it once per step; the base
        version changes no state and draws no random numbers, so it leaves runs without surgery
        unaffected."""
        pass

    def is_target(self, cell, **kwargs):
        pass

    def get_dosage(self, step, tumor_size):
        in_window = step >= self.start
        if self.duration is not None:
            in_window = in_window and (step < self.start + self.duration)
        if not in_window:
            return 0.
        if self.adaptive:
            dosage = 1. if tumor_size > self.max_tumor_size else 0.
        else:
            dosage = 1.
        self.dosage_trace.append((step, dosage))
        return dosage
