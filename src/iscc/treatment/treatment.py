import numpy as np

class Treatment(object):
    """Shared base for the therapies; defines the dosing schedule and per-cell effect.

    The therapies ([`Chemotherapy`][iscc.treatment.Chemotherapy],
    [`TargetedTherapy`][iscc.treatment.TargetedTherapy],
    [`Immunotherapy`][iscc.treatment.Immunotherapy],
    [`Surgery`][iscc.treatment.Surgery]) share the dosing / scheduling parameters below;
    each subclass adds its own targeting knobs and defines how a dose modifies a cell.
    Pass an instance to a tumour engine's ``grow(..., treatment=...)``.

    Parameters
    ----------
    adaptive : bool, optional
        If ``True``, dose only while the tumour exceeds ``max_tumor_size`` (adaptive
        therapy); if ``False`` (default), dose continuously within the active window.
    start : int, optional
        First step at which the therapy is active (default 0).
    duration : int, optional
        Number of steps the therapy stays active (default ``None`` = until the run ends).
    dosage_decay : float, optional
        Between-round dose decay factor (default 0.5).
    rounds : int, optional
        Number of dosing rounds (default 4).
    rate_multiplier : float, optional
        Factor applied to a fully-sensitive cell's affected rate under full dose
        (default 2.0); the exponent depends on the cell's ``treatment_resistance``.
    toxicity : float, optional
        Per-step probability of off-target action on a non-cancer cell (default 0.1).
    effectiveness : float, optional
        Per-step probability the therapy acts on a targeted cell (default 0.9).
    kill_rate : float, optional
        Death hazard imposed on a fully-sensitive cell under full dose in the genotype
        engine (default 1.5); set above the ``max_birth_rate`` cap so even high-fitness
        (driver-amplified) sensitive clones still regress while resistant clones escape.
    max_tumor_size : int, optional
        Size threshold that gates dosing when ``adaptive=True`` (default 100000).
    sites : {"both", "primary", "met"}, optional
        Compartment(s) the therapy acts on (default ``"both"`` = systemic): ``"primary"``
        (neoadjuvant / local) or ``"met"`` (adjuvant after primary resection).
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
                 mutagenicity_mode="uniform", mutagenicity_target="all"):
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
        # Which compartment(s) the therapy acts on (metastasis module, R9): "both" (systemic, default),
        # "met" (adjuvant after primary resection), or "primary" (neoadjuvant / local). The engine
        # gates the per-genotype death/immune override by the deme's compartment (GenotypeTumor._death_rate).
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
        # 1.0 (default) -> no genotype is ever touched -> byte-identical to before.
        # NOTE the effect saturates: iscc draws mutation as an alternative FATE to dispersal
        # (mut_prob = mutation_rate / (mutation_rate + dispersal_rate)), so mut_prob -> 1 at most,
        # i.e. at the shipped rates the mutant fraction can rise 0.25 -> 1.0, a ceiling of 4x.
        self.mutagenicity = float(mutagenicity)
        # WHO gets the mutator phenotype. "uniform" (default) raises it for every cancer clone in a
        # treated compartment, regardless of resistance — so a clone taking ZERO drug still mutates
        # 4x faster, permanently. That is the only drug effect in the engine NOT scaled by
        # (1 - treatment_resistance): the kill, immunotherapy's resistance stripping and the persister
        # cost all are. It also cuts against the mutator's own purpose — the elevated rate that makes
        # de novo resistance likely is inherited by the resistant clone, where its main effect is to
        # multiply the CNA rate that DELETES the resistance allele (reversion), ~4x.
        # "dose" scales the boost by the dose the clone actually receives,
        #     factor = 1 + (mutagenicity - 1) * (1 - tr),
        # exactly as `_kill_amount` scales the hazard: a sensitive cell gets the full multiplier, a
        # fully resistant one gets none. Mechanistically this says the mutagenesis comes from drug
        # that reaches the DNA — true of efflux/detoxification resistance, weaker for target-site
        # resistance where the drug still enters. Default "uniform" -> byte-identical to before.
        self.mutagenicity_mode = str(mutagenicity_mode)
        # WHAT the mutator accelerates. `mutagenicity` multiplies `mutation_rate`, i.e. the chance a
        # division takes the MUTATION fate at all; the SNV/CNA split happens downstream on
        # snv_prob/cnv_prob, so point mutations and copy-number events scale TOGETHER. That couples
        # the two processes this model cares about: resistance is ACQUIRED by an SNV landing on a
        # resistance locus and LOST by a CNA deleting the copy that carries it. Measured, the
        # acquisition:reversion ratio is 0.09 at mutagenicity 1.0 and 0.09 at 4.0 — turning the
        # mutator up cannot shift the balance, only the scale.
        # "snv" makes the drug a POINT MUTAGEN rather than a clastogen: cnv_prob is lowered by
        # exactly the factor that keeps the ABSOLUTE per-division CNA rate where it was, so extra
        # mutating divisions all become SNVs. Measured: reversion 1.10e-3 -> 5.83e-4 (back to the
        # no-mutator baseline) with acquisition held, ratio 0.09 -> 0.27. This is a claim about the
        # AGENT — platinums and alkylators are base-damaging and leave substitution signatures, which
        # is mechanistically distinct from inducing chromosome missegregation.
        # Default "all" -> both scale together -> byte-identical to before.
        self.mutagenicity_target = str(mutagenicity_target)
        self.dosage_trace = []  # list of (step, dosage) for every queried step

    def _apply(self, cell):
        pass

    def discrete_event(self, tumor, step):
        """One-shot state edits that are NOT rate modifiers (e.g. surgical resection). The base
        treatment does nothing; `Surgery` overrides this. The engine calls it once per step, so
        chemo-only / no-surgery runs stay byte-identical (no state change, no rng draw)."""
        pass

    def is_target(self, cell, **kwargs):
        pass

    def apply(self, cell, **kwargs):
        if self.is_target(cell, **kwargs) or (not self.is_target(cell, **kwargs) and np.random.binomial(1, self.toxicity)):
            if np.random.binomial(1, self.effectiveness):
                self._apply(cell)

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
