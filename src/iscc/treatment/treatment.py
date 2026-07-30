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
                 kill_rate=1.5, max_tumor_size=100_000, sites="both"):
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
