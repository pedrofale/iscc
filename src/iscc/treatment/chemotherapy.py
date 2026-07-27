from .treatment import Treatment
import numpy as np

class Chemotherapy(Treatment):
    """Systemic cytotoxic therapy: raises the death rate of sensitive cancer cells.

    Every cancer cell is a target (normal cells are hit only via off-target
    ``toxicity``). A treated cell's death rate is multiplied by
    ``rate_multiplier ** (1 - treatment_resistance)``, so resistant clones
    (``treatment_resistance`` → 1) escape while sensitive clones regress. Pass an
    instance to ``GenotypeTumor.grow(..., treatment=chemo)``; resistance is meant to
    **emerge** under this pressure rather than be pre-seeded.

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
        Factor applied to a fully-sensitive cell's death rate under full dose (default 2.0).
    toxicity : float, optional
        Per-step probability of off-target action on a non-cancer cell (default 0.1).
    effectiveness : float, optional
        Per-step probability the therapy acts on a targeted cell (default 0.9).
    kill_rate : float, optional
        Death hazard imposed on a fully-sensitive cell under full dose in the genotype
        engine (default 1.5); set above ``max_birth_rate`` so even high-fitness
        (driver-amplified) sensitive clones still regress.
    max_tumor_size : int, optional
        Size threshold that gates dosing when ``adaptive=True`` (default 100000).
    sites : {"both", "primary", "met"}, optional
        Compartment(s) the therapy acts on (default ``"both"`` = systemic): ``"primary"``
        (neoadjuvant / local) or ``"met"`` (adjuvant after primary resection).
    """

    def __init__(self, **kwargs):
        super(Chemotherapy, self).__init__(**kwargs)

    def _apply(self, cell): # if not resistant to treatment, 
        cell.evolutionary_parameters['death_rate'] = min(1., cell.evolutionary_parameters['death_rate'] * (self.rate_multiplier ** (1.-cell.evolutionary_parameters['treatment_resistance']))) # increase death rate inversely proportionally to treatment resistance
    
    def apply(self, cell, **kwargs):
        # Change some evolutionary parameter of the cell with some effectiveness. Chemo and TT: death rate. IT: immune resistance
        if self.is_target(cell, **kwargs) or (not self.is_target(cell, **kwargs) and np.random.binomial(1, self.toxicity)):
            if np.random.binomial(1, self.effectiveness):
                # Depending on wether the cell is sensitive to the treatment or not!
                # cell.evo_parameter = f(cell.evo_parameter, cell.treatment_effectiveness)
                self._apply(cell)
        return

    def is_target(self, cell, **kwargs):
        # Check if cell is targeted by this treatment
        if cell.type == 'cancer':
            return True
        else:
            return False