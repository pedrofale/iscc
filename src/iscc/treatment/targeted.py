from .treatment import Treatment

class TargetedTherapy(Treatment):
    """Targeted therapy: kills only cancer cells that express a given set of target sites.

    Like [`Chemotherapy`][iscc.treatment.Chemotherapy] it raises the death rate of sensitive
    cells (by ``rate_multiplier ** (1 - treatment_resistance)``), but a cell is a target only
    if it expresses the therapy's ``targets`` (mutated driver sites), modelling a
    biomarker-selected agent. Resistant clones escape and are selected for. Pass an
    instance to [`GenotypeTumor`][iscc.tumor.GenotypeTumor]`.grow(..., treatment=tt)`.

    Parameters
    ----------
    targets : list
        Genome coordinates (mutated driver sites) a cell must express to be targeted.
    adaptive : bool, optional
        If ``True``, dose only while the tumour exceeds ``max_tumor_size``; else dose
        continuously within the window (default ``False``).
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
        Death hazard on a fully-sensitive cell under full dose in the genotype engine
        (default 1.5); set above ``max_birth_rate`` so high-fitness sensitive clones regress.
    max_tumor_size : int, optional
        Size threshold that gates dosing when ``adaptive=True`` (default 100000).
    sites : {"both", "primary", "met"}, optional
        Compartment(s) the therapy acts on (default ``"both"`` = systemic).
    """

    def __init__(self, targets, **kwargs):
        # NB: must be super(TargetedTherapy, self); super(Treatment, self) would skip
        # Treatment.__init__ entirely and leave the dosing attributes unset.
        super(TargetedTherapy, self).__init__(**kwargs)
        self.targets = list(targets) # actual coordinates
    
    def _apply(self, cell):
        cell.evolutionary_parameters['death_rate'] = min(1., cell.evolutionary_parameters['death_rate'] * (self.rate_multiplier ** (1.-cell.evolutionary_parameters['treatment_resistance']))) # increase death rate inversely proportionally to treatment resistance

    def is_target(self, cell, mut_effects, need_all=True):
        # Check if cell is targeted by this treatment
        expressed = [cell.expresses(target, mut_effects) for target in self.targets] 
        frac_expressed = sum(expressed) / len(expressed)
        if frac_expressed == 1:
            return True
        else:
            return (frac_expressed > 0) and not need_all