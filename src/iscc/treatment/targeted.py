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
    **kwargs
        Forwarded to the [`Treatment`][iscc.treatment.Treatment] base. See
        [`Treatment`][iscc.treatment.Treatment] for the shared dosing / scheduling
        parameters (``start``, ``duration``, ``rate_multiplier``, ``effectiveness``, ...).
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