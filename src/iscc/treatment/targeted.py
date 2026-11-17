from .treatment import Treatment

class TargetedTherapy(Treatment):
    """Targeted therapy, which adds a death hazard only to cancer cells that carry its targets.

    Like [`Chemotherapy`][iscc.treatment.Chemotherapy], it adds a death hazard to sensitive cells,
    set by ``kill_rate`` and ``kill_mode`` and scaled down by the resistance of the clone, but a
    cancer clone is a target only if it expresses the therapy's ``targets`` (mutated driver sites),
    as for a biomarker-selected agent. Other cancer clones and normal cells take the off-target
    ``toxicity`` share of the dose. Resistant clones escape and are selected. Pass an instance to
    [`GenotypeTumor.grow`][iscc.tumor.GenotypeTumor.grow] as ``treatment=tt``.

    Parameters
    ----------
    targets : list
        Genome coordinates (mutated driver sites) a cell must express to be targeted. With
        ``response="mechanistic"`` a clone is targeted only if it CARRIES every target: a copy
        of the segment with the mutation at that site, or more copies of the segment than
        ``amplification_threshold``. A lineage that loses the target by CNA stops responding.
    **kwargs
        Forwarded to the [`Treatment`][iscc.treatment.Treatment] base. See
        [`Treatment`][iscc.treatment.Treatment] for the shared dosing / scheduling
        parameters (``start``, ``duration``, ``kill_rate``, ``effectiveness``, ...).
    """

    def __init__(self, targets, **kwargs):
        # NB: must be super(TargetedTherapy, self); super(Treatment, self) would skip
        # Treatment.__init__ entirely and leave the dosing attributes unset.
        super(TargetedTherapy, self).__init__(**kwargs)
        self.targets = list(targets) # actual coordinates
    
    def is_target(self, cell, mut_effects, need_all=True):
        # Check if cell is targeted by this treatment
        expressed = [cell.expresses(target, mut_effects) for target in self.targets] 
        frac_expressed = sum(expressed) / len(expressed)
        if frac_expressed == 1:
            return True
        else:
            return (frac_expressed > 0) and not need_all
