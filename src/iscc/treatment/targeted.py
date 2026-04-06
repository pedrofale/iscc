from .treatment import Treatment
import numpy as np

class TargetedTherapy(Treatment):
    def __init__(self, targets, **kwargs):
        super(Treatment, self).__init__(**kwargs)
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