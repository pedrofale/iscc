from .treatment import Treatment
import numpy as np

class Immunotherapy(Treatment):
    # Immunotherapy works by lowering the cell's immune resistance; the actual killing
    # is then done by the local immune microenvironment (see the death-rate model).
    affects = "immune_resistance"

    def __init__(self, immune_checkpoints, **kwargs):
        super(Immunotherapy, self).__init__(**kwargs)
        self.targets = list(immune_checkpoints)

    def _apply(self, cell):
        # Decrease immune resistance; cells with high treatment resistance are also
        # refractory to immunotherapy, so their resistance is stripped less (exponent
        # 1-treatment_resistance). rate_multiplier > 1 => the factor below is < 1.
        tr = cell.evolutionary_parameters['treatment_resistance']
        factor = self.rate_multiplier ** (-(1. - tr))
        cell.evolutionary_parameters['immune_resistance'] = max(
            0., cell.evolutionary_parameters['immune_resistance'] * factor)

    def apply(self, cell, mut_effects):
        # Change some evolutionary parameter of the cell with some effectiveness. Chemo and TT: death rate. IT: immune resistance
        if self.is_target(cell, mut_effects) or (not self.is_target(cell, mut_effects) and np.random.binomial(1, self.toxicity)):
            if np.random.binomial(1, self.effectiveness):
                # Depending on wether the cell is sensitive to the treatment or not!
                # cell.evo_parameter = f(cell.evo_parameter, cell.treatment_effectiveness)
                self._apply(cell)
        return

    def is_target(self, cell, mut_effects, need_all=True):
        # Check if cell is targeted by this treatment. With no checkpoints specified,
        # the therapy is broad: every cancer cell is a target.
        if not self.targets:
            return cell.type == 'cancer'
        expressed = [cell.expresses(target, mut_effects) for target in self.targets]
        frac_expressed = sum(expressed) / len(expressed)
        if frac_expressed == 1:
            return True
        else:
            return (frac_expressed > 0) and not need_all