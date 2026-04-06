from .treatment import Treatment
import numpy as np

class Chemotherapy(Treatment):
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