from .treatment import Treatment
import numpy as np

class Chemotherapy(Treatment):
    """Systemic cytotoxic therapy: raises the death rate of sensitive cancer cells.

    Every cancer cell is a target (normal cells are hit only via off-target
    ``toxicity``). A treated cell's death rate is multiplied by
    ``rate_multiplier ** (1 - treatment_resistance)``, so resistant clones
    (``treatment_resistance`` → 1) escape while sensitive clones regress. Pass an
    instance to [`GenotypeTumor`][iscc.tumor.GenotypeTumor]`.grow(..., treatment=chemo)`;
    resistance is meant to **emerge** under this pressure rather than be pre-seeded.

    Takes no targeting parameters of its own (every cancer cell is a target). See
    [`Treatment`][iscc.treatment.Treatment] for the shared dosing / scheduling parameters
    (``start``, ``duration``, ``rate_multiplier``, ``effectiveness``, ...).
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