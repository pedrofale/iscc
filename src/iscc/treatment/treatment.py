import numpy as np

class Treatment(object):
    def __init__(self, adaptive=True, dosage_decay=0.5, rounds=4,
                 rate_multiplier=2., toxicity=0.1, effectiveness=0.9,
                 max_tumor_size=100_000):
        self.rate_multiplier = rate_multiplier
        self.toxicity = toxicity
        self.effectiveness = effectiveness
        self.adaptive = adaptive
        self.dosage_decay = dosage_decay
        self.rounds = rounds
        self.max_tumor_size = max_tumor_size

    def _apply(self, cell):
        pass

    def is_target(self, cell, **kwargs):
        # Check if cell is targeted by this treatment
        pass

    def apply(self, cell, **kwargs):
        # Change some evolutionary parameter of the cell with some effectiveness. Chemo and TT: death rate. IT: immune resistance
        if self.is_target(cell, **kwargs) or (not self.is_target(cell, **kwargs) and np.random.binomial(1, self.toxicity)):
            if np.random.binomial(1, self.effectiveness):
                # Depending on wether the cell is sensitive to the treatment or not!
                # cell.evo_parameter = f(cell.evo_parameter, cell.treatment_effectiveness)
                self._apply(cell)
        return

    def get_dosage(self, tumor_size):
        # Check size of tumor, change since last dosage, and compute new dosage according to adaptive therapy params or ignore if not adaptive
        if self.adaptive:
            dosage = 1. if tumor_size > self.max_tumor_size else 0. # compute...
        else:
            dosage = 1. # or maximum tolerated dose...
        return dosage
