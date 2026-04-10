import numpy as np

class Treatment(object):
    def __init__(self, adaptive=False, start=0, duration=None,
                 dosage_decay=0.5, rounds=4,
                 rate_multiplier=2., toxicity=0.1, effectiveness=0.9,
                 max_tumor_size=100_000):
        self.adaptive = adaptive
        self.start = start
        self.duration = duration
        self.rate_multiplier = rate_multiplier
        self.toxicity = toxicity
        self.effectiveness = effectiveness
        self.dosage_decay = dosage_decay
        self.rounds = rounds
        self.max_tumor_size = max_tumor_size
        self.dosage_trace = []  # list of (step, dosage) for every queried step

    def _apply(self, cell):
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
