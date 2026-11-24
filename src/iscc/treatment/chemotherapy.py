from .treatment import Treatment

class Chemotherapy(Treatment):
    """Systemic cytotoxic therapy, which adds a death hazard to sensitive cancer cells while dosed.

    Every cancer cell is a target, and normal cells take the off-target ``toxicity`` share of the
    dose. The added hazard is set by ``kill_rate`` and ``kill_mode`` and scaled down by the
    resistance of the clone, as described in [`Treatment`][iscc.treatment.Treatment], so resistant
    clones (``treatment_resistance`` near 1) escape while sensitive clones regress. Pass an
    instance to [`GenotypeTumor.grow`][iscc.tumor.GenotypeTumor.grow] as ``treatment=chemo``. The
    therapy seeds no resistance, which arises by mutation and is selected under the dose.

    Takes no targeting parameters of its own. See [`Treatment`][iscc.treatment.Treatment] for the
    shared dosing parameters (``start``, ``duration``, ``kill_rate``, ``kill_mode``,
    ``effectiveness``, ...).
    """

    def __init__(self, **kwargs):
        super(Chemotherapy, self).__init__(**kwargs)

    def is_target(self, cell, **kwargs):
        # Check if cell is targeted by this treatment
        if cell.type == 'cancer':
            return True
        else:
            return False
