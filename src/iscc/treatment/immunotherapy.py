from .treatment import Treatment

class Immunotherapy(Treatment):
    """Immunotherapy: strips a cancer cell's immune resistance so the local immune
    microenvironment can kill it.

    Unlike the death-rate therapies it does not kill directly — it *lowers*
    ``immune_resistance`` (by ``rate_multiplier ** -(1 - treatment_resistance)``, so
    cells that are also treatment-resistant are stripped less), and the killing is done
    by the local immune density in the death-rate model. With no ``immune_checkpoints``
    given it acts on every cancer cell; otherwise only on cells expressing the listed
    checkpoints. Pass an instance to [`GenotypeTumor`][iscc.tumor.GenotypeTumor]`.grow(..., treatment=it)`.

    Parameters
    ----------
    immune_checkpoints : list
        Checkpoint sites a cell must express to be targeted; empty = broad (every
        cancer cell is a target). With ``response="mechanistic"`` a clone must instead carry
        them, as for [`TargetedTherapy`][iscc.treatment.TargetedTherapy]'s ``targets``.
    **kwargs
        Forwarded to the [`Treatment`][iscc.treatment.Treatment] base. See
        [`Treatment`][iscc.treatment.Treatment] for the shared dosing / scheduling
        parameters (``start``, ``duration``, ``rate_multiplier``, ``effectiveness``, ...).
    """

    # Immunotherapy works by lowering the cell's immune resistance; the actual killing
    # is then done by the local immune microenvironment (see the death-rate model).
    affects = "immune_resistance"

    def __init__(self, immune_checkpoints, **kwargs):
        super(Immunotherapy, self).__init__(**kwargs)
        self.targets = list(immune_checkpoints)

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
