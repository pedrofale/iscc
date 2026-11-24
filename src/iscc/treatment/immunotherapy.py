from .treatment import Treatment

class Immunotherapy(Treatment):
    """Immunotherapy, which lowers the immune resistance of cancer cells so that immune cells kill them.

    It adds no death hazard of its own. While dosed, the immune resistance of a targeted clone is
    multiplied by ``1 - x``, where ``x`` is the dose the clone takes (``effectiveness`` times
    ``1 - treatment_resistance``, see [`Treatment`][iscc.treatment.Treatment]), so clones that are
    also treatment-resistant keep more of it. Cancer clones that are not targets take the
    ``toxicity`` share instead. The killing is done by the immune cells of the deme,
    through the death hazard ``prob_kill`` (of ``immune_cell_params``) times the immune fraction of
    the deme times one minus the immune resistance. With no ``immune_checkpoints`` the therapy acts
    on every cancer cell, and otherwise only on cells that express the listed checkpoints. Pass an
    instance to [`GenotypeTumor.grow`][iscc.tumor.GenotypeTumor.grow] as ``treatment=it``.

    Parameters
    ----------
    immune_checkpoints : list
        Checkpoint sites a cell must express to be targeted; empty = broad (every
        cancer cell is a target). With ``response="mechanistic"`` a clone must instead carry
        them, as for [`TargetedTherapy`][iscc.treatment.TargetedTherapy]'s ``targets``.
    **kwargs
        Forwarded to the [`Treatment`][iscc.treatment.Treatment] base. See
        [`Treatment`][iscc.treatment.Treatment] for the shared dosing / scheduling
        parameters (``start``, ``duration``, ``effectiveness``, ...).
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
