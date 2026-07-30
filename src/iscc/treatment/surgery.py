from .treatment import Treatment


class Surgery(Treatment):
    """Surgical resection: a ONE-SHOT removal of an entire compartment at a
    fixed step — not a rate modifier. At (or after) ``start`` every cell in ``site`` (default
    ``"primary"``) is removed and the simulation continues on the remaining compartment(s).

    Implemented through the ``discrete_event`` hook (the base ``Treatment`` defines it as a no-op), so
    a run with no ``Surgery`` is byte-identical. ``get_dosage`` returns 0 always, so ``Surgery`` never
    contributes a death-rate modifier — its only effect is the discrete resection. The genealogy is
    not pruned, so a 2-band Muller still shows the resected compartment's band cliff to 0 at ``start``.

    Parameters
    ----------
    site : {"primary", "met"}, optional
        Compartment resected at ``start`` (default ``"primary"``).
    start : int, optional
        Step at (or after) which the one-shot resection fires (default 0).
    **kwargs
        Forwarded to the [`Treatment`][iscc.treatment.Treatment] base. See
        [`Treatment`][iscc.treatment.Treatment] for the shared dosing / scheduling
        parameters; note ``Surgery`` never doses (``get_dosage`` is always 0), so only
        ``start`` is consulted for its schedule.
    """

    affects = "resection"

    def __init__(self, start=0, site="primary", **kwargs):
        super(Surgery, self).__init__(start=start, **kwargs)
        self.site = site
        self._done = False

    def get_dosage(self, step, tumor_size):
        return 0.0  # a discrete event, not a dose

    def discrete_event(self, tumor, step):
        if not self._done and step >= self.start:
            tumor._resect(self.site)
            self._done = True
