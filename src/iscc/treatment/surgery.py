from .treatment import Treatment


class Surgery(Treatment):
    """Surgical resection: a ONE-SHOT removal of tissue at a fixed step — not a rate modifier. At
    (or after) ``start`` every cell in ``site`` (default ``"primary"``) is removed, or with ``region``
    only the cells of the excised specimen, and the simulation continues on what is left.

    Implemented through the ``discrete_event`` hook (the base ``Treatment`` defines it as a no-op), so
    a run with no ``Surgery`` is byte-identical. ``get_dosage`` returns 0 always, so ``Surgery`` never
    contributes a death-rate modifier — its only effect is the discrete resection. The genealogy is
    not pruned, so a 2-band Muller still shows the resected compartment's band cliff to 0 at ``start``.

    A whole-compartment resection cures that compartment by construction. A partial one (``region``)
    clears only the specimen's demes, cancer and normal residents alike; everything outside keeps
    growing. It appends the ground truth of the cut to the tumour's ``resections`` list and returns it
    from ``discrete_event``: a dict with

    * ``step``, ``time``, ``site``, ``region`` (the cleared demes, as global deme indices) and
      ``margin``;
    * ``removed``: cancer cells per genotype inside the region;
    * ``margin_cancer``: cancer cells per genotype in the rim of the region within ``margin`` demes
      of the cut, and ``margin_positive`` (any there at all) — the TRUE margin status;
    * ``residual``: a table (``deme``, ``row``, ``col``, ``distance``, ``genotype``, ``n``) of the
      cancer cells left in the compartment outside the region, with each deme's distance from the
      cut.

    Distances are in demes, centre to centre, to the nearest deme on the other side of the cut, so a
    deme touching the cut is at distance 1. The grid's own edge is not a cut. What a pathologist
    would report from the specimen is for the caller to decide.

    Parameters
    ----------
    site : {"primary", "met"}, optional
        Compartment resected at ``start`` (default ``"primary"``).
    start : int, optional
        Step at (or after) which the one-shot resection fires (default 0).
    region : array-like, optional
        The excised specimen, as global deme indices in the compartment (e.g. from
        [`Resection.bisect`][iscc.sample.Resection.bisect]) or a boolean mask over the
        compartment's demes (flat, or ``(side, side)``). Default None: the whole compartment, as
        before.
    margin : float, optional
        Width, in demes, of the rim just inside the cut that a pathologist would inspect
        (default 0: no rim). A deme is ~50 µm, so a 2 mm margin is ~40 demes; convert before
        passing. Needs ``region``.
    **kwargs
        Forwarded to the [`Treatment`][iscc.treatment.Treatment] base. See
        [`Treatment`][iscc.treatment.Treatment] for the shared dosing / scheduling
        parameters; note ``Surgery`` never doses (``get_dosage`` is always 0), so only
        ``start`` is consulted for its schedule.
    """

    affects = "resection"

    def __init__(self, start=0, site="primary", region=None, margin=0, **kwargs):
        super(Surgery, self).__init__(start=start, **kwargs)
        if margin < 0:
            raise ValueError(f"margin must be non-negative, got {margin}")
        if margin and region is None:
            raise ValueError("margin needs a region: a whole-compartment resection has no cut")
        self.site = site
        self.region = region
        self.margin = margin
        self._done = False

    def get_dosage(self, step, tumor_size):
        return 0.0  # a discrete event, not a dose

    def discrete_event(self, tumor, step):
        if not self._done and step >= self.start:
            if self.region is None:
                tumor._resect(self.site)
                self._done = True
                return None
            record = tumor._resect(self.site, region=self.region, margin=self.margin)
            self._done = True
            return record
