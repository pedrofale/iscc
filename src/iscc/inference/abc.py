"""Generic Approximate Bayesian Computation engine (DESIGN_inference A.1).

Dependency-light (numpy + scikit-learn + scipy; **no JAX, no R**). The engine is model-agnostic:
you give it priors and a ``simulate(theta) -> summary_vector`` callback (for the tumour model,
``inference.tumor`` wires this up), and it returns posterior samples for the parameters.

Two posteriors are produced, mirroring CINner's ABC-rf pipeline:
  * **Rejection ABC** — draw ``N`` parameter sets from the priors, simulate each, keep the
    ``accept_frac`` closest to the observed summary under a per-summary-standardised Euclidean
    distance. Transparent and assumption-free.
  * **Regression adjustment** — Beaumont et al. (2002) local-linear correction of the accepted
    sample (numpy only), tightening the posterior by regressing parameters on the residual
    summary distance. Plus a Random-Forest point estimate (``sklearn``), the Python analogue of
    CINner's ABC-rf, fit on the full reference table.

Simulations are independent and parallelised with ``multiprocessing``.
"""
from dataclasses import dataclass, field

import numpy as np


# --- priors -----------------------------------------------------------------
class Prior:
    """A product prior over named parameters.

    Each parameter maps to a sampler, given as one of:
      * ``(low, high)`` tuple -> ``Uniform(low, high)``;
      * a frozen ``scipy.stats`` distribution (anything with ``.rvs`` / ``.pdf``);
      * a callable ``rng -> sample`` (no density; logpdf falls back to a flat 0).
    """

    def __init__(self, specs):
        self.names = list(specs)
        self.specs = specs

    def sample(self, rng, n):
        """Draw ``n`` parameter sets -> ``(n, n_params)`` array, columns ordered as ``names``."""
        cols = []
        for name in self.names:
            spec = self.specs[name]
            if isinstance(spec, tuple):
                low, high = spec
                cols.append(rng.uniform(low, high, size=n))
            elif hasattr(spec, "rvs"):
                cols.append(spec.rvs(size=n, random_state=rng))
            elif callable(spec):
                cols.append(np.array([spec(rng) for _ in range(n)], dtype=float))
            else:
                raise TypeError(f"unsupported prior spec for {name!r}: {spec!r}")
        return np.column_stack(cols).astype(float)


# --- posterior container ----------------------------------------------------
@dataclass
class Posterior:
    """Result of an ABC run. ``samples`` is the (regression-adjusted) posterior."""
    names: list
    samples: np.ndarray            # accepted, regression-adjusted -> the posterior
    rejection_samples: np.ndarray  # accepted, *unadjusted*
    rf_estimate: np.ndarray        # Random-Forest point estimate at the observation
    all_theta: np.ndarray = field(repr=False, default=None)
    all_summaries: np.ndarray = field(repr=False, default=None)
    distances: np.ndarray = field(repr=False, default=None)

    def mean(self):
        return self.samples.mean(axis=0)

    def median(self):
        return np.median(self.samples, axis=0)

    def map(self):
        """Per-parameter MAP via a 1-D Gaussian-KDE mode of the posterior marginal."""
        from scipy.stats import gaussian_kde
        out = []
        for j in range(self.samples.shape[1]):
            col = self.samples[:, j]
            if np.ptp(col) == 0:
                out.append(float(col[0]))
                continue
            kde = gaussian_kde(col)
            grid = np.linspace(col.min(), col.max(), 256)
            out.append(float(grid[np.argmax(kde(grid))]))
        return np.asarray(out)

    def credible_interval(self, level=0.9):
        """Per-parameter equal-tailed credible interval -> ``(lo, hi)`` arrays."""
        a = (1 - level) / 2
        lo = np.quantile(self.samples, a, axis=0)
        hi = np.quantile(self.samples, 1 - a, axis=0)
        return lo, hi

    def as_dict(self, estimator="map"):
        est = getattr(self, estimator)() if estimator != "rf" else self.rf_estimate
        return dict(zip(self.names, est))


# --- simulation (top-level so multiprocessing can pickle it) -----------------
def _simulate_one(args):
    simulate, theta, names = args
    try:
        s = np.asarray(simulate(dict(zip(names, theta))), dtype=float)
    except Exception:
        s = None
    return s


def _run_simulations(simulate, thetas, names, n_workers):
    tasks = [(simulate, theta, names) for theta in thetas]
    if n_workers and n_workers > 1:
        import multiprocessing as mp
        with mp.Pool(n_workers) as pool:
            results = pool.map(_simulate_one, tasks)
    else:
        results = [_simulate_one(t) for t in tasks]
    return results


# --- distance / adjustment helpers ------------------------------------------
def _standardise(thetas, summaries, observed):
    """Filter to usable rows/columns and robustly standardise (median / MAD).

    Drops summary columns that are non-finite in the observation, then reference rows with any
    non-finite value in the surviving columns (e.g. extinct tumours return an all-``nan`` vector),
    then columns with ~zero spread (uninformative). Returns ``(thetas_kept, S_std, obs_std)`` so
    the row filtering stays consistent between parameters and summaries.
    """
    col = np.isfinite(observed)
    S = summaries[:, col]
    obs = observed[col]
    row = np.isfinite(S).all(axis=1)
    S = S[row]
    thetas = thetas[row]
    med = np.median(S, axis=0)
    mad = np.median(np.abs(S - med), axis=0) * 1.4826
    keep = mad > 1e-9
    scale = mad[keep]
    return thetas, S[:, keep] / scale, obs[keep] / scale


def _regression_adjust(theta_acc, S_acc, obs):
    """Beaumont et al. (2002) local-linear regression adjustment of accepted parameters.

    Regress each parameter on the (already standardised) summaries over the accepted set and
    correct for the residual distance to the observation: ``theta* = theta - (S - obs) @ beta``.
    Returns the adjusted parameter sample (same shape as ``theta_acc``).
    """
    n = S_acc.shape[0]
    X = S_acc - obs
    A = np.hstack([np.ones((n, 1)), X])
    beta, *_ = np.linalg.lstsq(A, theta_acc, rcond=None)
    return theta_acc - X @ beta[1:]


# --- engine -----------------------------------------------------------------
class ABC:
    def __init__(self, prior, simulate, n_workers=1, seed=0):
        """``prior``: a :class:`Prior`. ``simulate``: ``dict(name->value) -> summary vector``."""
        self.prior = prior
        self.simulate = simulate
        self.n_workers = n_workers
        self.rng = np.random.default_rng(seed)

    def reference_table(self, n_samples):
        """Draw ``n_samples`` priors and simulate them -> ``(theta, summaries)`` (failures dropped)."""
        thetas = self.prior.sample(self.rng, n_samples)
        results = _run_simulations(self.simulate, thetas, self.prior.names, self.n_workers)
        keep = [i for i, s in enumerate(results) if s is not None]
        thetas = thetas[keep]
        summaries = np.vstack([results[i] for i in keep])
        return thetas, summaries

    def run(self, observed, n_samples=1000, accept_frac=0.1, adjust=True,
            rf=True, project="rf", reference=None):
        """Run ABC against ``observed`` -> :class:`Posterior`.

        ``reference`` optionally reuses a precomputed ``(theta, summaries)`` table (so the same
        sims can serve several observations). ``accept_frac`` sets the rejection tolerance.

        ``project="rf"`` enables **semi-automatic ABC** (Fearnhead & Prangle 2012): a random
        forest is fit to map summaries -> parameters, and the rejection/regression step runs in
        that low-dimensional *predicted-parameter* space. Each projected coordinate targets one
        parameter on a comparable scale, so a single clean summary can no longer dominate the
        distance and starve the others (the failure mode of raw standardised-Euclidean matching
        when parameters are identified by different summaries). ``project=None`` uses the raw
        standardised summaries. The reference table is projected through the forest's
        out-of-bag predictions to avoid the train-on / match-on double-dip.
        """
        observed = np.asarray(observed, dtype=float)
        thetas_all, summaries = reference if reference is not None else self.reference_table(n_samples)

        thetas, Sstd, obs_std = _standardise(thetas_all, summaries, observed)

        forest = None
        rf_estimate = np.full(thetas.shape[1], np.nan)
        if rf or project == "rf":
            from sklearn.ensemble import RandomForestRegressor
            forest = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, oob_score=True,
                                           random_state=0, n_jobs=self.n_workers or 1)
            forest.fit(Sstd, thetas)
            rf_estimate = forest.predict(obs_std[None, :])[0]

        if project == "rf":
            # match in out-of-bag predicted-parameter space (honest projection of the reference)
            Smatch = np.atleast_2d(forest.oob_prediction_)
            obs_match = forest.predict(obs_std[None, :])[0]
            med = np.median(Smatch, axis=0)
            mad = np.median(np.abs(Smatch - med), axis=0) * 1.4826
            scale = np.where(mad > 1e-9, mad, 1.0)
            Smatch = Smatch / scale
            obs_match = obs_match / scale
        else:
            Smatch, obs_match = Sstd, obs_std

        dist = np.sqrt(((Smatch - obs_match) ** 2).sum(axis=1))
        n_acc = max(2, int(round(accept_frac * len(dist))))
        order = np.argsort(dist)[:n_acc]
        theta_acc = thetas[order]
        rejection = theta_acc.copy()

        samples = (_regression_adjust(theta_acc, Smatch[order], obs_match)
                   if adjust and len(order) > Smatch.shape[1] + 1 else theta_acc)

        return Posterior(
            names=self.prior.names, samples=samples, rejection_samples=rejection,
            rf_estimate=np.atleast_1d(rf_estimate), all_theta=thetas,
            all_summaries=summaries, distances=dist,
        )
