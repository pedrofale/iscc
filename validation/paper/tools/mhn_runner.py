"""Fit a Mutual Hazard Network to a cohort's events (Fig 5, Parts II and III).

Runs inside the mhn image (images/mhn.sh; the ``mhn`` package, spang-lab/LearnMHN), never in envs/iscc:

    bin/python-mhn mhn_runner.py <events.csv> <out_prefix> [--penalty sym_sparse] [--lam cv]

Input: a patients x events 0/1 CSV with a header of event names and the patient ids in its first
column, as ``iscc.integrations.progression.to_mhn_matrix(...).to_csv(path)`` writes it
(``--no-index`` for a file without the id column).

Model: the classical MHN (Schill et al. 2020), ``cMHNOptimizer``. The regularisation follows the
package's own demo (demo/demo.ipynb), its documented standard choice:

* ``--penalty sym_sparse`` (default): the symmetric penalty, which the authors recommend for
  mutational data (Schill et al. 2024); ``l1`` is the package default when no penalty is set.
* ``--lam cv`` (default): lambda by ``lambda_from_cv`` with the package defaults, a 9-step log grid
  over [0.1 / n, 10 / n], 5 folds and the one-standard-error rule. ``--lam default`` is ``train()``'s
  own 1 / n; a number fixes it.

Results are not rounded (``train``'s default rounds Theta to two decimals, which ties pairs when
they are ranked by |Theta|).

Output: ``<out_prefix>.theta.csv``, the events x events log-Theta: entry [i, j], i != j, is the
multiplicative effect (on the log scale) of event j being present on the rate of event i; the
diagonal is each event's base rate. And ``<out_prefix>.json``: the settings, the chosen lambda and
the CV scores, the optimiser's report, the versions in the image and the timings.

Temporary files go to a fresh directory under ``--work-dir`` (default ``$ISCC_WORK_DIR``, else
``$SNIC_TMP``; never /tmp), removed at the end.
"""
import argparse
import json
import os
import platform
import shutil
import sys
import tempfile
import time


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("events_csv")
    p.add_argument("out_prefix", help="writes <out_prefix>.theta.csv and <out_prefix>.json")
    p.add_argument("--penalty", default="sym_sparse", choices=["sym_sparse", "l1", "l2"])
    p.add_argument("--lam", default="cv", help="'cv' (default), 'default' (1 / n) or a number")
    p.add_argument("--nfolds", type=int, default=5)
    p.add_argument("--cv-steps", type=int, default=9)
    p.add_argument("--maxit", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0, help="numpy seed for the CV folds")
    p.add_argument("--no-index", action="store_true", help="the CSV has no patient-id column")
    p.add_argument("--work-dir", default=os.environ.get("ISCC_WORK_DIR") or os.environ.get("SNIC_TMP"))
    p.add_argument("--keep-work", action="store_true")
    return p.parse_args(argv)


def make_work_dir(base, prefix):
    if not base:
        sys.exit("no work directory: pass --work-dir or set ISCC_WORK_DIR / SNIC_TMP (never /tmp)")
    os.makedirs(base, exist_ok=True)
    work = tempfile.mkdtemp(prefix=prefix, dir=base)
    for sub in ("tmp", "mpl", "cache"):
        os.makedirs(os.path.join(work, sub), exist_ok=True)
    os.environ.update(TMPDIR=os.path.join(work, "tmp"), TEMP=os.path.join(work, "tmp"),
                      TMP=os.path.join(work, "tmp"), MPLCONFIGDIR=os.path.join(work, "mpl"),
                      XDG_CACHE_HOME=os.path.join(work, "cache"))
    tempfile.tempdir = os.path.join(work, "tmp")
    return work


def image_versions():
    try:
        with open("/opt/versions.txt") as f:
            return dict(line.split(" ", 1) for line in f.read().strip().splitlines())
    except OSError:
        return {}


def main(argv=None):
    args = parse_args(argv)
    work = make_work_dir(args.work_dir, "mhn-")
    cwd = os.getcwd()
    t0 = time.time()
    try:
        os.chdir(work)                               # anything the package writes lands here
        import numpy as np
        import pandas as pd
        from mhn.optimizers import cMHNOptimizer

        path = os.path.join(cwd, args.events_csv)
        X = pd.read_csv(path, header=0) if args.no_index else pd.read_csv(path, header=0, index_col=0)
        vals = X.to_numpy()
        if not np.isin(vals, [0, 1]).all():
            sys.exit("the event matrix must hold only 0 and 1")
        data = vals.astype(np.int32)
        n, d = data.shape
        events = [str(c) for c in X.columns]

        opt = cMHNOptimizer()
        opt.load_data_matrix(data)
        penalty = {"sym_sparse": opt.Penalty.SYM_SPARSE, "l1": opt.Penalty.L1,
                   "l2": opt.Penalty.L2}[args.penalty]
        opt.set_penalty(penalty)

        cv = None
        t_cv = time.time()
        if args.lam == "cv":
            np.random.seed(args.seed)
            lam, scores = opt.lambda_from_cv(nfolds=min(args.nfolds, n), steps=args.cv_steps,
                                             return_lambda_scores=True, pick_1se=True)
            grid = scores["Lambda Value"].to_numpy()
            cv = dict(nfolds=min(args.nfolds, n), steps=args.cv_steps, grid=[float(grid.min()), float(grid.max())],
                      grid_rule="package default, log grid over [0.1 / n, 10 / n]",
                      rule="one standard error", seed=args.seed,
                      scores=json.loads(scores.to_json(orient="records")),
                      at_grid_edge=bool(lam <= grid.min() or lam >= grid.max()))
        elif args.lam == "default":
            lam = 1.0 / n
        else:
            lam = float(args.lam)
        t_cv = time.time() - t_cv

        t_fit = time.time()
        res = opt.train(lam=float(lam), maxit=args.maxit, round_result=False)
        t_fit = time.time() - t_fit
        theta = np.asarray(res.log_theta)
        meta = getattr(res, "meta", {}) or {}

        out_theta = args.out_prefix + ".theta.csv"
        os.makedirs(os.path.dirname(os.path.join(cwd, out_theta)) or ".", exist_ok=True)
        pd.DataFrame(theta, index=events, columns=events).to_csv(os.path.join(cwd, out_theta))
        rec = dict(
            input=path, theta=os.path.join(cwd, out_theta), n_patients=int(n), n_events=int(d),
            events=events, event_frequencies=dict(zip(events, data.mean(0).round(4).tolist())),
            model="cMHN (mhn.optimizers.cMHNOptimizer)", penalty=args.penalty,
            lam_rule=args.lam, lam=float(lam), cv=cv, maxit=args.maxit, round_result=False,
            convention="theta[i, j] (i != j): log effect of event j on the rate of event i; "
                       "theta[i, i]: base log rate of event i",
            optimiser={k: (v.item() if hasattr(v, "item") else v) for k, v in meta.items()
                       if k in ("score", "message", "status", "nit")},
            versions=image_versions(), host=platform.node(), date=time.strftime("%Y-%m-%d %H:%M:%S"),
            timings=dict(cv_s=round(t_cv, 2), fit_s=round(t_fit, 2), total_s=round(time.time() - t0, 2)))
        with open(os.path.join(cwd, args.out_prefix + ".json"), "w") as f:
            json.dump(rec, f, indent=1, default=str)
        print(f"[mhn_runner] n={n} events={d} penalty={args.penalty} lambda={lam:.5g} "
              f"({args.lam}) in {rec['timings']['total_s']} s -> {out_theta}")
    finally:
        os.chdir(cwd)
        if not args.keep_work:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
