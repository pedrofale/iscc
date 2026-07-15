"""MHN runner — executed in the dedicated ``iscc-mhn`` env, never imported by the core env.

Fits a Mutual Hazard Network (Schill et al. 2020) to a cross-sectional patients x events binary
matrix and writes back the learned Theta. The off-diagonal Theta[i, j] is the multiplicative effect
of event j on the RATE of event i — MHN's analogue of iscc's planted E (they are NOT the same
parameter: iscc's E is a fitness/selection coefficient, MHN's Theta is a rate modifier, so we score
recovered EDGES, not values).

Usage:  python mhn_runner.py <input_matrix.csv> <output_theta.csv> [lambda]
        lambda omitted or "cv" -> pick the regularization by cross-validation.
"""
import sys

import numpy as np
import pandas as pd

from mhn.optimizers import cMHNOptimizer


def main():
    in_csv, out_csv = sys.argv[1], sys.argv[2]
    lam_arg = sys.argv[3] if len(sys.argv) > 3 else "cv"

    X = pd.read_csv(in_csv, index_col=0)
    data = X.values.astype(np.int32)

    opt = cMHNOptimizer()
    opt.load_data_matrix(data)

    if lam_arg == "cv":
        # MHN is regularized; picking lambda by CV is what the method prescribes, and it matters --
        # too little penalty and every pair gets a non-zero Theta (all false positives).
        try:
            lam = opt.lambda_from_cv(nfolds=min(5, len(data)), show_progressbar=False)
        except Exception as e:                     # tiny cohorts can break CV; fall back to the default
            print(f"[mhn_runner] CV failed ({e}); falling back to lam=1/n", file=sys.stderr)
            lam = 1.0 / len(data)
    else:
        lam = float(lam_arg)

    result = opt.train(lam=lam, maxit=5000)
    theta = np.asarray(result.log_theta if hasattr(result, "log_theta") else result.theta)

    pd.DataFrame(theta, index=X.columns, columns=X.columns).to_csv(out_csv)
    print(f"[mhn_runner] n={len(data)} events={data.shape[1]} lambda={lam:.5f} -> {out_csv}")


if __name__ == "__main__":
    main()
