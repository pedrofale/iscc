"""M3a: the real-tumour empirical (n, D, J1) table is valid.

Guards the committed `validation/data/noble_empirical_indices.csv` (computed from the published
Noble et al. 2022 phylogenies by `build_noble_empirical_indices.py`, using iscc's own index
functions). The full simulated-vs-real overlay is in `validation/validate_evolution_modes.py`.
"""
import os

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(REPO, "validation/data/noble_empirical_indices.csv")


def test_empirical_indices_table_is_valid():
    df = pd.read_csv(CSV)
    assert set(["tumour", "cancer_type", "n", "D", "J1", "n_clones"]).issubset(df.columns)
    assert len(df) >= 35                      # the Noble compilation (we have 43)
    assert df.cancer_type.nunique() >= 6      # ccRCC, NSCLC, breast, uveal, meso, AML

    # values fall in Noble's reported empirical ranges (n up to ~14, 1 <= D <= ~12, J1 in [0,1])
    assert df.n.between(0, 14).all()
    assert df.D.between(1.0, 12.0).all()
    j = df.J1.dropna()
    assert j.between(0, 1).all()
    assert df.n_clones.min() >= 1
