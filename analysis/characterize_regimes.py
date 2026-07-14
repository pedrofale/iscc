"""Layer 1 of the operating envelope (DESIGN_operating_envelope.md): the characterisation sweep.

Sweeps the key parameter AXES one at a time around a realistic baseline and records the phenotype
METRICS + degenerate-regime flags for every run, one tidy row per ``(axis, value, seed)``. The
output CSV is the map that the phase-diagram figure (``validation/validate_operating_envelope.py``)
and the manuscript "operating regimes" section are built on, and it is what the default thresholds in
``iscc.tumor.diagnostics`` were sanity-checked against.

This is a MAP, not an inference: small tumours, a coarse grid and a few fixed seeds per cell, so it
is fast and fully reproducible. Metrics are computed by the same read-only ``tumor.diagnose()`` the
built-in QC uses, so the atlas and the runtime warning agree by construction.

Usage:
    python analysis/characterize_regimes.py            # full sweep -> analysis/characterize_regimes.csv
    python analysis/characterize_regimes.py --quick    # tiny sweep (smoke / CI)
"""
import argparse
import copy
import os

import numpy as np
import pandas as pd

from iscc.tumor.models import GenotypeTumor
from iscc.tumor.diagnostics import diagnose

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = os.path.join(HERE, "characterize_regimes.csv")

# A realistic baseline (a scaled-down example_config that lands in the good region). Every axis
# sweep perturbs ONE group of knobs away from this and leaves the rest fixed.
BASELINE = dict(
    genome_params=dict(n_segments=5, segment_size=200),
    selection_params=dict(prop_driver=0.1, prop_dispersal=0.1, prop_immune_resistance=0.1,
                          prop_treatment_resistance=0.1, driver_effects=1.1, dispersal_effects=1.1,
                          immune_resistant_effects=1.1, treatment_resistant_effects=1.1),
    cancer_cell_params=dict(max_birth_rate=0.8, division_rate=0.3, death_rate=0.02,
                            mutation_rate=0.2, dispersal_rate=0.1, snv_prob=0.5, cnv_prob=0.5,
                            n_snvs_per_allele=0.3, amp_prob=0.5),
    deme_params=dict(carrying_capacity=10, initial_cancer_cells=5, initial_death_rate=0.1,
                     maximum_death_rate=1.0),
    spatial_params=dict(grid_size=22, structure_radius=0),
    microenv_params=None,
)

# A microenvironment (hypoxia) config for the gradient axis. "uniform" O2 source so the gradient
# depends on the tumour having an oxygenated margin -> a small tumour / long diffusion length has
# no core-rim contrast (the no-gradient degenerate regime).
HYPOXIA = dict(hypoxia=dict(strength=1.0, n_genes=40, o2_supply=0.3, o2_source="uniform"))


def _build(over):
    """Deep-merge ``over`` (nested param overrides) onto the baseline and build a tumour."""
    p = copy.deepcopy(BASELINE)
    for group, vals in over.items():
        if vals is None:
            p[group] = None
        elif isinstance(p[group], dict):
            p[group] = {**p[group], **vals}
        else:
            p[group] = vals
    return GenotypeTumor(
        genome_params=p["genome_params"], selection_params=p["selection_params"],
        cancer_cell_params=p["cancer_cell_params"], deme_params=p["deme_params"],
        spatial_params=p["spatial_params"], microenv_params=p["microenv_params"], seed=0)


# Each axis: name -> (list of (label, value, override-dict), n_steps). ``override`` is deep-merged.
def _axes(quick=False):
    def vals(full, q):
        return q if quick else full

    axes = {}
    # 1. SNV load: mutation_rate x n_snvs_per_allele
    axes["mutation_rate"] = ([("mutation_rate", mr, {"cancer_cell_params": {"mutation_rate": mr}})
                              for mr in vals([0.001, 0.02, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0],
                                             [0.001, 0.2, 4.0])], 1600)
    axes["n_snvs_per_allele"] = (
        [("n_snvs_per_allele", n, {"cancer_cell_params": {"n_snvs_per_allele": n}})
         for n in vals([0.01, 0.1, 0.3, 1.0, 3.0, 8.0], [0.01, 0.3, 8.0])], 1600)
    # 2. survival: death_rate (vs division_rate=0.3) and initial_cancer_cells
    axes["death_rate"] = ([("death_rate", dr, {"cancer_cell_params": {"death_rate": dr}})
                           for dr in vals([0.02, 0.1, 0.2, 0.28, 0.3, 0.35], [0.02, 0.3, 0.35])], 1600)
    # founder bottleneck: with death 0.15 vs division 0.3 a single founder has P(extinction)~0.5,
    # which vanishes as the initial cluster grows -> the ~7% one-cell founder-extinction problem.
    axes["initial_cancer_cells"] = (
        [("initial_cancer_cells", n, {"cancer_cell_params": {"death_rate": 0.15},
                                      "deme_params": {"initial_cancer_cells": n}})
         for n in vals([1, 2, 5, 10, 20], [1, 20])], 1600)
    # 3. spatial mixing: dispersal_rate relative to division_rate (0.3)
    axes["dispersal_rate"] = ([("dispersal_rate", d, {"cancer_cell_params": {"dispersal_rate": d}})
                               for d in vals([0.05, 0.1, 0.3, 1.0, 3.0, 8.0], [0.05, 1.0, 8.0])], 1800)
    # 4. selection strength: prop_driver x driver_effects
    axes["driver_effects"] = ([("driver_effects", e, {"selection_params": {"driver_effects": e}})
                               for e in vals([1.0, 1.1, 1.5, 2.0, 3.0], [1.0, 3.0])], 1600)
    axes["prop_driver"] = ([("prop_driver", p, {"selection_params": {"prop_driver": p,
                                                                     "driver_effects": 2.0}})
                            for p in vals([0.0, 0.05, 0.1, 0.3, 0.6], [0.0, 0.6])], 1600)
    # 5. tumour size: grid_size and carrying_capacity
    axes["carrying_capacity"] = ([("carrying_capacity", c, {"deme_params": {"carrying_capacity": c}})
                                  for c in vals([1, 2, 5, 10, 20], [1, 20])], 1600)
    axes["grid_size"] = ([("grid_size", g, {"spatial_params": {"grid_size": g}})
                          for g in vals([8, 12, 16, 22, 30], [8, 30])], 1600)
    # 6. CNA burden: amp_prob and max_cn
    axes["amp_prob"] = ([("amp_prob", a, {"cancer_cell_params": {"amp_prob": a, "cnv_prob": 0.9,
                                                                 "snv_prob": 0.1}})
                         for a in vals([0.2, 0.5, 0.8, 0.95, 1.0], [0.2, 1.0])], 1600)
    # max_cn is the copy-number viability cap; move max_ploidy with it so it actually binds, under a
    # CNA-heavy amplifying regime (else ploidy sits well below the cap and max_cn has no effect).
    axes["max_cn"] = ([("max_cn", m, {"cancer_cell_params": {"cnv_prob": 0.9, "snv_prob": 0.1,
                                                             "amp_prob": 1.0, "mutation_rate": 0.5},
                                      "selection_params": {"max_cn": m, "max_ploidy": m}})
                       for m in vals([4, 6, 10, 16, 24], [4, 24])], 1600)
    # 7. microenvironment gradient: the O2 diffusion length (o2_diffusion D relative to consumption
    #    k) versus a small tumour. Short diffusion length -> hypoxic core / oxygenated rim; a long
    #    diffusion length (large D) equilibrates O2 across the whole lesion -> no core-rim gradient.
    axes["hypoxia_diffusion"] = (
        [("o2_diffusion", D, {"spatial_params": {"grid_size": 10},
                              "deme_params": {"carrying_capacity": 3},
                              "microenv_params": _hypoxia(D)})
         for D in vals([0.3, 1.0, 3.0, 10.0, 40.0, 120.0], [0.3, 120.0])], 1000)
    return axes


def _hypoxia(D):
    h = copy.deepcopy(HYPOXIA)
    h["hypoxia"]["o2_diffusion"] = D
    h["hypoxia"]["o2_consumption"] = 1.0
    return h


def run(quick=False, seeds=(0, 1, 2)):
    if quick:
        seeds = (0,)
    rows = []
    for axis, (points, n_steps) in _axes(quick).items():
        for label, value, over in points:
            for seed in seeds:
                t = _build(over)
                t.grow(n_steps=n_steps, seed=seed)
                d = diagnose(t)
                row = dict(axis=axis, param=label, value=value, seed=seed, n_steps=n_steps)
                row.update(d.metrics)
                row["degenerate"] = not d.ok
                row["flags"] = "|".join(c.name for c in d.failures)
                rows.append(row)
                print(f"{axis:18} {label}={value:<8} seed={seed}  "
                      f"N={d['n_cancer']:<5} shannon={d['shannon']:.2f} "
                      f"conf={_f(d['clone_confinement'])} tmb_frac={d['tmb_frac']:.3f} "
                      f"{'DEGEN:'+row['flags'] if row['flags'] else 'ok'}")
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {len(df)} rows -> {OUT_CSV}")
    return df


def _f(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="tiny sweep for smoke/CI")
    args = ap.parse_args()
    run(quick=args.quick)
