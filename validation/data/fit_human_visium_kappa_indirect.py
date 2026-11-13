"""Visium kappa by indirect inference: which input kappa makes ``estimate_visium`` on a simulated section
match ``estimate_visium`` on the real one (82,165 over the layer's genes, ``fit_human_visium_kappa.py``)?

The direct fit reads all of a real section's dispersion around its mean composition as Dirichlet-
multinomial noise, but part of it is tissue heterogeneity, which iscc simulates itself; adding the
direct value on top counts that biology twice. Simulating a paper-regime section (needs
``validation/paper/regime.py``, on the paper branches) at several input kappas shows which reproduce
the real estimate. Result on 2026-11-13 (regime-v2, small scale, seed 1, 657 in-tissue spots):
input 82,165 -> 55,416; 300,000 -> 81,464; 1e9 -> 81,592. The simulated tissue alone reproduces the
real dispersion, so the data bound the technical kappa only from below; 300,000 is the smallest tested
value within 1% of the real estimate, and is ``HUMAN_VISIUM_KAPPA``.
"""
import sys
import time

import numpy as np

sys.path.insert(0, "validation")
sys.path.insert(0, "validation/paper")
import realistic_regime as RR  # noqa: E402
import regime as R  # noqa: E402
from iscc.data import Visium, estimate_visium  # noqa: E402
from iscc.data.human import HumanGenomeLayer, arm_genome  # noqa: E402

t0 = time.time()
cfg = arm_genome(total_genes=6000, driver_scale=0.4)
ep = R.expression()
ep["program_params"]["max_log_fold"] = 5.0
tum = R.grow(seed=1, scale="small", target_cancer=R.T["sweep_target"], genome=cfg["genome_params"],
             selection=cfg["selection_params"], expression_params=ep)
layer = HumanGenomeLayer(tum)
expr = layer.expression(tum)
sec = RR.section(tum, depth_frac=0.5, max_cells=60_000)
print(f"section {len(sec['cell_exp'])} cells, {time.time() - t0:.0f} s", flush=True)
for kappa in (82_165.0, 300_000.0, 1e9):
    t = time.time()
    vz = Visium(seed=2, kappa=kappa).run(sec, gene_lift=expr)
    tissue = np.asarray(vz.obs.n_cells) > 0
    est = estimate_visium(vz.spot_counts[tissue].to_numpy(), count_model="dm", coords=vz.spot_coords[tissue])
    print(f"input kappa {kappa:>13,.0f} -> estimate_visium {est.hypers.kappa:>9,.0f}  "
          f"({tissue.sum()} spots, {time.time() - t:.0f} s)", flush=True)
