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

Result on 2026-11-22, after the Visium depth change and with the layer emitting all 36,601 genes
(634 in-tissue spots), against the real 89,197 over all genes: input 89,197 -> 53,597; 300,000 ->
80,480; 1e9 -> 97,673. Over the arm genes alone the same sections give 74,276 and 90,126 at 300,000
and 1e9, against the real 82,165. The estimate no longer saturates, so the value is to be refitted on a
finer grid once the per-gene Visium platform factor is in.

With the platform factor (2026-11-22, inputs as arguments): 300,000 -> 71,934; 600,000 -> 81,254;
1e6 -> 90,198; 2e6 -> 88,508; 5e6 -> 89,653; 1e9 -> 88,257, against the real 89,197. A section of an
immune-variant R14 tumour gives 63,693 at 300,000 up to 74,768 at 1e9, below the real at any input.
The programme fold changes are calibrated next and change the tissue's own dispersion, so the value
is refitted after them.

Refitted on 2026-11-23, with the programme factor on cancer cells only, on the regrown immune R14 tumour
(disp0.9_seed1, 2,608 in-tissue spots, three Visium seeds per input, mean and sd): 300,000 -> 77,014
(1,201); 600,000 -> 85,698 (1,935); 1e6 -> 87,996 (1,935); 2e6 -> 90,274 (1,446); 5e6 -> 93,476
(3,140); 1e9 -> 93,523 (3,192), against the real 89,197. The real estimate is crossed at about 1.4e6
(log-linear between 1e6 and 2e6), which is ``HUMAN_VISIUM_KAPPA``; 1.4e6 itself was not run.

Usage::

    python validation/data/fit_human_visium_kappa_indirect.py 300000 1e6 1e9 [--tumour <.iscc>] [--seeds 3]
"""
import argparse
import sys
import time

import numpy as np

sys.path.insert(0, "validation")
sys.path.insert(0, "validation/paper")
import realistic_regime as RR  # noqa: E402
import regime as R  # noqa: E402
from iscc.data import Visium, estimate_visium  # noqa: E402
from iscc.data.human import HumanGenomeLayer, arm_genome  # noqa: E402
from iscc.tumor.models import GenotypeTumor  # noqa: E402

ap = argparse.ArgumentParser(description="Visium kappa by indirect inference")
ap.add_argument("kappas", nargs="*", type=float, default=[89_197, 300_000, 1e9])
ap.add_argument("--tumour", help="a stored .iscc tumour to section (default: grow the small immune-free one)")
ap.add_argument("--seeds", type=int, default=1, help="Visium seeds per input kappa (2, 3, ...)")
a = ap.parse_args()
t0 = time.time()
if a.tumour:
    tum = GenotypeTumor.load(a.tumour)
else:
    cfg = arm_genome(total_genes=6000, driver_scale=0.4)
    ep = R.expression()
    ep["program_params"]["max_log_fold"] = 5.0
    tum = R.grow(seed=1, scale="small", target_cancer=R.T["sweep_target"], genome=cfg["genome_params"],
                 selection=cfg["selection_params"], expression_params=ep)
layer = HumanGenomeLayer(tum)
expr = layer.expression(tum)
sec = RR.section(tum, depth_frac=0.5, max_cells=60_000)
print(f"section {len(sec['cell_exp'])} cells, {time.time() - t0:.0f} s", flush=True)
for kappa in a.kappas:
    ests = []
    for vs in range(2, 2 + a.seeds):
        t = time.time()
        vz = Visium(seed=vs, kappa=kappa).run(sec, gene_lift=expr)
        tissue = np.asarray(vz.obs.n_cells) > 0
        est = estimate_visium(vz.spot_counts[tissue].to_numpy(), count_model="dm", coords=vz.spot_coords[tissue])
        ests.append(est.hypers.kappa)
        print(f"input kappa {kappa:>13,.0f} seed {vs} -> estimate_visium {est.hypers.kappa:>9,.0f}  "
              f"({tissue.sum()} spots, {time.time() - t:.0f} s)", flush=True)
        del vz
    if a.seeds > 1:
        print(f"input kappa {kappa:>13,.0f} -> mean {np.mean(ests):>9,.0f}, sd {np.std(ests, ddof=1):,.0f} "
              f"over {a.seeds} seeds", flush=True)
