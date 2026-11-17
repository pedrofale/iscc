"""iscc end to end at a given tumour size, on paper regime v3, for the runtime comparison.

    python iscc_runtime_v3.py <n_cancer> <seed> <scale>

As iscc_runtime.py with --layers visium --low-memory (the "visium-low" variant), but the tumour is
grown through regime.grow, the regime's one source since the compaction (vault:
decision-paper-regime-v3: binomial tau deaths, Cao-Gillespie-Petzold steps at eps 0.01, driver effect
1.03, wgd_rate 0.01), stopping at the first leap that reaches n_cancer. Run from a copy of iscc at
paper1/standard-algorithms (cwd), with src on PYTHONPATH.

The timings in ``results/runtime.csv`` (variant ``v3``) are Slurm array 5323338 on Arrhenius, one core
per run, sizes 2,500 / 7,500 / 25,000 / 75,000 and seeds 1-3 (scale small / small / mid / cm), at
iscc commit faaf5d1, each run as::

    echo "TOOL=iscc VARIANT=v3 N_CANCER=$n SEED=$seed NODE=$(hostname) CPU=..."
    ISCC_PAPER_TIER=full PYTHONPATH=src /usr/bin/time -v python iscc_runtime_v3.py $n $seed $scale

and collected with ``runtime_collect.py results/runtime.csv <log dir> --merge``.
"""
import json, os, sys, time
sys.path.insert(0, os.path.join(os.getcwd(), "validation", "paper"))
sys.path.insert(0, os.path.join(os.getcwd(), "validation"))
import regime as RG
import realistic_regime as RR
from iscc.data import Visium

n, seed, scale = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
t0 = time.time()
t = RG.grow(seed, scale, n, max_steps=100_000, chunk=2)
t1 = time.time()
sec = RR.resect(t).slice(depth_frac=0.5, max_cells=2 * n, layers="visium")
t2 = time.time()
vis = Visium(seed=300 + seed).run(sec, grid_side=t.grid_size)
t3 = time.time()
print(json.dumps(dict(tool="iscc", variant="v3", commit=open("COMMIT").read().strip(), scale=scale, seed=seed,
                      n_cancer=int(t.get_cancer_size()), reached=bool(t.reached_target),
                      n_total=int(sum(sum(d.values()) for d in t.demes)), section_cells=int(len(sec["cell_type"])),
                      spots=int(vis.spot_counts.shape[0]), grow_s=t1 - t0, section_s=t2 - t1, visium_s=t3 - t2,
                      total_s=t3 - t0)))
