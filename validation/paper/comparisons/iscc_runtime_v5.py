"""iscc end to end at a given tumour size, on paper regime v5, for the runtime comparison (S4 Fig).

    python iscc_runtime_v5.py <n_cancer> <seed> <scale>

As ``iscc_runtime_v4.py``, on the regime of the v5 registry: the tumour is grown through
``regime.grow`` (stopping at the first leap that reaches n_cancer), a section is cut and Visium spots
are generated on the growth genes, the same work as the v4 timing. Run from a copy of iscc at the
commit of that regime (cwd, with a ``COMMIT`` file), with src on PYTHONPATH.

The timings in ``results/runtime.csv`` (variant ``v5``) are a Slurm array on Arrhenius's ``cpu``
partition (the AMD EPYC 9755 nodes every other tool was timed on), one core per run, sizes 2,500 /
7,500 / 25,000 / 75,000 and seeds 1-3 (scale small / small / mid / cm), each run as::

    echo "TOOL=iscc VARIANT=v5 N_CANCER=$n SEED=$seed NODE=$(hostname) CPU=..."
    ISCC_PAPER_TIER=full PYTHONPATH=src /usr/bin/time -v python iscc_runtime_v5.py $n $seed $scale

and collected with ``runtime_collect.py ... --merge`` into ``results/runtime.csv``.
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
print(json.dumps(dict(tool="iscc", variant="v5", commit=open("COMMIT").read().strip(), scale=scale, seed=seed,
                      n_cancer=int(t.get_cancer_size()), reached=bool(t.reached_target),
                      n_total=int(sum(sum(d.values()) for d in t.demes)), section_cells=int(len(sec["cell_type"])),
                      spots=int(vis.spot_counts.shape[0]), grow_s=t1 - t0, section_s=t2 - t1, visium_s=t3 - t2,
                      total_s=t3 - t0)))
