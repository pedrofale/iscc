"""iscc end to end at a given tumour size, for the runtime comparison.

    python iscc_runtime.py <n_cancer> <seed> <scale> [--layer]

Grows one paper-regime tumour (``validation/paper/regime.py``) to ``n_cancer`` cancer cells with the
engine options the paper switches on (demon's division law, independent random streams, the
confinement clock in generations), cuts a thin section through it, materialises at most twice
``n_cancer`` of its cells (OncoSpot's cell count at the same malignant count, its tumour fraction
being 0.5), generates Visium spots, and prints the time of each stage and the size reached.
``scale`` is the regime's tissue field (small, mid, cm), which sets how many normal cells surround
the tumour. Run it with the iscc engine (PR #12) on PYTHONPATH and
``ISCC_PAPER_DIR`` pointing at a checkout whose ``validation/paper`` holds the regime.

``--layer`` will add the human genome layer (genes, SNPs and the tools' files) once it is merged;
until then it stops with an error rather than timing something else.
"""
import argparse
import copy
import functools
import json
import os
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n_cancer", type=int)
    ap.add_argument("seed", type=int)
    ap.add_argument("scale", choices=["small", "mid", "cm"])
    ap.add_argument("--layer", action="store_true")
    ap.add_argument("--low-memory", action="store_true")
    ap.add_argument("--layers", default=None,
                    help="make_cell_data(layers=...) for the section, e.g. visium (needs the chunked engine)")
    a = ap.parse_args()
    if a.layer:
        raise NotImplementedError("the human genome layer is not merged yet")
    paper = os.path.join(os.environ["ISCC_PAPER_DIR"], "validation", "paper")
    sys.path.insert(0, paper)
    import regime as RG
    import realistic_regime as RR
    from iscc.data import Visium
    from iscc.tumor.models import GenotypeTumor

    lm = {"low_memory": True} if a.low_memory else {}
    RG.GenotypeTumor = functools.partial(GenotypeTumor, rng_streams="independent", **lm)
    origin = dict(copy.deepcopy(RG.ORIGIN_CONFINEMENT), clock="generations", generations=200)
    t0 = time.time()
    t = RG.grow(a.seed, a.scale, a.n_cancer, deme={"dispersal_law": "noble"},
                spatial={"origin_confinement": origin}, max_steps=100_000, chunk=2)
    t1 = time.time()
    kw = {} if a.layers is None else {"layers": a.layers}
    sec = RR.resect(t).slice(depth_frac=0.5, max_cells=2 * a.n_cancer, **kw)
    t2 = time.time()
    vis = Visium(seed=300 + a.seed).run(sec, grid_side=t.grid_size)
    t3 = time.time()
    total = int(sum(sum(d.values()) for d in t.demes))
    rec = dict(tool="iscc", layers=a.layers, low_memory=a.low_memory, scale=a.scale, seed=a.seed, n_cancer=int(t.get_cancer_size()), n_total=total,
               reached=bool(t.reached_target), section_cells=int(len(sec["cell_type"])),
               spots=int(vis.spot_counts.shape[0]), genes=int(vis.spot_counts.shape[1]),
               grow_s=t1 - t0, section_s=t2 - t1, visium_s=t3 - t2, total_s=t3 - t0)
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
