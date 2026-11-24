"""Maps of the structures only iscc produces, from a regrown R12 tumour (for ``supp_structures.py``).

R12 is the paper regime with the metastatic lattice (registry: 60,000 cancer cells at the sweep scale,
regime v3). Of the given seeds this keeps the one with the largest metastatic deposit and writes, as
8-bit RGB, the primary's tissue map as the paper's figures draw it (``common.tissue_map``: each deme's
dominant clone over lumen, gland wall and stroma, by cancer fraction, the gland walls outlined) and the
deposit's map (each deme's dominant clone over the host parenchyma, by cancer fraction), with the
counts the caption needs, into one small ``.npz``.

    ISCC_PAPER_OUT=<paper-v3> ISCC_PAPER_TIER=full python iscc_structures.py <out.npz> <seed> [<seed> ...]

Needs a node: each tumour pickle is about 400 MB.
"""
import json
import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import common as C  # noqa: E402
import registry as REG  # noqa: E402
import style as S  # noqa: E402


def _met_cancer(t):
    return int(sum(c for d in t.demes[t.n_primary_demes:] for g, c in d.items() if t._is_cancer(g)))


def met_map(t, clones, min_alpha=0.25):
    """(Gm, Gm, 3) RGB of the deposit: host parenchyma, each deme's dominant clone by cancer fraction."""
    G, base = t.met_grid_size, t.n_primary_demes
    rgb = np.tile(np.array(C._hex(S.COMPARTMENT["met"])), (G, G, 1)).astype(float)
    for i in range(G * G):
        deme = t.demes[base + i]
        cancer, total = Counter(), 0
        for g, c in deme.items():
            total += c
            if t._is_cancer(g):
                cancer[clones.of(g)] += c
        n = sum(cancer.values())
        if n:
            a = min_alpha + (1 - min_alpha) * n / max(total, 1)
            r, c = t.deme_coords[base + i]
            rgb[r, c] = (1 - a) * rgb[r, c] + a * np.array(C._hex(clones.colour[cancer.most_common(1)[0][0]]))
    return rgb


def main():
    out, seeds = sys.argv[1], [int(s) for s in sys.argv[2:]]
    best = None
    for s in seeds:
        t = REG.load("R12", f"seed{s}")
        m = _met_cancer(t)
        print(f"R12 seed{s}: met cancer cells {m}", flush=True)
        if best is None or m > best[1]:
            best = (s, m)
        del t
    s = best[0]
    t = REG.load("R12", f"seed{s}")
    cl = C.Clones(t)
    G = t.grid_size
    gland = np.asarray(t.gland_id)[:G * G]
    prim = [{g: c for g, c in t.demes[i].items() if t._is_cancer(g)} for i in range(G * G)]
    stroma = sum(sum(d.values()) for i, d in enumerate(prim) if gland[i] == -1)
    meta = dict(run="R12", task=f"seed{s}", seeds_scanned=seeds, primary_cancer=int(sum(sum(d.values()) for d in prim)),
                stroma_cancer=int(stroma), met_cancer=_met_cancer(t), n_glands=int(t.n_glands),
                grid=int(G), met_grid=int(t.met_grid_size), clones=len(cl.labels), time=float(t.time))
    np.savez_compressed(out, primary=(C.tissue_map(t, cl) * 255).round().astype(np.uint8),
                        met=(met_map(t, cl) * 255).round().astype(np.uint8), meta=json.dumps(meta))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
