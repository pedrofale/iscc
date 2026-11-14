"""Golden digests of the paper regime, under tau-leaping and the exact engine.

The paper's tumours are grown by ``validation/paper/regime.py``. This test grows that regime at a tiny
scale, through ``regime.grow`` itself (so through exactly the settings the paper passes), and pins:

- the deme counts at a few points of growth;
- every ``make_cell_data`` frame, after a pickle round trip (the figures read pickled tumours);
- one assay per modality: bulk WGS, scDNA, scRNA and Visium, on the dissociated sample and section the
  figures cut.

Only the field is shrunk (a 24 x 24 grid with two glands) and the acinus opens after 10 generations
instead of 200. Every other parameter is the regime's. Three of the registry's variants are grown:

- ``paper``: the regime itself, with the spatial history on (R0, R1, R3);
- ``open``: the basement-membrane gate and the stromal hazard off (R10's open arm), so cells reach
  the stroma;
- ``exact_passengers``: passenger coarsening off (R2).

A refactor must keep every digest: a change here is a change in what the paper simulates.

Set ``ISCC_GOLDEN_RECORD=1`` to print the digests instead of checking them.
"""
import hashlib
import os
import pickle
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "validation", "paper"))
import regime as RG  # noqa: E402
import realistic_regime as RR  # noqa: E402

from iscc.data import Visium, bulkDNA, scDNA, scRNA  # noqa: E402
from iscc.tumor.components.cell import Cell  # noqa: E402

SEED = 3
TINY = dict(grid_size=24, n_glands=2, min_gland_sep=11,
            origin_confinement=dict(RG.ORIGIN_CONFINEMENT, generations=10))
VARIANTS = {
    "paper": dict(target_cancer=1500, spatial_snapshot_every=2, spatial=TINY),
    "open": dict(target_cancer=1500,
                 spatial=dict(TINY, breach_gated_invasion=False, stromal_hazard=0.0)),
    "exact_passengers": dict(target_cancer=600, coarsen=False, spatial=TINY),
}
CHUNK = {"tau": 10, "exact": 1500}
CHECKPOINTS = (1, 3, 6)                 # chunks after which the deme counts are pinned
MAX_CELLS = 300
RECORD = os.environ.get("ISCC_GOLDEN_RECORD") == "1"

# Recorded at 1761f12 (paper1/integration-v3) with the regime made one source. The exact entries were
# re-pinned when the exact engine took Gillespie's clock: the acinus's confinement clock now counts
# the waiting times, so it opens at a different event (with confinement off the counts are unchanged).
# Five entries were re-pinned again when growth learned to stop at the target size: those variants
# cross their target inside a chunk, so the counts from that point on differ (tau exact_passengers
# reached its target at a chunk end and did not move).
GOLDEN = {
    ('tau', 'paper'): {
        'counts@1': '7d6bb7a3ceebbb1a', 'counts@3': '0dd5f3cd81ca655d',
        'counts@end': 'f07a6baf57729e00', 'births': 'b401ed2fc242085b',
        'spatial_snapshots': '7fe6425bd91e88d3', 'events': 'f2f631f629780d84',
        'stroma_pct': 0.0, 'cell_data.cell_cnv': '5b761225d3037e1d',
        'cell_data.cell_crd': '59994f19fc8eaf5c', 'cell_data.cell_deme': '0c8f872c1bc41feb',
        'cell_data.cell_evo': 'b7189475fa7ec2d0', 'cell_data.cell_exp': 'd31819b22cdf180b',
        'cell_data.cell_exp_m': 'c119a3b0290982eb', 'cell_data.cell_exp_p': '5ed6345611830cc9',
        'cell_data.cell_gland': 'e8c07a2d4bec6693', 'cell_data.cell_microenv': 'c5729cc9ffe03301',
        'cell_data.cell_program': '1d83a98fdeaf8aae', 'cell_data.cell_rna_baf': '2cc87031218fa68f',
        'cell_data.cell_rna_vaf': 'e7745c16a0935cb9', 'cell_data.cell_snv': 'c5c5e75e146a44d5',
        'cell_data.cell_snv_m': '629178a6b565537d', 'cell_data.cell_snv_p': 'ae3a8447a6d59083',
        'cell_data.cell_type': '677eef0cb1159bad', 'cell_data.cell_wgd': '89c07096f2f0d4c8',
        'wgs': 'b3fc21c4184be042', 'scdna': 'b3c483b91e503e0a',
        'scrna': '5a85a5ed02c5ccb9', 'visium': 'c710c4498228351e',
    },
    ('tau', 'open'): {
        'counts@1': '7d6bb7a3ceebbb1a', 'counts@3': '680fcf22feaf2809',
        'counts@end': 'd1ef886788b916f6', 'births': '3f2595f3e7afbca1',
        'spatial_snapshots': '7ab8fa84d365bf39', 'events': 'f2f631f629780d84',
        'stroma_pct': 3.077931, 'cell_data.cell_cnv': '0a0380e40d3f4f90',
        'cell_data.cell_crd': '390c21ecafefa9c0', 'cell_data.cell_deme': '05a3a3ed10407669',
        'cell_data.cell_evo': '8faff21205e9c5cd', 'cell_data.cell_exp': '98e46f3937e977eb',
        'cell_data.cell_exp_m': '660124df4c9c79b0', 'cell_data.cell_exp_p': 'f652f886b920d1c8',
        'cell_data.cell_gland': 'b23f861c3d5d34a6', 'cell_data.cell_microenv': 'f4b762579ccce72e',
        'cell_data.cell_program': '2ee01470259ed9cb', 'cell_data.cell_rna_baf': 'eaadd21c08252598',
        'cell_data.cell_rna_vaf': 'fbdeba5b87468cfd', 'cell_data.cell_snv': '3a6a10a60ebd90f6',
        'cell_data.cell_snv_m': 'e0c6b480891cdcee', 'cell_data.cell_snv_p': 'f8c15336b1383122',
        'cell_data.cell_type': 'bcaa9b0814b2b444', 'cell_data.cell_wgd': '96ee3928baabef5d',
        'wgs': 'ca407412022ab455', 'scdna': 'a5aa132aa211014d',
        'scrna': '4cb776aebdc6dd8c', 'visium': 'be3aa31c1780d41a',
    },
    ('tau', 'exact_passengers'): {
        'counts@1': 'e36baf2f9dc79b43', 'counts@3': '4a98a3daffc83f65',
        'counts@end': '4a98a3daffc83f65', 'births': '717328b7d6ffff31',
        'spatial_snapshots': '7ab8fa84d365bf39', 'events': '43a661cc2b804f1d',
        'stroma_pct': 0.0, 'cell_data.cell_cnv': '719476ce48cee01f',
        'cell_data.cell_crd': '9b7e2f8d793c15f2', 'cell_data.cell_deme': '978ead6e466061eb',
        'cell_data.cell_evo': '7402df4acb039990', 'cell_data.cell_exp': '0c2cadc3342ff039',
        'cell_data.cell_exp_m': 'b3f65040cef5726d', 'cell_data.cell_exp_p': 'c6865e772e7ba94b',
        'cell_data.cell_gland': '3b2b7f51ac81bdac', 'cell_data.cell_microenv': '7f348303f9d414c7',
        'cell_data.cell_program': 'beda8d95e121a3e4', 'cell_data.cell_rna_baf': '9b5418e296e1e935',
        'cell_data.cell_rna_vaf': '36d85a6a3dc18dcc', 'cell_data.cell_snv': '757121f07d826f1b',
        'cell_data.cell_type': 'cb3198eef68c27a4', 'cell_data.cell_wgd': 'b74374cff114956a',
        'wgs': '1e4b6cc3640d6d71', 'scdna': '410bbb2481e872db',
        'scrna': '2adcf893338374e6', 'visium': '2955b190bb132925',
    },
    ('exact', 'paper'): {
        'counts@1': '3d9a4b463d86e0d0', 'counts@3': '97749733009a70ea',
        'counts@end': '97749733009a70ea', 'births': '7aa7965b6e02a914',
        'spatial_snapshots': 'c36fb9168515d43c', 'events': '3a0094637d46ee31',
        'stroma_pct': 0.0, 'cell_data.cell_cnv': '0bfbf10e76f8d39f',
        'cell_data.cell_crd': '04cd02ab292d8039', 'cell_data.cell_deme': '85620f80e487eabd',
        'cell_data.cell_evo': 'b43872a38ab64ae3', 'cell_data.cell_exp': '0d0189a6a3f15ea3',
        'cell_data.cell_exp_m': '90ba3d925f131a8f', 'cell_data.cell_exp_p': '94da9545ab587f79',
        'cell_data.cell_gland': '5e820fb8a777eeeb', 'cell_data.cell_microenv': 'f8d05dabca285a70',
        'cell_data.cell_program': '685d8e30f563703f', 'cell_data.cell_rna_baf': 'b0013bb2f75f7dd6',
        'cell_data.cell_rna_vaf': '54a9d5ae0c4001da', 'cell_data.cell_snv': '2590a368a490a7f7',
        'cell_data.cell_snv_m': 'ac69f7469d217aeb', 'cell_data.cell_snv_p': '1346c83b0af2b14c',
        'cell_data.cell_type': '1dd17b4b5d5edc42', 'cell_data.cell_wgd': '54ca3051b964c5ab',
        'wgs': '8f3bf09949bc6835', 'scdna': 'c3d11c6d6e1ddba5',
        'scrna': 'a2fafb8b9e5db403', 'visium': '5b38887165fdeec9',
    },
    ('exact', 'open'): {
        'counts@1': '9a9469b0df0b7e23', 'counts@3': '1160db200c8f69e0',
        'counts@end': '1160db200c8f69e0', 'births': '54224839341d5d69',
        'spatial_snapshots': '7ab8fa84d365bf39', 'events': '3a0094637d46ee31',
        'stroma_pct': 1.133333, 'cell_data.cell_cnv': '344192c8d57908bb',
        'cell_data.cell_crd': '005debdcd4ee3e60', 'cell_data.cell_deme': 'bb7adc130914079a',
        'cell_data.cell_evo': '92d06545ca2502ce', 'cell_data.cell_exp': '4eed4bb34052c8fc',
        'cell_data.cell_exp_m': 'd5934019803c74e1', 'cell_data.cell_exp_p': 'a98a85f8e7b1ac55',
        'cell_data.cell_gland': '67cba3d42b3e2c56', 'cell_data.cell_microenv': '1aed7856061ef01d',
        'cell_data.cell_program': '5b421835ff78aa23', 'cell_data.cell_rna_baf': '01b5deb955f799ac',
        'cell_data.cell_rna_vaf': 'f2af070a0c00abd1', 'cell_data.cell_snv': 'ff50fdf5528185d3',
        'cell_data.cell_snv_m': 'bc8553eb99249611', 'cell_data.cell_snv_p': '30778039beacb0bf',
        'cell_data.cell_type': '0dcde433aefaa66e', 'cell_data.cell_wgd': 'a94d258276274121',
        'wgs': 'fae549cf97cb08c9', 'scdna': '4f9889be7a24000e',
        'scrna': '3432a171ea0eec16', 'visium': '21289dbf0226a34d',
    },
    ('exact', 'exact_passengers'): {
        'counts@1': 'a02508377c7ec772', 'counts@end': '511bc2ca5fabe185',
        'births': '03f041b0d3b8759b', 'spatial_snapshots': '7ab8fa84d365bf39',
        'events': 'bdf75f4875039487', 'stroma_pct': 0.0,
        'cell_data.cell_cnv': 'cd834b3663bdcea8', 'cell_data.cell_crd': '867e4622b990375a',
        'cell_data.cell_deme': 'ecbc62b57817b784', 'cell_data.cell_evo': 'e7d0c098591a0dcb',
        'cell_data.cell_exp': '6727d5fcbaea885e', 'cell_data.cell_exp_m': '03c5c180d3ccae79',
        'cell_data.cell_exp_p': '23c60bd61b8b0e6d', 'cell_data.cell_gland': '2925cf33d3c8c083',
        'cell_data.cell_microenv': 'b469fb45baf15378', 'cell_data.cell_program': 'caa4574621927527',
        'cell_data.cell_rna_baf': '45f8f2abe97042fa', 'cell_data.cell_rna_vaf': '17c70f9c115726dd',
        'cell_data.cell_snv': '6692c73060c6bf8d', 'cell_data.cell_type': '564a596624ec526b',
        'cell_data.cell_wgd': '9371bd5f5f96107e', 'wgs': '68d647752cb67dbe',
        'scdna': 'af848852d30983de', 'scrna': '38bbda87cf58bb6a',
        'visium': '88152c8fedd9c18e',
    },
}


def _digest(obj):
    h = hashlib.sha256()

    def feed(o):
        if isinstance(o, pd.DataFrame):
            feed(("frame", [str(c) for c in o.columns], [str(i) for i in o.index]))
            for c in range(o.shape[1]):
                feed(o.iloc[:, c].to_numpy())
        elif isinstance(o, pd.Series):
            feed(("series", [str(i) for i in o.index], o.to_numpy()))
        elif isinstance(o, np.ndarray):
            if o.dtype == object:
                feed(o.tolist())
            else:
                h.update(f"{o.dtype}{o.shape}".encode())
                h.update(np.ascontiguousarray(o).tobytes())
        elif isinstance(o, dict):
            h.update(b"{")
            for k in sorted(o, key=str):
                feed(str(k))
                feed(o[k])
            h.update(b"}")
        elif isinstance(o, (list, tuple)):
            h.update(b"[")
            for x in o:
                feed(x)
            h.update(b"]")
        elif isinstance(o, (np.generic, float, int, bool, str)) or o is None:
            h.update(repr(o.item() if isinstance(o, np.generic) else o).encode())
        else:
            raise TypeError(f"no digest for {type(o).__name__}")

    feed(obj)
    return h.hexdigest()[:16]


def _counts(t):
    """Every deme's genotype counts, keyed by creation ordinal."""
    return [t.step, [(i, sorted((int(t.genotypes[g].ord), int(c)) for g, c in d.items()))
                     for i, d in enumerate(t.demes) if d]]


def _run(engine, variant):
    Cell._genotype_counter = 0          # genotype ids are process-global; make them run-local
    out, n = {}, [0]

    def on_chunk(t):
        n[0] += 1
        if n[0] in CHECKPOINTS:
            out[f"counts@{n[0]}"] = _digest(_counts(t))

    t = RG.grow(SEED, "small", chunk=CHUNK[engine], max_steps=10 ** 9, on_chunk=on_chunk,
                update_mode=engine, **VARIANTS[variant])
    assert t.reached_target
    out["counts@end"] = _digest(_counts(t))
    out["births"] = _digest(t.genotype_births())
    out["spatial_snapshots"] = _digest(t.spatial_snapshots())
    out["events"] = _digest(t.events)
    out["stroma_pct"] = round(RR.stroma_cancer_pct(t), 6)
    t = pickle.loads(pickle.dumps(t, protocol=pickle.HIGHEST_PROTOCOL))
    t.programs.max_log_fold = RG.PROGRAM_MAX_LOG_FOLD          # as registry.load applies it

    cd = t.make_cell_data(max_cells=MAX_CELLS)
    for k in sorted(cd):
        out[f"cell_data.{k}"] = _digest(cd[k])

    spec = RR.resect(t)
    diss = spec.dissociate(max_cells=MAX_CELLS)
    cells = list(diss["cell_exp"].index)
    gid = diss["cell_type"]["cell_id"].astype(str).values
    cancer = [c for c, g in zip(cells, gid) if g in t.genotypes and t._is_cancer(g)]
    wgs = bulkDNA(breadth="wgs", seed=11).run(diss)
    out["wgs"] = _digest(wgs.observed_data)
    sc = scDNA(n_cells=min(40, len(cancer)), breadth="wgs", kappa=500.0, mu_depth=60.0,
               seed=21).run(diss, cell_subset=cancer[:40])
    out["scdna"] = _digest([sc.coverage, list(sc.cells)])
    rna = scRNA(n_cells=len(cells), seed=5).run(diss, cell_subset=cells)
    out["scrna"] = _digest(rna.observed_counts)
    sec = spec.slice(depth_frac=0.5, max_cells=MAX_CELLS, layers="visium")
    vis = Visium(seed=300, **RG.VISIUM).run(sec, grid_side=t.grid_size)
    out["visium"] = _digest([vis.spot_counts, np.asarray(vis.spot_coords)])
    return out


@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("engine", ["tau", "exact"])
def test_paper_regime_is_bit_identical(engine, variant):
    got = _run(engine, variant)
    if RECORD:
        print(f"\nGOLDEN[{(engine, variant)!r}] = {got!r}")
        return
    want = GOLDEN.get((engine, variant))
    assert want, "no golden digests recorded"
    assert got.keys() == want.keys()
    diff = {k: (got[k], want[k]) for k in want if got[k] != want[k]}
    assert not diff, diff
