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

# Recorded at 1761f12 (paper1/integration-v3) with the regime made one source.
GOLDEN = {
    ('tau', 'paper'): {
        'counts@1': '7d6bb7a3ceebbb1a', 'counts@3': '0dd5f3cd81ca655d',
        'counts@end': '935052cec449040a', 'births': 'b773fc6fde29b85b',
        'spatial_snapshots': '833f7debe6de79cf', 'events': 'f2f631f629780d84',
        'stroma_pct': 0.0, 'cell_data.cell_cnv': 'fa37be7fc0542ea3',
        'cell_data.cell_crd': '90236d823e99da06', 'cell_data.cell_deme': 'ee7c79710b0c70ef',
        'cell_data.cell_evo': '778b4a5f1b4730c8', 'cell_data.cell_exp': 'da3aa5c60c910c22',
        'cell_data.cell_exp_m': '7a1eca4478fabe51', 'cell_data.cell_exp_p': '79d82c8892889ec5',
        'cell_data.cell_gland': '202a0a59b663a66c', 'cell_data.cell_microenv': '47c008056ae7ec47',
        'cell_data.cell_program': '53ac28908d34a56b', 'cell_data.cell_rna_baf': 'e72b98450bfb7769',
        'cell_data.cell_rna_vaf': '92ba0422f1b91b2b', 'cell_data.cell_snv': 'c812e9ddd3543925',
        'cell_data.cell_snv_m': '3ef1531772afd1a0', 'cell_data.cell_snv_p': '3bff75a430242159',
        'cell_data.cell_type': '51b79eefa198655f', 'cell_data.cell_wgd': '54ca3051b964c5ab',
        'wgs': '7405f53efd6313d3', 'scdna': '8e52e5d3e86061be',
        'scrna': '164725e8c11d5741', 'visium': '3b48f0e87ec9c36a',
    },
    ('tau', 'open'): {
        'counts@1': '7d6bb7a3ceebbb1a', 'counts@3': '680fcf22feaf2809',
        'counts@end': 'f31ac8af3a91ca8c', 'births': 'cf30cd31542327f4',
        'spatial_snapshots': '7ab8fa84d365bf39', 'events': 'f2f631f629780d84',
        'stroma_pct': 8.654116, 'cell_data.cell_cnv': '5f29ec2c78a54028',
        'cell_data.cell_crd': '7cc6e2695ad93bc0', 'cell_data.cell_deme': 'f298ef6bf98888e7',
        'cell_data.cell_evo': '219866297dd111fb', 'cell_data.cell_exp': '251e4925cdd86818',
        'cell_data.cell_exp_m': '56dcef9218490152', 'cell_data.cell_exp_p': '527723762adfb05c',
        'cell_data.cell_gland': '8ed62ed5129931ce', 'cell_data.cell_microenv': 'aacae5d46141f040',
        'cell_data.cell_program': '0d9c23423d106bec', 'cell_data.cell_rna_baf': '52091ce5786545a2',
        'cell_data.cell_rna_vaf': 'ab76bad396698346', 'cell_data.cell_snv': '47434965c9afd609',
        'cell_data.cell_snv_m': '0482d7fff47ee650', 'cell_data.cell_snv_p': '9968c056404fe246',
        'cell_data.cell_type': '6d39b447abaaa096', 'cell_data.cell_wgd': '3fb431b80f90bcb0',
        'wgs': 'f750f2aaf8956f54', 'scdna': '540833177700e345',
        'scrna': '0b641709632a7b9d', 'visium': '704b86388b3e10cb',
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
        'counts@1': '8abe10b509074f5d', 'counts@3': '14e33bfd10a22783',
        'counts@end': '14e33bfd10a22783', 'births': 'b839bb4b15bc95ac',
        'spatial_snapshots': 'f523a8c79118f113', 'events': 'a884ab2b9cece4d9',
        'stroma_pct': 0.0, 'cell_data.cell_cnv': '230347f6f6d7baf8',
        'cell_data.cell_crd': 'bf6a49ce3bc8c8ed', 'cell_data.cell_deme': '6054359f3a829a8c',
        'cell_data.cell_evo': '79f99001326e9f86', 'cell_data.cell_exp': 'a8ae949ba5833660',
        'cell_data.cell_exp_m': '75ceb6876eb2cd02', 'cell_data.cell_exp_p': 'c3c1150552a0333f',
        'cell_data.cell_gland': 'd1fd563da00cdb4f', 'cell_data.cell_microenv': '8f285e46f9d40773',
        'cell_data.cell_program': 'f9013a8aeceba741', 'cell_data.cell_rna_baf': 'f8132f55e2495021',
        'cell_data.cell_rna_vaf': 'dce552ae7701b590', 'cell_data.cell_snv': '412788ffadb98527',
        'cell_data.cell_snv_m': '6ed82524a6444730', 'cell_data.cell_snv_p': '7e84b5c869badfb9',
        'cell_data.cell_type': 'f95ca5306a90e133', 'cell_data.cell_wgd': 'b74374cff114956a',
        'wgs': '063d140bf049a31b', 'scdna': 'a5815c01fd8fa6b4',
        'scrna': '68216ec8fc4a04ef', 'visium': '188a7e3f21d0ef28',
    },
    ('exact', 'open'): {
        'counts@1': '6b3ff17fdd45d733', 'counts@3': '384d26273268144c',
        'counts@end': '384d26273268144c', 'births': '191b4c6cb9273305',
        'spatial_snapshots': '7ab8fa84d365bf39', 'events': 'a884ab2b9cece4d9',
        'stroma_pct': 1.409214, 'cell_data.cell_cnv': 'ca4636e13b047742',
        'cell_data.cell_crd': '96b0d715f9394c80', 'cell_data.cell_deme': '5d5516da35d94bee',
        'cell_data.cell_evo': 'd171cc5f9c5f3b29', 'cell_data.cell_exp': '0c34e0a6619ce8ba',
        'cell_data.cell_exp_m': '561bcee2029406a0', 'cell_data.cell_exp_p': '878efc8528da207b',
        'cell_data.cell_gland': '7ca5fb47579464fe', 'cell_data.cell_microenv': '6a7d632f33a2530f',
        'cell_data.cell_program': 'd88d37181c19b6e8', 'cell_data.cell_rna_baf': 'a55e33624d22db1b',
        'cell_data.cell_rna_vaf': '1253be8fa9f3d9a8', 'cell_data.cell_snv': '438fff02ba0d94b6',
        'cell_data.cell_snv_m': '525bce9b352df01f', 'cell_data.cell_snv_p': '04f15b9a017b9b63',
        'cell_data.cell_type': 'fbc2e190758c51b1', 'cell_data.cell_wgd': 'baa7fe5f98c6325b',
        'wgs': '2830c57960ac190c', 'scdna': 'f9646ee8a4b37d81',
        'scrna': '744df9b5150f4959', 'visium': 'fb04b79732f11f8e',
    },
    ('exact', 'exact_passengers'): {
        'counts@1': '10c94bf85fb7c3ad', 'counts@end': 'b52846dd9aed6122',
        'births': 'd58b148307e0387b', 'spatial_snapshots': '7ab8fa84d365bf39',
        'events': 'a884ab2b9cece4d9', 'stroma_pct': 0.0,
        'cell_data.cell_cnv': '53c43ddec51d8486', 'cell_data.cell_crd': '15774000a9a75dec',
        'cell_data.cell_deme': 'ba9d1bbb86bb524f', 'cell_data.cell_evo': '310f5f78eeb3b3e7',
        'cell_data.cell_exp': '3494a5380e5b15c6', 'cell_data.cell_exp_m': '6610abd4899ced9a',
        'cell_data.cell_exp_p': '9a5979d1fd764dda', 'cell_data.cell_gland': 'fa63b054bddad07e',
        'cell_data.cell_microenv': '298802f7aee5de7a', 'cell_data.cell_program': '05de08956d117169',
        'cell_data.cell_rna_baf': '0a0e55939a674bad', 'cell_data.cell_rna_vaf': '6e0984323fe406e3',
        'cell_data.cell_snv': '5c60ae3f83f500c5', 'cell_data.cell_type': 'b5101e97d4315111',
        'cell_data.cell_wgd': 'ef32027d8837eb8b', 'wgs': 'f8ccbabaaca8820f',
        'scdna': '1873b942da239d39', 'scrna': '4fe382d3467078fa',
        'visium': '754159f5ac0a8e55',
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
