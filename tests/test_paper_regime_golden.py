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
# Re-pinned for paper regime v3 (vault: decision-paper-regime-v3): binomial tau deaths with the
# Cao-Gillespie-Petzold step at epsilon 0.01, driver effect 1.03, wgd_rate 0.01, and the program
# log-fold bound applied in the chunked materialisation. The bound alone moves only the expression
# outputs (cell_exp, cell_exp_m, cell_exp_p, cell_rna_baf, scRNA, Visium); the regime moves the rest.
GOLDEN = {('tau', 'paper'): {'counts@1': '1282a482de6262c3',
                    'counts@3': 'a2915b6abce15bac',
                    'counts@end': 'b3abaeb96805ee46',
                    'births': 'bbf80213f097fe65',
                    'spatial_snapshots': '869747b6871c8978',
                    'events': '43a661cc2b804f1d',
                    'stroma_pct': 0.0,
                    'cell_data.cell_cnv': '66816cd5fde5a2ab',
                    'cell_data.cell_crd': 'cca7e38b8fc401f2',
                    'cell_data.cell_deme': '985ac2ff95bb46d2',
                    'cell_data.cell_evo': '0c407b42f49cf0e7',
                    'cell_data.cell_exp': 'baf5b7924e5ed46c',
                    'cell_data.cell_exp_m': 'f41f0da2fcb1ab0e',
                    'cell_data.cell_exp_p': 'b6e4031f46956942',
                    'cell_data.cell_gland': 'acf3de0ed66b3356',
                    'cell_data.cell_microenv': 'ddfcca88c230f494',
                    'cell_data.cell_program': 'afcd350a1b55aae3',
                    'cell_data.cell_rna_baf': '506a63ac3c8d08cb',
                    'cell_data.cell_rna_vaf': '9fa0b17298fe79ec',
                    'cell_data.cell_snv': '714ff7482f9f7ea6',
                    'cell_data.cell_snv_m': '7467ad96c6c96108',
                    'cell_data.cell_snv_p': 'e04d3e0fe02ba042',
                    'cell_data.cell_type': 'bf1a9548b50bb3c4',
                    'cell_data.cell_wgd': 'eee00f2d813c49a7',
                    'wgs': '28c03718350174e8',
                    'scdna': '0567a19d9b46c6fd',
                    'scrna': '6dedeaecf8ca7212',
                    'visium': '0806867b43ac4df9'},
 ('tau', 'open'): {'counts@1': '1282a482de6262c3',
                   'counts@3': '3a729f2474df9eb0',
                   'counts@end': 'e54982411ab12df0',
                   'births': 'b77e52ebb6d9c410',
                   'spatial_snapshots': '7ab8fa84d365bf39',
                   'events': '43a661cc2b804f1d',
                   'stroma_pct': 0.732845,
                   'cell_data.cell_cnv': '4006a5f99d2c5d36',
                   'cell_data.cell_crd': '32132db9640d401d',
                   'cell_data.cell_deme': 'f7f4684f5957f54c',
                   'cell_data.cell_evo': 'e0d95688e2435603',
                   'cell_data.cell_exp': 'b5c63779b8789755',
                   'cell_data.cell_exp_m': '776c280d4432e11d',
                   'cell_data.cell_exp_p': 'bd1c3d929223f99e',
                   'cell_data.cell_gland': 'f36e0e2558a30dc3',
                   'cell_data.cell_microenv': '22e0cc41e8d4a00d',
                   'cell_data.cell_program': 'f3fba75bf8e846ec',
                   'cell_data.cell_rna_baf': '05543e59a68a8b83',
                   'cell_data.cell_rna_vaf': 'deb9d8fd5fe9c66a',
                   'cell_data.cell_snv': 'aa552f6d3f93b3a1',
                   'cell_data.cell_snv_m': 'dd5c4c907c3c9ac3',
                   'cell_data.cell_snv_p': '562a7ed7f6f63fe1',
                   'cell_data.cell_type': '6854970cd87db83a',
                   'cell_data.cell_wgd': '0fdfd91026f17c85',
                   'wgs': '2a5297bba939d498',
                   'scdna': 'bd4434c91f92af0a',
                   'scrna': '44833cdc9f599de5',
                   'visium': '1fc9fe55e690d3b3'},
 ('tau', 'exact_passengers'): {'counts@1': '5ac672396e0e5a7e',
                               'counts@3': 'd91099b939dae93a',
                               'counts@end': 'd91099b939dae93a',
                               'births': '0fa07fe70cd70333',
                               'spatial_snapshots': '7ab8fa84d365bf39',
                               'events': 'f2f631f629780d84',
                               'stroma_pct': 0.0,
                               'cell_data.cell_cnv': 'f680afcfd60e32a5',
                               'cell_data.cell_crd': 'ed89278856910737',
                               'cell_data.cell_deme': 'd9778b31c4934cb8',
                               'cell_data.cell_evo': 'aeeb0f5d12054d04',
                               'cell_data.cell_exp': '5ea96176a4b25a8b',
                               'cell_data.cell_exp_m': '9b25d38c71c059a6',
                               'cell_data.cell_exp_p': '7b46e0515b38d58b',
                               'cell_data.cell_gland': 'dc531ef7b0701e95',
                               'cell_data.cell_microenv': '12b75d7c9efcc81a',
                               'cell_data.cell_program': '7b29044328e8dc97',
                               'cell_data.cell_rna_baf': '63f9bad28c4ccc9a',
                               'cell_data.cell_rna_vaf': '6d442646cbb19211',
                               'cell_data.cell_snv': '79a31373d9a3140c',
                               'cell_data.cell_type': 'e55b52311380f53d',
                               'cell_data.cell_wgd': '1d2ccc5a88d04b9f',
                               'wgs': '544539387b74e2df',
                               'scdna': '0c7bc3d8102356be',
                               'scrna': '76fb7104d5841718',
                               'visium': '2ef18fc1c2517409'},
 ('exact', 'paper'): {'counts@1': 'a734ca650357cc43',
                      'counts@3': 'cfad3b411a29ef46',
                      'counts@end': 'cfad3b411a29ef46',
                      'births': '7394a71ef4fa8667',
                      'spatial_snapshots': '2a80f27af61e91d7',
                      'events': 'ec30c1e9135054c9',
                      'stroma_pct': 0.0,
                      'cell_data.cell_cnv': 'cc1bde7abe5f826e',
                      'cell_data.cell_crd': '34a551f3b0fb9271',
                      'cell_data.cell_deme': 'aef462c04172e69e',
                      'cell_data.cell_evo': 'b835948685e39165',
                      'cell_data.cell_exp': 'cd7debb718840a01',
                      'cell_data.cell_exp_m': '012f761629ba1a24',
                      'cell_data.cell_exp_p': '3fb22f3848e09b5c',
                      'cell_data.cell_gland': '4eb18ca9113c5131',
                      'cell_data.cell_microenv': '3678ea7ffe22aad1',
                      'cell_data.cell_program': '2163f34754620e0b',
                      'cell_data.cell_rna_baf': 'd7beb9760ea3d9b0',
                      'cell_data.cell_rna_vaf': '4ff8336609e6585c',
                      'cell_data.cell_snv': '8c002d6e72578871',
                      'cell_data.cell_snv_m': 'a96b8ae0342f1773',
                      'cell_data.cell_snv_p': 'a88e0b6fcde83f18',
                      'cell_data.cell_type': '01af29403356dbe2',
                      'cell_data.cell_wgd': '65c8cc85d24d755c',
                      'wgs': '96153eebd0bf77af',
                      'scdna': '258111f745946f3b',
                      'scrna': 'e37cf94f1ee9cf0e',
                      'visium': '8ab8e0c8a46a7ec2'},
 ('exact', 'open'): {'counts@1': '576f6d4a5c787cf2',
                     'counts@3': 'ec2994f23d3a05c6',
                     'counts@end': 'ec2994f23d3a05c6',
                     'births': '51560a0292a01c5f',
                     'spatial_snapshots': '7ab8fa84d365bf39',
                     'events': 'ec30c1e9135054c9',
                     'stroma_pct': 1.6,
                     'cell_data.cell_cnv': 'ee747b0868a95417',
                     'cell_data.cell_crd': 'f63d35821095f9a3',
                     'cell_data.cell_deme': '52b6485680386df6',
                     'cell_data.cell_evo': 'b1cf57e399443525',
                     'cell_data.cell_exp': '111c5c04fe9df0d9',
                     'cell_data.cell_exp_m': '5c15ffce947cd046',
                     'cell_data.cell_exp_p': 'b26b126a923c89e4',
                     'cell_data.cell_gland': 'de69c0ee54eb0037',
                     'cell_data.cell_microenv': '296114285ccbd7cb',
                     'cell_data.cell_program': '412ad4e75a423343',
                     'cell_data.cell_rna_baf': '1858b2590bdd589b',
                     'cell_data.cell_rna_vaf': '91efa6bce13c1cb7',
                     'cell_data.cell_snv': '1b037d6775cf6cca',
                     'cell_data.cell_snv_m': 'b8455b22bb2a03ac',
                     'cell_data.cell_snv_p': '5bc1901b949a0105',
                     'cell_data.cell_type': 'dad302ce5743665e',
                     'cell_data.cell_wgd': 'e04826df2cbd710e',
                     'wgs': 'ead2e858d0abf191',
                     'scdna': '2e194bf10de854c4',
                     'scrna': 'c9d4efe1415d414c',
                     'visium': '7d353a2aaf039ca6'},
 ('exact', 'exact_passengers'): {'counts@1': '361212c7ec5b6fe6',
                                 'counts@end': '6fcdcd1a577507af',
                                 'births': '1eef27600e1a8174',
                                 'spatial_snapshots': '7ab8fa84d365bf39',
                                 'events': '3725079ff1c83073',
                                 'stroma_pct': 0.0,
                                 'cell_data.cell_cnv': 'f8789a4d6ca1668e',
                                 'cell_data.cell_crd': '54ec645c71be88e0',
                                 'cell_data.cell_deme': 'c2084cb0033562a4',
                                 'cell_data.cell_evo': '58a11ec841c6f5fd',
                                 'cell_data.cell_exp': 'a4cf93f4eef61aae',
                                 'cell_data.cell_exp_m': '79f3926ed397f537',
                                 'cell_data.cell_exp_p': 'f4944ef012117648',
                                 'cell_data.cell_gland': '495ccc083ff3f81b',
                                 'cell_data.cell_microenv': '05850fbe3299afe8',
                                 'cell_data.cell_program': '978ace2d0f905f17',
                                 'cell_data.cell_rna_baf': 'f95004a47c56102d',
                                 'cell_data.cell_rna_vaf': '2048ccf82cb29de4',
                                 'cell_data.cell_snv': '9603d91d7e67db9f',
                                 'cell_data.cell_type': '91214ce5a70143d6',
                                 'cell_data.cell_wgd': 'b74374cff114956a',
                                 'wgs': '2a88fa337a87234b',
                                 'scdna': '18fe58c3a31bb6a9',
                                 'scrna': '5c6c489567b1af7e',
                                 'visium': 'f19b0cfd635abeb7'}}



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
