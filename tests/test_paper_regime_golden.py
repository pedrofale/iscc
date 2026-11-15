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
# Re-pinned when stromal survival was made to pay in the stroma (vault: decision-stromal-survival-selected):
# stromal_survival_cost 0.05, stromal_survival_effects 4.0 and stromal_hazard 0.9.
GOLDEN = {('tau', 'paper'): {'counts@1': '1282a482de6262c3',
                    'counts@3': 'a2915b6abce15bac',
                    'counts@end': '610958b604b3a50b',
                    'births': 'f7ac417796c805a0',
                    'spatial_snapshots': '0c5b181ade278d13',
                    'events': '43a661cc2b804f1d',
                    'stroma_pct': 0.0,
                    'cell_data.cell_cnv': '45bdc50298c56bdb',
                    'cell_data.cell_crd': '763198a3b2796d5c',
                    'cell_data.cell_deme': 'de65c25bbf79b8d3',
                    'cell_data.cell_evo': 'ed866d84d2b1b7c5',
                    'cell_data.cell_exp': '1116cb17cd9989b8',
                    'cell_data.cell_exp_m': 'bcfccdf6511c9394',
                    'cell_data.cell_exp_p': 'e6b4813a6777b9ec',
                    'cell_data.cell_gland': 'e8753e5fccd4c163',
                    'cell_data.cell_microenv': '7ca4eae6a13f9dfa',
                    'cell_data.cell_program': '416f6acfd01c4486',
                    'cell_data.cell_rna_baf': '8be1c1012ddb98fd',
                    'cell_data.cell_rna_vaf': 'a6a9ba25e6131a26',
                    'cell_data.cell_snv': '3e065423c4fea280',
                    'cell_data.cell_snv_m': '1a2c88df10d55e51',
                    'cell_data.cell_snv_p': '8eb12b3e2c8cb03d',
                    'cell_data.cell_type': '8261462b88066580',
                    'cell_data.cell_wgd': 'b965e5ffe07c7c83',
                    'wgs': 'b226273b138dbbc0',
                    'scdna': '9779fd308a16736c',
                    'scrna': '73ab527dac3f1500',
                    'visium': 'f219f708a7ec7d64'},
 ('tau', 'open'): {'counts@1': '1282a482de6262c3',
                   'counts@3': '3a729f2474df9eb0',
                   'counts@end': '4e896dad458f6e54',
                   'births': 'd6aaf2e634a211f2',
                   'spatial_snapshots': '7ab8fa84d365bf39',
                   'events': '43a661cc2b804f1d',
                   'stroma_pct': 1.066667,
                   'cell_data.cell_cnv': '0ec57f6c1d8baefa',
                   'cell_data.cell_crd': '3e62740cae180ac1',
                   'cell_data.cell_deme': '12f4e87b14058e96',
                   'cell_data.cell_evo': '21b7c45b789c7899',
                   'cell_data.cell_exp': '63ad2cd8f5607bba',
                   'cell_data.cell_exp_m': '30abc24a04155a45',
                   'cell_data.cell_exp_p': '20f5496879560b0d',
                   'cell_data.cell_gland': '609bbcce6661e0c3',
                   'cell_data.cell_microenv': '00e99b1fa92b09c2',
                   'cell_data.cell_program': 'd9fa0097c750c932',
                   'cell_data.cell_rna_baf': '427e2c3657dee511',
                   'cell_data.cell_rna_vaf': '2c7b9c3de62f39d7',
                   'cell_data.cell_snv': 'e5ca68ffba24634d',
                   'cell_data.cell_snv_m': '29ff3ecde4a90f0a',
                   'cell_data.cell_snv_p': 'f5b46866710375c5',
                   'cell_data.cell_type': 'e2bd07458f3a7cd8',
                   'cell_data.cell_wgd': 'f6263eefe7c5290a',
                   'wgs': 'dd8e5c27893ef544',
                   'scdna': 'c3e5a7e66c3baa16',
                   'scrna': '8949761bd3d44365',
                   'visium': 'dc9c64cc9031999c'},
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
                      'counts@3': '2211d64997ffb1e5',
                      'counts@end': '2211d64997ffb1e5',
                      'births': '4645270a80fdf452',
                      'spatial_snapshots': '408dddf5e97da880',
                      'events': 'ec30c1e9135054c9',
                      'stroma_pct': 0.0,
                      'cell_data.cell_cnv': '003e4d3d88bb1f81',
                      'cell_data.cell_crd': 'ff807d61e23f0eac',
                      'cell_data.cell_deme': '861a46b5f5c9ca92',
                      'cell_data.cell_evo': '57c1e870572a1b91',
                      'cell_data.cell_exp': 'b58f17fd5cc29171',
                      'cell_data.cell_exp_m': 'bdf89b3244635fe9',
                      'cell_data.cell_exp_p': 'd6876118bc95329d',
                      'cell_data.cell_gland': 'c072418c565dff06',
                      'cell_data.cell_microenv': '334d690a7d1889ce',
                      'cell_data.cell_program': '86194963b0d933b1',
                      'cell_data.cell_rna_baf': '4a25bfb742120cbe',
                      'cell_data.cell_rna_vaf': 'a740bf7c80344a98',
                      'cell_data.cell_snv': '638c587b25033fc3',
                      'cell_data.cell_snv_m': 'ffac08a5f7b8c150',
                      'cell_data.cell_snv_p': 'a72bdbb1b2548474',
                      'cell_data.cell_type': 'c56b6fdd9e55a395',
                      'cell_data.cell_wgd': 'f2882771395fa113',
                      'wgs': '4843095454b4ff4b',
                      'scdna': '184b146f0010b9c7',
                      'scrna': '3fe0da074bbc6b25',
                      'visium': '26dbf6cd79acc7c4'},
 ('exact', 'open'): {'counts@1': '576f6d4a5c787cf2',
                     'counts@3': '65b6ce2b22e93df1',
                     'counts@end': '65b6ce2b22e93df1',
                     'births': '46b1983f3bf379a2',
                     'spatial_snapshots': '7ab8fa84d365bf39',
                     'events': 'ec30c1e9135054c9',
                     'stroma_pct': 1.533333,
                     'cell_data.cell_cnv': '02c0b3134a16acd3',
                     'cell_data.cell_crd': '145cf8e9539905a8',
                     'cell_data.cell_deme': '04b175debabd3fe1',
                     'cell_data.cell_evo': 'c2dca6d048829d0b',
                     'cell_data.cell_exp': 'c648db6ff6878832',
                     'cell_data.cell_exp_m': '8aae838154a00e40',
                     'cell_data.cell_exp_p': '080860af79050974',
                     'cell_data.cell_gland': '3f80fadfde32cfc8',
                     'cell_data.cell_microenv': 'e19fdd9513776304',
                     'cell_data.cell_program': '03023f7800648322',
                     'cell_data.cell_rna_baf': '5097777926b47350',
                     'cell_data.cell_rna_vaf': '1c52ba3c5c314dbd',
                     'cell_data.cell_snv': '1a59a97954b28038',
                     'cell_data.cell_snv_m': '3f15d3b23582658b',
                     'cell_data.cell_snv_p': 'b0f75b4f5ff10e72',
                     'cell_data.cell_type': 'b97b16117a3499f9',
                     'cell_data.cell_wgd': '1d3a6c544ee82dc6',
                     'wgs': '04a0d42e470377f1',
                     'scdna': '868b0616f1fc1662',
                     'scrna': '1bef9e8d97fb5f69',
                     'visium': '86a48ea4e2223ad6'},
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
