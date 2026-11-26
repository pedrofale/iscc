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

Float arrays are digested at 24 bits of mantissa (about seven significant digits). numpy's SIMD paths
and OpenBLAS's kernels round the last bits differently on ARM and on x86 with and without AVX-512, so
the expression, program, allele-fraction and WGS tables differed between the CI runners at full
precision; at 32 bits they already agreed between an ARM laptop and an AVX-512 node.

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
FLOAT_BITS = 24                         # mantissa bits of a float array that enter its digest
RECORD =os.environ.get("ISCC_GOLDEN_RECORD") == "1"

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
# The six Visium entries were re-pinned when the Visium assay took Visium v1's spot (radius 0.55 at
# pitch 2), the capture field became normalised over the tissue spots, ambient RNA became a share of
# the library, and the default kappa became 300,000 (visium-depth). Nothing else moved. The old Visium
# digests had matched on an ARM laptop, so these do not depend on the platform.
# Re-pinned on x86 (Arrhenius, numpy 1.26.4, scipy 1.15.3) at ef30d3e, after checking the digests commit
# by commit along paper1/preprint: the scRNA defaults fitted to PBMC3k (rna-dropout) moved only the scrna
# entries, and germline sites staying in the mutation pool (germline; the regime carries germline) moved
# every growth entry. The viability rule of no segment at zero copies (nullisomy) moved no entry at
# this scale. Every entry that digests a float array was then re-pinned once more when float arrays
# began to enter their digest at FLOAT_BITS of mantissa.
# The expression entries (cell_exp, cell_exp_m, cell_exp_p, cell_program, scRNA and Visium) were
# re-pinned when the niche drive was confined to cancer cells (normal-programmes; vault:
# decision-normal-cells-no-programme-factor), on an ARM laptop; no growth, genotype, WGS or scDNA entry moved.
# The six Visium entries were re-pinned when the Visium concentration was refitted to 1,400,000
# (visium-kappa-refit); nothing else moved.
# Re-pinned when the stroma split into fibroblasts, endothelial and perivascular cells (stromal-subtypes;
# vault: decision-stromal-subtypes), on an ARM laptop, matching the x86 CI runners. Growth is unchanged:
# the time, the steps, the events and, after relabelling the genotype ids by rank, the cancer counts
# of every deme and the genealogy are identical in all six runs. The two subtype genotypes are created
# at seeding, so every later genotype takes an id and a creation ordinal two higher, which moves every
# entry keyed by them; the cell tables and the assays also hold the new cell types.
# The six scrna entries were re-pinned when the 10x preset's per-gene batch factor took its fit on two
# 10x runs of one sample (sigma_batch 0.10 -> 0.29; vault: decision-10x-per-gene-factor-fitted), on an
# ARM laptop; nothing else moved.
# The six scrna and six visium entries were re-pinned when the library of a cell and the depth of a spot
# came to follow their RNA content by default (content_exponent 1; vault:
# decision-library-size-from-rna-content), on an ARM laptop; nothing else moved.
# The cell_evo, scrna and visium entries were re-pinned when cell_evo gained each genotype's mean copy
# number and the RNA content of a cancer cell came to follow it (the epithelium times mean_cn / 2, which a
# doubling alone set before); nothing else moved (exact exact_passengers' visium did not move either).
# The six visium entries were re-pinned when the content came to take its share of the fitted spread of
# depth instead of adding to it, and Visium's content_exponent was refitted to 0.5 (vault:
# decision-library-size-from-rna-content); nothing else moved.
GOLDEN = {('exact', 'exact_passengers'): {'births': '13b6f004e1565d74',
                                 'cell_data.cell_cnv': 'e7c022a031b06903',
                                 'cell_data.cell_crd': '0fb76952e920c6b0',
                                 'cell_data.cell_deme': '51c12b0d07768932',
                                 'cell_data.cell_evo': '8b5a2ad77ac40050',
                                 'cell_data.cell_exp': 'b03ef2ad11e07283',
                                 'cell_data.cell_exp_m': 'cf28059dd2184e9b',
                                 'cell_data.cell_exp_p': 'eef47a18f9bb4b88',
                                 'cell_data.cell_gland': '3678666b38f0b6af',
                                 'cell_data.cell_lineage': '22e38b77e3aa5f21',
                                 'cell_data.cell_microenv': '128eed4098480d02',
                                 'cell_data.cell_program': 'b7fcaaf444fcdfc0',
                                 'cell_data.cell_rna_baf': '52e152efa34dedec',
                                 'cell_data.cell_rna_vaf': '4f15a16a5c8961aa',
                                 'cell_data.cell_snv': '1e88ac7e54917581',
                                 'cell_data.cell_type': 'ddfef7bd625eaef8',
                                 'cell_data.cell_wgd': '1d3a6c544ee82dc6',
                                 'counts@1': '001a153a00a505da',
                                 'counts@end': '04922b2dd3f68d8c',
                                 'events': '333e9dc87eb5526e',
                                 'scdna': '0e4eca7b67c4f25b',
                                 'scrna': '4f5c8de8876da8b9',
                                 'spatial_snapshots': '7ab8fa84d365bf39',
                                 'stroma_pct': 0.0,
                                 'visium': 'ac8bc9bb61635bdc',
                                 'wgs': '69c8402b9c2a46f1'},
 ('exact', 'open'): {'births': '7bc0688e75518ce2',
                     'cell_data.cell_cnv': 'da41ef7ec8344559',
                     'cell_data.cell_crd': '2386ebc8b7941481',
                     'cell_data.cell_deme': '83cdeaf1600aec98',
                     'cell_data.cell_evo': '72f548cc3c0fc148',
                     'cell_data.cell_exp': 'a8a463c663a85d21',
                     'cell_data.cell_exp_m': 'b5b080caecbb1c3d',
                     'cell_data.cell_exp_p': 'ca2f998849115b3a',
                     'cell_data.cell_gland': '62404d497385e630',
                     'cell_data.cell_lineage': '5f04abb505753ffc',
                     'cell_data.cell_microenv': 'df0c69dee431e7ed',
                     'cell_data.cell_program': '3b24c76096d4dd26',
                     'cell_data.cell_rna_baf': 'c892ce10d35603b8',
                     'cell_data.cell_rna_vaf': '297874d89345e79b',
                     'cell_data.cell_snv': 'af8498edd12a31b8',
                     'cell_data.cell_snv_m': '5d8000aeb997ba0c',
                     'cell_data.cell_snv_p': 'a22e7984b40730ce',
                     'cell_data.cell_type': '7394fc3c1142d4e6',
                     'cell_data.cell_wgd': '90cfd41bf004c503',
                     'counts@1': '8373b8a188a4de9b',
                     'counts@3': 'd06af24f4b15c6b9',
                     'counts@end': 'd06af24f4b15c6b9',
                     'events': '93329d3f9de28b40',
                     'scdna': 'b81a3d8015bf03b7',
                     'scrna': '66d2cfe79497cefd',
                     'spatial_snapshots': '7ab8fa84d365bf39',
                     'stroma_pct': 1.8,
                     'visium': '6af8f8ccc6b3f921',
                     'wgs': 'c6e1d92cbda9ffa9'},
 ('exact', 'paper'): {'births': '8dc564815e8bc8f5',
                      'cell_data.cell_cnv': '0dd83ce64038d284',
                      'cell_data.cell_crd': '2c325858b549e20e',
                      'cell_data.cell_deme': '9926341445ce5c83',
                      'cell_data.cell_evo': '8007234770fab986',
                      'cell_data.cell_exp': '8978e481ad176723',
                      'cell_data.cell_exp_m': '78fe0cc304f44ce9',
                      'cell_data.cell_exp_p': 'ad8d8248820ee88c',
                      'cell_data.cell_gland': '125db85b57d48bd3',
                      'cell_data.cell_lineage': 'd33db6ca9146ada5',
                      'cell_data.cell_microenv': '135ac99e3789014e',
                      'cell_data.cell_program': '3ed6622304c75fc6',
                      'cell_data.cell_rna_baf': '92d2e223b864817b',
                      'cell_data.cell_rna_vaf': 'fa6f90bb55d2fd6a',
                      'cell_data.cell_snv': '94d9ee868177d6c0',
                      'cell_data.cell_snv_m': 'aafeae2a130b7892',
                      'cell_data.cell_snv_p': 'c2fd6c0c53df16a5',
                      'cell_data.cell_type': '7052682f78e3dfed',
                      'cell_data.cell_wgd': '75595b2fd6c196f4',
                      'counts@1': '14e64af88cd8151e',
                      'counts@3': 'a116f19b9cfae3fa',
                      'counts@end': 'a116f19b9cfae3fa',
                      'events': '93329d3f9de28b40',
                      'scdna': 'a7c48d4ce2d61be1',
                      'scrna': 'cdeee3c891fc3c00',
                      'spatial_snapshots': '26c0fc71fb555423',
                      'stroma_pct': 0.0,
                      'visium': '2f167c1f664053b5',
                      'wgs': '43c37a09f8698406'},
 ('tau', 'exact_passengers'): {'births': '358cdbcbca547c77',
                               'cell_data.cell_cnv': '094f24c85e414ff4',
                               'cell_data.cell_crd': '50b920f9ab99084d',
                               'cell_data.cell_deme': 'c62d9b7d43b0f073',
                               'cell_data.cell_evo': '0ec88e6784d2d28d',
                               'cell_data.cell_exp': 'fa2f7a9bc033d535',
                               'cell_data.cell_exp_m': 'e1f5c2dccf46a892',
                               'cell_data.cell_exp_p': '43d520f2067b52bb',
                               'cell_data.cell_gland': '9d4ab3d7327ef009',
                               'cell_data.cell_lineage': '6c11acbc3dc63de2',
                               'cell_data.cell_microenv': '6ad52abc8c5e4c12',
                               'cell_data.cell_program': '653b1e98d7a85c63',
                               'cell_data.cell_rna_baf': '351a49b90d2d4e97',
                               'cell_data.cell_rna_vaf': '9ede53762035dfea',
                               'cell_data.cell_snv': '830b1952175b74b2',
                               'cell_data.cell_type': '86e036804ff8fe4f',
                               'cell_data.cell_wgd': 'd9b9375fea4dc355',
                               'counts@1': '294ff314c5e28053',
                               'counts@3': 'a80f25bd711f088c',
                               'counts@end': 'a80f25bd711f088c',
                               'events': 'f2f631f629780d84',
                               'scdna': '08d758263d12f86f',
                               'scrna': 'e5d5e1178875f718',
                               'spatial_snapshots': '7ab8fa84d365bf39',
                               'stroma_pct': 0.0,
                               'visium': '4f90485266ffde64',
                               'wgs': '705c1dc04b9f4bf5'},
 ('tau', 'open'): {'births': '5f460a9633553fab',
                   'cell_data.cell_cnv': '26e0b317135ce064',
                   'cell_data.cell_crd': '4686064eeac81dd3',
                   'cell_data.cell_deme': '1f1b7865083f1a4a',
                   'cell_data.cell_evo': '0728212884d0fde5',
                   'cell_data.cell_exp': '9aede8e8528b0418',
                   'cell_data.cell_exp_m': '7795263eacb7608b',
                   'cell_data.cell_exp_p': '397f875317a036ac',
                   'cell_data.cell_gland': 'cbcda149090ba495',
                   'cell_data.cell_lineage': '4a0a58616e33796d',
                   'cell_data.cell_microenv': '07db7c154553c827',
                   'cell_data.cell_program': 'baea80227520c30e',
                   'cell_data.cell_rna_baf': '6bd503939e0979f0',
                   'cell_data.cell_rna_vaf': 'e1ff9d5c25bf8c7f',
                   'cell_data.cell_snv': 'e1e41f3226cd5c02',
                   'cell_data.cell_snv_m': '056864c2bb117840',
                   'cell_data.cell_snv_p': '841516944f1b1f9c',
                   'cell_data.cell_type': '6a269dcd0296325b',
                   'cell_data.cell_wgd': '0fdfd91026f17c85',
                   'counts@1': 'bc49f17f1f8abc66',
                   'counts@3': '4848a93e2b280715',
                   'counts@end': '4159eeb4daba07de',
                   'events': '43a661cc2b804f1d',
                   'scdna': 'adccd637e950aec6',
                   'scrna': 'aedb4e7acebff733',
                   'spatial_snapshots': '7ab8fa84d365bf39',
                   'stroma_pct': 5.784574,
                   'visium': '40057b1a43ca63a8',
                   'wgs': 'e574fb6939da36f5'},
 ('tau', 'paper'): {'births': '986bd3027a9c415f',
                    'cell_data.cell_cnv': '56ea96c695b95fa3',
                    'cell_data.cell_crd': '31c116de7c53eda1',
                    'cell_data.cell_deme': 'e83adc641eb7284b',
                    'cell_data.cell_evo': '100ef9da616137ee',
                    'cell_data.cell_exp': '6471b03e72eb07fc',
                    'cell_data.cell_exp_m': 'e2c80f8b463b5116',
                    'cell_data.cell_exp_p': '5e989f6c5cb2404a',
                    'cell_data.cell_gland': 'b29a8eed535890dc',
                    'cell_data.cell_lineage': '0c51fa41cc002f0f',
                    'cell_data.cell_microenv': '539616ba27be3d5c',
                    'cell_data.cell_program': '073bbb1e6f686dcf',
                    'cell_data.cell_rna_baf': '01224f5aba8136d6',
                    'cell_data.cell_rna_vaf': 'ecfc97ec1ae9ff37',
                    'cell_data.cell_snv': 'e4e2790f05e899b9',
                    'cell_data.cell_snv_m': '1536ac6296bbe38d',
                    'cell_data.cell_snv_p': '0b20e87b58076953',
                    'cell_data.cell_type': '6b974ca683877d62',
                    'cell_data.cell_wgd': '04cc9f868b1249db',
                    'counts@1': 'bc49f17f1f8abc66',
                    'counts@3': '8581bde3ce4ec167',
                    'counts@end': '1d86e36d3b522701',
                    'events': '43a661cc2b804f1d',
                    'scdna': 'aeb20c3048a5a67f',
                    'scrna': 'e9acf92b6c96a357',
                    'spatial_snapshots': '45a606a53cf29aef',
                    'stroma_pct': 0.0,
                    'visium': 'b2c92d6a60347bb4',
                    'wgs': '419ef883d6e8067c'}}


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
                if o.dtype.kind == "f":                 # round off the CPU's last bits
                    m, e = np.frexp(o)
                    o = np.ldexp(np.round(m * 2.0 ** FLOAT_BITS) / 2.0 ** FLOAT_BITS, e)
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
    # kappa 2000 as given: the bulk default when the digests were recorded (now the HG002 fit), pinned
    # so the digest keeps checking the dissociated sample rather than the assay default
    wgs = bulkDNA(breadth="wgs", seed=11, kappa=2000.0).run(diss)
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
