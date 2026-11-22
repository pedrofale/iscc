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
# The six Visium entries were re-pinned when the Visium assay took Visium v1's spot (radius 0.55 at
# pitch 2), the capture field became normalised over the tissue spots, ambient RNA became a share of
# the library, and the default kappa became 300,000 (visium-depth). Nothing else moved. The old Visium
# digests had matched on an ARM laptop, so these do not depend on the platform.
# Re-pinned on x86 (Arrhenius, numpy 1.26.4, scipy 1.15.3) at 2dc2abc, after checking the digests commit
# by commit along paper1/preprint: the scRNA defaults fitted to PBMC3k (rna-dropout) moved only the scrna
# entries, and germline sites staying in the mutation pool (germline; the regime carries germline) moved
# every growth entry. The viability rule of no segment at zero copies (nullisomy) moved no entry at this scale.
GOLDEN = {('tau', 'paper'): {'counts@1': 'b81e2e80e8fbfe0e',
                    'counts@3': 'd0ce12fa70e65ac8',
                    'counts@end': '4828dc4bb8a4c616',
                    'births': '7f1ce890def21f81',
                    'spatial_snapshots': '321d93bf09aeed14',
                    'events': '43a661cc2b804f1d',
                    'stroma_pct': 0.0,
                    'cell_data.cell_cnv': '41474370c78fbaf7',
                    'cell_data.cell_crd': 'e9b5b64c18c0726f',
                    'cell_data.cell_deme': '89e258f91e82d039',
                    'cell_data.cell_evo': 'adc090250fb84294',
                    'cell_data.cell_exp': 'b6a31b9d09d0fbea',
                    'cell_data.cell_exp_m': '33810d609555fbe1',
                    'cell_data.cell_exp_p': 'bbd30257ab0820d6',
                    'cell_data.cell_gland': '7c0095de8ba03a82',
                    'cell_data.cell_microenv': '2b19ad91d0e29743',
                    'cell_data.cell_program': '5675fa865c9bca31',
                    'cell_data.cell_rna_baf': 'ad4c19e80047c2fd',
                    'cell_data.cell_rna_vaf': '9021b51df96141fa',
                    'cell_data.cell_snv': '18808f13bf5c3299',
                    'cell_data.cell_snv_m': '825b4a3f5ffd8273',
                    'cell_data.cell_snv_p': '0ad6684a2bc70b21',
                    'cell_data.cell_type': '9662ec092ec64ab6',
                    'cell_data.cell_wgd': '54ca3051b964c5ab',
                    'wgs': 'ca3d98540c5cc542',
                    'scdna': 'ec8ff33558c71b84',
                    'scrna': '9cb4e1973ea09bf3',
                    'visium': '56ee198dcba933e8'},
 ('tau', 'open'): {'counts@1': 'b81e2e80e8fbfe0e',
                   'counts@3': '58d853c71efeec7d',
                   'counts@end': '6ffc841892ff2445',
                   'births': '667ad817489e528b',
                   'spatial_snapshots': '7ab8fa84d365bf39',
                   'events': '43a661cc2b804f1d',
                   'stroma_pct': 5.784574,
                   'cell_data.cell_cnv': '77fd22689949614f',
                   'cell_data.cell_crd': '796f3ef9d920992e',
                   'cell_data.cell_deme': '63ccfd9e6c81ca17',
                   'cell_data.cell_evo': '225bf19144b7209b',
                   'cell_data.cell_exp': 'e67a09a4f86f6850',
                   'cell_data.cell_exp_m': '02734433dce819a3',
                   'cell_data.cell_exp_p': '2687f6c834d1db7b',
                   'cell_data.cell_gland': 'b0cabbe4f147be03',
                   'cell_data.cell_microenv': 'd53a9a2a5da2d2b5',
                   'cell_data.cell_program': 'c558cfe5dae7798f',
                   'cell_data.cell_rna_baf': 'b8a7ece23e0619b1',
                   'cell_data.cell_rna_vaf': '1a597d4027ad5066',
                   'cell_data.cell_snv': '0aa6082223aa200b',
                   'cell_data.cell_snv_m': 'b31241325c951c76',
                   'cell_data.cell_snv_p': 'b8455cbfbfea6a5c',
                   'cell_data.cell_type': '1eed471921f9c8c6',
                   'cell_data.cell_wgd': 'b4f275c53c79c5ff',
                   'wgs': '66088ae42c64b0e2',
                   'scdna': '0d6d30920a91fb74',
                   'scrna': 'f616dec95c67c9d6',
                   'visium': '83ec4d1ad244c14c'},
 ('tau', 'exact_passengers'): {'counts@1': '5ac672396e0e5a7e',
                               'counts@3': '3c7674802ae50a3e',
                               'counts@end': '3c7674802ae50a3e',
                               'births': '4eef5b125d0ea38d',
                               'spatial_snapshots': '7ab8fa84d365bf39',
                               'events': 'f2f631f629780d84',
                               'stroma_pct': 0.0,
                               'cell_data.cell_cnv': 'ad6c2aaa1c925878',
                               'cell_data.cell_crd': '0ed45c1f38ace31e',
                               'cell_data.cell_deme': 'd16f8ffd48a446d0',
                               'cell_data.cell_evo': '62f90515aef28c2f',
                               'cell_data.cell_exp': 'bb392bea894a6a42',
                               'cell_data.cell_exp_m': '6076447a8377e2a6',
                               'cell_data.cell_exp_p': '2cc28ef5698f15ba',
                               'cell_data.cell_gland': '36c7807d78da8978',
                               'cell_data.cell_microenv': '967aad4fec90d28a',
                               'cell_data.cell_program': '85a4571eb4400e2e',
                               'cell_data.cell_rna_baf': '47fa3354727ffd5e',
                               'cell_data.cell_rna_vaf': '94a1852527dc3a7e',
                               'cell_data.cell_snv': 'd82fdc0475f21d6f',
                               'cell_data.cell_type': 'c5db4ea6fab49bbe',
                               'cell_data.cell_wgd': '65c8cc85d24d755c',
                               'wgs': '7667fb16e4a9d8c9',
                               'scdna': 'c2f142919bc6fce6',
                               'scrna': '3f096b662a0673a7',
                               'visium': '824d7381c8a043ec'},
 ('exact', 'paper'): {'counts@1': '31809c9c71eb72d3',
                      'counts@3': 'a3afe258fbe281c8',
                      'counts@end': 'a3afe258fbe281c8',
                      'births': 'a0f00b5537108b4d',
                      'spatial_snapshots': '700a2cbf9cf89a77',
                      'events': '93329d3f9de28b40',
                      'stroma_pct': 0.0,
                      'cell_data.cell_cnv': 'd4f180a5dd0e2e86',
                      'cell_data.cell_crd': '52e6e92cf14e23a4',
                      'cell_data.cell_deme': 'e254b195338765d0',
                      'cell_data.cell_evo': '7431cb77b4b18878',
                      'cell_data.cell_exp': '3b6c976f3f58b157',
                      'cell_data.cell_exp_m': 'fe1eaef004e73410',
                      'cell_data.cell_exp_p': 'e1cda8600fdad9a3',
                      'cell_data.cell_gland': '808f984f5b9e3694',
                      'cell_data.cell_microenv': '612a1358a31669eb',
                      'cell_data.cell_program': '2d67bc0d4c35843c',
                      'cell_data.cell_rna_baf': '0c9e34ae2504ac17',
                      'cell_data.cell_rna_vaf': '396e12b26a8553ad',
                      'cell_data.cell_snv': 'a5456752b514bf35',
                      'cell_data.cell_snv_m': '7e6a118dd958b6d4',
                      'cell_data.cell_snv_p': '84115a7dbf62ee2f',
                      'cell_data.cell_type': '7bbead5ef36645db',
                      'cell_data.cell_wgd': 'f6b40469dbc11ff0',
                      'wgs': '2abbb99f1c85f16b',
                      'scdna': '3a4eadd7e8a84f46',
                      'scrna': 'ea59740069c059d0',
                      'visium': 'c6071427599ce4fc'},
 ('exact', 'open'): {'counts@1': '0523454c429e9f3c',
                     'counts@3': '4466a70bc1ec1c18',
                     'counts@end': '4466a70bc1ec1c18',
                     'births': '95e8897392b71728',
                     'spatial_snapshots': '7ab8fa84d365bf39',
                     'events': '93329d3f9de28b40',
                     'stroma_pct': 1.8,
                     'cell_data.cell_cnv': 'fb819271884b2db2',
                     'cell_data.cell_crd': '6f2315ba8f40ace9',
                     'cell_data.cell_deme': '1595b51bb54ed8f5',
                     'cell_data.cell_evo': 'cfd099c78a73e3bb',
                     'cell_data.cell_exp': '8303cacac936a800',
                     'cell_data.cell_exp_m': '2a961cabd843483f',
                     'cell_data.cell_exp_p': '7acf5a597b914170',
                     'cell_data.cell_gland': '2816344efd5ae03c',
                     'cell_data.cell_microenv': 'd4aa994ef44a271b',
                     'cell_data.cell_program': '0c5a102862bbf22a',
                     'cell_data.cell_rna_baf': '23cc3fcca6169e7a',
                     'cell_data.cell_rna_vaf': '17a6bacbb831f30b',
                     'cell_data.cell_snv': 'd84a8679a31692a8',
                     'cell_data.cell_snv_m': '5e525919a284e6a1',
                     'cell_data.cell_snv_p': 'ac79f0b4138a9ca8',
                     'cell_data.cell_type': 'd1218c07f013c005',
                     'cell_data.cell_wgd': 'b74374cff114956a',
                     'wgs': '5055ed6ebe5e6263',
                     'scdna': '7f6c3c93fbd00ce4',
                     'scrna': 'a51440043c629ecd',
                     'visium': '0023972e6c3d5fe5'},
 ('exact', 'exact_passengers'): {'counts@1': '4e364b79caf5166f',
                                 'counts@end': '57e31453ce71cab2',
                                 'births': 'e80767144d997560',
                                 'spatial_snapshots': '7ab8fa84d365bf39',
                                 'events': '333e9dc87eb5526e',
                                 'stroma_pct': 0.0,
                                 'cell_data.cell_cnv': 'f8789a4d6ca1668e',
                                 'cell_data.cell_crd': '882e3673c3016ceb',
                                 'cell_data.cell_deme': '2bcfb6adb5d1a65d',
                                 'cell_data.cell_evo': '7067cd1a1d000cb5',
                                 'cell_data.cell_exp': '3be48442cb3ecc2f',
                                 'cell_data.cell_exp_m': 'f138869b7498fc8a',
                                 'cell_data.cell_exp_p': 'c82b27c266802b94',
                                 'cell_data.cell_gland': '495ccc083ff3f81b',
                                 'cell_data.cell_microenv': '7adf2e69862ee463',
                                 'cell_data.cell_program': '249a0c0726601df8',
                                 'cell_data.cell_rna_baf': 'cec0dcedd18c2cf4',
                                 'cell_data.cell_rna_vaf': '8bcdd6e56b27b59e',
                                 'cell_data.cell_snv': '08d9259429e4d85d',
                                 'cell_data.cell_type': 'dc27d309b11e105b',
                                 'cell_data.cell_wgd': 'b74374cff114956a',
                                 'wgs': '0660573a3060e1b9',
                                 'scdna': '18fe58c3a31bb6a9',
                                 'scrna': 'd9efd52fe8964347',
                                 'visium': '9a543833d80d7623'}}



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
