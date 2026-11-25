"""Library sizes that follow the RNA content of a cell or a spot (``content_exponent``), and the warning
on Visium sections thinned to nearly one cell per spot."""
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from iscc.data import Visium, scRNA
from iscc.data.content import cell_rna_content, rna_content_table
from iscc.data.visium import SparseSpotWarning

PITCH = 2.0


def _section(density=12.0, radius=9.0, side=40, n_genes=30, seed=0):
    """A disk of tissue, epithelium on the left and T cells on the right, ``density`` cells per unit area."""
    rng = np.random.default_rng(seed)
    genes = [f"G{g}" for g in range(n_genes)]
    base = {k: rng.gamma(2.0, 1.0, n_genes) for k in ("epithelial", "immune")}
    ids, crd, exp, typ = [], [], [], []
    centre = np.array([side / 2.0, side / 2.0])
    for _ in range(int(4 * radius ** 2 * density)):
        p = centre + rng.uniform(-radius, radius, 2)
        if np.linalg.norm(p - centre) > radius:
            continue
        k = "epithelial" if p[1] < centre[1] else "immune"
        ids.append(f"C{len(ids)}")
        crd.append(p)
        exp.append(base[k] * rng.lognormal(0.0, 0.2, n_genes))
        typ.append(k)
    return {"cell_exp": pd.DataFrame(exp, index=ids, columns=genes),
            "cell_crd": pd.DataFrame(crd, index=ids, columns=["row", "col"]),
            "cell_type": pd.DataFrame(typ, index=ids, columns=["cell_id"])}, side


def test_content_table_is_the_atlas_relative_to_the_epithelium():
    t = rna_content_table()
    assert t["epithelial"] == 1.0
    assert t["immune"] == pytest.approx(0.284, abs=0.001)          # T and NK cells
    assert t["host"] == t["stromal"]


def test_cancer_cells_take_the_epithelium_and_double_with_the_genome():
    cd = {"cell_type": pd.DataFrame({"cell_id": ["epithelial", "immune", "c3", "c3"]}, index=list("abcd")),
          "cell_wgd": pd.DataFrame({"is_wgd": [False, False, False, True]}, index=list("abcd"))}
    c = cell_rna_content(cd)
    assert c.tolist() == pytest.approx([1.0, rna_content_table()["immune"], 1.0, 2.0])
    assert cell_rna_content(cd, cells=["d", "a"]).tolist() == pytest.approx([2.0, 1.0])


def test_scrna_library_follows_the_content_of_the_cell_type():
    cd, _ = _section()
    n = len(cd["cell_exp"])
    plain = scRNA(n_cells=n, seed=1, content_exponent=0.0).run(cd)
    scaled = scRNA(n_cells=n, seed=1).run(cd)                     # the default, 1
    typ = cd["cell_type"]["cell_id"].reindex(scaled.observed_counts.index)
    lib = scaled.observed_counts.sum(axis=1)
    ratio = lib[typ == "epithelial"].mean() / lib[typ == "immune"].mean()
    assert ratio == pytest.approx(1.0 / rna_content_table()["immune"], rel=0.15)    # about 3.5
    # the mean depth of the batch is kept
    assert lib.mean() == pytest.approx(plain.observed_counts.sum(axis=1).mean(), rel=0.05)


def test_scrna_default_is_proportional_to_the_content():
    cd, _ = _section()
    n = len(cd["cell_exp"])
    a = scRNA(n_cells=n, seed=1).run(cd).observed_counts.to_numpy()
    b = scRNA(n_cells=n, seed=1, content_exponent=1.0).run(cd).observed_counts.to_numpy()
    assert np.array_equal(a, b)


def test_visium_depth_follows_the_content_of_the_spot():
    cd, side = _section()
    kw = dict(seed=3, spot_pitch=PITCH, spot_radius=1.0, field_sigma=0.0, ambient_frac=0.0)
    plain = Visium(content_exponent=0.0, **kw).run(cd, grid_side=side)
    scaled = Visium(**kw).run(cd, grid_side=side)                 # the default, 1
    on = scaled.obs.n_cells > 0
    assert "rna_content" in scaled.obs and "rna_content" not in plain.obs
    rho_scaled = spearmanr(scaled.obs.library[on], scaled.obs.rna_content[on])[0]
    rho_plain = spearmanr(plain.obs.library[on], scaled.obs.rna_content[on])[0]
    assert rho_scaled > 0.5 > rho_plain
    assert scaled.obs.library[on].mean() == pytest.approx(plain.obs.library[on].mean(), rel=0.05)


def test_sparse_sections_warn_and_dense_ones_do_not():
    sparse, side = _section(density=0.15)
    with pytest.warns(SparseSpotWarning):
        Visium(seed=3, spot_pitch=PITCH, spot_radius=1.0).run(sparse, grid_side=side)
    dense, side = _section(density=12.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", SparseSpotWarning)
        Visium(seed=3, spot_pitch=PITCH, spot_radius=1.0).run(dense, grid_side=side)
