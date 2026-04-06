"""Shared fixtures for iscc tests."""
import numpy as np
import pandas as pd
import pytest

# Small genome so tests run fast
N_SEGMENTS = 2
SEGMENT_SIZE = 20
N_GENES = N_SEGMENTS * SEGMENT_SIZE

GENOME_PARAMS = {"n_segments": N_SEGMENTS, "segment_size": SEGMENT_SIZE}
SELECTION_PARAMS = {
    "prop_driver": 0.1,
    "prop_dispersal": 0.1,
    "prop_immune_resistance": 0.1,
    "prop_treatment_resistance": 0.1,
}
CANCER_CELL_PARAMS = {
    "division_rate": 0.5,
    "death_rate": 0.1,
    "max_birth_rate": 0.8,
    "mutation_rate": 0.5,
    "dispersal_rate": 0.2,
}
EPITHELIAL_CELL_PARAMS = {"division_rate": 0.0, "death_rate": 0.1}
STROMAL_CELL_PARAMS = {"division_rate": 0.0, "death_rate": 0.1}
IMMUNE_CELL_PARAMS = {"division_rate": 0.0, "death_rate": 0.1, "dispersal_rate": 0.1}
DEME_PARAMS = {"carrying_capacity": 2}


@pytest.fixture
def selection():
    from iscc.tumor.components.selection import Selection
    return Selection(n_segments=N_SEGMENTS, segment_size=SEGMENT_SIZE, **SELECTION_PARAMS)


@pytest.fixture
def cancer_cell(selection):
    from iscc.tumor.components.cell import CancerCell
    return CancerCell(
        n_segments=N_SEGMENTS,
        segment_size=SEGMENT_SIZE,
        n_onc=len(selection.get_oncogenes()),
        n_tsg=len(selection.get_tsgs()),
        n_disp=len(selection.get_dispersal_genes()),
        n_ir=len(selection.get_immune_resistant()),
        n_tr=len(selection.get_treatment_resistant()),
        **CANCER_CELL_PARAMS,
    )


@pytest.fixture
def glandular_tumor():
    from iscc.tumor.models.glandular import GlandularTumor
    return GlandularTumor(
        genome_params=GENOME_PARAMS,
        selection_params=SELECTION_PARAMS,
        cancer_cell_params=CANCER_CELL_PARAMS,
        epithelial_cell_params=EPITHELIAL_CELL_PARAMS,
        stromal_cell_params=STROMAL_CELL_PARAMS,
        immune_cell_params=IMMUNE_CELL_PARAMS,
        deme_params=DEME_PARAMS,
        grid_size=5,
        structure_radius=0,
    )


@pytest.fixture
def grown_tumor(glandular_tumor):
    glandular_tumor.grow(n_steps=10, seed=0)
    return glandular_tumor


@pytest.fixture
def synthetic_cell_data():
    """Minimal cell_data dict matching Tumor.make_cell_data() output."""
    rng = np.random.default_rng(0)
    n_cells = 20
    cell_names = [f"C{i}" for i in range(n_cells)]
    snv = pd.DataFrame(
        rng.integers(0, 2, size=(n_cells, N_GENES)).astype(float),
        index=cell_names,
        columns=[f"G_{s}_{p}" for s in range(N_SEGMENTS) for p in range(SEGMENT_SIZE)],
    )
    exp = pd.DataFrame(
        rng.beta(0.1, 1.0, size=(n_cells, N_GENES)),
        index=cell_names,
        columns=snv.columns,
    )
    cnv = pd.DataFrame(
        np.full((n_cells, N_GENES), 2.0),
        index=cell_names,
        columns=snv.columns,
    )
    return {"cell_snv": snv, "cell_exp": exp, "cell_cnv": cnv}
