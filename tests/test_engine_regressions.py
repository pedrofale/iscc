"""Regression tests for the engine-correctness bugs fixed in the AUDIT.md Section A/B pass.

Each test pins down one previously-broken behaviour so it cannot silently regress.
"""
import numpy as np
import pytest

from iscc.tumor.components.cell import CancerCell
from iscc.tumor.components.selection import Selection

from conftest import (
    N_SEGMENTS,
    SEGMENT_SIZE,
    SELECTION_PARAMS,
    CANCER_CELL_PARAMS,
)


def _make_selection(seed=0):
    np.random.seed(seed)
    return Selection(n_segments=N_SEGMENTS, segment_size=SEGMENT_SIZE, **SELECTION_PARAMS)


def _make_cancer_cell(selection):
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


# --- A1: genome segments must be independent objects -------------------------
def test_genome_segments_are_independent():
    cell = CancerCell(n_segments=4, segment_size=10)
    # Distinct dicts, distinct inner lists, distinct bitsets.
    for i in range(1, 4):
        assert cell.genome[0] is not cell.genome[i]
        assert cell.genome[0]['p'] is not cell.genome[i]['p']
        assert cell.genome[0]['p'][0] is not cell.genome[i]['p'][0]
    # Mutating one segment must not touch the others.
    cell.genome[0]['p'][0][3] = True
    assert all(not cell.genome[i]['p'][0][3] for i in range(1, 4))


# --- A2: driver counting uses len(), not sum(), of positions -----------------
def test_mutation_summary_counts_drivers_not_index_sum():
    selection = _make_selection()
    cell = _make_cancer_cell(selection)
    seg = 0
    mut_bits = np.zeros(SEGMENT_SIZE, dtype=bool)
    # Choose two oncogene positions if available, else fall back to any positions.
    onc = list(selection.onc[seg])
    if len(onc) >= 2:
        mut_bits[onc[:2]] = True
        expected = 2
    else:
        mut_bits[:2] = True
        expected = int(mut_bits[selection.onc[seg]].sum())
    before = cell.genome_summary['n_mut_onc']
    cell.update_genome_summary_mutation(selection, mut_bits, seg)
    delta = cell.genome_summary['n_mut_onc'] - before
    assert delta == expected
    # A count can never exceed the number of mutated positions.
    assert delta <= int(mut_bits.sum())


# --- A3: nullisomy_count counts zero-copy segments ---------------------------
def test_nullisomy_count_is_zero_copy_segments():
    selection = _make_selection()
    cell = _make_cancer_cell(selection)
    seg = 0
    # Delete the single allele on each haplotype of segment 0 -> copy number 0.
    # seg_cns starts at 2 (one 'p' + one 'm' allele).
    allele_muts = cell.genome[seg]['p'][0]
    cell.update_genome_summary_cnv(selection, allele_muts, seg, sign=-1)
    cell.update_genome_summary_cnv(selection, cell.genome[seg]['m'][0], seg, sign=-1)
    cell.genome_summary['seg_cns'][seg] = 0  # ensure exactly zero for the assertion
    cell.update_genome_summary_cnv(selection, np.zeros(SEGMENT_SIZE, dtype=bool), seg, sign=0)
    assert cell.genome_summary['nullisomy_count'] == int(
        np.sum(np.asarray(cell.genome_summary['seg_cns']) == 0)
    )
    assert cell.genome_summary['nullisomy_count'] >= 1


# --- A4: celltype baseline expression indexes drivers correctly --------------
def test_make_celltype_exps_sets_driver_genes(glandular_tumor):
    t = glandular_tumor
    tsgs = t.selection.get_tsgs()
    oncs = t.selection.get_oncogenes()
    for celltype, exp in t.celltype_exps.items():
        assert np.allclose(exp[tsgs], 0.8)
        assert np.allclose(exp[oncs], 0.01)


# --- A5: death_rate is not overwritten by the division-rate fitness ----------
def test_death_rate_not_changed_by_genome_selection():
    selection = _make_selection()
    cell = _make_cancer_cell(selection)
    death_before = cell.evolutionary_parameters['death_rate']
    cell.update_evolutionary_parameters(selection)
    # death rate stays at the configured baseline; genome-driven selection acts on
    # division/dispersal/resistance, not death (treatment changes death separately).
    assert cell.evolutionary_parameters['death_rate'] == death_before


# --- A6: gene_to_pos parses the actual gene-name format ----------------------
def test_gene_to_pos_roundtrip():
    selection = _make_selection()
    names = selection.get_gene_names()
    # G_{seg}_{pos}; pick segment 1, position 3 if it exists.
    target = f"{names[0].split('_')[0]}_1_3"
    seg, pos = selection.gene_to_pos(target)
    assert (seg, pos) == (1, 3)


# --- B7: expresses() indexing is valid and self-sufficient -------------------
def test_expresses_runs_without_baseline(monkeypatch):
    selection = _make_selection()
    cell = _make_cancer_cell(selection)
    # No baseline_exp set yet; expresses() should set one and return a bool.
    assert not hasattr(cell, 'baseline_exp')
    result = cell.expresses((0, 0), selection.mut_effects)
    assert isinstance(result, (bool, np.bool_))
    assert hasattr(cell, 'baseline_exp')


# --- B13: divide() gives the daughter its own evolutionary_parameters --------
def test_divide_isolates_evolutionary_parameters():
    selection = _make_selection()
    cell = _make_cancer_cell(selection)
    daughter = cell.divide()
    assert daughter.evolutionary_parameters is not cell.evolutionary_parameters
    daughter.evolutionary_parameters['death_rate'] = 0.99
    assert cell.evolutionary_parameters['death_rate'] != 0.99


# --- Phase 2: copy-on-write genotypes ----------------------------------------
def test_divide_shares_genome_until_mutation():
    """A non-mutating daughter shares the parent's genome object (copy-on-write)."""
    selection = _make_selection()
    cell = _make_cancer_cell(selection)
    daughter = cell.divide()
    # cheap path: genome/summary shared by reference, evolutionary_parameters not
    assert daughter.genome is cell.genome
    assert daughter.genome_summary is cell.genome_summary
    assert daughter.evolutionary_parameters is not cell.evolutionary_parameters


def test_mutation_does_not_corrupt_shared_clonemates():
    """Mutating one cell must not change a clonemate that shared its genome."""
    import numpy as np
    selection = _make_selection()
    cell = _make_cancer_cell(selection)
    daughter = cell.divide()
    assert daughter.genome is cell.genome  # sharing before mutation
    # snapshot the parent's genome/summary, then mutate the daughter
    parent_snv_before = cell.get_snvs().copy()
    parent_summary_before = dict(cell.genome_summary)
    daughter.mutate(np.random.default_rng(0), selection, n_snvs_per_allele=3)
    # daughter diverged...
    assert daughter.genome is not cell.genome
    # ...and the parent (and any other sharer) is untouched
    assert np.array_equal(cell.get_snvs(), parent_snv_before)
    assert dict(cell.genome_summary) == parent_summary_before


# --- B8: immune-cell dispersal migrates the cell and conserves the population -
def test_immune_dispersal_migrates_cell(glandular_tumor):
    from iscc.tumor.components.cell import ImmuneCell
    t = glandular_tumor
    src = t.grid[2][2]
    imm = ImmuneCell(n_segments=N_SEGMENTS, segment_size=SEGMENT_SIZE,
                     division_rate=0.0, death_rate=0.1, dispersal_rate=0.1)
    src.add_cell(imm)
    before = sum(len(d.cells) for d in t.deme_list)
    src.apply_event('dispersal', imm, rng=np.random.default_rng(0))
    after = sum(len(d.cells) for d in t.deme_list)
    assert before == after            # population conserved (moved, not duplicated/lost)
    assert imm not in src.cells       # left the source deme
    assert any(imm in d.cells for d in t.deme_list)  # landed in a neighbour


# --- B12/B14: a structured glandular tumor builds and grows end-to-end --------
def test_structured_glandular_tumor_builds_and_grows():
    from collections import Counter
    from iscc.tumor.models.glandular import GlandularTumor
    from conftest import (GENOME_PARAMS, SELECTION_PARAMS, CANCER_CELL_PARAMS,
                          EPITHELIAL_CELL_PARAMS, STROMAL_CELL_PARAMS,
                          IMMUNE_CELL_PARAMS, DEME_PARAMS)
    t = GlandularTumor(
        genome_params=GENOME_PARAMS, selection_params=SELECTION_PARAMS,
        cancer_cell_params=CANCER_CELL_PARAMS, epithelial_cell_params=EPITHELIAL_CELL_PARAMS,
        stromal_cell_params=STROMAL_CELL_PARAMS, immune_cell_params=IMMUNE_CELL_PARAMS,
        deme_params=DEME_PARAMS, grid_size=9, structure_radius=3,
    )
    types = Counter(c.type for d in t.deme_list for c in d.cells)
    # structure_radius > 0 must place all three normal/cancer populations
    assert {'epithelial', 'stromal', 'cancer'} <= set(types)
    # normal cells must share the tumor's genome size (segment_size propagated, B14)
    a_cell = next(iter(next(d for d in t.deme_list if d.cells).cells))
    assert a_cell.segment_size == SEGMENT_SIZE
    t.grow(n_steps=5, seed=0)
    assert t.cell_data['cell_snv'].shape[1] == t.n_genes
