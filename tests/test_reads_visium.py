"""F6 Visium-reads tests (reuses F7b; DESIGN_features §D / F6 reads bullet).

Visium reads ARE scRNA reads barcoded by SPOT, so the SAME invariants as scRNA must hold:
  * spot UMI totals conserved exactly (alt + ref == spot count; reads never invent coverage);
  * the read count equals the conserved spot UMI total;
  * one spatial barcode per spot;
  * per-base sequencing error is the SINGLE read-error source (non-ref fraction ~ e, not ~2e);
plus the NEW spatial piece:
  * the spot-level RNA-VAF is the expression-weighted clone mixture of its member cells — a 2-clone
    spot lands BETWEEN the two cells' VAFs (the clone-mixture signal deconvolution must untangle).
"""
import gzip
import os

import numpy as np
import pandas as pd

from iscc.data.reads import emit_visium_reads, spot_clone_mixture_vaf, SyntheticTranscriptome


def _read_fastq(path):
    with gzip.open(path, "rt") as fh:
        lines = fh.read().splitlines()
    return [(lines[i], lines[i + 1]) for i in range(0, len(lines), 4)]


GENES = [f"G_{g}" for g in range(8)]


def make_cell_data(grid=20, seed=0):
    """Single cell per integer coordinate; clone A (rows<grid/2) vs B, with RNA-VAF blocks so the
    two clones carry distinct mutant fractions (used to check the spot clone-mixture VAF)."""
    rng = np.random.default_rng(seed)
    ids, coords, exp, ctype, rvaf = [], [], [], [], []
    i = 0
    for r in range(grid):
        for c in range(grid):
            clone = "A" if r < grid / 2 else "B"
            e = rng.gamma(3.0, 1.0, len(GENES))
            v = np.full(len(GENES), 0.8 if clone == "A" else 0.2)   # clone-specific RNA-VAF
            exp.append(e); coords.append((r, c)); ctype.append(clone)
            rvaf.append(v); ids.append(f"C{i}"); i += 1
    exp = pd.DataFrame(exp, index=ids, columns=GENES)
    return {
        "cell_exp": exp,
        "cell_crd": pd.DataFrame(coords, index=ids, columns=["row", "col"]),
        "cell_type": pd.DataFrame(ctype, index=ids, columns=["cell_id"]),
        "cell_rna_vaf": pd.DataFrame(rvaf, index=ids, columns=GENES),
    }


def test_umi_totals_conserved_and_counts_only(tmp_path):
    cd = make_cell_data()
    res = emit_visium_reads(cd, grid_side=20, obs_fidelity=0.6, seed=3,
                            spot_pitch=2.0, spot_radius=0.6, outdir=str(tmp_path))
    assert np.array_equal(res["alt"].values + res["ref"].values, res["total"].values)
    assert res["status"] == "counts-only"
    assert res["spot_rna_vaf"].shape == res["obs_vaf"].shape


def test_read_count_equals_spot_umi_total(tmp_path):
    cd = make_cell_data()
    res = emit_visium_reads(cd, grid_side=20, obs_fidelity=0.6, seed=3, emit_fastq=True,
                            read_length=60, spot_pitch=2.0, spot_radius=0.6, outdir=str(tmp_path))
    assert res["status"] == "emitted:synthetic-fastq"
    assert res["n_reads"] == int(res["alt"].values.sum() + res["ref"].values.sum())
    assert len(_read_fastq(res["fastq"][0])) == res["n_reads"]


def test_one_spatial_barcode_per_spot(tmp_path):
    cd = make_cell_data()
    res = emit_visium_reads(cd, grid_side=20, seed=3, emit_fastq=True, read_length=50,
                            spot_pitch=2.0, spot_radius=0.6, outdir=str(tmp_path))
    n_emitting_spots = int((res["total"].values.sum(axis=1) > 0).sum())
    barcodes_in_r1 = {seq[:16] for _, seq in _read_fastq(res["fastq"][0])}
    # one distinct spatial barcode per spot that emitted any read
    assert len(res["barcodes"]) == res["total"].shape[0]
    assert len(barcodes_in_r1) == n_emitting_spots


def test_error_rate_is_single_source(tmp_path):
    """Hom-ref molecular content -> the FASTQ non-ref fraction at the variant site ~ e (the per-base
    error is the ONLY read-error source, not doubled by also folding it into the alt/ref split)."""
    e = 0.05
    grid = 20
    rng = np.random.default_rng(0)
    ids, coords, exp, rvaf = [], [], [], []
    i = 0
    for r in range(grid):
        for c in range(grid):
            exp.append(rng.gamma(4.0, 3.0, len(GENES)))   # well-expressed
            rvaf.append(np.zeros(len(GENES)))             # hom-ref: true RNA-VAF 0 everywhere
            coords.append((r, c)); ids.append(f"C{i}"); i += 1
    cd = {
        "cell_exp": pd.DataFrame(exp, index=ids, columns=GENES),
        "cell_crd": pd.DataFrame(coords, index=ids, columns=["row", "col"]),
        "cell_rna_vaf": pd.DataFrame(rvaf, index=ids, columns=GENES),
    }
    tx = SyntheticTranscriptome(GENES, seed=0, read_length=50)
    res = emit_visium_reads(cd, grid_side=grid, obs_fidelity=1.0, error_rate=e, seed=1,
                            emit_fastq=True, transcriptome=tx, spot_pitch=2.0, spot_radius=0.6,
                            outdir=str(tmp_path))
    vp = tx.var_pos
    nonref = tot = 0
    for rid, seq in _read_fastq(res["fastq"][1]):
        g = rid.split(":")[1]
        nonref += int(seq[vp] != tx.seq[g][vp]); tot += 1
    assert abs(nonref / tot - e) < 0.015              # ~e, NOT ~2e


# ---------------------------------------------------------------- clone-mixture VAF -----------
def test_spot_rna_vaf_is_expression_weighted_mixture():
    """A 2-clone spot's RNA-VAF is the expression-weighted average of its members' VAFs, so it
    lands strictly BETWEEN the two clones' values (here 0.2 and 0.8)."""
    genes = ["g0"]
    cell_exp = pd.DataFrame({"g0": [10.0, 30.0]}, index=["cA", "cB"])      # B expresses 3x
    cell_vaf = pd.DataFrame({"g0": [0.8, 0.2]}, index=["cA", "cB"])
    members = [np.array(["cA", "cB"])]
    srv = spot_clone_mixture_vaf(members, cell_exp, cell_vaf, genes)
    expected = (10 * 0.8 + 30 * 0.2) / (10 + 30)       # = 0.35
    assert abs(srv.iloc[0, 0] - expected) < 1e-9
    assert 0.2 < srv.iloc[0, 0] < 0.8                  # strictly between the two cells' VAFs


def test_spot_rna_vaf_zero_where_unexpressed():
    genes = ["g0", "g1"]
    cell_exp = pd.DataFrame({"g0": [5.0], "g1": [0.0]}, index=["cA"])     # g1 unexpressed
    cell_vaf = pd.DataFrame({"g0": [0.5], "g1": [0.9]}, index=["cA"])
    srv = spot_clone_mixture_vaf([np.array(["cA"])], cell_exp, cell_vaf, genes)
    assert srv.iloc[0, 0] == 0.5
    assert srv.iloc[0, 1] == 0.0                       # 0 where no expression


def test_full_path_clone_mixture_lands_between(tmp_path):
    """End-to-end: at the clone boundary a spot mixes A (VAF 0.8) and B (VAF 0.2) cells, so its
    spot_rna_vaf falls between — exactly the clone mixture spatial deconvolution must resolve."""
    cd = make_cell_data()
    res = emit_visium_reads(cd, grid_side=20, obs_fidelity=1.0, seed=3, spot_pitch=2.0,
                            spot_radius=1.5, outdir=str(tmp_path))   # wide radius -> mixed spots
    srv = res["spot_rna_vaf"].values
    frac = res["clone_frac"].values
    mixed = (frac < 1.0) & (res["n_cells"].values > 1) & (res["total"].values.sum(axis=1) > 0)
    vals = srv[mixed]
    vals = vals[vals > 0]
    assert mixed.sum() > 0
    assert np.any((vals > 0.2) & (vals < 0.8))         # mixed spots land between the clones
