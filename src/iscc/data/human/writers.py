"""Write human-gene assays in the formats copy-number tools read, with the truth beside them.

:func:`write_visium` writes a Space Ranger-style directory (for CalicoST, Clonalscope, CopyKAT on
spots, CancerFinder via the h5), CalicoST's ``snp_dir``, Clonalscope's references and the truth;
:func:`write_scrna` writes the same for single cells. Observations and truth go to separate folders,
and nothing in ``outs/`` reveals the truth.
"""
import gzip
import json
import os
import warnings

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp

GENOME = "GRCh38"
# Dirichlet-multinomial concentration for Visium counts over the human genes, by indirect inference
# against 10x's V1_Breast_Cancer_Block_A_Section_1. `estimate_visium` (count model "dm") on its 3,798
# in-tissue spots gives 82,165 over the layer's 35,061 genes (validation/data/fit_human_visium_kappa.py;
# 89,197 over all 36,601 features), but that reads the section's tissue heterogeneity as count noise.
# The same estimator on a simulated paper-regime section returns 55,416 at an input of 82,165, 81,464 at
# 300,000 and 81,592 with no count noise at all (fit_human_visium_kappa_indirect.py): iscc's simulated
# tissue already accounts for the real dispersion, so the technical concentration is only bounded from
# below, and 300,000 is the smallest tested value that reproduces the real estimate within 1%. The
# default `VisiumBatchHyperParams.kappa` (50) is calibrated for the small growth-gene axis; spread over
# the human genes it piles each spot's counts onto a few dozen genes.
HUMAN_VISIUM_KAPPA = 300_000.0


def barcodes(n, seed=0, suffix="-1"):
    """``n`` distinct 16-base barcodes in the 10x style (``ACGT...-1``), fixed by ``seed``."""
    rng = np.random.default_rng([int(seed), 0xBC])
    out, seen = [], set()
    while len(out) < n:
        for row in rng.integers(0, 4, size=(n - len(out), 16)):
            bc = "".join("ACGT"[i] for i in row) + suffix
            if bc not in seen:
                seen.add(bc)
                out.append(bc)
    return out


def _sparse(counts):
    from .layer import as_csr
    return as_csr(counts).astype(np.int32)


def write_10x_mtx(path, counts, bcs, genes):
    """A 10x ``filtered_feature_bc_matrix`` directory: ``matrix.mtx.gz`` (features x barcodes),
    ``barcodes.tsv.gz``, ``features.tsv.gz`` (Ensembl ID, symbol, ``Gene Expression``)."""
    os.makedirs(path, exist_ok=True)
    X = _sparse(counts)
    with gzip.open(os.path.join(path, "matrix.mtx.gz"), "wb") as f:
        scipy.io.mmwrite(f, X.T.tocoo(), field="integer")
    with gzip.open(os.path.join(path, "barcodes.tsv.gz"), "wt") as f:
        f.write("\n".join(bcs) + "\n")
    with gzip.open(os.path.join(path, "features.tsv.gz"), "wt") as f:
        for gid, name in zip(genes["gene_id"], genes["gene_name"]):
            f.write(f"{gid}\t{name}\tGene Expression\n")


def write_10x_h5(path, counts, bcs, genes):
    """A Cell Ranger v3 ``filtered_feature_bc_matrix.h5`` (readable by ``scanpy.read_10x_h5``)."""
    import h5py
    X = _sparse(counts)                     # barcodes x features in CSR == features x barcodes in CSC
    with h5py.File(path, "w") as f:
        f.attrs["filetype"] = "matrix"
        f.attrs["version"] = 2
        m = f.create_group("matrix")
        m.create_dataset("barcodes", data=np.array(bcs, dtype="S"))
        m.create_dataset("data", data=X.data.astype(np.int32), compression="gzip")
        m.create_dataset("indices", data=X.indices.astype(np.int64), compression="gzip")
        m.create_dataset("indptr", data=X.indptr.astype(np.int64))
        m.create_dataset("shape", data=np.array([X.shape[1], X.shape[0]], dtype=np.int32))
        ft = m.create_group("features")
        ft.create_dataset("id", data=np.array(genes["gene_id"], dtype="S"))
        ft.create_dataset("name", data=np.array(genes["gene_name"], dtype="S"))
        ft.create_dataset("feature_type", data=np.array(["Gene Expression"] * X.shape[1], dtype="S"))
        ft.create_dataset("genome", data=np.array([GENOME] * X.shape[1], dtype="S"))
        ft.create_dataset("_all_tag_keys", data=np.array(["genome"], dtype="S"))


def write_snp_dir(path, allele, bcs):
    """CalicoST's ``snp_dir``: ``cell_snp_Aallele.npz``, ``cell_snp_Ballele.npz`` (barcodes x SNPs,
    haplotypes 1 and 2), ``unique_snp_ids.npy`` and ``barcodes.txt``; SNPs no barcode covers are
    dropped. Also writes ``snps.tsv`` (positions, alleles, gene, phase) for other tools."""
    os.makedirs(path, exist_ok=True)
    a = allele.drop_empty()
    sp.save_npz(os.path.join(path, "cell_snp_Aallele.npz"), a.A.tocsr())
    sp.save_npz(os.path.join(path, "cell_snp_Ballele.npz"), a.B.tocsr())
    np.save(os.path.join(path, "unique_snp_ids.npy"), np.array(a.snps["snp_id"].tolist(), dtype=object),
            allow_pickle=True)
    with open(os.path.join(path, "barcodes.txt"), "w") as f:
        f.write("\n".join(bcs) + "\n")
    a.snps.to_csv(os.path.join(path, "snps.tsv"), sep="\t", index=False)


def write_copykat(path, counts, bcs, genes):
    """CopyKAT's input: a dense genes x cells UMI table with Ensembl IDs as row names
    (``copykat(rawmat=read.delim(path, row.names=1), id.type="E", ...)``)."""
    X = _sparse(counts).T.toarray()
    df = pd.DataFrame(X, index=pd.Index(genes["gene_id"], name="gene_id"), columns=bcs)
    df.to_csv(path, sep="\t", compression="gzip" if str(path).endswith(".gz") else None)


def write_clonalscope_refs(path, layer):
    """Clonalscope's references for these genes: ``genes.bed`` (``chr start end geneID``),
    ``sizes.txt`` (chromosome sizes, no header) and ``chrarm.tsv`` (the modelled arms, the segments
    of its no-WGS mode)."""
    os.makedirs(path, exist_ok=True)
    g = layer.genes
    g[["chrom", "start", "end", "gene_id"]].rename(columns={"chrom": "chr", "gene_id": "geneID"}).to_csv(
        os.path.join(path, "genes.bed"), sep="\t", index=False)
    layer.reference.chrom_sizes.to_csv(os.path.join(path, "sizes.txt"), sep="\t", index=False, header=False)
    arms = layer.reference.arms
    pd.DataFrame({"chr": arms["chrom"].str.replace("chr", "", regex=False).astype(int),
                  "arm": arms["arm"], "start": arms["start"], "end": arms["end"]}).to_csv(
        os.path.join(path, "chrarm.tsv"), sep="\t", index=False)


def _tissue_positions(assay, bcs, scale=100.0):
    """Space Ranger ``tissue_positions.csv``: every spot with its array row/column and pixels."""
    crd = np.asarray(assay.spot_coords, dtype=float)
    pitch = float(assay.hypers.spot_pitch)
    row_dy = pitch * np.sqrt(3.0) / 2.0
    array_row = np.round((crd[:, 0] - crd[:, 0].min()) / row_dy).astype(int)
    array_col = np.round(2.0 * (crd[:, 1] - crd[:, 1].min()) / pitch).astype(int)
    return pd.DataFrame({
        "barcode": bcs,
        "in_tissue": (np.asarray(assay.obs["n_cells"]) > 0).astype(int),
        "array_row": array_row, "array_col": array_col,
        "pxl_row_in_fullres": np.round(crd[:, 0] * scale).astype(int),
        "pxl_col_in_fullres": np.round(crd[:, 1] * scale).astype(int),
    })


def write_visium(out_dir, assay, layer, allele=None, tumor=None, seed=0):
    """Write a human-gene Visium assay for the copy-number tools, with its truth.

    Layout::

        out_dir/outs/filtered_feature_bc_matrix.h5          (in-tissue spots)
        out_dir/outs/filtered_feature_bc_matrix/            (the same as mtx)
        out_dir/outs/spatial/tissue_positions.csv           (all spots, Space Ranger 2 header)
        out_dir/outs/spatial/scalefactors_json.json
        out_dir/snp_dir/                                    (CalicoST; needs allele)
        out_dir/copykat_counts.tsv.gz                       (genes x in-tissue spots)
        out_dir/clonalscope/                                (genes.bed, sizes.txt, chrarm.tsv)
        out_dir/truth/spots.tsv, cn_segments.tsv, germline_snps.tsv

    Parameters
    ----------
    out_dir : str
    assay : Visium
        Run with ``gene_lift=layer.lift`` and ``kappa=HUMAN_VISIUM_KAPPA`` (warns otherwise).
    layer : HumanGenomeLayer
    allele : AlleleCounts, optional
        From :meth:`HumanGenomeLayer.visium_allele_counts`; without it no ``snp_dir`` is written.
    tumor : GenotypeTumor, optional
        Needed for the copy-number truth.
    seed : int, default 0
        Seed of the spot barcodes.
    """
    if assay.count_model == "dm" and float(assay.hypers.kappa) < 1000:
        warnings.warn(f"Visium kappa {assay.hypers.kappa:g} is the growth-gene scale; over human genes "
                      f"use Visium(kappa=HUMAN_VISIUM_KAPPA) or counts concentrate on a few genes",
                      stacklevel=2)
    outs = os.path.join(out_dir, "outs")
    os.makedirs(os.path.join(outs, "spatial"), exist_ok=True)
    spots = list(assay.spot_counts.index)
    bcs = barcodes(len(spots), seed=seed)
    in_tissue = np.asarray(assay.obs["n_cells"]) > 0
    keep = np.flatnonzero(in_tissue)
    counts = _sparse(assay.spot_counts)[keep]
    kept_bcs = [bcs[i] for i in keep]
    write_10x_h5(os.path.join(outs, "filtered_feature_bc_matrix.h5"), counts, kept_bcs, layer.genes)
    write_10x_mtx(os.path.join(outs, "filtered_feature_bc_matrix"), counts, kept_bcs, layer.genes)
    _tissue_positions(assay, bcs).to_csv(os.path.join(outs, "spatial", "tissue_positions.csv"), index=False)
    diam = 2.0 * float(assay.hypers.spot_radius) * 100.0
    with open(os.path.join(outs, "spatial", "scalefactors_json.json"), "w") as f:
        json.dump({"spot_diameter_fullres": diam, "fiducial_diameter_fullres": diam,
                   "tissue_hires_scalef": 1.0, "tissue_lowres_scalef": 1.0}, f)
    if allele is not None:
        sub = type(allele)(allele.A[keep].tocsr(), allele.B[keep].tocsr(), allele.snps, kept_bcs)
        write_snp_dir(os.path.join(out_dir, "snp_dir"), sub, kept_bcs)
    write_copykat(os.path.join(out_dir, "copykat_counts.tsv.gz"), counts, kept_bcs, layer.genes)
    write_clonalscope_refs(os.path.join(out_dir, "clonalscope"), layer)
    _write_spot_truth(os.path.join(out_dir, "truth"), assay, layer, bcs, tumor)


def write_scrna(out_dir, assay, layer, cell_data, allele=None, tumor=None, seed=0):
    """Write a human-gene scRNA assay for the copy-number tools, with its truth.

    Layout::

        out_dir/outs/filtered_feature_bc_matrix.h5 and filtered_feature_bc_matrix/
        out_dir/snp_dir/                         (per-cell haplotype counts; needs allele)
        out_dir/copykat_counts.tsv.gz            (genes x cells)
        out_dir/clonalscope/                     (genes.bed, sizes.txt, chrarm.tsv, celltype.tsv)
        out_dir/truth/cells.tsv, cn_segments.tsv, germline_snps.tsv

    ``clonalscope/celltype.tsv`` labels the true normal cells ``normal`` (the baseline Clonalscope
    normalises against) and the rest ``tumor``: it comes from the truth, as a cell-type annotation
    would.

    Parameters
    ----------
    out_dir : str
    assay : scRNA
        Run with ``gene_lift=layer.lift``.
    layer : HumanGenomeLayer
    cell_data : dict
        The sample the assay drew from.
    allele : AlleleCounts, optional
        From :meth:`HumanGenomeLayer.scrna_allele_counts`.
    tumor : GenotypeTumor, optional
        Needed for the truth.
    seed : int, default 0
        Seed of the cell barcodes.
    """
    outs = os.path.join(out_dir, "outs")
    os.makedirs(outs, exist_ok=True)
    cells = list(assay.observed_counts.index)
    bcs = barcodes(len(cells), seed=seed)
    counts = _sparse(getattr(assay, "counts_csr", None) if getattr(assay, "counts_csr", None) is not None
                     else assay.observed_counts)
    write_10x_h5(os.path.join(outs, "filtered_feature_bc_matrix.h5"), counts, bcs, layer.genes)
    write_10x_mtx(os.path.join(outs, "filtered_feature_bc_matrix"), counts, bcs, layer.genes)
    if allele is not None:
        write_snp_dir(os.path.join(out_dir, "snp_dir"), type(allele)(allele.A, allele.B, allele.snps, bcs), bcs)
    write_copykat(os.path.join(out_dir, "copykat_counts.tsv.gz"), counts, bcs, layer.genes)
    write_clonalscope_refs(os.path.join(out_dir, "clonalscope"), layer)
    truth_dir = os.path.join(out_dir, "truth")
    os.makedirs(truth_dir, exist_ok=True)
    gids = layer._gids(cell_data, cells)
    tumour = (layer.is_tumour(gids, tumor) if tumor is not None
              else np.array([str(g) not in ("epithelial", "stromal", "immune", "host") for g in gids]))
    pd.DataFrame({"barcode": bcs, "cell": cells, "genotype": [str(g) for g in gids],
                  "is_tumour": tumour, "is_doublet": assay.obs["is_doublet"].to_numpy()}).to_csv(
        os.path.join(truth_dir, "cells.tsv"), sep="\t", index=False)
    pd.DataFrame({"barcode": bcs, "celltype": np.where(tumour, "tumor", "normal")}).to_csv(
        os.path.join(out_dir, "clonalscope", "celltype.tsv"), sep="\t", index=False)
    if tumor is not None:
        layer.cn_segments(gids, tumor).to_csv(os.path.join(truth_dir, "cn_segments.tsv"), sep="\t", index=False)
    if layer.germline is not None:
        layer.germline.to_csv(os.path.join(truth_dir, "germline_snps.tsv"), sep="\t", index=False)


def _write_spot_truth(path, assay, layer, bcs, tumor):
    os.makedirs(path, exist_ok=True)
    placed = assay._grid["placed"]
    gids_all = layer._gids(placed)
    tumour_all = (layer.is_tumour(gids_all, tumor) if tumor is not None
                  else np.array([str(g) not in ("epithelial", "stromal", "immune", "host") for g in gids_all]))
    rows = []
    for s, (bc, m) in enumerate(zip(bcs, assay._grid["members"])):
        n = int(m.size)
        n_t = int(tumour_all[m].sum()) if n else 0
        rows.append((bc, assay.spot_names[s], n, n_t, (n_t / n) if n else 0.0,
                     assay.obs["clone"].iat[s], ",".join(map(str, gids_all[m])) if n else ""))
    pd.DataFrame(rows, columns=["barcode", "spot", "n_cells", "n_tumour", "tumour_fraction",
                                "dominant_genotype", "genotypes"]).to_csv(
        os.path.join(path, "spots.tsv"), sep="\t", index=False)
    if tumor is not None:
        layer.cn_segments(list(dict.fromkeys(gids_all)), tumor).to_csv(
            os.path.join(path, "cn_segments.tsv"), sep="\t", index=False)
    if layer.germline is not None:
        layer.germline.to_csv(os.path.join(path, "germline_snps.tsv"), sep="\t", index=False)
