"""Build the per-cell-type baselines and the Hallmark programmes of the human genome layer (L3).

Sources:
  1. **Human Breast Cell Atlas** (Kumar et al., Nature 2023, doi:10.1038/s41586-023-06252-9), the
     "scRNA-seq data - all cells" dataset on CELLxGENE (714,331 cells from 126 women, normal breast;
     dataset 842c6f5d-4a94-4eef-8510-8c792d1124bc, version 296226b4-4916-44a3-97cc-60bdfb95f2e1).
     CELLxGENE hosts its datasets under CC BY 4.0. Raw counts are in ``raw.X``.
     https://datasets.cellxgene.cziscience.com/296226b4-4916-44a3-97cc-60bdfb95f2e1.h5ad
  2. **MSigDB Hallmark gene sets**, v2025.1 (Liberzon et al., Cell Systems 2015), CC BY 4.0 (MSigDB
     v2022.1 and later). https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2025.1.Hs/h.all.v2025.1.Hs.symbols.gmt

A cell type's baseline is the mean over its cells of each gene's share of the cell's counts (10x 3' v3
cells only), so it is the expected composition of one cell; ``rna_content`` is the type's mean
counts per cell relative to the luminal epithelium. iscc's types map onto the atlas as:

  * ``epithelial`` (and so the cancer, through its cell of origin): luminal epithelium, the authors'
    ``LummHR-*`` (hormone-responsive) and ``Lumsec-*`` (secretory) cells. Basal cells are left out:
    iscc's ductal field has one epithelial type, lining the duct.
  * ``stromal``: fibroblasts.
  * ``immune``: T and NK cells, the cells that kill tumour cells in iscc's immune model.

Outputs (under ``src/iscc/data/human/reference/``):
  * ``hbca_type_baselines.tsv.gz`` — gene_id and one column per iscc type (GRCh38-2020-A genes; a
    gene the atlas does not measure is 0).
  * ``hbca_types.tsv`` — per type: atlas labels, cells, donors, mean counts, rna_content.
  * ``hallmark_programs.tsv`` — programme, hallmark set, gene_id, gene_name: the Hallmark sets that
    seed iscc's named programmes, matched to GRCh38-2020-A symbols.

Usage::

    python validation/data/build_atlas_baselines.py --h5ad <atlas.h5ad> --gmt <h.all...gmt>
"""
import argparse
import gzip
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "src", "iscc", "data", "human", "reference")

TYPES = {
    "epithelial": ("author_cell_type", ("LummHR", "Lumsec")),
    "stromal": ("cell_type", ("fibroblast",)),
    "immune": ("cell_type", ("T cell", "natural killer cell", "mature NK T cell")),
}
# iscc's named programmes and the Hallmark sets that seed them. `drug_resistance` has no Hallmark
# counterpart; xenobiotic metabolism (drug-metabolising enzymes and transporters) is the closest, and
# is a proxy. `immune_evasion` and the background programmes have none and stay on random genes.
PROGRAM_SETS = {
    "proliferation": ["HALLMARK_E2F_TARGETS", "HALLMARK_G2M_CHECKPOINT"],
    "emt": ["HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION"],
    "hypoxia": ["HALLMARK_HYPOXIA"],
    "drug_resistance": ["HALLMARK_XENOBIOTIC_METABOLISM"],
}


def _select(obs, column, patterns):
    lab = obs[column].astype(str)
    if column == "author_cell_type":
        return lab.str.startswith(patterns)
    return lab.apply(lambda s: any(p in s for p in patterns))


def type_profiles(h5ad, genes, chunk=20000):
    import anndata as ad
    a = ad.read_h5ad(h5ad, backed="r")
    obs = a.obs
    v3 = obs["assay"].astype(str) == "10x 3' v3"
    gene_pos = pd.Index(a.raw.var_names if a.raw is not None else a.var_names)
    col = gene_pos.get_indexer(genes["gene_id"])          # -1: not measured by the atlas
    prof, rows = {}, []
    for t, (column, patterns) in TYPES.items():
        sel = np.flatnonzero(_select(obs, column, patterns).to_numpy() & v3.to_numpy())
        acc = np.zeros(len(gene_pos))
        totals = []
        for lo in range(0, len(sel), chunk):
            X = (a.raw.X if a.raw is not None else a.X)[sel[lo:lo + chunk]]
            tot = np.asarray(X.sum(axis=1)).ravel()
            keep = tot > 0
            X = X[keep].multiply(1.0 / tot[keep][:, None]).tocsr()
            acc += np.asarray(X.sum(axis=0)).ravel()
            totals.append(tot[keep])
        totals = np.concatenate(totals)
        mean = acc / len(totals)
        p = np.where(col >= 0, mean[np.maximum(col, 0)], 0.0)
        prof[t] = p / p.sum()
        labels = sorted(set(obs[column].astype(str).to_numpy()[sel]))
        rows.append((t, column, ";".join(labels), len(totals), obs["donor_id"].iloc[sel].nunique(),
                     float(totals.mean())))
    info = pd.DataFrame(rows, columns=["type", "atlas_column", "atlas_labels", "n_cells", "n_donors",
                                       "mean_counts"])
    info["rna_content"] = info["mean_counts"] / float(info.loc[info.type == "epithelial", "mean_counts"].iat[0])
    n_missing = int((col < 0).sum())
    return pd.DataFrame(prof), info, n_missing


def hallmark_programs(gmt, genes):
    sets = {}
    with open(gmt) as f:
        for ln in f:
            name, _url, *syms = ln.rstrip("\n").split("\t")
            sets[name] = syms
    first = genes.drop_duplicates("gene_name").set_index("gene_name")["gene_id"]
    rows = []
    for prog, names in PROGRAM_SETS.items():
        seen = set()
        for s in names:
            for sym in sets[s]:
                if sym in first.index and sym not in seen:
                    seen.add(sym)
                    rows.append((prog, s, first[sym], sym))
    return pd.DataFrame(rows, columns=["programme", "hallmark_set", "gene_id", "gene_name"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--h5ad", required=True)
    ap.add_argument("--gmt", required=True)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    with gzip.open(os.path.join(args.out, "grch38_2020A_genes.tsv.gz"), "rt") as f:
        genes = pd.read_csv(f, sep="\t")
    prof, info, n_missing = type_profiles(args.h5ad, genes)
    out = pd.concat([genes[["gene_id"]], prof], axis=1)
    with gzip.open(os.path.join(args.out, "hbca_type_baselines.tsv.gz"), "wt", compresslevel=9) as f:
        out.to_csv(f, sep="\t", index=False, float_format="%.6g")
    info.to_csv(os.path.join(args.out, "hbca_types.tsv"), sep="\t", index=False)
    print(info.to_string())
    print(f"GRCh38-2020-A genes the atlas does not measure: {n_missing} of {len(genes)}")
    hp = hallmark_programs(args.gmt, genes)
    hp.to_csv(os.path.join(args.out, "hallmark_programs.tsv"), sep="\t", index=False)
    print(hp.groupby("programme").size().to_string())


if __name__ == "__main__":
    main()
