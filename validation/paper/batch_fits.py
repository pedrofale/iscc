"""Fig 5's scRNA batch model: each batch's technical parameters fitted to real data.

The design, batches from named technical sources and a panel validating them, was registered before
any run. This script makes
every fit ONCE and writes them to one summary file, before any statistic of the validation panel
(``batch_validation.py``) or any integration is computed::

    PYTHONPATH=src python validation/paper/batch_fits.py fit --out $D/batchval/fits.json \\
        --wu-tar $D/realdata/GSE176078/GSE176078_Wu_etal_2021_BRCA_scRNASeq.tar.gz \\
        --ding-dir $D/realdata/GSE132044 --pbmcsca-meta $D/realdata/pbmcsca/pbmcsca_meta.tsv.gz \\
        --orthology-dir $D/realdata/orthology --work $SNIC_TMP/fits
    PYTHONPATH=src python validation/paper/batch_fits.py set-ambient-range --fits $D/batchval/fits.json \\
        --low 0.03 --high 0.11 --source "..." --registered "<where the range was registered>"

The archives are extracted into ``--work`` (node-local scratch) and deleted with it. The pbmcsca
metadata TSV is made by ``tools/pbmcsca_meta.R``.

**Wu et al. 2021** (Nat Genet 53:1334-1347, doi:10.1038/s41588-021-00911-1; GEO GSE176078). Their
Methods (read on PMC9044823, 2026-11-16): "Chromium Single-Cell v2 3' and 5' Chemistry", "5,000 to
7,000 cells were targeted per well", Cell Ranger Single Cell v2.0 on GRCh38, EmptyDrops (DropletUtils
1.2.2) "with additional cutoffs for cells with a gene and unique molecular identifier (UMIs) count
greater than 200 and 250, respectively, and a mitochondrial percentage less than 20%". Each tumour
(``orig.ident``) is a batch:

* ``estimate_rna`` (``protocol="10x"``, ``fit_dropout=True``) on the tumour's T cells only
  (``celltype_major == "T-cells"``), so cell-type composition does not enter the technical fit:
  ``mu_lib``, ``sigma_lib``, ``dispersion``, ``dropout_mid``, ``dropout_shape``. A tumour whose
  fitted midpoint is 0 has no dropout. Genes with no count in those cells are dropped first; they
  change none of the estimates (library sizes are row sums; the dispersion and dropout fits use
  expressed genes only).
* Dissociation stress: the mean over the tumour's T cells of the mean log-normalised expression
  (``log1p(1e4 * count / library)``) of the dissociation-affected genes of van den Brink et al. 2017
  (Nat Methods 14:935-936, doi:10.1038/nmeth.4437), relative to the median over tumours, converted to
  ``iscc``'s multiplicative stress ``max(exp(score - median) - 1, 0)``. The 140 genes are the mouse
  genes listed in the paper's Supplementary Methods ("In silico purification", the R code
  ``Selection <- Data[c(...)]``, which it says holds the genes of Supplementary Table 5;
  Supplementary Text and Figures PDF, MOESM1, read 2026-11-16). They are mapped to human genes
  through MGI's homology report (HOM_MouseHumanSequence.rpt) and, for mouse symbols MGI no longer
  uses and for human genes the data name by an older symbol, through HGNC's previous and alias
  symbols (hgnc_complete_set); both files downloaded 2026-11-16 and kept beside the data.
* Ambient fraction, SoupX's principle (Young & Behjati 2020, GigaScience 9(12):giaa151,
  doi:10.1093/gigascience/giaa151): genes specific to one other cell type, counted in the T cells,
  are soup (:func:`ambient_estimate`, fixed before the fit). Each tumour by itself; a tumour whose
  estimate is not identifiable takes a value drawn per seed (per batch in the cohort) from the range
  in fits.json's ``ambient_range``, set by ``set-ambient-range``. The registered range
  is the soup share rho uniform on [0.03, 0.11], from Janssen et al. 2023 (Genome Biol 24:140,
  doi:10.1186/s13059-023-02978-x; Results, "Background noise fractions differ between replicates and
  cells", Fig. 2A, PMC10278251): in their scRNA-seq of dissociated tissue, replicate 3's cells mostly
  below 3% background and replicate 2's median around 11%. As first registered,
  the estimates are used only if identifiable in at least half the tumours; otherwise every tumour
  takes the range.
* Doublet rate: 10x's multiplet rate at the tumour's number of cells, from the Chromium Single Cell
  3' Reagent Kits v2 User Guide (CG00052 Rev B, p. 6: ~0.4% at ~500 cells recovered ... ~7.6% at
  ~10,000; read 2026-11-16), interpolated linearly in the table and extrapolated above 10,000 cells at
  7.6% per 10,000 (:data:`MULTIPLET_TABLE`). The same table is applied to every tumour, although Wu
  et al. used both the 3' and the 5' v2 kits; the 5' kit's own guide was not checked. The cells
  counted are the published, filtered ones, fewer than the barcodes the channel recovered.
* Recovery per patient: each type's default recovery on the logit scale plus a normal draw of sd 0.5
  (:func:`recovery_draws`), for the cohort's 50 patients.

**Ding et al. 2020** (Nat Biotechnol 38:737-746, doi:10.1038/s41587-020-0465-8; GEO GSE132044), PBMC1
in two 10x Chromium v2 runs, A (5,172 cells) and B (3,057 cells), every cell of both. Cell types
(:func:`transfer_labels`): the authors' annotation (``CellType`` of the Single Cell Portal study
SCP424's metadata, as packaged in SeuratData's pbmcsca 3.0.0, CC BY 4.0) where it exists; every other
cell (and any the authors marked "Unassigned", as registered) takes the majority label
of its 15 nearest labelled
cells on 50 principal components of the pooled counts, normalised to the median library size and
log-transformed; the leave-one-out accuracy of that rule on the labelled cells is recorded. The fits
per run "by type" (:func:`ding_fits`, :func:`combine_by_type`): ``estimate_rna`` (fit_dropout on)
within each cell type, and the ambient attempt in each type's cells against the other types'
markers; one vector per run, the cell-weighted mean over types of ``mu_lib``, ``sigma_lib``,
dispersion and the ambient fraction (a run's ambient fraction counts only when the types whose
estimate is identifiable hold more than half its cells, as registered); dropout only
when types holding more than half the run's cells have a fitted midpoint above 0, then the
cell-weighted mean of those types' midpoints and shapes.
No dissociation (blood).
"""
import argparse
import gzip
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for p in (os.path.join(REPO, "src"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402

# ------------------------------------------------------------------ sources (checked 2026-11-16/17)
SOURCES = {
    "wu2021": dict(citation="Wu SZ et al. A single-cell and spatially resolved atlas of human breast "
                            "cancers. Nat Genet 53, 1334-1347 (2021)",
                   doi="10.1038/s41588-021-00911-1", geo="GSE176078", pmc="PMC9044823",
                   file="GSE176078_Wu_etal_2021_BRCA_scRNASeq.tar.gz (558,829,202 bytes)",
                   methods="Methods (PMC9044823, read 2026-11-16): Chromium Single-Cell v2 3' and 5' chemistry; "
                           "5,000 to 7,000 cells targeted per well; Cell Ranger Single Cell v2.0 on GRCh38; "
                           "EmptyDrops (DropletUtils 1.2.2) with cells of > 200 genes, > 250 UMIs and < 20% "
                           "mitochondrial"),
    "ding2020": dict(citation="Ding J et al. Systematic comparison of single-cell and single-nucleus "
                              "RNA-sequencing methods. Nat Biotechnol 38, 737-746 (2020)",
                     doi="10.1038/s41587-020-0465-8", geo="GSE132044",
                     files=["GSE132044_pbmc_hg38_cell.tsv.gz", "GSE132044_pbmc_hg38_gene.tsv.gz",
                            "GSE132044_pbmc_hg38_count_matrix.mtx.gz (127,509,347 bytes)"]),
    "ding2020_labels": dict(source="SeuratData pbmcsca.SeuratData 3.0.0 (CC BY 4.0; md5 "
                                   "6eb6dad07367ea8861027200196b096c), metadata column CellType; its "
                                   "documentation names SCP424 as the source; exported by tools/pbmcsca_meta.R"),
    "vandenbrink2017": dict(citation="van den Brink SC et al. Single-cell sequencing reveals "
                                     "dissociation-induced gene expression in tissue subpopulations. "
                                     "Nat Methods 14, 935-936 (2017)",
                            doi="10.1038/nmeth.4437",
                            where="Supplementary Methods, 'In silico purification': the R code listing the "
                                  "140 dissociation-affected genes (mouse), which it says are those of "
                                  "Supplementary Table 5; MOESM1 PDF"),
    "soupx": dict(citation="Young MD, Behjati S. SoupX removes ambient RNA contamination from "
                           "droplet-based single-cell RNA sequencing data. GigaScience 9(12), giaa151 (2020)",
                  doi="10.1093/gigascience/giaa151"),
    "multiplet": dict(source="10x Genomics, Chromium Single Cell 3' Reagent Kits v2 User Guide, CG00052 "
                             "Rev B, p. 6 (multiplet rate by cells recovered), read 2026-11-16; applied to "
                             "Wu et al.'s 3' and 5' runs alike (the 5' kit's guide was not checked)"),
    "ambient_range": dict(citation="Janssen P et al. The effect of background noise and its removal on the "
                                   "analysis of single-cell expression data. Genome Biol 24, 140 (2023)",
                          doi="10.1186/s13059-023-02978-x", pmc="PMC10278251",
                          where="Results, 'Background noise fractions differ between replicates and cells' "
                                "(Fig. 2A): scRNA-seq replicate 3, most cells below 3% background; replicate 2, "
                                "median around 11%", registered="vault bc210df", rho=[0.03, 0.11]),
    "mgi_homology": dict(source="MGI HOM_MouseHumanSequence.rpt, downloaded 2026-11-16"),
    "hgnc": dict(source="HGNC hgnc_complete_set.txt, downloaded 2026-11-16"),
}

# van den Brink et al. 2017, Supplementary Methods: the 140 dissociation-affected genes (mouse).
VDB_MOUSE_GENES = (
    "Actg1 Ankrd1 Arid5a Atf3 Atf4 Bag3 Bhlhe40 Brd2 Btg1 Btg2 Ccnl1 Ccrn4l Cebpb Cebpd Cebpg "
    "Csrnp1 Cxcl1 Cyr61 Dcn Ddx3x Ddx5 Des Dnaja1 Dnajb1 Dnajb4 Dusp1 Dusp8 Egr1 Egr2 Eif1 Eif5 Erf "
    "Errfi1 Fam132b Fos Fosb Fosl2 Gadd45a Gcc1 Gem H3f3b Hipk3 Hsp90aa1 Hsp90ab1 Hspa1a Hspa1b "
    "Hspa5 Hspa8 Hspb1 Hsph1 Id3 Idi1 Ier2 Ier3 Ifrd1 Il6 Irf1 Irf8 Itpkc Jun Junb Jund Klf2 Klf4 "
    "Klf6 Klf9 Litaf Lmna Maff Mafk Mcl1 Midn Mir22hg Mt1 Mt2 Myadm Myc Myd88 Nckap5l Ncoa7 Nfkbia "
    "Nfkbiz Nop58 Nppc Nr4a1 Odc1 Osgin1 Oxnad1 Pcf11 Pde4b Per1 Phlda1 Pnp Pnrc1 Ppp1cc Ppp1r15a "
    "Pxdc1 Rap1b Rassf1 Rhob Rhoh Ripk1 Sat1 Sbno2 Sdc4 Serpine1 Skil Slc10a6 Slc38a2 Slc41a1 "
    "Socs3 Sqstm1 Srf Srsf5 Srsf7 Stat3 Tagln2 Tiparp Tnfaip3 Tnfaip6 Tpm3 Tppp3 Tra2a Tra2b Trib1 "
    "Tubb4b Tubb6 Ubc Usp2 Wac Zc3h12a Zfand5 Zfp36 Zfp36l1 Zfp36l2 Zyx Gadd45g Hspe1 Ier5 Kcne4"
).split()
assert len(VDB_MOUSE_GENES) == 140 and len(set(VDB_MOUSE_GENES)) == 140

# 10x multiplet rate (%) by cells recovered: CG00052 Rev B p. 6 (3' v2).
MULTIPLET_TABLE = ((500, 0.4), (1000, 0.8), (2000, 1.6), (3000, 2.3), (4000, 3.1), (5000, 3.9),
                   (6000, 4.6), (7000, 5.4), (8000, 6.1), (9000, 6.9), (10000, 7.6))

WU_T = "T-cells"
DING_RUNS = {"A": "PBMC1.10x-Chromium-v2-A", "B": "PBMC1.10x-Chromium-v2-B"}
DING_LABEL_METHOD = {"A": "10x Chromium (v2) A", "B": "10x Chromium (v2) B"}
UNASSIGNED = "Unassigned"
LABEL_TRANSFER = dict(k=15, n_pcs=50, seed=0)
MIN_TYPE_CELLS_FIT = 10            # a type is fitted within a run only with this many cells

# ---- the ambient attempt (SoupX's principle), fixed before any fit ----
AMBIENT = dict(
    marker_min_mean_cp10k=5.0,       # a marker is well expressed in its own type (so it is in the soup)
    marker_specificity=20.0,         # ... and its mean there is >= 20x its mean in every other type
    min_source_cells=50,             # a type's set is used in a batch only with >= 50 cells of it there
    min_t_cells=50,                  # ... >= 50 non-expressing cells kept
    min_counts=20,                   # ... and >= 20 marker counts observed in them
    exclude_quantile=0.999,          # SoupX's non-expressing cells: drop a cell whose marker counts
                                     # exceed this Poisson quantile at 100% contamination
    agreement=2.0,                   # identifiable: >= 2 sets, largest / smallest estimate <= 2
    min_sets=2,
)
AMBIENT_DRAW = "uniform on the soup share rho in [low, high], which is iscc's ambient_frac (a share of the library)"
RECOVERY_SD = 0.5
RECOVERY_SEED = 20261116
N_PATIENTS = 50


# ------------------------------------------------------------------ readers
def read_mtx(path):
    """A MatrixMarket coordinate file (plain or .gz) as a CSR of its stated shape, fast."""
    op = gzip.open if path.endswith(".gz") else open
    skip = 0
    with op(path, "rt") as f:
        for line in f:
            skip += 1
            if not line.startswith("%"):
                nr, nc, nnz = (int(x) for x in line.split())
                break
    df = pd.read_csv(path, sep=r"\s+", header=None, skiprows=skip,
                     dtype={0: np.int32, 1: np.int32, 2: np.float64}, engine="c")
    if len(df) != nnz:
        raise ValueError(f"{path}: header says {nnz} entries, read {len(df)}")
    return sp.csr_matrix((df[2].to_numpy(), (df[0].to_numpy() - 1, df[1].to_numpy() - 1)), shape=(nr, nc))


def multiplet_rate(n_cells):
    """10x's multiplet rate (a fraction) at ``n_cells`` cells recovered (:data:`MULTIPLET_TABLE`)."""
    x = np.array([c for c, _ in MULTIPLET_TABLE], float)
    y = np.array([r for _, r in MULTIPLET_TABLE], float) / 100
    n = float(n_cells)
    if n <= x[-1]:
        return float(np.interp(n, np.r_[0.0, x], np.r_[0.0, y]))
    return float(y[-1] * n / x[-1])


def load_wu(tar, work):
    """Extract Wu et al.'s archive into ``work``; return (counts cells x genes CSR, genes, meta)."""
    subprocess.run(["tar", "xzf", tar, "-C", work], check=True)
    d = os.path.join(work, "Wu_etal_2021_BRCA_scRNASeq")
    genes = pd.read_csv(os.path.join(d, "count_matrix_genes.tsv"), sep="\t", header=None)[0].astype(str).to_numpy()
    bcs = pd.read_csv(os.path.join(d, "count_matrix_barcodes.tsv"), sep="\t", header=None)[0].astype(str).to_numpy()
    meta = pd.read_csv(os.path.join(d, "metadata.csv"), index_col=0)
    X = read_mtx(os.path.join(d, "count_matrix_sparse.mtx"))
    if X.shape == (len(genes), len(bcs)):
        X = X.T.tocsr()
    if X.shape != (len(bcs), len(genes)):
        raise ValueError(f"Wu matrix {X.shape} matches neither genes {len(genes)} nor barcodes {len(bcs)}")
    meta = meta.reindex(bcs)
    if meta.isna().all(axis=1).any():
        raise ValueError("Wu metadata misses some barcodes of the matrix")
    missing = [c for c in ("orig.ident", "celltype_major") if c not in meta.columns]
    if missing:
        raise ValueError(f"Wu metadata lacks {missing} (columns: {list(meta.columns)}); stopping, not adapting")
    if WU_T not in set(meta["celltype_major"].astype(str)):
        raise ValueError(f"Wu metadata has no celltype_major {WU_T!r} (values: "
                         f"{sorted(set(meta['celltype_major'].astype(str)))}); stopping, not adapting")
    shutil.rmtree(d)
    return X, genes, meta


def load_ding(ding_dir, labels_path):
    """Ding et al.'s PBMC1 10x v2 runs A and B, every cell: (counts cells x genes CSR, genes, obs with
    the run, the barcode and the authors' label where pbmcsca has one, else NaN)."""
    cells = pd.read_csv(os.path.join(ding_dir, "GSE132044_pbmc_hg38_cell.tsv.gz"), header=None)[0].astype(str).to_numpy()
    genes = pd.read_csv(os.path.join(ding_dir, "GSE132044_pbmc_hg38_gene.tsv.gz"), header=None)[0].astype(str).to_numpy()
    X = read_mtx(os.path.join(ding_dir, "GSE132044_pbmc_hg38_count_matrix.mtx.gz"))
    if X.shape == (len(genes), len(cells)):
        X = X.T.tocsr()
    if X.shape != (len(cells), len(genes)):
        raise ValueError(f"Ding matrix {X.shape} matches neither genes {len(genes)} nor cells {len(cells)}")
    run = pd.Series(cells).str.rsplit(".", n=1).str[0].to_numpy()
    keep = np.isin(run, list(DING_RUNS.values()))
    X, cells, run = X[np.flatnonzero(keep)].tocsr(), cells[keep], run[keep]
    inv = {v: k for k, v in DING_RUNS.items()}
    obs = pd.DataFrame({"run": [inv[r] for r in run],
                        "barcode": pd.Series(cells).str.rsplit(".", n=1).str[1].to_numpy()}, index=cells)
    lab = pd.read_csv(labels_path, sep="\t", low_memory=False)
    missing = [c for c in ("cell", "Method", "CellType") if c not in lab.columns]
    if missing:
        raise ValueError(f"pbmcsca metadata lacks {missing} (columns: {list(lab.columns)}); stopping, not adapting")
    obs["author_label"] = _ding_labels(obs, lab)
    for r in DING_RUNS:
        n = int(obs.loc[obs["run"] == r, "author_label"].notna().sum())
        if n == 0:
            raise ValueError(f"no pbmcsca label matched run {r}'s cells; stopping, not adapting")
    return X, genes, obs


def _ding_labels(obs, lab):
    """Match the pbmcsca metadata to the GEO cells by run and barcode (NaN where it has none)."""
    out = pd.Series(index=obs.index, dtype=object)
    for r, method in DING_LABEL_METHOD.items():
        m = lab[lab["Method"] == method]
        if "Experiment" in m:
            m = m[m["Experiment"].astype(str).str.lower().str.contains("pbmc1")]
        if m.empty:
            raise ValueError(f"pbmcsca has no PBMC1 cells of {method!r}")
        bc = m["cell"].astype(str).str.extract(r"([ACGTN]{12,})")[0]
        if bc.isna().any():
            raise ValueError(f"pbmcsca cell names of {method!r} carry no barcode, e.g. {m['cell'].iloc[0]}")
        lookup = pd.Series(m["CellType"].astype(str).to_numpy(), index=bc.to_numpy())
        if not lookup.index.is_unique:
            raise ValueError(f"pbmcsca barcodes repeat within {method!r}")
        sel = (obs["run"] == r).to_numpy()
        out[sel] = lookup.reindex(obs.loc[sel, "barcode"]).to_numpy()
    return out


# ------------------------------------------------------------------ the embedding and the labels
def lognorm_pca(X, n_pcs=50, seed=0):
    """Principal components of the counts normalised to the median library size and log1p (the
    registration's normalisation, also S4's): scanpy's PCA (arpack, centred, not scaled)."""
    import anndata as ad
    import scanpy as sc
    X = sp.csr_matrix(X, dtype=np.float64)
    lib = np.asarray(X.sum(1)).ravel()
    med = float(np.median(lib))
    norm = sp.diags(med / np.maximum(lib, 1.0)) @ X
    norm.data = np.log1p(norm.data)
    a = ad.AnnData(X=norm.astype(np.float32))
    n = min(n_pcs, min(a.shape) - 1)
    sc.pp.pca(a, n_comps=n, zero_center=True, svd_solver="arpack", random_state=seed)
    return a.obsm["X_pca"].astype(np.float64), med


def transfer_labels(X, author, k=LABEL_TRANSFER["k"], n_pcs=LABEL_TRANSFER["n_pcs"], seed=LABEL_TRANSFER["seed"]):
    """Every cell's type: the authors' label where it exists, else the majority label of its ``k``
    nearest labelled cells on ``n_pcs`` PCs of the pooled counts (ties: the label of the nearest
    among the tied). "Unassigned" counts as no label. Also the rule's leave-one-out accuracy on the
    labelled cells (each labelled cell's ``k`` nearest other labelled cells)."""
    from sklearn.neighbors import NearestNeighbors
    author = pd.Series(author, dtype=object)
    lab = author.notna().to_numpy() & (author.astype(str) != UNASSIGNED).to_numpy()
    pcs, _ = lognorm_pca(X, n_pcs, seed)
    ref, ref_lab = pcs[lab], author.to_numpy()[lab].astype(str)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(ref)

    def vote(rows):
        out = []
        for r in rows:
            labs = ref_lab[r]
            vals, cnt = np.unique(labs, return_counts=True)
            top = vals[cnt == cnt.max()]
            out.append(top[0] if len(top) == 1 else next(x for x in labs if x in set(top)))
        return np.array(out, dtype=object)

    final = author.astype(object).copy()
    q = np.flatnonzero(~lab)
    if len(q):
        _, idx = nn.kneighbors(pcs[q], n_neighbors=k)
        final.iloc[q] = vote(idx)
    _, idx = nn.kneighbors(ref, n_neighbors=k + 1)
    loo = []
    for i, row in enumerate(idx):
        loo.append([j for j in row if j != i][:k])
    pred = vote(np.array(loo))
    acc = float((pred == ref_lab).mean())
    per_type = {t: float((pred[ref_lab == t] == t).mean()) for t in sorted(set(ref_lab))}
    rec = dict(rule=f"majority of the {k} nearest labelled cells on {n_pcs} PCs of the pooled counts "
                    "(median-library normalised, log1p); ties to the nearest", k=k, n_pcs=n_pcs,
               n_cells=int(len(author)), n_author_labelled=int(lab.sum()),
               n_author_unassigned=int((author.astype(str) == UNASSIGNED).sum()),
               n_transferred=int(len(q)), loo_accuracy=acc, loo_accuracy_by_type=per_type)
    return final.astype(str).to_numpy(), rec


# ------------------------------------------------------------------ gene sets
def vdb_human_genes(dataset_genes, orthology_dir):
    """The van den Brink genes as genes of ``dataset_genes``, with the route each took."""
    hom = pd.read_csv(os.path.join(orthology_dir, "HOM_MouseHumanSequence.rpt.gz"), sep="\t", dtype=str)
    mouse = hom[hom["Common Organism Name"].str.startswith("mouse")]
    human = hom[hom["Common Organism Name"] == "human"]
    by_class = human.groupby("DB Class Key")["Symbol"].apply(list).to_dict()
    m2class = dict(zip(mouse["Symbol"], mouse["DB Class Key"]))
    hg = pd.read_csv(os.path.join(orthology_dir, "hgnc_complete_set.txt.gz"), sep="\t", dtype=str,
                     usecols=["symbol", "prev_symbol", "alias_symbol"])
    other, names = {}, {}
    for _, r in hg.iterrows():
        allnames = [r["symbol"]] + [x for col in ("prev_symbol", "alias_symbol") if isinstance(r[col], str)
                                    for x in r[col].split("|")]
        names[r["symbol"]] = allnames
        for x in allnames:
            other.setdefault(x, set()).add(r["symbol"])
    present = set(dataset_genes)
    rows, out = [], []
    for g in VDB_MOUSE_GENES:
        if g in m2class:
            cands, route = by_class.get(m2class[g], []), "MGI homology"
        else:
            cands, route = [g.upper()], "upper-case (no current MGI symbol)"
        hits = []
        for c in cands:
            if c in present:
                hits.append(c)
                continue
            for cur in other.get(c, ()):
                hits += [x for x in names.get(cur, []) if x in present]
        hits = sorted(set(hits))
        rows.append(dict(mouse=g, route=route, human_candidates=cands, matched=hits))
        out += hits
    return sorted(set(out)), rows


def marker_sets(X, ctype, exclude=()):
    """Genes specific to each type not in ``exclude`` (:data:`AMBIENT`): mean in its type >= 5 counts
    per 10k and >= 20 x its mean in every other type. Means are of counts per 10k (cp10k)."""
    lib = np.asarray(X.sum(1)).ravel()
    norm = sp.diags(1e4 / np.maximum(lib, 1)) @ X
    types = sorted(set(ctype))
    M = np.vstack([np.asarray(norm[np.flatnonzero(ctype == t)].mean(0)).ravel() for t in types])
    sets = {}
    for i, t in enumerate(types):
        if t in exclude:
            continue
        others = np.delete(M, i, axis=0).max(0)
        ok = (M[i] >= AMBIENT["marker_min_mean_cp10k"]) & (M[i] >= AMBIENT["marker_specificity"] * np.maximum(others, 1e-12))
        sets[t] = np.flatnonzero(ok)
    return sets


def ambient_estimate(X, ctype, target_mask, sets):
    """SoupX's principle on one batch: for each other type's marker set G, the soup share of G is its
    share of the batch's pooled counts (the filtered matrices hold no empty droplets, so the pooled
    cells stand in for the soup, as iscc's own ambient model does), and the contamination of the
    target cells is their marker counts over the soup share times their library sizes, after dropping
    target cells whose marker counts exceed the Poisson ``exclude_quantile`` at 100% contamination
    (SoupX's non-expressing cells). Returns the per-set estimates and whether they identify one
    contamination fraction (:data:`AMBIENT`)."""
    from scipy.stats import poisson
    tot = float(X.sum())
    lib = np.asarray(X.sum(1)).ravel()
    t_idx = np.flatnonzero(target_mask)
    per = {}
    for k, genes in sets.items():
        n_src = int((ctype == k).sum())
        if n_src < AMBIENT["min_source_cells"] or len(genes) == 0:
            per[k] = dict(n_source_cells=n_src, n_genes=int(len(genes)), used=False, reason="too few source cells")
            continue
        s_k = float(X[:, genes].sum()) / tot
        o = np.asarray(X[t_idx][:, genes].sum(1)).ravel()
        L = lib[t_idx]
        thr = poisson.ppf(AMBIENT["exclude_quantile"], s_k * L)
        keep = o <= thr
        rec = dict(n_source_cells=n_src, n_genes=int(len(genes)), soup_share=s_k,
                   n_target_kept=int(keep.sum()), n_target_dropped=int((~keep).sum()), counts=float(o[keep].sum()))
        if keep.sum() < AMBIENT["min_t_cells"] or o[keep].sum() < AMBIENT["min_counts"]:
            per[k] = dict(rec, used=False, reason="too few target cells or counts")
            continue
        per[k] = dict(rec, used=True, rho=float(o[keep].sum() / (s_k * L[keep].sum())))
    rhos = [v["rho"] for v in per.values() if v.get("used")]
    ident = (len(rhos) >= AMBIENT["min_sets"] and min(rhos) > 0 and max(rhos) < 1
             and max(rhos) / min(rhos) <= AMBIENT["agreement"])
    rho = float(np.median(rhos)) if rhos else None
    return dict(per_set=per, n_sets=len(rhos), identifiable=bool(ident), rho=rho,
                # iscc draws the soup as ambient_frac of the library, so the soup share rho is ambient_frac
                ambient_frac=(rho if (ident and rho is not None and rho < 1) else None))


# ------------------------------------------------------------------ fits
def fit_rna(X):
    """``estimate_rna`` (10x, fit_dropout on) on a cells x genes count matrix, zero genes dropped."""
    from iscc.data.estimate import estimate_rna
    X = sp.csc_matrix(X)
    nz = np.flatnonzero(np.diff(X.indptr) > 0)
    dense = X[:, nz].toarray().astype(float)
    # The registered design fits the T cells as one population: pooled, not split into automatic types.
    est = estimate_rna(dense, protocol="10x", fit_dropout=True, cell_types=None)
    h = est.hypers
    return dict(mu_lib=float(h.mu_lib), sigma_lib=float(h.sigma_lib), dispersion=float(h.dispersion),
                dropout_mid=float(h.dropout_mid), dropout_shape=float(h.dropout_shape),
                cv_lib=float(est.cv_lib), n_cells=int(est.n_cells), n_genes_nonzero=int(len(nz)),
                fitted=list(est.fitted))


def stress_score(X, genes_idx):
    """Mean over cells of the mean log1p(1e4 * count / library) of the genes ``genes_idx``."""
    lib = np.asarray(X.sum(1)).ravel()
    sub = X[:, genes_idx].toarray().astype(float)
    ln = np.log1p(sub * (1e4 / np.maximum(lib, 1))[:, None])
    return float(ln.mean(axis=1).mean())


def recovery_draws(n_patients=N_PATIENTS, sd=RECOVERY_SD, seed=RECOVERY_SEED):
    """Per patient and biological type: expit(logit(default recovery) + N(0, sd^2))."""
    from iscc.constants import STROMAL_SUBTYPES
    from iscc.sample.dissociation import DEFAULT_RECOVERY
    rng = np.random.default_rng(seed)
    # the stromal subtypes take the draw of the stroma (merge_recovery), so the draws are those of the
    # types before the subtypes
    types = [t for t in DEFAULT_RECOVERY if t == "stromal" or t not in STROMAL_SUBTYPES]
    out = {}
    for p in range(n_patients):
        z = rng.normal(0.0, sd, size=len(types))
        out[str(p)] = {t: float(1 / (1 + np.exp(-(np.log(DEFAULT_RECOVERY[t] / (1 - DEFAULT_RECOVERY[t])) + zi))))
                       for t, zi in zip(types, z)}
    return dict(rule="expit(logit(default) + N(0, sd^2)) per patient and type", sd=sd, seed=seed,
                defaults=dict(DEFAULT_RECOVERY), types=types, per_patient=out)


def combine_by_type(by_type, ambient_by_type, n_run_cells):
    """One vector per run from the per-type fits (the clarification of 2026-11-17): cell-weighted means
    of ``mu_lib``, ``sigma_lib``, dispersion and the ambient fraction; dropout only when types holding
    more than half the run's cells have a midpoint above 0, then the cell-weighted mean of those
    types' midpoints and shapes. The ambient fraction is a mean over the types whose estimate is
    identifiable, and counts as the run's only when those types hold more than half its cells;
    otherwise the run takes the registered range (``None`` here)."""
    fitted = {t: v for t, v in by_type.items() if "mu_lib" in v}
    w = {t: float(v["n_cells"]) for t, v in fitted.items()}
    W = sum(w.values())
    comb = {p: float(sum(w[t] * fitted[t][p] for t in fitted) / W) for p in ("mu_lib", "sigma_lib", "dispersion")}
    drop = {t: v for t, v in fitted.items() if v["dropout_mid"] > 0}
    drop_cells = sum(w[t] for t in drop)
    if drop_cells > n_run_cells / 2:
        Wd = sum(w[t] for t in drop)
        comb["dropout_mid"] = float(sum(w[t] * drop[t]["dropout_mid"] for t in drop) / Wd)
        comb["dropout_shape"] = float(sum(w[t] * drop[t]["dropout_shape"] for t in drop) / Wd)
        comb["dropout_source"] = f"types with a midpoint above 0 hold {drop_cells:.0f} of {n_run_cells} cells"
    else:
        comb["dropout_mid"], comb["dropout_shape"] = 0.0, 1.0
        comb["dropout_source"] = (f"no dropout: types with a midpoint above 0 hold {drop_cells:.0f} of "
                                  f"{n_run_cells} cells (not more than half)")
    amb = {t: a for t, a in ambient_by_type.items() if a.get("identifiable")}
    amb_cells = sum(by_type[t]["n_cells"] for t in amb)
    if amb and amb_cells > n_run_cells / 2:
        Wa = sum(by_type[t]["n_cells"] for t in amb)
        comb["ambient_frac"] = float(sum(by_type[t]["n_cells"] * amb[t]["ambient_frac"] for t in amb) / Wa)
        comb["ambient_source"] = f"fitted: identifiable in types holding {amb_cells} of {n_run_cells} cells"
    else:
        comb["ambient_frac"] = None
        comb["ambient_source"] = (f"not identifiable (identifiable types hold {amb_cells} of {n_run_cells} "
                                  "cells): drawn per seed from the registered range")
    return comb


def wu_fits(X, genes, meta, orthology_dir):
    tumours = sorted(meta["orig.ident"].astype(str).unique())
    ctype = meta["celltype_major"].astype(str).to_numpy()
    tum = meta["orig.ident"].astype(str).to_numpy()
    vdb, vdb_rows = vdb_human_genes(genes, orthology_dir)
    gidx = {g: i for i, g in enumerate(genes)}
    vdb_idx = np.array([gidx[g] for g in vdb])
    sets = marker_sets(X, ctype, exclude=(WU_T,))
    per = {}
    for t in tumours:
        sel = tum == t
        Xt, ct = X[np.flatnonzero(sel)], ctype[sel]
        tmask = ct == WU_T
        rec = dict(n_cells=int(sel.sum()), n_t_cells=int(tmask.sum()),
                   composition={k: int(v) for k, v in pd.Series(ct).value_counts().items()})
        if tmask.sum() < 50:
            per[t] = dict(rec, skipped=f"{int(tmask.sum())} T cells (< 50)")
            continue
        XT = Xt[np.flatnonzero(tmask)]
        rec["rna"] = fit_rna(XT)
        rec["stress_score"] = stress_score(XT, vdb_idx)
        rec["ambient"] = ambient_estimate(Xt, ct, tmask, sets)
        rec["doublet_rate"] = multiplet_rate(rec["n_cells"])
        per[t] = rec
        print(f"[wu] {t}: {rec['n_t_cells']} T cells, mu_lib {rec['rna']['mu_lib']:.0f}, phi "
              f"{rec['rna']['dispersion']:.3f}, dropout_mid {rec['rna']['dropout_mid']:.3g}, stress score "
              f"{rec['stress_score']:.4f}, ambient {rec['ambient']['rho']} ({rec['ambient']['n_sets']} sets, "
              f"identifiable {rec['ambient']['identifiable']}), doublets {rec['doublet_rate']:.4f}", flush=True)
    fitted = [t for t in tumours if "rna" in per[t]]
    if len(fitted) < 2:
        raise ValueError(f"only {len(fitted)} Wu tumour(s) have >= 50 T cells; stopping, not adapting")
    scores = np.array([per[t]["stress_score"] for t in fitted])
    med = float(np.median(scores))
    for t in fitted:
        per[t]["stress_strength"] = float(max(np.exp(per[t]["stress_score"] - med) - 1.0, 0.0))
    n_ident = sum(per[t]["ambient"]["identifiable"] for t in fitted)
    half = n_ident >= len(fitted) / 2
    needs = []
    for t in fitted:
        a = per[t]["ambient"]
        if half and a["identifiable"]:
            per[t]["ambient_frac"], per[t]["ambient_source"] = float(a["ambient_frac"]), "fitted"
        else:
            per[t]["ambient_frac"] = None
            per[t]["ambient_source"] = ("not identifiable: drawn per batch from the registered range" if half else
                                        "the estimate is identifiable in fewer than half the tumours, so no "
                                        "tumour uses it: drawn per batch from the registered range")
            needs.append(t)
    vectors = {}
    for t in fitted:
        r = per[t]["rna"]
        vectors[t] = dict(mu_lib=r["mu_lib"], sigma_lib=r["sigma_lib"], dispersion=r["dispersion"],
                          dropout_mid=r["dropout_mid"],
                          dropout_shape=(r["dropout_shape"] if r["dropout_mid"] > 0 else None),
                          ambient_frac=per[t]["ambient_frac"], doublet_rate=per[t]["doublet_rate"],
                          stress_strength=per[t]["stress_strength"])
    return dict(tumours=tumours, fitted_tumours=fitted, per_tumour=per, vectors=vectors,
                stress=dict(genes_matched=vdb, n_matched=len(vdb), mapping=vdb_rows, median_score=med,
                            rule="stress = max(exp(score - median score over tumours) - 1, 0); score = mean "
                                 "over the tumour's T cells of the mean log1p(1e4 count / library) of the genes"),
                ambient=dict(settings=AMBIENT, non_expressing="the tumour's T cells",
                             marker_sets={k: [str(genes[i]) for i in v] for k, v in sets.items()},
                             n_identifiable=int(n_ident), n_tumours=len(fitted),
                             identifiable_in_at_least_half=bool(half), needs_range=needs))


def ding_fits(X, genes, obs):
    """Per run and cell type: estimate_rna and the ambient attempt (the type's own cells against the
    other types' markers); combined into one vector per run (:func:`combine_by_type`), with the run's
    doublet rate."""
    ctype = obs["cell_type"].astype(str).to_numpy()
    sets = marker_sets(X, ctype)
    runs, needs = {}, []
    for r in DING_RUNS:
        sel = (obs["run"] == r).to_numpy()
        Xr, ct = X[np.flatnonzero(sel)], ctype[sel]
        by_type, amb_type = {}, {}
        for t in sorted(set(ct)):
            idx = np.flatnonzero(ct == t)
            if len(idx) < MIN_TYPE_CELLS_FIT:
                by_type[t] = dict(n_cells=int(len(idx)), skipped=f"fewer than {MIN_TYPE_CELLS_FIT} cells")
                continue
            by_type[t] = fit_rna(Xr[idx])
            amb_type[t] = ambient_estimate(Xr, ct, ct == t, {k: v for k, v in sets.items() if k != t})
        comb = combine_by_type(by_type, amb_type, int(sel.sum()))
        if comb["ambient_frac"] is None:
            needs.append(r)
        runs[r] = dict(n_cells=int(sel.sum()), composition={k: int(v) for k, v in pd.Series(ct).value_counts().items()},
                       by_type=by_type, ambient_by_type=amb_type, combined=comb,
                       doublet_rate=multiplet_rate(int(sel.sum())))
        print(f"[ding] run {r}: {int(sel.sum())} cells; combined {comb}; doublets {runs[r]['doublet_rate']:.4f}",
              flush=True)
    return dict(runs=runs, marker_sets={k: [str(genes[i]) for i in v] for k, v in sets.items()},
                min_type_cells_fit=MIN_TYPE_CELLS_FIT, needs_range=needs,
                combine_rule=combine_by_type.__doc__.strip().split("\n\n")[0].replace("\n    ", " "))


# ------------------------------------------------------------------ the ambient range and its checks
def ambient_needs(fits):
    """The tumours and runs whose ambient fraction comes from the registered range."""
    return dict(wu=list(fits.get("wu", {}).get("ambient", {}).get("needs_range", [])),
                ding=list(fits.get("ding", {}).get("needs_range", [])))


def require_ambient_range(fits):
    """Refuse to go on (statistics, levels) when a tumour or run needs the range and fits.json does not
    carry it yet; return the range (or None when nothing needs it)."""
    needs = ambient_needs(fits)
    if (needs["wu"] or needs["ding"]) and not fits.get("ambient_range"):
        raise RuntimeError(f"the ambient fraction is not identifiable for {needs}; fits.json carries no "
                           "registered range yet (batch_fits.py set-ambient-range), so nothing that uses it runs")
    return fits.get("ambient_range")


def draw_ambient(rng_range, rng):
    """One ambient_frac drawn from the registered range (:data:`AMBIENT_DRAW`)."""
    rho = float(rng.uniform(rng_range["rho_low"], rng_range["rho_high"]))
    return rho, rho


def set_ambient_range(path, low, high, source, registered):
    with open(path) as f:
        fits = json.load(f)
    if fits.get("ambient_range"):
        sys.exit(f"{path} already carries an ambient range: {fits['ambient_range']}")
    if not 0 <= low < high < 1:
        sys.exit("the range is of the soup share rho: 0 <= low < high < 1")
    fits["ambient_range"] = dict(rho_low=float(low), rho_high=float(high),
                                 ambient_frac_low=float(low), ambient_frac_high=float(high),
                                 draw=AMBIENT_DRAW, source=source, registered=registered,
                                 set=time.strftime("%Y-%m-%dT%H:%M:%S"), needs=ambient_needs(fits))
    with open(path + ".part", "w") as f:
        json.dump(fits, f, indent=1)
    os.replace(path + ".part", path)
    print(f"[fits] ambient range {fits['ambient_range']} -> {path}")


# ------------------------------------------------------------------ main
def fit(a):
    if os.path.exists(a.out):
        sys.exit(f"{a.out} exists: the fits are made once (remove it deliberately to refit)")
    os.makedirs(a.work, exist_ok=True)
    t0 = time.time()
    import iscc
    out = dict(registered="exp-paper1-fig5-cohort, Amendment 2026-11-16 night, with the Clarifications "
                          "before any run (2026-11-17)",
               sources=SOURCES, created=time.strftime("%Y-%m-%dT%H:%M:%S"), host=platform.node(),
               versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                             iscc=getattr(iscc, "__version__", None)),
               commit=_commit(), multiplet_table_percent=MULTIPLET_TABLE, vdb_mouse_genes=list(VDB_MOUSE_GENES),
               ambient_range=None)
    try:
        Xd, gd, od = load_ding(a.ding_dir, a.pbmcsca_meta)
        labels, transfer = transfer_labels(Xd, od["author_label"])
        od["cell_type"] = labels
        for r in DING_RUNS:
            s = (od["run"] == r).to_numpy()
            transfer[f"run_{r}"] = dict(n_cells=int(s.sum()),
                                        n_author_labelled=int((od["author_label"][s].notna()
                                                               & (od["author_label"][s].astype(str) != UNASSIGNED)).sum()))
        out["ding"] = ding_fits(Xd, gd, od)
        out["ding"]["labels"] = dict(transfer=transfer, cell_type=dict(zip(od.index, labels)),
                                     source=dict(zip(od.index, np.where(od["author_label"].notna()
                                                                        & (od["author_label"].astype(str) != UNASSIGNED),
                                                                        "author", "transferred"))))
        print(f"[ding] labels: {transfer}", flush=True)
        del Xd
        if not a.skip_wu:
            Xw, gw, mw = load_wu(a.wu_tar, a.work)
            out["wu"] = wu_fits(Xw, gw, mw, a.orthology_dir)
            del Xw
        out["recovery"] = recovery_draws()
        needs = ambient_needs(out)
        out["what_is_fitted"] = dict(
            fitted=["mu_lib", "sigma_lib", "dispersion", "dropout_mid", "dropout_shape",
                    "stress_strength (from the van den Brink score)",
                    "ambient_frac where identifiable (SoupX principle)"],
            computed_from_cells_recovered=["doublet_rate (10x multiplet table)"],
            drawn_from_range=["recovery per patient and type (logit-normal, sd 0.5)"]
            + ([f"ambient_frac of {needs} (range to be registered)"] if (needs["wu"] or needs["ding"]) else []),
            fixed=dict(sigma_batch=0.10, depth_batch_sigma="0 wherever mu_lib is drawn per batch"))
    finally:
        shutil.rmtree(a.work, ignore_errors=True)
    out["wall_s"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out + ".part", "w") as f:
        json.dump(out, f, indent=1, default=_jsonable)
    os.replace(a.out + ".part", a.out)
    print(f"[fits] wrote {a.out} in {out['wall_s']} s; ambient range needed for {ambient_needs(out)}", flush=True)


def file_sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--out", required=True)
    f.add_argument("--wu-tar", required=True)
    f.add_argument("--ding-dir", required=True)
    f.add_argument("--pbmcsca-meta", required=True)
    f.add_argument("--orthology-dir", required=True)
    f.add_argument("--work", required=True, help="node-local scratch directory (removed at the end)")
    f.add_argument("--skip-wu", action="store_true", help="smoke: Ding only")
    r = sub.add_parser("set-ambient-range")
    r.add_argument("--fits", required=True)
    r.add_argument("--low", type=float, required=True, help="lowest soup share rho of a cell's UMIs")
    r.add_argument("--high", type=float, required=True)
    r.add_argument("--source", required=True, help="the verified source: paper, figure or table")
    r.add_argument("--registered", required=True, help="where the range was registered (vault commit)")
    a = p.parse_args(argv)
    if a.cmd == "fit":
        fit(a)
    else:
        set_ambient_range(a.fits, a.low, a.high, a.source, a.registered)


def _jsonable(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (tuple, set)):
        return list(o)
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def _commit():
    try:
        return subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None


if __name__ == "__main__":
    main()
