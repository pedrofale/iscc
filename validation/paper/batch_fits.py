"""Fig 5's scRNA batch model: each batch's technical parameters fitted to real data.

Registered in the vault note exp-paper1-fig5-cohort, section "Amendment, 2026-11-16 night: batches
from named technical sources, and a panel validating them"; the model choice is
decision-scrna-batches-from-technical-sources. This script makes every fit ONCE and writes them to
one summary file, before any statistic of the validation panel (``batch_validation.py``) or any
integration is computed::

    PYTHONPATH=src python validation/paper/batch_fits.py --out $D/batchval/fits.json \\
        --wu-tar $D/realdata/GSE176078/GSE176078_Wu_etal_2021_BRCA_scRNASeq.tar.gz \\
        --ding-dir $D/realdata/GSE132044 --pbmcsca-meta $D/realdata/pbmcsca/pbmcsca_meta.tsv.gz \\
        --orthology-dir $D/realdata/orthology --work $SNIC_TMP/fits

The archives are extracted into ``--work`` (node-local scratch) and deleted with it.

**Wu et al. 2021** (Nat Genet 53:1334, doi:10.1038/s41588-021-00911-1; GEO GSE176078; 10x Chromium
Single-Cell v2 3' and 5', Cell Ranger 2.0, EmptyDrops then genes > 200, UMIs > 250, mito < 20%, per
the paper's Methods, read on PMC9044823 on 2026-11-16). Each tumour (``orig.ident``) is a batch:

* ``estimate_rna`` (``protocol="10x"``, ``fit_dropout=True``) on the tumour's T cells only
  (``celltype_major == "T-cells"``), so cell-type composition does not enter the technical fit:
  ``mu_lib``, ``sigma_lib``, ``dispersion``, ``dropout_mid``, ``dropout_shape``. A tumour whose
  fitted midpoint is 0 has no dropout. Genes with no count in those cells are dropped first; they
  change none of the estimates (library sizes are row sums; the dispersion and dropout fits use
  expressed genes only).
* Dissociation stress: the mean over the tumour's T cells of the mean log-normalised expression
  (``log1p(1e4 * count / library)``) of the dissociation-affected genes of van den Brink et al. 2017
  (Nat Methods 14:935, doi:10.1038/nmeth.4437), relative to the median over tumours, converted to
  ``iscc``'s multiplicative stress ``max(exp(score - median) - 1, 0)``. The 140 genes are the mouse
  genes of the paper's Supplementary Table 5, copied from the list in its Supplementary Methods
  ("In silico purification", the R code ``Selection <- Data[c(...)]``; Supplementary Text and
  Figures PDF, MOESM1, read 2026-11-16). They are mapped to human genes through MGI's homology
  report (HOM_MouseHumanSequence.rpt) and, for mouse symbols MGI no longer uses and for human genes
  the data name by an older symbol, through HGNC's previous and alias symbols (hgnc_complete_set);
  both files downloaded 2026-11-16 and kept beside the data.
* Ambient fraction, SoupX's principle (Young & Behjati 2020, GigaScience 9:giaa151,
  doi:10.1093/gigascience/giaa151): genes specific to one other cell type, counted in the T cells,
  are soup. Fixed before the fit (:func:`ambient_estimate`); used only if identifiable in at least
  half the tumours.
* Doublet rate: 10x's published multiplet rate at the tumour's number of cells (Chromium Single Cell
  3' Reagent Kits v2 User Guide, CG00052 Rev B, p. 6: ~0.4% at ~500 cells recovered ... ~7.6% at
  ~10,000; the v3.1 guide CG000204 Rev D gives the same rates), interpolated linearly in the table
  and extrapolated above 10,000 cells at 7.6% per 10,000 (:data:`MULTIPLET_TABLE`).
* Recovery per patient: each type's default recovery on the logit scale plus a normal draw of sd 0.5
  (:func:`recovery_draws`), for the cohort's 50 patients.

**Ding et al. 2020** (Nat Biotechnol 38:737, doi:10.1038/s41587-020-0465-8; GEO GSE132044), PBMC1 in
two 10x Chromium v2 runs, A (5,172 cells) and B (3,057 cells): the same fits per run, on all its
cells by type (``estimate_rna`` within each cell type, combined over the types by their cell
numbers, :func:`ding_fits`), with the run's own ambient attempt and doublet rate, and no
dissociation (blood). Cell types are the authors' annotations (``CellType`` of the Single Cell
Portal study SCP424's metadata, as packaged in SeuratData's pbmcsca 3.0.0, CC BY 4.0); cells the
authors left "Unassigned" are left out of every fit and statistic.
"""
import argparse
import gzip
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

# ------------------------------------------------------------------ sources (checked 2026-11-16)
SOURCES = {
    "wu2021": dict(citation="Wu SZ et al. A single-cell and spatially resolved atlas of human breast "
                            "cancers. Nat Genet 53, 1334-1347 (2021)",
                   doi="10.1038/s41588-021-00911-1", geo="GSE176078",
                   file="GSE176078_Wu_etal_2021_BRCA_scRNASeq.tar.gz (558,829,202 bytes)",
                   methods="10x Chromium Single-Cell v2 3' and 5'; 5,000-7,000 cells targeted per well; "
                           "Cell Ranger 2.0 on GRCh38; EmptyDrops, then genes > 200, UMIs > 250, "
                           "mitochondrial < 20% (Methods, PMC9044823)"),
    "ding2020": dict(citation="Ding J et al. Systematic comparison of single-cell and single-nucleus "
                              "RNA-sequencing methods. Nat Biotechnol 38, 737-746 (2020)",
                     doi="10.1038/s41587-020-0465-8", geo="GSE132044",
                     files=["GSE132044_pbmc_hg38_cell.tsv.gz", "GSE132044_pbmc_hg38_gene.tsv.gz",
                            "GSE132044_pbmc_hg38_count_matrix.mtx.gz (127,509,347 bytes)"]),
    "ding2020_labels": dict(source="SeuratData pbmcsca.SeuratData 3.0.0 (CC BY 4.0; md5 "
                                   "6eb6dad07367ea8861027200196b096c), metadata column CellType, "
                                   "repackaging the Single Cell Portal study SCP424's meta.txt"),
    "vandenbrink2017": dict(citation="van den Brink SC et al. Single-cell sequencing reveals "
                                     "dissociation-induced gene expression in tissue subpopulations. "
                                     "Nat Methods 14, 935-936 (2017)",
                            doi="10.1038/nmeth.4437",
                            where="Supplementary Methods, 'In silico purification' (the 140 genes of "
                                  "Supplementary Table 5, mouse symbols), MOESM1 PDF"),
    "soupx": dict(citation="Young MD, Behjati S. SoupX removes ambient RNA contamination from "
                           "droplet-based single-cell RNA sequencing data. GigaScience 9, giaa151 (2020)",
                  doi="10.1093/gigascience/giaa151"),
    "multiplet": dict(source="10x Genomics, Chromium Single Cell 3' Reagent Kits v2 User Guide, "
                             "CG00052 Rev B, p. 6 (multiplet rate by cells recovered); identical "
                             "rates in Chromium Next GEM Single Cell 3' v3.1 User Guide CG000204 Rev D"),
    "mgi_homology": dict(source="MGI HOM_MouseHumanSequence.rpt, downloaded 2026-11-16"),
    "hgnc": dict(source="HGNC hgnc_complete_set.txt, downloaded 2026-11-16"),
}

# van den Brink et al. 2017, Supplementary Table 5, as listed in the Supplementary Methods (mouse).
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

# 10x multiplet rate (%) by cells recovered: CG00052 Rev B p. 6 (v2), the chemistry Wu et al. and
# Ding et al. used.
MULTIPLET_TABLE = ((500, 0.4), (1000, 0.8), (2000, 1.6), (3000, 2.3), (4000, 3.1), (5000, 3.9),
                   (6000, 4.6), (7000, 5.4), (8000, 6.1), (9000, 6.9), (10000, 7.6))

WU_T = "T-cells"
DING_RUNS = {"A": "PBMC1.10x-Chromium-v2-A", "B": "PBMC1.10x-Chromium-v2-B"}
DING_LABEL_METHOD = {"A": "10x Chromium (v2) A", "B": "10x Chromium (v2) B"}
DING_T_TYPES = ("CD4+ T cell", "Cytotoxic T cell")
UNASSIGNED = "Unassigned"

# ---- the ambient attempt (SoupX's principle), fixed before any fit ----
AMBIENT = dict(
    marker_min_mean_cp10k=5.0,       # a marker is well expressed in its own type (so it is in the soup)
    marker_specificity=20.0,         # ... and its mean there is >= 20x its mean in every other type
    min_source_cells=50,             # a type's set is used in a batch only with >= 50 cells of it there
    min_t_cells=50,                  # ... >= 50 non-expressing (T) cells kept
    min_counts=20,                   # ... and >= 20 marker counts observed in them
    exclude_quantile=0.999,          # SoupX's non-expressing cells: drop a T cell whose marker counts
                                     # exceed this Poisson quantile at 100% contamination
    agreement=2.0,                   # identifiable: >= 2 sets, largest / smallest estimate <= 2
    min_sets=2,
)
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
    df = pd.read_csv(path, sep=r"\s+", header=None, skiprows=skip, dtype={0: np.int32, 1: np.int32, 2: np.float64},
                     engine="c")
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
    shutil.rmtree(d)
    return X, genes, meta


def load_ding(ding_dir, labels_path):
    """Ding et al.'s PBMC1 10x v2 runs A and B: (counts cells x genes CSR, genes, obs with run and
    cell type). Cells the authors left "Unassigned" are dropped here (recorded by the caller)."""
    cells = pd.read_csv(os.path.join(ding_dir, "GSE132044_pbmc_hg38_cell.tsv.gz"), header=None)[0].astype(str).to_numpy()
    genes = pd.read_csv(os.path.join(ding_dir, "GSE132044_pbmc_hg38_gene.tsv.gz"), header=None)[0].astype(str).to_numpy()
    X = read_mtx(os.path.join(ding_dir, "GSE132044_pbmc_hg38_count_matrix.mtx.gz"))
    if X.shape == (len(genes), len(cells)):
        X = X.T.tocsr()
    if X.shape != (len(cells), len(genes)):
        raise ValueError(f"Ding matrix {X.shape} matches neither genes {len(genes)} nor cells {len(cells)}")
    run = pd.Series(cells).str.rsplit(".", n=1).str[0].to_numpy()
    keep = np.isin(run, list(DING_RUNS.values()))
    X, cells, run = X[keep], cells[keep], run[keep]
    inv = {v: k for k, v in DING_RUNS.items()}
    obs = pd.DataFrame({"run": [inv[r] for r in run], "barcode": pd.Series(cells).str.rsplit(".", n=1).str[1].to_numpy()},
                       index=cells)
    lab = pd.read_csv(labels_path, sep="\t", low_memory=False)
    obs["cell_type"] = _ding_labels(obs, lab)
    return X, genes, obs


def _ding_labels(obs, lab):
    """Match the pbmcsca metadata to the GEO cells by run and barcode."""
    out = pd.Series(index=obs.index, dtype=object)
    for r, method in DING_LABEL_METHOD.items():
        m = lab[lab["Method"] == method]
        if m.empty:
            raise ValueError(f"pbmcsca has no cells of {method!r}")
        bc = m["cell"].astype(str).str.extract(r"([ACGTN]{12,})")[0]
        if bc.isna().any():
            raise ValueError(f"pbmcsca cell names of {method!r} carry no barcode, e.g. {m['cell'].iloc[0]}")
        if "Experiment" in m:
            m = m[m["Experiment"].astype(str).str.lower().str.contains("pbmc1")]
            bc = m["cell"].astype(str).str.extract(r"([ACGTN]{12,})")[0]
        lookup = pd.Series(m["CellType"].astype(str).to_numpy(), index=bc.to_numpy())
        if not lookup.index.is_unique:
            raise ValueError(f"pbmcsca barcodes repeat within {method!r}")
        sel = obs["run"] == r
        out[sel] = lookup.reindex(obs.loc[sel, "barcode"]).to_numpy()
    return out


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
    other = {}                                           # any HGNC symbol -> the gene's current symbol + its old names
    names = {}
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
            # the data may name the human gene by an older or alias symbol, or MGI by a newer one
            for cur in other.get(c, ()):
                hits += [x for x in names.get(cur, []) if x in present]
        hits = sorted(set(hits))
        rows.append(dict(mouse=g, route=route, human_candidates=cands, matched=hits))
        out += hits
    return sorted(set(out)), rows


def marker_sets(X, ctype, t_types):
    """Genes specific to each non-T type (:data:`AMBIENT`): mean in its type >= 5 counts per 10k and
    >= 20 x its mean in every other type, T cells included. Means are of counts per 10k (cp10k)."""
    lib = np.asarray(X.sum(1)).ravel()
    norm = sp.diags(1e4 / np.maximum(lib, 1)) @ X
    types = sorted(set(ctype))
    means = {}
    for t in types:
        idx = np.flatnonzero(ctype == t)
        means[t] = np.asarray(norm[idx].mean(0)).ravel()
    M = np.vstack([means[t] for t in types])
    sets = {}
    for i, t in enumerate(types):
        if t in t_types:
            continue
        others = np.delete(M, i, axis=0).max(0)
        ok = (M[i] >= AMBIENT["marker_min_mean_cp10k"]) & (M[i] >= AMBIENT["marker_specificity"] * np.maximum(others, 1e-12))
        sets[t] = np.flatnonzero(ok)
    return sets


def ambient_estimate(X, ctype, t_mask, sets):
    """SoupX's principle on one batch: for each other type's marker set G, the soup share of G is
    its share of the batch's pooled counts (the filtered matrices hold no empty droplets, so the
    pooled cells stand in for the soup, as iscc's own ambient model does), and the contamination is
    the marker counts in the T cells over the soup share times their library sizes, after dropping
    T cells whose marker counts exceed the Poisson ``exclude_quantile`` at 100% contamination
    (SoupX's non-expressing cells). Returns the per-set estimates and whether they identify one
    contamination fraction (:data:`AMBIENT`)."""
    from scipy.stats import poisson
    tot = float(X.sum())
    lib = np.asarray(X.sum(1)).ravel()
    t_idx = np.flatnonzero(t_mask)
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
                   n_t_kept=int(keep.sum()), n_t_dropped=int((~keep).sum()), counts=float(o[keep].sum()))
        if keep.sum() < AMBIENT["min_t_cells"] or o[keep].sum() < AMBIENT["min_counts"]:
            per[k] = dict(rec, used=False, reason="too few T cells or counts")
            continue
        rho = float(o[keep].sum() / (s_k * L[keep].sum()))
        per[k] = dict(rec, used=True, rho=rho)
    rhos = [v["rho"] for v in per.values() if v.get("used")]
    ident = (len(rhos) >= AMBIENT["min_sets"] and min(rhos) > 0 and max(rhos) < 1
             and max(rhos) / min(rhos) <= AMBIENT["agreement"])
    rho = float(np.median(rhos)) if rhos else None
    return dict(per_set=per, n_sets=len(rhos), identifiable=bool(ident), rho=rho,
                # iscc adds ambient_frac x library soup counts on top of the cell's own library, so
                # the soup share of a cell is a / (1 + a): a = rho / (1 - rho)
                ambient_frac=(rho / (1 - rho) if (rho is not None and rho < 1) else None))


# ------------------------------------------------------------------ fits
def fit_rna(X):
    """``estimate_rna`` (10x, fit_dropout on) on a cells x genes count matrix, zero genes dropped."""
    from iscc.data.estimate import estimate_rna
    X = X.tocsc()
    nz = np.flatnonzero(np.diff(X.indptr) > 0)
    dense = X[:, nz].toarray().astype(float)
    est = estimate_rna(dense, protocol="10x", fit_dropout=True)
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
    from iscc.sample.dissociation import DEFAULT_RECOVERY
    rng = np.random.default_rng(seed)
    types = list(DEFAULT_RECOVERY)
    out = {}
    for p in range(n_patients):
        z = rng.normal(0.0, sd, size=len(types))
        out[str(p)] = {t: float(1 / (1 + np.exp(-(np.log(DEFAULT_RECOVERY[t] / (1 - DEFAULT_RECOVERY[t])) + zi))))
                       for t, zi in zip(types, z)}
    return dict(rule="expit(logit(default) + N(0, sd^2)) per patient and type", sd=sd, seed=seed,
                defaults=dict(DEFAULT_RECOVERY), types=types, per_patient=out)


def wu_fits(X, genes, meta, orthology_dir):
    tumours = sorted(meta["orig.ident"].astype(str).unique())
    ctype = meta["celltype_major"].astype(str).to_numpy()
    tum = meta["orig.ident"].astype(str).to_numpy()
    vdb, vdb_rows = vdb_human_genes(genes, orthology_dir)
    gidx = {g: i for i, g in enumerate(genes)}
    vdb_idx = np.array([gidx[g] for g in vdb])
    sets = marker_sets(X, ctype, (WU_T,))
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
    scores = np.array([per[t]["stress_score"] for t in fitted])
    med = float(np.median(scores))
    for t in fitted:
        d = per[t]["stress_score"] - med
        per[t]["stress_strength"] = float(max(np.exp(d) - 1.0, 0.0))
    n_ident = sum(per[t]["ambient"]["identifiable"] for t in fitted)
    use_ambient = n_ident >= len(fitted) / 2
    ident_vals = [per[t]["ambient"]["ambient_frac"] for t in fitted if per[t]["ambient"]["identifiable"]]
    for t in fitted:
        a = per[t]["ambient"]
        if use_ambient:
            if a["identifiable"]:
                per[t]["ambient_frac"], per[t]["ambient_source"] = float(a["ambient_frac"]), "fitted"
            else:
                per[t]["ambient_frac"], per[t]["ambient_source"] = float(np.median(ident_vals)), \
                    "median of the identifiable tumours (this tumour's estimate was not identifiable)"
        else:
            per[t]["ambient_frac"], per[t]["ambient_source"] = None, "not fitted (identifiable in fewer than half)"
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
                ambient=dict(settings=AMBIENT, marker_sets={k: [str(genes[i]) for i in v] for k, v in sets.items()},
                             n_identifiable=int(n_ident), n_tumours=len(fitted), used=bool(use_ambient)))


def ding_fits(X, genes, obs, orthology_dir):
    """Per run: estimate_rna within each cell type, combined over types by their cell numbers; the
    run's ambient attempt (T cells against the other types' markers) and doublet rate."""
    keep = obs["cell_type"].notna() & (obs["cell_type"] != UNASSIGNED)
    n_unassigned = {r: int(((obs["run"] == r) & ~keep).sum()) for r in DING_RUNS}
    n_unlabelled = {r: int(((obs["run"] == r) & obs["cell_type"].isna()).sum()) for r in DING_RUNS}
    n_all = {r: int((obs["run"] == r).sum()) for r in DING_RUNS}
    Xk, ok = X[np.flatnonzero(keep.to_numpy())], obs[keep]
    ctype = ok["cell_type"].astype(str).to_numpy()
    sets = marker_sets(Xk, ctype, DING_T_TYPES)
    runs = {}
    for r in DING_RUNS:
        sel = (ok["run"] == r).to_numpy()
        Xr, ct = Xk[np.flatnonzero(sel)], ctype[sel]
        by_type = {}
        for t in sorted(set(ct)):
            idx = np.flatnonzero(ct == t)
            if len(idx) < 10:
                by_type[t] = dict(n_cells=int(len(idx)), skipped="fewer than 10 cells")
                continue
            by_type[t] = fit_rna(Xr[idx])
        fitted = {t: v for t, v in by_type.items() if "mu_lib" in v}
        w = np.array([v["n_cells"] for v in fitted.values()], float)
        comb = {}
        for p in ("mu_lib", "sigma_lib", "dispersion", "dropout_mid", "dropout_shape"):
            comb[p] = float(np.sum(w * np.array([v[p] for v in fitted.values()])) / w.sum())
        amb = ambient_estimate(Xr, ct, np.isin(ct, DING_T_TYPES), sets)
        runs[r] = dict(n_cells_geo=n_all[r], n_unassigned=n_unassigned[r], n_unlabelled=n_unlabelled[r],
                       n_cells=int(sel.sum()), composition={k: int(v) for k, v in pd.Series(ct).value_counts().items()},
                       by_type=by_type, combined=comb, ambient=amb, doublet_rate=multiplet_rate(n_all[r]))
        print(f"[ding] run {r}: {int(sel.sum())} cells; combined {comb}; ambient {amb['rho']} "
              f"(identifiable {amb['identifiable']}); doublets {runs[r]['doublet_rate']:.4f}", flush=True)
    n_ident = sum(runs[r]["ambient"]["identifiable"] for r in runs)
    use = n_ident >= len(runs) / 2
    vals = [runs[r]["ambient"]["ambient_frac"] for r in runs if runs[r]["ambient"]["identifiable"]]
    for r in runs:
        if use:
            if runs[r]["ambient"]["identifiable"]:
                runs[r]["ambient_frac"], runs[r]["ambient_source"] = float(runs[r]["ambient"]["ambient_frac"]), "fitted"
            else:
                runs[r]["ambient_frac"], runs[r]["ambient_source"] = float(np.median(vals)), "median of the identifiable run(s)"
        else:
            runs[r]["ambient_frac"], runs[r]["ambient_source"] = None, "not fitted (identifiable in fewer than half)"
    return dict(runs=runs, marker_sets={k: [str(genes[i]) for i in v] for k, v in sets.items()},
                ambient_used=bool(use), combine_rule="cell-number-weighted mean over cell types with >= 10 cells "
                "of each estimate_rna parameter fitted within the type (a type whose fitted dropout midpoint "
                "is 0 enters it as 0, its shape as estimate_rna's 1.0)")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", required=True)
    p.add_argument("--wu-tar", required=True)
    p.add_argument("--ding-dir", required=True)
    p.add_argument("--pbmcsca-meta", required=True)
    p.add_argument("--orthology-dir", required=True)
    p.add_argument("--work", required=True, help="node-local scratch directory (removed at the end)")
    p.add_argument("--skip-wu", action="store_true", help="smoke: Ding only")
    a = p.parse_args(argv)
    if os.path.exists(a.out):
        sys.exit(f"{a.out} exists: the fits are made once (remove it deliberately to refit)")
    os.makedirs(a.work, exist_ok=True)
    t0 = time.time()
    import iscc
    out = dict(registered="exp-paper1-fig5-cohort, Amendment 2026-11-16 night", sources=SOURCES,
               created=time.strftime("%Y-%m-%dT%H:%M:%S"), host=platform.node(),
               versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                             iscc=getattr(iscc, "__version__", None)),
               commit=_commit(), multiplet_table_percent=MULTIPLET_TABLE,
               vdb_mouse_genes=list(VDB_MOUSE_GENES))
    try:
        Xd, gd, od = load_ding(a.ding_dir, a.pbmcsca_meta)
        out["ding"] = ding_fits(Xd, gd, od, a.orthology_dir)
        out["ding"]["label_match"] = {r: dict(n=int((od["run"] == r).sum()),
                                              labelled=int(((od["run"] == r) & od["cell_type"].notna()).sum()))
                                      for r in DING_RUNS}
        del Xd
        if not a.skip_wu:
            Xw, gw, mw = load_wu(a.wu_tar, a.work)
            out["wu"] = wu_fits(Xw, gw, mw, a.orthology_dir)
            del Xw
        out["recovery"] = recovery_draws()
        out["what_is_fitted"] = dict(
            fitted_per_tumour=["mu_lib", "sigma_lib", "dispersion", "dropout_mid", "dropout_shape",
                               "stress_strength (from the van den Brink score)"],
            computed_from_cells_recovered=["doublet_rate (10x multiplet table)"],
            ambient_frac=("fitted per tumour (SoupX principle)" if out.get("wu", {}).get("ambient", {}).get("used")
                          else "not identifiable in at least half the tumours: not fitted"),
            drawn_from_range=["recovery per patient and type (logit-normal, sd 0.5)"]
            + ([] if out.get("wu", {}).get("ambient", {}).get("used") else ["ambient_frac (range not yet stated)"]),
            fixed=dict(sigma_batch=0.10, depth_batch_sigma="0 wherever mu_lib is drawn per batch"))
    finally:
        shutil.rmtree(a.work, ignore_errors=True)
    out["wall_s"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out + ".part", "w") as f:
        json.dump(out, f, indent=1, default=_jsonable)
    os.replace(a.out + ".part", a.out)
    print(f"[fits] wrote {a.out} in {out['wall_s']} s", flush=True)


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
