"""Fig 4c and 4d on one tumour: the errors one modality hands the next, scored against the ground truth.

4c (``dna-rna``). clonealign assigns the tumour's scRNA cells to clones twice: once given the TRUE clones
and their copy-number profiles, once given the clones called from SIMULATED scDNA. The DNA chain is the
one a single-cell WGS study runs: DLP+-like libraries (``scDNA`` with ``estimate_dna`` fitted to the DLP+
reference in ``validation/data``), HMMcopy per cell on bins of ~500 kb (one growth locus each), then
UMAP + HDBSCAN on the called states, clusters with the same copy-number profile merged and flat (diploid)
clusters dropped. The gap between the two accuracies is the error the RNA step inherits from the DNA step.

4d (``spatial``). Clonalscope maps clones onto the tumour's Visium section (the Visium of Fig 4b: the same
section, seeds and settings) without a prior and with the DNA-derived prior from 4c's scDNA chain, handed
over through Clonalscope's own inputs: the segment table's ``states`` (the relative copy number of the DNA
pseudobulk per arm, what ``Segmentation_bulk`` gives from matched WGS) and ``U0``, its prior mean relative
coverage per cluster (a normal row and one row per DNA clone). Both runs take Fig 4b's normal reference
(``visium_reference``: the spots holding only non-epithelial normal cells) and cluster every other spot,
normal epithelium included; the writer's own celltype.tsv, which marks the spots with any true cancer
cell, is never read. CalicoST runs alongside (its clone labels; it reads no reference). Each tool's call per
spot is scored against the spot's true clone fractions by ``fig4_score.py``. ``--only`` redoes some of the
three tools and keeps the others' stored labels, refusing when the rebuilt spots differ from the stored.

The true clones are the ground truth's: lineage clades (``clones_from_clades``) holding at least 3% of the
sampled tumour cells, counted over the dissociated and the sectioned cells together so that scRNA, scDNA
and Visium share one set of labels, and clades whose arm-level profiles are proportional merged into one
clone, since no copy-number method can tell them apart (expression and single-cell depth are both
library-normalised).

Every input a tool reads and everything it writes live under ``--work`` (a job's $SNIC_TMP); only
per-tumour summaries reach ``--out``:

    OUT/c/<run>_<task>/  score.json, true_clones.tsv, dna_clones.tsv, dna_cells.tsv.gz, rna_cells.tsv.gz
    OUT/d/<run>_<task>/  score.json, spots.tsv.gz (truth, the reference flag and every tool's label per spot)

The spatial step reads ``OUT/c/<run>_<task>/dna_clones.tsv``, so it runs after ``dna-rna``. Usage
(``ISCC_PAPER_OUT`` and ``ISCC_PAPER_TIER`` locate the registry; ``ISCC_TOOLS_DIR``,
``ISCC_CLONEALIGN_RSCRIPT`` and ``ISCC_HMMCOPY_RSCRIPT`` the tools)::

    python validation/paper/fig4_chain.py dna-rna --run R14 --task seed1 --out OUT --work $SNIC_TMP
    python validation/paper/fig4_chain.py spatial --run R14 --task seed1 --out OUT --work $SNIC_TMP
    python validation/paper/fig4_chain.py info    --run R14 --task seed1 --out OUT --work $SNIC_TMP

``oracle-set`` (after dna-rna) runs clonealign on the true profiles of the clones the DNA step found
(``oracle_set.json``), which splits 4c's gap into missed clones and profile error. ``info`` only writes
``OUT/c/<run>_<task>/tumour.json`` (cancer cells, whole-genome-doubled fraction, and
whether the doubling swept the tumour), which ``dna-rna`` also writes.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
VAL = os.path.dirname(HERE)
TOOLS = os.path.join(HERE, "tools")
for p in (HERE, VAL, TOOLS):
    if p not in sys.path:
        sys.path.insert(0, p)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

MIN_CLONE_FRAC = 0.03       # Fig 4b's truth (make_layer_dataset --min-clone-frac)
N_DNA = 1500                # single-cell WGS libraries, a DLP+ run's order of magnitude
RNA_MAX_CELLS = 3000        # tumour cells handed to clonealign
RNA_GENES = 4000            # at most this many genes, the most expressed of the informative ones
COARSE_K = 4                # --coarse: the ground truth's clones in this many copy-number groups
# The Visium section's cell cap: none, as regime.SECTION_MAX_CELLS (not imported here, so that the module
# loads without the regime), so the spots hold the cells of the section; the tier's max_cells (8,000)
# left about 1.4 cells per spot and 50,000 2.3 to 6.1. The dissociated sample (scDNA, scRNA) keeps the
# tier's cap. The section is materialised with only the frames Visium and the allele counts read
# (expression, and expression per homolog).
SECTION_MAX_CELLS = int(os.environ.get("ISCC_SECTION_MAX_CELLS", 10 ** 8))
SECTION_LAYERS = ("cell_exp", "cell_exp_p", "cell_exp_m")
CLONEALIGN = dict(max_iter=200, n_repeats=2, seed=1, preprocess=1)   # Fig 5's settings + the package's
                                                                      # preprocess_for_clonealign
NORMAL, OTHER, TRUNK = "normal", "other", "above_clades"
SPATIAL_TOOLS = ("clonalscope", "clonalscope_dna", "calicost")   # 4d


def _lap(wall, name, t0):
    wall[name] = round(time.time() - t0, 1)
    print(f"  {name}: {wall[name]} s", flush=True)
    return time.time()


def _commit():
    try:
        return subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True).stdout.strip() or os.environ.get("ISCC_COMMIT", "")
    except OSError:
        return os.environ.get("ISCC_COMMIT", "")


# ---------------------------------------------------------------------------------------------------
# the tumour, its samples and the ground truth
# ---------------------------------------------------------------------------------------------------
def sample(run, task, section_max_cells=SECTION_MAX_CELLS, dosage="regime"):
    """The stored tumour, its human-gene layer, and the two samples of the paper's figures: the
    dissociated half (scRNA, scDNA, at the tier's max_cells) and the section (Visium, at
    ``section_max_cells``), cut as make_layer_dataset cuts them. Every step slices the same section,
    since the ground truth's clones are counted over both samples."""
    import realistic_regime as RR
    import regime as RG
    import registry as REG
    from iscc.data.human import HumanGenomeLayer
    t = REG.load(run, task)
    if dosage == "proportional":            # the counterfactual: every gene follows copy number, clonealign's law
        t.programs.dosage_sensitivity = np.ones_like(t.programs.dosage_sensitivity)
    layer = HumanGenomeLayer(t, allow_linear=t.genome_params.get("arms") is None)
    spec = RR.resect(t)
    dis = spec.dissociate(max_cells=RG.T["max_cells"])
    sec = spec.slice(depth_frac=0.5, max_cells=int(section_max_cells), layers=SECTION_LAYERS)
    return t, layer, dis, sec


def tumour_info(t, run, task):
    """The tumour's cancer-cell count and the fraction carrying a whole-genome doubling (as summaries.py
    counts it); a tumour with at least 95% doubled cells is marked swept."""
    from collections import Counter
    cnt = Counter()
    for deme in t.demes:
        for g, c in deme.items():
            if t._is_cancer(g):
                cnt[g] += c
    tot = sum(cnt.values())
    wgd = sum(c for g, c in cnt.items() if getattr(t.genotypes[g], "is_wgd", False)) / max(tot, 1)
    return dict(run=run, task=task, cancer=int(tot), wgd_frac=float(wgd), wgd_swept=bool(wgd >= 0.95),
                paper_out=os.environ.get("ISCC_PAPER_OUT"), tier=os.environ.get("ISCC_PAPER_TIER"),
                commit=_commit())


def write_tumour_info(t, a):
    out = os.path.join(a.out, "c", f"{a.run}_{a.task}")
    os.makedirs(out, exist_ok=True)
    info = tumour_info(t, a.run, a.task)
    with open(os.path.join(out, "tumour.json"), "w") as f:
        json.dump(info, f, indent=1)
    print("tumour:", json.dumps(info), flush=True)
    return info


def run_info(a):
    import registry as REG
    write_tumour_info(REG.load(a.run, a.task), a)


def segment_names(layer):
    return list(layer.arms) if layer.arms is not None else [f"seg{s}" for s in range(len(layer.segment_sizes))]


def _gids(cell_data, cells=None):
    g = cell_data["cell_type"].iloc[:, 0]
    return (g if cells is None else g.reindex(cells)).to_numpy(dtype=object)


def genotype_cn(t, layer, gids):
    """Total copy number per growth segment (an arm, on an arm genome) for each genotype."""
    cn_p, cn_m = layer.homolog_cn(list(gids), t)
    return (cn_p + cn_m).astype(int)


def _proportional_key(profile):
    p = np.asarray(profile, float)
    return tuple(np.round(p / p.mean(), 3)) if p.mean() > 0 else tuple(p)


def true_clones(t, layer, samples, min_frac=MIN_CLONE_FRAC):
    """The ground truth's clones over the union of ``samples``.

    Returns ``(clone_of, profiles, members)``: ``clone_of`` maps a genotype id to its clone label
    (``"above_clades"`` for a cancer genotype above every clade when those cells are as many as a clade must
    hold, ``"other"`` when they are fewer, absent for normal cells), ``profiles`` is the
    clones x segments frame of integer copy number (the median over the clone's cells), ``members`` maps a
    merged label to the clades it holds."""
    from iscc.integrations import clones_from_clades
    gids = np.concatenate([_gids(s) for s in samples])
    tum = layer.is_tumour(gids, t)
    n_t = int(tum.sum())
    saved = getattr(t, "cell_data", None)
    t.cell_data = {"cell_type": pd.DataFrame({"cell_id": gids}, index=[f"c{i}" for i in range(len(gids))])}
    try:
        lab = clones_from_clades(t, min_cells=max(20, int(round(min_frac * n_t))))  # Fig 4b's truth rule
    finally:
        t.cell_data = saved
    lab = lab.astype(object).to_numpy()
    clade_of = {}
    for g, lb, is_t in zip(gids, lab, tum):
        if is_t:
            clade_of[g] = OTHER if (lb is None or (isinstance(lb, float) and np.isnan(lb))) else str(lb)
    ug = list(clade_of)
    cn = dict(zip(ug, genotype_cn(t, layer, ug)))
    segs = segment_names(layer)
    # The cells above every clade ("other": the trunk and branches too small to be clades) have profiles
    # every modality sees; they are one more clone here when they are as many as a clade must hold.
    min_cells = max(20, int(round(min_frac * n_t)))
    n_other = sum(1 for g, is_t in zip(gids, tum) if is_t and clade_of[g] == OTHER)
    if n_other >= min_cells:
        clade_of = {g: (TRUNK if c == OTHER else c) for g, c in clade_of.items()}
    cells = [(clade_of[g], cn[g]) for g, is_t in zip(gids, tum) if is_t and clade_of[g] != OTHER]
    clades = sorted({c for c, _ in cells},
                    key=lambda c: int(c.split("_")[-1]) if c.split("_")[-1].isdigit() else 10**6)
    prof = {c: np.rint(np.median(np.stack([v for k, v in cells if k == c]), axis=0)).astype(int) for c in clades}
    groups = {}
    for c in clades:
        groups.setdefault(_proportional_key(prof[c]), []).append(c)
    members, merged_of = {}, {}
    for key, cs in groups.items():
        name = "+".join(cs)
        members[name] = cs
        for c in cs:
            merged_of[c] = name
    clone_of = {g: (merged_of.get(c, OTHER)) for g, c in clade_of.items()}
    profiles = pd.DataFrame([prof[cs[0]] for cs in members.values()], index=list(members), columns=segs)
    n_of = pd.Series([clone_of[g] for g, is_t in zip(gids, tum) if is_t]).value_counts()
    profiles.attrs["n_cells"] = {k: int(n_of.get(k, 0)) for k in members}
    return clone_of, profiles, members


def label_cells(gids, clone_of):
    """The ground-truth label of each cell: a clone, ``"other"``, or ``"normal"``."""
    return np.array([clone_of.get(g, NORMAL) for g in gids], dtype=object)


# ---------------------------------------------------------------------------------------------------
# the normal reference, as the tools are used in practice (Fig 4b and 4d)
# ---------------------------------------------------------------------------------------------------
# A tool that takes a normal reference (CopyKAT with one, Numbat, inferCNV, Clonalscope: all through
# clonalscope/celltype.tsv) is given the sample's microenvironment normals, the cells a user annotates by
# cell type from markers: every non-epithelial normal cell; on Visium, the spots whose cells are all such
# normals (a pathologist's stroma annotation). Normal epithelial cells are never in the reference, since a
# user cannot tell them from cancer cells; everything outside the reference is clustered. The writers'
# own celltype.tsv labels a barcode by its TRUE tumour content, which no user has, so it is always
# overwritten with this one.
def cell_classes(t, gids):
    """Per cell: 2 a cancer cell, 1 a normal epithelial cell, 0 a microenvironment normal (any other type)."""
    out = np.zeros(len(gids), dtype=np.int8)
    for i, g in enumerate(gids):
        ty = getattr(t.genotypes.get(g), "type", None)
        out[i] = 2 if ty == "cancer" else (1 if ty == "epithelial" else 0)
    return out


def visium_reference(cls, members, keep):
    """Per in-tissue spot of ``keep`` (the writer's order), 1 when every cell in it is a microenvironment
    normal (``cell_classes`` 0), else 0. ``members[s]`` indexes the cells of spot ``s`` in ``cls``."""
    return np.array([int(not (cls[members[s]] > 0).any()) for s in keep], dtype=np.int8)


def write_reference(ds, barcodes, reference):
    """Write the reference into ``<ds>/clonalscope/celltype.tsv`` (``normal`` where ``reference`` is 1,
    ``tumor`` elsewhere), replacing the writer's truth-derived one."""
    pd.DataFrame({"barcode": list(barcodes),
                  "celltype": np.where(np.asarray(reference).astype(int) == 1, "normal", "tumor")}).to_csv(
        os.path.join(ds, "clonalscope", "celltype.tsv"), sep="\t", index=False)


def coarse_clones(profiles, k=COARSE_K):
    """The ground truth's clones grouped into ``k`` copy-number groups (average linkage on the L1
    distance between their relative profiles). Returns clone -> group label and the groups' profiles
    (each group's largest clone's, by the cell counts true_clones attaches)."""
    from scipy.cluster.hierarchy import fcluster, linkage
    names = list(profiles.index)
    if len(names) <= k:
        return {n: n for n in names}, profiles.copy()
    rel = profiles.to_numpy(float)
    rel = 2.0 * rel / rel.mean(axis=1, keepdims=True)
    grp = fcluster(linkage(rel, method="average", metric="cityblock"), t=k, criterion="maxclust")
    out, rows = {}, {}
    for g in sorted(set(grp), key=lambda g: names.index(next(n for n, x in zip(names, grp) if x == g))):
        members = [n for n, x in zip(names, grp) if x == g]
        label = "|".join(members)
        n = profiles.attrs.get("n_cells", {})
        rows[label] = profiles.loc[max(members, key=lambda m: n.get(m, 0))].to_numpy()
        for n in members:
            out[n] = label
    return out, pd.DataFrame.from_dict(rows, orient="index", columns=profiles.columns)


# ---------------------------------------------------------------------------------------------------
# 4c: the DNA chain (scDNA -> HMMcopy -> clusters) and clonealign
# ---------------------------------------------------------------------------------------------------
def dlp_estimate():
    """``estimate_dna`` fitted to the DLP+ reference (OV2295, 500 kb bins; Laks et al. 2019)."""
    from iscc.data import estimate_dna
    z = np.load(os.path.join(VAL, "data", "dna_ref_dlp_sc_wgs.npz"), allow_pickle=True)
    return estimate_dna(z["coverage"], z["alt"], z["cn"], modality="sc", breadth=str(z["breadth"]),
                        depth_model=str(z["depth_model"]))


def hmmcopy_bins(layer, dna, snv_columns):
    """HMMcopy's bin table for the scDNA loci: each growth locus is a bin at its place on the human
    chromosome (``layer.grid``), with the assay's GC and mappability. Sorted along the genome."""
    pos = pd.Index(snv_columns).get_indexer(list(dna.genes))
    if (pos < 0).any():
        raise ValueError("scDNA loci not found among the growth genes")
    first = layer.grid.sort_values(["growth_gene", "lo"]).groupby("growth_gene").first()
    arms = layer.reference.arms
    g = first.loc[pos]
    chrom = arms["chrom"].to_numpy()[g["arm"].to_numpy()]
    bins = pd.DataFrame({"chr": chrom, "start": g["lo"].to_numpy() + 1, "end": g["hi"].to_numpy(),
                         "gc": np.asarray(dna.gc, float), "map": np.asarray(dna.mappability, float),
                         "segment": g["segment"].to_numpy()},
                        index=pd.Index([str(x) for x in dna.genes], name="locus"))
    bins["_c"] = bins["chr"].str.replace("chr", "", regex=False).astype(int)
    return bins.sort_values(["_c", "start"]).drop(columns="_c")


def run_hmmcopy(reads, bins, work, cores=1):
    """HMMcopy per cell with the DLP+ single-cell parameterisation (tools/hmmcopy_dlp.R in the r-cnv
    image), the cells split into ``cores`` chunks run side by side. Returns the bins x cells frame of
    copy-number states (NaN where a bin or a cell was not called) and the per-cell table (multiplier)."""
    rscript = os.environ.get("ISCC_HMMCOPY_RSCRIPT")
    if not rscript:
        raise RuntimeError("set ISCC_HMMCOPY_RSCRIPT (validation/paper/tools/bin/rscript-r-cnv)")
    os.makedirs(work, exist_ok=True)
    bins_csv = os.path.join(work, "bins.csv")
    bins[["chr", "start", "end", "gc", "map"]].to_csv(bins_csv)
    chunks = [c for c in np.array_split(np.asarray(reads.columns), max(1, cores)) if len(c)]
    procs = []
    for k, cols in enumerate(chunks):
        rc = os.path.join(work, f"reads_{k}.csv")
        reads[list(cols)].to_csv(rc)
        out = os.path.join(work, f"hmm_{k}")
        log = open(os.path.join(work, f"hmmcopy_{k}.log"), "w")
        procs.append((subprocess.Popen([rscript, os.path.join(TOOLS, "hmmcopy_dlp.R"), rc, bins_csv, out],
                                       stdout=log, stderr=subprocess.STDOUT), out, log))
    states, per_cell = [], []
    for p, out, log in procs:
        rc = p.wait()
        log.close()
        if rc != 0:
            raise RuntimeError(f"hmmcopy_dlp.R failed (exit {rc}); see {log.name}:\n"
                               + open(log.name).read()[-1500:])
        states.append(pd.read_csv(out + "_state.csv", index_col=0))
        per_cell.append(pd.read_csv(out + "_cell.csv", dtype={"cell": str}))
    print("   ", open(procs[0][2].name).read().strip().splitlines()[-1], flush=True)
    st = pd.concat(states, axis=1)
    st.index = st.index.astype(str)
    st.columns = st.columns.astype(str)
    return st.reindex(index=bins.index, columns=reads.columns), pd.concat(per_cell).set_index("cell")


def cluster_cells(cn_bins, seed=0):
    """UMAP + HDBSCAN on the called copy-number states, in the manner of the DLP+ pipeline (Laks et al.
    2019; scgenome's umap_hdbscan_cluster). Returns labels (-1 = noise) and the embedding."""
    import umap
    from sklearn.cluster import HDBSCAN
    X = np.asarray(cn_bins, float)
    emb = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=seed).fit_transform(X)
    mcs = max(15, int(round(0.02 * len(X))))
    lab = HDBSCAN(min_cluster_size=mcs).fit_predict(emb)
    return lab, emb, mcs


def called_clones(seg_cn, lab, ploidy=None):
    """Clone profiles from the clusters: the median relative copy number per segment, put back on the
    cluster's own scale (``ploidy``, each cell's median called state; the cluster's median of them) and
    rounded; clusters with the same (proportional) profile merged; flat clusters (every segment at one
    level: diploid cells) set aside as normal when any other cluster remains. The scale matters for a
    doubled tumour: one copy lost from four is 1.5 on the relative scale, and rounding there erases it.

    Returns ``(clone_of_cluster, profiles)``: cluster id -> clone label (``"normal"`` for a flat one,
    None for noise) and the clones x segments frame of copy number."""
    clusters = sorted(c for c in set(lab.tolist()) if c >= 0)
    if ploidy is None:
        ploidy = np.full(len(lab), 2.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # a segment no cell of the cluster called
        prof = {}
        for c in clusters:
            p = np.nanmedian(ploidy[lab == c])
            p = 2.0 if not np.isfinite(p) or p <= 0 else p
            prof[c] = np.nan_to_num(np.rint(np.nanmedian(seg_cn[lab == c], axis=0) * p / 2.0), nan=p).astype(int)
    flat = {c for c in clusters if np.unique(prof[c]).size == 1}
    if len(flat) == len(clusters):
        flat = set()
    groups = {}
    for c in clusters:
        if c not in flat:
            groups.setdefault(_proportional_key(prof[c]), []).append(c)
    out, rows = {c: NORMAL for c in flat}, {}
    for i, cs in enumerate(sorted(groups.values(), key=lambda cs: -sum((lab == c).sum() for c in cs))):
        name = f"dna_{i + 1}"
        rows[name] = prof[cs[0]]
        for c in cs:
            out[c] = name
    return out, pd.DataFrame.from_dict(rows, orient="index")


def dna_chain(t, layer, dis, clone_of, work, seed=0, n_cells=N_DNA, cores=1):
    from iscc.data import scDNA
    wall, t0 = {}, time.time()
    est = dlp_estimate()
    cells_all = list(dis["cell_exp"].index)
    rng = np.random.default_rng([seed, 0xD7A])
    pick = set(rng.choice(np.arange(len(cells_all)), size=min(n_cells, len(cells_all)), replace=False).tolist())
    cells = [c for i, c in enumerate(cells_all) if i in pick]
    dna = scDNA(breadth=est.breadth, depth_model=est.depth_model, seed=200 + seed, **est.dna_kwargs()).run(
        dis, cell_subset=cells)
    t0 = _lap(wall, "scdna", t0)
    bins = hmmcopy_bins(layer, dna, dis["cell_snv"].columns)
    reads = dna.coverage.T.copy()
    reads.index = reads.index.astype(str)
    reads = reads.loc[bins.index].astype(int)
    reads.columns = [str(c) for c in reads.columns]
    states, hmm_cells = run_hmmcopy(reads, bins, os.path.join(work, "hmmcopy"), cores=cores)
    t0 = _lap(wall, "hmmcopy", t0)
    # Bins outside a cell's "ideal" set (mappability, GC, depth) carry no state; a cell counts as called
    # when it has a state on at least half its bins, and only bins called in 90% of those cells are
    # clustered on, a missing state taking the cell's median.
    cn = states.to_numpy(float)                                # states are copy numbers 0..11
    # A bin with no reads at all in a cell whose median bin holds at least 50 (the DLP+ pipeline's
    # good-cell depth) is a homozygous deletion; HMMcopy's ideal-bin filter would otherwise drop it.
    rd = reads.to_numpy(float)
    deep = np.median(rd, axis=0) >= 50
    cn[(rd == 0) & ~np.isfinite(cn) & deep[None, :]] = 0.0
    ok = np.isfinite(cn).mean(axis=0) >= 0.5
    good_bins = np.isfinite(cn[:, ok]).mean(axis=1) >= 0.9 if ok.any() else np.zeros(len(cn), bool)
    segs = segment_names(layer)
    seg_of_bin = bins["segment"].to_numpy()
    # Clones are found on relative copy number, each cell scaled so that its median bin is 2: scRNA,
    # the modality the clones are handed to, is library-normalised and cannot see ploidy, and the
    # scaling absorbs HMMcopy's ploidy (multiplier) errors, which would otherwise split a clone in two.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN segments of failed cells
        med = np.nanmedian(cn, axis=0)
        rel = 2.0 * cn / np.where(med > 0, med, np.nan)[None, :]
        seg_cn = np.stack([np.nanmedian(rel[seg_of_bin == s], axis=0) for s in range(len(segs))], axis=1)
        X = rel[good_bins][:, ok].T
        X = np.where(np.isfinite(X), X, np.nanmedian(X, axis=1, keepdims=True))

    lab = np.full(len(cells), -2)                              # -2: HMMcopy failed
    emb = np.full((len(cells), 2), np.nan)
    if ok.sum() >= 2:
        l_ok, e_ok, mcs = cluster_cells(X, seed=seed)
        lab[ok], emb[ok] = l_ok, e_ok
    else:
        mcs = None
    clone_of_cluster, profiles = called_clones(seg_cn, lab, ploidy=med)
    t0 = _lap(wall, "cluster", t0)

    gids = _gids(dis, cells)
    truth = label_cells(gids, clone_of)
    called = np.array([clone_of_cluster.get(c) if c >= 0 else None for c in lab], dtype=object)
    # each called clone stands for the true clone most of its DNA cells belong to
    mapping, purity = {}, {}
    for name in profiles.index:
        tv = pd.Series(truth[called == name]).value_counts()
        mapping[name], purity[name] = str(tv.index[0]), float(tv.iat[0] / tv.sum())
    # how well HMMcopy called each cell: as absolute copy number (the multiplier is its ploidy guess),
    # and up to the cell's scale (the called states rescaled to the true median)
    true_cn = genotype_cn(t, layer, gids)[:, seg_of_bin].astype(float)            # cells x bins
    called_ok = cn[:, ok].T
    fin = np.isfinite(called_ok)
    if ok.any():
        conc = float(np.mean(true_cn[ok][fin] == called_ok[fin]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            scale = np.median(true_cn[ok], axis=1, keepdims=True) / np.nanmedian(called_ok, axis=1, keepdims=True)
        conc_rel = float(np.mean(np.rint(called_ok * scale)[fin] == true_cn[ok][fin]))
    else:
        conc = conc_rel = float("nan")
    from sklearn.metrics import adjusted_rand_score
    m = (lab >= 0) & (truth != OTHER)
    diag = dict(n_cells=len(cells), n_hmmcopy_ok=int(ok.sum()), n_noise=int((lab == -1).sum()),
                n_tumour=int((truth != NORMAL).sum()), hdbscan_min_cluster_size=mcs,
                n_clusters=int(len(set(lab[lab >= 0].tolist()))), n_called_clones=int(len(profiles)),
                n_true_clones_present=int(len(set(truth[(truth != NORMAL) & (truth != OTHER)].tolist()))),
                cluster_ari=float(adjusted_rand_score(truth[m], lab[m])) if m.sum() > 1 else float("nan"),
                hmmcopy_bin_concordance=conc, hmmcopy_bin_concordance_scaled=conc_rel,
                hmmcopy_wrong_ploidy=float(np.mean(np.abs(scale - 1) > 0.25)) if ok.any() else float("nan"),
                mean_reads_per_bin=float(reads.to_numpy().mean()),
                n_bins=int(len(bins)), n_bins_clustered=int(good_bins.sum()), dlp_fit=repr(est), wall_s=wall)
    cells_tab = pd.DataFrame({"cell": cells, "genotype": [str(g) for g in gids], "true_clone": truth,
                              "hmmcopy_ok": ok.astype(int),
                              "multiplier": hmm_cells["multiplier"].reindex(reads.columns).round(3).to_numpy(),
                              "cluster": lab, "called_clone": called,
                              "umap1": emb[:, 0], "umap2": emb[:, 1]})
    seg_tab = pd.DataFrame(seg_cn, columns=segs)
    return dict(profiles=profiles, mapping=mapping, purity=purity, diag=diag, cells=cells_tab,
                seg_cn=seg_tab, sizes={n: int((called == n).sum()) for n in profiles.index})


def rna_counts(t, layer, dis, seed=0):
    """The human-gene scRNA of the dissociated cells, as make_layer_dataset draws it for Fig 4b."""
    from iscc.data import scRNA
    expr = layer.expression(t)
    dcells = list(dis["cell_exp"].index)
    sc = scRNA(n_cells=len(dcells), seed=100 + seed, dropout_mid=0.0, n_reads=10_000).run(
        dis, cell_subset=dcells, gene_lift=expr)
    return sc


def score_assignment(truth, pred):
    """Accuracy and ARI of predicted true-clone labels (None = no call) against the truth."""
    from sklearn.metrics import adjusted_rand_score
    truth = np.asarray(truth, dtype=object)
    pred = np.asarray([p if p is not None else "__none__" for p in pred], dtype=object)
    recall = {str(c): float(np.mean(pred[truth == c] == c)) for c in pd.unique(truth)}
    return dict(accuracy=float(np.mean(pred == truth)), balanced_accuracy=float(np.mean(list(recall.values()))),
                ari=float(adjusted_rand_score(truth.astype(str), pred.astype(str))), recall=recall)


def run_clonealign(Y, L, work, max_iter=200, n_repeats=2, seed=1, preprocess=1):
    """validation/clonealign_runner.R in the clonealign image; ``preprocess=1`` applies the package's own
    preprocess_for_clonealign. Returns the (cells x clones) probabilities, columns in L's order."""
    import integration_common as IC
    in_dir, out_dir = os.path.join(work, "in"), os.path.join(work, "out")
    os.makedirs(in_dir, exist_ok=True)
    Y.to_csv(os.path.join(in_dir, "Y.csv"))
    L.to_csv(os.path.join(in_dir, "L.csv"))
    r = subprocess.run([IC.CLONEALIGN_RSCRIPT, os.path.join(VAL, "clonealign_runner.R"), in_dir, out_dir,
                        str(int(max_iter)), str(int(n_repeats)), str(int(seed)), str(int(preprocess))],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"clonealign failed (exit {r.returncode}); Y {Y.shape}, L {L.shape}\n{r.stderr[-2000:]}")
    print("   ", " | ".join(x for x in r.stdout.splitlines() if x.startswith("clonealign:")), flush=True)
    P = pd.read_csv(os.path.join(out_dir, "clone_probs.csv"), index_col=0).reindex(Y.index)
    P.columns = list(L.columns)
    P.attrs["run"] = {k: int(v) for k, v in pd.read_csv(os.path.join(out_dir, "run.csv")).iloc[0].items()}
    return P


def informative_genes(X, gene_seg, profiles, n_max=RNA_GENES, min_counts=20):
    """The genes clonealign is given for one set of clone profiles: expressed (more than ``min_counts``
    UMIs over the cells), on a segment whose copy number differs between the clones and lies in 1..6 in
    every clone (preprocess_for_clonealign's rules, applied first so the matrix handed over stays small),
    at most ``n_max`` of them, the most expressed."""
    prof = profiles.to_numpy(float)
    seg_ok = (prof.std(axis=0) > 0) & (prof.min(axis=0) > 0) & (prof.max(axis=0) <= 6)
    tot = np.asarray(X.sum(axis=0)).ravel()
    cand = np.flatnonzero((gene_seg >= 0) & seg_ok[gene_seg] & (tot > min_counts))   # -1: on no arm
    if cand.size > n_max:
        cand = cand[np.argsort(-tot[cand])[:n_max]]
    return np.sort(cand)


def clonealign_arm(X, cells, layer, profiles, work, name):
    """clonealign on cells ``X`` given clone ``profiles``; returns the clone each cell is assigned to and
    the runner's record. With one clone, or no informative gene, every cell goes to the largest clone."""
    gene_seg = layer.genes["segment"].to_numpy()
    g = informative_genes(X, gene_seg, profiles)
    if len(profiles) < 2 or g.size == 0:
        return np.array([profiles.index[0]] * X.shape[0], dtype=object), dict(n_genes=int(g.size), skipped=True)
    ids = layer.genes["gene_id"].astype(str).to_numpy()[g]
    Y = pd.DataFrame(X[:, g].toarray().astype(float), index=cells, columns=[f"{x}.{i}" for i, x in zip(g, ids)])
    L = pd.DataFrame(profiles.to_numpy(float)[:, gene_seg[g]].T, index=Y.columns, columns=list(profiles.index))
    P = run_clonealign(Y, L, os.path.join(work, name), **CLONEALIGN)
    pred = np.asarray(list(profiles.index), dtype=object)[np.nanargmax(P.to_numpy(float), axis=1)]
    return pred, dict(n_genes=int(g.size), **P.attrs["run"])


def rna_inputs(t, layer, dis, clone_of, a):
    """The tumour cells clonealign is given (at most ``a.rna_cells``, those with a true clone) and their
    human-gene counts; the same cells on every call with the same seed."""
    sc = rna_counts(t, layer, dis, seed=a.seed)
    cells = list(sc.obs.index)
    truth = label_cells(_gids(dis, cells), clone_of)
    keep = np.flatnonzero((truth != NORMAL) & (truth != OTHER))
    rng = np.random.default_rng([a.seed, 0xCA])
    if keep.size > a.rna_cells:
        keep = np.sort(rng.choice(keep, size=a.rna_cells, replace=False))
    return sc.counts_csr[keep].tocsc(), [str(cells[i]) for i in keep], truth[keep]


def run_oracle_set(a):
    """The decomposition of 4c's gap: clonealign given the TRUE profiles of only the true clones the DNA
    step found (those its called clones stand for). Against the truth arm, the loss is the clones the
    DNA step missed or merged; against the DNA arm, the rest is the error in the called profiles.
    Writes ``c/<run>_<task>/oracle_set.json``; needs dna-rna's dna_clones.tsv."""
    out = os.path.join(a.out, "c", f"{a.run}_{a.task}")
    work = os.path.join(a.work, f"c_{a.run}_{a.task}")
    os.makedirs(work, exist_ok=True)
    dna = pd.read_csv(os.path.join(out, "dna_clones.tsv"), sep="\t", index_col=0)
    t, layer, dis, sec = sample(a.run, a.task, a.section_max_cells)
    clone_of, true_prof, members = true_clones(t, layer, [dis, sec])
    X, rna_ids, t_fine = rna_inputs(t, layer, dis, clone_of, a)
    found = [c for c in pd.unique(dna["true_clone"]) if c in true_prof.index]
    rec = dict(run=a.run, task=a.task, commit=_commit(), clones_found=found, n_true_clones=int(len(true_prof)),
               share_of_cells_in_found=float(np.isin(t_fine, found).mean()))
    if found:
        pred, rec["clonealign_run"] = clonealign_arm(X, rna_ids, layer, true_prof.loc[found], work, "ca_oracle")
        rec["oracle_set_arm"] = score_assignment(t_fine, pred)
    with open(os.path.join(out, "oracle_set.json"), "w") as f:
        json.dump(rec, f, indent=1, default=str)
    print("oracle_set", json.dumps({k: rec[k] for k in ("clones_found", "share_of_cells_in_found")}),
          json.dumps({k: round(v, 3) for k, v in rec.get("oracle_set_arm", {}).items() if not isinstance(v, dict)}),
          flush=True)


def run_dna_rna(a):
    from sklearn.metrics import adjusted_rand_score
    out = os.path.join(a.out, "c" if a.dosage == "regime" else f"c_{a.dosage}", f"{a.run}_{a.task}")
    os.makedirs(out, exist_ok=True)
    work = os.path.join(a.work, f"c_{a.run}_{a.task}")
    os.makedirs(work, exist_ok=True)
    wall, t0 = {}, time.time()
    t, layer, dis, sec = sample(a.run, a.task, a.section_max_cells, dosage=a.dosage)
    t0 = _lap(wall, "load_and_sample", t0)
    tinfo = write_tumour_info(t, a)
    clone_of, true_prof, members = true_clones(t, layer, [dis, sec])
    group_of, coarse_prof = coarse_clones(true_prof)
    t0 = _lap(wall, "truth", t0)
    dna = dna_chain(t, layer, dis, clone_of, work, seed=a.seed, n_cells=a.n_dna, cores=a.cores)
    t0 = _lap(wall, "dna_chain", t0)

    X, rna_ids, t_fine = rna_inputs(t, layer, dis, clone_of, a)
    keep = np.arange(len(rna_ids))
    t0 = _lap(wall, "scrna", t0)
    t_coarse = np.array([group_of[c] for c in t_fine], dtype=object)
    segs = segment_names(layer)
    rec = dict(run=a.run, task=a.task, commit=_commit(), paper_out=os.environ.get("ISCC_PAPER_OUT"),
               tier=os.environ.get("ISCC_PAPER_TIER"), seed=a.seed, clonealign=CLONEALIGN, tumour=tinfo,
               section_max_cells=int(a.section_max_cells),
               n_rna_cells=int(len(keep)), true_clones=members, coarse_groups=sorted(set(group_of.values())),
               dna=dna["diag"], dna_mapping=dna["mapping"], dna_purity=dna["purity"])
    resolutions = (("fine", t_fine, true_prof),) + ((("coarse", t_coarse, coarse_prof),) if a.coarse else ())
    for res, tt, prof in resolutions:
        sizes = pd.Series(tt).value_counts()
        rec[res] = dict(n_true_clones=int(len(prof)), majority=float(sizes.iat[0] / sizes.sum()),
                        chance=float(1.0 / len(prof)))
    t0 = _lap(wall, "inputs", t0)

    # (i) the true clones and their profiles, at both resolutions
    pred_true, rec["fine"]["clonealign_true_run"] = clonealign_arm(X, rna_ids, layer, true_prof, work, "ca_true")
    rec["fine"]["truth_arm"] = score_assignment(t_fine, pred_true)
    t0 = _lap(wall, "clonealign_true", t0)
    pred_coarse = np.array([None] * len(keep), dtype=object)
    if a.coarse:
        if len(coarse_prof) < len(true_prof):
            pred_coarse, rec["coarse"]["clonealign_true_run"] = clonealign_arm(X, rna_ids, layer, coarse_prof,
                                                                             work, "ca_coarse")
        else:
            pred_coarse = np.array([group_of[c] for c in pred_true], dtype=object)
        rec["coarse"]["truth_arm"] = score_assignment(t_coarse, pred_coarse)
        t0 = _lap(wall, "clonealign_coarse", t0)

    # (ii) the clones called from simulated scDNA; each stands for the true clone most of its DNA cells
    # belong to, and for that clone's coarse group
    prof = dna["profiles"]
    if len(prof):
        called, rec["clonealign_dna_run"] = clonealign_arm(X, rna_ids, layer, prof, work, "ca_dna")
    else:
        called = np.array([None] * len(keep), dtype=object)
    pred_dna = np.array([dna["mapping"].get(c) if c is not None else None for c in called], dtype=object)
    pred_dna_coarse = np.array([group_of.get(c) if c is not None else None for c in pred_dna], dtype=object)
    for res, tt, pp in (("fine", t_fine, pred_dna), ("coarse", t_coarse, pred_dna_coarse))[:len(resolutions)]:
        rec[res]["dna_arm"] = score_assignment(tt, pp)
        rec[res]["dna_arm"]["ari_unmapped"] = float(adjusted_rand_score(tt.astype(str), called.astype(str)))
        rec[res]["gap"] = rec[res]["truth_arm"]["accuracy"] - rec[res]["dna_arm"]["accuracy"]
        rec[res]["gap_balanced"] = (rec[res]["truth_arm"]["balanced_accuracy"]
                                    - rec[res]["dna_arm"]["balanced_accuracy"])
    t0 = _lap(wall, "clonealign_dna", t0)
    rec["wall_s"] = wall

    # summaries only
    tp = true_prof.copy()
    tp.insert(0, "group", [group_of[c] for c in tp.index])
    tp.insert(0, "n_cells", [int(sum(1 for g in _gids(dis) if clone_of.get(g) == c)) for c in tp.index])
    tp.to_csv(os.path.join(out, "true_clones.tsv"), sep="\t", index_label="clone")
    dp = prof.copy()
    dp.columns = segs
    dp.insert(0, "purity", [dna["purity"][n] for n in dp.index])
    dp.insert(0, "true_clone", [dna["mapping"][n] for n in dp.index])
    dp.insert(0, "n_cells", [dna["sizes"][n] for n in dp.index])
    dp.to_csv(os.path.join(out, "dna_clones.tsv"), sep="\t", index_label="clone")
    dc = pd.concat([dna["cells"], dna["seg_cn"].round(2)], axis=1)
    dc.to_csv(os.path.join(out, "dna_cells.tsv.gz"), sep="\t", index=False)
    pd.DataFrame({"cell": rna_ids, "true_clone": t_fine, "true_group": t_coarse, "clonealign_true": pred_true,
                  "clonealign_coarse": pred_coarse, "clonealign_dna": called,
                  "clonealign_dna_mapped": pred_dna}).to_csv(os.path.join(out, "rna_cells.tsv.gz"), sep="\t",
                                                            index=False)
    with open(os.path.join(out, "score.json"), "w") as f:
        json.dump(rec, f, indent=1, default=str)
    for res, _, _ in resolutions:
        r = rec[res]
        print(res, json.dumps(dict(n_true_clones=r["n_true_clones"], majority=round(r["majority"], 3),
                                   truth=round(r["truth_arm"]["accuracy"], 3),
                                   truth_balanced=round(r["truth_arm"]["balanced_accuracy"], 3),
                                   dna=round(r["dna_arm"]["accuracy"], 3),
                                   dna_balanced=round(r["dna_arm"]["balanced_accuracy"], 3),
                                   gap=round(r["gap"], 3))), flush=True)
    print("dna:", json.dumps({k: v for k, v in dna["diag"].items() if k != "dlp_fit"}), flush=True)


# ---------------------------------------------------------------------------------------------------
# 4d: Visium, Clonalscope with and without the DNA prior, CalicoST
# ---------------------------------------------------------------------------------------------------
def write_prior(prior_dir, dna_clones_tsv, layer):
    """Clonalscope's DNA prior from the 4c chain's called clones: ``U0.tsv`` (a normal row of 1s and one
    row per DNA clone: its copy number over its gene-weighted mean, the relative coverage Clonalscope
    estimates per segment) and ``states.tsv`` (the DNA pseudobulk on the same scale, 1 within 5%)."""
    dc = pd.read_csv(dna_clones_tsv, sep="\t", index_col=0)
    segs = segment_names(layer)
    cn = dc[segs].to_numpy(float)
    w = layer.genes["segment"].value_counts().reindex(range(len(segs)), fill_value=0).to_numpy(float)
    ploidy = (cn * w).sum(axis=1, keepdims=True) / w.sum()
    rel = cn / ploidy
    U0 = pd.DataFrame(np.vstack([np.ones(len(segs)), rel]), index=[NORMAL] + list(dc.index), columns=segs)
    n = dc["n_cells"].to_numpy(float)
    bulk = (rel * n[:, None]).sum(axis=0) / n.sum()
    states = np.where(np.abs(bulk - 1) < 0.05, 1.0, np.round(bulk, 3))
    os.makedirs(prior_dir, exist_ok=True)
    U0.round(4).to_csv(os.path.join(prior_dir, "U0.tsv"), sep="\t", index_label="cluster")
    pd.DataFrame({"arm": segs, "state": states}).to_csv(os.path.join(prior_dir, "states.tsv"), sep="\t", index=False)
    return U0, states


def cn_oracle(X, gene_ids, layer, true_prof, reference, min_counts=20, pseudo=0.1):
    """The copy-number oracle of 4d: each spot takes the true clone whose copy-number profile its expression
    follows best, given the true profiles (no clone discovery). Per spot and segment, the mean over the
    segment's informative genes of log((count per 10,000 + pseudo) / (the reference spots' mean + pseudo));
    per clone, log2 copy number; each centred over the segments whose copy number differs between clones,
    and the clone with the highest Pearson correlation taken. Informative genes: on such a segment, with
    more than ``min_counts`` counts over the spots. ``X`` spots x genes (counts), ``reference`` the boolean
    mask of the reference spots. Returns the clone per spot and the number of segments and genes used."""
    seg_of = dict(zip(layer.genes["gene_id"].astype(str), layer.genes["segment"].to_numpy()))
    gseg = np.array([seg_of.get(str(g), -1) for g in gene_ids])
    prof = true_prof.to_numpy(float)
    segs = np.flatnonzero((prof.std(axis=0) > 0) & (prof.min(axis=0) > 0))
    X = np.asarray(X, dtype=float)
    lib = X.sum(axis=1, keepdims=True)
    N = 1e4 * X / np.maximum(lib, 1.0)
    base = N[reference].mean(axis=0) if reference.any() else N.mean(axis=0)
    expressed = X.sum(axis=0) > min_counts
    R = np.log((N + pseudo) / (base + pseudo))
    cols, used = [], []
    for k in segs:
        g = np.flatnonzero((gseg == k) & expressed)
        if g.size:
            cols.append(R[:, g].mean(axis=1))
            used.append(k)
    if len(used) < 2:
        return np.array([true_prof.index[0]] * X.shape[0], dtype=object), dict(n_segments=len(used), n_genes=0)
    E = np.column_stack(cols)
    E = E - E.mean(axis=1, keepdims=True)
    C = np.log2(prof[:, used])
    C = C - C.mean(axis=1, keepdims=True)
    En = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
    Cn = C / np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-12)
    pick = np.argmax(En @ Cn.T, axis=1)
    n_genes = int(sum(((gseg == k) & expressed).sum() for k in used))
    return np.asarray(list(true_prof.index), dtype=object)[pick], dict(n_segments=len(used), n_genes=n_genes)


def run_spatial(a):
    import regime as RG
    from iscc.data import Visium
    from iscc.data.human import HUMAN_VISIUM_KAPPA, write_visium
    tag = f"{a.run}_{a.task}"
    bleed = getattr(a, "bleed", "regime")
    out = os.path.join(a.out, "d" if bleed == "regime" else f"d_{bleed}", tag)
    os.makedirs(out, exist_ok=True)
    work = os.path.join(a.work, f"d_{tag}")
    os.makedirs(work, exist_ok=True)
    dna_tsv = os.path.join(a.out, "c", tag, "dna_clones.tsv")
    if not os.path.exists(dna_tsv):
        sys.exit(f"no DNA clones at {dna_tsv}: run dna-rna first")
    wall, t0 = {}, time.time()
    t, layer, dis, sec = sample(a.run, a.task, a.section_max_cells)
    t0 = _lap(wall, "load_and_sample", t0)
    clone_of, true_prof, members = true_clones(t, layer, [dis, sec])
    del dis
    t0 = _lap(wall, "truth", t0)
    expr = layer.expression(t)
    vkw = dict(RG.VISIUM) if bleed == "regime" else {**RG.VISIUM, "bleed_rate": 0.0}   # off: no spot swapping
    vz = Visium(seed=300 + a.seed, kappa=HUMAN_VISIUM_KAPPA, **vkw).run(sec, grid_side=t.grid_size,
                                                                           gene_lift=expr)
    av = layer.visium_allele_counts(vz, t, seed=a.seed)
    ds = os.path.join(work, "visium")
    write_visium(ds, vz, layer, allele=av, tumor=t, seed=a.seed)
    t0 = _lap(wall, "visium", t0)

    # the truth per in-tissue spot: its tumour cells' true clones
    from iscc.data.human.writers import barcodes
    placed, members_of_spot = vz._grid["placed"], vz._grid["members"]
    pg = _gids(placed, list(placed["cell_exp"].index))
    lab = label_cells(pg, clone_of)
    bcs = barcodes(len(vz.spot_counts.index), seed=a.seed)
    keep = np.flatnonzero(np.asarray(vz.obs["n_cells"]) > 0)
    clones = list(true_prof.index)
    rows = []
    for s in keep:
        m = members_of_spot[s]
        lm = lab[m]
        rows.append([bcs[s], int(m.size), int((lm != NORMAL).sum()), int((lm == OTHER).sum())]
                    + [int((lm == c).sum()) for c in clones])
    spots = pd.DataFrame(rows, columns=["barcode", "n_cells", "n_tumour", "n_other"] + [f"n:{c}" for c in clones])
    pos = pd.read_csv(os.path.join(ds, "outs", "spatial", "tissue_positions.csv"))
    spots = spots.merge(pos[["barcode", "array_row", "array_col"]], on="barcode", how="left")
    stored_key = list(spots.columns)

    # the normal reference as the tools are used (Fig 4b's protocol): the spots holding only
    # microenvironment normals; every other spot, normal epithelium included, is clustered
    cls = cell_classes(t, pg)
    ref = visium_reference(cls, members_of_spot, keep)
    write_reference(ds, [bcs[s] for s in keep], ref)
    spots["n_normal_epithelium"] = [int((cls[members_of_spot[s]] == 1).sum()) for s in keep]
    spots["reference"] = ref
    clustered = ref == 0
    oracle_calls, oracle_info = cn_oracle(vz.spot_counts.to_numpy()[keep], list(vz.spot_counts.columns), layer,
                                          true_prof, np.asarray(ref, bool))
    print("copy-number oracle:", json.dumps(oracle_info), flush=True)
    protocol = dict(kind="microenvironment normals: the spots holding only non-epithelial normal cells; "
                         "every other spot clustered (fig4_chain.visium_reference)",
                    n_reference=int(ref.sum()), n_clustered=int(clustered.sum()),
                    n_clustered_with_cancer=int((clustered & (spots["n_tumour"] > 0)).sum()),
                    n_clustered_normal_epithelium_only=int((clustered & (spots["n_tumour"] == 0)).sum()),
                    n_reference_with_cancer=int((~clustered & (spots["n_tumour"] > 0)).sum()))
    print("reference protocol:", json.dumps(protocol), flush=True)

    # the tools (all of them, or --only), CalicoST in the background since it is the slowest; a tool not
    # redone keeps its stored labels, which needs the rebuilt truth to be the stored one
    only = list(SPATIAL_TOOLS) if not getattr(a, "only", None) else \
        ([] if a.only == "none" else a.only.split(","))          # none: only the oracle, every tool kept
    unknown = set(only) - set(SPATIAL_TOOLS)
    if unknown:
        sys.exit(f"--only: unknown spatial tools {sorted(unknown)} (of {SPATIAL_TOOLS})")
    kept = [x for x in SPATIAL_TOOLS if x not in only]
    old_rec = {}
    if kept:
        old_p = os.path.join(out, "spots.tsv.gz")
        if not os.path.exists(old_p):
            sys.exit(f"--only {a.only}: no stored spots at {old_p} to keep {kept} from")
        old = pd.read_csv(old_p, sep="\t", dtype={f"call:{x}": str for x in SPATIAL_TOOLS})
        key = [c for c in stored_key if c in old.columns]
        if key != stored_key or not old[key].reset_index(drop=True).equals(spots[key].reset_index(drop=True)):
            raise RuntimeError("the rebuilt spots differ from the stored ones: rerun every spatial tool, not --only")
        for x in kept:
            if f"call:{x}" in old:
                spots[f"call:{x}"] = old[f"call:{x}"].to_numpy()
        with open(os.path.join(out, "score.json")) as f:
            old_rec = json.load(f)
    bin_dir = os.path.join(TOOLS, "bin")
    tools_dir = os.environ["ISCC_TOOLS_DIR"]
    runs = {x: os.path.join(work, x) for x in only}
    prior_dir = os.path.join(work, "prior")
    U0, states = write_prior(prior_dir, dna_tsv, layer)
    logs = {k: open(os.path.join(work, f"{k}.log"), "w") for k in runs}
    calicost = None
    if "calicost" in runs:
        calicost = subprocess.Popen([os.path.join(bin_dir, "python-calicost"), os.path.join(TOOLS, "calicost.py"),
                                     ds, runs["calicost"], "--seed", str(a.seed)],
                                    stdout=logs["calicost"], stderr=subprocess.STDOUT)
    status = {}
    for name, extra in (("clonalscope", []), ("clonalscope_dna", [prior_dir])):
        if name not in runs:
            continue
        t1 = time.time()
        r = subprocess.run([os.path.join(bin_dir, "rscript-r-cnv"), os.path.join(TOOLS, "clonalscope.R"), ds,
                            runs[name], os.path.join(tools_dir, "data", "clonalscope"), "20", str(200 + a.seed)]
                           + extra, stdout=logs[name], stderr=subprocess.STDOUT)
        status[name] = r.returncode
        wall[name] = round(time.time() - t1, 1)
        print(f"  {name}: exit {r.returncode}, {wall[name]} s", flush=True)
    if calicost is not None:
        t1 = time.time()
        status["calicost"] = calicost.wait()
        wall["calicost_wait"] = round(time.time() - t1, 1)
        print(f"  calicost: exit {status['calicost']}", flush=True)
    for f in logs.values():
        f.close()

    info = {x: dict(old_rec.get("runs", {}).get(x, {}), kept_from_commit=old_rec.get("commit")) for x in kept}
    for name, d in runs.items():
        p = os.path.join(d, "calls.tsv")
        if status[name] == 0 and os.path.exists(p):
            calls = pd.read_csv(p, sep="\t", dtype={"clone": str}).set_index("barcode")["clone"]
            spots[f"call:{name}"] = calls.reindex(spots["barcode"]).to_numpy()
            with open(os.path.join(d, "run.json")) as f:
                info[name] = json.load(f)
        else:
            info[name] = dict(failed=True, exit=status[name],
                              log_tail=open(os.path.join(work, f"{name}.log")).read()[-3000:])
    spots["call:oracle_cn"] = oracle_calls
    info["oracle_cn"] = oracle_info
    spots.to_csv(os.path.join(out, "spots.tsv.gz"), sep="\t", index=False)
    rec = dict(run=a.run, task=a.task, commit=_commit(), bleed=bleed, paper_out=os.environ.get("ISCC_PAPER_OUT"),
               tier=os.environ.get("ISCC_PAPER_TIER"), seed=a.seed, true_clones=members,
               section_max_cells=int(a.section_max_cells), n_section_cells=int(len(placed["cell_exp"])),
               n_spots=int(len(spots)), n_tumour_spots=int((spots["n_tumour"] > 0).sum()),
               mean_cells_per_spot=float(spots["n_cells"].mean()),
               mean_cells_per_tumour_spot=float(spots.loc[spots["n_tumour"] > 0, "n_cells"].mean()),
               reference_protocol=protocol, redone=only, kept=kept,
               prior=dict(U0=U0.round(3).to_dict(orient="index"), states=dict(zip(segment_names(layer), states))),
               runs=info, wall_s=wall)
    import fig4_score as FS
    rec["scores"] = FS.score_spots(spots, os.path.join(a.out, "c", tag))
    with open(os.path.join(out, "score.json"), "w") as f:
        json.dump(rec, f, indent=1, default=str)
    print(json.dumps(rec["scores"]), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("dna-rna", "spatial", "info", "oracle-set"))
    ap.add_argument("--run", default="R14")
    ap.add_argument("--task", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", required=True, help="scratch for every tool input and output (a job's $SNIC_TMP)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cores", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    ap.add_argument("--n-dna", type=int, default=N_DNA)
    ap.add_argument("--rna-cells", type=int, default=RNA_MAX_CELLS)
    ap.add_argument("--rna-genes", type=int, default=RNA_GENES)
    ap.add_argument("--section-max-cells", type=int, default=SECTION_MAX_CELLS,
                    help="cells in the Visium section (default $ISCC_SECTION_MAX_CELLS, else no cap)")
    ap.add_argument("--dosage", choices=("regime", "proportional"), default="regime",
                    help="dna-rna: the regime's buffered dosage, or every gene proportional to copy number (the "
                         "counterfactual of exp-paper1-dosage-counterfactual), written to OUT/c_proportional")
    ap.add_argument("--bleed", choices=("regime", "off"), default="regime",
                    help="spatial: the regime's spot swapping, or none (bleed_rate 0), written to OUT/d_off")
    ap.add_argument("--coarse", action="store_true",
                    help="also run clonealign on the copy-number groups of the true clones (a diagnostic)")
    ap.add_argument("--only", default=None,
                    help="spatial: comma-separated tools to redo (of clonalscope, clonalscope_dna, calicost); the "
                         "others keep their stored labels (default: all; none: only the copy-number oracle)")
    a = ap.parse_args()
    warnings.filterwarnings("ignore", category=FutureWarning)
    try:
        {"dna-rna": run_dna_rna, "spatial": run_spatial, "info": run_info,
         "oracle-set": run_oracle_set}[a.step](a)
    finally:
        if os.environ.get("FIG4_KEEP_WORK") != "1":
            for d in (f"c_{a.run}_{a.task}", f"d_{a.run}_{a.task}"):
                shutil.rmtree(os.path.join(a.work, d), ignore_errors=True)


if __name__ == "__main__":
    main()
